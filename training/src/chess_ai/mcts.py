"""Monte Carlo Tree Search — port of `src/game/ai/MCTS.ts`.

AlphaZero-style MCTS: each node tracks visit count + accumulated value + prior;
selection uses PUCT; leaf evaluations come from the NN; backprop flips sign
each ply (zero-sum game convention).

Batched search (`run_batched_mcts`) advances N games in lockstep, gathering all
pending leaf-eval requests into a single NN batch per simulation step so the
GPU stays busy. Mirrors the TS `runBatchedMCTS` exactly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .encoding import POLICY_SIZE, encode_board, move_to_index
from .engine import (
    ChessGameState,
    Move,
    Position,
    apply_move,
    expand_children,
    get_legal_moves,
    position_key,
)

# Mate-distance preference. Terminal checkmate scores shrink slightly
# with search depth so a mate the search can reach SOONER outranks the
# same mate further away. config.py deliberately keeps value_ply_decay
# at 1.0 (decaying WDL outcome labels turns "winning but far away" into
# "draw" and corrupted an earlier run), and states that "prefer faster
# wins should come from search terminal handling" — this is that
# handling. It is symmetric, so a loss-seeking (jester) search likewise
# prefers to be mated as soon as possible.
MATE_DEPTH_DISCOUNT = 0.01
MATE_MIN_MAGNITUDE = 0.5


def mate_value(depth: int) -> float:
    """Magnitude of a checkmate score found `depth` plies below the root."""
    return max(MATE_MIN_MAGNITUDE, 1.0 - MATE_DEPTH_DISCOUNT * depth)


C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
FPU_REDUCTION = 0.0


def set_mcts_params(
    c_puct: float | None = None,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float | None = None,
    fpu_reduction: float | None = None,
) -> None:
    """Override MCTS hyperparams at runtime (before starting self-play).

    Called by Trainer once at startup and by each MP worker at import time
    so every process uses the same values. Not thread-safe — intended to
    be called from a single thread before any MCTS search starts.

    c_puct           — PUCT exploration constant in UCB. AlphaZero used 1.0
                       for Go, ~2 for chess. Higher = more exploration.
    dirichlet_alpha  — concentration of noise added to the root prior.
                       ~0.3 for chess (avg ~30 legal moves). Lower = spikier.
    dirichlet_epsilon — mixing weight: prior = (1-eps)*prior + eps*noise.
                        0 disables root noise entirely.
    fpu_reduction    — First-Play Urgency reduction. Unvisited children get
                        an initial Q of parent_Q - fpu_reduction so PUCT
                        doesn't over-commit to the prior-peaked leading
                        move. 0.0 = old behavior (unvisited Q=0).
                        Leela-derived rigs use ~0.4–0.5.
    """
    global C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, FPU_REDUCTION
    if c_puct is not None:
        C_PUCT = c_puct
    if dirichlet_alpha is not None:
        DIRICHLET_ALPHA = dirichlet_alpha
    if dirichlet_epsilon is not None:
        DIRICHLET_EPSILON = dirichlet_epsilon
    if fpu_reduction is not None:
        FPU_REDUCTION = fpu_reduction

# Batched evaluator signature: takes an (B, 8*8*NUM_PLANES) numpy array and
# returns (policies [B, POLICY_SIZE], values [B] scalar from current-player's
# perspective). Used by the self-play loop to batch calls to the PyTorch net.
BatchedEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class MCTSNode:
    __slots__ = (
        "parent",
        "children",
        "move",
        "state",
        "visit_count",
        "total_value",
        "prior",
        "is_expanded",
        "is_terminal",
        "terminal_value",
        "pos_key",
        "depth",
    )

    def __init__(self, state: ChessGameState, parent: "MCTSNode | None" = None, move: Move | None = None):
        self.parent: MCTSNode | None = parent
        self.depth: int = 0 if parent is None else parent.depth + 1
        self.children: dict[int, MCTSNode] = {}
        self.move: Move | None = move
        self.state: ChessGameState = state
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        # Repetition-aware searches cache each node's position key here
        # (computed lazily on first descent arrival). A tree has exactly
        # one path to each node, so a repetition verdict is permanent.
        self.pos_key: bytes | None = None


@dataclass
class MCTSResult:
    policy: np.ndarray       # [POLICY_SIZE] visit-based policy target
    move: Move               # Sampled move
    root_value: float        # Root mean Q (for logging)
    # Q of the best VISITED root child from the mover's perspective —
    # "the value if I play my best move." This is the resignation
    # statistic: root_value (mean Q over all sims) is dragged down by
    # forced exploration of the mover's own bad moves, so triggering
    # resign on it systematically over-fires.
    best_q: float = 0.0


class MCTSSearch:
    """Single-game MCTS; fed by an external batched evaluator.

    Call sequence per simulation:
        board = search.select_leaf()   # None if terminal (already backpropped)
        # ... NN evaluates `board` ...
        search.supply_eval(policy, value)
    After N sims call `search.get_result()`.
    """

    def __init__(
        self,
        state: ChessGameState,
        board_encoder: "Callable[[ChessGameState], np.ndarray] | None" = None,
        position_counts: "dict[bytes, int] | None" = None,
        invert_turns: "str | None" = None,
    ):
        self.root = MCTSNode(state)
        self._pending_leaf: MCTSNode | None = None
        # None = Sage's 20-plane encode_board. Toy passes its 6-plane
        # encoder; the search itself is encoding-agnostic.
        self._encode = board_encoder or encode_board
        # In-tree repetition awareness. `position_counts` is the GAME's
        # position-occurrence history (the same dict self-play/eval use
        # for threefold adjudication; root occurrence included). When
        # given, any in-tree position that (a) would complete FIDE
        # threefold against the game history, or (b) repeats ANY position
        # on its own search path (lc0-style twofold rule — in-tree
        # shuffling makes no progress by definition) is scored as a
        # terminal draw. Without this the search is blind to the
        # shuffle-draws the game rules adjudicate, so a winning side
        # happily repeats — the root cause of self-play collapsing into
        # ~90% threefold draws.
        self._game_counts = position_counts
        self._root_key = position_key(state) if position_counts is not None else None
        # Misère ("Jester") selection: at nodes whose side-to-move is in
        # `invert_turns` (None | "white" | "black" | "both"), PUCT picks
        # the child MAXIMIZING the opponent's Q instead of minimizing it
        # — the mover is trying to LOSE. Everything else (backup signs,
        # terminal values, value labels, visit-count policy targets)
        # keeps truthful chess semantics: the value head stays an honest
        # "who is winning" estimator; only what the search WANTS flips.
        #   Jester-vs-winner game: invert_turns = jester's color (the
        #   opponent is modeled as trying to win — which is aligned with
        #   the jester's goal, so opponent plies keep normal selection).
        #   Jester-vs-jester game: invert_turns = "both".
        self._invert_turns = invert_turns
        self._check_terminal(self.root)

    def is_terminal(self) -> bool:
        return self.root.is_terminal

    def get_root_board(self) -> np.ndarray:
        return self._encode(self.root.state)

    def init_root(
        self,
        policy: np.ndarray,
        value: float,
        dirichlet_epsilon: float | None = None,
    ) -> None:
        """Populate the root's children with priors from the NN, add Dirichlet noise.

        `dirichlet_epsilon=None` uses the module-global (self-play);
        pass 0.0 to disable root noise entirely (eval/match play).
        """
        self._expand_with_policy(self.root, policy)
        _add_dirichlet_noise(self.root, dirichlet_epsilon)
        _backpropagate(self.root, value)

    def select_leaf(self) -> np.ndarray | None:
        """Descend from root to an unexpanded (or terminal) leaf."""
        node = self.root
        track = self._game_counts is not None
        path_keys: set = {self._root_key} if track else set()
        while node.is_expanded and not node.is_terminal:
            node = _select_child(node, self._invert_turns)
            if track and not node.is_terminal:
                if node.pos_key is None:
                    node.pos_key = position_key(node.state)
                    occ = self._game_counts.get(node.pos_key, 0)  # type: ignore[union-attr]
                    if node.pos_key in path_keys or occ + 1 >= 3:
                        node.is_terminal = True
                        node.is_expanded = True
                        node.terminal_value = 0.0
                path_keys.add(node.pos_key)

        if node.is_terminal:
            _backpropagate(node, node.terminal_value)
            self._pending_leaf = None
            return None

        self._pending_leaf = node
        return self._encode(node.state)

    def pending_leaf_turn(self) -> str | None:
        """Side-to-move at the leaf awaiting evaluation — lets the
        batched driver route the eval to the right net in dual-net
        (agent vs frozen-opponent) searches."""
        return self._pending_leaf.state.currentTurn if self._pending_leaf else None

    def supply_eval(self, policy: np.ndarray, value: float) -> None:
        if self._pending_leaf is None:
            return
        self._expand_with_policy(self._pending_leaf, policy)
        _backpropagate(self._pending_leaf, value)
        self._pending_leaf = None

    def get_result(
        self,
        rng: random.Random | None = None,
        temperature: float = 1.0,
    ) -> MCTSResult:
        rng = rng or random
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = sum(c.visit_count for c in self.root.children.values())
        if total_visits > 0:
            for idx, child in self.root.children.items():
                policy[idx % POLICY_SIZE] += child.visit_count / total_visits
        move = _sample_move(self.root, rng, temperature) if self.root.children else Move(Position(0, 0), Position(0, 0))
        root_value = (
            self.root.total_value / self.root.visit_count if self.root.visit_count > 0 else 0.0
        )
        visited_qs = [
            -c.total_value / c.visit_count
            for c in self.root.children.values()
            if c.visit_count > 0
        ]
        best_q = max(visited_qs) if visited_qs else root_value
        return MCTSResult(policy=policy, move=move, root_value=root_value, best_q=best_q)

    # --- internals ---

    def _check_terminal(self, node: MCTSNode) -> None:
        from .selfplay import _is_insufficient_material
        if _is_insufficient_material(node.state.board):
            node.is_terminal = node.is_expanded = True
            node.terminal_value = 0.0
            return
        s = node.state.status
        if s in ("checkmate", "stalemate", "draw"):
            node.is_terminal = True
            node.is_expanded = True
            # Terminal value is from the perspective of the player-to-move at the
            # terminal node. Checkmate = they have no moves and are in check, so
            # they've LOST: value = -1. Stalemate/draw: 0.
            node.terminal_value = -mate_value(node.depth) if s == "checkmate" else 0.0
        elif s in ("active", "check"):
            # Fresh post-apply_move state: apply_move already computed status via
            # get_legal_moves, so "active"/"check" guarantees >=1 legal move.
            # Skipping the recomputation here is a ~2x win on MCTS expansion.
            pass
        else:
            # Root node with status "waiting" or unexpected value — fall back to
            # the safe (but expensive) check. No legal moves means checkmate
            # when the side to move is in check (value -1 from their
            # perspective), stalemate otherwise. (An earlier version labeled
            # both cases 0.0, scoring mates as draws on this path.)
            moves = get_legal_moves(node.state)
            if not moves:
                from .engine import is_in_check
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = (
                    -mate_value(node.depth)
                    if is_in_check(node.state.board, node.state.currentTurn)
                    else 0.0
                )

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray) -> None:
        # `expand_children` bundles get_legal_moves + one apply_move per child
        # into a single Rust call (or a Python fallback loop). Replacing the
        # old "loop-of-apply_move" pattern is the main FFI-cost reduction in
        # the MCTS hot path — ~30 Python/Rust crossings per expansion → 1.
        children = expand_children(node.state)
        if not children:
            from .engine import is_in_check
            node.is_terminal = True
            node.is_expanded = True
            node.terminal_value = (
                -mate_value(node.depth)
                if is_in_check(node.state.board, node.state.currentTurn)
                else 0.0
            )
            return

        is_white = node.state.currentTurn == "white"

        # All legal promotions stay in the tree. The legacy from/to
        # policy head shares their prior and receives aggregated visits.
        from collections import Counter
        counts = Counter(move_to_index(move, is_white) for move, _ in children)
        prior_sum = sum(float(policy[idx]) for idx in counts)
        for move, child_state in children:
            mi = move_to_index(move, is_white)
            promotion_offset = {"rook": 1, "bishop": 2, "knight": 3}.get(move.promotion, 0)
            child = MCTSNode(child_state, parent=node, move=move)
            child.prior = (float(policy[mi]) / prior_sum / counts[mi]
                           if prior_sum > 0 else 1.0 / len(children))
            self._check_terminal(child)
            node.children[mi + promotion_offset * POLICY_SIZE] = child

        node.is_expanded = True


# --- Helpers (module-level so they can be shared by the search + tests) ---


def _select_child(node: MCTSNode, invert_turns: "str | None" = None) -> MCTSNode:
    # AlphaZero PUCT sign convention: `child.total_value` is stored in the
    # CHILD's perspective (_backpropagate flips sign as it walks up). At
    # select time we view it from the PARENT's perspective, so negate:
    # a child whose Q is -1 (child is losing) is a GREAT move for the
    # parent and should score +1 here.
    #
    # Historical bug note: this used to read `q = child.total_value / …`
    # (no negation), which inverted PUCT — MCTS systematically avoided
    # moves leading to terminal mates, because a mating move produced a
    # child node with total_value=-1 (from the child's perspective), and
    # without negation that -1 sank the move's score. The symptom was a
    # visit fraction of exactly 1/N on every mate-in-1 position across
    # tens of thousands of training gens, with the model never finding
    # checkmate despite any amount of further training.
    best_score = -math.inf
    best_child: MCTSNode | None = None
    sqrt_parent = math.sqrt(max(node.visit_count, 1))
    # Misère inversion (see MCTSSearch.__init__): at an inverted node the
    # mover WANTS to lose, so a child's Q is read as-is (the child's own
    # perspective IS the opponent's winning chances) instead of negated.
    inverted = invert_turns is not None and (
        invert_turns == "both" or node.state.currentTurn == invert_turns
    )
    # FPU (First-Play Urgency): unvisited children start at parent_Q -
    # FPU_REDUCTION in parent-perspective, matching the Rust search
    # (mcts.rs select_child). Previously this path hard-coded q=0 for
    # unvisited children, silently diverging from Rust whenever
    # CHESS_AI_PYTHON_MCTS=1 was set. Under inversion, a child's expected
    # Q is ~ -parent_q, so the FPU baseline negates too.
    parent_q = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
    q_unvisited = (-parent_q if inverted else parent_q) - (FPU_REDUCTION if node.parent else 0.0)
    sign = 1.0 if inverted else -1.0
    for child in node.children.values():
        q = sign * child.total_value / child.visit_count if child.visit_count > 0 else q_unvisited
        u = C_PUCT * child.prior * sqrt_parent / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def _backpropagate(node: MCTSNode, value: float) -> None:
    current: MCTSNode | None = node
    v = value
    while current is not None:
        current.visit_count += 1
        current.total_value += v
        v = -v
        current = current.parent


def _sample_move(root: MCTSNode, rng: random.Random, temperature: float = 1.0) -> Move:
    """Pick a move from the root's visit distribution.

    Sampling weight is visit_count ** (1 / temperature):

      temperature <= ~0  → argmax (greedy; pick the most-visited child)
      temperature == 1.0 → proportional to visit counts (exploration)
      temperature  > 1.0 → flatter than proportional (sloppier)

    AlphaZero uses τ=1 for the opening plies and τ→0 thereafter so the game
    commits to decisive best moves once the opening is committed. Without
    this annealing, self-play games keep sampling sub-optimal moves
    proportionally and the training signal stays mushy.

    """
    children = list(root.children.values())
    if not children:
        raise RuntimeError("_sample_move called on a root with no children")

    if temperature <= 1e-6:
        best_child = max(children, key=lambda c: c.visit_count)
        assert best_child.move is not None
        return best_child.move

    total_visits = sum(c.visit_count for c in children)
    if total_visits == 0:
        # No sims completed yet on any child — fall back to a uniform pick.
        return rng.choice([c.move for c in children if c.move is not None])  # type: ignore[return-value]

    if abs(temperature - 1.0) < 1e-6:
        weights = [float(c.visit_count) for c in children]
        total = float(total_visits)
    else:
        power = 1.0 / temperature
        weights = [float(c.visit_count) ** power for c in children]
        total = sum(weights)
        if total <= 0.0:
            return rng.choice([c.move for c in children if c.move is not None])  # type: ignore[return-value]

    r = rng.random() * total
    for child, w in zip(children, weights):
        r -= w
        if r <= 0:
            assert child.move is not None
            return child.move

    # Float drift guard.
    best_child = max(children, key=lambda c: c.visit_count)
    assert best_child.move is not None
    return best_child.move


def _add_dirichlet_noise(root: MCTSNode, epsilon: float | None = None) -> None:
    eps = DIRICHLET_EPSILON if epsilon is None else epsilon
    n = len(root.children)
    if n == 0 or eps <= 0.0:
        return
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * n)
    for i, child in enumerate(root.children.values()):
        child.prior = (1 - eps) * child.prior + eps * float(noise[i])


# --- Batched MCTS across many games ---


def _soften_policy(policies: np.ndarray, temperature: float) -> np.ndarray:
    """Apply p**(1/T) and renormalize row-wise. T>1 flattens the prior."""
    softened = np.power(np.maximum(policies, 1e-12), 1.0 / temperature)
    row_sums = softened.sum(axis=-1, keepdims=True)
    return softened / np.maximum(row_sums, 1e-12)


try:
    import chess_ai_rust as _rust_mcts

    _HAVE_RUST_MCTS = hasattr(_rust_mcts, "MctsSearch")
except ImportError:
    _HAVE_RUST_MCTS = False

# Opt-in flag; defaults on when the Rust extension is present. Set
# `CHESS_AI_PYTHON_MCTS=1` to force the Python implementation for
# parity testing / debugging.
import os as _os
USE_RUST_MCTS = _HAVE_RUST_MCTS and not _os.environ.get("CHESS_AI_PYTHON_MCTS")


def _state_to_dict(state: ChessGameState) -> dict:
    """Serialize a ChessGameState into the plain-dict form Rust MctsSearch
    expects. Mirrors `ChessGameState.to_dict()` but avoids the method call
    overhead and keeps this module self-contained."""
    cr = state.castlingRights
    ep = state.enPassantTarget
    return {
        "board": [
            [None if p is None else {"color": p.color, "type": p.type} for p in row]
            for row in state.board
        ],
        "currentTurn": state.currentTurn,
        "castlingRights": {
            "whiteKingside": cr.whiteKingside,
            "whiteQueenside": cr.whiteQueenside,
            "blackKingside": cr.blackKingside,
            "blackQueenside": cr.blackQueenside,
        },
        "enPassantTarget": None if ep is None else {"rank": ep.rank, "file": ep.file},
        "halfMoveClock": state.halfMoveClock,
        "fullMoveNumber": state.fullMoveNumber,
        "status": state.status,
    }


def _tuple_to_move(tup: tuple[int, int, int, int, str | None]) -> Move:
    from_r, from_f, to_r, to_f, promo = tup
    return Move(
        from_pos=Position(rank=from_r, file=from_f),
        to_pos=Position(rank=to_r, file=to_f),
        promotion=promo,
    )


class RustMCTSSearch:
    """Adapter sharing the Python batching/dual-net driver with Rust trees."""

    def __init__(self, state, board_encoder=None, position_counts=None, invert_turns=None):
        from types import SimpleNamespace
        self.root = SimpleNamespace(state=state)
        self._search = _rust_mcts.MctsSearch(
            _state_to_dict(state), C_PUCT, FPU_REDUCTION, invert_turns,
            list(position_counts.items()) if position_counts is not None else None,
        )

    def is_terminal(self):
        return self._search.is_terminal()

    def get_root_board(self):
        return np.frombuffer(self._search.get_root_board(), dtype=np.float32)

    def init_root(self, policy, value, dirichlet_epsilon=None):
        self._search.init_root(
            np.asarray(policy, dtype=np.float32).tobytes(), float(value),
            DIRICHLET_EPSILON if dirichlet_epsilon is None else dirichlet_epsilon,
            DIRICHLET_ALPHA, self._rng.randrange(2**63),
        )

    def select_leaf(self):
        board = self._search.select_leaf()
        return None if board is None else np.frombuffer(board, dtype=np.float32)

    def pending_leaf_turn(self):
        return self._search.pending_leaf_turn()

    def supply_eval(self, policy, value):
        self._search.supply_eval(np.asarray(policy, dtype=np.float32).tobytes(), float(value))

    def get_result(self, rng=None, temperature=1.0):
        rng = rng or random
        policy, move, value = self._search.get_result(temperature, rng.randrange(2**63))
        return MCTSResult(np.frombuffer(policy, dtype=np.float32).copy(),
                          _tuple_to_move(move), value, self._search.best_child_q())


def run_batched_mcts(
    states: list[ChessGameState],
    evaluator: BatchedEvaluator,
    num_simulations: int,
    rng: random.Random | None = None,
    temperatures: list[float] | None = None,
    policy_softening_temperature: float = 1.0,
    dirichlet_epsilon: float | None = None,
    board_encoder: "Callable[[ChessGameState], np.ndarray] | None" = None,
    position_counts: "list[dict[bytes, int]] | None" = None,
    invert_turns: "list[str | None] | None" = None,
    opponent_evaluator: BatchedEvaluator | None = None,
    agent_colors: "list[str | None] | None" = None,
    opponent_evaluators: list[BatchedEvaluator | None] | None = None,
) -> list[MCTSResult]:
    """Run MCTS for each input state, batching all NN evaluations across games.

    Both backends support repetition, selection inversion, and dual-net
    routing. `invert_turns[i]` is None, "white", "black", or "both";
    values retain the ordinary-outcome sign convention in all cases.
    `opponent_evaluators[i]` optionally selects a different frozen opponent
    per game; `agent_colors[i]` controls which evaluator supplies a leaf.

    `temperatures[i]` controls the move-selection temperature for game `i`.
    Defaults to τ=1.0 for every game (AlphaZero-style exploration). Pass a
    list of zeros to get argmax (greedy) selection for all games.

    `policy_softening_temperature` flattens the priors fed to MCTS at the
    ROOT only: >1.0 lets low-prior root moves accumulate enough PUCT
    exploration to actually get visited. Leaf expansions keep the raw
    policy — softening every node (as an earlier version did) flattens
    Q estimates throughout the tree and turns the whole search into
    exploration noise at low sim counts. The training target (MCTS visit
    distribution) is unchanged; this only widens root search.

    `dirichlet_epsilon=None` uses the module-global set via
    `set_mcts_params` (self-play exploration). Pass 0.0 for eval/match
    play — gating matches must measure the model, not model+noise.
    """
    rng = rng or random
    if temperatures is None:
        temperatures = [1.0] * len(states)
    elif len(temperatures) != len(states):
        raise ValueError(
            f"temperatures length {len(temperatures)} != states length {len(states)}"
        )
    eps = DIRICHLET_EPSILON if dirichlet_epsilon is None else dirichlet_epsilon
    for name, lst in (("position_counts", position_counts),
                      ("invert_turns", invert_turns),
                      ("agent_colors", agent_colors),
                      ("opponent_evaluators", opponent_evaluators)):
        if lst is not None and len(lst) != len(states):
            raise ValueError(f"{name} length {len(lst)} != states length {len(states)}")

    # Never silently use an obsolete extension without variant semantics.
    use_rust = USE_RUST_MCTS and board_encoder is None
    if use_rust and not hasattr(_rust_mcts.MctsSearch, "pending_leaf_turn"):
        raise RuntimeError("Rebuild chess_ai_rust: maturin develop --release in rust_engine")
    search_class = RustMCTSSearch if use_rust else MCTSSearch

    soften = policy_softening_temperature != 1.0

    searches = [
        search_class(
            s,
            board_encoder,
            position_counts[i] if position_counts else None,
            invert_turns[i] if invert_turns else None,
        )
        for i, s in enumerate(states)
    ]

    for search in searches:
        if isinstance(search, RustMCTSSearch):
            search._rng = rng

    def _routed_eval(items):
        out = [None] * len(items)
        groups = {}
        for k, (gi, board, turn) in enumerate(items):
            ac = agent_colors[gi] if agent_colors else None
            opponent = opponent_evaluators[gi] if opponent_evaluators else opponent_evaluator
            ev = opponent if opponent is not None and ac is not None and turn != ac else evaluator
            groups.setdefault(ev, []).append((k, board))
        for ev, batch in groups.items():
            policies, values = ev(np.stack([b for _, b in batch]))
            for j, (k, _) in enumerate(batch):
                out[k] = (policies[j], float(values[j]))
        return out

    active = [(i, s) for i, s in enumerate(searches) if not s.is_terminal()]
    if not active:
        return [s.get_result(rng, temperatures[i]) for i, s in enumerate(searches)]

    # Batch-evaluate the root positions (routed: a frozen-opponent-to-move
    # root belongs to the opponent net).
    root_evals = _routed_eval([
        (i, s.get_root_board(), s.root.state.currentTurn) for i, s in active
    ])
    for k, (_i, s) in enumerate(active):
        policy, value = root_evals[k]
        if soften:
            policy = _soften_policy(policy[None, :], policy_softening_temperature)[0]
        s.init_root(policy, value, eps)

    # Simulation loop. Leaf priors are NOT softened — root-only.
    for _ in range(num_simulations):
        pending: list[tuple[int, MCTSSearch, np.ndarray]] = []
        for i, s in active:
            board = s.select_leaf()
            if board is not None:
                pending.append((i, s, board))

        if pending:
            evals = _routed_eval([
                (i, b, s.pending_leaf_turn() or "white") for i, s, b in pending
            ])
            for j, (_i, s, _b) in enumerate(pending):
                policy, value = evals[j]
                s.supply_eval(policy, value)

    return [s.get_result(rng, temperatures[i]) for i, s in enumerate(searches)]
