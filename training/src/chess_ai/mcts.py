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
from .engine import ChessGameState, Move, Position, apply_move, expand_children, get_legal_moves

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
    )

    def __init__(self, state: ChessGameState, parent: "MCTSNode | None" = None, move: Move | None = None):
        self.parent: MCTSNode | None = parent
        self.children: dict[int, MCTSNode] = {}
        self.move: Move | None = move
        self.state: ChessGameState = state
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0


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

    def __init__(self, state: ChessGameState):
        self.root = MCTSNode(state)
        self._pending_leaf: MCTSNode | None = None
        self._check_terminal(self.root)

    def is_terminal(self) -> bool:
        return self.root.is_terminal

    def get_root_board(self) -> np.ndarray:
        return encode_board(self.root.state)

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
        while node.is_expanded and not node.is_terminal:
            node = _select_child(node)

        if node.is_terminal:
            _backpropagate(node, node.terminal_value)
            self._pending_leaf = None
            return None

        self._pending_leaf = node
        return encode_board(node.state)

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
                policy[idx] = child.visit_count / total_visits
        move = _sample_move(self.root, rng, temperature)
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
        s = node.state.status
        if s in ("checkmate", "stalemate", "draw"):
            node.is_terminal = True
            node.is_expanded = True
            # Terminal value is from the perspective of the player-to-move at the
            # terminal node. Checkmate = they have no moves and are in check, so
            # they've LOST: value = -1. Stalemate/draw: 0.
            node.terminal_value = -1.0 if s == "checkmate" else 0.0
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
                    -1.0
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
                -1.0
                if is_in_check(node.state.board, node.state.currentTurn)
                else 0.0
            )
            return

        is_white = node.state.currentTurn == "white"

        # Accumulate prior mass for just the moves we'll actually keep (one
        # MCTS child per unique policy-index — underpromotions collapse).
        seen: set[int] = set()
        prior_sum = 0.0
        for move, _child_state in children:
            mi = move_to_index(move, is_white)
            if mi not in seen:
                seen.add(mi)
                prior_sum += float(policy[mi])

        seen.clear()
        for move, child_state in children:
            mi = move_to_index(move, is_white)
            if mi in seen:
                continue
            seen.add(mi)
            child = MCTSNode(child_state, parent=node, move=move)
            child.prior = float(policy[mi]) / prior_sum if prior_sum > 0 else 1.0 / len(seen)
            self._check_terminal(child)
            node.children[mi] = child

        node.is_expanded = True


# --- Helpers (module-level so they can be shared by the search + tests) ---


def _select_child(node: MCTSNode) -> MCTSNode:
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
    # FPU (First-Play Urgency): unvisited children start at parent_Q -
    # FPU_REDUCTION in parent-perspective, matching the Rust search
    # (mcts.rs select_child). Previously this path hard-coded q=0 for
    # unvisited children, silently diverging from Rust whenever
    # CHESS_AI_PYTHON_MCTS=1 was set.
    parent_q = node.total_value / node.visit_count if node.visit_count > 0 else 0.0
    q_unvisited = parent_q - FPU_REDUCTION
    for child in node.children.values():
        q = -child.total_value / child.visit_count if child.visit_count > 0 else q_unvisited
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

    temperature == 1.0 → sample proportional to visit counts (exploration).
    temperature <= ~0  → argmax (greedy; pick the most-visited child).

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

    r = rng.random() * total_visits
    for child in children:
        r -= child.visit_count
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


def _run_batched_mcts_rust(
    states: list[ChessGameState],
    evaluator: BatchedEvaluator,
    num_simulations: int,
    rng: random.Random,
    temperatures: list[float],
    policy_softening_temperature: float,
    dirichlet_epsilon: float,
) -> list[MCTSResult]:
    """Rust-backed MCTS loop. Same semantics as the Python path; ~20× faster
    per-sim because the tree + PUCT + backprop live in Rust and the board
    encoding is emitted directly from Rust without marshalling."""
    soften = policy_softening_temperature != 1.0
    searches = [
        _rust_mcts.MctsSearch(_state_to_dict(s), C_PUCT, FPU_REDUCTION)
        for s in states
    ]
    active_idx = [i for i, s in enumerate(searches) if not s.is_terminal()]

    # Terminal-at-construction games return a zero-policy sentinel with
    # a dummy move. The outer self-play loop shouldn't call into MCTS for
    # already-terminal states, but we mirror Python's behavior defensively.
    if not active_idx:
        results = []
        for i, s in enumerate(searches):
            seed = rng.randrange(2**63)
            policy_bytes, move_tup, rv = s.get_result(temperatures[i], seed)
            policy = np.frombuffer(policy_bytes, dtype=np.float32).copy()
            if sum(move_tup[:4]) == 0 and move_tup[4] is None:
                # No legal moves — return the existing (zero) policy but
                # synthesize a placeholder Move; caller is expected to
                # check the state's status, not use this move.
                move = Move(Position(0, 0), Position(0, 0), None)
            else:
                move = _tuple_to_move(move_tup)
            results.append(MCTSResult(policy=policy, move=move, root_value=rv))
        return results

    # Batch-evaluate the root positions.
    root_boards_bytes = [searches[i].get_root_board() for i in active_idx]
    root_boards = np.stack(
        [np.frombuffer(b, dtype=np.float32) for b in root_boards_bytes]
    )
    root_policies, root_values = evaluator(root_boards)
    if soften:
        root_policies = _soften_policy(root_policies, policy_softening_temperature)

    for k, gi in enumerate(active_idx):
        searches[gi].init_root(
            root_policies[k].tobytes(),
            float(root_values[k]),
            dirichlet_epsilon,
            DIRICHLET_ALPHA,
            rng.randrange(2**63),
        )

    # Simulation loop. Leaf priors are NOT softened — softening is
    # root-only (see run_batched_mcts docstring).
    for _ in range(num_simulations):
        pending: list[tuple[int, bytes]] = []
        for gi in active_idx:
            board = searches[gi].select_leaf()
            if board is not None:
                pending.append((gi, board))
        if pending:
            boards = np.stack(
                [np.frombuffer(b, dtype=np.float32) for _, b in pending]
            )
            policies, values = evaluator(boards)
            for j, (gi, _) in enumerate(pending):
                searches[gi].supply_eval(policies[j].tobytes(), float(values[j]))

    # Extract results.
    results: list[MCTSResult | None] = [None] * len(searches)
    for gi, s in enumerate(searches):
        seed = rng.randrange(2**63)
        policy_bytes, move_tup, rv = s.get_result(temperatures[gi], seed)
        policy = np.frombuffer(policy_bytes, dtype=np.float32).copy()
        # hasattr guard: an older compiled extension without best_child_q
        # degrades to the mean-Q statistic instead of crashing self-play.
        bq = s.best_child_q() if hasattr(s, "best_child_q") else rv
        if gi in active_idx:
            move = _tuple_to_move(move_tup)
        else:
            # Terminal; synthesize a placeholder (caller shouldn't use it).
            move = Move(Position(0, 0), Position(0, 0), None)
        results[gi] = MCTSResult(policy=policy, move=move, root_value=rv, best_q=bq)
    return results  # type: ignore[return-value]


def run_batched_mcts(
    states: list[ChessGameState],
    evaluator: BatchedEvaluator,
    num_simulations: int,
    rng: random.Random | None = None,
    temperatures: list[float] | None = None,
    policy_softening_temperature: float = 1.0,
    dirichlet_epsilon: float | None = None,
) -> list[MCTSResult]:
    """Run MCTS for each input state, batching all NN evaluations across games.

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

    if USE_RUST_MCTS:
        return _run_batched_mcts_rust(
            states,
            evaluator,
            num_simulations,
            rng,
            temperatures,
            policy_softening_temperature,
            eps,
        )

    soften = policy_softening_temperature != 1.0

    searches = [MCTSSearch(s) for s in states]

    active = [s for s in searches if not s.is_terminal()]
    if not active:
        return [s.get_result(rng, temperatures[i]) for i, s in enumerate(searches)]

    # Batch-evaluate the root positions.
    root_boards = np.stack([s.get_root_board() for s in active])
    root_policies, root_values = evaluator(root_boards)
    if soften:
        root_policies = _soften_policy(root_policies, policy_softening_temperature)
    for i, s in enumerate(active):
        s.init_root(root_policies[i], float(root_values[i]), eps)

    # Simulation loop. Leaf priors are NOT softened — root-only.
    for _ in range(num_simulations):
        pending: list[tuple[int, np.ndarray]] = []
        for i, s in enumerate(active):
            board = s.select_leaf()
            if board is not None:
                pending.append((i, board))

        if pending:
            boards = np.stack([b for _, b in pending])
            policies, values = evaluator(boards)
            for j, (idx, _) in enumerate(pending):
                active[idx].supply_eval(policies[j], float(values[j]))

    return [s.get_result(rng, temperatures[i]) for i, s in enumerate(searches)]
