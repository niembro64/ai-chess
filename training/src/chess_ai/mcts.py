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
from .engine import ChessGameState, Move, apply_move, expand_children, get_legal_moves

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


def set_mcts_params(
    c_puct: float | None = None,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float | None = None,
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
    """
    global C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON
    if c_puct is not None:
        C_PUCT = c_puct
    if dirichlet_alpha is not None:
        DIRICHLET_ALPHA = dirichlet_alpha
    if dirichlet_epsilon is not None:
        DIRICHLET_EPSILON = dirichlet_epsilon

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
    root_value: float        # Root Q (for logging)


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

    def init_root(self, policy: np.ndarray, value: float) -> None:
        """Populate the root's children with priors from the NN, add Dirichlet noise."""
        self._expand_with_policy(self.root, policy)
        _add_dirichlet_noise(self.root)
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
        return MCTSResult(policy=policy, move=move, root_value=root_value)

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
            # the safe (but expensive) check.
            moves = get_legal_moves(node.state)
            if not moves:
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = 0.0

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray) -> None:
        # `expand_children` bundles get_legal_moves + one apply_move per child
        # into a single Rust call (or a Python fallback loop). Replacing the
        # old "loop-of-apply_move" pattern is the main FFI-cost reduction in
        # the MCTS hot path — ~30 Python/Rust crossings per expansion → 1.
        children = expand_children(node.state)
        if not children:
            node.is_terminal = True
            node.is_expanded = True
            node.terminal_value = 0.0
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
    for child in node.children.values():
        q = -child.total_value / child.visit_count if child.visit_count > 0 else 0.0
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


def _add_dirichlet_noise(root: MCTSNode) -> None:
    n = len(root.children)
    if n == 0:
        return
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * n)
    for i, child in enumerate(root.children.values()):
        child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * float(noise[i])


# --- Batched MCTS across many games ---


def _soften_policy(policies: np.ndarray, temperature: float) -> np.ndarray:
    """Apply p**(1/T) and renormalize row-wise. T>1 flattens the prior."""
    softened = np.power(np.maximum(policies, 1e-12), 1.0 / temperature)
    row_sums = softened.sum(axis=-1, keepdims=True)
    return softened / np.maximum(row_sums, 1e-12)


def run_batched_mcts(
    states: list[ChessGameState],
    evaluator: BatchedEvaluator,
    num_simulations: int,
    rng: random.Random | None = None,
    temperatures: list[float] | None = None,
    policy_softening_temperature: float = 1.0,
) -> list[MCTSResult]:
    """Run MCTS for each input state, batching all NN evaluations across games.

    `temperatures[i]` controls the move-selection temperature for game `i`.
    Defaults to τ=1.0 for every game (AlphaZero-style exploration). Pass a
    list of zeros to get argmax (greedy) selection for all games.

    `policy_softening_temperature` flattens the priors fed to MCTS: >1.0
    lets low-prior moves accumulate enough PUCT exploration to actually
    get visited. Applied to every NN policy read (root + leaf expansions)
    before priors are set. The training target (MCTS visit distribution)
    is unchanged; this only widens search. Use at self-play time when a
    collapsed policy head is overconfident on wrong moves and preventing
    MCTS from escaping — the trained-policy sharpness at eval time is
    preserved (eval callers leave this at 1.0).
    """
    rng = rng or random
    if temperatures is None:
        temperatures = [1.0] * len(states)
    elif len(temperatures) != len(states):
        raise ValueError(
            f"temperatures length {len(temperatures)} != states length {len(states)}"
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
        s.init_root(root_policies[i], float(root_values[i]))

    # Simulation loop.
    for _ in range(num_simulations):
        pending: list[tuple[int, np.ndarray]] = []
        for i, s in enumerate(active):
            board = s.select_leaf()
            if board is not None:
                pending.append((i, board))

        if pending:
            boards = np.stack([b for _, b in pending])
            policies, values = evaluator(boards)
            if soften:
                policies = _soften_policy(policies, policy_softening_temperature)
            for j, (idx, _) in enumerate(pending):
                active[idx].supply_eval(policies[j], float(values[j]))

    return [s.get_result(rng, temperatures[i]) for i, s in enumerate(searches)]
