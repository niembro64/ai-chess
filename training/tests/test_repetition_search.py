"""In-tree repetition awareness (mcts.py).

Without game history, MCTS is blind to threefold repetition: a winning
side sees no downside to shuffling, and self-play collapses into
repetition draws (observed: ~90% of games at gen 6k of the first 10x128
toy run). With `position_counts`, any in-tree move that would complete
FIDE threefold — or that repeats any position on its own search path —
is scored as a terminal draw, so the search steers a winning position
away from shuffles and a losing one toward them.
"""

import numpy as np

from chess_ai.encoding import POLICY_SIZE, move_to_index
from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves, position_key
from chess_ai.mcts import MCTSSearch


def _uniform_policy() -> np.ndarray:
    return np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)


def _drive(search: MCTSSearch, sims: int, leaf_value: float) -> None:
    """Run sims with a constant evaluator (value from the leaf mover's
    perspective)."""
    for _ in range(sims):
        board = search.select_leaf()
        if board is not None:
            search.supply_eval(_uniform_policy(), leaf_value)


def test_child_completing_threefold_is_scored_as_draw():
    state = create_initial_game_state()
    state.status = "active"

    # Pretend the position after 1.e4 already occurred twice in the game:
    # playing e4 now would complete FIDE threefold.
    e4 = next(
        m for m in get_legal_moves(state)
        if m.from_pos.rank == 6 and m.from_pos.file == 4
        and m.to_pos.rank == 4 and m.to_pos.file == 4
    )
    repeat_key = position_key(apply_move(state, e4))

    search = MCTSSearch(state, position_counts={repeat_key: 2})
    search.init_root(_uniform_policy(), 0.0, dirichlet_epsilon=0.0)
    # Every non-draw leaf evaluates to -0.9 (leaf mover losing) → from
    # the root mover's view those moves are worth ~+0.9, while the
    # repetition child pins at exactly 0. The search must explore e4 at
    # least once, mark it a terminal draw, and then prefer siblings.
    _drive(search, 300, leaf_value=-0.9)

    e4_child = search.root.children[move_to_index(e4, True)]
    assert e4_child.visit_count > 0, "repetition child never explored"
    assert e4_child.is_terminal, "threefold-completing child must be terminal"
    assert e4_child.terminal_value == 0.0
    assert not e4_child.children, "terminal draw must not be expanded"

    best = max(search.root.children.values(), key=lambda c: c.visit_count)
    assert best is not e4_child, (
        "search still prefers the repetition move despite winning eval"
    )


def test_losing_side_seeks_the_repetition_draw():
    state = create_initial_game_state()
    state.status = "active"

    e4 = next(
        m for m in get_legal_moves(state)
        if m.from_pos.rank == 6 and m.from_pos.file == 4
        and m.to_pos.rank == 4 and m.to_pos.file == 4
    )
    repeat_key = position_key(apply_move(state, e4))

    search = MCTSSearch(state, position_counts={repeat_key: 2})
    search.init_root(_uniform_policy(), 0.0, dirichlet_epsilon=0.0)
    # Every non-draw leaf evaluates to +0.9 (leaf mover WINNING) → the
    # root mover is losing everywhere (~-0.9) except the forced draw at
    # 0. Now the draw is the BEST move and should dominate visits.
    _drive(search, 300, leaf_value=0.9)

    best = max(search.root.children.values(), key=lambda c: c.visit_count)
    assert best is search.root.children[move_to_index(e4, True)], (
        "a losing search should steer INTO the repetition draw"
    )


def test_no_position_counts_keeps_old_behavior():
    state = create_initial_game_state()
    state.status = "active"

    search = MCTSSearch(state)
    search.init_root(_uniform_policy(), 0.0, dirichlet_epsilon=0.0)
    _drive(search, 50, leaf_value=0.0)

    # Without history, nothing may be marked terminal at depth 1 from the
    # start position, and no keys are computed.
    for child in search.root.children.values():
        assert not child.is_terminal
        assert child.pos_key is None
