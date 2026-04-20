"""Regression test for the AlphaZero MCTS sign convention.

The bug this guards against: `_select_child` used to read
    q = child.total_value / child.visit_count
when it should have been
    q = -child.total_value / child.visit_count

Without the negation, a mating move's child node — which correctly
stored terminal_value = -1.0 (the side-to-move at the checkmate leaf
has LOST, from their perspective) — contributed -1.0 to the PUCT
selection score at the parent. That made the mating move LESS
attractive than unvisited alternatives, so MCTS systematically avoided
mates: every mate-in-1 position in the eval set saw exactly 1/N visits
on the mating move across every search, and the model's 0-120-0 eval
scores went unexplained for tens of thousands of gradient steps.

With the correct sign, a child whose value is -1 contributes +1 at the
parent (the parent WINS when the child loses), and PUCT overwhelmingly
prefers the mating move after a single backup.

The test below uses a uniform-policy / zero-value stub evaluator — so
the NN itself contributes no signal. If MCTS still finds mate-in-1
from hand-crafted positions, it's because terminal-state backup is
working and sign-conventions agree. If it doesn't find any, the bug
is back.
"""

from __future__ import annotations

import random

import numpy as np

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import apply_move, get_legal_moves
from chess_ai.eval_positions import build_eval_positions
from chess_ai.mcts import run_batched_mcts


def _uniform_policy_evaluator(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stub NN: uniform policy prior, zero value. No chess knowledge."""
    batch = boards.shape[0]
    return (
        np.full((batch, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32),
        np.zeros(batch, dtype=np.float32),
    )


def _move_key(m) -> tuple[int, int, int, int]:
    return (m.from_pos.rank, m.from_pos.file, m.to_pos.rank, m.to_pos.file)


def test_mcts_finds_mate_in_1_with_uniform_policy():
    """Hand-crafted mate-in-1 positions should be findable by MCTS
    using only the terminal-state signal (no NN knowledge). 100 sims is
    plenty for a position with ~20 legal moves and one or more mating
    replies."""
    hand_positions = [
        p for p in build_eval_positions()
        if p.difficulty == "mate-in-1" and "random" not in p.name
    ]
    assert len(hand_positions) == 5, "Expected 5 hand-crafted mate-in-1 positions"

    rng = random.Random(42)
    found = 0
    failed_positions: list[str] = []
    for p in hand_positions:
        mating_keys = {
            _move_key(m) for m in get_legal_moves(p.state)
            if apply_move(p.state, m).status == "checkmate"
        }
        assert mating_keys, f"{p.name!r} has no mating moves — test fixture broken"

        result = run_batched_mcts(
            [p.state], _uniform_policy_evaluator, 100, rng, temperatures=[0.0]
        )[0]
        if _move_key(result.move) in mating_keys:
            found += 1
        else:
            failed_positions.append(p.name)

    # All 5 must succeed. Pre-fix this went 0/5 (or very close).
    assert found == 5, (
        f"MCTS with uniform policy found mate in only {found}/5 hand-crafted "
        f"positions. Failed: {failed_positions}. This is the _select_child "
        f"sign-flip bug — see the module docstring for details."
    )


def test_mcts_visits_concentrate_on_mating_move():
    """Beyond 'selects the mating move', we also expect MCTS to put the
    MAJORITY of its visits on the mate once it's found, because the
    terminal +1 value makes it dominate PUCT in subsequent sims."""
    p = next(
        p for p in build_eval_positions()
        if p.difficulty == "mate-in-1" and "back rank" in p.name
    )
    mating_keys = {
        _move_key(m) for m in get_legal_moves(p.state)
        if apply_move(p.state, m).status == "checkmate"
    }

    rng = random.Random(0)
    result = run_batched_mcts(
        [p.state], _uniform_policy_evaluator, 100, rng, temperatures=[0.0]
    )[0]

    # `result.policy` is the visit-count-normalized distribution over
    # the full 4096-move policy space. Sum the visit fraction on the
    # mating moves.
    from chess_ai.encoding import move_to_index
    is_white = p.state.currentTurn == "white"
    mating_indices = [
        move_to_index(m, is_white) for m in get_legal_moves(p.state)
        if apply_move(p.state, m).status == "checkmate"
    ]
    mate_visit_frac = float(sum(result.policy[i] for i in mating_indices))

    # With the bug, this was 1/100 = 0.01. With the fix, the mating
    # move should attract most of the budget once its Q settles at +1.
    assert mate_visit_frac > 0.5, (
        f"Mating move visit fraction = {mate_visit_frac:.3f}; expected > 0.5. "
        f"Sign-flip bug sets this to ~0.01."
    )
