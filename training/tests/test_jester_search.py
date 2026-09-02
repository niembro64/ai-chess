"""Misère ("Jester") MCTS selection — mcts.py invert_turns.

The value semantics stay truthful everywhere (value head = who is
winning; terminal mate = -1 for the mated side). Only PUCT selection
flips at inverted plies: the mover maximizes the opponent's Q. These
tests pin the three behaviors that make a good loser:

  1. flee your own mates (winning is failure),
  2. walk into forced mates against yourself,
  3. jester-vs-jester ("both") models an opponent who REFUSES to win,
     so an unforced self-mate line stops looking attractive.

Plus the dual-net routing used for jester-vs-frozen-Sage trees.
"""

from __future__ import annotations

import random

import numpy as np

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves
from chess_ai.eval_positions import build_eval_positions
from chess_ai.mcts import run_batched_mcts


def _uniform_evaluator(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = boards.shape[0]
    return (
        np.full((batch, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32),
        np.zeros(batch, dtype=np.float32),
    )


def _move_key(m) -> tuple[int, int, int, int]:
    return (m.from_pos.rank, m.from_pos.file, m.to_pos.rank, m.to_pos.file)


def _find_move(state, from_rf, to_rf):
    return next(
        m for m in get_legal_moves(state)
        if (m.from_pos.rank, m.from_pos.file) == from_rf
        and (m.to_pos.rank, m.to_pos.file) == to_rf
    )


def test_inverted_search_flees_mate_in_1():
    """From every hand-crafted mate-in-1 position, the inverted mover
    must NOT play the mate (winning is the worst outcome for a jester)."""
    hand_positions = [
        p for p in build_eval_positions()
        if p.difficulty == "mate-in-1" and "random" not in p.name
    ]
    assert len(hand_positions) == 5

    rng = random.Random(42)
    for p in hand_positions:
        mating_keys = {
            _move_key(m) for m in get_legal_moves(p.state)
            if apply_move(p.state, m).status == "checkmate"
        }
        # Skip the degenerate case where EVERY move mates (none here).
        assert len(mating_keys) < len(get_legal_moves(p.state))

        result = run_batched_mcts(
            [p.state], _uniform_evaluator, 100, rng,
            temperatures=[0.0],
            dirichlet_epsilon=0.0,
            invert_turns=[p.state.currentTurn],
        )[0]
        assert _move_key(result.move) not in mating_keys, (
            f"inverted search played the mate in {p.name!r}"
        )


def _fools_mate_minus_one():
    """Position after 1.f3 e5 — white to move; 2.g4?? allows Qh4#."""
    state = create_initial_game_state()
    state.status = "active"
    state = apply_move(state, _find_move(state, (6, 5), (5, 5)))   # f3
    state = apply_move(state, _find_move(state, (1, 4), (3, 4)))   # e5
    return state


def test_inverted_search_walks_into_fools_mate():
    """Jester (white) vs a winner: white's inverted plies + black's
    normal in-tree plies must discover that 2.g4 FORCES black's mate
    (black, modeled as a winner, gladly plays Qh4#)."""
    state = _fools_mate_minus_one()
    g4 = _find_move(state, (6, 6), (4, 6))

    result = run_batched_mcts(
        [state], _uniform_evaluator, 2000, random.Random(7),
        temperatures=[0.0],
        dirichlet_epsilon=0.0,
        invert_turns=["white"],
    )[0]
    assert _move_key(result.move) == _move_key(g4), (
        "jester-vs-winner search failed to walk into the fool's mate"
    )


def test_jester_vs_jester_opponent_refuses_to_win():
    """Same position, invert_turns='both': the in-tree opponent is also
    a jester and REFUSES to play Qh4#, so g4 no longer reads as a
    forced loss and must not dominate visits."""
    state = _fools_mate_minus_one()
    g4 = _find_move(state, (6, 6), (4, 6))

    result = run_batched_mcts(
        [state], _uniform_evaluator, 2000, random.Random(7),
        temperatures=[0.0],
        dirichlet_epsilon=0.0,
        invert_turns=["both"],
    )[0]
    g4_share = result.policy[_policy_index(g4, is_white=True)]
    assert g4_share < 0.5, (
        f"g4 dominated ({g4_share:.2f}) even though a jester opponent "
        f"never cooperates by delivering the mate"
    )


def _policy_index(move, is_white: bool) -> int:
    from chess_ai.encoding import move_to_index
    return move_to_index(move, is_white)


def test_dual_net_routing_by_leaf_turn():
    """agent_colors routes leaf evals: white-to-move leaves go to the
    agent net, black-to-move leaves to the frozen opponent net."""
    calls = {"agent": 0, "opp": 0}

    def agent_eval(boards):
        calls["agent"] += boards.shape[0]
        return _uniform_evaluator(boards)

    def opp_eval(boards):
        calls["opp"] += boards.shape[0]
        return _uniform_evaluator(boards)

    state = create_initial_game_state()
    state.status = "active"
    run_batched_mcts(
        [state], agent_eval, 40, random.Random(3),
        temperatures=[0.0],
        dirichlet_epsilon=0.0,
        invert_turns=["white"],
        opponent_evaluator=opp_eval,
        agent_colors=["white"],
    )
    # Root (white) + white-ply leaves hit the agent net; depth-1 black
    # replies hit the opponent net. Both must have been consulted.
    assert calls["agent"] > 0, "agent net never evaluated"
    assert calls["opp"] > 0, "opponent net never evaluated"


def test_mate_scores_prefer_the_nearest_mate():
    """Terminal checkmate scores shrink with depth, so an equally
    forced mate that arrives sooner outranks one further away. This is
    the 'prefer faster wins from search terminal handling' that
    config.py's value_ply_decay comment calls for — and being
    symmetric, it makes a loss-seeking search want to be mated ASAP."""
    from chess_ai.mcts import MATE_MIN_MAGNITUDE, mate_value

    assert mate_value(0) == 1.0
    assert mate_value(1) > mate_value(5) > mate_value(20)
    # Never decays past the floor, so a deep mate still dominates any
    # ordinary evaluation instead of fading into noise.
    assert mate_value(10_000) == MATE_MIN_MAGNITUDE
    assert mate_value(20) > 0.75


def test_inverted_search_takes_the_faster_self_mate():
    """Fool's mate is available in one move; the inverted search must
    still pick it rather than a slower route to the same loss."""
    state = _fools_mate_minus_one()
    g4 = _find_move(state, (6, 6), (4, 6))
    result = run_batched_mcts(
        [state], _uniform_evaluator, 2000, random.Random(11),
        temperatures=[0.0],
        dirichlet_epsilon=0.0,
        invert_turns=["white"],
    )[0]
    assert _move_key(result.move) == _move_key(g4)
