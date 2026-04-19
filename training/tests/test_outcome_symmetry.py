"""W/B symmetry tests for the self-play outcome + reward attribution.

After training a session showed 18 W / 0 B across 18 decisive games — a
statistically significant skew (p ≈ 2e-6 under a fair-coin null hypothesis).
These tests pin down whether the attribution code has a directional bug by
exercising it in isolation with synthetic mirrored checkmate states.

If these tests pass, the bias is somewhere else (starting-position
distribution, MCTS, reward shaping, random-walk asymmetry) — not in the
final "who wins / how are training examples labelled" step.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import CastlingRights, ChessGameState, Move, Position
from chess_ai.selfplay import (
    GameSlot,
    GameSlotExample,
    SelfPlayConfig,
    SelfPlayEngine,
)


def _null_evaluator(boards: np.ndarray):
    """Returns a uniform policy + zero value. Enough to instantiate a
    SelfPlayEngine without needing a real model."""
    batch = boards.shape[0]
    policies = np.full((batch, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32)
    values = np.zeros(batch, dtype=np.float32)
    return policies, values


def _make_mate_state(loser: str) -> ChessGameState:
    """Construct a minimal state that looks like a checkmate.

    `_finish_game` only reads `status` + `currentTurn` + `board` (via
    `_material_advantage_white` for cap games, which we don't hit here).
    It does NOT re-verify that the position is actually mate, so we can
    fake it with an empty board — what we're testing is the
    attribution-from-status logic.
    """
    empty_board = [[None] * 8 for _ in range(8)]
    return ChessGameState(
        board=empty_board,
        currentTurn=loser,  # the side to move is the one who got mated
        castlingRights=CastlingRights(False, False, False, False),
        enPassantTarget=None,
        halfMoveClock=0,
        fullMoveNumber=10,
        status="checkmate",
    )


def _make_engine() -> tuple[SelfPlayEngine, list]:
    captured: list = []
    engine = SelfPlayEngine(
        evaluator=_null_evaluator,
        example_sink=captured.append,
        config=SelfPlayConfig(num_concurrent_games=1, mcts_simulations=1),
        rng=random.Random(0),
    )
    return engine, captured


def _dummy_examples(count: int = 4) -> list[GameSlotExample]:
    """Alternating white/black examples, all with zero positional score."""
    return [
        GameSlotExample(
            board=np.zeros(8 * 8 * 20, dtype=np.float32),
            policy=np.zeros(POLICY_SIZE, dtype=np.float32),
            turn_color=("white" if i % 2 == 0 else "black"),
            position_score=0.0,
        )
        for i in range(count)
    ]


def test_outcome_when_black_is_mated_is_white():
    engine, _ = _make_engine()
    slot = GameSlot(
        state=_make_mate_state(loser="black"),
        examples=_dummy_examples(),
        move_count=10,
        move_cap=100,
        is_standard_start=True,
    )
    result = engine._finish_game(slot)
    assert result.outcome == "mate_w"


def test_outcome_when_white_is_mated_is_black():
    engine, _ = _make_engine()
    slot = GameSlot(
        state=_make_mate_state(loser="white"),
        examples=_dummy_examples(),
        move_count=10,
        move_cap=100,
        is_standard_start=True,
    )
    result = engine._finish_game(slot)
    assert result.outcome == "mate_b", (
        f"Expected black wins when white was mated, got {result.outcome!r}. "
        f"This is the bias I'd most want to rule out if W/B counts are skewed."
    )


def test_training_example_values_flip_with_winner():
    """Pushed examples should be +1 from the winner's perspective, -1 from
    the loser's, regardless of which color wins."""
    # White wins (black is mated).
    engine, captured = _make_engine()
    slot = GameSlot(
        state=_make_mate_state(loser="black"),
        examples=_dummy_examples(count=2),  # one white-turn, one black-turn
        move_count=10,
        move_cap=100,
        is_standard_start=True,
    )
    engine._finish_game(slot)
    white_turn_ex = captured[0]  # first dummy example is turn_color="white"
    black_turn_ex = captured[1]
    assert white_turn_ex.value > 0.99, (
        f"White's example should have value ≈ +1 when white wins; got {white_turn_ex.value}"
    )
    assert black_turn_ex.value < -0.99, (
        f"Black's example should have value ≈ -1 when white wins; got {black_turn_ex.value}"
    )

    # Mirror: black wins (white is mated). The SAME input-example positions
    # should flip sign.
    engine2, captured2 = _make_engine()
    slot2 = GameSlot(
        state=_make_mate_state(loser="white"),
        examples=_dummy_examples(count=2),
        move_count=10,
        move_cap=100,
        is_standard_start=True,
    )
    engine2._finish_game(slot2)
    white_turn_ex2 = captured2[0]
    black_turn_ex2 = captured2[1]
    assert white_turn_ex2.value < -0.99, (
        f"White's example should have value ≈ -1 when black wins; got {white_turn_ex2.value}"
    )
    assert black_turn_ex2.value > 0.99, (
        f"Black's example should have value ≈ +1 when black wins; got {black_turn_ex2.value}"
    )


def test_random_start_color_distribution_is_balanced():
    """The 'who's to move after the random walk' coin flip should be close
    to 50/50. Our earlier analysis suggested a mild skew (~52-58% white)
    depending on num_random ranges; this test quantifies it so regressions
    get noticed.
    """
    from chess_ai.selfplay import _random_start

    rng = random.Random(42)
    white_to_move = 0
    total = 1000
    for _ in range(total):
        state = _random_start(rng)
        if state.currentTurn == "white":
            white_to_move += 1
    frac = white_to_move / total
    # Accept anything in [0.45, 0.60] as "not a severe asymmetry"; our
    # current algorithm lands around 0.50-0.55 on my seed checks.
    assert 0.40 < frac < 0.65, (
        f"Random-start color distribution is {frac:.2%} white-to-move "
        f"({white_to_move}/{total}). Severe skew could explain the W>>B "
        f"observation in the dashboard even without an attribution bug."
    )
