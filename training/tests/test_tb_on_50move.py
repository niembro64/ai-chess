"""Tests for tablebase adjudication on 50-move-rule draws.

The motivating bug: training runs were seeing 54% of all self-play games
end in 50-move-rule draws (label value=0) even when the ending position
was a theoretical win per Syzygy. That starved the value head of
decisive signal and caused value-collapse.

The fix: `_finish_game` now probes the tablebase on `status == "draw"`
(our engine's 50-move-rule signal), not just on move-cap timeouts. These
tests pin that behavior using a mocked tablebase — they work even on
machines without the actual tables installed.
"""

from __future__ import annotations

import random
from unittest import mock

import numpy as np

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import CastlingRights, ChessGameState
from chess_ai.selfplay import (
    GameSlot,
    GameSlotExample,
    SelfPlayConfig,
    SelfPlayEngine,
)


def _null_evaluator(boards: np.ndarray):
    batch = boards.shape[0]
    return (
        np.full((batch, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32),
        np.zeros(batch, dtype=np.float32),
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


def _fifty_move_state(stm: str) -> ChessGameState:
    """A position where the engine reported 50-move-rule draw
    (status='draw'). The board contents don't matter for our test —
    `_finish_game` passes the state to `tablebase.probe_outcome`, which
    we mock."""
    return ChessGameState(
        board=[[None] * 8 for _ in range(8)],
        currentTurn=stm,  # type: ignore[arg-type]
        castlingRights=CastlingRights(False, False, False, False),
        enPassantTarget=None,
        halfMoveClock=100,
        fullMoveNumber=60,
        status="draw",
    )


def _slot_50move(stm: str) -> GameSlot:
    return GameSlot(
        state=_fifty_move_state(stm),
        examples=[
            GameSlotExample(
                board=np.zeros(8 * 8 * 20, dtype=np.float32),
                policy=np.zeros(POLICY_SIZE, dtype=np.float32),
                turn_color="white",
                position_score=0.0,
            ),
            GameSlotExample(
                board=np.zeros(8 * 8 * 20, dtype=np.float32),
                policy=np.zeros(POLICY_SIZE, dtype=np.float32),
                turn_color="black",
                position_score=0.0,
            ),
        ],
        move_count=50,
        move_cap=400,   # NOT hit — game ended via 50-move rule
        is_standard_start=True,
    )


def test_50move_draw_tb_win_for_white():
    """Tb says white wins; game ended via 50-move rule with white to
    move. Result should be `tb_w` with white_outcome=+1."""
    engine, captured = _make_engine()
    with mock.patch("chess_ai.selfplay.tablebase.probe_outcome", return_value=1):
        result = engine._finish_game(_slot_50move(stm="white"))
    assert result.outcome == "tb_w"
    assert result.white_outcome == 1.0
    assert result.tb_adjudicated is True
    # The white-turn example should carry +1, the black-turn example -1.
    assert captured[0].value > 0.99
    assert captured[1].value < -0.99


def test_50move_draw_tb_win_for_black():
    """Mirror: tb says side-to-move loses (black to move, so black loses
    = white wins). Verifies perspective conversion."""
    engine, captured = _make_engine()
    # Black to move, tb returns -1 (stm loses) → white wins.
    with mock.patch("chess_ai.selfplay.tablebase.probe_outcome", return_value=-1):
        result = engine._finish_game(_slot_50move(stm="black"))
    assert result.outcome == "tb_w"
    assert result.white_outcome == 1.0


def test_50move_draw_tb_draw_becomes_tb_d():
    """Tb says the position is a theoretical draw. Outcome is `tb_d`
    (not `draw_50`) so the stats panel shows it in the tablebase
    section, and tb_adjudications is incremented."""
    engine, _ = _make_engine()
    with mock.patch("chess_ai.selfplay.tablebase.probe_outcome", return_value=0):
        result = engine._finish_game(_slot_50move(stm="white"))
    assert result.outcome == "tb_d"
    assert result.white_outcome == 0.0
    assert result.tb_adjudicated is True


def test_50move_draw_no_tb_signal_stays_draw_50():
    """Tb unavailable or out of range (returns None). Falls back to the
    natural `draw_50` label with value=0 (unchanged from before the
    refactor)."""
    engine, captured = _make_engine()
    with mock.patch("chess_ai.selfplay.tablebase.probe_outcome", return_value=None):
        result = engine._finish_game(_slot_50move(stm="white"))
    assert result.outcome == "draw_50"
    assert result.white_outcome == 0.0
    assert result.tb_adjudicated is False
    assert captured[0].value == 0.0
    assert captured[1].value == 0.0


def test_tb_adjudicated_samples_get_discounted_policy_weight():
    """TB-adjudicated games should propagate the configured discount
    (default 0.5) to every TrainingExample's policy_weight. Non-TB games
    (e.g. natural 50-move draw when the tb is unavailable) keep 1.0 so
    their MCTS visit distributions train the policy head at full weight.
    """
    captured: list = []
    engine = SelfPlayEngine(
        evaluator=_null_evaluator,
        example_sink=captured.append,
        config=SelfPlayConfig(
            num_concurrent_games=1, mcts_simulations=1,
            tb_policy_weight=0.3,  # non-default so we can see it land
        ),
        rng=random.Random(0),
    )
    # Case 1: TB rescued the outcome → both examples get 0.3
    with mock.patch("chess_ai.selfplay.tablebase.probe_outcome", return_value=1):
        engine._finish_game(_slot_50move(stm="white"))
    assert len(captured) == 2
    assert captured[0].policy_weight == 0.3
    assert captured[1].policy_weight == 0.3

    # Case 2: no TB signal → examples keep policy_weight=1.0
    captured.clear()
    with mock.patch("chess_ai.selfplay.tablebase.probe_outcome", return_value=None):
        engine._finish_game(_slot_50move(stm="white"))
    assert len(captured) == 2
    assert captured[0].policy_weight == 1.0
    assert captured[1].policy_weight == 1.0


def test_stalemate_is_not_overridden_by_tb():
    """Stalemate is a forced legal draw — the side to move genuinely has
    no moves. We don't tb-override it, because the game is ALREADY over
    with a correct result."""
    engine, _ = _make_engine()
    slot = _slot_50move(stm="white")
    slot.state.status = "stalemate"
    # Even if tb would say "win for white", we should not probe for stalemate.
    with mock.patch(
        "chess_ai.selfplay.tablebase.probe_outcome", return_value=1
    ) as probe:
        result = engine._finish_game(slot)
    assert result.outcome == "stalemate"
    assert result.white_outcome == 0.0
    probe.assert_not_called()
