"""Rules-contract tests for Uncheck Chess v1."""

from chess_ai.engine import (
    CastlingRights,
    ChessGameState,
    Move,
    Piece,
    Position,
    apply_move,
    get_legal_moves,
)
from chess_ai.selfplay import GameSlot, GameSlotExample, SelfPlayConfig, SelfPlayEngine
from chess_ai.encoding import POLICY_SIZE
from chess_ai.uncheck import validate_curriculum
import numpy as np
import random


def _move(fr: str, to: str) -> Move:
    def pos(s: str) -> Position:
        return Position(8 - int(s[1]), ord(s[0]) - ord("a"))
    return Move(pos(fr), pos(to))


def _uci(move: Move) -> str:
    def square(pos: Position) -> str:
        return chr(ord("a") + pos.file) + str(8 - pos.rank)
    return square(move.from_pos) + square(move.to_pos)


def _canonical() -> ChessGameState:
    board = [[None] * 8 for _ in range(8)]
    board[0][0] = Piece("black", "rook")
    board[0][7] = Piece("black", "king")
    board[2][5] = Piece("black", "knight")
    board[5][3] = Piece("white", "king")
    board[6][0] = Piece("white", "pawn")
    return ChessGameState(
        board, "white", CastlingRights(False, False, False, False),
        None, 0, 1, "active", "uncheck-v1",
    )


def test_canonical_forced_rescue_failure():
    state = apply_move(_canonical(), _move("d3", "e4"))
    assert state.status == "active"
    assert [_uci(move) for move in get_legal_moves(state)] == ["a8a2"]
    terminal = apply_move(state, _move("a8", "a2"))
    assert terminal.status == "uncheck"
    assert terminal.currentTurn == "white"
    assert terminal.winner == "white"
    assert get_legal_moves(terminal) == []


def test_capture_is_compulsory_and_king_capture_is_forbidden():
    state = _canonical()
    state.currentTurn = "black"
    assert [_uci(move) for move in get_legal_moves(state)] == ["a8a2"]


def test_normal_chess_contract_remains_default():
    state = _canonical()
    state.ruleset = "normal"
    assert len(get_legal_moves(state)) > 1


def test_verified_curriculum_is_separate_and_sound():
    validate_curriculum()


def test_uncheck_actual_winner_uses_opposite_reference_labels():
    captured = []
    def evaluator(boards):
        return np.full((len(boards), POLICY_SIZE), 1 / POLICY_SIZE, np.float32), np.zeros(len(boards), np.float32)
    engine = SelfPlayEngine(evaluator, captured.append, SelfPlayConfig(
        num_concurrent_games=1, ruleset="uncheck-v1", invert_agent_selection=True,
    ), random.Random(0))
    terminal = _canonical()
    terminal.currentTurn = "white"
    terminal.status = "uncheck"
    terminal.winner = "white"
    examples = [
        GameSlotExample(np.zeros(8 * 8 * 20, np.float32), np.zeros(POLICY_SIZE, np.float32), color, 0)
        for color in ("white", "black")
    ]
    result = engine._finish_game(GameSlot(state=terminal, examples=examples, move_count=2))
    assert result.outcome == "uncheck_w"
    assert result.white_outcome == -1.0
    assert [example.value for example in captured] == [-1.0, 1.0]


def test_python_and_protocol3_rust_agree_on_uncheck_fixture():
    import pytest
    import chess_ai.engine as engine
    if not engine._HAVE_RUST or getattr(engine._rust, "ENGINE_PROTOCOL", 0) < 3:
        pytest.skip("protocol-3 Rust extension unavailable")
    state = _canonical()
    assert engine._get_legal_moves_python(state) == engine._rust_get_legal_moves(state)
    for move in engine._get_legal_moves_python(state):
        assert engine._apply_move_python(state, move).to_dict() == engine._rust_apply_move(state, move).to_dict()
