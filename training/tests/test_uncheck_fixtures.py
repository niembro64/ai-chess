"""Language-neutral Uncheck v1 rule fixtures, checked by Python and Rust."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import chess_ai.engine as engine
from chess_ai.engine import apply_move, get_legal_moves, is_in_check, position_key
from chess_ai.mcts import MCTSSearch
from chess_ai.uncheck import move_uci, state_from_fen


FIXTURES = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "uncheck-v1.json").read_text()
)


def _legal(state):
    return sorted(move_uci(move) for move in get_legal_moves(state))


@pytest.mark.parametrize("fixture", FIXTURES["legalCases"], ids=lambda x: x["id"])
def test_legal_move_fixtures(fixture):
    state = state_from_fen(fixture["fen"])
    legal = _legal(state)
    if "exact" in fixture:
        assert legal == fixture["exact"]
    for move in fixture.get("includes", ()):
        assert move in legal
    for move in fixture.get("excludes", ()):
        assert move not in legal


@pytest.mark.parametrize("fixture", FIXTURES["terminalRoots"], ids=lambda x: x["id"])
def test_terminal_root_fixtures(fixture):
    state = state_from_fen(fixture["fen"])
    assert is_in_check(state.board, state.currentTurn) == fixture.get("attacked", True)
    assert get_legal_moves(state) == []
    assert MCTSSearch(state).is_terminal()


@pytest.mark.parametrize("fixture", FIXTURES["moveCases"], ids=lambda x: x["id"])
def test_move_result_fixtures(fixture):
    state = state_from_fen(fixture["fen"])
    move = next((m for m in get_legal_moves(state) if move_uci(m) == fixture["move"]), None)
    assert move is not None
    result = apply_move(state, move)
    assert result.status == fixture["status"]
    if turn := fixture.get("turn"):
        assert result.currentTurn == turn
    if winner := fixture.get("winner"):
        assert result.winner == winner
    if attacked := fixture.get("attacked"):
        assert is_in_check(result.board, attacked)
    if empty := fixture.get("empty"):
        rank, file = 8 - int(empty[1]), ord(empty[0]) - ord("a")
        assert result.board[rank][file] is None


def test_uncheck_repetition_key_uses_only_effective_en_passant():
    unavailable = state_from_fen("7k/8/8/8/8/8/8/K7 w - d6 0 1")
    no_ep = unavailable.copy()
    no_ep.enPassantTarget = None
    assert position_key(unavailable) == position_key(no_ep)

    available = state_from_fen("7k/8/8/3pP3/8/8/8/K7 w - d6 0 2")
    no_ep = available.copy()
    no_ep.enPassantTarget = None
    assert position_key(available) != position_key(no_ep)


@pytest.mark.skipif(not engine._HAVE_RUST or getattr(engine._rust, "ENGINE_PROTOCOL", 0) < 3,
                    reason="protocol-3 Rust extension unavailable")
@pytest.mark.parametrize("fixture", FIXTURES["legalCases"], ids=lambda x: x["id"])
def test_rust_and_python_fixture_parity(fixture):
    state = state_from_fen(fixture["fen"])
    py_moves = engine._get_legal_moves_python(state)
    rust_moves = engine._rust_get_legal_moves(state)
    assert py_moves == rust_moves
    for move in py_moves:
        assert engine._apply_move_python(state, move).to_dict() == engine._rust_apply_move(state, move).to_dict()
