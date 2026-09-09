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
