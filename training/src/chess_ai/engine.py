"""Python port of `src/game/chess/ChessEngine.ts`.

Must match the TS implementation byte-for-byte on legal-move generation,
board encoding, and status detection. Any divergence corrupts training —
changes here MUST be mirrored in the TS engine and verified by the
parity test in `tests/test_parity.py`.

Board layout:
    rank 0 = row 8 (black's back rank)
    rank 7 = row 1 (white's back rank)
    file 0 = a, file 7 = h
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PieceColor = Literal["white", "black"]
PieceType = Literal["king", "queen", "rook", "bishop", "knight", "pawn"]
GameStatus = Literal["waiting", "active", "check", "checkmate", "stalemate", "draw"]


@dataclass(frozen=True)
class Piece:
    color: PieceColor
    type: PieceType

    def to_dict(self) -> dict:
        return {"color": self.color, "type": self.type}

    @classmethod
    def from_dict(cls, d: dict) -> "Piece":
        return cls(color=d["color"], type=d["type"])


@dataclass(frozen=True)
class Position:
    rank: int
    file: int

    def to_dict(self) -> dict:
        return {"rank": self.rank, "file": self.file}

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(rank=d["rank"], file=d["file"])


@dataclass(frozen=True)
class Move:
    from_pos: Position
    to_pos: Position
    promotion: PieceType | None = None

    def to_dict(self) -> dict:
        out: dict = {"from": self.from_pos.to_dict(), "to": self.to_pos.to_dict()}
        if self.promotion is not None:
            out["promotion"] = self.promotion
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Move":
        return cls(
            from_pos=Position.from_dict(d["from"]),
            to_pos=Position.from_dict(d["to"]),
            promotion=d.get("promotion"),
        )


@dataclass
class CastlingRights:
    whiteKingside: bool
    whiteQueenside: bool
    blackKingside: bool
    blackQueenside: bool

    def copy(self) -> "CastlingRights":
        return CastlingRights(
            whiteKingside=self.whiteKingside,
            whiteQueenside=self.whiteQueenside,
            blackKingside=self.blackKingside,
            blackQueenside=self.blackQueenside,
        )

    def to_dict(self) -> dict:
        return {
            "whiteKingside": self.whiteKingside,
            "whiteQueenside": self.whiteQueenside,
            "blackKingside": self.blackKingside,
            "blackQueenside": self.blackQueenside,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CastlingRights":
        return cls(
            whiteKingside=d["whiteKingside"],
            whiteQueenside=d["whiteQueenside"],
            blackKingside=d["blackKingside"],
            blackQueenside=d["blackQueenside"],
        )


Board = list[list[Piece | None]]


@dataclass
class ChessGameState:
    board: Board
    currentTurn: PieceColor
    castlingRights: CastlingRights
    enPassantTarget: Position | None
    halfMoveClock: int
    fullMoveNumber: int
    status: GameStatus

    def to_dict(self) -> dict:
        return {
            "board": [[(p.to_dict() if p else None) for p in row] for row in self.board],
            "currentTurn": self.currentTurn,
            "castlingRights": self.castlingRights.to_dict(),
            "enPassantTarget": self.enPassantTarget.to_dict() if self.enPassantTarget else None,
            "halfMoveClock": self.halfMoveClock,
            "fullMoveNumber": self.fullMoveNumber,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChessGameState":
        return cls(
            board=[[(Piece.from_dict(p) if p else None) for p in row] for row in d["board"]],
            currentTurn=d["currentTurn"],
            castlingRights=CastlingRights.from_dict(d["castlingRights"]),
            enPassantTarget=Position.from_dict(d["enPassantTarget"]) if d["enPassantTarget"] else None,
            halfMoveClock=d["halfMoveClock"],
            fullMoveNumber=d["fullMoveNumber"],
            status=d["status"],
        )

    def copy(self) -> "ChessGameState":
        return ChessGameState(
            board=[row[:] for row in self.board],
            currentTurn=self.currentTurn,
            castlingRights=self.castlingRights.copy(),
            enPassantTarget=self.enPassantTarget,
            halfMoveClock=self.halfMoveClock,
            fullMoveNumber=self.fullMoveNumber,
            status=self.status,
        )


# --- Setup ---


def _initial_board() -> Board:
    board: Board = [[None] * 8 for _ in range(8)]
    back_rank: list[PieceType] = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
    for f in range(8):
        board[0][f] = Piece("black", back_rank[f])
        board[1][f] = Piece("black", "pawn")
        board[7][f] = Piece("white", back_rank[f])
        board[6][f] = Piece("white", "pawn")
    return board


def create_initial_game_state() -> ChessGameState:
    return ChessGameState(
        board=_initial_board(),
        currentTurn="white",
        castlingRights=CastlingRights(True, True, True, True),
        enPassantTarget=None,
        halfMoveClock=0,
        fullMoveNumber=1,
        status="waiting",
    )


# --- Helpers ---


def _opposite(c: PieceColor) -> PieceColor:
    return "black" if c == "white" else "white"


def _in_bounds(r: int, f: int) -> bool:
    return 0 <= r < 8 and 0 <= f < 8


def _find_king(board: Board, color: PieceColor) -> Position:
    for r in range(8):
        for f in range(8):
            p = board[r][f]
            if p and p.color == color and p.type == "king":
                return Position(r, f)
    raise RuntimeError(f"King not found for {color}")


# --- Attack detection ---

_KNIGHT_OFFSETS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
_ROOK_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_QUEEN_DIRS = _ROOK_DIRS + _BISHOP_DIRS


def is_square_attacked_by(board: Board, pos: Position, by_color: PieceColor) -> bool:
    # Knight
    for dr, df in _KNIGHT_OFFSETS:
        tr, tf = pos.rank + dr, pos.file + df
        if _in_bounds(tr, tf):
            p = board[tr][tf]
            if p and p.color == by_color and p.type == "knight":
                return True

    # Pawn (from TS: "pawns attack upward from their perspective";
    # white pawns live at higher rank indices and move toward rank 0,
    # so a white pawn attacks squares at rank-1 with file±1 — which
    # means squares at higher rank attack lower-rank squares for white).
    pawn_dir = 1 if by_color == "white" else -1
    for df in (-1, 1):
        tr, tf = pos.rank + pawn_dir, pos.file + df
        if _in_bounds(tr, tf):
            p = board[tr][tf]
            if p and p.color == by_color and p.type == "pawn":
                return True

    # King
    for dr in range(-1, 2):
        for df in range(-1, 2):
            if dr == 0 and df == 0:
                continue
            tr, tf = pos.rank + dr, pos.file + df
            if _in_bounds(tr, tf):
                p = board[tr][tf]
                if p and p.color == by_color and p.type == "king":
                    return True

    # Rook / Queen
    for dr, df in _ROOK_DIRS:
        for i in range(1, 8):
            tr, tf = pos.rank + dr * i, pos.file + df * i
            if not _in_bounds(tr, tf):
                break
            p = board[tr][tf]
            if p:
                if p.color == by_color and p.type in ("rook", "queen"):
                    return True
                break

    # Bishop / Queen
    for dr, df in _BISHOP_DIRS:
        for i in range(1, 8):
            tr, tf = pos.rank + dr * i, pos.file + df * i
            if not _in_bounds(tr, tf):
                break
            p = board[tr][tf]
            if p:
                if p.color == by_color and p.type in ("bishop", "queen"):
                    return True
                break

    return False


def is_in_check(board: Board, color: PieceColor) -> bool:
    return is_square_attacked_by(board, _find_king(board, color), _opposite(color))


# --- Apply move to a board (mutates) ---


def _apply_move_to_board(board: Board, move: Move, castling: CastlingRights) -> tuple[Piece | None, bool, bool]:
    piece = board[move.from_pos.rank][move.from_pos.file]
    assert piece is not None
    captured = board[move.to_pos.rank][move.to_pos.file]
    is_en_passant = False
    is_castle = False

    # En passant capture
    if piece.type == "pawn" and move.to_pos.file != move.from_pos.file and captured is None:
        is_en_passant = True
        board[move.from_pos.rank][move.to_pos.file] = None

    # Castling
    if piece.type == "king" and abs(move.to_pos.file - move.from_pos.file) == 2:
        is_castle = True
        if move.to_pos.file == 6:
            board[move.from_pos.rank][5] = board[move.from_pos.rank][7]
            board[move.from_pos.rank][7] = None
        else:
            board[move.from_pos.rank][3] = board[move.from_pos.rank][0]
            board[move.from_pos.rank][0] = None

    # Move piece
    board[move.to_pos.rank][move.to_pos.file] = piece
    board[move.from_pos.rank][move.from_pos.file] = None

    # Promotion
    if move.promotion is not None:
        board[move.to_pos.rank][move.to_pos.file] = Piece(piece.color, move.promotion)

    # Castling rights updates
    if piece.type == "king":
        if piece.color == "white":
            castling.whiteKingside = False
            castling.whiteQueenside = False
        else:
            castling.blackKingside = False
            castling.blackQueenside = False
    if piece.type == "rook":
        if piece.color == "white":
            if move.from_pos.rank == 7 and move.from_pos.file == 7:
                castling.whiteKingside = False
            if move.from_pos.rank == 7 and move.from_pos.file == 0:
                castling.whiteQueenside = False
        else:
            if move.from_pos.rank == 0 and move.from_pos.file == 7:
                castling.blackKingside = False
            if move.from_pos.rank == 0 and move.from_pos.file == 0:
                castling.blackQueenside = False

    # Rook captured on its starting square
    if move.to_pos.rank == 0 and move.to_pos.file == 7:
        castling.blackKingside = False
    if move.to_pos.rank == 0 and move.to_pos.file == 0:
        castling.blackQueenside = False
    if move.to_pos.rank == 7 and move.to_pos.file == 7:
        castling.whiteKingside = False
    if move.to_pos.rank == 7 and move.to_pos.file == 0:
        castling.whiteQueenside = False

    return captured, is_en_passant, is_castle


# --- Pseudo-legal move generation ---


def _pseudo_legal_moves(
    board: Board,
    color: PieceColor,
    castling: CastlingRights,
    en_passant: Position | None,
) -> list[Move]:
    moves: list[Move] = []

    for rank in range(8):
        for file in range(8):
            piece = board[rank][file]
            if not piece or piece.color != color:
                continue
            from_pos = Position(rank, file)

            if piece.type == "pawn":
                direction = -1 if color == "white" else 1
                start_rank = 6 if color == "white" else 1
                promo_rank = 0 if color == "white" else 7

                # Forward one
                fr = rank + direction
                if _in_bounds(fr, file) and board[fr][file] is None:
                    if fr == promo_rank:
                        for promo in ("queen", "rook", "bishop", "knight"):
                            moves.append(Move(from_pos, Position(fr, file), promo))  # type: ignore[arg-type]
                    else:
                        moves.append(Move(from_pos, Position(fr, file)))

                    # Forward two from starting rank
                    if rank == start_rank:
                        fr2 = rank + direction * 2
                        if board[fr2][file] is None:
                            moves.append(Move(from_pos, Position(fr2, file)))

                # Captures (including en passant)
                for df in (-1, 1):
                    cr, cf = rank + direction, file + df
                    if not _in_bounds(cr, cf):
                        continue
                    target = board[cr][cf]
                    if target and target.color != color:
                        if cr == promo_rank:
                            for promo in ("queen", "rook", "bishop", "knight"):
                                moves.append(Move(from_pos, Position(cr, cf), promo))  # type: ignore[arg-type]
                        else:
                            moves.append(Move(from_pos, Position(cr, cf)))
                    if en_passant is not None and cr == en_passant.rank and cf == en_passant.file:
                        moves.append(Move(from_pos, Position(cr, cf)))

            elif piece.type == "knight":
                for dr, df in _KNIGHT_OFFSETS:
                    tr, tf = rank + dr, file + df
                    if not _in_bounds(tr, tf):
                        continue
                    target = board[tr][tf]
                    if not target or target.color != color:
                        moves.append(Move(from_pos, Position(tr, tf)))

            elif piece.type in ("bishop", "rook", "queen"):
                dirs = (
                    _BISHOP_DIRS if piece.type == "bishop"
                    else _ROOK_DIRS if piece.type == "rook"
                    else _QUEEN_DIRS
                )
                for dr, df in dirs:
                    for i in range(1, 8):
                        tr, tf = rank + dr * i, file + df * i
                        if not _in_bounds(tr, tf):
                            break
                        target = board[tr][tf]
                        if not target:
                            moves.append(Move(from_pos, Position(tr, tf)))
                        else:
                            if target.color != color:
                                moves.append(Move(from_pos, Position(tr, tf)))
                            break

            elif piece.type == "king":
                for dr in range(-1, 2):
                    for df in range(-1, 2):
                        if dr == 0 and df == 0:
                            continue
                        tr, tf = rank + dr, file + df
                        if not _in_bounds(tr, tf):
                            continue
                        target = board[tr][tf]
                        if not target or target.color != color:
                            moves.append(Move(from_pos, Position(tr, tf)))

                # Castling
                if color == "white" and rank == 7 and file == 4:
                    if (
                        castling.whiteKingside
                        and board[7][5] is None
                        and board[7][6] is None
                        and board[7][7] is not None
                        and board[7][7].type == "rook"
                        and board[7][7].color == "white"
                        and not is_square_attacked_by(board, Position(7, 4), "black")
                        and not is_square_attacked_by(board, Position(7, 5), "black")
                        and not is_square_attacked_by(board, Position(7, 6), "black")
                    ):
                        moves.append(Move(from_pos, Position(7, 6)))
                    if (
                        castling.whiteQueenside
                        and board[7][3] is None
                        and board[7][2] is None
                        and board[7][1] is None
                        and board[7][0] is not None
                        and board[7][0].type == "rook"
                        and board[7][0].color == "white"
                        and not is_square_attacked_by(board, Position(7, 4), "black")
                        and not is_square_attacked_by(board, Position(7, 3), "black")
                        and not is_square_attacked_by(board, Position(7, 2), "black")
                    ):
                        moves.append(Move(from_pos, Position(7, 2)))

                if color == "black" and rank == 0 and file == 4:
                    if (
                        castling.blackKingside
                        and board[0][5] is None
                        and board[0][6] is None
                        and board[0][7] is not None
                        and board[0][7].type == "rook"
                        and board[0][7].color == "black"
                        and not is_square_attacked_by(board, Position(0, 4), "white")
                        and not is_square_attacked_by(board, Position(0, 5), "white")
                        and not is_square_attacked_by(board, Position(0, 6), "white")
                    ):
                        moves.append(Move(from_pos, Position(0, 6)))
                    if (
                        castling.blackQueenside
                        and board[0][3] is None
                        and board[0][2] is None
                        and board[0][1] is None
                        and board[0][0] is not None
                        and board[0][0].type == "rook"
                        and board[0][0].color == "black"
                        and not is_square_attacked_by(board, Position(0, 4), "white")
                        and not is_square_attacked_by(board, Position(0, 3), "white")
                        and not is_square_attacked_by(board, Position(0, 2), "white")
                    ):
                        moves.append(Move(from_pos, Position(0, 2)))

    return moves


def _clone_board(board: Board) -> Board:
    return [row[:] for row in board]


def get_legal_moves(state: ChessGameState) -> list[Move]:
    pseudo = _pseudo_legal_moves(state.board, state.currentTurn, state.castlingRights, state.enPassantTarget)

    legal: list[Move] = []
    for move in pseudo:
        test_board = _clone_board(state.board)
        test_castling = state.castlingRights.copy()
        _apply_move_to_board(test_board, move, test_castling)
        if not is_in_check(test_board, state.currentTurn):
            legal.append(move)
    return legal


def apply_move(state: ChessGameState, move: Move) -> ChessGameState:
    new_state = state.copy()
    piece = new_state.board[move.from_pos.rank][move.from_pos.file]
    if piece is None:
        raise ValueError("No piece at source square")

    captured, is_en_passant, _ = _apply_move_to_board(new_state.board, move, new_state.castlingRights)

    # En passant target
    if piece.type == "pawn" and abs(move.to_pos.rank - move.from_pos.rank) == 2:
        new_state.enPassantTarget = Position(
            (move.from_pos.rank + move.to_pos.rank) // 2,
            move.from_pos.file,
        )
    else:
        new_state.enPassantTarget = None

    # Half-move clock
    if piece.type == "pawn" or captured is not None or is_en_passant:
        new_state.halfMoveClock = 0
    else:
        new_state.halfMoveClock += 1

    # Full move number
    if state.currentTurn == "black":
        new_state.fullMoveNumber += 1

    new_state.currentTurn = _opposite(state.currentTurn)

    # Update status
    next_legal = get_legal_moves(new_state)
    in_check = is_in_check(new_state.board, new_state.currentTurn)
    if len(next_legal) == 0:
        if in_check:
            new_state.status = "checkmate"
        else:
            new_state.status = "stalemate"
    elif in_check:
        new_state.status = "check"
    elif new_state.halfMoveClock >= 100:
        new_state.status = "draw"
    else:
        new_state.status = "active"

    return new_state
