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

# Optional Rust acceleration. When the `chess_ai_rust` extension is built
# (see training/rust_engine/), `get_legal_moves` dispatches to it for a
# ~10-30× speedup on the MCTS hot path. Falls back silently to the pure
# Python implementation if the extension isn't installed.
try:
    import chess_ai_rust as _rust  # type: ignore[import-not-found]
    _HAVE_RUST = True
except ImportError:
    _rust = None  # type: ignore[assignment]
    _HAVE_RUST = False

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
    # Hot path: `_in_bounds` inlined as `0 <= tr < 8 and 0 <= tf < 8` (saves ~2M
    # function calls per self-play step in profile).
    r0 = pos.rank
    f0 = pos.file

    # Knight
    for dr, df in _KNIGHT_OFFSETS:
        tr = r0 + dr
        tf = f0 + df
        if 0 <= tr < 8 and 0 <= tf < 8:
            p = board[tr][tf]
            if p is not None and p.color == by_color and p.type == "knight":
                return True

    # Pawn. White pawns live at higher rank indices and attack toward rank 0,
    # so a square at rank r is pawn-attacked from rank r+pawn_dir with file±1.
    pawn_dir = 1 if by_color == "white" else -1
    tr = r0 + pawn_dir
    if 0 <= tr < 8:
        for df in (-1, 1):
            tf = f0 + df
            if 0 <= tf < 8:
                p = board[tr][tf]
                if p is not None and p.color == by_color and p.type == "pawn":
                    return True

    # King (adjacent squares)
    for dr in (-1, 0, 1):
        tr = r0 + dr
        if 0 <= tr < 8:
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                tf = f0 + df
                if 0 <= tf < 8:
                    p = board[tr][tf]
                    if p is not None and p.color == by_color and p.type == "king":
                        return True

    # Rook / Queen rays
    for dr, df in _ROOK_DIRS:
        tr = r0 + dr
        tf = f0 + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            p = board[tr][tf]
            if p is not None:
                if p.color == by_color and (p.type == "rook" or p.type == "queen"):
                    return True
                break
            tr += dr
            tf += df

    # Bishop / Queen rays
    for dr, df in _BISHOP_DIRS:
        tr = r0 + dr
        tf = f0 + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            p = board[tr][tf]
            if p is not None:
                if p.color == by_color and (p.type == "bishop" or p.type == "queen"):
                    return True
                break
            tr += dr
            tf += df

    return False


def _is_pos_attacked_by_rc(
    board: Board, r0: int, f0: int, by_color: PieceColor
) -> bool:
    """Same as is_square_attacked_by but takes raw (rank, file) — avoids
    constructing a Position object in get_legal_moves' hot loop."""
    for dr, df in _KNIGHT_OFFSETS:
        tr = r0 + dr
        tf = f0 + df
        if 0 <= tr < 8 and 0 <= tf < 8:
            p = board[tr][tf]
            if p is not None and p.color == by_color and p.type == "knight":
                return True

    pawn_dir = 1 if by_color == "white" else -1
    tr = r0 + pawn_dir
    if 0 <= tr < 8:
        for df in (-1, 1):
            tf = f0 + df
            if 0 <= tf < 8:
                p = board[tr][tf]
                if p is not None and p.color == by_color and p.type == "pawn":
                    return True

    for dr in (-1, 0, 1):
        tr = r0 + dr
        if 0 <= tr < 8:
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                tf = f0 + df
                if 0 <= tf < 8:
                    p = board[tr][tf]
                    if p is not None and p.color == by_color and p.type == "king":
                        return True

    for dr, df in _ROOK_DIRS:
        tr = r0 + dr
        tf = f0 + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            p = board[tr][tf]
            if p is not None:
                if p.color == by_color and (p.type == "rook" or p.type == "queen"):
                    return True
                break
            tr += dr
            tf += df

    for dr, df in _BISHOP_DIRS:
        tr = r0 + dr
        tf = f0 + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            p = board[tr][tf]
            if p is not None:
                if p.color == by_color and (p.type == "bishop" or p.type == "queen"):
                    return True
                break
            tr += dr
            tf += df

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


def _serialize_board_for_rust(board: Board) -> list[list[dict | None]]:
    """Convert our list-of-Piece board into the dict format the Rust FFI expects."""
    return [
        [({"color": p.color, "type": p.type} if p else None) for p in row]
        for row in board
    ]


def _rust_get_legal_moves(state: ChessGameState) -> list[Move]:
    """Dispatch get_legal_moves to the Rust extension.

    Packs the Python state into the dict/primitive formats the Rust FFI accepts,
    calls the Rust implementation, and unpacks the result into Python Move
    dataclasses. Parity with the pure-Python path is asserted by
    tests/test_rust_engine_parity.py on 5000 positions.
    """
    cr = state.castlingRights
    castling = {
        "whiteKingside": cr.whiteKingside,
        "whiteQueenside": cr.whiteQueenside,
        "blackKingside": cr.blackKingside,
        "blackQueenside": cr.blackQueenside,
    }
    ep = None if state.enPassantTarget is None else {
        "rank": state.enPassantTarget.rank,
        "file": state.enPassantTarget.file,
    }
    tuples = _rust.get_legal_moves(
        _serialize_board_for_rust(state.board),
        state.currentTurn,
        castling,
        ep,
    )
    return [
        Move(
            from_pos=Position(t[0], t[1]),
            to_pos=Position(t[2], t[3]),
            promotion=t[4],
        )
        for t in tuples
    ]


def get_legal_moves(state: ChessGameState) -> list[Move]:
    """Compute all legal moves for the side to move.

    Uses the Rust extension when it's built; otherwise falls back to the
    pure-Python implementation below.
    """
    if _HAVE_RUST:
        return _rust_get_legal_moves(state)
    return _get_legal_moves_python(state)


def _rust_expand_children(state: ChessGameState) -> list[tuple["Move", ChessGameState]]:
    """One-call bundled legal-move enumeration + apply_move for all children.

    The MCTS tree expansion used to make N (~30) separate round-trips to
    Rust — one `apply_move_full` per legal move — each paying board-packing
    and state-dict construction at the FFI boundary. `generate_children`
    replaces all of that with a single round-trip, which is the biggest
    single marshalling win available in the hot path.
    """
    cr = state.castlingRights
    castling = {
        "whiteKingside": cr.whiteKingside,
        "whiteQueenside": cr.whiteQueenside,
        "blackKingside": cr.blackKingside,
        "blackQueenside": cr.blackQueenside,
    }
    ep = None if state.enPassantTarget is None else {
        "rank": state.enPassantTarget.rank,
        "file": state.enPassantTarget.file,
    }
    raw_children = _rust.generate_children(
        _serialize_board_for_rust(state.board),
        state.currentTurn,
        castling,
        ep,
        state.halfMoveClock,
        state.fullMoveNumber,
    )

    result: list[tuple[Move, ChessGameState]] = []
    for d in raw_children:
        move = Move(
            from_pos=Position(d["move_from_r"], d["move_from_f"]),
            to_pos=Position(d["move_to_r"], d["move_to_f"]),
            promotion=d["move_promotion"],
        )
        board_flat: list[int] = d["board"]
        child_board: Board = [
            [_PIECE_BY_INT[board_flat[r * 8 + f]] for f in range(8)]
            for r in range(8)
        ]
        cr_d = d["castlingRights"]
        child_cr = CastlingRights(
            whiteKingside=cr_d["whiteKingside"],
            whiteQueenside=cr_d["whiteQueenside"],
            blackKingside=cr_d["blackKingside"],
            blackQueenside=cr_d["blackQueenside"],
        )
        ep_d = d["enPassantTarget"]
        child_ep = None if ep_d is None else Position(rank=ep_d["rank"], file=ep_d["file"])
        child_state = ChessGameState(
            board=child_board,
            currentTurn=d["currentTurn"],
            castlingRights=child_cr,
            enPassantTarget=child_ep,
            halfMoveClock=d["halfMoveClock"],
            fullMoveNumber=d["fullMoveNumber"],
            status=d["status"],
        )
        result.append((move, child_state))
    return result


def expand_children(state: ChessGameState) -> list[tuple["Move", ChessGameState]]:
    """Enumerate legal moves and compute all child states in one shot.

    Semantically equivalent to `[(m, apply_move(state, m)) for m in get_legal_moves(state)]`
    — just much cheaper at runtime because the Rust path crosses the FFI
    boundary once instead of once per child.
    """
    if _HAVE_RUST:
        return _rust_expand_children(state)
    moves = _get_legal_moves_python(state)
    return [(m, _apply_move_python(state, m)) for m in moves]


def _get_legal_moves_python(state: ChessGameState) -> list[Move]:
    """Pure-Python implementation of `get_legal_moves`.

    Previously we cloned the entire board for every pseudo-legal move (~30 per
    call). Now we save only the 2–4 squares that each pseudo-move touches,
    mutate in place, test, and undo. We also track the king's position
    explicitly so we skip 29 redundant 64-square `_find_king` scans per call.
    ~3–5× faster on typical positions.
    """
    pseudo = _pseudo_legal_moves(
        state.board, state.currentTurn, state.castlingRights, state.enPassantTarget
    )
    board = state.board
    castling = state.castlingRights
    current_turn = state.currentTurn
    opponent = "black" if current_turn == "white" else "white"

    # Snapshot castling rights once; restore after each trial.
    saved_wk = castling.whiteKingside
    saved_wq = castling.whiteQueenside
    saved_bk = castling.blackKingside
    saved_bq = castling.blackQueenside

    # Locate own king once. Track where it ends up after each pseudo-move so we
    # can call `_is_pos_attacked_by_rc` directly (no per-iteration king scan).
    king_pos = _find_king(board, current_turn)
    king_r0 = king_pos.rank
    king_f0 = king_pos.file

    legal: list[Move] = []
    for move in pseudo:
        from_r = move.from_pos.rank
        from_f = move.from_pos.file
        to_r = move.to_pos.rank
        to_f = move.to_pos.file

        sq_from = board[from_r][from_f]
        sq_to = board[to_r][to_f]

        # Classify the move to know which extra squares need saving.
        is_pawn = sq_from is not None and sq_from.type == "pawn"
        is_king_move = sq_from is not None and sq_from.type == "king"
        is_en_passant = is_pawn and from_f != to_f and sq_to is None
        is_castle = is_king_move and abs(to_f - from_f) == 2

        sq_ep = None
        ep_r = ep_f = 0
        sq_rook_from = sq_rook_to = None
        rook_from_f = rook_to_f = 0

        if is_en_passant:
            ep_r, ep_f = from_r, to_f
            sq_ep = board[ep_r][ep_f]
        if is_castle:
            rook_from_f = 7 if to_f == 6 else 0
            rook_to_f = 5 if to_f == 6 else 3
            sq_rook_from = board[from_r][rook_from_f]
            sq_rook_to = board[from_r][rook_to_f]

        # Apply the move in place (mutates board + castling).
        _apply_move_to_board(board, move, castling)

        # King position after the move: only changed if this WAS the king moving.
        if is_king_move:
            k_r, k_f = to_r, to_f
        else:
            k_r, k_f = king_r0, king_f0

        # King-safety test on the mutated board.
        if not _is_pos_attacked_by_rc(board, k_r, k_f, opponent):
            legal.append(move)

        # Restore the (at most 4) squares we touched.
        board[from_r][from_f] = sq_from
        board[to_r][to_f] = sq_to
        if is_en_passant:
            board[ep_r][ep_f] = sq_ep
        if is_castle:
            board[from_r][rook_from_f] = sq_rook_from
            board[from_r][rook_to_f] = sq_rook_to

        # Restore castling rights (apply_move_to_board may have flipped flags).
        castling.whiteKingside = saved_wk
        castling.whiteQueenside = saved_wq
        castling.blackKingside = saved_bk
        castling.blackQueenside = saved_bq

    return legal


# Precomputed immutable Piece instances keyed by Rust's signed-int piece code
# (see training/rust_engine/src/lib.rs: sign = colour, magnitude = type).
# Used by `_rust_apply_move` to turn the returned flat int board back into
# our nested Piece list without constructing fresh Piece dicts per call.
_PIECE_BY_INT: dict[int, Piece | None] = {0: None}
for _c_sign, _c_name in ((1, "white"), (-1, "black")):
    for _pt_idx, _pt_name in enumerate(("king", "queen", "rook", "bishop", "knight", "pawn"), start=1):
        _PIECE_BY_INT[_c_sign * _pt_idx] = Piece(color=_c_name, type=_pt_name)  # type: ignore[arg-type]


def _rust_apply_move(state: ChessGameState, move: Move) -> ChessGameState:
    """Dispatch apply_move to the Rust extension.

    Sends (board, currentTurn, castlingRights, enPassantTarget, halfMoveClock,
    fullMoveNumber, move coords) across the FFI; receives a dict with the
    post-move state. Parity with the pure-Python path is exercised by
    tests/test_rust_engine_parity.py (the parity fixture was generated by
    the TS engine and Python's apply_move is byte-identical to it, so Rust
    matching Python is the same as Rust matching TS).
    """
    cr = state.castlingRights
    castling = {
        "whiteKingside": cr.whiteKingside,
        "whiteQueenside": cr.whiteQueenside,
        "blackKingside": cr.blackKingside,
        "blackQueenside": cr.blackQueenside,
    }
    ep = None if state.enPassantTarget is None else {
        "rank": state.enPassantTarget.rank,
        "file": state.enPassantTarget.file,
    }
    result = _rust.apply_move_full(
        _serialize_board_for_rust(state.board),
        state.currentTurn,
        castling,
        ep,
        state.halfMoveClock,
        state.fullMoveNumber,
        move.from_pos.rank,
        move.from_pos.file,
        move.to_pos.rank,
        move.to_pos.file,
        move.promotion,
    )

    board_flat: list[int] = result["board"]
    new_board: Board = [
        [_PIECE_BY_INT[board_flat[r * 8 + f]] for f in range(8)]
        for r in range(8)
    ]
    cr_res = result["castlingRights"]
    new_castling = CastlingRights(
        whiteKingside=cr_res["whiteKingside"],
        whiteQueenside=cr_res["whiteQueenside"],
        blackKingside=cr_res["blackKingside"],
        blackQueenside=cr_res["blackQueenside"],
    )
    ep_dict = result["enPassantTarget"]
    new_ep = None if ep_dict is None else Position(rank=ep_dict["rank"], file=ep_dict["file"])

    return ChessGameState(
        board=new_board,
        currentTurn=result["currentTurn"],
        castlingRights=new_castling,
        enPassantTarget=new_ep,
        halfMoveClock=result["halfMoveClock"],
        fullMoveNumber=result["fullMoveNumber"],
        status=result["status"],
    )


def apply_move(state: ChessGameState, move: Move) -> ChessGameState:
    """Apply a move to a game state, returning a new state.

    Uses the Rust extension when it's built; falls back to pure Python
    otherwise. The hot path for MCTS tree expansion.
    """
    if _HAVE_RUST:
        return _rust_apply_move(state, move)
    return _apply_move_python(state, move)


def _apply_move_python(state: ChessGameState, move: Move) -> ChessGameState:
    """Pure-Python implementation of `apply_move` (fallback)."""
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


def position_key(state: ChessGameState) -> bytes:
    """Stable byte-string identifying a chess position for FIDE-style
    threefold-repetition comparison. Includes board layout, side to
    move, castling rights, and en-passant target. Excludes the halfmove
    clock and full-move number — FIDE 9.2 compares by position only.

    Lives in engine (not selfplay) so the MCTS — which selfplay imports —
    can use it for in-tree repetition awareness without an import cycle.
    """
    parts = bytearray(64 + 1 + 4 + 2)
    i = 0
    for r in range(8):
        for f in range(8):
            p = state.board[r][f]
            if p is None:
                parts[i] = ord('.')
            else:
                tmap = {"king": 'k', "queen": 'q', "rook": 'r',
                        "bishop": 'b', "knight": 'n', "pawn": 'p'}
                ch = tmap[p.type]
                if p.color == "white":
                    ch = ch.upper()
                parts[i] = ord(ch)
            i += 1
    parts[i] = ord(state.currentTurn[0])  # 'w' or 'b'
    i += 1
    cr = state.castlingRights
    for flag in (cr.whiteKingside, cr.whiteQueenside,
                 cr.blackKingside, cr.blackQueenside):
        parts[i] = ord('1' if flag else '0')
        i += 1
    if state.enPassantTarget is not None:
        parts[i] = ord('a') + state.enPassantTarget.file
        parts[i + 1] = ord('1') + state.enPassantTarget.rank
    else:
        parts[i] = ord('-')
        parts[i + 1] = ord('-')
    return bytes(parts)
