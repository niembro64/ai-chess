"""Syzygy tablebase adjudication for cap-timeout games.

When a self-play game hits the move cap with few enough pieces on the board,
we can consult a Syzygy tablebase to get the theoretically correct result
(win / draw / loss with optimal play). That's a much cleaner training signal
than scoring the cap as a bare 0.

Tablebases must be downloaded separately and their directory path passed to
`open_tablebase(path)`. If no path is configured — or the pieces on the
board exceed the tablebase's max piece count — `probe_outcome()` returns
None and callers should fall back to the default cap handling (score = 0).

Tablebase sources (one-time download):
  * 3-4-5 piece: ~1 GB — Lichess S3 mirror or tablebase.lichess.ovh
  * Up to 5 pieces is plenty for our endgame curriculum. 6-piece tables are
    ~150 GB and unnecessary at this scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

try:
    import chess
    import chess.syzygy
    _HAVE_PYTHON_CHESS = True
except ImportError:
    _HAVE_PYTHON_CHESS = False


_TB = None
_TB_MAX_PIECES = 0


_PIECE_TYPE_MAP = {
    "king": chess.KING if _HAVE_PYTHON_CHESS else None,
    "queen": chess.QUEEN if _HAVE_PYTHON_CHESS else None,
    "rook": chess.ROOK if _HAVE_PYTHON_CHESS else None,
    "bishop": chess.BISHOP if _HAVE_PYTHON_CHESS else None,
    "knight": chess.KNIGHT if _HAVE_PYTHON_CHESS else None,
    "pawn": chess.PAWN if _HAVE_PYTHON_CHESS else None,
}


def open_tablebase(directory: str | Path | None, max_pieces: int = 5) -> bool:
    """Configure the tablebase directory. Returns True on success.

    Safe to call with directory=None or a non-existent path — probing just
    stays disabled in that case.
    """
    global _TB, _TB_MAX_PIECES
    _TB = None
    _TB_MAX_PIECES = 0

    if directory is None or not _HAVE_PYTHON_CHESS:
        return False
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return False
    try:
        _TB = chess.syzygy.open_tablebase(str(path))
        _TB_MAX_PIECES = max_pieces
        return True
    except Exception:
        _TB = None
        return False


def is_enabled() -> bool:
    return _TB is not None


def _our_state_to_python_chess(state) -> "chess.Board | None":
    """Convert our ChessGameState to a python-chess Board. None if piece count
    exceeds the tablebase's max.
    """
    if _TB is None:
        return None

    board = chess.Board.empty()

    piece_count = 0
    for r in range(8):
        for f in range(8):
            piece = state.board[r][f]
            if piece is None:
                continue
            piece_count += 1
            if piece_count > _TB_MAX_PIECES:
                return None
            color = chess.WHITE if piece.color == "white" else chess.BLACK
            piece_type = _PIECE_TYPE_MAP[piece.type]
            # Our rank 0 = black's back rank = python-chess rank 7.
            square = chess.square(f, 7 - r)
            board.set_piece_at(square, chess.Piece(piece_type, color))

    board.turn = chess.WHITE if state.currentTurn == "white" else chess.BLACK
    board.castling_rights = chess.BB_EMPTY  # Our cap games usually have lost castling rights; be safe.
    board.ep_square = None
    return board


def probe_outcome(state) -> Literal[-1, 0, 1] | None:
    """Look up the theoretical result of `state` under optimal play.

    Returns +1 if the side to move wins, -1 if they lose, 0 for a draw.
    Returns None if the tablebase is unavailable, the piece count is too
    high, or the position can't be probed (e.g. castling rights would
    complicate the lookup).
    """
    if _TB is None:
        return None
    pc_board = _our_state_to_python_chess(state)
    if pc_board is None:
        return None
    try:
        wdl = _TB.probe_wdl(pc_board)
    except Exception:
        return None
    # python-chess returns values in {-2, -1, 0, 1, 2}:
    #   2  = cursed win (win but 50-move rule may save)
    #   1  = win
    #   0  = draw
    #  -1  = blessed loss (loss but 50-move rule may save)
    #  -2  = loss
    if wdl >= 1:
        return 1
    if wdl <= -1:
        return -1
    return 0
