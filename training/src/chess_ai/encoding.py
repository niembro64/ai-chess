"""Python port of `encodeBoard` and `moveToIndex` from `src/game/ai/ChessNet.ts`.

Must emit byte-identical output to the TS `encodeBoard` — the parity test
in `tests/test_parity.py` enforces this.

Plane layout (channels-last, flattened [rank*8 + file, plane]):
     0- 5  own pieces (K Q R B N P)
     6-11  opp pieces
    12     bias (constant 1)
    13     halfMoveClock / 50  (clamped to 1)
    14     fullMoveNumber / 100 (clamped to 1)
    15-18  castling rights (own K-side, own Q-side, opp K-side, opp Q-side)
    19     en passant target
"""

from __future__ import annotations

import numpy as np

from .engine import ChessGameState, Move

NUM_PLANES = 20
POLICY_SIZE = 4096  # 64 × 64 (from × to)

_PIECE_TYPES = ("king", "queen", "rook", "bishop", "knight", "pawn")
_PIECE_TYPE_INDEX = {t: i for i, t in enumerate(_PIECE_TYPES)}


def encode_board(state: ChessGameState) -> np.ndarray:
    """Encode board to a flat [8*8*NUM_PLANES] float32 array matching the TS encoder."""
    data = np.zeros(8 * 8 * NUM_PLANES, dtype=np.float32)
    is_white = state.currentTurn == "white"

    for rank in range(8):
        for file in range(8):
            board_rank = rank if is_white else 7 - rank
            board_file = file if is_white else 7 - file
            piece = state.board[board_rank][board_file]
            if piece is not None:
                is_own = piece.color == state.currentTurn
                type_idx = _PIECE_TYPE_INDEX[piece.type]
                channel = type_idx if is_own else 6 + type_idx
                data[(rank * 8 + file) * NUM_PLANES + channel] = 1.0

    # Channel 12: bias
    for i in range(64):
        data[i * NUM_PLANES + 12] = 1.0

    # Channel 13: halfMoveClock / 50 (clamped)
    hmc = min(1.0, state.halfMoveClock / 50.0)
    if hmc > 0:
        for i in range(64):
            data[i * NUM_PLANES + 13] = hmc

    # Channel 14: fullMoveNumber / 100 (clamped)
    fmn = min(1.0, state.fullMoveNumber / 100.0)
    if fmn > 0:
        for i in range(64):
            data[i * NUM_PLANES + 14] = fmn

    # Channels 15-18: castling rights
    cr = state.castlingRights
    castling = [
        cr.whiteKingside if is_white else cr.blackKingside,
        cr.whiteQueenside if is_white else cr.blackQueenside,
        cr.blackKingside if is_white else cr.whiteKingside,
        cr.blackQueenside if is_white else cr.whiteQueenside,
    ]
    for c in range(4):
        if castling[c]:
            for i in range(64):
                data[i * NUM_PLANES + 15 + c] = 1.0

    # Channel 19: en passant
    ep = state.enPassantTarget
    if ep is not None:
        ep_rank = ep.rank if is_white else 7 - ep.rank
        ep_file = ep.file if is_white else 7 - ep.file
        data[(ep_rank * 8 + ep_file) * NUM_PLANES + 19] = 1.0

    return data


def move_to_index(move: Move, is_white: bool) -> int:
    """Map a Move to a policy index in [0, POLICY_SIZE).

    Mirrors the board flip used by `encode_board`: from black's perspective,
    rank/file are negated so that "own side" always occupies the bottom half.
    Underpromotions collide with the queen-promotion index (matches TS).
    """
    fr, ff = move.from_pos.rank, move.from_pos.file
    tr, tf = move.to_pos.rank, move.to_pos.file
    if not is_white:
        fr, ff = 7 - fr, 7 - ff
        tr, tf = 7 - tr, 7 - tf
    return (fr * 8 + ff) * 64 + (tr * 8 + tf)
