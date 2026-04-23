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
    """Encode board to a flat [8*8*NUM_PLANES] float32 array matching the TS encoder.

    Piece planes still require a Python loop over 64 squares (the board is a
    nested-list-of-Piece-dataclasses, so we need to resolve each cell). The
    constant planes (bias/counters/castling) are filled with numpy slice
    broadcasts, which skips 64 Python iterations per plane.

    A Rust implementation exists in rust_engine but benchmarks ~15% slower
    here because marshalling the Python board (nested dataclass list →
    list-of-dicts) into FFI dwarfs the scalar-loop saving. It'll become
    the fast path once MCTS owns state in Rust (no marshalling on the
    leaf-expansion boundary). Keeping the Rust function exported for
    that integration.
    """
    data = np.zeros((8, 8, NUM_PLANES), dtype=np.float32)
    is_white = state.currentTurn == "white"
    own_color = state.currentTurn
    board = state.board

    # Piece planes 0–11.
    if is_white:
        for r in range(8):
            row = board[r]
            for f in range(8):
                piece = row[f]
                if piece is None:
                    continue
                type_idx = _PIECE_TYPE_INDEX[piece.type]
                channel = type_idx if piece.color == own_color else 6 + type_idx
                data[r, f, channel] = 1.0
    else:
        for r in range(8):
            row = board[r]
            out_r = 7 - r
            for f in range(8):
                piece = row[f]
                if piece is None:
                    continue
                type_idx = _PIECE_TYPE_INDEX[piece.type]
                channel = type_idx if piece.color == own_color else 6 + type_idx
                data[out_r, 7 - f, channel] = 1.0

    # Channel 12: bias
    data[:, :, 12] = 1.0

    # Channel 13: halfMoveClock / 50
    hmc = min(1.0, state.halfMoveClock / 50.0)
    if hmc > 0:
        data[:, :, 13] = hmc

    # Channel 14: fullMoveNumber / 100
    fmn = min(1.0, state.fullMoveNumber / 100.0)
    if fmn > 0:
        data[:, :, 14] = fmn

    # Channels 15–18: castling rights
    cr = state.castlingRights
    if is_white:
        if cr.whiteKingside:
            data[:, :, 15] = 1.0
        if cr.whiteQueenside:
            data[:, :, 16] = 1.0
        if cr.blackKingside:
            data[:, :, 17] = 1.0
        if cr.blackQueenside:
            data[:, :, 18] = 1.0
    else:
        if cr.blackKingside:
            data[:, :, 15] = 1.0
        if cr.blackQueenside:
            data[:, :, 16] = 1.0
        if cr.whiteKingside:
            data[:, :, 17] = 1.0
        if cr.whiteQueenside:
            data[:, :, 18] = 1.0

    # Channel 19: en passant target
    ep = state.enPassantTarget
    if ep is not None:
        ep_r = ep.rank if is_white else 7 - ep.rank
        ep_f = ep.file if is_white else 7 - ep.file
        data[ep_r, ep_f, 19] = 1.0

    return data.reshape(-1)


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
