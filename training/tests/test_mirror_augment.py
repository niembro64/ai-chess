"""Semantic correctness of the file-mirror augmentation.

The encoder canonicalizes to the side-to-move perspective with a
180-degree rotation for black (rank AND file), so in *encoded* space the
file-mirror of a position is exactly the encoding of its color-flipped
counterpart: swap piece colors, mirror ranks, swap castling rights by
color (kingside stays kingside), flip the turn. mirror_batch must
therefore (a) leave the castling planes UNSWAPPED and (b) permute the
policy with the file-flip index map.

An earlier version swapped castling plane pairs [15,16] and [17,18],
which broke this equivalence on every sample with asymmetric castling
rights, making those input planes unlearnable. This test pins the
corrected behavior against the real encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chess_ai.encoding import NUM_PLANES, POLICY_SIZE, encode_board, move_to_index
from chess_ai.engine import ChessGameState, get_legal_moves
from chess_ai.selfplay import _get_mirror_policy_perm, mirror_batch

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity_positions.json"


def _color_flip_dict(d: dict) -> dict:
    """Color-flip a serialized state: swap piece colors, mirror ranks,
    swap castling rights white<->black, flip the turn. Move counters and
    status are carried over unchanged so the encodings stay comparable."""
    flipped_board = []
    for r in range(7, -1, -1):
        row = []
        for cell in d["board"][r]:
            if cell is None:
                row.append(None)
            else:
                row.append({
                    "color": "black" if cell["color"] == "white" else "white",
                    "type": cell["type"],
                })
        flipped_board.append(row)
    cr = d["castlingRights"]
    ep = d.get("enPassantTarget")
    return {
        "board": flipped_board,
        "currentTurn": "black" if d["currentTurn"] == "white" else "white",
        "castlingRights": {
            "whiteKingside": cr["blackKingside"],
            "whiteQueenside": cr["blackQueenside"],
            "blackKingside": cr["whiteKingside"],
            "blackQueenside": cr["whiteQueenside"],
        },
        "enPassantTarget": (
            {"rank": 7 - ep["rank"], "file": ep["file"]} if ep else None
        ),
        "halfMoveClock": d["halfMoveClock"],
        "fullMoveNumber": d["fullMoveNumber"],
        "status": d["status"],
    }


@pytest.fixture(scope="module")
def positions() -> list[dict]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture missing; run `npm run dump-parity`. Expected: {FIXTURE_PATH}")
    with FIXTURE_PATH.open() as f:
        data = json.load(f)
    # 200 positions is plenty; the fixture's random walk covers all
    # castling-rights combinations and both side-to-move cases early.
    return data["positions"][:200]


def test_board_mirror_equals_colorflip_encoding(positions: list[dict]):
    """mirror_batch(encode(P)) must equal encode(colorflip(P)) exactly."""
    with_rights = 0
    for pos in positions:
        state = ChessGameState.from_dict(pos["state"])
        flipped = ChessGameState.from_dict(_color_flip_dict(pos["state"]))

        cr = pos["state"]["castlingRights"]
        if any(cr.values()):
            with_rights += 1

        enc = encode_board(state)[None, :]
        dummy_policy = np.zeros((1, POLICY_SIZE), dtype=np.float32)
        mirrored, _ = mirror_batch(enc, dummy_policy, np.array([True]))

        expected = encode_board(flipped)[None, :]
        if not np.array_equal(mirrored, expected):
            m = mirrored.reshape(8, 8, NUM_PLANES)
            e = expected.reshape(8, 8, NUM_PLANES)
            bad_planes = sorted({int(p) for _, _, p in zip(*np.where(m != e))})
            raise AssertionError(
                f"mirror_batch != colorflip encoding; differing planes: {bad_planes}"
            )
    # Make sure the fixture actually exercised castling rights.
    assert with_rights > 20, f"only {with_rights} positions had castling rights"


def test_policy_perm_matches_colorflip_move_indices(positions: list[dict]):
    """For every legal move, the file-flip policy permutation must map its
    index to the color-flipped move's index in the flipped position."""
    perm = _get_mirror_policy_perm()
    checked = 0
    for pos in positions[:50]:
        state = ChessGameState.from_dict(pos["state"])
        is_white = state.currentTurn == "white"
        for move in get_legal_moves(state):
            i = move_to_index(move, is_white)
            # Color-flip of the move: ranks mirror, files stay.
            fr, ff = 7 - move.from_pos.rank, move.from_pos.file
            tr, tf = 7 - move.to_pos.rank, move.to_pos.file
            if not is_white:
                # flipped state has white to move: no rotation in indexing
                j = (fr * 8 + ff) * 64 + (tr * 8 + tf)
            else:
                # flipped state has black to move: 180-degree rotation
                j = ((7 - fr) * 8 + (7 - ff)) * 64 + ((7 - tr) * 8 + (7 - tf))
            assert perm[i] == j
            checked += 1
    assert checked > 500


def test_mirror_is_involution(positions: list[dict]):
    """Applying the mirror twice must reproduce the original batch."""
    boards = np.stack(
        [encode_board(ChessGameState.from_dict(p["state"])) for p in positions[:32]]
    )
    rng = np.random.default_rng(0)
    policies = rng.random((len(boards), POLICY_SIZE)).astype(np.float32)
    mask = np.ones(len(boards), dtype=bool)

    b1, p1 = mirror_batch(boards, policies, mask)
    b2, p2 = mirror_batch(b1, p1, mask)
    assert np.array_equal(b2, boards)
    assert np.array_equal(p2, policies)
