"""Rust engine byte-parity with the Python engine.

Skipped if the Rust extension isn't built yet. Build with:

    cd training/rust_engine && maturin develop --release
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import chess_ai_rust
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False

from chess_ai.engine import ChessGameState, get_legal_moves, is_in_check, is_square_attacked_by
from chess_ai.engine import Move, Position

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity_positions.json"


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    if not _HAVE_RUST:
        pytest.skip("Rust extension not built; `cd rust_engine && maturin develop --release`")
    if not FIXTURE_PATH.exists():
        pytest.skip("Parity fixture missing; run `npm run dump-parity` first")
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _rust_move_key(t: tuple) -> tuple:
    fr, ff, tr, tf, promo = t
    return (fr, ff, tr, tf, promo)


def _py_move_key(m: Move) -> tuple:
    return (m.from_pos.rank, m.from_pos.file, m.to_pos.rank, m.to_pos.file, m.promotion)


def _ts_move_key(m: dict) -> tuple:
    return (m["from"]["rank"], m["from"]["file"], m["to"]["rank"], m["to"]["file"], m.get("promotion"))


def test_rust_legal_moves_match_python(fixture_data: dict):
    """Byte-for-byte: Rust returns the same set of legal moves as Python."""
    mismatches = []
    for pos in fixture_data["positions"]:
        s = pos["state"]
        rust_moves = chess_ai_rust.get_legal_moves(
            s["board"], s["currentTurn"], s["castlingRights"], s["enPassantTarget"]
        )
        rust_set = {_rust_move_key(m) for m in rust_moves}

        state = ChessGameState.from_dict(s)
        py_moves = get_legal_moves(state)
        py_set = {_py_move_key(m) for m in py_moves}

        if rust_set != py_set:
            mismatches.append({
                "id": pos["id"],
                "only_rust": rust_set - py_set,
                "only_py": py_set - rust_set,
            })
            if len(mismatches) >= 3:
                break
    assert not mismatches, f"Rust vs Python mismatches (≤3 shown): {mismatches}"


def test_rust_legal_moves_match_fixture(fixture_data: dict):
    """Rust output also matches the canonical TS-generated fixture."""
    mismatches = []
    for pos in fixture_data["positions"]:
        s = pos["state"]
        rust_moves = chess_ai_rust.get_legal_moves(
            s["board"], s["currentTurn"], s["castlingRights"], s["enPassantTarget"]
        )
        rust_set = {_rust_move_key(m) for m in rust_moves}
        ts_set = {_ts_move_key(m) for m in pos["legalMoves"]}
        if rust_set != ts_set:
            mismatches.append({
                "id": pos["id"],
                "only_rust": rust_set - ts_set,
                "only_ts": ts_set - rust_set,
            })
            if len(mismatches) >= 3:
                break
    assert not mismatches, f"Rust vs TS fixture mismatches (≤3 shown): {mismatches}"


def test_rust_is_in_check_match(fixture_data: dict):
    for pos in fixture_data["positions"]:
        s = pos["state"]
        state = ChessGameState.from_dict(s)
        for color in ("white", "black"):
            r = chess_ai_rust.is_in_check_py(s["board"], color)
            p = is_in_check(state.board, color)
            assert r == p, f"is_in_check mismatch at id={pos['id']} color={color}: rust={r} py={p}"


def test_rust_is_square_attacked_match(fixture_data: dict):
    # Spot-check a handful of squares per position.
    probes = [(0, 0), (3, 4), (4, 3), (7, 7), (0, 4), (7, 4)]
    for pos in fixture_data["positions"][:500]:  # keep runtime sane
        s = pos["state"]
        state = ChessGameState.from_dict(s)
        for rank, file in probes:
            for color in ("white", "black"):
                r = chess_ai_rust.is_square_attacked_by_py(s["board"], rank, file, color)
                p = is_square_attacked_by(state.board, Position(rank, file), color)
                assert r == p, (
                    f"is_square_attacked_by mismatch at id={pos['id']} "
                    f"sq=({rank},{file}) by={color}: rust={r} py={p}"
                )
