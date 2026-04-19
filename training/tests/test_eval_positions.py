"""Sanity tests for the curated eval position set.

If any opening sequence becomes illegal after an engine change, or any
asymmetric setup produces a terminal position by accident, these tests
will fail fast instead of waiting for training to hit the first eval
match.
"""

from __future__ import annotations

from chess_ai.engine import apply_move, get_legal_moves
from chess_ai.eval_positions import build_eval_positions


def test_all_eval_positions_build():
    positions = build_eval_positions()
    # 50 mate-in-1 (5 hand + 45 random) + 5 asymmetric + 5 balanced = 60.
    # eval_games must be 120 to play each position once per color.
    assert len(positions) == 60
    names = [p.name for p in positions]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


def test_mate_in_1_count_is_50():
    """10× the original 5 hand-crafted positions — stress-tests pattern
    recognition across many positions, not just a handful."""
    positions = build_eval_positions()
    mate_in_1 = [p for p in positions if p.difficulty == "mate-in-1"]
    assert len(mate_in_1) == 50, f"expected 50 mate-in-1, got {len(mate_in_1)}"


def test_eval_positions_are_non_terminal():
    for p in build_eval_positions():
        assert p.state.status == "active", (
            f"position {p.name!r} has status {p.state.status!r}; "
            f"should be 'active' (non-terminal)"
        )
        legal = get_legal_moves(p.state)
        assert len(legal) > 0, f"position {p.name!r} has no legal moves"


def test_difficulty_mix_present():
    """We want a balance of difficulty levels — at least one trivial,
    at least one clear, and several balanced openings. Keeps future
    edits from accidentally collapsing the set to one category."""
    positions = build_eval_positions()
    difficulties = {p.difficulty for p in positions}
    assert "trivial" in difficulties
    assert "clear" in difficulties
    assert "balanced" in difficulties


def test_trivial_positions_have_material_imbalance():
    """Trivial-difficulty positions should have overwhelmingly
    asymmetric material. Checked by material value, not piece count —
    K+Q vs K is 2-vs-1 pieces but 9 points of advantage."""
    values = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}
    for p in build_eval_positions():
        if p.difficulty != "trivial":
            continue
        white_val = black_val = 0
        for r in range(8):
            for f in range(8):
                piece = p.state.board[r][f]
                if piece is None:
                    continue
                v = values[piece.type]
                if piece.color == "white":
                    white_val += v
                else:
                    black_val += v
        # Trivial = >= 5 points of material imbalance (rook or better).
        assert abs(white_val - black_val) >= 5, (
            f"{p.name!r} is tagged 'trivial' but material imbalance is "
            f"only {abs(white_val - black_val)} points "
            f"(W={white_val}, B={black_val})."
        )


def test_build_is_cached():
    """Repeated calls return the same object — expensive opening-sequence
    replay should only happen once."""
    assert build_eval_positions() is build_eval_positions()


def test_mate_in_1_positions_actually_have_mate_in_1():
    """Every position tagged 'mate-in-1' must have at least one legal
    move that immediately results in checkmate. Catches design errors
    in hand-crafted positions before they silently produce bogus eval
    data (a 'draw' in a position that was supposed to be mate).
    """
    for p in build_eval_positions():
        if p.difficulty != "mate-in-1":
            continue
        legal = get_legal_moves(p.state)
        mating_moves = []
        for m in legal:
            after = apply_move(p.state, m)
            if after.status == "checkmate":
                mating_moves.append(m)
        assert len(mating_moves) >= 1, (
            f"Position {p.name!r} is tagged 'mate-in-1' but no legal "
            f"move produces checkmate. Tried {len(legal)} legal moves."
        )
