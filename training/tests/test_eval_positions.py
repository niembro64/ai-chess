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
    # 20 mate-in-1 (5 hand + 15 random) + 10 endgame + 15 middlegame
    # + 15 opening = 60. eval_games must be 120 to play each position
    # once per color.
    assert len(positions) == 60
    names = [p.name for p in positions]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


def test_category_counts():
    """Each difficulty category hits its target count — guards against
    accidentally dropping a chunk of positions during a refactor."""
    positions = build_eval_positions()
    counts = {}
    for p in positions:
        counts[p.difficulty] = counts.get(p.difficulty, 0) + 1
    assert counts.get("mate-in-1") == 20, counts
    assert counts.get("endgame") == 10, counts
    assert counts.get("middlegame") == 15, counts
    assert counts.get("opening") == 15, counts


def test_eval_positions_are_non_terminal():
    for p in build_eval_positions():
        assert p.state.status == "active", (
            f"position {p.name!r} has status {p.state.status!r}; "
            f"should be 'active' (non-terminal)"
        )
        legal = get_legal_moves(p.state)
        assert len(legal) > 0, f"position {p.name!r} has no legal moves"


def test_difficulty_mix_present():
    """All four difficulty categories are represented — keeps future
    edits from accidentally collapsing the set to one category."""
    positions = build_eval_positions()
    difficulties = {p.difficulty for p in positions}
    assert "mate-in-1" in difficulties
    assert "endgame" in difficulties
    assert "middlegame" in difficulties
    assert "opening" in difficulties


def test_endgame_overwhelming_positions_have_material_imbalance():
    """A subset of endgames (the legacy 'trivial' ones like K+Q vs K)
    should have overwhelming material imbalance. Identified by name —
    technique-heavy endgames like Lucena/Philidor have small imbalances
    by design and aren't checked here."""
    values = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}
    overwhelming_names = {
        "K+Q vs K", "K+R vs K", "Lone king vs full army",
    }
    for p in build_eval_positions():
        if p.name not in overwhelming_names:
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
        assert abs(white_val - black_val) >= 5, (
            f"{p.name!r} should be overwhelmingly-winning but material "
            f"imbalance is only {abs(white_val - black_val)} points "
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
