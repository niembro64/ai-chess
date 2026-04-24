"""Curated starting positions for the auto-eval tournament.

Instead of playing every eval game from the standard opening (where two
weak models both draw by shuffling), we test across a balanced mix of
position types so most of the eval budget measures skills that are
actively improving (not saturated tactical patterns):

  * "mate-in-1"  — side-to-move has a forced mate in one. Tests value-
    head sanity / tactical pattern recognition. Past mid-training both
    models solve ~all of these; kept mainly as a regression sentinel.
  * "endgame"    — asymmetric or technically-difficult endgames.
    Ranges from "overwhelming" (K+Q vs K) to "subtle" (R+P vs R
    Lucena/Philidor, opposite-color bishops). Tests conversion
    technique — where most strength gains land past the opening.
  * "middlegame" — structural middlegame tests (IQP, minority attack,
    opposite-side castling attack, classical tabiya positions). Reached
    via deep move sequences from known lines. Tests positional
    understanding once the opening phase is over.
  * "opening"    — mainline openings after 4-10 moves. Standard
    relative-strength test; stronger opening+early-middlegame play
    wins more of these.

Every position is played exactly twice (one game per color assignment)
so any intrinsic imbalance averages across the pair.

Mix (total 60 positions → 120 games at eval_games=120):
    20 mate-in-1  (5 hand-crafted + 15 procedurally generated)
    10 endgame    (5 asymmetric + 5 technique-heavy)
    15 middlegame (15 themed structural positions)
    15 opening    (5 original + 10 modern lines)

Hardcoded, deterministic — fixed across every eval match so scores are
directly comparable gen-to-gen WITHIN a given position set. Changing
the set resets the comparison baseline (so eval.csv scores pre- vs.
post-change aren't directly comparable).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Literal

from .engine import (
    CastlingRights,
    ChessGameState,
    Move,
    Piece,
    Position,
    apply_move,
    create_initial_game_state,
    get_legal_moves,
    is_in_check,
)

Difficulty = Literal["mate-in-1", "endgame", "middlegame", "opening"]


@dataclass(frozen=True)
class EvalPosition:
    name: str
    state: ChessGameState
    difficulty: Difficulty


# --- Mainline opening sequences (opening-phase positions) ------------------

# Move sequences in UCI long algebraic (e.g. "e2e4"). Castling is encoded
# as the king's two-square move ("e1g1" or "e1c1"). Promotions end with a
# fifth char ("e7e8q") — none of our book lines need them.
#
# These are "opening phase" positions — 4-10 moves deep — meant to test
# early-game judgment. Deeper structural positions live in
# _MIDDLEGAME_SEQUENCES below.
_OPENING_SEQUENCES: tuple[tuple[str, str], ...] = (
    # Original 5 — kept for continuity with pre-rebalance eval runs.
    ("Italian Game",             "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5"),
    ("Ruy Lopez, Closed",        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7"),
    ("Sicilian Najdorf",         "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"),
    ("Queen's Gambit Declined",  "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7"),
    ("King's Indian Defense",    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8"),
    # 10 additional modern lines covering 1.e4 / 1.d4 / 1.c4 / 1.Nf3,
    # both classical and hypermodern systems.
    ("Scotch Game",              "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 c1e3"),
    ("Vienna Game",              "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 f4e5 f6e4"),
    ("French Classical",         "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7"),
    ("Caro-Kann Classical",      "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5"),
    ("Pirc Defense",             "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3 f8g7"),
    ("Alekhine's Defense",       "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3 g7g6"),
    ("English (1.c4 e5)",        "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 f8b4"),
    ("Dutch Leningrad",          "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6 g1f3 f8g7"),
    ("London System",            "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3 f8d6"),
    ("Petroff Defense",          "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4"),
)


def _uci_to_move(uci: str) -> Move:
    """Parse a UCI move string into our engine's Move dataclass.

    UCI rank 8 is our rank 0 (black back rank); UCI file 'a' is our file 0.
    """
    if len(uci) not in (4, 5):
        raise ValueError(f"bad UCI move: {uci!r}")
    from_file = ord(uci[0]) - ord("a")
    from_rank = 8 - int(uci[1])
    to_file = ord(uci[2]) - ord("a")
    to_rank = 8 - int(uci[3])
    promotion = None
    if len(uci) == 5:
        promotion = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}[uci[4]]
    return Move(
        from_pos=Position(from_rank, from_file),
        to_pos=Position(to_rank, to_file),
        promotion=promotion,
    )


def _play_sequence(uci_sequence: str) -> ChessGameState:
    state = create_initial_game_state()
    state.status = "active"
    for token in uci_sequence.split():
        move = _uci_to_move(token)
        # Match against the legal-moves list so the engine picks up flags
        # (castling, en passant) from its own generator rather than us
        # reconstructing them by hand.
        for lm in get_legal_moves(state):
            if (
                lm.from_pos.rank == move.from_pos.rank
                and lm.from_pos.file == move.from_pos.file
                and lm.to_pos.rank == move.to_pos.rank
                and lm.to_pos.file == move.to_pos.file
                and lm.promotion == move.promotion
            ):
                state = apply_move(state, lm)
                break
        else:
            raise ValueError(
                f"illegal move {token!r} in sequence {uci_sequence!r}"
            )
    state.status = "active"
    return state


# --- Asymmetric-material positions (built by hand) -------------------------


def _empty_board() -> list[list[Piece | None]]:
    return [[None] * 8 for _ in range(8)]


def _place(board, color: str, piece_type: str, square: str) -> None:
    """Place a piece at a square given in chess notation ('e1', 'a8', ...)."""
    file_idx = ord(square[0]) - ord("a")
    rank_idx = 8 - int(square[1])
    board[rank_idx][file_idx] = Piece(color, piece_type)  # type: ignore[arg-type]


def _state_from_board(board, to_move: str) -> ChessGameState:
    return ChessGameState(
        board=board,
        currentTurn=to_move,  # type: ignore[arg-type]
        castlingRights=CastlingRights(False, False, False, False),
        enPassantTarget=None,
        halfMoveClock=0,
        fullMoveNumber=1,
        status="active",
    )


def _build_kq_vs_k() -> ChessGameState:
    """K+Q vs K — textbook mate in ~10 moves. Competent models must win."""
    b = _empty_board()
    _place(b, "white", "king", "e1")
    _place(b, "white", "queen", "d1")
    _place(b, "black", "king", "e8")
    return _state_from_board(b, "white")


def _build_kr_vs_k() -> ChessGameState:
    """K+R vs K — textbook mate, harder than K+Q (requires king+rook coordination)."""
    b = _empty_board()
    _place(b, "white", "king", "e1")
    _place(b, "white", "rook", "a1")
    _place(b, "black", "king", "e8")
    return _state_from_board(b, "white")


def _build_lone_king_vs_army() -> ChessGameState:
    """Black has only a king; white has full back rank + extra material.

    Should be trivial to win for any model that has learned ANYTHING. If
    the champion can't mate this, the eval signal tells you immediately
    that the value head is broken.
    """
    b = _empty_board()
    # Full white back rank.
    back_rank = [("rook","a1"), ("knight","b1"), ("bishop","c1"), ("queen","d1"),
                 ("king","e1"), ("bishop","f1"), ("knight","g1"), ("rook","h1")]
    for piece_type, sq in back_rank:
        _place(b, "white", piece_type, sq)
    # Black king exposed in middle of the board (not on starting square)
    # so the model can't rely on memorizing a particular mating net.
    _place(b, "black", "king", "e6")
    return _state_from_board(b, "white")


def _build_kp_vs_k_winning() -> ChessGameState:
    """K+P vs K, clearly winning (king supports pawn, opponent far away).

    White K on e5, P on e6, Black K on g8 — white just pushes the e-pawn
    and promotes. Tests whether the model understands pawn promotion.
    """
    b = _empty_board()
    _place(b, "white", "king", "e5")
    _place(b, "white", "pawn", "e6")
    _place(b, "black", "king", "g8")
    return _state_from_board(b, "white")


def _build_up_a_knight_middlegame() -> ChessGameState:
    """Italian Game position with black's c6 knight removed.

    White is up a clean minor piece in a normal middlegame position.
    Tests whether the model can convert a material advantage in a real
    position (as opposed to the stripped-down endgame tests above).
    """
    state = _play_sequence("e2e4 e7e5 g1f3 b8c6 f1c4 f8c5")
    # Remove the black knight from c6. c6 = file 2, rank index 2 (8-6).
    state.board[2][2] = None
    state.status = "active"
    return state


# --- Mate-in-1 positions ---------------------------------------------------
#
# Simplest possible test of value-head sanity: the side to move has a
# single-move checkmate available. If a trained model can't find these,
# something is deeply wrong — the position evaluation is broken. These
# are much more direct than "K+Q vs K" (which also requires endgame
# technique). A random model gets some of them purely by luck; a weak
# but functional model should solve them consistently.
#
# Each is verified by `tests/test_eval_positions.py` — the test walks
# every legal move and asserts at least one leads to checkmate, so any
# design error here fails fast instead of producing a silent bogus
# "eval" result.


def _build_mate_rook_back_rank() -> ChessGameState:
    """Classic back-rank rook mate: W plays Re1-e8#.
    Black king trapped on g8 by own pawns f7/g7/h7; e-file is clean."""
    b = _empty_board()
    _place(b, "white", "king", "g1")
    _place(b, "white", "rook", "e1")
    _place(b, "black", "king", "g8")
    _place(b, "black", "pawn", "f7")
    _place(b, "black", "pawn", "g7")
    _place(b, "black", "pawn", "h7")
    return _state_from_board(b, "white")


def _build_mate_ladder_two_rooks() -> ChessGameState:
    """Ladder mate with two rooks. W plays Rb1-b8#.
    Ra7 cuts off the 7th rank so black king on h8 has no escape."""
    b = _empty_board()
    _place(b, "white", "king", "a1")
    _place(b, "white", "rook", "a7")
    _place(b, "white", "rook", "b1")
    _place(b, "black", "king", "h8")
    return _state_from_board(b, "white")


def _build_mate_queen_8th_rank() -> ChessGameState:
    """Queen to 8th rank with king support. W plays Qa1-a8#.
    White king on f6 covers f7, g7; queen from a8 covers rank."""
    b = _empty_board()
    _place(b, "white", "king", "f6")
    _place(b, "white", "queen", "a1")
    _place(b, "black", "king", "g8")
    _place(b, "black", "pawn", "h7")
    return _state_from_board(b, "white")


def _build_mate_queen_on_h_file() -> ChessGameState:
    """King-supported queen mate. W plays Qxh7#.
    The h7 pawn blocks the queen's h-file check in the start position
    (otherwise black would already be in check — illegal). White captures
    the pawn with Qxh7#; queen defended by K@g6 covers every escape."""
    b = _empty_board()
    _place(b, "white", "king", "g6")
    _place(b, "white", "queen", "h1")
    _place(b, "black", "king", "h8")
    _place(b, "black", "pawn", "h7")
    return _state_from_board(b, "white")


def _build_mate_ladder_with_7th_cover() -> ChessGameState:
    """Rook ladder mate with king support. W plays Rb1-b8#.
    Ra7 closes 7th rank; K on f6 covers g7 escape; Rb8 delivers check."""
    b = _empty_board()
    _place(b, "white", "king", "f6")
    _place(b, "white", "rook", "a7")
    _place(b, "white", "rook", "b1")
    _place(b, "black", "king", "h8")
    return _state_from_board(b, "white")


# Each entry: (name, builder, difficulty).
_HAND_MATE_IN_1_POSITIONS: tuple[tuple[str, Callable[[], ChessGameState], Difficulty], ...] = (
    ("Mate-in-1: back rank (rook)",   _build_mate_rook_back_rank,      "mate-in-1"),
    ("Mate-in-1: two rook ladder",    _build_mate_ladder_two_rooks,    "mate-in-1"),
    ("Mate-in-1: queen on 8th rank",  _build_mate_queen_8th_rank,      "mate-in-1"),
    ("Mate-in-1: queen on h-file",    _build_mate_queen_on_h_file,     "mate-in-1"),
    ("Mate-in-1: rook ladder + king", _build_mate_ladder_with_7th_cover,"mate-in-1"),
)


# --- Procedural mate-in-1 generation ---------------------------------------
#
# Hand-designed mate-in-1 positions are error-prone (easy to build an
# illegal position where the defender is already in check) and don't
# scale. Instead we generate many candidates randomly, filter to legal
# positions where the side-to-move has at least one mating move, and
# keep the first N that pass. The RNG seed is fixed so the same
# positions are produced on every run, giving the same deterministic
# eval behavior as hand-crafted positions.

_RANDOM_SEED = 4242           # fixed for determinism across runs
_RANDOM_MATE_TARGET = 15      # +5 hand-crafted = 20 mate-in-1 positions total
_RANDOM_MATE_MAX_TRIES = 20_000  # ceiling on retries; generation aborts if hit


def _sq(rank: int, file: int) -> str:
    """Board-index (rank 0 = 8th rank, file 0 = a) → chess notation ('e4')."""
    return f"{chr(ord('a') + file)}{8 - rank}"


def _random_mate_candidate(rng: random.Random) -> ChessGameState | None:
    """Build a random sparse position; return it if side-to-move has
    mate-in-1 AND the position is legal (opponent not already in check).
    Returns None to signal "try again" on any inconsistency.
    """
    board = _empty_board()
    squares = [(r, f) for r in range(8) for f in range(8)]
    rng.shuffle(squares)
    sq_iter = iter(squares)

    try:
        stm_color = rng.choice(["white", "black"])
        other_color = "black" if stm_color == "white" else "white"

        # Place attacker king.
        ar, af = next(sq_iter)
        _place(board, stm_color, "king", _sq(ar, af))

        # Place defender king at Chebyshev distance > 1 from attacker.
        placed_defender = False
        for _ in range(40):
            try:
                dr, df = next(sq_iter)
            except StopIteration:
                return None
            if max(abs(ar - dr), abs(af - df)) > 1:
                _place(board, other_color, "king", _sq(dr, df))
                placed_defender = True
                break
        if not placed_defender:
            return None

        # Place 1-3 attacker pieces. Bias toward queens/rooks (the
        # classic mating agents) over minor pieces.
        piece_pool = ["queen", "rook", "rook", "rook", "bishop", "knight"]
        for _ in range(rng.randint(1, 3)):
            try:
                pr, pf = next(sq_iter)
            except StopIteration:
                break
            pt = rng.choice(piece_pool)
            if pt == "pawn" and (pr == 0 or pr == 7):
                continue
            _place(board, stm_color, pt, _sq(pr, pf))

        # Optionally add a few defender pieces (pawns/knights/bishops)
        # to block escape squares or force specific mating nets.
        if rng.random() < 0.5:
            def_pool = ["pawn", "pawn", "knight", "bishop"]
            for _ in range(rng.randint(1, 3)):
                try:
                    pr, pf = next(sq_iter)
                except StopIteration:
                    break
                pt = rng.choice(def_pool)
                if pt == "pawn" and (pr == 0 or pr == 7):
                    continue
                _place(board, other_color, pt, _sq(pr, pf))
    except (StopIteration, KeyError):
        return None

    state = _state_from_board(board, stm_color)

    # Legality check: defender must not already be in check (would imply
    # the defender's last move was illegal). Attacker must not be in a
    # self-check either — we want a "make your move" starting state.
    if is_in_check(board, other_color):
        return None
    if is_in_check(board, stm_color):
        # Being in check is a legal state, but for mate-in-1 tests we
        # want clean "you have mate available" positions, not "you're in
        # check and happen to have a counter-mate available."
        return None

    legal = get_legal_moves(state)
    if not legal:
        return None

    # Does any legal move deliver checkmate?
    for m in legal:
        after = apply_move(state, m)
        if after.status == "checkmate":
            return state
    return None


def _generate_random_mate_in_1_positions(count: int, seed: int) -> list[ChessGameState]:
    rng = random.Random(seed)
    results: list[ChessGameState] = []
    tries = 0
    while len(results) < count and tries < _RANDOM_MATE_MAX_TRIES:
        tries += 1
        cand = _random_mate_candidate(rng)
        if cand is not None:
            results.append(cand)
    if len(results) < count:
        raise RuntimeError(
            f"Mate-in-1 generator exhausted: wanted {count}, got {len(results)} "
            f"in {tries} tries. Bump _RANDOM_MATE_MAX_TRIES or loosen filter."
        )
    return results


# --- Technique-heavy endgame positions -------------------------------------
#
# Second tier of endgame tests: not overwhelming material, but well-known
# positions that test specific endgame techniques. Unlike the asymmetric
# set above, these can be drawn (defender holding) rather than always
# won — that's the point. The challenger/champion split on these tells
# you whether the model has learned the technique, not just the material.


def _build_rook_endgame_winning() -> ChessGameState:
    """R+P vs R, pawn on 7th supported by king — Lucena-like winning.

    White K on c8 shelters from checks along the 8th; pawn on d7 one
    square from promotion; rook on a1 ready for the 'bridge' technique.
    Black king on f6 is cut off from the pawn's queening square by
    distance; rook on h2 is the only defender. White wins by lifting
    the rook to build a shield and walking the king out.
    """
    b = _empty_board()
    _place(b, "white", "king", "c8")
    _place(b, "white", "pawn", "d7")
    _place(b, "white", "rook", "a1")
    _place(b, "black", "king", "f6")
    _place(b, "black", "rook", "h2")
    return _state_from_board(b, "white")


def _build_rook_endgame_drawn() -> ChessGameState:
    """R+P vs R with defender's king in front of the pawn — Philidor-
    like drawing position.

    White king on d5, pawn on d4, rook on a7 (attacker). Black king on
    d7 (front-of-pawn defense), rook on h6 (sixth-rank defense pattern).
    A correctly-played defender holds; this tests whether the model
    knows the Philidor method (and whether the attacker can find wins
    that aren't actually there).
    """
    b = _empty_board()
    _place(b, "white", "king", "d5")
    _place(b, "white", "pawn", "d4")
    _place(b, "white", "rook", "a7")
    _place(b, "black", "king", "d7")
    _place(b, "black", "rook", "h6")
    return _state_from_board(b, "white")


def _build_opposite_color_bishops() -> ChessGameState:
    """Opposite-color bishops with W up a pawn — classic drawing tendency.

    White has light-square bishop + extra pawn vs. Black's dark-square
    bishop. In opposite-color bishop endgames, up-a-pawn often can't
    win because the defender's bishop covers all the squares the
    attacker's bishop can't. Tests endgame evaluation subtlety: a
    value head that always rewards extra material will overestimate
    White here.
    """
    b = _empty_board()
    _place(b, "white", "king", "e4")
    _place(b, "white", "bishop", "d3")   # light-square
    _place(b, "white", "pawn", "e5")
    _place(b, "white", "pawn", "f4")
    _place(b, "black", "king", "e7")
    _place(b, "black", "bishop", "g7")   # dark-square
    _place(b, "black", "pawn", "f5")
    return _state_from_board(b, "white")


def _build_two_bishops_mate() -> ChessGameState:
    """K+2B vs K — winning but requires bishop coordination.

    Two bishops controlling adjacent diagonals drive the enemy king
    to a corner. Mate takes ~20-30 moves with perfect play. Tests
    whether the model understands long-horizon mate forcing, not
    just "if mate available, play it" tactical patterns.
    """
    b = _empty_board()
    _place(b, "white", "king", "e4")
    _place(b, "white", "bishop", "d3")   # light-square
    _place(b, "white", "bishop", "e3")   # dark-square
    _place(b, "black", "king", "e6")
    return _state_from_board(b, "white")


def _build_rook_plus_two_pawns() -> ChessGameState:
    """R+2P vs R — clearly winning extra-pawn rook endgame.

    White has two connected passed pawns on the queenside and an
    active rook. Conversion requires standard technique (shoulder-
    barging, creating a passed pawn front). Easier than Lucena but
    not trivially winning — tests basic endgame conversion skill.
    """
    b = _empty_board()
    _place(b, "white", "king", "e3")
    _place(b, "white", "rook", "a1")
    _place(b, "white", "pawn", "b4")
    _place(b, "white", "pawn", "c4")
    _place(b, "black", "king", "e6")
    _place(b, "black", "rook", "a8")
    return _state_from_board(b, "white")


# Each entry: (name, builder, difficulty). All endgame-category now;
# the old "trivial" / "clear" split collapsed because their statistical
# power is the same (both test conversion under known technique).
_ENDGAME_POSITIONS: tuple[tuple[str, Callable[[], ChessGameState], Difficulty], ...] = (
    ("K+Q vs K",                      _build_kq_vs_k,                  "endgame"),
    ("K+R vs K",                      _build_kr_vs_k,                  "endgame"),
    ("Lone king vs full army",        _build_lone_king_vs_army,        "endgame"),
    ("K+P vs K (promotion race)",     _build_kp_vs_k_winning,          "endgame"),
    ("Up a knight (middlegame)",      _build_up_a_knight_middlegame,   "endgame"),
    ("R+P vs R (Lucena-like)",        _build_rook_endgame_winning,     "endgame"),
    ("R+P vs R (Philidor-like)",      _build_rook_endgame_drawn,       "endgame"),
    ("Opposite-color bishops +1P",    _build_opposite_color_bishops,   "endgame"),
    ("K+2B vs K (two-bishop mate)",   _build_two_bishops_mate,         "endgame"),
    ("R+2P vs R",                     _build_rook_plus_two_pawns,      "endgame"),
)


# --- Middlegame tabiya sequences -------------------------------------------
#
# Deeper opening → middlegame transitions, reached via well-known
# mainline move sequences. Each position embodies a characteristic
# structural theme (IQP, minority attack, opposite-side castling, etc.)
# so the eval tests whether the model has learned positional patterns
# beyond opening memorization. All sequences are standard theory; any
# engine-logic bug that makes a move illegal will be caught by
# test_eval_positions_are_non_terminal.

_MIDDLEGAME_SEQUENCES: tuple[tuple[str, str], ...] = (
    ("IQP (Tarrasch)",
     "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5 c4d5 e6d5 g1f3 b8c6 g2g3 g8f6 "
     "f1g2 f8e7 e1g1 e8g8"),
    ("Minority attack (QGD Exchange)",
     "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5 c1g5 f8e7 e2e3 e8g8 "
     "f1d3 b8d7 d1c2 f8e8"),
    ("Najdorf English Attack",
     "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e6 "
     "f2f3 b7b5 d1d2 c8b7"),
    ("King's Indian Classical",
     "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 "
     "e1g1 b8c6 d4d5 c6e7"),
    ("Sveshnikov Sicilian",
     "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 "
     "c1g5 a7a6 b5a3 b7b5"),
    ("French Advance (blocked)",
     "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6 f1e2 c5d4 "
     "c3d4 c8d7"),
    ("Caro-Kann Advance",
     "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5 c1e3 b8c6 "
     "e1g1 c5d4"),
    ("Nimzo-Indian Classical",
     "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3 b7b6 "
     "g1f3 c8b7 e2e3 d7d6"),
    ("QGA main line",
     "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5 e1g1 a7a6 "
     "b1c3 b7b5 c4d3"),
    ("English symmetric",
     "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 g1f3 g8f6 e1g1 e8g8 "
     "d2d3 d7d6 a1b1 a7a6"),
    ("Grünfeld Exchange",
     "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7 "
     "f1c4 c7c5 g1e2 b8c6"),
    ("Slav Main",
     "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 g1f3 d5c4 a2a4 c8f5 e2e3 e7e6 "
     "f1c4 f8b4"),
    ("Benoni Modern",
     "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6 e2e4 g7g6 "
     "g1f3 f8g7 f1e2 e8g8"),
    ("Catalan",
     "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1 d5c4 "
     "d1c2 a7a6"),
    ("Dragon Yugoslav (pre-castling)",
     "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 "
     "f2f3 e8g8"),
)


# --- Public API -------------------------------------------------------------


_CACHE: list[EvalPosition] | None = None


def build_eval_positions() -> list[EvalPosition]:
    """Return all curated eval positions. Cached after first call.

    Order: mate-in-1 (fastest failure signal if the value head is
    broken), then endgame (technique-heavy conversions), then
    middlegame (structural themes), then opening (standard repertoire).
    The eval loop pairs each position with two color assignments, so
    each entry yields exactly two games.

    Mix: 20 mate-in-1 + 10 endgame + 15 middlegame + 15 opening = 60
    positions = 120 games at eval_games=120.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    positions: list[EvalPosition] = []
    # Hand-crafted mate-in-1 patterns first.
    for name, builder, difficulty in _HAND_MATE_IN_1_POSITIONS:
        positions.append(EvalPosition(name=name, state=builder(), difficulty=difficulty))
    # Then procedurally generated random mate-in-1 positions.
    random_states = _generate_random_mate_in_1_positions(_RANDOM_MATE_TARGET, _RANDOM_SEED)
    for i, state in enumerate(random_states, start=1):
        positions.append(EvalPosition(
            name=f"Mate-in-1: random #{i:02d}",
            state=state,
            difficulty="mate-in-1",
        ))
    # Endgame positions (asymmetric material + technique-heavy).
    for name, builder, difficulty in _ENDGAME_POSITIONS:
        positions.append(EvalPosition(name=name, state=builder(), difficulty=difficulty))
    # Themed middlegame tabiyas reached via deep move sequences.
    for name, seq in _MIDDLEGAME_SEQUENCES:
        positions.append(EvalPosition(
            name=name, state=_play_sequence(seq), difficulty="middlegame",
        ))
    # Opening-phase positions (4-10 moves deep).
    for name, seq in _OPENING_SEQUENCES:
        positions.append(EvalPosition(
            name=name, state=_play_sequence(seq), difficulty="opening",
        ))

    _CACHE = positions
    return positions
