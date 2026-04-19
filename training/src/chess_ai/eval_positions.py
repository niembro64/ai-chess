"""Curated starting positions for the auto-eval tournament.

Instead of playing every eval game from the standard opening (where two
weak models both draw by shuffling), we test across a mix of difficulty
levels:

  * "trivial"  — one side has overwhelming material (e.g. K+Q vs K). A
    competent model MUST win these; any draw here is a sign the value
    head can't even convert a won position. Each is played once from
    each color, so both models face the same test. If one model fails
    to convert, the match score reflects it directly.
  * "clear"    — winning but requires technique (K+P vs K, middlegame
    up a knight). Tests "can you close out a slight advantage?"
  * "balanced" — mainline openings after 4-8 moves. Standard relative-
    strength test; stronger middlegame tech wins more of these.

Each position is played exactly twice (one game per color assignment)
so any intrinsic advantage in an asymmetric position averages out
across the pair. 5 asymmetric + 5 balanced × 2 colors = 20 games.

Hardcoded, deterministic — fixed across every eval match so scores are
directly comparable gen-to-gen.
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

Difficulty = Literal["mate-in-1", "trivial", "clear", "balanced"]


@dataclass(frozen=True)
class EvalPosition:
    name: str
    state: ChessGameState
    difficulty: Difficulty


# --- Mainline opening sequences (balanced positions) -----------------------

# Move sequences in UCI long algebraic (e.g. "e2e4"). Castling is encoded
# as the king's two-square move ("e1g1" or "e1c1"). Promotions end with a
# fifth char ("e7e8q") — none of our book lines need them.
_OPENING_SEQUENCES: tuple[tuple[str, str], ...] = (
    ("Italian Game",             "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5"),
    ("Ruy Lopez, Closed",        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7"),
    ("Sicilian Najdorf",         "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"),
    ("Queen's Gambit Declined",  "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7"),
    ("King's Indian Defense",    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8"),
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
_RANDOM_MATE_TARGET = 45      # +5 hand-crafted = 50 mate-in-1 positions total
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


# Each entry: (name, builder, difficulty). Order is stable — the trainer
# iterates this list and pairs each entry with two color assignments.
_ASYMMETRIC_POSITIONS: tuple[tuple[str, Callable[[], ChessGameState], Difficulty], ...] = (
    ("K+Q vs K",                      _build_kq_vs_k,                "trivial"),
    ("K+R vs K",                      _build_kr_vs_k,                "trivial"),
    ("Lone king vs full army",        _build_lone_king_vs_army,      "trivial"),
    ("K+P vs K (promotion race)",     _build_kp_vs_k_winning,        "clear"),
    ("Up a knight (middlegame)",      _build_up_a_knight_middlegame, "clear"),
)


# --- Public API -------------------------------------------------------------


_CACHE: list[EvalPosition] | None = None


def build_eval_positions() -> list[EvalPosition]:
    """Return all curated eval positions. Cached after first call.

    Order: mate-in-1 positions first (fastest failure signal if the value
    head is broken), then asymmetric-material endgames, then balanced
    openings. The eval loop iterates this list and pairs each position
    with two color assignments, so each entry yields exactly two games.
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
    # Asymmetric-material endgames.
    for name, builder, difficulty in _ASYMMETRIC_POSITIONS:
        positions.append(EvalPosition(name=name, state=builder(), difficulty=difficulty))
    # Mainline openings.
    for name, seq in _OPENING_SEQUENCES:
        positions.append(EvalPosition(name=name, state=_play_sequence(seq), difficulty="balanced"))

    _CACHE = positions
    return positions
