"""Verified starting positions and bounded proofs for Uncheck Chess v1.

This module intentionally does not use python-chess legality: its king-safety
and capture rules describe ordinary chess and would reject valid Uncheck play.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .engine import (
    CastlingRights,
    ChessGameState,
    Piece,
    Position,
    apply_move,
    get_legal_moves,
    position_key,
)


def state_from_fen(fen: str) -> ChessGameState:
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError(f"invalid FEN: {fen}")
    board = [[None] * 8 for _ in range(8)]
    pieces = {
        "k": "king", "q": "queen", "r": "rook", "b": "bishop",
        "n": "knight", "p": "pawn",
    }
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"invalid FEN board: {fen}")
    for rank, encoded in enumerate(ranks):
        file = 0
        for char in encoded:
            if char.isdigit():
                file += int(char)
            else:
                if char.lower() not in pieces or file >= 8:
                    raise ValueError(f"invalid FEN board: {fen}")
                board[rank][file] = Piece("white" if char.isupper() else "black", pieces[char.lower()])
                file += 1
        if file != 8:
            raise ValueError(f"invalid FEN rank: {fen}")
    if sum(p is not None and p.type == "king" for row in board for p in row) != 2:
        raise ValueError("Uncheck positions require exactly two kings")
    rights = fields[2]
    ep = fields[3]
    ep_pos = None if ep == "-" else Position(8 - int(ep[1]), ord(ep[0]) - ord("a"))
    return ChessGameState(
        board=board,
        currentTurn="white" if fields[1] == "w" else "black",
        castlingRights=CastlingRights("K" in rights, "Q" in rights, "k" in rights, "q" in rights),
        enPassantTarget=ep_pos,
        halfMoveClock=int(fields[4]),
        fullMoveNumber=int(fields[5]),
        status="active",
        ruleset="uncheck-v1",
    )


def move_uci(move) -> str:
    def sq(pos):
        return chr(ord("a") + pos.file) + str(8 - pos.rank)
    suffix = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}.get(move.promotion, "")
    return sq(move.from_pos) + sq(move.to_pos) + suffix


def forced_uncheck_moves(state: ChessGameState, max_plies: int = 5) -> tuple[str, ...]:
    """Moves that force the root side's actual Uncheck win within a bound."""
    target = state.currentTurn

    def prove(position: ChessGameState, remaining: int, counts: dict[bytes, int]) -> bool:
        if position.status == "uncheck":
            return position.winner == target
        if position.status in ("draw", "stalemate") or remaining == 0:
            return False
        legal = get_legal_moves(position)
        if not legal:
            return False
        target_turn = position.currentTurn == target
        outcomes = []
        for move in legal:
            child = apply_move(position, move)
            key = position_key(child)
            next_counts = dict(counts)
            next_counts[key] = next_counts.get(key, 0) + 1
            outcomes.append(False if next_counts[key] >= 3 else prove(child, remaining - 1, next_counts))
        return any(outcomes) if target_turn else all(outcomes)

    counts = {position_key(state): 1}
    winning = []
    for move in get_legal_moves(state):
        child = apply_move(state, move)
        if prove(child, max_plies - 1, counts):
            winning.append(move_uci(move))
    return tuple(winning)


@dataclass(frozen=True)
class UncheckPosition:
    name: str
    fen: str
    max_plies: int
    winning_moves: tuple[str, ...]

    @property
    def state(self) -> ChessGameState:
        return state_from_fen(self.fen)

    @property
    def difficulty(self) -> str:
        return "uncheck-tactic"


_TRAIN = (
    UncheckPosition("forced-knight-exposure", "r6k/8/5n2/8/8/3K4/P7/8 w - - 0 1", 3, ("d3e4", "d3c2", "d3d2", "d3e2", "a2a3")),
    UncheckPosition("forced-knight-exposure-black", "8/p7/3k4/8/8/5N2/8/R6K b - - 0 1", 3, ("a7a6", "d6c7", "d6d7", "d6e7", "d6e5")),
)

_EVAL = (
    UncheckPosition("heldout-mirrored-knight", "k6r/8/2n5/8/8/4K3/7P/8 w - - 0 1", 3, ("e3d4", "e3d2", "e3e2", "e3f2", "h2h3")),
)


def curriculum_positions(split: str = "train") -> tuple[UncheckPosition, ...]:
    return _TRAIN if split == "train" else _EVAL


def curriculum_start(rng: random.Random) -> ChessGameState:
    return rng.choice(_TRAIN).state


def validate_curriculum() -> None:
    for position in _TRAIN + _EVAL:
        actual = forced_uncheck_moves(position.state, position.max_plies)
        if actual != position.winning_moves:
            raise ValueError(f"invalid Uncheck proof {position.name}: {actual} != {position.winning_moves}")
