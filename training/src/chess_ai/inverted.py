"""Exact short selfmate curriculum, independent of ordinary-chess scores.

A proof succeeds only if the target can force its own checkmate against
EVERY legal reply. Budget exhaustion is unknown, never a winning label.
Training and held-out sets split base positions before symmetry augmentation.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess

from .engine import CastlingRights, ChessGameState, Piece, Position


class ProofBudgetExceeded(RuntimeError):
    pass


def state_from_fen(fen: str) -> ChessGameState:
    b = chess.Board(fen)
    if not b.is_valid():
        raise ValueError(f"Invalid curriculum FEN: {fen}")
    board = [[None] * 8 for _ in range(8)]
    for sq, p in b.piece_map().items():
        board[7 - chess.square_rank(sq)][chess.square_file(sq)] = Piece(
            "white" if p.color else "black", chess.piece_name(p.piece_type)
        )
    ep = b.ep_square
    status = (
        "checkmate"
        if b.is_checkmate()
        else "stalemate"
        if b.is_stalemate()
        else "draw"
        if b.halfmove_clock >= 100
        else "check"
        if b.is_check()
        else "active"
    )
    return ChessGameState(
        board=board,
        currentTurn="white" if b.turn else "black",
        castlingRights=CastlingRights(
            b.has_kingside_castling_rights(chess.WHITE),
            b.has_queenside_castling_rights(chess.WHITE),
            b.has_kingside_castling_rights(chess.BLACK),
            b.has_queenside_castling_rights(chess.BLACK),
        ),
        enPassantTarget=None if ep is None else Position(7 - chess.square_rank(ep), chess.square_file(ep)),
        halfMoveClock=b.halfmove_clock,
        fullMoveNumber=b.fullmove_number,
        status=status,
    )


def forced_selfmate_moves(fen: str, max_plies: int = 6, node_budget: int = 100_000) -> tuple[str, ...]:
    """Winning root moves within a bound; target is the root side to move."""
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid proof position: {fen}")
    target = board.turn
    nodes = 0

    def prove(remaining: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            raise ProofBudgetExceeded(f"Proof exceeded {node_budget} nodes")
        if board.is_checkmate():
            return board.turn == target
        if (
            remaining == 0
            or board.is_stalemate()
            or board.is_insufficient_material()
            or board.halfmove_clock >= 100
            or board.is_repetition(2)
        ):
            return False
        moves = list(board.legal_moves)
        # Checks constrain the resisting player; ordering affects cost only.
        moves.sort(key=board.gives_check, reverse=True)
        our_turn = board.turn == target
        for move in moves:
            board.push(move)
            try:
                success = prove(remaining - 1)
            finally:
                board.pop()
            if success == our_turn:
                return our_turn
        return not our_turn

    winning = []
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if prove(max_plies - 1):
                winning.append(move.uci())
        finally:
            board.pop()
    return tuple(winning)


@dataclass(frozen=True)
class SelfmatePosition:
    name: str
    fen: str
    plies: int
    winning_moves: tuple[str, ...]

    @property
    def state(self):
        return state_from_fen(self.fen)

    @property
    def difficulty(self):
        return f"selfmate-{self.plies // 2}"


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict, ...]:
    path = Path(__file__).with_name("selfmate_positions.json")
    return tuple(json.loads(path.read_text()))


def selfmate_positions(split: str = "train") -> list[SelfmatePosition]:
    if split not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    out = []
    for i, row in enumerate(_catalog()):
        if row["split"] != split:
            continue
        # Color reversal is symmetric for these castling-free positions.
        board = chess.Board(row["fen"])
        for flip in (False, True):
            b = board.mirror() if flip else board
            fen = b.fen()
            moves = (
                tuple(
                    chess.Move(
                        chess.square_mirror(chess.Move.from_uci(m).from_square),
                        chess.square_mirror(chess.Move.from_uci(m).to_square),
                        promotion=chess.Move.from_uci(m).promotion,
                    ).uci()
                    for m in row["winning_moves"]
                )
                if flip
                else tuple(row["winning_moves"])
            )
            out.append(SelfmatePosition(f"Selfmate {i:03d} {'B' if flip else 'W'}", fen, row["plies"], moves))
    return out


def curriculum_start(rng: random.Random) -> ChessGameState:
    # Balance difficulty instead of letting plentiful one-move proofs dominate.
    positions = selfmate_positions("train")
    depth = rng.choice(sorted({p.plies for p in positions}))
    return rng.choice([p for p in positions if p.plies == depth]).state


def stable_position_id(state: ChessGameState) -> str:
    """Include clocks and rights, unlike repetition identity or display names."""
    data = state.to_dict()
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
