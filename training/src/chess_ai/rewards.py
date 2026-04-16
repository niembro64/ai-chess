"""Multi-signal positional reward shaping — port of `evaluatePosition` in Trainer.ts.

Produces a scalar score from the given player's perspective summing
hand-crafted positional signals (material, center control, castling,
mobility, doubled/passed pawns, hanging pieces, rooks on open files, ...).

The TS side multiplies each signal by a `RewardWeights` coefficient; the same
weights dataclass is mirrored here so a training run can dial reward shaping
exactly like the browser trainer.

Parity with the TS implementation is enforced by
`tests/test_reward_parity.py`, which loads positions + their TS-computed
scores from the parity fixture and asserts the Python port matches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .engine import Board, ChessGameState, PieceColor, Position, is_in_check, is_square_attacked_by


@dataclass
class RewardWeights:
    winning: float = 1.0
    material: float = 0.3
    pawnVal: float = 1.0
    knightVal: float = 3.0
    bishopVal: float = 3.0
    rookVal: float = 5.0
    queenVal: float = 9.0
    check: float = 0.02
    development: float = 0.01
    centerOccupation: float = 0.01
    centerAttack: float = 0.005
    castled: float = 0.03
    uncastled: float = 0.03
    mobility: float = 0.001
    bishopPair: float = 0.02
    doubledPawns: float = 0.01
    passedPawns: float = 0.02
    hangingPieces: float = 0.005
    rookOpenFile: float = 0.01


DEFAULT_REWARD_WEIGHTS = RewardWeights()


_CENTER_SQUARES = (Position(3, 3), Position(3, 4), Position(4, 3), Position(4, 4))


def _piece_value(piece_type: str, w: RewardWeights) -> float:
    if piece_type == "pawn":
        return w.pawnVal
    if piece_type == "knight":
        return w.knightVal
    if piece_type == "bishop":
        return w.bishopVal
    if piece_type == "rook":
        return w.rookVal
    if piece_type == "queen":
        return w.queenVal
    return 0.0


def _opposite(c: PieceColor) -> PieceColor:
    return "black" if c == "white" else "white"


def evaluate_position(
    state: ChessGameState,
    color: PieceColor,
    w: RewardWeights,
    precomputed_move_count: int | None = None,
) -> float:
    """Scalar reward from `color`'s perspective. Mirrors evaluatePosition in Trainer.ts."""
    board: Board = state.board
    opp: PieceColor = _opposite(color)
    back_rank = 7 if color == "white" else 0
    move_number = state.fullMoveNumber

    score = 0.0

    own_material = 0.0
    opp_material = 0.0
    own_bishops = 0
    opp_bishops = 0
    own_developed = 0
    own_center_pieces = 0
    opp_center_pieces = 0
    own_pawn_files: list[int] = []
    opp_pawn_files: list[int] = []
    own_king_pos = Position(0, 0)

    for r in range(8):
        for f in range(8):
            piece = board[r][f]
            if piece is None:
                continue
            val = _piece_value(piece.type, w)
            if piece.color == color:
                own_material += val
                if piece.type == "bishop":
                    own_bishops += 1
                if piece.type == "king":
                    own_king_pos = Position(r, f)
                if piece.type not in ("pawn", "king") and r != back_rank:
                    own_developed += 1
                if (r == 3 or r == 4) and (f == 3 or f == 4):
                    own_center_pieces += 1
                if piece.type == "pawn":
                    own_pawn_files.append(f)
            else:
                opp_material += val
                if piece.type == "bishop":
                    opp_bishops += 1
                if (r == 3 or r == 4) and (f == 3 or f == 4):
                    opp_center_pieces += 1
                if piece.type == "pawn":
                    opp_pawn_files.append(f)

    max_material = w.queenVal + 2 * w.rookVal + 2 * w.bishopVal + 2 * w.knightVal + 8 * w.pawnVal
    if max_material <= 0:
        max_material = 1.0

    if w.material:
        score += ((own_material - opp_material) / max_material) * w.material

    if w.check and is_in_check(board, opp):
        score += w.check

    if w.development and move_number <= 20:
        score += min(own_developed, 6) * w.development

    if w.centerOccupation:
        score += (own_center_pieces - opp_center_pieces) * w.centerOccupation

    if w.centerAttack:
        for sq in _CENTER_SQUARES:
            if is_square_attacked_by(board, sq, color):
                score += w.centerAttack
            if is_square_attacked_by(board, sq, opp):
                score -= w.centerAttack

    king_on_castled_square = (
        color == "white"
        and own_king_pos.rank == 7
        and own_king_pos.file in (6, 2)
    ) or (
        color == "black"
        and own_king_pos.rank == 0
        and own_king_pos.file in (6, 2)
    )
    if w.castled and king_on_castled_square:
        score += w.castled
    if w.uncastled and not king_on_castled_square and move_number > 15:
        king_on_start = (
            (color == "white" and own_king_pos.rank == 7 and own_king_pos.file == 4)
            or (color == "black" and own_king_pos.rank == 0 and own_king_pos.file == 4)
        )
        if king_on_start:
            score -= w.uncastled

    if w.mobility and state.currentTurn == color and precomputed_move_count is not None:
        score += min(precomputed_move_count, 40) * w.mobility

    if w.bishopPair:
        if own_bishops >= 2:
            score += w.bishopPair
        if opp_bishops >= 2:
            score -= w.bishopPair

    if w.doubledPawns:
        file_count: dict[int, int] = {}
        for f in own_pawn_files:
            file_count[f] = file_count.get(f, 0) + 1
        for count in file_count.values():
            if count > 1:
                score -= w.doubledPawns * (count - 1)

    if w.passedPawns:
        for f in own_pawn_files:
            passed = True
            for of2 in opp_pawn_files:
                if abs(of2 - f) <= 1:
                    passed = False
                    break
            if passed:
                score += w.passedPawns

    if w.hangingPieces:
        for r in range(8):
            for f in range(8):
                piece = board[r][f]
                if piece is None or piece.color != color or piece.type == "king":
                    continue
                pos = Position(r, f)
                if is_square_attacked_by(board, pos, opp) and not is_square_attacked_by(board, pos, color):
                    score -= _piece_value(piece.type, w) * w.hangingPieces

    if w.rookOpenFile:
        for r in range(8):
            for f in range(8):
                piece = board[r][f]
                if piece is None or piece.color != color or piece.type != "rook":
                    continue
                has_own_pawn = f in own_pawn_files
                has_opp_pawn = f in opp_pawn_files
                if not has_own_pawn and not has_opp_pawn:
                    score += w.rookOpenFile
                elif not has_own_pawn:
                    score += w.rookOpenFile * 0.5

    return score


def reward_weights_to_dict(w: RewardWeights) -> dict:
    return asdict(w)
