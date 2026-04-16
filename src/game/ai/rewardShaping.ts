// Multi-signal positional reward shaping.
// Returns a scalar score from the given player's perspective combining material
// + tactical + positional signals, each scaled by a tunable weight.
//
// KEEP IN SYNC with `training/src/chess_ai/rewards.py` — any change here must
// be mirrored there, and the reward-parity test must be re-verified.

import type { ChessGameState, PieceColor, Board } from '@/types/chess';
import { isInCheck, isSquareAttackedBy } from '@/game/chess/ChessEngine';

export type RewardWeights = {
  winning: number;
  material: number;
  pawnVal: number;
  knightVal: number;
  bishopVal: number;
  rookVal: number;
  queenVal: number;
  check: number;
  development: number;
  centerOccupation: number;
  centerAttack: number;
  castled: number;
  uncastled: number;
  mobility: number;
  bishopPair: number;
  doubledPawns: number;
  passedPawns: number;
  hangingPieces: number;
  rookOpenFile: number;
};

export const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  winning: 1.0,
  material: 0.3,
  pawnVal: 1,
  knightVal: 3,
  bishopVal: 3,
  rookVal: 5,
  queenVal: 9,
  check: 0.02,
  development: 0.01,
  centerOccupation: 0.01,
  centerAttack: 0.005,
  castled: 0.03,
  uncastled: 0.03,
  mobility: 0.001,
  bishopPair: 0.02,
  doubledPawns: 0.01,
  passedPawns: 0.02,
  hangingPieces: 0.005,
  rookOpenFile: 0.01,
};

export const PIECE_VALUES: Record<string, number> = {
  pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9, king: 0,
};

export const CENTER_SQUARES = [
  { rank: 3, file: 3 }, { rank: 3, file: 4 }, // d5, e5
  { rank: 4, file: 3 }, { rank: 4, file: 4 }, // d4, e4
];

function oppositeColor(c: PieceColor): PieceColor {
  return c === 'white' ? 'black' : 'white';
}

function pieceValue(type: string, w: RewardWeights): number {
  switch (type) {
    case 'pawn': return w.pawnVal;
    case 'knight': return w.knightVal;
    case 'bishop': return w.bishopVal;
    case 'rook': return w.rookVal;
    case 'queen': return w.queenVal;
    default: return 0;
  }
}

export function evaluatePosition(
  state: ChessGameState,
  color: PieceColor,
  w: RewardWeights,
  precomputedMoveCount?: number,
): number {
  const board = state.board;
  const opp = oppositeColor(color);
  const backRank = color === 'white' ? 7 : 0;
  const moveNumber = state.fullMoveNumber;

  let score = 0;

  let ownMaterial = 0, oppMaterial = 0;
  let maxMaterial = 0;
  let ownBishops = 0, oppBishops = 0;
  let ownDeveloped = 0;
  let ownCenterPieces = 0, oppCenterPieces = 0;
  const ownPawnFiles: number[] = [];
  const oppPawnFiles: number[] = [];
  let ownKingPos = { rank: 0, file: 0 };

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[r][f];
      if (!piece) continue;
      const isOwn = piece.color === color;
      const val = pieceValue(piece.type, w);
      if (isOwn) {
        ownMaterial += val;
        if (piece.type === 'bishop') ownBishops++;
        if (piece.type === 'king') ownKingPos = { rank: r, file: f };
        if (piece.type !== 'pawn' && piece.type !== 'king' && r !== backRank) ownDeveloped++;
        if ((r === 3 || r === 4) && (f === 3 || f === 4)) ownCenterPieces++;
        if (piece.type === 'pawn') ownPawnFiles.push(f);
      } else {
        oppMaterial += val;
        if (piece.type === 'bishop') oppBishops++;
        if ((r === 3 || r === 4) && (f === 3 || f === 4)) oppCenterPieces++;
        if (piece.type === 'pawn') oppPawnFiles.push(f);
      }
    }
  }

  maxMaterial = w.queenVal + 2 * w.rookVal + 2 * w.bishopVal + 2 * w.knightVal + 8 * w.pawnVal;
  if (maxMaterial <= 0) maxMaterial = 1;

  if (w.material) score += ((ownMaterial - oppMaterial) / maxMaterial) * w.material;
  if (w.check && isInCheck(board, opp)) score += w.check;
  if (w.development && moveNumber <= 20) score += Math.min(ownDeveloped, 6) * w.development;
  if (w.centerOccupation) score += (ownCenterPieces - oppCenterPieces) * w.centerOccupation;

  if (w.centerAttack) {
    for (const sq of CENTER_SQUARES) {
      if (isSquareAttackedBy(board, sq, color)) score += w.centerAttack;
      if (isSquareAttackedBy(board, sq, opp)) score -= w.centerAttack;
    }
  }

  const kingOnCastledSquare =
    (color === 'white' && ownKingPos.rank === 7 && (ownKingPos.file === 6 || ownKingPos.file === 2)) ||
    (color === 'black' && ownKingPos.rank === 0 && (ownKingPos.file === 6 || ownKingPos.file === 2));
  if (w.castled && kingOnCastledSquare) score += w.castled;
  if (w.uncastled && !kingOnCastledSquare && moveNumber > 15) {
    const kingOnStart =
      (color === 'white' && ownKingPos.rank === 7 && ownKingPos.file === 4) ||
      (color === 'black' && ownKingPos.rank === 0 && ownKingPos.file === 4);
    if (kingOnStart) score -= w.uncastled;
  }

  if (w.mobility && state.currentTurn === color && precomputedMoveCount !== undefined) {
    score += Math.min(precomputedMoveCount, 40) * w.mobility;
  }

  if (w.bishopPair) {
    if (ownBishops >= 2) score += w.bishopPair;
    if (oppBishops >= 2) score -= w.bishopPair;
  }

  if (w.doubledPawns) {
    const fileCount = new Map<number, number>();
    for (const f of ownPawnFiles) fileCount.set(f, (fileCount.get(f) ?? 0) + 1);
    for (const count of fileCount.values()) {
      if (count > 1) score -= w.doubledPawns * (count - 1);
    }
  }

  if (w.passedPawns) {
    for (const f of ownPawnFiles) {
      let passed = true;
      for (const of2 of oppPawnFiles) {
        if (Math.abs(of2 - f) <= 1) { passed = false; break; }
      }
      if (passed) score += w.passedPawns;
    }
  }

  if (w.hangingPieces) {
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const piece = board[r][f];
        if (!piece || piece.color !== color || piece.type === 'king') continue;
        const pos = { rank: r, file: f };
        if (isSquareAttackedBy(board, pos, opp) && !isSquareAttackedBy(board, pos, color)) {
          score -= pieceValue(piece.type, w) * w.hangingPieces;
        }
      }
    }
  }

  if (w.rookOpenFile) {
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const piece = board[r][f];
        if (!piece || piece.color !== color || piece.type !== 'rook') continue;
        const hasOwnPawn = ownPawnFiles.includes(f);
        const hasOppPawn = oppPawnFiles.includes(f);
        if (!hasOwnPawn && !hasOppPawn) score += w.rookOpenFile;
        else if (!hasOwnPawn) score += w.rookOpenFile * 0.5;
      }
    }
  }

  return score;
}

// Simple material-only evaluation (UI display).
export function materialAdvantage(state: ChessGameState, color: PieceColor): number {
  const board: Board = state.board;
  let own = 0, opp = 0;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[r][f];
      if (!piece) continue;
      const val = PIECE_VALUES[piece.type] ?? 0;
      if (piece.color === color) own += val; else opp += val;
    }
  }
  return (own - opp) / 39;
}
