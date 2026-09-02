import type {
  Board,
  Square,
  Piece,
  PieceColor,
  PieceType,
  Position,
  Move,
  CastlingRights,
  ChessGameState,
} from '@/types/chess';

function createInitialBoard(): Board {
  const board: Board = Array.from({ length: 8 }, () => Array(8).fill(null) as Square[]);

  const backRank: PieceType[] = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook'];

  // Black pieces (rank 0 = row 8)
  for (let file = 0; file < 8; file++) {
    board[0][file] = { color: 'black', type: backRank[file] };
    board[1][file] = { color: 'black', type: 'pawn' };
  }

  // White pieces (rank 7 = row 1)
  for (let file = 0; file < 8; file++) {
    board[7][file] = { color: 'white', type: backRank[file] };
    board[6][file] = { color: 'white', type: 'pawn' };
  }

  return board;
}

export function createInitialGameState(): ChessGameState {
  return {
    board: createInitialBoard(),
    currentTurn: 'white',
    castlingRights: {
      whiteKingside: true,
      whiteQueenside: true,
      blackKingside: true,
      blackQueenside: true,
    },
    enPassantTarget: null,
    halfMoveClock: 0,
    fullMoveNumber: 1,
    status: 'waiting',
    winner: null,
    moveHistory: [],
    lastMove: null,
  };
}

function cloneBoard(board: Board): Board {
  return board.map(rank => rank.map(sq => sq ? { ...sq } : null));
}

export function cloneGameState(state: ChessGameState): ChessGameState {
  return {
    board: cloneBoard(state.board),
    currentTurn: state.currentTurn,
    castlingRights: { ...state.castlingRights },
    enPassantTarget: state.enPassantTarget ? { ...state.enPassantTarget } : null,
    halfMoveClock: state.halfMoveClock,
    fullMoveNumber: state.fullMoveNumber,
    status: state.status,
    drawReason: state.drawReason,
    winner: state.winner,
    moveHistory: [...state.moveHistory],
    lastMove: state.lastMove ? { ...state.lastMove } : null,
  };
}

// --- Position identity for threefold repetition (FIDE 9.2) ---
//
// Two positions are "the same" when piece placement, side to move,
// castling rights, and the en passant target all match. Move clocks are
// excluded. This mirrors training's `_position_key` byte-for-byte in
// spirit (including its one known FIDE nit: the ep square is included
// unconditionally, whereas FIDE only counts ep when the capture is
// actually legal — both sides under-detect identically, so browser and
// training adjudicate the same games the same way).

export function positionKey(state: ChessGameState): string {
  const parts: string[] = [];
  const typeChar: Record<PieceType, string> = {
    king: 'k', queen: 'q', rook: 'r', bishop: 'b', knight: 'n', pawn: 'p',
  };
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const p = state.board[r][f];
      if (!p) {
        parts.push('.');
      } else {
        const ch = typeChar[p.type];
        parts.push(p.color === 'white' ? ch.toUpperCase() : ch);
      }
    }
  }
  parts.push(state.currentTurn[0]);
  const cr = state.castlingRights;
  parts.push(cr.whiteKingside ? '1' : '0');
  parts.push(cr.whiteQueenside ? '1' : '0');
  parts.push(cr.blackKingside ? '1' : '0');
  parts.push(cr.blackQueenside ? '1' : '0');
  if (state.enPassantTarget) {
    parts.push(String.fromCharCode(97 + state.enPassantTarget.file));
    parts.push(String(state.enPassantTarget.rank));
  } else {
    parts.push('--');
  }
  return parts.join('');
}

// Rebuild the position-occurrence map for a game by replaying its
// moveHistory from the initial position. The STARTING position seeds the
// map with count 1 — it can itself be the repeated position (1.Nf3 Nf6
// 2.Ng1 Ng8 brings it back). Used by the AI (which only receives a
// snapshot); ChessServer maintains its copy incrementally instead.
// --- Own-side placement (bot house rule) ------------------------------
//
// Just ONE colour's piece placement — no side to move, no castling, no
// en passant. Bots are held to a stricter standard than the FIDE rules
// bind a human to: a bot may never move its army back into an
// arrangement it has already occupied, so it can never shuffle back and
// forth even when the full position differs. Humans remain bound only
// by the real rules.
export function ownSideKey(state: ChessGameState, color: PieceColor): string {
  const typeChar: Record<PieceType, string> = {
    king: 'k', queen: 'q', rook: 'r', bishop: 'b', knight: 'n', pawn: 'p',
  };
  const parts: string[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const p = state.board[r][f];
      parts.push(p && p.color === color ? typeChar[p.type] : '.');
    }
  }
  return parts.join('');
}

// Every arrangement `color`'s army has already occupied this game,
// including the current one.
export function buildOwnSideKeys(
  state: ChessGameState,
  color: PieceColor,
): Set<string> {
  const seen = new Set<string>();
  let s = createInitialGameState();
  seen.add(ownSideKey(s, color));
  for (const move of state.moveHistory) {
    s = applyMove(s, move);
    seen.add(ownSideKey(s, color));
  }
  return seen;
}

export function buildPositionCounts(state: ChessGameState): Map<string, number> {
  const counts = new Map<string, number>();
  let s = createInitialGameState();
  counts.set(positionKey(s), 1);
  for (const move of state.moveHistory) {
    s = applyMove(s, move);
    const key = positionKey(s);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

// --- Insufficient material (FIDE 5.2.2) ---
//
// True when NO sequence of legal moves can produce checkmate:
//   K vs K, K+B vs K, K+N vs K, and K+B vs K+B with both bishops on the
//   same square color. Combinations that still allow helpmates
//   (K+N vs K+N, K+B vs K+N, K+N+N vs K) are NOT insufficient. Ported
//   from training's `_is_insufficient_material`.
export function isInsufficientMaterial(board: Board): boolean {
  const minors: { color: PieceColor; type: PieceType; squareColor: number }[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const p = board[r][f];
      if (!p || p.type === 'king') continue;
      if (p.type === 'pawn' || p.type === 'rook' || p.type === 'queen') {
        return false; // mate is reachable
      }
      minors.push({ color: p.color, type: p.type, squareColor: (r + f) % 2 });
    }
  }
  if (minors.length === 0) return true;            // K vs K
  if (minors.length === 1) return true;            // K+B or K+N vs K
  if (minors.length === 2) {
    const [a, b] = minors;
    // Two bishops, one per side, on the same square color.
    return (
      a.type === 'bishop' && b.type === 'bishop' &&
      a.color !== b.color && a.squareColor === b.squareColor
    );
  }
  return false;
}

function getPieceAt(board: Board, pos: Position): Square {
  return board[pos.rank][pos.file];
}

function isInBounds(pos: Position): boolean {
  return pos.rank >= 0 && pos.rank < 8 && pos.file >= 0 && pos.file < 8;
}

function oppositeColor(color: PieceColor): PieceColor {
  return color === 'white' ? 'black' : 'white';
}

// Find king position for a given color
function findKing(board: Board, color: PieceColor): Position {
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const piece = board[rank][file];
      if (piece && piece.color === color && piece.type === 'king') {
        return { rank, file };
      }
    }
  }
  throw new Error(`King not found for ${color}`);
}

// Check if a square is attacked by the given color
export function isSquareAttackedBy(board: Board, pos: Position, byColor: PieceColor): boolean {
  // Check knight attacks
  const knightMoves = [
    [-2, -1], [-2, 1], [-1, -2], [-1, 2],
    [1, -2], [1, 2], [2, -1], [2, 1],
  ];
  for (const [dr, df] of knightMoves) {
    const target = { rank: pos.rank + dr, file: pos.file + df };
    if (isInBounds(target)) {
      const piece = getPieceAt(board, target);
      if (piece && piece.color === byColor && piece.type === 'knight') return true;
    }
  }

  // Check pawn attacks
  const pawnDir = byColor === 'white' ? 1 : -1; // Pawns attack "upward" from their perspective
  for (const df of [-1, 1]) {
    const target = { rank: pos.rank + pawnDir, file: pos.file + df };
    if (isInBounds(target)) {
      const piece = getPieceAt(board, target);
      if (piece && piece.color === byColor && piece.type === 'pawn') return true;
    }
  }

  // Check king attacks
  for (let dr = -1; dr <= 1; dr++) {
    for (let df = -1; df <= 1; df++) {
      if (dr === 0 && df === 0) continue;
      const target = { rank: pos.rank + dr, file: pos.file + df };
      if (isInBounds(target)) {
        const piece = getPieceAt(board, target);
        if (piece && piece.color === byColor && piece.type === 'king') return true;
      }
    }
  }

  // Check sliding pieces (rook, bishop, queen)
  // Rook/Queen: horizontal and vertical
  const rookDirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
  for (const [dr, df] of rookDirs) {
    for (let i = 1; i < 8; i++) {
      const target = { rank: pos.rank + dr * i, file: pos.file + df * i };
      if (!isInBounds(target)) break;
      const piece = getPieceAt(board, target);
      if (piece) {
        if (piece.color === byColor && (piece.type === 'rook' || piece.type === 'queen')) return true;
        break; // Blocked
      }
    }
  }

  // Bishop/Queen: diagonals
  const bishopDirs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  for (const [dr, df] of bishopDirs) {
    for (let i = 1; i < 8; i++) {
      const target = { rank: pos.rank + dr * i, file: pos.file + df * i };
      if (!isInBounds(target)) break;
      const piece = getPieceAt(board, target);
      if (piece) {
        if (piece.color === byColor && (piece.type === 'bishop' || piece.type === 'queen')) return true;
        break;
      }
    }
  }

  return false;
}

export function isInCheck(board: Board, color: PieceColor): boolean {
  const kingPos = findKing(board, color);
  return isSquareAttackedBy(board, kingPos, oppositeColor(color));
}

// Apply a move to a board (mutates). Returns captured piece if any.
function applyMoveToBoard(board: Board, move: Move, castlingRights: CastlingRights): {
  captured: Square;
  isEnPassant: boolean;
  isCastle: boolean;
} {
  const piece = board[move.from.rank][move.from.file]!;
  const captured = board[move.to.rank][move.to.file];
  let isEnPassant = false;
  let isCastle = false;

  // En passant capture
  if (piece.type === 'pawn' && move.to.file !== move.from.file && !captured) {
    isEnPassant = true;
    board[move.from.rank][move.to.file] = null; // Remove captured pawn
  }

  // Castling
  if (piece.type === 'king' && Math.abs(move.to.file - move.from.file) === 2) {
    isCastle = true;
    if (move.to.file === 6) {
      // Kingside
      board[move.from.rank][5] = board[move.from.rank][7];
      board[move.from.rank][7] = null;
    } else {
      // Queenside
      board[move.from.rank][3] = board[move.from.rank][0];
      board[move.from.rank][0] = null;
    }
  }

  // Move piece
  board[move.to.rank][move.to.file] = piece;
  board[move.from.rank][move.from.file] = null;

  // Promotion
  if (move.promotion) {
    board[move.to.rank][move.to.file] = { color: piece.color, type: move.promotion };
  }

  // Update castling rights
  if (piece.type === 'king') {
    if (piece.color === 'white') {
      castlingRights.whiteKingside = false;
      castlingRights.whiteQueenside = false;
    } else {
      castlingRights.blackKingside = false;
      castlingRights.blackQueenside = false;
    }
  }
  if (piece.type === 'rook') {
    if (piece.color === 'white') {
      if (move.from.rank === 7 && move.from.file === 7) castlingRights.whiteKingside = false;
      if (move.from.rank === 7 && move.from.file === 0) castlingRights.whiteQueenside = false;
    } else {
      if (move.from.rank === 0 && move.from.file === 7) castlingRights.blackKingside = false;
      if (move.from.rank === 0 && move.from.file === 0) castlingRights.blackQueenside = false;
    }
  }
  // If a rook is captured, remove that side's castling rights
  if (move.to.rank === 0 && move.to.file === 7) castlingRights.blackKingside = false;
  if (move.to.rank === 0 && move.to.file === 0) castlingRights.blackQueenside = false;
  if (move.to.rank === 7 && move.to.file === 7) castlingRights.whiteKingside = false;
  if (move.to.rank === 7 && move.to.file === 0) castlingRights.whiteQueenside = false;

  return { captured, isEnPassant, isCastle };
}

// Generate pseudo-legal moves (doesn't filter for leaving king in check)
function generatePseudoLegalMoves(board: Board, color: PieceColor, castlingRights: CastlingRights, enPassantTarget: Position | null): Move[] {
  const moves: Move[] = [];

  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const piece = board[rank][file];
      if (!piece || piece.color !== color) continue;
      const from = { rank, file };

      switch (piece.type) {
        case 'pawn': {
          const dir = color === 'white' ? -1 : 1;
          const startRank = color === 'white' ? 6 : 1;
          const promoRank = color === 'white' ? 0 : 7;

          // Forward one
          const fwd = { rank: rank + dir, file };
          if (isInBounds(fwd) && !getPieceAt(board, fwd)) {
            if (fwd.rank === promoRank) {
              for (const promo of ['queen', 'rook', 'bishop', 'knight'] as PieceType[]) {
                moves.push({ from, to: fwd, promotion: promo });
              }
            } else {
              moves.push({ from, to: fwd });
            }

            // Forward two from starting rank
            if (rank === startRank) {
              const fwd2 = { rank: rank + dir * 2, file };
              if (!getPieceAt(board, fwd2)) {
                moves.push({ from, to: fwd2 });
              }
            }
          }

          // Captures
          for (const df of [-1, 1]) {
            const cap = { rank: rank + dir, file: file + df };
            if (!isInBounds(cap)) continue;
            const target = getPieceAt(board, cap);
            if (target && target.color !== color) {
              if (cap.rank === promoRank) {
                for (const promo of ['queen', 'rook', 'bishop', 'knight'] as PieceType[]) {
                  moves.push({ from, to: cap, promotion: promo });
                }
              } else {
                moves.push({ from, to: cap });
              }
            }
            // En passant
            if (enPassantTarget && cap.rank === enPassantTarget.rank && cap.file === enPassantTarget.file) {
              moves.push({ from, to: cap });
            }
          }
          break;
        }

        case 'knight': {
          const offsets = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
          for (const [dr, df] of offsets) {
            const to = { rank: rank + dr, file: file + df };
            if (!isInBounds(to)) continue;
            const target = getPieceAt(board, to);
            if (!target || target.color !== color) {
              moves.push({ from, to });
            }
          }
          break;
        }

        case 'bishop': {
          const dirs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
          for (const [dr, df] of dirs) {
            for (let i = 1; i < 8; i++) {
              const to = { rank: rank + dr * i, file: file + df * i };
              if (!isInBounds(to)) break;
              const target = getPieceAt(board, to);
              if (!target) {
                moves.push({ from, to });
              } else {
                if (target.color !== color) moves.push({ from, to });
                break;
              }
            }
          }
          break;
        }

        case 'rook': {
          const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
          for (const [dr, df] of dirs) {
            for (let i = 1; i < 8; i++) {
              const to = { rank: rank + dr * i, file: file + df * i };
              if (!isInBounds(to)) break;
              const target = getPieceAt(board, to);
              if (!target) {
                moves.push({ from, to });
              } else {
                if (target.color !== color) moves.push({ from, to });
                break;
              }
            }
          }
          break;
        }

        case 'queen': {
          const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];
          for (const [dr, df] of dirs) {
            for (let i = 1; i < 8; i++) {
              const to = { rank: rank + dr * i, file: file + df * i };
              if (!isInBounds(to)) break;
              const target = getPieceAt(board, to);
              if (!target) {
                moves.push({ from, to });
              } else {
                if (target.color !== color) moves.push({ from, to });
                break;
              }
            }
          }
          break;
        }

        case 'king': {
          // Normal king moves
          for (let dr = -1; dr <= 1; dr++) {
            for (let df = -1; df <= 1; df++) {
              if (dr === 0 && df === 0) continue;
              const to = { rank: rank + dr, file: file + df };
              if (!isInBounds(to)) continue;
              const target = getPieceAt(board, to);
              if (!target || target.color !== color) {
                moves.push({ from, to });
              }
            }
          }

          // Castling
          if (color === 'white' && rank === 7 && file === 4) {
            if (castlingRights.whiteKingside &&
                !board[7][5] && !board[7][6] &&
                board[7][7]?.type === 'rook' && board[7][7]?.color === 'white' &&
                !isSquareAttackedBy(board, { rank: 7, file: 4 }, 'black') &&
                !isSquareAttackedBy(board, { rank: 7, file: 5 }, 'black') &&
                !isSquareAttackedBy(board, { rank: 7, file: 6 }, 'black')) {
              moves.push({ from, to: { rank: 7, file: 6 } });
            }
            if (castlingRights.whiteQueenside &&
                !board[7][3] && !board[7][2] && !board[7][1] &&
                board[7][0]?.type === 'rook' && board[7][0]?.color === 'white' &&
                !isSquareAttackedBy(board, { rank: 7, file: 4 }, 'black') &&
                !isSquareAttackedBy(board, { rank: 7, file: 3 }, 'black') &&
                !isSquareAttackedBy(board, { rank: 7, file: 2 }, 'black')) {
              moves.push({ from, to: { rank: 7, file: 2 } });
            }
          }
          if (color === 'black' && rank === 0 && file === 4) {
            if (castlingRights.blackKingside &&
                !board[0][5] && !board[0][6] &&
                board[0][7]?.type === 'rook' && board[0][7]?.color === 'black' &&
                !isSquareAttackedBy(board, { rank: 0, file: 4 }, 'white') &&
                !isSquareAttackedBy(board, { rank: 0, file: 5 }, 'white') &&
                !isSquareAttackedBy(board, { rank: 0, file: 6 }, 'white')) {
              moves.push({ from, to: { rank: 0, file: 6 } });
            }
            if (castlingRights.blackQueenside &&
                !board[0][3] && !board[0][2] && !board[0][1] &&
                board[0][0]?.type === 'rook' && board[0][0]?.color === 'black' &&
                !isSquareAttackedBy(board, { rank: 0, file: 4 }, 'white') &&
                !isSquareAttackedBy(board, { rank: 0, file: 3 }, 'white') &&
                !isSquareAttackedBy(board, { rank: 0, file: 2 }, 'white')) {
              moves.push({ from, to: { rank: 0, file: 2 } });
            }
          }
          break;
        }
      }
    }
  }

  return moves;
}

// Generate all legal moves for the current player
export function getLegalMoves(state: ChessGameState): Move[] {
  const pseudoMoves = generatePseudoLegalMoves(
    state.board,
    state.currentTurn,
    state.castlingRights,
    state.enPassantTarget,
  );

  // Filter: only moves that don't leave own king in check
  return pseudoMoves.filter(move => {
    const testBoard = cloneBoard(state.board);
    const testCastling = { ...state.castlingRights };
    applyMoveToBoard(testBoard, move, testCastling);
    return !isInCheck(testBoard, state.currentTurn);
  });
}

// Get legal moves for a specific square
export function getLegalMovesForSquare(state: ChessGameState, pos: Position): Move[] {
  return getLegalMoves(state).filter(
    m => m.from.rank === pos.rank && m.from.file === pos.file,
  );
}

// Apply a move to the game state and return a new state
export function applyMove(state: ChessGameState, move: Move): ChessGameState {
  const newState = cloneGameState(state);
  const piece = getPieceAt(newState.board, move.from);
  if (!piece) throw new Error('No piece at source square');

  const result = applyMoveToBoard(newState.board, move, newState.castlingRights);

  // Update en passant target
  if (piece.type === 'pawn' && Math.abs(move.to.rank - move.from.rank) === 2) {
    newState.enPassantTarget = {
      rank: (move.from.rank + move.to.rank) / 2,
      file: move.from.file,
    };
  } else {
    newState.enPassantTarget = null;
  }

  // Update half-move clock
  if (piece.type === 'pawn' || result.captured || result.isEnPassant) {
    newState.halfMoveClock = 0;
  } else {
    newState.halfMoveClock++;
  }

  // Update full move number
  if (state.currentTurn === 'black') {
    newState.fullMoveNumber++;
  }

  // Switch turn
  newState.currentTurn = oppositeColor(state.currentTurn);
  newState.moveHistory = [...state.moveHistory, move];
  newState.lastMove = move;

  // Update game status
  const nextLegalMoves = getLegalMoves(newState);
  const inCheck = isInCheck(newState.board, newState.currentTurn);

  if (nextLegalMoves.length === 0) {
    if (inCheck) {
      newState.status = 'checkmate';
      newState.winner = state.currentTurn;
    } else {
      newState.status = 'stalemate';
    }
  } else if (inCheck) {
    newState.status = 'check';
  } else if (newState.halfMoveClock >= 100) {
    // 50-move rule
    newState.status = 'draw';
    newState.drawReason = 'fifty-move';
  } else {
    newState.status = 'active';
  }

  return newState;
}

// Check if a move is legal
export function isLegalMove(state: ChessGameState, move: Move): boolean {
  const legalMoves = getLegalMoves(state);
  return legalMoves.some(
    m =>
      m.from.rank === move.from.rank &&
      m.from.file === move.from.file &&
      m.to.rank === move.to.rank &&
      m.to.file === move.to.file &&
      m.promotion === move.promotion,
  );
}

// Get piece unicode symbol
export function getPieceSymbol(piece: Piece): string {
  const symbols: Record<PieceColor, Record<PieceType, string>> = {
    white: { king: '\u2654', queen: '\u2655', rook: '\u2656', bishop: '\u2657', knight: '\u2658', pawn: '\u2659' },
    black: { king: '\u265A', queen: '\u265B', rook: '\u265C', bishop: '\u265D', knight: '\u265E', pawn: '\u265F' },
  };
  return symbols[piece.color][piece.type];
}

// Always-filled (silhouette) glyph for board rendering. Both colors use the
// same shape, then CSS colors them \u2014 yields a much clearer light/dark
// distinction than the default outline-vs-filled Unicode pair.
export function getFilledPieceGlyph(type: PieceType): string {
  const glyphs: Record<PieceType, string> = {
    king: '\u265A',
    queen: '\u265B',
    rook: '\u265C',
    bishop: '\u265D',
    knight: '\u265E',
    pawn: '\u265F',
  };
  return glyphs[type];
}

// Convert position to algebraic notation
export function posToAlgebraic(pos: Position): string {
  const file = String.fromCharCode(97 + pos.file); // a-h
  const rank = String(8 - pos.rank); // 1-8
  return `${file}${rank}`;
}
