export type PieceColor = 'white' | 'black';
export type PieceType = 'king' | 'queen' | 'rook' | 'bishop' | 'knight' | 'pawn';

export type Piece = {
  color: PieceColor;
  type: PieceType;
};

// Board is 8x8, indexed [rank][file] where rank 0 = row 8 (black's back rank), rank 7 = row 1 (white's back rank)
export type Square = Piece | null;
export type Board = Square[][];

export type Position = {
  rank: number; // 0-7 (0 = rank 8, 7 = rank 1)
  file: number; // 0-7 (0 = a, 7 = h)
};

export type Move = {
  from: Position;
  to: Position;
  promotion?: PieceType; // For pawn promotion
};

export type CastlingRights = {
  whiteKingside: boolean;
  whiteQueenside: boolean;
  blackKingside: boolean;
  blackQueenside: boolean;
};

export type GameStatus = 'waiting' | 'active' | 'check' | 'checkmate' | 'stalemate' | 'draw';

export type ChessGameState = {
  board: Board;
  currentTurn: PieceColor;
  castlingRights: CastlingRights;
  enPassantTarget: Position | null;
  halfMoveClock: number; // For 50-move rule
  fullMoveNumber: number;
  status: GameStatus;
  winner: PieceColor | null;
  moveHistory: Move[];
  lastMove: Move | null;
};

export type PlayerId = 1 | 2;

export function playerIdToColor(playerId: PlayerId): PieceColor {
  return playerId === 1 ? 'white' : 'black';
}

export function colorToPlayerId(color: PieceColor): PlayerId {
  return color === 'white' ? 1 : 2;
}
