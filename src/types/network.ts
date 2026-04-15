import type { PlayerId, Move, ChessGameState } from './chess';

// Client -> Server
export type NetworkPlayerActionMessage = { type: 'command'; data: ChessCommand };

// Server -> Client
export type NetworkServerSnapshotMessage =
  | { type: 'state'; data: NetworkGameSnapshot | string }
  | { type: 'playerAssignment'; playerId: PlayerId }
  | { type: 'gameStart'; playerIds: PlayerId[] }
  | { type: 'playerJoined'; playerId: PlayerId; playerName: string }
  | { type: 'playerLeft'; playerId: PlayerId };

// Combined (transport envelope)
export type NetworkMessage = NetworkPlayerActionMessage | NetworkServerSnapshotMessage;

// Commands from client to server
export type ChessCommand =
  | { type: 'move'; move: Move }
  | { type: 'resign' }
  | { type: 'offerDraw' }
  | { type: 'acceptDraw' }
  | { type: 'declineDraw' };

// Snapshot of game state sent to clients
export type NetworkGameSnapshot = {
  gameState: ChessGameState;
  drawOffer: PlayerId | null; // Which player is offering a draw
};

export type LobbyPlayer = {
  playerId: PlayerId;
  name: string;
  isHost: boolean;
};

export type NetworkRole = 'host' | 'client';
