import type { PlayerId } from './chess';
import type { ChessCommand, NetworkGameSnapshot } from './network';

export type SnapshotCallback = (state: NetworkGameSnapshot) => void;
export type GameOverCallback = (winner: PlayerId | null) => void;

export type GameConnection = {
  sendCommand(command: ChessCommand): void;
  onSnapshot(callback: SnapshotCallback): void;
  onGameOver(callback: GameOverCallback): void;
  disconnect(): void;
};
