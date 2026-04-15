// RemoteGameConnection - WebRTC bridge for remote clients

import type { GameConnection, SnapshotCallback, GameOverCallback } from './GameConnection';
import type { ChessCommand, NetworkGameSnapshot } from '@/types/network';
import { networkManager } from '../network/NetworkManager';

export class RemoteGameConnection implements GameConnection {
  private snapshotCallback: SnapshotCallback | null = null;
  private gameOverCallback: GameOverCallback | null = null;

  constructor() {
    networkManager.onStateReceived = (state: NetworkGameSnapshot) => {
      this.snapshotCallback?.(state);

      // Check for game over
      const gs = state.gameState;
      if (gs.status === 'checkmate' && gs.winner) {
        const winnerId = gs.winner === 'white' ? 1 : 2;
        this.gameOverCallback?.(winnerId as 1 | 2);
      } else if (gs.status === 'stalemate' || gs.status === 'draw') {
        this.gameOverCallback?.(null);
      }
    };
  }

  sendCommand(command: ChessCommand): void {
    networkManager.sendCommand(command);
  }

  onSnapshot(callback: SnapshotCallback): void {
    this.snapshotCallback = callback;
  }

  onGameOver(callback: GameOverCallback): void {
    this.gameOverCallback = callback;
  }

  disconnect(): void {
    this.snapshotCallback = null;
    this.gameOverCallback = null;
    networkManager.onStateReceived = undefined;
  }
}
