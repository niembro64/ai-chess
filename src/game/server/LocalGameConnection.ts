// LocalGameConnection - In-memory bridge between ChessServer and local client (host)

import type { GameConnection, SnapshotCallback, GameOverCallback } from './GameConnection';
import type { ChessServer } from './ChessServer';
import type { ChessCommand } from '@/types/network';

export class LocalGameConnection implements GameConnection {
  private server: ChessServer;
  private snapshotCallback: SnapshotCallback | null = null;
  private gameOverCallback: GameOverCallback | null = null;

  constructor(server: ChessServer) {
    this.server = server;

    server.addSnapshotListener((state) => {
      this.snapshotCallback?.(state);
    });

    server.addGameOverListener((winner) => {
      this.gameOverCallback?.(winner);
    });
  }

  sendCommand(command: ChessCommand): void {
    this.server.receiveCommand(command, 1); // Host is always player 1
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
  }
}
