// LocalGameConnection - In-memory bridge between ChessServer and local client (host)

import type { GameConnection, SnapshotCallback, GameOverCallback } from './GameConnection';
import type { ChessServer } from './ChessServer';
import type { ChessCommand } from '@/types/network';

export class LocalGameConnection implements GameConnection {
  private server: ChessServer;
  private playerId: 1 | 2;
  private snapshotCallback: SnapshotCallback | null = null;
  private gameOverCallback: GameOverCallback | null = null;

  // The local human isn't always player 1: in Visual Bot games the bot
  // takes white (player 1) and the human plays black (player 2).
  constructor(server: ChessServer, playerId: 1 | 2 = 1) {
    this.server = server;
    this.playerId = playerId;

    server.addSnapshotListener((state) => {
      this.snapshotCallback?.(state);
    });

    server.addGameOverListener((winner) => {
      this.gameOverCallback?.(winner);
    });
  }

  sendCommand(command: ChessCommand): void {
    this.server.receiveCommand(command, this.playerId);
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
