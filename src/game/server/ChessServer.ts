// ChessServer - Authoritative chess game server (runs in host's browser)

import {
  createInitialGameState,
  applyMove,
  isLegalMove,
  isInsufficientMaterial,
  positionKey,
} from '../chess/ChessEngine';
import type { ChessGameState, PlayerId } from '@/types/chess';
import { playerIdToColor, colorToPlayerId } from '@/types/chess';
import type { ChessCommand, NetworkGameSnapshot } from '@/types/network';
import type { SnapshotCallback, GameOverCallback } from './GameConnection';

export class ChessServer {
  private gameState: ChessGameState;
  private drawOffer: PlayerId | null = null;
  // Occurrence count per position (placement + turn + castling + ep) for
  // FIDE threefold repetition. The engine stays stateless per-position;
  // repetition needs game history, so the rules glue lives here — same
  // architecture as the Python training loop.
  private positionCounts: Map<string, number> = new Map();

  private snapshotListeners: SnapshotCallback[] = [];
  private gameOverListeners: GameOverCallback[] = [];

  constructor() {
    this.gameState = createInitialGameState();
  }

  // Start the game
  start(): void {
    this.gameState.status = 'active';
    // Seed the starting position with count 1 — it can itself be the
    // repeated position (1.Nf3 Nf6 2.Ng1 Ng8 brings it back).
    this.positionCounts = new Map([[positionKey(this.gameState), 1]]);
    this.emitSnapshot();
  }

  // Receive a command from a client
  receiveCommand(command: ChessCommand, fromPlayerId: PlayerId): void {
    switch (command.type) {
      case 'move': {
        // Validate it's this player's turn
        const playerColor = playerIdToColor(fromPlayerId);
        if (this.gameState.currentTurn !== playerColor) {
          console.warn(`Player ${fromPlayerId} tried to move out of turn`);
          return;
        }

        // Validate the move is legal
        if (!isLegalMove(this.gameState, command.move)) {
          console.warn(`Player ${fromPlayerId} attempted illegal move`);
          return;
        }

        // Clear any pending draw offer on a move
        this.drawOffer = null;

        // Apply the move
        this.gameState = applyMove(this.gameState, command.move);

        // Draws the stateless engine can't see: threefold repetition
        // (needs game history) and insufficient material (rules glue on
        // top of movegen). Engine-produced terminal statuses take
        // precedence — a move that mates wins even if it also creates
        // the third occurrence (FIDE 5.1).
        if (this.gameState.status === 'active' || this.gameState.status === 'check') {
          const key = positionKey(this.gameState);
          const count = (this.positionCounts.get(key) ?? 0) + 1;
          this.positionCounts.set(key, count);
          if (count >= 3) {
            this.gameState.status = 'draw';
            this.gameState.drawReason = 'repetition';
          } else if (isInsufficientMaterial(this.gameState.board)) {
            this.gameState.status = 'draw';
            this.gameState.drawReason = 'insufficient-material';
          }
        }
        this.emitSnapshot();

        // Check for game over
        if (this.gameState.status === 'checkmate' && this.gameState.winner) {
          const winnerId = colorToPlayerId(this.gameState.winner);
          for (const listener of this.gameOverListeners) {
            listener(winnerId);
          }
        } else if (this.gameState.status === 'stalemate' || this.gameState.status === 'draw') {
          for (const listener of this.gameOverListeners) {
            listener(null);
          }
        }
        break;
      }

      case 'resign': {
        const resignColor = playerIdToColor(fromPlayerId);
        const winnerColor = resignColor === 'white' ? 'black' : 'white';
        this.gameState.status = 'checkmate'; // Treat resign like checkmate for display
        this.gameState.winner = winnerColor;
        this.emitSnapshot();

        const winnerId = colorToPlayerId(winnerColor);
        for (const listener of this.gameOverListeners) {
          listener(winnerId);
        }
        break;
      }

      case 'offerDraw': {
        this.drawOffer = fromPlayerId;
        this.emitSnapshot();
        break;
      }

      case 'acceptDraw': {
        if (this.drawOffer && this.drawOffer !== fromPlayerId) {
          this.gameState.status = 'draw';
          this.gameState.drawReason = 'agreement';
          this.drawOffer = null;
          this.emitSnapshot();
          for (const listener of this.gameOverListeners) {
            listener(null);
          }
        }
        break;
      }

      case 'declineDraw': {
        this.drawOffer = null;
        this.emitSnapshot();
        break;
      }
    }
  }

  // Emit current state to all listeners
  private emitSnapshot(): void {
    const snapshot: NetworkGameSnapshot = {
      gameState: this.gameState,
      drawOffer: this.drawOffer,
    };

    for (const listener of this.snapshotListeners) {
      listener(snapshot);
    }
  }

  // Add a snapshot listener
  addSnapshotListener(callback: SnapshotCallback): void {
    this.snapshotListeners.push(callback);
  }

  // Add a game over listener
  addGameOverListener(callback: GameOverCallback): void {
    this.gameOverListeners.push(callback);
  }

  // Get current game state
  getGameState(): ChessGameState {
    return this.gameState;
  }

  // Stop and cleanup
  stop(): void {
    this.snapshotListeners.length = 0;
    this.gameOverListeners.length = 0;
  }
}
