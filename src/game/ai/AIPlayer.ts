// AIPlayer: loads a trained model from IndexedDB and generates moves via MCTS

import { ChessNet } from './ChessNet';
import { runMCTS } from './MCTS';
import type { ChessGameState, Move } from '@/types/chess';

export class AIPlayer {
  private net: ChessNet;
  private sims: number;

  private constructor(net: ChessNet, sims: number) {
    this.net = net;
    this.sims = sims;
  }

  // Try to load a trained model. Returns null if no saved model exists.
  static async create(sims: number = 50): Promise<AIPlayer | null> {
    // Create a placeholder network (architecture will be overwritten by load)
    const net = ChessNet.create({ numResBlocks: 1, numFilters: 16, learningRate: 0.01 });
    const loaded = await net.load();
    if (!loaded) {
      net.dispose();
      return null;
    }
    return new AIPlayer(net, sims);
  }

  // Create with an untrained model (random play) as fallback
  static createUntrained(sims: number = 25): AIPlayer {
    const net = ChessNet.create({ numResBlocks: 1, numFilters: 16, learningRate: 0.01 });
    return new AIPlayer(net, sims);
  }

  getMove(state: ChessGameState): Move {
    const result = runMCTS(state, this.net, this.sims);
    return result.move;
  }

  dispose(): void {
    this.net.dispose();
  }
}
