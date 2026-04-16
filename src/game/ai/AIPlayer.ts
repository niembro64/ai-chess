// AIPlayer: loads a trained model and generates moves via MCTS

import { ChessNet, type SerializedWeights } from './ChessNet';
import { runMCTS } from './MCTS';
import type { ChessGameState, Move } from '@/types/chess';

export type AIModelSource = 'trained' | 'preset' | 'untrained';

const DEFAULT_NET_CONFIG = { numResBlocks: 1, numFilters: 16, kernelSize: 3, valueHeadSize: 32, learningRate: 0.01 };

export class AIPlayer {
  private net: ChessNet;
  private sims: number;
  public readonly source: AIModelSource;

  private constructor(net: ChessNet, sims: number, source: AIModelSource) {
    this.net = net;
    this.sims = sims;
    this.source = source;
  }

  // Load trained model from IndexedDB (architecture comes from saved topology)
  static async createFromTrained(sims: number = 50): Promise<AIPlayer | null> {
    const net = await ChessNet.loadFromSaved(0.01);
    if (!net) return null;
    return new AIPlayer(net, sims, 'trained');
  }

  // Load from preset weights (hardcoded in project)
  static createFromPreset(weights: SerializedWeights, sims: number = 50): AIPlayer {
    const net = ChessNet.create(DEFAULT_NET_CONFIG);
    net.importWeights(weights);
    return new AIPlayer(net, sims, 'preset');
  }

  // Create with random weights (untrained)
  static createUntrained(sims: number = 25): AIPlayer {
    const net = ChessNet.create(DEFAULT_NET_CONFIG);
    return new AIPlayer(net, sims, 'untrained');
  }

  getMove(state: ChessGameState): Move {
    const result = runMCTS(state, this.net, this.sims);
    return result.move;
  }

  dispose(): void {
    this.net.dispose();
  }
}
