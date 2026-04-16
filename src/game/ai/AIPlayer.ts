// AIPlayer: loads a trained model and generates moves via MCTS

import { ChessNet, type NetConfig, type SerializedWeights, WDL_SIZE } from './ChessNet';
import { runMCTS } from './MCTS';
import type { ChessGameState, Move } from '@/types/chess';

export type AIModelSource = 'trained' | 'preset' | 'untrained';

const DEFAULT_NET_CONFIG: NetConfig = {
  numResBlocks: 6,
  numFilters: 64,
  kernelSize: 3,
  valueHeadSize: 64,
  seReduction: 8,
  learningRate: 0.01,
};

// Detect architecture from weight shapes so we create a matching model
function detectConfigFromWeights(weights: SerializedWeights): NetConfig {
  const firstShape = weights.shapes[0];
  const kernelSize = firstShape[0];
  const numFilters = firstShape[3];

  let resBlockConvs = 0;
  for (const shape of weights.shapes) {
    if (shape.length === 4 && shape[0] === kernelSize && shape[2] === numFilters && shape[3] === numFilters) {
      resBlockConvs++;
    }
  }
  const numResBlocks = Math.floor(resBlockConvs / 2);

  // SE reduction: detect from dense [numFilters, hidden] with hidden < numFilters
  let seReduction = 8;
  for (const shape of weights.shapes) {
    if (shape.length === 2 && shape[0] === numFilters && shape[1] > 0 && shape[1] < numFilters) {
      seReduction = Math.max(1, Math.round(numFilters / shape[1]));
      break;
    }
  }

  // Value head size: first dense [64, vhs] that isn't the [vhs, 3] WDL layer
  let valueHeadSize = 64;
  for (const shape of weights.shapes) {
    if (shape.length === 2 && shape[0] === 64 && shape[1] !== WDL_SIZE) {
      valueHeadSize = shape[1];
      break;
    }
  }

  return { numResBlocks, numFilters, kernelSize, valueHeadSize, seReduction, learningRate: 0.01 };
}

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

  // Load from preset weights (use embedded config or detect from weight shapes)
  static createFromPreset(weights: SerializedWeights, sims: number = 50): AIPlayer {
    const config: NetConfig = weights.config
      ? { ...weights.config, learningRate: 0.01 }
      : detectConfigFromWeights(weights);
    const net = ChessNet.create(config);
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
