// AIPlayer: loads trained weights and generates moves via MCTS.

import { ChessNet, type NetConfig, type SerializedWeights, WDL_SIZE } from './ChessNet';
import { runMCTSAsync } from './MCTS';
import type { ChessGameState, Move } from '@/types/chess';

// Figure out the ChessNet architecture from the weight tensor shapes so we
// can rebuild a matching model. Used only if the serialized JSON lacks an
// explicit `config` field (older dumps).
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

  let seReduction = 8;
  for (const shape of weights.shapes) {
    if (shape.length === 2 && shape[0] === numFilters && shape[1] > 0 && shape[1] < numFilters) {
      seReduction = Math.max(1, Math.round(numFilters / shape[1]));
      break;
    }
  }

  // value_fc2 is the only 2-D tensor whose output dim is WDL_SIZE; its
  // input dim is the value head width. (Matching on [64, !=WDL] instead
  // would collide with SE fc1 on 64-filter nets.)
  let valueHeadSize = 64;
  for (const shape of weights.shapes) {
    if (shape.length === 2 && shape[1] === WDL_SIZE) {
      valueHeadSize = shape[0];
      break;
    }
  }

  return { numResBlocks, numFilters, kernelSize, valueHeadSize, seReduction, learningRate: 0.01 };
}

export class AIPlayer {
  private net: ChessNet;
  private sims: number;

  private constructor(net: ChessNet, sims: number) {
    this.net = net;
    this.sims = sims;
  }

  static create(weights: SerializedWeights, sims: number = 50): AIPlayer {
    const config: NetConfig = weights.config
      ? { ...weights.config, learningRate: 0.01 }
      : detectConfigFromWeights(weights);
    const net = ChessNet.create(config);
    net.importWeights(weights);
    return new AIPlayer(net, sims);
  }

  // Match-play search: no root noise, argmax move selection, and the
  // async runner yields to the event loop so the UI stays responsive.
  async getMove(state: ChessGameState): Promise<Move> {
    const result = await runMCTSAsync(state, this.net, this.sims);
    return result.move;
  }

  dispose(): void {
    this.net.dispose();
  }
}
