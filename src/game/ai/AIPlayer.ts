// AIPlayer: loads trained weights and generates moves via MCTS.

import { ChessNet, type NetConfig, type SerializedWeights, WDL_SIZE } from './ChessNet';
import { runMCTSAsync } from './MCTS';
import {
  applyMove,
  buildPositionCounts,
  positionKey,
} from '@/game/chess/ChessEngine';
import type { ChessGameState, Move } from '@/types/chess';

// --- Repetition veto -------------------------------------------------
//
// The network's input encoding carries no history, so neither the net
// nor the search can perceive threefold repetition — a winning bot
// would happily shuffle into the auto-draw the game rules now enforce.
// Post-search fix that leaves the weights untouched: walk down the
// search's visit-ranked move list and play the best move that does NOT
// walk into a repetition, instead of blindly playing the top move.
//
// Below this root value the bot is losing badly enough that a
// repetition draw SAVES half a point — the veto turns off entirely so
// the search's choice (which may repeat) stands.
const REPETITION_OK_BELOW = -0.3;
// Above this root value the bot is clearly winning and shouldn't even
// allow a SECOND occurrence — drifting to the brink lets the opponent
// spring the third. Between the two thresholds only the game-ending
// third occurrence is vetoed.
const STRICT_AVOID_ABOVE = 0.3;

// Pick the best-ranked move that avoids repetition, or null to keep the
// search's own choice. Exported for tests.
export function pickNonRepeatingMove(
  state: ChessGameState,
  rankedMoves: { move: Move; visits: number }[],
  positionCounts: Map<string, number>,
  rootValue: number,
): Move | null {
  if (rankedMoves.length === 0) return null;
  if (rootValue <= REPETITION_OK_BELOW) return null;

  // Occurrence count the resulting position would REACH if played.
  const resulting = rankedMoves.map(({ move }) => {
    const key = positionKey(applyMove(state, move));
    return { move, wouldReach: (positionCounts.get(key) ?? 0) + 1 };
  });

  if (rootValue >= STRICT_AVOID_ABOVE) {
    const fresh = resulting.find(r => r.wouldReach < 2);
    if (fresh) return fresh.move;
  }
  const safe = resulting.find(r => r.wouldReach < 3);
  // Every move creates the third occurrence — the draw is forced; keep
  // the search's choice.
  return safe ? safe.move : null;
}

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
  // After the search, the repetition veto may bump the choice down the
  // visit ranking (see pickNonRepeatingMove).
  async getMove(state: ChessGameState): Promise<Move> {
    const result = await runMCTSAsync(state, this.net, this.sims);
    const counts = buildPositionCounts(state);
    const vetoed = pickNonRepeatingMove(
      state, result.rankedMoves, counts, result.rootValue,
    );
    return vetoed ?? result.move;
  }

  dispose(): void {
    this.net.dispose();
  }
}
