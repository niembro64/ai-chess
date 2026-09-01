// AIPlayer: loads trained weights and generates moves via MCTS.
//
// Like ToyPlayer, every move can emit a ToyThought for the Toy Mind
// panel. Sage's real input is the 20-plane tensor, but the panel's
// GAME STATE shows the same 6-plane ±1 view for both bots — derived
// here by re-encoding the position with Toy's encoder (identical to
// collapsing Sage's 12 piece planes to own−opp; the extra 8 planes —
// bias, clocks, castling, en passant — have no visual). The policy /
// value shapes are shared, so the rest of the thought is exact.

import { ChessNet, moveToIndex, type NetConfig, type SerializedWeights, WDL_SIZE } from './ChessNet';
import { runMCTSAsync } from './MCTS';
import { encodeToyBoard } from './ToyNet';
import type { ToyThought } from './ToyPlayer';
import {
  applyMove,
  buildPositionCounts,
  getLegalMoves,
  positionKey,
  posToAlgebraic,
} from '@/game/chess/ChessEngine';
import type { ChessGameState, Move } from '@/types/chess';

// --- Repetition veto -------------------------------------------------
//
// The network's input encoding carries no history, so neither the net
// nor the search can perceive that a position already occurred.
// House rule: a BOT may never move into a board state the game has
// already seen — not even once (humans may; the engine only enforces
// FIDE threefold). Post-search fix that leaves the weights untouched:
// walk down the search's visit-ranked move list and play the best move
// whose resulting position is FRESH. Only when every legal move
// repeats (forced) does the search's own choice stand.

// Pick the best-ranked move that reaches a never-seen position, or
// null (forced — keep the search's choice). Exported for tests.
export function pickFreshMove(
  state: ChessGameState,
  rankedMoves: { move: Move; visits: number }[],
  positionCounts: Map<string, number>,
): Move | null {
  if (rankedMoves.length === 0) return null;
  for (const { move } of rankedMoves) {
    const key = positionKey(applyMove(state, move));
    if ((positionCounts.get(key) ?? 0) === 0) return move;
  }
  return null;
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

export type AIPlayerOptions = {
  // What this net was TRAINED to want ('win' = Sage/Toy, 'lose' =
  // Jester). Determines the search's default goal direction.
  trainedGoal?: 'win' | 'lose';
  // GOAL INVERTED inference mode: pursue the OPPOSITE of the trained
  // goal — implemented as the misère search-selection flip (see
  // MCTSOptions.invertForTurn), never as "pick the lowest prediction"
  // (which shuffles randomly instead of planning). Root priors are
  // flattened when inverting, because the net's trained priors point
  // away from the new goal and would starve the search of the moves it
  // now needs. The value net stays a truthful evaluator throughout.
  goalInverted?: boolean;
  // Search-progress callback for the in-game thinking bar.
  onProgress?: (completed: number, total: number) => void;
};

export class AIPlayer {
  private net: ChessNet;
  private sims: number;
  private onThought: ((t: ToyThought) => void) | null;
  // True when the EFFECTIVE goal (trained goal, possibly inverted by
  // the inference mode) is to lose — drives the misère search flip.
  private seeksLoss: boolean;
  private flattenRootPriors: boolean;
  private onProgress: ((completed: number, total: number) => void) | null;

  private constructor(
    net: ChessNet,
    sims: number,
    onThought: ((t: ToyThought) => void) | null,
    options: AIPlayerOptions,
  ) {
    this.net = net;
    this.sims = sims;
    this.onThought = onThought;
    const trainedToLose = (options.trainedGoal ?? 'win') === 'lose';
    const inverted = options.goalInverted ?? false;
    this.seeksLoss = trainedToLose !== inverted;
    this.flattenRootPriors = inverted;
    this.onProgress = options.onProgress ?? null;
  }

  static create(
    weights: SerializedWeights,
    sims: number = 50,
    onThought?: (t: ToyThought) => void,
    options: AIPlayerOptions = {},
  ): AIPlayer {
    const config: NetConfig = weights.config
      ? { ...weights.config, learningRate: 0.01 }
      : detectConfigFromWeights(weights);
    const net = ChessNet.create(config);
    net.importWeights(weights);
    return new AIPlayer(net, sims, onThought ?? null, options);
  }

  // Terminal observation, same contract as ToyPlayer.observeTerminal:
  // the game ended on the bot's turn — run the forward pass anyway and
  // emit a thought whose legal mask is empty.
  observeTerminal(state: ChessGameState): void {
    if (!this.onThought) return;
    const rootEval = this.net.predict(state);
    const isWhite = state.currentTurn === 'white';
    const legalMask = new Uint8Array(4096);
    for (const m of getLegalMoves(state)) {
      legalMask[moveToIndex(m, isWhite)] = 1; // stays all-zero at mate
    }
    this.onThought({
      planes: encodeToyBoard(state),
      rawPolicy: rootEval.policy,
      visitPolicy: new Float32Array(4096),
      legalMask,
      value: rootEval.value,
      rootValue: rootEval.value,
      chosen: '',
      moves: [],
      blackToMove: !isWhite,
    });
  }

  // Match-play search: no root noise, argmax move selection, and the
  // async runner yields to the event loop so the UI stays responsive.
  // After the search, the repetition veto may bump the choice down the
  // visit ranking (see pickNonRepeatingMove).
  async getMove(state: ChessGameState): Promise<Move> {
    const result = await runMCTSAsync(state, this.net, this.sims, {
      invertForTurn: this.seeksLoss ? state.currentTurn : undefined,
      flattenRootPriors: this.flattenRootPriors,
      onProgress: (done, total) => this.onProgress?.(done, total),
    });
    const counts = buildPositionCounts(state);
    // House rule: bots never repeat a prior board state (see
    // pickFreshMove) — regardless of goal direction or effort.
    const vetoed = pickFreshMove(state, result.rankedMoves, counts);
    const move = vetoed ?? result.move;

    if (this.onThought) {
      const rootEval = this.net.predict(state);
      const isWhite = state.currentTurn === 'white';
      const legalMask = new Uint8Array(4096);
      for (const m of getLegalMoves(state)) {
        legalMask[moveToIndex(m, isWhite)] = 1;
      }
      const totalVisits = result.rankedMoves.reduce((s, r) => s + r.visits, 0) || 1;
      this.onThought({
        planes: encodeToyBoard(state),
        rawPolicy: rootEval.policy,
        visitPolicy: result.policy,
        legalMask,
        value: rootEval.value,
        rootValue: result.rootValue,
        chosen: posToAlgebraic(move.from) + posToAlgebraic(move.to),
        moves: result.rankedMoves.map(r => ({
          uci: posToAlgebraic(r.move.from) + posToAlgebraic(r.move.to),
          share: r.visits / totalVisits,
          index: moveToIndex(r.move, isWhite),
        })),
        blackToMove: !isWhite,
      });
    }
    return move;
  }

  dispose(): void {
    this.net.dispose();
  }
}
