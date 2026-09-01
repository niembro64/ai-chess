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

export type AIPlayerOptions = {
  // Misère ("Jester") mode: the bot plays to LOSE. Search selection
  // inverts at its own plies (see MCTSOptions.invertForTurn) and the
  // repetition veto flips (a draw SPOILS a loss in progress). The value
  // net stays a truthful "who is winning" estimator throughout.
  jester?: boolean;
  // See MCTSOptions.flattenRootPriors — needed while Jester runs on
  // winner weights.
  flattenRootPriors?: boolean;
};

export class AIPlayer {
  private net: ChessNet;
  private sims: number;
  private onThought: ((t: ToyThought) => void) | null;
  private jester: boolean;
  private flattenRootPriors: boolean;

  private constructor(
    net: ChessNet,
    sims: number,
    onThought: ((t: ToyThought) => void) | null,
    options: AIPlayerOptions,
  ) {
    this.net = net;
    this.sims = sims;
    this.onThought = onThought;
    this.jester = options.jester ?? false;
    this.flattenRootPriors = options.flattenRootPriors ?? false;
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
      invertForTurn: this.jester ? state.currentTurn : undefined,
      flattenRootPriors: this.flattenRootPriors,
    });
    const counts = buildPositionCounts(state);
    // Jester flips the repetition veto by negating the root value:
    // when it is successfully losing (very negative truthful value) a
    // repetition draw would SPOIL the loss — avoid it strictly; when it
    // is accidentally winning, a draw is an improvement — allow it.
    const vetoValue = this.jester ? -result.rootValue : result.rootValue;
    const vetoed = pickNonRepeatingMove(
      state, result.rankedMoves, counts, vetoValue,
    );
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
