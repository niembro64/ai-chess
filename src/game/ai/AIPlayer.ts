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
  buildOwnSideKeys,
  getLegalMoves,
  ownSideKey,
  posToAlgebraic,
} from '@/game/chess/ChessEngine';
import type { ChessGameState, Move, PieceColor } from '@/types/chess';

// --- Bot house rule: never repeat your own army's arrangement -------
//
// Stricter than the FIDE rules a human plays under. The engine still

// --- The bot's move ranking ------------------------------------------
//
// Every entry of the 4096-wide policy space, ordered purely by the
// EFFECTIVE distribution — the raw policy head at LOW effort, the
// search's visit shares at MEDIUM and HIGH. The ordering is the same
// whether we ask the bot for its trained goal or the opposite; only
// which entry gets played changes:
//
//   pursuing its trained goal -> the highest-probability LEGAL entry
//   asked for the opposite    -> the lowest-probability LEGAL entry
//
// Jester's weights already rate self-mating moves highest, so "asked to
// lose" takes its top entry while "asked to win" takes its bottom one;
// Sage is the mirror image.
export type RankedEntry = {
  move: Move | null;
  index: number;
  p: number;
  legal: boolean;
  // Legal, but the no-repeat house rule ruled it out for THIS move.
  // Carried on the entry so the visible list can mark it: a skipped top
  // row otherwise looks exactly like the bot ignoring its own top
  // prediction, which is how this surfaced as a bug report.
  blocked?: boolean;
};

// Algebraic name for any policy slot, legal or not, in the net's frame
// (rotated 180 degrees when black is to move).
export function uciForIndex(index: number, isWhite: boolean): string {
  const square = (sq: number) => {
    const r = isWhite ? Math.floor(sq / 8) : 7 - Math.floor(sq / 8);
    const f = isWhite ? sq % 8 : 7 - (sq % 8);
    return String.fromCharCode(97 + f) + String(8 - r);
  };
  return square(Math.floor(index / 64)) + square(index % 64);
}

export function rankByDistribution(
  distribution: Float32Array,
  legalMoves: Move[],
  isWhite: boolean,
  searchedMoves: { move: Move; visits: number }[] = [],
): RankedEntry[] {
  const byIndex = new Map<number, Move>();
  for (const move of legalMoves) {
    const index = moveToIndex(move, isWhite);
    if (!byIndex.has(index)) byIndex.set(index, move); // underpromotions collapse
  }
  // Keep the network shape while choosing each promotion by its search visits.
  for (const { move } of [...searchedMoves].sort((a, b) => a.visits - b.visits)) {
    byIndex.set(moveToIndex(move, isWhite), move);
  }
  const entries: RankedEntry[] = [];
  for (let index = 0; index < distribution.length; index++) {
    const move = byIndex.get(index) ?? null;
    entries.push({ move, index, p: distribution[index], legal: move !== null });
  }
  // The index tie-break is load-bearing: at 100 simulations spread
  // over ~30 moves several moves routinely share an identical visit
  // share, and an unspecified tie order would let the played move sit
  // below a row holding the same probability. Selection reads this
  // same array (see candidateMoves), so display and choice cannot
  // disagree.
  entries.sort((a, b) => b.p - a.p || a.index - b.index);
  return entries;
}

// The legal entries, in the order this bot would choose them.
export function candidateMoves(
  entries: RankedEntry[],
  lowestFirst: boolean,
): { move: Move; index: number }[] {
  const legal = entries
    .filter(e => e.move !== null)
    .map(e => ({ move: e.move as Move, index: e.index }));
  return lowestFirst ? legal.reverse() : legal;
}

// Mark the entries the no-repeat rule takes off the table, so the list
// the player reads is the list the bot chose from. Mutates in place and
// returns the played move: one pass, one source of truth.
//
// If EVERY legal move repeats, nothing is marked — the rule yields
// rather than leaving the bot with no move.
export function markBlockedAndPick(
  state: ChessGameState,
  entries: RankedEntry[],
  candidates: { move: Move; index: number }[],
  seenOwnSide: Set<string>,
  color: PieceColor,
): Move {
  const blocked = new Set<number>();
  for (const c of candidates) {
    if (seenOwnSide.has(ownSideKey(applyMove(state, c.move), color))) {
      blocked.add(c.index);
    }
  }
  if (blocked.size === candidates.length) blocked.clear();
  for (const e of entries) e.blocked = blocked.has(e.index);
  const pick = candidates.find(c => !blocked.has(c.index)) ?? candidates[0];
  return pick.move;
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
  // The SEARCH always runs in the direction this model's weights were
  // trained for, never in the direction we are currently asking of it.
  // That keeps the distribution — and therefore the whole right-hand
  // list — identical whether the bot is asked to win or to lose; only
  // which entry it plays changes.
  private searchSeeksLoss: boolean;
  // True when we are asking the OPPOSITE of what the weights learned,
  // which is exactly when the lowest-probability move is wanted.
  private goalInverted: boolean;
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
    this.searchSeeksLoss = (options.trainedGoal ?? 'win') === 'lose';
    this.goalInverted = options.goalInverted ?? false;
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
    // No legal moves at a terminal position, so every entry is illegal.
    const entries = rankByDistribution(rootEval.policy, getLegalMoves(state), isWhite);
    this.onThought({
      planes: encodeToyBoard(state),
      distribution: rootEval.policy,
      rawPolicy: rootEval.policy,
      entries: entries.map(e => ({
        uci: e.move
          ? posToAlgebraic(e.move.from) + posToAlgebraic(e.move.to)
          : uciForIndex(e.index, isWhite),
        index: e.index,
        p: e.p,
        legal: e.legal,
      })),
      value: rootEval.value,
      rootValue: rootEval.value,
      chosen: '',
      blackToMove: !isWhite,
    });
  }

  // Match-play search: no root noise, argmax move selection, and the
  // async runner yields to the event loop so the UI stays responsive.
  // After the search, the repetition veto may bump the choice down the
  // visit ranking (see pickNonRepeatingMove).
  async getMove(state: ChessGameState): Promise<Move> {
    const rootEval = this.net.predict(state);
    const isWhite = state.currentTurn === 'white';
    const color: PieceColor = isWhite ? 'white' : 'black';
    const legal = getLegalMoves(state);

    // The EFFECTIVE distribution: the raw policy head at LOW effort
    // (no search at all), the search's visit shares at MEDIUM and HIGH.
    // Whatever produced it is what the list shows and what the pick
    // reads from.
    let distribution = rootEval.policy;
    let rootValue = rootEval.value;
    let searchedMoves: { move: Move; visits: number }[] = [];
    if (this.sims > 0) {
      const result = await runMCTSAsync(state, this.net, this.sims, {
        invertForTurn: this.searchSeeksLoss ? 'both' : undefined,
        onProgress: (done, total) => this.onProgress?.(done, total),
      });
      distribution = result.policy;
      rootValue = result.rootValue;
      searchedMoves = result.rankedMoves;
    }

    const entries = rankByDistribution(distribution, legal, isWhite, searchedMoves);
    const candidates = candidateMoves(entries, this.goalInverted);
    const seenOwnSide = buildOwnSideKeys(state, color);
    // In competitive inverted chess a legal return move can be essential
    // to a forced selfmate. MCTS handles actual repetition draws in-tree.
    const move = this.searchSeeksLoss && !this.goalInverted
      ? candidates[0].move
      : markBlockedAndPick(state, entries, candidates, seenOwnSide, color);

    if (this.onThought) {
      this.onThought({
        planes: encodeToyBoard(state),
        distribution,
        rawPolicy: rootEval.policy,
        entries: entries.map(e => ({
          uci: e.move
            ? posToAlgebraic(e.move.from) + posToAlgebraic(e.move.to)
            : uciForIndex(e.index, isWhite),
          index: e.index,
          p: e.p,
          legal: e.legal,
          blocked: e.blocked === true,
        })),
        value: rootEval.value,
        rootValue,
        chosen: posToAlgebraic(move.from) + posToAlgebraic(move.to),
        blackToMove: !isWhite,
      });
    }
    return move;
  }

  dispose(): void {
    this.net.dispose();
  }
}
