// ToyPlayer: plays with the Toy net and narrates its thinking.
//
// Same skeleton as AIPlayer (MCTS + repetition veto) but every move
// also emits a ToyThought — the full record the Toy Mind panel
// visualizes: the exact 8x8x6 tensor the net saw, its raw policy
// before masking, the visit distribution after search, and the value.

import { runMCTSAsync } from './MCTS';
import { ToyNet, encodeToyBoard, isToyWeights, type ToySerializedWeights } from './ToyNet';
import {
  candidateMoves,
  markBlockedAndPick,
  rankByDistribution,
  uciForIndex,
} from './AIPlayer';
import {
  buildOwnSideKeys,
  getLegalMoves,
  posToAlgebraic,
} from '@/game/chess/ChessEngine';
import type { ChessGameState, Move, PieceColor } from '@/types/chess';

export type ToyThought = {
  // The exact input tensor (8*8*6 flat, mover perspective, ±1).
  planes: Float32Array;
  // The EFFECTIVE distribution the bot decided from: the raw policy
  // head at LOW effort, the search's visit shares at MEDIUM/HIGH.
  distribution: Float32Array;
  // The network's raw policy head, always — this is what the POLICY
  // HEAD grid paints, so that visual stays a picture of the net itself.
  rawPolicy: Float32Array;
  // Every policy slot, legal and illegal interleaved, ordered by
  // `distribution` (highest first). Identical whether the bot was asked
  // for its trained goal or the opposite; only `chosen` changes.
  // `blocked` marks a legal entry the no-repeat house rule ruled out
  // for this move, so the list stays the candidate pool the bot chose
  // from rather than showing rows that were never available.
  entries: {
    uci: string;
    index: number;
    p: number;
    legal: boolean;
    blocked?: boolean;
  }[];
  // Scalar value in [-1, 1] from the mover's perspective.
  value: number;
  rootValue: number;
  chosen: string;                                   // e.g. "e7e5"
  // True when the tensor is in black's rotated frame.
  blackToMove: boolean;
};

export type ToyPlayerOptions = {
  // GOAL INVERTED inference mode — same semantics as AIPlayerOptions
  // (Toy's trained goal is always 'win').
  goalInverted?: boolean;
  // Search-progress callback for the in-game thinking bar.
  onProgress?: (completed: number, total: number) => void;
};

export class ToyPlayer {
  private net: ToyNet;
  private sims: number;
  private onThought: ((t: ToyThought) => void) | null;
  private goalInverted: boolean;
  private onProgress: ((completed: number, total: number) => void) | null;

  private constructor(
    net: ToyNet,
    sims: number,
    onThought: ((t: ToyThought) => void) | null,
    options: ToyPlayerOptions,
  ) {
    this.net = net;
    this.sims = sims;
    this.onThought = onThought;
    this.goalInverted = options.goalInverted ?? false;
    this.onProgress = options.onProgress ?? null;
  }

  static create(
    weights: unknown,
    sims: number,
    onThought?: (t: ToyThought) => void,
    options: ToyPlayerOptions = {},
  ): ToyPlayer {
    if (!isToyWeights(weights)) {
      throw new Error('Not a toy-v1 weight file');
    }
    return new ToyPlayer(
      ToyNet.create(weights as ToySerializedWeights), sims, onThought ?? null, options,
    );
  }

  // Terminal observation: the game just ended on Toy's turn (it got
  // checkmated / stalemated / the draw landed on its move). There is no
  // move to pick, but the net can still LOOK at the position — run the
  // forward pass and emit a thought where every move is illegal: the
  // legal mask is empty, the whole policy board renders red, and the
  // SEARCH list is empty. Pedagogically great: you see exactly what a
  // mated position looks like to the network (and whether its value
  // head has learned to recognize doom).
  observeTerminal(state: ChessGameState): void {
    if (!this.onThought) return;
    const planes = encodeToyBoard(state);
    const [rootEval] = this.net.predictBatch([planes]);
    const isWhite = state.currentTurn === 'white';
    const entries = rankByDistribution(rootEval.policy, getLegalMoves(state), isWhite);
    this.onThought({
      planes,
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

  async getMove(state: ChessGameState): Promise<Move> {
    const planes = encodeToyBoard(state);
    const [rootEval] = this.net.predictBatch([planes]);
    const isWhite = state.currentTurn === 'white';
    const color: PieceColor = isWhite ? 'white' : 'black';
    const legal = getLegalMoves(state);

    let distribution = rootEval.policy;
    let rootValue = rootEval.value;
    if (this.sims > 0) {
      const result = await runMCTSAsync(state, this.net, this.sims, {
        encode: encodeToyBoard,
        onProgress: (done, total) => this.onProgress?.(done, total),
      });
      distribution = result.policy;
      rootValue = result.rootValue;
    }

    const entries = rankByDistribution(distribution, legal, isWhite);
    const candidates = candidateMoves(entries, this.goalInverted);
    const seenOwnSide = buildOwnSideKeys(state, color);
    const move = markBlockedAndPick(state, entries, candidates, seenOwnSide, color);

    if (this.onThought) {
      this.onThought({
        planes,
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
