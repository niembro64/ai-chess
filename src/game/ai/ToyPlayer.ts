// ToyPlayer: plays with the Toy net and narrates its thinking.
//
// Same skeleton as AIPlayer (MCTS + repetition veto) but every move
// also emits a ToyThought — the full record the Toy Mind panel
// visualizes: the exact 8x8x6 tensor the net saw, its raw policy
// before masking, the visit distribution after search, and the value.

import { runMCTSAsync } from './MCTS';
import { ToyNet, encodeToyBoard, isToyWeights, type ToySerializedWeights } from './ToyNet';
import { moveToIndex } from './ChessNet';
import { pickFreshMove, rankMovesByPolicy } from './AIPlayer';
import {
  buildPositionCounts,
  getLegalMoves,
  posToAlgebraic,
} from '@/game/chess/ChessEngine';
import type { ChessGameState, Move } from '@/types/chess';

export type ToyThought = {
  // The exact input tensor (8*8*6 flat, mover perspective, ±1).
  planes: Float32Array;
  // Net's instant opinion: softmax over all 4096, BEFORE legal masking.
  rawPolicy: Float32Array;
  // 1 = legal in this position (by policy index), for the illegal-mass tint.
  legalMask: Uint8Array;
  // Scalar value in [-1, 1] from the mover's (Toy's) perspective.
  value: number;
  rootValue: number;
  chosen: string;                                   // e.g. "e7e5"
  // EVERY legal move in the bot's own ranking — highest policy
  // probability first, or LOWEST first when it is asked for the
  // opposite of its trained goal, so the move it plays is always the
  // first entry. `index` is the move's slot in the 4096 policy vector,
  // which the panel uses to color each row like its policy-grid cell
  // and to anchor the leader line.
  moves: { uci: string; index: number }[];
  // True when the tensor is in black's rotated frame (Toy plays black).
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
    const legalMask = new Uint8Array(4096);
    for (const m of getLegalMoves(state)) {
      legalMask[moveToIndex(m, isWhite)] = 1; // stays all-zero at mate
    }
    this.onThought({
      planes,
      rawPolicy: rootEval.policy,
      legalMask,
      value: rootEval.value,
      rootValue: rootEval.value,
      chosen: '',
      moves: [],
      blackToMove: !isWhite,
    });
  }

  async getMove(state: ChessGameState): Promise<Move> {
    const planes = encodeToyBoard(state);
    const [rootEval] = this.net.predictBatch([planes]);

    const isWhite = state.currentTurn === 'white';
    const legal = getLegalMoves(state);

    // LOW effort plays straight off the policy head; MEDIUM and HIGH
    // hand the decision to the search (see AIPlayer.getMove).
    let ranked: { move: Move; index: number }[];
    let rootValue = rootEval.value;
    if (this.sims > 0) {
      const result = await runMCTSAsync(state, this.net, this.sims, {
        encode: encodeToyBoard,
        invertForTurn: this.goalInverted ? state.currentTurn : undefined,
        flattenRootPriors: this.goalInverted,
        onProgress: (done, total) => this.onProgress?.(done, total),
      });
      rootValue = result.rootValue;
      ranked = result.rankedMoves.map(r => ({
        move: r.move,
        index: moveToIndex(r.move, isWhite),
      }));
    } else {
      ranked = rankMovesByPolicy(legal, rootEval.policy, isWhite, this.goalInverted);
    }

    const counts = buildPositionCounts(state);
    const move = pickFreshMove(state, ranked, counts) ?? ranked[0].move;

    if (this.onThought) {
      const legalMask = new Uint8Array(4096);
      for (const m of legal) legalMask[moveToIndex(m, isWhite)] = 1;
      this.onThought({
        planes,
        rawPolicy: rootEval.policy,
        legalMask,
        value: rootEval.value,
        rootValue,
        chosen: posToAlgebraic(move.from) + posToAlgebraic(move.to),
        moves: ranked.map(r => ({
          uci: posToAlgebraic(r.move.from) + posToAlgebraic(r.move.to),
          index: r.index,
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
