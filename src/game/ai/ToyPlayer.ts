// ToyPlayer: plays with the Toy net and narrates its thinking.
//
// Same skeleton as AIPlayer (MCTS + repetition veto) but every move
// also emits a ToyThought — the full record the Toy Mind panel
// visualizes: the exact 8x8x6 tensor the net saw, its raw policy
// before masking, the visit distribution after search, and the value.

import { runMCTSAsync } from './MCTS';
import { ToyNet, encodeToyBoard, isToyWeights, type ToySerializedWeights } from './ToyNet';
import { moveToIndex } from './ChessNet';
import { pickFreshMove } from './AIPlayer';
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
  // After 128-sim MCTS: visit distribution (only legal moves have mass).
  visitPolicy: Float32Array;
  // 1 = legal in this position (by policy index), for the illegal-mass tint.
  legalMask: Uint8Array;
  // Scalar value in [-1, 1] from the mover's (Toy's) perspective.
  value: number;
  rootValue: number;
  chosen: string;                                   // e.g. "e7e5"
  // EVERY legal move, ordered by search visits (best first). `index`
  // is the move's slot in the 4096 policy vector — the panel uses it
  // to color each list entry exactly like its policy-grid cell and to
  // anchor the leader line.
  moves: { uci: string; share: number; index: number }[];
  // True when the tensor is in black's rotated frame (Toy plays black).
  blackToMove: boolean;
};

export type ToyPlayerOptions = {
  // GOAL INVERTED inference mode + move temperature — same semantics
  // as AIPlayerOptions (Toy's trained goal is always 'win').
  goalInverted?: boolean;
  temperature?: number;
};

export class ToyPlayer {
  private net: ToyNet;
  private sims: number;
  private onThought: ((t: ToyThought) => void) | null;
  private goalInverted: boolean;
  private temperature: number;

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
    this.temperature = options.temperature ?? 0;
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
      visitPolicy: new Float32Array(4096),
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

    const result = await runMCTSAsync(state, this.net, this.sims, {
      encode: encodeToyBoard,
      invertForTurn: this.goalInverted ? state.currentTurn : undefined,
      flattenRootPriors: this.goalInverted,
      moveTemperature: this.temperature,
    });

    const counts = buildPositionCounts(state);
    // House rule: bots never repeat a prior board state.
    const vetoed = pickFreshMove(state, result.rankedMoves, counts);
    const move = vetoed ?? result.move;

    if (this.onThought) {
      const isWhite = state.currentTurn === 'white';
      const legalMask = new Uint8Array(4096);
      for (const m of getLegalMoves(state)) {
        legalMask[moveToIndex(m, isWhite)] = 1;
      }
      const totalVisits = result.rankedMoves.reduce((s, r) => s + r.visits, 0) || 1;
      this.onThought({
        planes,
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
