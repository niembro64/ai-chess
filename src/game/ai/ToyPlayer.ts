// ToyPlayer: plays with the Toy net and narrates its thinking.
//
// Same skeleton as AIPlayer (MCTS + repetition veto) but every move
// also emits a ToyThought — the full record the Toy Mind panel
// visualizes: the exact 8x8x6 tensor the net saw, its raw policy
// before masking, the visit distribution after search, and the value.

import { runMCTSAsync } from './MCTS';
import { ToyNet, encodeToyBoard, isToyWeights, type ToySerializedWeights } from './ToyNet';
import { moveToIndex } from './ChessNet';
import { pickNonRepeatingMove } from './AIPlayer';
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
  topMoves: { uci: string; share: number }[];       // top-5 by visits
  // True when the tensor is in black's rotated frame (Toy plays black).
  blackToMove: boolean;
};

export class ToyPlayer {
  private net: ToyNet;
  private sims: number;
  private onThought: ((t: ToyThought) => void) | null;

  private constructor(net: ToyNet, sims: number, onThought: ((t: ToyThought) => void) | null) {
    this.net = net;
    this.sims = sims;
    this.onThought = onThought;
  }

  static create(
    weights: unknown,
    sims: number,
    onThought?: (t: ToyThought) => void,
  ): ToyPlayer {
    if (!isToyWeights(weights)) {
      throw new Error('Not a toy-v1 weight file');
    }
    return new ToyPlayer(ToyNet.create(weights as ToySerializedWeights), sims, onThought ?? null);
  }

  async getMove(state: ChessGameState): Promise<Move> {
    const planes = encodeToyBoard(state);
    const [rootEval] = this.net.predictBatch([planes]);

    const result = await runMCTSAsync(state, this.net, this.sims, {
      encode: encodeToyBoard,
    });

    const counts = buildPositionCounts(state);
    const vetoed = pickNonRepeatingMove(state, result.rankedMoves, counts, result.rootValue);
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
        topMoves: result.rankedMoves.slice(0, 5).map(r => ({
          uci: posToAlgebraic(r.move.from) + posToAlgebraic(r.move.to),
          share: r.visits / totalVisits,
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
