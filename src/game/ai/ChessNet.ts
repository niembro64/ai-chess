// Neural network for chess: ResNet with policy + value heads
// Input: 8x8x18 board planes (channels-last)
// Policy output: 4096 probabilities (64 from-squares × 64 to-squares)
// Value output: scalar in [-1, 1]

import * as tf from '@tensorflow/tfjs';
import type { ChessGameState, PieceType, Position, Move } from '@/types/chess';

// --- Board encoding ---

const PIECE_TYPES: PieceType[] = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn'];
const NUM_PLANES = 18;
const BOARD_FLOATS = 8 * 8 * NUM_PLANES; // 1152
export const POLICY_SIZE = 4096; // 64 * 64

export function encodeBoard(state: ChessGameState): Float32Array {
  const data = new Float32Array(BOARD_FLOATS);
  const isWhite = state.currentTurn === 'white';

  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const boardRank = isWhite ? rank : 7 - rank;
      const boardFile = isWhite ? file : 7 - file;
      const piece = state.board[boardRank][boardFile];

      if (piece) {
        const isOwn = piece.color === state.currentTurn;
        const typeIdx = PIECE_TYPES.indexOf(piece.type);
        const channel = isOwn ? typeIdx : 6 + typeIdx;
        data[(rank * 8 + file) * NUM_PLANES + channel] = 1;
      }
    }
  }

  // Channel 12: constant ones (bias plane)
  for (let i = 0; i < 64; i++) {
    data[i * NUM_PLANES + 12] = 1;
  }

  // Channels 13-16: castling rights
  const cr = state.castlingRights;
  const castling = [
    isWhite ? cr.whiteKingside : cr.blackKingside,
    isWhite ? cr.whiteQueenside : cr.blackQueenside,
    isWhite ? cr.blackKingside : cr.whiteKingside,
    isWhite ? cr.blackQueenside : cr.whiteQueenside,
  ];
  for (let c = 0; c < 4; c++) {
    if (castling[c]) {
      for (let i = 0; i < 64; i++) {
        data[i * NUM_PLANES + 13 + c] = 1;
      }
    }
  }

  // Channel 17: en passant target
  if (state.enPassantTarget) {
    const epRank = isWhite ? state.enPassantTarget.rank : 7 - state.enPassantTarget.rank;
    const epFile = isWhite ? state.enPassantTarget.file : 7 - state.enPassantTarget.file;
    data[(epRank * 8 + epFile) * NUM_PLANES + 17] = 1;
  }

  return data;
}

// --- Move encoding ---

export function moveToIndex(move: Move, isWhite: boolean): number {
  let fromRank = move.from.rank, fromFile = move.from.file;
  let toRank = move.to.rank, toFile = move.to.file;
  if (!isWhite) {
    fromRank = 7 - fromRank; fromFile = 7 - fromFile;
    toRank = 7 - toRank; toFile = 7 - toFile;
  }
  return (fromRank * 8 + fromFile) * 64 + (toRank * 8 + toFile);
}

export function indexToPosition(index: number, isWhite: boolean): { from: Position; to: Position } {
  const fromSquare = Math.floor(index / 64);
  const toSquare = index % 64;
  let fromRank = Math.floor(fromSquare / 8), fromFile = fromSquare % 8;
  let toRank = Math.floor(toSquare / 8), toFile = toSquare % 8;
  if (!isWhite) {
    fromRank = 7 - fromRank; fromFile = 7 - fromFile;
    toRank = 7 - toRank; toFile = 7 - toFile;
  }
  return {
    from: { rank: fromRank, file: fromFile },
    to: { rank: toRank, file: toFile },
  };
}

// --- Neural Network ---

export type NetConfig = {
  numResBlocks: number;
  numFilters: number;
  learningRate: number;
};

export class ChessNet {
  model: tf.LayersModel;
  private optimizer: tf.Optimizer;

  private constructor(model: tf.LayersModel, lr: number) {
    this.model = model;
    this.optimizer = tf.train.adam(lr);
  }

  static create(config: NetConfig): ChessNet {
    const model = buildModel(config.numResBlocks, config.numFilters);
    return new ChessNet(model, config.learningRate);
  }

  getParamCount(): number {
    return this.model.countParams();
  }

  // Single-position predict (for gameplay / single MCTS). Uses dataSync for speed.
  predict(state: ChessGameState): { policy: Float32Array; value: number } {
    const boardData = encodeBoard(state);
    const input = tf.tensor4d(boardData, [1, 8, 8, NUM_PLANES]);
    const outputs = this.model.predict(input) as tf.Tensor[];
    const policyData = outputs[0].dataSync() as Float32Array;
    const valueData = outputs[1].dataSync() as Float32Array;
    input.dispose();
    outputs[0].dispose();
    outputs[1].dispose();
    return {
      policy: new Float32Array(policyData),
      value: valueData[0],
    };
  }

  // Batched predict: evaluate multiple positions in one GPU call.
  predictBatch(boards: Float32Array[]): Array<{ policy: Float32Array; value: number }> {
    const batchSize = boards.length;
    if (batchSize === 0) return [];

    // Pack into a single buffer
    const buf = new Float32Array(batchSize * BOARD_FLOATS);
    for (let i = 0; i < batchSize; i++) {
      buf.set(boards[i], i * BOARD_FLOATS);
    }

    const input = tf.tensor4d(buf, [batchSize, 8, 8, NUM_PLANES]);
    const outputs = this.model.predict(input) as tf.Tensor[];
    const allPolicy = outputs[0].dataSync() as Float32Array;
    const allValue = outputs[1].dataSync() as Float32Array;
    input.dispose();
    outputs[0].dispose();
    outputs[1].dispose();

    const results: Array<{ policy: Float32Array; value: number }> = [];
    for (let i = 0; i < batchSize; i++) {
      results.push({
        policy: allPolicy.slice(i * POLICY_SIZE, (i + 1) * POLICY_SIZE),
        value: allValue[i],
      });
    }
    return results;
  }

  async train(
    boards: Float32Array[],
    policies: Float32Array[],
    values: number[],
  ): Promise<{ policyLoss: number; valueLoss: number; totalLoss: number }> {
    const batchSize = boards.length;

    const boardBuf = new Float32Array(batchSize * BOARD_FLOATS);
    const policyBuf = new Float32Array(batchSize * POLICY_SIZE);
    const valueBuf = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      boardBuf.set(boards[i], i * BOARD_FLOATS);
      policyBuf.set(policies[i], i * POLICY_SIZE);
      valueBuf[i] = values[i];
    }

    const inputTensor = tf.tensor4d(boardBuf, [batchSize, 8, 8, NUM_PLANES]);
    const policyTarget = tf.tensor2d(policyBuf, [batchSize, POLICY_SIZE]);
    const valueTarget = tf.tensor2d(valueBuf, [batchSize, 1]);

    let policyLoss = 0;
    let valueLoss = 0;

    const totalLossTensor = this.optimizer.minimize(() => {
      const outputs = this.model.apply(inputTensor, { training: true }) as tf.Tensor[];
      const pLoss = tf.neg(tf.sum(tf.mul(policyTarget, tf.log(tf.add(outputs[0], 1e-8))))).div(tf.scalar(batchSize));
      const vLoss = tf.losses.meanSquaredError(valueTarget, outputs[1]) as tf.Scalar;
      const total = tf.add(pLoss, vLoss) as tf.Scalar;
      policyLoss = pLoss.dataSync()[0];
      valueLoss = vLoss.dataSync()[0];
      return total;
    }, true) as tf.Scalar;

    const totalLoss = totalLossTensor!.dataSync()[0];
    totalLossTensor!.dispose();
    inputTensor.dispose();
    policyTarget.dispose();
    valueTarget.dispose();

    return { policyLoss, valueLoss, totalLoss };
  }

  async save(): Promise<void> {
    await this.model.save('indexeddb://chess-net');
  }

  async load(): Promise<boolean> {
    try {
      const loaded = await tf.loadLayersModel('indexeddb://chess-net');
      this.model.dispose();
      this.model = loaded;
      return true;
    } catch {
      return false;
    }
  }

  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}

// --- Model architecture ---

function buildModel(numResBlocks: number, numFilters: number): tf.LayersModel {
  const input = tf.input({ shape: [8, 8, NUM_PLANES] });

  let x = conv2d(input, numFilters, 3);
  for (let i = 0; i < numResBlocks; i++) {
    x = residualBlock(x, numFilters);
  }

  // Policy head
  let ph = tf.layers.conv2d({ filters: 2, kernelSize: 1, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  ph = tf.layers.batchNormalization().apply(ph) as tf.SymbolicTensor;
  ph = tf.layers.activation({ activation: 'relu' }).apply(ph) as tf.SymbolicTensor;
  ph = tf.layers.flatten().apply(ph) as tf.SymbolicTensor;
  const policyOutput = tf.layers.dense({
    units: POLICY_SIZE, activation: 'softmax', name: 'policy_output',
  }).apply(ph) as tf.SymbolicTensor;

  // Value head
  let vh = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  vh = tf.layers.batchNormalization().apply(vh) as tf.SymbolicTensor;
  vh = tf.layers.activation({ activation: 'relu' }).apply(vh) as tf.SymbolicTensor;
  vh = tf.layers.flatten().apply(vh) as tf.SymbolicTensor;
  vh = tf.layers.dense({ units: 32, activation: 'relu' }).apply(vh) as tf.SymbolicTensor;
  const valueOutput = tf.layers.dense({
    units: 1, activation: 'tanh', name: 'value_output',
  }).apply(vh) as tf.SymbolicTensor;

  return tf.model({ inputs: input, outputs: [policyOutput, valueOutput] });
}

function conv2d(x: tf.SymbolicTensor, filters: number, kernelSize: number): tf.SymbolicTensor {
  let out = tf.layers.conv2d({ filters, kernelSize, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  out = tf.layers.batchNormalization().apply(out) as tf.SymbolicTensor;
  out = tf.layers.activation({ activation: 'relu' }).apply(out) as tf.SymbolicTensor;
  return out;
}

function residualBlock(x: tf.SymbolicTensor, filters: number): tf.SymbolicTensor {
  let out = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  out = tf.layers.batchNormalization().apply(out) as tf.SymbolicTensor;
  out = tf.layers.activation({ activation: 'relu' }).apply(out) as tf.SymbolicTensor;
  out = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', useBias: false }).apply(out) as tf.SymbolicTensor;
  out = tf.layers.batchNormalization().apply(out) as tf.SymbolicTensor;
  out = tf.layers.add().apply([out, x]) as tf.SymbolicTensor;
  out = tf.layers.activation({ activation: 'relu' }).apply(out) as tf.SymbolicTensor;
  return out;
}
