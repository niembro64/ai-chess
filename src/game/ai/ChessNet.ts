// Neural network for chess: ResNet with SE blocks, policy head + WDL value head
// Input: 8x8x20 board planes (channels-last)
// Policy output: 4096 probabilities (64 from-squares × 64 to-squares)
// Value output: WDL distribution [P(win), P(draw), P(loss)]; scalar value = P(win) - P(loss)

import * as tf from '@tensorflow/tfjs';
import type { ChessGameState, PieceType, Position, Move } from '@/types/chess';
import { extractWeights, cpuPredict, cpuPredictBatch, type CPUWeights } from './CPUForward';

// --- Board encoding ---

const PIECE_TYPES: PieceType[] = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn'];
export const NUM_PLANES = 20;
const BOARD_FLOATS = 8 * 8 * NUM_PLANES;
export const POLICY_SIZE = 4096; // 64 * 64

// Plane layout (from current player's perspective; board is flipped for black so "own"
// always occupies the bottom half of the 8x8 input):
//   0- 5  own pieces (K Q R B N P)
//   6-11  opp pieces
//   12    bias (constant 1)
//   13    halfMoveClock / 50          (50-move rule awareness)
//   14    fullMoveNumber / 100        (game phase awareness)
//   15-18 castling rights (own K-side, own Q-side, opp K-side, opp Q-side)
//   19    en passant target
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

  // Channel 12: bias plane
  for (let i = 0; i < 64; i++) {
    data[i * NUM_PLANES + 12] = 1;
  }

  // Channel 13: halfMoveClock / 50 (clamp to 1)
  const hmc = Math.min(1, state.halfMoveClock / 50);
  if (hmc > 0) {
    for (let i = 0; i < 64; i++) data[i * NUM_PLANES + 13] = hmc;
  }

  // Channel 14: fullMoveNumber / 100 (clamp to 1)
  const fmn = Math.min(1, state.fullMoveNumber / 100);
  if (fmn > 0) {
    for (let i = 0; i < 64; i++) data[i * NUM_PLANES + 14] = fmn;
  }

  // Channels 15-18: castling rights
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
        data[i * NUM_PLANES + 15 + c] = 1;
      }
    }
  }

  // Channel 19: en passant target
  if (state.enPassantTarget) {
    const epRank = isWhite ? state.enPassantTarget.rank : 7 - state.enPassantTarget.rank;
    const epFile = isWhite ? state.enPassantTarget.file : 7 - state.enPassantTarget.file;
    data[(epRank * 8 + epFile) * NUM_PLANES + 19] = 1;
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
  kernelSize: number;       // 3 or 5
  valueHeadSize: number;    // Hidden units in value head (16-64)
  seReduction: number;      // Squeeze-and-excitation channel reduction ratio (typically 8)
  learningRate: number;
};

// Policy channels is fixed at 64 (one per destination square).
const POLICY_CHANNELS = 64;
export const WDL_SIZE = 3;

export type SerializedWeights = {
  config?: NetConfig;   // Optional explicit config; if absent, detected from shapes
  // dtype tag for the `data` entries. When "float16", each entry is a
  // base64-encoded little-endian fp16 blob (produced by the Python
  // exporter). Absent / "float32" means each entry is a legacy number[].
  dtype?: 'float16' | 'float32';
  shapes: number[][];
  // Heterogeneous for back-compat: fp32 fixtures are number[], fp16
  // exports are base64 strings. importWeights() detects per entry.
  data: (number[] | string)[];
};

/**
 * Decode a single IEEE 754 half-precision float (16 bits) to a number.
 * Standard conversion — handles subnormals, Inf, NaN, ±0. Used to read
 * fp16 weights serialized by the Python trainer.
 */
function fp16ToFp32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7C00) >> 10;
  const f = h & 0x03FF;
  if (e === 0) {
    if (f === 0) return s ? -0 : 0;
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 0x1F) {
    return f === 0 ? (s ? -Infinity : Infinity) : NaN;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function decodeFp16Base64(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = fp16ToFp32(u16[i]);
  return out;
}

export class ChessNet {
  model: tf.LayersModel;
  private config: NetConfig;
  private cpu: CPUWeights | null = null; // CPU-side weight cache for fast inference

  private constructor(model: tf.LayersModel, config: NetConfig) {
    this.model = model;
    this.config = config;
    this.syncCPU();
  }

  static create(config: NetConfig): ChessNet {
    const model = buildModel(config);
    return new ChessNet(model, config);
  }

  // Copy weights from TF.js model to CPU arrays for fast inference
  syncCPU(): void {
    this.cpu = extractWeights(
      this.model,
      this.config.kernelSize,
      this.config.numFilters,
      this.config.numResBlocks,
      this.config.valueHeadSize,
      this.config.seReduction,
    );
  }

  importWeights(weights: SerializedWeights): void {
    const tensors: tf.Tensor[] = [];
    for (let i = 0; i < weights.shapes.length; i++) {
      const entry = weights.data[i];
      // String = base64 fp16 (new format). Array = legacy fp32 list.
      // tf.tensor accepts Float32Array or number[], so no further
      // dtype massaging needed.
      const values = typeof entry === 'string' ? decodeFp16Base64(entry) : entry;
      tensors.push(tf.tensor(values, weights.shapes[i]));
    }
    this.model.setWeights(tensors);
    for (const t of tensors) t.dispose();
    this.syncCPU();
  }

  // Use CPU for small models (<=32 filters), GPU for larger ones
  private get useCPU(): boolean {
    return this.cpu !== null && this.config.numFilters <= 32;
  }

  predict(state: ChessGameState): { policy: Float32Array; wdl: Float32Array; value: number } {
    if (this.useCPU) {
      return cpuPredict(encodeBoard(state), this.cpu!);
    }
    return tf.tidy(() => {
      const input = tf.tensor4d(encodeBoard(state), [1, 8, 8, NUM_PLANES]);
      const outputs = this.model.predict(input) as tf.Tensor[];
      const wdl = new Float32Array(outputs[1].dataSync() as Float32Array);
      return {
        policy: new Float32Array(outputs[0].dataSync() as Float32Array),
        wdl,
        value: wdl[0] - wdl[2],
      };
    });
  }

  predictBatch(boards: Float32Array[]): Array<{ policy: Float32Array; wdl: Float32Array; value: number }> {
    if (boards.length === 0) return [];
    if (this.useCPU) {
      return cpuPredictBatch(boards, this.cpu!);
    }
    const batchSize = boards.length;
    const buf = new Float32Array(batchSize * BOARD_FLOATS);
    for (let i = 0; i < batchSize; i++) {
      buf.set(boards[i], i * BOARD_FLOATS);
    }
    return tf.tidy(() => {
      const input = tf.tensor4d(buf, [batchSize, 8, 8, NUM_PLANES]);
      const outputs = this.model.predict(input) as tf.Tensor[];
      const allPolicy = outputs[0].dataSync() as Float32Array;
      const allWdl = outputs[1].dataSync() as Float32Array;
      const results: Array<{ policy: Float32Array; wdl: Float32Array; value: number }> = [];
      for (let i = 0; i < batchSize; i++) {
        const wdl = allWdl.slice(i * WDL_SIZE, (i + 1) * WDL_SIZE);
        results.push({
          policy: allPolicy.slice(i * POLICY_SIZE, (i + 1) * POLICY_SIZE),
          wdl,
          value: wdl[0] - wdl[2],
        });
      }
      return results;
    });
  }

  dispose(): void {
    this.model.dispose();
  }
}

// --- Model architecture ---

function buildModel(cfg: NetConfig): tf.LayersModel {
  const input = tf.input({ shape: [8, 8, NUM_PLANES] });
  const ks = cfg.kernelSize;

  let x = convBlock(input, cfg.numFilters, ks);
  for (let i = 0; i < cfg.numResBlocks; i++) {
    x = residualBlockSE(x, cfg.numFilters, ks, cfg.seReduction);
  }

  // Policy head: Conv2D(filters→policyChannels, 1×1) → flatten → softmax
  let ph = tf.layers.conv2d({ filters: POLICY_CHANNELS, kernelSize: 1, padding: 'same', useBias: true }).apply(x) as tf.SymbolicTensor;
  ph = tf.layers.flatten().apply(ph) as tf.SymbolicTensor;
  const policyOutput = tf.layers.activation({
    activation: 'softmax', name: 'policy_output',
  }).apply(ph) as tf.SymbolicTensor;

  // Value head: Conv2D(filters→1, 1×1) → BN → ReLU → flatten → Dense(vhs, relu) → Dense(3, softmax)
  let vh = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  vh = tf.layers.batchNormalization().apply(vh) as tf.SymbolicTensor;
  vh = tf.layers.activation({ activation: 'relu' }).apply(vh) as tf.SymbolicTensor;
  vh = tf.layers.flatten().apply(vh) as tf.SymbolicTensor;
  vh = tf.layers.dense({ units: cfg.valueHeadSize, activation: 'relu' }).apply(vh) as tf.SymbolicTensor;
  const wdlOutput = tf.layers.dense({
    units: WDL_SIZE, activation: 'softmax', name: 'wdl_output',
  }).apply(vh) as tf.SymbolicTensor;

  return tf.model({ inputs: input, outputs: [policyOutput, wdlOutput] });
}

function convBlock(x: tf.SymbolicTensor, filters: number, kernelSize: number): tf.SymbolicTensor {
  let out = tf.layers.conv2d({ filters, kernelSize, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  out = tf.layers.batchNormalization().apply(out) as tf.SymbolicTensor;
  out = tf.layers.activation({ activation: 'relu' }).apply(out) as tf.SymbolicTensor;
  return out;
}

// Residual block with Squeeze-and-Excitation (channel-wise attention).
// Layer ordering determines weight serialization order — DO NOT REORDER without
// also updating CPUForward.extractWeights and the Python weight exporter.
function residualBlockSE(x: tf.SymbolicTensor, filters: number, kernelSize: number, seReduction: number): tf.SymbolicTensor {
  let out = tf.layers.conv2d({ filters, kernelSize, padding: 'same', useBias: false }).apply(x) as tf.SymbolicTensor;
  out = tf.layers.batchNormalization().apply(out) as tf.SymbolicTensor;
  out = tf.layers.activation({ activation: 'relu' }).apply(out) as tf.SymbolicTensor;
  out = tf.layers.conv2d({ filters, kernelSize, padding: 'same', useBias: false }).apply(out) as tf.SymbolicTensor;
  out = tf.layers.batchNormalization().apply(out) as tf.SymbolicTensor;

  // SE: global avg pool → dense(hidden, relu) → dense(filters, sigmoid) → per-channel scale
  const hidden = Math.max(1, Math.floor(filters / seReduction));
  let se = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(out) as tf.SymbolicTensor;
  se = tf.layers.dense({ units: hidden, activation: 'relu', useBias: true }).apply(se) as tf.SymbolicTensor;
  se = tf.layers.dense({ units: filters, activation: 'sigmoid', useBias: true }).apply(se) as tf.SymbolicTensor;
  se = tf.layers.reshape({ targetShape: [1, 1, filters] }).apply(se) as tf.SymbolicTensor;
  out = tf.layers.multiply().apply([out, se]) as tf.SymbolicTensor;

  out = tf.layers.add().apply([out, x]) as tf.SymbolicTensor;
  out = tf.layers.activation({ activation: 'relu' }).apply(out) as tf.SymbolicTensor;
  return out;
}
