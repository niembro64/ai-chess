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

// Detect architecture config from a loaded TF.js model.
// Uses model.weights (layer variable metadata) to read shapes WITHOUT
// creating or disposing any tensors.
//
// Expected weight ordering (per res block with SE):
//   conv1 W, bn1 γ β μ σ², conv2 W, bn2 γ β μ σ², SE_dense1 W b, SE_dense2 W b  (14 weights)
// Value head: conv W, BN γ β μ σ², dense1 W b, dense2(3) W b
function detectConfigFromModel(model: tf.LayersModel): Omit<NetConfig, 'learningRate'> {
  const shapes = model.weights.map(w => w.shape as number[]);

  const kernelSize = shapes[0][0];
  const numFilters = shapes[0][3];

  let resConvs = 0;
  for (const s of shapes) {
    if (s.length === 4 && s[0] === kernelSize && s[2] === numFilters && s[3] === numFilters) {
      resConvs++;
    }
  }
  const numResBlocks = Math.floor(resConvs / 2);

  // Detect SE reduction from dense weight [numFilters, hidden] where hidden < numFilters
  let seReduction = 8;
  for (const s of shapes) {
    if (s.length === 2 && s[0] === numFilters && s[1] > 0 && s[1] < numFilters) {
      seReduction = Math.max(1, Math.round(numFilters / s[1]));
      break;
    }
  }

  // Value head size: dense [64, vhs] followed later by [vhs, 3]
  let valueHeadSize = 64;
  for (let i = 0; i < shapes.length; i++) {
    const s = shapes[i];
    if (s.length === 2 && s[0] === 64 && s[1] !== WDL_SIZE) {
      valueHeadSize = s[1];
      break;
    }
  }

  return { numResBlocks, numFilters, kernelSize, valueHeadSize, seReduction };
}

export type SerializedWeights = {
  config?: NetConfig;   // Optional explicit config; if absent, detected from shapes
  shapes: number[][];
  data: number[][];
};

export class ChessNet {
  model: tf.LayersModel;
  private optimizer: tf.Optimizer;
  private lr: number;
  private config: NetConfig;
  private cpu: CPUWeights | null = null; // CPU-side weight cache for fast inference

  private constructor(model: tf.LayersModel, lr: number, config: NetConfig) {
    this.model = model;
    this.lr = lr;
    this.config = config;
    this.optimizer = tf.train.adam(lr);
    this.syncCPU();
  }

  static create(config: NetConfig): ChessNet {
    const model = buildModel(config);
    return new ChessNet(model, config.learningRate, config);
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

  getParamCount(): number {
    return this.model.countParams();
  }

  // Sample a few weights from the first trainable layer for UI display
  getSampleWeights(): number[] {
    const layers = this.model.trainableWeights;
    if (layers.length === 0) return [];
    // read() returns the model's own weight tensor -- do NOT dispose it
    const data = layers[0].read().dataSync() as Float32Array;
    const count = 8;
    const step = Math.max(1, Math.floor(data.length / count));
    const samples: number[] = [];
    for (let i = 0; i < count && i * step < data.length; i++) {
      samples.push(data[i * step]);
    }
    return samples;
  }

  getConfig(): NetConfig {
    return { ...this.config };
  }

  // Export all weights as a serializable object.
  // Uses model.weights (LayerVariable refs) to avoid the getWeights copy/dispose ambiguity.
  exportWeights(): SerializedWeights {
    const result: SerializedWeights = { config: { ...this.config }, shapes: [], data: [] };
    for (const w of this.model.weights) {
      result.shapes.push(w.shape as number[]);
      result.data.push(Array.from(w.read().dataSync() as Float32Array));
    }
    return result;
  }

  importWeights(weights: SerializedWeights): void {
    const tensors: tf.Tensor[] = [];
    for (let i = 0; i < weights.shapes.length; i++) {
      tensors.push(tf.tensor(weights.data[i], weights.shapes[i]));
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

  async train(
    boards: Float32Array[],
    policies: Float32Array[],
    values: number[],
  ): Promise<{ policyLoss: number; valueLoss: number; totalLoss: number }> {
    const batchSize = boards.length;

    const boardBuf = new Float32Array(batchSize * BOARD_FLOATS);
    const policyBuf = new Float32Array(batchSize * POLICY_SIZE);
    // Convert continuous value [-1,1] to WDL target: P(win)=max(0,v), P(loss)=max(0,-v),
    // P(draw)=1-P(win)-P(loss). v=1 → [1,0,0], v=-1 → [0,0,1], v=0 → [0,1,0].
    const wdlBuf = new Float32Array(batchSize * WDL_SIZE);
    for (let i = 0; i < batchSize; i++) {
      boardBuf.set(boards[i], i * BOARD_FLOATS);
      policyBuf.set(policies[i], i * POLICY_SIZE);
      const v = Math.max(-1, Math.min(1, values[i]));
      const w = Math.max(0, v);
      const l = Math.max(0, -v);
      wdlBuf[i * WDL_SIZE + 0] = w;
      wdlBuf[i * WDL_SIZE + 1] = 1 - w - l;
      wdlBuf[i * WDL_SIZE + 2] = l;
    }

    const inputTensor = tf.tensor4d(boardBuf, [batchSize, 8, 8, NUM_PLANES]);
    const policyTarget = tf.tensor2d(policyBuf, [batchSize, POLICY_SIZE]);
    const wdlTarget = tf.tensor2d(wdlBuf, [batchSize, WDL_SIZE]);

    let policyLoss = 0;
    let valueLoss = 0;

    const WEIGHT_DECAY = 1e-4;

    const totalLossTensor = this.optimizer.minimize(() => {
      const outputs = this.model.apply(inputTensor, { training: true }) as tf.Tensor[];
      const pLoss = tf.neg(tf.sum(tf.mul(policyTarget, tf.log(tf.add(outputs[0], 1e-8))))).div(tf.scalar(batchSize));
      // WDL cross-entropy loss: -mean(sum(target * log(pred), axis=1))
      const vLoss = tf.neg(tf.mean(tf.sum(tf.mul(wdlTarget, tf.log(tf.add(outputs[1], 1e-8))), 1))) as tf.Scalar;

      // L2 weight decay (simulates AdamW) — computed as a single sum to minimize intermediate tensors
      const allSquaredWeights = this.model.trainableWeights.map(w => tf.sum(tf.square(w.read())));
      const l2 = tf.addN(allSquaredWeights) as tf.Scalar;
      const decay = tf.mul(l2, tf.scalar(WEIGHT_DECAY)) as tf.Scalar;

      const total = tf.add(tf.add(pLoss, vLoss), decay) as tf.Scalar;
      policyLoss = pLoss.dataSync()[0];
      valueLoss = vLoss.dataSync()[0];
      return total;
    }, true) as tf.Scalar;

    const totalLoss = totalLossTensor!.dataSync()[0];
    totalLossTensor!.dispose();
    inputTensor.dispose();
    policyTarget.dispose();
    wdlTarget.dispose();

    return { policyLoss, valueLoss, totalLoss };
  }

  async save(): Promise<void> {
    await this.model.save('indexeddb://chess-net');
  }

  // Check if a saved model exists in IndexedDB without loading it
  static async hasSavedModel(): Promise<boolean> {
    try {
      const models = await tf.io.listModels();
      return 'indexeddb://chess-net' in models;
    } catch {
      return false;
    }
  }

  // Delete saved model from IndexedDB (for clearing corrupted saves)
  static async deleteSavedModel(): Promise<void> {
    try {
      await tf.io.removeModel('indexeddb://chess-net');
    } catch {
      // Model didn't exist
    }
  }

  async load(): Promise<boolean> {
    try {
      const loaded = await tf.loadLayersModel('indexeddb://chess-net');
      this.model.dispose();
      this.optimizer.dispose();
      this.model = loaded;
      this.optimizer = tf.train.adam(this.lr);
      this.config = { ...detectConfigFromModel(loaded), learningRate: this.lr };
      this.syncCPU();
      return true;
    } catch {
      return false;
    }
  }

  // Load directly from IndexedDB without creating a throwaway model first.
  // Returns null (and deletes the stored model) if the saved architecture is
  // incompatible with the current encoder (e.g., a model saved before the
  // plane-count / WDL-head upgrade).
  static async loadFromSaved(lr: number): Promise<ChessNet | null> {
    try {
      const model = await tf.loadLayersModel('indexeddb://chess-net');
      const firstConvShape = model.weights[0]?.shape as number[] | undefined;
      const savedPlanes = firstConvShape?.[2];
      if (savedPlanes !== NUM_PLANES) {
        model.dispose();
        await ChessNet.deleteSavedModel();
        return null;
      }
      const config = detectConfigFromModel(model);
      return new ChessNet(model, lr, { ...config, learningRate: lr });
    } catch {
      return null;
    }
  }

  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
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
