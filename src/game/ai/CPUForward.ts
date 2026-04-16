// Custom CPU forward pass for chess neural network inference.
// ~10-50x faster than TF.js WebGL for our tiny model because:
// - No GPU dispatch overhead (~0.1ms per op)
// - No tensor creation/disposal
// - Entire small model fits in cache
// - Direct array math, no framework abstraction
//
// Supports: input encoding with NUM_PLANES channels, ResNet with SE blocks,
//           policy head (softmax over POLICY_SIZE), WDL value head (softmax over 3).

import * as tf from '@tensorflow/tfjs';
import { POLICY_SIZE, NUM_PLANES, WDL_SIZE } from './ChessNet';

const BOARD = 8;
const BOARD_SQ = BOARD * BOARD; // 64

// --- Weight storage ---

export type CPUResBlock = {
  conv1: Float32Array;
  scale1: Float32Array;
  bias1: Float32Array;
  conv2: Float32Array;
  scale2: Float32Array;
  bias2: Float32Array;
  seHidden: number;
  seDense1W: Float32Array; // [filters * seHidden]
  seDense1B: Float32Array; // [seHidden]
  seDense2W: Float32Array; // [seHidden * filters]
  seDense2B: Float32Array; // [filters]
};

export type CPUWeights = {
  kernelSize: number;
  numFilters: number;
  numResBlocks: number;
  valueHeadSize: number;
  seReduction: number;
  // Fused BN: output = input * scale + bias (precomputed from gamma, beta, mean, var)
  // Conv weights stored as [kH * kW * inC * outC]
  initConv: Float32Array;
  initScale: Float32Array; // gamma / sqrt(var + eps)
  initBias: Float32Array;  // beta - mean * scale
  resBlocks: CPUResBlock[];
  policyConv: Float32Array; // [1 * 1 * filters * 64]
  policyBias: Float32Array; // [64]
  valueConv: Float32Array;  // [1 * 1 * filters * 1]
  valueScale: Float32Array;
  valueBias: Float32Array;
  valueDense1W: Float32Array; // [64 * valueHeadSize]
  valueDense1B: Float32Array;
  valueDense2W: Float32Array; // [valueHeadSize * 3]  (WDL)
  valueDense2B: Float32Array; // [3]
};

const BN_EPS = 0.001; // TF.js default BatchNorm epsilon

// Fuse BN params: scale = gamma / sqrt(var + eps), bias = beta - mean * scale
function fuseBN(gamma: Float32Array, beta: Float32Array, mean: Float32Array, variance: Float32Array): { scale: Float32Array; bias: Float32Array } {
  const n = gamma.length;
  const scale = new Float32Array(n);
  const bias = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    scale[i] = gamma[i] / Math.sqrt(variance[i] + BN_EPS);
    bias[i] = beta[i] - mean[i] * scale[i];
  }
  return { scale, bias };
}

// Extract weights from TF.js model into CPU-friendly format.
//
// TF.js `model.weights` layout (see `training/src/chess_ai/weight_io.py` for the
// full derivation — this MUST stay in sync):
//   [trainable (γ β for BN, no μ σ²), in layer topo order]
//   [non-trainable (μ σ² for all BNs), in BN creation order]
//
// Trainable section:
//   init conv, init bn (γ β),
//   per block: conv1, bn1 (γ β), conv2, bn2 (γ β), SE fc1 (W b), SE fc2 (W b),
//   heads (interleaved): value_conv, value_bn (γ β), policy_conv (W b),
//                         value_fc1 (W b), value_fc2 (W b)
//
// Non-trainable section (μ σ² per BN in creation order):
//   init_bn, (res blocks: bn1, bn2) × N, value_bn
export function extractWeights(model: tf.LayersModel, kernelSize: number, numFilters: number, numResBlocks: number, valueHeadSize: number, seReduction: number): CPUWeights {
  const allWeights = model.weights;
  let idx = 0;
  const next = () => {
    const w = allWeights[idx];
    idx++;
    return new Float32Array(w.read().dataSync() as Float32Array);
  };

  // ----- Trainable -----

  // Initial conv + BN γ/β
  const initConv = next();
  const initGamma = next();
  const initBeta = next();

  // Res block trainable weights (γ/β only for BNs; SE denses)
  interface PartialResBlock {
    conv1: Float32Array;
    g1: Float32Array;
    b1: Float32Array;
    conv2: Float32Array;
    g2: Float32Array;
    b2: Float32Array;
    seDense1W: Float32Array;
    seDense1B: Float32Array;
    seDense2W: Float32Array;
    seDense2B: Float32Array;
  }
  const partialBlocks: PartialResBlock[] = [];
  for (let b = 0; b < numResBlocks; b++) {
    partialBlocks.push({
      conv1: next(),
      g1: next(),
      b1: next(),
      conv2: next(),
      g2: next(),
      b2: next(),
      seDense1W: next(),
      seDense1B: next(),
      seDense2W: next(),
      seDense2B: next(),
    });
  }

  // Heads — order is interleaved (value conv+BN first, then policy, then value denses)
  const valueConv = next();
  const valueGamma = next();
  const valueBeta = next();
  const policyConv = next();
  const policyBias = next();
  const valueDense1W = next();
  const valueDense1B = next();
  const valueDense2W = next();
  const valueDense2B = next();

  // ----- Non-trainable: μ/σ² for every BN in creation order -----

  const initMean = next();
  const initVar = next();

  const blockBNStats: Array<{ m1: Float32Array; v1: Float32Array; m2: Float32Array; v2: Float32Array }> = [];
  for (let b = 0; b < numResBlocks; b++) {
    blockBNStats.push({
      m1: next(),
      v1: next(),
      m2: next(),
      v2: next(),
    });
  }

  const valueMean = next();
  const valueVar = next();

  if (idx !== allWeights.length) {
    throw new Error(
      `extractWeights consumed ${idx} weights but model has ${allWeights.length}. ` +
      `This means the TS/Python weight-order contract has drifted.`,
    );
  }

  // ----- Fuse BN params now that we have γ β μ σ² for each -----

  const initBN = fuseBN(initGamma, initBeta, initMean, initVar);
  const valueBN = fuseBN(valueGamma, valueBeta, valueMean, valueVar);

  const seHidden = Math.max(1, Math.floor(numFilters / seReduction));
  const resBlocks: CPUResBlock[] = [];
  for (let b = 0; b < numResBlocks; b++) {
    const p = partialBlocks[b];
    const s = blockBNStats[b];
    const bn1 = fuseBN(p.g1, p.b1, s.m1, s.v1);
    const bn2 = fuseBN(p.g2, p.b2, s.m2, s.v2);
    resBlocks.push({
      conv1: p.conv1, scale1: bn1.scale, bias1: bn1.bias,
      conv2: p.conv2, scale2: bn2.scale, bias2: bn2.bias,
      seHidden,
      seDense1W: p.seDense1W, seDense1B: p.seDense1B,
      seDense2W: p.seDense2W, seDense2B: p.seDense2B,
    });
  }

  return {
    kernelSize, numFilters, numResBlocks, valueHeadSize, seReduction,
    initConv, initScale: initBN.scale, initBias: initBN.bias,
    resBlocks,
    policyConv, policyBias,
    valueConv, valueScale: valueBN.scale, valueBias: valueBN.bias,
    valueDense1W, valueDense1B, valueDense2W, valueDense2B,
  };
}

// --- Forward pass primitives ---

// Conv2D with 'same' padding. Kernel layout: [kH, kW, inC, outC]
function conv2d(
  input: Float32Array, kernel: Float32Array,
  inC: number, outC: number, kSize: number,
  output: Float32Array,
): void {
  const pad = kSize >> 1;
  for (let oy = 0; oy < BOARD; oy++) {
    for (let ox = 0; ox < BOARD; ox++) {
      const outBase = (oy * BOARD + ox) * outC;
      // Zero output
      for (let oc = 0; oc < outC; oc++) output[outBase + oc] = 0;

      for (let ky = 0; ky < kSize; ky++) {
        const iy = oy + ky - pad;
        if (iy < 0 || iy >= BOARD) continue;
        for (let kx = 0; kx < kSize; kx++) {
          const ix = ox + kx - pad;
          if (ix < 0 || ix >= BOARD) continue;
          const inBase = (iy * BOARD + ix) * inC;
          const kBase = (ky * kSize + kx) * inC * outC;
          for (let ic = 0; ic < inC; ic++) {
            const inputVal = input[inBase + ic];
            if (inputVal === 0) continue; // Skip zeros (sparse input)
            const kOff = kBase + ic * outC;
            for (let oc = 0; oc < outC; oc++) {
              output[outBase + oc] += inputVal * kernel[kOff + oc];
            }
          }
        }
      }
    }
  }
}

// 1x1 convolution (much simpler, no padding needed)
function conv1x1(
  input: Float32Array, kernel: Float32Array,
  inC: number, outC: number,
  output: Float32Array,
): void {
  for (let pos = 0; pos < BOARD_SQ; pos++) {
    const inBase = pos * inC;
    const outBase = pos * outC;
    for (let oc = 0; oc < outC; oc++) {
      let sum = 0;
      for (let ic = 0; ic < inC; ic++) {
        sum += input[inBase + ic] * kernel[ic * outC + oc];
      }
      output[outBase + oc] = sum;
    }
  }
}

// 1x1 conv with bias (for policy head)
function conv1x1Bias(
  input: Float32Array, kernel: Float32Array, bias: Float32Array,
  inC: number, outC: number,
  output: Float32Array,
): void {
  for (let pos = 0; pos < BOARD_SQ; pos++) {
    const inBase = pos * inC;
    const outBase = pos * outC;
    for (let oc = 0; oc < outC; oc++) {
      let sum = bias[oc];
      for (let ic = 0; ic < inC; ic++) {
        sum += input[inBase + ic] * kernel[ic * outC + oc];
      }
      output[outBase + oc] = sum;
    }
  }
}

// Fused BatchNorm + ReLU: output = max(0, input * scale + bias)
function bnRelu(input: Float32Array, scale: Float32Array, bias: Float32Array, channels: number, output: Float32Array): void {
  for (let pos = 0; pos < BOARD_SQ; pos++) {
    const base = pos * channels;
    for (let c = 0; c < channels; c++) {
      const val = input[base + c] * scale[c] + bias[c];
      output[base + c] = val > 0 ? val : 0;
    }
  }
}

// Fused BatchNorm (no ReLU) — used before SE scaling so SE applies to BN output
function bn(input: Float32Array, scale: Float32Array, bias: Float32Array, channels: number, output: Float32Array): void {
  const n = BOARD_SQ * channels;
  for (let i = 0; i < n; i++) {
    const c = i % channels;
    output[i] = input[i] * scale[c] + bias[c];
  }
}

// ReLU in-place
function reluInPlace(data: Float32Array, size: number): void {
  for (let i = 0; i < size; i++) {
    if (data[i] < 0) data[i] = 0;
  }
}

// Residual add: output = a + b, then ReLU
function residualAddRelu(a: Float32Array, b: Float32Array, size: number, output: Float32Array): void {
  for (let i = 0; i < size; i++) {
    const val = a[i] + b[i];
    output[i] = val > 0 ? val : 0;
  }
}

// Dense layer: output[j] = bias[j] + sum_i(input[i] * kernel[i * outSize + j])
function dense(input: Float32Array, kernel: Float32Array, bias: Float32Array, inSize: number, outSize: number, output: Float32Array): void {
  for (let j = 0; j < outSize; j++) {
    let sum = bias[j];
    for (let i = 0; i < inSize; i++) {
      sum += input[i] * kernel[i * outSize + j];
    }
    output[j] = sum;
  }
}

// Softmax over a flat array
function softmax(input: Float32Array, size: number, output: Float32Array): void {
  let max = -Infinity;
  for (let i = 0; i < size; i++) {
    if (input[i] > max) max = input[i];
  }
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const e = Math.exp(input[i] - max);
    output[i] = e;
    sum += e;
  }
  const invSum = 1 / sum;
  for (let i = 0; i < size; i++) {
    output[i] *= invSum;
  }
}

// Squeeze-Excite: global avg pool (channels-last) → dense(hidden, relu) → dense(filters, sigmoid)
// Returns per-channel scale factor in seScale.
function seCompute(
  spatial: Float32Array,     // [BOARD_SQ * filters] post-BN activations
  filters: number,
  block: CPUResBlock,
  seAvgBuf: Float32Array,    // [filters]
  seHiddenBuf: Float32Array, // [hidden]
  seScaleBuf: Float32Array,  // [filters]
): void {
  const hidden = block.seHidden;
  // Global avg pool over 64 spatial positions
  for (let c = 0; c < filters; c++) seAvgBuf[c] = 0;
  for (let pos = 0; pos < BOARD_SQ; pos++) {
    const base = pos * filters;
    for (let c = 0; c < filters; c++) seAvgBuf[c] += spatial[base + c];
  }
  const inv = 1 / BOARD_SQ;
  for (let c = 0; c < filters; c++) seAvgBuf[c] *= inv;

  // Dense1 + ReLU: [filters] → [hidden]
  for (let j = 0; j < hidden; j++) {
    let s = block.seDense1B[j];
    for (let i = 0; i < filters; i++) {
      s += seAvgBuf[i] * block.seDense1W[i * hidden + j];
    }
    seHiddenBuf[j] = s > 0 ? s : 0;
  }

  // Dense2 + sigmoid: [hidden] → [filters]
  for (let j = 0; j < filters; j++) {
    let s = block.seDense2B[j];
    for (let i = 0; i < hidden; i++) {
      s += seHiddenBuf[i] * block.seDense2W[i * filters + j];
    }
    seScaleBuf[j] = 1 / (1 + Math.exp(-s));
  }
}

// Apply per-channel SE scale in place
function seApply(spatial: Float32Array, seScale: Float32Array, filters: number): void {
  for (let pos = 0; pos < BOARD_SQ; pos++) {
    const base = pos * filters;
    for (let c = 0; c < filters; c++) {
      spatial[base + c] *= seScale[c];
    }
  }
}

// --- Full forward pass ---

// Pre-allocated buffers to avoid GC pressure
let buf1: Float32Array | null = null;
let buf2: Float32Array | null = null;
let buf3: Float32Array | null = null;
let policyBuf: Float32Array | null = null;
let valueFlatBuf: Float32Array | null = null;
let valueDense1Buf: Float32Array | null = null;
let valueDense2Buf: Float32Array | null = null; // [WDL_SIZE]
let seAvgBuf: Float32Array | null = null;
let seHiddenBuf: Float32Array | null = null;
let seScaleBuf: Float32Array | null = null;

function ensureBuffers(filters: number, valueHeadSize: number, seReduction: number): void {
  const spatialSize = BOARD_SQ * filters;
  const hidden = Math.max(1, Math.floor(filters / seReduction));
  if (!buf1 || buf1.length < spatialSize) {
    buf1 = new Float32Array(spatialSize);
    buf2 = new Float32Array(spatialSize);
    buf3 = new Float32Array(spatialSize);
    policyBuf = new Float32Array(POLICY_SIZE);
    valueFlatBuf = new Float32Array(BOARD_SQ);
    valueDense1Buf = new Float32Array(valueHeadSize);
    valueDense2Buf = new Float32Array(WDL_SIZE);
    seAvgBuf = new Float32Array(filters);
    seHiddenBuf = new Float32Array(hidden);
    seScaleBuf = new Float32Array(filters);
  } else {
    if (!valueDense1Buf || valueDense1Buf.length < valueHeadSize) valueDense1Buf = new Float32Array(valueHeadSize);
    if (!valueDense2Buf) valueDense2Buf = new Float32Array(WDL_SIZE);
    if (!seAvgBuf || seAvgBuf.length < filters) seAvgBuf = new Float32Array(filters);
    if (!seHiddenBuf || seHiddenBuf.length < hidden) seHiddenBuf = new Float32Array(hidden);
    if (!seScaleBuf || seScaleBuf.length < filters) seScaleBuf = new Float32Array(filters);
  }
}

export function cpuPredict(
  board: Float32Array, // [8*8*NUM_PLANES] encoded board
  w: CPUWeights,
): { policy: Float32Array; wdl: Float32Array; value: number } {
  const f = w.numFilters;
  const ks = w.kernelSize;
  const vhs = w.valueHeadSize;
  ensureBuffers(f, vhs, w.seReduction);

  // Initial conv + BN + ReLU
  conv2d(board, w.initConv, NUM_PLANES, f, ks, buf1!);
  bnRelu(buf1!, w.initScale, w.initBias, f, buf2!);

  // Residual blocks with SE
  let current = buf2!;
  for (let b = 0; b < w.numResBlocks; b++) {
    const block = w.resBlocks[b];
    conv2d(current, block.conv1, f, f, ks, buf3!);
    bnRelu(buf3!, block.scale1, block.bias1, f, buf1!);
    conv2d(buf1!, block.conv2, f, f, ks, buf3!);
    // BN (no ReLU yet); SE is applied before residual add
    bn(buf3!, block.scale2, block.bias2, f, buf1!);
    // SE: compute scale from buf1, apply to buf1 in place
    seCompute(buf1!, f, block, seAvgBuf!, seHiddenBuf!, seScaleBuf!);
    seApply(buf1!, seScaleBuf!, f);
    // Residual add + ReLU: buf1 (SE-scaled) + current → buf3
    residualAddRelu(buf1!, current, BOARD_SQ * f, buf3!);
    // Swap: buf3 is now current, old current goes into buf3 slot
    const tmp = current;
    current = buf3!;
    buf3 = tmp;
  }

  // Policy head: 1x1 conv with bias → softmax
  conv1x1Bias(current, w.policyConv, w.policyBias, f, 64, policyBuf!);
  const policy = new Float32Array(POLICY_SIZE);
  softmax(policyBuf!, POLICY_SIZE, policy);

  // Value head: 1x1 conv → BN → ReLU → dense → ReLU → dense(3) → softmax
  conv1x1(current, w.valueConv, f, 1, valueFlatBuf!);
  for (let i = 0; i < BOARD_SQ; i++) {
    const val = valueFlatBuf![i] * w.valueScale[0] + w.valueBias[0];
    valueFlatBuf![i] = val > 0 ? val : 0;
  }
  dense(valueFlatBuf!, w.valueDense1W, w.valueDense1B, BOARD_SQ, vhs, valueDense1Buf!);
  reluInPlace(valueDense1Buf!, vhs);
  dense(valueDense1Buf!, w.valueDense2W, w.valueDense2B, vhs, WDL_SIZE, valueDense2Buf!);
  const wdl = new Float32Array(WDL_SIZE);
  softmax(valueDense2Buf!, WDL_SIZE, wdl);
  const value = wdl[0] - wdl[2];

  return { policy, wdl, value };
}

// Batch predict: just loop (no batch overhead on CPU)
export function cpuPredictBatch(
  boards: Float32Array[],
  w: CPUWeights,
): Array<{ policy: Float32Array; wdl: Float32Array; value: number }> {
  return boards.map(b => cpuPredict(b, w));
}
