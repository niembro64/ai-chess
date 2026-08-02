// ToyNet — the browser side of the "Toy" teaching network.
//
// Mirrors training/scripts/toy_train.py exactly:
//   input   8x8x6 channels-last (P N B R Q K), +1 mover / -1 opponent,
//           board rotated 180° when black is to move (Sage convention)
//   trunk   conv3x3 -> N residual blocks (conv-relu-conv + skip), no BN
//   policy  1x1 conv (4ch) -> flatten(256) -> FC -> softmax over 4096
//   value   1x1 conv (2ch) -> flatten(128) -> FC(64) -> FC(3) -> softmax
//           = WDL probabilities like Sage; scalar value = P(win) - P(loss)
//
// Weight JSON ("toy-v1") lists tensors by name in a fixed order, fp16
// base64, conv filters already in TF layout [h,w,in,out] and linear
// weights [in,out] — the exporter did the transposes.

import * as tf from '@tensorflow/tfjs';
import type { ChessGameState } from '@/types/chess';
import type { PieceType } from '@/types/chess';

export const TOY_NUM_PLANES = 6;
export const TOY_PIECE_CHANNEL: Record<PieceType, number> = {
  pawn: 0, knight: 1, bishop: 2, rook: 3, queen: 4, king: 5,
};
// For labeling the visualization slabs, in channel order.
export const TOY_CHANNEL_NAMES = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King'];

export type ToySerializedWeights = {
  kind: 'toy-v1';
  config: {
    numPlanes: number;
    numFilters: number;
    numResBlocks: number;
    policyChannels: number;
    valueChannels: number;
    valueHidden: number;
  };
  names: string[];
  shapes: number[][];
  data: string[];
};

export function isToyWeights(json: unknown): json is ToySerializedWeights {
  return !!json && typeof json === 'object' && (json as { kind?: string }).kind === 'toy-v1';
}

// Flat [8*8*6], idx = (rank*8 + file)*6 + channel. Matches encode_toy().
export function encodeToyBoard(state: ChessGameState): Float32Array {
  const x = new Float32Array(8 * 8 * TOY_NUM_PLANES);
  const whiteToMove = state.currentTurn === 'white';
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const p = state.board[r][f];
      if (!p) continue;
      const rr = whiteToMove ? r : 7 - r;
      const ff = whiteToMove ? f : 7 - f;
      const sign = (p.color === 'white') === whiteToMove ? 1 : -1;
      x[(rr * 8 + ff) * TOY_NUM_PLANES + TOY_PIECE_CHANNEL[p.type]] = sign;
    }
  }
  return x;
}

function decodeFp16Base64(b64: string): Float32Array {
  const bin = atob(b64);
  const n = bin.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const lo = bin.charCodeAt(i * 2);
    const hi = bin.charCodeAt(i * 2 + 1);
    const h = (hi << 8) | lo; // little-endian fp16
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    let v: number;
    if (exp === 0) {
      v = sign * frac * 2 ** -24;                    // subnormal
    } else if (exp === 0x1f) {
      v = frac ? NaN : sign * Infinity;
    } else {
      v = sign * (1 + frac / 1024) * 2 ** (exp - 15);
    }
    out[i] = v;
  }
  return out;
}

export class ToyNet {
  private w: Map<string, tf.Tensor>;
  private blocks: number;

  private constructor(w: Map<string, tf.Tensor>, blocks: number) {
    this.w = w;
    this.blocks = blocks;
  }

  static create(json: ToySerializedWeights): ToyNet {
    const w = new Map<string, tf.Tensor>();
    for (let i = 0; i < json.names.length; i++) {
      const values = decodeFp16Base64(json.data[i]);
      w.set(json.names[i], tf.tensor(values, json.shapes[i] as number[]));
    }
    return new ToyNet(w, json.config.numResBlocks);
  }

  // Raw single-position outputs plus intermediate logits — used by the
  // Toy Mind panel to show the pre-mask policy.
  predictBatch(boards: Float32Array[]): Array<{ policy: Float32Array; value: number }> {
    if (boards.length === 0) return [];
    const B = boards.length;
    const buf = new Float32Array(B * 8 * 8 * TOY_NUM_PLANES);
    for (let i = 0; i < B; i++) buf.set(boards[i], i * 8 * 8 * TOY_NUM_PLANES);

    const [policyT, valueT] = tf.tidy(() => {
      const g = (n: string) => this.w.get(n)! as tf.Tensor4D;
      const gm = (n: string) => this.w.get(n)! as tf.Tensor2D;
      const gb = (n: string) => this.w.get(n)!;

      let x = tf.tensor4d(buf, [B, 8, 8, TOY_NUM_PLANES]);
      x = tf.relu(tf.add(tf.conv2d(x, g('conv_in.w'), 1, 'same'), gb('conv_in.b'))) as tf.Tensor4D;
      for (let i = 0; i < this.blocks; i++) {
        const y1 = tf.relu(
          tf.add(tf.conv2d(x, g(`block${i}.conv1.w`), 1, 'same'), gb(`block${i}.conv1.b`)),
        ) as tf.Tensor4D;
        const y2 = tf.add(tf.conv2d(y1, g(`block${i}.conv2.w`), 1, 'same'), gb(`block${i}.conv2.b`));
        x = tf.relu(tf.add(x, y2)) as tf.Tensor4D;
      }
      const p = tf.relu(tf.add(tf.conv2d(x, g('policy_conv.w'), 1, 'same'), gb('policy_conv.b')));
      const logits = tf.add(
        tf.matMul(tf.reshape(p, [B, -1]) as tf.Tensor2D, gm('policy_fc.w')),
        gb('policy_fc.b'),
      );
      const policy = tf.softmax(logits as tf.Tensor2D);

      const v0 = tf.relu(tf.add(tf.conv2d(x, g('value_conv.w'), 1, 'same'), gb('value_conv.b')));
      const v1 = tf.relu(tf.add(
        tf.matMul(tf.reshape(v0, [B, -1]) as tf.Tensor2D, gm('value_fc1.w')),
        gb('value_fc1.b'),
      )) as tf.Tensor2D;
      const wdl = tf.softmax(
        tf.add(tf.matMul(v1, gm('value_fc2.w')), gb('value_fc2.b')) as tf.Tensor2D,
      );
      return [policy, wdl];
    });

    const policyData = policyT.dataSync() as Float32Array;
    const wdlData = valueT.dataSync() as Float32Array;
    policyT.dispose();
    valueT.dispose();

    const out: Array<{ policy: Float32Array; value: number }> = [];
    for (let i = 0; i < B; i++) {
      out.push({
        policy: policyData.slice(i * 4096, (i + 1) * 4096),
        // Scalar value = P(win) - P(loss), same convention as Sage.
        value: wdlData[i * 3] - wdlData[i * 3 + 2],
      });
    }
    return out;
  }

  dispose(): void {
    for (const t of this.w.values()) t.dispose();
    this.w.clear();
  }
}
