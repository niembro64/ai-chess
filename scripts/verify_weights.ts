// Round-trip verification: load PyTorch-exported weights into the TS ChessNet,
// run forward on the same boards the Python side used, dump outputs.
//
// Usage (from project root):
//   npm run verify-weights
//
// Pipeline:
//   1. Python runs `training/scripts/make_roundtrip_fixture.py` →
//        training/tests/fixtures/roundtrip_weights.json
//        training/tests/fixtures/roundtrip_boards.json
//        training/tests/fixtures/roundtrip_pytorch_outputs.json
//   2. This script → training/tests/fixtures/roundtrip_ts_outputs.json
//   3. `pytest training/tests/test_weight_roundtrip.py` compares.

import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { ChessNet, type SerializedWeights } from '../src/game/ai/ChessNet';

const FIXTURE_DIR = path.resolve('training/tests/fixtures');
const WEIGHTS_PATH = path.join(FIXTURE_DIR, 'roundtrip_weights.json');
const BOARDS_PATH = path.join(FIXTURE_DIR, 'roundtrip_boards.json');
const OUT_PATH = path.join(FIXTURE_DIR, 'roundtrip_ts_outputs.json');

async function main(): Promise<void> {
  if (!fs.existsSync(WEIGHTS_PATH) || !fs.existsSync(BOARDS_PATH)) {
    console.error(
      'Missing fixture files. Run `python training/scripts/make_roundtrip_fixture.py` first.',
    );
    process.exit(1);
  }

  // Use the default CPU backend — good enough for parity checks, and in Node
  // with plain @tensorflow/tfjs there's no GPU option anyway.
  await tf.setBackend('cpu');
  await tf.ready();

  const weights = JSON.parse(
    fs.readFileSync(WEIGHTS_PATH, 'utf8'),
  ) as SerializedWeights;
  const boardsDoc = JSON.parse(
    fs.readFileSync(BOARDS_PATH, 'utf8'),
  ) as { boards: number[][] };

  if (!weights.config) {
    throw new Error('roundtrip_weights.json is missing embedded config');
  }

  const net = ChessNet.create({
    ...weights.config,
    learningRate: 0.001,
  });
  net.importWeights(weights);

  const boards = boardsDoc.boards.map(b => new Float32Array(b));
  const results = net.predictBatch(boards);

  const policies = results.map(r => Array.from(r.policy));
  const wdls = results.map(r => Array.from(r.wdl));

  fs.writeFileSync(
    OUT_PATH,
    JSON.stringify({
      numFilters: weights.config.numFilters,
      numResBlocks: weights.config.numResBlocks,
      policies,
      wdls,
    }),
  );

  console.log(
    `Wrote ${OUT_PATH} (${policies.length} positions, policy size ${policies[0].length}, wdl size ${wdls[0].length})`,
  );

  net.dispose();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
