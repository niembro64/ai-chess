// Parity check: ToyNet.ts forward pass vs the PyTorch net that exported
// public/models/toy.json. The fixture (training/tests/fixtures/
// toy_parity.json, written by `toy_train.py --dump-fixture`) holds
// encoded boards plus PyTorch's softmax policies and tanh values for
// the SAME checkpoint. fp16 storage + fp32 math means small deviations;
// tolerance reflects that.
//
// Also pins the TS 6-plane encoder against the conventions the Python
// tests pin (sign, channel order, rotation).
//
// Usage: npx tsx scripts/test... -> npm run verify-toy

import fs from 'node:fs';
import path from 'node:path';
import '@tensorflow/tfjs';
import { ToyNet, encodeToyBoard, isToyWeights } from '../src/game/ai/ToyNet';
import { ToyPlayer, type ToyThought } from '../src/game/ai/ToyPlayer';
import {
  applyMove,
  createInitialGameState,
  getLegalMoves,
  posToAlgebraic,
} from '../src/game/chess/ChessEngine';

let failures = 0;
function check(cond: boolean, label: string): void {
  if (cond) console.log(`  PASS ${label}`);
  else { console.error(`  FAIL ${label}`); failures++; }
}

// --- encoder pins ------------------------------------------------------

console.log('encodeToyBoard:');
{
  const s = createInitialGameState();
  const x = encodeToyBoard(s);
  const at = (r: number, f: number, ch: number) => x[(r * 8 + f) * 6 + ch];
  check(at(6, 0, 0) === 1, 'white pawn rank2 = +1 in channel 0 (white to move)');
  check(at(1, 0, 0) === -1, 'black pawn rank7 = -1');
  check(at(7, 4, 5) === 1, 'own king at e1 = +1 in channel 5');
  check(at(0, 4, 5) === -1, 'opponent king = -1');

  const sb = createInitialGameState();
  sb.currentTurn = 'black';
  const xb = encodeToyBoard(sb);
  // Rotation property (see test_toy_smoke.py): black's view of the
  // start position equals white's view file-mirrored.
  let mirrorOk = true;
  for (let r = 0; r < 8 && mirrorOk; r++) {
    for (let f = 0; f < 8 && mirrorOk; f++) {
      for (let ch = 0; ch < 6; ch++) {
        if (xb[(r * 8 + f) * 6 + ch] !== x[(r * 8 + (7 - f)) * 6 + ch]) {
          mirrorOk = false;
          break;
        }
      }
    }
  }
  check(mirrorOk, 'black-to-move view = white view file-mirrored');
}

// --- forward parity vs PyTorch ------------------------------------------

console.log('forward parity:');
{
  const weightsPath = path.resolve('public/models/toy.json');
  const fixturePath = path.resolve('training/tests/fixtures/toy_parity.json');
  const weights = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as {
    boards: number[][]; policies: number[][]; values: number[];
  };
  if (!isToyWeights(weights)) throw new Error('public/models/toy.json is not toy-v1');

  const net = ToyNet.create(weights);
  const boards = fixture.boards.map(b => Float32Array.from(b));
  const results = net.predictBatch(boards);

  let maxPolicyErr = 0;
  let maxValueErr = 0;
  for (let i = 0; i < boards.length; i++) {
    maxValueErr = Math.max(maxValueErr, Math.abs(results[i].value - fixture.values[i]));
    const expected = fixture.policies[i];
    for (let j = 0; j < 4096; j++) {
      maxPolicyErr = Math.max(maxPolicyErr, Math.abs(results[i].policy[j] - expected[j]));
    }
    // The strongest functional check: identical argmax.
    let tsBest = 0, ptBest = 0;
    for (let j = 1; j < 4096; j++) {
      if (results[i].policy[j] > results[i].policy[tsBest]) tsBest = j;
      if (expected[j] > expected[ptBest]) ptBest = j;
    }
    check(tsBest === ptBest, `position ${i}: policy argmax matches (${tsBest})`);
  }
  console.log(`  max |Δpolicy| = ${maxPolicyErr.toExponential(2)}, max |Δvalue| = ${maxValueErr.toExponential(2)}`);
  check(maxPolicyErr < 3e-3, 'policy within fp16 tolerance');
  check(maxValueErr < 3e-3, 'value within fp16 tolerance');
  net.dispose();
}

// --- terminal observation (checkmated bot still shows its brain) --------

console.log('observeTerminal:');
{
  const weights = JSON.parse(fs.readFileSync(path.resolve('public/models/toy.json'), 'utf8'));
  let thought: ToyThought | null = null;
  const player = ToyPlayer.create(weights, 8, t => { thought = t; });

  // Fool's mate: white is checkmated with white to move.
  let s = createInitialGameState();
  s.status = 'active';
  for (const uci of ['f2f3', 'e7e5', 'g2g4', 'd8h4']) {
    const move = getLegalMoves(s).find(
      m => posToAlgebraic(m.from) + posToAlgebraic(m.to) === uci,
    )!;
    s = applyMove(s, move);
  }
  check(s.status === 'checkmate', 'position is checkmate');

  player.observeTerminal(s);
  check(thought !== null, 'terminal thought emitted');
  const t = thought! as ToyThought;
  check(
    t.entries.every(e => !e.legal),
    'every entry is illegal at a terminal position (all red)',
  );
  check(t.entries.length === 4096, 'the whole policy space is listed');
  check(
    t.entries.every((e, i) => i === 0 || t.entries[i - 1].p >= e.p),
    'entries are ordered by probability, highest first',
  );
  check(t.rawPolicy.length === 4096 && Math.abs(
    t.rawPolicy.reduce((a, b) => a + b, 0) - 1,
  ) < 1e-3, 'forward pass still produced a softmax policy');
  check(Number.isFinite(t.value), `value head evaluated the mate (${t.value.toFixed(2)})`);
  player.dispose();
}

if (failures > 0) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log('\nToy parity verified.');
