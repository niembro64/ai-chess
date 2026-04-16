// Dumps random chess positions produced by the TS engine into JSON for the
// Python parity test. The Python engine must reproduce every field here
// byte-for-byte (legal moves, encoded board, move indices).
//
// Usage:
//   npx tsx scripts/dump_parity.ts [numPositions] [outputPath]
//
// Default output path: training/tests/fixtures/parity_positions.json
// Default position count: 1000

import fs from 'node:fs';
import path from 'node:path';
import {
  createInitialGameState,
  getLegalMoves,
  applyMove,
} from '../src/game/chess/ChessEngine';
import {
  encodeBoard,
  moveToIndex,
  NUM_PLANES,
  POLICY_SIZE,
} from '../src/game/ai/ChessNet';
import {
  evaluatePosition,
  DEFAULT_REWARD_WEIGHTS,
} from '../src/game/ai/rewardShaping';
import type { ChessGameState } from '../src/types/chess';

const numPositions = Number.parseInt(process.argv[2] ?? '1000', 10);
const outPath = path.resolve(
  process.argv[3] ?? 'training/tests/fixtures/parity_positions.json',
);

// Seeded RNG (mulberry32) so dumps are reproducible across runs.
function makeRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = makeRng(42);

function serializeState(state: ChessGameState) {
  return {
    board: state.board.map(row =>
      row.map(sq => (sq ? { color: sq.color, type: sq.type } : null)),
    ),
    currentTurn: state.currentTurn,
    castlingRights: { ...state.castlingRights },
    enPassantTarget: state.enPassantTarget ? { ...state.enPassantTarget } : null,
    halfMoveClock: state.halfMoveClock,
    fullMoveNumber: state.fullMoveNumber,
    status: state.status,
  };
}

function isOver(status: string): boolean {
  return status === 'checkmate' || status === 'stalemate' || status === 'draw';
}

function generate(n: number) {
  const positions: unknown[] = [];
  let state: ChessGameState = { ...createInitialGameState(), status: 'active' };

  while (positions.length < n) {
    if (isOver(state.status)) {
      state = { ...createInitialGameState(), status: 'active' };
    }

    // 2% chance to restart from the opening to keep coverage dense there
    if (rng() < 0.02) {
      state = { ...createInitialGameState(), status: 'active' };
    }

    const legal = getLegalMoves(state);
    if (legal.length === 0) {
      state = { ...createInitialGameState(), status: 'active' };
      continue;
    }

    const isWhite = state.currentTurn === 'white';
    const encoded = encodeBoard(state);
    const moveIndices = legal.map(m => moveToIndex(m, isWhite));
    // Shaped reward from each color's perspective, using the default weights.
    const rewardWhite = evaluatePosition(state, 'white', DEFAULT_REWARD_WEIGHTS, legal.length);
    const rewardBlack = evaluatePosition(state, 'black', DEFAULT_REWARD_WEIGHTS, legal.length);

    positions.push({
      id: positions.length,
      state: serializeState(state),
      legalMoves: legal.map(m => ({
        from: { ...m.from },
        to: { ...m.to },
        ...(m.promotion ? { promotion: m.promotion } : {}),
      })),
      legalMoveIndices: moveIndices,
      encodedBoard: Array.from(encoded),
      rewardWhite,
      rewardBlack,
      legalMoveCount: legal.length,
    });

    const move = legal[Math.floor(rng() * legal.length)];
    state = applyMove(state, move);
  }

  return positions;
}

console.log(`Generating ${numPositions} parity positions → ${outPath}`);
const positions = generate(numPositions);

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(
  outPath,
  JSON.stringify({
    version: 2,
    seed: 42,
    num_planes: NUM_PLANES,
    policy_size: POLICY_SIZE,
    reward_weights: DEFAULT_REWARD_WEIGHTS,
    positions,
  }),
);

const bytes = fs.statSync(outPath).size;
console.log(
  `Wrote ${positions.length} positions (${(bytes / 1024 / 1024).toFixed(2)} MB)`,
);
