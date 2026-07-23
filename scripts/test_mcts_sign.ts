// Regression test for the MCTS Q-sign convention in the browser search.
//
// Child nodes store values from the child's side-to-move perspective, so
// selection must NEGATE child Q. With the sign inverted, search actively
// avoids mating moves (a mate-in-1 child holds terminalValue = -1 and
// sinks to the bottom) — the historical bug fixed in the Python/Rust
// training search but originally shipped un-ported here.
//
// Scenario: fool's mate. After 1.f3 e5 2.g4 black to move has Qd8-h4#.
// With uniform priors and a neutral value net, a correct 150-sim search
// must pick the mate; the sign-inverted search never does.
//
// Usage: npx tsx scripts/test_mcts_sign.ts

import {
  createInitialGameState,
  getLegalMoves,
  applyMove,
  posToAlgebraic,
} from '../src/game/chess/ChessEngine';
import { POLICY_SIZE } from '../src/game/ai/ChessNet';
import { MCTSSearch } from '../src/game/ai/MCTS';
import type { ChessGameState } from '../src/types/chess';

function playUci(state: ChessGameState, uci: string): ChessGameState {
  const move = getLegalMoves(state).find(
    m => posToAlgebraic(m.from) + posToAlgebraic(m.to) === uci,
  );
  if (!move) throw new Error(`move ${uci} not legal in position`);
  return applyMove(state, move);
}

let state = createInitialGameState();
state = playUci(state, 'f2f3');
state = playUci(state, 'e7e5');
state = playUci(state, 'g2g4');

const NUM_SIMS = 150;
const uniformPolicy = new Float32Array(POLICY_SIZE).fill(1 / POLICY_SIZE);

const search = new MCTSSearch(state);
if (search.isTerminal()) throw new Error('test position should not be terminal');

// Neutral net: uniform priors, value 0 — all signal must come from the
// terminal mate value propagating with correct signs.
search.initRoot(uniformPolicy, 0);
for (let sim = 0; sim < NUM_SIMS; sim++) {
  const board = search.selectLeaf();
  if (board !== null) {
    search.supplyEval(uniformPolicy, 0);
  }
}

const result = search.getResult();
const chosen = posToAlgebraic(result.move.from) + posToAlgebraic(result.move.to);

let failures = 0;
if (chosen !== 'd8h4') {
  console.error(`FAIL: expected mating move d8h4, search chose ${chosen}`);
  failures++;
}
if (result.rootValue < 0.5) {
  console.error(
    `FAIL: root value should approach +1 with a found mate, got ${result.rootValue.toFixed(3)}`,
  );
  failures++;
}

if (failures > 0) {
  process.exit(1);
}
console.log(
  `PASS: ${NUM_SIMS}-sim search picks ${chosen} with root value ${result.rootValue.toFixed(3)}`,
);
