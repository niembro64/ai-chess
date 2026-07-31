// Tests for threefold-repetition enforcement + the AI repetition veto.
//
// Usage: npx tsx scripts/test_repetition.ts   (or: npm run test-repetition)

import {
  applyMove,
  buildPositionCounts,
  createInitialGameState,
  getLegalMoves,
  isInsufficientMaterial,
  positionKey,
  posToAlgebraic,
} from '../src/game/chess/ChessEngine';
import { ChessServer } from '../src/game/server/ChessServer';
import { pickNonRepeatingMove } from '../src/game/ai/AIPlayer';
import type { Board, ChessGameState, Move, PieceColor, PieceType } from '../src/types/chess';

let failures = 0;
function check(cond: boolean, label: string): void {
  if (cond) {
    console.log(`  PASS ${label}`);
  } else {
    console.error(`  FAIL ${label}`);
    failures++;
  }
}

function uciMove(state: ChessGameState, uci: string): Move {
  const move = getLegalMoves(state).find(
    m => posToAlgebraic(m.from) + posToAlgebraic(m.to) === uci,
  );
  if (!move) throw new Error(`move ${uci} not legal`);
  return move;
}

// --- positionKey semantics -------------------------------------------

console.log('positionKey:');
{
  const s0 = createInitialGameState();
  const s1 = createInitialGameState();
  check(positionKey(s0) === positionKey(s1), 'identical states share a key');

  // Same placement, different side to move → different position.
  const flipped = createInitialGameState();
  flipped.currentTurn = 'black';
  check(positionKey(s0) !== positionKey(flipped), 'side to move distinguishes');

  // Same placement, different castling rights → different position.
  const noCastle = createInitialGameState();
  noCastle.castlingRights.whiteKingside = false;
  check(positionKey(s0) !== positionKey(noCastle), 'castling rights distinguish');

  // En passant target distinguishes.
  const afterE4 = applyMove(s0, uciMove(s0, 'e2e4'));
  const viaTwoSteps = (() => {
    let s = createInitialGameState();
    s = applyMove(s, uciMove(s, 'e2e3'));
    // Need same placement with e-pawn on e4 & black to move but NO ep:
    // play e3-e4 after a black null-ish shuffle is impossible; instead
    // just compare afterE4 (ep=e3) against a manual copy with ep nulled.
    return null;
  })();
  void viaTwoSteps;
  const noEp = { ...afterE4, enPassantTarget: null };
  check(positionKey(afterE4) !== positionKey(noEp as ChessGameState), 'en passant target distinguishes');
}

// --- buildPositionCounts ----------------------------------------------

console.log('buildPositionCounts:');
{
  let s = createInitialGameState();
  s.status = 'active';
  const startKey = positionKey(s);
  for (const uci of ['g1f3', 'g8f6', 'f3g1', 'f6g8']) {
    s = applyMove(s, uciMove(s, uci));
  }
  const counts = buildPositionCounts(s);
  check(counts.get(startKey) === 2, 'start position seeded, then recurs (count 2 after knight shuffle)');
}

// --- ChessServer: auto-draw on the third occurrence --------------------

console.log('ChessServer threefold enforcement:');
{
  const server = new ChessServer();
  let latest: ChessGameState | null = null;
  server.addSnapshotListener(snap => { latest = snap.gameState; });
  server.start();

  // 1.Nf3 Nf6 2.Ng1 Ng8 2nd occurrence of start; 3.Nf3 Nf6 4.Ng1 Ng8 → 3rd.
  const cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  const moves = [...cycle, ...cycle];
  for (let i = 0; i < moves.length; i++) {
    const state = latest!;
    const move = uciMove(state, moves[i]);
    server.receiveCommand({ type: 'move', move }, (i % 2 === 0 ? 1 : 2));
    if (i < moves.length - 1) {
      check(latest!.status !== 'draw', `no draw yet after ply ${i + 1}`);
    }
  }
  check(latest!.status === 'draw', 'draw declared on third occurrence');
  check((latest! as ChessGameState).drawReason === 'repetition', 'reason is repetition');
}

// --- isInsufficientMaterial --------------------------------------------

console.log('isInsufficientMaterial:');
{
  const emptyBoard = (): Board =>
    Array.from({ length: 8 }, () => Array(8).fill(null)) as Board;
  const place = (b: Board, r: number, f: number, color: PieceColor, type: PieceType) => {
    b[r][f] = { color, type };
  };

  const kk = emptyBoard();
  place(kk, 0, 4, 'black', 'king'); place(kk, 7, 4, 'white', 'king');
  check(isInsufficientMaterial(kk), 'K vs K');

  const kbk = emptyBoard();
  place(kbk, 0, 4, 'black', 'king'); place(kbk, 7, 4, 'white', 'king');
  place(kbk, 4, 2, 'white', 'bishop');
  check(isInsufficientMaterial(kbk), 'K+B vs K');

  const knk = emptyBoard();
  place(knk, 0, 4, 'black', 'king'); place(knk, 7, 4, 'white', 'king');
  place(knk, 4, 2, 'black', 'knight');
  check(isInsufficientMaterial(knk), 'K+N vs K');

  const kbkbSame = emptyBoard();
  place(kbkbSame, 0, 4, 'black', 'king'); place(kbkbSame, 7, 4, 'white', 'king');
  place(kbkbSame, 4, 2, 'white', 'bishop');   // (4+2)%2 = 0
  place(kbkbSame, 2, 0, 'black', 'bishop');   // (2+0)%2 = 0 — same color
  check(isInsufficientMaterial(kbkbSame), 'KB vs KB, same-color bishops');

  const kbkbDiff = emptyBoard();
  place(kbkbDiff, 0, 4, 'black', 'king'); place(kbkbDiff, 7, 4, 'white', 'king');
  place(kbkbDiff, 4, 2, 'white', 'bishop');   // 0
  place(kbkbDiff, 2, 1, 'black', 'bishop');   // 1 — opposite color
  check(!isInsufficientMaterial(kbkbDiff), 'KB vs KB, opposite bishops = NOT insufficient');

  const knkn = emptyBoard();
  place(knkn, 0, 4, 'black', 'king'); place(knkn, 7, 4, 'white', 'king');
  place(knkn, 4, 2, 'white', 'knight'); place(knkn, 3, 3, 'black', 'knight');
  check(!isInsufficientMaterial(knkn), 'K+N vs K+N = NOT insufficient (helpmate exists)');

  const kpk = emptyBoard();
  place(kpk, 0, 4, 'black', 'king'); place(kpk, 7, 4, 'white', 'king');
  place(kpk, 5, 3, 'white', 'pawn');
  check(!isInsufficientMaterial(kpk), 'K+P vs K = NOT insufficient');
}

// --- pickNonRepeatingMove (the AI veto) ---------------------------------

console.log('pickNonRepeatingMove:');
{
  const state = createInitialGameState();
  state.status = 'active';
  const nf3 = uciMove(state, 'g1f3');
  const e4 = uciMove(state, 'e2e4');
  const ranked = [
    { move: nf3, visits: 300 },  // search's favorite
    { move: e4, visits: 100 },
  ];
  const keyAfterNf3 = positionKey(applyMove(state, nf3));

  // Nf3's resulting position already occurred twice → playing it would be
  // the third occurrence. Neutral eval: veto must fall through to e4.
  const counts2 = new Map([[keyAfterNf3, 2]]);
  check(
    pickNonRepeatingMove(state, ranked, counts2, 0.0) === e4,
    'third-occurrence move vetoed, next-ranked move plays',
  );

  // Same map but the bot is losing badly: no veto (draw saves half a point).
  check(
    pickNonRepeatingMove(state, ranked, counts2, -0.5) === null,
    'no veto while losing — repetition draw is welcome',
  );

  // Winning strictly: even a SECOND occurrence is dodged.
  const counts1 = new Map([[keyAfterNf3, 1]]);
  check(
    pickNonRepeatingMove(state, ranked, counts1, 0.5) === e4,
    'second occurrence dodged when clearly winning',
  );
  // ...but with a neutral eval a mere second occurrence is fine.
  check(
    pickNonRepeatingMove(state, ranked, counts1, 0.0) === nf3,
    'second occurrence allowed at neutral eval',
  );

  // Every move repeats → forced; keep the search's choice.
  const keyAfterE4 = positionKey(applyMove(state, e4));
  const allRepeat = new Map([[keyAfterNf3, 2], [keyAfterE4, 2]]);
  check(
    pickNonRepeatingMove(state, ranked, allRepeat, 0.0) === null,
    'all moves repeat → forced draw, keep search choice',
  );
}

if (failures > 0) {
  console.error(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log('\nAll repetition tests passed.');
