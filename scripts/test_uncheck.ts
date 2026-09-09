import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  applyMove,
  createInitialGameState,
  getLegalMoves,
  isInCheck,
  positionKey,
  posToAlgebraic,
} from '../src/game/chess/ChessEngine';
import { MCTSSearch } from '../src/game/ai/MCTS';
import { ChessServer } from '../src/game/server/ChessServer';
import type { ChessGameState, GameStatus, Piece, PieceColor, PieceType } from '../src/types/chess';

type LegalCase = { id: string; fen: string; exact?: string[]; includes?: string[]; excludes?: string[] };
type TerminalCase = { id: string; fen: string; winner: PieceColor | null; attacked?: boolean };
type MoveCase = {
  id: string; fen: string; move: string; status: GameStatus; turn?: PieceColor;
  winner?: PieceColor; attacked?: PieceColor; empty?: string;
};
type Fixtures = { ruleset: 'uncheck-v1'; legalCases: LegalCase[]; terminalRoots: TerminalCase[]; moveCases: MoveCase[] };

const fixtures = JSON.parse(readFileSync(new URL('../fixtures/uncheck-v1.json', import.meta.url), 'utf8')) as Fixtures;
const types: Record<string, PieceType> = { k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' };

function stateFromFen(fen: string): ChessGameState {
  const [placement, turn, castling, ep, halfmove, fullmove] = fen.split(' ');
  const state = createInitialGameState(fixtures.ruleset);
  state.board = Array.from({ length: 8 }, () => Array<Piece | null>(8).fill(null));
  for (const [rank, encoded] of placement.split('/').entries()) {
    let file = 0;
    for (const char of encoded) {
      if (/\d/.test(char)) file += Number(char);
      else {
        state.board[rank][file++] = {
          color: char === char.toUpperCase() ? 'white' : 'black',
          type: types[char.toLowerCase()],
        };
      }
    }
  }
  state.currentTurn = turn === 'w' ? 'white' : 'black';
  state.castlingRights = {
    whiteKingside: castling.includes('K'), whiteQueenside: castling.includes('Q'),
    blackKingside: castling.includes('k'), blackQueenside: castling.includes('q'),
  };
  state.enPassantTarget = ep === '-' ? null : { rank: 8 - Number(ep[1]), file: ep.charCodeAt(0) - 97 };
  state.halfMoveClock = Number(halfmove);
  state.fullMoveNumber = Number(fullmove);
  state.status = 'active';
  state.winner = null;
  state.moveHistory = [];
  state.lastMove = null;
  return state;
}

function legalUci(state: ChessGameState): string[] {
  return getLegalMoves(state).map(move =>
    posToAlgebraic(move.from) + posToAlgebraic(move.to) +
    ({ queen: 'q', rook: 'r', bishop: 'b', knight: 'n' }[move.promotion ?? ''] ?? ''),
  ).sort();
}

function moveUci(move: ReturnType<typeof getLegalMoves>[number]): string {
  return posToAlgebraic(move.from) + posToAlgebraic(move.to) +
    ({ queen: 'q', rook: 'r', bishop: 'b', knight: 'n' }[move.promotion ?? ''] ?? '');
}

for (const fixture of fixtures.legalCases) {
  const legal = legalUci(stateFromFen(fixture.fen));
  if (fixture.exact) assert.deepEqual(legal, fixture.exact, fixture.id);
  for (const move of fixture.includes ?? []) assert.ok(legal.includes(move), `${fixture.id}: missing ${move}`);
  for (const move of fixture.excludes ?? []) assert.ok(!legal.includes(move), `${fixture.id}: unexpectedly allowed ${move}`);
}

for (const fixture of fixtures.terminalRoots) {
  const state = stateFromFen(fixture.fen);
  assert.equal(isInCheck(state.board, state.currentTurn), fixture.attacked ?? true, fixture.id);
  assert.deepEqual(legalUci(state), [], fixture.id);
  assert.equal(new MCTSSearch(state).isTerminal(), true, `${fixture.id}: MCTS root`);
}

for (const fixture of fixtures.moveCases) {
  const state = stateFromFen(fixture.fen);
  const move = getLegalMoves(state).find(candidate => moveUci(candidate) === fixture.move);
  assert.ok(move, `${fixture.id}: ${fixture.move} is not permitted`);
  const next = applyMove(state, move);
  assert.equal(next.status, fixture.status, `${fixture.id}: status`);
  if (fixture.turn) assert.equal(next.currentTurn, fixture.turn, `${fixture.id}: turn`);
  if (fixture.winner) assert.equal(next.winner, fixture.winner, `${fixture.id}: winner`);
  if (fixture.attacked) assert.equal(isInCheck(next.board, fixture.attacked), true, `${fixture.id}: attack`);
  if (fixture.empty) {
    const file = fixture.empty.charCodeAt(0) - 97;
    const rank = 8 - Number(fixture.empty[1]);
    assert.equal(next.board[rank][file], null, `${fixture.id}: ${fixture.empty} not empty`);
  }
}

{
  const unavailable = stateFromFen('7k/8/8/8/8/8/8/K7 w - d6 0 1');
  assert.equal(positionKey(unavailable), positionKey({ ...unavailable, enPassantTarget: null }), 'ineffective Uncheck en passant key');
  const available = stateFromFen('7k/8/8/3pP3/8/8/8/K7 w - d6 0 2');
  assert.notEqual(positionKey(available), positionKey({ ...available, enPassantTarget: null }), 'effective Uncheck en passant key');
}

{
  const server = new ChessServer('uncheck-v1');
  let latest: ChessGameState | null = null;
  server.addSnapshotListener(snapshot => { latest = snapshot.gameState; });
  server.start();
  const moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'];
  for (const [index, uci] of moves.entries()) {
    const move = getLegalMoves(latest!).find(candidate => moveUci(candidate) === uci);
    assert.ok(move, `repetition move ${uci}`);
    server.receiveCommand({ type: 'move', move }, index % 2 === 0 ? 1 : 2);
    if (index < moves.length - 1) assert.notEqual(latest!.status, 'draw', `early repetition at ${index + 1}`);
  }
  assert.equal(latest!.status, 'draw');
  assert.equal(latest!.drawReason, 'repetition');
}

console.log(`PASS: ${fixtures.legalCases.length + fixtures.terminalRoots.length + fixtures.moveCases.length} Uncheck v1 rule fixtures`);
