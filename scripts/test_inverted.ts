import assert from 'node:assert/strict';
import { createInitialGameState, getLegalMoves, applyMove, positionKey } from '../src/game/chess/ChessEngine';
import { MCTSSearch } from '../src/game/ai/MCTS';
import { POLICY_SIZE, moveToIndex } from '../src/game/ai/ChessNet';
import { rankByDistribution } from '../src/game/ai/AIPlayer';
import type { ChessGameState, Piece } from '../src/types/chess';

const policy = new Float32Array(POLICY_SIZE).fill(1 / POLICY_SIZE);
function forced(): ChessGameState {
  const state = createInitialGameState();
  state.board = Array.from({ length: 8 }, () => Array<Piece | null>(8).fill(null));
  for (const [square, piece] of Object.entries({ h8: ['white','king'], c6: ['white','queen'],
      b8: ['black','king'], a8: ['black','rook'], f5: ['black','bishop'], h5: ['black','knight'] })) {
    state.board[8-Number(square[1])][square.charCodeAt(0)-97] = {color:piece[0],type:piece[1]} as Piece;
  }
  state.castlingRights = { whiteKingside:false,whiteQueenside:false,blackKingside:false,blackQueenside:false };
  state.status = 'active';
  return state;
}
function search(state: ChessGameState, counts?: Map<string,number>) {
  const search = new MCTSSearch(state, undefined, 'both', counts);
  search.initRoot(policy, 0);
  for (let i=0;i<400;i++) if(search.selectLeaf()) search.supplyEval(policy,0);
  return search.getResult();
}
const state=forced();
const result=search(state);
const after=applyMove(state,result.move);
const replies=getLegalMoves(after);
assert.ok(replies.length>0 && replies.every(m=>applyMove(after,m).status==='checkmate'), 'must force selfmate against all replies');
const terminal=applyMove(after,replies[0]);
assert.equal(terminal.currentTurn,'white');

// Both forced alternatives repeat a previous position: draw-aware search
// must report neutral value instead of hallucinating an immediate win.
const counts = new Map([[positionKey(state),1]]);
for(const m of getLegalMoves(state)) counts.set(positionKey(applyMove(state,m)),2);
assert.equal(search(state,counts).rootValue,0);

const promoted=createInitialGameState();
promoted.board=Array.from({length:8},()=>Array<Piece|null>(8).fill(null));
promoted.board[0][7]={color:'black',type:'king'};
promoted.board[1][0]={color:'white',type:'pawn'};
promoted.board[7][7]={color:'white',type:'king'};
promoted.castlingRights=state.castlingRights;
promoted.status='active';
const promotionResult=search(promoted);
assert.equal(promotionResult.rankedMoves.filter(r=>r.move.promotion).length,4);
assert.ok(Math.abs(promotionResult.policy.reduce((a,b)=>a+b,0)-1)<1e-5);
const moves=getLegalMoves(promoted);
const knight=moves.find(m=>m.promotion==='knight')!;
const entries=rankByDistribution(policy,moves,true,[{move:knight,visits:100}]);
assert.equal(entries.find(e=>e.index===moveToIndex(knight,true))!.move!.promotion,'knight');
console.log('PASS: forced selfmate, in-tree repetition, and promotion choices');
