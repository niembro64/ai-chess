// Monte Carlo Tree Search (AlphaZero-style)
// Supports batched evaluation across multiple concurrent games

import type { ChessGameState, Move } from '@/types/chess';
import { getLegalMoves, applyMove, positionKey, buildPositionCounts, isInCheck, isInsufficientMaterial } from '@/game/chess/ChessEngine';
import { encodeBoard, moveToIndex, POLICY_SIZE } from './ChessNet';

// The search only needs "boards in, (policy, value) out" — satisfied by
// both ChessNet (Sage, 20 planes) and ToyNet (Toy, 6 planes).
export interface PolicyValueNet {
  predictBatch(boards: Float32Array[]): Array<{ policy: Float32Array; value: number }>;
}

// Board encoder matching the net's input format. Defaults to Sage's
// 20-plane encoding; Toy passes its own 6-plane encoder.
export type BoardEncoder = (state: ChessGameState) => Float32Array;

// Mate-distance preference, mirroring training/src/chess_ai/mcts.py:
// terminal checkmate scores shrink slightly with depth so a mate the
// search can reach SOONER outranks the same mate further away. Value
// labels are never decayed (that corrupted an earlier training run);
// the preference lives here, in terminal handling. Symmetric, so a
// loss-seeking search likewise wants to be mated as soon as possible.
const MATE_DEPTH_DISCOUNT = 0.01;
const MATE_MIN_MAGNITUDE = 0.5;

function mateValue(depth: number): number {
  return Math.max(MATE_MIN_MAGNITUDE, 1 - MATE_DEPTH_DISCOUNT * depth);
}

const C_PUCT = 1.5;
// First-Play Urgency: unvisited children start at parent-Q minus this
// penalty instead of 0, matching the training-side Rust search
// (training/config.py fpu_reduction).
const FPU_REDUCTION = 0.4;

export type MCTSOptions = {
  // Board encoder matching the net (defaults to Sage's 20-plane).
  encode?: BoardEncoder;
  // Misère ("Jester") selection: at nodes whose side-to-move matches
  // this, PUCT MAXIMIZES the child's Q instead of negating it — the
  // mover is trying to LOSE. Value semantics stay truthful everywhere
  // (backprop, terminals); only what the search WANTS flips. Mirrors
  // training/src/chess_ai/mcts.py invert_turns.
  invertForTurn?: 'white' | 'black' | 'both';
  // Called as simulations complete (at yield points) so the UI can show
  // search progress. Single-game async search only.
  onProgress?: (completed: number, total: number) => void;
  // Replace root priors with uniform after expansion. Needed when a
  // WINNER's policy net drives an inverted search (instant-Jester on
  // Sage weights): the priors point at good moves, so the worst moves
  // would never accumulate PUCT exploration without this.
  flattenRootPriors?: boolean;
};

class MCTSNode {
  parent: MCTSNode | null = null;
  depth = 0;
  children: Map<number, MCTSNode> = new Map();
  move: Move | null = null;
  state: ChessGameState;

  visitCount = 0;
  totalValue = 0;
  prior = 0;

  isExpanded = false;
  isTerminal = false;
  terminalValue = 0;
  posKey: string | null = null;

  constructor(state: ChessGameState) {
    this.state = state;
  }
}

export type MCTSResult = {
  policy: Float32Array;
  move: Move;
  rootValue: number;
  // Every root move ordered by visit count, best first. Lets callers
  // apply post-search selection filters (e.g. the repetition veto in
  // AIPlayer) by walking down the search's own ranking.
  rankedMoves: { move: Move; visits: number }[];
};

// --- Batched MCTS for training (multiple games at once) ---

export class MCTSSearch {
  private root: MCTSNode;
  private pendingLeaf: MCTSNode | null = null;
  private readonly gameCounts: Map<string, number>;
  private encode: BoardEncoder;
  private invertForTurn: 'white' | 'black' | 'both' | undefined;

  constructor(
    state: ChessGameState,
    encode: BoardEncoder = encodeBoard,
    invertForTurn?: 'white' | 'black' | 'both',
    positionCounts?: Map<string, number>,
  ) {
    this.root = new MCTSNode(state);
    this.root.posKey = positionKey(state);
    const counts = positionCounts ?? (state.moveHistory.length ? buildPositionCounts(state) : new Map());
    this.gameCounts = new Map(counts);
    if (!this.gameCounts.has(this.root.posKey)) this.gameCounts.set(this.root.posKey, 1);
    this.encode = encode;
    this.invertForTurn = invertForTurn;
    this.checkTerminal(this.root);
  }

  isTerminal(): boolean {
    return this.root.isTerminal;
  }

  getRootBoard(): Float32Array {
    return this.encode(this.root.state);
  }

  // Initialize root with the NN evaluation.
  initRoot(policy: Float32Array, value: number, flattenPriors = false): void {
    this.expandWithPolicy(this.root, policy);
    if (flattenPriors) {
      const n = this.root.children.size;
      for (const child of this.root.children.values()) child.prior = 1 / n;
    }
    backpropagate(this.root, value);
  }

  // Select to leaf. Returns encoded board if NN eval needed, null if terminal.
  selectLeaf(): Float32Array | null {
    let node = this.root;
    const path = new Set([this.root.posKey!]);
    while (node.isExpanded && !node.isTerminal) {
      node = selectChild(node, node === this.root, this.invertForTurn);
      node.posKey ??= positionKey(node.state);
      if (!node.isTerminal && (path.has(node.posKey) || (this.gameCounts.get(node.posKey) ?? 0) + 1 >= 3)) {
        node.isTerminal = true;
        node.isExpanded = true;
        node.terminalValue = 0;
      }
      path.add(node.posKey);
    }

    if (node.isTerminal) {
      backpropagate(node, node.terminalValue);
      this.pendingLeaf = null;
      return null;
    }

    this.pendingLeaf = node;
    return this.encode(node.state);
  }

  // Supply NN evaluation for the pending leaf
  supplyEval(policy: Float32Array, value: number): void {
    if (!this.pendingLeaf) return;
    this.expandWithPolicy(this.pendingLeaf, policy);
    backpropagate(this.pendingLeaf, value);
    this.pendingLeaf = null;
  }

  // Get final result after all simulations. The move is the most-visited
  // root child; callers that want a different rule read `rankedMoves`.
  getResult(): MCTSResult {
    const policy = new Float32Array(POLICY_SIZE);
    let totalVisits = 0;
    for (const child of this.root.children.values()) {
      totalVisits += child.visitCount;
    }
    if (totalVisits > 0) {
      for (const [idx, child] of this.root.children) {
        policy[idx % POLICY_SIZE] += child.visitCount / totalVisits;
      }
    }
    const rankedMoves = [...this.root.children.values()]
      .filter(c => c.move !== null)
      .map(c => ({ move: c.move!, visits: c.visitCount }))
      .sort((a, b) => b.visits - a.visits);
    return {
      policy,
      move: argmaxMove(this.root),
      rootValue: this.root.visitCount > 0 ? this.root.totalValue / this.root.visitCount : 0,
      rankedMoves,
    };
  }

  private checkTerminal(node: MCTSNode): void {
    if (isInsufficientMaterial(node.state.board)) {
      node.isTerminal = true; node.isExpanded = true; node.terminalValue = 0;
      return;
    }
    const s = node.state.status;
    if (s === 'checkmate' || s === 'stalemate' || s === 'draw') {
      node.isTerminal = true;
      node.isExpanded = true;
      node.terminalValue = s === 'checkmate' ? -mateValue(node.depth) : 0;
    } else {
      const moves = getLegalMoves(node.state);
      if (moves.length === 0) {
        node.isTerminal = true;
        node.isExpanded = true;
        node.terminalValue = isInCheck(node.state.board, node.state.currentTurn) ? -mateValue(node.depth) : 0;
      }
    }
  }

  private expandWithPolicy(node: MCTSNode, policy: Float32Array): void {
    const state = node.state;
    const legalMoves = getLegalMoves(state);
    if (legalMoves.length === 0) {
      node.isTerminal = true;
      node.isExpanded = true;
      node.terminalValue = 0;
      return;
    }

    const isWhite = node.state.currentTurn === 'white';
    const counts = new Map<number, number>();
    for (const move of legalMoves) {
      const index = moveToIndex(move, isWhite);
      counts.set(index, (counts.get(index) ?? 0) + 1);
    }
    let priorSum = 0;
    for (const index of counts.keys()) priorSum += policy[index];
    for (const move of legalMoves) {
      const index = moveToIndex(move, isWhite);
      const child = new MCTSNode(applyMove(node.state, move));
      child.parent = node;
      child.depth = node.depth + 1;
      child.move = move;
      child.prior = priorSum > 0 ? policy[index] / priorSum / counts.get(index)! : 1 / legalMoves.length;
      this.checkTerminal(child);
      const offset = move.promotion === 'rook' ? 1 : move.promotion === 'bishop' ? 2 : move.promotion === 'knight' ? 3 : 0;
      node.children.set(index + offset * POLICY_SIZE, child);
    }
    node.isExpanded = true;
  }
}

// Run batched MCTS across multiple games. One GPU call per simulation step.
export function runBatchedMCTS(
  states: ChessGameState[],
  net: PolicyValueNet,
  numSimulations: number,
  options: MCTSOptions = {},
): MCTSResult[] {
  const searches = states.map(
    s => new MCTSSearch(s, options.encode ?? encodeBoard, options.invertForTurn),
  );

  // Filter to non-terminal games
  const active = searches.filter(s => !s.isTerminal());
  if (active.length === 0) {
    return searches.map(s => s.getResult());
  }

  // Batch-evaluate root positions
  const rootBoards = active.map(s => s.getRootBoard());
  const rootResults = net.predictBatch(rootBoards);
  for (let i = 0; i < active.length; i++) {
    active[i].initRoot(
      rootResults[i].policy, rootResults[i].value,
      options.flattenRootPriors ?? false,
    );
  }

  // Run simulation steps with batched leaf evaluation
  for (let sim = 0; sim < numSimulations; sim++) {
    const needsEval: { idx: number; board: Float32Array }[] = [];

    for (let i = 0; i < active.length; i++) {
      const board = active[i].selectLeaf();
      if (board !== null) {
        needsEval.push({ idx: i, board });
      }
    }

    if (needsEval.length > 0) {
      const results = net.predictBatch(needsEval.map(e => e.board));
      for (let i = 0; i < needsEval.length; i++) {
        active[needsEval[i].idx].supplyEval(results[i].policy, results[i].value);
      }
    }
  }

  return searches.map(s => s.getResult());
}

// --- Single-game MCTS (for gameplay, not training) ---

export function runMCTS(
  state: ChessGameState,
  net: PolicyValueNet,
  numSimulations: number,
  options: MCTSOptions = {},
): MCTSResult {
  return runBatchedMCTS([state], net, numSimulations, options)[0];
}

// Async single-game search that yields to the event loop every few
// simulations so a long think (e.g. 400 sims) doesn't freeze the tab.
export async function runMCTSAsync(
  state: ChessGameState,
  net: PolicyValueNet,
  numSimulations: number,
  options: MCTSOptions = {},
  yieldEverySims = 16,
): Promise<MCTSResult> {
  const search = new MCTSSearch(state, options.encode ?? encodeBoard, options.invertForTurn);
  if (search.isTerminal()) {
    return search.getResult();
  }

  const [rootEval] = net.predictBatch([search.getRootBoard()]);
  search.initRoot(
    rootEval.policy, rootEval.value, options.flattenRootPriors ?? false,
  );

  for (let sim = 0; sim < numSimulations; sim++) {
    const board = search.selectLeaf();
    if (board !== null) {
      const [res] = net.predictBatch([board]);
      search.supplyEval(res.policy, res.value);
    }
    if ((sim + 1) % yieldEverySims === 0) {
      options.onProgress?.(sim + 1, numSimulations);
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
  }

  options.onProgress?.(numSimulations, numSimulations);
  return search.getResult();
}

// --- Shared helpers ---

function selectChild(
  node: MCTSNode,
  isRoot: boolean,
  invertForTurn?: 'white' | 'black' | 'both',
): MCTSNode {
  let bestScore = -Infinity;
  let bestChild: MCTSNode | null = null;
  const sqrtParent = Math.sqrt(node.visitCount);
  // Misère inversion (see MCTSOptions.invertForTurn): at an inverted
  // node the mover WANTS to lose, so a child's Q is read as-is (the
  // child's own perspective IS the opponent's winning chances).
  const inverted = invertForTurn !== undefined
    && (invertForTurn === 'both' || node.state.currentTurn === invertForTurn);
  // node.totalValue accumulates values from node's own side-to-move
  // perspective, so a child's Q seen from the selecting node is NEGATED.
  // Without the negation the search prefers moves that are good for the
  // opponent (mate-in-1s score -1 and sink to the bottom) — the same
  // historical bug documented and fixed in training/src/chess_ai/mcts.py.
  const parentQ = node.visitCount > 0 ? node.totalValue / node.visitCount : 0;
  // FPU at non-root nodes only: the training search pairs root FPU with
  // Dirichlet root noise, but we play noiseless — root FPU without noise
  // lets the first-visited child starve its unvisited siblings forever
  // (KataGo makes the same root exception). Under inversion a child's
  // expected Q is ~ -parentQ, so the FPU baseline negates too.
  const baseQ = inverted ? -parentQ : parentQ;
  const unvisitedQ = isRoot ? baseQ : baseQ - FPU_REDUCTION;
  const sign = inverted ? 1 : -1;

  for (const child of node.children.values()) {
    const q = child.visitCount > 0
      ? sign * child.totalValue / child.visitCount
      : unvisitedQ;
    const u = C_PUCT * child.prior * sqrtParent / (1 + child.visitCount);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestChild = child;
    }
  }
  return bestChild!;
}

function backpropagate(node: MCTSNode, value: number): void {
  let current: MCTSNode | null = node;
  while (current !== null) {
    current.visitCount++;
    current.totalValue += value;
    value = -value;
    current = current.parent;
  }
}

function argmaxMove(root: MCTSNode): Move {
  let bestVisits = -1;
  let bestMove: Move | null = null;
  for (const child of root.children.values()) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestMove = child.move;
    }
  }
  return bestMove!;
}
