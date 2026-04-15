// Monte Carlo Tree Search (AlphaZero-style)
// Supports batched evaluation across multiple concurrent games

import type { ChessGameState, Move } from '@/types/chess';
import { getLegalMoves, applyMove } from '@/game/chess/ChessEngine';
import { ChessNet, encodeBoard, moveToIndex, POLICY_SIZE } from './ChessNet';

const C_PUCT = 1.5;
const DIRICHLET_ALPHA = 0.3;
const DIRICHLET_EPSILON = 0.25;

class MCTSNode {
  parent: MCTSNode | null = null;
  children: Map<number, MCTSNode> = new Map();
  move: Move | null = null;
  state: ChessGameState;

  visitCount = 0;
  totalValue = 0;
  prior = 0;

  isExpanded = false;
  isTerminal = false;
  terminalValue = 0;

  constructor(state: ChessGameState) {
    this.state = state;
  }
}

export type MCTSResult = {
  policy: Float32Array;
  move: Move;
  rootValue: number;
};

// --- Batched MCTS for training (multiple games at once) ---

export class MCTSSearch {
  private root: MCTSNode;
  private pendingLeaf: MCTSNode | null = null;

  constructor(state: ChessGameState) {
    this.root = new MCTSNode(state);
    this.checkTerminal(this.root);
  }

  isTerminal(): boolean {
    return this.root.isTerminal;
  }

  getRootBoard(): Float32Array {
    return encodeBoard(this.root.state);
  }

  // Initialize root with NN evaluation + Dirichlet noise
  initRoot(policy: Float32Array, value: number): void {
    this.expandWithPolicy(this.root, policy);
    addDirichletNoise(this.root);
    backpropagate(this.root, value);
  }

  // Select to leaf. Returns encoded board if NN eval needed, null if terminal.
  selectLeaf(): Float32Array | null {
    let node = this.root;
    while (node.isExpanded && !node.isTerminal) {
      node = selectChild(node);
    }

    if (node.isTerminal) {
      backpropagate(node, node.terminalValue);
      this.pendingLeaf = null;
      return null;
    }

    this.pendingLeaf = node;
    return encodeBoard(node.state);
  }

  // Supply NN evaluation for the pending leaf
  supplyEval(policy: Float32Array, value: number): void {
    if (!this.pendingLeaf) return;
    this.expandWithPolicy(this.pendingLeaf, policy);
    backpropagate(this.pendingLeaf, value);
    this.pendingLeaf = null;
  }

  // Get final result after all simulations
  getResult(): MCTSResult {
    const policy = new Float32Array(POLICY_SIZE);
    let totalVisits = 0;
    for (const child of this.root.children.values()) {
      totalVisits += child.visitCount;
    }
    if (totalVisits > 0) {
      for (const [idx, child] of this.root.children) {
        policy[idx] = child.visitCount / totalVisits;
      }
    }
    return {
      policy,
      move: sampleMove(this.root),
      rootValue: this.root.visitCount > 0 ? this.root.totalValue / this.root.visitCount : 0,
    };
  }

  private checkTerminal(node: MCTSNode): void {
    const s = node.state.status;
    if (s === 'checkmate' || s === 'stalemate' || s === 'draw') {
      node.isTerminal = true;
      node.isExpanded = true;
      node.terminalValue = s === 'checkmate' ? -1 : 0;
    } else {
      const moves = getLegalMoves(node.state);
      if (moves.length === 0) {
        node.isTerminal = true;
        node.isExpanded = true;
        node.terminalValue = 0;
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

    const isWhite = state.currentTurn === 'white';
    const seenIndices = new Set<number>();
    let priorSum = 0;
    for (const move of legalMoves) {
      const mi = moveToIndex(move, isWhite);
      if (!seenIndices.has(mi)) {
        seenIndices.add(mi);
        priorSum += policy[mi];
      }
    }

    seenIndices.clear();
    for (const move of legalMoves) {
      const mi = moveToIndex(move, isWhite);
      if (seenIndices.has(mi)) continue;
      seenIndices.add(mi);

      const child = new MCTSNode(applyMove(state, move));
      child.parent = node;
      child.move = move;
      child.prior = priorSum > 0 ? policy[mi] / priorSum : 1 / seenIndices.size;
      this.checkTerminal(child);
      node.children.set(mi, child);
    }

    node.isExpanded = true;
  }
}

// Run batched MCTS across multiple games. One GPU call per simulation step.
export function runBatchedMCTS(
  states: ChessGameState[],
  net: ChessNet,
  numSimulations: number,
): MCTSResult[] {
  const searches = states.map(s => new MCTSSearch(s));

  // Filter to non-terminal games
  const active = searches.filter(s => !s.isTerminal());
  if (active.length === 0) {
    return searches.map(s => s.getResult());
  }

  // Batch-evaluate root positions
  const rootBoards = active.map(s => s.getRootBoard());
  const rootResults = net.predictBatch(rootBoards);
  for (let i = 0; i < active.length; i++) {
    active[i].initRoot(rootResults[i].policy, rootResults[i].value);
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
  net: ChessNet,
  numSimulations: number,
): MCTSResult {
  return runBatchedMCTS([state], net, numSimulations)[0];
}

// --- Shared helpers ---

function selectChild(node: MCTSNode): MCTSNode {
  let bestScore = -Infinity;
  let bestChild: MCTSNode | null = null;
  const sqrtParent = Math.sqrt(node.visitCount);

  for (const child of node.children.values()) {
    const q = child.visitCount > 0 ? child.totalValue / child.visitCount : 0;
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

function sampleMove(root: MCTSNode): Move {
  let totalVisits = 0;
  for (const child of root.children.values()) {
    totalVisits += child.visitCount;
  }

  let r = Math.random() * totalVisits;
  for (const child of root.children.values()) {
    r -= child.visitCount;
    if (r <= 0) return child.move!;
  }

  let bestVisits = 0;
  let bestMove: Move | null = null;
  for (const child of root.children.values()) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestMove = child.move;
    }
  }
  return bestMove!;
}

function addDirichletNoise(root: MCTSNode): void {
  const n = root.children.size;
  if (n === 0) return;
  const noise = dirichletSample(n, DIRICHLET_ALPHA);
  let i = 0;
  for (const child of root.children.values()) {
    child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * noise[i];
    i++;
  }
}

function dirichletSample(n: number, alpha: number): number[] {
  const samples: number[] = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const g = gammaSample(alpha);
    samples.push(g);
    sum += g;
  }
  if (sum > 0) {
    for (let i = 0; i < n; i++) samples[i] /= sum;
  } else {
    for (let i = 0; i < n; i++) samples[i] = 1 / n;
  }
  return samples;
}

function gammaSample(shape: number): number {
  if (shape < 1) return gammaSample(shape + 1) * Math.pow(Math.random(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do { x = randn(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function randn(): number {
  return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
}
