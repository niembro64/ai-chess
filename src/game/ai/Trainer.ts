// Training orchestrator: manages concurrent self-play games + neural network training
// Uses batched MCTS for fast GPU utilization across all concurrent games

import type { ChessGameState, PieceColor, Board } from '@/types/chess';
import { createInitialGameState, applyMove, getLegalMoves, isInCheck, isSquareAttackedBy } from '@/game/chess/ChessEngine';
import * as tf from '@tensorflow/tfjs';
import { ChessNet, encodeBoard, moveToIndex, POLICY_SIZE, type SerializedWeights } from './ChessNet';
import { runBatchedMCTS } from './MCTS';

// Try to register WebGPU backend (faster than WebGL for training)
let webgpuImported = false;
async function initBestBackend(): Promise<string> {
  // Try WebGPU first (Chrome 113+, much faster for ML)
  if (!webgpuImported) {
    try {
      await import('@tensorflow/tfjs-backend-webgpu');
      webgpuImported = true;
    } catch {
      // WebGPU package not available
    }
  }

  if (webgpuImported) {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      return `webgpu (${tf.backend().constructor.name})`;
    } catch {
      // WebGPU not supported in this browser, fall through
    }
  }

  // Fall back to WebGL (default, uses GPU via graphics shaders)
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    return 'webgl';
  } catch {
    // Last resort
    await tf.setBackend('cpu');
    await tf.ready();
    return 'cpu (no GPU)';
  }
}

// --- Position evaluation ---
// Comprehensive positional scoring for reward shaping.
// Returns a value roughly in [-1, 1] from the perspective of `color`.
// Combines material + tactical + positional signals.

const PIECE_VALUES: Record<string, number> = {
  pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9, king: 0,
};

// Center squares (most valuable for control)
const CENTER_SQUARES = [
  { rank: 3, file: 3 }, { rank: 3, file: 4 }, // d5, e5
  { rank: 4, file: 3 }, { rank: 4, file: 4 }, // d4, e4
];

function oppositeColor(c: PieceColor): PieceColor {
  return c === 'white' ? 'black' : 'white';
}

function pieceValue(type: string, w: RewardWeights): number {
  switch (type) {
    case 'pawn': return w.pawnVal;
    case 'knight': return w.knightVal;
    case 'bishop': return w.bishopVal;
    case 'rook': return w.rookVal;
    case 'queen': return w.queenVal;
    default: return 0;
  }
}

function evaluatePosition(state: ChessGameState, color: PieceColor, w: RewardWeights, precomputedMoveCount?: number): number {
  const board = state.board;
  const opp = oppositeColor(color);
  const backRank = color === 'white' ? 7 : 0;
  const moveNumber = state.fullMoveNumber;

  let score = 0;

  let ownMaterial = 0, oppMaterial = 0;
  let maxMaterial = 0; // For normalization
  let ownBishops = 0, oppBishops = 0;
  let ownDeveloped = 0;
  let ownCenterPieces = 0, oppCenterPieces = 0;
  const ownPawnFiles: number[] = [];
  const oppPawnFiles: number[] = [];
  let ownKingPos = { rank: 0, file: 0 };

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[r][f];
      if (!piece) continue;
      const isOwn = piece.color === color;
      const val = pieceValue(piece.type, w);
      if (isOwn) {
        ownMaterial += val;
        if (piece.type === 'bishop') ownBishops++;
        if (piece.type === 'king') ownKingPos = { rank: r, file: f };
        if (piece.type !== 'pawn' && piece.type !== 'king' && r !== backRank) ownDeveloped++;
        if ((r === 3 || r === 4) && (f === 3 || f === 4)) ownCenterPieces++;
        if (piece.type === 'pawn') ownPawnFiles.push(f);
      } else {
        oppMaterial += val;
        if (piece.type === 'bishop') oppBishops++;
        if ((r === 3 || r === 4) && (f === 3 || f === 4)) oppCenterPieces++;
        if (piece.type === 'pawn') oppPawnFiles.push(f);
      }
    }
  }

  // Max possible material for normalization (Q + 2R + 2B + 2N + 8P with current weights)
  maxMaterial = w.queenVal + 2 * w.rookVal + 2 * w.bishopVal + 2 * w.knightVal + 8 * w.pawnVal;
  if (maxMaterial <= 0) maxMaterial = 1;

  // Material
  if (w.material) score += ((ownMaterial - oppMaterial) / maxMaterial) * w.material;

  // Check
  if (w.check && isInCheck(board, opp)) score += w.check;

  // Development (early game only)
  if (w.development && moveNumber <= 20) score += Math.min(ownDeveloped, 6) * w.development;

  // Center occupation
  if (w.centerOccupation) {
    score += (ownCenterPieces - oppCenterPieces) * w.centerOccupation;
  }

  // Center attacks
  if (w.centerAttack) {
    for (const sq of CENTER_SQUARES) {
      if (isSquareAttackedBy(board, sq, color)) score += w.centerAttack;
      if (isSquareAttackedBy(board, sq, opp)) score -= w.centerAttack;
    }
  }

  // Castling
  const kingOnCastledSquare =
    (color === 'white' && ownKingPos.rank === 7 && (ownKingPos.file === 6 || ownKingPos.file === 2)) ||
    (color === 'black' && ownKingPos.rank === 0 && (ownKingPos.file === 6 || ownKingPos.file === 2));
  if (w.castled && kingOnCastledSquare) score += w.castled;
  if (w.uncastled && !kingOnCastledSquare && moveNumber > 15) {
    const kingOnStart =
      (color === 'white' && ownKingPos.rank === 7 && ownKingPos.file === 4) ||
      (color === 'black' && ownKingPos.rank === 0 && ownKingPos.file === 4);
    if (kingOnStart) score -= w.uncastled;
  }

  // Mobility
  if (w.mobility && state.currentTurn === color && precomputedMoveCount !== undefined) {
    score += Math.min(precomputedMoveCount, 40) * w.mobility;
  }

  // Bishop pair
  if (w.bishopPair) {
    if (ownBishops >= 2) score += w.bishopPair;
    if (oppBishops >= 2) score -= w.bishopPair;
  }

  // Doubled pawns
  if (w.doubledPawns) {
    const fileCount = new Map<number, number>();
    for (const f of ownPawnFiles) fileCount.set(f, (fileCount.get(f) ?? 0) + 1);
    for (const count of fileCount.values()) {
      if (count > 1) score -= w.doubledPawns * (count - 1);
    }
  }

  // Passed pawns
  if (w.passedPawns) {
    for (const f of ownPawnFiles) {
      let passed = true;
      for (const of2 of oppPawnFiles) {
        if (Math.abs(of2 - f) <= 1) { passed = false; break; }
      }
      if (passed) score += w.passedPawns;
    }
  }

  // Hanging pieces
  if (w.hangingPieces) {
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const piece = board[r][f];
        if (!piece || piece.color !== color || piece.type === 'king') continue;
        const pos = { rank: r, file: f };
        if (isSquareAttackedBy(board, pos, opp) && !isSquareAttackedBy(board, pos, color)) {
          score -= pieceValue(piece.type, w) * w.hangingPieces;
        }
      }
    }
  }

  // Rooks on open files
  if (w.rookOpenFile) {
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const piece = board[r][f];
        if (!piece || piece.color !== color || piece.type !== 'rook') continue;
        const hasOwnPawn = ownPawnFiles.includes(f);
        const hasOppPawn = oppPawnFiles.includes(f);
        if (!hasOwnPawn && !hasOppPawn) score += w.rookOpenFile;
        else if (!hasOwnPawn) score += w.rookOpenFile * 0.5;
      }
    }
  }

  return score;
}

// Simple material-only evaluation (for UI display)
function materialAdvantage(state: ChessGameState, color: PieceColor): number {
  let own = 0, opp = 0;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = state.board[r][f];
      if (!piece) continue;
      const val = PIECE_VALUES[piece.type] ?? 0;
      if (piece.color === color) own += val; else opp += val;
    }
  }
  return (own - opp) / 39;
}

// --- Types ---

export type TrainingExample = {
  board: Float32Array;
  policy: Float32Array;
  value: number;
};

export type LogEntry = { time: number; message: string };

// Snapshot of a game slot for UI display
export type GameSlotSnapshot = {
  moveCount: number;
  moveCap: number;
  isStandardStart: boolean;
  materialWhite: number;
  materialBlack: number;
  board: Board;
  currentTurn: PieceColor;
  status: string;
};

// Snapshot of a completed game for UI display
export type CompletedGameSnapshot = {
  gameNumber: number;
  moveCount: number;
  board: Board;
  outcome: string;
  outcomeClass: 'white' | 'black' | 'draw' | 'cap';
};

export type TrainingStats = {
  generation: number;
  gamesCompleted: number;
  gamesPerMinute: number;
  replayBufferSize: number;
  policyLoss: number;
  valueLoss: number;
  totalLoss: number;
  avgGameLength: number;
  whiteWins: number;
  blackWins: number;
  draws: number;
  activeGames: number;
  paramCount: number;
  lossHistory: Array<{ gen: number; policy: number; value: number; total: number }>;
  isRunning: boolean;
  log: LogEntry[];
  gameSlots: GameSlotSnapshot[];
  completedGames: CompletedGameSnapshot[];
  gpuBackend: string;
  sampleWeights: number[];
  tensorCount: number;  // TF.js live tensor count -- should be stable, not growing
  memoryMB: number;     // TF.js GPU memory usage
};

export type RewardWeights = {
  winning: number;        // Game outcome: checkmate = +winning, loss = -winning
  material: number;       // Overall material scaling
  pawnVal: number;        // Pawn piece value
  knightVal: number;      // Knight piece value
  bishopVal: number;      // Bishop piece value
  rookVal: number;        // Rook piece value
  queenVal: number;       // Queen piece value
  check: number;          // Delivering check
  development: number;    // Pieces off back rank (early game)
  centerOccupation: number; // Pieces on center squares
  centerAttack: number;   // Attacking center squares
  castled: number;        // King has castled
  uncastled: number;      // Penalty for king stuck on start square
  mobility: number;       // Legal move count
  bishopPair: number;     // Having both bishops
  doubledPawns: number;   // Penalty per doubled pawn
  passedPawns: number;    // Pawns with no opposing blockers
  hangingPieces: number;  // Penalty for undefended attacked pieces
  rookOpenFile: number;   // Rooks on open/semi-open files
};

export const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  winning: 1.0,
  material: 0.3,
  pawnVal: 1,
  knightVal: 3,
  bishopVal: 3,
  rookVal: 5,
  queenVal: 9,
  check: 0.02,
  development: 0.01,
  centerOccupation: 0.01,
  centerAttack: 0.005,
  castled: 0.03,
  uncastled: 0.03,
  mobility: 0.001,
  bishopPair: 0.02,
  doubledPawns: 0.01,
  passedPawns: 0.02,
  hangingPieces: 0.005,
  rookOpenFile: 0.01,
};

export type TrainerConfig = {
  numConcurrentGames: number;
  mctsSimulations: number;
  learningRate: number;
  trainingBatchSize: number;
  replayBufferMax: number;
  gradientStepsPerTrain: number;
  numResBlocks: number;
  numFilters: number;
  kernelSize: number;
  valueHeadSize: number;
  seReduction: number;
  rewards: RewardWeights;
};

export const DEFAULT_CONFIG: TrainerConfig = {
  numConcurrentGames: 8,
  mctsSimulations: 25,
  learningRate: 0.0001,
  trainingBatchSize: 128,
  replayBufferMax: 8000,
  gradientStepsPerTrain: 4,
  numResBlocks: 6,
  numFilters: 64,
  kernelSize: 3,
  valueHeadSize: 64,
  seReduction: 8,
  rewards: { ...DEFAULT_REWARD_WEIGHTS },
};

// --- Self-play game slot ---

type GameSlotExample = {
  board: Float32Array;
  policy: Float32Array;
  turnColor: PieceColor;
  positionScore: number; // Full positional evaluation for this player
};

type GameSlot = {
  state: ChessGameState;
  examples: GameSlotExample[];
  moveCount: number;
  moveCap: number;
  isStandardStart: boolean;
};

// --- Starting position generators ---

function normalStartingPosition(): ChessGameState {
  const state = createInitialGameState();
  return { ...state, status: 'active' };
}

function randomStartingPosition(): ChessGameState {
  // Mix of opening (40%), midgame (30%), late game (30%)
  const roll = Math.random();
  let numRandom: number;
  if (roll < 0.4) {
    numRandom = Math.floor(Math.random() * 7); // 0-6 moves (opening)
  } else if (roll < 0.7) {
    numRandom = 12 + Math.floor(Math.random() * 13); // 12-24 moves (midgame)
  } else {
    numRandom = 30 + Math.floor(Math.random() * 21); // 30-50 moves (late game)
  }

  let state = normalStartingPosition();

  for (let i = 0; i < numRandom; i++) {
    const moves = getLegalMoves(state);
    if (moves.length === 0) break;
    state = applyMove(state, moves[Math.floor(Math.random() * moves.length)]);
    if (state.status === 'checkmate' || state.status === 'stalemate' || state.status === 'draw') {
      state = normalStartingPosition();
      break;
    }
  }

  return state;
}

// --- Trainer ---

export class Trainer {
  private net: ChessNet | null = null;
  private config: TrainerConfig;
  private running = false;

  private replayBuffer: TrainingExample[] = [];
  private replayBufferHead = 0; // Ring buffer write pointer
  private games: GameSlot[] = [];

  private generation = 0;
  private gamesCompleted = 0;
  private gameLengths: number[] = [];
  private whiteWins = 0;
  private blackWins = 0;
  private draws = 0;
  private policyLoss = 0;
  private valueLoss = 0;
  private gpuBackend = '';
  private totalLoss = 0;
  private lossHistory: Array<{ gen: number; policy: number; value: number; total: number }> = [];
  private startTime = 0;
  private log: LogEntry[] = [];
  private completedGames: CompletedGameSnapshot[] = [];
  private lastSaveNotifyTime = 0;
  private statsCallCount = 0;
  private cachedSampleWeights: number[] = [];
  private cachedTensorCount = 0;
  private cachedMemoryMB = 0;

  public onStatsUpdate?: (stats: TrainingStats) => void;
  public onSaved?: () => void;

  constructor(config: Partial<TrainerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  private addLog(message: string): void {
    const time = this.startTime > 0 ? Date.now() - this.startTime : 0;
    this.log.push({ time, message });
    if (this.log.length > 200) this.log.shift();
  }

  private getCachedSampleWeights(): number[] {
    this.statsCallCount++;
    if (this.statsCallCount % 30 === 0) {
      if (this.net) this.cachedSampleWeights = this.net.getSampleWeights();
      const mem = tf.memory();
      this.cachedTensorCount = mem.numTensors;
      this.cachedMemoryMB = (mem.numBytes ?? 0) / (1024 * 1024);
    }
    return this.cachedSampleWeights;
  }

  private getGameSlotSnapshots(): GameSlotSnapshot[] {
    return this.games.map(g => ({
      moveCount: g.moveCount,
      moveCap: g.moveCap,
      isStandardStart: g.isStandardStart,
      materialWhite: materialAdvantage(g.state, 'white'),
      materialBlack: materialAdvantage(g.state, 'black'),
      board: g.state.board,
      currentTurn: g.state.currentTurn,
      status: g.state.status,
    }));
  }

  getStats(): TrainingStats {
    const elapsed = (Date.now() - this.startTime) / 60000;
    const avgLen = this.gameLengths.length > 0
      ? this.gameLengths.reduce((a, b) => a + b, 0) / this.gameLengths.length
      : 0;
    return {
      generation: this.generation,
      gamesCompleted: this.gamesCompleted,
      gamesPerMinute: elapsed > 0 ? this.gamesCompleted / elapsed : 0,
      replayBufferSize: this.replayBufferSize,
      policyLoss: this.policyLoss,
      valueLoss: this.valueLoss,
      totalLoss: this.totalLoss,
      avgGameLength: avgLen,
      whiteWins: this.whiteWins,
      blackWins: this.blackWins,
      draws: this.draws,
      activeGames: this.games.length,
      paramCount: this.net?.getParamCount() ?? 0,
      lossHistory: this.lossHistory,
      isRunning: this.running,
      log: this.log,
      gameSlots: this.getGameSlotSnapshots(),
      completedGames: this.completedGames,
      gpuBackend: this.gpuBackend,
      sampleWeights: this.getCachedSampleWeights(),
      tensorCount: this.cachedTensorCount,
      memoryMB: this.cachedMemoryMB,
    };
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.startTime = Date.now();

    // Initialize best available GPU backend
    this.gpuBackend = await initBestBackend();
    this.addLog(`GPU backend: ${this.gpuBackend}`);

    if (!this.net) {
      // Try to load a previously saved model directly (avoids create-then-dispose)
      this.net = await ChessNet.loadFromSaved(this.config.learningRate);
      if (this.net) {
        this.addLog(`Loaded saved model (${this.net.getParamCount().toLocaleString()} params)`);
      } else {
        this.net = ChessNet.create({
          numResBlocks: this.config.numResBlocks,
          numFilters: this.config.numFilters,
          kernelSize: this.config.kernelSize,
          valueHeadSize: this.config.valueHeadSize,
          seReduction: this.config.seReduction,
          learningRate: this.config.learningRate,
        });
        this.addLog(`New network: ${this.config.numResBlocks} res blocks, ${this.config.numFilters} filters, ${this.net.getParamCount().toLocaleString()} params`);
      }
    }

    this.games = [];
    for (let i = 0; i < this.config.numConcurrentGames; i++) {
      // 90% random starts, 10% standard opening
      this.games.push(this.createGameSlot(Math.random() > 0.1));
    }
    this.addLog(`${this.config.numConcurrentGames} games, ${this.config.mctsSimulations} MCTS sims, batched predict`);
    this.emitStats();
    await this.trainLoop();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.net) {
      await this.net.save();
      this.addLog('Saved and stopped');
    }
    this.emitStats();
  }

  async saveModel(): Promise<void> {
    if (this.net) {
      await this.net.save();
      this.addLog('Model saved');
    }
  }

  static async hasSavedModel(): Promise<boolean> {
    return ChessNet.hasSavedModel();
  }

  static async deleteSavedModel(): Promise<void> {
    await ChessNet.deleteSavedModel();
  }

  exportWeights(): SerializedWeights | null {
    return this.net?.exportWeights() ?? null;
  }

  async loadModel(): Promise<boolean> {
    // Dispose existing net if any, then load fresh from IndexedDB
    if (this.net) {
      this.net.dispose();
      this.net = null;
    }
    this.net = await ChessNet.loadFromSaved(this.config.learningRate);
    if (this.net) {
      this.addLog(`Model loaded (${this.net.getParamCount().toLocaleString()} params)`);
      return true;
    }
    return false;
  }

  updateConfig(config: Partial<TrainerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  private addToBuffer(example: TrainingExample): void {
    const max = this.config.replayBufferMax;
    if (this.replayBuffer.length < max) {
      this.replayBuffer.push(example);
    } else {
      // Ring buffer: overwrite oldest entry in O(1)
      this.replayBuffer[this.replayBufferHead] = example;
      this.replayBufferHead = (this.replayBufferHead + 1) % max;
    }
  }

  private get replayBufferSize(): number {
    return this.replayBuffer.length;
  }

  private createGameSlot(useRandom?: boolean): GameSlot {
    // 90% random starts, 10% standard opening
    const random = useRandom ?? (Math.random() > 0.1);
    const state = random ? randomStartingPosition() : normalStartingPosition();
    const baseCap = 5 + Math.floor(Math.random() * 26); // 5-30
    const moveCap = random ? baseCap : baseCap * 10; // Standard starts get 50-300
    return { state, examples: [], moveCount: 0, moveCap, isStandardStart: !random };
  }

  // Main loop: advance ALL games by one move using batched MCTS, then check for completions
  private async trainLoop(): Promise<void> {
    while (this.running) {
      // Collect current states from all active games
      const states = this.games.map(g => g.state);

      // Batched MCTS: one GPU call per simulation step across ALL games
      const results = runBatchedMCTS(states, this.net!, this.config.mctsSimulations);

      // Apply results to each game
      for (let i = 0; i < this.games.length; i++) {
        const slot = this.games[i];
        const { policy, move } = results[i];

        // Record training example
        const board = encodeBoard(slot.state);
        const isWhite = slot.state.currentTurn === 'white';
        const canonPolicy = new Float32Array(POLICY_SIZE);
        const legalMoves = getLegalMoves(slot.state);
        const seenIndices = new Set<number>();
        for (const m of legalMoves) {
          const mi = moveToIndex(m, isWhite);
          if (!seenIndices.has(mi)) {
            seenIndices.add(mi);
            canonPolicy[mi] = policy[mi];
          }
        }

        const posScore = evaluatePosition(slot.state, slot.state.currentTurn, this.config.rewards, legalMoves.length);

        slot.examples.push({
          board,
          policy: canonPolicy,
          turnColor: slot.state.currentTurn,
          positionScore: posScore,
        });

        // Apply move
        slot.state = applyMove(slot.state, move);
        slot.moveCount++;

        // Check game end (per-game random cap prevents shuffling games from hogging slots)
        const isOver =
          slot.state.status === 'checkmate' ||
          slot.state.status === 'stalemate' ||
          slot.state.status === 'draw' ||
          slot.moveCount >= slot.moveCap;

        if (isOver) {
          this.finishGame(slot);
          this.games[i] = this.createGameSlot();
        }
      }

      // Train every round as long as buffer has enough data
      if (this.replayBuffer.length >= this.config.trainingBatchSize) {
        await this.runTrainStep();
        // Sync weights to CPU cache only for small models that use CPU inference
        if (this.config.numFilters <= 32) {
          this.net!.syncCPU();
        }
      }

      // Autosave every 30 seconds
      const now = Date.now();
      if (this.net && now - this.lastSaveNotifyTime >= 30_000) {
        this.lastSaveNotifyTime = now;
        await this.net.save();
        this.onSaved?.();
      }

      // Yield to UI every step so boards update each move
      this.emitStats();
      await new Promise(r => setTimeout(r, 0));
    }
  }

  private finishGame(slot: GameSlot): void {
    const hitCap = slot.moveCount >= slot.moveCap &&
      slot.state.status !== 'checkmate' &&
      slot.state.status !== 'stalemate' &&
      slot.state.status !== 'draw';

    let whiteOutcome: number;
    let outcomeLabel: string;

    if (slot.state.status === 'checkmate' && slot.state.winner) {
      whiteOutcome = slot.state.winner === 'white' ? 1 : -1;
      if (slot.state.winner === 'white') this.whiteWins++; else this.blackWins++;
      outcomeLabel = `${slot.state.winner} wins`;
    } else if (hitCap) {
      // Capped: use material advantage so we don't mislabel a winning position as a draw
      const matAdv = materialAdvantage(slot.state, 'white');
      whiteOutcome = Math.max(-1, Math.min(1, matAdv * 3));
      this.draws++;
      outcomeLabel = `cap (${(matAdv * 39) >= 0 ? '+' : ''}${(matAdv * 39).toFixed(0)})`;
    } else {
      whiteOutcome = 0;
      this.draws++;
      outcomeLabel = slot.state.status === 'stalemate' ? 'stalemate' : 'draw';
    }

    const winWeight = this.config.rewards.winning;

    for (const ex of slot.examples) {
      const outcomeFromPerspective = ex.turnColor === 'white' ? whiteOutcome : -whiteOutcome;
      // Value = weighted game outcome + positional score (already weighted by individual reward weights)
      const value = outcomeFromPerspective * winWeight + ex.positionScore;
      this.addToBuffer({
        board: ex.board,
        policy: ex.policy,
        value: Math.max(-1, Math.min(1, value)),
      });
    }

    this.gamesCompleted++;
    this.gameLengths.push(slot.moveCount);
    if (this.gameLengths.length > 100) this.gameLengths.shift();

    // Track completed game for UI display
    const outcomeClass: 'white' | 'black' | 'draw' | 'cap' =
      (slot.state.status === 'checkmate' && slot.state.winner === 'white') ? 'white' :
      (slot.state.status === 'checkmate' && slot.state.winner === 'black') ? 'black' :
      hitCap ? 'cap' : 'draw';
    this.completedGames.push({
      gameNumber: this.gamesCompleted,
      moveCount: slot.moveCount,
      board: slot.state.board.map(r => [...r]),
      outcome: outcomeLabel,
      outcomeClass,
    });
    // Keep last 16 completed games
    if (this.completedGames.length > 16) this.completedGames.shift();

    this.addLog(`Game #${this.gamesCompleted}: ${outcomeLabel} in ${slot.moveCount} moves`);
  }

  private async runTrainStep(): Promise<void> {
    const bufSize = this.replayBuffer.length;
    if (bufSize < this.config.trainingBatchSize) return;

    const steps = this.config.gradientStepsPerTrain;
    let lastResult = { policyLoss: 0, valueLoss: 0, totalLoss: 0 };

    for (let step = 0; step < steps; step++) {
      // Sample a fresh random batch for each gradient step
      const batchSize = Math.min(this.config.trainingBatchSize, bufSize);
      const boards: Float32Array[] = [];
      const policies: Float32Array[] = [];
      const values: number[] = [];

      for (let i = 0; i < batchSize; i++) {
        const idx = Math.floor(Math.random() * bufSize);
        boards.push(this.replayBuffer[idx].board);
        policies.push(this.replayBuffer[idx].policy);
        values.push(this.replayBuffer[idx].value);
      }

      lastResult = await this.net!.train(boards, policies, values);
    }

    this.policyLoss = lastResult.policyLoss;
    this.valueLoss = lastResult.valueLoss;
    this.totalLoss = lastResult.totalLoss;

    this.generation++;
    this.lossHistory.push({
      gen: this.generation,
      policy: lastResult.policyLoss,
      value: lastResult.valueLoss,
      total: lastResult.totalLoss,
    });
    if (this.lossHistory.length > 500) this.lossHistory.shift();

    this.addLog(`Gen ${this.generation} (${steps}x): P=${lastResult.policyLoss.toFixed(3)} V=${lastResult.valueLoss.toFixed(3)}`);
  }

  private emitStats(): void {
    this.onStatsUpdate?.(this.getStats());
  }
}
