// Training orchestrator: manages concurrent self-play games + neural network training
// Uses batched MCTS for fast GPU utilization across all concurrent games

import type { ChessGameState, PieceColor, Board } from '@/types/chess';
import { createInitialGameState, applyMove, getLegalMoves } from '@/game/chess/ChessEngine';
import * as tf from '@tensorflow/tfjs';
import { ChessNet, encodeBoard, moveToIndex, POLICY_SIZE, type SerializedWeights } from './ChessNet';
import { runBatchedMCTS } from './MCTS';
import {
  evaluatePosition,
  materialAdvantage,
  DEFAULT_REWARD_WEIGHTS,
  type RewardWeights,
} from './rewardShaping';
export { DEFAULT_REWARD_WEIGHTS, type RewardWeights } from './rewardShaping';

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
