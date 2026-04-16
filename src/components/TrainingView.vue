<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { Trainer, DEFAULT_CONFIG, type TrainingStats, type TrainerConfig, type GameSlotSnapshot, type CompletedGameSnapshot } from '@/game/ai/Trainer';
import { getPieceSymbol } from '@/game/chess/ChessEngine';

const isMobile = window.innerWidth <= 600;

const trainer = new Trainer();
const stats = ref<TrainingStats | null>(null);
const config = ref<TrainerConfig>({
  ...DEFAULT_CONFIG,
  rewards: { ...DEFAULT_CONFIG.rewards },
  numConcurrentGames: isMobile ? 4 : DEFAULT_CONFIG.numConcurrentGames,
  mctsSimulations: isMobile ? 20 : DEFAULT_CONFIG.mctsSimulations,
});
const configLocked = ref(false);
const hasSavedModel = ref(false);

// Check for existing saved model on mount
onMounted(async () => {
  hasSavedModel.value = await Trainer.hasSavedModel();
});

trainer.onSaved = () => {
  saveMessage.value = 'Saved';
  setTimeout(() => { saveMessage.value = ''; }, 2000);
};

trainer.onStatsUpdate = (s) => {
  stats.value = {
    ...s,
    lossHistory: [...s.lossHistory],
    log: [...s.log],
    gameSlots: s.gameSlots.map(g => ({ ...g, board: g.board.map(r => [...r]) })),
    completedGames: s.completedGames.map(g => ({ ...g, board: g.board.map(r => [...r]) })),
  };
};

const startButtonLabel = computed(() => {
  if (stats.value?.isRunning) return 'Stop';
  if (hasSavedModel.value) return 'Continue Training';
  return 'Start Training';
});

async function handleStart() {
  if (stats.value?.isRunning) {
    await trainer.stop();
    hasSavedModel.value = true; // We just saved on stop
  } else {
    configLocked.value = true;
    trainer.updateConfig(config.value);
    await trainer.start();
    hasSavedModel.value = true; // Will autosave during training
  }
}

const saveMessage = ref('');

async function handleCopyWeights() {
  const weights = trainer.exportWeights();
  if (!weights) {
    saveMessage.value = 'No model to copy';
    setTimeout(() => { saveMessage.value = ''; }, 2000);
    return;
  }
  // Build JSON manually with sci notation for compact, readable numbers
  // Format: +1.234e+00 style (explicit signs on mantissa and exponent)
  const dataStr = weights.data.map(arr => {
    const nums = arr.map(raw => {
      const v = Number(raw);
      if (v === 0) return '0';
      return v.toExponential(2).replace('e+', 'e'); // "1.23e0", "-5.68e-2"
    });
    return `[${nums.join(',')}]`;
  }).join(',');

  const json = `{"shapes":${JSON.stringify(weights.shapes)},"data":[${dataStr}]}`;
  await navigator.clipboard.writeText(json);
  saveMessage.value = `Copied (${(json.length / 1024).toFixed(0)} KB)`;
  setTimeout(() => { saveMessage.value = ''; }, 3000);
}

async function handleReset() {
  if (!confirm('Delete saved model and start fresh?')) return;
  await Trainer.deleteSavedModel();
  hasSavedModel.value = false;
  saveMessage.value = 'Model deleted';
  setTimeout(() => { saveMessage.value = ''; }, 2000);
}

const emit = defineEmits<{
  (e: 'back'): void;
}>();

onUnmounted(() => {
  trainer.stop();
});

// Chart
// Chart dimensions (content area, excluding margins)
const chartMargin = { top: 8, right: 12, bottom: 20, left: 45 };
const chartTotalWidth = 480;
const chartTotalHeight = 130;
const chartW = chartTotalWidth - chartMargin.left - chartMargin.right;
const chartH = chartTotalHeight - chartMargin.top - chartMargin.bottom;

type ChartData = {
  points: string;
  yMin: number;
  yMax: number;
  yTicks: number[];
  xTicks: { x: number; label: string }[];
};

function buildChart(key: 'policy' | 'value' | 'total'): ChartData | null {
  const history = stats.value?.lossHistory ?? [];
  if (history.length < 2) return null;

  const vals = history.map(h => h[key]);
  let rawMin = Math.min(...vals);
  let rawMax = Math.max(...vals);
  // Add 5% padding
  const pad = (rawMax - rawMin) * 0.05 || 0.001;
  const yMin = Math.max(0, rawMin - pad);
  const yMax = rawMax + pad;
  const yRange = yMax - yMin || 1;

  const points = history.map((h, i) => {
    const x = chartMargin.left + (i / (history.length - 1)) * chartW;
    const y = chartMargin.top + chartH - ((h[key] - yMin) / yRange) * chartH;
    return `${x},${y}`;
  }).join(' ');

  // Generate ~4 Y-axis ticks
  const yTicks: number[] = [];
  const step = yRange / 4;
  for (let i = 0; i <= 4; i++) {
    yTicks.push(yMin + step * i);
  }

  // X-axis ticks (generation numbers)
  const xTicks: { x: number; label: string }[] = [];
  const totalGens = history.length;
  const firstGen = history[0].gen;
  const lastGen = history[totalGens - 1].gen;
  const genStep = Math.max(1, Math.floor((lastGen - firstGen) / 5));
  for (let g = firstGen; g <= lastGen; g += genStep) {
    const idx = history.findIndex(h => h.gen >= g);
    if (idx >= 0) {
      const x = chartMargin.left + (idx / (totalGens - 1)) * chartW;
      xTicks.push({ x, label: String(g) });
    }
  }

  return { points, yMin, yMax, yTicks, xTicks };
}

const policyChart = computed(() => buildChart('policy'));
const valueChart = computed(() => buildChart('value'));
const totalChart = computed(() => buildChart('total'));

function fmtTick(n: number): string {
  if (n >= 10) return n.toFixed(0);
  if (n >= 1) return n.toFixed(1);
  if (n >= 0.01) return n.toFixed(2);
  return n.toFixed(3);
}

function fmt(n: number, decimals = 2): string { return n.toFixed(decimals); }

// Log slider: maps slider position 0-100 to a logarithmic value range.
// Position 0 = off (value 0). Positions 1-100 map log-spaced from min to max.
function logSliderToValue(pos: number, min: number, max: number): number {
  if (pos <= 0) return 0;
  return min * Math.pow(max / min, (pos - 1) / 99);
}

function valueToLogSlider(value: number, min: number, max: number): number {
  if (value <= 0) return 0;
  const clamped = Math.max(min, Math.min(max, value));
  return 1 + 99 * Math.log(clamped / min) / Math.log(max / min);
}

function fmtReward(n: number): string {
  if (n === 0) return 'off';
  if (n >= 1) return n.toFixed(1);
  if (n >= 0.01) return n.toFixed(3);
  return n.toFixed(4);
}

// Reward weight slider definitions: [key, label, min, max]
type RwKey = keyof typeof DEFAULT_CONFIG.rewards;
const REWARD_SLIDERS: Array<[RwKey, string, number, number]> = [
  ['winning', 'Winning', 0.01, 5.0],
  ['material', 'Material', 0.01, 5.0],
  ['pawnVal', 'Pawn Value', 0.1, 10],
  ['knightVal', 'Knight Value', 0.1, 15],
  ['bishopVal', 'Bishop Value', 0.1, 15],
  ['rookVal', 'Rook Value', 0.1, 20],
  ['queenVal', 'Queen Value', 0.1, 30],
  ['check', 'Check', 0.001, 0.5],
  ['development', 'Development', 0.001, 0.2],
  ['centerOccupation', 'Center Occ.', 0.001, 0.2],
  ['centerAttack', 'Center Atk.', 0.001, 0.1],
  ['castled', 'Castled', 0.001, 0.5],
  ['uncastled', 'Uncastled', 0.001, 0.5],
  ['mobility', 'Mobility', 0.0001, 0.02],
  ['bishopPair', 'Bishop Pair', 0.001, 0.3],
  ['doubledPawns', 'Doubled Pawn', 0.001, 0.2],
  ['passedPawns', 'Passed Pawn', 0.001, 0.3],
  ['hangingPieces', 'Hanging', 0.001, 0.1],
  ['rookOpenFile', 'Rook Open', 0.001, 0.1],
];

function getRewardSliderPos(key: RwKey, min: number, max: number): number {
  return valueToLogSlider(config.value.rewards[key], min, max);
}

function setRewardSliderPos(key: RwKey, pos: number, min: number, max: number): void {
  config.value.rewards[key] = logSliderToValue(pos, min, max);
}
function fmtTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  return `${Math.floor(sec / 60)}:${(sec % 60).toString().padStart(2, '0')}`;
}

const totalGames = computed(() => {
  const s = stats.value;
  return s ? s.whiteWins + s.blackWins + s.draws : 0;
});

const winPct = computed(() => {
  const t = totalGames.value;
  if (t === 0) return { white: '0', black: '0', draw: '0' };
  const s = stats.value!;
  return {
    white: fmt(s.whiteWins / t * 100, 1),
    black: fmt(s.blackWins / t * 100, 1),
    draw: fmt(s.draws / t * 100, 1),
  };
});


const visibleLog = computed(() => (stats.value?.log ?? []).slice(-40).reverse());

// Mini board helper
function getPieceChar(slot: GameSlotSnapshot, rank: number, file: number): string {
  const piece = slot.board[rank]?.[file];
  if (!piece) return '';
  return getPieceSymbol(piece);
}

function miniSquareClass(rank: number, file: number): string {
  return (rank + file) % 2 === 0 ? 'mini-light' : 'mini-dark';
}

function getCompletedPieceChar(game: CompletedGameSnapshot, rank: number, file: number): string {
  const piece = game.board[rank]?.[file];
  if (!piece) return '';
  return getPieceSymbol(piece);
}

const reversedCompletedGames = computed(() => {
  return [...(stats.value?.completedGames ?? [])].reverse();
});
</script>

<template>
  <div class="training-wrapper">
    <!-- Header -->
    <div class="header">
      <button class="back-btn" @click="emit('back')">&larr; Back</button>
      <h1 class="title">AI Training</h1>
      <div class="header-controls">
        <button class="action-btn start-btn" :class="{ stop: stats?.isRunning, resume: !stats?.isRunning && hasSavedModel }" @click="handleStart">
          {{ startButtonLabel }}
        </button>
        <button class="action-btn copy-btn" @click="handleCopyWeights" :disabled="!stats || stats.isRunning">Copy Weights</button>
        <button class="action-btn reset-btn" @click="handleReset" :disabled="stats?.isRunning">Reset</button>
        <span class="save-msg" :class="{ visible: saveMessage }">{{ saveMessage || '&nbsp;' }}</span>
      </div>
    </div>

    <div class="content">
      <!-- Left: Config -->
      <div class="left-column">
        <div class="panel config-panel">
          <h2 class="panel-title">Training</h2>
          <div class="config-grid">
            <label for="cfg-games">Parallel Games</label>
            <input id="cfg-games" type="number" v-model.number="config.numConcurrentGames" min="1" max="64" />
            <label for="cfg-mcts">MCTS Sims</label>
            <input id="cfg-mcts" type="number" v-model.number="config.mctsSimulations" min="5" max="800" />
            <label for="cfg-lr">Learn Rate</label>
            <input id="cfg-lr" type="number" v-model.number="config.learningRate" min="0.0001" max="0.1" step="0.001" />
            <label for="cfg-batch">Batch Size</label>
            <input id="cfg-batch" type="number" v-model.number="config.trainingBatchSize" min="16" max="256" />
            <label for="cfg-buffer">Buffer Max</label>
            <input id="cfg-buffer" type="number" v-model.number="config.replayBufferMax" min="500" max="20000" step="500" />
            <label for="cfg-gradsteps">Grad Steps</label>
            <input id="cfg-gradsteps" type="number" v-model.number="config.gradientStepsPerTrain" min="1" max="32" />
          </div>
        </div>

        <div class="panel config-panel">
          <h2 class="panel-title">Network</h2>
          <template v-if="hasSavedModel && !configLocked">
            <p class="config-note">Saved model will be loaded. Reset to change architecture.</p>
          </template>
          <template v-else-if="stats">
            <div class="config-grid">
              <label>Params</label>
              <span class="config-readonly">{{ stats.paramCount.toLocaleString() }}</span>
            </div>
          </template>
          <template v-else>
            <div class="config-grid">
              <label for="cfg-blocks">Layers (Depth)</label>
              <input id="cfg-blocks" type="number" v-model.number="config.numResBlocks" min="1" max="10" />
              <label for="cfg-filters">Features (Width)</label>
              <input id="cfg-filters" type="number" v-model.number="config.numFilters" min="8" max="128" />
              <label for="cfg-kernel">Kernel Size</label>
              <input id="cfg-kernel" type="number" v-model.number="config.kernelSize" min="3" max="5" step="2" />
              <label for="cfg-vhead">Value Head</label>
              <input id="cfg-vhead" type="number" v-model.number="config.valueHeadSize" min="8" max="128" />
            </div>
          </template>
        </div>

        <!-- Reward weights with log sliders -->
        <div class="panel config-panel">
          <h2 class="panel-title">Rewards</h2>

          <div
            v-for="[key, label, min, max] in REWARD_SLIDERS"
            :key="key"
            class="reward-slider-row"
          >
            <span class="rw-label">{{ label }}</span>
            <input
              type="range"
              class="rw-slider"
              :value="getRewardSliderPos(key, min, max)"
              @input="setRewardSliderPos(key, Number(($event.target as HTMLInputElement).value), min, max)"
              min="0" max="100" step="1"
            />
            <span class="rw-value" :class="{ off: config.rewards[key] === 0 }">{{ fmtReward(config.rewards[key]) }}</span>
          </div>
        </div>
      </div>

      <!-- Center: Stats + Active Games -->
      <div class="center-column">
        <!-- Stats -->
        <div class="panel stats-panel">
          <h2 class="panel-title">Stats</h2>
          <template v-if="stats">
            <div class="stats-grid">
              <div class="stat-card"><span class="stat-label">Gen</span><span class="stat-value big">{{ stats.generation }}</span></div>
              <div class="stat-card"><span class="stat-label">Games</span><span class="stat-value big">{{ stats.gamesCompleted }}</span></div>
              <div class="stat-card"><span class="stat-label">Games/min</span><span class="stat-value">{{ fmt(stats.gamesPerMinute, 1) }}</span></div>
              <div class="stat-card"><span class="stat-label">Buffer</span><span class="stat-value">{{ stats.replayBufferSize }}</span></div>
              <div class="stat-card"><span class="stat-label">Avg Len</span><span class="stat-value">{{ fmt(stats.avgGameLength, 0) }}</span></div>
              <div class="stat-card"><span class="stat-label">Params</span><span class="stat-value">{{ stats.paramCount.toLocaleString() }}</span></div>
            <div class="stat-card" v-if="stats.gpuBackend"><span class="stat-label">GPU</span><span class="stat-value gpu-val">{{ stats.gpuBackend }}</span></div>
            <div class="stat-card"><span class="stat-label">Tensors</span><span class="stat-value" :class="{ 'tensor-warn': stats.tensorCount > 200 }">{{ stats.tensorCount }}</span></div>
            <div class="stat-card"><span class="stat-label">GPU MB</span><span class="stat-value">{{ fmt(stats.memoryMB, 1) }}</span></div>
            </div>

            <!-- Sample weights -->
            <div v-if="stats.sampleWeights.length > 0" class="weights-row">
              <span class="weights-label">Weights:</span>
              <span v-for="(w, i) in stats.sampleWeights" :key="i"
                class="weight-val"
                :class="{ positive: w > 0, negative: w < 0 }"
              >{{ w >= 0 ? '+' : '' }}{{ w.toFixed(4) }}</span>
            </div>

            <!-- Loss -->
            <div class="loss-row">
              <div class="loss-card">
                <span class="loss-label">Move Prediction</span>
                <span class="loss-sublabel">how wrong the move choices are</span>
                <span class="loss-value" style="color:#4a9eff">{{ fmt(stats.policyLoss, 4) }}</span>
              </div>
              <div class="loss-card">
                <span class="loss-label">Board Evaluation</span>
                <span class="loss-sublabel">how wrong the position scoring is</span>
                <span class="loss-value" style="color:#44aa44">{{ fmt(stats.valueLoss, 4) }}</span>
              </div>
              <div class="loss-card">
                <span class="loss-label">Combined</span>
                <span class="loss-sublabel">total error being minimized</span>
                <span class="loss-value" style="color:#ffaa00">{{ fmt(stats.totalLoss, 4) }}</span>
              </div>
            </div>

            <!-- Charts: separate Y-axes so both lines are visible -->
            <div v-if="stats.lossHistory.length >= 2" class="charts-row">
              <!-- Move Prediction chart -->
              <div v-if="policyChart" class="chart-box">
                <div class="chart-title" style="color:#4a9eff">Move Prediction</div>
                <svg class="loss-chart" :viewBox="`0 0 ${chartTotalWidth} ${chartTotalHeight}`">
                  <!-- Grid lines -->
                  <line v-for="(tick, i) in policyChart.yTicks" :key="'pg'+i"
                    :x1="chartMargin.left" :x2="chartMargin.left + chartW"
                    :y1="chartMargin.top + chartH - ((tick - policyChart.yMin) / (policyChart.yMax - policyChart.yMin)) * chartH"
                    :y2="chartMargin.top + chartH - ((tick - policyChart.yMin) / (policyChart.yMax - policyChart.yMin)) * chartH"
                    stroke="#2a2a3e" stroke-width="0.5" />
                  <!-- Y-axis labels -->
                  <text v-for="(tick, i) in policyChart.yTicks" :key="'py'+i"
                    :x="chartMargin.left - 4"
                    :y="chartMargin.top + chartH - ((tick - policyChart.yMin) / (policyChart.yMax - policyChart.yMin)) * chartH + 3"
                    fill="#666" font-size="9" font-family="monospace" text-anchor="end">{{ fmtTick(tick) }}</text>
                  <!-- X-axis labels -->
                  <text v-for="(tick, i) in policyChart.xTicks" :key="'px'+i"
                    :x="tick.x" :y="chartTotalHeight - 3"
                    fill="#555" font-size="8" font-family="monospace" text-anchor="middle">{{ tick.label }}</text>
                  <!-- Axes -->
                  <line :x1="chartMargin.left" :x2="chartMargin.left" :y1="chartMargin.top" :y2="chartMargin.top + chartH" stroke="#444" stroke-width="1" />
                  <line :x1="chartMargin.left" :x2="chartMargin.left + chartW" :y1="chartMargin.top + chartH" :y2="chartMargin.top + chartH" stroke="#444" stroke-width="1" />
                  <!-- Data line -->
                  <polyline :points="policyChart.points" fill="none" stroke="#4a9eff" stroke-width="1.5" />
                </svg>
              </div>

              <!-- Board Evaluation chart -->
              <div v-if="valueChart" class="chart-box">
                <div class="chart-title" style="color:#44aa44">Board Evaluation</div>
                <svg class="loss-chart" :viewBox="`0 0 ${chartTotalWidth} ${chartTotalHeight}`">
                  <!-- Grid lines -->
                  <line v-for="(tick, i) in valueChart.yTicks" :key="'vg'+i"
                    :x1="chartMargin.left" :x2="chartMargin.left + chartW"
                    :y1="chartMargin.top + chartH - ((tick - valueChart.yMin) / (valueChart.yMax - valueChart.yMin)) * chartH"
                    :y2="chartMargin.top + chartH - ((tick - valueChart.yMin) / (valueChart.yMax - valueChart.yMin)) * chartH"
                    stroke="#2a2a3e" stroke-width="0.5" />
                  <!-- Y-axis labels -->
                  <text v-for="(tick, i) in valueChart.yTicks" :key="'vy'+i"
                    :x="chartMargin.left - 4"
                    :y="chartMargin.top + chartH - ((tick - valueChart.yMin) / (valueChart.yMax - valueChart.yMin)) * chartH + 3"
                    fill="#666" font-size="9" font-family="monospace" text-anchor="end">{{ fmtTick(tick) }}</text>
                  <!-- X-axis labels -->
                  <text v-for="(tick, i) in valueChart.xTicks" :key="'vx'+i"
                    :x="tick.x" :y="chartTotalHeight - 3"
                    fill="#555" font-size="8" font-family="monospace" text-anchor="middle">{{ tick.label }}</text>
                  <!-- Axes -->
                  <line :x1="chartMargin.left" :x2="chartMargin.left" :y1="chartMargin.top" :y2="chartMargin.top + chartH" stroke="#444" stroke-width="1" />
                  <line :x1="chartMargin.left" :x2="chartMargin.left + chartW" :y1="chartMargin.top + chartH" :y2="chartMargin.top + chartH" stroke="#444" stroke-width="1" />
                  <!-- Data line -->
                  <polyline :points="valueChart.points" fill="none" stroke="#44aa44" stroke-width="1.5" />
                </svg>
              </div>

              <!-- Combined chart -->
              <div v-if="totalChart" class="chart-box">
                <div class="chart-title" style="color:#ffaa00">Combined</div>
                <svg class="loss-chart" :viewBox="`0 0 ${chartTotalWidth} ${chartTotalHeight}`">
                  <line v-for="(tick, i) in totalChart.yTicks" :key="'tg'+i"
                    :x1="chartMargin.left" :x2="chartMargin.left + chartW"
                    :y1="chartMargin.top + chartH - ((tick - totalChart.yMin) / (totalChart.yMax - totalChart.yMin)) * chartH"
                    :y2="chartMargin.top + chartH - ((tick - totalChart.yMin) / (totalChart.yMax - totalChart.yMin)) * chartH"
                    stroke="#2a2a3e" stroke-width="0.5" />
                  <text v-for="(tick, i) in totalChart.yTicks" :key="'ty'+i"
                    :x="chartMargin.left - 4"
                    :y="chartMargin.top + chartH - ((tick - totalChart.yMin) / (totalChart.yMax - totalChart.yMin)) * chartH + 3"
                    fill="#666" font-size="9" font-family="monospace" text-anchor="end">{{ fmtTick(tick) }}</text>
                  <text v-for="(tick, i) in totalChart.xTicks" :key="'tx'+i"
                    :x="tick.x" :y="chartTotalHeight - 3"
                    fill="#555" font-size="8" font-family="monospace" text-anchor="middle">{{ tick.label }}</text>
                  <line :x1="chartMargin.left" :x2="chartMargin.left" :y1="chartMargin.top" :y2="chartMargin.top + chartH" stroke="#444" stroke-width="1" />
                  <line :x1="chartMargin.left" :x2="chartMargin.left + chartW" :y1="chartMargin.top + chartH" :y2="chartMargin.top + chartH" stroke="#444" stroke-width="1" />
                  <polyline :points="totalChart.points" fill="none" stroke="#ffaa00" stroke-width="1.5" />
                </svg>
              </div>
            </div>

            <!-- Outcomes -->
            <div class="outcome-bar-container">
              <div class="outcome-bar">
                <div class="outcome-segment white-seg" :style="{ width: winPct.white + '%' }"></div>
                <div class="outcome-segment draw-seg" :style="{ width: winPct.draw + '%' }"></div>
                <div class="outcome-segment black-seg" :style="{ width: winPct.black + '%' }"></div>
              </div>
              <div class="outcome-labels">
                <span>W {{ winPct.white }}%</span>
                <span>D {{ winPct.draw }}%</span>
                <span>B {{ winPct.black }}%</span>
              </div>
            </div>
          </template>
          <div v-else class="no-stats">Press "Start Training"</div>
        </div>

        <!-- Active games with mini boards -->
        <div v-if="stats && stats.gameSlots.length > 0" class="panel games-panel">
          <h2 class="panel-title">Active Games</h2>
          <div class="mini-boards-grid">
            <div v-for="(slot, i) in stats.gameSlots" :key="i" class="mini-board-card" :class="{ 'standard-start': slot.isStandardStart }">
              <div class="mini-board-header">
                <span class="mini-id">#{{ i + 1 }}</span>
                <span v-if="slot.isStandardStart" class="std-badge">STD</span>
                <span class="mini-moves">{{ slot.moveCount }}/{{ slot.moveCap }}</span>
                <span class="mini-turn" :class="slot.currentTurn">{{ slot.currentTurn === 'white' ? 'W' : 'B' }}</span>
              </div>
              <div class="mini-board">
                <div v-for="rank in 8" :key="rank" class="mini-row">
                  <div
                    v-for="file in 8"
                    :key="file"
                    class="mini-square"
                    :class="miniSquareClass(rank - 1, file - 1)"
                  >
                    <span v-if="getPieceChar(slot, rank - 1, file - 1)" class="mini-piece">
                      {{ getPieceChar(slot, rank - 1, file - 1) }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Completed games -->
        <div v-if="reversedCompletedGames.length > 0" class="panel games-panel completed-panel">
          <h2 class="panel-title">Completed (last {{ reversedCompletedGames.length }})</h2>
          <div class="mini-boards-grid">
            <div
              v-for="game in reversedCompletedGames"
              :key="game.gameNumber"
              class="mini-board-card completed-card"
              :class="'outcome-' + game.outcomeClass"
            >
              <div class="mini-board-header">
                <span class="mini-id">#{{ game.gameNumber }}</span>
                <span class="mini-moves">{{ game.moveCount }}m</span>
                <span class="mini-outcome" :class="game.outcomeClass">{{ game.outcome }}</span>
              </div>
              <div class="mini-board">
                <div v-for="rank in 8" :key="rank" class="mini-row">
                  <div
                    v-for="file in 8"
                    :key="file"
                    class="mini-square"
                    :class="miniSquareClass(rank - 1, file - 1)"
                  >
                    <span v-if="getCompletedPieceChar(game, rank - 1, file - 1)" class="mini-piece">
                      {{ getCompletedPieceChar(game, rank - 1, file - 1) }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Log -->
      <div class="panel log-panel">
        <h2 class="panel-title">Log</h2>
        <div class="log-list">
          <div v-for="(entry, i) in visibleLog" :key="i" class="log-entry">
            <span class="log-time">{{ fmtTime(entry.time) }}</span>
            <span class="log-msg">{{ entry.message }}</span>
          </div>
          <div v-if="visibleLog.length === 0" class="log-empty">No activity</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.training-wrapper { width: 100%; height: 100%; background: #1a1a2e; display: flex; flex-direction: column; overflow-y: auto; }
.header { display: flex; align-items: center; gap: 10px; padding: 10px 16px; background: rgba(20,20,35,.95); border-bottom: 1px solid #333; flex-shrink: 0; flex-wrap: wrap; }
.back-btn { font-family: monospace; font-size: 13px; padding: 6px 14px; background: rgba(60,60,60,.8); border: 1px solid #555; border-radius: 6px; color: #ccc; cursor: pointer; }
.back-btn:hover { background: rgba(80,80,80,.9); color: white; }
.title { font-family: monospace; font-size: 22px; color: #fff; margin: 0; flex: 1; }
.header-controls { display: flex; align-items: center; gap: 8px; }
.action-btn { font-family: monospace; font-size: 12px; padding: 6px 16px; border: none; border-radius: 6px; cursor: pointer; color: white; white-space: nowrap; }
.action-btn:hover { filter: brightness(1.2); }
.action-btn:disabled { opacity: .5; cursor: not-allowed; }
.start-btn { background: #44aa44; min-width: 140px; text-align: center; }
.start-btn.stop { background: #cc4444; }
.start-btn.resume { background: #4a9eff; }
.copy-btn { background: #666; }
.reset-btn { background: #993333; }
.save-msg { font-family: monospace; font-size: 11px; color: #44aa44; min-width: 90px; opacity: 0; transition: opacity 0.15s; }
.save-msg.visible { opacity: 1; }

.content { display: flex; gap: 12px; padding: 12px 16px; flex: 1; min-height: 0; overflow: hidden; }

.left-column { width: 300px; flex-shrink: 0; overflow-y: auto; }
.center-column { flex: 1; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; min-width: 0; }
.panel { background: rgba(20,20,35,.95); border: 1px solid #333; border-radius: 8px; padding: 12px; }
.log-panel { width: 260px; flex-shrink: 0; display: flex; flex-direction: column; overflow: hidden; }

@media (max-width: 900px) {
  .content { flex-direction: column; overflow-y: auto; }
  .left-column { width: 100%; }
  .center-column { overflow: visible; }
  .log-panel { width: 100%; max-height: 200px; }
}

@media (max-width: 480px) {
  .header { gap: 6px; padding: 8px 10px; }
  .title { font-size: 16px; }
  .header-controls { flex-wrap: wrap; gap: 4px; }
  .action-btn { font-size: 11px; padding: 5px 10px; }
  .start-btn { min-width: 110px; }
  .content { padding: 8px; gap: 8px; }
  .charts-row { flex-direction: column; }
  .chart-box { min-width: 0; }
  .loss-row { flex-wrap: wrap; gap: 10px; }
  .stats-grid { grid-template-columns: repeat(auto-fill, minmax(75px, 1fr)); gap: 4px; }
  .mini-boards-grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 6px; }
  .mini-square { width: 10px; height: 10px; }
  .mini-piece { font-size: 8px; }
  .games-panel { max-height: none; }
}
.panel-title { font-family: monospace; font-size: 12px; color: #aaa; margin: 0 0 10px; text-transform: uppercase; letter-spacing: 1px; }

/* Config */
.config-grid { display: grid; grid-template-columns: 1fr auto; gap: 4px 8px; align-items: center; }
.config-grid label { font-family: monospace; font-size: 10px; color: #888; }
.config-grid input[type="number"] { width: 64px; font-family: monospace; font-size: 11px; padding: 2px 4px; background: rgba(0,0,0,.3); border: 1px solid #444; border-radius: 3px; color: #ccc; text-align: right; }
.config-grid input[type="range"] { width: 64px; accent-color: #4a9eff; }
.config-grid input:disabled { opacity: .4; }
.config-grid input:focus { outline: none; border-color: #4a9eff; }
.config-note { font-family: monospace; font-size: 10px; color: #666; margin: 0; line-height: 1.4; }
.config-readonly { font-family: monospace; font-size: 12px; color: #aaa; }
/* Reward sliders */
.reward-slider-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.rw-label { font-family: monospace; font-size: 10px; color: #888; width: 75px; flex-shrink: 0; }
.rw-slider {
  flex: 1;
  min-width: 80px;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: rgba(255,255,255,.1);
  outline: none;
  cursor: pointer;
}
.rw-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #4a9eff;
  border: 2px solid #1a1a2e;
  cursor: pointer;
  box-shadow: 0 0 4px rgba(74,158,255,.4);
}
.rw-slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #4a9eff;
  border: 2px solid #1a1a2e;
  cursor: pointer;
  box-shadow: 0 0 4px rgba(74,158,255,.4);
}
.rw-slider:disabled { opacity: .4; cursor: not-allowed; }
.rw-slider:disabled::-webkit-slider-thumb { background: #666; box-shadow: none; }
.rw-slider:disabled::-moz-range-thumb { background: #666; box-shadow: none; }
.rw-value { font-family: 'Courier New', monospace; font-size: 10px; color: #aaa; width: 45px; flex-shrink: 0; text-align: right; font-variant-numeric: tabular-nums; }
.rw-value.off { color: #555; }

@media (max-width: 480px) {
  .reward-slider-row { gap: 4px; margin-bottom: 8px; }
  .rw-label { font-size: 9px; width: 65px; }
  .rw-slider { height: 8px; }
  .rw-slider::-webkit-slider-thumb { width: 24px; height: 24px; }
  .rw-slider::-moz-range-thumb { width: 24px; height: 24px; }
}

/* Stats */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 6px; margin-bottom: 12px; }
.stat-card { background: rgba(0,0,0,.2); border: 1px solid rgba(255,255,255,.05); border-radius: 5px; padding: 6px 8px; }
.stat-label { font-family: monospace; font-size: 9px; color: #666; text-transform: uppercase; display: block; }
.stat-value { font-family: monospace; font-size: 13px; color: #ccc; font-weight: bold; }
.stat-value.big { font-size: 18px; color: #fff; }
.gpu-val { color: #44aa44; font-size: 11px; }
.tensor-warn { color: #ff4444 !important; }

.weights-row { display: flex; align-items: center; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.weights-label { font-family: monospace; font-size: 9px; color: #555; text-transform: uppercase; }
.weight-val { font-family: 'Courier New', monospace; font-size: 11px; font-variant-numeric: tabular-nums; padding: 2px 4px; background: rgba(0,0,0,.2); border-radius: 3px; }
.weight-val.positive { color: #5b8; }
.weight-val.negative { color: #b85; }

.loss-row { display: flex; gap: 16px; margin-bottom: 12px; }
.loss-card { display: flex; flex-direction: column; gap: 2px; }
.loss-label { font-family: monospace; font-size: 9px; color: #666; text-transform: uppercase; }
.loss-sublabel { font-family: monospace; font-size: 8px; color: #444; }
.loss-value { font-family: 'Courier New', monospace; font-size: 16px; font-weight: bold; font-variant-numeric: tabular-nums; }

.charts-row { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
.chart-box { flex: 1; min-width: 200px; }
.chart-title { font-family: monospace; font-size: 10px; font-weight: bold; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.loss-chart { width: 100%; height: 130px; background: rgba(0,0,0,.2); border: 1px solid rgba(255,255,255,.05); border-radius: 4px; }

.outcome-bar-container { max-width: 350px; margin-bottom: 8px; }
.outcome-bar { display: flex; height: 18px; border-radius: 3px; overflow: hidden; background: #333; }
.outcome-segment { transition: width .3s ease; }
.white-seg { background: #e0e0e0; }
.draw-seg { background: #888; }
.black-seg { background: #333; border-left: 1px solid #555; }
.outcome-labels { display: flex; justify-content: space-between; margin-top: 3px; font-family: monospace; font-size: 10px; color: #888; }

/* Mini boards */
.mini-boards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 8px; }
.mini-board-card { background: rgba(0,0,0,.2); border: 1px solid rgba(255,255,255,.05); border-radius: 5px; padding: 4px; }
.mini-board-header { display: flex; align-items: center; gap: 4px; margin-bottom: 3px; padding: 0 2px; font-family: monospace; font-size: 9px; }
.mini-id { color: #555; }
.mini-moves { color: #888; flex: 1; }
.mini-turn { font-weight: bold; }
.mini-turn.white { color: #ddd; }
.mini-turn.black { color: #888; }
.mini-board { display: flex; flex-direction: column; border: 1px solid #444; border-radius: 2px; overflow: hidden; }
.mini-row { display: flex; }
.mini-square { width: 12px; height: 12px; display: flex; align-items: center; justify-content: center; }
.mini-light { background: #c8b080; }
.mini-dark { background: #8b6b3d; }
.mini-piece { font-size: 10px; line-height: 1; }
.standard-start { border-color: rgba(74, 158, 255, 0.4); }
.std-badge { font-size: 7px; color: #4a9eff; font-weight: bold; background: rgba(74,158,255,.15); padding: 0 3px; border-radius: 2px; }

/* Completed games */
.completed-panel { opacity: 0.85; }
.completed-card { position: relative; }
.completed-card.outcome-white { border-color: rgba(255, 255, 255, 0.4); }
.completed-card.outcome-black { border-color: rgba(100, 100, 255, 0.3); }
.completed-card.outcome-draw { border-color: rgba(255, 170, 0, 0.3); }
.completed-card.outcome-cap { border-color: rgba(255, 80, 80, 0.3); }
.completed-card .mini-board { opacity: 0.6; }
.mini-outcome { font-size: 8px; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 50px; }
.mini-outcome.white { color: #eee; }
.mini-outcome.black { color: #88f; }
.mini-outcome.draw { color: #aa8800; }
.mini-outcome.cap { color: #ff6666; }

/* Log */
.log-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1px; }
.log-entry { display: flex; gap: 6px; padding: 2px 4px; font-family: monospace; font-size: 9px; background: rgba(0,0,0,.15); border-radius: 2px; }
.log-time { color: #444; flex-shrink: 0; width: 36px; }
.log-msg { color: #888; word-break: break-word; }
.log-empty { font-family: monospace; font-size: 11px; color: #444; text-align: center; padding: 20px; }

.no-stats { font-family: monospace; font-size: 13px; color: #555; padding: 30px 0; text-align: center; }
</style>
