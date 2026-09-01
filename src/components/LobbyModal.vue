<script setup lang="ts">
import { ref, computed } from 'vue';
import type { PlayerId } from '@/types/chess';
import type { LobbyPlayer } from '@/types/network';
import {
  EFFORT_LEVELS,
  GRID_ASKED,
  GRID_MODELS,
  MODELS,
  botFace,
  goalLabel,
  isInverted,
  pieceTint,
  type Effort,
  type Goal,
  type ModelId,
} from '@/game/ai/models';
import BotIcon from './BotIcon.vue';
import PieceIcon from './PieceIcon.vue';
import type { BotIconName } from './botIcons';

const props = defineProps<{
  visible: boolean;
  isHost: boolean;
  roomCode: string;
  players: LobbyPlayer[];
  localPlayerId: PlayerId;
  error: string | null;
  isConnecting: boolean;
}>();

const emit = defineEmits<{
  (e: 'host'): void;
  (e: 'join', roomCode: string): void;
  (e: 'start'): void;
  (e: 'cancel'): void;
  (e: 'playBot', model: ModelId, opts: {
    goalInverted: boolean;
    effort: Effort;
    playColor: 'white' | 'black';
  }): void;
}>();

// --- AI setup ---------------------------------------------------------
//
// The model grid crosses MODEL × GOAL: each row is a network (by what
// it was TRAINED to do), each column is what we ASK it to do at play
// time — left = its trained goal, right = inverted (the misère search
// flip, not "pick the worst move"). Toy sits below as its own option.

// Grid selection: which network, and what we ask of it.
const pickedModel = ref<ModelId>('sage');
const askedGoal = ref<Goal>('win');
const playColor = ref<'white' | 'black'>('white');
const effort = ref<Effort>('medium');

const goalInverted = computed(() => isInverted(pickedModel.value, askedGoal.value));

// The previews show the exact piece colors the game will start with:
// your standard set, and the bot's tinted set in the opposite color.
function tintStyle(model: ModelId | null, color: 'white' | 'black'): Record<string, string> {
  const t = pieceTint(model, color);
  return { color: t.fill, '--piece-outline': t.outline };
}

function startBot(): void {
  emit('playBot', pickedModel.value, {
    goalInverted: goalInverted.value,
    effort: effort.value,
    playColor: playColor.value,
  });
}

const joinCode = ref('');
const codeCopied = ref(false);

async function copyCode() {
  try {
    await navigator.clipboard.writeText(props.roomCode);
    codeCopied.value = true;
    setTimeout(() => {
      codeCopied.value = false;
    }, 2000);
  } catch {
    const codeEl = document.querySelector('.room-code') as HTMLElement;
    if (codeEl) {
      const range = document.createRange();
      range.selectNodeContents(codeEl);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
  }
}

function getPlayerColor(playerId: PlayerId): string {
  return playerId === 1 ? '#ffffff' : '#333333';
}

function handleHost() {
  emit('host');
}

function handleJoinSubmit() {
  if (joinCode.value.length >= 4) {
    emit('join', joinCode.value.toUpperCase());
  }
}

function handleStart() {
  emit('start');
}

function handleCancel() {
  joinCode.value = '';
  emit('cancel');
}

const canStart = computed(() => {
  return props.isHost && props.players.length >= 2;
});

const isInLobby = computed(() => {
  return props.roomCode !== '';
});

const canJoin = computed(() => {
  return joinCode.value.length >= 4;
});
</script>

<template>
  <div v-if="visible" class="lobby-overlay">
    <div class="lobby-modal">
      <!-- Initial screen -->
      <template v-if="!isInLobby && !isConnecting">
        <h1 class="title">AI CHESS</h1>
        <p class="subtitle">Can you lose to a bot trying to lose?</p>

        <div class="main-actions">
          <button class="lobby-btn host-btn" @click="handleHost">Play Online</button>

          <div class="join-row">
            <input
              v-model="joinCode"
              class="code-input"
              type="text"
              maxlength="4"
              placeholder="CODE"
              @keyup.enter="handleJoinSubmit"
            />
            <button
              class="lobby-btn join-btn"
              :disabled="!canJoin"
              @click="handleJoinSubmit"
            >Join</button>
          </div>

          <!-- ============ VS AI setup ============
               Weights are fetched lazily on start — only the model you
               actually play gets downloaded. -->
          <div class="setup">
            <div class="setup-title">AI Model</div>
            <div class="model-table">
              <div class="mt-corner"></div>
              <div v-for="m in GRID_MODELS" :key="`h-${m}`" class="mt-colhead">
                TRAINED TO {{ goalLabel(MODELS[m].trainedGoal) }}
              </div>
              <template v-for="asked in GRID_ASKED" :key="`r-${asked}`">
                <div class="mt-rowhead"><span>ASKED TO {{ goalLabel(asked) }}</span></div>
                <button
                  v-for="m in GRID_MODELS"
                  :key="`${m}-${asked}`"
                  class="mt-cell"
                  :class="[
                    `m-${m}`,
                    { active: pickedModel === m && askedGoal === asked },
                  ]"
                  @click="pickedModel = m; askedGoal = asked"
                >
                  <BotIcon class="mt-face" :name="botFace(m, asked) as BotIconName" />
                  <span class="mt-name">{{ MODELS[m].name }}</span>
                </button>
              </template>
            </div>
            <button
              class="mt-toy"
              :class="{ active: pickedModel === 'toy' }"
              @click="pickedModel = 'toy'; askedGoal = 'win'"
            >
              <BotIcon class="mt-toy-face" name="toy" />
              <span class="mt-name">{{ MODELS.toy.name }}</span>
              <span class="mt-toy-sub">tiny net · watch it think</span>
            </button>

            <div class="setup-title">You Play</div>
            <div class="color-row">
              <button
                v-for="c in (['white', 'black'] as const)"
                :key="c"
                class="color-cell"
                :class="{ active: playColor === c }"
                :aria-label="`Play as ${c}`"
                @click="playColor = c"
              >
                <span
                  class="king"
                  :style="tintStyle(playColor === c ? null : pickedModel, c)"
                >
                  <PieceIcon type="king" />
                </span>
              </button>
            </div>

            <div class="setup-title">Model Effort</div>
            <div class="effort-seg">
              <button
                v-for="(lvl, key) in EFFORT_LEVELS"
                :key="key"
                :class="{ active: effort === key }"
                @click="effort = key as Effort"
              >{{ lvl.label }}</button>
            </div>
            <p class="setup-explain">
              Effort is how much the model actually thinks before moving:
              at <b>Low</b> it barely looks ahead, at <b>High</b> it runs its
              full search. It always plays the best move it found.
            </p>

            <button class="lobby-btn start-btn" @click="startBot">
              Play {{ MODELS[pickedModel].name }}
            </button>
          </div>
        </div>

        <div v-if="error" class="error-message">{{ error }}</div>
      </template>

      <!-- Connecting screen -->
      <template v-else-if="isConnecting">
        <h1 class="title">CONNECTING...</h1>
        <div class="connecting-spinner"></div>
        <div class="footer-row">
          <button class="lobby-btn cancel-btn" @click="handleCancel">Cancel</button>
        </div>
      </template>

      <!-- Lobby screen -->
      <template v-else-if="isInLobby">
        <h1 class="title">GAME LOBBY</h1>

        <div class="room-code-display" @click="copyCode">
          <span class="room-label">Share code:</span>
          <span class="room-code">{{ roomCode }}</span>
          <button class="copy-btn" :class="{ copied: codeCopied }" :title="codeCopied ? 'Copied!' : 'Copy'">
            {{ codeCopied ? '\u2713' : '\u29C9' }}
          </button>
        </div>

        <div class="players-section">
          <h2 class="players-title">Players ({{ players.length }}/2)</h2>
          <ul class="player-list">
            <li
              v-for="player in players"
              :key="player.playerId"
              class="player-item"
              :class="{ 'is-local': player.playerId === localPlayerId }"
            >
              <span
                class="player-color"
                :style="{ backgroundColor: getPlayerColor(player.playerId) }"
              ></span>
              <span class="player-name">{{ player.name }}</span>
              <span v-if="player.isHost" class="host-badge">HOST</span>
              <span v-if="player.playerId === localPlayerId" class="you-badge">YOU</span>
            </li>
          </ul>
        </div>

        <p v-if="players.length < 2" class="waiting-hint">Waiting for opponent to join...</p>

        <div v-if="error" class="error-message">{{ error }}</div>

        <div class="footer-row">
          <button class="lobby-btn cancel-btn" @click="handleCancel">Leave</button>
          <div class="footer-spacer"></div>
          <button
            v-if="isHost"
            class="lobby-btn start-btn"
            :disabled="!canStart"
            @click="handleStart"
          >Start Game</button>
          <span v-else class="waiting-text">Waiting for host...</span>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.lobby-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  /* Layered backdrop matches the in-game gradient palette. */
  background:
    radial-gradient(ellipse at 30% 0%, rgba(99, 102, 241, 0.15), transparent 55%),
    rgba(7, 6, 15, 0.7);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  display: flex;
  align-items: safe center;
  justify-content: safe center;
  overflow-y: auto;
  z-index: 3000;
}

.lobby-modal {
  position: relative;
  background: linear-gradient(165deg, rgba(40, 38, 70, 0.85), rgba(20, 19, 38, 0.92));
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  padding: 34px 38px;
  min-width: min(420px, 90dvw);
  max-width: 90dvw;
  /* The AI setup section makes this screen tall — cap it and scroll
     inside rather than overflowing the viewport on short phones. */
  max-height: 92dvh;
  overflow-y: auto;
  overscroll-behavior: contain;
  text-align: center;
  backdrop-filter: blur(20px) saturate(1.4);
  -webkit-backdrop-filter: blur(20px) saturate(1.4);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 0 1px rgba(212, 175, 95, 0.15),
    0 0 80px rgba(99, 102, 241, 0.25),
    0 24px 60px rgba(0, 0, 0, 0.5);
}

@media (max-width: 480px) {
  .lobby-modal {
    padding: 22px 16px;
    border-radius: 16px;
  }
  /* Tighter cartoons + type so the whole setup still fits a phone. */
  .model-face {
    width: 38px;
    height: 38px;
  }
  .model-cell {
    padding: 8px 4px 7px;
  }
  .model-line {
    font-size: 9px;
  }
  .setup-explain {
    font-size: 10px;
  }
}

.title {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 36px;
  font-weight: 700;
  letter-spacing: 4px;
  background: linear-gradient(135deg, #f7c058 10%, #ffffff 50%, #5ae3d8 90%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
  text-shadow: 0 0 40px rgba(99, 102, 241, 0.3);
}

.subtitle {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 400;
  color: #94a3b8;
  letter-spacing: 0.5px;
  margin: 10px 0 24px 0;
}

.main-actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: stretch;
  width: 220px;
  margin: 0 auto 8px;
}

.join-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.join-row .join-btn {
  flex-shrink: 0;
}

.join-row .code-input {
  flex: 1;
  min-width: 0;
}

.lobby-btn {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.4px;
  padding: 11px 28px;
  border: 1px solid transparent;
  border-radius: 10px;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.2s ease,
              background 0.2s ease, border-color 0.2s ease;
}

.lobby-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}

.lobby-btn:active:not(:disabled) {
  transform: translateY(0);
}

.lobby-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.host-btn {
  background: linear-gradient(165deg, #5ae3d8, #14b8a6);
  color: #07060f;
  width: 100%;
  box-shadow: 0 6px 18px rgba(94, 234, 212, 0.25);
}

.host-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #6ef0e5, #2dd4bf);
  box-shadow: 0 8px 24px rgba(94, 234, 212, 0.45);
}

/* --- VS AI setup ------------------------------------------------------ */

.setup {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 4px;
  text-align: left;
}

.setup-title {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-top: 2px;
}

/* Model table — one connected button group, not four loose buttons.
   COLUMNS are the networks (by what they were trained to do); ROWS are
   what we ask of them, labelled with 90deg text down the left gutter. */
.model-table {
  display: grid;
  grid-template-columns: 18px 1fr 1fr;
  grid-template-rows: auto 1fr 1fr;
  border: 1.5px solid rgba(255, 255, 255, 0.14);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.03);
}

.mt-colhead {
  padding: 6px 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.8px;
  color: #94a3b8;
  text-align: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
}

.mt-corner {
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
}

.mt-rowhead {
  display: flex;
  align-items: center;
  justify-content: center;
  border-right: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
}

/* Bottom-to-top so the label reads upward alongside its row. */
.mt-rowhead span {
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  font-family: 'JetBrains Mono', monospace;
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.8px;
  color: #94a3b8;
  white-space: nowrap;
}

.mt-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 9px 4px 8px;
  border: none;
  /* Hairlines between cells keep the group reading as one control. */
  border-left: 1px solid rgba(255, 255, 255, 0.09);
  border-top: 1px solid rgba(255, 255, 255, 0.09);
  background: transparent;
  color: #e2e8f0;
  cursor: pointer;
  transition: background 0.15s;
}

.mt-cell:first-of-type,
.mt-rowhead + .mt-cell {
  border-left: none;
}

.mt-cell:hover {
  background: rgba(255, 255, 255, 0.06);
}

/* The neighbouring cells' hairline borders are painted after this one
   in DOM order, which clipped the selected ring on every cell except
   the first. Lift the active cell into its own stacking level so the
   full ring shows on all four. */
.mt-cell.active {
  position: relative;
  z-index: 1;
  background: var(--accent-bg, rgba(94, 234, 212, 0.16));
  box-shadow: inset 0 0 0 1.5px var(--accent, #5ae3d8);
}

.mt-cell.m-sage { --accent: #4ade80; --accent-bg: rgba(74, 222, 128, 0.16); }
.mt-cell.m-jester { --accent: #c084fc; --accent-bg: rgba(192, 132, 252, 0.16); }

.mt-face {
  width: 42px;
  height: 42px;
}

.mt-name {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: #cbd5e1;
}

.mt-cell.active .mt-name {
  color: var(--accent);
}

/* Toy is a different kind of thing (tiny teaching net), so it sits
   outside the grid as its own wide button. */
.mt-toy {
  --accent: #2dd4bf;
  --accent-bg: rgba(45, 212, 191, 0.16);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 9px;
  padding: 7px 10px;
  border: 1.5px solid rgba(255, 255, 255, 0.14);
  border-radius: 11px;
  background: rgba(255, 255, 255, 0.03);
  cursor: pointer;
}

.mt-toy:hover { background: rgba(255, 255, 255, 0.07); }

.mt-toy.active {
  border-color: var(--accent);
  background: var(--accent-bg);
}

.mt-toy.active .mt-name { color: var(--accent); }

.mt-toy-face {
  width: 26px;
  height: 26px;
}

.mt-toy-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: #64748b;
}

/* You-play picker: each option previews the REAL piece colors the game
   will start with — your standard set beside the bot's tinted one. */
.color-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.color-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  padding: 10px 6px;
  border: 1.5px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}

.color-cell:hover { background: rgba(255, 255, 255, 0.08); }

.color-cell.active {
  border-color: #5ae3d8;
  background: rgba(94, 234, 212, 0.12);
  box-shadow: 0 0 0 1px #5ae3d8;
}

/* White king always on the left, black on the right. Your side shows
   the standard set; the other king wears the chosen model's tint —
   exactly the two colors the game will start with. */
.king {
  width: 34px;
  height: 34px;
  filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.55));
}

/* Effort segmented control. */
.effort-seg {
  display: flex;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 9px;
  overflow: hidden;
}

.effort-seg button {
  flex: 1;
  padding: 8px 4px;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 1px;
  color: #94a3b8;
  background: transparent;
  border: none;
  cursor: pointer;
}

.effort-seg button.active {
  color: #07060f;
  background: #5ae3d8;
}

.setup-explain {
  margin: 0;
  font-size: 10.5px;
  line-height: 1.5;
  color: #64748b;
}

.setup-explain b {
  color: #94a3b8;
  font-weight: 600;
}

.start-btn {
  width: 100%;
  margin-top: 2px;
  background: linear-gradient(165deg, #f7c058, #e09b2d);
  color: #07060f;
  box-shadow: 0 6px 18px rgba(247, 192, 88, 0.3);
}

.start-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #ffd075, #f0a93a);
  box-shadow: 0 8px 24px rgba(247, 192, 88, 0.5);
}

.preset-btn {
  background: linear-gradient(165deg, #6366f1, #4f46e5);
  color: white;
  width: 100%;
  box-shadow: 0 6px 18px rgba(99, 102, 241, 0.3);
}

.preset-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #7c7ff5, #5d56e8);
}

.join-btn {
  background: linear-gradient(165deg, #6366f1, #4f46e5);
  color: white;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.join-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #7c7ff5, #5d56e8);
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
}

.start-btn {
  background: linear-gradient(165deg, #5ae3d8, #14b8a6);
  color: #07060f;
  box-shadow: 0 4px 12px rgba(94, 234, 212, 0.3);
}

.start-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #6ef0e5, #2dd4bf);
  box-shadow: 0 6px 20px rgba(94, 234, 212, 0.45);
}

.cancel-btn {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.15);
  color: #cbd5e1;
}

.cancel-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.12);
  border-color: rgba(255, 255, 255, 0.3);
}

.code-input {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 20px;
  font-weight: 600;
  text-align: center;
  width: 110px;
  padding: 9px 10px;
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid rgba(99, 102, 241, 0.45);
  border-radius: 10px;
  color: #f1f5f9;
  text-transform: uppercase;
  letter-spacing: 6px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.code-input::placeholder {
  color: #475569;
  letter-spacing: 6px;
  font-weight: 400;
}

.code-input:focus {
  outline: none;
  border-color: rgba(99, 102, 241, 0.85);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18),
              0 0 18px rgba(99, 102, 241, 0.3);
}

.footer-row {
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: center;
  margin-top: 20px;
}

.footer-spacer {
  flex: 1;
}

.room-code-display {
  display: flex;
  align-items: center;
  gap: 14px;
  background: linear-gradient(165deg, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.2));
  padding: 14px 22px;
  border-radius: 12px;
  margin-bottom: 22px;
  cursor: pointer;
  border: 1px solid rgba(94, 234, 212, 0.3);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  justify-content: center;
}

.room-code-display:hover {
  border-color: rgba(94, 234, 212, 0.6);
  box-shadow: 0 0 24px rgba(94, 234, 212, 0.2);
}

.room-label {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 11px;
  font-weight: 500;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}

.room-code {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 28px;
  color: #5ae3d8;
  letter-spacing: 8px;
  font-weight: 700;
  user-select: all;
  text-shadow: 0 0 18px rgba(94, 234, 212, 0.5);
}

.copy-btn {
  font-size: 16px;
  width: 32px;
  height: 32px;
  padding: 0;
  background: rgba(94, 234, 212, 0.12);
  border: 1px solid rgba(94, 234, 212, 0.5);
  border-radius: 8px;
  color: #5ae3d8;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.copy-btn:hover {
  background: rgba(94, 234, 212, 0.25);
  box-shadow: 0 0 12px rgba(94, 234, 212, 0.4);
}

.copy-btn.copied {
  background: rgba(94, 234, 212, 0.4);
  border-color: #5ae3d8;
  color: #07060f;
  font-weight: 700;
}

.players-section {
  margin-bottom: 15px;
}

.players-title {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 12px;
  font-weight: 600;
  color: #94a3b8;
  margin: 0 0 14px 0;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.player-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.player-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 10px;
  margin-bottom: 8px;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.player-item.is-local {
  background: linear-gradient(90deg, rgba(99, 102, 241, 0.18), rgba(94, 234, 212, 0.06));
  border-color: rgba(99, 102, 241, 0.45);
}

.player-color {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.25);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
}

.player-name {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  font-weight: 500;
  color: #f1f5f9;
  flex: 1;
  text-align: left;
}

.host-badge {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  background: linear-gradient(135deg, #5ae3d8, #14b8a6);
  color: #07060f;
  padding: 3px 9px;
  border-radius: 5px;
}

.you-badge {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  background: linear-gradient(135deg, #6366f1, #4f46e5);
  color: white;
  padding: 3px 9px;
  border-radius: 5px;
}

.waiting-text {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  color: #94a3b8;
  padding: 14px 20px;
  font-style: italic;
}

.waiting-hint {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 12px;
  color: #64748b;
  margin-bottom: 10px;
  font-style: italic;
}

.error-message {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  color: #ff8888;
  background: rgba(255, 60, 60, 0.1);
  border: 1px solid rgba(255, 80, 80, 0.3);
  padding: 11px 16px;
  border-radius: 8px;
  margin-top: 16px;
}

.connecting-spinner {
  width: 44px;
  height: 44px;
  border: 3px solid rgba(99, 102, 241, 0.18);
  border-top-color: #5ae3d8;
  border-right-color: #6366f1;
  border-radius: 50%;
  margin: 24px auto;
  animation: spin 1s linear infinite;
  box-shadow: 0 0 24px rgba(99, 102, 241, 0.3);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
