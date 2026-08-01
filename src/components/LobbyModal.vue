<script setup lang="ts">
import { ref, computed } from 'vue';
import type { PlayerId } from '@/types/chess';
import type { LobbyPlayer } from '@/types/network';

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
  (e: 'playBot', model: 'sage' | 'toy'): void;
}>();

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
        <p class="subtitle">Online Multiplayer Chess</p>

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

          <!-- Bot opponents. Weights are fetched lazily on click — only
               the model you pick gets downloaded. -->
          <button class="lobby-btn ai-btn" @click="emit('playBot', 'sage')">
            <span class="bot-name">Play Against Sage AI</span>
            <span class="bot-tag">Strong AI - No Brain Visuals</span>
          </button>
          <button class="lobby-btn ai-btn toy-btn" @click="emit('playBot', 'toy')">
            <span class="bot-name">Play Toy Bot</span>
            <span class="bot-tag">Weak AI - Watch Input and Output Activations</span>
          </button>
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
  padding: 44px 54px;
  min-width: min(420px, 90dvw);
  max-width: 90dvw;
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
    padding: 28px 22px;
    border-radius: 16px;
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

.ai-btn {
  background: linear-gradient(165deg, #a855f7, #7c3aed);
  color: white;
  width: 100%;
  box-shadow: 0 6px 18px rgba(168, 85, 247, 0.3);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.ai-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #b569f9, #8b4ef0);
  box-shadow: 0 8px 24px rgba(168, 85, 247, 0.5);
}

/* Toy gets the teal accent to read as "the other one". */
.toy-btn {
  background: linear-gradient(165deg, #2dd4bf, #0d9488);
  box-shadow: 0 6px 18px rgba(45, 212, 191, 0.3);
}

.toy-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #46e4cf, #14b8a6);
  box-shadow: 0 8px 24px rgba(45, 212, 191, 0.5);
}

.bot-name {
  font-weight: 700;
}

.bot-tag {
  font-size: 11px;
  font-weight: 400;
  opacity: 0.85;
  letter-spacing: 0.3px;
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
