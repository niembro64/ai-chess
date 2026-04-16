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
  hasPresetModel: boolean;
}>();

const emit = defineEmits<{
  (e: 'host'): void;
  (e: 'join', roomCode: string): void;
  (e: 'start'): void;
  (e: 'cancel'): void;
  (e: 'playAi', source: 'trained' | 'preset'): void;
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
          <button class="lobby-btn host-btn" @click="handleHost">Host Game</button>

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

          <button class="lobby-btn ai-btn" @click="emit('playAi', 'trained')">Play vs Your AI</button>
          <button v-if="props.hasPresetModel" class="lobby-btn preset-btn" @click="emit('playAi', 'preset')">Play vs Preset AI</button>
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
  background: rgba(10, 10, 20, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
}

.lobby-modal {
  position: relative;
  background: rgba(20, 20, 35, 0.98);
  border: 2px solid #4444aa;
  border-radius: 16px;
  padding: 40px 50px;
  min-width: min(420px, 90vw);
  max-width: 90vw;
  text-align: center;
  box-shadow: 0 0 60px rgba(68, 68, 170, 0.3);
}

@media (max-width: 480px) {
  .lobby-modal {
    padding: 24px 20px;
    border-radius: 12px;
  }
}

.title {
  font-family: monospace;
  font-size: 32px;
  color: #ffffff;
  margin: 0;
  text-shadow: 0 0 20px rgba(68, 68, 170, 0.5);
}

.subtitle {
  font-family: monospace;
  font-size: 14px;
  color: #888;
  margin: 8px 0 20px 0;
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
  font-family: monospace;
  font-size: 16px;
  padding: 10px 28px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.lobby-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.host-btn {
  background: #44aa44;
  color: white;
  width: 100%;
}

.host-btn:hover:not(:disabled) {
  background: #55cc55;
}

.ai-btn {
  background: #9944cc;
  color: white;
  width: 100%;
}

.ai-btn:hover:not(:disabled) {
  background: #aa55dd;
}

.preset-btn {
  background: #4a9eff;
  color: white;
  width: 100%;
}

.preset-btn:hover:not(:disabled) {
  background: #5aafff;
}

.join-btn {
  background: #4a9eff;
  color: white;
}

.join-btn:hover:not(:disabled) {
  background: #5aafff;
}

.start-btn {
  background: #44aa44;
  color: white;
}

.start-btn:hover:not(:disabled) {
  background: #55cc55;
}

.cancel-btn {
  background: #666;
  color: white;
}

.cancel-btn:hover {
  background: #777;
}

.code-input {
  font-family: monospace;
  font-size: 20px;
  text-align: center;
  width: 110px;
  padding: 8px 10px;
  background: rgba(0, 0, 0, 0.3);
  border: 2px solid #4444aa;
  border-radius: 8px;
  color: white;
  text-transform: uppercase;
  letter-spacing: 4px;
}

.code-input::placeholder {
  color: #555;
  letter-spacing: 4px;
}

.code-input:focus {
  outline: none;
  border-color: #6666cc;
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
  gap: 12px;
  background: rgba(0, 0, 0, 0.3);
  padding: 12px 20px;
  border-radius: 10px;
  margin-bottom: 20px;
  cursor: pointer;
  border: 1px solid rgba(74, 158, 255, 0.3);
  transition: all 0.2s ease;
  justify-content: center;
}

.room-code-display:hover {
  border-color: rgba(74, 158, 255, 0.6);
  background: rgba(0, 0, 0, 0.4);
}

.room-label {
  font-family: monospace;
  font-size: 13px;
  color: #888;
}

.room-code {
  font-family: monospace;
  font-size: 28px;
  color: #4a9eff;
  letter-spacing: 6px;
  font-weight: bold;
  user-select: all;
  text-shadow: 0 0 10px rgba(74, 158, 255, 0.4);
}

.copy-btn {
  font-size: 16px;
  width: 32px;
  height: 32px;
  padding: 0;
  background: rgba(74, 158, 255, 0.2);
  border: 1px solid #4a9eff;
  border-radius: 8px;
  color: #4a9eff;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.copy-btn:hover {
  background: rgba(74, 158, 255, 0.4);
}

.copy-btn.copied {
  background: rgba(68, 170, 68, 0.3);
  border-color: #44aa44;
  color: #44aa44;
}

.players-section {
  margin-bottom: 15px;
}

.players-title {
  font-family: monospace;
  font-size: 16px;
  color: #aaa;
  margin: 0 0 15px 0;
}

.player-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.player-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 15px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
  margin-bottom: 8px;
}

.player-item.is-local {
  background: rgba(68, 68, 170, 0.2);
  border: 1px solid rgba(68, 68, 170, 0.4);
}

.player-color {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.3);
}

.player-name {
  font-family: monospace;
  font-size: 16px;
  color: white;
  flex: 1;
  text-align: left;
}

.host-badge {
  font-family: monospace;
  font-size: 11px;
  background: #44aa44;
  color: white;
  padding: 3px 8px;
  border-radius: 4px;
}

.you-badge {
  font-family: monospace;
  font-size: 11px;
  background: #4a9eff;
  color: white;
  padding: 3px 8px;
  border-radius: 4px;
}

.waiting-text {
  font-family: monospace;
  font-size: 14px;
  color: #888;
  padding: 14px 20px;
}

.waiting-hint {
  font-family: monospace;
  font-size: 13px;
  color: #666;
  margin-bottom: 10px;
}

.error-message {
  font-family: monospace;
  font-size: 14px;
  color: #ff6666;
  background: rgba(255, 0, 0, 0.1);
  padding: 10px 15px;
  border-radius: 6px;
  margin-top: 15px;
}

.connecting-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(68, 68, 170, 0.3);
  border-top-color: #4a9eff;
  border-radius: 50%;
  margin: 20px auto;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
