<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue';
import type { PlayerId, Move, ChessGameState, PieceColor } from '@/types/chess';
import { playerIdToColor } from '@/types/chess';
import type { NetworkGameSnapshot, LobbyPlayer, NetworkRole } from '@/types/network';
import type { GameConnection } from '@/types/game';
import { createInitialGameState, posToAlgebraic } from '@/game/chess/ChessEngine';
import { networkManager } from '@/game/network/NetworkManager';
import { ChessServer } from '@/game/server/ChessServer';
import { LocalGameConnection } from '@/game/server/LocalGameConnection';
import { RemoteGameConnection } from '@/game/server/RemoteGameConnection';
import { AIPlayer, type AIModelSource } from '@/game/ai/AIPlayer';
import { PRESET_WEIGHTS } from '@/game/ai/presetWeights';
import LobbyModal from './LobbyModal.vue';
import ChessBoard from './ChessBoard.vue';

// Lobby state
const showLobby = ref(true);
const isHost = ref(false);
const roomCode = ref('');
const lobbyPlayers = ref<LobbyPlayer[]>([]);
const localPlayerId = ref<PlayerId>(1);
const lobbyError = ref<string | null>(null);
const isConnecting = ref(false);
const gameStarted = ref(false);
const networkRole = ref<NetworkRole | null>(null);

// Game state
const gameState = ref<ChessGameState>(createInitialGameState());
const drawOffer = ref<PlayerId | null>(null);

// Server & connection
let currentServer: ChessServer | null = null;
let activeConnection: GameConnection | null = null;

// AI opponent
let aiPlayer: AIPlayer | null = null;
const playingVsAi = ref(false);
const aiThinking = ref(false);
const aiModelStatus = ref<AIModelSource | 'none'>('none');

// Computed
const localColor = computed<PieceColor>(() => playerIdToColor(localPlayerId.value));
const isMyTurn = computed(() => gameState.value.currentTurn === localColor.value);
const isGameOver = computed(() =>
  gameState.value.status === 'checkmate' ||
  gameState.value.status === 'stalemate' ||
  gameState.value.status === 'draw'
);

const statusText = computed(() => {
  const gs = gameState.value;
  switch (gs.status) {
    case 'waiting':
      return 'Waiting to start...';
    case 'active':
      if (playingVsAi.value && !isMyTurn.value) {
        return aiThinking.value ? 'AI is thinking...' : "AI's turn";
      }
      return isMyTurn.value ? 'Your turn' : "Opponent's turn";
    case 'check':
      return isMyTurn.value ? 'You are in check!' : 'Opponent is in check';
    case 'checkmate':
      if (gs.winner === localColor.value) return 'Checkmate - You win!';
      return playingVsAi.value ? 'Checkmate - AI wins' : 'Checkmate - You lose';
    case 'stalemate':
      return 'Stalemate - Draw';
    case 'draw':
      return 'Draw';
    default:
      return '';
  }
});

const turnIndicator = computed(() => {
  if (isGameOver.value) return '';
  const extra = playingVsAi.value ? ` (${aiModelStatus.value === 'trained' ? 'trained model' : 'random model'})` : '';
  return (gameState.value.currentTurn === 'white' ? 'White to move' : 'Black to move') + extra;
});

const moveHistoryDisplay = computed(() => {
  const moves = gameState.value.moveHistory;
  const display: string[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    const moveNum = Math.floor(i / 2) + 1;
    const whiteMove = formatMove(moves[i]);
    const blackMove = i + 1 < moves.length ? formatMove(moves[i + 1]) : '';
    display.push(`${moveNum}. ${whiteMove}${blackMove ? ' ' + blackMove : ''}`);
  }
  return display;
});

function formatMove(move: Move): string {
  const from = posToAlgebraic(move.from);
  const to = posToAlgebraic(move.to);
  const promo = move.promotion ? `=${move.promotion[0].toUpperCase()}` : '';
  return `${from}${to}${promo}`;
}

// --- AI move scheduling ---

function scheduleAiMove(): void {
  if (!playingVsAi.value || !currentServer || !aiPlayer) return;
  if (isGameOver.value) return;
  if (gameState.value.currentTurn === localColor.value) return; // Human's turn

  aiThinking.value = true;
  // Use setTimeout so the UI updates before AI starts computing
  setTimeout(() => {
    if (!aiPlayer || !currentServer || isGameOver.value) {
      aiThinking.value = false;
      return;
    }
    const move = aiPlayer.getMove(gameState.value);
    aiThinking.value = false;
    // Send AI's move as player 2
    currentServer.receiveCommand({ type: 'move', move }, 2);
  }, 50);
}

// --- Lobby handlers ---

async function handleHost(): Promise<void> {
  try {
    isConnecting.value = true;
    lobbyError.value = null;

    const code = await networkManager.hostGame();
    roomCode.value = code;
    isHost.value = true;
    networkRole.value = 'host';
    localPlayerId.value = 1;

    lobbyPlayers.value = [{
      playerId: 1,
      name: 'White',
      isHost: true,
    }];

    setupNetworkCallbacks();
    isConnecting.value = false;
  } catch (err) {
    lobbyError.value = (err as Error).message || 'Failed to host game';
    isConnecting.value = false;
  }
}

async function handleJoin(code: string): Promise<void> {
  try {
    isConnecting.value = true;
    lobbyError.value = null;

    await networkManager.joinGame(code);
    roomCode.value = code;
    isHost.value = false;
    networkRole.value = 'client';

    setupNetworkCallbacks();
    isConnecting.value = false;
  } catch (err) {
    lobbyError.value = (err as Error).message || 'Failed to join game';
    isConnecting.value = false;
  }
}

async function handlePlayAi(source: 'trained' | 'preset'): Promise<void> {
  isConnecting.value = true;
  lobbyError.value = null;

  if (source === 'preset' && PRESET_WEIGHTS) {
    aiPlayer = AIPlayer.createFromPreset(PRESET_WEIGHTS, 50);
    aiModelStatus.value = 'preset';
  } else {
    // Try trained model from IndexedDB, fall back to untrained
    const trained = await AIPlayer.createFromTrained(50);
    if (trained) {
      aiPlayer = trained;
      aiModelStatus.value = 'trained';
    } else {
      aiPlayer = AIPlayer.createUntrained(25);
      aiModelStatus.value = 'untrained';
    }
  }

  playingVsAi.value = true;
  networkRole.value = null;
  localPlayerId.value = 1;
  isConnecting.value = false;

  startGameWithPlayers([1, 2]);
}

function handleLobbyStart(): void {
  networkManager.startGame();
}

function handleLobbyCancel(): void {
  networkManager.disconnect();
  networkRole.value = null;
  roomCode.value = '';
  isHost.value = false;
  lobbyPlayers.value = [];
  lobbyError.value = null;
  isConnecting.value = false;
}

function setupNetworkCallbacks(): void {
  networkManager.onPlayerJoined = (player: LobbyPlayer) => {
    const existing = lobbyPlayers.value.find(p => p.playerId === player.playerId);
    if (!existing) {
      lobbyPlayers.value = [...lobbyPlayers.value, player];
    }
  };

  networkManager.onPlayerLeft = (playerId: PlayerId) => {
    lobbyPlayers.value = lobbyPlayers.value.filter(p => p.playerId !== playerId);
  };

  networkManager.onPlayerAssignment = (playerId: PlayerId) => {
    localPlayerId.value = playerId;
  };

  networkManager.onGameStart = (playerIds: PlayerId[]) => {
    startGameWithPlayers(playerIds);
  };

  networkManager.onError = (error: string) => {
    lobbyError.value = error;
  };
}

function startGameWithPlayers(playerIds: PlayerId[]): void {
  showLobby.value = false;
  gameStarted.value = true;

  if (networkRole.value !== 'client') {
    currentServer = new ChessServer();
    const localConnection = new LocalGameConnection(currentServer);
    activeConnection = localConnection;

    localConnection.onSnapshot((snapshot: NetworkGameSnapshot) => {
      gameState.value = snapshot.gameState;
      drawOffer.value = snapshot.drawOffer;

      // If playing vs AI and it's AI's turn, schedule its move
      if (playingVsAi.value) {
        scheduleAiMove();
      }
    });

    if (networkRole.value === 'host') {
      currentServer.addSnapshotListener((state: NetworkGameSnapshot) => {
        networkManager.broadcastState(state);
      });
      networkManager.onCommandReceived = (command, fromPlayerId) => {
        currentServer?.receiveCommand(command, fromPlayerId as PlayerId);
      };
    }

    currentServer.start();
  } else {
    const remoteConnection = new RemoteGameConnection();
    activeConnection = remoteConnection;
    remoteConnection.onSnapshot((snapshot: NetworkGameSnapshot) => {
      gameState.value = snapshot.gameState;
      drawOffer.value = snapshot.drawOffer;
    });
  }

  console.log('Game started with players:', playerIds);
}

function handleMove(move: Move): void {
  if (!activeConnection) return;
  activeConnection.sendCommand({ type: 'move', move });
}

function handleResign(): void {
  if (!activeConnection || isGameOver.value) return;
  if (confirm('Are you sure you want to resign?')) {
    activeConnection.sendCommand({ type: 'resign' });
  }
}

function handleOfferDraw(): void {
  if (!activeConnection || isGameOver.value) return;
  activeConnection.sendCommand({ type: 'offerDraw' });
}

function handleAcceptDraw(): void {
  if (!activeConnection) return;
  activeConnection.sendCommand({ type: 'acceptDraw' });
}

function handleDeclineDraw(): void {
  if (!activeConnection) return;
  activeConnection.sendCommand({ type: 'declineDraw' });
}

function returnToLobby(): void {
  if (currentServer) {
    currentServer.stop();
    currentServer = null;
  }
  if (aiPlayer) {
    aiPlayer.dispose();
    aiPlayer = null;
  }
  activeConnection?.disconnect();
  activeConnection = null;
  networkManager.disconnect();

  gameStarted.value = false;
  showLobby.value = true;
  networkRole.value = null;
  playingVsAi.value = false;
  aiThinking.value = false;
  aiModelStatus.value = 'none';
  lobbyPlayers.value = [];
  roomCode.value = '';
  isHost.value = false;
  gameState.value = createInitialGameState();
  drawOffer.value = null;
}

onUnmounted(() => {
  if (aiPlayer) {
    aiPlayer.dispose();
    aiPlayer = null;
  }
});
</script>

<template>
  <div class="game-wrapper">
    <!-- Lobby Modal -->
    <LobbyModal
      :visible="showLobby"
      :is-host="isHost"
      :room-code="roomCode"
      :players="lobbyPlayers"
      :local-player-id="localPlayerId"
      :error="lobbyError"
      :is-connecting="isConnecting"
      :has-preset-model="PRESET_WEIGHTS !== null"
      @host="handleHost"
      @join="handleJoin"
      @start="handleLobbyStart"
      @cancel="handleLobbyCancel"
      @play-ai="handlePlayAi"
    />

    <!-- Game UI (visible when game started) -->
    <div v-if="gameStarted" class="game-area">
      <div class="game-layout">
        <!-- Left sidebar: move history -->
        <div class="sidebar">
          <div class="sidebar-header">
            <h2 class="sidebar-title">Moves</h2>
          </div>
          <div class="move-list">
            <div
              v-for="(line, i) in moveHistoryDisplay"
              :key="i"
              class="move-line"
            >
              {{ line }}
            </div>
            <div v-if="moveHistoryDisplay.length === 0" class="no-moves">
              No moves yet
            </div>
          </div>
        </div>

        <!-- Center: chess board -->
        <div class="board-area">
          <!-- Status bar -->
          <div class="status-bar">
            <span class="status-text" :class="{
              'my-turn': isMyTurn && !isGameOver,
              'game-over': isGameOver,
              'in-check': gameState.status === 'check',
            }">
              {{ statusText }}
            </span>
            <span v-if="turnIndicator" class="turn-indicator">
              {{ turnIndicator }}
            </span>
          </div>

          <!-- Draw offer banner -->
          <div v-if="drawOffer && drawOffer !== localPlayerId && !isGameOver" class="draw-offer-banner">
            <span>Opponent offers a draw</span>
            <button class="draw-btn accept" @click="handleAcceptDraw">Accept</button>
            <button class="draw-btn decline" @click="handleDeclineDraw">Decline</button>
          </div>

          <ChessBoard
            :game-state="gameState"
            :local-player-id="localPlayerId"
            @move="handleMove"
          />

          <!-- Player info -->
          <div class="player-info">
            <span class="player-label">
              You are playing as
              <strong :style="{ color: localColor === 'white' ? '#fff' : '#aaa' }">
                {{ localColor === 'white' ? 'White' : 'Black' }}
              </strong>
            </span>
          </div>

          <!-- Game controls -->
          <div class="game-controls">
            <template v-if="!isGameOver">
              <button class="control-btn" @click="handleOfferDraw" :disabled="drawOffer === localPlayerId">
                {{ drawOffer === localPlayerId ? 'Draw Offered' : 'Offer Draw' }}
              </button>
              <button class="control-btn resign-btn" @click="handleResign">Resign</button>
            </template>
            <template v-else>
              <button class="control-btn lobby-btn" @click="returnToLobby">Return to Lobby</button>
            </template>
          </div>
        </div>

        <!-- Right sidebar: captured pieces -->
        <div class="sidebar">
          <div class="sidebar-header">
            <h2 class="sidebar-title">Game Info</h2>
          </div>
          <div class="game-info-content">
            <div class="info-row">
              <span class="info-label">Mode:</span>
              <span class="info-value">{{ playingVsAi ? 'vs AI' : networkRole === 'host' ? 'Host' : networkRole === 'client' ? 'Client' : 'Local' }}</span>
            </div>
            <div v-if="playingVsAi" class="info-row">
              <span class="info-label">AI Model:</span>
              <span class="info-value" :class="{
                'ai-trained': aiModelStatus === 'trained',
                'ai-preset': aiModelStatus === 'preset',
                'ai-untrained': aiModelStatus === 'untrained',
              }">
                {{ aiModelStatus === 'trained' ? 'Your Model' : aiModelStatus === 'preset' ? 'Preset' : 'Random' }}
              </span>
            </div>
            <div v-if="!playingVsAi && roomCode" class="info-row">
              <span class="info-label">Room:</span>
              <span class="info-value">{{ roomCode }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Move #:</span>
              <span class="info-value">{{ gameState.fullMoveNumber }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Status:</span>
              <span class="info-value">{{ gameState.status }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Background when no game -->
    <div v-if="!gameStarted && !showLobby" class="empty-bg"></div>
  </div>
</template>

<style scoped>
.game-wrapper {
  width: 100%;
  height: 100%;
  position: relative;
  background: #1a1a2e;
}

.game-area {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow-y: auto;
}

.game-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  padding: 20px;
}

@media (max-width: 900px) {
  .game-layout {
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 10px;
  }
}

.sidebar {
  width: 200px;
  background: rgba(20, 20, 35, 0.95);
  border: 1px solid #333;
  border-radius: 8px;
  overflow: hidden;
}

@media (max-width: 900px) {
  .sidebar {
    width: 100%;
    max-width: 400px;
  }
  .sidebar:first-child {
    order: 2;
  }
  .sidebar:last-child {
    order: 3;
  }
  .board-area {
    order: 1;
  }
}

.sidebar-header {
  background: rgba(68, 68, 170, 0.2);
  border-bottom: 1px solid #333;
  padding: 10px 14px;
}

.sidebar-title {
  font-family: monospace;
  font-size: 14px;
  color: #aaa;
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.move-list {
  padding: 10px 14px;
  max-height: 500px;
  overflow-y: auto;
}

.move-line {
  font-family: monospace;
  font-size: 13px;
  color: #ccc;
  padding: 3px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.no-moves {
  font-family: monospace;
  font-size: 13px;
  color: #555;
  font-style: italic;
}

.board-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 20px;
  background: rgba(20, 20, 35, 0.95);
  border: 1px solid #333;
  border-radius: 8px;
  min-width: min(400px, 90vw);
  justify-content: center;
}

.status-text {
  font-family: monospace;
  font-size: 16px;
  color: #888;
}

.status-text.my-turn {
  color: #4a9eff;
  font-weight: bold;
}

.status-text.game-over {
  color: #ffaa00;
  font-weight: bold;
}

.status-text.in-check {
  color: #ff4444;
  font-weight: bold;
}

.turn-indicator {
  font-family: monospace;
  font-size: 12px;
  color: #666;
}

.draw-offer-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  background: rgba(255, 170, 0, 0.15);
  border: 1px solid rgba(255, 170, 0, 0.4);
  border-radius: 8px;
  font-family: monospace;
  font-size: 13px;
  color: #ffaa00;
}

.draw-btn {
  font-family: monospace;
  font-size: 12px;
  padding: 4px 12px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.draw-btn.accept {
  background: #44aa44;
  color: white;
}

.draw-btn.accept:hover {
  background: #55cc55;
}

.draw-btn.decline {
  background: #666;
  color: white;
}

.draw-btn.decline:hover {
  background: #777;
}

.player-info {
  font-family: monospace;
  font-size: 14px;
  color: #888;
}

.player-label strong {
  font-size: 15px;
}

.game-controls {
  display: flex;
  gap: 12px;
}

.control-btn {
  font-family: monospace;
  font-size: 13px;
  padding: 8px 20px;
  background: rgba(60, 60, 60, 0.8);
  border: 1px solid #555;
  border-radius: 6px;
  color: #ccc;
  cursor: pointer;
  transition: all 0.15s ease;
}

.control-btn:hover:not(:disabled) {
  background: rgba(80, 80, 80, 0.9);
  border-color: #777;
  color: white;
}

.control-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.resign-btn {
  border-color: rgba(255, 80, 80, 0.3);
  color: #ff6666;
}

.resign-btn:hover {
  background: rgba(255, 40, 40, 0.2) !important;
  border-color: rgba(255, 80, 80, 0.6) !important;
}

.lobby-btn {
  background: #4444aa;
  border-color: #5555cc;
  color: white;
}

.lobby-btn:hover {
  background: #5555cc !important;
}

.game-info-content {
  padding: 10px 14px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.info-label {
  font-family: monospace;
  font-size: 12px;
  color: #666;
}

.info-value {
  font-family: monospace;
  font-size: 12px;
  color: #aaa;
}

.info-value.ai-trained {
  color: #44aa44;
  font-weight: bold;
}

.info-value.ai-preset {
  color: #4a9eff;
  font-weight: bold;
}

.info-value.ai-untrained {
  color: #cc8844;
}

.empty-bg {
  width: 100%;
  height: 100%;
  background: #1a1a2e;
}

/* Scrollbar styling for move list */
.move-list::-webkit-scrollbar {
  width: 6px;
}

.move-list::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
}

.move-list::-webkit-scrollbar-thumb {
  background: #444;
  border-radius: 3px;
}
</style>
