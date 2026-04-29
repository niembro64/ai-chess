<script setup lang="ts">
import { ref, computed, onUnmounted, watch, nextTick } from 'vue';
import type { PlayerId, Move, ChessGameState, PieceColor } from '@/types/chess';
import { playerIdToColor } from '@/types/chess';
import type { NetworkGameSnapshot, LobbyPlayer, NetworkRole } from '@/types/network';
import type { GameConnection } from '@/types/game';
import { createInitialGameState, posToAlgebraic } from '@/game/chess/ChessEngine';
import { networkManager } from '@/game/network/NetworkManager';
import { ChessServer } from '@/game/server/ChessServer';
import { LocalGameConnection } from '@/game/server/LocalGameConnection';
import { RemoteGameConnection } from '@/game/server/RemoteGameConnection';
import { AIPlayer } from '@/game/ai/AIPlayer';
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

// AI opponent. The bot is the single model in `PRESET_WEIGHTS`
// (produced by the Python trainer and deployed via deploy_to_browser.py).
let aiPlayer: AIPlayer | null = null;
const playingVsBot = ref(false);
const aiThinking = ref(false);

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
      if (playingVsBot.value && !isMyTurn.value) {
        return aiThinking.value ? 'AI is thinking...' : "AI's turn";
      }
      return isMyTurn.value ? 'Your turn' : "Opponent's turn";
    case 'check':
      return isMyTurn.value ? 'You are in check!' : 'Opponent is in check';
    case 'checkmate':
      if (gs.winner === localColor.value) return 'Checkmate - You win!';
      return playingVsBot.value ? 'Checkmate - AI wins' : 'Checkmate - You lose';
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
  const extra = playingVsBot.value ? ' (bot)' : '';
  return (gameState.value.currentTurn === 'white' ? 'White to move' : 'Black to move') + extra;
});

// Player chips: opponent shown above the board, local player below.
// Color reflects the piece color each side is actually playing, regardless
// of board orientation — the chip on top of the screen is the side you
// look at across the board, which is always the opponent.
const opponentColor = computed<PieceColor>(() => localColor.value === 'white' ? 'black' : 'white');
const opponentName = computed(() => {
  if (playingVsBot.value) return 'Bot';
  if (networkRole.value) return 'Opponent';
  return 'Player 2';
});
const localName = computed(() => 'You');
const isOpponentTurn = computed(() => gameState.value.currentTurn === opponentColor.value);
const isLocalTurn = computed(() => gameState.value.currentTurn === localColor.value);

const moveListEl = ref<HTMLElement | null>(null);

// Keep the latest move visible: pin the move list to the bottom whenever
// the move count changes.
watch(
  () => gameState.value.moveHistory.length,
  () => {
    nextTick(() => {
      const el = moveListEl.value;
      if (el) el.scrollTop = el.scrollHeight;
    });
  },
);

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

// --- Bot move scheduling ---

function scheduleBotMove(): void {
  if (!playingVsBot.value || !currentServer || !aiPlayer) return;
  if (isGameOver.value) return;
  if (gameState.value.currentTurn === localColor.value) return; // Human's turn

  aiThinking.value = true;
  // Let the UI update before the bot starts computing.
  setTimeout(() => {
    if (!aiPlayer || !currentServer || isGameOver.value) {
      aiThinking.value = false;
      return;
    }
    const move = aiPlayer.getMove(gameState.value);
    aiThinking.value = false;
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

function handlePlayBot(): void {
  if (!PRESET_WEIGHTS) {
    lobbyError.value = 'No bot weights available. See training/README.md.';
    return;
  }
  isConnecting.value = true;
  lobbyError.value = null;

  // 400 sims at play time. The training run's eval cadence uses 100,
  // and self-play uses 100, but those are tuned for throughput; play
  // strength scales meaningfully with deeper search. The aiThinking
  // spinner covers the extra latency.
  aiPlayer = AIPlayer.create(PRESET_WEIGHTS, 400);
  playingVsBot.value = true;
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

      // If playing vs Bot and it's bot's turn, schedule its move
      if (playingVsBot.value) {
        scheduleBotMove();
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
  playingVsBot.value = false;
  aiThinking.value = false;
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
      :has-bot-model="PRESET_WEIGHTS !== null"
      @host="handleHost"
      @join="handleJoin"
      @start="handleLobbyStart"
      @cancel="handleLobbyCancel"
      @play-bot="handlePlayBot"
    />

    <!-- Game UI (visible when game started) -->
    <div v-if="gameStarted" class="game-area">
      <div class="game-layout">
        <!-- Left sidebar: move history -->
        <div class="sidebar">
          <div class="sidebar-header">
            <h2 class="sidebar-title">Moves</h2>
          </div>
          <div class="move-list" ref="moveListEl">
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

          <!-- Opponent chip: shown above the board, since the board is
               flipped to put the local player on the bottom edge. -->
          <div class="player-chip" :class="{ 'is-active': isOpponentTurn && !isGameOver }">
            <span class="chip-swatch" :class="opponentColor"></span>
            <span class="chip-name">{{ opponentName }}</span>
            <span class="chip-color">{{ opponentColor === 'white' ? 'White' : 'Black' }}</span>
            <span v-if="isOpponentTurn && !isGameOver" class="chip-turn-dot"></span>
          </div>

          <ChessBoard
            :game-state="gameState"
            :local-player-id="localPlayerId"
            @move="handleMove"
          />

          <!-- Local player chip: always below the board. -->
          <div class="player-chip is-local" :class="{ 'is-active': isLocalTurn && !isGameOver }">
            <span class="chip-swatch" :class="localColor"></span>
            <span class="chip-name">{{ localName }}</span>
            <span class="chip-color">{{ localColor === 'white' ? 'White' : 'Black' }}</span>
            <span v-if="isLocalTurn && !isGameOver" class="chip-turn-dot"></span>
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
              <span class="info-value">{{ playingVsBot ? 'vs Bot' : networkRole === 'host' ? 'Host' : networkRole === 'client' ? 'Client' : 'Local' }}</span>
            </div>
            <div v-if="!playingVsBot && roomCode" class="info-row">
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
  /* Inherit the layered ambient gradient from index.html — no opaque
     fill here so the body background shows through. */
  background: transparent;
  color: #e2e8f0;
}

.game-area {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: safe center;
  justify-content: safe center;
  /* No scrollbars: the board sizing in ChessBoard.vue is height-aware,
     and the mobile breakpoint hides nice-to-have panels so everything
     fits in the viewport. */
  overflow: hidden;
}

.game-layout {
  display: flex;
  gap: 24px;
  align-items: stretch;
  padding: 20px;
  max-height: 100dvh;
  box-sizing: border-box;
}

@media (max-width: 900px) {
  .game-layout {
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 8px;
    max-width: 100dvw;
    box-sizing: border-box;
  }
}

.sidebar {
  width: 220px;
  /* Glassmorphic surface: blurred backdrop + faint accent border + soft
     drop shadow so the panels read as floating cards over the gradient. */
  background: linear-gradient(165deg, rgba(40, 38, 70, 0.55), rgba(20, 19, 38, 0.7));
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  backdrop-filter: blur(14px) saturate(1.3);
  -webkit-backdrop-filter: blur(14px) saturate(1.3);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.05),
    0 8px 24px rgba(0, 0, 0, 0.35);
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

@media (max-width: 900px) {
  /* Mobile: drop both sidebars entirely so the board, status bar, and
     controls fit on a single screen with no scroll. */
  .sidebar {
    display: none;
  }
  .board-area {
    order: 1;
  }
}

.sidebar-header {
  /* Subtle accent gradient with a 1px gold-ish underline. */
  background: linear-gradient(90deg, rgba(99, 102, 241, 0.18), rgba(94, 234, 212, 0.10));
  border-bottom: 1px solid rgba(212, 175, 95, 0.25);
  padding: 12px 16px;
}

.sidebar-title {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 11px;
  font-weight: 600;
  color: #cbd5e1;
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.move-list {
  padding: 10px 6px 10px 14px;
  flex: 1 1 0;
  min-height: 0;
  /* `scroll` (not `auto`) reserves the gutter even when content fits,
     so the layout never shifts when a new move pushes it past the
     overflow boundary. */
  overflow-y: scroll;
  /* Firefox scrollbar styling. */
  scrollbar-width: thin;
  scrollbar-color: rgba(99, 102, 241, 0.65) rgba(255, 255, 255, 0.04);
}
/* WebKit / Blink: a slim glassy track with an indigo→teal gradient thumb
   that matches the rest of the UI accent palette. */
.move-list::-webkit-scrollbar {
  width: 8px;
}
.move-list::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 4px;
  margin: 6px 0;
}
.move-list::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.75), rgba(94, 234, 212, 0.6));
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 0 6px rgba(99, 102, 241, 0.35);
}
.move-list::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(124, 127, 245, 0.95), rgba(110, 240, 229, 0.85));
}

.move-line {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12.5px;
  color: #d1d5db;
  padding: 5px 4px;
  border-radius: 4px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  letter-spacing: 0.3px;
}

.move-line:nth-child(odd) {
  background: rgba(255, 255, 255, 0.015);
}

.move-line:last-child {
  background: rgba(99, 102, 241, 0.18);
  color: #f1f5f9;
  border-bottom: none;
  font-weight: 500;
}

.no-moves {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12.5px;
  color: #64748b;
  font-style: italic;
  padding: 4px 0;
}

.board-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

@media (max-width: 900px) {
  .board-area {
    gap: 6px;
  }
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 24px;
  background: linear-gradient(165deg, rgba(40, 38, 70, 0.55), rgba(20, 19, 38, 0.7));
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  backdrop-filter: blur(14px) saturate(1.3);
  -webkit-backdrop-filter: blur(14px) saturate(1.3);
  min-width: min(440px, 92dvw);
  justify-content: center;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.05),
    0 8px 24px rgba(0, 0, 0, 0.3);
}

@media (max-width: 900px) {
  .status-bar {
    padding: 6px 12px;
    gap: 10px;
  }
}

.status-text {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 15px;
  font-weight: 500;
  color: #94a3b8;
  letter-spacing: 0.3px;
}

.status-text.my-turn {
  color: #5ae3d8;
  font-weight: 600;
  text-shadow: 0 0 14px rgba(94, 234, 212, 0.5);
}

.status-text.game-over {
  color: #f7c058;
  font-weight: 600;
  text-shadow: 0 0 14px rgba(247, 192, 88, 0.5);
}

.status-text.in-check {
  color: #ff7777;
  font-weight: 600;
  text-shadow: 0 0 14px rgba(255, 80, 80, 0.55);
  animation: check-text-pulse 1.2s ease-in-out infinite;
}

@keyframes check-text-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.turn-indicator {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  color: #64748b;
  letter-spacing: 0.5px;
  padding-left: 16px;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
}

.draw-offer-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 18px;
  background: linear-gradient(90deg, rgba(247, 192, 88, 0.15), rgba(247, 192, 88, 0.06));
  border: 1px solid rgba(247, 192, 88, 0.4);
  border-radius: 10px;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: #f7c058;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  box-shadow: 0 0 24px rgba(247, 192, 88, 0.2);
}

.draw-btn {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 12px;
  font-weight: 500;
  padding: 5px 14px;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.draw-btn.accept {
  background: linear-gradient(165deg, #5ae3d8, #34c4b8);
  color: #0f0d1f;
  font-weight: 600;
}

.draw-btn.accept:hover {
  background: linear-gradient(165deg, #6ef0e5, #46d4c8);
  box-shadow: 0 0 16px rgba(94, 234, 212, 0.5);
}

.draw-btn.decline {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.18);
  color: #cbd5e1;
}

.draw-btn.decline:hover {
  background: rgba(255, 255, 255, 0.14);
  border-color: rgba(255, 255, 255, 0.3);
}

/* Player chips above + below the board: leave no doubt who is which
   color, and highlight whose turn it is with an accent border + a
   pulsing dot. The chip on top is always the opponent because the
   board is flipped for the local player. */
.player-chip {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 14px;
  background: linear-gradient(165deg, rgba(40, 38, 70, 0.55), rgba(20, 19, 38, 0.7));
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  backdrop-filter: blur(10px) saturate(1.3);
  -webkit-backdrop-filter: blur(10px) saturate(1.3);
  font-family: 'Inter', system-ui, sans-serif;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.player-chip.is-active {
  border-color: rgba(94, 234, 212, 0.65);
  box-shadow: 0 0 0 1px rgba(94, 234, 212, 0.4),
              0 0 18px rgba(94, 234, 212, 0.3);
}

.chip-swatch {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}

.chip-swatch.white {
  background: linear-gradient(155deg, #ffffff, #d4d4d4);
  border: 1.5px solid rgba(0, 0, 0, 0.15);
}

.chip-swatch.black {
  background: linear-gradient(155deg, #2c2c2c, #0a0a0a);
  border: 1.5px solid rgba(255, 255, 255, 0.2);
}

.chip-name {
  font-size: 13px;
  font-weight: 600;
  color: #f1f5f9;
  letter-spacing: 0.3px;
}

.chip-color {
  font-size: 11px;
  font-weight: 500;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  padding-left: 8px;
  border-left: 1px solid rgba(255, 255, 255, 0.12);
}

.chip-turn-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #5ae3d8;
  box-shadow: 0 0 10px rgba(94, 234, 212, 0.85);
  animation: turn-dot-pulse 1.4s ease-in-out infinite;
  margin-left: 2px;
}

@keyframes turn-dot-pulse {
  0%, 100% { opacity: 0.5; transform: scale(0.85); }
  50%      { opacity: 1;   transform: scale(1.15); }
}

@media (max-width: 900px) {
  .player-chip {
    padding: 4px 12px;
    gap: 8px;
  }
  .chip-name {
    font-size: 12px;
  }
  .chip-color {
    font-size: 10px;
    padding-left: 6px;
  }
}

.game-controls {
  display: flex;
  gap: 12px;
}

.control-btn {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.3px;
  padding: 9px 22px;
  background: linear-gradient(165deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  color: #cbd5e1;
  cursor: pointer;
  transition: transform 0.15s ease, border-color 0.15s ease,
              background 0.15s ease, box-shadow 0.15s ease, color 0.15s ease;
}

.control-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.04));
  border-color: rgba(99, 102, 241, 0.6);
  color: #f1f5f9;
  box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.4),
              0 0 18px rgba(99, 102, 241, 0.3);
  transform: translateY(-1px);
}

.control-btn:active:not(:disabled) {
  transform: translateY(0);
}

.control-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.resign-btn {
  border-color: rgba(255, 80, 80, 0.35);
  color: #ff7c7c;
}

.resign-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, rgba(255, 60, 60, 0.18), rgba(255, 60, 60, 0.06)) !important;
  border-color: rgba(255, 100, 100, 0.7) !important;
  color: #ff9999 !important;
  box-shadow: 0 0 0 1px rgba(255, 80, 80, 0.5),
              0 0 18px rgba(255, 80, 80, 0.35) !important;
}

.lobby-btn {
  background: linear-gradient(165deg, #6366f1, #4f46e5);
  border-color: rgba(99, 102, 241, 0.7);
  color: white;
  font-weight: 600;
}

.lobby-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, #7c7ff5, #5d56e8) !important;
  border-color: rgba(124, 127, 245, 0.9) !important;
  box-shadow: 0 0 0 1px rgba(124, 127, 245, 0.5),
              0 0 24px rgba(99, 102, 241, 0.5) !important;
}

.game-info-content {
  padding: 12px 16px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.info-row:last-child {
  border-bottom: none;
}

.info-label {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 11px;
  font-weight: 500;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.info-value {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12.5px;
  color: #e2e8f0;
  font-weight: 500;
}

.empty-bg {
  width: 100%;
  height: 100%;
  background: transparent;
}

</style>
