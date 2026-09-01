<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue';
import type { PlayerId, Move, ChessGameState, PieceColor } from '@/types/chess';
import { playerIdToColor } from '@/types/chess';
import type { NetworkGameSnapshot, LobbyPlayer, NetworkRole } from '@/types/network';
import type { GameConnection } from '@/types/game';
import { applyMove, createInitialGameState, posToAlgebraic } from '@/game/chess/ChessEngine';
import { networkManager } from '@/game/network/NetworkManager';
import { ChessServer } from '@/game/server/ChessServer';
import { LocalGameConnection } from '@/game/server/LocalGameConnection';
import { RemoteGameConnection } from '@/game/server/RemoteGameConnection';
import { AIPlayer } from '@/game/ai/AIPlayer';
import { ToyPlayer, type ToyThought } from '@/game/ai/ToyPlayer';
import { MODELS, fetchModelJson, type ModelId } from '@/game/ai/models';
import type { SerializedWeights } from '@/game/ai/ChessNet';
import LobbyModal from './LobbyModal.vue';
import ChessBoard from './ChessBoard.vue';
import { defineAsyncComponent } from 'vue';

// Lazy-load the Toy Mind panel's chunk only when someone actually
// plays Toy, so Sage games pay nothing for it.
const ToyMindPanel = defineAsyncComponent(() => import('./ToyMindPanel.vue'));

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

// --- Move-history navigation ---
//
// `viewPly` selects which position the BOARD displays: null = live game,
// N = the position after N plies (0 = starting position). Past positions
// are reconstructed by replaying moveHistory from the initial position —
// the game itself keeps running underneath (the bot still moves; new
// moves append while you browse).
const viewPly = ref<number | null>(null);
const totalPlies = computed(() => gameState.value.moveHistory.length);
const isViewingHistory = computed(
  () => viewPly.value !== null && viewPly.value < totalPlies.value,
);
// The ply currently shown on the board (live = latest).
const shownPly = computed(() => viewPly.value ?? totalPlies.value);

const displayState = computed<ChessGameState>(() => {
  if (viewPly.value === null || viewPly.value >= totalPlies.value) {
    return gameState.value;
  }
  let s = createInitialGameState();
  for (let i = 0; i < viewPly.value; i++) {
    s = applyMove(s, gameState.value.moveHistory[i]);
  }
  return s;
});

function historyStart(): void {
  if (totalPlies.value > 0) viewPly.value = 0;
}
function historyBack(): void {
  const cur = shownPly.value;
  if (cur > 0) viewPly.value = cur - 1;
}
function historyForward(): void {
  if (viewPly.value === null) return;
  const next = viewPly.value + 1;
  viewPly.value = next >= totalPlies.value ? null : next;
}
function historyLive(): void {
  viewPly.value = null;
}
function historyJumpTo(ply: number): void {
  viewPly.value = ply >= totalPlies.value ? null : ply;
}

function handleHistoryKeys(e: KeyboardEvent): void {
  if (!gameStarted.value || showLobby.value) return;
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    historyBack();
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    historyForward();
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    historyStart();
  } else if (e.key === 'ArrowDown' || e.key === 'Escape') {
    e.preventDefault();
    historyLive();
  }
}
// --- Screen wake lock ---
//
// Keep the phone screen from dimming/sleeping while a game is open:
// waiting through the bot's think time means long stretches with no
// touch input, which otherwise trips the OS screen timeout mid-game.
// Screen Wake Lock API — secure contexts (https/localhost) only;
// supported on Android Chrome/Edge and iOS 16.4+. Silently a no-op
// where unsupported or denied (battery saver): the game still works,
// the screen just dims as usual.
let wakeLock: WakeLockSentinel | null = null;

async function requestWakeLock(): Promise<void> {
  if (!('wakeLock' in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request('screen');
  } catch {
    wakeLock = null;
  }
}

function releaseWakeLock(): void {
  wakeLock?.release().catch(() => {});
  wakeLock = null;
}

// The OS auto-releases the lock whenever the tab is hidden; re-acquire
// when the player comes back to an in-progress game.
function handleVisibilityChange(): void {
  if (document.visibilityState === 'visible' && gameStarted.value) {
    requestWakeLock();
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleHistoryKeys);
  document.addEventListener('visibilitychange', handleVisibilityChange);
});
onUnmounted(() => {
  window.removeEventListener('keydown', handleHistoryKeys);
  document.removeEventListener('visibilitychange', handleVisibilityChange);
  releaseWakeLock();
});

// Server & connection
let currentServer: ChessServer | null = null;
let activeConnection: GameConnection | null = null;

// AI opponent. Weights are fetched lazily from public/models/ when a
// bot is picked in the lobby (see models.ts) — Sage and Toy are
// separate downloads and only the chosen one loads.
let aiPlayer: AIPlayer | ToyPlayer | null = null;
const playingVsBot = ref(false);
const aiThinking = ref(false);
const botModelId = ref<ModelId | null>(null);
// Toy's latest thought record, rendered by the Toy Mind panel.
const toyThought = ref<ToyThought | null>(null);
// Both bots emit thoughts now — the panel shows for any bot game.
const showToyPanel = computed(() => botModelId.value !== null && toyThought.value !== null);

// Piece tinting: Sage plays green, Jester purple (light shade as
// white, dark as black — see ChessBoard.vue). Toy keeps standard sets.
const botTheme = computed<'sage' | 'jester' | null>(() =>
  playingVsBot.value && (botModelId.value === 'sage' || botModelId.value === 'jester')
    ? botModelId.value
    : null,
);
const botColor = computed<PieceColor | null>(() =>
  playingVsBot.value ? (localPlayerId.value === 1 ? 'black' : 'white') : null,
);

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
        return aiThinking.value ? 'AI is thinking' : "AI's turn";
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
      switch (gs.drawReason) {
        case 'repetition': return 'Draw - threefold repetition';
        case 'fifty-move': return 'Draw - 50-move rule';
        case 'insufficient-material': return 'Draw - insufficient material';
        case 'agreement': return 'Draw - by agreement';
        default: return 'Draw';
      }
    default:
      return '';
  }
});

const turnIndicator = computed(() => {
  if (isGameOver.value) return '';
  const side = gameState.value.currentTurn === 'white' ? 'White to move' : 'Black to move';
  if (!playingVsBot.value) return side;
  // Label WHOSE turn it actually is, not just "there's a bot somewhere".
  return side + (gameState.value.currentTurn === localColor.value ? ' (you)' : ' (bot)');
});

// Player chips: opponent shown above the board, local player below.
// Color reflects the piece color each side is actually playing, regardless
// of board orientation — the chip on top of the screen is the side you
// look at across the board, which is always the opponent.
const opponentColor = computed<PieceColor>(() => localColor.value === 'white' ? 'black' : 'white');
const opponentName = computed(() => {
  if (playingVsBot.value) return botModelId.value ? MODELS[botModelId.value].name : 'Bot';
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

// Move list rows: one per full move, each half-move individually
// clickable to jump the board to the position AFTER that move.
const moveHistoryDisplay = computed(() => {
  const moves = gameState.value.moveHistory;
  const rows: { num: number; halves: { label: string; ply: number }[] }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    const halves = [{ label: formatMove(moves[i]), ply: i + 1 }];
    if (i + 1 < moves.length) {
      halves.push({ label: formatMove(moves[i + 1]), ply: i + 2 });
    }
    rows.push({ num: Math.floor(i / 2) + 1, halves });
  }
  return rows;
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
  if (isGameOver.value) {
    // Game ended on the bot's turn (it got checkmated / stalemated):
    // no move to make, but the mind panel still shows the net's view
    // of the terminal position — with every move invalid.
    if (gameState.value.currentTurn !== localColor.value) {
      aiPlayer.observeTerminal(gameState.value);
    }
    return;
  }
  if (gameState.value.currentTurn === localColor.value) return; // Human's turn

  aiThinking.value = true;
  // Let the UI update before the bot starts computing.
  setTimeout(async () => {
    if (!aiPlayer || !currentServer || isGameOver.value) {
      aiThinking.value = false;
      return;
    }
    const move = await aiPlayer.getMove(gameState.value);
    aiThinking.value = false;
    if (!currentServer || isGameOver.value) return;
    // The bot is whichever seat the human isn't in (Toy plays white).
    const botPlayerId: PlayerId = localPlayerId.value === 1 ? 2 : 1;
    currentServer.receiveCommand({ type: 'move', move }, botPlayerId);
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

async function handlePlayBot(
  model: ModelId,
  opts: { goalInverted: boolean; temperature: number },
): Promise<void> {
  isConnecting.value = true;
  lobbyError.value = null;

  let weights: unknown;
  try {
    weights = await fetchModelJson(model);
  } catch (err) {
    lobbyError.value = (err as Error).message || `Failed to load ${MODELS[model].name}`;
    isConnecting.value = false;
    return;
  }

  try {
    if (model === 'toy') {
      aiPlayer = ToyPlayer.create(
        weights, MODELS.toy.sims,
        t => { toyThought.value = t; },
        { goalInverted: opts.goalInverted, temperature: opts.temperature },
      );
    } else {
      aiPlayer = AIPlayer.create(
        weights as SerializedWeights,
        MODELS[model].sims,
        t => { toyThought.value = t; },
        {
          trainedGoal: MODELS[model].trainedGoal,
          goalInverted: opts.goalInverted,
          temperature: opts.temperature,
        },
      );
    }
  } catch (err) {
    lobbyError.value = (err as Error).message || 'Model weights are invalid';
    isConnecting.value = false;
    return;
  }

  botModelId.value = model;
  toyThought.value = null;
  playingVsBot.value = true;
  networkRole.value = null;
  // Toy plays WHITE and moves first, so the Toy Mind panel fills with
  // its first thought immediately — no waiting for the human's move.
  // Sage games keep the human on white.
  localPlayerId.value = model === 'toy' ? 2 : 1;
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
  viewPly.value = null;
  requestWakeLock();

  if (networkRole.value !== 'client') {
    currentServer = new ChessServer();
    const localConnection = new LocalGameConnection(currentServer, localPlayerId.value);
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
  if (isViewingHistory.value) return; // board is frozen; belt-and-suspenders
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
  botModelId.value = null;
  toyThought.value = null;
  aiThinking.value = false;
  lobbyPlayers.value = [];
  roomCode.value = '';
  isHost.value = false;
  gameState.value = createInitialGameState();
  drawOffer.value = null;
  viewPly.value = null;
  releaseWakeLock();
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
      @host="handleHost"
      @join="handleJoin"
      @start="handleLobbyStart"
      @cancel="handleLobbyCancel"
      @play-bot="handlePlayBot"
    />

    <!-- Game UI (visible when game started) -->
    <div
      v-if="gameStarted"
      class="game-area"
      :class="{ 'has-panel': showToyPanel, 'toy-mode': botModelId !== null }"
    >
      <div class="game-stack">
      <div class="game-layout">
        <!-- Left sidebar: move history -->
        <div class="sidebar">
          <div class="sidebar-header">
            <h2 class="sidebar-title">Moves</h2>
          </div>
          <div class="move-list" ref="moveListEl">
            <div
              v-for="row in moveHistoryDisplay"
              :key="row.num"
              class="move-line"
            >
              <span class="move-num">{{ row.num }}.</span>
              <button
                v-for="half in row.halves"
                :key="half.ply"
                class="mv"
                :class="{ current: shownPly === half.ply }"
                @click="historyJumpTo(half.ply)"
              >
                {{ half.label }}
              </button>
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
            <!-- Wave-dot thinking animation, here in the status bar:
                 bot games hide the player chips (toy-mode), so the
                 chip-mounted dots never show there. -->
            <span
              v-if="aiThinking"
              class="thinking-dots"
              aria-label="AI is thinking"
            ><span></span><span></span><span></span></span>
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
            <!-- Wave-dot "thinking" indicator on the opponent chip when
                 it's their turn — separate semantic from the local chip's
                 turn-dot (which means "your move, go ahead"). -->
            <span
              v-if="isOpponentTurn && !isGameOver"
              class="thinking-dots"
              role="status"
              aria-label="opponent is thinking"
            >
              <span></span><span></span><span></span>
            </span>
          </div>

          <ChessBoard
            class="board-slot"
            :class="{ 'history-view': isViewingHistory }"
            :game-state="displayState"
            :frozen="isViewingHistory"
            :local-player-id="localPlayerId"
            :bot-theme="botTheme"
            :bot-color="botColor"
            @move="handleMove"
          />

          <!-- Move-history navigation: step through past positions.
               Also bound to arrow keys (←/→ step, ↑ start, ↓/Esc live). -->
          <div class="history-nav">
            <button class="nav-btn" :disabled="shownPly === 0" @click="historyStart" title="Start position (↑)">«</button>
            <button class="nav-btn" :disabled="shownPly === 0" @click="historyBack" title="Previous move (←)">‹</button>
            <span class="nav-pos" :class="{ viewing: isViewingHistory }">
              {{ !isViewingHistory ? 'live' : shownPly === 0 ? 'start' : `move ${shownPly} / ${totalPlies}` }}
            </span>
            <button class="nav-btn" :disabled="!isViewingHistory" @click="historyForward" title="Next move (→)">›</button>
            <button class="nav-btn" :disabled="!isViewingHistory" @click="historyLive" title="Back to live (↓)">»</button>
          </div>

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
              <span class="info-value">{{ playingVsBot ? `vs ${opponentName}` : networkRole === 'host' ? 'Host' : networkRole === 'client' ? 'Client' : 'Local' }}</span>
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

      <!-- The bot's mind: what the net saw, thought, and chose — same
           panel for Sage and Toy (Sage's GAME STATE is the 6-plane
           view of its position; policy/value shapes are shared). -->
      <ToyMindPanel
        v-if="showToyPanel"
        :thought="toyThought!"
        :flipped="localColor === 'black'"
        :bot-name="botModelId ? MODELS[botModelId].name : 'Bot'"
        class="toy-mind-row"
      />
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

.game-stack {
  display: flex;
  flex-direction: column;
  align-items: center;
  max-height: 100%;
}

/* When the Toy Mind panel is open the page is a study tool — allow
   vertical scroll so board + panel can both be taller than the fold. */
.game-area.has-panel {
  overflow-y: auto;
  align-items: safe flex-start;
}

.toy-mind-row {
  margin: 0 20px 24px;
  /* Never let the flex column squeeze the panel — the page scrolls
     instead (game-area.has-panel enables overflow-y). */
  flex-shrink: 0;
  max-width: calc(100vw - 16px);
}

/* Visual-bot mode sheds furniture the mode doesn't need, at EVERY
   width: player chips and Offer Draw / Resign (the game-over Return
   to Lobby button survives). Priority goes to the three visuals + the
   board, mirroring the mobile philosophy. */
.game-area.toy-mode .player-chip {
  display: none;
}
.game-area.toy-mode .game-controls .control-btn:not(.lobby-btn) {
  display: none;
}

/* Desktop Visual-bot: left/right split — Toy Mind panel on the left,
   board column on the right, sidebars gone. (The mobile block below
   handles the top/bottom variant.) */
@media (min-width: 901px) {
  .game-area.toy-mode .game-stack {
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 18px;
  }
  .game-area.toy-mode .toy-mind-row {
    order: -1;
    margin: 0 0 0 12px;
  }
  .game-area.toy-mode .sidebar {
    display: none;
  }
  .game-area.has-panel {
    align-items: safe center;
  }
  /* The board's size formula budgets ~488px for the sidebars, which
     toy-mode hides — rebudget for the panel instead so panel + board
     fit side by side down to ~1150px-wide screens. */
  .game-area.toy-mode :deep(.square) {
    --sq: min(72px, calc((100dvw - 810px) / 8), calc((100dvh - 250px) / 8));
  }
}

/* --- Mobile: thumbs-first layout ----------------------------------
   The chessboard is the only thing the player touches constantly, so
   it pins to the BOTTOM of the screen (thumb zone) at full width. All
   the info furniture — status, chips, controls, and the Toy Mind
   panel — compresses into whatever is left at the top; the panel gets
   a capped height and scrolls internally. Implemented with flex
   `order` so the desktop DOM/layout is untouched. */
@media (max-width: 900px) {
  .game-stack {
    min-height: 100dvh;
    width: 100%;
  }
  .game-layout {
    flex: 1 1 auto;
    width: 100%;
    min-height: 0;
  }
  .board-area {
    flex: 1 1 auto;
    min-height: 0;
    width: 100%;
  }
  /* Info at the top, in reading order — but the status bar starts the
     bottom cluster: margin-top:auto pushes status + arrows + board
     together against the bottom edge, right where the thumbs are. */
  .status-bar { order: 1; padding: 4px 10px; margin-top: auto; }
  .draw-offer-banner { order: 2; }
  .player-chip { order: 3; }
  .player-chip.is-local { order: 4; }
  .game-controls { order: 5; }
  .control-btn { padding: 6px 14px; font-size: 12px; }
  /* ...the history arrows last of the info stack (they're interactive,
     so closest to the thumbs)... */
  .history-nav { order: 6; }
  /* ...and the board pinned to the bottom edge. */
  .board-slot {
    order: 10;
    padding-bottom: max(4px, env(safe-area-inset-bottom));
  }
  /* Toy Mind rides above everything and GROWS: it takes all the
     vertical space left above the bottom cluster (status bar, arrows,
     board), and its three-column strip (GAME STATE | POLICY | SEARCH)
     contains its visuals inside that allotted height. No page scroll. */
  .toy-mind-row {
    order: -1;
    margin: 4px 4px 0;
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    /* The stack centers children (shrink-to-fit); the panel must span
       the full width regardless of its contents' intrinsic size. */
    align-self: stretch;
  }
  /* The panel owns the leftover height, so the board cluster stops
     growing and sits content-sized against the bottom edge. */
  .game-area.has-panel .game-layout {
    flex: 0 0 auto;
  }
  .game-area.has-panel {
    overflow: hidden;
  }
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
  display: flex;
  align-items: center;
  gap: 4px;
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12.5px;
  color: #d1d5db;
  padding: 3px 4px;
  border-radius: 4px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  letter-spacing: 0.3px;
}

.move-line:nth-child(odd) {
  background: rgba(255, 255, 255, 0.015);
}

.move-num {
  color: #64748b;
  min-width: 26px;
}

/* Individual half-moves are buttons: click one to jump the board to
   the position after that move. */
.mv {
  font: inherit;
  letter-spacing: inherit;
  color: inherit;
  background: transparent;
  border: none;
  border-radius: 4px;
  padding: 2px 6px;
  cursor: pointer;
  transition: background 0.12s ease, color 0.12s ease;
}

.mv:hover {
  background: rgba(99, 102, 241, 0.25);
  color: #f1f5f9;
}

.mv.current {
  background: rgba(99, 102, 241, 0.35);
  color: #f1f5f9;
  font-weight: 600;
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

/* Three small teal dots that wave in sequence — the canonical "AI is
   thinking" affordance. Used on the opponent chip during their turn,
   distinct from the local chip's turn-dot which says "your move." */
.thinking-dots {
  display: inline-flex;
  align-items: flex-end;
  gap: 3px;
  margin-left: 4px;
  height: 8px;
}
.thinking-dots > span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #5ae3d8;
  box-shadow: 0 0 6px rgba(94, 234, 212, 0.7);
  animation: thinking-wave 1.3s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
.thinking-dots > span:nth-child(2) { animation-delay: 0.15s; }
.thinking-dots > span:nth-child(3) { animation-delay: 0.3s; }

@keyframes thinking-wave {
  0%, 70%, 100% {
    transform: translateY(0);
    opacity: 0.35;
  }
  35% {
    transform: translateY(-4px);
    opacity: 1;
  }
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

/* --- Move-history navigation ------------------------------------- */

/* Mute the board while a past position is shown: desaturate + dim so
   there's no mistaking history for the live game. */
.board-slot {
  transition: filter 0.25s ease;
}
.board-slot.history-view {
  filter: saturate(0.35) brightness(0.78);
}

.history-nav {
  display: flex;
  align-items: center;
  gap: 6px;
}

.nav-btn {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 16px;
  line-height: 1;
  width: 34px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(165deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 7px;
  color: #cbd5e1;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
  padding: 0 0 2px;
}

.nav-btn:hover:not(:disabled) {
  background: linear-gradient(165deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.04));
  border-color: rgba(99, 102, 241, 0.6);
  color: #f1f5f9;
}

.nav-btn:disabled {
  opacity: 0.35;
  cursor: default;
}

.nav-pos {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.6px;
  color: #64748b;
  min-width: 96px;
  text-align: center;
  text-transform: uppercase;
}

/* Amber "you are in the past" accent, matching the muted board. */
.nav-pos.viewing {
  color: #f7c058;
  text-shadow: 0 0 10px rgba(247, 192, 88, 0.4);
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
