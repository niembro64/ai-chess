<script setup lang="ts">
import { ref, computed } from 'vue';
import type { PlayerId, PieceColor, Position, Move, ChessGameState, PieceType } from '@/types/chess';
import { playerIdToColor } from '@/types/chess';
import { getLegalMovesForSquare, getPieceSymbol } from '@/game/chess/ChessEngine';

const props = defineProps<{
  gameState: ChessGameState;
  localPlayerId: PlayerId;
}>();

const emit = defineEmits<{
  (e: 'move', move: Move): void;
}>();

const selectedSquare = ref<Position | null>(null);
const legalTargets = ref<Position[]>([]);
const promotionPending = ref<{ from: Position; to: Position } | null>(null);

const localColor = computed<PieceColor>(() => playerIdToColor(props.localPlayerId));
const isFlipped = computed(() => localColor.value === 'black');
const isMyTurn = computed(() => props.gameState.currentTurn === localColor.value);
const isGameOver = computed(() =>
  props.gameState.status === 'checkmate' ||
  props.gameState.status === 'stalemate' ||
  props.gameState.status === 'draw'
);

// Get ranks and files in display order (flipped for black)
const displayRanks = computed(() => {
  const ranks = [0, 1, 2, 3, 4, 5, 6, 7];
  return isFlipped.value ? [...ranks].reverse() : ranks;
});

const displayFiles = computed(() => {
  const files = [0, 1, 2, 3, 4, 5, 6, 7];
  return isFlipped.value ? [...files].reverse() : files;
});

function getSquareColor(rank: number, file: number): string {
  return (rank + file) % 2 === 0 ? 'light' : 'dark';
}

function isSelected(rank: number, file: number): boolean {
  return selectedSquare.value?.rank === rank && selectedSquare.value?.file === file;
}

function isLegalTarget(rank: number, file: number): boolean {
  return legalTargets.value.some(t => t.rank === rank && t.file === file);
}

function isLastMoveSquare(rank: number, file: number): boolean {
  const lm = props.gameState.lastMove;
  if (!lm) return false;
  return (lm.from.rank === rank && lm.from.file === file) ||
         (lm.to.rank === rank && lm.to.file === file);
}

function isKingInCheck(rank: number, file: number): boolean {
  if (props.gameState.status !== 'check' && props.gameState.status !== 'checkmate') return false;
  const piece = props.gameState.board[rank][file];
  return piece?.type === 'king' && piece.color === props.gameState.currentTurn;
}

function handleSquareClick(rank: number, file: number): void {
  if (isGameOver.value) return;
  if (promotionPending.value) return;

  const piece = props.gameState.board[rank][file];

  // If we already have a selection and this is a legal target, make the move
  if (selectedSquare.value && isLegalTarget(rank, file)) {
    const from = selectedSquare.value;
    const to = { rank, file };

    // Check if this is a pawn promotion
    const movingPiece = props.gameState.board[from.rank][from.file];
    const promoRank = localColor.value === 'white' ? 0 : 7;
    if (movingPiece?.type === 'pawn' && to.rank === promoRank) {
      promotionPending.value = { from, to };
      selectedSquare.value = null;
      legalTargets.value = [];
      return;
    }

    emit('move', { from, to });
    selectedSquare.value = null;
    legalTargets.value = [];
    return;
  }

  // If clicking own piece, select it
  if (piece && piece.color === localColor.value && isMyTurn.value) {
    selectedSquare.value = { rank, file };
    const moves = getLegalMovesForSquare(props.gameState, { rank, file });
    legalTargets.value = moves.map(m => m.to);
  } else {
    selectedSquare.value = null;
    legalTargets.value = [];
  }
}

function handlePromotion(pieceType: PieceType): void {
  if (!promotionPending.value) return;
  emit('move', {
    from: promotionPending.value.from,
    to: promotionPending.value.to,
    promotion: pieceType,
  });
  promotionPending.value = null;
}

function cancelPromotion(): void {
  promotionPending.value = null;
}

function fileLabel(file: number): string {
  return String.fromCharCode(97 + file);
}

function rankLabel(rank: number): string {
  return String(8 - rank);
}
</script>

<template>
  <div class="chess-board-wrapper">
    <div class="board-container">
      <!-- Rank labels (left side) -->
      <div class="rank-labels">
        <div v-for="rank in displayRanks" :key="'rl' + rank" class="rank-label">
          {{ rankLabel(rank) }}
        </div>
      </div>

      <div class="board-and-files">
        <!-- The board -->
        <div class="chess-board">
          <div v-for="rank in displayRanks" :key="'r' + rank" class="board-row">
            <div
              v-for="file in displayFiles"
              :key="'s' + rank + '-' + file"
              class="square"
              :class="[
                getSquareColor(rank, file),
                {
                  selected: isSelected(rank, file),
                  'legal-target': isLegalTarget(rank, file),
                  'last-move': isLastMoveSquare(rank, file),
                  'king-check': isKingInCheck(rank, file),
                },
              ]"
              @click="handleSquareClick(rank, file)"
            >
              <span v-if="gameState.board[rank][file]" class="piece">
                {{ getPieceSymbol(gameState.board[rank][file]!) }}
              </span>
              <span v-if="isLegalTarget(rank, file) && !gameState.board[rank][file]" class="move-dot"></span>
              <span v-if="isLegalTarget(rank, file) && gameState.board[rank][file]" class="capture-ring"></span>
            </div>
          </div>
        </div>

        <!-- File labels (bottom) -->
        <div class="file-labels">
          <div v-for="file in displayFiles" :key="'fl' + file" class="file-label">
            {{ fileLabel(file) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Promotion dialog -->
    <div v-if="promotionPending" class="promotion-overlay" @click="cancelPromotion">
      <div class="promotion-dialog" @click.stop>
        <h3>Promote pawn to:</h3>
        <div class="promotion-choices">
          <button
            v-for="pt in (['queen', 'rook', 'bishop', 'knight'] as PieceType[])"
            :key="pt"
            class="promotion-btn"
            @click="handlePromotion(pt)"
          >
            {{ getPieceSymbol({ color: localColor, type: pt }) }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chess-board-wrapper {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.board-container {
  display: flex;
  align-items: stretch;
}

.rank-labels {
  display: flex;
  flex-direction: column;
  justify-content: stretch;
}

.rank-label {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  font-family: monospace;
  font-size: 13px;
  color: #888;
}

.board-and-files {
  display: flex;
  flex-direction: column;
}

.chess-board {
  border: 3px solid #4444aa;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 0 30px rgba(68, 68, 170, 0.3);
}

.board-row {
  display: flex;
}

.square {
  width: 72px;
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  position: relative;
  transition: background-color 0.1s ease;
}

.square.light {
  background: #e8d5b5;
}

.square.dark {
  background: #b58863;
}

.square.selected {
  background: rgba(255, 255, 100, 0.6) !important;
}

.square.last-move.light {
  background: #cdd26a;
}

.square.last-move.dark {
  background: #aaa23a;
}

.square.king-check {
  background: rgba(255, 50, 50, 0.6) !important;
}

.square:hover {
  filter: brightness(1.1);
}

.piece {
  font-size: 48px;
  line-height: 1;
  pointer-events: none;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

.move-dot {
  position: absolute;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.25);
  pointer-events: none;
}

.capture-ring {
  position: absolute;
  width: 64px;
  height: 64px;
  border-radius: 50%;
  border: 5px solid rgba(0, 0, 0, 0.25);
  pointer-events: none;
}

.file-labels {
  display: flex;
}

.file-label {
  width: 72px;
  text-align: center;
  font-family: monospace;
  font-size: 13px;
  color: #888;
  padding-top: 4px;
}

/* Promotion dialog */
.promotion-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 5000;
}

.promotion-dialog {
  background: rgba(20, 20, 35, 0.98);
  border: 2px solid #4444aa;
  border-radius: 12px;
  padding: 20px 30px;
  text-align: center;
  box-shadow: 0 0 40px rgba(68, 68, 170, 0.4);
}

.promotion-dialog h3 {
  font-family: monospace;
  color: #ccc;
  margin: 0 0 16px 0;
  font-size: 16px;
}

.promotion-choices {
  display: flex;
  gap: 8px;
}

.promotion-btn {
  width: 64px;
  height: 64px;
  font-size: 42px;
  background: rgba(60, 60, 60, 0.8);
  border: 2px solid #666;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.promotion-btn:hover {
  background: rgba(68, 68, 170, 0.4);
  border-color: #4444aa;
  transform: scale(1.1);
}
</style>
