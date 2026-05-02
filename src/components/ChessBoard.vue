<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';
import type { PlayerId, PieceColor, Position, Move, ChessGameState, PieceType } from '@/types/chess';
import { playerIdToColor } from '@/types/chess';
import { getLegalMovesForSquare, getFilledPieceGlyph } from '@/game/chess/ChessEngine';

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

// Smooth piece slide on every move (FLIP technique).
//
// After Vue commits the new gameState to the DOM, the moved piece is
// already at its destination square. We:
//   1. Find the destination square by data-square attribute.
//   2. Compute the visual offset back to its origin (in pixels).
//   3. Set a synchronous transform that places the piece visually back
//      at the origin square, with no transition.
//   4. Force a layout reflow so the browser commits that transform.
//   5. Animate transform → translate(0, 0) with ease-in-out.
//
// Works for captures (the captured piece disappears instantly when the
// state updates; the capturing piece slides on top), and for promotions
// (the new queen glyph slides; minor cosmetic that the pawn-glyph isn't
// shown during travel — acceptable). Castling animates the king; the
// rook teleports — fine for a first cut.
const boardRef = ref<HTMLElement | null>(null);

const SLIDE_MS = 280;
const SLIDE_EASING = 'cubic-bezier(0.4, 0, 0.2, 1)';

watch(
  () => props.gameState.lastMove,
  async (newMove, oldMove) => {
    if (!newMove) return;
    // Same logical move — skip (e.g. parent re-render with no real change).
    if (
      oldMove &&
      oldMove.from.rank === newMove.from.rank &&
      oldMove.from.file === newMove.from.file &&
      oldMove.to.rank === newMove.to.rank &&
      oldMove.to.file === newMove.to.file
    ) return;

    await nextTick();
    const board = boardRef.value;
    if (!board) return;

    const dest = board.querySelector<HTMLElement>(
      `[data-square="${newMove.to.rank}-${newMove.to.file}"]`,
    );
    const piece = dest?.querySelector<HTMLElement>('.piece');
    if (!dest || !piece) return;

    const sq = dest.offsetWidth;
    let dx = (newMove.from.file - newMove.to.file) * sq;
    let dy = (newMove.from.rank - newMove.to.rank) * sq;
    if (isFlipped.value) {
      dx = -dx;
      dy = -dy;
    }

    // FLIP: invert (place piece back at source) → play (animate to zero).
    piece.style.transition = 'none';
    piece.style.transform = `translate(${dx}px, ${dy}px)`;
    // Force a reflow so the inverted transform commits before the next paint.
    void piece.offsetHeight;
    piece.style.transition = `transform ${SLIDE_MS}ms ${SLIDE_EASING}`;
    piece.style.transform = 'translate(0, 0)';

    const cleanup = () => {
      piece.style.transition = '';
      piece.style.transform = '';
      piece.removeEventListener('transitionend', cleanup);
    };
    piece.addEventListener('transitionend', cleanup);
  },
);
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
        <div class="chess-board" ref="boardRef">
          <div v-for="rank in displayRanks" :key="'r' + rank" class="board-row">
            <div
              v-for="file in displayFiles"
              :key="'s' + rank + '-' + file"
              class="square"
              :data-square="`${rank}-${file}`"
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
              <span
                v-if="gameState.board[rank][file]"
                class="piece"
                :class="gameState.board[rank][file]!.color"
              >
                {{ getFilledPieceGlyph(gameState.board[rank][file]!.type) }}
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
            :class="localColor"
            @click="handlePromotion(pt)"
          >
            {{ getFilledPieceGlyph(pt) }}
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
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: #94a3b8;
  letter-spacing: 0.5px;
}

.board-and-files {
  display: flex;
  flex-direction: column;
}

.chess-board {
  /* Layered border: outer 3px purple frame, inner 1px gold inlay, plus
     a soft ambient glow and a heavy drop shadow to lift the board off
     the page. */
  border: 3px solid transparent;
  border-radius: 6px;
  overflow: hidden;
  background-clip: padding-box;
  box-shadow:
    0 0 0 1px rgba(212, 175, 95, 0.35),
    0 0 0 4px rgba(99, 102, 241, 0.85),
    0 0 40px rgba(99, 102, 241, 0.35),
    0 14px 36px rgba(0, 0, 0, 0.55);
}

.board-row {
  display: flex;
}

.square {
  /* Size by min of: design max, width budget, height budget. The width
     budget reserves space for sidebars + page padding (desktop sidebars
     take ~448px); the height budget reserves space for status bar +
     opponent chip + local chip + controls + gaps so the board never
     forces a scroll. The mobile overrides below relax the width budget
     once sidebars are hidden. */
  --sq: min(72px, calc((100dvw - 488px) / 8), calc((100dvh - 250px) / 8));
  width: var(--sq);
  height: var(--sq);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  position: relative;
  transition: background-color 0.1s ease;
}

@media (max-width: 900px) {
  .square {
    --sq: min(72px, calc((100dvw - 56px) / 8), calc((100dvh - 220px) / 8));
  }
}

.square.light {
  background: #ead7b5;
}

.square.dark {
  background: #b58863;
}

/* Selected: warm gold inset ring instead of a harsh yellow flood. The
   piece stays clearly visible. */
.square.selected {
  box-shadow: inset 0 0 0 4px rgba(247, 192, 88, 0.95),
              inset 0 0 16px rgba(247, 192, 88, 0.35);
}

/* Last-move tint: cool teal layered on top of the wood, instead of the
   muddy yellow-green that fought with the warm board palette. */
.square.last-move::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(94, 234, 212, 0.32);
  pointer-events: none;
}

.square.king-check {
  animation: king-check-pulse 1.2s ease-in-out infinite;
}

@keyframes king-check-pulse {
  0%, 100% {
    box-shadow:
      inset 0 0 0 3px rgba(255, 80, 80, 0.85),
      inset 0 0 24px rgba(255, 80, 80, 0.5);
  }
  50% {
    box-shadow:
      inset 0 0 0 4px rgba(255, 120, 120, 1),
      inset 0 0 36px rgba(255, 80, 80, 0.85);
  }
}

.square:hover::after {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.08);
  pointer-events: none;
}

.piece {
  /* Track the square's --sq variable: pieces should be ~2/3 of the square. */
  font-size: calc(var(--sq) * 0.66);
  line-height: 1;
  pointer-events: none;
  z-index: 1;
  /* Drop shadow gives pieces weight against the board. Outline (the
     contrasting edge) is done via text-shadow on the color variants below
     so it stays crisp regardless of board color. */
  filter: drop-shadow(0 3px 4px rgba(0, 0, 0, 0.5));
}

/* White pieces: cream fill, dark outline. Eight-direction text-shadow
   simulates a hard 1px stroke around the glyph. */
.piece.white {
  color: #f6efde;
  text-shadow:
    -1px -1px 0 #1a1410,
     1px -1px 0 #1a1410,
    -1px  1px 0 #1a1410,
     1px  1px 0 #1a1410,
     0   -1.2px 0 #1a1410,
     0    1.2px 0 #1a1410,
    -1.2px 0 0 #1a1410,
     1.2px 0 0 #1a1410;
}

/* Black pieces: deep charcoal fill with a thin warm-cream rim so they
   read clearly even on the dark squares. */
.piece.black {
  color: #15110d;
  text-shadow:
    -1px -1px 0 rgba(246, 239, 222, 0.55),
     1px -1px 0 rgba(246, 239, 222, 0.55),
    -1px  1px 0 rgba(246, 239, 222, 0.55),
     1px  1px 0 rgba(246, 239, 222, 0.55);
}

.move-dot {
  position: absolute;
  width: 28%;
  height: 28%;
  border-radius: 50%;
  background: rgba(94, 234, 212, 0.85);
  box-shadow: 0 0 12px rgba(94, 234, 212, 0.5);
  pointer-events: none;
}

.capture-ring {
  position: absolute;
  width: 88%;
  height: 88%;
  border-radius: 50%;
  border: 4px solid rgba(94, 234, 212, 0.85);
  box-shadow: 0 0 18px rgba(94, 234, 212, 0.45),
              inset 0 0 12px rgba(94, 234, 212, 0.35);
  pointer-events: none;
}

.file-labels {
  display: flex;
}

.file-label {
  flex: 1;
  text-align: center;
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  font-size: min(12px, 3dvw);
  font-weight: 500;
  color: #94a3b8;
  padding-top: 6px;
  letter-spacing: 0.5px;
}

.rank-label {
  font-size: min(12px, 3dvw) !important;
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
  background: linear-gradient(165deg, rgba(40, 38, 70, 0.95), rgba(22, 21, 42, 0.97));
  border: 1px solid rgba(99, 102, 241, 0.45);
  border-radius: 16px;
  padding: 24px 32px;
  text-align: center;
  backdrop-filter: blur(14px) saturate(1.4);
  -webkit-backdrop-filter: blur(14px) saturate(1.4);
  box-shadow:
    0 0 0 1px rgba(212, 175, 95, 0.2) inset,
    0 0 60px rgba(99, 102, 241, 0.45),
    0 18px 40px rgba(0, 0, 0, 0.55);
}

.promotion-dialog h3 {
  font-family: 'Inter', system-ui, sans-serif;
  color: #e2e8f0;
  margin: 0 0 18px 0;
  font-size: 15px;
  font-weight: 500;
  letter-spacing: 0.4px;
}

.promotion-choices {
  display: flex;
  gap: 10px;
}

.promotion-btn {
  width: 64px;
  height: 64px;
  font-size: 42px;
  background: linear-gradient(155deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 10px;
  cursor: pointer;
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.promotion-btn.white {
  color: #f6efde;
  text-shadow:
    -1px -1px 0 #1a1410,
     1px -1px 0 #1a1410,
    -1px  1px 0 #1a1410,
     1px  1px 0 #1a1410,
     0   -1.2px 0 #1a1410,
     0    1.2px 0 #1a1410,
    -1.2px 0 0 #1a1410,
     1.2px 0 0 #1a1410;
}

.promotion-btn.black {
  color: #15110d;
  text-shadow:
    -1px -1px 0 rgba(246, 239, 222, 0.6),
     1px -1px 0 rgba(246, 239, 222, 0.6),
    -1px  1px 0 rgba(246, 239, 222, 0.6),
     1px  1px 0 rgba(246, 239, 222, 0.6);
}

.promotion-btn:hover {
  border-color: rgba(99, 102, 241, 0.7);
  box-shadow:
    0 0 0 1px rgba(99, 102, 241, 0.5),
    0 0 24px rgba(99, 102, 241, 0.4);
  transform: scale(1.08);
}
</style>
