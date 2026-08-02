<script setup lang="ts">
// BoardGrid — the POLICY HEAD renderer as a reusable component.
//
// Draws a rows×cols grid of 8×8 boards on one canvas: dark separator
// lines between boards, per-column headers along the bottom and
// per-row headers along the left (same font/color as the policy
// head's a-h / 1-8 coordinates), cells drawn at base geometry × an
// integer scale k so the canvas displays 1:1 crisp at its natural
// size.
//
//   POLICY HEAD:  rows=8 cols=8  (from-square grid of destination boards)
//   GAME STATE:   1×6 desktop / 6×1 mobile (one board per piece plane)
//
// The parent supplies DISPLAY-ORDERED cell colors — any board
// flipping / net-frame rotation happens upstream. index =
// (boardRow*cols + boardCol)*mini² + cellR*mini + cellF.

import { onMounted, ref, watch } from 'vue';

const props = withDefaults(
  defineProps<{
    colors: string[];             // rows*cols*mini² cell fills, display order
    rows?: number;                // boards down
    cols?: number;                // boards across
    mini?: number;                // cells per board side
    k?: number;                   // integer draw scale (1 inline, 2 modal)
    cellPx?: number;              // base px per cell
    gap?: number;                 // base px between boards
    colLabels?: string[] | null;  // one per board column, drawn beneath
    rowLabels?: string[] | null;  // one per board row, drawn at left
    ring?: number | null;         // display-order cell index to ring amber
    checker?: boolean;            // checkerboard the boards
  }>(),
  {
    rows: 8, cols: 8, mini: 8, k: 1, cellPx: 5, gap: 1,
    colLabels: null, rowLabels: null, ring: null, checker: false,
  },
);

const emit = defineEmits<{
  (e: 'cellClick', index: number): void;
  (e: 'cellMove', index: number | null, clientX: number, clientY: number): void;
}>();

const canvasEl = ref<HTMLCanvasElement | null>(null);

const boardPx = () => props.mini * props.cellPx;
const padL = () => (props.rowLabels ? 16 : 0);
const padB = () => (props.colLabels ? 14 : 0);
const gridW = () => props.cols * boardPx() + (props.cols - 1) * props.gap;
const gridH = () => props.rows * boardPx() + (props.rows - 1) * props.gap;
const baseW = () => padL() + gridW() + (props.rowLabels ? 2 : 0);
const baseH = () => gridH() + padB();

// Base-coordinate pixel center of a cell — for the parent's leader
// lines (scale by rect.width / baseW() to get on-screen coords).
function cellCenter(index: number): { x: number; y: number } {
  const m = props.mini;
  const board = Math.floor(index / (m * m));
  const cell = index % (m * m);
  const bR = Math.floor(board / props.cols);
  const bC = board % props.cols;
  const cR = Math.floor(cell / m);
  const cF = cell % m;
  return {
    x: padL() + bC * (boardPx() + props.gap) + cF * props.cellPx + props.cellPx / 2,
    y: bR * (boardPx() + props.gap) + cR * props.cellPx + props.cellPx / 2,
  };
}

function draw(): void {
  const canvas = canvasEl.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d')!;
  const k = props.k;
  canvas.width = baseW() * k;
  canvas.height = baseH() * k;
  ctx.scale(k, k);
  ctx.fillStyle = 'rgba(10, 9, 20, 0.9)';
  ctx.fillRect(0, 0, baseW(), baseH());

  if (props.checker) {
    for (let r = 0; r < props.rows; r++) {
      for (let c = 0; c < props.cols; c++) {
        ctx.fillStyle = (r + c) % 2 === 0
          ? 'rgba(139, 143, 217, 0.10)'
          : 'rgba(139, 143, 217, 0.035)';
        ctx.fillRect(
          padL() + c * (boardPx() + props.gap),
          r * (boardPx() + props.gap),
          boardPx(),
          boardPx(),
        );
      }
    }
  }

  const m = props.mini;
  for (let i = 0; i < props.colors.length; i++) {
    const board = Math.floor(i / (m * m));
    const cell = i % (m * m);
    const bR = Math.floor(board / props.cols);
    const bC = board % props.cols;
    const cR = Math.floor(cell / m);
    const cF = cell % m;
    ctx.fillStyle = props.colors[i];
    ctx.fillRect(
      padL() + bC * (boardPx() + props.gap) + cF * props.cellPx,
      bR * (boardPx() + props.gap) + cR * props.cellPx,
      props.cellPx - 0.5,
      props.cellPx - 0.5,
    );
  }

  // Headers in the policy head's coordinate style.
  if (props.colLabels || props.rowLabels) {
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    if (props.colLabels) {
      for (let c = 0; c < props.cols; c++) {
        ctx.fillText(
          props.colLabels[c] ?? '',
          padL() + c * (boardPx() + props.gap) + boardPx() / 2,
          gridH() + 11,
        );
      }
    }
    if (props.rowLabels) {
      for (let r = 0; r < props.rows; r++) {
        ctx.fillText(
          props.rowLabels[r] ?? '',
          padL() / 2 - 1,
          r * (boardPx() + props.gap) + boardPx() / 2 + 3,
        );
      }
    }
  }

  if (props.ring !== null && props.ring !== undefined) {
    const c = cellCenter(props.ring);
    ctx.strokeStyle = '#f7c058';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// Mouse/touch position → display-order cell index, or null outside
// the cells (headers, gaps beyond the boards).
function decode(e: MouseEvent): number | null {
  const canvas = canvasEl.value;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const s = baseW() / (rect.width || 1);
  const x = (e.clientX - rect.left) * s - padL();
  const y = (e.clientY - rect.top) * s;
  const bC = Math.floor(x / (boardPx() + props.gap));
  const bR = Math.floor(y / (boardPx() + props.gap));
  const cF = Math.floor((x - bC * (boardPx() + props.gap)) / props.cellPx);
  const cR = Math.floor((y - bR * (boardPx() + props.gap)) / props.cellPx);
  if (
    bC < 0 || bC >= props.cols || bR < 0 || bR >= props.rows ||
    cF < 0 || cF >= props.mini || cR < 0 || cR >= props.mini
  ) {
    return null;
  }
  return (bR * props.cols + bC) * props.mini * props.mini + cR * props.mini + cF;
}

function onClick(e: MouseEvent): void {
  const idx = decode(e);
  if (idx !== null) emit('cellClick', idx);
}

function onMove(e: MouseEvent): void {
  emit('cellMove', decode(e), e.clientX, e.clientY);
}

function onLeave(): void {
  emit('cellMove', null, 0, 0);
}

onMounted(draw);
watch(
  () => [
    props.colors, props.rows, props.cols, props.k, props.gap,
    props.colLabels, props.rowLabels, props.ring,
  ],
  draw,
);

defineExpose({ canvasEl, cellCenter, baseW });
</script>

<template>
  <canvas
    ref="canvasEl"
    class="board-grid"
    @click="onClick"
    @mousemove="onMove"
    @mouseleave="onLeave"
  ></canvas>
</template>

<style scoped>
.board-grid {
  /* No rounding: radius clips the corner cells. */
  border-radius: 0;
  display: block;
}
</style>
