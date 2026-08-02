<script setup lang="ts">
// BoardGrid — the POLICY HEAD's visual language as a reusable canvas.
//
// An 8×8 board of squares, each holding a mini×mini block of colored
// cells, drawn at base geometry × an integer scale k (ctx.scale) so
// every instance stays crisp. mini=8 renders the 4096-cell policy
// head; mini=1 renders a plain 8×8 board (the GAME STATE planes).
//
// The parent supplies DISPLAY-ORDERED cell colors — any board
// flipping / net-frame rotation happens upstream. This component is
// pure geometry and paint: index = (outerR*8+outerF)*mini² +
// subR*mini + subF, top-left origin.

import { onMounted, ref, watch } from 'vue';

const props = withDefaults(
  defineProps<{
    colors: string[];            // (8*mini)² cell fills, display order
    mini?: number;               // sub-cells per board square
    k?: number;                  // integer draw scale (1 inline, 2 modal)
    cellPx?: number;             // base px per sub-cell
    fileLabels?: string[] | null; // 8 bottom labels — enables bottom pad
    rankLabels?: string[] | null; // 8 left labels — enables left pad
    ring?: number | null;        // display-order cell index to ring amber
    checker?: boolean;           // checkerboard the outer squares
  }>(),
  { mini: 1, k: 1, cellPx: 5, fileLabels: null, rankLabels: null, ring: null, checker: false },
);

const emit = defineEmits<{
  (e: 'cellClick', index: number): void;
  (e: 'cellMove', index: number | null, clientX: number, clientY: number): void;
}>();

const canvasEl = ref<HTMLCanvasElement | null>(null);

const GAP = 1;
const outerPx = () => props.mini * props.cellPx;
const padL = () => (props.rankLabels ? 16 : 0);
const padB = () => (props.fileLabels ? 14 : 0);
const boardPx = () => 8 * outerPx() + 7 * GAP;
const baseW = () => padL() + boardPx() + (props.rankLabels ? 2 : 0);
const baseH = () => boardPx() + padB();

// Base-coordinate pixel center of a cell — for the parent's leader
// lines (scale by rect.width / baseW() to get on-screen coords).
function cellCenter(index: number): { x: number; y: number } {
  const m = props.mini;
  const outer = Math.floor(index / (m * m));
  const sub = index % (m * m);
  const oR = Math.floor(outer / 8);
  const oF = outer % 8;
  const sR = Math.floor(sub / m);
  const sF = sub % m;
  return {
    x: padL() + oF * (outerPx() + GAP) + sF * props.cellPx + props.cellPx / 2,
    y: oR * (outerPx() + GAP) + sR * props.cellPx + props.cellPx / 2,
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
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        ctx.fillStyle = (r + f) % 2 === 0
          ? 'rgba(139, 143, 217, 0.10)'
          : 'rgba(139, 143, 217, 0.035)';
        ctx.fillRect(padL() + f * (outerPx() + GAP), r * (outerPx() + GAP), outerPx(), outerPx());
      }
    }
  }

  const m = props.mini;
  for (let i = 0; i < props.colors.length; i++) {
    const outer = Math.floor(i / (m * m));
    const sub = i % (m * m);
    const oR = Math.floor(outer / 8);
    const oF = outer % 8;
    const sR = Math.floor(sub / m);
    const sF = sub % m;
    ctx.fillStyle = props.colors[i];
    ctx.fillRect(
      padL() + oF * (outerPx() + GAP) + sF * props.cellPx,
      oR * (outerPx() + GAP) + sR * props.cellPx,
      props.cellPx - 0.5,
      props.cellPx - 0.5,
    );
  }

  if (props.fileLabels || props.rankLabels) {
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i < 8; i++) {
      if (props.fileLabels) {
        ctx.fillText(props.fileLabels[i], padL() + i * (outerPx() + GAP) + outerPx() / 2, boardPx() + 11);
      }
      if (props.rankLabels) {
        ctx.fillText(props.rankLabels[i], padL() / 2 - 1, i * (outerPx() + GAP) + outerPx() / 2 + 3);
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
// the cells (labels, gaps beyond the board).
function decode(e: MouseEvent): number | null {
  const canvas = canvasEl.value;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const s = baseW() / (rect.width || 1);
  const x = (e.clientX - rect.left) * s - padL();
  const y = (e.clientY - rect.top) * s;
  const oF = Math.floor(x / (outerPx() + GAP));
  const oR = Math.floor(y / (outerPx() + GAP));
  const sF = Math.floor((x - oF * (outerPx() + GAP)) / props.cellPx);
  const sR = Math.floor((y - oR * (outerPx() + GAP)) / props.cellPx);
  if (oF < 0 || oF > 7 || oR < 0 || oR > 7 || sF < 0 || sF >= props.mini || sR < 0 || sR >= props.mini) {
    return null;
  }
  return (oR * 8 + oF) * props.mini * props.mini + sR * props.mini + sF;
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
watch(() => [props.colors, props.ring, props.fileLabels, props.rankLabels, props.k], draw);

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
