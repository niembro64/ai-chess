<script setup lang="ts">
// Inline SVG chess piece. Fill comes from the CSS `color` in effect
// (currentColor); the contrasting outline from the `--piece-outline`
// custom property. See pieceIcons.ts for why this replaced the Unicode
// glyphs (iOS renders U+265F as emoji and the rest from a different
// font than desktop — the board looked chaotic on iPhone).
import type { PieceType } from '@/types/chess';
import { PIECE_ICONS } from './pieceIcons';

defineProps<{ type: PieceType }>();
</script>

<template>
  <svg viewBox="0 0 45 45" aria-hidden="true" v-html="PIECE_ICONS[type]"></svg>
</template>

<style scoped>
svg {
  display: block;
  width: 100%;
  height: 100%;
  overflow: visible;
}
/* v-html content carries no scope attribute, so style it via :deep. */
svg :deep(*) {
  fill: currentColor;
  stroke: var(--piece-outline, transparent);
  stroke-width: 1.1;
  stroke-linejoin: round;
}
</style>
