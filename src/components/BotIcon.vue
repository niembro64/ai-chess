<script setup lang="ts">
// One cartoon bot portrait. See botIcons.ts for the artwork and why
// each bot has two moods (model × goal).
import { BOT_ICONS, type BotIconName } from './botIcons';

defineProps<{ name: BotIconName }>();
</script>

<template>
  <svg viewBox="0 0 64 64" aria-hidden="true" v-html="BOT_ICONS[name]"></svg>
</template>

<style scoped>
svg {
  display: block;
  width: 100%;
  height: 100%;
  overflow: visible;
}
</style>
