<script setup lang="ts">
import { ref } from 'vue';
import GameView from './components/GameView.vue';
import TrainingView from './components/TrainingView.vue';

const mode = ref<'play' | 'train'>('play');
</script>

<template>
  <GameView v-if="mode === 'play'" />
  <TrainingView v-else @back="mode = 'play'" />

  <!-- Mode switch (visible on the play screen) -->
  <button
    v-if="mode === 'play'"
    class="train-toggle"
    @click="mode = 'train'"
    title="AI Training"
  >
    Train AI
  </button>
</template>

<style>
#app {
  width: 100%;
  height: 100%;
}

* {
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
}

.train-toggle {
  position: fixed;
  bottom: 16px;
  right: 16px;
  z-index: 4000;
  font-family: monospace;
  font-size: 13px;
  padding: 8px 16px;
  background: rgba(20, 20, 35, 0.9);
  border: 1px solid #4444aa;
  border-radius: 6px;
  color: #aaa;
  cursor: pointer;
  transition: all 0.15s;
}

.train-toggle:hover {
  background: rgba(68, 68, 170, 0.3);
  border-color: #6666cc;
  color: white;
}
</style>
