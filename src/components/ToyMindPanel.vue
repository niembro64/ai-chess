<script setup lang="ts">
// Toy Mind — visualizes exactly what the Toy net saw, guessed, and chose.
//
// Left: the 8x8x6 input tensor as a rotatable three.js stack — six grid
// slabs (one per piece-type channel), spheres where the tensor is
// nonzero. Sphere COLOR follows the real game color (your pieces are
// white spheres); sphere POSITION is the honest net frame, which is
// rotated 180° because Toy plays black. That mismatch is the lesson.
//
// Middle: two 64x64 canvases — the net's raw softmax (with probability
// it wasted on illegal moves tinted red) and the distribution after
// MCTS. Watching search sharpen the fuzzy prior is the point.
//
// Right: scalar value gauge + top moves by visit share.

import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { ToyThought } from '@/game/ai/ToyPlayer';
import { TOY_CHANNEL_NAMES, TOY_NUM_PLANES } from '@/game/ai/ToyNet';

const props = defineProps<{ thought: ToyThought }>();

const sceneEl = ref<HTMLDivElement | null>(null);
const rawCanvas = ref<HTMLCanvasElement | null>(null);
const searchCanvas = ref<HTMLCanvasElement | null>(null);

// --- three.js scene ----------------------------------------------------

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let sphereGroup: THREE.Group | null = null;
let rafId = 0;

const SLAB_GAP = 1.5;

function initScene(): void {
  const el = sceneEl.value!;
  const w = el.clientWidth || 380;
  const h = el.clientHeight || 320;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.setSize(w, h);
  el.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);
  camera.position.set(11, 8.5, 13);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 6;
  controls.maxDistance = 40;
  controls.target.set(0, (TOY_NUM_PLANES - 1) * SLAB_GAP * 0.5, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(6, 12, 8);
  scene.add(dir);

  // Six channel slabs: faint 8x8 grids, one per piece type, stacked in Y.
  for (let ch = 0; ch < TOY_NUM_PLANES; ch++) {
    const grid = new THREE.GridHelper(8, 8, 0x8b8fd9, 0x4c4f86);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.4;
    grid.position.y = ch * SLAB_GAP;
    scene.add(grid);

    const label = makeLabelSprite(TOY_CHANNEL_NAMES[ch]);
    label.position.set(-5.4, ch * SLAB_GAP, 0);
    scene.add(label);
  }

  sphereGroup = new THREE.Group();
  scene.add(sphereGroup);

  const animate = () => {
    rafId = requestAnimationFrame(animate);
    controls!.update();
    renderer!.render(scene!, camera!);
  };
  animate();
}

function makeLabelSprite(text: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 48;
  const ctx = canvas.getContext('2d')!;
  ctx.font = '600 26px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#cbd5e1';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 64, 24);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
  sprite.scale.set(2.4, 0.9, 1);
  return sprite;
}

const whiteMat = new THREE.MeshLambertMaterial({
  color: 0xf6efde, transparent: true, opacity: 0.92,
});
const blackMat = new THREE.MeshLambertMaterial({
  color: 0x17131f, transparent: true, opacity: 0.92,
});
const sphereGeo = new THREE.SphereGeometry(0.34, 20, 16);

function updateSpheres(): void {
  if (!sphereGroup) return;
  sphereGroup.clear();
  const { planes, blackToMove } = props.thought;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      for (let ch = 0; ch < TOY_NUM_PLANES; ch++) {
        const v = planes[(r * 8 + f) * TOY_NUM_PLANES + ch];
        if (v === 0) continue;
        // +1 = the mover's piece. Toy is the mover when this panel
        // updates; Toy plays black in bot games — so +1 maps to a dark
        // sphere and -1 (the human's white army) to a light one. When
        // Toy somehow plays white, the mapping flips with it.
        const moverIsBlack = blackToMove;
        const isBlackPiece = (v > 0) === moverIsBlack;
        const mesh = new THREE.Mesh(sphereGeo, isBlackPiece ? blackMat : whiteMat);
        mesh.position.set(f - 3.5, ch * SLAB_GAP, r - 3.5);
        sphereGroup.add(mesh);
      }
    }
  }
}

// --- 2D policy canvases --------------------------------------------------

const GRID = 256;   // 4px per cell
const PAD = 18;

function drawPolicy(
  canvas: HTMLCanvasElement,
  probs: Float32Array,
  legalMask: Uint8Array | null,
): void {
  const ctx = canvas.getContext('2d')!;
  const size = GRID + PAD;
  canvas.width = size;
  canvas.height = size;
  ctx.fillStyle = 'rgba(10, 9, 20, 0.9)';
  ctx.fillRect(0, 0, size, size);

  // Normalize brightness to the strongest cell so the grid stays
  // readable whether the distribution is a fuzzy prior (max ~1%) or a
  // sharp post-search spike (max ~90%) — we're showing SHAPE here;
  // absolute magnitudes live in the visits list.
  let maxP = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > maxP) maxP = probs[i];
  if (maxP <= 0) maxP = 1;

  const cell = GRID / 64;
  for (let from = 0; from < 64; from++) {
    for (let to = 0; to < 64; to++) {
      const p = probs[from * 64 + to];
      if (p <= 1e-7) continue;
      const t = Math.sqrt(p / maxP); // sqrt lifts the mid-range
      const illegal = legalMask !== null && legalMask[from * 64 + to] === 0;
      ctx.fillStyle = illegal
        ? `rgba(248, 90, 90, ${0.3 + 0.7 * t})`
        : `rgba(94, 234, 212, ${0.25 + 0.75 * t})`;
      ctx.fillRect(PAD + to * cell, from * cell, Math.max(1.5, cell - 0.5), Math.max(1.5, cell - 0.5));
    }
  }

  // Axis ticks: file-rank block boundaries every 8 squares.
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.beginPath();
  for (let i = 8; i < 64; i += 8) {
    ctx.moveTo(PAD + i * cell, 0);
    ctx.lineTo(PAD + i * cell, GRID);
    ctx.moveTo(PAD, i * cell);
    ctx.lineTo(PAD + GRID, i * cell);
  }
  ctx.stroke();

  ctx.fillStyle = '#64748b';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < 8; i++) {
    // Ranks of the FROM square down the left, TO blocks along the bottom.
    ctx.fillText(String(i + 1), PAD / 2, i * 8 * cell + 4 * cell + 3);
    ctx.fillText(String(i + 1), PAD + i * 8 * cell + 4 * cell, GRID + 12);
  }
}

function redraw(): void {
  if (rawCanvas.value) drawPolicy(rawCanvas.value, props.thought.rawPolicy, props.thought.legalMask);
  if (searchCanvas.value) drawPolicy(searchCanvas.value, props.thought.visitPolicy, null);
  updateSpheres();
}

watch(() => props.thought, redraw);

onMounted(() => {
  initScene();
  redraw();
});

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId);
  controls?.dispose();
  renderer?.dispose();
  sphereGeo.dispose();
  whiteMat.dispose();
  blackMat.dispose();
  if (renderer && sceneEl.value?.contains(renderer.domElement)) {
    sceneEl.value.removeChild(renderer.domElement);
  }
});

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}
</script>

<template>
  <div class="toy-mind">
    <div class="tm-header">
      <span class="tm-title">Toy Mind</span>
      <span class="tm-sub">
        what the net saw · rotated to Toy's view, its pieces are +1 (dark spheres)
      </span>
    </div>
    <div class="tm-body">
      <div class="tm-col tm-scene-col">
        <div ref="sceneEl" class="tm-scene"></div>
        <div class="tm-caption">input tensor 8×8×6 — drag to rotate</div>
      </div>

      <div class="tm-col">
        <canvas ref="rawCanvas" class="tm-grid"></canvas>
        <div class="tm-caption">
          raw policy (64 from × 64 to)
          <span class="tm-illegal">■ mass on illegal moves</span>
        </div>
      </div>

      <div class="tm-col">
        <canvas ref="searchCanvas" class="tm-grid"></canvas>
        <div class="tm-caption">after {{ 128 }}-sim search</div>
      </div>

      <div class="tm-col tm-stats">
        <div class="tm-value">
          <div class="tm-value-track">
            <div
              class="tm-value-fill"
              :class="{ neg: thought.value < 0 }"
              :style="{ height: `${Math.abs(thought.value) * 50}%` }"
            ></div>
            <div class="tm-value-zero"></div>
          </div>
          <div class="tm-value-num">
            {{ thought.value >= 0 ? '+' : '' }}{{ thought.value.toFixed(2) }}
          </div>
          <div class="tm-caption">value<br />(Toy's eval)</div>
        </div>
        <div class="tm-moves">
          <div
            v-for="m in thought.topMoves"
            :key="m.uci"
            class="tm-move"
            :class="{ chosen: m.uci === thought.chosen }"
          >
            <span class="tm-move-uci">{{ m.uci }}</span>
            <span class="tm-move-share">{{ pct(m.share) }}</span>
          </div>
          <div class="tm-caption">visits</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.toy-mind {
  background: linear-gradient(165deg, rgba(40, 38, 70, 0.55), rgba(20, 19, 38, 0.7));
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  backdrop-filter: blur(14px) saturate(1.3);
  -webkit-backdrop-filter: blur(14px) saturate(1.3);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 8px 24px rgba(0, 0, 0, 0.35);
  overflow: hidden;
  max-width: 1100px;
}

.tm-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  background: linear-gradient(90deg, rgba(45, 212, 191, 0.16), rgba(99, 102, 241, 0.10));
  border-bottom: 1px solid rgba(212, 175, 95, 0.25);
  padding: 10px 16px;
}

.tm-title {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 11px;
  font-weight: 600;
  color: #cbd5e1;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.tm-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: #64748b;
}

.tm-body {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  padding: 14px 16px 12px;
  align-items: flex-start;
  justify-content: center;
}

.tm-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.tm-scene {
  width: 360px;
  height: 300px;
  border-radius: 10px;
  overflow: hidden;
  background: radial-gradient(ellipse at 50% 40%, rgba(60, 58, 110, 0.35), rgba(12, 11, 24, 0.6));
  cursor: grab;
}
.tm-scene:active { cursor: grabbing; }

.tm-grid {
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(10, 9, 20, 0.9);
}

.tm-caption {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: #64748b;
  text-align: center;
  letter-spacing: 0.4px;
}

.tm-illegal {
  color: #f87171;
  margin-left: 6px;
}

.tm-stats {
  flex-direction: row;
  gap: 18px;
  align-items: flex-start;
}

.tm-value {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.tm-value-track {
  position: relative;
  width: 18px;
  height: 200px;
  border-radius: 9px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;
}

.tm-value-fill {
  position: absolute;
  bottom: 50%;
  left: 0;
  right: 0;
  background: linear-gradient(180deg, #5ae3d8, #34c4b8);
}

.tm-value-fill.neg {
  bottom: auto;
  top: 50%;
  background: linear-gradient(180deg, #f87171, #dc2626);
}

.tm-value-zero {
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 1px;
  background: rgba(255, 255, 255, 0.35);
}

.tm-value-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  color: #e2e8f0;
}

.tm-moves {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 118px;
}

.tm-move {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #d1d5db;
  padding: 3px 8px;
  border-radius: 5px;
  background: rgba(255, 255, 255, 0.03);
}

.tm-move.chosen {
  background: rgba(94, 234, 212, 0.18);
  color: #5ae3d8;
  font-weight: 600;
}

.tm-move-share {
  color: #94a3b8;
}
</style>
