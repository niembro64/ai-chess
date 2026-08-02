<script setup lang="ts">
// Toy Mind — what the Toy net saw, guessed, and chose, laid out to
// teach the architecture by its own structure:
//
//   INPUT            NETWORK OUTPUT                SEARCH
//   8x8x6 tensor  →  POLICY (64x64) + VALUE     →  every legal move,
//   (rotatable)      one grid, one gauge           ordered by visits
//
// The SEARCH list is colorized with the exact colormap of the policy
// grid (each entry's chip = its cell's color), and a leader line runs
// from the top-visited move to its cell in the matrix.

import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { ToyThought } from '@/game/ai/ToyPlayer';
import { TOY_CHANNEL_NAMES, TOY_NUM_PLANES } from '@/game/ai/ToyNet';

const props = defineProps<{ thought: ToyThought }>();

const bodyEl = ref<HTMLDivElement | null>(null);
const sceneEl = ref<HTMLDivElement | null>(null);
const gridCanvas = ref<HTMLCanvasElement | null>(null);
const moveListEl = ref<HTMLDivElement | null>(null);
const leader = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
const hover = ref<{ x: number; y: number; text: string } | null>(null);

// --- shared colormap: grid cells AND list chips ------------------------

function maxProb(): number {
  let m = 0;
  const p = props.thought.rawPolicy;
  for (let i = 0; i < p.length; i++) if (p[i] > m) m = p[i];
  return m || 1;
}

function cellColor(p: number, maxP: number, illegal: boolean): string {
  const t = Math.sqrt(Math.min(1, p / maxP)); // sqrt lifts the mid-range
  return illegal
    ? `rgba(248, 90, 90, ${0.3 + 0.7 * t})`
    : `rgba(94, 234, 212, ${0.25 + 0.75 * t})`;
}

// Chip background for a list entry = the exact color of its grid cell.
function chipStyle(index: number): Record<string, string> {
  const p = props.thought.rawPolicy[index];
  return { background: cellColor(p, maxProb(), false) };
}

// --- three.js input scene ----------------------------------------------

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let sphereGroup: THREE.Group | null = null;
let rafId = 0;

const SLAB_GAP = 1.5;

function initScene(): void {
  const el = sceneEl.value!;
  const w = el.clientWidth || 360;
  const h = el.clientHeight || 300;

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

const whiteMat = new THREE.MeshLambertMaterial({ color: 0xf6efde, transparent: true, opacity: 0.92 });
const blackMat = new THREE.MeshLambertMaterial({ color: 0x17131f, transparent: true, opacity: 0.92 });
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
        // +1 = the mover's piece; Toy usually plays black, so +1 maps
        // to a dark sphere and -1 (the human's white army) to a light
        // one. Positions are the honest rotated net frame.
        const isBlackPiece = (v > 0) === blackToMove;
        const mesh = new THREE.Mesh(sphereGeo, isBlackPiece ? blackMat : whiteMat);
        mesh.position.set(f - 3.5, ch * SLAB_GAP, r - 3.5);
        sphereGroup.add(mesh);
      }
    }
  }
}

// --- policy grid ---------------------------------------------------------

const GRID = 256;   // 4px per cell
const PAD = 18;
const CELL = GRID / 64;

function drawGrid(): void {
  const canvas = gridCanvas.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d')!;
  const size = GRID + PAD;
  canvas.width = size;
  canvas.height = size;
  ctx.fillStyle = 'rgba(10, 9, 20, 0.9)';
  ctx.fillRect(0, 0, size, size);

  const { rawPolicy, legalMask } = props.thought;
  const maxP = maxProb();

  for (let from = 0; from < 64; from++) {
    for (let to = 0; to < 64; to++) {
      const p = rawPolicy[from * 64 + to];
      if (p <= 1e-7) continue;
      const illegal = legalMask[from * 64 + to] === 0;
      ctx.fillStyle = cellColor(p, maxP, illegal);
      ctx.fillRect(PAD + to * CELL, from * CELL, Math.max(1.5, CELL - 0.5), Math.max(1.5, CELL - 0.5));
    }
  }

  // Rank-block separators.
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.beginPath();
  for (let i = 8; i < 64; i += 8) {
    ctx.moveTo(PAD + i * CELL, 0);
    ctx.lineTo(PAD + i * CELL, GRID);
    ctx.moveTo(PAD, i * CELL);
    ctx.lineTo(PAD + GRID, i * CELL);
  }
  ctx.stroke();

  ctx.fillStyle = '#64748b';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < 8; i++) {
    ctx.fillText(String(i + 1), PAD / 2, i * 8 * CELL + 4 * CELL + 3);
    ctx.fillText(String(i + 1), PAD + i * 8 * CELL + 4 * CELL, GRID + 12);
  }

  // Ring the top-visited move's cell (the leader line's target).
  const top = props.thought.moves[0];
  if (top) {
    const row = Math.floor(top.index / 64);
    const col = top.index % 64;
    ctx.strokeStyle = '#f7c058';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(PAD + col * CELL + CELL / 2, row * CELL + CELL / 2, 5.5, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// Translate a net-frame square index to a real board name ("g6").
function squareName(netIndex: number): string {
  let r = Math.floor(netIndex / 8);
  let f = netIndex % 8;
  if (props.thought.blackToMove) {
    r = 7 - r;
    f = 7 - f;
  }
  return String.fromCharCode(97 + f) + String(8 - r);
}

function onGridMove(e: MouseEvent): void {
  const canvas = gridCanvas.value!;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left - PAD;
  const y = e.clientY - rect.top;
  const to = Math.floor(x / CELL);
  const from = Math.floor(y / CELL);
  if (to < 0 || to > 63 || from < 0 || from > 63) {
    hover.value = null;
    return;
  }
  const idx = from * 64 + to;
  const p = props.thought.rawPolicy[idx];
  const illegal = props.thought.legalMask[idx] === 0;
  hover.value = {
    x: e.clientX - bodyEl.value!.getBoundingClientRect().left + 14,
    y: e.clientY - bodyEl.value!.getBoundingClientRect().top - 10,
    text: `${squareName(from)}→${squareName(to)} · ${(p * 100).toFixed(2)}%${illegal ? ' · illegal' : ''}`,
  };
}

// --- leader line: top list entry → its grid cell -------------------------

function updateLeader(): void {
  leader.value = null;
  const body = bodyEl.value;
  const canvas = gridCanvas.value;
  const list = moveListEl.value;
  const top = props.thought.moves[0];
  if (!body || !canvas || !list || !top) return;
  const first = list.querySelector<HTMLElement>('.tm-move');
  if (!first) return;

  const bodyRect = body.getBoundingClientRect();
  const cellRect = canvas.getBoundingClientRect();
  const fromRect = first.getBoundingClientRect();

  const row = Math.floor(top.index / 64);
  const col = top.index % 64;
  // Canvas is rendered 1:1 (width attribute == CSS width), so cell
  // coordinates map directly.
  const cx = cellRect.left - bodyRect.left + PAD + col * CELL + CELL / 2;
  const cy = cellRect.top - bodyRect.top + row * CELL + CELL / 2;
  const lx = fromRect.left - bodyRect.left - 4;
  const ly = fromRect.top - bodyRect.top + fromRect.height / 2;
  // Pull both endpoints back along the line so the arrow TIPS rest on
  // the cell's ring and the entry's outline instead of covering them.
  const dx = cx - lx;
  const dy = cy - ly;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  leader.value = {
    x1: lx + ux * 4,
    y1: ly + uy * 4,
    x2: cx - ux * 9,
    y2: cy - uy * 9,
  };
}

// --- lifecycle ------------------------------------------------------------

function refresh(): void {
  drawGrid();
  updateSpheres();
  nextTick(updateLeader);
}

watch(() => props.thought, refresh);

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  initScene();
  refresh();
  resizeObserver = new ResizeObserver(() => updateLeader());
  if (bodyEl.value) resizeObserver.observe(bodyEl.value);
});

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId);
  resizeObserver?.disconnect();
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
      <span class="tm-legend">
        <span class="lg lg-legal">■</span> probability on legal moves
        <span class="lg lg-illegal">■</span> wasted on illegal moves (masked away)
        <span class="lg lg-zero">■</span> ~zero
      </span>
    </div>

    <div ref="bodyEl" class="tm-body">
      <!-- ============ GAME STATE INPUT ============ -->
      <section class="tm-section">
        <h3 class="tm-section-title">Game State Input</h3>
        <div ref="sceneEl" class="tm-scene"></div>
        <div class="tm-caption">
          8×8×6 tensor — drag to rotate<br />
          Toy's view (rotated); its pieces are +1 (dark)
        </div>
      </section>

      <!-- ============ NETWORK OUTPUT ============ -->
      <section class="tm-section">
        <h3 class="tm-section-title">Network output</h3>
        <div class="tm-out">
          <div class="tm-out-policy">
            <canvas
              ref="gridCanvas"
              class="tm-grid"
              @mousemove="onGridMove"
              @mouseleave="hover = null"
            ></canvas>
            <div class="tm-caption">
              <span class="tm-sublabel">policy output head</span><br />
              64 from-squares × 64 to-squares · hover to decode
            </div>
          </div>
          <div class="tm-out-value">
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
            <div class="tm-caption"><span class="tm-sublabel">value output head</span></div>
          </div>
        </div>
      </section>

      <!-- ============ SEARCH ============ -->
      <section class="tm-section">
        <h3 class="tm-section-title">Search</h3>
        <div ref="moveListEl" class="tm-moves">
          <div
            v-for="(m, i) in thought.moves"
            :key="m.uci"
            class="tm-move"
            :class="{ chosen: m.uci === thought.chosen, top: i === 0 }"
          >
            <span class="tm-chip" :style="chipStyle(m.index)"></span>
            <span class="tm-move-uci">{{ m.uci }}</span>
            <span class="tm-move-share">{{ pct(m.share) }}</span>
          </div>
        </div>
        <div class="tm-caption">
          all {{ thought.moves.length }} legal moves by visit share ·
          chip = that move's color in the policy grid
        </div>
      </section>

      <!-- Leader line: top-visited move ↔ its policy cell, arrowheads
           pointing at both (the amber-outlined list entry and the
           amber-ringed grid cell). -->
      <svg v-if="leader" class="tm-leader" aria-hidden="true">
        <defs>
          <marker
            id="tm-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f7c058" />
          </marker>
        </defs>
        <line
          :x1="leader.x1" :y1="leader.y1" :x2="leader.x2" :y2="leader.y2"
          stroke="#f7c058" stroke-width="1.4" stroke-dasharray="5 4" opacity="0.85"
          marker-start="url(#tm-arrow)" marker-end="url(#tm-arrow)"
        />
      </svg>

      <div
        v-if="hover"
        class="tm-tooltip"
        :style="{ left: `${hover.x}px`, top: `${hover.y}px` }"
      >{{ hover.text }}</div>
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
  max-width: 1160px;
}

.tm-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px 16px;
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

.tm-legend {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: #94a3b8;
}

.lg { margin-left: 10px; margin-right: 3px; }
.lg-legal { color: #5ae3d8; }
.lg-illegal { color: #f87171; }
.lg-zero { color: #1c1a2c; text-shadow: 0 0 0 1px #444; }

.tm-body {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  padding: 14px 16px 14px;
  align-items: stretch;
  justify-content: center;
}

.tm-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 10px 14px 12px;
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.02);
}

.tm-section-title {
  margin: 0;
  align-self: flex-start;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 2.2px;
}

.tm-sublabel {
  color: #5ae3d8;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-right: 6px;
}

.tm-scene {
  width: 340px;
  height: 290px;
  border-radius: 10px;
  overflow: hidden;
  background: radial-gradient(ellipse at 50% 40%, rgba(60, 58, 110, 0.35), rgba(12, 11, 24, 0.6));
  cursor: grab;
}
.tm-scene:active { cursor: grabbing; }

.tm-out {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.tm-out-policy,
.tm-out-value {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.tm-grid {
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(10, 9, 20, 0.9);
  cursor: crosshair;
}

.tm-caption {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: #64748b;
  text-align: center;
  letter-spacing: 0.4px;
  line-height: 1.5;
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
  gap: 3px;
  min-width: 150px;
  max-height: 300px;
  overflow-y: auto;
  padding-right: 6px;
  scrollbar-width: thin;
  scrollbar-color: rgba(99, 102, 241, 0.65) rgba(255, 255, 255, 0.04);
}

.tm-move {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #d1d5db;
  padding: 2px 8px;
  border-radius: 5px;
  background: rgba(255, 255, 255, 0.03);
  flex-shrink: 0;
}

.tm-move.chosen {
  color: #5ae3d8;
  font-weight: 600;
}

/* Top-visited move: amber outline matching the ring around its cell in
   the policy grid — the two ends of the leader line dress alike. */
.tm-move.top {
  outline: 1.4px solid #f7c058;
  outline-offset: 1px;
}

.tm-chip {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  flex-shrink: 0;
}

.tm-move-uci { flex: 1; }

.tm-move-share {
  color: #94a3b8;
}

.tm-leader {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.tm-tooltip {
  position: absolute;
  pointer-events: none;
  background: rgba(12, 11, 24, 0.95);
  border: 1px solid rgba(94, 234, 212, 0.4);
  border-radius: 6px;
  padding: 4px 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #e2e8f0;
  white-space: nowrap;
  z-index: 5;
}
</style>
