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

// Every cell is hue-coded by legality (teal = legal, red = illegal)
// and BRIGHTNESS-coded by probability. Nothing renders black: the
// lowest-probability cells sit at a dark version of their hue, so the
// legal/illegal split stays readable across the whole board.
function cellColor(p: number, maxP: number, illegal: boolean): string {
  const t = Math.sqrt(Math.min(1, p / maxP)); // sqrt lifts the mid-range
  const a = 0.15 + 0.85 * t;                  // dark floor, never invisible
  return illegal
    ? `rgba(248, 90, 90, ${a})`
    : `rgba(94, 234, 212, ${a})`;
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

// --- policy grid: a board of boards --------------------------------------
//
// The 64x64 matrix rearranged so chess players can read it: a big 8x8
// board in REAL coordinates (files a-h, ranks 8..1 top-down, same
// orientation as the game board), where each big square holds a tiny
// 8x8 board of that square's DESTINATIONS. "What does Toy want to do
// with the g7 pawn?" = find g7, read its mini-board.

const MINI = 5;                    // px per destination cell
const OUTER = MINI * 8;            // 40px per from-square
const GAP = 1;
const PAD_L = 16;                  // rank labels
const PAD_B = 14;                  // file labels
const BOARD = 8 * OUTER + 7 * GAP;

// Net-frame square index -> real board coords {r, f} (engine layout:
// r 0 = rank 8 at top, f 0 = file a). The net rotates 180° for black.
function netToReal(netIndex: number): { r: number; f: number } {
  let r = Math.floor(netIndex / 8);
  let f = netIndex % 8;
  if (props.thought.blackToMove) {
    r = 7 - r;
    f = 7 - f;
  }
  return { r, f };
}

// Pixel center of a policy entry's mini-cell on the canvas.
function cellCenter(policyIndex: number): { x: number; y: number } {
  const from = netToReal(Math.floor(policyIndex / 64));
  const to = netToReal(policyIndex % 64);
  return {
    x: PAD_L + from.f * (OUTER + GAP) + to.f * MINI + MINI / 2,
    y: from.r * (OUTER + GAP) + to.r * MINI + MINI / 2,
  };
}

function drawGrid(): void {
  const canvas = gridCanvas.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d')!;
  canvas.width = PAD_L + BOARD + 2;
  canvas.height = BOARD + PAD_B;
  ctx.fillStyle = 'rgba(10, 9, 20, 0.9)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const { rawPolicy, legalMask } = props.thought;
  const maxP = maxProb();

  // Checkerboard tint + border for the big from-squares.
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const ox = PAD_L + f * (OUTER + GAP);
      const oy = r * (OUTER + GAP);
      ctx.fillStyle = (r + f) % 2 === 0
        ? 'rgba(139, 143, 217, 0.10)'   // light square
        : 'rgba(139, 143, 217, 0.035)'; // dark square
      ctx.fillRect(ox, oy, OUTER, OUTER);
    }
  }

  // Destination mini-cells, converted net frame -> real coords. Every
  // cell is drawn — legality picks the hue, probability the brightness.
  for (let i = 0; i < 4096; i++) {
    const p = rawPolicy[i];
    const from = netToReal(Math.floor(i / 64));
    const to = netToReal(i % 64);
    ctx.fillStyle = cellColor(p, maxP, legalMask[i] === 0);
    ctx.fillRect(
      PAD_L + from.f * (OUTER + GAP) + to.f * MINI,
      from.r * (OUTER + GAP) + to.r * MINI,
      MINI - 0.5,
      MINI - 0.5,
    );
  }

  // File / rank labels in real coordinates, like the game board.
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < 8; i++) {
    ctx.fillText(String.fromCharCode(97 + i), PAD_L + i * (OUTER + GAP) + OUTER / 2, BOARD + 11);
    ctx.fillText(String(8 - i), PAD_L / 2 - 1, i * (OUTER + GAP) + OUTER / 2 + 3);
  }

  // Ring the top-visited move's destination mini-cell (leader target).
  const top = props.thought.moves[0];
  if (top) {
    const c = cellCenter(top.index);
    ctx.strokeStyle = '#f7c058';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function onGridMove(e: MouseEvent): void {
  const canvas = gridCanvas.value!;
  const rect = canvas.getBoundingClientRect();
  // The canvas CSS-scales down on narrow screens; convert display px
  // back to canvas px before doing cell math.
  const scale = rect.width / canvas.width || 1;
  const x = (e.clientX - rect.left) / scale - PAD_L;
  const y = (e.clientY - rect.top) / scale;
  const fromF = Math.floor(x / (OUTER + GAP));
  const fromR = Math.floor(y / (OUTER + GAP));
  const toF = Math.floor((x - fromF * (OUTER + GAP)) / MINI);
  const toR = Math.floor((y - fromR * (OUTER + GAP)) / MINI);
  if (fromF < 0 || fromF > 7 || fromR < 0 || fromR > 7 || toF < 0 || toF > 7 || toR < 0 || toR > 7) {
    hover.value = null;
    return;
  }
  // Real display coords -> net-frame policy index.
  const black = props.thought.blackToMove;
  const nFrom = (black ? 7 - fromR : fromR) * 8 + (black ? 7 - fromF : fromF);
  const nTo = (black ? 7 - toR : toR) * 8 + (black ? 7 - toF : toF);
  const idx = nFrom * 64 + nTo;
  const p = props.thought.rawPolicy[idx];
  const illegal = props.thought.legalMask[idx] === 0;
  const name = (f: number, r: number) => String.fromCharCode(97 + f) + String(8 - r);
  hover.value = {
    x: e.clientX - bodyEl.value!.getBoundingClientRect().left + 14,
    y: e.clientY - bodyEl.value!.getBoundingClientRect().top - 10,
    text: `${name(fromF, fromR)}→${name(toF, toR)} · ${(p * 100).toFixed(2)}%${illegal ? ' · illegal' : ''}`,
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
  // Anchor the list end on the top entry's color CHIP — it wears the
  // same amber circle as the grid cell, so the line runs circle to
  // circle.
  const chip = list.querySelector<HTMLElement>('.tm-move .tm-chip');
  if (!chip) return;

  const bodyRect = body.getBoundingClientRect();
  const cellRect = canvas.getBoundingClientRect();
  const chipRect = chip.getBoundingClientRect();

  // Account for CSS downscaling on narrow screens: canvas-space cell
  // coordinates map to display space via the rendered/intrinsic ratio.
  const scale = cellRect.width / canvas.width || 1;
  const c = cellCenter(top.index);
  const cx = cellRect.left - bodyRect.left + c.x * scale;
  const cy = cellRect.top - bodyRect.top + c.y * scale;
  const lx = chipRect.left - bodyRect.left + chipRect.width / 2;
  const ly = chipRect.top - bodyRect.top + chipRect.height / 2;
  // Pull both endpoints back along the line so the arrow TIPS rest on
  // the two amber circles instead of covering them.
  const dx = cx - lx;
  const dy = cy - ly;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  leader.value = {
    x1: lx + ux * 12,
    y1: ly + uy * 12,
    x2: cx - ux * 10,
    y2: cy - uy * 10,
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
        <span class="lg lg-legal">■</span> legal moves
        <span class="lg lg-illegal">■</span> illegal moves
        <span class="lg lg-bright"></span> brightness shows output probability
      </span>
    </div>

    <div ref="bodyEl" class="tm-body">
      <!-- ============ NETWORK INPUT ============ -->
      <section class="tm-section">
        <h3 class="tm-section-title">Network Input</h3>
        <div class="tm-subtitle">game state</div>
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
            <div class="tm-subtitle">policy head</div>
            <canvas
              ref="gridCanvas"
              class="tm-grid"
              @mousemove="onGridMove"
              @mouseleave="hover = null"
            ></canvas>
            <div class="tm-caption">
              each square holds a mini-board of its destinations · hover to decode
            </div>
          </div>
          <div class="tm-out-value">
            <div class="tm-subtitle">value head</div>
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
/* Dark-to-bright ramp: the brightness axis of the colormap. */
.lg-bright {
  display: inline-block;
  width: 26px;
  height: 9px;
  border-radius: 2px;
  vertical-align: baseline;
  background: linear-gradient(90deg, rgba(94, 234, 212, 0.15), rgba(94, 234, 212, 1));
  border: 1px solid rgba(255, 255, 255, 0.12);
}

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

/* Subtitle line ABOVE each visual (game state / policy head / value head). */
.tm-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  font-weight: 700;
  color: #5ae3d8;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}

.tm-scene {
  width: min(340px, calc(100vw - 90px));
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
  /* Scale down on narrow screens; hover/leader math converts via the
     rendered/intrinsic ratio. */
  max-width: calc(100vw - 90px);
  height: auto;
}

@media (max-width: 900px) {
  .tm-body {
    padding: 10px 8px 12px;
    gap: 12px;
  }
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

/* Top-visited move: its color CHIP wears the same amber circle as the
   move's mini-cell in the policy grid — the leader line runs circle to
   circle, marking the SAME square twice. */
.tm-move.top .tm-chip {
  position: relative;
}

.tm-move.top .tm-chip::after {
  content: '';
  position: absolute;
  inset: -5px;
  border: 1.6px solid #f7c058;
  border-radius: 50%;
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
