<script setup lang="ts">
// Toy Mind — visualizes what the Toy net saw, guessed, and chose.
//
//   NETWORK INPUT          NETWORK OUTPUT               SEARCH
//   GAME STATE tensor  →   POLICY HEAD + VALUE HEAD  →  all legal moves
//   (rotatable 3D)         (board of boards)            by visit share
//
// Interactions:
// - Tap GAME STATE  → fullscreen modal of the 3D tensor.
// - Tap POLICY HEAD → fullscreen modal: big policy board + SEARCH list.
// - Click a SEARCH row → the amber circles + leader line re-target that
//   move (reset to the top move whenever Toy thinks again).

import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { ToyThought } from '@/game/ai/ToyPlayer';
import { TOY_CHANNEL_NAMES, TOY_NUM_PLANES } from '@/game/ai/ToyNet';

// `flipped` mirrors the game board's orientation: when the human plays
// black the board renders 180° rotated, and the POLICY HEAD follows so
// both boards read the same way (h1 top-left, your pieces at bottom).
const props = defineProps<{ thought: ToyThought; flipped?: boolean }>();

const bodyEl = ref<HTMLDivElement | null>(null);
const sceneEl = ref<HTMLDivElement | null>(null);
const gridCanvas = ref<HTMLCanvasElement | null>(null);
const moveListEl = ref<HTMLDivElement | null>(null);
const leader = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
const hover = ref<{ x: number; y: number; text: string } | null>(null);

// Modals + THE selection.
//
// One selection for the whole panel, stored as a policy index (0-4095).
// Clicking a SEARCH row selects that move; clicking a tiny square in
// the policy modal selects that cell (legal or illegal). The amber
// ring, the row highlight, the leader line, and the decode text all
// derive from this single value; it resets to the search's top move
// whenever Toy thinks again.
const expanded = ref<null | 'state' | 'policy'>(null);
const modalSceneEl = ref<HTMLDivElement | null>(null);
const modalPolicyWrap = ref<HTMLDivElement | null>(null);
const modalGridCanvas = ref<HTMLCanvasElement | null>(null);
const modalMovesEl = ref<HTMLDivElement | null>(null);
const modalLeader = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
const selIndex = ref<number | null>(null);

// --- shared colormap: grid cells AND list chips ------------------------
//
// Hue = legality (teal legal / red illegal); brightness = probability,
// MIN-MAX NORMALIZED PER CLASS and linearly interpolated: the lowest
// probability in each family maps to its darkest shade, the highest to
// its brightest. Full dynamic range in both families, every position.

const policyStats = computed(() => {
  const { rawPolicy, legalMask } = props.thought;
  let lMin = Infinity, lMax = -Infinity, iMin = Infinity, iMax = -Infinity;
  for (let i = 0; i < rawPolicy.length; i++) {
    const p = rawPolicy[i];
    if (legalMask[i]) {
      if (p < lMin) lMin = p;
      if (p > lMax) lMax = p;
    } else {
      if (p < iMin) iMin = p;
      if (p > iMax) iMax = p;
    }
  }
  if (!Number.isFinite(lMin)) { lMin = 0; lMax = 1; }
  if (!Number.isFinite(iMin)) { iMin = 0; iMax = 1; }
  return { lMin, lMax, iMin, iMax };
});

const DARKEST = 0.12;  // alpha of the darkest shade — never invisible

function cellColor(p: number, illegal: boolean): string {
  const s = policyStats.value;
  const min = illegal ? s.iMin : s.lMin;
  const max = illegal ? s.iMax : s.lMax;
  const t = max > min ? (p - min) / (max - min) : 1;
  const a = DARKEST + (1 - DARKEST) * t;
  return illegal
    ? `rgba(248, 90, 90, ${a})`
    : `rgba(94, 234, 212, ${a})`;
}

function chipStyle(index: number): Record<string, string> {
  return { background: cellColor(props.thought.rawPolicy[index], false) };
}

// Visit-share bar behind each SEARCH row: the top move gets a full bar,
// the rest scale relative to it.
function rowStyle(share: number): Record<string, string> {
  const top = props.thought.moves[0]?.share || 1;
  const pct = Math.max(2, (share / top) * 100);
  return {
    background: `linear-gradient(90deg, rgba(94, 234, 212, 0.22) ${pct}%, rgba(255, 255, 255, 0.03) ${pct}%)`,
  };
}

function selectIndex(idx: number): void {
  selIndex.value = idx;
  redrawGrids();
  nextTick(updateLeaders);
}

// --- three.js input scene ----------------------------------------------

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let sphereGroup: THREE.Group | null = null;
let rafId = 0;

const SLAB_GAP = 1.5;

// --- Idle auto-spin -----------------------------------------------------
//
// The scene drifts in a slow azimuthal rotation whenever the user isn't
// touching it. `autoVel` EMAs toward its target each frame:
//   - grabbing the scene → target 0 with a very short time constant, so
//     the auto-spin gets out of the way instantly and the drag feels 1:1
//   - releasing → target SPIN with a long time constant, so the user's
//     fling (OrbitControls damping) decays while the idle drift ramps
//     back up — a smooth velocity handoff, no snap
//   - initial mount starts at 0 and eases up the same way
const IDLE_SPIN = 0.25;      // rad/s ≈ one revolution every ~25s
const RAMP_TAU = 1.6;        // s — gentle pickup toward idle spin
const GRAB_TAU = 0.12;       // s — near-instant yield to the user's hand
const SPIN_AXIS = new THREE.Vector3(0, 1, 0);
let autoVel = 0;
let userHolding = false;
let lastFrameT = 0;

function initScene(): void {
  const el = sceneEl.value!;
  const w = el.clientWidth || 340;
  const h = el.clientHeight || 290;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.setSize(w, h);
  el.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 4;
  controls.maxDistance = 60;
  controls.addEventListener('start', () => { userHolding = true; });
  controls.addEventListener('end', () => { userHolding = false; });
  fitCamera();

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

  lastFrameT = performance.now();
  const animate = () => {
    rafId = requestAnimationFrame(animate);
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastFrameT) / 1000);
    lastFrameT = now;

    // EMA the auto-spin velocity toward its current target.
    const target = userHolding ? 0 : IDLE_SPIN;
    const tau = userHolding ? GRAB_TAU : RAMP_TAU;
    autoVel += (target - autoVel) * (1 - Math.exp(-dt / tau));

    if (Math.abs(autoVel) > 1e-4 && camera && controls) {
      const offset = camera.position.clone().sub(controls.target);
      offset.applyAxisAngle(SPIN_AXIS, autoVel * dt);
      camera.position.copy(controls.target).add(offset);
    }

    controls!.update();
    renderer!.render(scene!, camera!);
  };
  animate();
}

// Frame the whole tensor stack (slabs + channel labels) inside the
// current viewport, whatever its size/aspect: back the camera off along
// a pleasing iso direction until the scene's bounding sphere fits the
// narrower of the two view angles.
const SCENE_RADIUS = 7.4;

function fitCamera(): void {
  if (!camera || !controls) return;
  const target = new THREE.Vector3(0, (TOY_NUM_PLANES - 1) * SLAB_GAP * 0.5, 0);
  const fovV = (camera.fov * Math.PI) / 180;
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect);
  const halfMin = Math.min(fovV, fovH) / 2;
  const dist = (SCENE_RADIUS / Math.sin(halfMin)) * 1.02;
  const dir = new THREE.Vector3(1, 0.55, 1.15).normalize();
  camera.position.copy(target.clone().add(dir.multiplyScalar(dist)));
  camera.updateProjectionMatrix();
  controls.target.copy(target);
  controls.update();
}

// Re-home the single renderer between the inline card and the modal —
// one scene, one WebGL context, two possible parents. Refits the
// camera so the stack always starts fully framed in the new viewport.
function mountSceneTo(el: HTMLElement): void {
  if (!renderer || !camera) return;
  el.appendChild(renderer.domElement);
  const w = el.clientWidth || 340;
  const h = el.clientHeight || 290;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  fitCamera();
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
        const isBlackPiece = (v > 0) === blackToMove;
        const mesh = new THREE.Mesh(sphereGeo, isBlackPiece ? blackMat : whiteMat);
        mesh.position.set(f - 3.5, ch * SLAB_GAP, r - 3.5);
        sphereGroup.add(mesh);
      }
    }
  }
}

// Tap (not drag) on the scene expands it — distinguish from an orbit
// drag by pointer travel.
let downX = 0;
let downY = 0;
function scenePointerDown(e: PointerEvent): void {
  downX = e.clientX;
  downY = e.clientY;
}
function scenePointerUp(e: PointerEvent): void {
  if (Math.hypot(e.clientX - downX, e.clientY - downY) < 8) {
    openModal('state');
  }
}

// --- modal handling ------------------------------------------------------

function openModal(which: 'state' | 'policy'): void {
  expanded.value = which;
  nextTick(() => {
    if (which === 'state' && modalSceneEl.value) {
      mountSceneTo(modalSceneEl.value);
    }
    if (which === 'policy') {
      redrawGrids();
    }
    updateLeaders();
  });
}

function closeModal(): void {
  const was = expanded.value;
  expanded.value = null;
  hover.value = null;
  nextTick(() => {
    if (was === 'state' && sceneEl.value) {
      mountSceneTo(sceneEl.value);
    }
    updateLeaders();
  });
}

// --- policy grid: a board of boards --------------------------------------
//
// Big 8x8 board in REAL coordinates (files a-h, ranks 8..1 top-down),
// each square holding a tiny 8x8 board of destination probabilities.
// Drawn at base geometry × an integer scale k (1 inline, 2 in the
// modal) via ctx.scale, so both canvases stay crisp.

const MINI = 5;
const OUTER = MINI * 8;
const GAP = 1;
const PAD_L = 16;
const PAD_B = 14;
const BOARD = 8 * OUTER + 7 * GAP;
const BASE_W = PAD_L + BOARD + 2;
const BASE_H = BOARD + PAD_B;

function netToReal(netIndex: number): { r: number; f: number } {
  let r = Math.floor(netIndex / 8);
  let f = netIndex % 8;
  if (props.thought.blackToMove) {
    r = 7 - r;
    f = 7 - f;
  }
  return { r, f };
}

// Real board coords -> DISPLAY coords, matching the game board's
// orientation (180° rotation when the human plays black).
function realToDisp(s: { r: number; f: number }): { r: number; f: number } {
  return props.flipped ? { r: 7 - s.r, f: 7 - s.f } : s;
}

// Base-coordinate pixel center of a policy entry's mini-cell.
function cellCenter(policyIndex: number): { x: number; y: number } {
  const from = realToDisp(netToReal(Math.floor(policyIndex / 64)));
  const to = realToDisp(netToReal(policyIndex % 64));
  return {
    x: PAD_L + from.f * (OUTER + GAP) + to.f * MINI + MINI / 2,
    y: from.r * (OUTER + GAP) + to.r * MINI + MINI / 2,
  };
}

// Human-readable description of any policy index ("g2→g4 · 1.2% · legal").
function describeIndex(idx: number): string {
  const from = netToReal(Math.floor(idx / 64));
  const to = netToReal(idx % 64);
  const name = (s: { r: number; f: number }) =>
    String.fromCharCode(97 + s.f) + String(8 - s.r);
  const p = props.thought.rawPolicy[idx];
  const legal = props.thought.legalMask[idx] === 1;
  return `${name(from)}→${name(to)} · ${(p * 100).toFixed(2)}% · ${legal ? 'legal' : 'illegal'}`;
}

const selectionText = computed(() =>
  selIndex.value !== null ? describeIndex(selIndex.value) : '',
);

function drawGridInto(canvas: HTMLCanvasElement, k: number): void {
  const ctx = canvas.getContext('2d')!;
  canvas.width = BASE_W * k;
  canvas.height = BASE_H * k;
  ctx.scale(k, k);
  ctx.fillStyle = 'rgba(10, 9, 20, 0.9)';
  ctx.fillRect(0, 0, BASE_W, BASE_H);

  const { rawPolicy, legalMask } = props.thought;

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const ox = PAD_L + f * (OUTER + GAP);
      const oy = r * (OUTER + GAP);
      ctx.fillStyle = (r + f) % 2 === 0
        ? 'rgba(139, 143, 217, 0.10)'
        : 'rgba(139, 143, 217, 0.035)';
      ctx.fillRect(ox, oy, OUTER, OUTER);
    }
  }

  for (let i = 0; i < 4096; i++) {
    const from = realToDisp(netToReal(Math.floor(i / 64)));
    const to = realToDisp(netToReal(i % 64));
    ctx.fillStyle = cellColor(rawPolicy[i], legalMask[i] === 0);
    ctx.fillRect(
      PAD_L + from.f * (OUTER + GAP) + to.f * MINI,
      from.r * (OUTER + GAP) + to.r * MINI,
      MINI - 0.5,
      MINI - 0.5,
    );
  }

  // Labels follow the display orientation, same as the game board's
  // coordinate strips.
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < 8; i++) {
    const fileChar = String.fromCharCode(97 + (props.flipped ? 7 - i : i));
    const rankNum = props.flipped ? i + 1 : 8 - i;
    ctx.fillText(fileChar, PAD_L + i * (OUTER + GAP) + OUTER / 2, BOARD + 11);
    ctx.fillText(String(rankNum), PAD_L / 2 - 1, i * (OUTER + GAP) + OUTER / 2 + 3);
  }

  // Ring THE selected cell (search row or modal cell pick — one ring).
  if (selIndex.value !== null) {
    const c = cellCenter(selIndex.value);
    ctx.strokeStyle = '#f7c058';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function redrawGrids(): void {
  if (gridCanvas.value) drawGridInto(gridCanvas.value, 1);
  if (expanded.value === 'policy' && modalGridCanvas.value) {
    drawGridInto(modalGridCanvas.value, 2);
  }
}

// Decode a mouse/touch position on a policy canvas into a policy index
// + human-readable text, or null when outside the cells.
function decodeCell(
  e: MouseEvent,
  canvas: HTMLCanvasElement,
): { index: number; text: string } | null {
  const rect = canvas.getBoundingClientRect();
  const scale = rect.width / canvas.width || 1;
  const k = canvas.width / BASE_W;
  const x = (e.clientX - rect.left) / (scale * k) - PAD_L;
  const y = (e.clientY - rect.top) / (scale * k);
  const dFromF = Math.floor(x / (OUTER + GAP));
  const dFromR = Math.floor(y / (OUTER + GAP));
  const dToF = Math.floor((x - dFromF * (OUTER + GAP)) / MINI);
  const dToR = Math.floor((y - dFromR * (OUTER + GAP)) / MINI);
  if (dFromF < 0 || dFromF > 7 || dFromR < 0 || dFromR > 7 || dToF < 0 || dToF > 7 || dToR < 0 || dToR > 7) {
    return null;
  }
  // Display coords -> real board coords (un-flip), then real -> net frame.
  const flip = props.flipped ? (v: number) => 7 - v : (v: number) => v;
  const fromR = flip(dFromR), fromF = flip(dFromF);
  const toR = flip(dToR), toF = flip(dToF);
  const black = props.thought.blackToMove;
  const nFrom = (black ? 7 - fromR : fromR) * 8 + (black ? 7 - fromF : fromF);
  const nTo = (black ? 7 - toR : toR) * 8 + (black ? 7 - toF : toF);
  const idx = nFrom * 64 + nTo;
  const p = props.thought.rawPolicy[idx];
  const illegal = props.thought.legalMask[idx] === 0;
  const name = (f: number, r: number) => String.fromCharCode(97 + f) + String(8 - r);
  return {
    index: idx,
    text: `${name(fromF, fromR)}→${name(toF, toR)} · ${(p * 100).toFixed(2)}%${illegal ? ' · illegal' : ' · legal'}`,
  };
}

// Hover decode exists only in the MODAL (desktop nicety); the inline
// canvas is purely a "tap to expand" trigger.
function onModalGridMove(e: MouseEvent): void {
  if (!modalGridCanvas.value || !modalPolicyWrap.value) return;
  const cell = decodeCell(e, modalGridCanvas.value);
  if (!cell) {
    hover.value = null;
    return;
  }
  const cRect = modalPolicyWrap.value.getBoundingClientRect();
  hover.value = {
    x: e.clientX - cRect.left + 14,
    y: e.clientY - cRect.top - 10,
    text: cell.text,
  };
}

// Click inside the modal selects that tiny square — same selection the
// SEARCH rows drive.
function onModalGridClick(e: MouseEvent): void {
  if (!modalGridCanvas.value) return;
  const cell = decodeCell(e, modalGridCanvas.value);
  if (cell) selectIndex(cell.index);
}

// --- leader lines: selected list entry → its grid cell -------------------

function computeLeader(
  container: HTMLElement,
  canvas: HTMLCanvasElement,
  list: HTMLElement,
): { x1: number; y1: number; x2: number; y2: number } | null {
  // The leader only exists when the selection corresponds to a SEARCH
  // row (a legal move); an illegal-cell selection rings the grid alone.
  if (selIndex.value === null) return null;
  const sel = props.thought.moves.find(m => m.index === selIndex.value);
  if (!sel) return null;
  const chip = list.querySelector<HTMLElement>('.tm-move.selected .tm-chip');
  if (!chip) return null;

  const cRect = container.getBoundingClientRect();
  const cellRect = canvas.getBoundingClientRect();
  const chipRect = chip.getBoundingClientRect();

  const k = canvas.width / BASE_W;
  const scale = (cellRect.width / canvas.width || 1) * k;
  const c = cellCenter(sel.index);
  const cx = cellRect.left - cRect.left + c.x * scale;
  const cy = cellRect.top - cRect.top + c.y * scale;
  const lx = chipRect.left - cRect.left + chipRect.width / 2;
  const ly = chipRect.top - cRect.top + chipRect.height / 2;
  const dx = cx - lx;
  const dy = cy - ly;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  return { x1: lx + ux * 12, y1: ly + uy * 12, x2: cx - ux * 10, y2: cy - uy * 10 };
}

function updateLeaders(): void {
  leader.value =
    bodyEl.value && gridCanvas.value && moveListEl.value
      ? computeLeader(bodyEl.value, gridCanvas.value, moveListEl.value)
      : null;
  modalLeader.value =
    expanded.value === 'policy' && modalPolicyWrap.value && modalGridCanvas.value && modalMovesEl.value
      ? computeLeader(modalPolicyWrap.value, modalGridCanvas.value, modalMovesEl.value)
      : null;
}

// --- lifecycle ------------------------------------------------------------

function refresh(): void {
  // New thought → the selection resets to the search's top choice.
  selIndex.value = props.thought.moves[0]?.index ?? null;
  redrawGrids();
  updateSpheres();
  nextTick(updateLeaders);
}

watch(() => props.thought, refresh);

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  initScene();
  refresh();
  resizeObserver = new ResizeObserver(() => updateLeaders());
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
      <section class="tm-section tm-sec-state">
        <h3 class="tm-section-title">Network Input</h3>
        <div class="tm-subtitle">game state</div>
        <div
          ref="sceneEl"
          class="tm-scene"
          @pointerdown="scenePointerDown"
          @pointerup="scenePointerUp"
        ></div>
        <div class="tm-caption">
          8×8×6 tensor — drag to rotate, tap to expand<br />
          Toy's view — its pieces are +1, yours are −1
        </div>
      </section>

      <!-- ============ NETWORK OUTPUT ============ -->
      <section class="tm-section tm-sec-output">
        <h3 class="tm-section-title">Network Output</h3>
        <div class="tm-out">
          <div class="tm-out-policy">
            <div class="tm-subtitle">policy head</div>
            <canvas
              ref="gridCanvas"
              class="tm-grid"
              @click="openModal('policy')"
            ></canvas>
            <div class="tm-caption">
              each square holds a mini-board of its destinations · tap to expand
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
      <section class="tm-section tm-sec-search">
        <h3 class="tm-section-title">Search</h3>
        <div ref="moveListEl" class="tm-moves">
          <div
            v-for="m in thought.moves"
            :key="m.uci"
            class="tm-move"
            :class="{ chosen: m.uci === thought.chosen, selected: m.index === selIndex }"
            :style="rowStyle(m.share)"
            @click="selectIndex(m.index)"
          >
            <span class="tm-chip" :style="chipStyle(m.index)"></span>
            <span class="tm-move-uci">{{ m.uci }}</span>
            <span class="tm-move-share">{{ pct(m.share) }}</span>
          </div>
        </div>
        <div v-if="thought.moves.length === 0" class="tm-gameover">
          game over — no legal moves
        </div>
        <div v-else class="tm-caption">
          all {{ thought.moves.length }} legal moves by visit share · click one
          to point at its policy cell
        </div>
      </section>

      <!-- Leader line: selected move → its policy cell. -->
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

    </div>

    <!-- ============ Fullscreen modals ============ -->
    <!-- Teleported to <body>: the panel's backdrop-filter makes it the
         containing block for position:fixed, which would imprison and
         clip the overlay inside the panel strip. -->
    <Teleport to="body">
    <div v-if="expanded" class="tm-modal" @click.self="closeModal()">
      <button class="tm-modal-close" @click="closeModal()">✕</button>

      <!-- GAME STATE expanded: the same three.js scene, re-homed big. -->
      <div v-if="expanded === 'state'" class="tm-modal-card">
        <div class="tm-subtitle">game state</div>
        <div ref="modalSceneEl" class="tm-modal-scene"></div>
        <div class="tm-caption">drag to rotate · tap outside to close</div>
      </div>

      <!-- POLICY expanded: big board + the SEARCH list beneath it. -->
      <div v-else ref="modalPolicyWrap" class="tm-modal-card tm-modal-policy">
        <div class="tm-subtitle">policy head</div>
        <canvas
          ref="modalGridCanvas"
          class="tm-modal-grid"
          @click="onModalGridClick"
          @mousemove="onModalGridMove"
          @mouseleave="hover = null"
        ></canvas>
        <div class="tm-picked" :class="{ empty: selIndex === null }">
          {{ selIndex !== null ? selectionText : 'tap a tiny square to inspect it' }}
        </div>
        <div class="tm-subtitle">search</div>
        <div ref="modalMovesEl" class="tm-moves tm-modal-moves">
          <div
            v-for="m in thought.moves"
            :key="m.uci"
            class="tm-move"
            :class="{ chosen: m.uci === thought.chosen, selected: m.index === selIndex }"
            :style="rowStyle(m.share)"
            @click="selectIndex(m.index)"
          >
            <span class="tm-chip" :style="chipStyle(m.index)"></span>
            <span class="tm-move-uci">{{ m.uci }}</span>
            <span class="tm-move-share">{{ pct(m.share) }}</span>
          </div>
        </div>

        <svg v-if="modalLeader" class="tm-leader" aria-hidden="true">
          <line
            :x1="modalLeader.x1" :y1="modalLeader.y1"
            :x2="modalLeader.x2" :y2="modalLeader.y2"
            stroke="#f7c058" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.9"
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
    </Teleport>
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
  /* Desktop: compact 2x2 grid sized for the side-by-side game layout —
     GAME STATE top-left, POLICY beneath it, SEARCH as a full-height
     right column (the one thing allowed to scroll). The mobile block
     at the end of this file overrides back to a horizontal strip. */
  display: grid;
  grid-template-columns: auto minmax(165px, 195px);
  grid-template-rows: auto auto;
  grid-template-areas:
    "state  search"
    "output search";
  gap: 10px;
  padding: 10px 12px;
  align-items: stretch;
  justify-content: center;
}

.tm-sec-state { grid-area: state; }
.tm-sec-output { grid-area: output; }
.tm-sec-search {
  grid-area: search;
  min-height: 0;
}
.tm-sec-search .tm-moves {
  flex: 1 1 0;
  min-height: 0;
  max-height: none;
}

.tm-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 8px 12px 10px;
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

.tm-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  font-weight: 700;
  color: #5ae3d8;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}

.tm-scene {
  width: 260px;
  height: 192px;
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
  /* No rounding: radius clips the corner mini-cells of the a8/h8/a1/h1
     squares. */
  border-radius: 0;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(10, 9, 20, 0.9);
  cursor: pointer;
  max-width: calc(100vw - 90px);
  height: auto;
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
  cursor: pointer;
}

.tm-move.chosen {
  color: #5ae3d8;
  font-weight: 600;
}

/* Selected move (defaults to the top of the list; click any row to
   re-target): its chip wears the same amber circle as its mini-cell in
   the policy grid — the leader line runs circle to circle. */
.tm-move.selected .tm-chip {
  position: relative;
}

.tm-move.selected .tm-chip::after {
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

.tm-gameover {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  color: #f7c058;
  padding: 8px 12px;
  border: 1px dashed rgba(247, 192, 88, 0.5);
  border-radius: 6px;
  letter-spacing: 0.5px;
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

/* --- Fullscreen modals ------------------------------------------------ */

.tm-modal {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(8, 7, 16, 0.72);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}

.tm-modal-close {
  position: absolute;
  top: max(10px, env(safe-area-inset-top));
  right: 14px;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.25);
  background: rgba(20, 19, 38, 0.8);
  color: #e2e8f0;
  font-size: 15px;
  cursor: pointer;
  z-index: 61;
}

.tm-modal-card {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  max-width: 96vw;
  max-height: 92dvh;
}

.tm-modal-scene {
  width: min(92vw, 700px);
  height: min(70dvh, 620px);
  border-radius: 12px;
  overflow: hidden;
  background: radial-gradient(ellipse at 50% 40%, rgba(60, 58, 110, 0.35), rgba(12, 11, 24, 0.6));
  cursor: grab;
}
.tm-modal-scene:active { cursor: grabbing; }

.tm-modal-grid {
  border-radius: 0;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(10, 9, 20, 0.95);
  width: min(92vw, 58dvh);
  height: auto;
  cursor: crosshair;
}

.tm-modal-moves {
  max-height: 24dvh;
  min-width: min(70vw, 260px);
}

.tm-picked {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  color: #e2e8f0;
  padding: 3px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
}

.tm-picked.empty {
  color: #64748b;
  font-weight: 400;
  border-color: transparent;
  background: transparent;
}

/* --- Mobile compression — MUST stay the last block (ties the base
   rules on specificity; wins by cascade order). ---------------------- */
@media (max-width: 900px) {
  .tm-header {
    display: none;
  }
  .tm-body {
    display: flex;
    flex-wrap: nowrap;
    gap: 4px;
    padding: 4px;
    align-items: stretch;
  }
  .tm-sec-search .tm-moves {
    max-height: 168px;
  }
  .tm-section {
    padding: 3px 4px 4px;
    gap: 3px;
    min-width: 0;
  }
  /* No section titles on mobile — the teal subtitles carry the labels.
     POLICY HEAD gets the width; GAME STATE and SEARCH cede it. */
  .tm-section-title {
    display: none;
  }
  .tm-sec-state { flex: 0 1 25%; }
  .tm-sec-output { flex: 0 1 46%; }
  .tm-sec-search { flex: 1 1 25%; }
  .tm-subtitle {
    font-size: 8px;
    letter-spacing: 0.8px;
  }
  .tm-caption {
    display: none;
  }
  .tm-scene {
    width: 100%;
    height: 150px;
  }
  .tm-out {
    flex-direction: column;
    gap: 3px;
    align-items: center;
  }
  .tm-grid {
    max-width: 100%;
  }
  .tm-out-value {
    display: none;
  }
  .tm-moves {
    min-width: 0;
    width: 100%;
    max-height: 168px;
    padding-right: 2px;
  }
  .tm-move {
    font-size: 9.5px;
    padding: 1px 3px;
    gap: 3px;
  }
  .tm-move-share {
    font-size: 9px;
  }
  .tm-chip {
    width: 9px;
    height: 9px;
  }
  .tm-move.selected .tm-chip::after {
    inset: -4px;
  }
}
</style>
