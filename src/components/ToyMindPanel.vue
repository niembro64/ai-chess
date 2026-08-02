<script setup lang="ts">
// Toy Mind — visualizes what the Toy net saw, guessed, and chose.
//
//   NETWORK INPUT           NETWORK OUTPUT               SEARCH
//   GAME STATE planes   →   POLICY HEAD + VALUE HEAD  →  all legal moves
//   (six 8×8 grids,         (board of boards)            by visit share
//    one per piece type)
//
// Interactions:
// - Tap GAME STATE  → fullscreen modal of the six piece planes.
// - Tap POLICY HEAD → fullscreen modal: big policy board + SEARCH list.
// - Click a SEARCH row → the amber circles + leader line re-target that
//   move (reset to the top move whenever Toy thinks again).
//
// Both the POLICY HEAD and the GAME STATE render through the SAME
// BoardGrid canvas component (the policy head's renderer): the policy
// head is an 8×8 grid of destination boards, the game state is a 1×6
// strip of piece planes on desktop and a 6×1 column on mobile, with
// the piece names as the strip's row/column headers.

import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import BoardGrid from './BoardGrid.vue';
import type { ToyThought } from '@/game/ai/ToyPlayer';
import { TOY_CHANNEL_NAMES, TOY_NUM_PLANES } from '@/game/ai/ToyNet';

// `flipped` mirrors the game board's orientation: when the human plays
// black the board renders 180° rotated, and the POLICY HEAD follows so
// both boards read the same way (h1 top-left, your pieces at bottom).
const props = defineProps<{ thought: ToyThought; flipped?: boolean }>();

const bodyEl = ref<HTMLDivElement | null>(null);
const gridRef = ref<InstanceType<typeof BoardGrid> | null>(null);
const moveListEl = ref<HTMLDivElement | null>(null);
const leader = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
const hover = ref<{ x: number; y: number; text: string } | null>(null);

// Modals + THE selection.
//
// One selection for the whole panel, stored as a NET policy index
// (0-4095). Clicking a SEARCH row selects that move; clicking a tiny
// square in the policy modal selects that cell (legal or illegal). The
// amber ring, the row highlight, the leader line, and the decode text
// all derive from this single value; it resets to the search's top
// move whenever Toy thinks again.
const expanded = ref<null | 'state' | 'policy'>(null);
const modalPolicyWrap = ref<HTMLDivElement | null>(null);
const modalGridRef = ref<InstanceType<typeof BoardGrid> | null>(null);
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

type SearchRow = { uci: string; index: number; share: number; p: number; legal: boolean };

function chipStyle(row: SearchRow): Record<string, string> {
  return { background: cellColor(row.p, !row.legal) };
}

// Bar behind each SEARCH row. Legal rows: visit share, the top move
// gets a full teal bar and the rest scale relative to it. Illegal
// rows: raw prior relative to the biggest illegal prior, in red.
function rowStyle(row: SearchRow): Record<string, string> {
  if (row.legal) {
    const top = props.thought.moves[0]?.share || 1;
    const pct = Math.max(2, (row.share / top) * 100);
    return {
      background: `linear-gradient(90deg, rgba(94, 234, 212, 0.22) ${pct}%, rgba(255, 255, 255, 0.03) ${pct}%)`,
    };
  }
  const iMax = policyStats.value.iMax || 1;
  const pct = Math.max(2, (row.p / iMax) * 100);
  return {
    background: `linear-gradient(90deg, rgba(248, 90, 90, 0.16) ${pct}%, rgba(255, 255, 255, 0.03) ${pct}%)`,
  };
}

function selectIndex(idx: number): void {
  selIndex.value = idx;
  nextTick(() => {
    updateLeaders();
    // Keep the selected row visible in whichever list is on screen.
    for (const list of [moveListEl.value, modalMovesEl.value]) {
      list?.querySelector('.tm-move.selected')?.scrollIntoView({ block: 'nearest' });
    }
  });
}

// --- coordinate frames -----------------------------------------------

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

// NET square (0-63) -> DISPLAY square. Both hops are 180° rotations,
// so the composition is an involution: the same function converts
// display squares back to net squares.
function sqDisp(netSq: number): number {
  const s = realToDisp(netToReal(netSq));
  return s.r * 8 + s.f;
}

// NET policy index <-> DISPLAY policy index (self-inverse).
function policyDisp(idx: number): number {
  return sqDisp(Math.floor(idx / 64)) * 64 + sqDisp(idx % 64);
}

// Human-readable description of any NET policy index
// ("g2→g4 · 1.2% · legal").
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

// --- policy grid data (display-ordered, fed to BoardGrid) --------------

const policyColors = computed(() => {
  const { rawPolicy, legalMask } = props.thought;
  const colors = new Array<string>(4096);
  for (let i = 0; i < 4096; i++) {
    colors[policyDisp(i)] = cellColor(rawPolicy[i], legalMask[i] === 0);
  }
  return colors;
});

const fileLabels = computed(() =>
  Array.from({ length: 8 }, (_, i) => String.fromCharCode(97 + (props.flipped ? 7 - i : i))),
);
const rankLabels = computed(() =>
  Array.from({ length: 8 }, (_, i) => String(props.flipped ? i + 1 : 8 - i)),
);

const ringDisp = computed(() =>
  selIndex.value !== null ? policyDisp(selIndex.value) : null,
);

// --- SEARCH rows: the whole policy, colored by legality ----------------
//
// Legal moves first, ordered by visit share (the search's opinion),
// then EVERY illegal move ordered by the net's raw prior — same teal /
// red families as the policy grid. The default selection stays the
// top legal move.

function uciOf(netIndex: number): string {
  const from = netToReal(Math.floor(netIndex / 64));
  const to = netToReal(netIndex % 64);
  const name = (s: { r: number; f: number }) =>
    String.fromCharCode(97 + s.f) + String(8 - s.r);
  return name(from) + name(to);
}

const searchRows = computed<SearchRow[]>(() => {
  const { moves, rawPolicy, legalMask } = props.thought;
  const rows: SearchRow[] = moves.map(m => ({
    uci: m.uci, index: m.index, share: m.share, p: rawPolicy[m.index], legal: true,
  }));
  const illegal: SearchRow[] = [];
  for (let i = 0; i < rawPolicy.length; i++) {
    if (!legalMask[i]) {
      illegal.push({ uci: uciOf(i), index: i, share: 0, p: rawPolicy[i], legal: false });
    }
  }
  illegal.sort((a, b) => b.p - a.p);
  return rows.concat(illegal);
});

const illegalCount = computed(() => searchRows.value.length - props.thought.moves.length);

function rawPct(x: number): string {
  return `${(x * 100).toFixed(2)}%`;
}

// --- game state strip: one 8×8 plane per piece type ---------------------
//
// Literal piece colors (white piece / black piece / grey empty), in the
// same display orientation as the POLICY HEAD and the game board. The
// strip runs 1×6 on desktop (full names as column headers) and 6×1 on
// mobile (K Q R B N P letters as row headers) — same flat color array
// either way, only the BoardGrid layout props change.

const STATE_GREY = '#454b5e';
const STATE_WHITE = '#f2ead8';
const STATE_BLACK = '#14111f';

// Kings first — channel order in the tensor is P N B R Q K.
const STATE_ORDER = [5, 4, 3, 2, 1, 0];
const STATE_NAMES = STATE_ORDER.map(ch => TOY_CHANNEL_NAMES[ch]);
const STATE_LETTERS = ['K', 'Q', 'R', 'B', 'N', 'P'];

// The mobile breakpoint mirrors this stylesheet's @media split — the
// strip's orientation is a prop, not a CSS concern.
const mq = window.matchMedia('(max-width: 900px)');
const isMobile = ref(mq.matches);
const onMqChange = (): void => { isMobile.value = mq.matches; };

const stateRows = computed(() => (isMobile.value ? 6 : 1));
const stateCols = computed(() => (isMobile.value ? 1 : 6));
const stateRowLabels = computed(() => (isMobile.value ? STATE_LETTERS : null));
const stateColLabels = computed(() => (isMobile.value ? null : STATE_NAMES));

const stateColors = computed(() => {
  const { planes, blackToMove } = props.thought;
  const colors = new Array<string>(STATE_ORDER.length * 64).fill(STATE_GREY);
  STATE_ORDER.forEach((ch, board) => {
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const v = planes[(r * 8 + f) * TOY_NUM_PLANES + ch];
        if (v === 0) continue;
        // +1 is the mover's piece; the mover is black iff blackToMove.
        const isBlackPiece = (v > 0) === blackToMove;
        const d = realToDisp(netToReal(r * 8 + f));
        colors[board * 64 + d.r * 8 + d.f] = isBlackPiece ? STATE_BLACK : STATE_WHITE;
      }
    }
  });
  return colors;
});

// --- modal handling ------------------------------------------------------

function openModal(which: 'state' | 'policy'): void {
  expanded.value = which;
  nextTick(updateLeaders);
}

function closeModal(): void {
  expanded.value = null;
  hover.value = null;
  nextTick(updateLeaders);
}

// Hover decode exists only in the MODAL (desktop nicety); the inline
// canvas is purely a "tap to expand" trigger.
function onModalGridMove(index: number | null, clientX: number, clientY: number): void {
  if (index === null || !modalPolicyWrap.value) {
    hover.value = null;
    return;
  }
  const cRect = modalPolicyWrap.value.getBoundingClientRect();
  hover.value = {
    x: clientX - cRect.left + 14,
    y: clientY - cRect.top - 10,
    text: describeIndex(policyDisp(index)),
  };
}

// Click inside the modal selects that tiny square — same selection the
// SEARCH rows drive.
function onModalGridClick(index: number): void {
  selectIndex(policyDisp(index));
}

// --- leader lines: selected list entry → its grid cell -------------------

function computeLeader(
  container: HTMLElement,
  grid: InstanceType<typeof BoardGrid>,
  list: HTMLElement,
): { x1: number; y1: number; x2: number; y2: number } | null {
  // Every policy index has a SEARCH row now (legal and illegal), so
  // the leader exists whenever the selected row's chip is in the DOM.
  if (selIndex.value === null) return null;
  const canvas = grid.canvasEl;
  if (!canvas) return null;
  const chip = list.querySelector<HTMLElement>('.tm-move.selected .tm-chip');
  if (!chip) return null;

  const cRect = container.getBoundingClientRect();
  const cellRect = canvas.getBoundingClientRect();
  const chipRect = chip.getBoundingClientRect();

  const scale = cellRect.width / grid.baseW();
  const c = grid.cellCenter(policyDisp(selIndex.value));
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
    bodyEl.value && gridRef.value && moveListEl.value
      ? computeLeader(bodyEl.value, gridRef.value, moveListEl.value)
      : null;
  modalLeader.value =
    expanded.value === 'policy' && modalPolicyWrap.value && modalGridRef.value && modalMovesEl.value
      ? computeLeader(modalPolicyWrap.value, modalGridRef.value, modalMovesEl.value)
      : null;
}

// --- lifecycle ------------------------------------------------------------

function refresh(): void {
  // New thought → the selection resets to the search's top choice and
  // the lists scroll back to the top (where that row lives).
  selIndex.value = props.thought.moves[0]?.index ?? null;
  nextTick(() => {
    updateLeaders();
    for (const list of [moveListEl.value, modalMovesEl.value]) {
      if (list) list.scrollTop = 0;
    }
  });
}

watch(() => props.thought, refresh);

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  refresh();
  mq.addEventListener('change', onMqChange);
  resizeObserver = new ResizeObserver(() => updateLeaders());
  if (bodyEl.value) resizeObserver.observe(bodyEl.value);
});

onBeforeUnmount(() => {
  mq.removeEventListener('change', onMqChange);
  resizeObserver?.disconnect();
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
        <div class="tm-states" @click="openModal('state')">
          <BoardGrid
            class="tm-states-grid"
            :colors="stateColors"
            :rows="stateRows"
            :cols="stateCols"
            :cell-px="7"
            :gap="2"
            :col-labels="stateColLabels"
            :row-labels="stateRowLabels"
          />
        </div>
        <div class="tm-caption">
          one 8×8 plane per piece type · grey = empty · tap to expand
        </div>
      </section>

      <!-- ============ NETWORK OUTPUT ============ -->
      <section class="tm-section tm-sec-output">
        <h3 class="tm-section-title">Network Output</h3>
        <div class="tm-out">
          <div class="tm-out-policy">
            <div class="tm-subtitle">policy head</div>
            <div class="tm-grid-wrap">
              <BoardGrid
                ref="gridRef"
                class="tm-grid"
                :colors="policyColors"
                :mini="8"
                :col-labels="fileLabels"
                :row-labels="rankLabels"
                :ring="ringDisp"
                checker
                @click="openModal('policy')"
              />
            </div>
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
            v-for="m in searchRows"
            :key="m.uci"
            class="tm-move"
            :class="{ chosen: m.uci === thought.chosen, selected: m.index === selIndex, illegal: !m.legal }"
            :style="rowStyle(m)"
            @click="selectIndex(m.index)"
          >
            <span class="tm-chip" :style="chipStyle(m)"></span>
            <span class="tm-move-uci">{{ m.uci }}</span>
            <span class="tm-move-share">{{ m.legal ? pct(m.share) : rawPct(m.p) }}</span>
          </div>
        </div>
        <div v-if="thought.moves.length === 0" class="tm-gameover">
          game over — no legal moves
        </div>
        <div v-else class="tm-caption">
          {{ thought.moves.length }} legal moves by visit share, then
          {{ illegalCount }} illegal by prior · click one to point at its
          policy cell
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

      <!-- GAME STATE expanded: the same strip, drawn big. -->
      <div v-if="expanded === 'state'" class="tm-modal-card">
        <div class="tm-subtitle">game state</div>
        <BoardGrid
          class="tm-modal-states-grid"
          :colors="stateColors"
          :rows="stateRows"
          :cols="stateCols"
          :cell-px="7"
          :k="2"
          :gap="2"
          :col-labels="stateColLabels"
          :row-labels="stateRowLabels"
        />
        <div class="tm-caption">
          white / black pieces per plane · grey = empty · tap outside to close
        </div>
      </div>

      <!-- POLICY expanded: big board + the SEARCH list beneath it. -->
      <div v-else ref="modalPolicyWrap" class="tm-modal-card tm-modal-policy">
        <div class="tm-subtitle">policy head</div>
        <BoardGrid
          ref="modalGridRef"
          class="tm-modal-grid"
          :colors="policyColors"
          :mini="8"
          :k="2"
          :col-labels="fileLabels"
          :row-labels="rankLabels"
          :ring="ringDisp"
          checker
          @cell-click="onModalGridClick"
          @cell-move="onModalGridMove"
        />
        <div class="tm-picked" :class="{ empty: selIndex === null }">
          {{ selIndex !== null ? selectionText : 'tap a tiny square to inspect it' }}
        </div>
        <div class="tm-subtitle">search</div>
        <div ref="modalMovesEl" class="tm-moves tm-modal-moves">
          <div
            v-for="m in searchRows"
            :key="m.uci"
            class="tm-move"
            :class="{ chosen: m.uci === thought.chosen, selected: m.index === selIndex, illegal: !m.legal }"
            :style="rowStyle(m)"
            @click="selectIndex(m.index)"
          >
            <span class="tm-chip" :style="chipStyle(m)"></span>
            <span class="tm-move-uci">{{ m.uci }}</span>
            <span class="tm-move-share">{{ m.legal ? pct(m.share) : rawPct(m.p) }}</span>
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

/* --- GAME STATE: the piece-plane strip ------------------------------- */
/* One BoardGrid canvas, same visual language as the POLICY HEAD:
   desktop 1×6 with full names as bottom headers, displayed 1:1 at its
   base geometry (~346px — matches the policy board's width) so it
   stays as crisp as the policy head. The mobile block at the end
   flips it to 6×1 with letter row headers and scales by height. */

.tm-states {
  display: flex;
  justify-content: center;
  cursor: pointer;
}

.tm-states-grid {
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(10, 9, 20, 0.9);
  max-width: 100%;
  height: auto;
}

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

.tm-grid-wrap {
  display: flex;
  justify-content: center;
}

.tm-grid {
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

.tm-move.illegal .tm-move-uci {
  color: #d99a9a;
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

/* The modal strip is drawn at k=2 (backing store 2× base); the
   max-constraints fit either orientation and cap it near its natural
   2× size so it stays sharp. */
.tm-modal-states-grid {
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(10, 9, 20, 0.95);
  max-width: min(92vw, 700px);
  max-height: 72dvh;
  width: auto;
  height: auto;
}

.tm-modal-grid {
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
  /* The panel fills its row (GameView grows .toy-mind-row to the
     bottom cluster) — flex chains, never percentage heights, down to
     each section; the canvases contain via the absolute pattern. */
  .toy-mind {
    display: flex;
    flex-direction: column;
    flex: 1 1 0;
    min-height: 0;
  }
  .tm-body {
    display: flex;
    flex-wrap: nowrap;
    gap: 4px;
    padding: 4px;
    align-items: stretch;
    flex: 1 1 0;
    min-height: 0;
  }
  .tm-section {
    padding: 3px 4px 4px;
    gap: 3px;
    min-width: 0;
    min-height: 0;
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
    flex-shrink: 0;
  }
  .tm-caption {
    display: none;
  }
  /* GAME STATE strip runs 6×1 here (BoardGrid props flip at this same
     breakpoint via matchMedia). flex-basis 0 is load-bearing: the
     column's tall natural canvas must never inflate the strip — it
     takes whatever height the POLICY HEAD leaves and scales into it,
     aspect kept, never clipped. */
  .tm-states {
    flex: 1 1 0;
    min-height: 0;
    align-self: stretch;
    position: relative;
  }
  /* Absolutely positioned inside the (definite) .tm-states box:
     percentage max-sizes then resolve against the real container on
     every engine — flex-derived percentage heights are unreliable on
     WebKit and were letting the last plane overflow. width/height
     auto + both max constraints scale the canvas to fit, aspect kept,
     margin:auto centers it. */
  .tm-states-grid {
    position: absolute;
    inset: 0;
    margin: auto;
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
  }
  .tm-out {
    flex-direction: column;
    gap: 3px;
    align-items: center;
    flex: 1 1 0;
    min-height: 0;
    width: 100%;
  }
  .tm-out-policy {
    flex: 1 1 0;
    min-height: 0;
    width: 100%;
  }
  /* POLICY HEAD containment — same absolute pattern as the GAME
     STATE: scaled to fit both the section's width and height,
     centered, never clipped (wide-short screens used to chop its
     bottom). Absolute is safe here ONLY because .toy-mind-row
     align-self: stretch fixes the panel's width — the section widths
     are %-based, nothing depends on this canvas's intrinsic size. */
  .tm-grid-wrap {
    flex: 1 1 0;
    min-height: 0;
    align-self: stretch;
    position: relative;
  }
  .tm-grid {
    position: absolute;
    inset: 0;
    margin: auto;
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
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
