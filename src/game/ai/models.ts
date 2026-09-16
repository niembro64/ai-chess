// Model registry + lazy weight loading.
//
// Weights live in public/models/ as static assets and are fetched ONLY
// when the player picks an opponent in the lobby — nothing model-sized
// ships in the JS bundle. (Sage's 15MB blob used to be imported into
// the bundle and downloaded by every visitor.)

export type ModelId = 'sage' | 'toy' | 'jester';

export const MODELS: Record<ModelId, {
  name: string;
  file: string;
  sims: number;
  // The model's value convention; Jester's checkpoint uses the UnCheck
  // turn-boundary objective. Older goal-inversion internals remain for
  // compatibility, but the lobby offers trained pairings only.
  trainedGoal: 'win' | 'lose';
}> = {
  sage: {
    name: 'Sage',
    file: 'models/sage.json',
    sims: 400,
    trainedGoal: 'win',
  },
  toy: {
    name: 'Toy',
    file: 'models/toy.json',
    sims: 128,
    trainedGoal: 'win',
  },
  jester: {
    name: 'Jester',
    file: 'models/jester.json',
    sims: 400,
    trainedGoal: 'lose',
  },
};

// --- Effort ------------------------------------------------------------
//
// EFFORT is the difficulty dial, and it buys SEARCH — which is where
// most of a network's playing strength lives. Published numbers are
// stark: AlphaGo Zero's raw network (one forward pass, argmax) rates
// ~3,000 Elo against ~5,200 for the same weights with MCTS, and Leela
// at 1 node/move sits just above 2,300 while a 128x10 net at 1,500
// nodes plays ~3,050. Search is worth hundreds to thousands of Elo, so
// the ladder runs from no search at all to the model's full budget:
//
//   LOW     the policy head alone — one forward pass, no lookahead
//   MEDIUM  100 simulations
//   HIGH    the model's full budget (400 for Sage and Jester)
//
// At LOW the model chooses its highest-ranked legal move. At MEDIUM
// and HIGH its search uses the game and value convention it trained on.

export type Effort = 'low' | 'medium' | 'high';

export const EFFORT_LEVELS: Record<Effort, { label: string; sims: number | null }> = {
  // null = the model's own full budget.
  low: { label: 'Low', sims: 0 },
  medium: { label: 'Medium', sims: 100 },
  high: { label: 'High', sims: null },
};

export function effortSims(id: ModelId, effort: Effort): number {
  const sims = EFFORT_LEVELS[effort].sims;
  // 0 means "no search at all" — the policy head plays directly.
  return sims === null ? MODELS[id].sims : Math.min(sims, MODELS[id].sims);
}

// --- Piece tints -------------------------------------------------------
//
// Single source of truth for board piece colors, so the lobby's "you
// play" preview and the live board agree. The HUMAN always uses the
// standard cream/charcoal set; the BOT's pieces carry its identity
// color — Sage green, Jester purple — light as white, dark as black.
// Toy keeps the standard set.

export type PieceTint = { fill: string; outline: string };

export const STANDARD_TINTS: Record<'white' | 'black', PieceTint> = {
  white: { fill: '#f6efde', outline: '#1a1410' },
  black: { fill: '#15110d', outline: 'rgba(246, 239, 222, 0.55)' },
};

export const BOT_TINTS: Record<'sage' | 'jester', Record<'white' | 'black', PieceTint>> = {
  sage: {
    white: { fill: '#d9f2d0', outline: '#14381d' },
    black: { fill: '#0e2b15', outline: 'rgba(190, 235, 195, 0.6)' },
  },
  jester: {
    white: { fill: '#ecdcf8', outline: '#381852' },
    black: { fill: '#1f0e30', outline: 'rgba(225, 200, 250, 0.6)' },
  },
};

export function pieceTint(model: ModelId | null, color: 'white' | 'black'): PieceTint {
  if (model === 'sage' || model === 'jester') return BOT_TINTS[model][color];
  return STANDARD_TINTS[color];
}

// --- The two rulesets ---------------------------------------------------
//
// The app plays chess two ways, each paired with its trained bot:
//
//   'win'   NORMAL chess.   You win by checkmating your opponent.
//   'lose'  UNCHECK chess. You win when your turn begins with your king
//           attacked; captures are compulsory and kings may enter attack.
//
// Sage plays Chess and Jester plays UnCheck Chess in the lobby.

export type Goal = 'win' | 'lose';

export const GRID_MODELS: readonly ModelId[] = ['sage', 'jester'];
// Retained for older callers that persist the former cross-pairing flag.
export const GRID_ASKED: readonly Goal[] = ['win', 'lose'];

/** Display name of a variant. 'win' is ordinary chess. */
export function goalLabel(goal: Goal): string {
  return goal === 'win' ? 'NORMAL' : 'UNCHECK';
}

/** One-line statement of a variant's win condition. */
export function variantRule(goal: Goal): string {
  return goal === 'win'
    ? 'Checkmate your opponent to win.'
    : 'Begin your turn with your king attacked to win; captures are compulsory.';
}

export function isInverted(model: ModelId, asked: Goal): boolean {
  return MODELS[model].trainedGoal !== asked;
}

/**
 * True when the game uses Uncheck rules. The compatibility flag still
 * records whether the requested row opposes the model's trained ranking.
 */
export function isUncheckVariant(model: ModelId, goalInverted: boolean): boolean {
  return (MODELS[model].trainedGoal === 'lose') !== goalInverted;
}

/** @deprecated Use isUncheckVariant. */
export const isInvertedVariant = isUncheckVariant;

// The sweaty variant appears only during active search.
export function botFace(model: ModelId, thinking = false): string {
  if (model === 'jester') return thinking ? 'jester-straining' : 'jester-gleeful';
  return thinking ? 'sage-flustered' : 'sage-calm';
}

export async function fetchModelJson(id: ModelId): Promise<unknown> {
  const url = import.meta.env.BASE_URL + MODELS[id].file;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${MODELS[id].name} weights (HTTP ${res.status})`);
  }
  return res.json();
}
