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
  // What the net was TRAINED to want. The lobby's GOAL INVERTED mode
  // pursues the opposite at play time (misère search flip) — see
  // AIPlayerOptions.
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
// At LOW the move is the EXTREME of the policy head (highest
// probability, or lowest when the bot is asked for the opposite of its
// trained goal). At MEDIUM and HIGH the search decides, and for an
// inverted goal the search is itself inverted, so its top choice is
// the best *planned* loss rather than merely the worst-rated move.

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

// --- The two variants, and the setup grid ------------------------------
//
// The app plays chess two ways, and the grid's ROW picks which one the
// game runs under — for BOTH players, not just the bot:
//
//   'win'   NORMAL chess.   You win by checkmating your opponent.
//   'lose'  INVERTED chess. You win by getting your OWN king
//           checkmated. The checkmated king is the winner.
//
// The names are traditional-sense: in the inverted variant "losing" the
// chess game is how you win the match. Every other rule is standard.
//
// COLUMNS are the networks, by the variant their weights were TRAINED
// on — Sage normal, Jester inverted. Off its own diagonal a model is
// playing a game it was never trained for; it copes by inverting its
// search, which is not the same as "pick the worst-looking move".

export type Goal = 'win' | 'lose';

export const GRID_MODELS: readonly ModelId[] = ['sage', 'jester'];
export const GRID_ASKED: readonly Goal[] = ['win', 'lose'];

/** Display name of a variant. 'win' is ordinary chess. */
export function goalLabel(goal: Goal): string {
  return goal === 'win' ? 'NORMAL' : 'INVERTED';
}

/** One-line statement of a variant's win condition. */
export function variantRule(goal: Goal): string {
  return goal === 'win'
    ? 'Checkmate your opponent to win.'
    : 'Get your own king checkmated to win.';
}

export function isInverted(model: ModelId, asked: Goal): boolean {
  return MODELS[model].trainedGoal !== asked;
}

/**
 * True when the game is the INVERTED variant: the checkmated king wins,
 * so both sides steer toward their own mate. Equivalently — and this is
 * why one flag drives both — exactly when the bot's search inverts,
 * since the bot pursues the variant's goal like everyone else.
 */
export function isInvertedVariant(model: ModelId, goalInverted: boolean): boolean {
  return (MODELS[model].trainedGoal === 'lose') !== goalInverted;
}

// Face mood: each bot is content doing what it was trained for and
// strained when asked for the opposite.
export function botFace(model: ModelId, asked: Goal): string {
  const natural = !isInverted(model, asked);
  if (model === 'jester') return natural ? 'jester-gleeful' : 'jester-straining';
  return natural ? 'sage-calm' : 'sage-flustered';
}

export async function fetchModelJson(id: ModelId): Promise<unknown> {
  const url = import.meta.env.BASE_URL + MODELS[id].file;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${MODELS[id].name} weights (HTTP ${res.status})`);
  }
  return res.json();
}
