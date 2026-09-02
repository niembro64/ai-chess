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
// EFFORT is the difficulty dial: how much SEARCH the bot runs before
// moving. Search is where most of the playing strength lives — a single
// forward pass is only the network's instinct, and it can't verify
// tactics or use the value head at all (the value of the current
// position is identical for every candidate move; only lookahead makes
// it discriminate). So LOW sees barely any consequences while HIGH runs
// the model's full search.
//
// The bot always plays the best move its search found — there is no
// randomness knob. Every level keeps SOME search, deliberately: the
// goal-inversion modes need a tree to plan a loss in. Zero-sim
// inversion would just take the argmin of the prior, which shuffles
// aimlessly instead of planning.

export type Effort = 'low' | 'medium' | 'high';

export const EFFORT_LEVELS: Record<Effort, {
  label: string;
  // Fraction of the model's full search budget.
  simsFraction: number;
}> = {
  low: { label: 'Low', simsFraction: 0.05 },
  medium: { label: 'Medium', simsFraction: 0.2 },
  high: { label: 'High', simsFraction: 1 },
};

export function effortSims(id: ModelId, effort: Effort): number {
  return Math.max(8, Math.round(MODELS[id].sims * EFFORT_LEVELS[effort].simsFraction));
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

// --- Setup grid --------------------------------------------------------
//
// COLUMNS are the networks, by what their weights were TRAINED to do;
// ROWS are what we ASK them to do at play time. A cell is inverted
// whenever the asked goal differs from the trained one — inversion is
// the misère search flip, never "pick the worst-looking move".

export type Goal = 'win' | 'lose';

export const GRID_MODELS: readonly ModelId[] = ['sage', 'jester'];
export const GRID_ASKED: readonly Goal[] = ['win', 'lose'];

export function goalLabel(goal: Goal): string {
  return goal === 'win' ? 'WIN' : 'LOSE';
}

export function isInverted(model: ModelId, asked: Goal): boolean {
  return MODELS[model].trainedGoal !== asked;
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
