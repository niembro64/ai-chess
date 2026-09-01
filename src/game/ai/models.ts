// Model registry + lazy weight loading.
//
// Weights live in public/models/ as static assets and are fetched ONLY
// when the player picks an opponent in the lobby — nothing model-sized
// ships in the JS bundle. (Sage's 15MB blob used to be imported into
// the bundle and downloaded by every visitor.)

export type ModelId = 'sage' | 'toy' | 'jester';

export const MODELS: Record<ModelId, {
  name: string;
  tagline: string;
  file: string;
  sims: number;
  // What the net was TRAINED to want. The lobby's GOAL INVERTED mode
  // pursues the opposite at play time (misère search flip) — see
  // AIPlayerOptions.
  trainedGoal: 'win' | 'lose';
}> = {
  sage: {
    name: 'Sage',
    tagline: 'Trained to checkmate opponent',
    file: 'models/sage.json',
    sims: 400,
    trainedGoal: 'win',
  },
  toy: {
    name: 'Toy',
    tagline: 'Trained to checkmate opponent — watch it think',
    file: 'models/toy.json',
    sims: 128,
    trainedGoal: 'win',
  },
  jester: {
    name: 'Jester',
    tagline: 'Trained to checkmate itself',
    file: 'models/jester.json',
    sims: 400,
    trainedGoal: 'lose',
  },
};

// --- Effort ------------------------------------------------------------
//
// EFFORT is the difficulty dial, and it moves the two levers that
// actually change how hard a bot tries: how much SEARCH it runs before
// moving, and how strictly it plays the search's best move. Search is
// where most of the playing strength lives — a single forward pass is
// the network's instinct, and it can't verify tactics or use the value
// head at all (the value of the current position is identical for every
// candidate move; only lookahead makes it discriminate). So LOW barely
// looks ahead AND wanders; HIGH runs the model's full search and always
// plays its top choice.
//
// Every level keeps SOME search, deliberately: the goal-inversion modes
// need a tree to plan a loss in. Zero-sim inversion would just take the
// argmin of the prior, which shuffles aimlessly instead of planning.

export type Effort = 'low' | 'medium' | 'high';

export const EFFORT_LEVELS: Record<Effort, {
  label: string;
  // Fraction of the model's full search budget.
  simsFraction: number;
  // Move-selection temperature over visit counts: 0 = always the top
  // choice, 1 = sample proportional to visits.
  temperature: number;
}> = {
  low: { label: 'Low', simsFraction: 0.05, temperature: 1.0 },
  medium: { label: 'Medium', simsFraction: 0.2, temperature: 0.35 },
  high: { label: 'High', simsFraction: 1, temperature: 0 },
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

export async function fetchModelJson(id: ModelId): Promise<unknown> {
  const url = import.meta.env.BASE_URL + MODELS[id].file;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${MODELS[id].name} weights (HTTP ${res.status})`);
  }
  return res.json();
}
