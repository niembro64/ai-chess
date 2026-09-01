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

export async function fetchModelJson(id: ModelId): Promise<unknown> {
  const url = import.meta.env.BASE_URL + MODELS[id].file;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${MODELS[id].name} weights (HTTP ${res.status})`);
  }
  return res.json();
}
