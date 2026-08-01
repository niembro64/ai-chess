// Model registry + lazy weight loading.
//
// Weights live in public/models/ as static assets and are fetched ONLY
// when the player picks an opponent in the lobby — nothing model-sized
// ships in the JS bundle. (Sage's 15MB blob used to be imported into
// the bundle and downloaded by every visitor.)

export type ModelId = 'sage' | 'toy';

export const MODELS: Record<ModelId, {
  name: string;
  tagline: string;
  file: string;
  sims: number;
}> = {
  sage: {
    name: 'Sage',
    tagline: 'Strong AI - No Brain Visuals',
    file: 'models/sage.json',
    sims: 400,
  },
  toy: {
    name: 'Toy',
    tagline: 'Weak AI - Watch Input and Output Activations',
    file: 'models/toy.json',
    sims: 128,
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
