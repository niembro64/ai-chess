// Preset model weights for "Play vs Preset AI"
//
// The presetWeights.txt file contains raw JSON from "Copy Weights".
// Vite imports it as a string, we parse it at runtime.

import type { SerializedWeights } from './ChessNet';
import weightsRaw from './presetWeights.txt?raw';

let parsed: SerializedWeights | null = null;

try {
  parsed = JSON.parse(weightsRaw) as SerializedWeights;
} catch {
  parsed = null;
}

export const PRESET_WEIGHTS: SerializedWeights | null = parsed;
