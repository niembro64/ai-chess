"""Copy a trained checkpoint's weight JSON into the browser preset slot.

Writes `<checkpoint_dir>/latest.json` → `src/game/ai/presetWeights.txt`.
The browser's `presetWeights.ts` imports that file as a raw string and
parses it at runtime, so the only thing required is overwriting the file
and rebuilding (or hot-reloading in dev).

Usage:
    python training/scripts/deploy_to_browser.py runs/initial_10x128/latest.json
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESET_PATH = PROJECT_ROOT / "src" / "game" / "ai" / "presetWeights.txt"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python training/scripts/deploy_to_browser.py <path/to/latest.json>", file=sys.stderr)
        sys.exit(2)

    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        print(f"source weights JSON not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Validate it's a SerializedWeights JSON before copying
    import json
    with src.open() as f:
        data = json.load(f)
    if not {"shapes", "data"}.issubset(data):
        print(f"{src} doesn't look like a SerializedWeights JSON", file=sys.stderr)
        sys.exit(1)

    if "config" not in data:
        print(
            "WARNING: weights JSON has no embedded config; "
            "browser will fall back to shape-detection heuristics.",
            file=sys.stderr,
        )

    PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, PRESET_PATH)
    size_kb = PRESET_PATH.stat().st_size / 1024
    print(f"Wrote {PRESET_PATH} ({size_kb:.0f} KB)")
    print("Next: restart your dev server (`npm run dev`) or rebuild (`npm run build`).")


if __name__ == "__main__":
    main()
