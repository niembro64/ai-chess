"""Start a FRESH Toy training run (random init) on this box.

Same pipeline, dashboard, and helper suite as Sage — just the tiny
6-plane ToyNet and toy-scaled knobs (config.build_toy_config).
Checkpoints land in runs_toy/latest; deploy to the site with:
    cp runs_toy/latest/latest.json ../public/models/toy.json

    python scripts/train_toy_new.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _toy_launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(mode="new")
