"""Continue Toy training from runs_toy/latest/latest.pt.

Errors out if no checkpoint exists — never silently starts over.

    python scripts/train_toy_continue.py
    python scripts/train_toy_continue.py --resume runs_toy/latest/archive/gen-5000.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _toy_launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(mode="continue")
