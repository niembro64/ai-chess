"""Fresh training run on the Windows/WSL box (9900K + 1080 Ti).

Wipes any stale champion.pt / eval.csv so previous results don't
pollute the new trajectory; bootstraps a random-init champion at gen 0.
For warm-starting from the existing latest.pt, use train_windows_continue.py.

    python scripts/train_windows_new.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(
        mode="new",
        num_workers=16,         # 9900K: 8 physical / 16 logical cores
        games_per_worker=16,
        batch_size=256,         # fits comfortably in 11 GiB
    )
