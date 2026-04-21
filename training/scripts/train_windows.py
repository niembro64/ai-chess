"""Training entrypoint for the Windows/WSL box (9900K + 1080 Ti).

Reads everything from `training/config.py` except the three hardware-
specific knobs set below. Use `--resume <ckpt>` to continue from a saved
checkpoint.

    python scripts/train_windows.py                # fresh run
    python scripts/train_windows.py --resume runs/latest/latest.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(
        num_workers=16,         # 9900K: 8 physical / 16 logical cores
        games_per_worker=16,
        batch_size=256,         # fits comfortably in 11 GiB
    )
