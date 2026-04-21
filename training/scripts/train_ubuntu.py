"""Training entrypoint for the Ubuntu box (4-core CPU + RTX 3090).

Reads everything from `training/config.py` except the three hardware-
specific knobs set below. Use `--resume <ckpt>` to continue from a saved
checkpoint.

    python scripts/train_ubuntu.py                 # fresh run
    python scripts/train_ubuntu.py --resume runs/latest/latest.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(
        num_workers=4,          # one per CPU core
        games_per_worker=24,    # 3090 eats big batches; keep it fed
        batch_size=512,         # 24 GiB VRAM, no reason to starve it
    )
