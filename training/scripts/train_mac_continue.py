"""Continue training on the Mac (Apple Silicon, MPS) from the latest checkpoint.

Auto-resumes from `runs/latest/latest.pt`. Errors out if the file is
missing — never silently falls back to a fresh run. For starting from
scratch, use train_mac_new.py.

    python scripts/train_mac_continue.py
    python scripts/train_mac_continue.py --resume runs/latest/archive/gen-50000.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(
        mode="continue",
        num_workers=20,
        games_per_worker=16,
        batch_size=1024,
        mp_batch_wait_ms=25.0,
    )
