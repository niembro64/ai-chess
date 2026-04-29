"""Continue training on the Windows/WSL box from the latest checkpoint.

Auto-resumes from `runs/latest/latest.pt`. Errors out if the file is
missing — never silently falls back to a fresh run. For starting from
scratch, use train_windows_new.py.

    python scripts/train_windows_continue.py
    python scripts/train_windows_continue.py --resume runs/latest/archive/gen-50000.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(
        mode="continue",
        num_workers=16,
        games_per_worker=16,
        batch_size=256,
    )
