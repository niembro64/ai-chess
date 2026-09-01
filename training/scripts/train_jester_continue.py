"""Start a RESUMED Jester training run on this box.

Full Sage architecture + pipeline with MISÈRE incentives — the net
learns to LOSE. Needs the frozen Sage champion at
config.JESTER_OPPONENT_CKPT. Checkpoints land in runs_jester/latest.

    python scripts/train_jester_continue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _jester_launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(mode="continue")
