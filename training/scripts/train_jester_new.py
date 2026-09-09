"""Start a fresh protocol-3 JESTER Uncheck run on this box.

Use --init-from to copy every legacy JESTER model tensor while starting
fresh optimizer, replay, counters, curriculum, and evaluation state.

    python scripts/train_jester_new.py --ruleset uncheck-v1 --init-from SOURCE.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _jester_launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(mode="new")
