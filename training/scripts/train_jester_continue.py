"""Resume a protocol-3 JESTER Uncheck run on this box.

Resume is accepted only when ruleset, value convention, and protocol match.

    python scripts/train_jester_continue.py --ruleset uncheck-v1 --resume RUN/latest.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _jester_launcher import launch  # noqa: E402


if __name__ == "__main__":
    launch(mode="continue")
