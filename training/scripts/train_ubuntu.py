"""Training entrypoint for the Ubuntu box (4-core CPU + RTX 3090).

Thin wrapper around train.py with hardware-specific defaults:
  * 4 workers (CPU has 4 cores — one per worker, none left idle)
  * 24 games per worker (3090 chews through big batches; keep it fed)
  * batch size 512 (3090 has 24 GiB, no reason to starve it)

All flags from train.py still work — defaults are just overridden here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import train  # noqa: E402


if __name__ == "__main__":
    parser = train.build_parser()
    parser.set_defaults(
        num_workers=4,
        games_per_worker=24,
        batch_size=512,
    )
    train.main(parser)
