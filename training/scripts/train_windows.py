"""Training entrypoint for the Windows/WSL box (9900K + 1080 Ti).

Thin wrapper around train.py with hardware-specific defaults:
  * 8 workers (9900K: 8 physical cores)
  * 16 games per worker
  * batch size 256 (fits comfortably in 11 GiB)

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
        num_workers=16,
        games_per_worker=16,
        batch_size=256,
    )
    train.main(parser)
