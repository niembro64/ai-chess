"""Fresh training run on the Ubuntu box (4-core CPU + RTX 3090).

Wipes any stale champion.pt / eval.csv so previous results don't
pollute the new trajectory; bootstraps a random-init champion at gen 0.
For warm-starting from the existing latest.pt, use train_ubuntu_continue.py.

    python scripts/train_ubuntu_new.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    # Post-Rust-MCTS tuning: each worker's MCTS is ~20× cheaper now, so
    # workers spend most of their time waiting on NN evals (I/O-bound)
    # rather than burning a core on Python. Oversubscribing cores by 2×
    # keeps the inference-server queue deeper, which lets it build
    # bigger GPU batches (3090 was at ~33% util on first restart).
    launch(
        mode="new",
        num_workers=8,          # 2× cores; MCTS is Rust, workers are I/O-bound
        games_per_worker=24,    # 8×24 = 192 concurrent games → big batches
        batch_size=1024,        # was 512; VRAM at 1.2/24GB, headroom to spare
    )
