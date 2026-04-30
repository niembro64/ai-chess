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
    # rather than burning a core on Python.
    #
    # Tuning history on the 4-core / 3090 box:
    #   * 8 workers × 24 games (192 in-flight) saturated inf-batch peak
    #     (avg 191.7 / peak 192) but left GPU at 39%, VRAM at 10%, CPU
    #     at 39%, and trainer at sleep_starved 50ms / 57ms iter (88%
    #     starved). Pipe was full but too narrow.
    #   * Bumped to 12 × 32 = 384 in-flight (2×). 3-4× core
    #     oversubscription is fine for I/O-bound workers; VRAM headroom
    #     absorbs the wider inference batches; the 3090 should now have
    #     enough work to push past 70%.
    # Watch `inf-batch avg / peak / inf-wait` after restart: if avg
    # tracks peak with wait below mp_batch_wait_ms, scale up further;
    # if avg stalls below peak, the wait-window or batching policy is
    # the new bottleneck.
    launch(
        mode="new",
        num_workers=12,         # 3× cores; MCTS is Rust, workers I/O-bound
        games_per_worker=32,    # 12×32 = 384 concurrent games (2× prior)
        batch_size=1024,        # VRAM at 2.5/24GB, headroom for more
    )
