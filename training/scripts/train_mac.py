"""Training entrypoint for the Mac (Apple Silicon, MPS backend).

Reads everything from `training/config.py` except the hardware-specific
knobs set below. Use `--resume <ckpt>` to continue from a saved
checkpoint.

    python scripts/train_mac.py                    # fresh run
    python scripts/train_mac.py --resume runs/latest/latest.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _launcher import launch  # noqa: E402


if __name__ == "__main__":
    # Tuned for M1 Max (10 cores, 32GB unified memory, MPS).
    #
    # The dominant cost on MPS is kernel-launch overhead, NOT per-element
    # compute — forward at batch=384 was measured at ~617ms (1.6ms/sample,
    # 60-80× slower per-sample than a 3090 at batch=1024). The win comes
    # from fatter batches: more samples per kernel launch amortizes that
    # fixed overhead. So:
    #
    # - num_workers=20 (2× cores) → deeper inference queue so the server
    #   has more requests to coalesce each fire-step.
    # - games_per_worker=16 → 20×16 = 320 concurrent games; more parallel
    #   MCTS trees means each inference-server wait window fills faster.
    # - batch_size=1024 → amortize MPS overhead on the gradient step.
    #   Unified memory has 16GB+ headroom over baseline OS usage.
    # - mp_batch_wait_ms=25 → let the inference server coalesce longer
    #   before firing a forward pass. On CUDA we run 12ms; MPS needs
    #   ~2× that window to build batches that justify the kernel cost.
    launch(
        num_workers=20,
        games_per_worker=16,
        batch_size=1024,
        mp_batch_wait_ms=25.0,
    )
