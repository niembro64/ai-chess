"""Build a warm-start checkpoint from a previous run's weights.

Takes the model WEIGHTS from an existing checkpoint (champion.pt or
latest.pt) and writes `<out>/latest.pt` that train_*_continue.py can
resume from — with everything else reset:

- optimizer state: dropped (load_checkpoint skips the missing key and
  keeps a fresh AdamW; the old moments belong to a different loss
  landscape — and champion.pt never had optimizer state anyway).
- stats/counters: zeroed EXCEPT stats.generation (--generation, picks
  the resume LR from config.py's schedule) and
  stats.value_warmup_until_gen (--value-warmup, see below).
- replay buffer: not part of checkpoints; starts empty either way.
- champion.pt / eval.csv in <out> are NOT touched. Copy the source
  run's champion.pt into <out> to gate fine-tuning progress against
  the warm-start baseline itself.

LESSON FROM THE JULY 2026 FINE-TUNE (default --generation 85000):
resuming converged weights at 3e-4 with a fresh AdamW lost ~170 Elo
within the first 10k steps and never recovered — fresh Adam moments
make early updates full-sized for every parameter, and a value head
facing a new label regime pumps huge gradients through the shared
trunk. Resume at the LR the weights were actually converged at
(85_000 → 3e-5 on the current schedule) and let --value-warmup handle
the value head's catch-up instead.

--value-warmup N freezes trunk + policy for the first N generations
after the warm start: only value_* parameters train, at the hot
TrainConfig.value_warmup_lr (default 3e-4), so the value head adapts
to its new labels in isolation before full fine-tuning begins at the
schedule LR. 0 disables.

Side effect on the dashboard: the progress bar starts at
generation/target_gens — cosmetic only.

Usage:
    python scripts/warm_start_from_checkpoint.py \
        runs/2026-05-plateau-294k/champion.pt
    python scripts/warm_start_from_checkpoint.py old.pt \
        --generation 85000 --value-warmup 4000 --out runs/latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write <out>/latest.pt seeding a fine-tune run from "
                    "an existing checkpoint's model weights."
    )
    parser.add_argument("source", type=Path,
                        help="Existing checkpoint (.pt) to take weights from")
    parser.add_argument("--generation", type=int, default=85_000,
                        help="Generation counter to seed — picks the resume "
                             "LR from config.py's schedule. Default 85000 "
                             "(→ 3e-5): resume converged weights at the LR "
                             "they were converged at, NOT hotter (see "
                             "module docstring for the 170-Elo lesson).")
    parser.add_argument("--value-warmup", type=int, default=4_000,
                        help="Freeze trunk+policy for this many generations "
                             "so the value head adapts to its new label "
                             "regime in isolation first (at "
                             "TrainConfig.value_warmup_lr). 0 disables. "
                             "Default 4000.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "latest",
                        help="Output run directory (default runs/latest)")
    args = parser.parse_args()

    state = torch.load(args.source, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise SystemExit(f"{args.source} has no model_state_dict")

    arch = state.get("model_arch")
    if arch is None:
        raise SystemExit(
            f"{args.source} has no model_arch — too old to warm-start safely"
        )

    n_params = sum(v.numel() for v in state["model_state_dict"].values())
    checkpoint = {
        "model_state_dict": state["model_state_dict"],
        "model_arch": arch,
        # No optimizer_state_dict on purpose — see module docstring.
        "stats": {
            "generation": args.generation,
            "value_warmup_until_gen": (
                args.generation + args.value_warmup if args.value_warmup > 0 else 0
            ),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "latest.pt"
    torch.save(checkpoint, out_path)

    src_gen = state.get("champion_gen") or state.get("stats", {}).get("generation")
    print(f"warm-start written: {out_path}")
    print(f"  source:       {args.source} (gen {src_gen}, {n_params / 1e6:.2f}M params)")
    print(f"  arch:         {arch}")
    print(f"  generation:   {args.generation} (sets resume LR via config.py schedule)")
    if args.value_warmup > 0:
        print(f"  value-warmup: gens {args.generation}..{args.generation + args.value_warmup} "
              f"(trunk+policy frozen, value head only)")
    else:
        print("  value-warmup: disabled")
    print("Resume with the *_continue.py entrypoint for your box.")


if __name__ == "__main__":
    main()
