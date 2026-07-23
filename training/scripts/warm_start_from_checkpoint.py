"""Build a warm-start checkpoint from a previous run's weights.

Takes the model WEIGHTS from an existing checkpoint (champion.pt or
latest.pt) and writes `<out>/latest.pt` that train_*_continue.py can
resume from — with everything else reset:

- optimizer state: dropped (load_checkpoint skips the missing key and
  keeps a fresh AdamW; the old moments belong to a different loss
  landscape — and champion.pt never had optimizer state anyway).
- stats/counters: zeroed EXCEPT stats.generation, set via --generation
  so the LR schedule resumes at a sane fine-tune rate. Resuming the old
  gen counter (e.g. 294k) would pin LR at the schedule's terminal floor
  and make retraining glacial; resetting to 0 would restart at 1e-3,
  hot enough to churn the pretrained policy. The default 30_000 lands
  on 3e-4 in config.py's schedule — a standard fine-tune rate.
- replay buffer: not part of checkpoints; starts empty either way.
- champion.pt / eval.csv in <out> are NOT touched. If champion.pt is
  absent, the first eval bootstraps the current model as champion; you
  can instead copy the source run's champion.pt into <out> to gate
  fine-tuning progress against the warm-start baseline itself.

Side effect on the dashboard: the progress bar starts at
generation/target_gens (e.g. 30%) — cosmetic only.

Usage:
    python scripts/warm_start_from_checkpoint.py \
        runs/2026-05-plateau-294k/champion.pt
    python scripts/warm_start_from_checkpoint.py old.pt \
        --generation 30000 --out runs/latest
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
    parser.add_argument("--generation", type=int, default=30_000,
                        help="Generation counter to seed — picks the resume "
                             "LR from config.py's schedule (default 30000)")
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
        "stats": {"generation": args.generation},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "latest.pt"
    torch.save(checkpoint, out_path)

    src_gen = state.get("champion_gen") or state.get("stats", {}).get("generation")
    print(f"warm-start written: {out_path}")
    print(f"  source:      {args.source} (gen {src_gen}, {n_params / 1e6:.2f}M params)")
    print(f"  arch:        {arch}")
    print(f"  generation:  {args.generation} (sets resume LR via config.py schedule)")
    print("Resume with the *_continue.py entrypoint for your box.")


if __name__ == "__main__":
    main()
