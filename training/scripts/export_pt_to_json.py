"""Convert a PyTorch checkpoint (.pt) to the browser SerializedWeights JSON.

The trainer auto-writes `latest.json` next to `latest.pt` on every
checkpoint, but champions / archives are only saved as `.pt`. When the
challenger has regressed past the champion (plateau detector still ticking
upward), the weights you want in the browser are the champion's, not
latest's. This script regenerates the browser JSON from any .pt.

Inverse of `restore_from_browser_json.py`.

Usage:

    cd training
    # Default: champion.pt → champion.json next to it.
    python scripts/export_pt_to_json.py runs/latest/champion.pt

    # Explicit output:
    python scripts/export_pt_to_json.py \\
        runs/latest/archive/gen-294000.pt \\
        --out runs/latest/archive/gen-294000.json

The output JSON is the same format as `latest.json` — feed it directly
to `deploy_to_browser.py` or copy into `src/game/ai/presetWeights.txt`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRAINING_DIR / "src"))
sys.path.insert(0, str(TRAINING_DIR))

import config as cfg  # noqa: E402
from chess_ai.model import ChessNet  # noqa: E402
from chess_ai.weight_io import export_weights  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("pt", type=Path, help="Source .pt checkpoint")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path (default: source path with .json extension)",
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate to embed in the JSON (informational only). "
             "Default: pull from saved stats.current_lr if present, else 0.",
    )
    args = p.parse_args()

    src: Path = args.pt
    if not src.exists():
        print(f"ERROR: source .pt not found: {src}", file=sys.stderr)
        return 1
    out: Path = args.out or src.with_suffix(".json")

    print(f"→ loading {src}")
    # weights_only=False: see Trainer.load_checkpoint comment.
    state = torch.load(src, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        print(f"ERROR: {src} has no model_state_dict (not a trainer checkpoint?)",
              file=sys.stderr)
        return 1

    # Architecture: prefer the embedded model_arch, fall back to current
    # config.py. The embedded arch is authoritative because the .pt was
    # saved by THIS run; config.py may have been edited since.
    arch = state.get("model_arch") or {}
    num_res_blocks = int(arch.get("num_res_blocks", cfg.NUM_RES_BLOCKS))
    num_filters    = int(arch.get("num_filters",    cfg.NUM_FILTERS))
    kernel_size    = int(arch.get("kernel_size",    cfg.KERNEL_SIZE))
    value_head     = int(arch.get("value_head_size", cfg.VALUE_HEAD_SIZE))
    se_reduction   = int(arch.get("se_reduction",   cfg.SE_REDUCTION))
    print(
        f"  arch: {num_res_blocks} blocks × {num_filters} filters "
        f"(value_head={value_head}, se={se_reduction})"
    )

    model = ChessNet(
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        kernel_size=kernel_size,
        value_head_size=value_head,
        se_reduction=se_reduction,
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Resolve LR for the JSON header. Informational only — the browser
    # doesn't use it for anything beyond display in the model panel.
    lr = args.lr
    if lr is None:
        saved_stats = state.get("stats")
        if isinstance(saved_stats, dict):
            lr = float(saved_stats.get("current_lr") or 0.0)
        else:
            lr = 0.0

    print("→ exporting weights")
    serialized = export_weights(model, learning_rate=lr)

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ writing {out}")
    with out.open("w") as f:
        json.dump(serialized, f)

    size_kb = out.stat().st_size / 1024
    print()
    print(f"Done. {out} ({size_kb:.0f} KB)")
    print("Next: python scripts/deploy_to_browser.py "
          f"{out.relative_to(TRAINING_DIR) if out.is_relative_to(TRAINING_DIR) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
