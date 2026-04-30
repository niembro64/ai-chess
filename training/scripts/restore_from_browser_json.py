"""Rebuild runs/latest/latest.pt from a browser-format weights JSON.

The browser preset (`src/game/ai/presetWeights.txt`) is the same JSON
that `latest.json` was at the time of `pull_from_ubuntu.sh --browser`.
This script unpacks it back into a PyTorch checkpoint that
`Trainer.load_checkpoint()` (and therefore `*_continue.py`) can resume
from. Use it when latest.pt is corrupt / lost / overwritten by a bad
run, but you still trust the deployed browser weights.

Caveats vs. resuming from a real .pt file:

  * Optimizer state (Adam m/v moment estimates) is not in the JSON, so
    those restart from zero. The first ~few hundred grad steps after
    resume will be slightly noisier than usual; weights converge back
    quickly with the LR schedule still in the right place.
  * stats.generation must be supplied via --gen. Without it, the
    LR schedule would walk back to its initial high value and burn
    the converged weights. The JSON doesn't carry the gen number;
    you set it from whatever the dashboard / commit message said when
    the JSON was deployed.

Usage:

    cd training
    # Default: reads ../src/game/ai/presetWeights.txt, writes
    # runs/latest/latest.pt with stats.generation=<N>.
    python scripts/restore_from_browser_json.py --gen 105000

    # Or point at any other JSON / output location:
    python scripts/restore_from_browser_json.py \\
        --source path/to/latest.json \\
        --out runs/latest/latest.pt \\
        --gen 105000
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
sys.path.insert(0, str(TRAINING_DIR))  # for `import config`

import config as cfg  # noqa: E402
from chess_ai.model import ChessNet  # noqa: E402
from chess_ai.train import TrainStats  # noqa: E402
from chess_ai.weight_io import import_weights  # noqa: E402


def _default_source() -> Path:
    """Browser preset file in the repo, the same JSON we ship to clients."""
    project_root = TRAINING_DIR.parent
    return project_root / "src" / "game" / "ai" / "presetWeights.txt"


def _default_out() -> Path:
    """Where Trainer.load_checkpoint expects a resume target by default."""
    return cfg.CHECKPOINT_DIR / "latest.pt"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--source", type=Path, default=_default_source(),
        help="Path to browser-format JSON (default: src/game/ai/presetWeights.txt)",
    )
    p.add_argument(
        "--out", type=Path, default=_default_out(),
        help=f"Output checkpoint path (default: {_default_out()})",
    )
    p.add_argument(
        "--gen", type=int, required=True,
        help="Generation count to restore (drives LR schedule). "
             "Set this to whatever gen the JSON was exported at — usually "
             "the dashboard 'gradient updates' value when --browser was last "
             "pulled. WITHOUT this, the LR schedule walks back to its "
             "initial value and destroys the trained weights.",
    )
    args = p.parse_args()

    src: Path = args.source
    out: Path = args.out
    if not src.exists():
        print(f"ERROR: source JSON not found: {src}", file=sys.stderr)
        return 1

    print(f"→ reading {src}")
    with src.open("r") as f:
        serialized = json.load(f)

    # Build a ChessNet matching the JSON's architecture so import_weights
    # can populate it tensor-by-tensor. The JSON's "config" dict is
    # authoritative — we don't trust the project's config.py here in case
    # it's been edited since the JSON was exported.
    arch = serialized["config"]
    model = ChessNet(
        num_res_blocks=arch["numResBlocks"],
        num_filters=arch["numFilters"],
        kernel_size=arch["kernelSize"],
        value_head_size=arch["valueHeadSize"],
        se_reduction=arch["seReduction"],
    )
    print(
        f"  arch: {arch['numResBlocks']} blocks × {arch['numFilters']} filters "
        f"(value_head={arch['valueHeadSize']}, se={arch['seReduction']})"
    )

    print("→ importing weights into ChessNet")
    import_weights(model, serialized)

    # Synthesize a TrainStats with the gen counter set, so Trainer
    # .load_checkpoint() restores it and the LR schedule sees the right
    # gen on the first step. Other fields are display-only.
    stats = TrainStats(generation=args.gen, target_gens=cfg.build_config().target_gens)

    # Match the structure save_checkpoint() writes. Optimizer state is
    # intentionally omitted — JSON doesn't carry it. aux_material is also
    # absent (the browser export ignores it). model_arch lets downstream
    # tools (compare_checkpoints, deploy_to_browser) self-describe.
    state = {
        "model_state_dict": model.state_dict(),
        "stats": stats.__dict__,
        "model_arch": {
            "num_res_blocks": model.num_res_blocks,
            "num_filters": model.num_filters,
            "kernel_size": model.kernel_size,
            "value_head_size": model.value_head_size,
            "se_reduction": model.se_reduction,
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ writing {out}")
    torch.save(state, out)

    print()
    print(f"Done. {out} now contains the browser weights at gen={args.gen}.")
    print("Resume training with the matching *_continue.py entrypoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
