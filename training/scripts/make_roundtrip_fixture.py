"""Generate the round-trip fixture for PyTorch ↔ TF.js parity testing.

Produces three files under `training/tests/fixtures/`:

  roundtrip_weights.json         # SerializedWeights exported from PyTorch
  roundtrip_boards.json          # N encoded boards (also produced by Python
                                 # encoder, proven equal to TS via parity test)
  roundtrip_pytorch_outputs.json # PyTorch forward pass outputs on those boards

The TS side (`scripts/verify_weights.ts`) consumes the first two and emits
`roundtrip_ts_outputs.json`; `test_weight_roundtrip.py` then asserts the
outputs match to within 1e-4.

Run:
    python training/scripts/make_roundtrip_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_ai.encoding import encode_board  # noqa: E402
from chess_ai.engine import ChessGameState  # noqa: E402
from chess_ai.model import NUM_PLANES, ChessNet, encoded_to_nchw  # noqa: E402
from chess_ai.weight_io import export_weights  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures"
PARITY_FIXTURE = FIXTURE_DIR / "parity_positions.json"
WEIGHTS_OUT = FIXTURE_DIR / "roundtrip_weights.json"
BOARDS_OUT = FIXTURE_DIR / "roundtrip_boards.json"
PY_OUT = FIXTURE_DIR / "roundtrip_pytorch_outputs.json"

# Config picked to exercise the TF.js GPU forward path (numFilters > 32) and
# hit the round-trip with meaningfully varied layer shapes.
CONFIG = dict(
    num_res_blocks=4,
    num_filters=64,
    kernel_size=3,
    value_head_size=32,
    se_reduction=8,
)
N_POSITIONS = 16


def main() -> None:
    if not PARITY_FIXTURE.exists():
        sys.exit(
            f"Missing parity fixture at {PARITY_FIXTURE}. Run `npm run dump-parity` first."
        )

    with PARITY_FIXTURE.open() as f:
        parity = json.load(f)
    positions = parity["positions"][:N_POSITIONS]
    assert parity["num_planes"] == NUM_PLANES

    torch.manual_seed(42)
    model = ChessNet(**CONFIG)
    # Randomize BN running stats too, so the test catches mismatched γ/β/μ/σ² ordering.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 1.5)
    model.eval()

    encoded_batch = []
    for pos in positions:
        state = ChessGameState.from_dict(pos["state"])
        encoded_batch.append(encode_board(state))
    encoded = np.stack(encoded_batch).astype(np.float32)

    x_flat = torch.from_numpy(encoded)  # [B, 8*8*NUM_PLANES]
    x = encoded_to_nchw(x_flat, NUM_PLANES)

    with torch.no_grad():
        policy, wdl = model(x)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    with WEIGHTS_OUT.open("w") as f:
        json.dump(export_weights(model), f)

    with BOARDS_OUT.open("w") as f:
        json.dump({"boards": encoded.tolist()}, f)

    with PY_OUT.open("w") as f:
        json.dump(
            {
                "config": {k: v for k, v in CONFIG.items()},
                "policies": policy.cpu().numpy().tolist(),
                "wdls": wdl.cpu().numpy().tolist(),
            },
            f,
        )

    print(f"Wrote: {WEIGHTS_OUT.name}, {BOARDS_OUT.name}, {PY_OUT.name}")
    print("Next: run `npm run verify-weights` from the project root, then `pytest`.")


if __name__ == "__main__":
    main()
