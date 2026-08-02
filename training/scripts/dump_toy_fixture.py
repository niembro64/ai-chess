"""Dump a TS-parity fixture for the Toy net: positions + this
checkpoint's outputs, consumed by scripts/verify_toy_weights.ts.

The fixture must match the checkpoint deployed to public/models/toy.json
— regenerate both from the same .pt.

    python scripts/dump_toy_fixture.py                       # runs_toy/latest/latest.pt
    python scripts/dump_toy_fixture.py --checkpoint path.pt --out f.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves  # noqa: E402
from chess_ai.toy import ToyNet, encode_toy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs_toy" / "latest" / "latest.pt")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tests" / "fixtures" / "toy_parity.json")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ToyNet()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    rng = random.Random(42)
    s = create_initial_game_state()
    s.status = "active"
    states = [s]
    for _ in range(7):
        moves = get_legal_moves(s)
        if not moves:
            break
        s = apply_move(s, rng.choice(moves))
        if s.status in ("checkmate", "stalemate", "draw"):
            break
        states.append(s)

    boards = np.stack([encode_toy(st) for st in states])
    x = torch.from_numpy(boards.reshape(-1, 8, 8, 6)).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        policies, wdl = model(x)
    values = (wdl[:, 0] - wdl[:, 2]).numpy()

    fixture = {
        "boards": [b.tolist() for b in boards],
        "policies": [p.tolist() for p in policies.numpy()],
        "values": [float(v) for v in values],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(fixture, f)
    print(f"fixture with {len(states)} positions -> {args.out}")


if __name__ == "__main__":
    main()
