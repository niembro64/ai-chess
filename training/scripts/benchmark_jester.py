"""Read-only held-out selfmate/search-depth benchmark; never promotes weights."""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from diagnose_checkpoint import _load_model
from chess_ai.inverted import augmented_selfmates, selfmate_positions
from chess_ai.jester_eval import Match, move_uci, play_matches
from chess_ai.mcts import run_batched_mcts, set_mcts_params
from chess_ai.selfplay import make_pytorch_evaluator
from chess_ai.engine import position_key
from chess_ai.train import pick_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sims", type=int, nargs="+", default=[96, 256, 400])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-conversions", action="store_true", help="Play complete resistant tactics, including held-out geometry")
    parser.add_argument(
        "--fumbler-games",
        type=int,
        default=0,
        help="Optional cooperative diagnostic, never used for promotion",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _ = _load_model(args.checkpoint)
    device = pick_device(args.device)
    model.to(device).eval()
    evaluator = make_pytorch_evaluator(model, device)
    set_mcts_params(c_puct=1.5, fpu_reduction=0.4)
    positions = selfmate_positions("eval")
    for sims in args.sims:
        start = time.monotonic()
        results = run_batched_mcts(
            [p.state for p in positions],
            evaluator,
            sims,
            random.Random(0),
            temperatures=[0] * len(positions),
            dirichlet_epsilon=0,
            invert_turns=["both"] * len(positions),
            position_counts=[{position_key(p.state): 1} for p in positions],
        )
        scores = [move_uci(r.move) in p.winning_moves for r, p in zip(results, positions)]
        by_depth = {
            d: sum(ok for p, ok in zip(positions, scores) if p.plies == d)
            / sum(p.plies == d for p in positions)
            for d in (2, 4, 6)
        }
        print(
            json.dumps(
                dict(
                    checkpoint=str(args.checkpoint),
                    device=str(device),
                    sims=sims,
                    solved=sum(scores),
                    total=len(scores),
                    accuracy=sum(scores) / len(scores),
                    by_plies=by_depth,
                    seconds=round(time.monotonic() - start, 2),
                )
            ),
            flush=True,
        )
    if args.full_conversions:
        geometry = augmented_selfmates("eval")
        extra = [p for depth in (2, 4, 6) for p in random.Random(910 + depth).sample(
            [p for p in geometry if p.plies == depth], 6)]
        starts = list(positions) + extra
        played = play_matches(
            [Match(p.state, p.state.currentTurn, evaluator, f"conversion/{i}", p.difficulty)
             for i, p in enumerate(starts)], evaluator, args.sims[-1], 12, 32)
        print(json.dumps(dict(checkpoint=str(args.checkpoint), diagnostic="resistant-full-conversion",
                              wins=sum(r == "win" for _, r in played), games=len(played),
                              geometry_wins=sum(r == "win" for _, r in played[len(positions):]),
                              geometry_games=len(extra))), flush=True)
    if args.fumbler_games:
        if args.fumbler_games < 2 or args.fumbler_games % 2:
            parser.error("--fumbler-games must be an even number for paired colors")
        from chess_ai.eval_positions import build_rotating_opening_positions

        openings = build_rotating_opening_positions(args.fumbler_games // 2, random.Random(20260906))
        played = play_matches(
            [Match(p.state.copy(), color, evaluator, f"helper/{i}/{color}", "helper", helper=True)
             for i, p in enumerate(openings) for color in ("white", "black")],
            evaluator, args.sims[-1], 120, 32)
        print(json.dumps(dict(checkpoint=str(args.checkpoint), diagnostic="cooperative-fumbler-only",
                              games=len(played), **{outcome: sum(r == outcome for _, r in played)
                                                   for outcome in ("win", "loss", "draw", "cap")})), flush=True)



if __name__ == "__main__":
    main()
