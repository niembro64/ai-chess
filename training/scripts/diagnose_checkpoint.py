"""Poke the model heads on mate-in-1 positions and report what's broken.

Usage:
    # scp the checkpoint from the server first, then:
    python scripts/diagnose_checkpoint.py /tmp/latest.json

    # or against a .pt checkpoint directly
    python scripts/diagnose_checkpoint.py runs/latest/latest.pt

Answers the question "why does eval keep returning 0-120-0?" by
splitting the failure into three independent components:

  (1) Raw policy head — given a mate-in-1 position, does the NN
      assign non-trivial probability to the mating move?
  (2) Raw value head  — does the NN think this position is winning
      (WDL triple skewed toward win)?
  (3) MCTS integration — given both heads, does the 60-sim search
      actually find and commit to the mating move?

Each failure mode has a different fix. Without splitting them, we're
layering interventions without knowing which layer is broken.
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

from chess_ai.encoding import NUM_PLANES, encode_board, move_to_index  # noqa: E402
from chess_ai.engine import apply_move, get_legal_moves  # noqa: E402
from chess_ai.eval_positions import build_eval_positions  # noqa: E402
from chess_ai.mcts import run_batched_mcts  # noqa: E402
from chess_ai.model import ChessNet, encoded_to_nchw  # noqa: E402
from chess_ai.selfplay import make_pytorch_evaluator  # noqa: E402
from chess_ai.weight_io import import_weights  # noqa: E402


def _load_model(path: Path) -> tuple[ChessNet, dict]:
    """Load a trained model from either latest.json or latest.pt."""
    if path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        cfg = data["config"]
        model = ChessNet(
            num_res_blocks=cfg["numResBlocks"],
            num_filters=cfg["numFilters"],
            kernel_size=cfg["kernelSize"],
            value_head_size=cfg["valueHeadSize"],
            se_reduction=cfg["seReduction"],
        )
        import_weights(model, data)
        return model, cfg
    state = torch.load(path, map_location="cpu", weights_only=False)
    arch = state["model_arch"]
    model = ChessNet(**arch)
    model.load_state_dict(state["model_state_dict"])
    return model, arch


def _move_to_uci(m, is_white: bool = True) -> str:
    """Board coordinates → UCI notation. Doesn't mirror — always reports
    the move in absolute-board terms so human chess readers can follow."""
    del is_white  # absolute coords for readability
    frank = 8 - m.from_pos.rank
    ffile = chr(ord("a") + m.from_pos.file)
    trank = 8 - m.to_pos.rank
    tfile = chr(ord("a") + m.to_pos.file)
    return f"{ffile}{frank}{tfile}{trank}"


def _policy_index_to_uci(idx: int, is_white: bool) -> str:
    """Reverse move_to_index: a canonical-view policy index → UCI
    in absolute-board terms."""
    from_sq = idx // 64
    to_sq = idx % 64
    fr, ff = from_sq // 8, from_sq % 8
    tr, tf = to_sq // 8, to_sq % 8
    if not is_white:
        fr, ff = 7 - fr, 7 - ff
        tr, tf = 7 - tr, 7 - tf
    return f"{chr(ord('a') + ff)}{8 - fr}{chr(ord('a') + tf)}{8 - tr}"


def _diagnose_position(model: ChessNet, position, mcts_sims: int) -> dict:
    state = position.state
    is_white = state.currentTurn == "white"

    # Find the mating move(s).
    mating_moves = []
    for m in get_legal_moves(state):
        if apply_move(state, m).status == "checkmate":
            mating_moves.append(m)
    mating_indices = [move_to_index(m, is_white) for m in mating_moves]
    mating_ucis = [_move_to_uci(m) for m in mating_moves]

    # --- Raw NN output ---
    model.eval()
    with torch.no_grad():
        board_flat = torch.from_numpy(encode_board(state)).unsqueeze(0)
        board_nchw = encoded_to_nchw(board_flat, NUM_PLANES)
        policy_t, wdl_t = model(board_nchw)
        policy = policy_t[0].numpy()
        wdl = wdl_t[0].numpy()

    mating_probs = [float(policy[i]) for i in mating_indices]
    sorted_idx = np.argsort(-policy)
    best_mating_rank = min(int(np.where(sorted_idx == i)[0][0]) + 1 for i in mating_indices)
    top5 = [(_policy_index_to_uci(int(i), is_white), float(policy[int(i)])) for i in sorted_idx[:5]]

    # --- MCTS result ---
    rng = random.Random(0)
    evaluator = make_pytorch_evaluator(model, torch.device("cpu"))
    mcts_result = run_batched_mcts([state], evaluator, mcts_sims, rng, temperatures=[0.0])[0]
    chosen_idx = move_to_index(mcts_result.move, is_white)
    mcts_found = chosen_idx in mating_indices
    mcts_mating_visits = [float(mcts_result.policy[i]) for i in mating_indices]

    return {
        "name": position.name,
        "stm": state.currentTurn,
        "wdl": wdl,
        "mating_ucis": mating_ucis,
        "top5": top5,
        "mating_probs": mating_probs,
        "best_mating_rank": best_mating_rank,
        "mcts_move": _move_to_uci(mcts_result.move),
        "mcts_found": mcts_found,
        "mcts_mating_visits": mcts_mating_visits,
        "mcts_root_value": mcts_result.root_value,
    }


def _print_report(reports: list[dict]) -> None:
    n = len(reports)
    for r in reports:
        print(f"\n{'─' * 78}")
        print(f"{r['name']}   (side to move: {r['stm']})")
        print(f"Mating move(s): {', '.join(r['mating_ucis'])}")
        w, d, l = r["wdl"]
        scalar = w - l
        print()
        print(f"  value head WDL:   W={w:.3f}  D={d:.3f}  L={l:.3f}   scalar v={scalar:+.3f}")
        print(f"    (mate-in-1 expects v ≈ +0.7 to +1.0 from stm's perspective)")
        print()
        print(f"  policy top 5:")
        for uci, prob in r["top5"]:
            tag = "  ← mate" if uci in r["mating_ucis"] else ""
            print(f"    {uci}   prob={prob:.4f}{tag}")
        print(f"  policy on mating move(s):")
        for uci, prob in zip(r["mating_ucis"], r["mating_probs"]):
            print(f"    {uci}   prob={prob:.4f}")
        print(f"    best mating move ranks #{r['best_mating_rank']} / 4,096")
        print()
        print(f"  MCTS ({len(r['mcts_mating_visits'])} mating move(s) tested):")
        print(f"    chose: {r['mcts_move']}   "
              f"{'← MATE ✓' if r['mcts_found'] else '(not a mate)'}")
        visits_str = ", ".join(f"{v:.3f}" for v in r["mcts_mating_visits"])
        print(f"    visit fraction on mating moves: {visits_str}")
        print(f"    root value: {r['mcts_root_value']:+.3f}")

    # --- Aggregate summary ---
    policy_rank1 = sum(1 for r in reports if r["best_mating_rank"] == 1)
    policy_top5 = sum(1 for r in reports if r["best_mating_rank"] <= 5)
    policy_top20 = sum(1 for r in reports if r["best_mating_rank"] <= 20)
    value_sees_win = sum(1 for r in reports if (r["wdl"][0] - r["wdl"][2]) > 0.3)
    mcts_found = sum(1 for r in reports if r["mcts_found"])

    avg_w = float(np.mean([r["wdl"][0] for r in reports]))
    avg_d = float(np.mean([r["wdl"][1] for r in reports]))
    avg_l = float(np.mean([r["wdl"][2] for r in reports]))

    print()
    print("=" * 78)
    print(f"SUMMARY  ({n} mate-in-1 positions tested)")
    print("=" * 78)
    print()
    print("RAW POLICY HEAD")
    print(f"  mating move at rank 1:    {policy_rank1}/{n}   ({100*policy_rank1/n:.0f}%)")
    print(f"  mating move in top 5:     {policy_top5}/{n}   ({100*policy_top5/n:.0f}%)")
    print(f"  mating move in top 20:    {policy_top20}/{n}   ({100*policy_top20/n:.0f}%)")
    print()
    print("RAW VALUE HEAD  (averaged across positions)")
    print(f"  W={avg_w:.3f}  D={avg_d:.3f}  L={avg_l:.3f}   scalar v̄={avg_w - avg_l:+.3f}")
    print(f"  positions where v > +0.3:   {value_sees_win}/{n}   ({100*value_sees_win/n:.0f}%)")
    print()
    print("MCTS  (60 sims)")
    print(f"  found the mating move:     {mcts_found}/{n}   ({100*mcts_found/n:.0f}%)")

    # --- Diagnosis ---
    print()
    print("─" * 78)
    print("DIAGNOSIS")
    print("─" * 78)
    policy_good = policy_top5 >= 0.6 * n
    value_good = value_sees_win >= 0.6 * n
    mcts_good = mcts_found >= 0.6 * n

    if policy_good and value_good and mcts_good:
        print("Everything looks fine. If eval still shows 0 wins, the problem is")
        print("elsewhere (eval scoring, color rotation, game loop bug).")
    elif policy_good and value_good and not mcts_good:
        print("Policy and value heads BOTH see the mate, but MCTS doesn't commit.")
        print("→ MCTS bottleneck. Try:")
        print("   - more sims in eval (eval_mcts_sims 60 → 200)")
        print("   - lower c_puct to let visit-value override policy prior")
        print("   - higher dirichlet_epsilon (more exploration at root)")
    elif policy_good and not value_good:
        print("Policy head sees the mate, but value head does NOT recognize the")
        print("position as winning.")
        print("→ Value head is collapsed or miscalibrated. Try:")
        print("   - value_draw_weight 0.3 → 0.1 (stronger class rebalancing)")
        print("   - mask cap-bucket games from value loss (keep their policy signal)")
    elif not policy_good and value_good:
        print("Value head knows it's winning, but policy has no idea which move to play.")
        print("→ Policy targets are noisy — MCTS visits during self-play aren't")
        print("  concentrating on good moves. Possible causes:")
        print("   - too few self-play MCTS sims (30 was cut to 60 but still low)")
        print("   - policy_label_smoothing diluting useful policy signal")
    elif avg_d > 0.85:
        print("Value head is severely collapsed — predicting ~draw on every")
        print(f"  position (average D = {avg_d:.3f}).")
        print("  Class weighting at 0.3 is not enough. Either:")
        print("   - drop to value_draw_weight=0.05")
        print("   - or mask cap-bucket games from value loss entirely")
        print("   - and consider: is the model ALSO not learning policy?")
    else:
        print("Both heads are broken to some degree. Training isn't working.")
        print(f"  policy top-5 rate: {100*policy_top5/n:.0f}%")
        print(f"  value-sees-win rate: {100*value_sees_win/n:.0f}%")
        print("  This usually means gradients aren't shaping the model at all.")
        print("  Check: is the replay buffer populated? Are grad steps actually firing?")
        print("  Verify value_loss is > 0.01 (not pinned to exactly zero).")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose model heads on mate-in-1 positions.")
    ap.add_argument("checkpoint", type=Path, help="Path to latest.json or latest.pt")
    ap.add_argument("--num-positions", type=int, default=8,
                    help="How many mate-in-1 positions to probe (default 8).")
    ap.add_argument("--mcts-sims", type=int, default=60,
                    help="MCTS simulations per position (default 60, matches eval).")
    args = ap.parse_args()

    print(f"Loading: {args.checkpoint}")
    model, cfg = _load_model(args.checkpoint)
    model.eval()

    blocks = cfg.get("numResBlocks", cfg.get("num_res_blocks"))
    filters = cfg.get("numFilters", cfg.get("num_filters"))
    print(f"Model: {blocks} blocks × {filters} filters")

    positions = [p for p in build_eval_positions() if p.difficulty == "mate-in-1"]
    positions = positions[: args.num_positions]
    print(f"Probing {len(positions)} mate-in-1 positions, {args.mcts_sims} MCTS sims each.")

    reports = [_diagnose_position(model, p, args.mcts_sims) for p in positions]
    _print_report(reports)


if __name__ == "__main__":
    main()
