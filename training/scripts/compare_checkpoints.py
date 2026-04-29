"""Play two checkpoints head-to-head and report Win/Draw/Loss + Elo estimate.

Running a tournament between checkpoints is the only way to know whether
gen N is actually stronger than gen M — loss going down doesn't guarantee
it, and self-play statistics can improve for degenerate reasons.

Usage:
    python scripts/compare_checkpoints.py \\
        --a runs/exp-old/latest.pt \\
        --b runs/latest/latest.pt \\
        --games 40 \\
        --mcts-sims 50 \\
        --device cuda

The script alternates colors (A as white on even games, black on odd) so
a first-move bias doesn't hide real strength differences.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import ChessGameState, apply_move, create_initial_game_state, get_legal_moves
from chess_ai.mcts import run_batched_mcts
from chess_ai.model import NUM_PLANES, ChessNet, encoded_to_nchw
from chess_ai.selfplay import _endgame_start, _random_start
from chess_ai.train import pick_device


def _infer_arch_from_state_dict(sd: dict) -> dict:
    """Fallback for checkpoints saved before model_arch was stored.

    Reads the shapes of known tensors to recover the architecture. We only
    need fields that ChessNet.__init__ accepts.
    """
    # Count res blocks via the "res_blocks.{N}.conv1.weight" keys.
    n_blocks = 0
    while f"res_blocks.{n_blocks}.conv1.weight" in sd:
        n_blocks += 1

    init_conv_w = sd["init_conv.weight"]  # [outC, inC, kH, kW]
    num_filters = init_conv_w.shape[0]
    kernel_size = init_conv_w.shape[2]

    value_fc1_w = sd["value_fc1.weight"]  # [out, in=64]
    value_head_size = value_fc1_w.shape[0]

    # SE reduction from the first res block's SE fc1.
    se_fc1_w = sd["res_blocks.0.se.fc1.weight"]  # [hidden, filters]
    hidden = se_fc1_w.shape[0]
    # ChessNet uses hidden = max(1, filters // reduction), so reduction = filters // hidden.
    se_reduction = max(1, num_filters // max(1, hidden))

    return {
        "num_res_blocks": n_blocks,
        "num_filters": num_filters,
        "kernel_size": kernel_size,
        "value_head_size": value_head_size,
        "se_reduction": se_reduction,
    }


def _load_model(path: Path, device: torch.device) -> ChessNet:
    """Reconstruct a ChessNet matching the saved arch, then load the state dict."""
    # weights_only=False because our checkpoints embed dataclasses
    # (TrainConfig, RewardWeights). PyTorch 2.6+ default refuses those.
    state = torch.load(path, map_location=device, weights_only=False)
    arch = state.get("model_arch")
    if arch is None:
        # Legacy checkpoint — infer from tensor shapes.
        arch = _infer_arch_from_state_dict(state["model_state_dict"])
    model = ChessNet(**arch)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def _make_evaluator(model: ChessNet, device: torch.device):
    def evaluate(boards: np.ndarray):
        with torch.no_grad():
            x_flat = torch.from_numpy(boards).to(device)
            x = encoded_to_nchw(x_flat, NUM_PLANES)
            policy, wdl = model(x)
        pol = policy.cpu().numpy()
        w = wdl.cpu().numpy()
        values = w[:, 0] - w[:, 2]
        return pol, values

    return evaluate


def _pick_starting_state(
    rng: random.Random, endgame_prob: float, random_prob: float
) -> ChessGameState:
    roll = rng.random()
    if roll < endgame_prob:
        return _endgame_start(rng)
    if roll < endgame_prob + random_prob:
        return _random_start(rng)
    s = create_initial_game_state()
    s.status = "active"
    return s


def play_game(
    evaluators: tuple,
    white_idx: int,
    mcts_sims: int,
    move_cap: int,
    rng: random.Random,
    start_state: ChessGameState,
) -> str:
    """Play one game between evaluators[0] and evaluators[1].

    evaluators[white_idx] plays white, the other plays black.
    Returns "a", "b", or "draw".
    """
    state = start_state
    moves_played = 0

    while True:
        if state.status in ("checkmate", "stalemate", "draw"):
            break
        if moves_played >= move_cap:
            return "draw"

        # Which evaluator's turn is it?
        if state.currentTurn == "white":
            ev_idx = white_idx
        else:
            ev_idx = 1 - white_idx

        # Greedy: τ=0 (argmax of visits).
        results = run_batched_mcts(
            [state], evaluators[ev_idx], mcts_sims, rng, temperatures=[0.0]
        )
        move = results[0].move
        state = apply_move(state, move)
        moves_played += 1

    if state.status == "checkmate":
        # The player to move lost (they have no moves).
        loser_color = state.currentTurn
        winner_color = "white" if loser_color == "black" else "black"
        winner_is_white = winner_color == "white"
        winner_idx = white_idx if winner_is_white else 1 - white_idx
        return "a" if winner_idx == 0 else "b"

    return "draw"


def _score_to_elo(score: float) -> float:
    """Convert A's score fraction in [0, 1] to an Elo difference (A minus B)."""
    # Clamp to avoid infinity at the boundaries.
    s = max(0.01, min(0.99, score))
    return -400.0 * math.log10(1.0 / s - 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Path to checkpoint A (.pt)")
    p.add_argument("--b", required=True, help="Path to checkpoint B (.pt)")
    p.add_argument("--games", type=int, default=40, help="Number of games to play")
    p.add_argument("--mcts-sims", type=int, default=50)
    p.add_argument("--move-cap", type=int, default=200, help="Plies before calling it a draw")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--endgame-start-prob", type=float, default=0.0,
                   help="Fraction of games starting from a theoretical endgame "
                        "(KQvK, etc.). Useful for evaluating endgame technique.")
    p.add_argument("--random-start-prob", type=float, default=0.0,
                   help="Fraction of games starting from a random-walk midgame.")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Loading A: {args.a}")
    model_a = _load_model(Path(args.a), device)
    print(f"Loading B: {args.b}")
    model_b = _load_model(Path(args.b), device)

    eval_a = _make_evaluator(model_a, device)
    eval_b = _make_evaluator(model_b, device)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wins_a = 0
    wins_b = 0
    draws = 0
    start_time = time.time()

    print(f"\nPlaying {args.games} games ({args.mcts_sims} sims/move, cap {args.move_cap})...\n")

    for g in range(args.games):
        # Alternate colors so A plays white on even games, black on odd.
        white_idx = g % 2
        start_state = _pick_starting_state(
            rng, args.endgame_start_prob, args.random_start_prob
        )
        result = play_game(
            (eval_a, eval_b), white_idx, args.mcts_sims, args.move_cap, rng, start_state
        )
        if result == "a":
            wins_a += 1
        elif result == "b":
            wins_b += 1
        else:
            draws += 1

        decisive = wins_a + wins_b
        total = wins_a + wins_b + draws
        rate = total / (time.time() - start_time + 1e-9)
        print(
            f"  game {g + 1:3d}/{args.games}  "
            f"A plays {'W' if white_idx == 0 else 'B'}  "
            f"result={result:5s}  "
            f"[A {wins_a} / draws {draws} / B {wins_b}]  "
            f"(decisive: {decisive}, rate: {rate:.2f} games/s)"
        )

    total = wins_a + wins_b + draws
    score_a = (wins_a + 0.5 * draws) / max(1, total)
    elo_diff = _score_to_elo(score_a)

    print(f"\n--- Result ---")
    print(f"A wins:   {wins_a}  ({wins_a / total:.1%})")
    print(f"Draws:    {draws}  ({draws / total:.1%})")
    print(f"B wins:   {wins_b}  ({wins_b / total:.1%})")
    print(f"A score:  {score_a:.3f}")
    print(f"Elo diff: A - B = {elo_diff:+.1f}")

    # 95% CI via normal approximation on score. Not rigorous for small N.
    if total >= 5:
        variance = score_a * (1 - score_a) / total
        stderr = math.sqrt(variance)
        ci_low = _score_to_elo(max(0.01, score_a - 1.96 * stderr))
        ci_high = _score_to_elo(min(0.99, score_a + 1.96 * stderr))
        print(f"  95% CI:   [{ci_low:+.1f}, {ci_high:+.1f}]")


if __name__ == "__main__":
    main()
