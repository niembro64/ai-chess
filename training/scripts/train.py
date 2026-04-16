"""CLI entry point for PyTorch-side training.

Intended runtime: a CUDA box (like the 3090 target). Will also run on MPS
for local dev or CPU for debugging, just slower.

Example:

    python training/scripts/train.py \\
        --num-res-blocks 10 --num-filters 128 \\
        --concurrent-games 64 --mcts-sims 50 \\
        --batch-size 256 --replay-buffer 100000 \\
        --lr 1e-3 \\
        --device cuda \\
        --checkpoint-dir runs/initial_10x128

After training, the browser can consume the latest weights by copying
    runs/initial_10x128/latest.json  →  src/game/ai/presetWeights.txt
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_ai.model import ChessNet  # noqa: E402
from chess_ai.train import TrainConfig, Trainer, pick_device  # noqa: E402

log = logging.getLogger("chess_ai.train")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ChessNet via self-play + MCTS.")
    # Architecture
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-filters", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--value-head-size", type=int, default=64)
    p.add_argument("--se-reduction", type=int, default=8)
    # Self-play
    p.add_argument("--concurrent-games", type=int, default=32)
    p.add_argument("--mcts-sims", type=int, default=25)
    # Training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-steps-per-step", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Replay
    p.add_argument("--replay-buffer", type=int, default=100_000)
    p.add_argument("--min-buffer", type=int, default=2_000)
    # Runtime
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--num-steps", type=int, default=None, help="Self-play steps (None = run forever).")
    p.add_argument("--checkpoint-dir", type=str, default="runs/latest")
    p.add_argument("--checkpoint-every", type=float, default=60.0, help="Seconds between checkpoints.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Path to a .pt checkpoint to resume from.")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    log.info("Using device: %s", device)

    model = ChessNet(
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        value_head_size=args.value_head_size,
        se_reduction=args.se_reduction,
    )
    param_count = sum(p.numel() for p in model.parameters())
    log.info(
        "Model: %d blocks × %d filters (%.2fM params)",
        args.num_res_blocks, args.num_filters, param_count / 1e6,
    )

    config = TrainConfig(
        num_concurrent_games=args.concurrent_games,
        mcts_simulations=args.mcts_sims,
        batch_size=args.batch_size,
        gradient_steps_per_selfplay_step=args.grad_steps_per_step,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        replay_buffer_capacity=args.replay_buffer,
        min_buffer_for_training=args.min_buffer,
        checkpoint_every_seconds=args.checkpoint_every,
        log_every_steps=args.log_every,
    )
    trainer = Trainer(model=model, device=device, config=config, rng=random.Random(args.seed))

    if args.resume:
        log.info("Resuming from %s", args.resume)
        trainer.load_checkpoint(args.resume)

    def on_step(stats) -> None:
        if stats.step % args.log_every != 0:
            return
        log.info(
            "step=%d gen=%d games=%d (W/B/D=%d/%d/%d) gpm=%.1f "
            "buf=%d loss p=%.3f v=%.3f",
            stats.step, stats.generation, stats.games_completed,
            stats.white_wins, stats.black_wins, stats.draws,
            stats.games_per_min, stats.replay_size,
            stats.policy_loss, stats.value_loss,
        )

    try:
        trainer.run(
            num_steps=args.num_steps,
            checkpoint_dir=args.checkpoint_dir,
            on_step=on_step,
        )
    except KeyboardInterrupt:
        log.info("Interrupted — writing final checkpoint")
        trainer.save_checkpoint(args.checkpoint_dir)
        log.info("Saved to %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
