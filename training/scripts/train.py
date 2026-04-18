"""CLI entry point for PyTorch-side training.

Intended runtime: a CUDA box (like the 3090 target). Will also run on MPS
for local dev or CPU for debugging, just slower.

Checkpoints always live in a single directory (default: `runs/latest/`).
Rerunning the command overwrites `latest.pt` + `latest.json` in that
directory — there is no "v1"/"v2" scheme. If you want to experiment with
a different config in parallel, point `--checkpoint-dir` somewhere else
for that single run; the default stays `runs/latest`.

Example (defaults shown):

    python training/scripts/train.py \\
        --num-res-blocks 10 --num-filters 128 \\
        --num-workers 3 --games-per-worker 24 --mcts-sims 30 \\
        --device cuda

After training, deploy the latest weights to the browser:

    python training/scripts/deploy_to_browser.py training/runs/latest/latest.json
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

from chess_ai.dashboard import DashboardLogger  # noqa: E402
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
    p.add_argument("--concurrent-games", type=int, default=32,
                   help="Single-process only: games advanced per step.")
    p.add_argument("--mcts-sims", type=int, default=25)
    # Multi-process self-play. Enabling --num-workers spawns that many CPU
    # worker processes + one GPU inference server that batches their eval
    # requests. Each worker runs --games-per-worker concurrent games.
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = single-process (legacy). >=1 enables MP mode.")
    p.add_argument("--games-per-worker", type=int, default=16,
                   help="MP mode: concurrent games per worker process.")
    p.add_argument("--mp-batch-wait-ms", type=float, default=5.0,
                   help="MP mode: how long the inference server waits for more "
                        "requests before dispatching a batch.")
    p.add_argument("--weight-broadcast-every", type=int, default=50,
                   help="MP mode: gradient steps between pushing fresh weights "
                        "to the inference server.")
    # Training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-steps-per-step", type=int, default=1)
    p.add_argument("--min-examples-between-grad-steps", type=int, default=32,
                   help="Rate-limit gradient updates: require this many new "
                        "training examples since the last step before taking "
                        "another. Prevents overtraining on a thin buffer.")
    p.add_argument("--target-gens", type=int, default=100_000,
                   help="'Well trained' milestone used for the dashboard "
                        "progress bar and ETA. Training continues past this "
                        "point — it's just a visible target.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Self-play scheduling
    p.add_argument("--endgame-start-prob", type=float, default=0.0,
                   help="Fraction of new episodes seeded from a theoretical "
                        "endgame (KQvK, KRvK, KPvK, ...). Curriculum: 0.40 "
                        "early (bootstrap), 0.15 mid, 0.05 late. Must satisfy "
                        "endgame_prob + random_prob <= 1.0 (the remainder is "
                        "standard openings).")
    p.add_argument("--random-start-prob", type=float, default=0.3,
                   help="Fraction of new episodes that start from a random "
                        "mid-/end-game position reached by a random walk "
                        "(short cap). Coverage tool. Single roll with "
                        "--endgame-start-prob; the remainder is standard "
                        "openings. Typical: 0.15-0.25.")
    p.add_argument("--temperature-threshold-plies", type=int, default=15,
                   help="Plies from episode start during which move selection "
                        "samples proportional to MCTS visits (τ=1). After this, "
                        "the game commits to argmax (τ→0).")
    # MCTS hyperparams (optional overrides; None = module default).
    p.add_argument("--c-puct", type=float, default=None,
                   help="PUCT exploration constant. Module default: 1.5. "
                        "Higher = more exploration. AlphaZero chess used ~2.")
    p.add_argument("--dirichlet-alpha", type=float, default=None,
                   help="Dirichlet noise concentration at the root. Module "
                        "default: 0.3. Lower = spikier noise.")
    p.add_argument("--dirichlet-epsilon", type=float, default=None,
                   help="Root-noise mixing weight. Module default: 0.25. "
                        "0 disables Dirichlet noise entirely.")
    # Data augmentation
    p.add_argument("--mirror-augment-prob", type=float, default=0.5,
                   help="Probability of left-right file mirror per sampled "
                        "batch row. 0 disables. 0.5 = free 2x data multiplier.")
    p.add_argument("--no-amp", dest="use_amp", action="store_false",
                   help="Disable mixed-precision (fp16) training on CUDA. "
                        "AMP gives ~2x forward/backward speedup on Pascal "
                        "and newer with no measurable quality loss.")
    p.set_defaults(use_amp=True)
    p.add_argument("--policy-label-smoothing", type=float, default=0.0,
                   help="Epsilon for uniform smoothing on the MCTS policy "
                        "target. 0.01-0.03 prevents the policy head from "
                        "collapsing to zero on unvisited moves. 0 disables.")
    p.add_argument("--aux-material-weight", type=float, default=0.1,
                   help="Weight on the auxiliary material-balance head loss. "
                        "KataGo-style multi-task training: dense gradient "
                        "signal on the trunk without polluting value/policy. "
                        "0 disables the aux head entirely.")
    # Replay
    p.add_argument("--replay-buffer", type=int, default=100_000)
    p.add_argument("--min-buffer", type=int, default=2_000)
    # Runtime
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--num-steps", type=int, default=None, help="Self-play steps (None = run forever).")
    p.add_argument("--checkpoint-dir", type=str, default="runs/latest",
                   help="Where to write latest.pt + latest.json + stats.csv. "
                        "Default is the single canonical spot (runs/latest/). "
                        "Override only for side-by-side experiments.")
    p.add_argument("--checkpoint-every", type=float, default=60.0, help="Seconds between checkpoints.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Path to a .pt checkpoint to resume from.")
    p.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        help="Disable the Rich TUI dashboard; use plain text logging instead.",
    )
    p.set_defaults(dashboard=True)
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
        num_workers=args.num_workers,
        games_per_worker=args.games_per_worker,
        mp_batch_wait_ms=args.mp_batch_wait_ms,
        weight_broadcast_every=args.weight_broadcast_every,
        batch_size=args.batch_size,
        gradient_steps_per_selfplay_step=args.grad_steps_per_step,
        min_examples_between_grad_steps=args.min_examples_between_grad_steps,
        target_gens=args.target_gens,
        endgame_start_prob=args.endgame_start_prob,
        random_start_prob=args.random_start_prob,
        temperature_threshold_plies=args.temperature_threshold_plies,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        mirror_augment_prob=args.mirror_augment_prob,
        use_amp=args.use_amp,
        policy_label_smoothing=args.policy_label_smoothing,
        aux_material_weight=args.aux_material_weight,
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

    def plain_on_step(stats) -> None:
        if stats.step % args.log_every != 0:
            return
        pct = stats.generation / max(1, stats.target_gens) * 100
        log.info(
            "gen=%d/%d (%.1f%%) gen/min=%.1f games=%d (%.1f/min) "
            "W/B/D/Cap=%d/%d/%d/%d buf=%d loss p=%.3f v=%.3f",
            stats.generation, stats.target_gens, pct,
            stats.gen_per_min,
            stats.games_completed, stats.games_per_min,
            stats.white_wins, stats.black_wins, stats.draws, stats.caps,
            stats.replay_size,
            stats.policy_loss, stats.value_loss,
        )

    model_summary = (
        f"blocks={args.num_res_blocks}\n"
        f"filters={args.num_filters}\n"
        f"value-head={args.value_head_size}\n"
        f"se-reduction={args.se_reduction}\n"
        f"params={param_count / 1e6:.2f}M\n"
        f"lr={args.lr:.1e}\n"
        f"games={args.concurrent_games}\n"
        f"sims={args.mcts_sims}\n"
        f"batch={args.batch_size}"
    )

    if args.dashboard:
        with DashboardLogger(
            args.checkpoint_dir,
            model_summary=model_summary,
            device_summary=str(device),
            on_log=None,   # Dashboard panel shows events; don't double-log to stdout
        ) as dash:
            def dashboard_on_step(stats) -> None:
                dash.on_step(stats)
            # Hook checkpoint saves so they show up in the events pane.
            _orig_save = trainer.save_checkpoint
            def save_and_log(directory):
                paths = _orig_save(directory)
                dash.log(f"checkpoint → {paths['json'].name} + {paths['pt'].name}")
                return paths
            trainer.save_checkpoint = save_and_log  # type: ignore[assignment]
            try:
                trainer.run(
                    num_steps=args.num_steps,
                    checkpoint_dir=args.checkpoint_dir,
                    on_step=dashboard_on_step,
                )
            except KeyboardInterrupt:
                dash.log("interrupted — writing final checkpoint")
                trainer.save_checkpoint(args.checkpoint_dir)
                dash.log(f"saved to {args.checkpoint_dir}")
    else:
        try:
            trainer.run(
                num_steps=args.num_steps,
                checkpoint_dir=args.checkpoint_dir,
                on_step=plain_on_step,
            )
        except KeyboardInterrupt:
            log.info("Interrupted — writing final checkpoint")
            trainer.save_checkpoint(args.checkpoint_dir)
            log.info("Saved to %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
