"""Zero-option training entrypoint.

Reads all settings from `training/config.py`. Edit that file to change
anything; run this file as-is. Optionally pass `--resume` to continue
from a saved checkpoint.

Usage:

    python scripts/run.py                      # fresh run
    python scripts/run.py --resume runs/latest/latest.pt

Intended to be the single launch command for both local dev (laptop) and
the remote training box (Ubuntu + 3090). The syzygy path, device, and
number of workers are all picked up from config.py.
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
sys.path.insert(0, str(ROOT))  # so `import config` works

import config as cfg  # noqa: E402
from chess_ai.dashboard import DashboardLogger  # noqa: E402
from chess_ai.model import ChessNet  # noqa: E402
from chess_ai.train import Trainer, pick_device  # noqa: E402

log = logging.getLogger("chess_ai.train")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train ChessNet (config-driven).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pt checkpoint to resume from.")
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    device = pick_device(cfg.DEVICE)
    log.info("Using device: %s", device)

    # Tablebase setup (silent no-op when path is missing).
    from chess_ai import tablebase
    syzygy_path_str = str(cfg.SYZYGY_PATH) if cfg.SYZYGY_PATH else None
    if tablebase.open_tablebase(syzygy_path_str, cfg.SYZYGY_MAX_PIECES):
        log.info("Syzygy tablebase: %s (max %d pieces)",
                 syzygy_path_str, cfg.SYZYGY_MAX_PIECES)
    elif syzygy_path_str and Path(syzygy_path_str).exists():
        log.warning("Syzygy path exists but no tables loaded: %s", syzygy_path_str)

    model = ChessNet(
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        num_filters=cfg.NUM_FILTERS,
        kernel_size=cfg.KERNEL_SIZE,
        value_head_size=cfg.VALUE_HEAD_SIZE,
        se_reduction=cfg.SE_REDUCTION,
    )
    param_count = sum(p.numel() for p in model.parameters())
    log.info(
        "Model: %d blocks × %d filters (%.2fM params)",
        cfg.NUM_RES_BLOCKS, cfg.NUM_FILTERS, param_count / 1e6,
    )

    config = cfg.build_config()
    trainer = Trainer(model=model, device=device, config=config,
                      rng=random.Random(cfg.SEED))

    if args.resume:
        log.info("Resuming from %s", args.resume)
        trainer.load_checkpoint(args.resume)
    else:
        # Fresh run — nuke any stale champion.pt left over from a
        # previous training session. A leftover champion acts as an
        # unintended adversary: it's a snapshot from a totally different
        # training regime (possibly with bugs we've since fixed), yet
        # the eval system treats it as our benchmark. The plateau
        # counter and Δelo numbers then track progress against that
        # alien model instead of against our own training trajectory.
        # Simplest fix: delete it so the first eval bootstraps cleanly.
        champ_path = cfg.CHECKPOINT_DIR / "champion.pt"
        if champ_path.exists():
            champ_path.unlink()
            log.info("Cleared stale champion.pt (fresh run — next eval will bootstrap)")

    model_summary = cfg.model_summary_lines(
        lr=config.learning_rate,
        param_count=param_count,
        concurrent_games=config.num_concurrent_games,
        sims=config.mcts_simulations,
        batch=config.batch_size,
    )

    def plain_on_step(stats) -> None:
        if stats.step % config.log_every_steps != 0:
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

    def plain_on_eval(result: dict) -> None:
        if result.get("note") == "bootstrap":
            log.info("eval: champion bootstrapped at gen %d", result.get("gen", 0))
            return
        tag = "NEW CHAMPION" if result.get("new_champion") else "no change"
        log.info(
            "eval gen=%d  %d-%d-%d  score=%.3f  Δelo=%+.1f  %s  (plateau %d/%d)",
            result["gen"], result["wins"], result["draws"], result["losses"],
            result["score"], result["elo_diff"], tag,
            result.get("plateau_counter", 0), config.max_plateau_evals,
        )

    ckpt_dir = str(cfg.CHECKPOINT_DIR)

    if cfg.ENABLE_DASHBOARD:
        with DashboardLogger(
            ckpt_dir,
            model_summary=model_summary,
            device_summary=str(device),
            on_log=None,
        ) as dash:
            # Hook checkpoint saves so they show up in the events pane.
            _orig_save = trainer.save_checkpoint
            def save_and_log(directory):
                paths = _orig_save(directory)
                dash.log(f"checkpoint → {paths['json'].name} + {paths['pt'].name}")
                return paths
            trainer.save_checkpoint = save_and_log  # type: ignore[assignment]

            def eval_to_dash(result: dict) -> None:
                dash.on_eval(
                    result,
                    threshold=config.eval_score_threshold,
                    plateau_max=config.max_plateau_evals,
                )

            try:
                trainer.run(
                    num_steps=None,
                    checkpoint_dir=ckpt_dir,
                    on_step=dash.on_step,
                    on_eval=eval_to_dash,
                    on_eval_progress=dash.on_eval_progress,
                )
            except KeyboardInterrupt:
                dash.log("interrupted — writing final checkpoint")
                trainer.save_checkpoint(ckpt_dir)
                dash.log(f"saved to {ckpt_dir}")
            # Plateau-stop: the main loop exits quietly when
            # _stop_requested flips. Surface it to the events pane so
            # the user doesn't wonder why it ended.
            if trainer._stop_requested:
                dash.log(
                    f"PLATEAU STOP — {config.max_plateau_evals} consecutive "
                    f"failed evals. Final checkpoint written."
                )
                trainer.save_checkpoint(ckpt_dir)
    else:
        try:
            trainer.run(
                num_steps=None,
                checkpoint_dir=ckpt_dir,
                on_step=plain_on_step,
                on_eval=plain_on_eval,
            )
        except KeyboardInterrupt:
            log.info("Interrupted — writing final checkpoint")
            trainer.save_checkpoint(ckpt_dir)
            log.info("Saved to %s", ckpt_dir)
        if trainer._stop_requested:
            log.info(
                "PLATEAU STOP — %d consecutive failed evals. Final checkpoint written.",
                config.max_plateau_evals,
            )
            trainer.save_checkpoint(ckpt_dir)


if __name__ == "__main__":
    main()
