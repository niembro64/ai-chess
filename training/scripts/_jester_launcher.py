"""Shared launcher for the Jester entrypoints (train_jester_new / _continue).

Mirrors scripts/_toy_launcher.py but builds the full Sage-architecture
ChessNet with MISÈRE incentives: search selection inverts at the
jester's plies while every value label stays truthful (see
chess_ai/mcts.py MCTSSearch).

The goal is to get its OWN king checkmated before the opponent gets
theirs. Self-play is therefore mostly jester-vs-jester mirror games —
the matchup the shipped bot meets, where the loss must be forced
against an opponent who refuses to deliver it — with a sparring side
holding a sustained temperature so the games actually finish. Eval
gating plays the champion jester and counts games the challenger LOSES
as wins. Checkpoints land in runs_jester/latest.
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


def launch(*, mode: str) -> None:
    if mode not in ("new", "continue"):
        raise ValueError(f"mode must be 'new' or 'continue', got {mode!r}")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train Jester (misère ChessNet).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Explicit checkpoint path (overrides mode-based "
                             "auto-detection).")
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    device = pick_device(cfg.DEVICE)
    log.info("Using device: %s", device)

    from chess_ai import tablebase
    syzygy_path_str = str(cfg.SYZYGY_PATH) if cfg.SYZYGY_PATH else None
    if tablebase.open_tablebase(syzygy_path_str, cfg.SYZYGY_MAX_PIECES):
        log.info("Syzygy tablebase: %s (max %d pieces)",
                 syzygy_path_str, cfg.SYZYGY_MAX_PIECES)

    config = cfg.build_jester_config()

    opponent = Path(config.jester_opponent_checkpoint)
    if not opponent.exists():
        raise FileNotFoundError(
            f"Jester needs the frozen winner checkpoint at {opponent} "
            f"(the Sage champion). Point config.JESTER_OPPONENT_CKPT at a "
            f"valid ChessNet .pt."
        )

    model = ChessNet(
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        num_filters=cfg.NUM_FILTERS,
        kernel_size=cfg.KERNEL_SIZE,
        value_head_size=cfg.VALUE_HEAD_SIZE,
        se_reduction=cfg.SE_REDUCTION,
    )
    param_count = sum(p.numel() for p in model.parameters())
    log.info("Jester (misère ChessNet): %d blocks × %d filters (%.2fM params); "
             "frozen opponent: %s",
             cfg.NUM_RES_BLOCKS, cfg.NUM_FILTERS, param_count / 1e6, opponent)

    trainer = Trainer(model=model, device=device, config=config,
                      rng=random.Random(cfg.SEED))

    ckpt_dir_path = cfg.CHECKPOINT_DIR_JESTER
    auto_latest = ckpt_dir_path / "latest.pt"
    if args.resume:
        resume_path: Path | None = Path(args.resume)
    elif mode == "continue":
        if not auto_latest.exists():
            raise FileNotFoundError(
                f"continue mode requires {auto_latest} to exist. "
                f"Start with train_jester_new.py, or pass --resume <path>."
            )
        resume_path = auto_latest
    else:
        resume_path = None

    if resume_path is not None:
        log.info("Resuming from %s", resume_path)
        trainer.load_checkpoint(str(resume_path))
    else:
        # Fresh run: clear stale champion/eval so the trajectory is clean,
        # then crown the random-init weights so the first eval measures
        # real progress. Same housekeeping as Sage's _launcher.
        for stale in ("champion.pt", "eval.csv"):
            p = ckpt_dir_path / stale
            if p.exists():
                p.unlink()
                log.info("Cleared stale %s (fresh run)", stale)
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        trainer._save_champion(ckpt_dir_path, gen=0)
        log.info("Bootstrapped champion from initial weights (gen 0)")

    model_summary = (
        f"family=jester (misère)\n"
        f"blocks={cfg.NUM_RES_BLOCKS}\n"
        f"filters={cfg.NUM_FILTERS}\n"
        f"params={param_count / 1e6:.2f}M\n"
        f"lr={config.learning_rate:.1e}\n"
        f"games={config.num_concurrent_games}\n"
        f"sims={config.mcts_simulations}\n"
        f"mirror={config.jester_selfplay_prob:.0%}\n"
        f"batch={config.batch_size}"
    )

    from _launcher import _build_system_summary

    ckpt_dir = str(ckpt_dir_path)
    with DashboardLogger(
        ckpt_dir,
        model_summary=model_summary,
        device_summary=_build_system_summary(device),
        on_log=None,
    ) as dash:
        _orig_save = trainer.save_checkpoint
        def save_and_log(directory):
            paths = _orig_save(directory)
            names = " + ".join(p.name for p in paths.values())
            dash.log(f"checkpoint → {names}")
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
        if trainer._stop_requested:
            dash.log(
                f"PLATEAU STOP — {config.max_plateau_evals} consecutive "
                f"failed evals. Final checkpoint written."
            )
            trainer.save_checkpoint(ckpt_dir)
