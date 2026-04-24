"""Shared launcher for the hardware-specific training entrypoints.

`train_ubuntu.py` and `train_windows.py` each call `launch(...)` with
their own hardware-tuned overrides. Everything else — MCTS depth,
eval cadence, architecture, lr schedule, etc. — is read from
`training/config.py`. Edit that file to change training behavior.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import sys
from pathlib import Path

import torch


def _build_system_summary(device: torch.device) -> str:
    """Multi-line k=v summary of the host + device for the dashboard panel.

    Cheap probes only — this runs once at startup. GPU name / VRAM via
    torch.cuda when applicable; CPU count via os.cpu_count(). Missing
    fields are just omitted.
    """
    lines = [f"device={device}"]

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024**3)
        lines.append(f"gpu={props.name}")
        lines.append(f"vram={vram_gb:.1f}GB")
        cuda_ver = torch.version.cuda or "?"
        lines.append(f"cuda={cuda_ver}")
    elif device.type == "mps":
        lines.append("gpu=Apple Silicon (MPS)")

    cpu_count = os.cpu_count() or 0
    lines.append(f"cpu-cores={cpu_count}")
    lines.append(f"host={platform.node()}")
    lines.append(f"os={platform.system()} {platform.release()}")
    lines.append(f"python={platform.python_version()}")
    lines.append(f"torch={torch.__version__}")
    return "\n".join(lines)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `import config` works

import config as cfg  # noqa: E402
from chess_ai.dashboard import DashboardLogger  # noqa: E402
from chess_ai.model import ChessNet  # noqa: E402
from chess_ai.train import Trainer, pick_device  # noqa: E402

log = logging.getLogger("chess_ai.train")


def launch(
    *,
    num_workers: int,
    games_per_worker: int,
    batch_size: int,
    mp_batch_wait_ms: float | None = None,
    min_examples_between_grad_steps: int | None = None,
) -> None:
    """Run training using `config.py` + the caller's hardware overrides.

    The first three overrides tune workload for the specific machine
    (core count, VRAM). The last two are optional inference/gradient
    cadence knobs — Mac (MPS) benefits from much larger inference
    batches than Ubuntu (CUDA) because MPS has higher kernel-launch
    overhead, so those boxes pass them; Ubuntu/Windows inherit the
    config.py defaults.
    """
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
    # Hardware overrides from the caller (train_ubuntu / train_windows).
    # These three values are the only machine-specific knobs; everything
    # else is shared across boxes via config.py.
    config.num_workers = num_workers
    config.games_per_worker = games_per_worker
    config.batch_size = batch_size
    if mp_batch_wait_ms is not None:
        config.mp_batch_wait_ms = mp_batch_wait_ms
    if min_examples_between_grad_steps is not None:
        config.min_examples_between_grad_steps = min_examples_between_grad_steps
    log.info(
        "Hardware overrides: num_workers=%d games_per_worker=%d batch_size=%d "
        "mp_batch_wait_ms=%.1f min_examples_between_grad_steps=%d",
        num_workers, games_per_worker, batch_size,
        config.mp_batch_wait_ms, config.min_examples_between_grad_steps,
    )

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
        champ_path = cfg.CHECKPOINT_DIR / "champion.pt"
        if champ_path.exists():
            champ_path.unlink()
            log.info("Cleared stale champion.pt (fresh run)")
        # Same cleanup applies to eval.csv. Otherwise a fresh training
        # run appends new eval matches to a CSV that still contains
        # rows from prior (potentially buggy) runs, polluting any
        # downstream trajectory plotting.
        eval_csv_path = cfg.CHECKPOINT_DIR / "eval.csv"
        if eval_csv_path.exists():
            eval_csv_path.unlink()
            log.info("Cleared stale eval.csv (fresh run)")
        # Snapshot the fresh (random-init) weights as champion.pt. This
        # way the first eval at `eval_every_gens` plays the trained
        # model against the initial random-init network — a real
        # baseline — instead of silently bootstrapping and reporting
        # no-contest. Gives us a meaningful Elo number for early gens.
        cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        trainer._save_champion(cfg.CHECKPOINT_DIR, gen=0)
        log.info("Bootstrapped champion from initial weights (gen 0)")

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
            device_summary=_build_system_summary(device),
            on_log=None,
        ) as dash:
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
