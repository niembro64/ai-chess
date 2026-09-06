"""Launch competitive JESTER with isolated runs and explicit weight-only forks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config as cfg
from chess_ai import tablebase
from chess_ai.dashboard import DashboardLogger
from chess_ai.model import ChessNet
from chess_ai.train import Trainer, pick_device

log = logging.getLogger("chess_ai.train")


def launch(*, mode: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--resume", type=Path, help="Resume a competitive checkpoint including optimizer/counters"
    )
    source.add_argument(
        "--init-from", type=Path, help="Copy only weights; fresh replay, optimizer and counters"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=cfg.CHECKPOINT_DIR_JESTER)
    parser.add_argument(
        "--opponent", type=Path, action="append", default=[], help="Frozen JESTER checkpoint (repeatable)"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sims", type=int)
    parser.add_argument("--steps", type=int, help="Optional finite smoke run (main-loop iterations)")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(cfg.SEED)
    torch.set_num_threads(1)
    random.seed(cfg.SEED)
    # Explicitly clear process-global tables too. Setting config alone cannot
    # undo a tablebase opened earlier by another launcher in this interpreter.
    tablebase.open_tablebase(None)
    config = cfg.build_jester_config()
    if args.workers is not None:
        config.num_workers = args.workers
    if args.sims is not None:
        config.mcts_simulations = args.sims
        config.eval_mcts_sims = args.sims
    device = pick_device(cfg.DEVICE)
    if device.type != "cuda" and args.workers is None:
        config.num_workers = 0
    if config.num_workers:
        from chess_ai import mcts

        if not mcts.USE_RUST_MCTS or not hasattr(mcts._rust_mcts.MctsSearch, "pending_leaf_turn"):
            raise RuntimeError("Competitive workers require rebuilt Rust MCTS: maturin develop --release")
    directory = args.checkpoint_dir.resolve()
    resume = args.resume
    if mode == "continue" and args.init_from is None and resume is None:
        resume = directory / "latest.pt"
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    if resume is None and any(
        (directory / name).exists() for name in ("latest.pt", "champion.pt", "stats.csv")
    ):
        raise FileExistsError(f"{directory} already contains a run; use continue or a new --checkpoint-dir")
    directory.mkdir(parents=True, exist_ok=True)

    pool = directory / "opponents"
    pool.mkdir(exist_ok=True)
    sources = args.opponent or ([args.init_from] if args.init_from else [])
    for path in sources:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        target = pool / f"{digest[:16]}.pt"
        if not target.exists():
            shutil.copy2(path, target)
    config.jester_opponent_checkpoints = tuple(str(p) for p in sorted(pool.glob("*.pt")))
    model = ChessNet(
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        num_filters=cfg.NUM_FILTERS,
        kernel_size=cfg.KERNEL_SIZE,
        value_head_size=cfg.VALUE_HEAD_SIZE,
        se_reduction=cfg.SE_REDUCTION,
    )
    trainer = Trainer(model, device, config, random.Random(cfg.SEED))
    trainer.stats.curriculum_prob = config.jester_curriculum_prob
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        old = checkpoint.get("config", {})
        if old.get("jester_gate") != "head_to_head" or old.get("syzygy_path") is not None:
            raise ValueError(
                "Legacy cooperative run: use --init-from with a new directory instead of --resume"
            )
        trainer.load_checkpoint(resume)
        if trainer._mp_self_play:
            trainer._mp_self_play.set_curriculum_prob(trainer.stats.curriculum_prob)
        else:
            trainer.engine.config.curriculum_start_prob = trainer.stats.curriculum_prob
    else:
        provenance = {"initialization": "random", "seed": cfg.SEED}
        if args.init_from:
            checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            provenance.update(
                initialization="weights-only",
                source=str(args.init_from.resolve()),
                source_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                source_generation=checkpoint.get(
                    "champion_gen", checkpoint.get("stats", {}).get("generation")
                ),
            )
        (directory / "initialization.json").write_text(json.dumps(provenance, indent=2) + "\n")
        trainer._save_champion(directory, gen=0)
        trainer.save_checkpoint(directory)
    log.info(
        "Competitive JESTER: %s, %d workers x %d games, %d sims; %d frozen opponents; no Syzygy or cooperative mates",
        device,
        config.num_workers,
        config.games_per_worker,
        config.mcts_simulations,
        len(config.jester_opponent_checkpoints),
    )
    log.info("Checkpoints and evidence: %s", directory)
    summary = (
        f"family=jester (competitive)\nblocks={cfg.NUM_RES_BLOCKS}\nfilters={cfg.NUM_FILTERS}\n"
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n"
        f"lr={config.learning_rate:.1e}\nworkers={config.num_workers}\nsims={config.mcts_simulations}\n"
        f"curriculum={trainer.stats.curriculum_prob:.0%}\nbatch={config.batch_size}"
    )
    from _launcher import _build_system_summary

    with DashboardLogger(
        str(directory),
        model_summary=summary,
        enabled=not args.no_dashboard,
        device_summary=_build_system_summary(device),
        on_log=None,
    ) as dash:
        original_save = trainer.save_checkpoint

        def save(path):
            result = original_save(path)
            dash.log("checkpoint → latest.pt + latest.json")
            return result

        trainer.save_checkpoint = save
        try:
            trainer.run(
                num_steps=args.steps,
                checkpoint_dir=str(directory),
                on_step=dash.on_step,
                on_eval=lambda result: dash.on_eval(
                    result, threshold=config.eval_score_threshold, plateau_max=config.max_plateau_evals
                ),
                on_eval_progress=dash.on_eval_progress,
            )
        except KeyboardInterrupt:
            dash.log("interrupted — preserving checkpoint")
        finally:
            trainer.save_checkpoint(directory)
