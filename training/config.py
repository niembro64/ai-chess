"""Single-source training config.

Edit this file to change anything about training. Run with:

    python scripts/run.py

No CLI flags, no options to remember — just tweak here and relaunch.

The separate `scripts/train.py` still exists for one-off experiments that
need argparse knobs; this file is the canonical path for normal runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from chess_ai.train import TrainConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Everything lives relative to this file, so editor cwd / tmux pane cwd
# doesn't matter.
ROOT = Path(__file__).resolve().parent

# Single canonical checkpoint directory. Overwritten on every run — no
# v1/v2/latest_new nonsense.
CHECKPOINT_DIR = ROOT / "runs" / "latest"

# Syzygy tablebases. We probe a few conventional locations and use the first
# one that exists; set the CHESS_AI_SYZYGY env var to override. Silently
# disabled when none of the candidates are present.
def _find_syzygy() -> Path | None:
    override = os.environ.get("CHESS_AI_SYZYGY")
    candidates = [Path(override)] if override else [
        Path.home() / "syzygy",    # where the server keeps them
        ROOT / "syzygy",           # conventional in-repo location
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None

SYZYGY_PATH: Path | None = _find_syzygy()
SYZYGY_MAX_PIECES = 5  # matches 3-4-5 piece tables (.rtbw)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

DEVICE = "auto"  # "auto" | "cuda" | "mps" | "cpu"
SEED = 42
ENABLE_DASHBOARD = True

# ---------------------------------------------------------------------------
# Model architecture (must match for resume)
# ---------------------------------------------------------------------------
#
# Sized for browser deployment: ~5.7M params, ~60MB JSON after weight
# quantization. Big enough to meaningfully encode chess knowledge
# (the previous 10×128 / 3M model was on the low end), small enough to
# fit under GitHub's 100MB hard limit and run in-browser without misery.
# AlphaZero used 20×256 (~46M) but they had 1000× the compute.

NUM_RES_BLOCKS = 12
NUM_FILTERS = 160
KERNEL_SIZE = 3
VALUE_HEAD_SIZE = 64
SE_REDUCTION = 8

# ---------------------------------------------------------------------------
# Training config — all other knobs
# ---------------------------------------------------------------------------

def build_config() -> TrainConfig:
    """Build the canonical TrainConfig. Edit values in-place."""
    return TrainConfig(
        # ---- Self-play, multiprocess (6 workers + 1 inference server) ----
        # Early runs showed the GPU idling ~95% of wall time waiting for
        # examples (`sleep_starved` dominated iter time). Three-part fix
        # tuned on a 3090: more workers, shallower MCTS for early gens,
        # and more frequent grad steps on thinner batches. Revisit once
        # gen/min stops rising with more workers.
        num_workers=6,
        games_per_worker=32,
        # Self-play MCTS depth. 30 was leaving the value head undertrained
        # because shallow search + weak value head = shuffling policy.
        # 60 doubles the depth and matches `eval_mcts_sims` — self-play
        # shouldn't be weaker than eval. Expect gen/min to drop from ~286
        # to ~150-200 on the 3090.
        mcts_simulations=60,
        mp_batch_wait_ms=5.0,
        weight_broadcast_every=50,

        # ---- Training hyperparams ----
        batch_size=512,
        gradient_steps_per_selfplay_step=1,
        min_examples_between_grad_steps=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        # Step-decay schedule — rough AlphaZero-style. Steps are gentler
        # (3× per step) than AZ's 10× because our total gen budget is
        # smaller; sharper steps would destabilize mid-training.
        lr_schedule=(
            (0,       1e-3),
            (30_000,  3e-4),
            (60_000,  1e-4),
            (85_000,  3e-5),
        ),
        use_amp=True,
        policy_label_smoothing=0.03,
        aux_material_weight=0.1,
        mirror_augment_prob=0.5,

        # ---- Self-play curriculum ----
        # Temperature window: first N plies sample moves proportionally to
        # MCTS visit counts (τ=1); after that it switches to argmax (τ→0).
        # Longer window = more opening variety in self-play. AlphaZero
        # used 30 plies; we were at 15. Bumped for coverage.
        temperature_threshold_plies=30,
        endgame_start_prob=0.40,
        # Random-walk starts produced ~0% decisive signal in the last run
        # (1 mate out of 1,239 games). Cut from 0.20 so more games go to
        # the standard bucket where the model actually learns to convert.
        random_start_prob=0.10,

        # ---- MCTS (None = module default) ----
        c_puct=None,
        dirichlet_alpha=None,
        dirichlet_epsilon=None,

        # ---- Replay buffer ----
        replay_buffer_capacity=200_000,
        min_buffer_for_training=2_000,

        # ---- Target (for progress bar + ETA; not a hard stop) ----
        target_gens=100_000,

        # ---- Syzygy tablebase adjudication (cap-timeouts) ----
        syzygy_path=str(SYZYGY_PATH) if SYZYGY_PATH else None,
        syzygy_max_pieces=SYZYGY_MAX_PIECES,

        # ---- Checkpointing ----
        checkpoint_every_seconds=60.0,
        archive_every_gens=1_000,   # snapshot every 1k gens → runs/latest/archive/
        keep_archives=20,

        # ---- Auto-eval (ALWAYS ON) ----
        # Every N gradient updates, pause training and play a match vs the
        # reigning champion. Challenger promotes to champion if it scores
        # above `eval_score_threshold`. Training stops after
        # `max_plateau_evals` consecutive failed evals.
        eval_every_gens=5_000,
        eval_games=20,
        # Deeper search for eval than self-play. Weak models can't find
        # mate in K+Q vs K with only 30 sims; 60 gives them a real chance
        # to demonstrate capability. Evals are rare, the extra compute is
        # free (one match ≈ one 5k-gen training cycle anyway).
        eval_mcts_sims=60,
        # Longer cap (400 plies) lets weak models actually reach mate during
        # eval; otherwise every early match drifts to "draw at cap" and the
        # plateau detector misfires while the model is genuinely learning.
        eval_move_cap=400,
        eval_score_threshold=0.54,   # ≈ +30 Elo
        # Plateau grace period. Early evals are mostly draws (noise, not
        # signal) so we need a long buffer before stop-training fires.
        # ~10 failed evals × 1000 gens ≈ 10k gens of headroom.
        max_plateau_evals=10,

        # ---- Logging ----
        log_every_steps=10,
    )


# Model summary string for the dashboard "model" panel.
def model_summary_lines(lr: float, param_count: float, concurrent_games: int, sims: int, batch: int) -> str:
    return (
        f"blocks={NUM_RES_BLOCKS}\n"
        f"filters={NUM_FILTERS}\n"
        f"value-head={VALUE_HEAD_SIZE}\n"
        f"se-reduction={SE_REDUCTION}\n"
        f"params={param_count / 1e6:.2f}M\n"
        f"lr={lr:.1e}\n"
        f"games={concurrent_games}\n"
        f"sims={sims}\n"
        f"batch={batch}"
    )
