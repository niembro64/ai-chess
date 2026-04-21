"""Single-source training config.

Edit this file to change anything about training. Launch with the
hardware-specific entrypoint for your box:

    python scripts/train_ubuntu.py     # 4-core + RTX 3090
    python scripts/train_windows.py    # 9900K + 1080 Ti

The two entrypoints share `scripts/_launcher.py` and differ only in
their hardware-specific overrides (num_workers, games_per_worker,
batch_size). Every other knob lives here.
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
        # Self-play MCTS depth. 200 sims produced ~45h ETA on the 3090
        # AND policy-head collapse — visit distributions were sharp
        # enough to reinforce whatever biases the policy developed
        # early, driving it toward confidently-wrong moves. 100 matches
        # `eval_mcts_sims` so self-play and eval see the same search
        # strength (which was the original motivation for lifting sims
        # past 25-30 in the first place).
        mcts_simulations=100,
        mp_batch_wait_ms=5.0,
        weight_broadcast_every=50,

        # ---- Training hyperparams ----
        batch_size=512,
        gradient_steps_per_selfplay_step=1,
        # 8 was cutting grad-step rate-limit 4× below the old working
        # value (32). At 8, each sample got reused in ~64 consecutive
        # batches before fresher data displaced it, which overfits the
        # policy head onto whatever early biases the MCTS visit target
        # happened to have. 32 is the AlphaZero-adjacent rule-of-thumb.
        min_examples_between_grad_steps=32,
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
        # 0.03 mixes 3% uniform into the MCTS visit target. In a broken
        # run where MCTS visits were biased toward wrong moves, label
        # smoothing prevented the policy from peaking on anything — but
        # it also prevented the policy from converging on the CORRECT
        # moves when MCTS found them. Diagnostic showed policy ranking
        # mates at #20-40 / 4096 despite v=+1.0 on the same positions,
        # consistent with smoothed targets + biased MCTS = no learning.
        # 0.0 = use the MCTS visit distribution as-is.
        policy_label_smoothing=0.0,
        # Value-head class balancing. With the MCTS sign-fix + cap-mask
        # in place, decisive rate in self-play runs at ~60-65% and cap
        # games no longer feed noise into the value loss. That leaves
        # a roughly 50/50 decisive-vs-legitimate-draw split, so we only
        # lightly down-weight draws. (Set higher when decisive rate
        # drops, lower when it's dominated by tb_d / stalemate / etc.)
        value_draw_weight=0.5,
        # Disabled. aux_material_weight feeds material-balance signal
        # through the shared trunk, which can bias trunk features
        # toward material-changing moves (captures, pushes) and away
        # from mates that don't change material. With the value head
        # already training well (diagnostic: avg v=+0.68 on winning
        # positions), the aux head is redundant at best, distortive
        # at worst — simplify it out until we have evidence we need it.
        aux_material_weight=0.0,
        mirror_augment_prob=0.5,

        # ---- Self-play curriculum ----
        # Temperature window: first N plies sample moves proportionally to
        # MCTS visit counts (τ=1); after that it switches to argmax (τ→0).
        # Longer window = more opening variety in self-play. AlphaZero
        # used 30 plies; we were at 15. Bumped for coverage.
        temperature_threshold_plies=30,
        # 40% was distorting the training distribution — 3-5 piece
        # syzygy positions are cheap decisive-value labels but far
        # from the middlegames the model actually plays. AlphaZero
        # used 0%; 0.15 keeps the value-label benefit from endgame
        # adjudication without skewing the policy head's distribution.
        endgame_start_prob=0.15,
        # Random-walk starts consistently produce ~0-3% decisive signal
        # (3 mates out of 331 random-origin games in the current run).
        # They're nearly pure waste — cap-timeouts with no tb rescue.
        # Zeroed out; that 10% of compute now flows to endgame (heavy
        # decisive signal via tb) + standard (where natural mate rate
        # is now 46% after the MCTS fix).
        random_start_prob=0.0,

        # ---- MCTS ----
        # Dirichlet noise at root injects exploration into self-play
        # prior probabilities. Module default is 0.25 (AZ's setting);
        # we bump to 0.35 because our self-play policy kept over-
        # committing — the model collapsed into a narrow style that
        # exploits itself but misses basic tactics in eval against
        # differently-trained opponents (e.g. mate-in-1s the policy
        # ranked low). More root noise = more off-policy exploration
        # during training, which diversifies the examples that feed
        # back into the policy head. None for the others keeps module
        # defaults (c_puct=1.5, dirichlet_alpha=0.3).
        c_puct=None,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.35,

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
        # 5k cadence gives more data points on the learning trajectory
        # — 10k left only ~10 evals across a 100k run, which is too
        # sparse to see plateau trends before they're entrenched. With
        # eval_mcts_sims back down to 100 and the rest of the match
        # budget stable, eval wall-time fraction stays manageable.
        eval_every_gens=5_000,
        # 120 games = 60 curated positions × 2 color assignments. The
        # position mix is 50 mate-in-1 (5 hand-crafted + 45 random) +
        # 5 asymmetric + 5 balanced openings. Each position plays
        # exactly once per color so fairness is preserved end-to-end.
        eval_games=120,
        # Deeper search for eval than self-play (60 sims). A trained
        # model's policy can become peaky to the point where MCTS at
        # 60 sims never explores mate shots ranked low in prior. 100
        # sims gives enough PUCT exploration budget to get past the
        # peakiness (combined with dirichlet_epsilon=0.35 during self-
        # play preventing the policy from going too peaky in the first
        # place). 200 was briefly tried — 3× match time for marginal
        # diagnostic gain; 100 is the sweet spot.
        eval_mcts_sims=100,
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
