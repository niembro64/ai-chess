"""Single-source training config.

Edit this file to change anything about training. Launch with the
hardware-specific entrypoint for your box, choosing between *_new
(fresh run) and *_continue (warm-start from runs/latest/latest.pt):

    python scripts/train_ubuntu_new.py        # 4-core + RTX 3090, fresh
    python scripts/train_ubuntu_continue.py   # 4-core + RTX 3090, resume
    python scripts/train_windows_new.py       # 9900K + 1080 Ti, fresh
    python scripts/train_windows_continue.py  # 9900K + 1080 Ti, resume
    python scripts/train_mac_new.py           # Apple Silicon (MPS), fresh
    python scripts/train_mac_continue.py      # Apple Silicon (MPS), resume

All six entrypoints share `scripts/_launcher.py` and differ only in
their hardware-specific overrides (num_workers, games_per_worker,
batch_size, mp_batch_wait_ms). Every other knob lives here.
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
        # Self-play MCTS depth. Sims are the dominant lever on how much
        # improvement signal the visit target carries over the raw
        # prior: simulation of the production 100-sim stack showed the
        # visit argmax matching the true best move only ~45% of the
        # time vs ~64% at AZ's 800. The earlier "200 sims caused
        # collapse" episode was misattributed — the collapse came from
        # the target-corruption bugs since fixed (castling-plane mirror
        # swap, value ply-decay draw labels, all-node softening), not
        # from search depth. 200 halves game throughput vs 100 but each
        # game teaches roughly twice as much; revisit upward if the
        # value head is learning (value_loss well below ~0.9).
        mcts_simulations=200,
        # Max time the inference server waits to accumulate requests before
        # dispatching a GPU batch. History: 5ms (pre-Rust, 4 workers) →
        # 8ms (Rust MCTS, 8 workers) → 12ms (once GPU hit 92% we bumped
        # again to let even bigger batches queue up). Above ~15ms each
        # worker feels the latency and per-game throughput drops; at 12
        # the GPU is processing near-optimal batch sizes with minimal
        # worker-side stall.
        mp_batch_wait_ms=12.0,
        weight_broadcast_every=50,

        # ---- Training hyperparams ----
        batch_size=512,
        gradient_steps_per_selfplay_step=1,
        # Post-Rust-MCTS era the self-play pipeline flows ~4× faster,
        # which means each sample in the 200k buffer gets re-sampled
        # many more times per minute than under the Python-MCTS regime.
        # At 32, effective sample reuse was ~44×/minute (vs AlphaZero's
        # target of ~8× *lifetime* per position). Bumping to 64 halves
        # the reuse rate and lets each grad step see fresher data —
        # trades raw gen/min for actual learning per step. ETA goes up
        # but convergence-per-gen should improve.
        # (Original comment: 8 was cutting grad-step rate-limit 4× below
        # the old working value (32). At 8, each sample got reused in
        # ~64 consecutive batches before fresher data displaced it,
        # which overfits the policy head onto whatever early biases the
        # MCTS visit target happened to have.)
        min_examples_between_grad_steps=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        # Step-decay schedule — rough AlphaZero-style. Steps are gentler
        # (3× per step) than AZ's 10× because our total gen budget is
        # smaller; sharper steps would destabilize mid-training. The
        # final entry is an explicit floor: runs routinely blow past
        # target_gens (the first long run spent its last 309k gens —
        # 78% of all gradient steps — parked at the previous 85k
        # terminus), so give the schedule one more step instead of
        # silently freezing.
        lr_schedule=(
            (0,       1e-3),
            (30_000,  3e-4),
            (60_000,  1e-4),
            (85_000,  3e-5),
            (150_000, 1e-5),
        ),
        use_amp=True,
        # Mix uniform probability into the MCTS visit target, spread over
        # each sample's visited-move support (train.py). Keep this SMALL:
        # the 0.10 used for most of the first long run — combined with
        # full-4096 spreading — put the loss floor at ~3 nats and trained
        # the head toward flatness; the model fit that corrupted target
        # almost perfectly and plateaued. 0.03 over ~30 legal moves still
        # gives every legal move a ~1e-3 floor (vs 7e-6 under 4096-wide
        # spreading at the same eps), which is plenty of anti-collapse
        # insurance now that the value head actually gets signal (see
        # value_ply_decay) and mirror augmentation is fixed.
        policy_label_smoothing=0.03,
        # Value-head class balancing. With the MCTS sign-fix + cap-mask
        # in place, decisive rate in self-play runs at ~60-65% and cap
        # games no longer feed noise into the value loss. That leaves
        # a roughly 50/50 decisive-vs-legitimate-draw split, so we only
        # lightly down-weight draws. (Set higher when decisive rate
        # drops, lower when it's dominated by tb_d / stalemate / etc.)
        value_draw_weight=0.5,
        # Per-ply decay on value targets — DISABLED (1.0 = AZ convention:
        # decisive games label ±1 at every ply). The previous 0.99 was a
        # training-run-corrupting mistake: the WDL conversion turns label
        # magnitude into draw probability (P(draw) = 1 - |v|), so at 0.99
        # every position more than ~69 plies from game end in a WON game
        # got a majority-draw target. The value head was explicitly
        # trained to output ~0 for all openings/middlegames, leaving
        # MCTS Q with no early-game guidance and resign unreachable.
        # (The Leela/KataGo "0.99" being imitated does not decay the WDL
        # outcome label — their mate-distance signal lives in separate
        # heads.) "Prefer faster wins" should come from search terminal
        # handling, not from corrupting outcome labels.
        value_ply_decay=1.0,
        # Soften self-play MCTS priors — ROOT ONLY (mcts.py). Diagnostic
        # on the 32k-gen checkpoint showed a collapsed policy with prior
        # 0.67–0.78 on the wrong move vs 0.001–0.015 on the mating move;
        # root softening gives low-prior moves enough exploration budget
        # to get visited. An earlier version softened EVERY node's priors
        # (root + all leaf expansions), which flattened Q estimates
        # throughout the tree and reduced the whole search to exploration
        # noise at low sim counts. Eval keeps the trained policy's
        # sharpness intact (softening is a self-play-only argument).
        self_play_policy_softening_temperature=1.5,
        # Down-weight policy loss on TB-adjudicated game samples. Those
        # games went to cap/50-move without MCTS finding a forcing line;
        # Syzygy rescued the value label (correct, kept at full weight),
        # but the visit distributions reflect "best guess while lost,"
        # not "this is the right plan." At gen ~570, 35% of games are
        # TB-adjudicated — without this discount a third of the policy
        # gradient trains the network to imitate meandering play.
        # 0.5 halves their contribution; drops to near-no-op as the
        # model learns to finish games on its own.
        tb_policy_weight=0.5,
        # Resignation: end self-play games early when the side-to-move's
        # best visited root-child Q ("even my best move loses") is at or
        # below `resign_threshold` after at least `resign_min_plies`
        # plies. Saves the cost of running already-decided games to
        # mate / cap. AZ-standard knobs: -0.85 / 0.10 / 20.
        # `resign_disabled_prob` is a held-back fraction of would-be
        # resignations that play on; the false-positive rate they reveal
        # is tracked in TrainStats.resign_truth_games/resign_truth_fp
        # and written to stats.csv — keep FP under ~5%, raise the
        # threshold toward -1.0 if it runs higher. Resigned games land
        # in their own resign_w/resign_b outcome buckets so mate-rate
        # diagnostics stay honest. Set `resign_threshold <= -1.0` to
        # disable.
        resign_threshold=-0.85,
        resign_disabled_prob=0.10,
        resign_min_plies=20,
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
        # prior probabilities. Back to AZ's 0.25: the 0.35 bump (an
        # anti-collapse patch) meant ~1/3 of every 100-sim budget chased
        # noise moves, and at that ratio ~17% of the visit-count
        # TRAINING TARGET was noise-following — the exploration knob was
        # eating the improvement operator. Exploration should come from
        # sims (raised to 200) and root softening, not from drowning
        # the target. Noise is self-play-only; eval passes epsilon=0.
        # None for the others keeps module defaults (c_puct=1.5,
        # dirichlet_alpha=0.3).
        c_puct=None,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.25,
        # Widen PUCT at self-play only (eval keeps c_puct=1.5 default
        # for sharp exploitation). 2.0 gives low-prior moves more
        # exploration bonus than the 1.5 default without the flattening
        # overshoot of the earlier 2.5 (stacked with noise + softening
        # it pushed root Q modulation of the visit distribution down to
        # a ~2-5x tilt on a noisy prior — barely an improvement
        # operator). AZ's own chess setting was ~2.0-2.5 at 800 sims;
        # at our 200 sims 2.0 is the right side to err on.
        self_play_c_puct=2.0,
        # First-Play Urgency reduction — AlphaZero/Leela-standard PUCT
        # refinement. Unvisited children get an initial Q of
        # `parent_Q - fpu_reduction` instead of 0. This pessimizes
        # prior-peaked unvisited children once their siblings have been
        # explored and found OK, shifting sim budget toward the moves
        # most likely to beat the current best. Complements policy
        # softening: softening flattens the input prior, FPU makes PUCT
        # less dependent on that prior once search is underway. 0.4 is
        # the Leela default and produces the expected behavior on
        # our c_puct=1.5–2.5 range.
        fpu_reduction=0.4,

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
        # 10k cadence: at ~50 gen/min and ~110 min/match, the previous
        # 5k cadence had eval matches consuming ~50% of wall-clock
        # (one match per ~100 min of training, each match ~110 min).
        # Doubling the cadence cuts eval-time fraction to ~25% without
        # giving up plateau-detection resolution: at this throughput
        # we still get ~2 evals/day, well within the noise envelope of
        # the 140-game match.
        eval_every_gens=10_000,
        # eval_games is informational — the trainer derives the true
        # game count from len(positions) * 2 (curated 60 + rotating K).
        # See TrainConfig.eval_games / eval_rotating_openings.
        eval_games=140,
        # Deeper search for eval than self-play (60 sims). A trained
        # model's policy can become peaky to the point where MCTS at
        # 60 sims never explores mate shots ranked low in prior. 100
        # sims matches the self-play search depth so eval measures the
        # model the same way training stresses it (combined with
        # dirichlet_epsilon=0.35 during self-play preventing the policy
        # from going too peaky in the first place). 200 was briefly
        # tried — 3× match time for marginal diagnostic gain; 100 is
        # the sweet spot.
        eval_mcts_sims=100,
        # Longer cap (400 plies) lets weak models actually reach mate during
        # eval; otherwise every early match drifts to "draw at cap" and the
        # plateau detector misfires while the model is genuinely learning.
        eval_move_cap=400,
        # Score required to dethrone the champion. 0.54 ≈ +28 Elo at
        # 140 games (~1σ above 0.5, where SE ≈ 0.042). Tighter than
        # the prior 0.51 gate which sat well inside noise and would
        # routinely promote drifts that weren't real strength gains.
        eval_score_threshold=0.54,
        # Inject 10 fresh random-walk opening positions into each eval
        # match (× 2 colors → 20 of the 140 games). Each match draws
        # new positions so the eval distribution tracks current self-
        # play without becoming gameable. Curated 60 still drive the
        # bulk of the signal (technique / structure / mate-in-1).
        eval_rotating_openings=10,
        # Plateau grace period. Early evals are mostly draws (noise, not
        # signal) so we need a long buffer before stop-training fires.
        # 10 failed evals × eval_every_gens (10k) = 100k gens of headroom.
        max_plateau_evals=10,

        # ---- Logging ----
        log_every_steps=10,
    )


# ---------------------------------------------------------------------------
# Toy — the teaching-sized net, trained by the SAME pipeline
# ---------------------------------------------------------------------------
#
# Toy inherits every architecture-independent helper from build_config()
# verbatim — label smoothing, draw down-weighting, cap masking, mirror
# augmentation, resign + truth-check, Syzygy adjudication, endgame
# curriculum, dirichlet/FPU/softening, eval gating, plateau stop — and
# overrides only scale and throughput. Architecture lives in
# chess_ai/toy.py (10 BN-free blocks x 128 filters ≈ 4.0M params —
# ~71% of Sage — behind the minimal 6-plane input).
#
# Throughput note: the Rust MCTS and the multiprocess inference server
# are hard-wired to the 20-plane encoding, so Toy runs single-process
# Python MCTS with many lockstep games batching their NN calls — the
# same architecture Sage used pre-Rust. Fine at Toy's size.

CHECKPOINT_DIR_TOY = ROOT / "runs_toy" / "latest"
CHECKPOINT_DIR_JESTER = ROOT / "runs_jester" / "latest"
# Frozen winner the jester trains against (and models in-tree): the
# Sage champion checkpoint on this box.
JESTER_OPPONENT_CKPT = ROOT / "runs" / "latest" / "champion.pt"


def build_toy_config() -> TrainConfig:
    cfg = build_config()
    cfg.num_workers = 0                 # single-process (see note above)
    cfg.num_concurrent_games = 96       # lockstep games per batched GPU call
    cfg.mcts_simulations = 64
    cfg.batch_size = 256
    cfg.min_examples_between_grad_steps = 64
    cfg.replay_buffer_capacity = 60_000
    cfg.min_buffer_for_training = 2_000
    # Toy converges in far fewer gens than Sage; schedule scaled down.
    cfg.learning_rate = 1e-3
    cfg.lr_schedule = (
        (0,       1e-3),
        (15_000,  3e-4),
        (35_000,  1e-4),
        (60_000,  3e-5),
    )
    cfg.target_gens = 40_000
    cfg.eval_every_gens = 2_000
    cfg.eval_mcts_sims = 64
    cfg.eval_move_cap = 400
    cfg.archive_every_gens = 1_000
    # More decisive-signal supply than Sage's 0.15: the 10x128 run's
    # standard-start games were 95% shuffle-draws at gen 6k while
    # endgame-origin games resolved 75% (mates + tablebase labels).
    # Paired with in-tree repetition awareness (mcts.py) to break the
    # 90%-threefold collapse.
    cfg.endgame_start_prob = 0.35
    return cfg


# ---------------------------------------------------------------------------
# Jester — the misère bot: full Sage architecture, inverted incentives.
# ---------------------------------------------------------------------------
#
# Trains to LOSE — specifically, to get its OWN king checkmated before
# the opponent manages to get theirs checkmated. Search selection
# inverts at the jester's plies while every value label stays truthful
# (chess_ai/mcts.py MCTSSearch).
#
# The genre matters. Playing the frozen Sage is a HELPMATE: the
# opponent wants to mate you, so "stop defending" solves it. Playing
# another jester is a SELFMATE: the opponent refuses to mate you, so
# you have to force it — and that is the game the shipped bot actually
# plays, against a human who is also racing to be mated. Self-play is
# therefore mostly mirror games, with a small Sage share kept for
# bootstrap (early on it is the only dense source of "this is what
# being mated looks like") and for robustness when a human accidentally
# plays a strong winning move.
#
# Greedy mirror play never terminates — neither loss-seeker will
# deliver the mate the other wants, so the game shuffles to a threefold
# draw and teaches nothing. One side of every mirror game therefore
# spars at a sustained temperature, which both restores the terminal
# signal and models the imperfect human.
#
# Runs on the Python MCTS path (inversion/dual-net are not in the Rust
# search yet), single-process like Toy.

def build_jester_config() -> TrainConfig:
    cfg = build_config()
    cfg.num_workers = 0
    cfg.num_concurrent_games = 64
    cfg.mcts_simulations = 96
    cfg.batch_size = 256
    cfg.min_examples_between_grad_steps = 64
    cfg.replay_buffer_capacity = 80_000
    cfg.min_buffer_for_training = 2_000
    cfg.learning_rate = 1e-3
    cfg.lr_schedule = (
        (0,       1e-3),
        (15_000,  3e-4),
        (35_000,  1e-4),
        (60_000,  3e-5),
    )
    cfg.target_gens = 40_000
    cfg.eval_every_gens = 2_000
    cfg.eval_mcts_sims = 96
    # Mirror eval games are slower than the old vs-Sage ones: both sides
    # are ducking the mate, so decisive lines take longer to appear.
    cfg.eval_move_cap = 220
    cfg.archive_every_gens = 1_000
    # Resignation is meaningless in misère play: the jester's truthful
    # best-Q is *supposed* to be terrible, so the trigger would fire on
    # every healthy position. Disabled outright.
    cfg.resign_threshold = -2.0
    cfg.jester_mode = True
    # Mostly mirror (the production matchup); the rest vs frozen Sage.
    cfg.jester_selfplay_prob = 0.85
    cfg.jester_opponent_checkpoint = str(JESTER_OPPONENT_CKPT)
    # τ=1 is visit-proportional — AlphaZero's own opening-play setting,
    # sustained here for the whole game rather than annealed away.
    cfg.jester_spar_temperature = 1.0
    cfg.eval_temperature = 1.0
    return cfg


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
