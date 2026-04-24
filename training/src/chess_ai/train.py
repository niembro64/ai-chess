"""Training orchestrator: self-play + gradient updates + checkpointing.

Single-process loop that owns the model, runs batched self-play to produce
training examples, and periodically does gradient updates on a replay
buffer. Mirrors the strategy in `Trainer.ts` but on CUDA/MPS via PyTorch.

Two checkpoint artifacts on every save:
  * `latest.pt`  — native PyTorch state dict (resume-able)
  * `latest.json` — browser-compatible SerializedWeights JSON
  ChessNet.importWeights() loads the JSON directly. Point the browser at it
  by copying into `src/game/ai/presetWeights.txt`.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("chess_ai.train")

from .model import (
    NUM_PLANES,
    WDL_SIZE,
    ChessNet,
    MaterialAuxHead,
    encoded_to_nchw,
    material_target_from_board,
)
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights
from .selfplay import ReplayBuffer, SelfPlayConfig, make_local_selfplay_engine, mirror_batch
from .weight_io import export_weights


@dataclass
class TrainConfig:
    # Self-play
    num_concurrent_games: int = 32
    mcts_simulations: int = 25
    # Multi-process self-play. num_workers=0 → single-process (legacy path).
    # num_workers>=1 → spawn that many CPU workers + one GPU inference server.
    # Each worker runs `games_per_worker` self-play games in parallel.
    num_workers: int = 0
    games_per_worker: int = 16
    mp_batch_wait_ms: float = 5.0
    weight_broadcast_every: int = 50   # gradient steps between inference-server weight refreshes
    # Training
    batch_size: int = 256
    gradient_steps_per_selfplay_step: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Optional step-decay schedule. Each entry is (min_gen, lr). The
    # Trainer picks the highest-threshold entry whose `min_gen <=
    # stats.generation` and applies that lr to the optimizer. Must be
    # sorted ascending by `min_gen`. None keeps `learning_rate` flat.
    # AlphaZero used stepped decay (0.2 → 0.02 → 0.002 → 0.0002); a
    # gentler version helps late-stage polishing without destabilizing.
    lr_schedule: tuple[tuple[int, float], ...] | None = None
    # Rate-limit gradient steps: require at least this many *new* training
    # examples to have arrived since the last update before taking another
    # gradient step. Prevents overtraining on tiny buffers in MP mode where
    # the trainer main-loop spins much faster than workers produce data.
    # AlphaZero-style ratio is ~1 grad step per 1000+ positions; 32 is a
    # reasonable default for our smaller scale.
    min_examples_between_grad_steps: int = 32
    # Starting-position mix + move-selection temperature (plumbed from the
    # SelfPlayConfig defaults for CLI convenience).
    endgame_start_prob: float = 0.0
    random_start_prob: float = 0.3
    temperature_threshold_plies: int = 15
    # MCTS search hyperparams. Applied via mcts.set_mcts_params() at startup
    # (and on each MP worker at import time). None = keep module default.
    c_puct: float | None = None
    dirichlet_alpha: float | None = None
    dirichlet_epsilon: float | None = None
    # Self-play-only override for c_puct. MP workers set their MCTS
    # globals from this; main-process eval keeps `c_puct` above. Used
    # to widen PUCT exploration during self-play (recovery from
    # collapse) while keeping eval's search strength tuned for
    # exploitation of the trained policy. None = use `c_puct`.
    self_play_c_puct: float | None = None
    # Left-right (file) mirror augmentation on sampled batches. Chess is
    # symmetric about the file axis, so this is a free 2x data multiplier.
    # 0.0 disables. 0.5 mirrors half the batch each step (standard).
    mirror_augment_prob: float = 0.5
    # Per-sample multiplier on the value-head loss for samples whose
    # target is "draw" (WDL = [0, 1, 0]). When < 1.0 the gradient
    # descent pays less attention to drawn examples relative to decisive
    # ones (win/loss), which directly counteracts value-head collapse in
    # training distributions where draws dominate (our self-play runs
    # see ~80% draw labels). 1.0 preserves the unweighted cross-entropy.
    # Starting point: 0.3 under-weights draws ~3× without ignoring them.
    value_draw_weight: float = 1.0
    # Per-ply value-target decay. Positions i plies from game end get a
    # value target of outcome * decay^i. 1.0 reproduces the AlphaZero
    # convention (every position in a winning game labeled +1). <1.0
    # teaches the value head to distinguish mate-in-1 from mate-in-100
    # — crucial when self-play MCTS sims are too low (we use 100, AZ
    # used 800) to reach terminal nodes often enough for Q to carry
    # mate-speed signal. Leela / KataGo use ~0.99.
    value_ply_decay: float = 1.0
    # Soften NN policy priors during self-play MCTS (>1.0 = flatter). The
    # training target (MCTS visit distribution) is unchanged; only the
    # input prior to MCTS is softened, so trained-policy sharpness at
    # eval time is preserved. Fixes policy-collapse where MCTS can't
    # explore past an overconfident wrong-move prior at low sim counts.
    # Typical: 1.3–1.6 when recovering from a collapsed policy; 1.0
    # (no-op) otherwise.
    self_play_policy_softening_temperature: float = 1.0
    # Mixed-precision training on CUDA: forward/backward in fp16 with a
    # GradScaler. 2-3x speedup on Pascal (1080 Ti) with no measurable
    # quality loss. No-op on CPU/MPS.
    use_amp: bool = True
    # Label smoothing on the MCTS policy target: mix a tiny uniform prior
    # over all legal moves to prevent the policy head from collapsing
    # probability to exactly 0 on moves the current search didn't visit.
    # Typical: 0.0-0.03. 0 disables.
    policy_label_smoothing: float = 0.0
    # Auxiliary head (KataGo-style): material balance prediction. Lives
    # outside ChessNet so browser weights are unchanged. Multi-task
    # training gives the trunk dense gradient signal on a quantity the
    # main heads also need. 0 disables the aux head + loss entirely.
    aux_material_weight: float = 0.1
    # Syzygy tablebase directory (None disables). When set and a cap game
    # has few enough pieces, we adjudicate the outcome via the tablebase
    # instead of scoring it 0.
    syzygy_path: str | None = None
    syzygy_max_pieces: int = 5
    # Replay
    replay_buffer_capacity: int = 100_000
    min_buffer_for_training: int = 2_000
    # "Well trained" milestone — drives the dashboard progress bar and ETA.
    # Not a hard stop; training runs until interrupted. Tune to your compute
    # budget and model size: 100k gradient updates is a reasonable ceiling
    # for a 3M-param chess network at our self-play throughput.
    target_gens: int = 100_000
    # Window (in seconds) used to compute recent gen/min + games/min rates.
    # Short = noisy but responsive; long = smoother but slow to reflect change.
    rate_window_seconds: float = 120.0
    # Checkpointing
    checkpoint_every_seconds: float = 60.0
    # Periodically copy `latest.pt` off to `archive/gen-<N>.pt` so we have a
    # trail of snapshots for compare_checkpoints + plateau detection. 0
    # disables archival entirely.
    archive_every_gens: int = 0
    # Cap on retained archives (oldest are deleted as new ones are written).
    # 0 = unlimited.
    keep_archives: int = 20
    # Auto-eval cadence. Every N gradient updates we pause training and play
    # a small tournament against the reigning "champion" checkpoint. 0
    # disables auto-eval.
    eval_every_gens: int = 0
    eval_games: int = 20
    eval_mcts_sims: int = 30
    eval_move_cap: int = 200
    # Score (wins + 0.5*draws) / total the challenger must exceed to dethrone
    # the champion. 0.54 ≈ +30 Elo — enough to be above CI noise with 20-game
    # evals.
    eval_score_threshold: float = 0.54
    # Stop training after this many consecutive evals where the challenger
    # failed to dethrone the champion. 0 disables the plateau detector
    # (training runs indefinitely). Typical: 3 — one bad eval could be
    # noise, but three in a row is a real plateau.
    max_plateau_evals: int = 0
    # Logging
    log_every_steps: int = 10
    # Reward shaping
    rewards: RewardWeights = field(default_factory=lambda: RewardWeights(**DEFAULT_REWARD_WEIGHTS.__dict__))


@dataclass
class TrainStats:
    step: int = 0
    generation: int = 0              # Count of gradient steps taken so far
    target_gens: int = 0             # Goal for "well trained" (for progress bar / ETA)
    games_completed: int = 0
    # Granular end-state counters. Every finished game increments exactly one.
    # See selfplay.GameResult for the meaning of each bucket.
    mate_w: int = 0                  # over-the-board white checkmate
    mate_b: int = 0                  # over-the-board black checkmate
    stalemate: int = 0               # no legal moves, not in check
    draw_50: int = 0                 # 50-move rule
    tb_w: int = 0                    # cap-timeout → Syzygy says white wins
    tb_b: int = 0                    # cap-timeout → Syzygy says black wins
    tb_d: int = 0                    # cap-timeout → Syzygy says draw
    cap: int = 0                     # cap-timeout, no tablebase signal
    # Aggregates (computed on the fly via properties) are exposed below for
    # backward compat with existing callers/CSV readers.
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    replay_size: int = 0
    # Rates measured over a recent sliding window (not cumulative) so they
    # reflect the current pace after the worker startup transient.
    gen_per_min: float = 0.0
    games_per_min: float = 0.0
    # Seconds-remaining estimate to reach target_gens at the current rate.
    # None if we haven't completed any gradient updates yet.
    eta_seconds: float | None = None
    # EMA-smoothed per-phase durations (ms). For bottleneck diagnosis.
    # Main loop phases:
    t_drain_ms: float = 0.0          # drain_examples + drain_results from workers
    t_broadcast_ms: float = 0.0      # broadcast fresh weights to inference server
    t_sleep_ms: float = 0.0          # yield when buffer is not ready (high = worker-starved)
    t_iter_ms: float = 0.0           # total wall time per main-loop iteration
    # train_step phases (CUDA-synchronized so GPU async doesn't lie):
    t_sample_ms: float = 0.0         # replay buffer sample
    t_h2d_ms: float = 0.0            # numpy -> tensor -> device transfer
    t_forward_ms: float = 0.0        # forward pass
    t_backward_ms: float = 0.0       # backward pass
    t_optim_ms: float = 0.0          # optimizer step
    # Aux head losses (0 when the head isn't active).
    aux_material_loss: float = 0.0

    # --- Aggregate views over the granular counters ---------------------
    # These exist so CSV readers / old callers / the dashboard footer can
    # still say "total W" without knowing about the mate_w/tb_w split.

    @property
    def white_wins(self) -> int:
        """All games won by white, OTB mate + tablebase-adjudicated."""
        return self.mate_w + self.tb_w

    @property
    def black_wins(self) -> int:
        """All games won by black, OTB mate + tablebase-adjudicated."""
        return self.mate_b + self.tb_b

    @property
    def draws(self) -> int:
        """All drawn games: stalemate + 50-move + tablebase-adjudicated."""
        return self.stalemate + self.draw_50 + self.tb_d

    @property
    def caps(self) -> int:
        """Cap-timeouts WITHOUT tablebase adjudication (pure unresolved)."""
        return self.cap

    @property
    def tb_adjudications(self) -> int:
        """All cap-timeouts adjudicated via Syzygy (any outcome)."""
        return self.tb_w + self.tb_b + self.tb_d

    # --- Per-origin breakdown -------------------------------------------
    # Same 8 outcome buckets, split three ways by how the game was
    # initialized. Useful for answering "are my endgame inits producing
    # real play signal, or mostly tablebase-distilled labels?" — see
    # selfplay.GameSlot.origin. The sum across origins equals the global
    # aggregate for each bucket.
    origin_outcomes: dict[str, dict[str, int]] = field(default_factory=lambda: {
        origin: {
            "mate_w": 0, "mate_b": 0,
            "stalemate": 0, "draw_50": 0,
            "tb_w": 0, "tb_b": 0, "tb_d": 0,
            "cap": 0,
        }
        for origin in ("standard", "endgame", "random")
    })


def values_to_wdl_targets(values: np.ndarray) -> np.ndarray:
    """Convert continuous values ∈ [-1, 1] to WDL targets [P(win), P(draw), P(loss)].

    Matches ChessNet.ts: P(win) = max(0, v), P(loss) = max(0, -v), P(draw) = 1 - w - l.
    """
    v = np.clip(values, -1.0, 1.0)
    w = np.maximum(0.0, v)
    l = np.maximum(0.0, -v)
    d = 1.0 - w - l
    return np.stack([w, d, l], axis=1).astype(np.float32)


class Trainer:
    def __init__(
        self,
        model: ChessNet,
        device: torch.device,
        config: TrainConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config or TrainConfig()
        self.device = device
        self.model = model.to(device)

        # Apply MCTS hyperparam overrides before self-play starts.
        from .mcts import set_mcts_params
        set_mcts_params(
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

        # Optional material aux head (KataGo-style). Only instantiated when
        # aux_material_weight > 0 so inference-only setups pay nothing.
        self.aux_material: MaterialAuxHead | None = None
        if self.config.aux_material_weight > 0:
            self.aux_material = MaterialAuxHead(model.num_filters).to(device)

        params_to_optimize = list(model.parameters())
        if self.aux_material is not None:
            params_to_optimize += list(self.aux_material.parameters())
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.rng = rng or random.Random()
        self.buffer = ReplayBuffer(capacity=self.config.replay_buffer_capacity)

        self._mp_self_play = None
        self.engine = None

        if self.config.num_workers > 0:
            # Multiprocessing mode: spawn workers + inference server. The
            # trainer no longer runs self-play inline; examples flow from
            # workers into the buffer via `mp_self_play.drain_examples`.
            from .mpselfplay import (
                ModelArch,
                MultiprocessingConfig,
                MultiprocessingSelfPlay,
            )
            mp_cfg = MultiprocessingConfig(
                num_workers=self.config.num_workers,
                games_per_worker=self.config.games_per_worker,
                mcts_simulations=self.config.mcts_simulations,
                batch_wait_ms=self.config.mp_batch_wait_ms,
                endgame_start_prob=self.config.endgame_start_prob,
                random_start_prob=self.config.random_start_prob,
                temperature_threshold_plies=self.config.temperature_threshold_plies,
                # Workers use self_play_c_puct when set; falls back to c_puct.
                # Main-process eval below keeps config.c_puct.
                c_puct=self.config.self_play_c_puct
                    if self.config.self_play_c_puct is not None
                    else self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
                syzygy_path=self.config.syzygy_path,
                syzygy_max_pieces=self.config.syzygy_max_pieces,
                rewards=self.config.rewards,
                value_ply_decay=self.config.value_ply_decay,
                policy_softening_temperature=self.config.self_play_policy_softening_temperature,
            )
            self._mp_self_play = MultiprocessingSelfPlay(
                arch=ModelArch.from_model(self.model),
                initial_state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                device=self.device,
                config=mp_cfg,
            )
        else:
            self.engine = make_local_selfplay_engine(
                model=self.model,
                device=device,
                replay_buffer=self.buffer,
                config=SelfPlayConfig(
                    num_concurrent_games=self.config.num_concurrent_games,
                    mcts_simulations=self.config.mcts_simulations,
                    endgame_start_prob=self.config.endgame_start_prob,
                    random_start_prob=self.config.random_start_prob,
                    temperature_threshold_plies=self.config.temperature_threshold_plies,
                    rewards=self.config.rewards,
                    value_ply_decay=self.config.value_ply_decay,
                    policy_softening_temperature=self.config.self_play_policy_softening_temperature,
                ),
                rng=self.rng,
            )

        self.stats = TrainStats(target_gens=self.config.target_gens)
        self._start_time = time.time()
        self._last_checkpoint_time = 0.0
        self._last_archive_gen = 0
        # Auto-eval state.
        self._last_eval_gen = 0
        self._champion_model: ChessNet | None = None
        self._champion_gen = 0
        self._plateau_counter = 0
        self._eval_history: list[dict] = []
        self._on_eval: Callable[[dict], None] | None = None
        # Per-game progress callback during an in-flight eval match.
        # Args: (games_done, total_games, wins, draws, losses, per_diff).
        # `per_diff` maps difficulty -> {"w", "d", "l"} so the dashboard
        # can show where the challenger is stronger/weaker mid-match.
        # Lets the dashboard render activity while training is paused
        # for the match (which can be 10+ minutes on 120 games).
        self._on_eval_progress: Callable[..., None] | None = None
        # Set by the plateau detector (committed separately) to request a
        # clean shutdown on the next main-loop iteration.
        self._stop_requested = False
        # (timestamp, generation, games_completed) samples used for windowed rates.
        self._rate_samples: deque[tuple[float, int, int]] = deque()
        # EMA smoothing factor for per-phase timings.
        self._timing_alpha = 0.1
        self._is_cuda = self.device.type == "cuda"

        # Mixed-precision GradScaler. Only active on CUDA + config.use_amp.
        self._amp_enabled = self._is_cuda and self.config.use_amp
        self._scaler = torch.amp.GradScaler("cuda") if self._amp_enabled else None

    def _device_sync(self) -> None:
        """Block until pending GPU work is done. No-op on CPU/MPS."""
        if self._is_cuda:
            torch.cuda.synchronize()

    def _maybe_update_lr(self) -> None:
        """Apply the step-decay schedule if one is configured.

        Picks the highest-threshold entry `(min_gen, lr)` where
        `min_gen <= self.stats.generation` and sets that lr on every
        optimizer param group. No-op when `lr_schedule` is None.
        """
        schedule = self.config.lr_schedule
        if not schedule:
            return
        gen = self.stats.generation
        target_lr = schedule[0][1]
        for threshold, lr in schedule:
            if gen >= threshold:
                target_lr = lr
            else:
                break
        # Skip the param_group write when the value hasn't changed — avoids
        # a per-step Python-side tensor dict mutation for no reason.
        if self.optimizer.param_groups[0].get("lr") == target_lr:
            return
        for g in self.optimizer.param_groups:
            g["lr"] = target_lr

    def _ema(self, attr: str, sample_ms: float) -> None:
        """Exponential moving average update on a TrainStats timing field."""
        prev = getattr(self.stats, attr)
        setattr(self.stats, attr, prev + self._timing_alpha * (sample_ms - prev) if prev > 0 else sample_ms)

    def _update_rates(self) -> None:
        """Recompute gen/min + games/min over the last `rate_window_seconds`.

        Using a sliding window (rather than cumulative averages) keeps the
        dashboard responsive after the initial worker warm-up, and gives an
        honest ETA that reflects the current pace, not the startup drag.
        """
        now = time.time()
        self._rate_samples.append((now, self.stats.generation, self.stats.games_completed))
        cutoff = now - self.config.rate_window_seconds
        while self._rate_samples and self._rate_samples[0][0] < cutoff:
            self._rate_samples.popleft()

        if len(self._rate_samples) >= 2:
            first = self._rate_samples[0]
            last = self._rate_samples[-1]
            dt_min = (last[0] - first[0]) / 60.0
            if dt_min > 0:
                self.stats.gen_per_min = (last[1] - first[1]) / dt_min
                self.stats.games_per_min = (last[2] - first[2]) / dt_min

        # ETA to target, based on the windowed rate.
        if self.stats.gen_per_min > 0 and self.stats.generation < self.stats.target_gens:
            remaining = self.stats.target_gens - self.stats.generation
            self.stats.eta_seconds = remaining / self.stats.gen_per_min * 60.0
        elif self.stats.generation >= self.stats.target_gens:
            self.stats.eta_seconds = 0.0
        else:
            self.stats.eta_seconds = None

    def train_step(self) -> dict[str, float]:
        """One gradient step. Samples a batch from the replay buffer and updates weights."""
        self.model.train()

        t0 = time.perf_counter()
        boards_np, policies_np, values_np, outcome_known_np = self.buffer.sample(
            self.config.batch_size, self.rng
        )
        if self.config.mirror_augment_prob > 0:
            # Use numpy's random state so we don't reseed from a Python
            # Random; a quick uniform-sample mask is cheap.
            mask = np.random.random(len(boards_np)) < self.config.mirror_augment_prob
            boards_np, policies_np = mirror_batch(boards_np, policies_np, mask)

        # Policy label smoothing: mix a small uniform over the full policy
        # space into each target. Cheap regularizer that stops the head from
        # collapsing to zero on unvisited moves.
        eps = self.config.policy_label_smoothing
        if eps > 0:
            policies_np = (1 - eps) * policies_np + eps / policies_np.shape[1]
        t1 = time.perf_counter()

        x_flat = torch.from_numpy(boards_np).to(self.device)
        x = encoded_to_nchw(x_flat, NUM_PLANES)
        policy_target = torch.from_numpy(policies_np).to(self.device)
        wdl_target = torch.from_numpy(values_to_wdl_targets(values_np)).to(self.device)
        outcome_known = torch.from_numpy(outcome_known_np).to(self.device)
        self._device_sync()
        t2 = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self._amp_enabled
            else _nullcontext()
        )
        with amp_ctx:
            h = self.model.trunk_features(x)
            pred_policy, pred_wdl = self.model.heads(h)
            # Cross-entropy losses (targets sum to 1; pred outputs are softmax-normalized).
            policy_loss = -(policy_target * torch.log(pred_policy.clamp(min=1e-8))).sum(dim=1).mean()

            per_sample_value = -(wdl_target * torch.log(pred_wdl.clamp(min=1e-8))).sum(dim=1)
            draw_w = self.config.value_draw_weight
            # Base weighting: down-weight draws (class-imbalance correction).
            if draw_w != 1.0:
                # Draw samples have wdl_target = [0, 1, 0] exactly (values
                # from selfplay are strictly -1 / 0 / +1, so the middle
                # column of the WDL target is 1 for every drawn sample).
                is_draw = wdl_target[:, 1] > 0.5
                sample_w = torch.where(
                    is_draw,
                    torch.full_like(per_sample_value, draw_w),
                    torch.ones_like(per_sample_value),
                )
            else:
                sample_w = torch.ones_like(per_sample_value)
            # Hard mask out cap-timeout samples — their value=0 label is
            # a no-signal placeholder, not an observed draw. Keeping them
            # in the value loss pulls the head toward "predict draw" on
            # unlabeled positions. Policy loss still uses every sample
            # (the MCTS visit distributions are useful regardless).
            effective_w = sample_w * outcome_known
            total_weight = effective_w.sum().clamp_min(1e-8)
            value_loss = (per_sample_value * effective_w).sum() / total_weight

            total = policy_loss + value_loss

            # Auxiliary material head: MSE against material balance computed
            # directly from the input tensor. Shares the trunk; its gradient
            # flows back through the residual tower.
            aux_material_loss = torch.tensor(0.0, device=self.device)
            if self.aux_material is not None and self.config.aux_material_weight > 0:
                # Target is fp32 regardless of AMP; the head output cast is fine.
                mat_target = material_target_from_board(x.float())
                mat_pred = self.aux_material(h)
                aux_material_loss = F.mse_loss(mat_pred.float(), mat_target)
                total = total + self.config.aux_material_weight * aux_material_loss
        self._device_sync()
        t3 = time.perf_counter()

        if self._scaler is not None:
            self._scaler.scale(total).backward()
        else:
            total.backward()
        self._device_sync()
        t4 = time.perf_counter()

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self._device_sync()
        t5 = time.perf_counter()

        self._ema("t_sample_ms", (t1 - t0) * 1000.0)
        self._ema("t_h2d_ms", (t2 - t1) * 1000.0)
        self._ema("t_forward_ms", (t3 - t2) * 1000.0)
        self._ema("t_backward_ms", (t4 - t3) * 1000.0)
        self._ema("t_optim_ms", (t5 - t4) * 1000.0)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total.item(),
            "aux_material_loss": float(aux_material_loss.item()) if self.aux_material is not None else 0.0,
        }

    def selfplay_step(self) -> None:
        """Advance all self-play games by one move (single-process mode)."""
        self.model.eval()
        finished = self.engine.step()
        for r in finished:
            self._record_outcome(r.outcome, getattr(r, "origin", "standard"))

    def _record_outcome(self, outcome: str, origin: str = "standard") -> None:
        # Dispatch on the granular outcome label. See selfplay.GameResult for
        # the full set. `origin` selects which per-origin sub-bucket to
        # increment alongside the global counter.
        bucket = getattr(self.stats, outcome, None)
        if isinstance(bucket, int):
            setattr(self.stats, outcome, bucket + 1)
        else:
            # Unknown outcome string — fall through to "cap" so totals still
            # balance. Should never happen unless someone adds a new outcome
            # in selfplay.py without updating TrainStats.
            self.stats.cap += 1
            outcome = "cap"
        # Per-origin bucket. Unknown origins get bundled into "standard" so
        # per-origin totals stay consistent with the global aggregate.
        origin_bucket = self.stats.origin_outcomes.get(origin) or self.stats.origin_outcomes["standard"]
        if outcome in origin_bucket:
            origin_bucket[outcome] += 1
        self.stats.games_completed += 1

    def run(
        self,
        num_steps: int | None = None,
        checkpoint_dir: str | Path | None = None,
        on_step: Callable[[TrainStats], None] | None = None,
        on_eval: Callable[[dict], None] | None = None,
        on_eval_progress: Callable[..., None] | None = None,
    ) -> None:
        """Main loop: alternate self-play and gradient updates forever (or num_steps).

        `on_eval` fires once per auto-eval match completion with the match
        summary dict (see `_run_eval_match`). The dashboard uses it to refresh
        the validation panel; external callers can log / alert on it.
        """
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._on_eval = on_eval
        self._on_eval_progress = on_eval_progress

        if self._mp_self_play is not None:
            self._run_mp(num_steps, ckpt_dir, on_step)
            return

        step = 0
        buffer_size_before = len(self.buffer)
        examples_since_grad = 0
        while True:
            if num_steps is not None and step >= num_steps:
                break
            if self._stop_requested:
                break
            step += 1

            self.selfplay_step()
            examples_since_grad += max(0, len(self.buffer) - buffer_size_before)
            buffer_size_before = len(self.buffer)

            min_new = self.config.min_examples_between_grad_steps
            if (
                len(self.buffer) >= self.config.min_buffer_for_training
                and examples_since_grad >= min_new
            ):
                last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
                for _ in range(self.config.gradient_steps_per_selfplay_step):
                    last_losses = self.train_step()
                    self.stats.generation += 1
                    self._maybe_update_lr()
                self.stats.policy_loss = last_losses["policy_loss"]
                self.stats.value_loss = last_losses["value_loss"]
                self.stats.total_loss = last_losses["total_loss"]
                self.stats.aux_material_loss = last_losses.get("aux_material_loss", 0.0)
                examples_since_grad = 0

            self.stats.step = step
            self.stats.replay_size = len(self.buffer)
            self._update_rates()

            if on_step is not None:
                on_step(self.stats)

            # Checkpoint
            if ckpt_dir is not None and (time.time() - self._last_checkpoint_time) >= self.config.checkpoint_every_seconds:
                self.save_checkpoint(ckpt_dir)
                self._last_checkpoint_time = time.time()
            if ckpt_dir is not None:
                self.maybe_archive_checkpoint(ckpt_dir)
                self.maybe_run_eval(ckpt_dir)

    def _run_mp(
        self,
        num_steps: int | None,
        ckpt_dir: Path | None,
        on_step: Callable[[TrainStats], None] | None,
    ) -> None:
        """Trainer main loop when workers run in separate processes.

        Responsibilities (this process only):
          * drain completed training examples into the replay buffer
          * run gradient updates once the buffer has enough data
          * periodically broadcast fresh weights to the inference server
          * checkpoint
        """
        assert self._mp_self_play is not None
        self._mp_self_play.start()

        step = 0
        examples_since_grad = 0
        try:
            while True:
                if num_steps is not None and step >= num_steps:
                    break
                if self._stop_requested:
                    break
                step += 1
                iter_start = time.perf_counter()

                # Drain as many training examples as are available right now.
                t_a = time.perf_counter()
                drained = self._mp_self_play.drain_examples(self.buffer)
                examples_since_grad += drained

                # Drain any completed-game outcomes and update the counters.
                for result in self._mp_self_play.drain_results():
                    self._record_outcome(result.outcome, getattr(result, "origin", "standard"))
                t_b = time.perf_counter()
                self._ema("t_drain_ms", (t_b - t_a) * 1000.0)

                # Gradient updates are rate-limited by new-example arrival so
                # we don't over-train on a thin buffer (classic RL failure
                # mode that collapses the policy distribution).
                min_new = self.config.min_examples_between_grad_steps
                if (
                    len(self.buffer) >= self.config.min_buffer_for_training
                    and examples_since_grad >= min_new
                ):
                    last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
                    for _ in range(self.config.gradient_steps_per_selfplay_step):
                        last_losses = self.train_step()
                        self.stats.generation += 1
                        # Push fresh weights on a cadence.
                        if self.stats.generation % self.config.weight_broadcast_every == 0:
                            t_c = time.perf_counter()
                            self._mp_self_play.broadcast_weights(self.model.state_dict())
                            self._ema("t_broadcast_ms", (time.perf_counter() - t_c) * 1000.0)
                    self.stats.policy_loss = last_losses["policy_loss"]
                    self.stats.value_loss = last_losses["value_loss"]
                    self.stats.total_loss = last_losses["total_loss"]
                    examples_since_grad = 0
                else:
                    # Either buffer isn't ready yet, or we're waiting for
                    # fresh examples. Yield so we don't hot-loop.
                    t_s = time.perf_counter()
                    time.sleep(0.05)
                    self._ema("t_sleep_ms", (time.perf_counter() - t_s) * 1000.0)

                self.stats.step = step
                self.stats.replay_size = len(self.buffer)
                self._update_rates()
                self._ema("t_iter_ms", (time.perf_counter() - iter_start) * 1000.0)

                if on_step is not None:
                    on_step(self.stats)

                # Checkpoint
                if ckpt_dir is not None and (time.time() - self._last_checkpoint_time) >= self.config.checkpoint_every_seconds:
                    self.save_checkpoint(ckpt_dir)
                    self._last_checkpoint_time = time.time()
                if ckpt_dir is not None:
                    self.maybe_archive_checkpoint(ckpt_dir)
                    self.maybe_run_eval(ckpt_dir)
        finally:
            self._mp_self_play.stop()

        # Final save on clean exit
        if ckpt_dir is not None:
            self.save_checkpoint(ckpt_dir)

    def save_checkpoint(self, directory: str | Path) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pt_path = directory / "latest.pt"
        json_path = directory / "latest.json"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": self.stats.__dict__,
            "config": self.config.__dict__,
            # Model architecture, so the checkpoint is self-describing for
            # downstream tools (compare_checkpoints, deploy_to_browser) and
            # we don't need to re-pass --num-filters etc. on the CLI.
            "model_arch": {
                "num_res_blocks": self.model.num_res_blocks,
                "num_filters": self.model.num_filters,
                "kernel_size": self.model.kernel_size,
                "value_head_size": self.model.value_head_size,
                "se_reduction": self.model.se_reduction,
            },
        }
        # Aux heads are saved separately so they can be absent without
        # breaking ChessNet's own state-dict strictness.
        if self.aux_material is not None:
            checkpoint["aux_material_state_dict"] = self.aux_material.state_dict()
        torch.save(checkpoint, pt_path)

        self.model.eval()
        weights = export_weights(self.model, learning_rate=self.config.learning_rate)
        with json_path.open("w") as f:
            json.dump(weights, f)

        return {"pt": pt_path, "json": json_path}

    # ------------------------------------------------------------------
    # Auto-eval vs champion
    # ------------------------------------------------------------------

    def _save_champion(self, ckpt_dir: Path, gen: int) -> None:
        """Copy current model weights into `ckpt_dir / champion.pt` and note
        the generation.
        """
        payload = {
            "model_state_dict": self.model.state_dict(),
            "model_arch": {
                "num_res_blocks": self.model.num_res_blocks,
                "num_filters": self.model.num_filters,
                "kernel_size": self.model.kernel_size,
                "value_head_size": self.model.value_head_size,
                "se_reduction": self.model.se_reduction,
            },
            "champion_gen": gen,
        }
        torch.save(payload, ckpt_dir / "champion.pt")
        self._champion_gen = gen

    def _load_champion_model(self, ckpt_dir: Path) -> ChessNet:
        """Instantiate a ChessNet matching the saved champion and load its
        weights. Cached on self._champion_model for reuse across evals.
        """
        path = ckpt_dir / "champion.pt"
        state = torch.load(path, map_location=self.device)
        arch = state["model_arch"]
        champ = ChessNet(**arch).to(self.device)
        champ.load_state_dict(state["model_state_dict"])
        champ.eval()
        self._champion_gen = int(state.get("champion_gen", 0))
        return champ

    def _make_model_evaluator(self, m: ChessNet):
        """Return a batched MCTS-style evaluator bound to the given model."""
        from .selfplay import make_pytorch_evaluator
        return make_pytorch_evaluator(m, self.device)

    def _play_eval_game(
        self,
        challenger_eval,
        champion_eval,
        challenger_plays_white: bool,
        mcts_sims: int,
        move_cap: int,
        starting_state: "ChessGameState | None" = None,
    ) -> str:
        """One eval game. Returns "challenger", "champion", or "draw".

        `starting_state`, if given, is deep-copied so repeated matches from
        the same curated position don't share mutable state. Falls back to
        the standard initial position when None (legacy behavior).
        """
        import copy

        from .engine import apply_move, create_initial_game_state
        from .mcts import run_batched_mcts

        if starting_state is None:
            state = create_initial_game_state()
            state.status = "active"
        else:
            state = copy.deepcopy(starting_state)
            state.status = "active"
        moves_played = 0
        while True:
            if state.status in ("checkmate", "stalemate", "draw"):
                break
            if moves_played >= move_cap:
                return "draw"
            if state.currentTurn == "white":
                ev = challenger_eval if challenger_plays_white else champion_eval
            else:
                ev = champion_eval if challenger_plays_white else challenger_eval
            # Greedy play: τ=0 (argmax visits).
            result = run_batched_mcts([state], ev, mcts_sims, self.rng, temperatures=[0.0])
            state = apply_move(state, result[0].move)
            moves_played += 1

        if state.status == "checkmate":
            loser = state.currentTurn
            winner_color = "white" if loser == "black" else "black"
            winner_is_white = winner_color == "white"
            winner_is_challenger = winner_is_white == challenger_plays_white
            return "challenger" if winner_is_challenger else "champion"
        return "draw"

    def _run_eval_match(self, ckpt_dir: Path) -> dict:
        """Play `eval_games` games vs the champion, alternating colors.

        Updates the champion and resets the plateau counter if the challenger
        clears `eval_score_threshold`. Returns a summary dict and appends it
        to self._eval_history + eval.csv.
        """
        import csv
        import math

        gen = self.stats.generation
        champ_path = ckpt_dir / "champion.pt"

        # Bootstrap: if no champion exists yet, crown the current model and
        # skip the match (we can't evaluate against ourselves).
        if not champ_path.exists():
            self._save_champion(ckpt_dir, gen)
            result = {
                "gen": gen,
                "champion_gen": gen,
                "games": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "score": 0.5,
                "elo_diff": 0.0,
                "new_champion": True,
                "plateau_counter": 0,
                "note": "bootstrap",
            }
            self._eval_history.append(result)
            return result

        # Fire an immediate "0 games done" progress tick before any of
        # the slow setup (loading champion, building evaluators) so the
        # dashboard's validation panel shows `running 0/N` as soon as
        # the match begins. Otherwise the first visible update waits on
        # game-1 completion, which at 100 MCTS sims is 30-60s of
        # apparent-silence in the panel.
        match_start = time.time()
        if self._on_eval_progress is not None:
            try:
                self._on_eval_progress(
                    0, self.config.eval_games, 0, 0, 0, {},
                    current=None, recent=[], elapsed_s=0.0,
                )
            except Exception:
                pass

        # Load (or reload) champion. Reload whenever it changed.
        if self._champion_model is None or self._champion_gen_on_disk(ckpt_dir) != self._champion_gen:
            self._champion_model = self._load_champion_model(ckpt_dir)

        # Flip current model to eval for the match so BN uses running stats.
        was_training = self.model.training
        self.model.eval()
        try:
            challenger_eval = self._make_model_evaluator(self.model)
            champion_eval = self._make_model_evaluator(self._champion_model)

            # Pull the curated position set. Built once per trainer and
            # cached; building plays a few moves per opening through our
            # own engine so every position is guaranteed legal.
            from .eval_positions import build_eval_positions
            positions = build_eval_positions()

            wins = draws = losses = 0
            total = self.config.eval_games
            # Per-difficulty breakdown. Lets the dashboard show how the
            # match is going in each category (mate-in-1, trivial, clear,
            # balanced) separately — much more informative than one score.
            per_diff: dict[str, dict[str, int]] = {}
            # Rolling outcomes so the dashboard can show a recent-results
            # strip ("W W D L W …") that gives an immediate visual read
            # on momentum during the long eval match.
            from collections import deque
            recent: deque[str] = deque(maxlen=14)
            # Each position plays twice — once with challenger as white,
            # once as black — so any intrinsic imbalance in asymmetric
            # positions (K+Q vs K, etc.) averages across the pair.
            for g in range(total):
                position = positions[(g // 2) % len(positions)]
                challenger_white = (g % 2 == 0)
                current = {
                    "game": g + 1,
                    "pos_name": position.name,
                    "difficulty": position.difficulty,
                    "challenger_white": challenger_white,
                }
                # Pre-game tick so the dashboard's "now" row updates
                # before we disappear into a 30-60s game-play block.
                if self._on_eval_progress is not None:
                    try:
                        self._on_eval_progress(
                            g, total, wins, draws, losses, per_diff,
                            current=current,
                            recent=list(recent),
                            elapsed_s=time.time() - match_start,
                        )
                    except Exception:
                        pass
                outcome = self._play_eval_game(
                    challenger_eval,
                    champion_eval,
                    challenger_white,
                    self.config.eval_mcts_sims,
                    self.config.eval_move_cap,
                    starting_state=position.state,
                )
                bucket = per_diff.setdefault(
                    position.difficulty, {"w": 0, "d": 0, "l": 0}
                )
                if outcome == "challenger":
                    wins += 1
                    bucket["w"] += 1
                    recent.append("W")
                elif outcome == "champion":
                    losses += 1
                    bucket["l"] += 1
                    recent.append("L")
                else:
                    draws += 1
                    bucket["d"] += 1
                    recent.append("D")
                # Post-game tick. Wrapped in try/except because a
                # dashboard hiccup shouldn't abort the whole eval match.
                if self._on_eval_progress is not None:
                    try:
                        self._on_eval_progress(
                            g + 1, total, wins, draws, losses, per_diff,
                            current=current,
                            recent=list(recent),
                            elapsed_s=time.time() - match_start,
                        )
                    except Exception:
                        pass
        finally:
            if was_training:
                self.model.train()

        total = wins + draws + losses
        score = (wins + 0.5 * draws) / max(1, total)
        # Clamp before the logit to avoid infinities at boundaries.
        s = max(0.01, min(0.99, score))
        elo_diff = -400.0 * math.log10(1.0 / s - 1.0)

        new_champion = score >= self.config.eval_score_threshold
        if new_champion:
            self._save_champion(ckpt_dir, gen)
            self._champion_model = None  # force reload next eval
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1

        result = {
            "gen": gen,
            "champion_gen": self._champion_gen,
            "games": total,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": round(score, 4),
            "elo_diff": round(elo_diff, 1),
            "new_champion": new_champion,
            "plateau_counter": self._plateau_counter,
            # Wall-clock duration of the match. Useful for spotting
            # eval regressions (e.g. a match that suddenly takes 2×
            # longer could mean MCTS is stuck exploring dead-end lines).
            "duration_s": round(time.time() - match_start, 1),
            # Per-difficulty breakdown — lets the dashboard / event log
            # show which bucket the challenger dominated. Excluded from
            # eval.csv below (dict values don't fit a flat CSV).
            "per_diff": per_diff,
        }
        self._eval_history.append(result)

        # Append to eval.csv. Exclude per_diff (nested dict).
        csv_path = ckpt_dir / "eval.csv"
        write_header = not csv_path.exists()
        csv_row = {k: v for k, v in result.items() if k != "per_diff"}
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)

        return result

    def _champion_gen_on_disk(self, ckpt_dir: Path) -> int:
        """Peek at champion.pt's champion_gen without loading weights."""
        try:
            state = torch.load(ckpt_dir / "champion.pt", map_location="cpu")
            return int(state.get("champion_gen", 0))
        except Exception:
            return 0

    def maybe_run_eval(self, ckpt_dir: Path) -> dict | None:
        """Run an eval match if enough generations have passed since the
        previous one. Returns the match summary or None.

        Also consults the plateau stop condition: if max_plateau_evals > 0
        and the plateau counter reaches that threshold, sets
        self._stop_requested so the main loop exits cleanly.
        """
        if self.config.eval_every_gens <= 0:
            return None
        gen = self.stats.generation
        if gen - self._last_eval_gen < self.config.eval_every_gens:
            return None
        if gen == 0:
            return None
        self._last_eval_gen = gen
        result = self._run_eval_match(ckpt_dir)
        if self._on_eval is not None:
            try:
                self._on_eval(result)
            except Exception:
                pass
        if (
            self.config.max_plateau_evals > 0
            and self._plateau_counter >= self.config.max_plateau_evals
        ):
            self._stop_requested = True
        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def maybe_archive_checkpoint(self, ckpt_dir: Path) -> Path | None:
        """Write `archive/gen-<N>.pt` when the generation cadence says it's time.

        Returns the archive path on success, None if nothing was written.
        Retention is applied after each write: oldest archives beyond
        `keep_archives` are deleted.
        """
        if self.config.archive_every_gens <= 0:
            return None
        gen = self.stats.generation
        if gen - self._last_archive_gen < self.config.archive_every_gens:
            return None
        if gen == 0:
            return None

        archive_dir = ckpt_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"gen-{gen}.pt"

        # Reuse save_checkpoint by writing to a temp dir then moving the .pt.
        # Simpler: just replicate the save payload here (cheap vs train_step).
        payload = {
            "model_state_dict": self.model.state_dict(),
            "stats": self.stats.__dict__,
            "config": self.config.__dict__,
            "model_arch": {
                "num_res_blocks": self.model.num_res_blocks,
                "num_filters": self.model.num_filters,
                "kernel_size": self.model.kernel_size,
                "value_head_size": self.model.value_head_size,
                "se_reduction": self.model.se_reduction,
            },
        }
        if self.aux_material is not None:
            payload["aux_material_state_dict"] = self.aux_material.state_dict()
        # Archives intentionally exclude optimizer state — they're for eval /
        # comparison, not for resuming training. Saves disk at scale.
        torch.save(payload, archive_path)

        self._last_archive_gen = gen
        self._enforce_archive_retention(archive_dir)
        return archive_path

    def _enforce_archive_retention(self, archive_dir: Path) -> None:
        if self.config.keep_archives <= 0:
            return
        # Sort archives by the generation embedded in the filename.
        def _gen_of(p: Path) -> int:
            try:
                return int(p.stem.removeprefix("gen-"))
            except ValueError:
                return -1

        archives = sorted(archive_dir.glob("gen-*.pt"), key=_gen_of)
        excess = len(archives) - self.config.keep_archives
        for p in archives[:max(0, excess)]:
            try:
                p.unlink()
            except OSError:
                pass

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if self.aux_material is not None and "aux_material_state_dict" in state:
            self.aux_material.load_state_dict(state["aux_material_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])


def pick_device(preferred: str = "auto") -> torch.device:
    """Pick the best available device (CUDA > MPS > CPU).

    With multiple CUDA GPUs, rank them by (compute capability, total memory)
    and pick the best — so a 3090 beats a 1080 Ti beats a 1070. Logs all
    visible GPUs and which one was chosen.
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred in ("cuda", "auto") and torch.cuda.is_available():
        count = torch.cuda.device_count()
        scored = []
        for i in range(count):
            p = torch.cuda.get_device_properties(i)
            # Sort key: higher compute capability first, then more memory.
            scored.append(((p.major, p.minor, p.total_memory), i, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        for (cc, mem), idx, props in (((s[0][:2], s[0][2]), s[1], s[2]) for s in scored):
            log.info(
                "CUDA:%d %s sm_%d%d mem=%.1fGiB",
                idx, props.name, cc[0], cc[1], mem / (1024 ** 3),
            )
        best_idx = scored[0][1]
        best_props = scored[0][2]
        log.info("Picked CUDA:%d (%s)", best_idx, best_props.name)
        return torch.device(f"cuda:{best_idx}")
    if preferred in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
