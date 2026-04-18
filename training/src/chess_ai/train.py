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
    # Left-right (file) mirror augmentation on sampled batches. Chess is
    # symmetric about the file axis, so this is a free 2x data multiplier.
    # 0.0 disables. 0.5 mirrors half the batch each step (standard).
    mirror_augment_prob: float = 0.5
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
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0                   # Stalemate / 50-move / actual draw only
    caps: int = 0                    # Hit move-cap before anything decisive
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
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
                syzygy_path=self.config.syzygy_path,
                syzygy_max_pieces=self.config.syzygy_max_pieces,
                rewards=self.config.rewards,
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
                ),
                rng=self.rng,
            )

        self.stats = TrainStats(target_gens=self.config.target_gens)
        self._start_time = time.time()
        self._last_checkpoint_time = 0.0
        self._last_archive_gen = 0
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
        boards_np, policies_np, values_np = self.buffer.sample(self.config.batch_size, self.rng)
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
            value_loss = -(wdl_target * torch.log(pred_wdl.clamp(min=1e-8))).sum(dim=1).mean()
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
            self._record_outcome(r.outcome)

    def _record_outcome(self, outcome: str) -> None:
        if outcome == "white":
            self.stats.white_wins += 1
        elif outcome == "black":
            self.stats.black_wins += 1
        elif outcome == "cap":
            self.stats.caps += 1
        else:
            # "draw" or "stalemate" — anything decisive that isn't a
            # checkmate or a move-cap timeout.
            self.stats.draws += 1
        self.stats.games_completed += 1

    def run(
        self,
        num_steps: int | None = None,
        checkpoint_dir: str | Path | None = None,
        on_step: Callable[[TrainStats], None] | None = None,
    ) -> None:
        """Main loop: alternate self-play and gradient updates forever (or num_steps)."""
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self._mp_self_play is not None:
            self._run_mp(num_steps, ckpt_dir, on_step)
            return

        step = 0
        buffer_size_before = len(self.buffer)
        examples_since_grad = 0
        while True:
            if num_steps is not None and step >= num_steps:
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
                step += 1
                iter_start = time.perf_counter()

                # Drain as many training examples as are available right now.
                t_a = time.perf_counter()
                drained = self._mp_self_play.drain_examples(self.buffer)
                examples_since_grad += drained

                # Drain any completed-game outcomes and update the counters.
                for result in self._mp_self_play.drain_results():
                    self._record_outcome(result.outcome)
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
    """Pick the best available device (CUDA > MPS > CPU)."""
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
