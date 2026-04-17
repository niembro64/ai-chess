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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .model import NUM_PLANES, WDL_SIZE, ChessNet, encoded_to_nchw
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights
from .selfplay import ReplayBuffer, SelfPlayConfig, make_local_selfplay_engine
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
    # Replay
    replay_buffer_capacity: int = 100_000
    min_buffer_for_training: int = 2_000
    # Checkpointing
    checkpoint_every_seconds: float = 60.0
    # Logging
    log_every_steps: int = 10
    # Reward shaping
    rewards: RewardWeights = field(default_factory=lambda: RewardWeights(**DEFAULT_REWARD_WEIGHTS.__dict__))


@dataclass
class TrainStats:
    step: int = 0
    generation: int = 0              # Count of gradient steps
    games_completed: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    replay_size: int = 0
    games_per_min: float = 0.0


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
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
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
                    rewards=self.config.rewards,
                ),
                rng=self.rng,
            )

        self.stats = TrainStats()
        self._start_time = time.time()
        self._last_checkpoint_time = 0.0

    def train_step(self) -> dict[str, float]:
        """One gradient step. Samples a batch from the replay buffer and updates weights."""
        self.model.train()
        boards_np, policies_np, values_np = self.buffer.sample(self.config.batch_size, self.rng)

        x_flat = torch.from_numpy(boards_np).to(self.device)
        x = encoded_to_nchw(x_flat, NUM_PLANES)
        policy_target = torch.from_numpy(policies_np).to(self.device)
        wdl_target = torch.from_numpy(values_to_wdl_targets(values_np)).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        pred_policy, pred_wdl = self.model(x)

        # Cross-entropy losses (targets sum to 1; pred outputs are softmax-normalized).
        policy_loss = -(policy_target * torch.log(pred_policy.clamp(min=1e-8))).sum(dim=1).mean()
        value_loss = -(wdl_target * torch.log(pred_wdl.clamp(min=1e-8))).sum(dim=1).mean()

        total = policy_loss + value_loss
        total.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total.item(),
        }

    def selfplay_step(self) -> None:
        """Advance all self-play games by one move (single-process mode only).

        Note: games_completed / W/B/D counters are tracked by the engine, not
        here. In MP mode self-play happens in worker processes and we can't
        sample per-step GameResults cheaply; instead the trainer counts
        examples received via the example queue.
        """
        self.model.eval()
        finished = self.engine.step()
        for r in finished:
            if r.outcome == "white":
                self.stats.white_wins += 1
            elif r.outcome == "black":
                self.stats.black_wins += 1
            else:
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
        while True:
            if num_steps is not None and step >= num_steps:
                break
            step += 1

            self.selfplay_step()

            if len(self.buffer) >= self.config.min_buffer_for_training:
                last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
                for _ in range(self.config.gradient_steps_per_selfplay_step):
                    last_losses = self.train_step()
                    self.stats.generation += 1
                self.stats.policy_loss = last_losses["policy_loss"]
                self.stats.value_loss = last_losses["value_loss"]
                self.stats.total_loss = last_losses["total_loss"]

            self.stats.step = step
            self.stats.replay_size = len(self.buffer)
            elapsed_min = (time.time() - self._start_time) / 60.0
            self.stats.games_per_min = self.stats.games_completed / elapsed_min if elapsed_min > 0 else 0.0

            if on_step is not None:
                on_step(self.stats)

            # Checkpoint
            if ckpt_dir is not None and (time.time() - self._last_checkpoint_time) >= self.config.checkpoint_every_seconds:
                self.save_checkpoint(ckpt_dir)
                self._last_checkpoint_time = time.time()

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
        try:
            while True:
                if num_steps is not None and step >= num_steps:
                    break
                step += 1

                # Drain as many training examples as are available right now.
                self._mp_self_play.drain_examples(self.buffer)

                # Drain any completed-game outcomes and update the counters.
                for result in self._mp_self_play.drain_results():
                    if result.outcome == "white":
                        self.stats.white_wins += 1
                    elif result.outcome == "black":
                        self.stats.black_wins += 1
                    else:
                        # Includes both "cap" and "draw"/"stalemate" — all
                        # render as draws in the outcomes panel.
                        self.stats.draws += 1
                    self.stats.games_completed += 1

                # Gradient updates run as fast as the buffer allows.
                if len(self.buffer) >= self.config.min_buffer_for_training:
                    last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
                    for _ in range(self.config.gradient_steps_per_selfplay_step):
                        last_losses = self.train_step()
                        self.stats.generation += 1
                        # Push fresh weights on a cadence.
                        if self.stats.generation % self.config.weight_broadcast_every == 0:
                            self._mp_self_play.broadcast_weights(self.model.state_dict())
                    self.stats.policy_loss = last_losses["policy_loss"]
                    self.stats.value_loss = last_losses["value_loss"]
                    self.stats.total_loss = last_losses["total_loss"]
                else:
                    # Nothing to train on yet: yield so we don't hot-loop the
                    # queue drain.
                    time.sleep(0.05)

                self.stats.step = step
                self.stats.replay_size = len(self.buffer)
                elapsed_min = (time.time() - self._start_time) / 60.0
                self.stats.games_per_min = (
                    self.stats.games_completed / elapsed_min if elapsed_min > 0 else 0.0
                )

                if on_step is not None:
                    on_step(self.stats)

                # Checkpoint
                if ckpt_dir is not None and (time.time() - self._last_checkpoint_time) >= self.config.checkpoint_every_seconds:
                    self.save_checkpoint(ckpt_dir)
                    self._last_checkpoint_time = time.time()
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

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "stats": self.stats.__dict__,
                "config": self.config.__dict__,
            },
            pt_path,
        )

        self.model.eval()
        weights = export_weights(self.model, learning_rate=self.config.learning_rate)
        with json_path.open("w") as f:
            json.dump(weights, f)

        return {"pt": pt_path, "json": json_path}

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
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
