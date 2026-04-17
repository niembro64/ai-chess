"""Smoke test: multiprocessing self-play actually spawns workers + inference
server, examples flow through the queue, gradient steps run.

Skipped on platforms where torch's spawn start method has known issues
(rare, but e.g., some test runners preload CUDA before fork).
"""

from __future__ import annotations

import random
import time

import torch

from chess_ai.model import ChessNet
from chess_ai.train import TrainConfig, Trainer


def test_mp_selfplay_fills_buffer_and_trains():
    # Tiny config so the whole test runs in <30s on a laptop CPU.
    model = ChessNet(
        num_res_blocks=2,
        num_filters=16,
        kernel_size=3,
        value_head_size=16,
        se_reduction=4,
    )

    config = TrainConfig(
        mcts_simulations=4,
        num_workers=2,
        games_per_worker=4,
        mp_batch_wait_ms=2.0,
        weight_broadcast_every=5,
        batch_size=16,
        gradient_steps_per_selfplay_step=1,
        learning_rate=1e-3,
        replay_buffer_capacity=500,
        min_buffer_for_training=16,   # low so we start training quickly
        checkpoint_every_seconds=1e9, # don't bother with disk during test
    )

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        config=config,
        rng=random.Random(7),
    )

    # Run for a bounded number of main-loop iterations. Each iteration drains
    # whatever examples arrived, so real self-play progress depends on wall
    # clock rather than the step count. Give it up to ~20s to produce data.
    deadline = time.time() + 20.0
    step_cap = 500
    step = 0

    def on_step(stats):
        pass

    # Run inline for stop condition control (can't easily stop Trainer.run
    # mid-loop without SIGINT). We mimic its inner loop.
    trainer._mp_self_play.start()
    try:
        examples_seen = 0
        while step < step_cap and time.time() < deadline:
            step += 1
            drained = trainer._mp_self_play.drain_examples(trainer.buffer)
            examples_seen += drained

            if len(trainer.buffer) >= config.min_buffer_for_training:
                for _ in range(config.gradient_steps_per_selfplay_step):
                    trainer.train_step()
                    trainer.stats.generation += 1

            time.sleep(0.05)

        assert examples_seen > 0, "No TrainingExamples flowed through the example queue"
        assert len(trainer.buffer) > 0, "Replay buffer is still empty"
        assert trainer.stats.generation > 0, "No gradient steps were run"
    finally:
        trainer._mp_self_play.stop()
