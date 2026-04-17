"""Smoke test the full train loop: self-play → buffer → gradient updates → checkpoint.

Uses a tiny model + tiny buffer so it finishes in seconds. We don't assert
the model improves — we just assert the plumbing runs without crashes, that
losses are finite, and that the checkpoint artifacts exist and are loadable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from chess_ai.model import ChessNet
from chess_ai.train import TrainConfig, Trainer, values_to_wdl_targets
from chess_ai.weight_io import import_weights


def test_values_to_wdl_targets_shape_and_sum():
    import numpy as np

    v = np.array([1.0, -1.0, 0.0, 0.5, -0.3, 10.0, -10.0], dtype=np.float32)
    wdl = values_to_wdl_targets(v)
    assert wdl.shape == (7, 3)
    assert wdl.sum(axis=1).tolist() == pytest.approx([1.0] * 7, abs=1e-6)
    # v=1 → pure win; v=-1 → pure loss; v=0 → pure draw
    assert wdl[0].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert wdl[1].tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert wdl[2].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_train_loop_runs_and_checkpoints(tmp_path: Path):
    torch.manual_seed(0)
    model = ChessNet(
        num_res_blocks=2,
        num_filters=16,
        kernel_size=3,
        value_head_size=16,
        se_reduction=4,
    )

    config = TrainConfig(
        num_concurrent_games=4,
        mcts_simulations=4,
        batch_size=16,
        gradient_steps_per_selfplay_step=1,
        # Tiny smoke config: don't rate-limit gradient steps, we just want
        # to verify the whole loop runs + checkpoints.
        min_examples_between_grad_steps=1,
        learning_rate=1e-3,
        replay_buffer_capacity=500,
        min_buffer_for_training=8,        # Low threshold so training kicks in fast
        checkpoint_every_seconds=0.0,     # Checkpoint every step (for testing)
    )

    trainer = Trainer(
        model=model, device=torch.device("cpu"), config=config, rng=random.Random(7)
    )

    checkpoint_dir = tmp_path / "ckpt"
    trainer.run(num_steps=40, checkpoint_dir=checkpoint_dir)

    # Artifacts exist
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "latest.json").exists()

    # JSON is valid SerializedWeights (has config + shapes + data, shapes match data count)
    with (checkpoint_dir / "latest.json").open() as f:
        serialized = json.load(f)
    assert "config" in serialized
    assert "shapes" in serialized and "data" in serialized
    assert len(serialized["shapes"]) == len(serialized["data"])

    # Round-trip: load JSON into a fresh PyTorch model, confirm weights are equal.
    fresh = ChessNet(
        num_res_blocks=serialized["config"]["numResBlocks"],
        num_filters=serialized["config"]["numFilters"],
        kernel_size=serialized["config"]["kernelSize"],
        value_head_size=serialized["config"]["valueHeadSize"],
        se_reduction=serialized["config"]["seReduction"],
    )
    import_weights(fresh, serialized)

    # Forward pass on a fixed input should match between the two models
    fresh.eval()
    model.eval()
    x = torch.randn(2, 20, 8, 8)
    with torch.no_grad():
        p1, w1 = model(x)
        p2, w2 = fresh(x)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(w1, w2, atol=1e-5)

    # Stats have sensible values
    assert trainer.stats.step == 40
    assert trainer.stats.generation > 0
    assert trainer.stats.replay_size > 0
    # Losses are finite
    import math
    assert math.isfinite(trainer.stats.policy_loss)
    assert math.isfinite(trainer.stats.value_loss)
