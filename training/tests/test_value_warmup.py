"""Value-head-only warmup tests.

Warm-starting converged weights whose value head must relearn a new
label regime sends huge early value gradients through the shared trunk
(the July 2026 fine-tune lost ~170 Elo in its first 10k steps this
way). While `stats.generation < stats.value_warmup_until_gen`, the
trainer freezes every non-`value_*` parameter and runs the optimizer at
`config.value_warmup_lr`; on expiry everything unfreezes and the LR
returns to the schedule.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_ai.model import ChessNet
from chess_ai.train import TrainConfig, Trainer


def _make_trainer(warmup_until: int, generation: int) -> Trainer:
    torch.manual_seed(0)
    np.random.seed(0)
    model = ChessNet(
        num_res_blocks=2, num_filters=16, kernel_size=3,
        value_head_size=16, se_reduction=4,
    )
    config = TrainConfig(
        batch_size=8,
        replay_buffer_capacity=64,
        min_buffer_for_training=1,
        num_workers=0,
        num_concurrent_games=1,
        mcts_simulations=1,
        use_amp=False,
        aux_material_weight=0.0,
        mirror_augment_prob=0.0,
        learning_rate=1e-3,
        lr_schedule=((0, 1e-3), (100, 3e-5)),
        value_warmup_lr=7e-4,   # distinctive value for assertions
    )
    trainer = Trainer(model=model, device=torch.device("cpu"), config=config)
    trainer.stats.generation = generation
    trainer.stats.value_warmup_until_gen = warmup_until

    from chess_ai.encoding import NUM_PLANES, POLICY_SIZE
    from chess_ai.selfplay import TrainingExample
    rng = np.random.default_rng(0)
    for v in (1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0):
        policy = rng.random(POLICY_SIZE).astype(np.float32)
        policy /= policy.sum()
        trainer.buffer.add(TrainingExample(
            board=rng.random(8 * 8 * NUM_PLANES).astype(np.float32),
            policy=policy,
            value=float(v),
        ))
    return trainer


def _snapshot(model: ChessNet) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_warmup_trains_only_value_head():
    trainer = _make_trainer(warmup_until=1000, generation=500)
    before = _snapshot(trainer.model)
    trainer.train_step()
    after = _snapshot(trainer.model)

    value_moved = trunk_moved = 0
    for name in before:
        changed = not torch.equal(before[name], after[name])
        if name.startswith("value_"):
            value_moved += changed
        else:
            trunk_moved += changed
    assert value_moved > 0, "value head must train during warmup"
    assert trunk_moved == 0, "trunk/policy must be frozen during warmup"


def test_warmup_uses_hot_lr_then_returns_to_schedule():
    trainer = _make_trainer(warmup_until=1000, generation=500)
    trainer.train_step()
    trainer._maybe_update_lr()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)

    # Cross the warmup boundary: LR must come from the schedule again
    # (gen 1000 ≥ threshold 100 → 3e-5) and the trunk must unfreeze.
    trainer.stats.generation = 1000
    trainer._maybe_update_lr()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)

    before = _snapshot(trainer.model)
    trainer.train_step()
    after = _snapshot(trainer.model)
    trunk_moved = sum(
        not torch.equal(before[n], after[n])
        for n in before if not n.startswith("value_")
    )
    assert trunk_moved > 0, "trunk must train again after warmup expires"
    assert all(p.requires_grad for p in trainer.model.parameters())


def test_no_warmup_is_default_behavior():
    trainer = _make_trainer(warmup_until=0, generation=500)
    before = _snapshot(trainer.model)
    trainer.train_step()
    after = _snapshot(trainer.model)
    trunk_moved = sum(
        not torch.equal(before[n], after[n])
        for n in before if not n.startswith("value_")
    )
    assert trunk_moved > 0


def test_warmup_survives_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer(warmup_until=1000, generation=500)
    trainer.train_step()
    trainer.save_checkpoint(tmp_path)

    fresh = _make_trainer(warmup_until=0, generation=0)
    fresh.load_checkpoint(tmp_path / "latest.pt")
    assert fresh.stats.value_warmup_until_gen == 1000
    # generation increments in the run loop, not train_step itself.
    assert fresh.stats.generation == 500
    # LR restored to the warmup rate, not the schedule rate.
    assert fresh.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)
