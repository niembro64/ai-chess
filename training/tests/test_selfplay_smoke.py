"""Smoke test for the self-play pipeline.

Runs a tiny model through a handful of batched-MCTS steps to make sure the
whole pipeline wires together without exceptions. Doesn't check quality —
the network is random-initialized — just that games progress, games finish,
and the replay buffer fills.
"""

from __future__ import annotations

import random

import pytest
import torch

from chess_ai.model import ChessNet
from chess_ai.selfplay import ReplayBuffer, SelfPlayConfig, make_local_selfplay_engine


@pytest.fixture(scope="module")
def tiny_model() -> ChessNet:
    torch.manual_seed(0)
    m = ChessNet(
        num_res_blocks=2,
        num_filters=16,
        kernel_size=3,
        value_head_size=16,
        se_reduction=4,
    )
    m.eval()
    return m


def test_selfplay_produces_examples(tiny_model: ChessNet):
    buffer = ReplayBuffer(capacity=500)
    engine = make_local_selfplay_engine(
        model=tiny_model,
        device=torch.device("cpu"),
        replay_buffer=buffer,
        config=SelfPlayConfig(num_concurrent_games=4, mcts_simulations=4),
        rng=random.Random(123),
    )

    total_finished = 0
    for _ in range(60):
        finished = engine.step()
        total_finished += len(finished)
        if total_finished >= 2:
            break

    assert total_finished >= 1, "Self-play should finish at least one game within 60 steps"
    assert len(buffer) > 0, "Replay buffer should be populated once games finish"

    # Verify replay sample() works and returns the expected shapes.
    boards, policies, values = buffer.sample(batch_size=8)
    assert boards.shape == (8, 8 * 8 * 20)
    assert policies.shape == (8, 4096)
    assert values.shape == (8,)
    assert (values >= -1.0).all() and (values <= 1.0).all()
