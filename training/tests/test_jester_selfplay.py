"""Jester-mode self-play + Trainer integration.

Covers the asymmetric game ecology (jester-vs-frozen slots with an
assigned agent color, jester-vs-jester mirror slots), the
agent-plies-only example gating, and the full Trainer loop in jester
mode with a frozen-opponent checkpoint.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from chess_ai.encoding import POLICY_SIZE
from chess_ai.model import ChessNet
from chess_ai.selfplay import ReplayBuffer, SelfPlayConfig, SelfPlayEngine
from chess_ai.train import TrainConfig, Trainer, _model_arch_dict


def _uniform_evaluator(boards: np.ndarray):
    batch = boards.shape[0]
    return (
        np.full((batch, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32),
        np.zeros(batch, dtype=np.float32),
    )


def _tiny_chessnet() -> ChessNet:
    return ChessNet(
        num_res_blocks=1, num_filters=16, kernel_size=3,
        value_head_size=8, se_reduction=4,
    )


def test_slot_mix_and_example_gating():
    """Slots split between mirror (agent_color=None) and vs-frozen games
    per agent_selfplay_prob; in vs-frozen games only agent plies are
    recorded as training examples."""
    examples: list = []
    config = SelfPlayConfig(
        num_concurrent_games=12,
        mcts_simulations=2,
        endgame_start_prob=0.0,
        random_start_prob=0.0,
        invert_agent_selection=True,
        frozen_evaluator=_uniform_evaluator,
        agent_selfplay_prob=0.5,
    )
    engine = SelfPlayEngine(
        _uniform_evaluator, examples.append, config, random.Random(11)
    )

    colors = {g.agent_color for g in engine.games}
    assert None in colors, "no mirror (jester-vs-jester) slots assigned"
    assert colors & {"white", "black"}, "no vs-frozen slots assigned"

    for _ in range(6):
        engine.step()

    for slot in engine.games:
        if slot.agent_color is None:
            # Mirror game: every ply is the agent — one example per move.
            assert len(slot.examples) == slot.move_count
        else:
            # vs-frozen: only the agent's plies were recorded.
            assert len(slot.examples) <= (slot.move_count + 1) // 2 + 1
            for ex in slot.examples:
                assert ex.turn_color == slot.agent_color


def test_no_frozen_evaluator_means_all_mirror():
    config = SelfPlayConfig(
        num_concurrent_games=6,
        mcts_simulations=2,
        invert_agent_selection=True,
        frozen_evaluator=None,
    )
    engine = SelfPlayEngine(
        _uniform_evaluator, lambda ex: None, config, random.Random(5)
    )
    assert all(g.agent_color is None for g in engine.games)


def test_jester_trainer_end_to_end(tmp_path):
    """The real Trainer in jester mode: frozen opponent loaded from a
    checkpoint, dual-net inverted self-play, gradient steps, truthful
    value labels — end to end on CPU."""
    torch.manual_seed(0)

    frozen = _tiny_chessnet()
    torch.save(
        {
            "model_state_dict": frozen.state_dict(),
            "model_arch": _model_arch_dict(frozen),
        },
        tmp_path / "frozen.pt",
    )

    model = _tiny_chessnet()
    config = TrainConfig(
        num_workers=0,
        num_concurrent_games=4,
        mcts_simulations=4,
        batch_size=16,
        min_examples_between_grad_steps=1,
        replay_buffer_capacity=2_000,
        min_buffer_for_training=8,
        use_amp=False,
        aux_material_weight=0.0,
        # All-endgame starts: short caps guarantee games finish inside
        # the step budget regardless of seed.
        endgame_start_prob=1.0,
        random_start_prob=0.0,
        resign_threshold=-2.0,           # disabled (misère play)
        checkpoint_every_seconds=1e9,
        eval_every_gens=10_000_000,      # no eval during smoke
        jester_mode=True,
        jester_selfplay_prob=0.5,
        jester_opponent_checkpoint=str(tmp_path / "frozen.pt"),
    )
    trainer = Trainer(
        model=model, device=torch.device("cpu"), config=config,
        rng=random.Random(7),
    )
    assert trainer._frozen_opponent is not None
    trainer.run(num_steps=300, checkpoint_dir=tmp_path)

    assert trainer.stats.generation > 0, "gradient steps must have run"
    assert trainer.stats.games_completed > 0
    assert (tmp_path / "latest.pt").exists()


def test_jester_mode_rejects_mp_workers():
    import pytest

    model = _tiny_chessnet()
    config = TrainConfig(
        num_workers=2, jester_mode=True, jester_opponent_checkpoint="x.pt"
    )
    with pytest.raises(ValueError, match="jester_mode"):
        Trainer(model=model, device=torch.device("cpu"), config=config,
                rng=random.Random(1))
