"""Smoke tests for Toy as a first-class pipeline citizen (chess_ai.toy).

Pins the conventions the browser port depends on — the 6-plane encoding
(perspective rotation + sign), the fixed weight-export order/shapes
with the WDL value head — and proves the SAGE trainer machinery runs
ToyNet end to end (self-play → buffer → gradient steps → checkpoint →
family-tagged reload).
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
import torch

from chess_ai.engine import create_initial_game_state
from chess_ai.toy import ToyNet, encode_toy, PIECE_CHANNEL
from chess_ai.train import TrainConfig, Trainer, _build_model_from_arch


def test_encoding_rotation_symmetry():
    """The 180° mover-perspective rotation (rank AND file — same
    convention as Sage) means black's view of the start position equals
    white's view FILE-MIRRORED: the start position is rank-mirror
    color-symmetric, and rotation = rank-mirror + file-mirror."""
    s_white = create_initial_game_state()
    s_black = create_initial_game_state()
    s_black.currentTurn = "black"
    ew = encode_toy(s_white).reshape(8, 8, 6)
    eb = encode_toy(s_black).reshape(8, 8, 6)
    assert np.array_equal(eb, ew[:, ::-1, :])
    assert not np.array_equal(eb, ew)  # rotation swaps king/queen files


def test_encoding_signs_and_channels():
    s = create_initial_game_state()
    x = encode_toy(s).reshape(8, 8, 6)
    assert x[6, 0, PIECE_CHANNEL["pawn"]] == 1.0     # own pawn = +1
    assert x[1, 0, PIECE_CHANNEL["pawn"]] == -1.0    # opponent pawn = -1
    assert x[7, 4, PIECE_CHANNEL["king"]] == 1.0
    assert x[0, 4, PIECE_CHANNEL["king"]] == -1.0
    assert not x[2:6].any()


def test_forward_contract_matches_chessnet():
    """ToyNet.heads must return (policy_probs, wdl_probs) like ChessNet —
    both softmax-normalized — so the Trainer works without branches."""
    torch.manual_seed(0)
    model = ToyNet()
    boards = np.stack([encode_toy(create_initial_game_state()) for _ in range(3)])
    x = torch.from_numpy(boards.reshape(-1, 8, 8, 6)).permute(0, 3, 1, 2).contiguous()
    policy, wdl = model(x)
    assert policy.shape == (3, 4096)
    assert wdl.shape == (3, 3)
    assert torch.allclose(policy.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(wdl.sum(dim=1), torch.ones(3), atol=1e-5)
    # Value-head params named value_* so the warmup freeze finds them.
    value_params = [n for n, _ in model.named_parameters() if n.startswith("value_")]
    assert len(value_params) == 6


def test_export_roundtrip_and_order():
    torch.manual_seed(0)
    model = ToyNet()
    blob = model.export_browser_json()
    assert blob["kind"] == "toy-v1"
    assert blob["names"][0] == "conv_in.w"
    assert blob["names"][-1] == "value_fc2.b"
    idx = blob["names"].index("value_fc2.w")
    assert blob["shapes"][idx] == [64, 3], "WDL head: value_fc2 maps 64 -> 3"
    pf = blob["names"].index("policy_fc.w")
    raw = np.frombuffer(base64.b64decode(blob["data"][pf]), dtype="<f2")
    expected = model.policy_fc.weight.detach().numpy().T.astype("<f2").ravel()
    assert np.array_equal(raw, expected)


def test_trainer_runs_toynet_end_to_end(tmp_path):
    """The real Trainer — buffer, self-play engine, mirror augment,
    smoothing, draw weighting, checkpointing — drives ToyNet through
    gradient steps via the 6-plane encoder and Python MCTS."""
    torch.manual_seed(0)
    from chess_ai.toy import encode_toy as enc
    model = ToyNet()
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
        # All-endgame starts: caps of 80-200 plies guarantee games finish
        # inside the step budget regardless of seed.
        endgame_start_prob=1.0,
        random_start_prob=0.0,
        resign_threshold=-2.0,           # disabled
        checkpoint_every_seconds=1e9,
        eval_every_gens=10_000_000,      # no eval during smoke
        mirror_augment_prob=0.5,
        policy_label_smoothing=0.03,
    )
    import random as _random
    trainer = Trainer(model=model, device=torch.device("cpu"), config=config,
                      rng=_random.Random(7), board_encoder=enc)
    trainer.run(num_steps=260, checkpoint_dir=tmp_path)

    assert trainer.stats.generation > 0, "gradient steps must have run"
    assert trainer.stats.games_completed > 0
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "latest.json").exists()

    import json
    with (tmp_path / "latest.json").open() as f:
        blob = json.load(f)
    assert blob["kind"] == "toy-v1"

    # Family-tagged checkpoint reloads as a ToyNet.
    state = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert state["model_arch"] == {"family": "toy"}
    rebuilt = _build_model_from_arch(state["model_arch"])
    assert isinstance(rebuilt, ToyNet)
    rebuilt.load_state_dict(state["model_state_dict"])


def test_mp_workers_rejected_with_custom_encoder():
    """The multiprocess path is hard-wired to 20-plane Rust encoding —
    a custom encoder there must fail loudly, not corrupt silently."""
    model = ToyNet()
    config = TrainConfig(num_workers=2)
    with pytest.raises(ValueError, match="board_encoder"):
        Trainer(model=model, device=torch.device("cpu"), config=config,
                board_encoder=encode_toy)
