"""Smoke tests for the Toy mini-trainer (scripts/toy_train.py).

Pins the conventions the browser port depends on: the 6-plane encoding
(perspective rotation + sign), the fixed weight-export order/shapes,
and that a self-play game actually runs end to end on a random net.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import toy_train  # noqa: E402
from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves  # noqa: E402


def test_encoding_rotation_symmetry():
    """The 180° mover-perspective rotation (rank AND file — same
    convention as Sage) means black's view of the start position equals
    white's view FILE-MIRRORED: the start position is rank-mirror
    color-symmetric, and rotation = rank-mirror + file-mirror. Pins the
    rotation direction and the sign flip in one shot."""
    s_white = create_initial_game_state()
    s_black = create_initial_game_state()
    s_black.currentTurn = "black"
    ew = toy_train.encode_toy(s_white).reshape(8, 8, 6)
    eb = toy_train.encode_toy(s_black).reshape(8, 8, 6)
    assert np.array_equal(eb, ew[:, ::-1, :])
    # And NOT identical — rotation swaps the king/queen files.
    assert not np.array_equal(eb, ew)


def test_encoding_signs_and_channels():
    s = create_initial_game_state()
    x = toy_train.encode_toy(s).reshape(8, 8, 6)
    # White to move: own (white) pawns on rank index 6 → +1 in channel 0.
    assert x[6, 0, toy_train.PIECE_CHANNEL["pawn"]] == 1.0
    # Opponent (black) pawns on rank index 1 → -1.
    assert x[1, 0, toy_train.PIECE_CHANNEL["pawn"]] == -1.0
    # Kings at e-file: white king at (7,4) → +1 in channel 5.
    assert x[7, 4, toy_train.PIECE_CHANNEL["king"]] == 1.0
    assert x[0, 4, toy_train.PIECE_CHANNEL["king"]] == -1.0
    # Empty middle is all zeros.
    assert not x[2:6].any()


def test_forward_shapes_and_value_range():
    torch.manual_seed(0)
    model = toy_train.ToyNet()
    boards = np.stack([
        toy_train.encode_toy(create_initial_game_state()) for _ in range(3)
    ])
    logits, value = model(toy_train.flat_to_nchw(boards))
    assert logits.shape == (3, 4096)
    assert value.shape == (3,)
    assert (value.abs() <= 1.0).all()


def test_export_roundtrip_and_order():
    torch.manual_seed(0)
    model = toy_train.ToyNet()
    blob = toy_train.export_toy_json(model)
    assert blob["kind"] == "toy-v1"
    assert blob["names"][0] == "conv_in.w"
    assert blob["names"][-1] == "value_fc2.b"
    assert len(blob["names"]) == len(blob["shapes"]) == len(blob["data"])
    # Decode a tensor and compare against the fp16-cast original.
    idx = blob["names"].index("policy_fc.w")
    raw = np.frombuffer(base64.b64decode(blob["data"][idx]), dtype="<f2")
    expected = model.policy_fc.weight.detach().numpy().T.astype("<f2").ravel()
    assert np.array_equal(raw, expected)
    assert blob["shapes"][idx] == [8 * 8 * 4, 4096]


def test_self_play_game_runs():
    torch.manual_seed(0)
    import random
    model = toy_train.ToyNet()
    model.eval()
    net_eval = toy_train.make_net_eval(model, torch.device("cpu"))
    rng = random.Random(1)
    examples, label = toy_train.play_game(net_eval, sims=8, rng=rng)
    assert len(examples) > 0
    assert label in ("mate", "stalemate", "draw", "repetition", "insufficient", "cap")
    planes, policy, value = examples[0]
    assert planes.shape == (8 * 8 * 6,)
    assert policy.shape == (4096,)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert value in (-1.0, 0.0, 1.0)


def test_mcts_finds_mate_in_one():
    """Fool's mate: with a neutral random net at 60 sims, terminal values
    alone must steer the visit distribution onto Qh4#. Pins the Q-sign
    convention in the toy MCTS (the classic bug this repo has hit twice)."""
    import random
    torch.manual_seed(0)
    model = toy_train.ToyNet()
    model.eval()
    net_eval = toy_train.make_net_eval(model, torch.device("cpu"))

    s = create_initial_game_state()
    s.status = "active"
    from chess_ai.encoding import move_to_index

    def play(uci):
        nonlocal s
        from chess_ai.engine import Position
        for m in get_legal_moves(s):
            key = (
                chr(97 + m.from_pos.file) + str(8 - m.from_pos.rank)
                + chr(97 + m.to_pos.file) + str(8 - m.to_pos.rank)
            )
            if key == uci:
                s = apply_move(s, m)
                return
        raise AssertionError(f"{uci} not legal")

    play("f2f3"); play("e7e5"); play("g2g4")

    visit_policy, root = toy_train.mcts(s, net_eval, sims=60, rng=random.Random(0), root_noise=False)
    best = max(root.children.values(), key=lambda c: c.visits)
    key = (
        chr(97 + best.move.from_pos.file) + str(8 - best.move.from_pos.rank)
        + chr(97 + best.move.to_pos.file) + str(8 - best.move.to_pos.rank)
    )
    assert key == "d8h4", f"expected mate d8h4, MCTS chose {key}"
