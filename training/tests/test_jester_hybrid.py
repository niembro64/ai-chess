import random

import numpy as np
import pytest

from chess_ai import mcts
from chess_ai.encoding import POLICY_SIZE, encode_board
from chess_ai.engine import apply_move, get_legal_moves, position_key
from chess_ai.inverted import augmented_selfmates, forced_selfmate_moves, state_from_fen
from chess_ai.jester_eval import Match, move_uci, play_matches
from chess_ai.replay import JesterReplayBuffer
from chess_ai.selfplay import GameSlot, GameSlotExample, SelfPlayConfig, SelfPlayEngine, TrainingExample

FEN = "rk5K/8/2Q5/5b1n/8/8/8/8 w - - 0 1"


def uniform(boards):
    return np.full((len(boards), POLICY_SIZE), 1 / POLICY_SIZE, np.float32), np.zeros(len(boards), np.float32)


@pytest.mark.parametrize("rust", [False, True])
def test_root_owner_matches_independent_single_network_search(monkeypatch, rust):
    if rust and not mcts._HAVE_RUST_MCTS:
        pytest.skip("Rust required")
    monkeypatch.setattr(mcts, "USE_RUST_MCTS", rust)
    state = state_from_fen(FEN)

    def second(boards):
        policies, values = uniform(boards)
        policies[:, ::3] *= 10
        policies /= policies.sum(axis=1, keepdims=True)
        return policies, values + .7

    def forbidden(boards):
        pytest.fail("Default net must never supply another actor's leaves")

    options = dict(temperatures=[0], dirichlet_epsilon=0, invert_turns=["both"])
    for ev in (uniform, second):
        expected = mcts.run_batched_mcts([state], ev, 80, random.Random(9), **options)[0]
        actual = mcts.run_batched_mcts([state], forbidden, 80, random.Random(9), root_evaluators=[ev], **options)[0]
        assert move_uci(expected.move) == move_uci(actual.move)
        np.testing.assert_array_equal(expected.policy, actual.policy)
        assert expected.root_value == actual.root_value


def test_reserved_replay_survives_long_game_flood_and_caps_have_no_value_label():
    buffer = JesterReplayBuffer(100)
    board = encode_board(state_from_fen(FEN))
    policy = uniform([0])[0][0]
    for source in ("curriculum", "bridge", "competitive", "bootstrap"):
        buffer.add(TrainingExample(board, policy, -1, source=source))
    for _ in range(1000):
        buffer.add(TrainingExample(board, policy, 0, outcome_known=False))
    assert buffer.source_sizes == dict(curriculum=1, bridge=1, competitive=1, bootstrap=1, cap=5)
    _, _, values, known, _ = buffer.sample(100, random.Random(1))
    assert buffer.last_sample_counts == dict(curriculum=30, bridge=30, competitive=25, bootstrap=10, cap=5)
    assert sum(known == 0) == 5
    assert np.all(values[known == 1] == -1)


def test_replay_waits_for_observed_outcomes():
    buffer = JesterReplayBuffer(100)
    buffer.add(TrainingExample(encode_board(state_from_fen(FEN)), uniform([0])[0][0], 0, False))
    assert not buffer.ready
    with pytest.raises(ValueError, match="observed"):
        buffer.sample(10)


def test_helper_does_not_become_imitation_target_and_bridge_is_resistant():
    state = state_from_fen(FEN)
    after = apply_move(state, next(m for m in get_legal_moves(state) if move_uci(m) == "c6c7"))
    examples = []
    engine = SelfPlayEngine(uniform, examples.append, SelfPlayConfig(
        num_concurrent_games=1, invert_agent_selection=True, mcts_simulations=8,
        curriculum_start_prob=0, helper_start_prob=0, bridge_start_prob=1,
        resign_threshold=-2), random.Random(1))
    slot = GameSlot(state=after, agent_color="white", spar_color="black", matchup="bootstrap")
    slot.trajectory.append(state.copy())
    slot.examples.append(GameSlotExample(encode_board(state), uniform([0])[0][0], "white", 0))
    engine.games = [slot]
    result = engine.step()[0]
    assert result.variant_outcome == "own_mate"
    assert len(examples) == 1 and examples[0].source == "bootstrap"
    assert examples[0].value == -1  # Ordinary sign, selection supplies inversion.
    assert position_key(state) in engine.bridge_pool
    bridge = engine._new_slot()
    assert bridge.origin == "bridge" and bridge.spar_color is None and bridge.matchup == "mirror"
    assert bridge.position_history == {position_key(bridge.state): 1}


def test_game_contribution_is_bounded_without_relabeling_caps():
    examples = []
    engine = SelfPlayEngine(uniform, examples.append, SelfPlayConfig(
        num_concurrent_games=1, invert_agent_selection=True, max_examples_per_game=8), random.Random(2))
    state = state_from_fen(FEN)
    slot = GameSlot(state=state, move_count=200, move_cap=200)
    slot.examples = [GameSlotExample(encode_board(state), uniform([0])[0][0], "white", 0) for _ in range(200)]
    engine._finish_game(slot)
    assert len(examples) == 8 and all(not ex.outcome_known for ex in examples)


def test_geometric_curriculum_proofs_and_split():
    train, heldout = augmented_selfmates("train"), augmented_selfmates("eval")
    assert not {p.fen for p in train} & {p.fen for p in heldout}
    assert len(train) > 54 and len(heldout) > 18
    for split in (train, heldout):
        for p in random.Random(16).sample(list(split), 12):
            assert set(forced_selfmate_moves(p.fen, p.plies)) == set(p.winning_moves)


def test_evaluation_logs_actual_moves_and_uses_resistant_reply():
    state = state_from_fen(FEN)
    after = apply_move(state, next(m for m in get_legal_moves(state) if move_uci(m) == "c6c7"))
    game, outcome = play_matches([Match(after, "white", uniform, "forced", "tactic")], uniform, 8, 8, 2)[0]
    assert outcome == "win" and game.moves == ["b8c7"] and game.start
