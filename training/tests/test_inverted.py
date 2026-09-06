"""Competitive inverted rules, exact proofs, search parity and gate evidence."""

import random
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

from chess_ai import mcts, tablebase
from chess_ai.encoding import POLICY_SIZE, encode_board
from chess_ai.engine import apply_move, get_legal_moves, position_key
from chess_ai.inverted import (
    ProofBudgetExceeded,
    forced_selfmate_moves,
    selfmate_positions,
    stable_position_id,
    state_from_fen,
)
from chess_ai.jester_eval import Match, move_uci, play_matches, score_interval
from chess_ai.selfplay import GameSlot, GameSlotExample, SelfPlayConfig, SelfPlayEngine
from chess_ai.train import TrainConfig, Trainer
from chess_ai.model import ChessNet

FORCED = "rk5K/8/2Q5/5b1n/8/8/8/8 w - - 0 1"


def uniform(boards):
    return np.full((len(boards), POLICY_SIZE), 1 / POLICY_SIZE, np.float32), np.zeros(len(boards), np.float32)


def tiny_trainer(**options):
    return Trainer(
        ChessNet(num_res_blocks=1, num_filters=16, value_head_size=8, se_reduction=4),
        torch.device("cpu"),
        TrainConfig(
            num_concurrent_games=2,
            aux_material_weight=0.0,
            replay_buffer_capacity=100,
            use_amp=False,
            **options,
        ),
    )


def test_resisting_opponent_can_be_forced_to_mate():
    state = state_from_fen(FORCED)
    move = next(m for m in get_legal_moves(state) if move_uci(m) == "c6c7")
    after = apply_move(state, move)
    replies = get_legal_moves(after)
    assert [move_uci(m) for m in replies] == ["b8c7"]
    terminal = apply_move(after, replies[0])
    assert terminal.status == "checkmate" and terminal.currentTurn == "white"
    assert "c6c7" in forced_selfmate_moves(FORCED, 2)
    with pytest.raises(ProofBudgetExceeded):
        forced_selfmate_moves(FORCED, 6, node_budget=1)


def test_catalog_proofs_and_holdout_are_valid():
    train, heldout = selfmate_positions("train"), selfmate_positions("eval")
    assert {p.plies for p in train} == {2, 4, 6}
    assert {p.plies for p in heldout} == {2, 4, 6}
    assert not {p.fen for p in train} & {p.fen for p in heldout}
    for p in train + heldout:
        assert set(forced_selfmate_moves(p.fen, p.plies)) == set(p.winning_moves), p.name
        if p.plies > 2:
            assert not forced_selfmate_moves(p.fen, p.plies - 2), p.name


@pytest.mark.parametrize("status,expected", [("active", "cap"), ("draw", "draw_50")])
def test_jester_never_uses_ordinary_tablebase_even_if_open(monkeypatch, status, expected):
    state = state_from_fen("k7/8/2KQ4/8/8/8/8/8 w - - 0 1")
    state.status = status
    captured = []
    engine = SelfPlayEngine(
        uniform,
        captured.append,
        SelfPlayConfig(num_concurrent_games=1, invert_agent_selection=True),
        random.Random(2),
    )
    monkeypatch.setattr(
        tablebase, "probe_outcome", lambda _: pytest.fail("ordinary tablebase used for inverted label")
    )
    slot = GameSlot(state=state, move_count=10, move_cap=10)
    slot.examples = [GameSlotExample(encode_board(state), uniform([0])[0][0], "white", 0.0)]
    result = engine._finish_game(slot)
    assert result.outcome == expected
    assert not result.tb_adjudicated
    assert captured[0].value == 0
    assert captured[0].outcome_known == (status == "draw")


@pytest.mark.parametrize("rust", [False, True])
def test_search_finds_forced_selfmate_with_both_colors_inverted(monkeypatch, rust):
    if rust and not mcts._HAVE_RUST_MCTS:
        pytest.skip("Rust extension required")
    monkeypatch.setattr(mcts, "USE_RUST_MCTS", rust)
    mcts.set_mcts_params(c_puct=1.5, fpu_reduction=0.4)
    state = state_from_fen(FORCED)
    result = mcts.run_batched_mcts(
        [state],
        uniform,
        400,
        random.Random(2),
        temperatures=[0.0],
        dirichlet_epsilon=0.0,
        invert_turns=["both"],
        position_counts=[{position_key(state): 1}],
    )[0]
    assert move_uci(result.move) in forced_selfmate_moves(FORCED, 2)
    assert abs(result.policy.sum() - 1) < 1e-5


@pytest.mark.parametrize("rust", [False, True])
def test_all_promotions_remain_searchable(monkeypatch, rust):
    if rust and not mcts._HAVE_RUST_MCTS:
        pytest.skip("Rust extension required")
    state = state_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1")
    promotions = {}

    # Pin the prior to a promotion. All four legal choices must reach
    # evaluation, rather than being discarded by the 4096-index collision.
    def evaluator(boards):
        policies, values = uniform(boards)
        for b in boards:
            board = np.asarray(b).reshape(8, 8, 20)
            # At black-to-move leaves white a8 appears at canonical h1.
            for kind, channel in [("queen", 7), ("rook", 8), ("bishop", 9), ("knight", 10)]:
                if board[7, 7, channel]:
                    promotions[kind] = True
        return policies, values

    monkeypatch.setattr(mcts, "USE_RUST_MCTS", rust)
    result = mcts.run_batched_mcts(
        [state], evaluator, 100, random.Random(0), temperatures=[0.0], dirichlet_epsilon=0.0
    )[0]
    # Minor-piece promotions can be terminal dead draws and skip evaluation;
    # the tree structure is checked separately below for the Python backend.
    assert promotions.get("queen") and promotions.get("rook")
    assert np.isclose(result.policy.sum(), 1)
    search = mcts.MCTSSearch(state)
    search.init_root(uniform([0])[0][0], 0.0, 0.0)
    assert len([c for c in search.root.children.values() if c.move.promotion]) == 4


def test_rust_position_history_is_not_ignored():
    if not mcts._HAVE_RUST_MCTS:
        pytest.skip("Rust extension required")
    state = state_from_fen(FORCED)
    child = apply_move(state, next(m for m in get_legal_moves(state) if move_uci(m) == "c6c7"))
    search = mcts._rust_mcts.MctsSearch(
        mcts._state_to_dict(child), 1.5, 0.4, "both", [(position_key(child), 1)]
    )
    assert bytes(search.root_position_key()) == position_key(child)
    # A mate still takes precedence over a synthetic repetition history.
    result = mcts.run_batched_mcts(
        [child],
        uniform,
        20,
        random.Random(0),
        temperatures=[0],
        invert_turns=["both"],
        position_counts=[{position_key(child): 1}],
    )[0]
    assert apply_move(child, result.move).status == "checkmate"


@pytest.mark.parametrize("rust", [False, True])
def test_search_scores_game_history_threefold_as_draw(monkeypatch, rust):
    if rust and not mcts._HAVE_RUST_MCTS:
        pytest.skip("Rust extension required")
    monkeypatch.setattr(mcts, "USE_RUST_MCTS", rust)
    state = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    history = {position_key(state): 1}
    history.update({position_key(apply_move(state, move)): 2 for move in get_legal_moves(state)})
    calls = []

    def evaluator(boards):
        calls.append(len(boards))
        return uniform(boards)

    result = mcts.run_batched_mcts(
        [state], evaluator, 80, random.Random(0), temperatures=[0],
        dirichlet_epsilon=0, invert_turns=["both"], position_counts=[history],
    )[0]
    assert calls == [1]  # Every child is a draw; only the root needs the net.
    assert result.root_value == 0


def test_competitive_match_counts_own_mate_and_keeps_caps_separate():
    state = state_from_fen(FORCED)
    after = apply_move(state, next(m for m in get_legal_moves(state) if move_uci(m) == "c6c7"))
    results = play_matches([Match(after, "white", uniform, "forced", "tactic")], uniform, 8, 8, 2)
    assert results[0][1] == "win"
    capped = play_matches([Match(state, "white", uniform, "cap", "tactic")], uniform, 8, 0, 2)
    assert capped[0][1] == "cap"
    score, lower, _ = score_interval(capped * 10)
    assert score == 0.5 and lower == 0


def test_gate_cannot_promote_draws_or_unresolved_caps():
    for outcome in ["draw", "cap"]:
        results = [(SimpleNamespace(pair=str(i // 2)), outcome) for i in range(80)]
        score, lower, upper = score_interval(results)
        assert score == 0.5 and lower <= 0.5 <= upper
    winning = [(SimpleNamespace(pair=str(i // 2)), "win") for i in range(80)]
    assert score_interval(winning)[1] > 0.5


def test_fumbler_cache_uses_board_and_search_settings():
    trainer = tiny_trainer()
    calls = []
    trainer._play_fumbler_game = lambda *a, **kw: (calls.append(a), 10)[1]
    a = SimpleNamespace(name="same display name", state=state_from_fen(FORCED))
    b = SimpleNamespace(name="same display name", state=state_from_fen("k7/8/2KQ4/8/8/8/8/8 w - - 0 1"))
    trainer._play_fumbler_pair(uniform, uniform, "white", 4, 30, a)
    trainer._play_fumbler_pair(uniform, uniform, "white", 4, 30, a)
    assert len(calls) == 3
    trainer._play_fumbler_pair(uniform, uniform, "white", 4, 30, b)
    assert len(calls) == 5
    trainer._play_fumbler_pair(uniform, uniform, "white", 8, 30, b)
    assert len(calls) == 7
    assert stable_position_id(a.state) != stable_position_id(b.state)


def test_competitive_config_removes_sage_and_cooperation():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("training_config", Path(__file__).parents[1] / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cfg = config.build_jester_config()
    assert cfg.syzygy_path is None and cfg.endgame_start_prob == 0
    assert cfg.jester_spar_random_prob == cfg.jester_spar_accept_mate_prob == cfg.jester_spar_temperature == 0
    assert cfg.jester_gate == "head_to_head"
    assert cfg.jester_opponent_checkpoint == ""
    assert cfg.mcts_simulations >= 256 and cfg.value_draw_weight == 1


def test_competitive_gate_runs_and_logs_without_promoting_all_caps(tmp_path, monkeypatch):
    from chess_ai import jester_eval

    trainer = tiny_trainer(
        jester_mode=True,
        jester_curriculum_prob=0,
        eval_mcts_sims=4,
        eval_move_cap=0,
        eval_rotating_openings=0,
    )
    trainer._save_champion(tmp_path, 0)
    # Real batched evaluator and score path, with an intentionally unresolved
    # match horizon. A small held-out slice keeps this integration check fast.
    monkeypatch.setattr(jester_eval, "selfmate_positions", lambda split: selfmate_positions(split)[:2])
    result = trainer._run_eval_match(tmp_path)
    assert result["gate"] == "competitive-inverted"
    assert result["caps"] == result["games"] and result["draws"] == 0
    assert not result["new_champion"]
    assert (tmp_path / "eval.csv").exists() and (tmp_path / "eval_games.jsonl").exists()


def test_competitive_multiprocess_produces_fresh_examples(tmp_path):
    import time
    from chess_ai.train import _model_arch_dict

    model = ChessNet(num_res_blocks=1, num_filters=16, value_head_size=8, se_reduction=4)
    path = tmp_path / "opponent.pt"
    torch.save(dict(model_state_dict=model.state_dict(), model_arch=_model_arch_dict(model)), path)
    cfg = TrainConfig(
        jester_mode=True,
        num_workers=2,
        games_per_worker=4,
        mcts_simulations=8,
        jester_curriculum_prob=1.0,
        jester_opponent_checkpoints=(str(path),),
        jester_selfplay_prob=0.5,
        jester_move_cap=8,
        min_buffer_for_training=4,
        batch_size=4,
        replay_buffer_capacity=100,
        aux_material_weight=0.0,
        use_amp=False,
        mp_batch_wait_ms=1,
        checkpoint_every_seconds=1e9,
    )
    trainer = Trainer(model, torch.device("cpu"), cfg)
    mp = trainer._mp_self_play
    mp.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(trainer.buffer) < 4:
            mp.check_health()
            mp.drain_examples(trainer.buffer)
            time.sleep(0.05)
        assert len(trainer.buffer) >= 4
        before = next(model.parameters()).detach().clone()
        losses = trainer.train_step()
        assert all(np.isfinite(v) for v in losses.values())
        assert not torch.equal(before, next(model.parameters()).detach())
        assert mp.snapshot_inf_stats()["dispatches"] > 0
    finally:
        mp.stop()
