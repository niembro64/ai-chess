"""Resign mechanism tests.

Resignation lets self-play end games early once the side-to-move's MCTS
best visited-child Q (`MCTSResult.best_q` — "even my best move loses")
drops below a threshold. The training labels are identical to a
checkmate (the resigning side gets value=-1, opponent +1), but resigns
land in their own resign_w / resign_b outcome buckets so mate-rate
diagnostics stay honest. Diagnostic counters live on the engine:
`resigns` (actual resigns), `resign_truth_checks` (held-back would-be
resigns that play on), and `resign_truth_finished` / `resign_truth_fps`
(held-back games scored against their real outcome — the resign
false-positive rate).

These tests don't run a real network — they directly poke `MCTSResult`
returns and verify the engine takes the right branch.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from chess_ai.mcts import MCTSResult
from chess_ai.model import ChessNet
from chess_ai.selfplay import ReplayBuffer, SelfPlayConfig, make_local_selfplay_engine


@pytest.fixture
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


def _build_engine(tiny_model: ChessNet, **cfg_kwargs):
    """Build a self-play engine with overridable resign knobs."""
    buffer = ReplayBuffer(capacity=500)
    engine = make_local_selfplay_engine(
        model=tiny_model,
        device=torch.device("cpu"),
        replay_buffer=buffer,
        config=SelfPlayConfig(
            num_concurrent_games=2,
            mcts_simulations=4,
            **cfg_kwargs,
        ),
        rng=random.Random(123),
    )
    return engine


def _patched_mcts_with_value(value: float):
    """Return a function that replaces run_batched_mcts with a stub that
    yields the given root_value AND best_q (the resign trigger) plus an
    arbitrary policy / move for every game in the batch."""
    def _stub(states, evaluator, sims, rng, temperatures, **kwargs):
        results = []
        for st in states:
            policy = np.zeros(4096, dtype=np.float32)
            # Pick the first legal move so apply_move doesn't blow up.
            from chess_ai.engine import get_legal_moves
            from chess_ai.encoding import move_to_index
            legal = get_legal_moves(st)
            mv = legal[0]
            policy[move_to_index(mv, st.currentTurn == "white")] = 1.0
            results.append(
                MCTSResult(policy=policy, move=mv, root_value=value, best_q=value)
            )
        return results
    return _stub


def test_resign_triggers_below_threshold(tiny_model: ChessNet):
    """best_q <= threshold past min_plies → engine ends the game."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-0.85,
        resign_disabled_prob=0.0,    # never truth-check
        resign_min_plies=0,          # allow resign immediately
    )
    with patch("chess_ai.selfplay.run_batched_mcts", _patched_mcts_with_value(-0.95)):
        finished = engine.step()
    # Both games hit the threshold; both should resign on this single step.
    assert len(finished) == 2
    assert engine.resigns == 2
    assert engine.resign_truth_checks == 0
    # Resigns get their own outcome buckets (resigning side loses).
    for r in finished:
        assert r.outcome in ("resign_w", "resign_b")
        assert r.outcome_label.endswith("resigns")
        assert r.resign_truth_fp is None


def test_resign_trigger_uses_best_q_not_root_value(tiny_model: ChessNet):
    """A pessimistic mean Q (root_value) must NOT trigger resignation
    when the best child Q says the position is fine — the mean is
    dragged down by forced exploration of the mover's own bad moves."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-0.85,
        resign_disabled_prob=0.0,
        resign_min_plies=0,
    )

    def _stub(states, evaluator, sims, rng, temperatures, **kwargs):
        from chess_ai.encoding import move_to_index
        from chess_ai.engine import get_legal_moves
        results = []
        for st in states:
            policy = np.zeros(4096, dtype=np.float32)
            legal = get_legal_moves(st)
            mv = legal[0]
            policy[move_to_index(mv, st.currentTurn == "white")] = 1.0
            results.append(
                MCTSResult(policy=policy, move=mv, root_value=-0.95, best_q=0.1)
            )
        return results

    with patch("chess_ai.selfplay.run_batched_mcts", _stub):
        finished = engine.step()
    assert engine.resigns == 0
    assert len(finished) == 0


def test_resign_skipped_above_threshold(tiny_model: ChessNet):
    """root_value above threshold → no resign, normal step."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-0.85,
        resign_disabled_prob=0.0,
        resign_min_plies=0,
    )
    with patch("chess_ai.selfplay.run_batched_mcts", _patched_mcts_with_value(0.0)):
        finished = engine.step()
    # No game ended on this single step from an even position.
    assert len(finished) == 0
    assert engine.resigns == 0


def test_resign_skipped_under_min_plies(tiny_model: ChessNet):
    """Threshold met but move_count < resign_min_plies → still no resign."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-0.85,
        resign_disabled_prob=0.0,
        resign_min_plies=10,         # block resign in opening
    )
    with patch("chess_ai.selfplay.run_batched_mcts", _patched_mcts_with_value(-0.95)):
        finished = engine.step()
    # move_count starts at 0; first step puts it at 1 — below min_plies.
    assert engine.resigns == 0
    assert len(finished) == 0


def test_resign_truth_check_holds_back(tiny_model: ChessNet):
    """resign_disabled_prob=1.0 → every threshold-met game truth-checks."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-0.85,
        resign_disabled_prob=1.0,    # always hold back (no actual resigns)
        resign_min_plies=0,
    )
    with patch("chess_ai.selfplay.run_batched_mcts", _patched_mcts_with_value(-0.95)):
        finished = engine.step()
    # Threshold met but truth-check fires — counted as held-back, not resigned.
    assert engine.resigns == 0
    assert engine.resign_truth_checks == 2
    # Game does NOT end on this step (the move is applied normally).
    assert len(finished) == 0


def test_resign_truth_check_measures_false_positives(tiny_model: ChessNet):
    """A held-out would-be resignation that goes on to NOT lose must be
    scored as a false positive on the engine counters and the
    GameResult. Build it synthetically: mark the slot as truth-checked
    for white, then finish via stalemate (a draw — white didn't lose)."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-0.85,
        resign_disabled_prob=1.0,
        resign_min_plies=0,
    )
    slot = engine.games[0]
    slot.resign_truth_color = "white"
    slot.state.status = "stalemate"
    result = engine._finish_game(slot)
    assert result.resign_truth_fp is True
    assert engine.resign_truth_finished == 1
    assert engine.resign_truth_fps == 1

    # And the true-positive case: white was flagged and white lost.
    slot2 = engine.games[1]
    slot2.resign_truth_color = "white"
    slot2.state.status = "checkmate"
    slot2.state.currentTurn = "white"  # white to move with no moves = white lost
    result2 = engine._finish_game(slot2)
    assert result2.resign_truth_fp is False
    assert engine.resign_truth_finished == 2
    assert engine.resign_truth_fps == 1


def test_resign_disabled_when_threshold_below_neg1(tiny_model: ChessNet):
    """threshold <= -1.0 disables resign entirely."""
    engine = _build_engine(
        tiny_model,
        resign_threshold=-1.5,
        resign_disabled_prob=0.0,
        resign_min_plies=0,
    )
    with patch("chess_ai.selfplay.run_batched_mcts", _patched_mcts_with_value(-0.99)):
        finished = engine.step()
    # Even with extreme value, resign branch never fires.
    assert engine.resigns == 0
    assert engine.resign_truth_checks == 0
    assert len(finished) == 0
