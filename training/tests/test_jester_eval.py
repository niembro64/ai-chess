"""Jester eval gating + the machinery that makes misère games finish.

The gate plays challenger jester vs CHAMPION jester — the matchup the
shipped bot actually meets, where the opponent refuses to deliver the
mate you want (a *selfmate*). Playing the frozen winner instead is a
*helpmate*: the opponent is trying to mate you, so "stop defending"
solves it, the score saturates, and the gate stops discriminating.

Two greedy loss-seekers never finish a game, so both self-play mirror
games and eval games need a sustained move-selection temperature. These
tests pin that, the misère scoring, and the kwarg whose absence killed
the first run at its very first eval match.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves
from chess_ai.model import ChessNet
from chess_ai.selfplay import SelfPlayConfig, SelfPlayEngine
from chess_ai.train import TrainConfig, Trainer

FOOLS_MATE = [((6, 5), (5, 5)), ((1, 4), (3, 4)), ((6, 6), (4, 6)), ((0, 3), (4, 7))]


def _uniform_evaluator(boards: np.ndarray):
    batch = boards.shape[0]
    return (
        np.full((batch, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32),
        np.zeros(batch, dtype=np.float32),
    )


def _tiny_trainer() -> Trainer:
    model = ChessNet(
        num_res_blocks=1, num_filters=16, kernel_size=3,
        value_head_size=8, se_reduction=4,
    )
    return Trainer(
        model=model,
        device=torch.device("cpu"),
        config=TrainConfig(num_workers=0, use_amp=False),
        rng=random.Random(3),
    )


def _script_fools_mate(monkeypatch) -> list[float]:
    """Force _play_eval_game down the fool's-mate line and record the
    temperature each ply was asked to play at."""
    seen_temperatures: list[float] = []
    ply = {"i": 0}

    def fake_search(states, evaluator, sims, rng, temperatures=None, **kwargs):
        seen_temperatures.append((temperatures or [0.0])[0])
        from_rf, to_rf = FOOLS_MATE[ply["i"]]
        ply["i"] += 1
        move = next(
            m for m in get_legal_moves(states[0])
            if (m.from_pos.rank, m.from_pos.file) == from_rf
            and (m.to_pos.rank, m.to_pos.file) == to_rf
        )
        return [SimpleNamespace(move=move, policy=None, root_value=0.0)]

    monkeypatch.setattr("chess_ai.mcts.run_batched_mcts", fake_search)
    return seen_temperatures


def _fresh_start():
    state = create_initial_game_state()
    state.status = "active"
    return state


# --- the crash -----------------------------------------------------------


def test_play_eval_game_accepts_the_jester_kwarg(monkeypatch):
    """Regression: _run_eval_match passes jester=..., and the signature
    used to lack it. Every jester eval match died on its first game with
    TypeError, which killed the first run at gen 7,503 — the first time
    eval_every_gens came round."""
    _script_fools_mate(monkeypatch)
    trainer = _tiny_trainer()
    outcome = trainer._play_eval_game(
        _uniform_evaluator, _uniform_evaluator, True, 4, 50,
        starting_state=_fresh_start(), jester=True, temperature=1.0,
    )
    assert outcome in ("challenger", "champion", "draw")


# --- misère scoring ------------------------------------------------------


def test_challenger_mated_first_wins_the_misere_game(monkeypatch):
    """White is mated by 4.Qh4#. As the challenger, white got its own
    king checkmated — that is the objective, so it scores a win."""
    _script_fools_mate(monkeypatch)
    trainer = _tiny_trainer()
    assert trainer._play_eval_game(
        _uniform_evaluator, _uniform_evaluator, True, 4, 50,
        starting_state=_fresh_start(), jester=True, temperature=1.0,
    ) == "challenger"


def test_champion_mated_first_wins_the_misere_game(monkeypatch):
    """Same game, challenger playing black: black delivered the mate
    instead of receiving one, so the CHAMPION reached the goal first."""
    _script_fools_mate(monkeypatch)
    trainer = _tiny_trainer()
    assert trainer._play_eval_game(
        _uniform_evaluator, _uniform_evaluator, False, 4, 50,
        starting_state=_fresh_start(), jester=True, temperature=1.0,
    ) == "champion"


def test_ordinary_scoring_is_untouched(monkeypatch):
    """The same mated challenger LOSES a normal (Sage) eval game — the
    inversion must be confined to jester=True."""
    _script_fools_mate(monkeypatch)
    trainer = _tiny_trainer()
    assert trainer._play_eval_game(
        _uniform_evaluator, _uniform_evaluator, True, 4, 50,
        starting_state=_fresh_start(), jester=False,
    ) == "champion"


def test_jester_eval_plays_at_the_requested_temperature(monkeypatch):
    """Greedy misère play never terminates, so eval games must be
    sampled, not argmaxed. τ has to reach the search on every ply."""
    seen = _script_fools_mate(monkeypatch)
    trainer = _tiny_trainer()
    trainer._play_eval_game(
        _uniform_evaluator, _uniform_evaluator, True, 4, 50,
        starting_state=_fresh_start(), jester=True, temperature=1.0,
    )
    assert seen == [1.0] * 4


# --- sparring side of mirror self-play -----------------------------------


def test_mirror_slots_get_a_sparring_side():
    """Every mirror game names one color as the sparring partner; games
    against the frozen winner keep an agent color and no spar side."""
    config = SelfPlayConfig(
        num_concurrent_games=16,
        mcts_simulations=2,
        endgame_start_prob=0.0,
        random_start_prob=0.0,
        invert_agent_selection=True,
        frozen_evaluator=_uniform_evaluator,
        agent_selfplay_prob=0.5,
    )
    engine = SelfPlayEngine(
        _uniform_evaluator, lambda ex: None, config, random.Random(11)
    )
    mirrors = [g for g in engine.games if g.agent_color is None]
    versus = [g for g in engine.games if g.agent_color is not None]
    assert mirrors and versus, "expected both game types at prob=0.5"
    assert all(g.spar_color in ("white", "black") for g in mirrors)
    assert all(g.spar_color is None for g in versus)
    assert {g.spar_color for g in mirrors} == {"white", "black"}, (
        "the sparring seat must vary — the agent has to learn to force "
        "the loss as either color"
    )


def test_sparring_side_never_anneals_to_greedy():
    """The agent side anneals to τ=0 like ordinary self-play; the spar
    side holds its temperature for the whole game. Without that both
    sides go greedy and shuffle to a threefold draw."""
    config = SelfPlayConfig(
        num_concurrent_games=4,
        mcts_simulations=2,
        endgame_start_prob=0.0,
        random_start_prob=0.0,
        temperature_threshold_plies=0,   # agent side is greedy immediately
        invert_agent_selection=True,
        frozen_evaluator=None,           # all mirror
        agent_selfplay_prob=1.0,
        spar_temperature=1.3,
    )
    engine = SelfPlayEngine(
        _uniform_evaluator, lambda ex: None, config, random.Random(2)
    )
    seen: list[list[float]] = []
    import chess_ai.selfplay as selfplay_module

    real = selfplay_module.run_batched_mcts

    def spy(states, evaluator, sims, rng, temperatures=None, **kwargs):
        seen.append(list(temperatures or []))
        return real(states, evaluator, sims, rng, temperatures, **kwargs)

    selfplay_module.run_batched_mcts = spy
    try:
        for _ in range(3):
            engine.step()
    finally:
        selfplay_module.run_batched_mcts = real

    for step_temps, in zip(seen):
        assert set(step_temps) <= {0.0, 1.3}
    # Across three plies both seats must have moved, so both the greedy
    # agent temperature and the sustained spar temperature must appear.
    flat = [t for step in seen for t in step]
    assert 1.3 in flat, "spar side never played at its sustained temperature"
    assert 0.0 in flat, "agent side never annealed to greedy"


# --- general temperature in the sampler ----------------------------------


def _root_with_visits(visits: list[int]):
    from chess_ai.mcts import MCTSNode

    root = MCTSNode(state=None)
    for index, count in enumerate(visits):
        child = MCTSNode(
            state=None, parent=root, move=SimpleNamespace(index=index),
        )
        child.visit_count = count
        root.children[index] = child
    return root


def test_temperature_zero_is_argmax():
    from chess_ai.mcts import _sample_move

    root = _root_with_visits([1, 97, 2])
    for seed in range(8):
        assert _sample_move(root, random.Random(seed), 0.0).index == 1


@pytest.mark.parametrize("temperature", [1.0, 2.5])
def test_higher_temperature_picks_the_top_move_less_often(temperature):
    """τ=1 samples proportional to visits; τ>1 is flatter still. Both
    must actually explore — that is what puts blunders back into mirror
    play and lets the game end."""
    from chess_ai.mcts import _sample_move

    root = _root_with_visits([80, 10, 10])
    rng = random.Random(0)
    picks = [_sample_move(root, rng, temperature).index for _ in range(400)]
    top_share = picks.count(0) / len(picks)
    assert 0.05 < top_share < 0.95, (
        f"τ={temperature} gave top-move share {top_share:.2f} — not sampling"
    )
    if temperature > 1.0:
        assert top_share < 0.8, "τ>1 must be flatter than proportional"
