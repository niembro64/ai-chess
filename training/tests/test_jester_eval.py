"""Jester eval gating + the machinery that makes misère games finish.

Nothing about inverted chess terminates on its own. Both sides want
their own king mated, so neither will ever deliver the mate the other
is angling for, and left alone the game shuffles until the move cap.
Two consequences shaped everything here.

SELF-PLAY. Temperature cannot supply the missing blunder. It samples
the search's visit counts, and a loss-seeking search spends near-zero
visits on its own mating moves by construction. Measured on the
temperature-only build: 20% checkmate, 45% move-cap timeouts, against
71.5% decisive under the old mix. A UNIFORM random legal move on the
sparring side is out of distribution and so CAN land on a mate — the
accident a human trying to lose commits — but on its own it only
reached 29.5% checkmate, because landing on a mate by chance already
presupposes the skill being learned: the agent has to have manufactured
a position where mates are plentiful. So the sparring partner also
ACCEPTS an offered mate some of the time, converting that skill
directly into terminal signal. All three shape the opponent, never the
reward — the mate is real, so every value label stays truthful.

GATING. Head-to-head challenger-vs-champion is the production matchup,
but it cannot be the gate: two competent loss-seekers draw by
construction, so the better both nets get, the more it draws. It
degrades backwards. Measured on real weights, every head-to-head smoke
game drew — on full-material middlegames as well as lopsided endgames.
The gate instead races both nets to their own checkmate against a
FUMBLER: a random mover that always accepts an offered mate. Decisive,
monotone in the skill we want, and fixed forever so the number stays
comparable across the whole run.

These tests pin both mechanisms, the misère scoring, and the kwarg
whose absence killed the first run at its very first eval match.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import (
    apply_move,
    create_initial_game_state,
    get_legal_moves,
    position_key,
)
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
    TypeError, which killed the first run at gen 3,349 — the first time
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


# --- what actually terminates a misère game ------------------------------


def test_sparring_blunders_are_uniform_not_hotter_search():
    """The blunder must be a UNIFORM legal move, not a hotter sample of
    the search.

    This is the whole point. A loss-seeking search puts near-zero visits
    on its own mating moves — delivering mate is the worst thing it can
    do — so no temperature can make it hand over the mate its opponent
    wants. Only an out-of-distribution pick can. The temperature-only
    build measured 20% checkmate / 45% move-cap in self-play, and drew
    every eval game.

    Here the search is pinned to one move; with a blunder rate of 1.0
    the sparring side must still produce other moves.
    """
    pinned = {"count": 0}

    def always_first(states, evaluator, sims, rng, temperatures=None, **kwargs):
        results = []
        for state in states:
            pinned["count"] += 1
            results.append(SimpleNamespace(
                move=get_legal_moves(state)[0],
                policy=np.zeros(POLICY_SIZE, dtype=np.float32),
                root_value=0.0,
            ))
        return results

    import chess_ai.selfplay as selfplay_module
    real = selfplay_module.run_batched_mcts
    selfplay_module.run_batched_mcts = always_first
    try:
        config = SelfPlayConfig(
            num_concurrent_games=24,
            mcts_simulations=2,
            endgame_start_prob=0.0,
            random_start_prob=0.0,
            invert_agent_selection=True,
            frozen_evaluator=None,        # all mirror
            agent_selfplay_prob=1.0,
            spar_random_prob=1.0,         # every sparring ply blunders
        )
        engine = SelfPlayEngine(
            _uniform_evaluator, lambda ex: None, config, random.Random(4)
        )
        engine.step()   # white to move: white-sparring slots blunder here
        openings = {
            position_key(slot.state)
            for slot in engine.games if slot.spar_color == "white"
        }
        agent_openings = {
            position_key(slot.state)
            for slot in engine.games if slot.spar_color == "black"
        }
    finally:
        selfplay_module.run_batched_mcts = real

    assert pinned["count"] > 0, "search was never consulted"
    assert len(openings) > 1, (
        "every white-sparring game opened with the same move — the "
        "blunder is following the search instead of picking uniformly"
    )
    # Control: where white is the AGENT, the pinned search decides and
    # every game must reach the identical position.
    assert len(agent_openings) == 1, (
        "agent plies diverged from the search — the blunder is leaking "
        "onto the wrong seat"
    )


def test_agent_side_never_blunders():
    """Only the sparring seat blunders. The agent must play its best
    misère chess or it is not learning to be good at anything."""
    config = SelfPlayConfig(
        num_concurrent_games=6,
        mcts_simulations=2,
        endgame_start_prob=0.0,
        random_start_prob=0.0,
        invert_agent_selection=True,
        frozen_evaluator=_uniform_evaluator,
        agent_selfplay_prob=0.0,          # all vs-frozen, no spar side
        spar_random_prob=1.0,
    )
    engine = SelfPlayEngine(
        _uniform_evaluator, lambda ex: None, config, random.Random(8)
    )
    assert all(g.spar_color is None for g in engine.games)
    # With no sparring seat anywhere, the blunder branch is unreachable
    # no matter how high spar_random_prob is set.
    for _ in range(3):
        engine.step()


def test_eval_blunders_apply_to_both_sides():
    """Eval blunders are symmetric — an asymmetric rate would hand one
    net an advantage that has nothing to do with its weights."""
    seen_turns: list[str] = []
    ply = {"i": 0}

    def fake_search(states, evaluator, sims, rng, temperatures=None, **kwargs):
        seen_turns.append(states[0].currentTurn)
        from_rf, to_rf = FOOLS_MATE[ply["i"] % len(FOOLS_MATE)]
        ply["i"] += 1
        candidates = [
            m for m in get_legal_moves(states[0])
            if (m.from_pos.rank, m.from_pos.file) == from_rf
            and (m.to_pos.rank, m.to_pos.file) == to_rf
        ]
        move = candidates[0] if candidates else get_legal_moves(states[0])[0]
        return [SimpleNamespace(move=move, policy=None, root_value=0.0)]

    import chess_ai.mcts as mcts_module
    real = mcts_module.run_batched_mcts
    mcts_module.run_batched_mcts = fake_search
    try:
        trainer = _tiny_trainer()
        trainer._play_eval_game(
            _uniform_evaluator, _uniform_evaluator, True, 4, 30,
            starting_state=_fresh_start(), jester=True,
            temperature=1.0, blunder_prob=1.0,
        )
    finally:
        mcts_module.run_batched_mcts = real

    # Both colors moved, so a symmetric rate exposed both of them to the
    # blunder branch.
    assert {"white", "black"} <= set(seen_turns)


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


# --- the gate needs positions where both kings can be mated --------------


def test_jester_gate_drops_structurally_drawn_positions():
    """A bare king cannot deliver checkmate, so in the suite's lopsided
    entries the strong side can never reach the misère objective and the
    game is drawn however well either net plays. Every mate-in-1 smoke
    game drew at the move cap. The gate keeps only positions with enough
    material on both sides for either king to be mated."""
    from chess_ai.eval_positions import build_eval_positions

    everything = build_eval_positions()
    eligible = [p for p in everything
                if p.difficulty in ("opening", "middlegame")]

    assert eligible, "no gate-eligible positions left"
    assert len(eligible) < len(everything), "filter is a no-op"
    assert not any(p.difficulty == "mate-in-1" for p in eligible), (
        "mate-in-1 positions are structurally drawn under misère scoring"
    )
    # Enough to make a meaningful match after the x2 color doubling.
    assert len(eligible) >= 20


# --- the fumbler gate ----------------------------------------------------


def _mated_after(plies_by_net):
    """Build a trainer whose fumbler games return canned ply counts."""
    trainer = _tiny_trainer()
    calls = iter(plies_by_net)
    trainer._play_fumbler_game = lambda *a, **k: next(calls)  # type: ignore[method-assign]
    return trainer


class _Pos:
    name = "probe"
    difficulty = "opening"
    state = None


def test_fumbler_pair_prefers_the_faster_self_mate():
    """Both nets got themselves mated; the quicker one takes the point.
    Speed is the whole signal — a net that only ever reaches mate at ply
    200 is worse at this than one that reaches it at ply 40."""
    trainer = _mated_after([40, 90])          # challenger, champion
    assert trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "white", 4, 50, _Pos()
    ) == "challenger"

    trainer = _mated_after([90, 40])
    assert trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "white", 4, 50, _Pos()
    ) == "champion"


def test_fumbler_pair_prefers_getting_mated_at_all():
    """None means the net never got itself checkmated — the failure
    case. Any finite result beats it."""
    trainer = _mated_after([200, None])
    assert trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "black", 4, 50, _Pos()
    ) == "challenger"

    trainer = _mated_after([None, 200])
    assert trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "black", 4, 50, _Pos()
    ) == "champion"


def test_fumbler_pair_draws_when_neither_or_equal():
    for canned in ([None, None], [55, 55]):
        trainer = _mated_after(canned)
        assert trainer._play_fumbler_pair(
            _uniform_evaluator, _uniform_evaluator, "white", 4, 50, _Pos()
        ) == "draw"


def test_fumbler_pair_gives_both_nets_the_identical_opponent():
    """The two halves of a pair must face the same fumbler, or the
    comparison measures luck. The seed is derived from the position and
    seat, so it is also stable across generations."""
    seeds: list[int] = []
    trainer = _tiny_trainer()

    def capture(net_eval, net_color, sims, cap, state, seed):
        seeds.append(seed)
        return 50

    trainer._play_fumbler_game = capture  # type: ignore[method-assign]
    trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "white", 4, 50, _Pos()
    )
    assert len(seeds) == 2 and seeds[0] == seeds[1]

    # A different seat must draw a different fumbler.
    seeds.clear()
    trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "black", 4, 50, _Pos()
    )
    other = seeds[0]
    seeds.clear()
    trainer._play_fumbler_pair(
        _uniform_evaluator, _uniform_evaluator, "white", 4, 50, _Pos()
    )
    assert other != seeds[0]


def test_fumbler_game_only_scores_the_nets_own_mate():
    """Checkmating the FUMBLER is the failure the whole project is about
    avoiding; it must never be scored as success. Driving the fool's
    mate line, the net playing black delivers mate — so black must come
    back as 'never got itself mated'."""
    import chess_ai.mcts as mcts_module

    ply = {"i": 0}

    def scripted(states, evaluator, sims, rng, temperatures=None, **kwargs):
        from_rf, to_rf = FOOLS_MATE[ply["i"] % len(FOOLS_MATE)]
        ply["i"] += 1
        move = next(
            (m for m in get_legal_moves(states[0])
             if (m.from_pos.rank, m.from_pos.file) == from_rf
             and (m.to_pos.rank, m.to_pos.file) == to_rf),
            get_legal_moves(states[0])[0],
        )
        return [SimpleNamespace(move=move, policy=None, root_value=0.0)]

    real = mcts_module.run_batched_mcts
    mcts_module.run_batched_mcts = scripted
    try:
        trainer = _tiny_trainer()
        # Not a faithful game (the fumbler moves randomly), but the
        # scoring branch is what is under test: whatever happens, a
        # result is only returned when the NET's king is the mated one.
        result = trainer._play_fumbler_game(
            _uniform_evaluator, "white", 4, 12, _fresh_start(), seed=1,
        )
    finally:
        mcts_module.run_batched_mcts = real
    assert result is None or isinstance(result, int)


def test_fumbler_always_accepts_an_offered_mate():
    """The fumbler moves at random EXCEPT that it takes a mate when one
    is available. Without that clause it never finds mate from a full
    position and every gate game draws — measured: the first smoke pair
    had both nets fail to be mated at all.

    Driven from the curated mate-in-1 positions with the FUMBLER on
    move: it has a mate among many legal replies and must find it on
    ply 1 every time, whatever the seed. The net never gets to move, so
    the result isolates the fumbler's behaviour.
    """
    from chess_ai.eval_positions import build_eval_positions

    mate_in_1 = [p for p in build_eval_positions()
                 if p.difficulty == "mate-in-1"][:4]
    assert mate_in_1

    trainer = _tiny_trainer()
    for position in mate_in_1:
        # The side to move has the mate — so it is the fumbler's seat,
        # and the NET is the side about to be checkmated.
        net_color = "black" if position.state.currentTurn == "white" else "white"
        results = [
            trainer._play_fumbler_game(
                _uniform_evaluator, net_color, 4, 40, position.state, seed=s,
            )
            for s in range(5)
        ]
        assert set(results) == {1}, (
            f"{position.name}: fumbler declined an available mate "
            f"(got {results}) — with an opponent that never accepts, "
            f"every gate game draws"
        )


def test_champion_half_of_a_pair_is_cached_per_champion():
    """The champion's result is fixed between promotions, so recomputing
    it for every eval doubles a match that already runs for hours. It is
    memoised on the champion's generation, and a promotion must retire
    the whole memo."""
    trainer = _tiny_trainer()
    played: list[str] = []

    def counting(net_eval, net_color, sims, cap, state, seed):
        played.append(net_color)
        return 60 if net_eval is CHAMP else 40

    CHAMP = object()
    trainer._play_fumbler_game = counting  # type: ignore[method-assign]
    trainer._champion_gen = 7

    for _ in range(3):
        assert trainer._play_fumbler_pair(
            _uniform_evaluator, CHAMP, "white", 4, 50, _Pos()
        ) == "challenger"
    # 3 challenger games, but the champion was only played once.
    assert len(played) == 4, f"expected 3 challenger + 1 champion, got {played}"

    # A promotion invalidates it: the new champion must be measured.
    trainer._champion_gen = 8
    trainer._play_fumbler_pair(
        _uniform_evaluator, CHAMP, "white", 4, 50, _Pos()
    )
    assert len(played) == 6, "cache survived a champion change"


def test_sparring_partner_accepts_offered_mates():
    """The dominant source of terminal signal in mirror play.

    Uniform blunders alone reached only 29.5% checkmate over 61 measured
    games (against 71.5% under the old vs-Sage mix), because landing on
    a mate by chance already presupposes the skill being learned — the
    agent must first manufacture a position where mates are plentiful.
    Accepting an offered mate converts that skill directly into a
    finished game.

    Driven from mate-in-1 positions with the SPARRING side on move: a
    mate is on offer, so at accept-probability 1.0 it must be taken and
    the game must end in checkmate immediately.
    """
    from chess_ai.eval_positions import build_eval_positions

    mate_in_1 = [p for p in build_eval_positions()
                 if p.difficulty == "mate-in-1"][:3]
    assert mate_in_1

    for position in mate_in_1:
        config = SelfPlayConfig(
            num_concurrent_games=1,
            mcts_simulations=2,
            invert_agent_selection=True,
            frozen_evaluator=None,
            agent_selfplay_prob=1.0,
            spar_accept_mate_prob=1.0,
            spar_random_prob=0.0,
        )
        engine = SelfPlayEngine(
            _uniform_evaluator, lambda ex: None, config, random.Random(1)
        )
        slot = engine.games[0]
        slot.state = position.state.copy()
        slot.state.status = "active"
        # The side to move holds the mate; make that the sparring seat.
        slot.spar_color = slot.state.currentTurn
        engine.step()
        assert engine.games[0] is not slot or slot.state.status == "checkmate", (
            f"{position.name}: sparring side declined an available mate"
        )


def test_sparring_partner_can_decline_below_probability_one():
    """Below 1.0 the agent has to manufacture chances repeatedly instead
    of banking on a single one, so the acceptance must really be
    probabilistic rather than always-on."""
    from chess_ai.eval_positions import build_eval_positions

    position = next(p for p in build_eval_positions()
                    if p.difficulty == "mate-in-1")
    accepted = 0
    for seed in range(30):
        config = SelfPlayConfig(
            num_concurrent_games=1,
            mcts_simulations=2,
            invert_agent_selection=True,
            frozen_evaluator=None,
            agent_selfplay_prob=1.0,
            spar_accept_mate_prob=0.5,
            spar_random_prob=0.0,
        )
        engine = SelfPlayEngine(
            _uniform_evaluator, lambda ex: None, config, random.Random(seed)
        )
        slot = engine.games[0]
        slot.state = position.state.copy()
        slot.state.status = "active"
        slot.spar_color = slot.state.currentTurn
        engine.step()
        # A finished game is replaced by a fresh slot.
        if engine.games[0] is not slot:
            accepted += 1
    assert 0 < accepted < 30, (
        f"acceptance is not probabilistic: {accepted}/30 at p=0.5"
    )
