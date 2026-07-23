"""Tests for per-ply decay on the self-play value target.

Without decay, every position in a winning game gets the same value
label (±1). That leaves MCTS Q-values undifferentiated across all
winning moves, so search can't prefer faster wins over slower ones.
Decay scales the label magnitude by `decay ** plies_from_end`, so the
value head learns to rank mate-in-1 > mate-in-20 > "merely winning."

These tests build a tiny synthetic game with a known length and check
that the emitted TrainingExample values match the expected decay
schedule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chess_ai.encoding import NUM_PLANES, POLICY_SIZE
from chess_ai.rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights
from chess_ai.selfplay import (
    GameSlot,
    GameSlotExample,
    SelfPlayConfig,
    SelfPlayEngine,
    TrainingExample,
)


def _fake_slot(n_examples: int, turn_colors: list[str]) -> GameSlot:
    """Build a GameSlot with n_examples pre-populated. State is a dummy
    post-mate state (white just mated black, so currentTurn='black' and
    status='checkmate'). Turn colors list the per-example side-to-move
    in the order they were recorded.
    """
    assert len(turn_colors) == n_examples
    from chess_ai.engine import create_initial_game_state
    state = create_initial_game_state()
    state.status = "checkmate"
    state.currentTurn = "black"  # white just moved and mated
    slot = GameSlot(
        state=state,
        move_count=n_examples,  # approximate
        examples=[],
        move_cap=1000,
        origin="standard",
    )
    for tc in turn_colors:
        slot.examples.append(
            GameSlotExample(
                board=np.zeros(8 * 8 * NUM_PLANES, dtype=np.float32),
                policy=np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32),
                turn_color=tc,
                position_score=0.0,
            )
        )
    return slot


def _collect_examples(slot: GameSlot, *, decay: float) -> list[TrainingExample]:
    """Run _finish_game with the given decay, return the emitted examples."""
    collected: list[TrainingExample] = []

    def sink(ex: TrainingExample) -> None:
        collected.append(ex)

    cfg = SelfPlayConfig(
        num_concurrent_games=1,
        rewards=RewardWeights(**DEFAULT_REWARD_WEIGHTS.__dict__),
        value_ply_decay=decay,
    )
    # Construct an engine-ish shim: we only need _finish_game + its deps.
    # SelfPlayEngine.__init__ wants an evaluator; pass a no-op.
    def noop_eval(boards):
        return np.zeros((len(boards), POLICY_SIZE), dtype=np.float32), np.zeros(
            (len(boards),), dtype=np.float32
        )

    engine = SelfPlayEngine(evaluator=noop_eval, example_sink=sink, config=cfg)
    engine._finish_game(slot)
    return collected


def test_decay_one_reproduces_alphazero_convention():
    """With decay=1.0, every position in a winning game labels to ±1 (up
    to sign from side-to-move). This is the AZ convention and the
    existing default — guards against silent regressions in the
    pre-decay code path."""
    slot = _fake_slot(10, ["white"] * 10)
    out = _collect_examples(slot, decay=1.0)
    # White just mated black → white_outcome = +1. All white-stm
    # examples get value +1.
    assert len(out) == 10
    for ex in out:
        assert ex.value == pytest.approx(1.0)
        assert ex.outcome_known is True


def test_decay_applies_per_ply():
    """With decay < 1, position k plies from terminal gets value
    decay**k. Last example (the position the winning move was played
    from, k=0) keeps the FULL ±1 magnitude; first example (k=n-1) is
    exponentially smaller."""
    n = 10
    decay = 0.9
    slot = _fake_slot(n, ["white"] * n)
    out = _collect_examples(slot, decay=decay)
    assert len(out) == n

    for i, ex in enumerate(out):
        plies_from_end = n - 1 - i
        expected = (decay ** plies_from_end) * 1.0  # outcome = +1
        assert ex.value == pytest.approx(expected, rel=1e-6), (
            f"example {i}: got {ex.value}, expected {expected} "
            f"(plies_from_end={plies_from_end})"
        )
    # The winning move's position must carry the undecayed label.
    assert out[-1].value == pytest.approx(1.0)


def test_decay_flips_sign_by_side_to_move():
    """Value is recorded from the side-to-move's perspective. In a
    white-wins game, white-stm examples get +decay^k, black-stm examples
    get -decay^k."""
    n = 6
    decay = 0.95
    # Alternating turns, as in a real game from white's first move.
    turn_colors = ["white", "black", "white", "black", "white", "black"]
    slot = _fake_slot(n, turn_colors)
    out = _collect_examples(slot, decay=decay)
    for i, ex in enumerate(out):
        plies_from_end = n - 1 - i
        mag = decay ** plies_from_end
        expected = mag if turn_colors[i] == "white" else -mag
        assert ex.value == pytest.approx(expected, rel=1e-6)


def test_decay_does_not_affect_drawn_games():
    """On a drawn outcome, white_outcome=0 → value=0 at every ply
    regardless of decay. Decay only scales the magnitude of a nonzero
    label."""
    from chess_ai.engine import create_initial_game_state
    n = 5
    state = create_initial_game_state()
    state.status = "stalemate"  # draw
    state.currentTurn = "black"
    slot = GameSlot(
        state=state,
        move_count=n,
        examples=[],
        move_cap=1000,
        origin="standard",
    )
    for _ in range(n):
        slot.examples.append(
            GameSlotExample(
                board=np.zeros(8 * 8 * NUM_PLANES, dtype=np.float32),
                policy=np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32),
                turn_color="white",
                position_score=0.0,
            )
        )

    out = _collect_examples(slot, decay=0.9)
    for ex in out:
        assert ex.value == pytest.approx(0.0)
