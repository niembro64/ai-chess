"""Tests for self-play policy softening.

Without softening, a collapsed policy head that puts prior 0.7+ on one
wrong move and 0.001 on the mating move cannot be rescued by MCTS at
low sim counts — PUCT's exploration bonus for a 0.001-prior move is
dwarfed by the value signal flowing through the 0.7-prior move.
Softening flattens the prior with `p**(1/T)` + renormalize so low-prior
good moves get enough visit budget to be discovered.

These tests pin down two invariants:
  (1) T=1.0 is a no-op (priors unchanged).
  (2) T>1.0 monotonically closes the ratio between max and min priors.
And a smoke test that MCTS still runs with softening enabled.
"""

from __future__ import annotations

import random

import numpy as np

from chess_ai.encoding import POLICY_SIZE
from chess_ai.mcts import _soften_policy, run_batched_mcts
from chess_ai.engine import create_initial_game_state


def test_soften_identity_at_temperature_one():
    policies = np.random.dirichlet([0.3] * POLICY_SIZE, size=4).astype(np.float32)
    out = _soften_policy(policies, 1.0)
    np.testing.assert_allclose(out, policies, rtol=1e-5, atol=1e-6)


def test_soften_flattens_sharp_prior():
    """Collapsed-style prior: one move at 0.78, rest uniformly tiny."""
    sharp = np.full((1, POLICY_SIZE), (1.0 - 0.78) / (POLICY_SIZE - 1), dtype=np.float32)
    sharp[0, 0] = 0.78
    softened = _soften_policy(sharp, 1.5)

    # Max prob dropped; ratio max/min shrank.
    assert softened[0, 0] < sharp[0, 0]
    sharp_ratio = sharp.max() / sharp.min()
    soft_ratio = softened.max() / softened.min()
    assert soft_ratio < sharp_ratio

    # Still a valid probability distribution.
    np.testing.assert_allclose(softened.sum(axis=-1), 1.0, rtol=1e-5)


def test_soften_monotone_in_temperature():
    """Higher T = flatter prior (smaller max/min ratio)."""
    sharp = np.full((1, POLICY_SIZE), 1e-4, dtype=np.float32)
    sharp[0, 0] = 0.7
    sharp /= sharp.sum(axis=-1, keepdims=True)

    ratios = []
    for T in [1.0, 1.2, 1.5, 2.0]:
        s = _soften_policy(sharp, T)
        ratios.append(s.max() / s.min())
    # Ratios should monotonically decrease as T grows.
    for a, b in zip(ratios, ratios[1:]):
        assert b < a, f"softening not monotonic in T: ratios={ratios}"


def test_run_batched_mcts_accepts_softening_param():
    """Smoke: passing policy_softening_temperature doesn't crash and
    returns the same structure as the baseline call."""
    state = create_initial_game_state()
    state.status = "active"

    def stub_eval(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = boards.shape[0]
        p = np.full((b, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32)
        v = np.zeros(b, dtype=np.float32)
        return p, v

    rng = random.Random(0)
    baseline = run_batched_mcts([state], stub_eval, 8, rng)
    softened = run_batched_mcts(
        [state], stub_eval, 8, rng, policy_softening_temperature=1.5
    )
    assert len(baseline) == 1 and len(softened) == 1
    assert softened[0].policy.shape == (POLICY_SIZE,)
    # Visits still sum to something; result is sane.
    assert softened[0].policy.sum() > 0.999
