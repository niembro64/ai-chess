"""Tests for the Rust MCTS backend.

Rust MctsSearch mirrors chess_ai.mcts.MCTSSearch. These tests pin down
the non-parity-sensitive invariants (any MCTS implementation must
satisfy them) since bit-exact parity with Python is impossible: Rust
uses ChaCha8 for Dirichlet noise + move sampling, Python uses numpy
Mersenne Twister / the stdlib `random` module, so even with matched
seeds the actual random draws differ. That's fine for training —
both are unbiased samples from the same distributions.

The sign-convention regression (mate-in-1 must be findable with
uniform policy + zero value, purely via terminal backup) is tested
in test_mcts_sign_convention.py against whichever backend is
active — switching `CHESS_AI_PYTHON_MCTS=1` covers the Python path.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from chess_ai.encoding import POLICY_SIZE
from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves
from chess_ai.eval_positions import build_eval_positions
from chess_ai.mcts import _HAVE_RUST_MCTS, run_batched_mcts
import chess_ai.mcts as _mcts

pytestmark = pytest.mark.skipif(
    not _HAVE_RUST_MCTS,
    reason="Rust MCTS not built — run `maturin develop` in rust_engine/",
)


def _uniform_eval(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = boards.shape[0]
    return (
        np.full((b, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32),
        np.zeros(b, dtype=np.float32),
    )


def test_rust_mcts_finds_mate_in_1_with_uniform_policy():
    """Same invariant as test_mcts_sign_convention but explicitly through
    the Rust path: terminal backup with negated child-Q must surface the
    mating move even when the prior is uniform."""
    assert _mcts.USE_RUST_MCTS, "test is only meaningful on the Rust path"

    hand_positions = [
        p
        for p in build_eval_positions()
        if p.difficulty == "mate-in-1" and "random" not in p.name
    ]
    found = 0
    total = 0
    for pos in hand_positions:
        total += 1
        results = run_batched_mcts(
            [pos.state], _uniform_eval, 100, random.Random(0), temperatures=[0.0]
        )
        picked = results[0].move
        # Check whether the picked move is a mating move by applying it.
        after = apply_move(pos.state, picked)
        if after.status == "checkmate":
            found += 1
    # Hand-crafted mate-in-1 positions should be nearly all solvable at
    # 100 sims with only terminal-backup signal. Small slack for any
    # position where the mating move shares a policy index with a non-mating
    # underpromotion move (shouldn't happen on hand-crafted boards).
    assert found / total >= 0.9, f"Rust MCTS solved only {found}/{total} mate-in-1"


def test_rust_mcts_policy_normalized_and_move_is_legal():
    state = create_initial_game_state()
    state.status = "active"
    results = run_batched_mcts(
        [state], _uniform_eval, 50, random.Random(0), temperatures=[1.0]
    )
    r = results[0]
    assert abs(r.policy.sum() - 1.0) < 1e-5
    assert r.policy.min() >= 0.0
    legal = get_legal_moves(state)
    # The sampled move must be among the legal moves.
    assert any(
        (m.from_pos.rank, m.from_pos.file, m.to_pos.rank, m.to_pos.file, m.promotion)
        == (
            r.move.from_pos.rank,
            r.move.from_pos.file,
            r.move.to_pos.rank,
            r.move.to_pos.file,
            r.move.promotion,
        )
        for m in legal
    )


def test_rust_mcts_batch_multiple_states():
    """Batch of multiple games should produce one result per game."""
    s = create_initial_game_state()
    s.status = "active"
    states = [s, s, s]
    results = run_batched_mcts(
        states, _uniform_eval, 30, random.Random(0), temperatures=[1.0, 1.0, 0.0]
    )
    assert len(results) == 3
    for r in results:
        assert abs(r.policy.sum() - 1.0) < 1e-5


def test_rust_mcts_deterministic_under_same_seed():
    """Same state + same seed → same policy (Rust path uses ChaCha8 with
    Python-provided seeds, so reproducibility is under our control)."""
    s = create_initial_game_state()
    s.status = "active"
    r1 = run_batched_mcts(
        [s], _uniform_eval, 80, random.Random(42), temperatures=[1.0]
    )[0]
    r2 = run_batched_mcts(
        [s], _uniform_eval, 80, random.Random(42), temperatures=[1.0]
    )[0]
    np.testing.assert_allclose(r1.policy, r2.policy, atol=1e-6)
