"""Reward-shaping parity: Python `evaluate_position` vs TS `evaluatePosition`.

The training pipeline uses the shaped reward as part of each training
example's value target. If Python and TS score a position differently,
Python-trained weights won't reproduce the browser trainer's behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chess_ai.engine import ChessGameState
from chess_ai.rewards import RewardWeights, evaluate_position

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity_positions.json"

# Reward calculation involves many float adds; ulp-level drift is expected,
# but real bugs show up at >1e-6.
ABS_TOL = 1e-6


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture missing. Run `npm run dump-parity` from the project root first."
        )
    with FIXTURE_PATH.open() as f:
        data = json.load(f)
    if data.get("version", 1) < 2 or "reward_weights" not in data:
        pytest.skip(
            "Parity fixture predates reward parity fields. Re-run `npm run dump-parity`."
        )
    return data


def test_reward_parity(fixture_data: dict):
    weights_dict = fixture_data["reward_weights"]
    weights = RewardWeights(**weights_dict)

    max_diff_white = 0.0
    max_diff_black = 0.0
    for pos in fixture_data["positions"]:
        state = ChessGameState.from_dict(pos["state"])
        legal_move_count = pos.get("legalMoveCount", len(pos["legalMoves"]))

        py_white = evaluate_position(state, "white", weights, legal_move_count)
        py_black = evaluate_position(state, "black", weights, legal_move_count)
        ts_white = pos["rewardWhite"]
        ts_black = pos["rewardBlack"]

        dw = abs(py_white - ts_white)
        db = abs(py_black - ts_black)
        max_diff_white = max(max_diff_white, dw)
        max_diff_black = max(max_diff_black, db)

        if not math.isclose(py_white, ts_white, abs_tol=ABS_TOL):
            raise AssertionError(
                f"Reward diff (white) at id={pos['id']}: py={py_white} ts={ts_white} "
                f"diff={dw:g}"
            )
        if not math.isclose(py_black, ts_black, abs_tol=ABS_TOL):
            raise AssertionError(
                f"Reward diff (black) at id={pos['id']}: py={py_black} ts={ts_black} "
                f"diff={db:g}"
            )
