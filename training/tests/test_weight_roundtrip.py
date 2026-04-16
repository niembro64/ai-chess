"""PyTorch ↔ TF.js weight round-trip parity.

This test is the gate for Python-side training. If PyTorch-exported weights
don't produce the same forward-pass outputs when loaded into the browser's
TF.js ChessNet, then training on one side and deploying to the other is
meaningless.

To run:
    python training/scripts/make_roundtrip_fixture.py   # PyTorch side
    npm run verify-weights                               # TS side
    cd training && pytest tests/test_weight_roundtrip.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PY_OUTPUTS = FIXTURE_DIR / "roundtrip_pytorch_outputs.json"
TS_OUTPUTS = FIXTURE_DIR / "roundtrip_ts_outputs.json"

# Forward-pass tolerance. Tight enough to catch real bugs (BN epsilon drift,
# transposed conv kernels, wrong flatten axis) but loose enough for the small
# per-op float32 accumulation differences between PyTorch's own kernels and
# our hand-rolled CPUForward implementation.
ABS_TOL = 1e-4


@pytest.fixture(scope="module")
def outputs() -> tuple[dict, dict]:
    if not PY_OUTPUTS.exists():
        pytest.skip(
            f"Missing {PY_OUTPUTS.name}. "
            "Run `python training/scripts/make_roundtrip_fixture.py` first."
        )
    if not TS_OUTPUTS.exists():
        pytest.skip(
            f"Missing {TS_OUTPUTS.name}. Run `npm run verify-weights` first."
        )
    with PY_OUTPUTS.open() as f:
        py = json.load(f)
    with TS_OUTPUTS.open() as f:
        ts = json.load(f)
    return py, ts


def test_output_shapes_match(outputs: tuple[dict, dict]):
    py, ts = outputs
    assert len(py["policies"]) == len(ts["policies"])
    assert len(py["wdls"]) == len(ts["wdls"])
    assert len(py["policies"][0]) == len(ts["policies"][0])
    assert len(py["wdls"][0]) == len(ts["wdls"][0]) == 3


def test_policy_parity(outputs: tuple[dict, dict]):
    py, ts = outputs
    py_p = np.asarray(py["policies"], dtype=np.float32)
    ts_p = np.asarray(ts["policies"], dtype=np.float32)
    diff = np.abs(py_p - ts_p)
    max_diff = float(diff.max())
    assert max_diff < ABS_TOL, (
        f"Policy max abs diff = {max_diff:g} exceeds tolerance {ABS_TOL:g}. "
        f"Top offenders: {diff.flatten().argsort()[-5:]}"
    )


def test_wdl_parity(outputs: tuple[dict, dict]):
    py, ts = outputs
    py_w = np.asarray(py["wdls"], dtype=np.float32)
    ts_w = np.asarray(ts["wdls"], dtype=np.float32)
    # Every WDL row should sum to 1 on both sides
    assert np.allclose(py_w.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(ts_w.sum(axis=1), 1.0, atol=1e-5)
    diff = np.abs(py_w - ts_w)
    max_diff = float(diff.max())
    assert max_diff < ABS_TOL, f"WDL max abs diff = {max_diff:g} exceeds tolerance {ABS_TOL:g}."
