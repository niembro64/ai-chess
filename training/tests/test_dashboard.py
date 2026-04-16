"""Smoke test the dashboard renderer + CSV logger.

Forces non-TTY mode (no real live-view to capture in pytest), but verifies:
- Context manager enters/exits without error
- `on_step` updates the rolling buffers and writes CSV rows
- All render helpers produce objects without raising
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from chess_ai.dashboard import DashboardLogger
from chess_ai.train import TrainStats


def _stats(step: int, gen: int, **overrides) -> TrainStats:
    s = TrainStats(
        step=step,
        generation=gen,
        games_completed=step // 5,
        white_wins=step // 15,
        black_wins=step // 17,
        draws=step // 13,
        policy_loss=10.0 / (1 + gen * 0.1),
        value_loss=2.0 / (1 + gen * 0.2),
        total_loss=12.0 / (1 + gen * 0.15),
        replay_size=step * 10,
        games_per_min=30.0,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_dashboard_csv_writes_header_and_rows(tmp_path: Path, monkeypatch):
    # Force non-TTY path so Live is skipped.
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ckpt = tmp_path / "run"
    with DashboardLogger(
        ckpt,
        model_summary="blocks=2\nfilters=16",
        device_summary="cpu",
    ) as dash:
        for step in range(1, 21):
            dash.on_step(_stats(step=step, gen=step - 1))

    csv_path = ckpt / "stats.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    # Header columns are well-formed
    assert {"step", "gen", "policy_loss", "value_loss", "total_loss"}.issubset(rows[0].keys())
    # Values round-trip as strings; confirm monotonic step progression
    assert [int(r["step"]) for r in rows] == list(range(1, 21))


def test_dashboard_renders_without_live(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    with DashboardLogger(
        tmp_path / "run2",
        model_summary="blocks=2\nfilters=16",
        device_summary="cpu",
    ) as dash:
        # Feed a handful of datapoints and ensure render helpers work.
        for step in range(1, 30):
            dash.on_step(_stats(step=step, gen=step - 1))
        layout = dash._render(_stats(step=100, gen=99))
        # Layout has children — rendering should not raise
        assert layout is not None


def test_dashboard_handles_no_stats_yet(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    with DashboardLogger(
        tmp_path / "run3",
        model_summary="blocks=2\nfilters=16",
        device_summary="cpu",
    ) as dash:
        # Render with no stats at all (fresh start)
        assert dash._render(None) is not None
