"""Live training dashboard + CSV logger.

Designed for the SSH + tmux workflow: the trainer paints a Rich-based TUI
in the current terminal pane so reattaching to tmux immediately shows
current losses, game outcomes, games/min, and an ASCII loss-curve plot.
All the same data also streams to `runs/<name>/stats.csv` for offline
plotting with matplotlib/pandas/whatever.

Gracefully falls back to plain-text logging when stdout isn't a TTY (e.g.,
piped to a file, running under nohup without a terminal).

Usage (inside `train.py`):

    dash = DashboardLogger(checkpoint_dir, model_summary=..., on_log=logger.info)
    with dash:
        trainer.run(on_step=dash.on_step, ...)

The Trainer's `on_step(stats)` callback just forwards to this.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import plotext as plt
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

# Keep at most this many points on each rolling line in the ASCII plot.
_LOSS_HISTORY_POINTS = 500

CSV_FIELDS = (
    "time", "step", "gen", "games",
    "white_wins", "black_wins", "draws", "caps",
    "games_per_min", "replay_size",
    "policy_loss", "value_loss", "total_loss",
)


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class _LossHistory:
    """Rolling buffer of (gen, policy, value, total) tuples for plotting."""

    def __init__(self, max_points: int = _LOSS_HISTORY_POINTS):
        self._buf: deque[tuple[int, float, float, float]] = deque(maxlen=max_points)
        self._last_gen = -1

    def append(self, gen: int, policy: float, value: float, total: float) -> None:
        # Only record when generation advances (avoids duplicate points when the
        # trainer reports stats on every self-play step regardless of training).
        if gen == self._last_gen:
            return
        self._last_gen = gen
        self._buf.append((gen, policy, value, total))

    def render(self, width: int, height: int) -> str:
        if len(self._buf) < 2:
            return "waiting for gradient steps…"
        gens = [b[0] for b in self._buf]
        policy = [b[1] for b in self._buf]
        value = [b[2] for b in self._buf]
        total = [b[3] for b in self._buf]
        plt.clf()
        plt.theme("dark")
        plt.plot(gens, policy, label="policy", color="cyan")
        plt.plot(gens, value, label="value", color="magenta")
        plt.plot(gens, total, label="total", color="yellow")
        plt.plotsize(width, height)
        plt.xlabel("gen")
        plt.ylabel("loss")
        return plt.build()


class DashboardLogger:
    """Rich TUI + CSV logger. Plug into `Trainer.run(on_step=...)`."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        model_summary: str,
        device_summary: str,
        run_name: str | None = None,
        refresh_per_second: float = 4.0,
        on_log: Callable[[str], None] | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.checkpoint_dir / "stats.csv"
        self.run_name = run_name or self.checkpoint_dir.name

        self.model_summary = model_summary
        self.device_summary = device_summary

        self._start_time = time.time()
        self._loss_history = _LossHistory()
        self._events: deque[tuple[float, str]] = deque(maxlen=8)
        self._on_log = on_log

        # Rich setup. When stdout isn't a TTY we disable the live dashboard
        # and just print log lines + write CSV. This keeps `train.py > log.txt`
        # and non-interactive CI invocations sane.
        self._is_tty = sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"
        self._console = Console()
        self._live: Live | None = None
        self._refresh_per_second = refresh_per_second

        self._csv_writer: csv.DictWriter | None = None
        self._csv_fp = None

    # -- Context manager --

    def __enter__(self) -> "DashboardLogger":
        self._csv_fp = self.csv_path.open("a", newline="", buffering=1)
        write_header = self.csv_path.stat().st_size == 0
        self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=CSV_FIELDS)
        if write_header:
            self._csv_writer.writeheader()

        if self._is_tty:
            self._live = Live(
                self._render(None),
                console=self._console,
                refresh_per_second=self._refresh_per_second,
                screen=False,        # Inline (doesn't take over the tmux pane)
                transient=False,
            )
            self._live.__enter__()
        else:
            self.log(f"Dashboard disabled (non-TTY). CSV stats → {self.csv_path}")
        self.log(f"Run: {self.run_name}")
        self.log(f"Model: {self.model_summary}")
        self.log(f"Device: {self.device_summary}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live is not None:
            try:
                self._live.__exit__(exc_type, exc_val, exc_tb)
            finally:
                self._live = None
        if self._csv_fp is not None:
            self._csv_fp.close()
            self._csv_fp = None

    # -- Public API --

    def log(self, message: str) -> None:
        """Emit a log line (persisted in the events panel and optionally stdout)."""
        self._events.append((time.time(), message))
        if self._on_log is not None:
            self._on_log(message)

    def on_step(self, stats) -> None:
        """Trainer callback: update loss history, refresh TUI, append CSV row."""
        self._loss_history.append(
            stats.generation, stats.policy_loss, stats.value_loss, stats.total_loss
        )

        if self._csv_writer is not None:
            self._csv_writer.writerow(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "step": stats.step,
                    "gen": stats.generation,
                    "games": stats.games_completed,
                    "white_wins": stats.white_wins,
                    "black_wins": stats.black_wins,
                    "draws": stats.draws,
                    "caps": getattr(stats, "caps", 0),
                    "games_per_min": round(stats.games_per_min, 2),
                    "replay_size": stats.replay_size,
                    "policy_loss": round(stats.policy_loss, 4),
                    "value_loss": round(stats.value_loss, 4),
                    "total_loss": round(stats.total_loss, 4),
                }
            )

        if self._live is not None:
            self._live.update(self._render(stats))

    # -- Rendering --

    def _render(self, stats) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top", size=9),
            Layout(name="outcomes", size=7),
            Layout(name="loss"),
            Layout(name="events", size=10),
        )
        layout["top"].split_row(
            Layout(self._progress_panel(stats), name="progress"),
            Layout(self._model_panel(), name="model"),
        )
        layout["header"].update(self._header_panel())
        layout["outcomes"].update(self._outcomes_panel(stats))
        layout["loss"].update(self._loss_panel())
        layout["events"].update(self._events_panel())
        return layout

    def _header_panel(self) -> Panel:
        elapsed = _format_duration(time.time() - self._start_time)
        header = Table.grid(expand=True, padding=(0, 1))
        header.add_column(justify="left", ratio=3)
        header.add_column(justify="right", ratio=1)
        header.add_row(
            Text(f"ChessNet Training — {self.run_name}", style="bold cyan"),
            Text(f"uptime {elapsed}", style="dim"),
        )
        return Panel(header, border_style="cyan")

    def _progress_panel(self, stats) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", justify="right")
        t.add_column()
        if stats is None:
            for label in ("step", "gen", "games", "games/min", "buffer"):
                t.add_row(label, "—")
        else:
            t.add_row("step", f"{stats.step:,}")
            t.add_row("gen", f"{stats.generation:,}")
            t.add_row("games", f"{stats.games_completed:,}")
            t.add_row("games/min", f"{stats.games_per_min:,.1f}")
            t.add_row("buffer", f"{stats.replay_size:,}")
        return Panel(t, title="progress", border_style="blue")

    def _model_panel(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", justify="right")
        t.add_column()
        for line in self.model_summary.split("\n"):
            if "=" in line:
                k, _, v = line.partition("=")
                t.add_row(k.strip(), v.strip())
            else:
                t.add_row("", line)
        t.add_row("device", self.device_summary)
        t.add_row("csv", str(self.csv_path))
        return Panel(t, title="model", border_style="blue")

    def _outcomes_panel(self, stats) -> Panel:
        w = getattr(stats, "white_wins", 0) if stats else 0
        b = getattr(stats, "black_wins", 0) if stats else 0
        d = getattr(stats, "draws", 0) if stats else 0
        c = getattr(stats, "caps", 0) if stats else 0
        real_total = w + b + d + c
        total = max(1, real_total)

        def row(label: str, count: int, style: str) -> tuple:
            pct = count / total * 100
            bar = ProgressBar(total=100, completed=pct, width=30, style=style, complete_style=style)
            return (
                Text(label, style=f"bold {style}"),
                Text(f"{count:>5}", style=style),
                Text(f"{pct:5.1f}%", style="dim"),
                bar,
            )

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=3)
        table.add_column(width=6, justify="right")
        table.add_column(width=8, justify="right")
        table.add_column(ratio=1)
        # W / B = checkmate. D = true draw (stalemate, 50-move rule).
        # Cap = hit the move-cap timeout before anything decisive.
        table.add_row(*row("W", w, "green"))
        table.add_row(*row("B", b, "red"))
        table.add_row(*row("D", d, "yellow"))
        table.add_row(*row("Cap", c, "cyan"))
        return Panel(table, title=f"outcomes  (n={real_total})", border_style="blue")

    def _loss_panel(self) -> Panel:
        # plotext renders into a size in cell units. Leave margins for the panel border.
        size = self._console.size
        plot_w = max(40, size.width - 6)
        plot_h = max(8, 12)  # plot takes the remaining loss-section height roughly

        chart = self._loss_history.render(plot_w, plot_h)

        last = None
        if self._loss_history._buf:
            _, p, v, t = self._loss_history._buf[-1]
            last = Text(
                f"policy={p:.3f}   value={v:.3f}   total={t:.3f}",
                style="dim",
            )

        group_items: list = [Text.from_ansi(chart)]
        if last is not None:
            group_items.append(last)
        return Panel(Group(*group_items), title="loss (rolling)", border_style="magenta")

    def _events_panel(self) -> Panel:
        lines = []
        for ts, msg in self._events:
            ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            lines.append(Text.assemble((ts_str, "dim"), "  ", msg))
        if not lines:
            lines.append(Text("(no events yet)", style="dim italic"))
        return Panel(Group(*lines), title="events", border_style="dim")
