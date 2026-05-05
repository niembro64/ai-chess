"""Live training dashboard + CSV logger.

Designed for the SSH + tmux workflow: the trainer paints a Rich-based TUI
in the current terminal pane so reattaching to tmux immediately shows
current losses, game outcomes, games/min, and an ASCII loss-curve plot.
All the same data also streams to `runs/<name>/stats.csv` for offline
plotting with matplotlib/pandas/whatever.

Gracefully falls back to plain-text logging when stdout isn't a TTY (e.g.,
piped to a file, running under nohup without a terminal).

Usage (inside the launcher):

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

from .sysmon import SystemMonitor, SystemSnapshot, StaticSystemInfo, probe_static_info

# Keep at most this many points on each rolling line in the ASCII plot.
_LOSS_HISTORY_POINTS = 500

CSV_FIELDS = (
    "time", "step", "gen", "target_gens", "games",
    # Aggregates (kept in the CSV for back-compat with older plotting scripts
    # that read the pre-split schema). These are computed on the fly from the
    # granular buckets below.
    "white_wins", "black_wins", "draws", "caps", "tb_adjudications",
    # Granular end-state buckets — canonical source of truth.
    "mate_w", "mate_b", "stalemate", "draw_50",
    "draw_repetition", "draw_insufficient",
    "tb_w", "tb_b", "tb_d", "cap",
    "gen_per_min", "games_per_min", "replay_size",
    "eta_seconds",
    "policy_loss", "value_loss", "total_loss",
    "t_drain_ms", "t_broadcast_ms", "t_sleep_ms", "t_iter_ms",
    "t_sample_ms", "t_h2d_ms", "t_forward_ms", "t_backward_ms", "t_optim_ms",
)


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _bar_pct(pct: float, width: int = 10) -> str:
    """Fixed-width unicode bar for utilization percentages (0-100)."""
    pct = max(0.0, min(100.0, pct))
    full = int(pct * width / 100.0)
    return "█" * full + "░" * (width - full)


def _util_style(pct: float) -> str:
    """Color utilization: red = idle (bad for training), green = hot."""
    if pct < 30:
        return "red"
    if pct < 70:
        return "yellow"
    return "green"


def _fractional_bar(pct: float, width: int, fg: str) -> Text:
    """Horizontal bar with 1/8-character resolution.

    Unlike rich's ProgressBar this (a) uses unicode partial blocks so tiny
    percentages still show a sliver instead of rounding to empty, and (b)
    fills against a visible track so categorical breakdowns don't all look
    80% empty when one bucket dominates.

    Character ladder (1/8 increments): ▏ ▎ ▍ ▌ ▋ ▊ ▉ █.
    """
    pct = max(0.0, min(100.0, pct))
    cells = pct * width / 100.0
    full = int(cells)
    frac = cells - full

    partials = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    p_idx = int(round(frac * 8))
    if p_idx == 8:
        full += 1
        p_idx = 0
    partial_char = partials[p_idx]

    rest = width - full - (1 if partial_char else 0)
    rest = max(0, rest)

    t = Text()
    if full > 0:
        t.append("█" * full, style=fg)
    if partial_char:
        t.append(partial_char, style=fg)
    if rest > 0:
        t.append("░" * rest, style="grey27")
    return t


def _format_eta(seconds: float | None) -> str:
    """Friendly remaining-time label for the dashboard ETA row."""
    if seconds is None:
        return "—"
    if seconds <= 0:
        return "done"
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{total}s"


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
        # Rolling window of recent auto-eval match results (the raw dicts
        # `Trainer._run_eval_match` returns). Newest last. Sized to hold a
        # full training run's worth (eval_every_gens=5000 over 100k gens =
        # 20 matches) with headroom for longer runs, so the score chart
        # panel can plot the full trajectory.
        self._eval_history: deque[dict] = deque(maxlen=60)
        # Score threshold the trainer is applying. Set from on_eval's first
        # match result so the panel can draw a clear "pass/fail" line.
        self._eval_threshold: float = 0.54
        # Max plateau budget, same mechanism as above.
        self._plateau_max: int = 0
        # Live progress during an in-flight eval match. None when idle,
        # {done, total, wins, draws, losses} while a match is running.
        # The validation panel renders a progress bar when this is set so
        # the user sees activity during the 10+ minutes a big match takes.
        self._eval_progress: dict | None = None

        # Rich setup. When stdout isn't a TTY we disable the live dashboard
        # and just print log lines + write CSV. This keeps `... > log.txt`
        # and non-interactive CI invocations sane.
        self._is_tty = sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"
        self._console = Console()
        self._live: Live | None = None
        self._refresh_per_second = refresh_per_second

        self._csv_writer: csv.DictWriter | None = None
        self._csv_fp = None

        # Hardware telemetry. Best-effort: missing libs (pynvml on macOS,
        # psutil unavailable) result in a hidden panel, not a crash.
        self._sysmon = SystemMonitor()
        self._sys_snapshot: SystemSnapshot | None = None
        # One-shot probe of CPU model name, driver version, disk space.
        # Cheap enough at startup; everything else in the hardware panel
        # comes from the live snapshot.
        self._static_info: StaticSystemInfo = probe_static_info(str(self.checkpoint_dir))

        # Cache of the most recent on_step stats so on_eval_progress can
        # rebuild the full layout (Rich's Live.refresh() only repaints
        # the stored renderable — it doesn't re-evaluate the panel
        # callables, so without a layout rebuild the live eval panel
        # appears frozen during a match).
        self._last_stats = None

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

    def on_eval_progress(
        self,
        games_done: int,
        total: int,
        wins: int,
        draws: int,
        losses: int,
        per_diff: dict | None = None,
        *,
        current: dict | None = None,
        recent: list[str] | None = None,
        elapsed_s: float = 0.0,
    ) -> None:
        """Trainer callback: fires twice per game in a live eval match
        (once before, once after) so the panel updates without waiting
        on the 30-60s game-play block.

        The validation panel reads `_eval_progress` and renders a live
        scoreboard with match ETA, per-difficulty W/D/L, a recent-games
        strip, and the name of the position currently being played.

        Triggers a full layout rebuild (not just Live.refresh()) because
        Rich only re-renders the stored Layout on refresh — the Panel
        contents are fixed when _render() builds them. Without the
        rebuild the eval panel appears frozen during a match since the
        training main loop's on_step cadence pauses during eval.
        """
        was_idle = self._eval_progress is None
        self._eval_progress = {
            "done": games_done, "total": total,
            "wins": wins, "draws": draws, "losses": losses,
            "per_diff": per_diff or {},
            "current": current,
            "recent": list(recent or []),
            "elapsed_s": float(elapsed_s),
        }
        # Events-log breadcrumb on match start. Symmetric with the
        # match-end log in on_eval(). Matters because an eval match
        # takes ~5-10 min and the live panel is only visible while the
        # user is looking; the events pane persists the lifecycle even
        # if they miss the live view.
        if was_idle and self._last_stats is not None:
            gen = getattr(self._last_stats, "generation", 0)
            self.log(f"eval match started: challenger gen {gen:,}")
        if self._live is not None:
            try:
                self._live.update(self._render(self._last_stats))
            except Exception:
                pass

    def on_eval(
        self,
        result: dict,
        *,
        threshold: float | None = None,
        plateau_max: int | None = None,
    ) -> None:
        """Trainer callback: record an auto-eval match and log a summary line.

        `result` matches what `Trainer._run_eval_match` returns:
            {gen, champion_gen, games, wins, draws, losses, score, elo_diff,
             new_champion, plateau_counter, [note]}
        """
        self._eval_progress = None   # match done; clear the live bar
        self._eval_history.append(result)
        if threshold is not None:
            self._eval_threshold = float(threshold)
        if plateau_max is not None:
            self._plateau_max = int(plateau_max)

        # Events pane gets a one-line summary. Bootstrap matches (no games
        # played yet) get a briefer line.
        if result.get("note") == "bootstrap":
            self.log(
                f"eval: champion bootstrapped at gen {result['gen']:,}"
            )
            return
        tag = "★ NEW CHAMPION" if result.get("new_champion") else "no change"
        # Human-friendly duration: "12m34s" instead of "754.3s".
        duration_s = float(result.get("duration_s") or 0.0)
        if duration_s > 0:
            mm, ss = divmod(int(duration_s), 60)
            dur = f" in {mm}m{ss:02d}s"
        else:
            dur = ""
        self.log(
            f"eval gen {result['gen']:,}{dur}: "
            f"{result['wins']}-{result['draws']}-{result['losses']} "
            f"score={result['score']:.3f} Δelo={result['elo_diff']:+.0f}  {tag}"
        )
        # Per-difficulty summary (optional — only if trainer passed it).
        # Highlights where the challenger out/underperformed, e.g.
        # "mate-1 0.72  trivial 0.50  openings 0.40". Quick look-see for
        # the events pane; the validation panel shows it graphically too.
        per_diff = result.get("per_diff") or {}
        if per_diff:
            parts = []
            for name in ("mate-in-1", "endgame", "middlegame", "opening"):
                stats = per_diff.get(name)
                if not stats:
                    continue
                n = stats["w"] + stats["d"] + stats["l"]
                if n == 0:
                    continue
                score = (stats["w"] + 0.5 * stats["d"]) / n
                label = {"mate-in-1": "mate-1", "middlegame": "midgame", "opening": "openings"}.get(name, name)
                parts.append(f"{label}={score:.2f}")
            if parts:
                self.log("  by difficulty:  " + "  ".join(parts))

    def on_step(self, stats) -> None:
        """Trainer callback: update loss history, refresh TUI, append CSV row."""
        self._last_stats = stats
        self._loss_history.append(
            stats.generation, stats.policy_loss, stats.value_loss, stats.total_loss
        )

        if self._csv_writer is not None:
            eta = getattr(stats, "eta_seconds", None)
            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "step": stats.step,
                "gen": stats.generation,
                "target_gens": getattr(stats, "target_gens", 0),
                "games": stats.games_completed,
                # Aggregates (computed from granular buckets).
                "white_wins": stats.white_wins,
                "black_wins": stats.black_wins,
                "draws": stats.draws,
                "caps": getattr(stats, "caps", 0),
                "tb_adjudications": getattr(stats, "tb_adjudications", 0),
                # Granular buckets.
                "mate_w": getattr(stats, "mate_w", 0),
                "mate_b": getattr(stats, "mate_b", 0),
                "stalemate": getattr(stats, "stalemate", 0),
                "draw_50": getattr(stats, "draw_50", 0),
                "draw_repetition": getattr(stats, "draw_repetition", 0),
                "draw_insufficient": getattr(stats, "draw_insufficient", 0),
                "tb_w": getattr(stats, "tb_w", 0),
                "tb_b": getattr(stats, "tb_b", 0),
                "tb_d": getattr(stats, "tb_d", 0),
                "cap": getattr(stats, "cap", 0),
                "gen_per_min": round(getattr(stats, "gen_per_min", 0.0), 2),
                "games_per_min": round(stats.games_per_min, 2),
                "replay_size": stats.replay_size,
                "eta_seconds": "" if eta is None else round(eta, 1),
                "policy_loss": round(stats.policy_loss, 4),
                "value_loss": round(stats.value_loss, 4),
                "total_loss": round(stats.total_loss, 4),
            }
            for t_field in (
                "t_drain_ms", "t_broadcast_ms", "t_sleep_ms", "t_iter_ms",
                "t_sample_ms", "t_h2d_ms", "t_forward_ms", "t_backward_ms", "t_optim_ms",
            ):
                row[t_field] = round(getattr(stats, t_field, 0.0), 2)
            self._csv_writer.writerow(row)

        if self._live is not None:
            self._live.update(self._render(stats))

    # -- Rendering --

    def _render(self, stats) -> Layout:
        # Refresh hardware snapshot each render tick. Cheap (psutil
        # delta-read + cached NVML handles) and keeps the panel live.
        self._sys_snapshot = self._sysmon.snapshot()

        # Eval chart gets its own row between middle and hardware — only
        # meaningful once we have ≥2 eval results. We still allocate the
        # row unconditionally (Rich doesn't easily support dynamic layout
        # dimensions) so the panel stays in a stable position; early in a
        # run it shows a 'waiting for eval matches' placeholder.
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top", size=9),
            Layout(name="middle", size=24),
            Layout(name="eval_chart", size=10),
            Layout(name="hardware", size=7),
            Layout(name="timings", size=9),
            Layout(name="loss"),
            Layout(name="events", size=10),
        )
        layout["top"].split_row(
            Layout(self._progress_panel(stats), name="progress"),
            Layout(self._model_panel(), name="model"),
        )
        layout["middle"].split_row(
            Layout(self._outcomes_panel(stats), name="outcomes"),
            Layout(self._eval_panel(), name="eval"),
        )
        layout["header"].update(self._header_panel())
        layout["eval_chart"].update(self._eval_chart_panel())
        layout["hardware"].update(self._hardware_panel())
        layout["timings"].update(self._timings_panel(stats))
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
        """Progress toward the 'well-trained' target + current pace + ETA.

        Rendered layout:
            gradient updates  N / target  (X.X%)
                              [▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱]
            gen / min         <windowed rate>
            ETA               <Dd Hh Mm>
            games             N  (X.X / min)
            buffer            N
        """
        t = Table.grid(padding=(0, 2), expand=True)
        t.add_column(style="dim", justify="right", width=18)
        t.add_column(ratio=1)

        if stats is None:
            for label in (
                "gradient updates", "progress", "gen / min",
                "ETA", "games", "buffer",
            ):
                t.add_row(label, "—")
            return Panel(t, title="progress", border_style="blue")

        target = max(1, getattr(stats, "target_gens", 0) or 1)
        pct = min(100.0, stats.generation / target * 100)

        t.add_row(
            "gradient updates",
            f"{stats.generation:,} / {target:,}  ({pct:.2f}%)",
        )
        t.add_row(
            "",
            ProgressBar(
                total=target,
                completed=min(stats.generation, target),
                width=46,
                style="blue",
                complete_style="bright_blue",
            ),
        )
        t.add_row("gen / min", f"{stats.gen_per_min:,.1f}")
        t.add_row("ETA", _format_eta(getattr(stats, "eta_seconds", None)))
        t.add_row(
            "games",
            f"{stats.games_completed:,}  ({stats.games_per_min:,.1f} / min)",
        )
        t.add_row("buffer", f"{stats.replay_size:,}")

        return Panel(t, title="progress", border_style="blue")

    def _model_panel(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", justify="right")
        t.add_column()

        def _add_kv_lines(blob: str) -> None:
            for line in blob.split("\n"):
                if not line:
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    t.add_row(k.strip(), v.strip())
                else:
                    t.add_row("", line)

        _add_kv_lines(self.model_summary)
        # device_summary (os/python/torch/host/cuda) was being clipped
        # here by the fixed panel height; it now lives in the hardware
        # panel where it fits. Keeping just model arch + csv here keeps
        # this panel focused on "what we're training."
        t.add_row("csv", str(self.csv_path))
        return Panel(t, title="model", border_style="blue")

    def _outcomes_panel(self, stats) -> Panel:
        """Granular end-state breakdown.

        Eight buckets grouped into four sections. Over-the-board checkmates
        and natural draws are the model's own play; the tablebase section
        is what Syzygy told us about cap-timeouts; unresolved is the
        remaining "we have no signal" residue.
        """
        buckets: dict[str, int] = {
            "mate_w": 0, "mate_b": 0,
            "stalemate": 0, "draw_50": 0,
            "draw_repetition": 0, "draw_insufficient": 0,
            "tb_w": 0, "tb_b": 0, "tb_d": 0,
            "cap": 0,
        }
        if stats is not None:
            for k in buckets:
                buckets[k] = getattr(stats, k, 0)
            # Back-compat: if we're fed an older stats object that still has
            # the pre-split aggregates, fall back to those. (No granular data
            # available — we'll just show W/B/D/Cap totals with zeroed splits.)
            legacy_mode = (
                all(v == 0 for v in buckets.values())
                and (
                    getattr(stats, "white_wins", 0)
                    + getattr(stats, "black_wins", 0)
                    + getattr(stats, "draws", 0)
                    + getattr(stats, "caps", 0)
                ) > 0
            )
        else:
            legacy_mode = False

        # Bar width scales with the terminal; keep it bounded on either end
        # so it doesn't collapse in narrow panes or overflow in wide ones.
        # Outcomes is in a half-row split with the eval panel, so we base off
        # half the terminal width.
        bar_w = max(12, min(26, self._console.size.width // 8))

        total_actual = sum(buckets.values())
        total = max(1, total_actual)

        def row(label: str, count: int, fg: str, indent: bool = True) -> tuple:
            pct = count / total * 100
            bar = _fractional_bar(pct, width=bar_w, fg=fg)
            lbl = Text(("  " + label) if indent else label, style=fg)
            return (
                lbl,
                Text(f"{count:>6,}", style=fg),
                Text(f"{pct:5.1f}%", style="dim"),
                bar,
            )

        def section(title: str) -> tuple:
            return (
                Text(title, style="dim italic"),
                Text("", style="dim"),
                Text("", style="dim"),
                Text("", style="dim"),
            )

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=13)       # label (indented within sections)
        table.add_column(width=8, justify="right")   # count
        table.add_column(width=7, justify="right")   # percent
        table.add_column(ratio=1)        # bar

        if legacy_mode:
            # Old-format stats — fall back to the 4-bucket view the previous
            # version of the dashboard showed.
            w = getattr(stats, "white_wins", 0)
            b = getattr(stats, "black_wins", 0)
            d = getattr(stats, "draws", 0)
            c = getattr(stats, "caps", 0)
            legacy_total = max(1, w + b + d + c)

            def legacy_row(lbl: str, count: int, fg: str) -> tuple:
                pct = count / legacy_total * 100
                return (
                    Text(lbl, style=fg),
                    Text(f"{count:>6,}", style=fg),
                    Text(f"{pct:5.1f}%", style="dim"),
                    _fractional_bar(pct, width=bar_w, fg=fg),
                )
            table.add_row(*legacy_row("W", w, "green"))
            table.add_row(*legacy_row("B", b, "red"))
            table.add_row(*legacy_row("D", d, "yellow"))
            table.add_row(*legacy_row("Cap", c, "cyan"))
            return Panel(
                table,
                title=f"outcomes  (n={w+b+d+c:,})",
                border_style="blue",
            )

        table.add_row(*section("checkmate"))
        table.add_row(*row("W mate", buckets["mate_w"], "bright_green"))
        table.add_row(*row("B mate", buckets["mate_b"], "bright_red"))

        table.add_row(*section("drawn"))
        table.add_row(*row("stalemate", buckets["stalemate"], "yellow"))
        table.add_row(*row("50-move", buckets["draw_50"], "yellow"))
        table.add_row(*row("3-fold", buckets["draw_repetition"], "yellow"))
        table.add_row(*row("insuff-mat", buckets["draw_insufficient"], "yellow"))

        table.add_row(*section("tablebase"))
        table.add_row(*row("TB → W", buckets["tb_w"], "green"))
        table.add_row(*row("TB → B", buckets["tb_b"], "red"))
        table.add_row(*row("TB → D", buckets["tb_d"], "bright_yellow"))

        table.add_row(*section("unresolved"))
        table.add_row(*row("cap", buckets["cap"], "magenta"))

        decisive = buckets["mate_w"] + buckets["mate_b"] + buckets["tb_w"] + buckets["tb_b"]
        decisive_frac = decisive / total * 100
        title = (
            f"outcomes  (n={total_actual:,}, decisive={decisive_frac:.1f}%)"
        )

        # Per-origin breakdown. Skipped when the run is single-origin (e.g.
        # endgame_start_prob=0 and random_start_prob=0) so we don't waste
        # rows on a redundant view. Rendered as free-form Text lines below
        # the table (escaping the 4-column grid so the metrics don't wrap).
        origin_stats = getattr(stats, "origin_outcomes", None) if stats is not None else None
        nonempty_origins: list[tuple[str, dict[str, int]]] = []
        if origin_stats:
            for name in ("standard", "endgame", "random"):
                d = origin_stats.get(name)
                if d and sum(d.values()) > 0:
                    nonempty_origins.append((name, d))

        if len(nonempty_origins) <= 1:
            return Panel(table, title=title, border_style="blue")

        origin_lines = [
            Text(
                "by origin  (natural mate vs tb-adjudicated)",
                style="dim italic",
            )
        ]
        for name, d in nonempty_origins:
            o_total = max(1, sum(d.values()))
            mates = d["mate_w"] + d["mate_b"]
            tbs = d["tb_w"] + d["tb_b"] + d["tb_d"]
            mate_pct = mates / o_total * 100
            tb_pct = tbs / o_total * 100
            # Conversion ratio: of all decisive games, how many were
            # natural mates vs tb-adjudicated. High = model converting on
            # its own; low = value head is freeloading on Syzygy. We only
            # display it once the decisive-game denominator is large
            # enough to be signal rather than noise — otherwise a single
            # lucky mate reads as "100%" which is actively misleading.
            decisive_o = mates + d["tb_w"] + d["tb_b"]
            MIN_CONV_DENOM = 10
            if decisive_o >= MIN_CONV_DENOM:
                conv = mates / decisive_o * 100
                conv_text: tuple[str, str] = (
                    f"{conv:4.1f}%",
                    "bright_green" if conv >= 50 else "yellow" if conv >= 20 else "red",
                )
            else:
                conv_text = (f"  —  (n={decisive_o})", "dim")
            origin_lines.append(
                Text.assemble(
                    (f"  {name:<10}", "bright_white"),
                    (f"{o_total:>6,}  ", "white"),
                    ("mate ", "dim"),
                    (f"{mate_pct:4.1f}%", "bright_green" if mate_pct >= 20 else "yellow"),
                    ("  tb ", "dim"),
                    (f"{tb_pct:4.1f}%", "cyan"),
                    ("  conv ", "dim"),
                    conv_text,
                )
            )

        return Panel(
            Group(table, *origin_lines),
            title=title,
            border_style="blue",
        )

    def _progress_row(self, progress: dict) -> Text:
        """Top progress line: fractional-block bar + W/D/L tally."""
        done = int(progress["done"])
        total = max(1, int(progress["total"]))
        pct = done / total * 100
        w = int(progress["wins"])
        d = int(progress["draws"])
        l = int(progress["losses"])
        bar = _fractional_bar(pct, width=20, fg="bright_magenta")
        return Text.assemble(
            bar,
            (f"  {done}/{total}  ", "bright_white"),
            ("W ", "dim"), (f"{w}", "bright_green"),
            ("  D ", "dim"), (f"{d}", "yellow"),
            ("  L ", "dim"), (f"{l}", "red"),
        )

    def _progress_score_row(self, progress: dict) -> Text:
        """Running score + projected Elo + threshold + elapsed/ETA —
        'how the fight is going' and 'when will it end'."""
        import math
        w = int(progress["wins"])
        d = int(progress["draws"])
        l = int(progress["losses"])
        done = w + d + l
        if done == 0:
            # Still show elapsed even when no game has finished yet.
            elapsed = float(progress.get("elapsed_s", 0.0))
            if elapsed > 0:
                return Text.assemble(
                    ("elapsed ", "dim"),
                    (_format_duration(elapsed), "bright_white"),
                )
            return Text("")
        score = (w + 0.5 * d) / done
        s_clamp = max(0.01, min(0.99, score))
        elo = -400.0 * math.log10(1.0 / s_clamp - 1.0)
        score_color = (
            "bright_green" if score >= self._eval_threshold
            else "yellow" if score >= 0.5
            else "red"
        )

        # ETA: extrapolate from games-per-second so far.
        elapsed = float(progress.get("elapsed_s", 0.0))
        total = max(1, int(progress.get("total", 0)))
        eta_str = ""
        if elapsed > 0 and done > 0:
            per_game = elapsed / done
            remaining = (total - done) * per_game
            eta_str = f"   elapsed {_format_duration(elapsed)}   ETA {_format_duration(remaining)}"

        return Text.assemble(
            ("score ", "dim"),
            (f"{score:.3f}", score_color),
            ("   Δelo ", "dim"),
            (f"{elo:+.0f}", "bright_green" if elo >= 0 else "red"),
            (f"   (thresh {self._eval_threshold:.2f})", "dim"),
            (eta_str, "dim"),
        )

    def _progress_now_row(self, progress: dict) -> Text | None:
        """'Currently playing' row: which curated position the eval is on
        right now, plus which side the challenger is taking. Returns None
        when there's no active game (between ticks or match-finished)."""
        current = progress.get("current")
        if not current:
            return None
        game_no = int(current.get("game", 0))
        total = int(progress.get("total", 0))
        pos_name = str(current.get("pos_name", "?"))
        difficulty = str(current.get("difficulty", "?"))
        challenger_white = bool(current.get("challenger_white"))
        side = "WHITE" if challenger_white else "BLACK"
        side_color = "bright_white" if challenger_white else "grey50"
        # Truncate long position names so the panel doesn't break layout.
        if len(pos_name) > 28:
            pos_name = pos_name[:25] + "…"
        return Text.assemble(
            (f"game {game_no}/{total}  ", "dim"),
            (pos_name, "bright_white"),
            (f"  ({difficulty})", "cyan"),
            ("  challenger ", "dim"),
            (side, side_color),
        )

    def _progress_recent_row(self, progress: dict) -> Text | None:
        """Last N game outcomes as a colored strip — 'W W D L W W …'.
        Newest on the right so the eye naturally reads it as history →
        present."""
        recent = progress.get("recent") or []
        if not recent:
            return None
        parts = []
        for r in recent:
            if r == "W":
                parts.append((r, "bright_green"))
            elif r == "L":
                parts.append((r, "red"))
            else:
                parts.append((r, "yellow"))
            parts.append((" ", "dim"))
        return Text.assemble(*parts, (" →", "dim"))

    def _progress_diff_rows(self, progress: dict) -> list[Text]:
        """Per-difficulty breakdown rows: name  W-D-L  (score).

        Shows only categories with at least one completed game so early
        in a match the row list grows naturally rather than showing
        empty buckets.
        """
        per_diff = progress.get("per_diff", {}) or {}
        out: list[Text] = []
        for name in ("mate-in-1", "trivial", "clear", "balanced"):
            stats = per_diff.get(name)
            if not stats:
                continue
            n = stats["w"] + stats["d"] + stats["l"]
            if n == 0:
                continue
            bucket_score = (stats["w"] + 0.5 * stats["d"]) / n
            score_color = (
                "bright_green" if bucket_score >= 0.6
                else "yellow" if bucket_score >= 0.4
                else "red"
            )
            # Display label — "mate-in-1" is long, abbreviate it.
            label = {"mate-in-1": "mate-1", "middlegame": "midgame", "opening": "openings"}.get(name, name)
            out.append(Text.assemble(
                (f"  {label:<8}", "bright_white"),
                (f"{stats['w']:>3}-{stats['d']:>2}-{stats['l']:>2}  ", "dim"),
                (f"({bucket_score:.2f})", score_color),
            ))
        return out

    def _eval_panel(self) -> Panel:
        """Auto-eval match history + plateau status.

        Split into three stacked blocks:
            1. header line: current champion gen + plateau streak / cap
            2. score sparkline across recent matches, with a threshold line
            3. small history table (5-6 rows): gen, W-D-L, score, Δelo, status
        """
        hist = list(self._eval_history)

        body = Table.grid(padding=(0, 1), expand=True)
        body.add_column(style="dim", justify="right", width=10)
        body.add_column(ratio=1)

        # --- Header block: champion + plateau
        if not hist:
            # Before the first eval completes there's no history; show a
            # one-line status placeholder + full live progress if a match
            # is in flight. Symmetric with the have-history branch below
            # so first-eval users aren't left staring at "waiting…".
            if self._eval_progress is None:
                body.add_row("status", Text("waiting for first eval match…", style="dim italic"))
                return Panel(body, title="validation", border_style="magenta")
            body.add_row("running", self._progress_row(self._eval_progress))
            score_line = self._progress_score_row(self._eval_progress)
            if str(score_line):
                body.add_row("", score_line)
            now_line = self._progress_now_row(self._eval_progress)
            if now_line is not None:
                body.add_row("now", now_line)
            recent_line = self._progress_recent_row(self._eval_progress)
            if recent_line is not None:
                body.add_row("recent", recent_line)
            for row in self._progress_diff_rows(self._eval_progress):
                body.add_row("", row)
            return Panel(body, title="validation", border_style="magenta")

        latest = hist[-1]
        champ_gen = latest.get("champion_gen", 0) or 0
        streak = int(latest.get("plateau_counter", 0) or 0)
        body.add_row("champion", Text(f"gen {champ_gen:,}", style="bright_white"))

        if self._plateau_max > 0:
            full = "▓" * streak
            rest = "░" * max(0, self._plateau_max - streak)
            plateau_style = "red" if streak >= self._plateau_max else "yellow" if streak > 0 else "green"
            plateau_text = Text.assemble(
                (full, plateau_style),
                (rest, "grey27"),
                (f"  {streak} / {self._plateau_max}", plateau_style),
            )
            body.add_row("plateau", plateau_text)
        else:
            body.add_row("plateau", Text("(disabled)", style="dim"))

        # --- Live progress (only shown when a match is currently running)
        if self._eval_progress is not None:
            body.add_row("running", self._progress_row(self._eval_progress))
            score_line = self._progress_score_row(self._eval_progress)
            if str(score_line):
                body.add_row("", score_line)
            now_line = self._progress_now_row(self._eval_progress)
            if now_line is not None:
                body.add_row("now", now_line)
            recent_line = self._progress_recent_row(self._eval_progress)
            if recent_line is not None:
                body.add_row("recent", recent_line)
            for row in self._progress_diff_rows(self._eval_progress):
                body.add_row("", row)

        # --- Sparkline of scores across recent matches
        # Map each score into one of eight sparkline bars using a fixed
        # [0.25, 0.75] window so the threshold line is always in the middle.
        bars = "▁▂▃▄▅▆▇█"
        def to_bar(s: float) -> str:
            lo, hi = 0.25, 0.75
            frac = max(0.0, min(1.0, (s - lo) / (hi - lo)))
            return bars[min(len(bars) - 1, int(frac * len(bars)))]
        spark = "".join(to_bar(float(h.get("score", 0.5))) for h in hist if h.get("games", 0) > 0)
        if spark:
            body.add_row(
                "scores",
                Text.assemble(
                    (spark, "cyan"),
                    (f"   threshold {self._eval_threshold:.2f}", "dim"),
                ),
            )

        # --- Per-difficulty trend (one row per bucket, sparkline + latest)
        # Only shown when at least two of the last few matches carry the
        # `score_<diff>` columns (added in the per-difficulty CSV change).
        # Older eval rows that predate the columns simply contribute no
        # bar — the sparkline stays short rather than being faked from
        # the aggregate. Skipped during a live match: the live panel
        # above already shows per-bucket W/D/L for the in-flight match,
        # and stacking another four rows would push the panel past its
        # height budget.
        diff_buckets: tuple[tuple[str, str], ...] = (
            ("mate-in-1",   "score_mate_in_1"),
            ("endgame",     "score_endgame"),
            ("middlegame",  "score_middlegame"),
            ("opening",     "score_opening"),
        )
        diff_window = list(hist)[-12:]   # most-recent slice for the sparkline
        show_diff_trend = self._eval_progress is None
        for label, key in (diff_buckets if show_diff_trend else ()):
            vals: list[float] = []
            for h in diff_window:
                v = h.get(key)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, str) and v not in ("", None):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if not vals:
                continue
            bucket_spark = "".join(to_bar(v) for v in vals)
            latest_v = vals[-1]
            latest_style = (
                "bright_green" if latest_v >= 0.7
                else "yellow" if latest_v >= 0.5
                else "red"
            )
            body.add_row(
                label,
                Text.assemble(
                    (bucket_spark, "cyan"),
                    (f"   {latest_v:.2f}", latest_style),
                ),
            )

        body.add_row("", Text(""))  # spacer

        # --- History table (most recent first, up to what fits)
        # Columns: challenger gen, opponent gen, W-D-L, score, Δelo, dur.
        # The "vs" column is the champion-at-match-time — makes it obvious
        # each score is measured against a moving baseline, not against
        # gen 0 random-init weights. Duration helps spot eval regressions
        # (if matches suddenly take 2× longer MCTS is stuck).
        head = Text.assemble(
            ("gen      ", "dim bold"),
            ("vs       ", "dim bold"),
            ("W-D-L     ", "dim bold"),
            ("score  ", "dim bold"),
            ("Δelo    ", "dim bold"),
            ("dur", "dim bold"),
        )
        body.add_row("", head)

        # When a match is live we've already consumed ~6 rows for the
        # running/score/now/recent/per-diff block, so keep the table
        # shorter. When idle, show a full 10 rows of history. The panel
        # is sized (22-row middle split) to fit either case.
        rows_to_show = 5 if self._eval_progress is not None else 10
        for h in reversed(list(hist)[-rows_to_show:]):
            gen_str = f"{h.get('gen', 0):>6,}"
            if h.get("note") == "bootstrap":
                row_text = Text.assemble(
                    (f"{gen_str}  ", "dim"),
                    ("bootstrap champion", "dim italic"),
                )
                body.add_row("", row_text)
                continue
            w, d, l = h.get("wins", 0), h.get("draws", 0), h.get("losses", 0)
            score = float(h.get("score", 0.5))
            elo = float(h.get("elo_diff", 0.0))
            # opponent_gen was added after the early runs; fall back to
            # champion_gen for back-compat with historical eval records
            # that predate the field.
            opp_gen = h.get("opponent_gen")
            if opp_gen is None:
                opp_gen = h.get("champion_gen", 0)
            opp_str = f"{opp_gen:>6,}"
            # Duration: "5m03" format. Missing/zero duration → "—".
            dur_s = float(h.get("duration_s") or 0.0)
            if dur_s > 0:
                mm, ss = divmod(int(dur_s), 60)
                dur_str = f"{mm}m{ss:02d}"
            else:
                dur_str = "  — "
            score_style = "bright_green" if score >= self._eval_threshold else "yellow"
            elo_style = "bright_green" if elo >= 0 else "red"
            star = " ★" if h.get("new_champion") else "  "
            row_text = Text.assemble(
                (f"{gen_str}  ", "white"),
                (f"{opp_str}  ", "dim"),
                (f"{w:>2}-{d:>2}-{l:>2}    ", "dim"),
                (f"{score:>5.3f}  ", score_style),
                (f"{elo:+5.1f}", elo_style),
                (star, "bright_yellow bold"),
                (f"  {dur_str}", "dim"),
            )
            body.add_row("", row_text)

        title = "validation"
        border = "magenta"
        if self._plateau_max > 0 and streak >= self._plateau_max:
            title += "  — PLATEAU STOP"
            border = "red"
        return Panel(body, title=title, border_style=border)

    def _eval_chart_panel(self) -> Panel:
        """Score trajectory across eval matches, with threshold line.

        Plots score (y-axis, 0.40-0.70 window centered on the threshold)
        vs challenger gen (x-axis). A horizontal line marks the promotion
        threshold — points above it were promotions (★), points below
        were non-promotions. Gives a single-glance view of whether the
        model is trending toward or away from promotions.

        Placeholder rendered until there are ≥2 eval results — a single
        point would just be noise and plotext needs ≥2 x-values to axis
        correctly anyway.
        """
        hist = [h for h in self._eval_history
                if h.get("games", 0) > 0 and h.get("note") != "bootstrap"]

        if len(hist) < 2:
            msg = Text(
                "waiting for eval matches — chart appears after the 2nd eval",
                style="dim italic",
            )
            return Panel(msg, title="eval scores", border_style="magenta")

        gens = [int(h["gen"]) for h in hist]
        scores = [float(h["score"]) for h in hist]
        # Promotions get a separate scatter series so they stand out.
        promo_gens = [int(h["gen"]) for h in hist if h.get("new_champion")]
        promo_scores = [float(h["score"]) for h in hist if h.get("new_champion")]
        threshold = [self._eval_threshold] * len(gens)

        size = self._console.size
        plot_w = max(40, size.width - 6)
        plot_h = max(6, 7)

        # Auto-window the y-axis around the actual data range so tiny
        # swings near the threshold stay visible. Pad by 0.03 on each
        # side, clamp to [0.35, 0.75] to avoid absurd scales on early
        # outlier results, and make sure the threshold line is always
        # inside the window.
        s_min = min(min(scores), self._eval_threshold)
        s_max = max(max(scores), self._eval_threshold)
        y_lo = max(0.35, s_min - 0.03)
        y_hi = min(0.75, s_max + 0.03)
        # Ensure at least a 0.08 window so short flat runs don't degenerate
        # into a single-line plot with invisible variation.
        if y_hi - y_lo < 0.08:
            mid = (y_hi + y_lo) / 2
            y_lo, y_hi = max(0.35, mid - 0.04), min(0.75, mid + 0.04)

        plt.clf()
        plt.theme("dark")
        plt.plot(gens, threshold, label=f"thresh {self._eval_threshold:.2f}",
                 color="yellow")
        plt.plot(gens, scores, label="score", color="cyan", marker="braille")
        if promo_gens:
            # No label on the scatter: plotext's legend renderer indexes
            # marker[s][i] for i in 0..2 and crashes on single-char markers
            # like ♥. The green hearts on the line are obvious without a
            # legend entry, and the tail line below names the promotion gens.
            plt.scatter(promo_gens, promo_scores,
                        color="green", marker="heart")
        plt.plotsize(plot_w, plot_h)
        plt.ylim(y_lo, y_hi)
        plt.xlabel("gen")
        plt.ylabel("score")
        try:
            chart = plt.build()
        except Exception as e:  # plotext is fragile; never let it kill training
            chart = f"(chart unavailable: {type(e).__name__}: {e})"

        # Tail summary line: latest score, delta, direction arrow.
        latest = hist[-1]
        latest_score = float(latest["score"])
        latest_elo = float(latest.get("elo_diff", 0.0))
        latest_gen = int(latest["gen"])
        opp = latest.get("opponent_gen") or latest.get("champion_gen", 0)
        tail_style = (
            "bright_green" if latest_score >= self._eval_threshold
            else "yellow" if latest_score >= 0.5
            else "red"
        )
        if len(hist) >= 2:
            prev_score = float(hist[-2]["score"])
            delta = latest_score - prev_score
            trend = "↑" if delta > 0.005 else "↓" if delta < -0.005 else "→"
            trend_str = f"   {trend} {delta:+.3f} vs previous"
        else:
            trend_str = ""
        tail = Text.assemble(
            (f"latest: gen {latest_gen:,} vs gen {opp:,}  ", "dim"),
            (f"score {latest_score:.3f}", tail_style),
            (f"   Δelo {latest_elo:+.1f}", "dim"),
            (trend_str, "dim"),
        )

        return Panel(Group(Text.from_ansi(chart), tail),
                     title="eval scores", border_style="magenta")

    def _timings_panel(self, stats) -> Panel:
        """EMA-smoothed per-phase durations for bottleneck diagnosis (ms)."""
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="dim", justify="right", width=14)
        table.add_column(justify="right", width=10)
        table.add_column(style="dim", justify="right", width=14)
        table.add_column(justify="right", width=10)
        table.add_column(style="dim", justify="right", width=14)
        table.add_column(justify="right", width=10)

        def fmt(name: str) -> str:
            if stats is None:
                return "—"
            v = getattr(stats, name, 0.0)
            return f"{v:,.2f} ms" if v > 0 else "—"

        # Layout: main-loop phases on top, train_step phases on bottom.
        table.add_row(
            "iter (total)", fmt("t_iter_ms"),
            "drain", fmt("t_drain_ms"),
            "sleep (starved)", fmt("t_sleep_ms"),
        )
        table.add_row(
            "broadcast", fmt("t_broadcast_ms"),
            "sample", fmt("t_sample_ms"),
            "h2d", fmt("t_h2d_ms"),
        )
        table.add_row(
            "forward", fmt("t_forward_ms"),
            "backward", fmt("t_backward_ms"),
            "optim", fmt("t_optim_ms"),
        )

        # Inference-server batching health. Avg batch = mean batch size
        # the GPU saw per dispatch; peak = all-time max; wait = mean
        # time between first request received and dispatch. Low avg vs
        # peak with low wait → server fires too eagerly. High wait near
        # batch_wait_ms → workers can't fill the window fast enough.
        if stats is not None and getattr(stats, "inf_avg_batch", 0.0) > 0:
            avg_b = getattr(stats, "inf_avg_batch", 0.0)
            peak = getattr(stats, "inf_peak_batch", 0)
            wait_ms = getattr(stats, "inf_avg_wait_ms", 0.0)
            disp_pm = getattr(stats, "inf_dispatches_per_min", 0.0)
            table.add_row(
                "inf-batch avg", f"{avg_b:,.1f}",
                "inf-peak", f"{peak:,d}",
                "inf-wait", f"{wait_ms:,.2f} ms",
            )
            table.add_row(
                "inf disp/min", f"{disp_pm:,.0f}",
                "", "",
                "", "",
            )

        return Panel(table, title="timings (EMA, per loop/step)", border_style="yellow")

    def _hardware_panel(self) -> Panel:
        """Live CPU / RAM / GPU utilization. Sampled per render tick."""
        snap = self._sys_snapshot
        if snap is None or not snap.have_data:
            # Libraries unavailable (pynvml missing, psutil missing) or
            # failed to init. Show a placeholder so the user knows the
            # panel is there but silent, rather than hiding it entirely.
            t = Table.grid(expand=True, padding=(0, 1))
            t.add_column()
            t.add_row(Text("telemetry unavailable (install psutil / pynvml)", style="dim"))
            return Panel(t, title="hardware", border_style="yellow")

        # Two-row grid: GPU row(s) on top, host row at bottom.
        table = Table.grid(expand=True, padding=(0, 2))
        # 4 info groups per row so a single GPU row fits comfortably at
        # typical terminal widths.
        table.add_column(style="dim", justify="right", width=10)
        table.add_column(width=26)
        table.add_column(style="dim", justify="right", width=10)
        table.add_column(width=26)
        table.add_column(style="dim", justify="right", width=10)
        table.add_column(width=22)

        for i, g in enumerate(snap.gpus):
            label = f"gpu{i}" if len(snap.gpus) > 1 else "gpu"
            name = g.name if g.name else "?"
            util_bar = _bar_pct(g.util_pct, width=10)
            util_txt = Text.assemble(
                (f"{util_bar}  ", _util_style(g.util_pct)),
                (f"{g.util_pct:5.1f}%  ", _util_style(g.util_pct)),
                (f"{name}", "dim"),
            )
            vram_pct = (
                100.0 * g.mem_used_gb / g.mem_total_gb if g.mem_total_gb > 0 else 0.0
            )
            vram_txt = (
                f"{g.mem_used_gb:4.1f} / {g.mem_total_gb:4.1f} GB  "
                f"({vram_pct:4.1f}%)"
            )
            extra_txt = f"{g.temp_c:.0f}°C"
            if g.power_limit_w > 0:
                extra_txt += f"   {g.power_w:.0f} / {g.power_limit_w:.0f} W"
            elif g.power_w > 0:
                extra_txt += f"   {g.power_w:.0f} W"
            table.add_row(label, util_txt, "vram", vram_txt, "temp", extra_txt)

        # Host row: CPU util + RAM.
        cpu_bar = _bar_pct(snap.cpu_util_pct, width=10)
        # Apple Silicon P/E breakdown when known; falls back to plain
        # core count otherwise.
        if self._static_info.cpu_p_cores or self._static_info.cpu_e_cores:
            core_label = (
                f"({self._static_info.cpu_p_cores}P+{self._static_info.cpu_e_cores}E)"
            )
        else:
            core_label = f"({snap.cpu_cores} cores)"
        cpu_txt = Text.assemble(
            (f"{cpu_bar}  ", _util_style(snap.cpu_util_pct)),
            (f"{snap.cpu_util_pct:5.1f}%  ", _util_style(snap.cpu_util_pct)),
            (core_label, "dim"),
        )
        ram_pct = (
            100.0 * snap.ram_used_gb / snap.ram_total_gb if snap.ram_total_gb > 0 else 0.0
        )
        ram_txt = (
            f"{snap.ram_used_gb:4.1f} / {snap.ram_total_gb:4.1f} GB  "
            f"({ram_pct:4.1f}%)"
        )
        table.add_row("cpu", cpu_txt, "ram", ram_txt, "", "")

        # Static info row: CPU model (truncated), NVIDIA driver, disk
        # free for runs/. Probed once at startup — not live — but valuable
        # at-a-glance context for bug reports.
        cpu_model = self._static_info.cpu_model or "?"
        # Truncate overly-long Intel names like
        # "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz" → fits column width.
        if len(cpu_model) > 34:
            cpu_model = cpu_model[:31] + "…"
        extras_txt = self._static_info.nvidia_driver or ""
        disk_txt = ""
        if self._static_info.disk_total_gb > 0:
            disk_txt = (
                f"{self._static_info.disk_free_gb:.0f} / "
                f"{self._static_info.disk_total_gb:.0f} GB free"
            )
        table.add_row(
            "model", Text(cpu_model, style="dim"),
            "driver", Text(extras_txt or "—", style="dim"),
            "disk", Text(disk_txt or "—", style="dim"),
        )

        # Environment line — pulls os/host/python/torch/cuda out of the
        # device_summary blob the launcher builds. Rendered as its own
        # full-width Text below the table (not a table row) because a
        # 6-column grid would truncate the long dotted list.
        env_fields = {}
        for line in (self.device_summary or "").split("\n"):
            if "=" in line:
                k, _, v = line.partition("=")
                env_fields[k.strip()] = v.strip()
        env_parts = []
        for key in ("device", "host", "os", "python", "torch", "cuda"):
            val = env_fields.get(key)
            if val:
                env_parts.append(f"{key}={val}")
        env_line = Text.assemble(
            ("              env  ", "dim"),
            ("  ·  ".join(env_parts) if env_parts else "—", "dim"),
        )

        return Panel(Group(table, env_line), title="hardware (live)",
                     border_style="yellow")

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
