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
    "time", "step", "gen", "target_gens", "games",
    # Aggregates (kept in the CSV for back-compat with older plotting scripts
    # that read the pre-split schema). These are computed on the fly from the
    # granular buckets below.
    "white_wins", "black_wins", "draws", "caps", "tb_adjudications",
    # Granular end-state buckets — canonical source of truth.
    "mate_w", "mate_b", "stalemate", "draw_50",
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
        # `Trainer._run_eval_match` returns). Newest last. Sized to fit the
        # validation panel's history table.
        self._eval_history: deque[dict] = deque(maxlen=10)
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

    def on_eval_progress(
        self,
        games_done: int,
        total: int,
        wins: int,
        draws: int,
        losses: int,
        per_diff: dict | None = None,
    ) -> None:
        """Trainer callback: fires after each game in a live eval match.

        The validation panel reads `_eval_progress` and renders a live
        scoreboard so the user sees activity during the 10+ min it takes
        to run 120 games. Per-difficulty breakdown shows which bucket is
        carrying the match (e.g., challenger dominating mate-in-1 but
        losing balanced openings). Triggers a manual Live refresh so
        the bar advances visibly between games (training main loop is
        paused during eval, so on_step wouldn't fire on its own cadence).
        """
        self._eval_progress = {
            "done": games_done, "total": total,
            "wins": wins, "draws": draws, "losses": losses,
            "per_diff": per_diff or {},
        }
        if self._live is not None:
            try:
                self._live.refresh()
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
        self.log(
            f"eval gen {result['gen']:,}: "
            f"{result['wins']}-{result['draws']}-{result['losses']} "
            f"score={result['score']:.3f} Δelo={result['elo_diff']:+.0f}  {tag}"
        )

    def on_step(self, stats) -> None:
        """Trainer callback: update loss history, refresh TUI, append CSV row."""
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
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top", size=9),
            Layout(name="middle", size=19),
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
        """Granular end-state breakdown.

        Eight buckets grouped into four sections. Over-the-board checkmates
        and natural draws are the model's own play; the tablebase section
        is what Syzygy told us about cap-timeouts; unresolved is the
        remaining "we have no signal" residue.
        """
        buckets: dict[str, int] = {
            "mate_w": 0, "mate_b": 0,
            "stalemate": 0, "draw_50": 0,
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
        """Running score + projected Elo + threshold — 'how the fight is going'."""
        import math
        w = int(progress["wins"])
        d = int(progress["draws"])
        l = int(progress["losses"])
        done = w + d + l
        if done == 0:
            return Text("")
        score = (w + 0.5 * d) / done
        s_clamp = max(0.01, min(0.99, score))
        elo = -400.0 * math.log10(1.0 / s_clamp - 1.0)
        score_color = (
            "bright_green" if score >= self._eval_threshold
            else "yellow" if score >= 0.5
            else "red"
        )
        return Text.assemble(
            ("score ", "dim"),
            (f"{score:.3f}", score_color),
            ("   Δelo ", "dim"),
            (f"{elo:+.0f}", "bright_green" if elo >= 0 else "red"),
            (f"   (thresh {self._eval_threshold:.2f})", "dim"),
        )

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
            label = {"mate-in-1": "mate-1", "balanced": "openings"}.get(name, name)
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
            body.add_row("status", Text("waiting for first eval match…", style="dim italic"))
            if self._eval_progress is not None:
                body.add_row("", self._progress_row(self._eval_progress))
            title = "validation"
            return Panel(body, title=title, border_style="magenta")

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

        body.add_row("", Text(""))  # spacer

        # --- History table (most recent first, up to what fits)
        head = Text.assemble(
            ("gen        ", "dim bold"),
            ("W-D-L     ", "dim bold"),
            ("score  ", "dim bold"),
            ("Δelo  ", "dim bold"),
        )
        body.add_row("", head)

        # Show up to 5 rows newest-first. Anything older falls off. We
        # budget 5 data rows + 4 header rows + 1 spacer = 10, fits 15-row panel.
        for h in reversed(hist[-5:]):
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
            score_style = "bright_green" if score >= self._eval_threshold else "yellow"
            elo_style = "bright_green" if elo >= 0 else "red"
            star = " ★" if h.get("new_champion") else "  "
            row_text = Text.assemble(
                (f"{gen_str}  ", "white"),
                (f"{w:>2}-{d:>2}-{l:>2}    ", "dim"),
                (f"{score:>5.3f}  ", score_style),
                (f"{elo:+5.1f}", elo_style),
                (star, "bright_yellow bold"),
            )
            body.add_row("", row_text)

        title = "validation"
        border = "magenta"
        if self._plateau_max > 0 and streak >= self._plateau_max:
            title += "  — PLATEAU STOP"
            border = "red"
        return Panel(body, title=title, border_style=border)

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
        return Panel(table, title="timings (EMA, per loop/step)", border_style="yellow")

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
