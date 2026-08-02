"""Live terminal dashboard for the Toy trainer — Sage's dashboard, miniaturized.

Four panels, one screen: progress/throughput, game outcomes (watch the
repetition rate fall as the net learns there's more to chess than
shuffling), a rolling loss chart, and live hardware telemetry (reuses
chess_ai.sysmon). Renders with rich.Live; toy_train.py activates it
only when stdout is a real terminal.
"""

from __future__ import annotations

import time
from collections import deque

import plotext as plt
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chess_ai.sysmon import SystemMonitor

OUTCOME_STYLES = [
    ("mate", "bright_green"),
    ("stalemate", "yellow"),
    ("draw", "yellow"),          # engine 50-move status
    ("repetition", "magenta"),
    ("insufficient", "yellow"),
    ("cap", "cyan"),
]


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class ToyDashboard:
    def __init__(self, total_iters: int, device: str, param_count: int) -> None:
        self.total_iters = total_iters
        self.device = device
        self.param_count = param_count
        self.t0 = time.time()
        self.iteration = 0
        self.games_total = 0
        self.buffer_len = 0
        self.p_loss = 0.0
        self.v_loss = 0.0
        self.selfplay_s = 0.0
        self.train_s = 0.0
        self.outcomes: dict[str, int] = {}
        self.history: deque[tuple[int, float, float]] = deque(maxlen=400)
        self._console = Console()
        self._monitor = SystemMonitor()
        self._live: Live | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "ToyDashboard":
        self._live = Live(
            self._render(), console=self._console,
            refresh_per_second=2, screen=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.__exit__(*exc)

    # -- updates ------------------------------------------------------------

    def on_iteration(
        self,
        iteration: int,
        labels: dict[str, int],
        buffer_len: int,
        p_loss: float,
        v_loss: float,
        selfplay_s: float,
        train_s: float,
    ) -> None:
        self.iteration = iteration
        self.buffer_len = buffer_len
        self.p_loss = p_loss
        self.v_loss = v_loss
        self.selfplay_s = selfplay_s
        self.train_s = train_s
        for k, n in labels.items():
            self.outcomes[k] = self.outcomes.get(k, 0) + n
            self.games_total += n
        self.history.append((iteration, p_loss, v_loss))
        if self._live:
            self._live.update(self._render())

    # -- rendering ------------------------------------------------------------

    def _render(self) -> Group:
        return Group(
            self._header(),
            self._progress_and_outcomes(),
            self._loss_panel(),
            self._hardware_panel(),
        )

    def _header(self) -> Panel:
        uptime = _fmt_duration(time.time() - self.t0)
        return Panel(
            Text.assemble(
                ("TOY TRAINING", "bold cyan"),
                (f"   {self.param_count / 1e6:.2f}M params · {self.device}", "dim"),
                (f"   uptime {uptime}", "dim"),
            ),
            border_style="cyan",
        )

    def _progress_and_outcomes(self) -> Table:
        wrap = Table.grid(expand=True)
        wrap.add_column(ratio=1)
        wrap.add_column(ratio=1)

        prog = Table.grid(padding=(0, 2))
        prog.add_column(justify="right", style="dim")
        prog.add_column()
        elapsed_min = max(1e-9, (time.time() - self.t0) / 60)
        pct = self.iteration / max(1, self.total_iters) * 100
        prog.add_row("iteration", f"{self.iteration:,} / {self.total_iters:,}  ({pct:.1f}%)")
        prog.add_row("games", f"{self.games_total:,}  ({self.games_total / elapsed_min:.1f}/min)")
        prog.add_row("buffer", f"{self.buffer_len:,}")
        prog.add_row("selfplay", f"{self.selfplay_s:.1f}s / iter")
        prog.add_row("train", f"{self.train_s:.2f}s / iter")

        out = Table.grid(padding=(0, 1))
        out.add_column(width=13, style="dim")
        out.add_column(width=7, justify="right")
        out.add_column(width=7, justify="right", style="dim")
        out.add_column(ratio=1)
        total = max(1, self.games_total)
        for key, style in OUTCOME_STYLES:
            n = self.outcomes.get(key, 0)
            share = n / total * 100
            bar = "█" * int(share / 5) + "░" * (20 - int(share / 5))
            out.add_row(key, f"{n:,}", f"{share:4.1f}%", Text(bar, style=style))

        wrap.add_row(
            Panel(prog, title="progress", border_style="blue"),
            Panel(out, title=f"outcomes (n={self.games_total:,})", border_style="blue"),
        )
        return wrap

    def _loss_panel(self) -> Panel:
        if len(self.history) < 2:
            return Panel(Text("collecting…", style="dim italic"),
                         title="loss", border_style="magenta")
        iters = [h[0] for h in self.history]
        p = [h[1] for h in self.history]
        v = [h[2] for h in self.history]
        size = self._console.size
        plt.clf()
        plt.theme("dark")
        plt.plot(iters, p, label="policy", color="cyan", marker="braille")
        plt.plot(iters, v, label="value", color="magenta", marker="braille")
        plt.plotsize(max(40, size.width - 8), 9)
        try:
            chart = plt.build()
        except Exception as e:  # plotext is fragile; never kill training
            chart = f"(chart unavailable: {type(e).__name__})"
        tail = f"policy={self.p_loss:.3f}   value={self.v_loss:.4f}"
        return Panel(Group(Text.from_ansi(chart), Text(tail, style="bold")),
                     title="loss (rolling)", border_style="magenta")

    def _hardware_panel(self) -> Panel:
        snap = self._monitor.snapshot()
        if not snap.have_data:
            return Panel(Text("no telemetry", style="dim"), border_style="green")
        parts = [f"cpu {snap.cpu_util_pct:4.0f}% ({snap.cpu_cores} cores)",
                 f"ram {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB"]
        for g in snap.gpus:
            parts.append(
                f"gpu {g.util_pct:.0f}%  vram {g.mem_used_gb:.1f}/{g.mem_total_gb:.1f} GB"
                + (f"  {g.temp_c:.0f}°C" if g.temp_c else "")
            )
        return Panel(Text("   ·   ".join(parts)), title="hardware", border_style="green")
