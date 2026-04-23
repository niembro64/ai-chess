"""Lightweight host + GPU telemetry for the dashboard.

Probes CPU / RAM / GPU utilization at a configurable cadence. All
operations are best-effort — missing libraries (pynvml on macOS, psutil
unavailable) just yield empty snapshots, so the dashboard degrades
gracefully instead of crashing.

Designed to be cheap enough to call every dashboard refresh (~4 Hz).
The nvmlDeviceGetUtilizationRates call is ~1-2 ms on the first read
and ~50-100 µs cached thereafter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

try:
    import pynvml  # type: ignore
    _HAVE_PYNVML = True
except ImportError:
    _HAVE_PYNVML = False


@dataclass
class GpuSnapshot:
    name: str = ""
    util_pct: float = 0.0          # 0-100
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    temp_c: float = 0.0
    power_w: float = 0.0
    power_limit_w: float = 0.0


@dataclass
class SystemSnapshot:
    cpu_util_pct: float = 0.0      # whole-box aggregate, 0-100
    cpu_cores: int = 0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpus: list[GpuSnapshot] = field(default_factory=list)
    # True iff *any* live telemetry source is responding. When False the
    # dashboard hides the hardware panel instead of showing empty rows.
    have_data: bool = False


class SystemMonitor:
    """Maintains open handles to psutil / pynvml. Call `snapshot()` from
    the dashboard refresh loop."""

    def __init__(self) -> None:
        self._nvml_inited = False
        self._nvml_handles: list = []
        self._initialized = False
        # psutil's cpu_percent needs a prior call to establish a baseline
        # for its delta computation. The first call always returns 0.0.
        self._psutil_primed = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if _HAVE_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_inited = True
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    self._nvml_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except Exception:
                # Driver not installed, not loaded, permission issue, etc.
                # Fall through — the snapshot will just omit GPU rows.
                self._nvml_inited = False
                self._nvml_handles = []
        if _HAVE_PSUTIL:
            # Prime cpu_percent so the next call returns a real value.
            psutil.cpu_percent(interval=None)
            self._psutil_primed = True

    def snapshot(self) -> SystemSnapshot:
        self._ensure_init()
        snap = SystemSnapshot()
        if _HAVE_PSUTIL:
            try:
                snap.cpu_util_pct = psutil.cpu_percent(interval=None)
                snap.cpu_cores = psutil.cpu_count(logical=True) or 0
                mem = psutil.virtual_memory()
                snap.ram_used_gb = mem.used / (1024 ** 3)
                snap.ram_total_gb = mem.total / (1024 ** 3)
                snap.have_data = True
            except Exception:
                pass

        if self._nvml_inited:
            for i, h in enumerate(self._nvml_handles):
                g = GpuSnapshot()
                try:
                    # nvmlDeviceGetName returns bytes on older pynvml, str on newer.
                    raw = pynvml.nvmlDeviceGetName(h)
                    g.name = raw.decode() if isinstance(raw, bytes) else raw
                except Exception:
                    g.name = f"gpu{i}"
                try:
                    rates = pynvml.nvmlDeviceGetUtilizationRates(h)
                    g.util_pct = float(rates.gpu)
                except Exception:
                    pass
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    g.mem_used_gb = mem.used / (1024 ** 3)
                    g.mem_total_gb = mem.total / (1024 ** 3)
                except Exception:
                    pass
                try:
                    g.temp_c = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass
                try:
                    g.power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    pass
                try:
                    g.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                except Exception:
                    pass
                snap.gpus.append(g)
            if snap.gpus:
                snap.have_data = True
        return snap

    def close(self) -> None:
        if self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_inited = False
            self._nvml_handles = []
