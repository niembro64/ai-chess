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


@dataclass
class StaticSystemInfo:
    """Host info probed once at startup. Unlike SystemSnapshot this is
    immutable — everything here is a fixed property of the hardware
    (CPU model) or driver install (NVIDIA driver version) that doesn't
    change while training is running. Splitting it out means the live
    snapshot path stays cheap (no repeated subprocess calls)."""
    cpu_model: str = ""
    cpu_physical_cores: int = 0
    cpu_logical_cores: int = 0
    # Apple Silicon only: performance/efficiency core split. Both 0
    # on non-Apple or pre-M1 hardware.
    cpu_p_cores: int = 0
    cpu_e_cores: int = 0
    nvidia_driver: str = ""
    # Disk free at startup, in GB, for the checkpoint directory's
    # filesystem. Updated rarely (not every render) but cheap enough
    # for occasional refresh.
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0


def _read_cpu_model() -> str:
    """Best-effort readable CPU model name across Linux/Mac/Windows.

    `platform.processor()` is near-useless on Linux (returns 'x86_64')
    and Mac (returns 'arm'), so we fall through to /proc/cpuinfo or
    sysctl for something like 'Apple M1 Max' / 'Intel(R) Core(TM)
    i7-9700K CPU @ 3.60GHz'.
    """
    import platform
    import subprocess
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=1.0,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif system == "Windows":
            p = platform.processor()
            if p:
                return p
    except Exception:
        pass
    return platform.processor() or "unknown"


def _read_apple_pe_cores() -> tuple[int, int]:
    """(P-cores, E-cores) on Apple Silicon; (0, 0) elsewhere.

    Uses sysctl's perflevel hierarchy: perflevel0 = performance cores,
    perflevel1 = efficiency cores. Pre-M1 Macs and non-Macs return
    (0, 0) and the caller should treat that as 'unknown'.
    """
    import platform
    import subprocess
    if platform.system() != "Darwin":
        return (0, 0)
    try:
        p = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True, text=True, timeout=1.0,
        )
        e = subprocess.run(
            ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
            capture_output=True, text=True, timeout=1.0,
        )
        if p.returncode == 0 and e.returncode == 0:
            return (int(p.stdout.strip()), int(e.stdout.strip()))
    except Exception:
        pass
    return (0, 0)


def probe_static_info(checkpoint_dir: str | None = None) -> StaticSystemInfo:
    """One-shot probe of host info the dashboard wants to display in the
    header/hardware panel. Cheap enough to call at dashboard startup.
    Everything best-effort: missing probes leave fields at their default."""
    import platform
    import shutil

    info = StaticSystemInfo()
    info.cpu_model = _read_cpu_model()
    info.cpu_p_cores, info.cpu_e_cores = _read_apple_pe_cores()

    if _HAVE_PSUTIL:
        try:
            info.cpu_physical_cores = psutil.cpu_count(logical=False) or 0
            info.cpu_logical_cores = psutil.cpu_count(logical=True) or 0
        except Exception:
            pass

    if _HAVE_PYNVML and platform.system() != "Darwin":
        try:
            pynvml.nvmlInit()
            raw = pynvml.nvmlSystemGetDriverVersion()
            info.nvidia_driver = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass

    if checkpoint_dir is not None:
        try:
            usage = shutil.disk_usage(checkpoint_dir)
            info.disk_free_gb = usage.free / (1024 ** 3)
            info.disk_total_gb = usage.total / (1024 ** 3)
        except Exception:
            pass

    return info


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
