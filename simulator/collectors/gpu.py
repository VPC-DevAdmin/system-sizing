"""NVIDIA GPU telemetry collector (roadmap 1.2).

Samples SM utilization, VRAM, power, SM clock, and throttle state at
1 Hz for every visible GPU. Two backends, picked at first use:

* **NVML** via ``pynvml`` (package ``nvidia-ml-py``) — in-process,
  ~free per sample. Preferred when importable and the driver responds.
* **nvidia-smi** subprocess with CSV output — always present wherever
  the driver is installed; ~50 ms per sample, fine at 1 Hz.

Multi-GPU hosts aggregate per sample: SM util and clock average across
devices (the "how busy/fast is the GPU complex" view), VRAM and power
sum (capacity and wall draw), throttled is any-device.

Like every collector, absence is not an error: ``is_available()``
returns False on CPU-only hosts and the run completes with NULLs in
the GPU telemetry columns.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# nvidia-smi clocks_throttle_reasons.active bitmask values that mean
# "actually slowed down" (as opposed to idle or app-configured clocks):
# SwPowerCap 0x4 | HwSlowdown 0x8 | SwThermalSlowdown 0x20 |
# HwThermalSlowdown 0x40 | HwPowerBrakeSlowdown 0x80.
_THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


@dataclass
class GpuSample:
    """One 1 Hz reading aggregated across all visible GPUs."""
    sm_util_pct: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    power_w: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    throttled: Optional[bool] = None


def _mean(vals: list[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _combine_devices(devices: list[dict]) -> Optional[GpuSample]:
    """Fold per-device readings into one GpuSample (see module doc)."""
    if not devices:
        return None
    utils = [d["sm_util_pct"] for d in devices if d.get("sm_util_pct") is not None]
    used = [d["vram_used_gb"] for d in devices if d.get("vram_used_gb") is not None]
    total = [d["vram_total_gb"] for d in devices if d.get("vram_total_gb") is not None]
    power = [d["power_w"] for d in devices if d.get("power_w") is not None]
    clocks = [d["sm_clock_mhz"] for d in devices if d.get("sm_clock_mhz") is not None]
    throttles = [d["throttled"] for d in devices if d.get("throttled") is not None]
    return GpuSample(
        sm_util_pct=_mean(utils),
        vram_used_gb=sum(used) if used else None,
        vram_total_gb=sum(total) if total else None,
        power_w=sum(power) if power else None,
        sm_clock_mhz=_mean(clocks),
        throttled=any(throttles) if throttles else None,
    )


def parse_smi_csv(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``
    output (one line per device) into per-device dicts. Tolerates
    '[N/A]' / 'N/A' fields (e.g. power on some SKUs) as None."""

    def _num(s: str) -> Optional[float]:
        s = s.strip()
        if not s or "N/A" in s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    devices: list[dict] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        util, mem_used_mib, mem_total_mib, power_w, clock_mhz, reasons = parts[:6]
        throttled: Optional[bool] = None
        r = reasons.strip()
        if r.startswith("0x"):
            try:
                throttled = bool(int(r, 16) & _THROTTLE_MASK)
            except ValueError:
                pass
        used = _num(mem_used_mib)
        total = _num(mem_total_mib)
        devices.append({
            "sm_util_pct": _num(util),
            "vram_used_gb": used / 1024.0 if used is not None else None,
            "vram_total_gb": total / 1024.0 if total is not None else None,
            "power_w": _num(power_w),
            "sm_clock_mhz": _num(clock_mhz),
            "throttled": throttled,
        })
    return devices


_SMI_QUERY = (
    "utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,"
    "clocks_throttle_reasons.active"
)


class GpuCollector:
    """Availability-probing 1 Hz GPU sampler. Construction is free;
    the backend is resolved on the first ``is_available()`` call."""

    name = "gpu"

    def __init__(self) -> None:
        self._mode: Optional[str] = None   # "nvml" | "smi" | "unavailable"
        self._nvml = None
        self._handles: list = []
        self.status = "not_started"

    def is_available(self) -> bool:
        if self._mode is None:
            self._resolve_backend()
        return self._mode in ("nvml", "smi")

    def _resolve_backend(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                self._nvml = pynvml
                self._handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
                ]
                self._mode = "nvml"
                self.status = "ok"
                return
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 — ImportError or NVML init failure
            pass
        if shutil.which("nvidia-smi") is not None:
            # Confirm the driver actually responds before claiming smi.
            try:
                probe = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={_SMI_QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    self._mode = "smi"
                    self.status = "ok"
                    return
            except Exception:  # noqa: BLE001
                pass
        self._mode = "unavailable"
        self.status = "not_available"

    def sample(self) -> Optional[GpuSample]:
        """One aggregated reading; None when unavailable or on a
        transient failure (caller records NULLs for that second)."""
        if not self.is_available():
            return None
        try:
            if self._mode == "nvml":
                return self._sample_nvml()
            return self._sample_smi()
        except Exception as e:  # noqa: BLE001
            log.debug("gpu sample failed: %s", e)
            return None

    def _sample_nvml(self) -> Optional[GpuSample]:
        nv = self._nvml
        devices: list[dict] = []
        for h in self._handles:
            d: dict = {}
            try:
                d["sm_util_pct"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:  # noqa: BLE001
                d["sm_util_pct"] = None
            try:
                mem = nv.nvmlDeviceGetMemoryInfo(h)
                d["vram_used_gb"] = mem.used / (1024 ** 3)
                d["vram_total_gb"] = mem.total / (1024 ** 3)
            except Exception:  # noqa: BLE001
                d["vram_used_gb"] = d["vram_total_gb"] = None
            try:
                d["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:  # noqa: BLE001
                d["power_w"] = None
            try:
                d["sm_clock_mhz"] = float(
                    nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
                )
            except Exception:  # noqa: BLE001
                d["sm_clock_mhz"] = None
            try:
                reasons = nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
                d["throttled"] = bool(reasons & _THROTTLE_MASK)
            except Exception:  # noqa: BLE001
                d["throttled"] = None
            devices.append(d)
        return _combine_devices(devices)

    def _sample_smi(self) -> Optional[GpuSample]:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        return _combine_devices(parse_smi_csv(r.stdout))

    def close(self) -> None:
        if self._mode == "nvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._nvml = None
            self._handles = []
            self._mode = None


def aggregate_gpu_samples(samples: list[GpuSample]) -> dict:
    """Window aggregates from per-second GpuSamples, keyed to match the
    ``cohort_measurements`` aggregate columns. Empty dict when no
    sample carried GPU data."""
    utils = [s.sm_util_pct for s in samples if s.sm_util_pct is not None]
    used = [s.vram_used_gb for s in samples if s.vram_used_gb is not None]
    totals = [s.vram_total_gb for s in samples if s.vram_total_gb is not None]
    power = [s.power_w for s in samples if s.power_w is not None]
    clocks = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz is not None]
    throttles = [s.throttled for s in samples if s.throttled is not None]
    agg: dict = {}
    if utils:
        agg["gpu_sm_util_pct_avg"] = sum(utils) / len(utils)
        agg["gpu_sm_util_pct_peak"] = max(utils)
    if used:
        agg["gpu_vram_used_gb_avg"] = sum(used) / len(used)
        agg["gpu_vram_used_gb_peak"] = max(used)
    if totals:
        agg["gpu_vram_total_gb"] = max(totals)
    if power:
        agg["gpu_power_w_avg"] = sum(power) / len(power)
        agg["gpu_power_w_peak"] = max(power)
    if clocks:
        agg["gpu_sm_clock_mhz_avg"] = sum(clocks) / len(clocks)
        agg["gpu_sm_clock_mhz_min"] = min(clocks)
    if throttles:
        agg["gpu_throttle_fraction"] = sum(1 for t in throttles if t) / len(throttles)
    return agg
