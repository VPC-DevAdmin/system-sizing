"""Host-detail collector — what the CPUs, memory, disks and NICs are
actually doing, per second.

The existing telemetry answers "how busy is the CPU"; this collector
answers the follow-ups an operator asks next: busy doing WHAT
(user/system/iowait/irq split), how busy is EACH core (per-core
utilization — a single pegged tokenizer thread hides completely in a
host average), is the scheduler backed up (run queue, context
switches), where the memory sits (cached vs dirty vs swap), and
whether storage or network is under pressure (per-NVMe throughput,
IOPS and device busy%; NIC rates).

Everything reads /proc and psutil counters — no perf, no root. Each
``sample()`` returns a JSON-able dict (stored as one JSON column per
interval row rather than thirty new columns). First call returns
counter-less fields only; rates need two samples.

Optional IPMI system power: RAPL is unavailable on this fleet's
kernel (5.15 + Granite Rapids needs ~6.5), but Dell iDRAC exposes
chassis power via ``ipmitool dcmi power reading``. Probed once; most
hosts need the operator to grant /dev/ipmi0 access (docs/deploy.md),
and the collector degrades to absent rather than failing.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Optional

log = logging.getLogger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# /proc/stat cpu columns: user nice system idle iowait irq softirq steal
_STAT_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq",
                "softirq", "steal")

# Disk name prefixes worth reporting (whole devices, not partitions).
_DISK_PREFIXES = ("nvme", "sd", "vd", "dm-")


def _read_proc_stat() -> tuple[dict[int, tuple], dict, Optional[int], Optional[int]]:
    """(per_cpu_ticks, aggregate_ticks, ctxt, intr) from /proc/stat."""
    per_cpu: dict[int, tuple] = {}
    agg: dict = {}
    ctxt = intr = None
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu "):
                nums = [int(x) for x in line.split()[1:9]]
                agg = dict(zip(_STAT_FIELDS[:len(nums)], nums, strict=False))
            elif line.startswith("cpu"):
                parts = line.split()
                try:
                    cpu_id = int(parts[0][3:])
                except ValueError:
                    continue
                per_cpu[cpu_id] = tuple(int(x) for x in parts[1:9])
            elif line.startswith("ctxt "):
                ctxt = int(line.split()[1])
            elif line.startswith("intr "):
                intr = int(line.split()[1])
    return per_cpu, agg, ctxt, intr


def _probe_ipmi() -> bool:
    """One-time check: can this user read chassis power via IPMI?"""
    if shutil.which("ipmitool") is None:
        return False
    try:
        r = subprocess.run(
            ["ipmitool", "dcmi", "power", "reading"],
            capture_output=True, text=True, timeout=8,
        )
        return r.returncode == 0 and "Instantaneous power reading" in r.stdout
    except Exception:  # noqa: BLE001
        return False


def _read_ipmi_power_w() -> Optional[float]:
    try:
        r = subprocess.run(
            ["ipmitool", "dcmi", "power", "reading"],
            capture_output=True, text=True, timeout=8,
        )
        for line in r.stdout.splitlines():
            if "Instantaneous power reading" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:  # noqa: BLE001
        return None
    return None


class HostCollector:
    """1 Hz host-detail sampler. Construction is free; availability is
    resolved on first sample. status: ok | not_available."""

    name = "host_detail"

    def __init__(self) -> None:
        self.status = "not_started"
        self._prev_cpu: Optional[dict[int, tuple]] = None
        self._prev_agg: Optional[dict] = None
        self._prev_ctxt: Optional[int] = None
        self._prev_intr: Optional[int] = None
        self._prev_t: Optional[float] = None
        self._prev_disk: Optional[dict] = None
        self._prev_net: Optional[tuple] = None
        self._ipmi: Optional[bool] = None    # None = not probed yet

    def is_available(self) -> bool:
        return os.path.exists("/proc/stat")

    def sample(self) -> Optional[dict]:
        if not self.is_available():
            self.status = "not_available"
            return None
        try:
            out = self._sample_inner()
            self.status = "ok"
            return out
        except Exception as e:  # noqa: BLE001
            log.debug("host sample failed: %s", e)
            return None

    def _sample_inner(self) -> dict:
        now = time.monotonic()
        per_cpu, agg, ctxt, intr = _read_proc_stat()
        out: dict = {}

        if self._prev_cpu is not None and self._prev_t is not None:
            dt = max(1e-3, now - self._prev_t)
            # Per-core utilization (%). idle = idle + iowait, matching
            # the existing cpu_util definition.
            cores: list[Optional[float]] = []
            for cpu_id in sorted(per_cpu):
                cur, prev = per_cpu[cpu_id], self._prev_cpu.get(cpu_id)
                if prev is None:
                    cores.append(None)
                    continue
                d = [c - p for c, p in zip(cur, prev, strict=False)]
                total = sum(d)
                idle = d[3] + (d[4] if len(d) > 4 else 0)
                cores.append(
                    round(100.0 * (1 - idle / total), 1) if total > 0 else None
                )
            out["cores_util_pct"] = cores

            # What the host's cycles went to, as % of ALL cpu-ticks.
            if self._prev_agg and agg:
                d = {k: agg.get(k, 0) - self._prev_agg.get(k, 0)
                     for k in _STAT_FIELDS}
                total = sum(d.values())
                if total > 0:
                    out["cpu_breakdown_pct"] = {
                        "user": round(100 * (d["user"] + d["nice"]) / total, 1),
                        "system": round(100 * d["system"] / total, 1),
                        "iowait": round(100 * d["iowait"] / total, 1),
                        "irq": round(100 * (d["irq"] + d["softirq"]) / total, 1),
                        "steal": round(100 * d["steal"] / total, 1),
                    }
            if ctxt is not None and self._prev_ctxt is not None:
                out["ctx_switches_s"] = round((ctxt - self._prev_ctxt) / dt)
            if intr is not None and self._prev_intr is not None:
                out["interrupts_s"] = round((intr - self._prev_intr) / dt)

        self._prev_cpu, self._prev_agg = per_cpu, agg
        self._prev_ctxt, self._prev_intr = ctxt, intr

        try:
            out["load1"] = round(os.getloadavg()[0], 2)
        except OSError:
            pass

        # Memory breakdown beyond used_gb.
        try:
            kv: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    rest = rest.strip().removesuffix("kB").strip()
                    try:
                        kv[key.strip()] = int(rest)
                    except ValueError:
                        pass
            gb = 1024 * 1024
            out["mem"] = {
                "available_gb": round(kv.get("MemAvailable", 0) / gb, 1),
                "cached_gb": round(kv.get("Cached", 0) / gb, 1),
                "dirty_mb": round(kv.get("Dirty", 0) / 1024, 1),
                "swap_used_gb": round(
                    (kv.get("SwapTotal", 0) - kv.get("SwapFree", 0)) / gb, 2),
            }
        except OSError:
            pass

        if _HAS_PSUTIL:
            # Storage pressure: per-device rates from counter deltas.
            try:
                disks = psutil.disk_io_counters(perdisk=True)
                cur = {
                    name: (d.read_bytes, d.write_bytes,
                           d.read_count + d.write_count,
                           getattr(d, "busy_time", 0))
                    for name, d in disks.items()
                    if name.startswith(_DISK_PREFIXES)
                    and not (name.startswith("nvme") and "p" in name[4:])
                }
                if self._prev_disk is not None and self._prev_t is not None:
                    dt = max(1e-3, now - self._prev_t)
                    per: dict[str, dict] = {}
                    for name, c in cur.items():
                        p = self._prev_disk.get(name)
                        if p is None:
                            continue
                        per[name] = {
                            "read_mb_s": round((c[0] - p[0]) / dt / 1e6, 1),
                            "write_mb_s": round((c[1] - p[1]) / dt / 1e6, 1),
                            "iops": round((c[2] - p[2]) / dt),
                            "util_pct": round(
                                min(100.0, 100 * (c[3] - p[3]) / (dt * 1000)), 1),
                        }
                    if per:
                        out["disk"] = per
                self._prev_disk = cur
            except Exception:  # noqa: BLE001
                pass

            # Network (all NICs except loopback, summed).
            try:
                n = psutil.net_io_counters(pernic=True)
                rx = sum(v.bytes_recv for k, v in n.items() if k != "lo")
                tx = sum(v.bytes_sent for k, v in n.items() if k != "lo")
                if self._prev_net is not None and self._prev_t is not None:
                    dt = max(1e-3, now - self._prev_t)
                    out["net"] = {
                        "rx_mb_s": round((rx - self._prev_net[0]) / dt / 1e6, 2),
                        "tx_mb_s": round((tx - self._prev_net[1]) / dt / 1e6, 2),
                    }
                self._prev_net = (rx, tx)
            except Exception:  # noqa: BLE001
                pass

        # Chassis power (Dell iDRAC over IPMI) — probed once.
        if self._ipmi is None:
            self._ipmi = _probe_ipmi()
        if self._ipmi:
            p = _read_ipmi_power_w()
            if p is not None:
                out["system_power_w"] = p

        self._prev_t = now
        return out
