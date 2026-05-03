"""Effective CPU frequency, filtered to the engine's bound CPU set.

If the workload is bound to socket 0 (CPUs 0-31 on a dual-socket box)
and we read all 256 logical CPUs, idle cores on socket 1 drag the mean
to half-nominal — the number lies. The collector accepts a CPU-id
filter from ``cpu_binding.expand_thread_binding`` and aggregates only
those.

Three-tier read with fallback (all earned the hard way on R470 GNR):
  1. ``cpufreq/scaling_cur_freq`` — the documented stable interface.
  2. ``cpufreq/cpuinfo_cur_freq`` — when scaling reports 0 / unreadable
     under intel_pstate=active / HWP, this MSR-backed file works.
  3. ``/proc/cpuinfo`` ``cpu MHz`` line — when the entire ``cpufreq/``
     subtree is absent (Ubuntu 6.8 + GNR), the kernel still publishes
     live per-CPU MHz via /proc.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class FrequencyCollector:
    cpu_filter: Optional[set[int]] = None  # None == use all enumerated CPUs
    _CPUFREQ_ROOT: str = "/sys/devices/system/cpu"
    _PROC_CPUINFO: str = "/proc/cpuinfo"

    @property
    def cpufreq_root(self) -> Path:
        return Path(self._CPUFREQ_ROOT)

    @property
    def proc_cpuinfo(self) -> Path:
        return Path(self._PROC_CPUINFO)

    @property
    def available(self) -> bool:
        return self.cpufreq_root.is_dir() or self.proc_cpuinfo.exists()

    def _enumerate_cpus(self) -> list[int]:
        """Return CPU ids to sample, intersected with the filter if any."""
        ids: list[int] = []
        if self.cpufreq_root.is_dir():
            for entry in self.cpufreq_root.iterdir():
                m = re.match(r"^cpu(\d+)$", entry.name)
                if m:
                    ids.append(int(m.group(1)))
        # When sysfs is empty, /proc/cpuinfo is the source of truth.
        if not ids and self.proc_cpuinfo.exists():
            try:
                for line in self.proc_cpuinfo.read_text().splitlines():
                    if line.startswith("processor"):
                        try:
                            ids.append(int(line.split(":", 1)[1].strip()))
                        except ValueError:
                            continue
            except OSError:
                pass
        if self.cpu_filter:
            ids = [i for i in ids if i in self.cpu_filter]
        return sorted(set(ids))

    def _read_sysfs_mhz(self, cpu_id: int) -> float | None:
        """Tier 1+2: scaling_cur_freq, then cpuinfo_cur_freq if zero/missing."""
        base = self.cpufreq_root / f"cpu{cpu_id}" / "cpufreq"
        for fname in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            f = base / fname
            if not f.exists():
                continue
            try:
                khz = int(f.read_text().strip())
            except (OSError, ValueError):
                continue
            if khz > 0:
                return khz / 1000.0  # → MHz
        return None

    def _read_proc_cpuinfo_mhz(self) -> dict[int, float]:
        """Tier 3: parse /proc/cpuinfo for live ``cpu MHz`` per processor."""
        out: dict[int, float] = {}
        try:
            text = self.proc_cpuinfo.read_text()
        except OSError:
            return out
        cur_proc: int | None = None
        for line in text.splitlines():
            if line.startswith("processor"):
                try:
                    cur_proc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    cur_proc = None
            elif line.lower().startswith("cpu mhz") and cur_proc is not None:
                try:
                    mhz = float(line.split(":", 1)[1].strip())
                    out[cur_proc] = mhz
                except ValueError:
                    pass
        return out

    def sample(self) -> tuple[float | None, float | None, float | None]:
        """Return (mean_mhz, stddev_mhz, min_mhz) across the filtered CPUs.

        Any CPU whose frequency can't be read is silently skipped — better
        a slightly-narrower aggregate than a NULL row.
        """
        cpus = self._enumerate_cpus()
        if not cpus:
            return None, None, None
        proc_map = None  # lazy-load tier-3
        values: list[float] = []
        for cpu_id in cpus:
            mhz = self._read_sysfs_mhz(cpu_id)
            if mhz is None:
                if proc_map is None:
                    proc_map = self._read_proc_cpuinfo_mhz()
                mhz = proc_map.get(cpu_id)
            if mhz is not None:
                values.append(mhz)
        if not values:
            return None, None, None
        if len(values) == 1:
            return values[0], 0.0, values[0]
        return statistics.fmean(values), statistics.pstdev(values), min(values)
