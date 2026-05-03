"""Package power via Intel RAPL energy counters.

Reads ``/sys/class/powercap/intel-rapl:*/energy_uj`` deltas. Handles
counter rollover via ``max_energy_range_uj``. No-op (with explicit
``not_supported`` status) on hosts without RAPL.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
UTC = timezone.utc


def detect_rapl_energy_path(rapl_root: str | Path = "/sys/class/powercap") -> Path | None:
    root = Path(rapl_root)
    if not root.exists():
        return None
    candidates = sorted(root.rglob("energy_uj"))
    for c in candidates:
        if "intel-rapl" in c.parent.name.lower():
            return c
    return candidates[0] if candidates else None


@dataclass
class PowerSample:
    sampled_at: datetime
    power_w: float


class PowerProbe:
    """Sample package power. Compatible with start()/stop() lifecycle.

    ``stop()`` returns a dict with ``power_w_avg``, ``power_w_peak``,
    ``status``. ``status`` is ``not_supported`` on hosts without RAPL.
    """

    def __init__(self, sample_interval_s: float = 1.0,
                 rapl_root: str | Path = "/sys/class/powercap"):
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        self.sample_interval_s = sample_interval_s
        self.rapl_root = Path(rapl_root)
        self._energy_path = detect_rapl_energy_path(self.rapl_root)
        self._max_range_path = (
            self._energy_path.with_name("max_energy_range_uj")
            if self._energy_path is not None else None
        )
        self._baseline_uj: Optional[int] = None
        self._baseline_t: Optional[datetime] = None
        self._samples: list[PowerSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._supported = self._energy_path is not None

    @property
    def supported(self) -> bool:
        return self._supported

    def start(self) -> None:
        if not self._supported or self._thread is not None:
            return
        self._samples = []
        self._stop_event.clear()
        try:
            self._baseline_uj = self._read_uj()
        except Exception as e:
            log.debug("RAPL initial read failed: %s", e)
            self._supported = False
            return
        self._baseline_t = datetime.now(UTC)
        self._thread = threading.Thread(target=self._loop, name="power-probe", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | str | None]:
        if not self._supported:
            return {"power_w_avg": None, "power_w_peak": None,
                    "status": "not_supported"}
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        # one final delta
        sample = self._compute()
        if sample is not None:
            with self._lock:
                self._samples.append(sample)
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {"power_w_avg": None, "power_w_peak": None,
                    "status": "supported_but_no_samples"}
        vals = [s.power_w for s in samples]
        return {
            "power_w_avg": sum(vals) / len(vals),
            "power_w_peak": max(vals),
            "status": "ok",
        }

    def _loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            sample = self._compute()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)

    def _compute(self) -> PowerSample | None:
        if self._energy_path is None or self._baseline_uj is None or self._baseline_t is None:
            return None
        try:
            cur_uj = self._read_uj()
        except Exception:
            return None
        cur_t = datetime.now(UTC)
        elapsed = (cur_t - self._baseline_t).total_seconds()
        if elapsed <= 0:
            return None
        delta = cur_uj - self._baseline_uj
        if delta < 0:
            max_range = self._read_max_range()
            if max_range is None:
                return None
            delta = (max_range - self._baseline_uj) + cur_uj
        self._baseline_uj = cur_uj
        self._baseline_t = cur_t
        return PowerSample(sampled_at=cur_t, power_w=(delta / 1_000_000.0) / elapsed)

    def _read_uj(self) -> int:
        assert self._energy_path is not None
        return int(self._energy_path.read_text(encoding="utf-8").strip())

    def _read_max_range(self) -> int | None:
        if self._max_range_path is None or not self._max_range_path.exists():
            return None
        try:
            return int(self._max_range_path.read_text(encoding="utf-8").strip())
        except Exception:
            return None
