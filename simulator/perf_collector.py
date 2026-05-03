"""perf-stat collector — system-wide PMU counters with GNR-aware AMX probe.

Critical lessons from prior R470 / Granite Rapids work, baked in:

  * **System-wide attach (`-a`), never `-p <pid>`.** vLLM with TP>1
    spawns worker subprocesses via mp; ``perf stat -p`` does not follow
    forks of arbitrary depth. We were attaching to the FastAPI front-end
    while the actual TMUL kernels ran in unmonitored worker children.
  * **AMX raw-event fallback for GNR.** Intel's perfmon JSON ships
    ``amx_ops.tmul_bf16`` with the SPR encoding (0xCE); on GNR the
    correct event is ``EXE.AMX_BUSY`` at code 0xB7 / umask 0x02.
    We pre-probe ``perf list`` for symbolic names and fall back to a
    raw event spec when the symbolic name is unrecognised.
  * **No `-- sleep` workload.** ``perf stat -a -- sleep 1000000`` is the
    "old" way to keep perf running until SIGINT. On Linux 6.8 + GNR,
    perf's signal-handling fails to clean up the sleep child cleanly;
    SIGINT hangs and the wait timeout kicks in, dropping all output.
    We launch ``perf stat -a -I 1000 -e ...`` with no workload and
    SIGINT on stop — exits cleanly.
  * **Partial-output resilience.** Even when SIGKILL is required, the
    CSV usually has most of the events. Always parse what's on disk.

Output: a dict of canonical PMU keys. Missing events are absent from
the dict, never zero.
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


# ── AMX event candidates ──────────────────────────────────────────────
# Order: most specific symbolic name first; raw fallback last. The raw
# fallback is the GNR EXE.AMX_BUSY encoding, which Intel's perfmon JSON
# does NOT publish under a symbolic name on every kernel.
AMX_CANDIDATE_EVENTS: list[str] = [
    "exe.amx_busy",
    "amx_ops_retired.base",
    "amx_ops_retired",
    "amx_ops.tmul_bf16",
]
AMX_RAW_FALLBACK = "cpu/event=0xB7,umask=0x02,name=exe_amx_busy/"

# ── Default extrapolation events ──────────────────────────────────────
# These are the cheap-to-collect signals that pay back the most on a
# bottleneck-attribution dashboard. Probed via ``perf list`` first; the
# unsupported ones are quietly dropped.
DEFAULT_EVENTS: tuple[str, ...] = (
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "longest_lat_cache.reference",
    "longest_lat_cache.miss",
    "cycle_activity.stalls_mem_any",
    "cycle_activity.stalls_l3_miss",
    "offcore_requests.all_data_rd",
    "offcore_requests.demand_data_rd",
    "mem_load_l3_miss_retired.local_dram",
    "mem_load_l3_miss_retired.remote_dram",
)


# ── perf list capability probe ────────────────────────────────────────


_PERF_LIST_CACHE: str | None = None


def _perf_list_text(perf_path: str, force_refresh: bool = False) -> str:
    """Memoised lower-case ``perf list`` output for capability matching."""
    global _PERF_LIST_CACHE
    if _PERF_LIST_CACHE is not None and not force_refresh:
        return _PERF_LIST_CACHE
    try:
        result = subprocess.run(
            [perf_path, "list"], capture_output=True, text=True, timeout=10
        )
        _PERF_LIST_CACHE = result.stdout.lower()
    except (subprocess.TimeoutExpired, OSError) as e:
        log.debug("perf list failed: %s", e)
        _PERF_LIST_CACHE = ""
    return _PERF_LIST_CACHE


def _probe_raw_event(perf_path: str, raw_spec: str) -> bool:
    """Verify a raw event spec actually counts (rc=0 with no 'not supported')."""
    try:
        r = subprocess.run(
            [perf_path, "stat", "-a", "-e", raw_spec, "--", "sleep", "0"],
            capture_output=True, text=True, timeout=3,
        )
        combined = (r.stdout + r.stderr).lower()
        return r.returncode == 0 and "not supported" not in combined
    except (subprocess.TimeoutExpired, OSError):
        return False


def resolve_events(
    candidates: tuple[str, ...] | list[str],
    perf_path: str,
) -> list[str]:
    """Return only the candidates that appear in ``perf list`` (lowercased)."""
    listing = _perf_list_text(perf_path)
    if not listing:
        return []
    return [c for c in candidates if c.lower() in listing]


def resolve_amx_event(perf_path: str) -> str | None:
    """Pick the best supported AMX event spec for this host.

    Returns:
      * a symbolic name from ``AMX_CANDIDATE_EVENTS`` when ``perf list``
        knows it, OR
      * ``AMX_RAW_FALLBACK`` (the GNR 0xB7/0x02 spec) when the raw probe
        confirms it counts, OR
      * ``None`` when nothing AMX-related is available.
    """
    symbolic = resolve_events(AMX_CANDIDATE_EVENTS, perf_path)
    if symbolic:
        return symbolic[0]
    if _probe_raw_event(perf_path, AMX_RAW_FALLBACK):
        return AMX_RAW_FALLBACK
    return None


# ── Collector ─────────────────────────────────────────────────────────


@dataclass
class PerfStatCollector:
    """System-wide ``perf stat -x,`` collector with AMX capability probing."""

    raw_output_path: Path
    perf_path: str | None = None
    extra_events: tuple[str, ...] = DEFAULT_EVENTS
    include_amx: bool = True
    interval_ms: int = 1000
    popen_factory: Callable | None = None  # test seam
    _process: object = field(default=None, init=False, repr=False)
    _status: str = field(default="not_started", init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _amx_event: str | None = field(default=None, init=False)
    _events: list[str] = field(default_factory=list, init=False)
    _stderr_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def status(self) -> str:
        return self._status

    @property
    def amx_event_name(self) -> str | None:
        """The AMX event name we ended up using (or None)."""
        if self._amx_event is None:
            return None
        # Strip raw-event "cpu/.../" wrapping for storage as the canonical
        # name; readers want "exe_amx_busy" not the raw spec.
        if self._amx_event.startswith("cpu/") and "name=" in self._amx_event:
            for part in self._amx_event.strip("cpu/").rstrip("/").split(","):
                if part.startswith("name="):
                    return part.split("=", 1)[1]
        return self._amx_event

    def start(self) -> None:
        perf = self.perf_path or shutil.which("perf")
        if perf is None:
            self._status = "perf_not_found"
            return

        events: list[str] = list(resolve_events(self.extra_events, perf))
        # 'cycles' / 'instructions' are hardware events not always shown
        # under their own label in `perf list`; include them unconditionally.
        for fallback in ("cycles", "instructions"):
            if fallback not in events and fallback in self.extra_events:
                events.insert(0, fallback)

        if self.include_amx:
            self._amx_event = resolve_amx_event(perf)
            if self._amx_event is not None:
                events.append(self._amx_event)

        if not events:
            self._status = "no_supported_events"
            return
        self._events = events

        self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write events to disk via perf's CSV output. We use stderr→file
        # because `perf stat` writes events to stderr by default.
        cmd = [
            perf, "stat", "-a",
            "-I", str(self.interval_ms),
            "-x,",
            "-o", str(self.raw_output_path),
            "-e", ",".join(events),
        ]
        # NB: no `-- sleep ...` workload — that's the SIGINT-hang trap.

        try:
            popen = self.popen_factory or subprocess.Popen
            self._process = popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            log.warning("perf launch failed: %s", e)
            self._status = "perf_launch_failed"
            return
        self._status = "running"

    def stop(self) -> dict:
        if self._status != "running" or self._process is None:
            return self._parse(default_status=self._status)

        proc = self._process
        self._process = None
        try:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        # Always parse: partial output is still valuable.
        return self._parse(default_status="stopped")

    # ---- parsing ------------------------------------------------------

    def _parse(self, default_status: str) -> dict:
        out: dict = {"status": default_status}
        if not self.raw_output_path.exists():
            return out
        try:
            text = self.raw_output_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return out
        # ``perf stat -x,`` lines after `-I 1000` look like:
        #   <ts>,<value>,<unit>,<event>,<run_time>,<pcnt>,...
        # The non-interval form (no -I) drops the leading ts column.
        agg: dict[str, float] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            # Determine the column indices: heuristic — when first column
            # is a float, it's the timestamp form (-I); when the second
            # column is a float/<not supported>, it's the no-I form.
            has_ts = False
            try:
                float(parts[0])
                has_ts = True
            except ValueError:
                pass
            if has_ts:
                value_str = parts[1].strip()
                event = parts[3].strip() if len(parts) > 3 else ""
            else:
                value_str = parts[0].strip()
                event = parts[2].strip() if len(parts) > 2 else ""
            if not event or value_str in ("", "<not supported>", "<not counted>"):
                continue
            try:
                v = float(value_str.replace(",", ""))
            except ValueError:
                continue
            # Sum across intervals → total counter for the window.
            agg[event] = agg.get(event, 0.0) + v

        # Map raw counters → canonical keys.
        canonical: dict = {"status": "ok" if agg else default_status}
        canonical["pmu_cycles"] = agg.get("cycles")
        canonical["pmu_instructions"] = agg.get("instructions")
        if canonical["pmu_cycles"] and canonical["pmu_instructions"]:
            canonical["pmu_ipc"] = canonical["pmu_instructions"] / canonical["pmu_cycles"]
        canonical["pmu_cache_references"] = agg.get("cache-references")
        canonical["pmu_cache_misses"] = agg.get("cache-misses")
        canonical["pmu_stalls_mem_any"] = agg.get("cycle_activity.stalls_mem_any")
        canonical["pmu_stalls_l3_miss"] = agg.get("cycle_activity.stalls_l3_miss")
        canonical["pmu_llc_reference"] = agg.get("longest_lat_cache.reference")
        canonical["pmu_llc_miss"] = agg.get("longest_lat_cache.miss")
        local = agg.get("mem_load_l3_miss_retired.local_dram")
        remote = agg.get("mem_load_l3_miss_retired.remote_dram")
        if local is not None or remote is not None:
            l = local or 0.0
            r = remote or 0.0
            denom = l + r
            if denom > 0:
                canonical["mem_local_fraction"] = l / denom
                canonical["mem_remote_fraction"] = r / denom

        # AMX: handle both symbolic names and the raw-event renaming.
        amx_total = 0.0
        amx_seen = False
        for event_name, v in agg.items():
            low = event_name.lower()
            if "amx" in low:
                amx_total += v
                amx_seen = True
        if amx_seen:
            canonical["pmu_amx_ops"] = amx_total
            canonical["amx_perf_event_name"] = self.amx_event_name

        # Stall ratio is what the bottleneck-attribution heuristic reads.
        cycles = canonical.get("pmu_cycles")
        stalls_mem = canonical.get("pmu_stalls_mem_any")
        if cycles and stalls_mem and cycles > 0:
            canonical["pmu_stall_mem_ratio"] = stalls_mem / cycles

        return canonical
