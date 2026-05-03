"""Pin the GNR / R470 telemetry lessons learned the hard way.

These tests don't exercise live perf — they pin behaviour we MUST keep
across refactors so we never re-discover the same telemetry bugs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running pytest from the repo root without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.amx_utilization import (
    AmxUtilization,
    _parse_line,
    aggregate,
    _Dispatch,
    parse_amx_utilization,
)
from simulator.bandwidth import (
    BandwidthCollector,
    _discover_imc_events,
    bandwidth_summary,
)
from simulator.cpu_binding import expand_thread_binding, flatten_for_taskset
from simulator.frequency import FrequencyCollector
from simulator.perf_collector import (
    AMX_CANDIDATE_EVENTS,
    AMX_RAW_FALLBACK,
)


# ── CPU binding ────────────────────────────────────────────────────────


def test_expand_binding_handles_pipe_separated_groups() -> None:
    """The vLLM ``VLLM_CPU_OMP_THREADS_BIND`` shape: per-worker groups
    joined by ``|``. The frequency collector treats them as one set."""
    out = expand_thread_binding("0-15|16-31|32-47|48-63")
    assert out == set(range(0, 64))


def test_expand_binding_handles_comma_within_group() -> None:
    """Interleaved layouts like ``0-7,16-23|8-15,24-31`` are a real
    vLLM CPU-NUMA thing — must merge cleanly."""
    out = expand_thread_binding("0-7,16-23|8-15,24-31")
    assert out == set(range(0, 32))


def test_expand_binding_empty_or_garbage_returns_empty_set() -> None:
    assert expand_thread_binding(None) == set()
    assert expand_thread_binding("") == set()
    # Empty set means "no filter, use all CPUs" downstream, not crash.
    assert expand_thread_binding("not,parseable,values") == set()


def test_flatten_for_taskset_collapses_contiguous_groups() -> None:
    """SGLang has no native bind-string flag, so we wrap launch with
    ``taskset -c <list>``. The flat form must compact contiguous CPUs
    back into a range — keeps the command line readable."""
    assert flatten_for_taskset("0-7|8-15|16-23|24-31") == "0-31"


def test_flatten_for_taskset_keeps_disjoint_runs_separate() -> None:
    """Interleaved binding stays interleaved: e.g. socket-0-only
    physical cores when SMT siblings are excluded."""
    assert flatten_for_taskset("0-3|8-11") == "0-3,8-11"


def test_flatten_for_taskset_returns_none_for_empty_input() -> None:
    """Caller treats None as 'don't wrap with taskset'."""
    assert flatten_for_taskset(None) is None
    assert flatten_for_taskset("") is None


# ── Bandwidth: per-IMC discovery on GNR ───────────────────────────────


def test_discover_imc_events_on_gnr_layout(tmp_path) -> None:
    """GNR exposes ``uncore_imc_0`` ... ``uncore_imc_N`` per memory
    controller. We MUST discover them and emit per-PMU event names —
    the bare ``uncore_imc/`` alias resolves to ZERO PMUs on GNR (silent
    zero-output failure)."""
    fake = tmp_path / "devices"
    fake.mkdir()
    for n in ["uncore_imc_0", "uncore_imc_1", "uncore_imc_2",
              "uncore_imc",  # legacy alias — must skip
              "uncore_imc_free_running_0",  # different event set — skip
              "cpu"]:
        (fake / n).mkdir()
    reads, writes = _discover_imc_events(root=fake)
    assert reads == [
        "uncore_imc_0/cas_count_read/",
        "uncore_imc_1/cas_count_read/",
        "uncore_imc_2/cas_count_read/",
    ]
    assert writes == [
        "uncore_imc_0/cas_count_write/",
        "uncore_imc_1/cas_count_write/",
        "uncore_imc_2/cas_count_write/",
    ]


def test_discover_imc_events_returns_none_on_spr_legacy_only(tmp_path) -> None:
    """SPR / EMR have only the aggregate ``uncore_imc`` alias. Discovery
    returns None and the collector falls back to the legacy event spec."""
    fake = tmp_path / "devices"
    fake.mkdir()
    (fake / "uncore_imc").mkdir()
    (fake / "cpu").mkdir()
    assert _discover_imc_events(root=fake) is None


# ── Bandwidth: parse modern perf output ───────────────────────────────


def test_bandwidth_parser_sums_across_imc_pmus() -> None:
    """Multi-IMC GNR hosts emit one CSV line per PMU per timestamp.
    We MUST sum, not overwrite — the overwrite bug reported ~8% of
    actual on a 12-IMC chip."""
    c = BandwidthCollector(interval_ms=1000)
    c._raw_lines = [
        "1.000000,1000000,,uncore_imc_0/cas_count_read/,100.00,\n",
        "1.000000,1000000,,uncore_imc_1/cas_count_read/,100.00,\n",
        "1.000000,1000000,,uncore_imc_2/cas_count_read/,100.00,\n",
        "1.000000,500000,,uncore_imc_0/cas_count_write/,100.00,\n",
        "1.000000,500000,,uncore_imc_1/cas_count_write/,100.00,\n",
        "1.000000,500000,,uncore_imc_2/cas_count_write/,100.00,\n",
    ]
    c._parse_raw_output()
    assert len(c._samples) == 1
    # 3 × 1_000_000 CAS × 64 bytes / 1.0s / 1e9 = 0.192 GB/s
    assert c._samples[0].read_gb_s == pytest.approx(0.192, abs=1e-6)
    assert c._samples[0].write_gb_s == pytest.approx(0.096, abs=1e-6)


def test_bandwidth_parser_handles_modern_perf_scaled_mib() -> None:
    """Linux 6.8 + perf 6.8 emit ``238.47 MiB`` instead of raw CAS counts.
    The old isdigit() filter dropped every such line — null bandwidth
    on every row. Float parsing + unit-scaling is the fix."""
    c = BandwidthCollector(interval_ms=1000)
    c._raw_lines = [
        "1.000604066,238.47,MiB,uncore_imc/cas_count_read/,8009419950,100.00,,\n",
        "1.000604066,207.08,MiB,uncore_imc/cas_count_write/,8007561959,100.00,,\n",
    ]
    c._parse_raw_output()
    assert len(c._samples) == 1
    # 238.47 MiB = ~250 MB → 0.25 GB/s
    assert c._samples[0].read_gb_s == pytest.approx(0.2501, abs=0.001)
    assert c._samples[0].write_gb_s == pytest.approx(0.2172, abs=0.001)


def test_bandwidth_parser_skips_unknown_units() -> None:
    """Better a low-BW row than a silently wrong one."""
    c = BandwidthCollector(interval_ms=1000)
    c._raw_lines = [
        "1.000000,123,bogounit,uncore_imc/cas_count_read/,1e9,100.00,,\n",
        "1.000000,238.47,MiB,uncore_imc/cas_count_read/,1e9,100.00,,\n",
    ]
    c._parse_raw_output()
    assert len(c._samples) == 1
    assert c._samples[0].read_gb_s == pytest.approx(0.2501, abs=0.001)


def test_bandwidth_summary_handles_empty_input() -> None:
    out = bandwidth_summary([])
    assert out["memory_bw_read_gb_s_avg"] is None


# ── Frequency: bound-CPU filter, tiered fallback ──────────────────────


def test_frequency_filters_to_bound_cpus(tmp_path, monkeypatch) -> None:
    """When the workload is bound to socket 0 (CPUs 0-31), frequency
    aggregation MUST exclude idle CPUs from the other socket — those
    drag the mean to half-nominal."""
    root = tmp_path / "cpu"
    for cpu_id, khz in [
        (0, 3_000_000), (1, 3_000_000), (2, 3_000_000), (3, 3_000_000),
        (4, 800_000),   (5, 800_000),   (6, 800_000),   (7, 800_000),  # idle other-socket
    ]:
        (root / f"cpu{cpu_id}" / "cpufreq").mkdir(parents=True)
        (root / f"cpu{cpu_id}" / "cpufreq" / "scaling_cur_freq").write_text(str(khz))

    fc = FrequencyCollector(cpu_filter={0, 1, 2, 3})
    fc._CPUFREQ_ROOT = str(root)
    mean, _, min_mhz = fc.sample()
    # Filtered: 3000 MHz across all four bound CPUs.
    assert mean == pytest.approx(3000.0)
    assert min_mhz == pytest.approx(3000.0)


def test_frequency_falls_back_to_cpuinfo_cur_freq(tmp_path) -> None:
    """On HWP-active GNR, ``scaling_cur_freq`` reads zero. The MSR-backed
    ``cpuinfo_cur_freq`` works. (R470 production observation.)"""
    root = tmp_path / "cpu"
    (root / "cpu0" / "cpufreq").mkdir(parents=True)
    (root / "cpu0" / "cpufreq" / "scaling_cur_freq").write_text("0")
    (root / "cpu0" / "cpufreq" / "cpuinfo_cur_freq").write_text("2900000")
    fc = FrequencyCollector(cpu_filter={0})
    fc._CPUFREQ_ROOT = str(root)
    mean, _, _ = fc.sample()
    assert mean == pytest.approx(2900.0)


def test_frequency_falls_back_to_proc_cpuinfo(tmp_path) -> None:
    """When the entire ``cpufreq/`` subtree is missing (Ubuntu 6.8 + GNR),
    the kernel still publishes per-CPU ``cpu MHz`` via /proc/cpuinfo."""
    root = tmp_path / "cpu"
    for cpu_id in range(2):
        (root / f"cpu{cpu_id}").mkdir(parents=True)  # no cpufreq subdir
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\ncpu MHz\t\t: 2900.0\n\n"
        "processor\t: 1\ncpu MHz\t\t: 2700.0\n"
    )
    fc = FrequencyCollector(cpu_filter={0, 1})
    fc._CPUFREQ_ROOT = str(root)
    fc._PROC_CPUINFO = str(cpuinfo)
    mean, _, min_mhz = fc.sample()
    assert mean == pytest.approx(2800.0)
    assert min_mhz == pytest.approx(2700.0)


# ── AMX events: GNR uses 0xB7/0x02, NOT the SPR 0xCE encoding ─────────


def test_amx_raw_fallback_uses_gnr_encoding() -> None:
    """The Intel perfmon JSON ships ``amx_ops.tmul_bf16`` with the SPR
    encoding (0xCE); on GNR that silently counts ZERO. The raw fallback
    spec is event=0xB7, umask=0x02 (EXE.AMX_BUSY)."""
    assert "event=0xB7" in AMX_RAW_FALLBACK
    assert "umask=0x02" in AMX_RAW_FALLBACK
    # The raw spec must NOT include the SPR-only 0xCE encoding.
    assert "0xCE" not in AMX_RAW_FALLBACK


def test_amx_candidates_include_gnr_busy_event_first() -> None:
    """``exe.amx_busy`` is the GNR symbolic name; if it's published in
    perf list we should pick it before the legacy candidates."""
    assert AMX_CANDIDATE_EVENTS[0] == "exe.amx_busy"


# ── AMX dispatch parsing: oneDNN verbose v0 / v1 schemas ──────────────


def test_amx_parser_v1_classifies_amx_kernel() -> None:
    line = (
        "onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx,"
        "undef,src:bf16::blocked:ab::f0,attr-scratchpad:user,,128x3584:3584x37888,1.18311"
    )
    ev = _parse_line(line)
    assert ev is not None
    assert ev.is_amx is True
    assert ev.time_ms == 1.18311


def test_amx_parser_distinguishes_avx512_core_from_avx512_core_amx() -> None:
    """The AMX regex must require an explicit ``amx`` substring — plain
    ``avx512_core`` is non-AMX."""
    line = (
        "onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx512_core,"
        "undef,src:f32,attr-scratchpad:user,,128x128,0.2"
    )
    ev = _parse_line(line)
    assert ev is not None and ev.is_amx is False


def test_amx_aggregate_distinguishes_empty_from_zero_fraction() -> None:
    """Empty log (verbose disabled) → fraction None.
    Log with only non-AMX dispatches → fraction 0.0.
    These are different signals; the verdict layer needs to tell them apart."""
    assert aggregate([]).onednn_amx_time_fraction is None
    only_non_amx = aggregate([
        _Dispatch("matmul", "brg_matmul:avx512_core_bf16", "x", 1.0, False)
    ])
    assert only_non_amx.onednn_amx_time_fraction == 0.0


def test_amx_aggregate_top_shapes_ranked_by_total_ms() -> None:
    events = [
        _Dispatch("matmul", "brg_matmul:avx10_1_512_amx", "hot", 0.1, True),
        _Dispatch("matmul", "brg_matmul:avx10_1_512_amx", "hot", 0.2, True),
        _Dispatch("matmul", "brg_matmul:avx10_1_512_amx", "cold", 1.0, True),
    ]
    out = aggregate(events)
    assert out.onednn_matmul_time_by_shape[0]["shape"] == "cold"
    assert out.onednn_matmul_time_by_shape[1]["shape"] == "hot"
    assert out.onednn_matmul_time_by_shape[1]["calls"] == 2
