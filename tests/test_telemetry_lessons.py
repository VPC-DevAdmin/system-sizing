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
from simulator.cpu_binding import (
    derive_sglang_thread_binding,
    expand_thread_binding,
    flatten_for_taskset,
)
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


def test_vllm_dual_socket_config_loads_with_replicas() -> None:
    """The R7735 dual-socket config has a replicas list — yaml gives us
    list[dict], the loader must convert to ReplicaConfig instances or
    the engine sees garbage."""
    from simulator.config import load_config, ReplicaConfig
    cfg = load_config("config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml")
    assert cfg.engine.type == "vllm_dual_socket"
    assert len(cfg.engine.replicas) == 2
    assert all(isinstance(r, ReplicaConfig) for r in cfg.engine.replicas)
    r0, r1 = cfg.engine.replicas
    # Pinning must NOT include SMT siblings (cores 64-127 on this box).
    assert r0.cpuset_cpus == "0-31"
    assert r0.cpuset_mems == "0"
    assert r1.cpuset_cpus == "32-63"
    assert r1.cpuset_mems == "1"
    # Per-replica OMP env must mirror the cpuset exactly (mismatch
    # silently kills BF16 throughput).
    assert r0.env["VLLM_CPU_OMP_THREADS_BIND"] == "0-31"
    assert r1.env["VLLM_CPU_OMP_THREADS_BIND"] == "32-63"


def test_vllm_dual_socket_engine_command_shape() -> None:
    """Exercise the per-replica docker run command construction without
    actually launching containers — pin the flags that the working
    runbook proved out."""
    from simulator.config import load_config
    from simulator.engines.vllm_dual_socket import VllmDualSocketEngine

    cfg = load_config("config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml")
    eng = VllmDualSocketEngine(cfg.engine)
    r0 = cfg.engine.replicas[0]
    cmd = eng._build_replica_command(r0, "test-r0")

    # Critical NUMA + security flags from the runbook
    assert "--cpuset-cpus" in cmd
    assert "0-31" in cmd
    assert "--cpuset-mems" in cmd
    assert "0" in cmd
    assert "--security-opt" in cmd
    assert "seccomp=unconfined" in cmd
    assert "--cap-add" in cmd
    assert "SYS_NICE" in cmd
    # vLLM-specific env flow-through. KV pool size is optimizer-validated
    # at 120 GB per replica (resolves the c=16 long-prompt 900 s tail);
    # see the YAML's optimizer-findings header for the rationale.
    assert any("VLLM_CPU_KVCACHE_SPACE=120" in s for s in cmd)
    assert any("VLLM_CPU_OMP_THREADS_BIND=0-31" in s for s in cmd)
    # vLLM serve flags
    assert "--dtype" in cmd and "bfloat16" in cmd
    assert "--max-model-len" in cmd and "8192" in cmd
    assert "--port" in cmd and "8000" in cmd
    assert "--served-model-name" in cmd and "qwen3_30b_a3b" in cmd


def test_intel_gemma4_single_replica_command_shape() -> None:
    """Single-socket Intel Xeon variant: one replica, all 64 cores, no
    NUMA membership pin (cpuset_mems blank). Pins the hand-validated
    docker invocation for Gemma 4 26B-A4B on Xeon 6761P."""
    from simulator.config import load_config
    from simulator.engines.vllm_dual_socket import VllmDualSocketEngine

    cfg = load_config("config/xeon_vllm_gemma4_26b_a4b.yaml")
    assert len(cfg.engine.replicas) == 1, (
        "Single-socket profile should declare exactly one replica"
    )

    eng = VllmDualSocketEngine(cfg.engine)
    cmd = eng._build_replica_command(cfg.engine.replicas[0], "test-r0")

    # All 64 cores, single replica.
    assert "--cpuset-cpus" in cmd and "0-63" in cmd
    # No NUMA-mems pin on single socket — flag must be ABSENT.
    assert "--cpuset-mems" not in cmd
    # vLLM env flow-through.
    assert any("VLLM_CPU_KVCACHE_SPACE=64" in s for s in cmd)
    assert any("VLLM_CPU_OMP_THREADS_BIND=0-63" in s for s in cmd)
    assert any("OMP_NUM_THREADS=64" in s for s in cmd)
    # Model + name match the Gemma 4 hand-validated invocation.
    assert "/models/gemma-4-26B-A4B-it" in cmd
    assert "gemma4_26b_a4b" in cmd
    assert "--dtype" in cmd and "bfloat16" in cmd
    assert "--port" in cmd and "8000" in cmd


def test_optimizer_profile_registry_has_intel_gemma4() -> None:
    """The engine_optimizer profile registry must expose the Intel
    Gemma 4 profile alongside the original AMD one. Validates the
    profile-level invariants (no NUMA mems pin since the host is
    single-socket) and the priority-list contents (headline shapes
    present, AMD-specific block_32 absent)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    eo = importlib.import_module("engine_optimizer")

    assert "intel_gemma4" in eo.PROFILES
    assert "amd_dual_socket" in eo.PROFILES
    assert eo.DEFAULT_PROFILE == "amd_dual_socket"  # backwards compat

    intel_configs = eo.PROFILES["intel_gemma4"]
    assert intel_configs, "intel_gemma4 profile must contain configs"

    # Single-socket invariant: NO replica anywhere in the Intel profile
    # may pin cpuset_mems — the host has only one NUMA node.
    for c in intel_configs:
        for r in c.replicas:
            assert r.cpuset_mems is None, (
                f"intel_gemma4/{c.name}/{r.name} must not pin NUMA mems "
                f"on single-socket Xeon (cpuset_mems={r.cpuset_mems!r})"
            )

    by_name = {c.name: c for c in intel_configs}

    # Headline shapes (★★★) — these are the primary axes of variation.
    assert "baseline" in by_name
    assert "dual_replica_32c" in by_name
    assert "tp2" in by_name
    assert "tp4" in by_name
    # dual_replica_32c is the only Intel config with two replicas;
    # everything else is single-replica.
    assert len(by_name["dual_replica_32c"].replicas) == 2
    assert {r.cpuset_cpus for r in by_name["dual_replica_32c"].replicas} == {
        "0-31", "32-63",
    }

    # Primary expected-helpful (★★).
    assert "kv_xl_96" in by_name
    assert "chunked_prefill" in by_name

    # Secondary diagnostics (★).
    assert "batch_budget" in by_name
    assert "batch_8192" in by_name
    assert "no_compile" in by_name
    assert "kv_xl_128" in by_name
    assert "kv_xl_chunked_prefill" in by_name

    # AMD-specific configs that should NOT be in the Intel profile.
    assert "block_32" not in by_name, (
        "block_32 was the AMD low-concurrency winner; not motivated for "
        "Granite Rapids — should not appear in intel_gemma4 profile"
    )


def test_optimizer_profile_registry_has_amd_gemma4() -> None:
    """The amd_gemma4 profile mirrors intel_gemma4's priority structure
    on the AMD dual-socket shape, dropping configs Qwen3 already proved
    are model-agnostic failures (tp2_cross_socket Gloo init, kv_xl 160
    GB startup) and keeping the AMD-specific block_32 (was the Qwen3
    low-concurrency winner)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    eo = importlib.import_module("engine_optimizer")

    assert "amd_gemma4" in eo.PROFILES
    amd_configs = eo.PROFILES["amd_gemma4"]
    assert amd_configs, "amd_gemma4 profile must contain configs"

    by_name = {c.name: c for c in amd_configs}

    # Headline shapes (★★★).
    for expected in ("baseline", "single_replica_64c", "kv_xl_140"):
        assert expected in by_name, f"missing headline shape: {expected}"
    # baseline is the dual-replica NUMA-pinned production shape.
    base = by_name["baseline"]
    assert len(base.replicas) == 2
    assert {r.cpuset_cpus for r in base.replicas} == {"0-31", "32-63"}
    assert {r.cpuset_mems for r in base.replicas} == {"0", "1"}, (
        "baseline must NUMA-pin replicas — it's the dual-socket AMD shape"
    )
    # baseline KV is the Qwen3-validated 120 GB.
    for r in base.replicas:
        assert r.env["VLLM_CPU_KVCACHE_SPACE"] == "120"

    # single_replica_64c covers all cores with NO NUMA pin.
    sr = by_name["single_replica_64c"]
    assert len(sr.replicas) == 1
    assert sr.replicas[0].cpuset_cpus == "0-63"
    assert sr.replicas[0].cpuset_mems is None

    # Primary expected-helpful (★★) and secondary diagnostics (★).
    for expected in (
        "chunked_prefill", "block_32",
        "batch_budget", "batch_8192", "no_compile",
        "kv_smaller_80", "kv_xl_chunked_prefill",
    ):
        assert expected in by_name, f"missing config: {expected}"

    # block_32 must remain — it was the Qwen3 AMD winner. Distinct from
    # the intel_gemma4 profile, which drops it.
    assert "block_32" in by_name, (
        "block_32 was the Qwen3 AMD low-concurrency winner; keep it in "
        "amd_gemma4 to test transfer to Gemma."
    )

    # Configs that must be ABSENT — Qwen3 proved these are AMD-stack
    # failures, not model-specific.
    assert "tp2_cross_socket" not in by_name, (
        "tp2_cross_socket failed Gloo distributed-init on Qwen3 — "
        "model-agnostic infrastructure issue. Don't waste cycles "
        "retesting on Gemma."
    )
    assert "tp2_chunked_prefill" not in by_name, (
        "Same Gloo init failure as tp2_cross_socket."
    )


def test_amd_gemma4_yaml_loads_and_builds_command() -> None:
    """The Gemma 4 AMD run-sweep config builds the same dual-replica
    NUMA-pinned shape as the Qwen3 config, but with the Gemma model."""
    from simulator.config import load_config
    from simulator.engines.vllm_dual_socket import VllmDualSocketEngine

    cfg = load_config("config/r7735_vllm_dual_socket_gemma4_26b_a4b.yaml")
    assert cfg.engine.type == "vllm_dual_socket"
    assert cfg.engine.model_id == "google/gemma-4-26B-A4B-it"
    assert cfg.engine.served_model_name == "gemma4_26b_a4b"
    assert len(cfg.engine.replicas) == 2

    eng = VllmDualSocketEngine(cfg.engine)
    cmd = eng._build_replica_command(cfg.engine.replicas[0], "test-r0")

    # NUMA pinning matches the dual-socket production shape.
    assert "--cpuset-cpus" in cmd and "0-31" in cmd
    assert "--cpuset-mems" in cmd and "0" in cmd
    # KV pool inherits the Qwen3-validated 120 GB.
    assert any("VLLM_CPU_KVCACHE_SPACE=120" in s for s in cmd)
    # Block-size 32 carries forward from the Qwen3 production knobs.
    assert "--block-size" in cmd and "32" in cmd
    # Gemma model + name.
    assert "/models/gemma-4-26B-A4B-it" in cmd
    assert "gemma4_26b_a4b" in cmd


def test_engine_api_model_name_uses_served_name_when_set() -> None:
    """Real bug from the AMD R7735 dual_socket first-light run: the
    simulator was sending ``Qwen/Qwen3-30B-A3B-Instruct-2507`` as the
    OpenAI ``model`` field, but vLLM was launched with
    ``--served-model-name qwen3_30b_a3b`` and only registered that
    name. vLLM rejected every request with 404.

    Fix: ``Engine.api_model_name`` returns ``served_model_name`` when
    set, falling back to the HF id when absent. Vanilla single-engine
    configs that don't set served_model_name keep working unchanged.
    """
    from simulator.config import load_config
    from simulator.engines.vllm_dual_socket import VllmDualSocketEngine

    cfg = load_config("config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml")
    eng = VllmDualSocketEngine(cfg.engine)
    # served_model_name is set in this config — that's the name vLLM
    # registers under, and it's what the API call must use.
    assert eng.api_model_name == "qwen3_30b_a3b"
    # The HF id is preserved for tokenizer corpus / run metadata.
    assert eng.model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_engine_api_model_name_falls_back_to_hf_id_when_unset() -> None:
    """For configs that don't set served_model_name (vanilla vLLM,
    sglang without explicit naming), api_model_name == model_id."""
    from simulator.config import Config
    from simulator.engines.base import Engine

    cfg = Config()
    cfg.engine.type = "vllm"
    cfg.engine.model_id = "Qwen/Qwen2.5-7B-Instruct"
    cfg.engine.served_model_name = None
    # Bare Engine() works for property checks without launching anything.
    e = Engine.__new__(Engine)
    e.cfg = cfg.engine
    e._proc = None
    e._log_file = None
    e._log_path = None
    assert e.api_model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert e.model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_vllm_dual_socket_exposes_per_replica_urls() -> None:
    """The simulator's runner reads ``engine.replica_urls`` and builds
    one AsyncOpenAI client per replica. For dual-socket NUMA-pinned
    vLLM, that's two URLs at the per-replica ports — no proxy in
    between."""
    from simulator.config import load_config
    from simulator.engines.vllm_dual_socket import VllmDualSocketEngine

    cfg = load_config("config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml")
    eng = VllmDualSocketEngine(cfg.engine)
    assert eng.replica_urls == [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8001/v1",
    ]
    # api_key is "EMPTY" by default — direct vLLM doesn't auth.
    assert eng.api_key == "EMPTY"


def test_pool_manager_load_balances_users_with_sticky_assignment() -> None:
    """Per the runbook: load-balanced sticky assignment guarantees ±1
    distribution across replicas. Verify both properties: balance and
    stickiness (same user_id → same replica every time)."""
    from openai import AsyncOpenAI
    from simulator.personas import COHORTS
    from simulator.pool_manager import PoolManager
    from simulator.tokenizer_corpus import TokenCorpus
    from simulator.virtual_user import SharedState

    # Build a PoolManager with 2 stub clients (we only exercise the
    # client-routing logic — never actually fire requests).
    clients = [
        AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x")
        for p in (8000, 8001)
    ]
    pool = PoolManager(
        cohort=COHORTS["chat_heavy"],
        clients=clients,
        model_id="test",
        corpus=TokenCorpus("test"),
        state=SharedState(),
        request_timeout_s=10,
    )
    # Assign 1000 unique user_ids; counts should differ by at most 1.
    assignments: dict[str, object] = {}
    for i in range(1000):
        uid = f"user-{i}"
        assignments[uid] = pool._client_for_user(uid)
    # Stickiness: re-asking for the same user returns the same client.
    for uid, expected in assignments.items():
        assert pool._client_for_user(uid) is expected
    # Balance: the explicit counter scheme guarantees ±1, not just
    # statistical balance.
    counts = pool._user_replica_counts
    assert max(counts) - min(counts) <= 1, f"unbalanced: {counts}"
    assert sum(counts) == 1000


def test_pool_manager_single_replica_no_routing_overhead() -> None:
    """For single-backend engines (vllm direct, sglang), the router
    should short-circuit to the only client and never grow the
    assignment map."""
    from openai import AsyncOpenAI
    from simulator.personas import COHORTS
    from simulator.pool_manager import PoolManager
    from simulator.tokenizer_corpus import TokenCorpus
    from simulator.virtual_user import SharedState

    only = AsyncOpenAI(base_url="http://127.0.0.1:30000/v1", api_key="x")
    pool = PoolManager(
        cohort=COHORTS["chat_heavy"],
        clients=[only],
        model_id="test",
        corpus=TokenCorpus("test"),
        state=SharedState(),
        request_timeout_s=10,
    )
    for i in range(100):
        assert pool._client_for_user(f"u-{i}") is only
    # No need to track assignments for single-replica engines.
    assert pool._user_replica_assignments == {}


def test_flatten_for_taskset_returns_none_for_empty_input() -> None:
    """Caller treats None as 'don't wrap with taskset'."""
    assert flatten_for_taskset(None) is None
    assert flatten_for_taskset("") is None


# ── SGLang TP-aligned thread binding ──────────────────────────────────


def test_sglang_bind_passes_through_when_tp_matches_groups() -> None:
    """An explicit TP-grouped string is honoured as-is."""
    assert derive_sglang_thread_binding("0-15|16-31|32-47|48-63", 4) \
           == "0-15|16-31|32-47|48-63"


def test_sglang_bind_splits_single_range_when_divisible() -> None:
    """A single contiguous range is split evenly into TP groups."""
    assert derive_sglang_thread_binding("0-31", 4) == "0-7|8-15|16-23|24-31"
    assert derive_sglang_thread_binding("0-63", 8) == \
           "0-7|8-15|16-23|24-31|32-39|40-47|48-55|56-63"


def test_sglang_bind_tp1_with_single_range_stays_single_range() -> None:
    assert derive_sglang_thread_binding("0-31", 1) == "0-31"


def test_sglang_bind_tp1_with_no_bind_returns_empty() -> None:
    """TP=1 doesn't actually require the env var — the SGLang assert
    only trips for TP>1."""
    assert derive_sglang_thread_binding(None, 1) == ""
    assert derive_sglang_thread_binding("", 1) == ""


def test_sglang_bind_raises_when_groups_disagree_with_tp() -> None:
    """SGLang's internal assert says TP must equal len(env.split('|')).
    We surface the misconfig at launch time instead."""
    with pytest.raises(ValueError, match="2 groups but tp=4"):
        derive_sglang_thread_binding("0-15|16-31", 4)


def test_sglang_bind_raises_when_range_not_divisible() -> None:
    """30 cores can't split evenly across TP=4."""
    with pytest.raises(ValueError, match="not divisible"):
        derive_sglang_thread_binding("0-29", 4)


def test_sglang_bind_raises_when_tp_gt_1_without_bind() -> None:
    with pytest.raises(ValueError, match="cpu_bind is required"):
        derive_sglang_thread_binding(None, 4)


def test_sglang_bind_raises_for_non_contiguous_single_group() -> None:
    """If you have a sparse CPU set, encode the |-grouping yourself —
    the auto-split only handles contiguous ranges."""
    with pytest.raises(ValueError, match="non-contiguous"):
        derive_sglang_thread_binding("0,1,2,8,9,10", 2)


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
