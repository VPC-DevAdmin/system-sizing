import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from simulator.infrastructure.dynamo_runtime import commands
from simulator.infrastructure.sessions import finite_replay, verify_tier
from simulator.infrastructure.spec import ExperimentSpec


def test_session_contract_rejects_unmatched_resources():
    s = dict(project_id="P02", min_requests=8)
    with pytest.raises(ValidationError, match="one GPU"):
        ExperimentSpec(**s, gpu_ids=[0, 1])
    with pytest.raises(ValidationError, match="frozen corpus"):
        ExperimentSpec(**s, corpus_size=16)
    with pytest.raises(ValidationError, match="exceed GPU"):
        ExperimentSpec(**s, session_cpu_gb=4)
    with pytest.raises(ValidationError, match="finite resume"):
        ExperimentSpec(project_id="P02")


def test_session_launch_keeps_gpu_budget_and_prefix_cache_matched():
    spec = ExperimentSpec(project_id="P02", min_requests=8, session_tier="cpu_disk")
    plan = dict(
        port=19200,
        model="m",
        served_model_name="m",
        context_limit=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="x",
        gpu_ids=[0],
        sessions={k: v for k, v in spec.model_dump().items() if k.startswith("session_")},
    )
    a, b = [commands(dict(plan, arm=arm))[1] for arm in ("baseline", "treatment")]
    assert "--enable-prefix-caching" in a[1] and "--enable-prefix-caching" in b[1]
    assert "--disaggregation-mode" not in b[1]
    assert a[1] == b[1][:-2]
    assert json.loads(b[1][-1])["kv_connector"] == "DynamoConnector"
    assert b[2]["DYN_KVBM_DISK_CACHE_DIR"].startswith("/tmp/")
    assert "DYN_KVBM_CPU_CACHE_GB" not in a[2]


def test_tier_evidence_requires_requested_reload_and_no_reset(tmp_path):
    before = "kvbm_onboard_blocks_h2d 10\nkvbm_onboard_blocks_d2d 4"
    (tmp_path / "resume-before-kvbm.prom").write_text(before)
    (tmp_path / "resume-after-kvbm.prom").write_text(before.replace("h2d 10", "h2d 20"))
    verify_tier(tmp_path, SimpleNamespace(session_tier="cpu"))
    with pytest.raises(RuntimeError, match="did not reload"):
        verify_tier(tmp_path, SimpleNamespace(session_tier="cpu_disk"))
    (tmp_path / "resume-after-kvbm.prom").write_text(before.replace("h2d 10", "h2d 5"))
    with pytest.raises(RuntimeError, match="reset"):
        verify_tier(tmp_path, SimpleNamespace(session_tier="cpu"))


def test_finite_replay_requests_each_session_exactly_once(monkeypatch, tmp_path):
    from simulator.infrastructure import sessions

    seen = []

    async def request(client, url, model, item, spec, request_id, key):
        await asyncio.sleep(0.001)
        seen.append(item["id"])
        return dict(
            transport_ok=True,
            correct=True,
            elapsed_s=0.1,
            ttft_s=0.01,
            usage={"prompt_tokens": 100},
        )

    monkeypatch.setattr(sessions, "request_one", request)
    engine = SimpleNamespace(replica_urls=["http://test"], api_model_name="m", api_key="EMPTY")
    spec = ExperimentSpec(project_id="P02", min_requests=8, concurrency=4)
    metrics = asyncio.run(
        finite_replay(engine, [{"id": i} for i in range(8)], spec, tmp_path / "rows.jsonl")
    )
    assert sorted(seen) == list(range(8)) and metrics["attempted"] == 8
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 8


def test_resume_latency_gate_is_independent_of_correctness(monkeypatch, tmp_path):
    from simulator.infrastructure import sessions

    async def capture(*args):
        pass

    async def replay(engine, items, spec, path):
        return dict(error_fraction=0, quality_fraction=1, gates_passed=True, ttft_p95_s=6)

    monkeypatch.setattr(sessions, "finite_replay", replay)
    monkeypatch.setattr(sessions, "verify_hot", lambda directory: None)
    engine = SimpleNamespace(arm="baseline", capture_metrics=capture)
    spec = ExperimentSpec(project_id="P02", min_requests=8, session_ttft_slo_s=5)
    result = asyncio.run(
        sessions.measure_sessions(
            engine, [{"id": i, "prompt": "document"} for i in range(24)], spec, tmp_path
        )
    )
    assert not result["gates_passed"] and not result["resume_ttft_gate_passed"]


def test_session_disk_monitor_stops_before_exhaustion(monkeypatch):
    from simulator.config import EngineConfig
    from simulator.infrastructure.sessions import OwnedSessions

    spec = ExperimentSpec(project_id="P02", min_requests=8)
    engine = OwnedSessions(EngineConfig(port=19200), spec, "treatment", "a" * 32)
    engine.storage_root = "/test"
    monkeypatch.setattr(
        "simulator.infrastructure.sessions.shutil.disk_usage",
        lambda p: SimpleNamespace(free=9 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="below 10 GiB"):
        engine.get_metrics()


def test_disk_tier_uses_ephemeral_block_volume_and_readonly_device_metadata(monkeypatch):
    from simulator.config import EngineConfig
    from simulator.infrastructure.dynamo import OwnedDynamo
    from simulator.infrastructure.sessions import OwnedSessions

    spec = ExperimentSpec(project_id="P02", min_requests=8, session_tier="cpu_disk")
    engine = OwnedSessions(
        EngineConfig(port=19200, gpu_image="pinned-image"), spec, "treatment", "a" * 32
    )
    monkeypatch.setattr(
        OwnedDynamo,
        "_build_docker_command",
        lambda e: ["docker", "run", "pinned-image", "/runtime.py"],
    )
    monkeypatch.setattr("simulator.infrastructure.sessions.Path.is_dir", lambda p: True)
    argv = engine._build_docker_command()
    assert argv == [
        "docker",
        "run",
        "-v",
        "/tmp",
        "-v",
        "/run/udev:/run/udev:ro",
        "pinned-image",
        "/runtime.py",
    ]


def test_resume_report_uses_ttft_and_only_enabled_tiers(tmp_path):
    from simulator.infrastructure.report import report_sections

    spec = ExperimentSpec(project_id="P02", min_requests=8)
    manifest = {
        "spec": spec.model_dump(),
        "harness": "session replay",
        "source_result": {"model": "m"},
        "cache_policy": "prime-pressure-resume",
        "created_at": 0,
    }
    state = {
        "results": [
            {
                "block": 0,
                "arm": "baseline",
                "metrics": {
                    "attempted": 8,
                    "quality_fraction": 1,
                    "qualified_per_hour": 100,
                    "latency_p95_s": 9.5,
                    "ttft_p95_s": 0.359,
                    "gates_passed": True,
                },
            }
        ]
    }
    p = tmp_path / "block-00-baseline/replica-0"
    p.mkdir(parents=True)
    (p / "dynamo-plan.json").write_text(
        json.dumps(
            {
                "arm": "baseline",
                "prefill_workers": 1,
                "sessions": {
                    "session_cpu_gb": 32,
                    "session_disk_gb": 64,
                    "session_gpu_gb": 4,
                    "session_tier": "cpu",
                },
            }
        )
    )
    _, _, rows, configs = report_sections(
        manifest,
        state,
        {},
        {"measurement": None, "status": "not_qualified", "reason": "pilot"},
        tmp_path,
    )
    assert rows[0][5] == "0.359"
    assert "host_cache_gb: 0" in configs[0] and "disk_cache_gb: 0" in configs[0]
    assert "prefill_workers" not in configs[0]


def test_hot_diagnostic_rejects_only_common_prefix_hits(tmp_path):
    from simulator.infrastructure.sessions import verify_hot

    before = "vllm:prefix_cache_queries_total 100\nvllm:prefix_cache_hits_total 20"
    (tmp_path / "hot-before-worker-0.prom").write_text(before)
    (tmp_path / "hot-after-worker-0.prom").write_text(
        "vllm:prefix_cache_queries_total 4100\nvllm:prefix_cache_hits_total 100"
    )
    with pytest.raises(RuntimeError, match="90%"):
        verify_hot(tmp_path)
    (tmp_path / "hot-after-worker-0.prom").write_text(
        "vllm:prefix_cache_queries_total 4100\nvllm:prefix_cache_hits_total 3900"
    )
    verify_hot(tmp_path)


def test_slow_resume_is_excluded_from_qualified_throughput():
    from simulator.infrastructure.comparison import compare
    from simulator.infrastructure.driver import summarize

    spec = ExperimentSpec(project_id="P02", min_requests=4, session_count=4, paired_blocks=1)
    rows = [
        dict(transport_ok=True, correct=True, elapsed_s=10, ttft_s=6, usage={}) for _ in range(4)
    ]
    slow = summarize(rows, 40, spec)
    assert slow["correct"] == 4 and slow["qualified"] == 0 and slow["qualified_per_hour"] == 0
    assert not slow["gates_passed"]
    fast = summarize([dict(r, ttft_s=1) for r in rows], 40, spec)
    result = compare(
        [dict(block=0, arm="baseline", metrics=fast), dict(block=0, arm="treatment", metrics=slow)],
        spec,
    )
    assert result["metrics"]["qualified_per_hour"]["change_percent"] == -100
