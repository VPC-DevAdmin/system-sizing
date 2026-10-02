import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from simulator.infrastructure import session_capacity as capacity
from simulator.infrastructure.dynamo_runtime import commands
from simulator.infrastructure.session_capacity_worker import SessionReplicas, cache_evidence
from simulator.infrastructure.spec import ExperimentSpec


def spec(**kw):
    args = dict(
        project_id="P02",
        session_mode="capacity",
        session_gpu_gb=None,
        duration_s=300,
        warmup_s=0,
        min_requests=2,
        paired_blocks=1,
        session_population_grid=[4, 8],
        session_rate_grid=[0.02, 0.04],
        document_records=16,
        gpu_ids=[0, 4],
    )
    return ExperimentSpec(**(args | kw))


def test_capacity_contract():
    assert spec()
    for kw in [
        dict(session_gpu_gb=4),
        dict(session_tier="cpu_disk"),
        dict(duration_s=60),
        dict(session_population_grid=[8, 4]),
        dict(session_population_grid=[8]),
        dict(session_rate_grid=[float("nan")]),
        dict(concurrency=4096, session_max_pending=16),
    ]:
        with pytest.raises(ValidationError):
            spec(**kw)


def test_trace_is_frozen_with_natural_replacement_and_unique_prefixes():
    s = spec().model_copy(update={"session_turns": 2})
    trace = capacity.make_trace(s, 4, 2, 0)
    assert trace == capacity.make_trace(s, 4, 2, 0)
    assert trace != capacity.make_trace(s, 4, 2, 1)
    assert all(a["at_s"] < b["at_s"] for a, b in zip(trace, trace[1:], strict=False))
    assert trace[-1]["at_s"] < s.duration_s
    assert any(e["generation"] > 0 for e in trace)
    a = capacity.document_item(s, dict(slot=0, generation=0, turn=0), 0)
    b = capacity.document_item(s, dict(slot=0, generation=0, turn=1), 0)
    c = capacity.document_item(s, dict(slot=1, generation=0, turn=0), 0)
    assert a["prompt"].split("\nQuestion")[0] == b["prompt"].split("\nQuestion")[0]
    assert a["prompt"].split("\nQuestion")[0] != c["prompt"].split("\nQuestion")[0]
    assert len(a["expected"]) == s.session_answer_records
    for row in a["expected"]:
        assert json.dumps(row) in a["prompt"]


def cells(s):
    return [
        dict(
            block=0,
            arm=arm,
            population=pop,
            per_session_rate=rate,
            rate=pop * rate,
            metrics=dict(
                stable_capacity=pop == 4,
                overload_observed=pop == 8,
                client_limited=False,
                quality_fraction=1,
                trace_sha256=f"{rate}-{pop}",
                cpu_reload_observed=arm == "treatment",
            ),
        )
        for rate in s.session_rate_grid
        for pop in s.session_population_grid
        for arm in ("baseline", "treatment")
    ]


def test_capacity_requires_complete_matched_boundaries_and_reload():
    s = spec()
    rows = cells(s)

    def first():
        return capacity.compare_capacity(rows, s)["by_cadence"][0]["blocks"][0]

    assert first()["comparison_ready"]
    assert first()["exploratory_capacity_uplift_percent"] is None
    assert not capacity.compare_capacity(rows, s)["qualified"]
    rows[1]["metrics"]["cpu_reload_observed"] = False
    assert not first()["comparison_ready"]
    rows[1]["metrics"]["cpu_reload_observed"] = True
    rows[2]["metrics"]["client_limited"] = True
    assert not first()["arms"]["baseline"]["boundary_resolved"]
    rows[2]["metrics"]["client_limited"] = False
    rows[0]["metrics"]["trace_sha256"] = "different"
    assert first()["exploratory_capacity_uplift_percent"] is None
    rows.pop(0)
    assert not first()["arms"]["baseline"]["complete"]


def test_highest_passing_population_is_not_a_maximum():
    s = spec()
    rows = cells(s)
    for r in rows:
        r["metrics"].update(stable_capacity=True, overload_observed=False)
    block = capacity.compare_capacity(rows, s)["by_cadence"][0]["blocks"][0]
    assert block["arms"]["baseline"]["best"]["population"] == 8
    assert not block["comparison_ready"]


def test_runtime_keeps_prefix_caching_without_artificial_gpu_cap():
    s = spec()
    plan = dict(
        port=19200,
        model="m",
        served_model_name="m",
        context_limit=8192,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="x",
        gpu_ids=[0],
        sessions={k: v for k, v in s.model_dump().items() if k.startswith("session_")},
    )
    a, b = [commands(dict(plan, arm=arm))[1] for arm in ["baseline", "treatment"]]
    assert "--enable-prefix-caching" in a[1]
    assert "--kv-cache-memory-bytes" not in a[1] and "--kv-cache-memory-bytes" not in b[1]
    assert a[1] == b[1][:-2]
    assert "DYN_KVBM_DISK_CACHE_GB" not in b[2]


def test_cache_evidence_fails_closed_on_missing_reset_and_nonfinite(tmp_path):
    with pytest.raises(RuntimeError):
        cache_evidence(tmp_path, "baseline")
    p = tmp_path / "replica-0"
    p.mkdir()
    for label, n in [("before", 10), ("after", 20)]:
        (p / f"{label}-worker-0.prom").write_text(f"vllm:prefix_cache_hits_total {n}\n")
        (p / f"{label}-kvbm.prom").write_text(f"kvbm_onboard_blocks_h2d {n}\n")
    assert cache_evidence(tmp_path, "treatment")[0]["cpu_reload_observed"]
    (p / "after-worker-0.prom").write_text("vllm:prefix_cache_hits_total nan\n")
    with pytest.raises(ValueError):
        cache_evidence(tmp_path, "treatment")
    (p / "after-worker-0.prom").write_text("vllm:prefix_cache_hits_total 1\n")
    with pytest.raises(RuntimeError):
        cache_evidence(tmp_path, "treatment")


def test_replica_ports_numa_and_cancel_event():
    from simulator.config import EngineConfig

    cfg = EngineConfig(
        type="vllm_cuda",
        model_id="test",
        gpu_device_ids=[0, 4],
        replica_devices=[[0, 4]],
        port=19200,
    )
    e = SessionReplicas(
        cfg,
        spec(),
        "baseline",
        "owner",
        {0: dict(cpus=[0, 1], node=0), 4: dict(cpus=[64, 65], node=1)},
    )
    assert [c.cfg.port for c in e.children] == [19200, 19300]
    assert all(c.stopping is e.stopping for c in e.children)
    assert e.children[1].cfg.docker_extra_args[-4:] == [
        "--cpuset-cpus",
        "64,65",
        "--cpuset-mems",
        "1",
    ]
    assert e.children[1].cfg.gpu_device_ids == [4]


def test_open_loop_replay_grades_sse_and_preserves_cold_work(monkeypatch, tmp_path):
    s = spec().model_copy(update=dict(duration_s=0.3, concurrency=32, min_requests=1))
    trace = [dict(at_s=0.01 + i * 0.015, slot=i % 4, generation=0, turn=i // 4) for i in range(18)]
    expected = {
        capacity.document_item(s, e, 0)["prompt"]: capacity.document_item(s, e, 0)["expected"]
        for e in trace
    }
    original = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        answer = json.dumps(expected[body["messages"][0]["content"]])
        chunks = [
            {"choices": [{"delta": {"content": answer}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            },
        ]
        return httpx.Response(
            200,
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
        )

    monkeypatch.setattr(
        capacity.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    engine = SimpleNamespace(replica_urls=["http://test/v1"], api_model_name="m", api_key="EMPTY")
    metrics = asyncio.run(capacity.replay(engine, s, trace, 0, tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == metrics["scheduled"] == 18
    assert metrics["correct"] == 18 and metrics["cold_requests"] == 4
    assert metrics["sustained_output_tokens_s"] > 0
    assert metrics["stable_capacity"]
    assert not metrics["error_counts"]


def test_admission_rejections_count_and_do_not_prove_server_saturation(monkeypatch, tmp_path):
    async def slow(*args):
        await asyncio.sleep(0.05)
        return dict(transport_ok=True, correct=True, ttft_s=0.01, usage={"completion_tokens": 1})

    monkeypatch.setattr(capacity, "request_one", slow)
    s = spec().model_copy(
        update=dict(duration_s=0.08, concurrency=1, session_max_pending=2, min_requests=1)
    )
    trace = [dict(at_s=0.001 * i, slot=i, generation=0, turn=0) for i in range(10)]
    e = SimpleNamespace(replica_urls=["http://test"], api_model_name="m", api_key="EMPTY")
    m = asyncio.run(capacity.replay(e, s, trace, 0, tmp_path))
    assert m["attempted"] == 10 and m["error_counts"]["admission_limit"] >= 8
    assert m["client_limited"] and not m["stable_capacity"]
    assert m["elapsed_s_including_drain"] >= 0.1


def test_campaign_freezes_grid_and_exports_separate_reports(tmp_path, monkeypatch):
    from zipfile import ZipFile

    from simulator.benchmarks.storage import read_json, write_json
    from simulator.infrastructure.campaign import create, verify
    from simulator.infrastructure.report import export

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    s = spec()
    p = create(tmp_path, {"model": "test/model"}, s, "c" * 32)
    manifest = verify(p)
    assert len(manifest["schedule"]) == 8
    for cell in manifest["schedule"]:
        assert cell["rate"] == cell["population"] * cell["per_session_rate"]
    assert "session_capacity.py" in manifest["adapter_sources"]
    assert manifest["project"]["outputs"][0]["title"] == "Document-analysis capacity"
    state = read_json(p / "state.json")
    state.update(phase="cancelled", results=[])
    write_json(p / "state.json", state)
    export(p, capacity.compare_capacity([], s))
    for output in manifest["project"]["outputs"]:
        folder = p / "outputs" / output["id"]
        claim = read_json(folder / "claim.json")
        assert claim["status"] == "not_qualified" and not claim["positive_claim_supported"]
        with ZipFile(folder / "white-paper.docx") as z:
            xml = z.read("word/document.xml").decode()
            assert "CPU" in xml and "synthetic" in xml
    assert list(p.glob("*.zip"))


def test_cancelled_replay_retains_started_requests(monkeypatch, tmp_path):
    async def never(*args):
        await asyncio.sleep(5)

    monkeypatch.setattr(capacity, "request_one", never)
    s = spec().model_copy(update=dict(duration_s=1))
    e = SimpleNamespace(replica_urls=["http://test"], api_model_name="m", api_key="EMPTY")

    async def run():
        task = asyncio.create_task(
            capacity.replay(e, s, [dict(at_s=0, slot=0, generation=0, turn=0)], 0, tmp_path)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    rows = [json.loads(r) for r in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["error"] == "cancelled"
