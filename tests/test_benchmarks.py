"""Contract, evidence integrity and lifecycle regression checks for external harnesses."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import zipfile
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from simulator.benchmarks.adapters import ArtifactError, command, parse, verdict
from simulator.benchmarks.campaign import create_campaign, recorded_config
from simulator.benchmarks.process import Lease, run_process
from simulator.benchmarks.report import export_bundle
from simulator.benchmarks.spec import BenchmarkSpec, digest, resolve_spec, winners
from simulator.benchmarks.storage import campaign_path, read_json, sha256, write_json
from simulator.config import Config


def source(tmp_path, *, legacy=False):
    run = tmp_path / "run_01"
    run.mkdir()
    cfg = asdict(Config())
    cfg["engine"]["model_id"] = "org/model"
    with sqlite3.connect(run / "run.db") as db:
        db.execute("CREATE TABLE cohort_run (model_id TEXT,config_json TEXT)")
        db.execute(
            "INSERT INTO cohort_run VALUES (?,?)",
            (
                "org/model",
                json.dumps({"mode": "headline_sweep"} if legacy else {"resolved_config": cfg}),
            ),
        )
    return {
        "model": "org/model",
        "engine": "vllm",
        "out_tok_s": 100,
        "concurrency": 8,
        "max_num_seqs": 64,
        "input_tokens": 128,
        "output_tokens": 64,
        "confirmed": True,
        "run_dir": str(run),
    }


@pytest.mark.parametrize(
    "data",
    [
        {"harnesses": ["unknown"]},
        {"harnesses": ["aiperf", "aiperf"]},
        {"recipe": "slo", "rates": [1]},
        {"recipe": "sweep"},
        {"rates": [float("nan")]},
        {"concurrency": [0]},
        {"duration_s": 0},
        {"output_tokens": 128},
        {"recipe": "slo", "rates": [1, 1]},
        {"ignored_option": True},
        {"max_hours": float("inf")},
    ],
)
def test_invalid_specs_are_rejected(data):
    with pytest.raises(ValidationError):
        BenchmarkSpec(**data)


def test_winner_across_quantizations_not_sizes():
    catalog = [
        {"id": "a", "family": "family", "params_b": 30},
        {"id": "b", "family": "family", "params_b": 30},
        {"id": "c", "family": "family", "params_b": 70},
    ]
    rows = [
        {"model": m, "confirmed": True, "out_tok_s": rate}
        for m, rate in [("a", 10), ("b", 20), ("c", 15)]
    ]
    rows += [
        {"model": "d", "out_tok_s": 9999},
        {"model": "e", "confirmed": True, "out_tok_s": float("nan")},
        {"model": "a", "confirmed": True, "out_tok_s": 900, "steady_state": False},
    ]
    assert {r["model"] for r in winners(rows, catalog)} == {"b", "c"}


def test_exact_config_and_identity(tmp_path):
    row = source(tmp_path)
    spec = BenchmarkSpec(harnesses=["aiperf"])
    path = create_campaign(tmp_path, row, spec, source={"kind": "manual"})
    assert create_campaign(tmp_path, row, spec, source={"kind": "manual"}) == path
    assert read_json(path / "manifest.json")["configuration"]["engine"]["model_id"] == row["model"]
    for name, expected in read_json(path / "manifest.json")["adapter_sources"].items():
        assert sha256(path / "adapter-source" / name) == expected
    other = create_campaign(
        tmp_path, row, spec.model_copy(update={"seed": 2}), source={"kind": "manual"}
    )
    assert other != path


def test_legacy_config_refused(tmp_path):
    row = source(tmp_path, legacy=True)
    with pytest.raises(ValueError, match="legacy"):
        recorded_config(tmp_path, row)


def test_paths_and_symlinks_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        campaign_path(tmp_path, "../other")
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / ("a" * 24)).symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        campaign_path(tmp_path, "a" * 24)


def test_load_resolution_requires_recorded_conditions():
    with pytest.raises(ValueError, match="token lengths"):
        resolve_spec(BenchmarkSpec(), {})
    assert resolve_spec(
        BenchmarkSpec(), {"input_tokens": 128, "output_tokens": 64, "concurrency": 16}
    )["concurrency"] == [16]


def test_commands_have_no_shell_and_distinguish_schedules(tmp_path):
    spec = BenchmarkSpec(recipe="slo", rates=[2], ttft_p95_ms=100).model_dump()
    for h in ["aiperf", "guidellm"]:
        args = command(
            h,
            h,
            spec,
            "http://localhost:1",
            "org/model",
            "tokenizer",
            tmp_path / "prompts.jsonl",
            tmp_path,
            2,
            300,
        )
        assert args[0] == (h if h == "aiperf" else "python")
        assert "poisson" in " ".join(args)
        assert "sh" not in args
    assert "kind=static,value=42" in args


def test_missing_tpot_cannot_pass_gate():
    spec = BenchmarkSpec(recipe="slo", rates=[1], tpot_p95_ms=20).model_dump()
    metrics = {
        "successful": 200,
        "duration_s": 300,
        "error_fraction": 0,
        "ttft_p95_ms": 1,
        "tpot_p95_ms": None,
    }
    assert verdict(metrics, spec, {"routed": {"a": 200}})[0] == "inconclusive"


def test_client_limit_and_unvisited_replica_invalidate():
    spec = BenchmarkSpec().model_dump()
    metrics = {"successful": 200, "duration_s": 300, "error_fraction": 0}
    assert verdict(metrics, spec, {"routed": {"a": 0}})[0] == "inconclusive"
    assert (
        verdict(metrics, spec, {"routed": {"a": 200}, "max_loop_lag_ms": 101})[0] == "inconclusive"
    )
    metrics["error_fraction"] = 0.2
    assert verdict(metrics, spec, {"routed": {"a": 200}})[0] == "failed_requirements"


def test_routing_gate_distinguishes_isolated_pause_from_sustained_delay():
    spec = BenchmarkSpec().model_dump()
    metrics = {"successful": 200, "duration_s": 300, "error_fraction": 0}
    routing = {
        "routed": {"a": 200},
        "max_loop_lag_ms": 500,
        "loop_lag_samples": 1000,
        "loop_lag_over_100ms": 1,
    }
    assert verdict(metrics, spec, routing)[0] == "measured"
    routing["loop_lag_over_100ms"] = 60
    assert verdict(metrics, spec, routing)[0] == "inconclusive"
    routing["loop_lag_over_100ms"] = 1
    routing["max_worker_lag_over_100ms_fraction"] = 0.1
    assert verdict(metrics, spec, routing)[0] == "inconclusive"


def test_unknown_artifact_schema_not_guessed(tmp_path):
    write_json(tmp_path / "profile_export_aiperf.json", {"schema_version": "new"})
    with pytest.raises(ArtifactError, match="schema"):
        parse("aiperf", tmp_path)


def test_resource_lease_exclusive_and_released(tmp_path):
    p = tmp_path / "lease"
    with Lease(p), pytest.raises(RuntimeError):
        with Lease(p):
            pass
    with Lease(p):
        pass


def test_process_cancel_terminates_and_preserves_log(tmp_path):
    async def check():
        task = asyncio.create_task(
            run_process(
                [sys.executable, "-u", "-c", "import time;print('started');time.sleep(30)"],
                tmp_path / "process.log",
                timeout=40,
            )
        )
        for _ in range(100):
            if (tmp_path / "process.log").exists() and "started" in (
                tmp_path / "process.log"
            ).read_text():
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "started" in (tmp_path / "process.log").read_text()

    asyncio.run(check())


def test_export_evidence_checksums_and_native_docx(tmp_path):
    row = source(tmp_path)
    path = create_campaign(
        tmp_path, row, BenchmarkSpec(harnesses=["aiperf"]), source={"kind": "manual"}
    )
    state = read_json(path / "state.json")
    state.update(phase="complete", outcome="incomplete")
    write_json(path / "state.json", state)
    bundle = export_bundle(path)
    with zipfile.ZipFile(bundle) as z:
        hashes = json.loads(z.read("SHA256SUMS.json"))
        assert "manifest.json" in hashes and "report.docx" in hashes
        for name, checksum in hashes.items():
            assert checksum == sha256(path / name)
    with zipfile.ZipFile(path / "report.docx") as z:
        text = z.read("word/document.xml").decode()
        assert "Third party inference benchmark report" in text
        assert "certification" in text


def test_api_rejects_changed_selection_and_bad_settings(tmp_path):
    from simulator.service import create_app

    with TestClient(create_app(runs_base=tmp_path, catalog_dir=tmp_path / "catalog")) as client:
        assert client.get("/api/roofline/benchmarks").status_code == 200
        assert (
            client.post(
                "/api/roofline/benchmarks", json={"result_id": "stale", "spec": {}}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/roofline/benchmarks", json={"result_id": "stale", "spec": {"unknown": 1}}
            ).status_code
            == 422
        )
        assert client.get("/api/roofline/benchmarks/bad").status_code == 422


def test_api_records_manual_selection_without_switching_precision(tmp_path, monkeypatch):
    from simulator.roofline import State, save_state
    from simulator.service import create_app

    row = source(tmp_path)
    save_state(tmp_path / "roofline.json", State(status="finished", results=[row]))
    called = []

    async def execute(path, base):
        called.append(read_json(path / "manifest.json")["source_result"])
        return {"outcome": "measured"}

    monkeypatch.setattr("simulator.service.benchmarks.execute", execute)
    with TestClient(create_app(runs_base=tmp_path, catalog_dir=tmp_path / "catalog")) as client:
        res = client.post(
            "/api/roofline/benchmarks",
            json={"result_id": digest(row), "spec": {"harnesses": ["aiperf"]}},
        )
        assert res.status_code == 202
    assert called == [row]


def test_boolean_concurrency_and_fractional_duration_rejected():
    for data in ({"concurrency": [True]}, {"duration_s": 10.5}):
        with pytest.raises(ValidationError):
            BenchmarkSpec(**data)


def test_output_length_and_offered_rate_mismatch_inconclusive():
    spec = BenchmarkSpec(recipe="slo", rates=[10], ttft_p95_ms=100).model_dump()
    spec.update(output_tokens=128, _load_point=10)
    m = {
        "successful": 200,
        "duration_s": 300,
        "error_fraction": 0,
        "ttft_p95_ms": 1,
        "output_tokens_mean": 64,
    }
    outcome, reasons = verdict(m, spec, {"routed": {"a": 200}})
    assert outcome == "inconclusive"
    assert any("output length" in r for r in reasons)
    assert any("arrivals" in r for r in reasons)


def test_completed_evidence_tampering_refused(tmp_path):
    from simulator.benchmarks.campaign import execute

    path = create_campaign(
        tmp_path, source(tmp_path), BenchmarkSpec(harnesses=["aiperf"]), source={}
    )
    state = read_json(path / "state.json")
    state.update(phase="complete")
    write_json(path / "state.json", state)
    export_bundle(path)
    (path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        asyncio.run(execute(path, tmp_path))


def test_orphan_is_visible_as_resumable(tmp_path):
    from simulator.benchmarks.campaign import list_campaigns

    path = create_campaign(
        tmp_path, source(tmp_path), BenchmarkSpec(harnesses=["aiperf"]), source={}
    )
    state = read_json(path / "state.json")
    state.update(phase="measuring")
    write_json(path / "state.json", state)
    assert list_campaigns(tmp_path)[0]["phase"] == "interrupted"
    with Lease(path / ".worker.lock"):
        assert list_campaigns(tmp_path)[0]["phase"] == "measuring"


def test_search_report_keeps_partial_models(tmp_path):
    from simulator.benchmarks.report import export_search

    path = create_campaign(
        tmp_path, source(tmp_path), BenchmarkSpec(harnesses=["aiperf"]), source={}
    )
    export_bundle(path)
    folder = export_search(
        tmp_path,
        [
            {"id": path.name, "model": "org/model", "status": "measured"},
            {"model": "failed-model", "status": "incomplete", "error": "legacy"},
        ],
    )
    with zipfile.ZipFile(folder / "evidence.zip") as archive:
        assert f"models/{path.name}.zip" in archive.namelist()
        assert b"failed-model" in archive.read("search.json")


def test_pin_snapshot_preserves_hf_blob_mount():
    from simulator.benchmarks.assets import pin_model

    cfg = Config().engine
    cfg.type = "vllm_cuda_multi"
    pin_model(cfg, "/cache/hub/models--org--model/snapshots/abc")
    assert cfg.docker_volumes["/cache/hub"] == "/cache/hub:ro"
    assert cfg.model_local_path == "/cache/hub/models--org--model/snapshots/abc"
    assert cfg.served_model_name == cfg.model_id


def test_worker_recovery_preserves_attempts_and_shared_launch(tmp_path, monkeypatch):
    from simulator.benchmarks import worker

    cfg = Config()
    cfg.engine.type = "mock"
    spec = BenchmarkSpec(
        harnesses=["aiperf", "guidellm"],
        max_attempts=2,
        duration_s=10,
        warmup_s=0,
        repetitions=1,
        min_requests=1,
    ).model_dump()
    spec.update(input_tokens=8, output_tokens=8, concurrency=[2])
    write_json(tmp_path / "manifest.json", {"recipe": spec, "configuration": asdict(cfg)})
    write_json(
        tmp_path / "state.json",
        {
            "attempts": [
                {
                    "harness": "aiperf",
                    "point": 2,
                    "repetition": 1,
                    "outcome": "running",
                    "number": 1,
                }
            ]
        },
    )
    tok = tmp_path / "tokenizer"
    tok.mkdir()
    (tmp_path / "tokenizer-cache/hub/models--capsim--benchmark-tokenizer/snapshots/local").mkdir(
        parents=True
    )
    (tmp_path / "prompts.jsonl").write_text("{}\n")
    write_json(tmp_path / "dataset.json", {"sha256": sha256(tmp_path / "prompts.jsonl")})
    calls = []

    class Engine:
        api_model_name = "model"

        def __init__(self):
            self.cfg = cfg.engine

        def launch(self, **kwargs):
            calls.append("launch")

        def shutdown(self):
            calls.append("shutdown")

    engine = Engine()
    monkeypatch.setattr("simulator.engines.make_engine", lambda *a: engine)
    monkeypatch.setattr(worker, "fingerprint", lambda cfg: {})
    monkeypatch.setattr(worker, "capture_environment", lambda cfg: {})
    monkeypatch.setattr(worker, "local_snapshot", lambda model: str(tok))

    async def prepare(*a):
        return {"binary": "tool"}

    async def probe(*a):
        pass

    routing_stats = {"active": 0, "routed": {"a": 10}}

    async def measure(*a):
        routing_stats["routed"]["a"] = 20
        calls.append("measure")
        return 1 if calls.count("measure") == 1 else 0

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def routing(*a, **kwargs):
        yield "http://local", routing_stats

    monkeypatch.setattr(worker, "prepare", prepare)
    monkeypatch.setattr(worker, "probe", probe)
    monkeypatch.setattr(worker, "run_measurement", measure)
    monkeypatch.setattr(worker, "routing_endpoint", routing)
    monkeypatch.setattr(
        worker, "parse", lambda *a: {"successful": 20, "duration_s": 10, "error_fraction": 0}
    )
    monkeypatch.setattr(worker, "export_bundle", lambda *a: calls.append("export"))
    asyncio.run(worker.campaign(tmp_path, tmp_path))
    result = read_json(tmp_path / "state.json")
    assert result["outcome"] == "measured"
    assert [a["outcome"] for a in result["attempts"]] == [
        "interrupted",
        "failed",
        "measured",
        "measured",
    ]
    assert calls.count("launch") == 2  # first launch fails; retry reused by second harness
    assert calls[-2:] == ["shutdown", "export"]  # Seal logs after shutdown writers finish.
    assert calls.count("shutdown") == 2


def test_missing_nvidia_smi_is_a_clear_refusal(monkeypatch):
    from simulator.benchmarks import worker

    class Engine:
        cfg = Config().engine
        cfg.type = "vllm_cuda"

    def missing(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(worker.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="Cannot verify exclusive GPU access"):
        worker.assert_gpu_idle(Engine())


def test_llamacpp_skips_gpu_exclusivity_check(monkeypatch):
    from simulator.benchmarks import worker

    class Engine:
        cfg = Config().engine
        cfg.type = "llamacpp"

    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: pytest.fail("no nvidia-smi"))
    worker.assert_gpu_idle(Engine())


def test_busy_gpu_refused_before_engine_launch(tmp_path, monkeypatch):
    from simulator.benchmarks import worker

    class Engine:
        cfg = Config().engine
        cfg.type = "vllm_cuda"

        def launch(self, **kwargs):
            pytest.fail("Must not launch or clean up containers on occupied GPUs")

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *a, **kw: type("P", (), {"returncode": 0, "stdout": "12345\n"})(),
    )
    with pytest.raises(RuntimeError, match="already active"):
        asyncio.run(worker.safe_launch(Engine(), tmp_path))


def test_automatic_finalists_select_one_quantization_and_continue_after_failure(
    tmp_path, monkeypatch
):
    from simulator.benchmarks import campaign
    from simulator.roofline import State

    rows = [
        {
            "model": m,
            "engine": "mock",
            "max_num_seqs": 8,
            "output_tokens": 8,
            "confirmed": True,
            "out_tok_s": v,
        }
        for m, v in [("fp8", 20), ("bf16", 10), ("other", 5)]
    ]
    catalog = [
        {"id": m, "family": f, "params_b": p}
        for m, f, p in [("fp8", "model", 30), ("bf16", "model", 30), ("other", "other", 70)]
    ]
    monkeypatch.setattr("simulator.model_catalog.load_model_catalog", lambda: catalog)
    seen = []

    def create(base, row, spec, **kw):
        seen.append(row["model"])
        if row["model"] == "other":
            raise ValueError("legacy configuration")
        return tmp_path / "winner"

    async def execute(*a):
        return {"outcome": "measured"}

    monkeypatch.setattr(campaign, "create_campaign", create)
    monkeypatch.setattr(campaign, "execute", execute)
    monkeypatch.setattr("simulator.benchmarks.report.export_search", lambda *a: tmp_path / "report")
    st = State(results=rows)
    asyncio.run(
        campaign.run_finalists(
            tmp_path, st, BenchmarkSpec(harnesses=["aiperf"]), tmp_path / "roofline.json"
        )
    )
    assert seen == ["fp8", "other"]
    assert [r["status"] for r in st.plan["benchmark_campaigns"]] == ["measured", "incomplete"]


def test_finalist_failure_during_execution_leaves_one_terminal_link(tmp_path, monkeypatch):
    """A harness crash must not strand a 'running' row beside an 'incomplete' one."""
    import sqlite3

    from simulator.benchmarks import campaign
    from simulator.roofline import State

    rows = [
        {"model": "m", "engine": "mock", "max_num_seqs": 8, "output_tokens": 8,
         "confirmed": True, "out_tok_s": 20}
    ]
    monkeypatch.setattr(
        "simulator.model_catalog.load_model_catalog",
        lambda: [{"id": "m", "family": "model", "params_b": 30}],
    )
    monkeypatch.setattr(campaign, "create_campaign", lambda base, row, spec, **kw: tmp_path / "w")

    async def execute(*a):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(campaign, "execute", execute)
    monkeypatch.setattr("simulator.benchmarks.report.export_search", lambda *a: tmp_path / "report")
    st = State(results=rows)
    asyncio.run(
        campaign.run_finalists(
            tmp_path, st, BenchmarkSpec(harnesses=["aiperf"]), tmp_path / "roofline.json"
        )
    )
    links = st.plan["benchmark_campaigns"]
    assert len(links) == 1
    assert links[0]["id"] == "w"
    assert links[0]["status"] == "incomplete"
    assert "OperationalError" in links[0]["error"]
    assert st.plan["benchmark_report"] == "report"


def test_hardware_lease_path_ignores_tmpdir(monkeypatch, tmp_path):
    from simulator.benchmarks.process import hardware_lease_path

    monkeypatch.delenv("CAPSIM_HARDWARE_LOCK", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "a"))
    first = hardware_lease_path()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "b"))
    assert hardware_lease_path() == first
    assert not first.is_relative_to(tmp_path)
    monkeypatch.setenv("CAPSIM_HARDWARE_LOCK", str(tmp_path / "shared.lock"))
    assert hardware_lease_path() == tmp_path / "shared.lock"


def test_credentials_in_engine_arguments_are_not_exported():
    from simulator.benchmarks.storage import redact

    data = {
        "docker_extra_args": ["-e", "HF_TOKEN=private", "--api-key", "another"],
        "input_tokens": 128,
        "vllm_extra_env": {"HF_TOKEN": "private"},
    }
    cleaned = redact(data)
    assert "private" not in json.dumps(cleaned)
    assert "another" not in json.dumps(cleaned)
    assert cleaned["input_tokens"] == 128


def test_guard_stops_tool_when_supervisor_is_killed(tmp_path):
    import os
    import signal
    import subprocess
    import time

    import psutil

    child_pid = tmp_path / "child.pid"
    code = (
        "import asyncio,sys;from pathlib import Path;from simulator.benchmarks.process import run_process;"
        "asyncio.run(run_process([sys.executable,'-c',"
        + repr(
            f"import os,time;from pathlib import Path;Path({str(child_pid)!r}).write_text(str(os.getpid()));time.sleep(60)"
        )
        + f"],Path({str(tmp_path / 'child.log')!r}),timeout=70,termination_grace=.3))"
    )
    owner = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 10
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid.exists()
        pid = int(child_pid.read_text())
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=3)
        while psutil.pid_exists(pid) and time.monotonic() < deadline:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()


@pytest.mark.parametrize("workers", [1, 2])
def test_router_visits_every_replica_and_drains(tmp_path, workers):
    import socket

    import httpx

    from simulator.benchmarks.proxy import refresh_routing, routing_endpoint
    from simulator.engines.mock import MockEngine

    async def check():
        engines = []
        for _ in range(2):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            cfg = Config().engine
            cfg.type = "mock"
            cfg.port = port
            cfg.mock_ttft_ms = 1
            cfg.mock_tpot_ms = 1
            engine = MockEngine(cfg)
            await asyncio.to_thread(engine.launch)
            engines.append(engine)
        pool = type(
            "Pool",
            (),
            {
                "replica_urls": [e.base_url for e in engines],
                "api_model_name": "mock",
                "api_key": "EMPTY",
            },
        )()
        try:
            async with routing_endpoint(
                pool, 10, workers=workers, directory=tmp_path / "router"
            ) as (endpoint, stats):
                async with httpx.AsyncClient() as client:
                    for _ in range(4):
                        response = await client.post(
                            endpoint + "/v1/chat/completions",
                            json={
                                "model": "mock",
                                "messages": [{"role": "user", "content": "hello"}],
                                "max_completion_tokens": 2,
                                "stream": True,
                            },
                        )
                        assert response.status_code == 200
                await refresh_routing(stats)
                assert sorted(stats["routed"].values()) == [2, 2]
                assert stats["active"] == 0 and stats["errors"] == 0
                # Disconnect before the declared body is complete. No upstream
                # request should be counted and the active lease must be released.
                port = int(endpoint.rsplit(":", 1)[1])
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(
                    b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                    b"Content-Length: 4096\r\nContent-Type: application/json\r\n\r\n{"
                )
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                for _ in range(40):
                    await asyncio.sleep(0.05)
                    await refresh_routing(stats)
                    if stats["client_disconnects"]:
                        break
                assert stats["client_disconnects"] == 1
                assert sorted(stats["routed"].values()) == [2, 2]
                assert stats["active"] == 0 and stats["errors"] == 0
                await refresh_routing(stats, reset=True)
                assert sum(stats["routed"].values()) == 0
                assert stats["client_disconnects"] == 0
                if workers > 1:
                    stats.processes[0].kill()
                    await stats.processes[0].wait()
                    with pytest.raises(RuntimeError, match="routing worker exited"):
                        await refresh_routing(stats)
        finally:
            for engine in engines:
                await asyncio.to_thread(engine.shutdown)

    asyncio.run(check())


def test_pin_model_keeps_shared_hub_blob_symlinks_resolvable(tmp_path):
    from simulator.benchmarks.assets import pin_model

    hub = tmp_path / "hub"
    repo = hub / "models--org--model"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (hub / "blobs/ab").mkdir(parents=True)
    (hub / "blobs/ab/shared").write_bytes(b"weights")
    (repo / "blobs/hash").symlink_to("../../blobs/ab/shared")
    (snapshot / "model.safetensors").symlink_to("../../blobs/hash")
    cfg = Config().engine
    cfg.type = "vllm_cuda_multi"
    pin_model(cfg, str(snapshot))
    assert cfg.docker_volumes[str(hub)] == str(hub) + ":ro"
    assert cfg.model_local_path == str(snapshot)
    assert (snapshot / "model.safetensors").read_bytes() == b"weights"


def test_source_shape_mismatch_is_not_silently_reproduced(tmp_path):
    row = source(tmp_path)
    write_json(
        __import__("pathlib").Path(row["run_dir"]) / "headline_sweep.json",
        {"shape": {"input_tokens": 32, "output_tokens": 1024}},
    )
    with pytest.raises(ValueError, match="source measurement"):
        create_campaign(tmp_path, row, BenchmarkSpec(harnesses=["aiperf"]), source={})


def test_high_concurrency_raises_inherited_file_limit(monkeypatch):
    import resource

    from simulator.benchmarks.worker import configure_file_limit

    current = [1024, 1048576]
    monkeypatch.setattr(resource, "getrlimit", lambda _: tuple(current))
    monkeypatch.setattr(
        resource, "setrlimit", lambda _, values: current.__setitem__(slice(None), values)
    )
    assert configure_file_limit({"concurrency": [8192]})["soft"] >= 20480
    assert current[1] == 1048576
    current[:] = [1024, 4096]
    with pytest.raises(RuntimeError, match="hard limit"):
        configure_file_limit({"concurrency": [8192]})


def test_guidellm_native_window_metrics_and_proven_boundary_cancellations(tmp_path):
    def stat(mean):
        return {
            "successful": {"mean": mean, "percentiles": {"p95": mean}},
            "total": {"mean": mean * 2, "percentiles": {"p95": mean * 2}},
        }

    metrics = {
        key: stat(value)
        for key, value in {
            "output_tokens_per_second": 100,
            "requests_per_second": 2,
            "output_token_count": 256,
            "prompt_token_count": 163,
            "time_to_first_token_ms": 20,
            "time_per_output_token_ms": 3,
            "inter_token_latency_ms": 2,
            "request_latency": 1,
        }.items()
    }
    metrics["request_totals"] = {"successful": 200, "errored": 1, "incomplete": 2}
    row = {
        "metrics": metrics,
        "start_time": 1000,
        "end_time": 1300,
        "scheduler_metrics": {
            "request_start_delay_avg": 0.001,
            "request_targeted_start_delay_avg": 3,
        },
        "scheduler_state": {
            "end_processing_constraints": {
                "max_duration": {"metadata": {"duration_exceeded": True, "stop_time": 1300}}
            }
        },
        "requests": {
            "incomplete": [
                {
                    "request_start_time": 1200,
                    "request_end_time": t,
                    "info": {"status": "cancelled", "timings": {"resolve_end": t}},
                }
                for t in [1300.1, 1299.0]
            ]
        },
    }
    doc = {"metadata": {"version": 2, "guidellm_version": "0.7.4"}, "benchmarks": [row]}
    write_json(tmp_path / "benchmarks.json", doc)
    actual = parse("guidellm", tmp_path)
    assert actual["output_tok_s"] == 200
    assert actual["completed_output_tok_s"] == 100
    assert actual["boundary_cancelled"] == 1  # Earlier cancellation remains a failure.
    assert actual["error_fraction"] == pytest.approx(2 / 202)
    assert actual["client_delay_ms"] == 1
    assert actual["targeted_start_delay_ms"] == 3000
    # Native exports can retain a request dispatched just after max_duration.
    # It must not invalidate accounting for the two requests in the window.
    row["requests"]["incomplete"].append(
        {
            "request_start_time": 1300.01,
            "request_end_time": 1300.5,
            "info": {"status": "cancelled", "timings": {"resolve_end": 1300.5}},
        }
    )
    write_json(tmp_path / "benchmarks.json", doc)
    assert parse("guidellm", tmp_path)["boundary_cancelled"] == 1
    # Missing retained records cannot prove the native incomplete count.
    row["requests"]["incomplete"][0]["request_end_time"] = None
    write_json(tmp_path / "benchmarks.json", doc)
    assert parse("guidellm", tmp_path)["boundary_cancelled"] == 0
    row["requests"]["incomplete"][0]["request_end_time"] = 1300.1
    spec = BenchmarkSpec().model_dump()
    spec["output_tokens"] = 256
    assert verdict(actual, spec, {"routed": {"a": 203}})[0] == "measured"
    spec.update(recipe="slo")
    assert verdict(actual, spec, {"routed": {"a": 203}})[0] == "inconclusive"
    row["scheduler_state"]["end_processing_constraints"] = {"max_errors": {}}
    write_json(tmp_path / "benchmarks.json", doc)
    assert parse("guidellm", tmp_path)["boundary_cancelled"] == 0


def test_slo_cannot_pass_when_boundary_censoring_hides_latency():
    metrics = {
        "successful": 200,
        "duration_s": 300,
        "error_fraction": 0,
        "boundary_cancelled_fraction": 0.2,
        "ttft_p95_ms": 1,
    }
    spec = BenchmarkSpec(recipe="slo", rates=[1.0], ttft_p95_ms=100).model_dump()
    assert verdict(metrics, spec, {"routed": {"a": 200}})[0] == "inconclusive"


def test_duplicate_campaign_caller_cannot_disturb_active_execution(tmp_path, monkeypatch):
    from simulator.benchmarks import campaign

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def running(*args):
            calls.append("worker")
            entered.set()
            await release.wait()
            return {"outcome": "measured"}

        monkeypatch.setattr(campaign, "_execute", running)
        first = asyncio.create_task(campaign.execute(tmp_path, tmp_path))
        await entered.wait()
        try:
            with pytest.raises(RuntimeError, match="busy"):
                await campaign.execute(tmp_path, tmp_path)
            assert calls == ["worker"]
        finally:
            release.set()
            assert await first == {"outcome": "measured"}

    asyncio.run(check())


def test_campaign_rejects_worker_still_cleaning_up(tmp_path):
    from simulator.benchmarks.campaign import execute

    with Lease(tmp_path / ".worker.lock"):
        with pytest.raises(RuntimeError, match="busy"):
            asyncio.run(execute(tmp_path, tmp_path))


def test_worker_module_uses_selected_source_over_current_checkout(tmp_path):
    import subprocess

    from simulator.benchmarks.process import module_argv

    selected = tmp_path / "selected"
    stale = tmp_path / "stale"
    for root in (selected, stale):
        package = root / "example_worker"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
    (stale / "example_worker" / "main.py").write_text("raise RuntimeError('stale source')")
    (selected / "example_worker" / "main.py").write_text(
        "import json,sys; print(json.dumps(sys.argv[1:]))"
    )
    result = subprocess.run(
        module_argv(str(selected), "example_worker.main", "request path", "8192"),
        cwd=stale,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["request path", "8192"]


def test_report_links_wait_for_complete_bundle_and_finished_export(tmp_path):
    from simulator.benchmarks.campaign import list_campaigns

    path = create_campaign(
        tmp_path, source(tmp_path), BenchmarkSpec(harnesses=["aiperf"]), source={}
    )
    state = read_json(path / "state.json")
    state.update(phase="complete", outcome="measured")
    write_json(path / "state.json", state)
    (path / "report.docx").write_bytes(b"document fixture")
    assert not list_campaigns(tmp_path)[0]["report_ready"]
    (path / "evidence.zip").write_bytes(b"bundle fixture")
    with Lease(path / ".worker.lock"):
        current = list_campaigns(tmp_path)[0]
        assert current["phase"] == "exporting" and not current["report_ready"]
    current = list_campaigns(tmp_path)[0]
    assert current["phase"] == "complete" and current["report_ready"]


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
def test_router_closes_upstream_when_headers_cannot_be_sent(failure):
    import httpx
    from starlette.requests import ClientDisconnect

    from simulator.benchmarks.proxy import _ForwardedStream

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"body"

    async def check():
        upstream = httpx.Response(200, stream=Body())
        assert not upstream.is_closed
        stats = {"active": 1}
        response = _ForwardedStream(upstream, stats)

        async def send(message):
            raise failure()

        async def receive():
            await asyncio.sleep(60)

        with pytest.raises((failure, ClientDisconnect)):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert upstream.is_closed
        assert stats["active"] == 0

    asyncio.run(check())


def test_combined_report_medians_exclude_inconclusive_and_separate_harnesses(tmp_path):
    from docx import Document

    from simulator.benchmarks.report import export_search

    path = create_campaign(
        tmp_path, source(tmp_path), BenchmarkSpec(harnesses=["aiperf", "guidellm"]), source={}
    )
    state = read_json(path / "state.json")
    state["attempts"] = [
        {
            "harness": harness,
            "point": 8,
            "repetition": repeat,
            "outcome": outcome,
            "metrics": {"output_tok_s": rate},
        }
        for harness, repeat, outcome, rate in (
            ("aiperf", 1, "measured", 100),
            ("aiperf", 2, "measured", 300),
            ("aiperf", 3, "inconclusive", 9999),
            ("guidellm", 1, "measured", 1000),
        )
    ]
    write_json(path / "state.json", state)
    folder = export_search(
        tmp_path, [{"id": path.name, "model": "org/model", "status": "needs_review"}]
    )
    rows = [
        [cell.text for cell in row.cells] for row in Document(folder / "report.docx").tables[0].rows
    ]
    assert rows[1:] == [
        ["org/model\naiperf", "8 concurrent", "200.00", "2 / 3"],
        ["org/model\nguidellm", "8 concurrent", "1,000.00", "1 / 3"],
    ]
