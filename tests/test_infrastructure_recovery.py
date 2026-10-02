import asyncio
import json

import httpx
import pytest

from simulator.infrastructure.allocation_recovery import AllocationFailure, run_allocation


def test_retries_are_bounded_and_failed_evidence_paths_are_distinct(tmp_path):
    calls, failures = [], []

    async def operation(path, attempt):
        calls.append(path)
        raise AllocationFailure("worker died")

    result = asyncio.run(
        run_allocation(tmp_path, operation, lambda *args: failures.append(args), enabled=True)
    )
    assert not result
    assert calls == [tmp_path, tmp_path / "retry-01"]
    assert len(failures) == 2


@pytest.mark.parametrize(
    "error",
    [
        asyncio.CancelledError(),
        RuntimeError("Owned cleanup failed"),
        ValueError("bad configuration"),
    ],
)
def test_cancellation_cleanup_and_programming_errors_are_not_retried(tmp_path, error):
    calls = []

    async def operation(path, attempt):
        calls.append(attempt)
        raise error

    with pytest.raises(type(error)):
        asyncio.run(
            run_allocation(
                tmp_path, operation, lambda *args: pytest.fail("must not retry"), enabled=True
            )
        )
    assert calls == [0]


@pytest.mark.parametrize("permanent", [False, True])
def test_worker_preserves_completed_windows_retries_after_cleanup_and_continues(
    tmp_path, monkeypatch, permanent
):
    from simulator.infrastructure import worker
    from simulator.infrastructure.campaign import create
    from simulator.infrastructure.spec import ExperimentSpec

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    spec = ExperimentSpec(
        project_id="P01",
        gpu_ids=[0, 1],
        paired_blocks=1,
        p01_goal_seek=True,
        p01_prefill_grid=[1],
        p01_document_grid=[64],
        p01_answer_grid=[4],
        p01_request_rates=[1, 2, 4],
        warmup_s=0,
        duration_s=20,
        min_requests=10,
        context_limit=16384,
    )
    path = create(tmp_path, {"model": "test/model"}, spec, "d" * 32)
    events, engines = [], []

    class Engine:
        def __init__(self, *args):
            self.number = len(engines)
            self.calls = 0
            self.replica_urls = ["http://test"]
            engines.append(self)

        def shutdown(self):
            events.append(("cleanup", self.number))

    async def launch(engine, directory):
        events.append(("launch", engine.number))

    async def noop(*args):
        pass

    async def sampler(*args):
        await asyncio.Event().wait()

    async def measure(engine, corpus, settings, evidence, start, rate):
        engine.calls += 1
        events.append(("measure", engine.number, rate))
        if (engine.number == 0 and engine.calls == 2) or (permanent and engine.number == 1):
            raise httpx.ConnectError("engine disappeared")
        return dict(gates_passed=True, latency_p95_s=1, queue_p95_s=0)

    monkeypatch.setattr(worker, "hardware_lease_path", lambda: tmp_path / "hardware.lock")
    monkeypatch.setattr(worker, "inspect_host", lambda: {})
    monkeypatch.setattr(worker, "capture_environment", lambda *a: {})
    monkeypatch.setattr(
        worker, "fingerprint", lambda *a: dict(snapshot=str(tmp_path), image={"id": "sha256:test"})
    )
    monkeypatch.setattr(worker, "pin_model", lambda *a: None)
    monkeypatch.setattr(worker, "launch_owned", launch)
    monkeypatch.setattr(worker, "probe", noop)
    monkeypatch.setattr(worker, "capture_metrics", noop)
    monkeypatch.setattr(worker, "telemetry", sampler)
    monkeypatch.setattr(worker, "export", lambda *a: None)
    monkeypatch.setattr("simulator.infrastructure.dynamo.OwnedDynamo", Engine)
    monkeypatch.setattr("simulator.infrastructure.dynamo.verify_transfer", lambda *a: None)
    monkeypatch.setattr("simulator.infrastructure.dynamo.verify_worker_activity", lambda *a: None)
    monkeypatch.setattr("simulator.infrastructure.tenants.measure_tenant", measure)
    asyncio.run(worker.campaign(path))
    state = json.loads((path / "state.json").read_text())
    assert state["phase"] == "complete", state
    assert len(state["results"]) == (4 if permanent else 6)
    assert len(state["attempts"]) == (2 if permanent else 1)
    assert state["attempts"][0]["point"]["rate"] == 2
    assert events.index(("cleanup", 0)) < events.index(("launch", 1))
    assert ("measure", 1, 1) not in events  # successful first window was retained
    assert ("measure", 1, 2) in events
    assert ("launch", 2) in events  # next allocation still ran
    assert len(list(path.rglob("failure.json"))) == (2 if permanent else 1)
    assert json.loads((path / "comparison.json").read_text())["complete_search"] is not permanent
