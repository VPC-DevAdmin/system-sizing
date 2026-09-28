import json

import pytest

from simulator.infrastructure.comparison import compare
from simulator.infrastructure.spec import ExperimentSpec
from simulator.infrastructure.speculative import verify_activity


def test_load_comparisons_do_not_hide_failed_or_missing_cells():
    spec = ExperimentSpec(
        project_id="P04",
        speculative_concurrencies=[1, 16],
        paired_blocks=5,
        mode="confirmation",
        duration_s=600,
        warmup_s=120,
    )
    rows = [
        dict(
            block=b,
            arm=a,
            concurrency=c,
            metrics=dict(
                gates_passed=True,
                qualified_per_hour=100 if a == "baseline" else 150,
                latency_p95_s=2,
                ttft_p95_s=0.1,
            ),
        )
        for b in range(5)
        for a in ("baseline", "treatment")
        for c in (1, 16)
    ]
    rows[-1]["metrics"]["gates_passed"] = False
    result = compare(rows, spec)
    assert not result["qualified"] and result["metrics"] == {}
    assert result["by_concurrency"]["1"]["qualified"]
    assert not result["by_concurrency"]["16"]["qualified"]
    result = compare(rows[:-2], spec)
    assert result["by_concurrency"]["1"]["reason"] == "Incomplete paired experiment"
    with pytest.raises(ValueError, match="Duplicate"):
        compare(rows + rows[:1], spec)


def test_native_drafting_required_on_each_replica_but_zero_acceptance_is_valid(tmp_path):
    for index in range(2):
        for phase, draft in [("before", 10), ("after", 20)]:
            (tmp_path / f"{phase}-replica-{index}.prom").write_text(
                f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {draft}\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 0\n'
            )
    verify_activity(tmp_path, 2)
    result = json.loads((tmp_path / "speculative-verification.json").read_text())
    assert all(r["accepted_fraction"] == 0 for r in result["replicas"])
    (tmp_path / "after-replica-1.prom").write_text((tmp_path / "before-replica-1.prom").read_text())
    with pytest.raises(RuntimeError, match="every replica"):
        verify_activity(tmp_path, 2)


def test_each_load_has_its_own_frozen_outputs_and_reports(tmp_path, monkeypatch):
    from simulator.infrastructure import campaign
    from simulator.infrastructure.report import export

    monkeypatch.setattr(
        campaign,
        "recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "model"}},
    )
    spec = ExperimentSpec(project_id="P04", speculative_concurrencies=[1, 16], paired_blocks=1)
    path = campaign.create(tmp_path, {"model": "model"}, spec, "b" * 32)
    manifest = campaign.verify(path)
    assert len(manifest["project"]["outputs"]) == 6
    state = json.loads((path / "state.json").read_text())
    state["results"] = [
        dict(
            block=0,
            arm=a,
            concurrency=c,
            metrics=dict(
                gates_passed=c == 1,
                attempted=32,
                quality_fraction=1,
                qualified_per_hour=c * 100,
                latency_p95_s=c,
                ttft_p95_s=0.1,
            ),
        )
        for a in ("baseline", "treatment")
        for c in (1, 16)
    ]
    (path / "state.json").write_text(json.dumps(state))
    export(path, compare(state["results"], spec))
    claims = [json.loads(p.read_text()) for p in (path / "outputs").glob("*/claim.json")]
    assert {c["concurrency"] for c in claims} == {1, 16}
    assert not any(c["positive_claim_supported"] for c in claims)
    text = (path / "outputs/P04-O2-C16/claim-brief.md").read_text()
    assert "**Concurrency:** 16" in text and "1,600.000" in text
    assert "100.000" not in text


def test_grid_reuses_each_owned_launch_and_cleans_up(monkeypatch, tmp_path):
    import asyncio
    import contextlib
    from dataclasses import asdict

    from simulator.config import Config
    from simulator.infrastructure import worker

    spec = ExperimentSpec(
        project_id="P04", speculative_concurrencies=[1, 4], paired_blocks=1, warmup_s=0
    )
    cfg = Config()
    manifest = {
        "spec": spec.model_dump(),
        "configuration": asdict(cfg),
        "schedule": [{"block": 0, "arm": a} for a in ("baseline", "treatment")],
        "id": "test",
        "source_result": {"model": "model"},
    }
    (tmp_path / "state.json").write_text(json.dumps({"results": []}))
    (tmp_path / "corpus.json").write_text("[]")
    monkeypatch.setattr(worker, "verify", lambda p: manifest)
    monkeypatch.setattr(worker, "Lease", lambda p: contextlib.nullcontext())
    monkeypatch.setattr(worker, "inspect_host", lambda: {})
    monkeypatch.setattr(
        worker, "fingerprint", lambda e: {"snapshot": "/model", "image": {"id": "image"}}
    )
    monkeypatch.setattr(worker, "capture_environment", lambda c: {})
    monkeypatch.setattr(worker, "pin_model", lambda *a: None)
    monkeypatch.setattr(worker, "load_config", lambda p: cfg)
    launches = []
    stops = []
    loads = []

    class Engine:
        replica_urls = ["http://one"]

        def __init__(self, *a, **kw):
            pass

        def shutdown(self):
            stops.append(True)

    monkeypatch.setattr(worker, "OwnedReplicas", Engine)

    async def launch(e, d):
        launches.append(str(d))

    async def probe(*a):
        pass

    async def telemetry(*a):
        await asyncio.Event().wait()

    async def measure(e, corpus, spec, path, *a):
        loads.append(spec.concurrency)
        return dict(gates_passed=True, qualified_per_hour=10, latency_p95_s=1, ttft_p95_s=0.1)

    monkeypatch.setattr(worker, "launch_owned", launch)
    monkeypatch.setattr(worker, "probe", probe)
    monkeypatch.setattr(worker, "telemetry", telemetry)
    monkeypatch.setattr(worker, "measure", measure)
    monkeypatch.setattr(worker, "capture_metrics", probe)
    monkeypatch.setattr(worker, "export", lambda *a: None)
    from simulator.infrastructure import speculative

    monkeypatch.setattr(speculative, "verify_activity", lambda *a: None)
    monkeypatch.setattr(speculative, "export_candidate", lambda *a: None)
    asyncio.run(worker.campaign(tmp_path))
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["phase"] == "complete", state
    assert len(launches) == len(stops) == 2 and loads == [1, 4, 1, 4]
    assert len(state["results"]) == 4
