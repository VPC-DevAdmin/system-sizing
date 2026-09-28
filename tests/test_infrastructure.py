import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from simulator.infrastructure.campaign import campaign_path, create, verify
from simulator.infrastructure.catalog import listing
from simulator.infrastructure.comparison import compare
from simulator.infrastructure.driver import request_one, summarize
from simulator.infrastructure.spec import ExperimentSpec
from simulator.infrastructure.worker import arm_config
from simulator.infrastructure.workload import build_corpus, grade
from simulator.service import create_app


def spec(**kwargs):
    return ExperimentSpec(project_id="P04", **kwargs)


def test_catalog_has_ten_projects_and_separate_outputs():
    projects = listing()
    assert len(projects) == 10
    assert len({o["id"] for p in projects for o in p["outputs"]}) == 31
    assert [p["id"] for p in projects if p["implementation_status"] == "pilot_available"] == [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
        "P10",
    ]


def test_confirmation_rejects_short_runs_and_bad_gpu_lists():
    with pytest.raises(ValidationError):
        spec(mode="confirmation")
    with pytest.raises(ValidationError):
        spec(gpu_ids=[0, 0])
    with pytest.raises(ValidationError):
        spec(response_slo_s=float("nan"))
    with pytest.raises(ValidationError):
        spec(unknown=True)
    assert spec(mode="confirmation", paired_blocks=5, duration_s=600, warmup_s=120)


def test_frozen_tasks_and_strict_correctness():
    corpus = build_corpus(16, 8, 42)
    assert corpus == build_corpus(16, 8, 42)
    assert corpus != build_corpus(16, 8, 43)
    expected = corpus[0]["expected"]
    assert grade(json.dumps(expected), expected)
    assert not grade("```json\n" + json.dumps(expected) + "\n```", expected)
    assert not grade(json.dumps(expected[::-1]), expected)


def test_failures_remain_in_denominator_and_latency_does_not_imply_quality():
    s = spec(min_requests=2)
    rows = [
        dict(transport_ok=True, correct=True, elapsed_s=2, ttft_s=0.1),
        dict(transport_ok=False, correct=False, elapsed_s=120, ttft_s=None),
    ]
    result = summarize(rows, 120, s)
    assert result["quality_fraction"] == 0.5
    assert result["error_fraction"] == 0.5
    assert result["qualified_per_hour"] == 30
    assert not result["gates_passed"]


def results(n=5, fail=False):
    return [
        {
            "block": i,
            "arm": arm,
            "metrics": {
                "gates_passed": not fail,
                "qualified_per_hour": 100 if arm == "baseline" else 120,
                "latency_p95_s": 10 if arm == "baseline" else 8,
            },
        }
        for i in range(n)
        for arm in ("baseline", "treatment")
    ]


def test_paired_comparison_keeps_failures_and_pilot_limit():
    assert not compare(results(2), spec())["qualified"]
    confirmation = spec(mode="confirmation", paired_blocks=5, duration_s=600, warmup_s=120)
    result = compare(results(), confirmation)
    assert result["qualified"]
    assert result["metrics"]["qualified_per_hour"]["change_percent"] == 20
    assert result["metrics"]["latency_p95_s"]["change_percent"] == 20
    assert not compare(results(fail=True), confirmation)["qualified"]
    assert not compare(results()[:-1], confirmation)["qualified"]
    with pytest.raises(ValueError):
        compare(results() + results()[:1], confirmation)


def test_arm_only_adds_speculation_and_does_not_mutate_source():
    cfg = {"engine": {"type": "vllm_cuda_multi", "vllm_extra_flags": []}}
    a, b = (arm_config(cfg, spec(), arm) for arm in ("baseline", "treatment"))
    assert cfg["engine"]["vllm_extra_flags"] == []
    assert b["engine"]["vllm_extra_flags"][:-2] == a["engine"]["vllm_extra_flags"]
    assert "--speculative-config" not in a["engine"]["vllm_extra_flags"]
    with pytest.raises(ValueError):
        arm_config(
            {"engine": {"vllm_extra_flags": ["--speculative-config={}"]}}, spec(), "baseline"
        )


def test_uuid_path_refuses_traversal_and_symlink(tmp_path):
    with pytest.raises(ValueError):
        campaign_path(tmp_path, "../../secret")
    p = tmp_path / "infrastructure"
    p.mkdir()
    (p / ("a" * 32)).symlink_to(tmp_path)
    with pytest.raises(ValueError):
        campaign_path(tmp_path, "a" * 32)


def test_create_idempotency_new_runs_and_tamper_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    row = {"model": "test/model"}
    a = create(tmp_path, row, spec(), "a" * 32)
    assert create(tmp_path, row, spec(), "a" * 32) == a
    assert create(tmp_path, row, spec(), "b" * 32) != a
    with pytest.raises(ValueError):
        create(tmp_path, row, spec(concurrency=8), "a" * 32)
    assert verify(a)["spec"]["project_id"] == "P04"
    (a / "corpus.json").write_text("[]")
    with pytest.raises(ValueError, match="workload changed"):
        verify(a)


def test_stream_requires_done_and_stop_and_grades_json():
    item = build_corpus(1, 4, 42)[0]

    async def run(done, reason):
        chunks = [
            {
                "choices": [
                    {"delta": {"content": json.dumps(item["expected"])}, "finish_reason": None}
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": reason}]},
        ]
        body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
        if done:
            body += "data: [DONE]\n\n"
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            return await request_one(client, "http://localhost/v1", "m", item, spec(), "r")

    assert asyncio.run(run(True, "stop"))["correct"]
    assert not asyncio.run(run(True, "length"))["correct"]
    assert not asyncio.run(run(False, "stop"))["transport_ok"]


def test_service_catalog_and_both_ui_entrypoints(tmp_path):
    with TestClient(create_app(runs_base=tmp_path, catalog_dir=tmp_path / "catalog")) as client:
        response = client.get("/api/infrastructure")
        assert response.status_code == 200
        assert len(response.json()["projects"]) == 10
        assert response.json()["readiness"] is None
        assert "tab-infrastructure" in client.get("/").text
        assert "infrastructure.css" in client.get("/workspace.html").text
        assert "Infrastructure.init" in client.get("/app.js").text
        assert client.get("/api/infrastructure/campaigns/not-an-id").status_code == 422


def test_owned_cleanup_accepts_verified_auto_remove_race(monkeypatch):
    import threading

    from simulator.config import EngineConfig
    from simulator.infrastructure.deployment import OwnedVllm

    replies = iter(
        [
            SimpleNamespace(returncode=0, stderr="", stdout=""),
            SimpleNamespace(returncode=0, stderr="", stdout="false"),
            SimpleNamespace(returncode=1, stderr="removal already in progress", stdout=""),
            SimpleNamespace(returncode=1, stderr="error: no such object: owned", stdout=""),
        ]
    )
    monkeypatch.setattr(
        "simulator.infrastructure.deployment.subprocess.run", lambda *a, **k: next(replies)
    )
    engine = OwnedVllm(EngineConfig(), threading.Event())
    engine._container_id = "owned"
    engine.shutdown()
    assert engine._container_id is None


def test_owned_cleanup_does_not_hide_docker_failure(monkeypatch):
    import threading

    from simulator.config import EngineConfig
    from simulator.infrastructure.deployment import OwnedVllm

    monkeypatch.setattr(
        "simulator.infrastructure.deployment.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="daemon unavailable", stdout=""),
    )
    engine = OwnedVllm(EngineConfig(), threading.Event())
    engine._container_id = "owned"
    with pytest.raises(RuntimeError, match="verify cleanup"):
        engine.shutdown()
    assert engine._container_id == "owned"


def test_energy_integrates_only_bracketed_window_and_keeps_boundary():
    from simulator.infrastructure.power import integrate_energy

    samples = [{"timestamp": t, "watts": 100 + 10 * t} for t in range(4)]
    result = integrate_energy(samples, 0.5, 2.5, boundary="gpu_board", max_gap_s=1)
    assert result["joules"] == 230
    assert not result["whole_server"]
    with pytest.raises(ValueError, match="bracket"):
        integrate_energy(samples, -1, 2, boundary="pdu_outlet", max_gap_s=1)
    with pytest.raises(ValueError, match="gap"):
        integrate_energy(samples, 0, 3, boundary="pdu_outlet", max_gap_s=0.5)


def test_power_change_restores_all_devices_after_partial_failure(tmp_path, monkeypatch):
    from simulator.benchmarks.storage import read_json
    from simulator.infrastructure.power import PowerTransaction

    state = {"GPU-a": 600.0, "GPU-b": 600.0}
    calls = []

    def run(argv):
        if argv[0] == "nvidia-smi":
            return "\n".join(f"{k},{v},300,600" for k, v in state.items())
        device, value = argv[-3], float(argv[-1])
        calls.append((device, value))
        if device == "GPU-b" and value == 400:
            raise RuntimeError("test failure")
        state[device] = value
        return ""

    monkeypatch.setattr("simulator.infrastructure.power._run", run)
    journal = tmp_path / "power.json"
    with pytest.raises(RuntimeError, match="test failure"):
        PowerTransaction(journal, {"GPU-a": 400.0, "GPU-b": 400.0}).apply()
    assert state == {"GPU-a": 600.0, "GPU-b": 600.0}
    assert read_json(journal)["phase"] == "restored"
    assert ("GPU-a", 600.0) in calls and ("GPU-b", 600.0) in calls


def test_json_grader_rejects_type_coercion_and_duplicate_fields():
    assert not grade('{"quantity":true}', {"quantity": 1})
    assert not grade('{"quantity":1,"quantity":1}', {"quantity": 1})


def test_structured_output_schema_does_not_embed_answers():
    item = build_corpus(1, 4, 42)[0]
    recorded = []

    async def run():
        def handler(request):
            recorded.append(json.loads(request.content))
            return httpx.Response(400)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await request_one(client, "http://localhost", "m", item, spec(), "r")

    asyncio.run(run())
    schema = recorded[0]["response_format"]["json_schema"]["schema"]
    encoded = json.dumps(schema)
    assert item["expected"][0]["record_id"] not in encoded
    assert schema["items"]["additionalProperties"] is False


def test_cost_requires_equal_work_and_explicit_wall_boundary():
    from simulator.infrastructure.cost import AccountedWork, CostLedger, compare_cost

    ledger = CostLedger(
        currency="USD",
        priced_on="2026-09-27",
        source="Test fixture only",
        acquisition=87660,
        support_per_year=0,
        software_per_year=0,
        operations_per_year=0,
        amortization_years=1,
        electricity_per_kwh=0,
        cooling_multiplier=1,
    )
    a = AccountedWork(
        campaign_id="a",
        contract_sha256="a" * 64,
        demand_trace_sha256="b" * 64,
        accounting_duration_s=3600,
        accepted_tasks=100,
        wall_energy_joules=3600000,
        energy_boundary="server_psu_input",
        includes_idle_and_reserve=True,
        quality_and_slo_passed=True,
    )
    b = a.model_copy(update={"campaign_id": "b", "accounting_duration_s": 1800})
    result = compare_cost(a, b, ledger, ledger)
    assert result["reduction_percent"] == 50
    assert not result["baseline"]["capital_savings_claim"]
    with pytest.raises(ValueError, match="equal accepted"):
        compare_cost(a, b.model_copy(update={"accepted_tasks": 200}), ledger, ledger)
    with pytest.raises(ValidationError):
        AccountedWork(**{**a.model_dump(), "energy_boundary": "gpu_board"})
    with pytest.raises(ValidationError):
        CostLedger(**{**ledger.model_dump(), "allocated_share": 0.5})


def test_parallel_launch_joins_before_cleanup():
    import threading

    from simulator.infrastructure.deployment import OwnedReplicas

    cfg = SimpleNamespace(replica_devices=[[0], [1]], port=19200)
    engine = OwnedReplicas.__new__(OwnedReplicas)
    engine.cfg = cfg
    engine.stopping = threading.Event()
    exited = threading.Event()
    cleaned = []

    class Child:
        def __init__(self, fails):
            self.fails = fails

        def launch(self, path):
            if self.fails:
                raise RuntimeError("launch failed")
            assert engine.stopping.wait(2)
            exited.set()

        def shutdown(self):
            assert exited.is_set()
            cleaned.append(self.fails)

    engine.children = [Child(True), Child(False)]
    with pytest.raises(RuntimeError, match="launch failed"):
        engine.launch("/tmp/unused")
    assert sorted(cleaned) == [False, True]


def test_recovery_refuses_container_with_wrong_ownership(tmp_path, monkeypatch):
    from simulator.benchmarks.storage import write_json
    from simulator.infrastructure.recovery import recover

    path = tmp_path / ("a" * 32)
    write_json(path / "state.json", {"phase": "launching"})
    write_json(
        path / "block-00-baseline/replica-0/resource.json",
        {"owner": path.name, "name": "capsim-infra-" + "b" * 32, "phase": "creating"},
    )
    monkeypatch.setattr(
        "simulator.infrastructure.recovery.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout='{"capsim.infrastructure":"other"}', stderr=""
        ),
    )
    with pytest.raises(ValueError, match="label does not match"):
        recover(path)


def test_exported_outputs_keep_pilot_gate_and_portable_identity(tmp_path, monkeypatch):
    import hashlib
    import shutil
    import zipfile

    from simulator.infrastructure.report import export

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    source = create(tmp_path, {"model": "test/model"}, spec(), "f" * 32)
    portable = tmp_path / "portable-copy"
    shutil.copytree(source, portable)
    export(portable, {"qualified": False, "reason": "Incomplete paired experiment", "metrics": {}})
    claim = json.loads((portable / "outputs/P04-O1/claim.json").read_text())
    assert claim["campaign_id"] == source.name
    assert claim["status"] == "not_qualified"
    assert not claim["positive_claim_supported"]
    with zipfile.ZipFile(portable / "evidence.zip") as archive:
        checksums = json.loads(archive.read("SHA256SUMS.json"))
        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == expected
            for name, expected in checksums.items()
        )
        assert "report-generation.json" in checksums


def test_mtp_treatment_changes_only_draft_config():
    settings = spec(speculative_method="mtp", speculative_tokens=3)
    cfg = {"engine": {"vllm_extra_flags": []}}
    a = arm_config(cfg, settings, "baseline")
    b = arm_config(cfg, settings, "treatment")
    assert a["engine"]["vllm_extra_flags"] == b["engine"]["vllm_extra_flags"][:-2]
    draft = json.loads(b["engine"]["vllm_extra_flags"][-1])
    assert draft == {"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"}


def test_correct_but_late_answers_cannot_qualify():
    rows = [dict(transport_ok=True, correct=True, elapsed_s=5, ttft_s=0.1)] * 20
    result = summarize(rows, 25, spec(min_requests=10, response_slo_s=2))
    assert result["quality_fraction"] == 1
    assert result["slo_fraction"] == 0
    assert not result["gates_passed"]


def test_cache_config_removes_inherited_dtype_and_keeps_same_backend():
    from simulator.infrastructure.cache import cache_config

    settings = ExperimentSpec(project_id="P03", context_limit=65536)
    source = {
        "engine": {
            "vllm_extra_flags": ["--kv-cache-dtype", "fp8", "--attention-backend=FLASHINFER"]
        }
    }
    for arm, dtype in [
        ("reference", "bfloat16"),
        ("baseline", "fp8"),
        ("treatment", "int4_per_token_head"),
    ]:
        cfg = cache_config(source, settings, arm)["engine"]
        assert cfg["kv_cache_dtype"] == dtype
        assert cfg["attention_backend"] == "TRITON_ATTN"
        assert cfg["vllm_extra_flags"].count("--kv-cache-dtype") == 1
        assert "--attention-backend=FLASHINFER" not in cfg["vllm_extra_flags"]
        assert "--speculative-config" not in cfg["vllm_extra_flags"]
    assert source["engine"]["vllm_extra_flags"][1] == "fp8"


def test_cache_grid_is_exploratory_and_reference_quality_is_required():
    from simulator.infrastructure.cache import compare_cache

    settings = ExperimentSpec(
        project_id="P03",
        paired_blocks=1,
        context_limit=65536,
        cache_record_counts=[16, 64],
        cache_concurrencies=[1, 4],
    )
    rows = [
        dict(
            block=0,
            arm=arm,
            document_records=r,
            concurrency=c,
            metrics={
                "gates_passed": True,
                "prompt_tokens_min": r * 100,
                "attempted": 30,
                "quality_fraction": 1,
                "error_fraction": 0,
            },
        )
        for arm in ("reference", "baseline", "treatment")
        for r in (16, 64)
        for c in ((1,) if arm == "reference" else (1, 4))
    ]
    result = compare_cache(rows, settings)
    assert result["complete"] and result["reference_quality_passed"]
    assert not result["qualified"]
    assert result["boundaries"]["treatment"]["largest_tested_qualified_document_records"] == 64
    rows[0]["metrics"]["gates_passed"] = False
    assert compare_cache(rows, settings)["reference_quality_passed"]
    rows[0]["metrics"]["quality_fraction"] = 0
    result = compare_cache(rows, settings)
    assert not result["reference_quality_passed"]
    assert not result["reference_quality_by_document_records"][16]
    assert result["reference_quality_by_document_records"][64]
    assert all(p["passed"] == (p["document_records"] == 64) for p in result["points"])
    with pytest.raises(ValueError, match="Duplicate"):
        compare_cache(rows + rows[:1], settings)
    with pytest.raises(ValidationError):
        ExperimentSpec(project_id="P03", context_limit=65536, cache_record_counts=[64, 16])
    with pytest.raises(ValidationError):
        ExperimentSpec(
            project_id="P03",
            mode="confirmation",
            paired_blocks=5,
            duration_s=600,
            warmup_s=120,
            context_limit=65536,
        )


def test_cache_creation_freezes_all_shapes_and_reference_schedule(tmp_path, monkeypatch):
    import simulator.infrastructure.campaign as campaign

    monkeypatch.setattr(campaign, "EXECUTABLE", {"P03", "P04"})
    monkeypatch.setattr(
        campaign,
        "recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    settings = ExperimentSpec(
        project_id="P03",
        paired_blocks=1,
        context_limit=16384,
        cache_record_counts=[16, 64],
        cache_concurrencies=[1, 4],
    )
    path = create(tmp_path, {"model": "test/model"}, settings, "d" * 32)
    manifest = verify(path)
    assert {s["arm"] for s in manifest["schedule"]} == {"reference", "baseline", "treatment"}
    assert set(json.loads((path / "corpus.json").read_text())) == {"16", "64"}
    from simulator.infrastructure.report import export

    export(path, {"qualified": False, "metrics": {}, "reason": "Cache experiment incomplete"})
    report = (path / "outputs/P03-O1/claim-brief.md").read_text()
    assert "INT4 per-token/head" in report
    assert "Prompt n-gram" not in report
    assert "No KV-memory reduction claim" in report


def test_cache_runtime_refuses_silent_fallback_or_wrong_precision():
    from simulator.infrastructure.cache import observe_runtime

    text = "Using AttentionBackendEnum.TRITON_ATTN backend. dtype=torch.bfloat16, kv_cache_dtype=int4_per_token_head, GPU KV cache size: 123,456 tokens"
    assert observe_runtime(text, "treatment")["reported_cache_token_capacity"] == 123456
    for bad in [
        text.replace("TRITON_ATTN", "FLASHINFER"),
        text.replace("int4_per_token_head", "fp8"),
        text.replace("torch.bfloat16", "torch.float16"),
    ]:
        with pytest.raises(ValueError, match="unverified"):
            observe_runtime(bad, "treatment")


def test_frozen_corpus_budget_rejects_unbounded_grid_generation():
    with pytest.raises(ValidationError, match="record budget"):
        ExperimentSpec(
            project_id="P03",
            context_limit=131072,
            corpus_size=4096,
            cache_record_counts=[512, 768],
            cache_concurrencies=[1],
        )


def test_invalidated_campaign_blocks_claim_download_but_retains_evidence(tmp_path, monkeypatch):
    from simulator.benchmarks.storage import read_json, write_json

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    path = create(tmp_path, {"model": "test/model"}, spec(), "e" * 32)
    state = read_json(path / "state.json")
    state.update(phase="complete", outcome="invalidated")
    write_json(path / "state.json", state)
    (path / "evidence.zip").write_bytes(b"original evidence fixture")
    with TestClient(create_app(runs_base=tmp_path, catalog_dir=tmp_path / "catalog")) as client:
        prefix = "/api/infrastructure/campaigns/" + path.name + "/export/"
        assert client.get(prefix + "docx?output_id=P04-O1").status_code == 409
        assert client.get(prefix + "json?output_id=P04-O1").status_code == 409
        assert client.get(prefix + "zip").status_code == 200


def test_bf16_reference_rejects_checkpoint_fp8_default():
    from simulator.infrastructure.cache import observe_runtime

    log = "Using AttentionBackendEnum.TRITON_ATTN backend. dtype=torch.bfloat16, kv_cache_dtype=fp8_e4m3"
    with pytest.raises(ValueError, match="unverified"):
        observe_runtime(log, "reference")


def test_fixed_cache_cohort_is_identical_and_distributed(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace

    from simulator.infrastructure import driver

    seen = []

    async def request(client, url, model, item, spec, identity, key):
        await asyncio.sleep(0.001)
        seen.append((identity, item["id"], url))
        return dict(
            transport_ok=True, correct=item["id"] != 2, elapsed_s=0.1, ttft_s=0.01, usage={}
        )

    monkeypatch.setattr(driver, "request_one", request)
    engine = SimpleNamespace(
        replica_urls=["http://one", "http://two"], api_model_name="m", api_key="EMPTY"
    )
    settings = ExperimentSpec(
        project_id="P03", context_limit=65536, min_requests=16, cache_samples=16
    )
    result = asyncio.run(
        driver.measure_finite(
            engine, [{"id": i} for i in range(16)], settings, tmp_path / "requests.jsonl"
        )
    )
    assert sorted(int(i) for i, _, _ in seen) == list(range(16))
    assert all(url == engine.replica_urls[int(i) % 2] for i, _, url in seen)
    assert result["attempted"] == 16 and result["correct"] == 15
    assert not result["gates_passed"]
    with pytest.raises(ValidationError, match="Cache samples"):
        ExperimentSpec(project_id="P03", context_limit=65536, cache_samples=16, min_requests=30)
