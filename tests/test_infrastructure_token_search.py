import pytest

from simulator.headline_search import scrape_complete
from simulator.infrastructure.spec import ExperimentSpec
from simulator.infrastructure.token_search import aggregate_metrics, compare, usable


def spec(**kw):
    return ExperimentSpec(
        **(
            dict(
                project_id="P01",
                gpu_ids=list(range(8)),
                p01_token_search=True,
                p01_prefill_grid=[2, 4, 6],
                context_limit=32768,
            )
            | kw
        )
    )


def test_token_contract():
    spec()
    for kw in [
        dict(p01_goal_seek=True),
        dict(p01_frontend_format="mistral"),
        dict(p01_request_rates=[1, 2, 4]),
        dict(p01_streams=[512, 128]),
        dict(p01_token_shapes=[(32768, 128)]),
        dict(p01_token_shapes=[(128, 256), (128, 256)]),
        dict(p01_prefill_grid=[1], p01_placement_grid=["socket_local"]),
    ]:
        with pytest.raises(ValueError):
            spec(**kw)


def test_role_counter_aggregation_and_missing_worker():
    a = {"generation_tokens_total": 10, "prompt_tokens_total": 1000, "num_running": 1}
    b = {"generation_tokens_total": 100, "prompt_tokens_total": 1000, "num_running": 2}
    m = aggregate_metrics([a, b], ["prefill", "decode"])
    assert m["generation_tokens_total"] == 100
    assert m["prompt_tokens_total"] == 1000
    assert m["num_running"] == 3
    assert scrape_complete(m)
    assert not scrape_complete(aggregate_metrics([a], ["prefill", "decode"]))
    assert not scrape_complete(aggregate_metrics([{}], ["aggregated"]))


def test_no_false_peak_from_unsettled_or_failed_rungs():
    rung = dict(
        concurrency=128,
        out_tok_s=100,
        steady_state=True,
        samples=100,
        errors=0,
        no_content=0,
        scrape_gaps=0,
        ttft_p95_ms=200,
    )
    rows = [
        dict(block=0, shape="i128-o256", prefill=0, placement="socket_local", rungs=[rung]),
        dict(
            block=0,
            shape="i128-o256",
            prefill=4,
            placement="socket_local",
            rungs=[rung | dict(out_tok_s=120), rung | dict(out_tok_s=1000, steady_state=False)],
        ),
    ]
    c = compare(rows)
    assert c["peak_uplift"]["uplift_percent"] == pytest.approx(20)
    rows[0]["rungs"].append(rung | dict(out_tok_s=150, steady_state=False))
    assert compare(rows)["peak_uplift"] is None
    assert compare(rows)["unresolved_baselines"]
    assert not usable(rung | dict(errors=10))
    assert not usable(rung | dict(samples=0))
    assert compare(rows[1:])["peak_uplift"] is None


def test_limit_preflight_raises_only_soft_limit(monkeypatch):
    import simulator.infrastructure.token_search as t

    state = [1024, 1048576]
    monkeypatch.setattr(t.resource, "getrlimit", lambda _: tuple(state))
    monkeypatch.setattr(t.resource, "setrlimit", lambda _, v: state.__setitem__(slice(None), v))
    assert t.ensure_file_limit(8192)["soft"] == 17408
    assert state[1] == 1048576
    state[:] = [1024, 2048]
    with pytest.raises(RuntimeError):
        t.ensure_file_limit(8192)


@pytest.mark.parametrize(
    "failure_type", [RuntimeError, __import__("httpx").ConnectError, TimeoutError]
)
def test_persistent_launch_and_retry_reuses_finished_shapes(tmp_path, monkeypatch, failure_type):
    import asyncio
    import json

    from simulator.infrastructure import token_worker as w

    async def idle(*args):
        pass

    monkeypatch.setattr(w, "wait_idle", idle)
    from simulator.infrastructure.campaign import create

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    settings = spec(
        paired_blocks=1,
        p01_prefill_grid=[2],
        p01_token_shapes=[(128, 256), (4096, 256)],
        p01_streams=[128, 512],
    )
    path = create(tmp_path, {"model": "test/model"}, settings, "e" * 32)
    manifest = json.loads((path / "manifest.json").read_text())
    events = []

    class Fake:
        worker_roles = []

        def __init__(self, *args):
            self.number = sum(x[0] == "construct" for x in events)
            events.append(("construct", self.number))

        def shutdown(self):
            events.append(("stop", self.number))

    async def noop(*args):
        pass

    async def measure(cfg, cohort, **kw):
        engine = kw["existing_engine"].engine
        folder = kw["run_dir"]
        events.append(("measure", engine.number, folder.name))
        if engine.number == 0 and folder.name == "i4096-o256":
            raise failure_type("simulated serving failure")
        row = dict(
            concurrency=128,
            out_tok_s=100,
            steady_state=True,
            samples=100,
            errors=0,
            no_content=0,
            scrape_gaps=0,
            ttft_p95_ms=200,
        )
        f = folder / "headline_sweep.json"
        f.write_text(json.dumps(dict(status="ok", stop_reason="grid", rungs=[row], peak=row)))
        return f

    monkeypatch.setattr(w, "hardware_lease_path", lambda: tmp_path / "host.lock")
    monkeypatch.setattr(w, "ensure_file_limit", lambda *a: {})
    monkeypatch.setattr(
        w, "fingerprint", lambda *a: dict(snapshot=str(tmp_path), image={"id": "test"})
    )
    monkeypatch.setattr(w, "pin_model", lambda *a: None)
    monkeypatch.setattr(w, "capture_environment", lambda *a: {})
    monkeypatch.setattr(w, "OwnedDynamo", Fake)
    for name in ["launch_owned", "probe", "capture_metrics"]:
        monkeypatch.setattr(w, name, noop)
    for name in ["verify_transfer", "verify_worker_activity"]:
        monkeypatch.setattr(w, name, lambda *a: None)
    monkeypatch.setattr(w, "run_headline_sweep", measure)
    monkeypatch.setattr(w, "export", lambda *a: None)
    result = asyncio.run(w.run(path, manifest, settings))
    assert result["phase"] == "complete", result
    assert len(result["results"]) == 4
    assert len(result["attempts"]) == 1
    assert ("measure", 1, "i128-o256") not in events
    assert events.index(("stop", 0)) < events.index(("construct", 1))
    from simulator.infrastructure.token_report import render

    render(path, manifest, result, compare(result["results"]))
    assert len(list(path.glob("outputs/*/white-paper.docx"))) == 3
    assert (path / "token-curves.csv").exists()


def test_recovery_settings_are_pinned_and_symmetric():
    import json

    from simulator.infrastructure.dynamo import arm_config
    from simulator.infrastructure.dynamo_runtime import commands

    settings = spec(
        p01_runtime_image="sha256:" + "a" * 64,
        p01_attention_backend="TRITON_ATTN",
        p01_chat_processor="vllm",
        p01_frontend_format="mistral",
        p01_kv_lease_s=300,
    )
    for arm in ("baseline", "treatment"):
        assert (
            arm_config({"engine": {}}, settings, arm)["engine"]["gpu_image"]
            == settings.p01_runtime_image
        )
        plan = dict(
            port=19200,
            model="/snapshot",
            served_model_name="model",
            context_limit=32768,
            max_num_seqs=256,
            gpu_memory_utilization=0.9,
            kv_cache_dtype="fp8",
            namespace="test",
            gpu_ids=[0, 1],
            prefill_workers=1,
            arm=arm,
            transfer_settings=settings.model_dump(),
        )
        assert commands(plan)[0][2]["DYN_CHAT_PROCESSOR"] == "vllm"
        assert commands(plan)[0][1][-4:] == [
            "--tokenizer-mode",
            "mistral",
            "--config-format",
            "mistral",
        ]
        for _, argv, _ in commands(plan)[1:]:
            assert argv[argv.index("--attention-backend") + 1] == "TRITON_ATTN"
            if arm == "treatment":
                assert json.loads(argv[-1])["kv_connector_extra_config"]["kv_lease_duration"] == 300
    with pytest.raises(ValueError):
        spec(p01_runtime_image="mutable:latest")


def test_drain_requires_complete_idle_scrapes(monkeypatch):
    import asyncio

    from simulator.infrastructure import token_search as t

    samples = iter(
        [
            {},
            {"num_running": 1, "queue_depth": 0},
            {"num_running": 0, "queue_depth": 0},
            {"num_running": 0, "queue_depth": 0},
        ]
    )
    observed = []

    def metrics(_):
        item = next(samples)
        observed.append(item)
        return {"replicas_total": 8, "replicas_scraped": 8} | item

    async def sleep(_):
        pass

    monkeypatch.setattr(t.CounterView, "get_metrics", metrics)
    monkeypatch.setattr(t.asyncio, "sleep", sleep)
    asyncio.run(t.wait_idle(object(), 10))
    assert len(observed) == 4
