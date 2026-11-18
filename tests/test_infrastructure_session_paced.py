import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from simulator.infrastructure import session_paced as paced
from simulator.infrastructure.session_capacity import compare_capacity
from simulator.infrastructure.session_cells import export_cells
from simulator.infrastructure.spec import ExperimentSpec
from simulator.infrastructure.wall_power import sample_power


def spec(**kw):
    args = dict(
        project_id="P02",
        session_mode="capacity",
        session_gpu_gb=None,
        duration_s=1800,
        warmup_s=0,
        min_requests=2,
        paired_blocks=1,
        session_population_grid=[56, 96],
        session_prime_documents=True,
        session_pacing="think",
        session_think_s=300,
        concurrency=128,
        document_records=16,
        gpu_ids=[0, 4],
    )
    return ExperimentSpec(**(args | kw))


def test_paced_contract():
    s = spec()
    # One cadence per cell, derived from the think time.
    assert s.session_rate_grid == [pytest.approx(1 / 300)]
    assert spec(session_pacing="surge", session_population_grid=[224], concurrency=256)
    for kw in [
        dict(session_prime_documents=False),
        dict(session_arrival_rates=[0.3]),
        dict(session_workload="knowledge_worker"),
        dict(concurrency=64),  # fewer client streams than users
        dict(duration_s=600),  # two expected questions per user, below the minimum returns
        dict(session_pacing="surge", session_surge_shares=[0.6, 0.6]),
        dict(session_pacing="surge", session_surge_shares=[0.001]),
    ]:
        with pytest.raises(ValidationError):
            spec(**kw)


def fake(**kw):
    args = dict(
        seed=42,
        duration_s=2.0,
        session_pacing="think",
        session_think_s=0.2,
        session_surge_shares=[0.25, 0.5],
        session_surge_quiet_s=0.05,
        session_surge_window_s=0.05,
        session_ttft_slo_s=5,
        session_document_pad_repeats=0,
        session_answer_records=2,
        document_records=16,
        request_timeout_s=1.0,
        response_slo_s=60,
        concurrency=64,
        min_requests=1,
        min_quality=0.95,
        max_error_fraction=0.01,
        min_slo_fraction=0.95,
        project_id="P02",
    )
    return SimpleNamespace(**(args | kw))


ENGINE = SimpleNamespace(
    replica_urls=["http://a/v1", "http://b/v1"], api_model_name="m", api_key="k"
)


def run(spec_, population, tmp_path, monkeypatch, service_s=0.01, slow=()):
    seen = []

    async def request_one(client, url, model, item, spec, request_id, api_key="EMPTY"):
        slot = int(request_id.split("-")[0])
        seen.append((slot, url))
        await asyncio.sleep(10 if slot in slow else service_s)
        return dict(
            request_id=request_id,
            task_id=item["id"],
            transport_ok=True,
            correct=True,
            ttft_s=service_s / 2,
            elapsed_s=service_s,
            output="",
            usage=dict(prompt_tokens=100, completion_tokens=10),
        )

    monkeypatch.setattr(paced, "request_one", request_one)
    metrics = asyncio.run(paced.replay_paced(ENGINE, spec_, population, 0, tmp_path / "cell"))
    rows = [json.loads(x) for x in (tmp_path / "cell" / "requests.jsonl").read_text().splitlines()]
    return metrics, rows, seen


def test_think_users_ask_one_question_at_a_time(tmp_path, monkeypatch):
    metrics, rows, seen = run(fake(), 8, tmp_path, monkeypatch)
    assert metrics["pacing"] == "think" and metrics["questions_per_user_min"] >= 1
    assert {slot for slot, _ in seen} == set(range(8))
    # Fixed affinity, the same mapping priming uses.
    assert all(url == ENGINE.replica_urls[slot % 2] for slot, url in seen)
    for slot in range(8):
        mine = sorted((r for r in rows if r["session"] == slot), key=lambda r: r["arrival_s"])
        assert [r["turn"] for r in mine] == list(range(1, len(mine) + 1))
        # The next question never starts before the previous answer finished.
        assert all(a["finish_s"] <= b["arrival_s"] for a, b in zip(mine, mine[1:], strict=False))
        assert all(r["arrival_s"] < 2.0 for r in mine)
    assert metrics["waits"]["within_first_token"]["2"] == 1
    assert metrics["stable_capacity"] and metrics["service_capacity"]
    # The plan digest is identical for both arms; arrival times are not part of it.
    again, _, _ = run(fake(), 8, tmp_path / "again", monkeypatch, service_s=0.05)
    assert again["trace_sha256"] == metrics["trace_sha256"]


def test_timeouts_stay_in_every_fraction(tmp_path, monkeypatch):
    metrics, rows, _ = run(fake(), 4, tmp_path, monkeypatch, slow={0, 1})
    late = [r for r in rows if r["session"] in (0, 1)]
    assert late and all(r["error"] == paced.TIMEOUT for r in late)
    waits = metrics["waits"]
    assert 0 < waits["timeout_fraction"] < 1
    assert waits["within_first_token"]["60"] == pytest.approx(waits["correct_fraction"])
    assert waits["first_token_p95_s"] == 1.0  # censored at the deadline, not dropped
    assert metrics["overload_observed"] and not metrics["service_capacity"]


def test_surges_release_disjoint_users(tmp_path, monkeypatch):
    s = fake(session_pacing="surge")
    plan = paced.pacing_plan(s, 16, 0)
    groups = [g["users"] for g in plan["surges"]]
    assert [len(g) for g in groups] == [4, 8]
    assert not set(groups[0]) & set(groups[1])
    metrics, rows, _ = run(s, 16, tmp_path, monkeypatch)
    assert len(rows) == 12 and all(r["turn"] == 1 for r in rows)
    assert [x["users"] for x in metrics["surges"]] == [4, 8]
    assert [x["share"] for x in metrics["surges"]] == [0.25, 0.5]
    assert all(0 < x["time_to_clear_s"] < 1 and x["questions"] == x["users"] for x in metrics["surges"])
    # Each surge waits out the quiet period first.
    assert metrics["surges"][1]["start_s"] > metrics["surges"][0]["start_s"] + 0.05


def cell(arm, users, share, block=0):
    within = {str(t): share for t in paced.WAIT_TARGETS_S}
    return dict(
        block=block,
        arm=arm,
        population=users,
        metrics=dict(waits=dict(within_first_token=within, correct_fraction=0.99)),
    )


def test_users_served_needs_every_smaller_rung_to_pass():
    s = spec()
    results = [
        cell("baseline", 56, 0.99),
        cell("baseline", 96, 0.4),
        cell("baseline", 144, 0.99),  # a pass above a failed rung does not count
        cell("treatment", 56, 0.99),
        cell("treatment", 96, 0.99),
        cell("treatment", 144, 0.99),
    ]
    out = compare_capacity(results, s)
    row = next(x for x in out["users_served"] if x["first_token_target_s"] == 5)
    assert row["arms"]["baseline"] == dict(
        users_served=56, tested=[56, 96, 144], limit_resolved=True
    )
    # The tier passed its top rung, so its limit is open and stays marked as such.
    assert row["arms"]["treatment"]["users_served"] == 144
    assert not row["arms"]["treatment"]["limit_resolved"]
    assert row["users_ratio"] == pytest.approx(144 / 56)
    assert out["by_cadence"] == [] and not out["qualified"]


def test_cells_export_reads_only_evidence(tmp_path):
    t0 = 1_000_000.0
    directory = tmp_path / "cell-a"
    directory.mkdir()
    rows = [
        dict(arrival_s=20 + i, finish_s=22 + i, ttft_s=0.5, transport_ok=True, correct=True)
        for i in range(10)
    ]
    (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def native(at, queries, gpu, host, cached, evicted):
        values = {
            'vllm:prefix_cache_queries_total{engine="0"}': queries,
            'vllm:prefix_cache_hits_total{engine="0"}': gpu,
            'vllm:external_prefix_cache_hits_total{engine="0"}': host,
            'vllm:prefix_cache_queries_created{engine="0"}': 5e9,
        }
        pool = [dict(cached_blocks=cached, total_blocks=100, evicted_blocks=evicted, replica=0)]
        return dict(timestamp=t0 + at, samples=[dict(replica=0, values=values)], cpu_pool=pool)

    samples = [native(-1, 0, 0, 0, 10, 3), native(25, 1000, 400, 500, 40, 3), native(35, 2000, 800, 1100, 60, 9)]
    (directory / "native-queue-cache.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in samples) + '{"partial'
    )
    (directory / "telemetry.jsonl").write_text(
        json.dumps(
            dict(
                timestamp=t0 + 25,
                host_cpu_pct=12.0,
                gpu_devices=[dict(sm_util_pct=80.0), dict(sm_util_pct=40.0)],
            )
        )
        + "\n"
    )
    (directory / "power.csv").write_text(f"{t0 + 21:.3f},3000.0,\n{t0 + 26:.3f},,URLError\n{t0 + 28:.3f},3200.0,\n")
    waits = paced.wait_views(
        [dict(r, elapsed_s=2.0) for r in rows], SimpleNamespace(request_timeout_s=300)
    )
    metrics = dict(
        waits=waits,
        started_at=t0,
        final_window_completions=10,
        sustained_correct_requests_s=0.5,
        correct=10,
        cpu_reload_observed=True,
        cache_work=dict(
            prefix_query_tokens=2000,
            gpu_hit_tokens=800,
            cpu_hit_tokens=1100,
            uncached_prefix_tokens=100,
            accounting_valid=True,
        ),
    )
    (tmp_path / "state.json").write_text(
        json.dumps(
            dict(results=[dict(block=0, arm="treatment", population=96, metrics=metrics, evidence="cell-a")])
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                spec=dict(
                    session_pacing="surge",
                    session_context_tokens=96000,
                    session_cpu_gb=192,
                    gpu_ids=[0, 1],
                    session_situation="S2",
                )
            )
        )
    )
    (record,) = export_cells(tmp_path)
    assert json.loads((tmp_path / "cells.json").read_text()) == [record]
    assert record["situation"] == "S2" and record["arm"] == "xeon_tier" and record["users"] == 96
    assert record["host_cache_gib_per_gpu"] == 192
    assert record["wait"]["first_token_p50_s"] == 0.5
    assert record["cache"]["rebuilt_tokens_per_question"] == 10
    assert record["cache"]["host_cache_used_fraction_peak"] == 0.6
    assert record["cache"]["host_evicted_blocks"] == 6  # relative to the pre-measurement sample
    assert record["machine"]["gpu_busy_pct"] == 60 and record["machine"]["host_cpu_busy_pct"] == 12
    assert record["machine"]["wall_power_w"] == 3100
    series = json.loads((directory / "timeline.json").read_text())
    at20 = next(b for b in series if b["at_s"] == 20)
    assert at20["asked"] == 10 and at20["answered"] == 8 and at20["waiting"] == 2
    assert (at20["gpu_hit_tokens"], at20["host_hit_tokens"], at20["rebuilt_tokens"]) == (400, 500, 100)
    assert next(b for b in series if b["at_s"] == 30)["rebuilt_tokens"] == 0


def test_power_sampler_survives_failed_reads(tmp_path):
    env = tmp_path / "idrac.env"
    env.write_text("IDRAC_HOST=h\nIDRAC_USER=u\nIDRAC_PASS=p\n")
    calls = []

    def read(values):
        calls.append(values["IDRAC_HOST"])
        if len(calls) == 2:
            raise OSError("down")
        return 3000.0

    async def go():
        task = asyncio.create_task(sample_power(env, tmp_path / "power.csv", 0.01, read))
        await asyncio.sleep(0.08)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())
    lines = (tmp_path / "power.csv").read_text().splitlines()
    assert len(lines) >= 3 and lines[0].endswith(",3000.0,") and lines[1].endswith(",,OSError")


def test_paced_schedule_honours_the_chosen_arms(tmp_path, monkeypatch):
    from simulator.infrastructure.campaign import create

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "m"}},
    )
    s = spec(session_population_grid=[336, 480, 640], concurrency=1024, session_cache_arms=["treatment"])
    path = create(tmp_path, {"model": "m"}, s, "d" * 32)
    schedule = json.loads((path / "manifest.json").read_text())["schedule"]
    assert [(c["arm"], c["population"]) for c in schedule] == [
        ("treatment", 336),
        ("treatment", 480),
        ("treatment", 640),
    ]
    assert all(c["per_session_rate"] == pytest.approx(1 / 300) for c in schedule)
    assert schedule[0]["rate"] == pytest.approx(336 / 300)


def test_connector_reports_host_pool_occupancy_and_evictions(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    name = "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector"
    module = ModuleType(name)

    class Pool:
        num_gpu_blocks = 101
        cached_block_hash_to_block = {"a": 1, "b": 2}

        def _maybe_evict_cached_block(self, block):
            return block == "cached"

    class Base:
        def __init__(self):
            self.worker_handler = None
            self.scheduler_manager = SimpleNamespace(cpu_block_pool=Pool())

        def build_connector_meta(self, output):
            return "meta"

    module.SimpleCPUOffloadConnector = Base
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("CAPSIM_CPU_EVIDENCE_DIR", str(tmp_path))
    location = Path(__file__).parents[1] / "simulator/infrastructure/native_cpu_connector.py"
    loaded = importlib.util.spec_from_file_location("test_native_pool", location)
    wrapper = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(wrapper)
    connector = wrapper.ObservedCPUOffloadConnector()
    pool = connector.scheduler_manager.cpu_block_pool
    # The pool's own answers are unchanged; only cached-block evictions are counted.
    assert pool._maybe_evict_cached_block("cached") is True
    assert pool._maybe_evict_cached_block("empty") is False
    connector._pool_written = 0
    assert connector.build_connector_meta(None) == "meta"
    (record,) = [json.loads(p.read_text()) for p in tmp_path.glob("cpu-pool-*.json")]
    assert (record["cached_blocks"], record["total_blocks"], record["evicted_blocks"]) == (2, 100, 1)
    assert not list(tmp_path.glob("native-cpu-*.json"))  # the copy-counter contract is untouched
