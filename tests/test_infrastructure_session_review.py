import asyncio
from types import SimpleNamespace

import pytest

from simulator.infrastructure.session_capacity import make_trace
from simulator.infrastructure.session_saturation import (
    cache_work_metrics,
    next_cells,
    paired_block_interval,
    repeated_load_comparisons,
)
from simulator.infrastructure.spec import ExperimentSpec


def spec(**kw):
    return ExperimentSpec(
        **(
            dict(
                project_id="P02",
                session_mode="capacity",
                session_gpu_gb=None,
                warmup_s=0,
                duration_s=300,
                min_requests=2,
                session_population_grid=[4],
                session_arrival_rates=[1],
                session_prime_documents=True,
            )
            | kw
        )
    )


def test_explicit_rate_cannot_bypass_return_coverage():
    with pytest.raises(ValueError, match="four expected returns"):
        spec(session_population_grid=[128], duration_s=120)
    with pytest.raises(ValueError, match="four expected returns"):
        spec(session_prime_documents=False, session_population_grid=[64], duration_s=300)
    assert spec(session_population_grid=[64], duration_s=300)


def test_primed_trace_has_no_replacements_and_covers_every_document():
    s = spec()
    trace = make_trace(s, 4, 1, 0)
    assert trace == make_trace(s, 4, 1, 0)
    assert trace != make_trace(s, 4, 1, 1)
    assert all(e["generation"] == 0 and e["turn"] > 0 for e in trace)
    assert min(sum(e["slot"] == i for e in trace) for i in range(4)) >= 4
    with pytest.raises(ValueError, match="Frozen trace"):
        make_trace(s.model_copy(update={"duration_s": 0.01}), 4, 1, 0)


def test_capacity_multiples_require_empirical_capacity_metadata():
    with pytest.raises(ValueError, match="measured GPU capacity"):
        spec(session_capacity_multiples=[2])
    s = spec(
        session_population_grid=[8, 16],
        session_gpu_capacity_documents=4,
        session_capacity_multiples=[2, 4],
        session_capacity_evidence="calibration.json",
    )
    assert s.session_population_grid == [8, 16]
    with pytest.raises(ValueError, match="must match"):
        spec(
            session_population_grid=[9],
            session_gpu_capacity_documents=4,
            session_capacity_multiples=[2],
            session_capacity_evidence="calibration.json",
        )


def test_native_accounting_excludes_external_substring_and_preempted_families():
    delta = {
        '0.prom:vllm:prefix_cache_queries_total{model="m"}': 1000,
        '0.prom:vllm:prefix_cache_hits_total{model="m"}': 100,
        '0.prom:vllm:external_prefix_cache_hits_total{model="m"}': 600,
        '0.prom:vllm:preempted_prefix_cache_hits_total{model="m"}': 900,
    }
    ev = [dict(worker_deltas=delta, cpu_backend="simple_cpu")]
    result = cache_work_metrics(ev, 10)["cache_work"]
    assert result["uncached_prefix_tokens"] == 300
    assert result["uncached_prefix_tokens_per_correct_request"] == 30
    assert result["uncached_prefix_share"] == 0.3
    assert result["recomputed_historical_share"] is None
    delta['0.prom:vllm:external_prefix_cache_hits_total{model="m"}'] = 1001
    assert not cache_work_metrics(ev, 10)["cache_work"]["accounting_valid"]
    del delta['0.prom:vllm:external_prefix_cache_hits_total{model="m"}']
    assert cache_work_metrics(ev, 10)["cache_work"]["uncached_prefix_tokens"] is None
    assert cache_work_metrics([], 10)["cache_work"]["uncached_prefix_tokens"] is None


def test_repeat_interval_uses_independent_pairs_and_never_one_block():
    assert paired_block_interval([(10, 20)])["throughput_change_percent"] is None
    assert paired_block_interval([(10, 20)] * 2)["bootstrap_95_percent"] is None
    r = paired_block_interval([(10, 20)] * 5)
    assert r["bootstrap_95_percent"] == [100, 100]
    assert not r["qualified"]
    with pytest.raises(ValueError):
        paired_block_interval([(float("nan"), 1)])


def test_failed_block_is_not_dropped_to_improve_summary():
    s = spec(paired_blocks=2)
    rows = [
        dict(
            population=4,
            rate=1,
            block=b,
            arm=a,
            metrics=dict(
                trace_sha256=str(b),
                stable_capacity=True,
                sustained_correct_requests_s=1,
                cpu_reload_observed=a == "treatment",
            ),
        )
        for b in range(2)
        for a in ("baseline", "treatment")
    ]
    assert repeated_load_comparisons(rows, s)[0]["independent_pairs"] == 2
    rows[-1]["metrics"]["stable_capacity"] = False
    r = repeated_load_comparisons(rows, s)[0]
    assert r["excluded_blocks"] == [1] and r["throughput_change_percent"] is None


def test_zero_refinement_still_extends_to_overload():
    s = spec(paired_blocks=1, session_refinement_rounds=0)
    rows = [
        dict(block=0, population=4, rate=1, arm=a, metrics={}) for a in ("baseline", "treatment")
    ]
    assert {r["rate"] for r in next_cells(rows, s, {})} == {2}


def test_diagnostic_fixture_keeps_records_and_question_fixed():
    from simulator.infrastructure.session_diagnostics import fixture

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: list(range(len(messages[0]["content"].split())))
    )
    a, b = [fixture(tokenizer, n, 42) for n in (8192, 32768)]
    assert a["expected"] == b["expected"]
    assert a["prompt"].split("Background note:")[0] == b["prompt"].split("Background note:")[0]
    assert a["prompt"].split("Return only")[1] == b["prompt"].split("Return only")[1]
    assert a["prompt_sha256"] != b["prompt_sha256"]
    assert all(__import__("json").dumps(r) in a["prompt"] for r in a["expected"])


def test_calibration_rejects_mismatch_and_requires_two_blocks(tmp_path):
    import hashlib
    import json

    from simulator.infrastructure.session_calibration import load_calibration

    file = tmp_path / "calibration.json"
    s = spec(
        session_population_grid=[8],
        session_capacity_multiples=[2],
        session_gpu_capacity_documents=4,
        session_capacity_evidence=str(file),
    )
    cfg = {"engine": {"kv_cache_dtype": "bfloat16"}}
    assets = {"snapshot": "revision", "image": "digest"}
    record = dict(
        assets=assets,
        gpu_ids=s.gpu_ids,
        context_limit=s.context_limit,
        document_records=s.document_records,
        configuration_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        points=[
            dict(
                block=b,
                population=n,
                repeat_gpu_hit_fraction=hit,
                arm="baseline",
                quality_fraction=1,
                measured_returns_per_document=8,
                evidence_sha256="a" * 64,
            )
            for b in range(2)
            for n, hit in [(4, 0.99), (8, 0.5)]
        ],
    )
    file.write_text(json.dumps(record))
    assert load_calibration(s, assets, cfg)["record"] == record
    with pytest.raises(ValueError, match="does not match"):
        load_calibration(s, {"snapshot": "other"}, cfg)
    record["points"] = record["points"][:2]
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="repeated hot"):
        load_calibration(s, assets, cfg)


def test_priming_stops_on_bad_answers_and_preserves_requests(tmp_path, monkeypatch):
    import json

    from simulator.infrastructure import session_capacity as mod

    async def wrong(*args):
        return dict(
            transport_ok=True, correct=False, output="[]", elapsed_s=0.1, ttft_s=0.1, usage={}
        )

    async def capture(*args):
        pytest.fail("Bad priming must not enter steady measurement")

    monkeypatch.setattr(mod, "request_one", wrong)
    e = SimpleNamespace(
        replica_urls=["http://test"], api_model_name="m", api_key="EMPTY", capture=capture
    )
    with pytest.raises(RuntimeError, match="priming failed"):
        asyncio.run(mod.prime_documents(e, spec(), 4, 0, tmp_path))
    assert len((tmp_path / "priming-requests.jsonl").read_text().splitlines()) == 4
    assert json.loads((tmp_path / "priming-metrics.json").read_text())["correct"] == 0


def test_confirmation_freezes_rates_and_requires_five_pairs():
    with pytest.raises(ValueError, match="five pairs"):
        spec(mode="confirmation", duration_s=600, paired_blocks=2)
    s = spec(mode="confirmation", duration_s=600, paired_blocks=5)
    rows = [
        dict(block=0, population=4, rate=1, arm=a, metrics={}) for a in ("baseline", "treatment")
    ]
    assert next_cells(rows, s, {}) == []


def test_diagnostics_exercises_real_request_contract(monkeypatch, tmp_path):
    import json

    import httpx

    from simulator.infrastructure import session_diagnostics as mod

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: list(range(len(messages[0]["content"].split())))
    )
    items = {
        (target, seed): mod.fixture(tokenizer, target, seed)
        for target in (8192, 32768, 65536, 96000)
        for seed in range(10)
    }
    expected = {v["prompt"]: v["expected"] for v in items.values()}
    monkeypatch.setattr(mod, "fixture", lambda tokenizer, target, seed: items[target, seed])
    original = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "medium"
        answer = json.dumps(expected[body["messages"][0]["content"]])
        chunk = {"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]}
        return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

    monkeypatch.setattr(
        mod.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    plan = dict(
        model_revision="pinned",
        runtime_digest="pinned",
        kv_dtype="bfloat16",
        prefix_caching=False,
        model="openai/gpt-oss-120b",
        url="http://test/v1",
        mode="isolated",
        seeds=list(range(10)),
        api_key="do-not-store",
    )
    output = tmp_path / "diagnosis"
    result = asyncio.run(mod.run(plan, tokenizer, output))
    assert len(result) == 4 and all(r["correct"] == r["attempted"] == 10 for r in result)
    assert "do-not-store" not in (output / "plan.json").read_text()


def test_padded_document_reaches_context_and_keeps_prefix_stable():
    import json

    from simulator.infrastructure.session_capacity import document_item, padding_repeats

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: list(range(len(messages[0]["content"].split())))
    )
    s = spec(
        session_document_padding=True,
        session_context_tokens=4000,
        document_records=16,
        context_limit=8192,
        max_output_tokens=256,
    )
    n = padding_repeats(tokenizer, s, 0)
    padded = s.model_copy(update={"session_document_pad_repeats": n})
    a = document_item(padded, dict(slot=0, generation=0, turn=0), 0)
    b = document_item(padded, dict(slot=0, generation=0, turn=3), 0)
    words = len(a["prompt"].split())
    assert 4000 <= words < 4000 + 32
    assert a["prompt"].split("\nQuestion")[0] == b["prompt"].split("\nQuestion")[0]
    assert all(json.dumps(r) in b["prompt"] for r in b["expected"])
    # Records sit after the padding, adjacent to the question.
    assert a["prompt"].rfind("Background note") < a["prompt"].find('{"record_id"')
    with pytest.raises(ValueError, match="output headroom"):
        spec(session_document_padding=True, session_context_tokens=120000, context_limit=125000, max_output_tokens=8192)


def test_deadline_failures_count_as_overload(monkeypatch, tmp_path):
    from simulator.infrastructure import session_capacity as mod

    async def never_finishes(*args):
        await asyncio.sleep(10)

    monkeypatch.setattr(mod, "request_one", never_finishes)
    s = spec(duration_s=300, min_requests=1).model_copy(
        update=dict(duration_s=0.2, request_timeout_s=0.05, concurrency=64, session_slo_grid_s=[0.05])
    )
    trace = [dict(at_s=0.002 * i, slot=i % 4, generation=0, turn=i // 4 + 1) for i in range(60)]
    e = SimpleNamespace(replica_urls=["http://test"], api_model_name="m", api_key="EMPTY")
    m = asyncio.run(mod.replay(e, s, trace, 0, tmp_path))
    assert m["error_counts"]["deadline_including_admission"] == 60
    assert m["final_window_deadline_failures"] > 0
    assert m["overload_observed"] is True and m["stable_capacity"] is False
