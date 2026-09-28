import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from simulator.infrastructure.call_report import compare_calls
from simulator.infrastructure.call_worker import process_call
from simulator.infrastructure.calls import call_corpus, grade_summary, word_error_rate
from simulator.infrastructure.search_report import compare_search
from simulator.infrastructure.spec import ExperimentSpec


def test_search_rejects_invalid_splits_rates_and_minimum_samples():
    args = dict(
        project_id="P01",
        gpu_ids=list(range(8)),
        paired_blocks=1,
        p01_prefill_grid=[2, 3, 4, 5],
        p01_request_rates=[2, 4, 6],
    )
    ExperimentSpec(**args)
    for updates in [
        dict(p01_prefill_grid=[2, 8]),
        dict(p01_request_rates=[4, 2]),
        dict(p01_request_rates=[]),
        dict(p01_request_rates=[0.01]),
    ]:
        with pytest.raises(ValueError):
            ExperimentSpec(**(args | updates))


def test_search_does_not_drop_failures_or_extrapolate():
    spec = ExperimentSpec(
        project_id="P01",
        gpu_ids=[0, 1, 2, 3],
        paired_blocks=1,
        p01_prefill_grid=[1, 2],
        p01_request_rates=[1, 2],
    )
    rows = [
        dict(
            block=0,
            prefill=p,
            rate=r,
            metrics=dict(gates_passed=r == 1, ttft_p95_s=1, latency_p95_s=4 if p == 0 else 3),
        )
        for p in [0, 1, 2]
        for r in [1, 2]
    ]
    result = compare_search(rows, spec)
    assert result["complete_grid"] and result["highest_passing_rates"] == {"0": 1, "1": 1, "2": 1}
    assert len(result["matched_rate_latency"]) == 2
    assert result["matched_rate_latency"][0]["ttft_reduction_percent"] == 0
    assert result["matched_completion_latency"][0]["latency_reduction_percent"] == 25
    assert not compare_search(rows[:-1], spec)["complete_grid"]
    with pytest.raises(ValueError):
        compare_search(rows + [rows[0]], spec)


def test_call_quality_rejects_wrong_summary_and_bad_transcript():
    row = call_corpus()[0]
    assert word_error_rate(row["reference"], row["reference"].upper()) == 0
    assert word_error_rate("one two three", "one wrong three") == pytest.approx(1 / 3)
    assert grade_summary(row["expected"], row["expected"])
    assert not grade_summary(row["expected"] | {"deadline": "friday"}, row["expected"])
    assert not grade_summary(row["expected"] | {"extra": "invented"}, row["expected"])


def test_pipeline_feeds_asr_text_not_reference_and_gates_both_stages():
    row = call_corpus()[0]
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        if request.url.path == "/transcribe":
            return httpx.Response(
                200,
                json=dict(
                    text="incorrect transcript", audio_seconds=30, server_s=0.1, asr_queue_s=0
                ),
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(row["expected"])}, "finish_reason": "stop"}
                ]
            },
        )

    spec = ExperimentSpec(
        project_id="P06", gpu_ids=[0, 1], tenant_workflow="post_call", min_requests=1
    )
    llm = SimpleNamespace(
        cfg=SimpleNamespace(host="localhost", port=19000), api_model_name="m", api_key="EMPTY"
    )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await process_call(client, llm, row, spec)

    result = asyncio.run(run())
    assert result["transport_ok"] and result["summary_correct"] and not result["correct"]
    assert "incorrect transcript" in seen[1]["messages"][0]["content"]
    assert row["reference"] not in seen[1]["messages"][0]["content"]


def test_consolidation_claim_requires_passing_matched_demand():
    spec = ExperimentSpec(
        project_id="P06",
        gpu_ids=[0, 1],
        tenant_workflow="post_call",
        min_requests=1,
        paired_blocks=1,
        call_request_rates=[0.1],
    )
    rows = [
        dict(
            block=0,
            arm=a,
            rate=0.1,
            metrics=dict(
                gates_passed=a != "mig", qualified_audio_hours_per_hour=2, latency_p95_s=3
            ),
        )
        for a in ["dedicated", "shared", "mig"]
    ]
    result = compare_calls(rows, spec)
    assert result["complete_grid"] and len(result["passing_matched_rates"]) == 1
    assert result["passing_matched_rates"][0]["placement"] == "shared"
    rows[0]["metrics"]["gates_passed"] = False
    assert compare_calls(rows, spec)["passing_matched_rates"] == []


def test_report_reads_engine_only_and_full_launch_documents(tmp_path):
    from simulator.infrastructure.workflow_report import recorded_methods

    (tmp_path / "manifest.json").write_text(json.dumps({"created_at": 0}))
    block = tmp_path / "block-00-shared"
    block.mkdir()
    launch = block / "launch-resolved.yaml"
    launch.write_text("max_model_len: 8192\nkv_cache_dtype: fp8\n")
    direct = dict(recorded_methods(tmp_path))["Resolved common engine settings"]
    launch.write_text("engine:\n  max_model_len: 8192\n  kv_cache_dtype: fp8\n")
    assert dict(recorded_methods(tmp_path))["Resolved common engine settings"] == direct
    assert json.loads(direct)["max_model_len"] == 8192
