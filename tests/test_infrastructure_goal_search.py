import asyncio
import json

import httpx
import pytest

from simulator.infrastructure.driver import request_one
from simulator.infrastructure.goal_search import compare_goal, next_rate, windows
from simulator.infrastructure.spec import ExperimentSpec
from simulator.infrastructure.workload import build_corpus


def settings(**updates):
    return ExperimentSpec(
        **(
            dict(
                project_id="P01",
                gpu_ids=list(range(4)),
                paired_blocks=1,
                p01_goal_seek=True,
                p01_document_grid=[64],
                p01_answer_grid=[4, 8],
                p01_prefill_grid=[1, 2],
                p01_request_rates=[1, 2, 4, 6, 8],
                context_limit=32768,
            )
            | updates
        )
    )


def test_shape_contract_checks_context_output_and_budget():
    settings()
    for update in [
        dict(p01_prefill_grid=[]),
        dict(p01_request_rates=[1, 2]),
        dict(p01_answer_grid=[8, 4]),
        dict(p01_answer_grid=[16]),
        dict(p01_document_grid=[256]),
        dict(corpus_size=4096),
        dict(p01_document_grid=[4], p01_answer_grid=[8]),
    ]:
        with pytest.raises(ValueError):
            settings(**update)


def test_adaptive_boundary_cannot_loop_or_invent_measurements():
    grid = [1, 2, 4, 6, 8]
    for threshold in [0, 1, 2, 4, 6, 8, 10]:
        observed = {}
        while (rate := next_rate(grid, observed)) is not None:
            assert rate not in observed and rate in grid
            observed[rate] = rate <= threshold
        passing = [r for r, ok in observed.items() if ok]
        assert max(passing, default=0) == max([r for r in grid if r <= threshold], default=0)
        assert len(observed) <= len(grid)


def measured_rows(spec):
    rows = []
    for block in range(spec.paired_blocks):
        for prefill in [0, *spec.p01_prefill_grid]:
            for point in windows(spec, dict(block=block, prefill=prefill), rows):
                # Aggregate passes 2, fails 4; treatment passes 6, fails 8.
                passed = point["rate"] <= (2 if prefill == 0 else 6)
                rows.append(
                    dict(
                        block=block,
                        prefill=prefill,
                        shape=point["id"],
                        rate=point["rate"],
                        metrics=dict(gates_passed=passed, latency_p95_s=10 if prefill == 0 else 7),
                    )
                )
    return rows


def test_peaks_require_matched_shapes_completed_blocks_and_baseline_boundary():
    spec = settings(paired_blocks=2)
    rows = measured_rows(spec)
    result = compare_goal(rows, spec)
    assert result["complete_search"]
    peak = result["peak_capacity_uplift"]
    assert peak["observed_uplift_percent"] == 200
    assert peak["conservative_uplift_percent"] == 100
    assert result["peak_absolute_capacity"]["passing_rate"] == 6
    assert result["peak_latency_reduction"]["reduction_percent"] == pytest.approx(30)
    assert not compare_goal(rows[:-1], spec)["complete_search"]
    assert compare_goal([r for r in rows if r["prefill"]], spec)["peak_capacity_uplift"] is None
    with pytest.raises(ValueError):
        compare_goal(rows + [rows[0]], spec)
    for row in rows:
        row["metrics"]["gates_passed"] = True
    assert compare_goal(rows, spec)["peak_capacity_uplift"] is None


def test_answer_shape_controls_schema_and_exact_grading():
    spec = settings()
    item = build_corpus(1, 64, 42, 8)[0]

    def handle(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["minItems"] == schema["maxItems"] == 8
        output = json.dumps(item["expected"])
        chunk = dict(choices=[dict(delta=dict(content=output), finish_reason="stop")])
        return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await request_one(client, "http://test", "m", item, spec, "one")

    assert asyncio.run(run())["correct"]


def test_campaign_freezes_every_shape_and_validates_tampering(tmp_path, monkeypatch):
    from simulator.infrastructure.campaign import create, verify

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    path = create(tmp_path, {"model": "test/model"}, settings(corpus_size=16), "c" * 32)
    manifest = verify(path)
    corpus = json.loads((path / "corpus.json").read_text())
    assert set(corpus) == {"d64-a4", "d64-a8"}
    assert len(corpus["d64-a8"][0]["expected"]) == 8
    assert len(manifest["schedule"]) == 3
    assert manifest["goal_search"]["objective"]
    corpus["d64-a8"][0]["prompt"] = "altered"
    (path / "corpus.json").write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="workload changed"):
        verify(path)


def test_headroom_scales_with_demand_without_changing_engine_allocation():
    from simulator.infrastructure.goal_search import client_concurrency, rate_grid

    spec = settings(concurrency=96, duration_s=20, min_requests=10)
    assert client_concurrency(spec, 1) == 96
    assert client_concurrency(spec, 12) == 240
    assert client_concurrency(spec, 50) == 512
    assert rate_grid(spec) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert spec.concurrency == 96


def test_client_throttling_cannot_establish_a_server_capacity_boundary():
    spec = settings()
    rows = measured_rows(spec)
    for row in rows:
        if row["prefill"] == 0 and not row["metrics"]["gates_passed"]:
            row["metrics"]["client_limit_detected"] = True
    result = compare_goal(rows, spec)
    assert result["peak_capacity_uplift"] is None
    assert all(e["boundary"] == "client_limited" for e in result["envelopes"] if e["prefill"] == 0)


def test_placement_cells_do_not_collide_or_share_a_search_boundary():
    spec = settings(
        gpu_ids=list(range(8)),
        p01_prefill_grid=[3],
        p01_placement_grid=["socket_local", "cross_socket"],
        p01_answer_grid=[4],
    )
    rows = []
    for prefill, placement in [(0, "socket_local"), (3, "socket_local"), (3, "cross_socket")]:
        cell = dict(block=0, prefill=prefill, placement=placement)
        for point in windows(spec, cell, rows):
            limit = 6 if prefill and placement == "socket_local" else 2
            rows.append(
                dict(
                    cell,
                    shape=point["id"],
                    rate=point["rate"],
                    metrics=dict(
                        gates_passed=point["rate"] <= limit, latency_p95_s=7 if prefill else 10
                    ),
                )
            )
    result = compare_goal(rows, spec)
    assert result["complete_search"]
    assert result["peak_capacity_uplift"]["placement"] == "socket_local"
    assert len(result["envelopes"]) == 3


def test_shared_curve_points_are_measured_even_when_upper_limit_passes():
    spec = settings(p01_answer_grid=[4], p01_prefill_grid=[1])
    rows = []
    seen = []
    for point in windows(spec, dict(block=0, prefill=0), rows):
        seen.append(point["rate"])
        rows.append(
            dict(
                block=0,
                prefill=0,
                shape=point["id"],
                rate=point["rate"],
                metrics=dict(gates_passed=True, latency_p95_s=10),
            )
        )
    assert seen == spec.p01_request_rates
    assert len(seen) == len(set(seen))


def test_non_monotonic_baseline_cannot_produce_capacity_headline():
    spec = settings()
    rows = measured_rows(spec)
    for row in rows:
        if row["prefill"] == 0 and row["rate"] == 6:
            row["metrics"]["gates_passed"] = True
    comparison = compare_goal(rows, spec)
    assert comparison["peak_capacity_uplift"] is None
    assert all(
        e["boundary"] == "non_monotonic" for e in comparison["envelopes"] if not e["prefill"]
    )


def test_cuda_diagnostics_cannot_generate_performance_peaks():
    spec = settings(p01_cuda_launch_blocking=True)
    result = compare_goal(measured_rows(spec), spec)
    assert result["diagnostic_only"]
    assert result["peak_capacity_uplift"] is None
    assert result["peak_absolute_capacity"] is None
    assert result["peak_latency_reduction"] is None


def test_expanded_client_headroom_keeps_engine_concurrency():
    from simulator.infrastructure.goal_search import client_concurrency

    spec = settings(
        concurrency=128, duration_s=180, request_timeout_s=120, p01_client_max_concurrency=8192
    )
    assert client_concurrency(spec, 40) == 4800
    assert spec.concurrency == 128
    assert client_concurrency(spec, 100) == 8192


def test_throughput_counts_correct_work_and_requires_complete_usage():
    from simulator.infrastructure.driver import summarize

    rows = [
        dict(
            transport_ok=True,
            correct=True,
            elapsed_s=3,
            ttft_s=0.2,
            usage=dict(prompt_tokens=100, completion_tokens=20),
        ),
        dict(
            transport_ok=True,
            correct=False,
            elapsed_s=4,
            ttft_s=0.3,
            usage=dict(prompt_tokens=200, completion_tokens=40),
        ),
        dict(transport_ok=False, correct=False, elapsed_s=5, ttft_s=None, usage=None),
    ]
    m = summarize(rows, 10, settings())
    assert m["completed_requests_per_s"] == 0.2
    assert m["correct_requests_per_s"] == 0.1
    assert m["output_tokens_per_s"] == 6
    assert m["total_tokens_per_s"] == 36
    assert m["attempted"] == 3
    rows[1]["usage"]["completion_tokens"] = True
    m = summarize(rows, 10, settings())
    assert m["output_usage_coverage"] == 0.5
    assert m["output_tokens_per_s"] is None
    assert m["total_tokens_per_s"] is None
    assert m["input_tokens_per_s"] == 30
