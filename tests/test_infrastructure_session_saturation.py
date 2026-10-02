import json
from types import SimpleNamespace

import pytest

from simulator.infrastructure.session_native import queue_delta, select_metrics
from simulator.infrastructure.session_saturation import (
    analyze_requests,
    compare_saturation,
    next_cells,
)


def spec():
    return SimpleNamespace(
        duration_s=10,
        min_requests=1,
        min_slo_fraction=0.95,
        request_timeout_s=600,
        session_slo_grid_s=[2, 10],
        session_arrival_rates=[1, 2],
        session_max_arrival_rate=16,
        session_refinement_rounds=2,
        paired_blocks=1,
        session_population_grid=[128],
        seed=42,
    )


def rows():
    return [
        dict(
            arrival_s=4,
            dispatch_s=4,
            finish_s=8,
            elapsed_s=4,
            session=0,
            task_id="a",
            transport_ok=True,
            correct=True,
        ),
        dict(
            arrival_s=6,
            dispatch_s=7,
            finish_s=12,
            elapsed_s=6,
            session=1,
            task_id="b",
            transport_ok=True,
            correct=True,
        ),
        dict(
            arrival_s=7,
            finish_s=607,
            elapsed_s=600,
            session=2,
            task_id="c",
            transport_ok=False,
            correct=False,
        ),
    ]


def cells(rates=(1, 2), saturated=False):
    return [
        dict(
            block=0,
            population=128,
            arm=arm,
            rate=rate,
            per_session_rate=rate / 128,
            metrics=dict(
                trace_sha256=str(rate),
                stable_capacity=rate == 1,
                saturation_observed=saturated and rate > 1,
                sustained_correct_requests_s=1,
                mean_outstanding_requests=2 if arm == "baseline" else 3,
                native_queue_mean_s=0.1 if rate == 1 else 4,
                cpu_reload_observed=arm == "treatment",
                slo_analysis=[dict(response_slo_s=n, passed=True) for n in (2, 10)],
            ),
        )
        for rate in rates
        for arm in ("baseline", "treatment")
    ]


def test_exact_occupancy_and_slo_keeps_timed_out_arrivals():
    result = analyze_requests(rows(), spec())
    assert result["mean_outstanding_requests"] == 2  # (3 + 4 + 3)/5
    assert result["mean_dispatched_requests"] == 1.2  # 3+3, timed out row not dispatched
    assert result["slo_analysis"][1]["accepted_fraction"] == 0.5
    assert not result["slo_analysis"][1]["passed"]
    assert result["resident_document_count"] is None


def test_extension_uses_same_new_rate_for_both_arms_and_stops_at_cap():
    s = spec()
    todo = next_cells(cells(), s, {})
    assert len(todo) == 2 and {c["rate"] for c in todo} == {4}
    assert {c["arm"] for c in todo} == {"baseline", "treatment"}
    assert not next_cells(cells((1, 16)), s, {})
    assert not compare_saturation(cells((1, 16)), s)["by_working_set"][0]["saturation_resolved"]


def test_refinement_requires_two_overload_points_not_one():
    s = spec()
    r = {}
    assert {c["rate"] for c in next_cells(cells((1, 2), True), s, r)} == {4}
    assert {c["rate"] for c in next_cells(cells((1, 2, 4), True), s, r)} == {1.5}
    assert r == {"0:128": 1}


def test_comparison_requires_native_queue_knee_and_matched_traces():
    s = spec()
    data = cells((1, 2, 4), True)
    group = compare_saturation(data, s)["by_working_set"][0]
    assert group["saturation_resolved"]
    assert group["slo_comparisons"][0]["concurrent_request_change_percent"] is None
    data[-1]["metrics"]["trace_sha256"] = "wrong"
    assert not compare_saturation(data, s)["by_working_set"][0]["saturation_resolved"]
    data[-1]["metrics"]["trace_sha256"] = "4"
    for r in data:
        r["metrics"]["native_queue_mean_s"] = None
    assert not compare_saturation(data, s)["by_working_set"][0]["saturation_resolved"]


def test_queue_histogram_delta_and_missing_residency(tmp_path):
    p = tmp_path / "replica-0"
    p.mkdir()

    def text(n):
        return f'vllm:request_queue_time_seconds_sum{{model="m"}} {n * 2}\nvllm:request_queue_time_seconds_count{{model="m"}} {n}\nvllm:request_queue_time_seconds_bucket{{le="4.0",model="m"}} {n}\n'

    (p / "before-worker-0.prom").write_text(text(10))
    (p / "after-worker-0.prom").write_text(text(20))
    r = queue_delta(tmp_path)
    assert r["native_queue_mean_s"] == 2 and r["native_queue_p95_bucket_upper_s"] == 4
    assert r["native_queue_samples"] == 10
    (p / "after-worker-0.prom").write_text(text(1))
    with pytest.raises(ValueError):
        queue_delta(tmp_path)
    with pytest.raises(ValueError):
        select_metrics("vllm:request_queue_time_seconds_sum NaN")


def test_saturation_manifest_uses_total_arrivals(tmp_path, monkeypatch):
    from simulator.infrastructure.campaign import create
    from simulator.infrastructure.spec import ExperimentSpec

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "m"}},
    )
    s = ExperimentSpec(
        project_id="P02",
        session_mode="capacity",
        session_gpu_gb=None,
        warmup_s=0,
        duration_s=300,
        min_requests=2,
        paired_blocks=1,
        session_population_grid=[4, 8],
        session_arrival_rates=[0.5, 1],
        session_max_arrival_rate=128,
        concurrency=2048,
    )
    path = create(tmp_path, {"model": "m"}, s, "d" * 32)
    schedule = json.loads((path / "manifest.json").read_text())["schedule"]
    assert len(schedule) == 8 and {r["rate"] for r in schedule} == {0.5, 1}
    assert all(r["per_session_rate"] == r["rate"] / r["population"] for r in schedule)


def test_deferred_waits_for_busy_and_two_idle_observations(tmp_path, monkeypatch):
    from simulator.benchmarks.storage import read_json, write_json
    from simulator.infrastructure import session_deferred as deferred

    p = tmp_path / "plan.json"
    write_json(p, {"api": "http://test"})
    monkeypatch.setattr(deferred, "idle_reasons", lambda plan: ["busy"])
    assert deferred.check(p, True)["phase"] == "waiting"
    monkeypatch.setattr(deferred, "idle_reasons", lambda plan: [])
    first = deferred.check(p)
    assert first["phase"] == "waiting" and first["idle_since"]
    monkeypatch.setattr(deferred.time, "time", lambda: first["idle_since"] + 121)
    assert deferred.check(p)["phase"] == "ready"
    (tmp_path / "hold").touch()
    assert deferred.check(p, True)["phase"] == "waiting"
    (tmp_path / "hold").unlink()
    state = read_json(tmp_path / "readiness.json")
    state["phase"] = "running"
    write_json(tmp_path / "readiness.json", state)
    assert deferred.check(p, True)["phase"] == "running"


def test_client_overload_refines_below_limit_instead_of_turning_load_up():
    s = spec()
    data = cells((1, 2, 4), True)
    for r in data:
        if r["rate"] == 4:
            r["metrics"].update(client_limited=True, saturation_observed=False)
    assert {r["rate"] for r in next_cells(data, s, {})} == {3}


def test_reanalysis_preserves_original_evidence(tmp_path):
    from simulator.benchmarks.storage import write_json
    from simulator.infrastructure.session_reanalyze import reanalyze
    from simulator.infrastructure.spec import ExperimentSpec

    s = ExperimentSpec(
        project_id="P02",
        session_mode="capacity",
        session_gpu_gb=None,
        warmup_s=0,
        duration_s=300,
        min_requests=1,
        session_arrival_rates=[1, 2],
        session_population_grid=[4, 8],
    )
    write_json(tmp_path / "manifest.json", {"spec": s.model_dump()})
    write_json(tmp_path / "state.json", {"results": []})
    original = (tmp_path / "state.json").read_bytes()
    out = reanalyze(tmp_path, [5, 20, 60])
    assert out.exists() and (tmp_path / "state.json").read_bytes() == original
    with pytest.raises(ValueError):
        reanalyze(tmp_path, [601])


@pytest.mark.parametrize(
    "encoded",
    [
        [1, 2, 3, 4],
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [[1, 2, 3, 4]]},
    ],
)
def test_token_length_supports_transformers_four_and_five(encoded):
    from simulator.infrastructure.session_capacity import prompt_token_count

    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: encoded)
    assert prompt_token_count(tokenizer, "test") == 4


def test_token_length_refuses_unknown_structure():
    from simulator.infrastructure.session_capacity import prompt_token_count

    with pytest.raises(ValueError):
        prompt_token_count(
            SimpleNamespace(apply_chat_template=lambda *a, **kw: "unexpected"), "test"
        )
