"""Explicit explored-rate envelopes and matched-rate comparisons; no peak extrapolation."""

import copy
import json
import statistics


def compare_search(rows, spec):
    capacities = {}
    matched = []
    completed = []
    expected = {
        (b, p, r)
        for b in range(spec.paired_blocks)
        for p in [0, *spec.p01_prefill_grid]
        for r in spec.p01_request_rates
    }
    keys = [(x["block"], x["prefill"], x["rate"]) for x in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate search cell")
    complete = set(keys) == expected
    for p in [0, *spec.p01_prefill_grid]:
        passing = []
        for rate in spec.p01_request_rates:
            cells = [x["metrics"] for x in rows if x["prefill"] == p and x["rate"] == rate]
            if len(cells) == spec.paired_blocks and all(x["gates_passed"] for x in cells):
                passing.append(rate)
        capacities[str(p)] = max(passing) if passing else None
    for p in spec.p01_prefill_grid:
        for rate in spec.p01_request_rates:
            a = [x["metrics"] for x in rows if x["prefill"] == 0 and x["rate"] == rate]
            b = [x["metrics"] for x in rows if x["prefill"] == p and x["rate"] == rate]
            if len(a) == len(b) == spec.paired_blocks and all(m["gates_passed"] for m in a + b):
                av = statistics.mean(m["ttft_p95_s"] for m in a)
                bv = statistics.mean(m["ttft_p95_s"] for m in b)
                end_a = statistics.mean(m["latency_p95_s"] for m in a)
                end_b = statistics.mean(m["latency_p95_s"] for m in b)
                completed.append(
                    dict(
                        prefill=p,
                        rate=rate,
                        baseline_p95_s=end_a,
                        treatment_p95_s=end_b,
                        latency_reduction_percent=100 * (1 - end_b / end_a) if end_a else None,
                    )
                )
                matched.append(
                    dict(
                        prefill=p,
                        rate=rate,
                        baseline_ttft_p95_s=av,
                        treatment_ttft_p95_s=bv,
                        ttft_reduction_percent=100 * (1 - bv / av) if av else None,
                    )
                )
    base = capacities["0"]
    return dict(
        qualified=False,
        reason="Exploratory allocation search; highest passing tested rates, not confirmed maxima",
        metrics={},
        complete_grid=complete,
        highest_passing_rates=capacities,
        matched_rate_latency=matched,
        matched_completion_latency=completed,
        capacity_change_percent={
            str(p): 100 * (capacities[str(p)] / base - 1) if base and capacities[str(p)] else None
            for p in spec.p01_prefill_grid
        },
    )


def render(path, manifest, state, comparison):
    from .workflow_report import render_outputs

    manifest = copy.deepcopy(manifest)
    manifest["project"]["outputs"][2]["title"] = (
        "Completed-response latency at matched arrival rate"
    )
    spec = manifest["spec"]
    rows = []
    for r in state["results"]:
        m = r["metrics"]
        rows.append(
            [
                r["block"] + 1,
                "aggregate"
                if r["prefill"] == 0
                else f"{r['prefill']}P{len(spec['gpu_ids']) - r['prefill']}D",
                r["rate"],
                f"{m['qualified']}/{m['attempted']}",
                m["ttft_p95_s"],
                m["latency_p95_s"],
                "Pass" if m["gates_passed"] else "Fail",
            ]
        )
    methods = [
        (
            "Models and hardware",
            manifest["source_result"]["model"] + "; GPUs " + str(spec["gpu_ids"]),
        ),
        (
            "Search",
            f"Prefill counts {spec['p01_prefill_grid']}; rates {spec['p01_request_rates']} requests/second; same grid and GPU budget for aggregate comparator",
        ),
        (
            "Targets",
            f"TTFT {spec['p01_ttft_slo_s']} s, completion {spec['response_slo_s']} s; quality {spec['min_quality']}; qualified fraction {spec['min_slo_fraction']}; queue growth limit {spec['p01_queue_growth_s']} s",
        ),
        (
            "Timing",
            f"{spec['duration_s']} seconds plus drain, {spec['warmup_s']} seconds disjoint warmup per cell; queue included",
        ),
        (
            "Workload",
            f"{spec['document_records']}-record synthetic documents; exact extraction grading, natural output termination; no claimed 8K/1K reproduction",
        ),
        (
            "Scheduling",
            "Randomized deployment order; ascending rates within each launch; shared aggregate baseline; all cells retained",
        ),
        (
            "Interpretation",
            "Selection is exploratory. Boundary rates require bracketing; a highest passing grid point is not maximum capacity. Numerical results apply only to the measured workload.",
        ),
        (
            "Evidence",
            "Pinned images/models, launch configuration, topology, request rows, native worker metrics and transfers, source snapshot and checksums in ZIP",
        ),
    ]
    methods.append(
        (
            "Dynamo transfer and batching",
            json.dumps({k: v for k, v in spec.items() if k.startswith("p01_")}),
        )
    )
    methods.append(
        (
            "Role placement",
            "The first N GPUs in the recorded GPU order perform prefill; remaining GPUs decode. Physical UUID bindings and per-worker measurements are retained. Routing is round-robin and may cross NUMA groups.",
        )
    )
    methods.append(("Dynamo image", manifest.get("dynamo", {}).get("image", "See evidence")))
    values = [
        comparison.get("highest_passing_rates"),
        comparison.get("matched_rate_latency"),
        comparison.get("matched_completion_latency", []),
    ]
    render_outputs(
        path,
        manifest,
        state,
        comparison,
        ["Block", "Split", "Rate / s", "Qualified", "p95 TTFT s", "p95 total s", "Gate"],
        rows,
        methods,
        values,
    )
