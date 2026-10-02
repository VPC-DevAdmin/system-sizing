"""Separate peak narratives with the complete measured curves in the evidence."""

import csv
import json
from xml.sax.saxutils import escape

from .workflow_report import render_outputs


def peak_text(index, value, spec):
    if not value:
        return [
            "No positive matched peak was established for this output within the measured search."
        ]
    split = f"{value['prefill']} prefill and {len(spec['gpu_ids']) - value['prefill']} decode GPUs"
    lines = [
        f"Selected workload {value['shape']} with {split}; placement {value.get('placement', 'pooled').replace('_', ' ')}."
    ]
    if index == 0:
        lines.append(
            f"Split serving passed at {value['treatment_rate']:g} requests per second. Aggregate passed at {value['baseline_rate']:g} requests per second, a {value['observed_uplift_percent']:.1f}% difference between the highest passing tested rates."
        )
        lines.append(
            f"Using the aggregate failing boundary instead gives a conservative measured advantage of {value['conservative_uplift_percent']:.1f}%. The exact capacity boundary between tested rates remains unknown."
        )
    elif index == 1:
        lines.append(
            f"The highest passing tested split-serving document rate was {value['passing_rate']:g} requests per second. This peak covers the tested disaggregated candidates for this document and answer size; the aggregate comparator may be faster."
        )
    else:
        lines.append(
            f"At {value['rate']:g} requests per second, p95 completion fell from {value['baseline_p95_s']:.3f} to {value['treatment_p95_s']:.3f} seconds, a {value['reduction_percent']:.1f}% reduction. Both deployments passed the service targets at this load."
        )
    for key, label in [("prompt_tokens_range", "Input"), ("output_tokens_range", "Output")]:
        tokens = value.get(key)
        if tokens:
            lines.append(
                f"{label} tokens observed for this shape and split: {tokens[0]} to {tokens[1]}."
            )
    if value.get("boundary", value.get("treatment_boundary")) == "grid_ceiling":
        lines.append(
            "The treatment passed at the upper search limit. Expand the rate range to locate its boundary."
        )
    return lines


def render(path, manifest, state, comparison):
    spec = manifest["spec"]
    rows = []
    columns = [
        "shape",
        "block",
        "prefill",
        "placement",
        "rate",
        "qualified_per_hour",
        "completed_requests_per_s",
        "correct_requests_per_s",
        "output_tokens_per_s",
        "input_tokens_per_s",
        "total_tokens_per_s",
        "output_usage_coverage",
        "input_usage_coverage",
        "request_ttft_p95_s",
        "queue_p95_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "latency_p50_s",
        "latency_p95_s",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "output_tokens_min",
        "output_tokens_max",
        "deployment_attempt",
        "effective_client_concurrency",
        "client_limit_detected",
        "gates_passed",
    ]
    with (path / "goal-search-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in state["results"]:
            m = row["metrics"]
            writer.writerow({k: row.get(k, m.get(k)) for k in columns})
            rows.append(
                [
                    row["shape"],
                    "Aggregate"
                    if row["prefill"] == 0
                    else f"{row['prefill']}P {row.get('placement', 'pooled').replace('socket_local', 'local').replace('cross_socket', 'cross')}",
                    row["rate"],
                    f"{m['qualified']}/{m['attempted']}",
                    m["ttft_p95_s"],
                    m["latency_p95_s"],
                    "Pass" if m["gates_passed"] else "Fail",
                ]
            )
    # Portable, dependency-free scatter curves. Failed cells are crosses; no interpolation.
    shapes = sorted(
        {
            (r["shape"], placement)
            for r in state["results"]
            for placement in spec.get("p01_placement_grid", ["pooled"])
        }
    )
    height = max(1, len(shapes)) * 280
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}"><rect width="100%" height="100%" fill="white"/>'
    ]
    colors = [
        "#222222",
        "#0066aa",
        "#b34a00",
        "#168030",
        "#7523ab",
        "#a50054",
        "#347575",
        "#866000",
    ]
    for i, (shape, placement) in enumerate(shapes):
        cells = [
            r
            for r in state["results"]
            if r["shape"] == shape
            and (r["prefill"] == 0 or r.get("placement", "pooled") == placement)
        ]
        top = i * 280
        xmax = max(spec["p01_request_rates"])
        ymax = max([r["metrics"]["qualified_per_hour"] / 3600 for r in cells] + [1])
        svg.append(
            f'<text x="60" y="{top + 24}" font-family="sans-serif" font-size="16">{escape(shape + " " + placement)} — measured qualified requests/s (drain included)</text>'
        )
        svg.append(f'<path d="M60 {top + 45} V{top + 225} H670" stroke="black" fill="none"/>')
        for j in range(5):
            x = 60 + 600 * j / 4
            y = top + 225 - 170 * j / 4
            svg.append(
                f'<text x="{x}" y="{top + 244}" font-size="12">{xmax * j / 4:g}</text><text x="5" y="{y}" font-size="12">{ymax * j / 4:.2f}</text>'
            )
        svg.append(f'<text x="280" y="{top + 269}" font-size="13">Offered requests / second</text>')
        for p in sorted({r["prefill"] for r in cells}):
            color = colors[p]
            label = "Aggregate" if p == 0 else f"{p}P{len(spec['gpu_ids']) - p}D"
            svg.append(
                f'<text x="710" y="{top + 50 + p * 22}" fill="{color}" font-size="13">{label}</text>'
            )
            for r in cells:
                if r["prefill"] != p:
                    continue
                x = 60 + 600 * r["rate"] / xmax
                y = top + 225 - 170 * (r["metrics"]["qualified_per_hour"] / 3600) / ymax
                if r["metrics"]["gates_passed"]:
                    svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
                else:
                    svg.append(
                        f'<path d="M{x - 5} {y - 5} l10 10 m-10 0 l10 -10" stroke="{color}" stroke-width="2"/>'
                    )
    svg.append("</svg>")
    (path / "goal-search-curves.svg").write_text("".join(svg))
    methods = [
        (
            "Recovery",
            f"Recorded serving failures: {len(state.get('attempts', []))}. Each allocation permits one retry after owned cleanup. Successful windows are retained; failed attempt files remain in their original directories. Settings are unchanged across retries.",
        ),
        (
            "Execution",
            f"Mode {spec.get('p01_execution', 'default')}; CUDA launch blocking {spec.get('p01_cuda_launch_blocking', False)}. Diagnostic timings must not be used as production performance claims.",
        ),
        ("Model", manifest["configuration"]["engine"]["model_id"]),
        (
            "Goal",
            "Find the largest measured same-shape capacity advantage over aggregate. Absolute capacity and latency peaks are separate outputs.",
        ),
        (
            "Workload",
            "Synthetic logistics extraction. dN-aM means N document records and M requested answer records. Natural termination; exact JSON grading. Token ranges for every cell are in goal-search-curves.csv; these are measured token shapes, not exact token quotas or domain-expertise validation.",
        ),
        ("Search", json.dumps(manifest.get("goal_search", {}))),
        (
            "Baseline",
            "One aggregate TP1 worker per GPU; equal GPU budget, model, precision, context and batching settings. This searches allocation and workload shape, not every possible aggregate engine configuration.",
        ),
        (
            "Service targets",
            f"TTFT <= {spec['p01_ttft_slo_s']} s; completion <= {spec['response_slo_s']} s; quality >= {spec['min_quality']}; qualified fraction >= {spec['min_slo_fraction']}; queue growth <= {spec['p01_queue_growth_s']} s.",
        ),
        (
            "Timing",
            f"{spec['duration_s']} seconds plus drain; {spec['warmup_s']} seconds disjoint warmup per cell; {spec['paired_blocks']} block(s). Client queue and failed admissions included.",
        ),
        (
            "Placement",
            "GPU PCI addresses, NUMA nodes, CPU affinity, pool membership and allowed KV pairs are recorded in topology-plan.json. Socket-local mode uses isolated namespaces and discovery stores; cross-socket mode swaps decode membership. Asymmetric pools use least-outstanding routing. Host memory binding is not asserted.",
        ),
        (
            "Load generator",
            f"Client concurrency ceiling: {spec.get('p01_client_max_concurrency', 512)}. Above the legacy ceiling of 512, headroom covers arrivals within the shorter of measurement duration and request timeout; otherwise the completion target is used. Engine concurrency and closed-loop warmup retain the configured concurrency. Per-cell concurrency and queue measurements are retained. Client-limited cells cannot establish server capacity. Throughput includes drain time and counts completed responses; correct requests/s separately counts exact-graded answers. Token rates require usage on every completed response. TTFT includes client queue; request TTFT excludes it. These are finite-cohort rates, not steady-state capacity measurements.",
        ),
        (
            "Peak selection",
            "Compare passing rates within the identical shape. Conservative uplift uses the baseline failing rate above its passing boundary; a baseline still passing at the grid ceiling cannot establish a capacity advantage. Highest tested points are not global maxima. Declared arrival rates are shared measured curve points before boundary refinement. A failed lower rate followed by a passing higher rate is flagged non_monotonic and excluded from capacity peak selection. Bracketing can still miss unmeasured pass islands.",
        ),
        (
            "Evidence",
            "Every attempted cell, including failures, is retained. CSV and SVG curves, native worker/transfer metrics, physical placement, pinned artifacts and request rows are in the ZIP. Unmeasured rates are not interpolated measurements.",
        ),
        (
            "Status",
            f"Search finished across declared shapes and splits: {comparison.get('complete_search', False)}. Peak selection is exploratory and has no held-out confirmation.",
        ),
    ]
    values = [
        comparison.get("peak_capacity_uplift"),
        comparison.get("peak_absolute_capacity"),
        comparison.get("peak_latency_reduction"),
    ]
    render_outputs(
        path,
        manifest,
        state,
        comparison,
        ["Shape", "Split", "Rate / s", "Qualified", "p95 TTFT s", "p95 total s", "Gate"],
        rows,
        methods,
        values,
    )
