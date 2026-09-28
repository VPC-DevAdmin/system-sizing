"""Separate post-call throughput, latency and resource-count outputs."""

import statistics


def compare_calls(rows, spec):
    keys = [(r["block"], r["arm"], r["rate"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate call cell")
    expected = {
        (b, a, r)
        for b in range(spec.paired_blocks)
        for a in ["dedicated", "shared", "mig"]
        for r in spec.call_request_rates
    }
    matched = []
    for arm in ["shared", "mig"]:
        for rate in spec.call_request_rates:
            a = [x["metrics"] for x in rows if x["arm"] == "dedicated" and x["rate"] == rate]
            b = [x["metrics"] for x in rows if x["arm"] == arm and x["rate"] == rate]
            if len(a) == len(b) == spec.paired_blocks and all(x["gates_passed"] for x in a + b):
                baseline = statistics.mean(x["qualified_audio_hours_per_hour"] for x in a)
                treatment = statistics.mean(x["qualified_audio_hours_per_hour"] for x in b)
                matched.append(
                    dict(
                        placement=arm,
                        rate=rate,
                        gpu_reduction_percent=50,
                        baseline_audio_hours_per_hour=baseline,
                        treatment_audio_hours_per_hour=treatment,
                        observed_throughput_retained_percent=100 * treatment / baseline
                        if baseline
                        else None,
                        baseline_p95_s=statistics.mean(x["latency_p95_s"] for x in a),
                        treatment_p95_s=statistics.mean(x["latency_p95_s"] for x in b),
                    )
                )
    return dict(
        qualified=False,
        metrics={},
        reason="Exploratory synthetic post-call workflow; not real-call capacity or free-form summary acceptance",
        complete_grid=set(keys) == expected,
        passing_matched_rates=matched,
    )


def render(path, manifest, state, comparison):
    from .calls import ASR_MODEL, ASR_REVISION
    from .workflow_report import render_outputs

    s = manifest["spec"]
    rows = []
    for r in state["results"]:
        m = r["metrics"]
        rows.append(
            [
                r["block"] + 1,
                r["arm"],
                r["rate"],
                f"{m['qualified']}/{m['attempted']}",
                m["qualified_audio_hours_per_hour"],
                m["latency_p95_s"],
                "Pass" if m["gates_passed"] else "Fail",
            ]
        )
    methods = [
        (
            "Speech model",
            ASR_MODEL + " revision " + ASR_REVISION + "; FP16, beam 5, English, two workers",
        ),
        (
            "Summary model",
            manifest["source_result"]["model"]
            + "; frozen source precision, temperature zero, 256 output tokens, thinking disabled",
        ),
        (
            "Placements",
            f"Dedicated ASR GPU {s['gpu_ids'][0]} and summary GPU {s['gpu_ids'][1]}; software sharing or two 2g.48gb MIG partitions on GPU {s['gpu_ids'][-1]}; {s['tenant_kv_gib']} GiB LLM KV in every placement",
        ),
        (
            "Workload",
            "Eight synthetic eSpeak support-call recordings; separate ninth warmup recording. Same WAVs and prompts in every placement. ASR output, never reference text, feeds the language model.",
        ),
        (
            "Summary task",
            "Exact four-field structured summary: issue, action, owner, deadline. No free-form summary quality or real contact-center claim.",
        ),
        (
            "Quality",
            f"Normalized word error rate <= {s['call_max_wer']}; exact normalized summary fields; quality fraction >= {s['min_quality']}",
        ),
        (
            "Timing",
            f"Rates {s['call_request_rates']} calls/s, {s['duration_s']} seconds plus drain; {s['concurrency']} in-flight calls; two separate warmup calls. End-to-end target {s['response_slo_s']} seconds; at least {s['min_slo_fraction']} qualified.",
        ),
        (
            "Accounting",
            "Audio-hours/hour equals qualified audio seconds divided by elapsed wall seconds including drain. Queue, rejected arrivals and stage failures remain in totals. Passing a finite window does not establish sustainable capacity; growing queues require a longer steady-load follow-up.",
        ),
        (
            "Handoff",
            "Host-mediated transcript text in all placements. Transcript bytes and client handoff timing are recorded. These do not isolate PCIe transfer latency or demonstrate avoiding PCIe.",
        ),
        (
            "Evidence",
            "Original WAVs, reference transcripts, summaries, ASR outputs, stage times, image/model fingerprints, package versions and MIG restoration journals in ZIP",
        ),
    ]
    matched = comparison.get("passing_matched_rates", [])
    fields = [
        [
            "placement",
            "rate",
            "baseline_audio_hours_per_hour",
            "treatment_audio_hours_per_hour",
            "observed_throughput_retained_percent",
        ],
        ["placement", "rate", "baseline_p95_s", "treatment_p95_s"],
        ["placement", "rate", "gpu_reduction_percent"],
    ]
    measurements = [[{k: r[k] for k in keys} for r in matched] for keys in fields]
    render_outputs(
        path,
        manifest,
        state,
        comparison,
        ["Block", "Placement", "Calls / s", "Qualified", "Audio h / h", "p95 total s", "Gate"],
        rows,
        methods,
        measurements,
    )
