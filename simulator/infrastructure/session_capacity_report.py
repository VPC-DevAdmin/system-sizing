"""Separate capacity, latency and mechanism outputs from one capacity campaign."""

from docx import Document
from docx.shared import Inches, Pt

from ..benchmarks.storage import write_json


def render(path, manifest, state, comparison):
    if manifest["spec"].get("session_workload") == "knowledge_worker":
        from .knowledge_workers import render_worker_reports

        return render_worker_reports(path, manifest, state, comparison)
    from .report import format_table, number

    spec = manifest["spec"]
    methods = {
        "Comparison": "GPU-only prefix caching versus the same runtime with CPU KV offload. Automatic GPU cache sizing in both arms; no artificially reduced GPU cache.",
        "Allocation": f"{len(spec['gpu_ids'])} selected GPUs {spec['gpu_ids']}, one TP1 replica per GPU with sticky document routing and NUMA-local host memory. No extrapolation to an untested GPU count.",
        "CPU memory": f"{spec['session_cpu_gb']} GB per replica; additional host capacity must be budgeted separately from GPU count.",
        "Workload": f"Synthetic exact-match document extraction, {spec['document_records']} records per document, {spec['session_answer_records']} requested per answer, {spec['session_turns']} questions before replacing a document. Independent questions can overlap; this is not a sequential conversation.",
        "Demand": "Paired seeded Poisson arrivals at equal per-session question cadence. Every declared population and cadence is tested. First-touch and replacement costs are included, with no separate eviction phase.",
        "Measurement": f"{spec['duration_s']} seconds of arrivals plus drain per cell; {spec['paired_blocks']} randomized paired blocks. Sustained throughput and backlog slope use the final half; all-window throughput includes drain. Readiness probing precedes measurement; there is no corpus warmup.",
        "Capacity": "Largest tested document population sustaining correct work without growing backlog. Client scheduling or admission limits invalidate saturation evidence. A passing upper boundary requires extending the grid. Latency-target capacity is reported separately.",
        "Attribution": "CPU reloads and raw cache counters are retained. Prompt-token usage is logical input volume, not proof of recomputed tokens or GPU time saved. Review native counters and telemetry before assigning a cause.",
        "Scope": "Active document populations are not simultaneous GPU streams. Synthetic extraction cannot establish broad document-analysis quality. Independent confirmation, boundary refinement and representative tasks are required before a marketing claim.",
        "Evidence": "Shared ZIP contains manifests, pinned model and image identities, resolved settings, topology, arrival traces, scored outputs, queue samples, raw metrics and GPU telemetry.",
    }
    if spec.get("session_arrival_rates"):
        methods["Demand"] = (
            "Identical open-loop Poisson arrival traces per A/B pair; a total-rate sweep for each document working set. Bounded extension through saturation and paired midpoint refinement; every adaptive cell is journaled."
        )
        methods["Capacity"] = (
            "Time-weighted concurrent outstanding requests at sustainable throughput, selected separately for each identical response-time SLO. Arrival-cohort scoring includes timeouts, errors and unfinished work. Queue buildup is not capacity. Supported reusable-document populations are separate from physical cache residency."
        )
    if spec.get("session_prime_documents"):
        methods["Measurement"] += (
            " First touches are scored and billed in separate priming evidence. Measured requests reuse the fixed corpus without replacements."
        )
        methods["Workload"] = (
            f"Fixed {spec['document_records']}-record documents; shuffled round-robin assignment within Poisson arrival times. Minimum {spec.get('session_min_returns', 4)} measured returns per document."
        )
    tables = {
        "P02-O1": (
            ["Block / arm", "Documents", "Question/s per doc", "Correct/s", "Stable"],
            lambda r: [
                f"{r['block'] + 1} / {r['arm']}",
                str(r["population"]),
                number(r["per_session_rate"]),
                number(r["metrics"]["sustained_correct_requests_s"]),
                str(r["metrics"]["stable_capacity"]),
            ],
        ),
        "P02-O2": (
            ["Block / arm", "Documents", "Total req/s", "TTFT p50 s", "TTFT p95 s", "SLO pass"],
            lambda r: [
                f"{r['block'] + 1} / {r['arm']}",
                str(r["population"]),
                number(r["rate"]),
                number(r["metrics"]["ttft_p50_s"]),
                number(r["metrics"]["ttft_p95_s"]),
                str(r["metrics"]["service_capacity"]),
            ],
        ),
        "P02-O3": (
            [
                "Block / arm",
                "Documents",
                "Total req/s",
                "Queue slope",
                "CPU reload",
                "Cold / total",
            ],
            lambda r: [
                f"{r['block'] + 1} / {r['arm']}",
                str(r["population"]),
                number(r["rate"]),
                number(r["metrics"]["backlog_slope_requests_s"]),
                str(r["metrics"]["cpu_reload_observed"]),
                f"{r['metrics']['cold_requests']} / {r['metrics']['scheduled']}",
            ],
        ),
    }
    for output in manifest["project"]["outputs"]:
        folder = path / "outputs" / output["id"]
        folder.mkdir(parents=True, exist_ok=True)
        reason = f"{state['phase']}: exploratory synthetic workload; no confirmed marketing claim."
        write_json(
            folder / "claim.json",
            dict(
                id=output["id"],
                title=output["title"],
                campaign_id=path.name,
                status="not_qualified",
                reason=reason,
                positive_claim_supported=False,
                mode="pilot",
                measurement=comparison if output["id"] == "P02-O1" else state["results"],
            ),
        )
        doc = Document()
        section = doc.sections[0]
        section.top_margin = section.bottom_margin = Inches(0.7)
        section.left_margin = section.right_margin = Inches(0.7)
        doc.styles["Normal"].font.size = Pt(10)
        doc.add_heading(output["title"], 0)
        doc.add_paragraph(reason)
        if state.get("error"):
            doc.add_paragraph("Run error: " + state["error"])
        if output["id"] == "P02-O1":
            for cadence in comparison.get("by_cadence", []):
                for b in cadence["blocks"]:
                    value = b["exploratory_capacity_uplift_percent"]
                    if value is not None:
                        doc.add_paragraph(
                            f"Block {b['block'] + 1}, mean {cadence['mean_seconds_between_questions']:g} seconds per question: {value:+.1f}% change in largest passing tested document population. Boundaries and attribution ready: {b['comparison_ready']}. This is a grid observation, not a proven maximum."
                        )
        if comparison.get("by_working_set"):
            doc.add_heading("Saturation and concurrency", 1)
            doc.add_paragraph(
                "Arrival rates are extended through two overload points per arm, then refined around the knee. Measurement timeouts censor unfinished requests; SLO analysis includes every arrival, including failures. Distinct resident document counts are unavailable from block telemetry."
            )
            for group in comparison["by_working_set"]:
                doc.add_paragraph(
                    f"Block {group['block'] + 1}, {group['population']} documents: saturation resolved = {group['saturation_resolved']}."
                )
                for arm, values_ in group["arms"].items():
                    best = values_["best"]
                    if best:
                        m = best["metrics"]
                        doc.add_paragraph(
                            f"{arm}: {m['mean_outstanding_requests']:.2f} mean concurrent requests at {m['sustained_correct_requests_s']:.2f} correct completions/s; knee bracket {values_['knee_bracket_requests_s']} requests/s."
                        )
                if output["id"] == "P02-O1":
                    for row in group["slo_comparisons"]:
                        doc.add_paragraph(
                            f"Response SLO {row['response_slo_s']:g}s: concurrent-request change {number(row['concurrent_request_change_percent'])}%; throughput change {number(row['throughput_change_percent'])}%. A larger queue alone is not extra capacity."
                        )
        headers, values = tables[output["id"]]
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        for c, v in zip(table.rows[0].cells, headers, strict=True):
            c.text = v
        for row in state["results"]:
            for c, v in zip(table.add_row().cells, values(row), strict=True):
                c.text = v
        format_table(table, [6.6 / len(headers)] * len(headers))
        if output["id"] == "P02-O3":
            doc.add_heading("Prefix work accounting", 1)
            for row in state["results"]:
                work = row["metrics"].get("cache_work", {})
                doc.add_paragraph(
                    f"Block {row['block'] + 1}, {row['arm']}, {row['population']} documents at {row['rate']:g}/s: uncached prefix tokens per correct request {number(work.get('uncached_prefix_tokens_per_correct_request'))}; uncached share {number(work.get('uncached_prefix_share'))}. Historical recomputation remains unavailable without request-level attribution."
                )
        for repeat in comparison.get("repeated_load_comparisons", []):
            doc.add_paragraph(
                f"Matched load {repeat['arrival_rate']:g}/s, {repeat['population']} documents: {repeat['independent_pairs']} independent pairs; throughput change {number(repeat['throughput_change_percent'])}%; paired block interval {repeat['bootstrap_95_percent']}. Excluded blocks {repeat['excluded_blocks']}; exploratory only."
            )
        doc.add_heading("Methodology and limits", 1)
        for k, v in methods.items():
            para = doc.add_paragraph()
            para.add_run(k + ": ").bold = True
            para.add_run(v)
        doc.save(folder / "white-paper.docx")
        (folder / "claim-brief.md").write_text(
            "# "
            + output["title"]
            + "\n\n"
            + reason
            + "\n\n"
            + "\n\n".join(f"**{k}:** {v}" for k, v in methods.items())
        )
