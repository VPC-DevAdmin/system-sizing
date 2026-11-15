"""Readable per-output reports and a portable, checksummed evidence archive."""

import csv
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import yaml
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from ..benchmarks.storage import read_json, sha256, write_json


def number(value):
    return "Unavailable" if value is None else f"{value:,.3f}"


def chart(directory, title, metric):
    if not metric or "baseline_mean" not in metric:
        return
    if metric.get("unit"):
        title += " (" + metric["unit"] + ")"
    a, b = metric["baseline_mean"], metric["treatment_mean"]
    maximum = max(a, b)
    rows = []
    for i, (name, value, color) in enumerate(
        [("Baseline", a, "#3B5265"), ("Treatment", b, "#087D8C")]
    ):
        y = 70 + i * 55
        rows.append(
            f'<text x="12" y="{y + 18}" font-size="14">{name}</text><rect x="110" y="{y}" width="{400 * value / maximum:.1f}" height="28" fill="{color}"/><text x="{125 + 400 * value / maximum:.1f}" y="{y + 19}" font-size="14">{value:.3f}</text>'
        )
    (directory / "comparison.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="205" viewBox="0 0 700 205"><rect width="700" height="205" fill="white"/><g font-family="Arial" fill="#12212c"><text x="12" y="30" font-size="17">'
        + escape(title[:75])
        + "</text>"
        + "".join(rows)
        + '<text x="12" y="190" font-size="12">Means of paired blocks. Consult the quality gates and qualification status.</text></g></svg>'
    )


def report_sections(manifest, state, comparison, claim, path):
    spec = manifest["spec"]
    metric = claim["measurement"]
    label = (
        "Exploratory pilot"
        if spec["mode"] == "pilot"
        else "Qualified measurement"
        if claim["status"] == "qualified_measurement"
        else "Not qualified for publication"
    )
    summary = f"{label}. {claim['reason']}."
    if metric and "change_percent" in metric:
        if metric.get("unit"):
            summary += f" Measurement unit: {metric['unit']}."
        summary += (
            f" The observed change was {metric['change_percent']:.2f} percent "
            "(positive values indicate improvement)."
        )
        interval = metric.get("paired_bootstrap_95_interval")
        summary += (
            f" The paired bootstrap 95 percent interval was {interval[0]:.2f} to {interval[1]:.2f} percent."
            if interval
            else " There are too few paired blocks to report a confidence interval."
        )
    methods = [
        ("Harness", manifest["harness"]),
        ("Model", manifest["source_result"]["model"]),
        ("Run purpose", spec["mode"]),
        ("GPU indices", ", ".join(map(str, spec["gpu_ids"]))),
        ("Deployment", "One TP1 replica per selected GPU; identical assignment in both arms"),
        ("Concurrency", str(claim.get("concurrency", spec["concurrency"]))),
        ("Paired blocks", str(spec["paired_blocks"])),
        ("Measured time per arm", f"{spec['duration_s']} seconds plus request drain"),
        ("Warmup per arm", f"{spec['warmup_s']} seconds on a disjoint corpus"),
        ("Request timeout", f"{spec['request_timeout_s']} seconds"),
        ("Maximum output", f"{spec['max_output_tokens']} tokens; natural EOS"),
        ("Context limit", f"{spec['context_limit']} tokens"),
        (
            "Workload",
            f"{spec['corpus_size']} synthetic logistics documents, {spec['document_records']} records each",
        ),
        (
            "Grader",
            "Exact JSON record equality including requested order, every field, and value; truncations fail",
        ),
        (
            "Structured output",
            "JSON schema constrains format in both arms; answers remain graded"
            if spec.get("structured_output")
            else "Unconstrained JSON generation",
        ),
        (
            "Sampling",
            f"Temperature 0, seed {spec['seed']}; thinking disabled in chat template options",
        ),
        ("Quality threshold", f"{spec['min_quality']:.1%} correct among all attempted requests"),
        ("Error threshold", f"At most {spec['max_error_fraction']:.1%} transport errors"),
        (
            "SLO acceptance",
            f"At least {spec['min_slo_fraction']:.1%} of attempted requests correct and within response target"
            if "min_slo_fraction" in spec
            else "No separate SLO-fraction gate recorded in this earlier run",
        ),
        ("Response target", f"{spec['response_slo_s']} seconds per correct answer"),
        ("Minimum sample", f"{spec['min_requests']} requests in every arm"),
        ("Cache policy", manifest["cache_policy"]),
        (
            "Treatment",
            f"Native MTP with Triton draft MoE backend, {spec['speculative_tokens']} draft tokens"
            if spec.get("speculative_method") == "mtp"
            else f"Prompt n-gram speculation, {spec['speculative_tokens']} draft tokens, lookup range 2 to 5",
        ),
        (
            "Scheduling",
            "Arm order randomized independently within paired blocks; engines restarted per arm",
        ),
        ("Rate denominator", "Total wall-clock serving window including drained requests"),
    ]
    if spec["project_id"] == "P10":
        methods = [
            (
                key,
                (
                    f"{spec['attention_baseline']} versus {spec['attention_treatment']}; {spec.get('attention_kv_dtype', 'fp8')} KV and BF16 model compute in both"
                    if key == "Treatment"
                    else value
                ),
            )
            for key, value in methods
        ]
        methods.append(
            (
                "Profiling",
                "Separate engine launch; one diagnostic request with max_iterations=4 configured. All A/B measurements use launches without a profiler.",
            )
        )
        methods.append(
            (
                "Numerical check",
                f"Four first-token probes per pair; same top token, at least 95% top-20 overlap, maximum common logprob difference {spec['attention_logprob_tolerance']}",
            )
        )
        profile_file = path / "block-00-profile/profile-summary.json"
        if profile_file.exists():
            profile = read_json(profile_file)
            methods.append(
                (
                    "Profile evidence",
                    f"{profile['kernel_events']} CUDA kernel events; raw traces and top-kernel summary in evidence",
                )
            )
        if (path / "numerical-parity.json").exists():
            methods.append(
                (
                    "Numerical gate",
                    "Pass" if read_json(path / "numerical-parity.json")["passed"] else "Fail",
                )
            )
    if spec["project_id"] == "P03":
        replacements = {
            "Concurrency": ", ".join(map(str, spec["cache_concurrencies"]))
            + " (separate grid cells)",
            "Workload": f"Frozen synthetic documents with {spec['cache_record_counts']} records; {spec['corpus_size']} documents per size",
            "Treatment": "INT4 per-token/head cache versus FP8 cache using checkpoint/runtime scales; BF16 cache quality reference; TRITON_ATTN in all arms",
            "Scheduling": "Arm order randomized per block; ascending size/concurrency grid within each owned launch",
            "Grader": "Exact JSON extraction at every point; BF16 reference at the same document size must pass correctness, sample and error gates before a point is accepted",
        }
        methods = [(key, replacements.get(key, value)) for key, value in methods]
        if "cache_samples" in spec:
            methods = [
                (
                    key,
                    (
                        f"Exactly {spec['cache_samples']} matched documents once per cell; duration setting does not apply"
                        if key == "Measured time per arm"
                        else value
                    ),
                )
                for key, value in methods
            ]
            methods.append(("Matched samples per cell", str(spec["cache_samples"])))
        methods.append(
            (
                "BF16 reference load",
                "Lowest configured concurrency at each document size; used to check task correctness",
            )
        )
        methods.append(
            (
                "Capacity limit",
                "Reports largest passing tested points only; no established maximum and no qualified marketing claim",
            )
        )
    if spec["project_id"] == "P01":
        replacements = {
            "Deployment": "Same total GPU budget; baseline one TP1 worker per GPU; treatment splits prefill and decode",
            "Treatment": f"Dynamo with {spec['prefill_workers']} prefill workers and {len(spec['gpu_ids']) - spec['prefill_workers']} decode workers; NIXL KV transfer",
        }
        methods = [(key, replacements.get(key, value)) for key, value in methods]
        checks = [read_json(f) for f in path.glob("block-*/transfer-verification.json")]
        verified = len(checks) == spec["paired_blocks"] and all(c.get("verified") for c in checks)
        methods.append(
            (
                "Transfer evidence",
                "Increasing NIXL byte counters verified in every measured treatment arm"
                if verified
                else "Transfer not established for every treatment arm; inspect the failed or incomplete attempt",
            )
        )
        methods.append(
            (
                "Source configuration",
                "Model and precision seeded from roofline; matched Dynamo launch settings replace original engine flags in both arms",
            )
        )
        methods.append(
            (
                "Load scope",
                "Same closed-loop concurrency; offered request rate is not held fixed; no open-loop latency claim",
            )
        )
    if spec["project_id"] == "P02":
        replacements = {
            "Deployment": "One TP1 worker; equal fixed GPU KV allocation in both arms",
            "Treatment": f"Dynamo KVBM {spec['session_tier']}; GPU {spec['session_gpu_gb']} GiB, host {spec['session_cpu_gb']} GB, disk {spec['session_disk_gb']} GB when enabled",
            "Measured time per arm": "Finite resume cohort; configured duration does not set its length",
            "Workload": f"Prime {spec['session_count']} document sessions, apply {spec['session_churn']} disjoint documents as cache pressure, resume each session once",
            "Rate denominator": "Resume cohort elapsed time only; prime and pressure costs recorded separately",
            "Scheduling": "Randomized arm order; identical finite document-session traces in each arm",
        }
        methods = [(k, replacements.get(k, v)) for k, v in methods]
        methods.append(
            (
                "Resume TTFT target",
                f"p95 at most {spec['session_ttft_slo_s']} seconds; separate from the complete-response target"
                if "session_ttft_slo_s" in spec
                else "No separate resume TTFT target was specified in this earlier pilot",
            )
        )
        if manifest.get("session_hot_diagnostic"):
            methods.append(
                (
                    "GPU hot diagnostic",
                    "One immediate return after priming; at least 90% native GPU cache hits required; not a hot-session capacity measurement",
                )
            )
        methods.append(
            (
                "Primary output",
                "p95 time to first content token when resuming; completed-response latency also retained",
            )
        )
        methods.append(
            (
                "Tier proof",
                "Native onboard-block deltas required for requested CPU or disk tier; configured cache sizes alone do not establish retention",
            )
        )
    environments = sorted(path.glob("block-*/environment.json"))
    if environments:
        environment = read_json(environments[0])
        gpu_rows = list(csv.reader(io.StringIO(environment.get("gpu_inventory", ""))))
        hardware = sorted(
            {row[1].strip() + " " + row[2].strip() for row in gpu_rows if len(row) >= 3}
        )
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in environment.get("cpu_inventory", "").splitlines()
                if line.startswith("Model name:")
            ),
            "See evidence inventory",
        )
        methods[0:0] = [
            ("Host", environment.get("hostname", "Unknown")),
            ("GPU type", "; ".join(hardware)),
            ("CPU", cpu),
        ]
    if (path / "report-revision.json").exists():
        methods.append(("Report revision", read_json(path / "report-revision.json")["note"]))
    methods.insert(
        0, ("Created UTC", datetime.fromtimestamp(manifest["created_at"], timezone.utc).isoformat())
    )
    rows = []
    for result in state["results"]:
        m = result["metrics"]
        row = [
            str(result["block"] + 1),
            result["arm"],
            str(m["attempted"]),
            f"{m['quality_fraction']:.1%}",
            number(m["qualified_per_hour"]),
            number(m["latency_p95_s"]),
            "Pass" if m["gates_passed"] else "Fail",
        ]
        if spec["project_id"] == "P02":
            row[5] = number(m.get("ttft_p95_s"))
        if spec["project_id"] == "P03":
            row[2:2] = [str(result["document_records"]), str(result["concurrency"])]
        rows.append(row)
    grouped = {}
    for config in sorted(path.glob("block-*/launch-resolved.yaml")):
        engine = yaml.safe_load(config.read_text())["engine"]
        selected = {
            k: engine.get(k)
            for k in (
                "gpu_image",
                "model_local_path",
                "served_model_name",
                "max_model_len",
                "gpu_memory_utilization",
                "replica_devices",
                "vllm_extra_flags",
            )
        }
        grouped.setdefault(json.dumps(selected, sort_keys=True), []).append(config.parent.name)
    engines = []
    for encoded, names in grouped.items():
        selected = json.loads(encoded)
        engines.append(
            "Applies to "
            + ", ".join(names)
            + "\n"
            + "\n".join(
                key + ": " + (json.dumps(value) if not isinstance(value, str) else value)
                for key, value in selected.items()
            )
        )
    if spec["project_id"] in ("P01", "P02"):
        engines = []
        planfiles = sorted(path.glob("block-*/replica-0/dynamo-plan.json"))
        # Plain vLLM P02 replicas (session_vllm.py) record their own plan.
        planfiles += sorted(path.glob("block-*/**/replica-0/session-plan.json"))
        for planfile in planfiles:
            plan = read_json(planfile)
            if spec["project_id"] == "P02":
                settings = plan.pop("sessions", {})
                plan.pop("prefill_workers", None)
                plan["prefix_caching"] = True
                plan["gpu_kv_gib"] = settings.get("session_gpu_gb")
                plan["host_cache_gb"] = (
                    settings.get("session_cpu_gb") if plan["arm"] == "treatment" else 0
                )
                plan["disk_cache_gb"] = (
                    settings.get("session_disk_gb")
                    if plan["arm"] == "treatment" and settings.get("session_tier") == "cpu_disk"
                    else 0
                )
            elif plan["arm"] == "baseline":
                plan.pop("prefill_workers", None)
                plan["roles"] = ["aggregated"] * len(spec["gpu_ids"])
            runtime = planfile.parent / "runtime-versions.json"
            if runtime.exists():
                plan["runtime_packages"] = read_json(runtime)["packages"]
            plan["image"] = plan.get("image") or manifest.get("dynamo", {}).get("image")
            plan["worker_commands"] = str(
                (planfile.parent / "service-commands.json").relative_to(path)
            )
            engines.append(
                "Applies to "
                + planfile.parent.parent.name
                + "\n"
                + "\n".join(
                    key + ": " + (value if isinstance(value, str) else json.dumps(value))
                    for key, value in plan.items()
                )
            )
    return summary, methods, rows, engines


def format_table(table, widths, header=True):
    table.autofit = False
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + edge)
        for key, value in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            border.set(qn("w:" + key), value)
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for i, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if i == 0 and header:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for j, cell in enumerate(row.cells):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for edge in ("top", "left", "bottom", "right"):
                item = OxmlElement("w:" + edge)
                item.set(qn("w:w"), "75")
                item.set(qn("w:type"), "dxa")
                margins.append(item)
            properties.append(margins)
            shade = OxmlElement("w:shd")
            shade.set(
                qn("w:fill"), "E7EFF5" if i == 0 and header else "FFFFFF" if i % 2 else "F5F7F9"
            )
            properties.append(shade)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if len(widths) == 2 or j == 1
                    else WD_ALIGN_PARAGRAPH.CENTER
                )
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.bold = bool(i == 0 and header)
                    run.font.color.rgb = RGBColor(0, 0, 0)


def export(path, comparison):
    manifest = read_json(path / "manifest.json")
    state = read_json(path / "state.json")
    spec = manifest["spec"]
    if spec.get("p01_token_search"):
        from .token_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    if spec.get("p01_goal_seek"):
        from .goal_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    if spec["project_id"] == "P01" and spec.get("p01_prefill_grid"):
        from .search_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    if spec["project_id"] == "P06" and spec.get("tenant_workflow") == "post_call":
        from .call_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    if spec["project_id"] == "P02" and spec.get("session_mode") == "capacity":
        from .session_capacity_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    if spec["project_id"] == "P06":
        from .tenant_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    if spec["project_id"] == "P05":
        from .diffusion_report import render

        render(path, manifest, state, comparison)
        archive_evidence(path, state)
        return
    mapping = {
        "P01-O1": "qualified_per_hour",
        "P01-O2": "latency_p95_s",
        "P02-O2": "ttft_p95_s",
        "P04-O1": "latency_p95_s",
        "P04-O2": "qualified_per_hour",
        "P03-O1": "largest_tested_qualified_document_records",
        "P03-O2": "largest_tested_qualified_concurrency",
        "P10-O1": "latency_p95_s",
        "P10-O2": "qualified_per_hour",
    }
    limits = (
        "This experiment measures exact record extraction from synthetic logistics documents at one "
        "closed-loop concurrency. Peak capacity, other task types, broad model quality and wall energy "
        "require their own measurements. Pilots provide exploratory evidence. No independent certification "
        "is claimed. Failed requests and incorrect answers remain in the totals. Native engine logs, "
        "per-request outputs, frozen corpora, settings, assets, host information and source files are "
        "included in the shared evidence ZIP."
    )
    if spec["project_id"] == "P03":
        limits = (
            "This exploratory grid reports the largest passing tested input and concurrency points, not "
            "a proven capacity maximum. Concurrency is compared at the largest configured document size. "
            "BF16 reference at the same document size must pass task-quality, sample-count and error gates; its speed is not a quality requirement. "
            "Exact request token counts and per-cell results are preserved in the evidence. No KV-memory "
            "reduction claim is produced without measured allocation accounting. The same Triton attention "
            "backend is selected in every arm; this is not a comparison against the original roofline backend. "
            "Synthetic extraction does not establish broad long-context model quality. All outputs remain exploratory."
        )
    if spec["project_id"] == "P02":
        limits = (
            "This pilot replays a finite cohort of document sessions after a fixed cache-pressure trace. "
            "It does not establish maximum retained conversations, simultaneous active-user capacity, "
            "multi-turn conversation quality, or production disk durability. Resume throughput excludes "
            "priming and cache-pressure work, which is recorded separately. CPU-only and CPU-plus-disk "
            "are distinct treatments. Native reload evidence is required; cache capacities are not summed. "
            "The cache is ephemeral and removed with the owned container. No certification is claimed."
        )
    for output in manifest["project"]["outputs"]:
        directory = path / "outputs" / output["id"]
        scoped_comparison = comparison.get("by_concurrency", {}).get(
            str(output.get("concurrency")), comparison
        )
        scoped_state = (
            dict(
                state,
                results=[
                    r for r in state["results"] if r.get("concurrency") == output["concurrency"]
                ],
            )
            if "concurrency" in output
            else state
        )
        metric = scoped_comparison.get("metrics", {}).get(
            mapping.get(output.get("base_output_id", output["id"]))
        )
        usable = metric and "baseline_mean" in metric
        reason = (
            scoped_comparison.get("reason", "Experiment incomplete")
            if usable
            else "Required measurement unavailable"
        )
        claim = {
            "id": output["id"],
            "title": output["title"],
            "campaign_id": manifest["id"],
            "status": "qualified_measurement"
            if scoped_comparison.get("qualified") and usable
            else "not_qualified",
            "reason": reason,
            "measurement": metric,
            "mode": spec["mode"],
            "scope": scoped_comparison.get("scope"),
            "concurrency": output.get("concurrency", spec["concurrency"]),
            "quality_scope": manifest["quality_scope"],
            "certification": "No certification claimed",
            "positive_claim_supported": bool(
                scoped_comparison.get("qualified")
                and usable
                and metric.get("paired_bootstrap_95_interval")
                and metric["paired_bootstrap_95_interval"][0] > 0
            ),
        }
        if output["id"] == "P10-O3" and (path / "block-00-profile/profile-summary.json").exists():
            claim.update(
                reason="Exploratory CUDA trace diagnostic; no kernel-only speedup claim",
                profile=read_json(path / "block-00-profile/profile-summary.json"),
            )
        write_json(directory / "claim.json", claim)
        summary, methods, rows, engines = report_sections(
            manifest, scoped_state, scoped_comparison, claim, path
        )
        if output["id"] == "P10-O3" and (path / "block-00-profile/profile-summary.json").exists():
            summary = "Exploratory CUDA trace diagnostic. Profiled timings are excluded from performance comparisons. See the trace summary and raw evidence."
        text = "# " + output["title"] + "\n\n" + summary + "\n\n## Methodology\n\n"
        text += "\n".join(f"- **{key}:** {value}" for key, value in methods)
        text += "\n\n## Paired measurements\n\n"
        headers = [
            "Block",
            "Arm",
            "Attempts",
            "Correct",
            "Qualified per hour",
            "p95 seconds",
            "Gates",
        ]
        widths = [0.45, 0.85, 0.75, 0.85, 1.55, 1.05, 0.9]
        if spec["project_id"] == "P02":
            headers[5] = "p95 TTFT seconds"
        if spec["project_id"] == "P03":
            headers[2:2] = ["Records", "Conc."]
            widths = [0.4, 0.75, 0.6, 0.65, 0.65, 0.65, 1.3, 0.9, 0.6]
        text += "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
        text += "\n".join("| " + " | ".join(row) + " |" for row in rows)
        text += "\n\n## Interpretation limits\n\n" + limits
        text += "\n\n## Effective configurations\n\n" + "\n\n".join(
            "```\n" + e + "\n```" for e in engines
        )
        if claim.get("profile"):
            text += "\n\n## CUDA kernel diagnostic\n\nSummed kernel microseconds; overlapping durations are not elapsed time.\n\n"
            text += "\n".join(
                f"- {name}: {duration:.3f} microseconds"
                for name, duration in claim["profile"]["top_kernels"][:10]
            )
        (directory / "claim-brief.md").write_text(text)
        doc = Document()
        section = doc.sections[0]
        section.top_margin = section.bottom_margin = Inches(0.7)
        section.left_margin = section.right_margin = Inches(0.75)
        for name in ("Normal", "Title", "Heading 1", "Heading 2"):
            style = doc.styles[name]
            style.font.name = "Calibri"
            style.font.color.rgb = RGBColor(0, 0, 0)
        for style in doc.styles:
            for border in list(style.element.iter(qn("w:pBdr"))):
                border.getparent().remove(border)
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        doc.styles["Normal"].font.size = Pt(11)
        doc.styles["Normal"].paragraph_format.space_after = Pt(5)
        doc.add_paragraph(output["title"].replace("-", " ").replace("/", " ").capitalize(), "Title")
        doc.add_paragraph(summary)
        doc.add_paragraph(f"Campaign {manifest['id']} | Output {output['id']}")
        doc.add_heading("Paired measurements", 1)
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        for cell, value in zip(table.rows[0].cells, headers, strict=True):
            cell.text = value
        for row in rows:
            for cell, value in zip(table.add_row().cells, row, strict=True):
                cell.text = value
        format_table(table, widths)
        if not rows:
            doc.add_paragraph("No complete measurement arms are available.")
        if claim.get("profile"):
            doc.add_heading("CUDA kernel diagnostic", 1)
            doc.add_paragraph(
                "Ten largest summed kernel durations from the separate baseline diagnostic. Durations can overlap and are not wall time. Names below are limited to 160 characters; full names and traces are in the evidence."
            )
            kernels = doc.add_table(rows=1, cols=2)
            kernels.style = "Table Grid"
            kernels.rows[0].cells[0].text = "Kernel"
            kernels.rows[0].cells[1].text = "Summed microseconds"
            for name, duration in claim["profile"]["top_kernels"][:10]:
                cells = kernels.add_row().cells
                cells[0].text = name[:160] + ("…" if len(name) > 160 else "")
                cells[1].text = f"{duration:,.3f}"
            format_table(kernels, [4.7, 1.7])
        doc.add_heading("Interpretation limits", 1)
        doc.add_paragraph(limits)
        doc.add_heading("Methodology", 1)
        table = doc.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        for key, value in methods:
            cells = table.add_row().cells
            cells[0].text, cells[1].text = key, value
        format_table(table, [1.9, 4.5], header=False)
        doc.add_heading("Effective serving configurations", 1)
        doc.add_paragraph(
            "These settings were resolved for each arm. The evidence archive preserves the full configuration and image and model fingerprints."
        )
        for engine in engines:
            paragraph = doc.add_paragraph(engine)
            if spec["project_id"] == "P10":
                paragraph.paragraph_format.line_spacing = 1
            for run in paragraph.runs:
                run.font.size = Pt(9)
        doc.save(directory / "white-paper.docx")
        chart(directory, output["title"], metric)
    archive_evidence(path, state)


def archive_evidence(path, state):
    renderer = path / "report-renderer.py"
    renderer.write_bytes(Path(__file__).read_bytes())
    write_json(
        path / "report-generation.json",
        {
            "renderer_sha256": sha256(renderer),
            "measurement_manifest_sha256": sha256(path / "manifest.json"),
            "note": "Report rendering does not change measurements or qualification criteria",
        },
    )
    files = [
        p
        for p in path.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.suffix not in (".zip", ".lock")
        and p.name not in ("SHA256SUMS.json", "state.json", "worker.log")
    ]
    write_json(path / "final-state.json", state)
    files.append(path / "final-state.json")
    files = sorted(set(files))
    checksums = {str(p.relative_to(path)): sha256(p) for p in files}
    write_json(path / "SHA256SUMS.json", checksums)
    descriptor, temporary = tempfile.mkstemp(prefix=".evidence-", suffix=".zip", dir=path)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in files + [path / "SHA256SUMS.json"]:
                archive.write(p, str(p.relative_to(path)))
        os.replace(temporary, path / "evidence.zip")
    finally:
        Path(temporary).unlink(missing_ok=True)
