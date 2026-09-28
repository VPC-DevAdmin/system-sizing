"""Editable methodology report and a hashed evidence bundle."""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .storage import read_json, sha256, write_json


def _new_document():
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.7)
    section.left_margin = section.right_margin = Inches(0.75)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Title", "Heading 1", "Heading 2"):
        doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    for style in doc.styles:
        for border in list(style.element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    doc.core_properties.author = "capsim"
    doc.core_properties.comments = ""
    doc.core_properties.created = doc.core_properties.modified = datetime.now(timezone.utc)
    return doc


def _table(doc, headers, rows):
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + edge)
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn("w:" + key), value)
        borders.append(border)
    t._tbl.tblPr.append(borders)
    for c, value in zip(t.rows[0].cells, headers, strict=True):
        c.text = value
    props = t.rows[0]._tr.get_or_add_trPr()
    props.append(OxmlElement("w:tblHeader"))
    for values in rows:
        for c, value in zip(t.add_row().cells, values, strict=True):
            c.text = "Unavailable" if value is None else str(value)
    for index, r in enumerate(t.rows):
        r._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for column, cell in enumerate(r.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if index == 0:
                fill = OxmlElement("w:shd")
                fill.set(qn("w:fill"), "E8EEF5")
                cell._tc.get_or_add_tcPr().append(fill)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if column == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = False
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0, 0, 0)
                    run.bold = index == 0
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(4)
    spacer.add_run().font.size = Pt(1)
    return t


def export_report(directory: Path) -> Path:
    manifest = read_json(directory / "manifest.json")
    state = read_json(directory / "state.json")
    doc = _new_document()
    doc.add_heading("Third party inference benchmark report", 0)
    row = manifest["source_result"]
    doc.add_paragraph(f"{row['model']} on {row['engine']}")
    doc.add_paragraph(
        f"Experiment {state['id']} · Outcome: {state.get('outcome') or state['phase']}"
    )
    if state.get("created_at"):
        created = datetime.fromtimestamp(state["created_at"], timezone.utc).isoformat(
            timespec="seconds"
        )
        finished = max((a.get("ended_at", 0) for a in state.get("attempts", [])), default=0)
        timing = f"Campaign created: {created}."
        if finished:
            timing += (
                " Last attempt finished: "
                + datetime.fromtimestamp(finished, timezone.utc).isoformat(timespec="seconds")
                + ". Attempt times include setup and artifact processing."
            )
        doc.add_paragraph(timing)
    doc.add_paragraph(
        "This report records performance measured with the named harnesses under "
        "the conditions below. It does not establish independent certification, "
        "model quality, or concurrent user capacity."
    )
    if (directory / "analysis-revision.json").exists():
        revision = read_json(directory / "analysis-revision.json")
        doc.add_paragraph(
            "Analysis revision: " + revision["reason"] + " "
            "The original verdict, report and evidence ZIP are preserved under analysis-history. "
            "The revision record identifies the analysis code and original artifact checksums. "
            "Native benchmark measurements were not changed."
        )

    def details(title, data):
        doc.add_heading(title, 1)
        for key, val in data.items():
            if val is None:
                val = "Not explicitly set"
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False, sort_keys=True)
            # Break long identifiers at safe boundaries for readable Word wrapping.
            val = str(val).replace("/", "/\u200b").replace(", ", ",\u200b ")
            p = doc.add_paragraph()
            p.add_run(key.replace("_", " ") + ": ").bold = True
            p.add_run(val)

    doc.add_heading("Measured results", 1)
    attempts = state.get("attempts", [])
    completed = [a for a in attempts if a.get("metrics")]
    _table(
        doc,
        ["Harness", "Load", "Repeat", "Output tok/s", "TTFT p95 ms", "Outcome"],
        [
            (
                a["harness"],
                a["point"],
                a["repetition"],
                round(a["metrics"]["output_tok_s"], 2),
                round(a["metrics"]["ttft_p95_ms"], 2),
                a["outcome"],
            )
            for a in completed
        ],
    )
    if not completed:
        doc.add_paragraph(
            "No valid parsed measurements are available. Inspect the attempt history."
        )
    for a in completed:
        m = a["metrics"]
        doc.add_paragraph(
            f"{a['harness']} load {a['point']}, repeat {a['repetition']}: "
            f"{m['successful']:g} successful; {m['failed']:g} failed; "
            f"incomplete requests: {m.get('incomplete') if m.get('incomplete') is not None else 'unavailable'}; "
            f"proven duration-boundary cancellations: {m.get('boundary_cancelled', 0)}; "
            f"error fraction {m['error_fraction']:.6g}; measured {m['duration_s']:.2f} seconds."
        )
        optional_metrics = {
            key: m.get(key)
            for key in (
                "requests_s",
                "input_tokens_mean",
                "output_tokens_mean",
                "tpot_p95_ms",
                "itl_p95_ms",
                "latency_p95_ms",
                "completed_output_tok_s",
                "client_delay_ms",
                "targeted_start_delay_ms",
            )
        }
        doc.add_paragraph(
            "; ".join(
                f"{key.replace('_', ' ')}: {f'{value:.2f}' if isinstance(value, (int, float)) else 'unavailable'}"
                for key, value in optional_metrics.items()
            )
        )
        routing_file = directory / a["directory"] / "routing.json"
        if routing_file.exists():
            routing = read_json(routing_file)
            fraction = routing.get("max_worker_lag_over_100ms_fraction")
            doc.add_paragraph(
                f"Routing workers: {routing.get('workers', 1)}; maximum loop delay: "
                f"{routing.get('max_loop_lag_ms', 0):.2f} ms; worst worker fraction of "
                f"samples above 100 ms: {f'{fraction:.4%}' if fraction is not None else 'unavailable'}."
            )
    doc.add_paragraph(
        "Requests use streaming Chat Completions with temperature 0 and the recorded ignore_eos setting. Recorded reasoning effort is forwarded when reasoning is enabled. Native token-count and latency definitions are retained by harness."
    )
    if manifest.get("adapter_version", 1) >= 2:
        doc.add_paragraph(
            "GuideLLM throughput is its total token rate inside the fixed measurement window, "
            "including tokens from partial requests at the boundary. Its completed-request rate is "
            "shown separately. AIPerf reports completed-request throughput over its native duration, "
            "including draining. Latency percentiles use successful requests. Proven duration-boundary "
            "cancellations remain visible and are excluded from the request-error fraction; early or "
            "unproven cancellations count as failures. SLO runs are inconclusive when the boundary-cancelled "
            "fraction exceeds the configured error limit."
        )
    else:
        doc.add_paragraph(
            "Adapter version 1 reports GuideLLM successful-request throughput and counts all incomplete requests in its error fraction."
        )
    details("Selection and scope", manifest["selection"])
    if row.get("steady_state") is False:
        doc.add_paragraph(
            "The source roofline remeasurement did not reach its stability criterion. "
            "Its throughput is unconfirmed; the external measurements above assess the recorded settings independently."
        )
    details("Methodology", manifest["recipe"])
    if manifest.get("adapter_version", 1) >= 2:
        doc.add_paragraph(
            "Routing saturation is flagged when more than 5% of 250 ms loop-delay samples during active requests "
            "on any worker exceed 100 ms. Maximum delay remains visible. GuideLLM mean "
            "request-start delay above 100 ms is a separate client check; Poisson SLO runs use "
            "targeted-start delay, including scheduler queuing. These checks detect "
            "possible client limitations; they do not prove unlimited load-generator headroom."
        )
    else:
        doc.add_paragraph(
            "Adapter version 1 flags maximum routing delay or mean GuideLLM targeted-start delay above 100 ms."
        )
    details("Harness versions", manifest["harness_versions"])
    if (directory / "environment.json").exists():
        env = read_json(directory / "environment.json")
        cpu_fields = (
            "Model name",
            "Socket(s)",
            "Core(s) per socket",
            "Thread(s) per core",
            "NUMA node(s)",
            "CPU(s)",
        )
        cpu = {
            line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
            for line in env.get("cpu_inventory", "").splitlines()
            if ":" in line and line.split(":", 1)[0].strip() in cpu_fields
        }
        if cpu:
            details("CPU topology", cpu)
        details(
            "System and resolved assets",
            {
                k: v
                for k, v in read_json(directory / "environment.json").items()
                if k not in {"cpu_inventory", "supervisor_runtime"}
            },
        )
        if env.get("supervisor_runtime"):
            runtime = env["supervisor_runtime"]
            doc.add_paragraph(
                f"Supervisor Python: {runtime['executable']}. "
                f"The complete inventory of {len(runtime['packages'])} installed packages "
                "and versions is retained in environment.json in the evidence bundle."
            )
    if (directory / "dataset.json").exists():
        details("Dataset", read_json(directory / "dataset.json"))
    from dataclasses import asdict

    from ..config import EngineConfig

    default_engine = asdict(EngineConfig())
    core = {
        "type",
        "model_id",
        "quantization",
        "quantization_kind",
        "gpu_image",
        "replica_devices",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "served_model_name",
    }
    settings = manifest["configuration"]["engine"]
    details(
        "Recorded engine configuration",
        {k: v for k, v in settings.items() if k in core or default_engine.get(k) != v},
    )
    doc.add_paragraph(
        "The evidence bundle contains the recorded configuration in engine.yaml and the pinned launch configuration in launch-resolved.yaml."
    )
    doc.add_heading("Attempts and recovery", 1)
    for a in attempts:
        doc.add_heading(
            f"{a['harness']} load {a['point']} repeat {a['repetition']} attempt {a['number']}", 2
        )
        outcome = doc.add_paragraph(
            f"Outcome: {a.get('outcome', 'interrupted')}. " + "; ".join(a.get("reasons", []))
        )
        outcome.paragraph_format.keep_with_next = True
        doc.add_paragraph(f"Evidence directory: {a['directory']}")
    if state.get("error"):
        doc.add_paragraph(state["error"])
    doc.add_heading("Limits and interpretation", 1)
    for note in manifest["limitations"]:
        doc.add_paragraph(note, style="List Bullet")
    doc.add_paragraph(
        "Each repetition is shown separately. No cross-harness percentile averaging "
        "or conversion between inter-token latency and time per output token is used. "
        "Missing metrics remain unavailable. Hardware telemetry covers the local host; "
        "GPU power is not whole-system wall power."
    )
    doc.add_heading("Reproduction and evidence", 1)
    doc.add_paragraph(
        "The evidence ZIP contains this editable report, experiment manifest, "
        "dependency locks, exact commands for each attempt, dataset, raw outputs, "
        "telemetry, and SHA256SUMS.json. Restore the recorded model and runtime assets "
        "on matching hardware and rerun the recipe through capsim. Local paths and the "
        "temporary routing endpoint must be resolved for the new machine."
    )
    doc.add_paragraph(
        "The manifest and original harness artifacts are authoritative if this editable "
        "document is subsequently changed."
    )
    # Avoid default author metadata from python-docx's bundled template.
    doc.core_properties.author = "capsim"
    doc.core_properties.title = "Third party inference benchmark report"
    doc.core_properties.comments = ""
    target = directory / "report.docx"
    temp = directory / ".report.docx"
    doc.save(temp)
    os.replace(temp, target)
    return target


def export_bundle(directory: Path) -> Path:
    export_report(directory)
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.name not in {"evidence.zip", ".evidence.zip", "SHA256SUMS.json", "worker.log"}
        and not p.name.startswith(".")
    )
    # Do not dereference artifact directory symlinks during export.
    files = [
        p
        for p in files
        if all(not (directory / q).is_symlink() for q in p.relative_to(directory).parents)
    ]
    write_json(
        directory / "SHA256SUMS.json", {str(p.relative_to(directory)): sha256(p) for p in files}
    )
    temp = directory / ".evidence.zip"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + [directory / "SHA256SUMS.json"]:
            z.write(p, str(p.relative_to(directory)))
    target = directory / "evidence.zip"
    os.replace(temp, target)
    return target


def export_search(base: Path, links: list[dict]) -> Path:
    """One report for a roofline search, retaining full per-model reports and bundles."""
    import copy
    import statistics

    from docx import Document

    from .spec import digest
    from .storage import campaign_path

    records = []
    for link in links:
        entry = dict(link)
        if link.get("id"):
            path = campaign_path(base, link["id"])
            entry["state_sha256"] = sha256(path / "state.json")
        records.append(entry)
    folder = base / "benchmark-reports" / digest(records)[:24]
    folder.mkdir(parents=True, exist_ok=True)
    doc = _new_document()
    doc.core_properties.title = "Roofline third party benchmark results"
    doc.add_heading("Roofline third party benchmark results", 0)
    doc.add_paragraph(
        "Named-harness performance measurements. Results are not independent certification. "
        "Per-model methods, attempts and limitations follow. Missing reports remain listed."
    )
    for record in records:
        doc.add_paragraph(
            f"{record.get('model', 'Selection')}: {record.get('status', 'unknown')}. "
            f"{record.get('error', '')}"
        )
    summary = []
    for record in records:
        if not record.get("id"):
            continue
        path = campaign_path(base, record["id"])
        manifest = read_json(path / "manifest.json")
        groups = {}
        for attempt in read_json(path / "state.json").get("attempts", []):
            if attempt.get("outcome") == "measured" and attempt.get("metrics"):
                key = (attempt["harness"], attempt["point"])
                groups.setdefault(key, {})[attempt["repetition"]] = attempt["metrics"][
                    "output_tok_s"
                ]
        for (harness, point), rates in sorted(groups.items()):
            unit = "req/s" if manifest["recipe"]["recipe"] == "slo" else "concurrent"
            summary.append(
                (
                    record.get("model", "Selection") + "\n" + harness,
                    f"{point:g} {unit}",
                    f"{statistics.median(rates.values()):,.2f}",
                    f"{len(rates)} / {manifest['recipe']['repetitions']}",
                )
            )
    if summary:
        doc.add_heading("Throughput summary", 1)
        table = _table(
            doc, ["Model and harness", "Load", "Median output tok/s", "Measured repeats"], summary
        )
        from docx.shared import Inches

        table.autofit = False
        for column, width in zip(table.columns, (3.0, 1.3, 1.4, 1.3), strict=True):
            column.width = Inches(width)
        for row in table.rows:
            for cell, width in zip(row.cells, (3.0, 1.3, 1.4, 1.3), strict=True):
                cell.width = Inches(width)
        doc.add_paragraph(
            "Medians include only measured repetitions at the same load within each harness. "
            "Native timing and token-count definitions differ between tools. "
            "GuideLLM includes tokens from partial requests inside its fixed window; "
            "AIPerf includes request drain time. Per-repetition rates, latency, errors, "
            "boundary cancellations and method details follow."
        )
    for record in records:
        if record.get("id"):
            path = campaign_path(base, record["id"])
            if (path / "report.docx").exists():
                doc.add_page_break()
                source = Document(path / "report.docx")
                for element in source.element.body:
                    if not element.tag.endswith("}sectPr"):
                        doc.element.body.insert(-1, copy.deepcopy(element))
    doc.core_properties.author = "capsim"
    doc.save(folder / ".report.docx")
    os.replace(folder / ".report.docx", folder / "report.docx")
    write_json(folder / "search.json", records)
    files = {"report.docx": folder / "report.docx", "search.json": folder / "search.json"}
    for record in records:
        if record.get("id"):
            path = campaign_path(base, record["id"]) / "evidence.zip"
            if path.is_file() and not path.is_symlink():
                files[f"models/{record['id']}.zip"] = path
    write_json(folder / "SHA256SUMS.json", {name: sha256(path) for name, path in files.items()})
    with zipfile.ZipFile(folder / ".evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in {**files, "SHA256SUMS.json": folder / "SHA256SUMS.json"}.items():
            archive.write(path, name)
    os.replace(folder / ".evidence.zip", folder / "evidence.zip")
    return folder
