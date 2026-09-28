"""Compact editable reports with separate output records and full measurement rows."""

import json

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from ..benchmarks.storage import write_json


def render_outputs(path, manifest, state, comparison, headers, rows, methods, values):
    from .report import format_table

    methods = list(methods) + recorded_methods(path)
    reason = comparison["reason"]
    if state["phase"] != "complete":
        reason = "Incomplete experiment: " + str(state.get("error", state["phase"]))
    for index, output in enumerate(manifest["project"]["outputs"]):
        folder = path / "outputs" / output["id"]
        folder.mkdir(parents=True, exist_ok=True)
        claim = dict(
            id=output["id"],
            title=output["title"],
            campaign_id=path.name,
            status="not_qualified",
            reason=reason,
            measurement=values[index],
            positive_claim_supported=False,
            mode="pilot",
        )
        write_json(folder / "claim.json", claim)
        doc = Document()
        sec = doc.sections[0]
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11)
        sec.top_margin = sec.bottom_margin = Inches(0.65)
        sec.left_margin = sec.right_margin = Inches(0.7)
        for name in ["Normal", "Title", "Heading 1", "Heading 2"]:
            doc.styles[name].font.name = "Calibri"
            doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
        doc.styles["Normal"].font.size = Pt(10)
        doc.styles["Normal"].paragraph_format.space_after = Pt(5)
        doc.styles["Normal"].paragraph_format.line_spacing = 1.05
        doc.styles["Title"].font.size = Pt(24)
        for style in doc.styles:
            for border in list(style.element.iter(qn("w:pBdr"))):
                border.getparent().remove(border)
        doc.add_paragraph(output["title"].replace("-", " ").replace("/", " per "), "Title")
        doc.add_paragraph(reason)
        doc.add_paragraph("Campaign " + path.name + " | Output " + output["id"])
        if values[index]:
            doc.add_heading("Measured comparison", 1)
        elif not manifest["spec"].get("p01_goal_seek"):
            doc.add_paragraph("No complete passing matched comparison is available.")
        # Distinct output interpretation without concealing shared evidence.
        if manifest["spec"].get("p01_goal_seek"):
            from .goal_report import peak_text

            for text in peak_text(index, values[index], manifest["spec"]):
                doc.add_paragraph(text)
        elif manifest["spec"]["project_id"] == "P01":
            if index == 0:
                for split, rate in (values[index] or {}).items():
                    doc.add_paragraph(
                        f"{'Aggregate' if split == '0' else split + ' prefill GPUs'}: highest passing tested arrival rate {rate if rate is not None else 'not established'} requests/second."
                    )
            elif index == 1:
                for r in values[index] or []:
                    doc.add_paragraph(
                        f"{r['prefill']} prefill GPUs, {r['rate']:g} requests/second: p95 TTFT {r['baseline_ttft_p95_s']:.3f} to {r['treatment_ttft_p95_s']:.3f} seconds."
                    )
            else:
                for r in values[index] or []:
                    doc.add_paragraph(
                        f"{r['prefill']} prefill GPUs, {r['rate']:g} requests/second: p95 completion {r['baseline_p95_s']:.3f} to {r['treatment_p95_s']:.3f} seconds ({r['latency_reduction_percent']:.2f}% reduction)."
                    )
        else:
            for r in values[index] or []:
                if index == 0:
                    text = f"{r['placement']} at {r['rate']:g} calls/second: qualified audio-hours/hour {r['baseline_audio_hours_per_hour']:.3f} dedicated versus {r['treatment_audio_hours_per_hour']:.3f} consolidated."
                elif index == 1:
                    text = f"{r['placement']} at {r['rate']:g} calls/second: p95 total latency {r['baseline_p95_s']:.3f} dedicated versus {r['treatment_p95_s']:.3f} seconds consolidated."
                else:
                    text = f"{r['placement']} at {r['rate']:g} calls/second passed on one GPU versus two dedicated GPUs: 50% fewer GPUs for this offered workload."
                doc.add_paragraph(text)
        doc.add_heading("All measured cells", 1)
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        for cell, text in zip(table.rows[0].cells, headers, strict=True):
            cell.text = text
        for row in rows:
            for cell, value in zip(table.add_row().cells, row, strict=True):
                cell.text = f"{value:.3f}" if isinstance(value, float) else str(value)
        format_table(
            table,
            [1.0, 0.8, 0.65, 0.8, 1.0, 1.0, 0.6]
            if manifest["spec"].get("p01_goal_seek")
            else [0.45, 1.0, 0.65, 0.8, 1.1, 1.1, 0.6],
        )
        doc.add_heading("Methodology and limits", 1)
        for key, value in methods:
            p = doc.add_paragraph()
            p.add_run(key + ": ").bold = True
            p.add_run(str(value))
        doc.save(folder / "white-paper.docx")
        (folder / "claim-brief.md").write_text(
            "# "
            + output["title"]
            + "\n\n"
            + reason
            + "\n\n```json\n"
            + json.dumps(values[index], indent=2)
            + "\n```\n\n"
            + "\n".join(f"- **{k}:** {v}" for k, v in methods)
        )


def recorded_methods(path):
    """Put the recorded machine and runtime identity in the document itself."""
    from datetime import datetime, timezone
    from pathlib import Path

    import yaml

    from ..benchmarks.storage import read_json

    manifest = read_json(path / "manifest.json")
    methods = [
        ("Recorded time", datetime.fromtimestamp(manifest["created_at"], timezone.utc).isoformat())
    ]
    environments = [path / "environment.json", *sorted(path.glob("block-*/environment.json"))]
    for file in environments:
        if not file.exists():
            continue
        env = read_json(file)
        cpu = {}
        for line in env.get("cpu_inventory", "").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                cpu[key.strip()] = value.strip()
        gpu = env.get("gpu_inventory", "").splitlines()
        methods += [
            ("Host", env.get("hostname", "Unknown") + "; " + env.get("platform", "Unknown")),
            (
                "Processor",
                f"{cpu.get('Socket(s)', '?')} sockets; {cpu.get('Model name', 'Unknown')}; {cpu.get('Core(s) per socket', '?')} cores/socket",
            ),
            ("Host memory", f"{env.get('ram_bytes', 0) / 2**30:.1f} GiB"),
            (
                "GPU and driver",
                "; ".join(gpu[0].split(",")[1:]).strip() if gpu else "See environment inventory",
            ),
        ]
        break
    assets = [path / "summary-assets.json", *sorted(path.glob("block-*/assets.json"))]
    for file in assets:
        if file.exists():
            record = read_json(file)
            methods += [
                ("Language model revision", Path(record.get("snapshot", "")).name),
                ("Language runtime image", record.get("image", {}).get("id", "See evidence")),
            ]
            break
    launches = sorted(path.glob("block-*/launch-resolved.yaml"))
    if launches:
        resolved = yaml.safe_load(launches[0].read_text())
        engine = resolved.get("engine", resolved)
        settings = {
            key: engine.get(key)
            for key in [
                "max_model_len",
                "max_num_seqs",
                "kv_cache_dtype",
                "quantization",
                "quantization_kind",
                "vllm_extra_flags",
            ]
            if engine.get(key) is not None
        }
        methods.append(("Resolved common engine settings", json.dumps(settings)))
    if (path / "speech-assets.json").exists():
        methods.append(("Speech runtime image", read_json(path / "speech-assets.json")["image"]))
    if (path / "speech-effective.json").exists():
        methods.append(
            (
                "Speech runtime packages",
                json.dumps(read_json(path / "speech-effective.json")["packages"]),
            )
        )
    return methods
