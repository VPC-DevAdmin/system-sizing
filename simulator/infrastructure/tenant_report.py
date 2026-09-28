"""Separate density, interference and fixed-set resource reports."""

import statistics

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from ..benchmarks.storage import read_json, write_json


def render(path, manifest, state, comparison):
    from .report import format_table, number

    placements = comparison.get("placements", {})
    valid = all(placements.get(a, {}).get("steady_contract_passed") for a in ("dedicated", "mig"))
    density = (
        {
            "baseline_services_per_gpu": 1,
            "mig_services_per_gpu": 2,
            "change_percent": 100,
            "software_sharing_services_per_gpu": placements.get("shared", {}).get(
                "services_per_gpu"
            ),
        }
        if valid
        else None
    )
    resource = (
        {"baseline_gpus": 2, "mig_gpus": 1, "gpu_reduction_percent": 50, "cost": "Not measured"}
        if valid
        else None
    )
    interference = {}
    for arm in ("dedicated", "shared", "mig"):
        means = {}
        for phase in ("steady", "noisy"):
            rows = [
                r["metrics"]
                for r in state["results"]
                if r["arm"] == arm and r["phase"] == phase and r["tenant"] == 0
            ]
            if len(rows) == manifest["spec"]["paired_blocks"] and all(
                r["latency_p95_s"] is not None and r["gates_passed"] for r in rows
            ):
                means[phase] = statistics.mean(r["latency_p95_s"] for r in rows)
        if means.get("steady", 0) > 0 and "noisy" in means:
            interference[arm] = {
                "victim_p95_increase_percent": 100 * (means["noisy"] / means["steady"] - 1),
                "steady_p95_s": means["steady"],
                "noisy_p95_s": means["noisy"],
            }
    measurements = {"P06-O1": density, "P06-O2": interference or None, "P06-O3": resource}
    spec = manifest["spec"]
    methods = [
        ("Model", manifest["source_result"]["model"]),
        ("Harness", manifest["harness"]),
        (
            "Physical GPU selection",
            str(spec["gpu_ids"]) + "; last GPU hosts software sharing and MIG",
        ),
        ("MIG profile", "Two 2g.48gb GPU instances with one compute instance each"),
        (
            "Memory controls",
            f"Explicit {spec['tenant_kv_gib']} GiB KV allocation per service in every placement",
        ),
        (
            "Workload",
            f"Two independent endpoints; {spec['document_records']}-record extraction documents; exact JSON grading",
        ),
        (
            "Offered load",
            f"{spec['tenant_requests_per_second']} requests/second per tenant; same schedule in all placements",
        ),
        (
            "Noisy phase",
            f"Tenant 1 increases to {spec['tenant_noise_multiplier']} times its rate; tenant 0 demand stays fixed",
        ),
        (
            "Queueing",
            "Response time includes client admission queue; rejected arrivals count as failures",
        ),
        (
            "Duration",
            f"{spec['duration_s']} seconds per phase plus drain; {spec['paired_blocks']} randomized blocks",
        ),
        (
            "Targets",
            f"At least {spec['min_quality']:.0%} correct and {spec['min_slo_fraction']:.0%} correct within {spec['response_slo_s']} seconds",
        ),
        ("Warmup", f"{spec['warmup_s']} seconds per phase on a disjoint corpus"),
        (
            "Restoration",
            "Original disabled MIG mode must be verified after all owned containers stop; journals retained",
        ),
        (
            "Limits",
            "Fixed two-service set, not maximum packing density. Hardware partitioning does not establish confidential computing. No cost or wall-energy measurement.",
        ),
    ]
    if (path / "host.json").exists():
        host = read_json(path / "host.json")
        methods.append(("Host", host.get("host", "Unknown")))
        cpu = host.get("commands", {}).get("cpu", {}).get("stdout", "")
        methods.append(
            (
                "CPU",
                next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in cpu.splitlines()
                        if line.startswith("Model name:")
                    ),
                    "See environment inventory",
                ),
            )
        )
    methods.append(
        (
            "Reproduction evidence",
            "Shared ZIP includes the frozen manifest and corpus, each launch-resolved YAML, image/model fingerprints, runtime commands, native metrics, per-request outputs and MIG restoration journal.",
        )
    )
    for output in manifest["project"]["outputs"]:
        folder = path / "outputs" / output["id"]
        folder.mkdir(parents=True, exist_ok=True)
        metric = measurements[output["id"]]
        reason = (
            "Exploratory placement result; no confirmed marketing claim"
            if metric
            else "Required complete, passing placement measurement unavailable"
        )
        if state.get("phase") in ("failed", "cancelled"):
            reason = (
                "Experiment "
                + state["phase"]
                + ": "
                + str(state.get("error", "Incomplete placement measurement"))
            )
        write_json(
            folder / "claim.json",
            {
                "id": output["id"],
                "title": output["title"],
                "campaign_id": path.name,
                "status": "not_qualified",
                "reason": reason,
                "measurement": metric,
                "positive_claim_supported": False,
                "mode": "pilot",
            },
        )
        doc = Document()
        sec = doc.sections[0]
        sec.top_margin = sec.bottom_margin = Inches(0.7)
        sec.left_margin = sec.right_margin = Inches(0.75)
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11)
        doc.styles["Normal"].font.name = "Calibri"
        doc.styles["Normal"].font.size = Pt(10)
        for style_name in ("Normal", "Title", "Heading 1", "Heading 2"):
            doc.styles[style_name].font.name = "Calibri"
            doc.styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
        for style in doc.styles:
            for border in list(style.element.iter(qn("w:pBdr"))):
                border.getparent().remove(border)
        doc.add_paragraph(output["title"].replace("-", " ").capitalize(), "Title")
        doc.add_paragraph(reason)
        if metric:
            if output["id"] == "P06-O1":
                doc.add_paragraph(
                    "The tested service set used two GPUs in the dedicated placement and one GPU with two MIG partitions. Both services passed the steady workload gates. This is 100 percent more tested services per GPU for this fixed set."
                )
                shared = density["software_sharing_services_per_gpu"]
                doc.add_paragraph(
                    f"Software sharing supported {shared:g} qualified services per GPU on the same test."
                    if shared is not None
                    else "Software sharing did not pass every steady service gate."
                )
            elif output["id"] == "P06-O2":
                for arm, value in interference.items():
                    doc.add_paragraph(
                        f"{arm.capitalize()}: victim p95 response time changed from {value['steady_p95_s']:.3f} to {value['noisy_p95_s']:.3f} seconds ({value['victim_p95_increase_percent']:+.2f} percent). Both victim windows passed their gates."
                    )
            else:
                doc.add_paragraph(
                    "The fixed set required two dedicated GPUs or one GPU with two MIG partitions, a 50 percent reduction in GPU count. Prices, licensing, operating cost and whole-server power were not measured; no cost-reduction result is reported."
                )
        doc.add_heading("Every tenant measurement", 1)
        table = doc.add_table(rows=1, cols=7)
        table.style = "Table Grid"
        headers = ["Block", "Placement", "Phase / tenant", "Attempts", "Correct", "p95 s", "Gates"]
        for cell, text in zip(table.rows[0].cells, headers, strict=True):
            cell.text = text
        for row in state["results"]:
            m = row["metrics"]
            values = [
                str(row["block"] + 1),
                row["arm"],
                f"{row['phase']} / {row['tenant']}",
                str(m["attempted"]),
                f"{m['quality_fraction']:.1%}",
                number(m["latency_p95_s"]),
                "Pass" if m["gates_passed"] else "Fail",
            ]
            for cell, text in zip(table.add_row().cells, values, strict=True):
                cell.text = text
        format_table(table, [0.45, 0.85, 1.35, 0.75, 0.8, 0.8, 0.65])
        doc.add_heading("Methodology and limits", 1)
        for key, value in methods:
            para = doc.add_paragraph()
            para.add_run(key + ": ").bold = True
            para.add_run(value)
        doc.save(folder / "white-paper.docx")
        (folder / "claim-brief.md").write_text(
            "# "
            + output["title"]
            + "\n\n"
            + reason
            + "\n\n"
            + str(metric)
            + "\n\n"
            + "\n".join(f"- **{k}:** {v}" for k, v in methods)
        )
