"""Raw generation reports and a separate, explicit acceptance-review ledger."""

import hashlib
import json
import random
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from ..benchmarks.storage import read_json, write_json

RUBRIC = {
    "subject": "Requested objects, counts and colors are correct",
    "composition": "Requested layout and background are present",
    "integrity": "No major visual defects that prevent use as a concept image",
}


def inventory(path):
    rows = []
    for f in sorted(path.glob("block-*/replica-0/images.jsonl")):
        plan = read_json(f.parent / "plan.json")
        for row in [json.loads(x) for x in f.read_text().splitlines()]:
            if row["ok"]:
                image = f.parent / row["image"]
                if (
                    Path(row["image"]).name != row["image"]
                    or image.suffix != ".png"
                    or image.is_symlink()
                    or hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("Image review evidence was modified or has an invalid path")
                identity = hashlib.sha256(
                    (path.name + str(f.relative_to(path)) + row["id"]).encode()
                ).hexdigest()[:24]
                rows.append(
                    dict(
                        row,
                        blind_id=identity,
                        arm=plan["arm"],
                        steps=plan["steps"],
                        block=plan["block"],
                        path=str((f.parent / row["image"]).relative_to(path)),
                    )
                )
    random.Random(90210).shuffle(rows)
    return rows


def render(path, manifest, state, comparison):
    rows = inventory(path)
    write_json(path / "blind-review-map.json", rows)
    review = (
        read_json(path / "image-review.json")
        if (path / "image-review.json").exists()
        else {"items": {}}
    )
    accepted = {}
    complete_review = bool(rows) and all(r["blind_id"] in review["items"] for r in rows)
    if complete_review:
        for row in rows:
            item = review["items"][row["blind_id"]]
            if item.get("reviewer_kind") == "human" and all(
                item["scores"].get(k) is True for k in RUBRIC
            ):
                key = f"{row['arm']}-{row['steps']}"
                accepted[key] = accepted.get(key, 0) + 1
    human_review = complete_review and all(
        review["items"][r["blind_id"]].get("reviewer_kind") == "human" for r in rows
    )
    methods = {
        "Harness": manifest["harness"],
        "Campaign": path.name,
        "Teacher": "Qwen/Qwen-Image @ " + manifest["diffusion_models"]["baseline"][1],
        "Student": "nvidia/Qwen-Image-Flash @ " + manifest["diffusion_models"]["treatment"][1],
        "Student provenance": "Released FastGen/DMD2-distilled checkpoint; inference uses Diffusers. No distillation run here.",
        "Image size / precision": "1024 × 1024 / bfloat16",
        "Workload": f"{manifest['spec']['diffusion_samples']} prompts/seeds per cell; one image at a time; identical frozen list",
        "Teacher steps": str(manifest["spec"]["diffusion_teacher_steps"]),
        "Student steps": "4; runtime requires the packaged static shift-3 trajectory [1, 0.9, 0.75, 0.5, 0] and aborts on mismatch",
        "Guidance": "Teacher CFG 4.0; student 1.0 because teacher guidance is distilled into the student",
        "Timing": manifest["load_policy"],
        "Warmup": "One disjoint prompt/seed per fresh pipeline; excluded from rate and recorded separately",
        "Ordering": "Randomized arm/step order within each block; fresh pipeline per cell",
        "Acceptance rubric": "; ".join(RUBRIC.values()),
        "Review status": "All generated images reviewed by a named human"
        if human_review
        else "Human acceptance review pending",
        "Limits": "Exploratory concept images only. Technical PNG integrity does not establish usable image quality. No energy or cost claim without measurement.",
    }
    if (path / "report-revision.json").exists():
        methods["Report revision"] = read_json(path / "report-revision.json")["note"]
    effective_files = sorted(path.glob("block-*/replica-0/effective.json"))
    if effective_files:
        effective = read_json(effective_files[0])
        methods["GPU"] = effective["gpu"]
        methods["Runtime"] = json.dumps(effective["packages"])
    if (path / "host.json").exists():
        host = read_json(path / "host.json")
        methods["Host"] = host.get("host", "Unknown")
        cpu = host.get("commands", {}).get("cpu", {}).get("stdout", "")
        methods["CPU"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu.splitlines()
                if line.startswith("Model name:")
            ),
            "See host inventory",
        )
    methods["Reproduction evidence"] = (
        "Shared ZIP includes pinned asset inventory, Dockerfile and image ID, every container command, effective pipeline settings, prompts/seeds, timing records, PNG hashes and review audit."
    )
    raw = comparison.get("raw_generation", {})
    for output in manifest["project"]["outputs"]:
        folder = path / "outputs" / output["id"]
        folder.mkdir(parents=True, exist_ok=True)
        measurement = None
        if output["id"] in ("P05-O1", "P05-O2") and human_review:
            measurement = {}
            for key, value in raw.items():
                seconds = sum(
                    r["metrics"]["elapsed_s"]
                    for r in state["results"]
                    if f"{r['arm']}-{r['steps']}" == key
                )
                count = accepted.get(key, 0)
                measurement[key] = {
                    "accepted": count,
                    "attempted": value["attempted"],
                    "accepted_per_hour": count * 3600 / seconds if seconds else None,
                    "seconds_per_accepted_image": seconds / count if count else None,
                }
        reason = (
            "Exploratory reviewed images; no confirmed marketing claim"
            if measurement
            else (
                "Human acceptance review pending"
                if output["id"] in ("P05-O1", "P05-O2")
                else "Whole-server energy was not measured"
                if output["id"] == "P05-O3"
                else "Cost inputs and local distillation costs were not measured"
            )
        )
        if state.get("phase") in ("failed", "cancelled"):
            reason = (
                "Experiment "
                + state["phase"]
                + ": "
                + str(state.get("error", "Incomplete generation"))
            )
        claim = {
            "id": output["id"],
            "title": output["title"],
            "campaign_id": path.name,
            "status": "not_qualified",
            "reason": reason,
            "measurement": measurement,
            "positive_claim_supported": False,
            "mode": "pilot",
            "raw_generation": raw,
            "review_complete": human_review,
        }
        write_json(folder / "claim.json", claim)
        doc = Document()
        section = doc.sections[0]
        section.top_margin = section.bottom_margin = Inches(0.7)
        section.left_margin = section.right_margin = Inches(0.75)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        doc.styles["Normal"].font.name = "Calibri"
        doc.styles["Normal"].font.size = Pt(10)
        doc.styles["Normal"].paragraph_format.space_after = Pt(3)
        doc.styles["Normal"].paragraph_format.line_spacing = 1
        doc.styles["Title"].font.size = Pt(24)
        doc.styles["Title"].paragraph_format.space_after = Pt(8)
        doc.styles["Heading 1"].font.size = Pt(14)
        doc.styles["Heading 1"].paragraph_format.space_before = Pt(8)
        doc.styles["Heading 1"].paragraph_format.space_after = Pt(4)
        for style_name in ("Normal", "Title", "Heading 1", "Heading 2"):
            doc.styles[style_name].font.name = "Calibri"
            doc.styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
        for style in doc.styles:
            for border in list(style.element.iter(qn("w:pBdr"))):
                border.getparent().remove(border)
        doc.add_paragraph(
            output["title"].replace("-", " ").replace("/", " per ").capitalize(), "Title"
        )
        doc.add_paragraph(reason)
        doc.add_heading("Raw generation measurements", 1)
        table = doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        for cell, text in zip(
            table.rows[0].cells,
            ["Arm / steps", "Completed", "Attempted", "Pipeline images/hour"],
            strict=True,
        ):
            cell.text = text
        for key, value in raw.items():
            for cell, text in zip(
                table.add_row().cells,
                [
                    key,
                    str(value["generated"]),
                    str(value["attempted"]),
                    f"{value['images_per_hour']:.2f}",
                ],
                strict=True,
            ):
                cell.text = text
        doc.add_paragraph(
            "Raw pipeline rates exclude evidence-image file writing. Rejected attempts remain in the time denominator. These are not accepted-image rates."
        )
        if measurement:
            doc.add_heading("Reviewed outcomes", 1)
            for key, value in measurement.items():
                doc.add_paragraph(key + ": " + json.dumps(value))
        doc.add_heading("Methodology and limits", 1)
        for key, value in methods.items():
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
            + "\n".join(f"- **{k}:** {v}" for k, v in methods.items())
            + "\n\n## Raw generation\n\n"
            + json.dumps(raw, indent=2)
        )
    write_json(
        path / "review-template.json",
        {
            "rubric": RUBRIC,
            "reviewer": "",
            "reviewer_kind": "human",
            "items": [
                {
                    "blind_id": r["blind_id"],
                    "prompt": r["prompt"],
                    "scores": {k: None for k in RUBRIC},
                }
                for r in rows
            ],
        },
    )
