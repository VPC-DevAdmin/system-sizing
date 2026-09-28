import json

import pytest

from simulator.infrastructure.diffusion import corpus, raw_comparison
from simulator.infrastructure.spec import ExperimentSpec


def test_diffusion_schedule_needs_no_llm_result_and_preserves_prompts(tmp_path, monkeypatch):
    from simulator.infrastructure import campaign
    from simulator.infrastructure.report import export

    monkeypatch.setattr(
        campaign, "recorded_config", lambda *a: pytest.fail("P05 must not use an LLM configuration")
    )
    spec = ExperimentSpec(project_id="P05", paired_blocks=1, diffusion_samples=4)
    path = campaign.create(tmp_path, {"model": "Qwen-Image / Qwen-Image-Flash"}, spec, "f" * 32)
    manifest = campaign.verify(path)
    assert {(r["arm"], r["steps"]) for r in manifest["schedule"]} == {
        ("baseline", 20),
        ("baseline", 50),
        ("treatment", 4),
    }
    assert json.loads((path / "corpus.json").read_text()) == corpus(spec)
    export(path, {"qualified": False, "metrics": {}})
    report = (path / "outputs/P05-O1/claim-brief.md").read_text()
    assert "Human acceptance review pending" in report and "FastGen" in report
    assert "logistics" not in report


def test_failed_generation_stays_in_rate_denominator():
    result = raw_comparison(
        [
            {
                "arm": "treatment",
                "steps": 4,
                "metrics": {"attempted": 2, "completed": 1, "elapsed_s": 20},
            }
        ]
    )
    assert result["raw_generation"]["treatment-4"]["images_per_hour"] == 180
    assert not result["qualified"]


def test_model_review_does_not_become_human_acceptance(tmp_path, monkeypatch):
    from simulator.infrastructure import campaign
    from simulator.infrastructure.image_review import submit
    from simulator.infrastructure.report import export

    spec = ExperimentSpec(project_id="P05", paired_blocks=1, diffusion_samples=4)
    path = campaign.create(tmp_path, {"model": "images"}, spec, "a" * 32)
    folder = path / "block-00-treatment-steps-4/replica-0"
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps({"arm": "treatment", "steps": 4, "block": 0}))
    (folder / "image-0000.png").write_bytes(b"fixture-image")
    (folder / "images.jsonl").write_text(
        json.dumps(
            {
                "id": "image-0000",
                "ok": True,
                "prompt": "one mug",
                "image": "image-0000.png",
                "elapsed_s": 1,
                "sha256": __import__("hashlib").sha256(b"fixture-image").hexdigest(),
            }
        )
        + "\n"
    )
    state = json.loads((path / "state.json").read_text())
    state.update(
        phase="complete",
        results=[
            {
                "arm": "treatment",
                "steps": 4,
                "metrics": {"attempted": 1, "completed": 1, "elapsed_s": 1},
            }
        ],
    )
    (path / "state.json").write_text(json.dumps(state))
    comparison = raw_comparison(state["results"])
    (path / "comparison.json").write_text(json.dumps(comparison))
    export(path, comparison)
    from simulator.infrastructure.diffusion_report import inventory

    identity = inventory(path)[0]["blind_id"]
    items = [
        {"blind_id": identity, "scores": {"subject": True, "composition": True, "integrity": True}}
    ]
    submit(path, "Automated reviewer", "model", items)
    claim = json.loads((path / "outputs/P05-O1/claim.json").read_text())
    assert not claim["review_complete"] and claim["measurement"] is None
    submit(path, "Named customer reviewer", "human", items)
    claim = json.loads((path / "outputs/P05-O1/claim.json").read_text())
    assert claim["review_complete"] and claim["measurement"]["treatment-4"]["accepted"] == 1
    assert not claim["positive_claim_supported"]
    assert json.loads((path / "state.json").read_text())["outcome"] == "reviewed_pilot"
    assert len(list((path / "evidence-revisions").glob("*.zip"))) == 2
    items[0]["scores"]["subject"] = "true"
    with pytest.raises(ValueError, match="boolean"):
        submit(path, "reviewer", "human", items)


def test_failed_archive_rebuild_preserves_existing_zip(tmp_path, monkeypatch):
    from simulator.infrastructure import report

    (tmp_path / "manifest.json").write_text("{}")
    previous = b"previous immutable archive"
    (tmp_path / "evidence.zip").write_bytes(previous)

    def fail(*args, **kwargs):
        raise OSError("disk write failure")

    monkeypatch.setattr(report.zipfile.ZipFile, "write", fail)
    with pytest.raises(OSError, match="disk write failure"):
        report.archive_evidence(tmp_path, {"phase": "complete"})
    assert (tmp_path / "evidence.zip").read_bytes() == previous
    assert not list(tmp_path.glob(".evidence-*"))
