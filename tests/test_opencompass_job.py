import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "oc"))
import hle_prompts as hp  # noqa: E402
import opencompass_job as ocj  # noqa: E402
import planner_campaign as pc  # noqa: E402


def _job(**oc):
    base = {"tests": ["hle", "aa_lcr"], "max_tokens": 100000, "temperature": 1.0, "top_p": 1.0,
            "api_kwargs": {"reasoning_effort": "high"}, "concurrency": 16}
    base.update(oc)
    return {"id": "OC-gptoss120b-1gpu-high", "kind": "opencompass", "model": "openai/gpt-oss-120b",
            "max_model_len": 131072, "opencompass": base}


def _settings(text: str) -> dict:
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "S")
    return ast.literal_eval(node.value)


def test_config_is_python_and_carries_the_settings():
    text = ocj.config_text(_job(), "http://127.0.0.1:9100", "openai/gpt-oss-120b")
    ast.parse(text)
    s = _settings(text)
    assert s["url"] == "http://127.0.0.1:9100/v1"
    assert s["sampling"] == {"temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"}
    assert s["max_seq_len"] == 131072 - 512
    assert s["tests"] == ["hle", "aa_lcr"]
    assert s["judge"] is None
    assert s["hle_judge_prompt"] == hp.HLE_JUDGE_PROMPT
    assert "{prediction}" in s["hle_judge_prompt"] and "{answer}" in s["hle_judge_prompt"]


def test_judge_settings_only_when_grading():
    judge = {"url": "http://127.0.0.1:9200/v1", "served": "qwen35", "model": "Qwen/Qwen3.5-397B"}
    s = _settings(ocj.config_text(_job(), "http://x/v1", "m", judge))
    assert s["judge"] == judge


def test_unknown_test_is_refused():
    with pytest.raises(ValueError):
        ocj.config_text(_job(tests=["critpt"]), "http://x/v1", "m")


def test_hle_prompts_follow_answer_type():
    assert "Exact Answer:" in hp.hle_system_prompt("exactMatch")
    assert "Exact Answer:" not in hp.hle_system_prompt("multipleChoice")
    assert "Answer: {your chosen answer}" in hp.hle_system_prompt("multipleChoice")


def test_final_answer_drops_reasoning_and_empties_cut_offs():
    assert hp.final_answer("thinking" + hp.FINAL_MARK + "Exact Answer: 4") == "Exact Answer: 4"
    assert hp.final_answer("still thinking when the tokens ran out") == ""


@pytest.mark.parametrize("reply,grade", [
    ("extracted_final_answer: 4\nreasoning: matches\ncorrect: yes\nconfidence: 90", "A"),
    ("correct: **no**", "B"),
    ("Correct - Yes", "A"),
    ("I could not decide", "unknown"),
    ("correct: no ... on reflection\ncorrect: yes", "A"),
])
def test_hle_judgement(reply, grade):
    assert hp.hle_judgement(reply) == grade


def test_answer_summary_counts_cut_offs(tmp_path):
    job = _job()
    pdir = tmp_path / "oc" / ocj.REUSE / "predictions" / ocj.abbr(job)
    pdir.mkdir(parents=True)
    rows = {"0": {"prediction": "r" + hp.FINAL_MARK + "a"}, "1": {"prediction": "ran out"}}
    (pdir / "HLE-text_0.json").write_text(json.dumps(rows))
    (pdir / "HLE-text_1.json").write_text(json.dumps({"2": {"prediction": "r" + hp.FINAL_MARK + "b"}}))
    (pdir / "AA-LCR-v1.1.json").write_text(json.dumps({"0": {"prediction": "x" + hp.FINAL_MARK + "y"}}))
    per = ocj.answer_summary(tmp_path, job)
    assert per["HLE-text"] == {"answers": 3, "cut_off": 1, "cut_off_share": 0.3333}
    assert per["AA-LCR-v1.1"]["cut_off"] == 0


def _queue(tmp_path, jobs):
    q = pc.Queue.__new__(pc.Queue)
    q.box, q.doc, q.jobs, q.runs_dir = "xe7740", {}, jobs, tmp_path
    return q


def test_opencompass_job_done_only_with_an_ok_summary(tmp_path):
    q = _queue(tmp_path, [{"id": "OC-x", "kind": "opencompass"}])
    assert pc._job_outcome(q, "OC-x") == "failed"
    (tmp_path / "OC-x").mkdir()
    (tmp_path / "OC-x" / "opencompass_summary.json").write_text(json.dumps({"status": "failed"}))
    assert pc._job_outcome(q, "OC-x") == "failed"
    (tmp_path / "OC-x" / "opencompass_summary.json").write_text(json.dumps({"status": "ok"}))
    assert pc._job_outcome(q, "OC-x") == "done"
    # A LiveBench summary does not count for an OpenCompass job.
    (tmp_path / "OC-x" / "opencompass_summary.json").unlink()
    (tmp_path / "OC-x" / "livebench_summary.json").write_text(json.dumps({"status": "ok"}))
    assert pc._job_outcome(q, "OC-x") == "failed"
