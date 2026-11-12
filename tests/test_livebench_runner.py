import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import livebench_job as lbj  # noqa: E402
import planner_campaign as pc  # noqa: E402


def _queue(tmp_path, jobs):
    q = pc.Queue.__new__(pc.Queue)
    q.box, q.doc, q.jobs, q.runs_dir = "xe7740", {}, jobs, tmp_path
    return q


def test_livebench_job_done_only_with_an_ok_summary(tmp_path):
    q = _queue(tmp_path, [{"id": "LB-qwen36-1gpu-think", "kind": "livebench"}])
    assert pc._job_outcome(q, "LB-qwen36-1gpu-think") == "failed"
    d = tmp_path / "LB-qwen36-1gpu-think"
    d.mkdir()
    (d / "livebench_summary.json").write_text(json.dumps({"status": "failed"}))
    assert pc._job_outcome(q, "LB-qwen36-1gpu-think") == "failed"
    (d / "livebench_summary.json").write_text(json.dumps({"status": "ok"}))
    assert pc._job_outcome(q, "LB-qwen36-1gpu-think") == "done"


def test_capacity_jobs_still_use_the_sweep_check(tmp_path):
    q = _queue(tmp_path, [{"id": "H1-x-1gpu-h_assistant"}])
    (tmp_path / "H1-x-1gpu-h_assistant").mkdir()
    (tmp_path / "H1-x-1gpu-h_assistant" / "livebench_summary.json").write_text(json.dumps({"status": "ok"}))
    assert pc._job_outcome(q, "H1-x-1gpu-h_assistant") == "failed"


def test_model_config_carries_the_mode():
    job = {"id": "LB-qwen36-1gpu-think", "livebench": {
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 32768,
        "api_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}}
    import yaml
    cfg = yaml.safe_load(lbj.model_config_yaml(job, "nvidia/Qwen3.6-35B-A3B-NVFP4"))
    assert cfg["display_name"] == "capsim-lb-qwen36-1gpu-think"
    assert cfg["api_name"] == {"local": "nvidia/Qwen3.6-35B-A3B-NVFP4"}
    kw = cfg["api_kwargs"]["local"]
    assert kw["temperature"] == 0.6 and kw["top_p"] == 0.95
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_livebench_ids_never_look_like_main_capacity_runs():
    import re
    assert not re.match(r"^H(?:\d+|C)-", "LB-qwen36-1gpu-think")


def test_scores_keep_the_newest_judgment_per_question(tmp_path, monkeypatch):
    monkeypatch.setattr(lbj, "LB_REPO", tmp_path)
    d = tmp_path / "livebench" / "data" / "live_bench" / "math" / "olympiad" / "model_judgment"
    d.mkdir(parents=True)
    rows = [{"model": "m", "question_id": "q1", "score": 0, "tstamp": 1},
            {"model": "m", "question_id": "q1", "score": 1, "tstamp": 2},
            {"model": "m", "question_id": "q2", "score": 0, "tstamp": 1},
            {"model": "other", "question_id": "q1", "score": 1, "tstamp": 3}]
    (d / "ground_truth_judgment.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = lbj.scores("m")
    assert s["graded"] == 2
    assert s["categories"] == {"math": 50.0}
