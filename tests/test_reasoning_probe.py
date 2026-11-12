import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import planner_campaign as pc  # noqa: E402
import reasoning_probe_job as rpj  # noqa: E402
from simulator.rescore import Window, _margin  # noqa: E402


def _row(wl, reasoning, answer=100, cap=False, err=None):
    r = {"workload": wl, "answer_target": 120}
    if err:
        r["error"] = err
        return r
    r.update({"reasoning_tokens": reasoning, "answer_tokens": answer, "hit_cap": cap,
              "first_answer_s": 1.0})
    return r


def test_summary_uses_hwave_ids_and_keeps_mean_median_p90():
    rows = [_row("document-qa", x) for x in (100, 200, 300, 400, 1000)]
    s = rpj.summarize(rows, expect_reasoning=True)
    w = s["workloads"]["h_document_qa"]
    assert s["status"] == "ok"
    assert w["reasoning_mean"] == 400.0
    assert w["reasoning_median"] == 300.0
    assert w["reasoning_p90"] == pytest.approx(760.0)


def test_summary_fails_when_reasoning_was_not_separated():
    rows = [_row("assistant", 0) for _ in range(10)]
    assert rpj.summarize(rows, expect_reasoning=True)["status"] == "failed"
    assert rpj.summarize(rows, expect_reasoning=False)["status"] == "ok"
    thinking = [_row("assistant", 300) for _ in range(10)]
    assert rpj.summarize(thinking, expect_reasoning=False)["status"] == "failed"
    # Control tokens beyond the answer, but no reasoning text: still off.
    off = [{**_row("assistant", 3), "reasoning_text_tokens": 0} for _ in range(10)]
    assert rpj.summarize(off, expect_reasoning=False)["status"] == "ok"


def test_summary_fails_on_too_many_errors():
    rows = [_row("assistant", 50) for _ in range(9)] + [_row("assistant", 0, err="boom")]
    assert rpj.summarize(rows, expect_reasoning=True)["status"] == "failed"


def test_turn_cap_adds_room_inside_the_context():
    class T:
        max_output = 2000 + 8000
    assert rpj.turn_cap(T, 32768, 1000, None) == 2000 + 32768
    assert rpj.turn_cap(T, 32768, 120_000, 131_072) == 131_072 - 120_000 - 256


def _queue(tmp_path):
    q = pc.Queue.__new__(pc.Queue)
    q.box, q.doc, q.jobs, q.runs_dir = "xe7740", {}, [], tmp_path
    return q


def test_measured_allowance_reads_the_probe_mean(tmp_path):
    (tmp_path / "R-x").mkdir()
    (tmp_path / "R-x" / "reasoning_probe_summary.json").write_text(json.dumps(
        {"status": "ok", "workloads": {"h_service_agent": {"reasoning_mean": 812.6}}}))
    job = {"id": "H4-x-h_service_agent-think", "reasoning_from": "R-x", "workloads": ["h_service_agent"]}
    assert pc.measured_allowance(_queue(tmp_path), job) == 813


def test_measured_allowance_refuses_a_failed_probe(tmp_path):
    (tmp_path / "R-x").mkdir()
    (tmp_path / "R-x" / "reasoning_probe_summary.json").write_text(json.dumps({"status": "failed"}))
    job = {"id": "H4-x", "reasoning_from": "R-x", "workloads": ["h_assistant"]}
    with pytest.raises(ValueError):
        pc.measured_allowance(_queue(tmp_path), job)


def test_probe_job_counts_as_a_score_job(tmp_path):
    q = _queue(tmp_path)
    q.jobs = [{"id": "R-x", "kind": "reasoning_probe"}]
    assert pc._job_outcome(q, "R-x") == "failed"
    (tmp_path / "R-x").mkdir()
    (tmp_path / "R-x" / "reasoning_probe_summary.json").write_text(json.dumps({"status": "ok"}))
    assert pc._job_outcome(q, "R-x") == "done"


def test_answer_start_after_mean_reasoning():
    # First token at 0.5 s, 20 ms per token. With 1,000 reasoning tokens
    # the answer starts at 20.5 s, so it misses a 2 s limit, though the
    # model happened to start content early on the filler text.
    turns = [(500, 600, 20, 500, 0, 1200, None)]
    limits = {"ttft_s": 2.0, "ttft_per_4k_s": 1.0, "tpot_ms": 30}
    win = Window(rate_per_min=1.0, stable=True, values={}, turns=turns)
    assert _margin(win, limits) > 0
    assert _margin(win, {**limits, "answer_after_tokens": 1000}) < 0
    assert _margin(win, {**limits, "answer_after_tokens": 50}) > 0
