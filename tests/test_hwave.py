"""H wave: heavy workloads at 128K context (config/personas/hwave.yaml)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

from simulator.persona_loader import parse_persona, serialize_persona
from simulator.personas import PERSONAS
from simulator.rescore import Window, _margin, binding_limit
from simulator.tokenizer_corpus import TokenCorpus

ROOT = Path(__file__).resolve().parents[1]


def _spec(**kw):
    spec = {
        "input_tokens": {"constant": 100}, "output_tokens": {"constant": 10},
        "turns_per_session": {"constant": 2}, "sessions_before_leaving": {"constant": 1},
        "inter_session_gap_seconds": {"constant": 1}, "read_time_seconds": {"constant": 0},
        "active_think_seconds": {"constant": 0},
        "sla": {"ttft_target_seconds": 1, "ttft_failure_seconds": 2,
                "tpot_target_ms": 10, "tpot_failure_ms": 20},
    }
    spec.update(kw)
    return spec


def test_hwave_personas_load():
    for pid in ("h_assistant", "h_document_qa", "h_coding_agent", "h_research_agent",
                "h_service_agent", "h_document_pipeline", "h_deep_analysis",
                "h_ctx_128k", "h_out_16k"):
        assert pid in PERSONAS, pid
    assert PERSONAS["h_coding_agent"].initial_context_tokens is not None
    assert PERSONAS["h_assistant"].initial_context_tokens is None


def test_initial_context_round_trips():
    p = parse_persona("x", _spec(initial_context_tokens={"constant": 5000}))
    assert p.initial_context_tokens.sample_int(random.Random(0)) == 5000
    assert serialize_persona(p)["initial_context_tokens"] == {"constant": 5000}
    assert "initial_context_tokens" not in serialize_persona(parse_persona("y", _spec()))


def test_long_prompts_reach_their_target():
    # The paragraph loop used to stop near 24K tokens.
    corpus = TokenCorpus("no-such-model/for-heuristic-counting")
    text = corpus.make_text(100_000, random.Random(1))
    assert corpus.count(text) >= 95_000


def _window(turns, **values):
    return Window(rate_per_min=1.0, stable=True, values=values, turns=turns)


def test_uncached_allowance_ignores_history():
    # 4K new input on top of 120K history, first token at 4 s. With the
    # whole prompt counted the allowance is ~32 s; with new input only
    # it is 2.5 s, so the turn fails.
    turns = [(4000, None, 20, 4000, 120_000, 100, None)]
    limits = {"ttft_s": 2.0, "ttft_per_4k_s": 1.0, "tpot_ms": 30}
    assert _margin(_window(turns), limits) > 0
    assert _margin(_window(turns), {**limits, "uncached_only": True}) < 0


def test_binding_limit():
    assert binding_limit(None, None) is None
    assert binding_limit(_window([], kv_cache_pct=60.0, preemptions=0), None) == "compute"
    assert binding_limit(_window([], kv_cache_pct=97.0), None) == "memory"
    assert binding_limit(_window([], kv_cache_pct=0.97), None) == "memory"
    assert binding_limit(_window([], kv_cache_pct=50.0),
                         _window([], kv_cache_pct=50.0, preemptions=3)) == "memory"


def test_hwave_queue_jobs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import planner_queues as pq
    jobs = pq.hwave_xe7740()
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids))
    assert all(j["max_model_len"] == 131072 for j in jobs)
    by = {j["id"]: j for j in jobs}
    # Flag lists add to the model's own: Beam keeps its reasoning parser.
    flags = by["H3.1-beam-8gpu-h_coding_agent-kvoff"]["engine_extra"]["vllm_extra_flags"]
    assert "--reasoning-parser" in flags and "--kv-offloading-size" in flags
    # Models whose tuning never finished must not wait on it.
    assert "tune_key" not in by["H2-qwen35-4gpu-h_assistant"]
    for j in jobs:
        if j["id"].startswith("H2-"):
            assert j["after"].startswith("HP-")
    r770 = pq.hwave_r770()
    assert {j["cpu_engine"]["max_model_len"] for j in r770} == {131072}


def test_twave_personas_and_jobs():
    for pid in ("t_ticket_triage", "t_document_intake", "t_privacy_screening", "t_agent_support"):
        assert pid in PERSONAS, pid
        assert PERSONAS[pid].initial_context_tokens is None
    sys.path.insert(0, str(ROOT / "scripts"))
    import planner_campaign as pc
    import planner_queues as pq
    jobs = pq.twave_r770()
    ids = [j["id"] for j in jobs]
    assert len(ids) == 12 and len(set(ids)) == 12
    assert all(j["id"].startswith("HC-") and j["devices"] == 1 for j in jobs)
    assert {j["cpu_engine"]["max_model_len"] for j in jobs} == {32768}
    # gpt-oss keeps its reasoning parser and gets room to reason.
    oss = [j for j in jobs if "gptoss20b" in j["id"]]
    assert all(j["reasoning_overhead_tokens"] == 256 for j in oss)
    assert all("--reasoning-parser" in j["cpu_engine"]["sglang_extra_flags"] for j in oss)
    sp = pc.PLANNER_SIMULATION["system_prompt_tokens"]
    assert all(sp[w] > 0 for w in pq.TWAVE)


def test_campaign_system_prompts_cover_hwave():
    sys.path.insert(0, str(ROOT / "scripts"))
    import planner_campaign as pc
    sp = pc.PLANNER_SIMULATION["system_prompt_tokens"]
    for pid in ("h_assistant", "h_coding_agent", "h_research_agent", "h_service_agent"):
        assert sp[pid] > 0
