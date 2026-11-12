import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cloud_token_probe as ctp  # noqa: E402


def test_sessions_match_the_planner_shapes():
    sessions = ctp.build_sessions()
    by = {w.id: [s for s in sessions if s.workload == w.id] for w in ctp.WORKLOADS}
    for w in ctp.WORKLOADS:
        assert len(by[w.id]) == w.sessions
        assert all(len(s.turns) == w.turns for s in by[w.id])
        # System prompts reach the workload's size (estimate within 5%).
        assert abs(ctp.est_tokens(by[w.id][0].system) - w.system_tokens) <= 0.05 * w.system_tokens + 50
    assert ctp.build_sessions()[5].turns[0].text == sessions[5].turns[0].text  # seeded
    # Every session of a workload sends the same system prompt, as the lab does.
    for w in ctp.WORKLOADS:
        assert len({s.system for s in by[w.id]}) == 1, w.id


def test_long_prompt_rate_card():
    haiku = ctp.MODELS["claude-haiku"]
    short = ctp.call_cost(haiku, 100_000, 0, 0, 1_000)
    long = ctp.call_cost(haiku, 100_001, 0, 0, 1_000)
    assert abs(short - (100_000 * 0.10 + 1_000 * 0.50) / 1e6) < 1e-12
    assert abs(long - 5 * (100_001 * 0.10 + 1_000 * 0.50) / 1e6) < 1e-12
    # Cache reads and writes bill at their own prices.
    assert abs(ctp.call_cost(haiku, 10_000, 6_000, 4_000, 0) - (6_000 * 0.01 + 4_000 * 0.125) / 1e6) < 1e-12


def test_plan_orders_low_expected_ceiling():
    p = ctp.plan(ctp.build_sessions())
    for rows in p.values():
        for r in rows.values():
            assert 0 < r["low"] <= r["expected"] <= r["ceiling"]


def test_adapters_read_each_providers_usage():
    m = ctp.MODELS
    oa = NS(responses=NS(create=lambda **k: NS(
        usage=NS(input_tokens=5000, output_tokens=400,
                 input_tokens_details=NS(cached_tokens=3000, cache_write_tokens=0),
                 output_tokens_details=NS(reasoning_tokens=0)),
        output_text="hi", status="completed")))
    u = ctp.call_openai(oa, m["gpt6-luna"], "sys", [{"role": "user", "content": "q"}], 900)
    assert (u.prompt, u.cache_read, u.output, u.reasoning) == (5000, 3000, 400, 0)

    msg = NS(usage=NS(input_tokens=200, cache_read_input_tokens=4000, cache_creation_input_tokens=800,
                      output_tokens=700, output_tokens_details=NS(thinking_tokens=250)),
             content=[NS(type="thinking"), NS(type="text", text="answer")], stop_reason="end_turn")

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return msg
    an = NS(messages=NS(stream=lambda **k: Stream()))
    u = ctp.call_anthropic(an, m["claude-haiku"], "sys", [{"role": "user", "content": "q"}], 900)
    assert (u.prompt, u.cache_read, u.cache_write, u.output, u.reasoning, u.text) == (5000, 4000, 800, 700, 250, "answer")



def test_google_adapter_reads_usage():
    import pytest
    pytest.importorskip("google.genai")
    m = ctp.MODELS
    resp = NS(usage_metadata=NS(prompt_token_count=6000, cached_content_token_count=2000,
                                candidates_token_count=300, thoughts_token_count=50),
              candidates=[NS(finish_reason=NS(name="STOP"))], text="ok")
    go = NS(models=NS(generate_content=lambda **k: resp))
    u = ctp.call_google(go, m["gemini-flash-lite"], "sys", [{"role": "user", "content": "q"}], 900)
    assert (u.prompt, u.cache_read, u.output, u.reasoning) == (6000, 2000, 350, 50)


def test_budget_guard_stops_before_overspending(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    calls = []

    def fake(client, m, system, msgs, max_out):
        calls.append(1)
        return ctp.Usage(ctp.est_tokens(system) + sum(ctp.est_tokens(x["content"]) for x in msgs),
                         0, 0, max_out, 0, "x", "max_tokens")
    monkeypatch.setitem(ctp.CALLERS, "openai", fake)
    monkeypatch.setattr(ctp, "make_client", lambda p: None)
    ctp.run(ctp.build_sessions(), ["gpt6-luna"], 0.05, tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows and len(rows) == len(calls)
    assert sum(r["cost"] for r in rows) <= 0.05
