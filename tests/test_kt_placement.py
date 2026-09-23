"""The KTransformers hot/cold placement driver: its accounting, not the
engine (which only the box can run)."""

from __future__ import annotations

import asyncio
import json

import httpx

from simulator import kt_placement as kp


def test_window_counts_tokens_that_arrived_inside_it():
    """A request straddling the window contributes only its in-window
    tokens -- the rate is streamed tokens over the window, with no
    finish-time wave to quantise it."""
    a = kp.Request(domain="code", replica=0, started=0.0, ttft=1.0, ended=12.0,
                   tokens=11, arrivals=[1.0 + i for i in range(11)])
    b = kp.Request(domain="chinese", replica=1, started=5.0, ttft=2.0, ended=None,
                   arrivals=[7.0, 8.0, 9.0, 30.0])
    c = kp.Request(domain="code", replica=0, started=6.0, ended=9.0, error="HTTP 500")
    w = kp.window_stats([a, b, c], 5.0, 10.0)
    assert w["stream_tok_s"] == (5 + 3) / 5.0      # a: 5,6,7,8,9; b: 7,8,9
    assert w["finished"] == 1 and w["succeeded"] == 0 and w["success_rate"] == 0.0
    w2 = kp.window_stats([a, b, c], 0.0, 20.0)
    assert w2["succeeded"] == 1 and w2["answers_by_domain"] == {"code": 1}
    assert w2["ttft_p50_ms"] == 1000.0 and w2["tpot_p50_ms"] == 1100.0


def test_node_busy_is_per_socket():
    a = {0: (100, 50), 1: (100, 50), 2: (100, 50), 3: (100, 50)}
    b = {0: (200, 60), 1: (200, 140), 2: (200, 60), 3: (200, 140)}
    assert kp.node_busy(a, b, {0: [0, 2], 1: [1, 3]}) == {0: 0.9, 1: 0.1}


def test_dump_pairs_are_new_files_in_time_order_per_replica(tmp_path):
    for r in ("r0", "r1"):
        d = tmp_path / r
        d.mkdir()
        for t in ("100.1", "200.2"):
            (d / f"expert_distribution_recorder_{t}.pt").write_bytes(b"x")
            (d / f"gpu_expert_distribution_{t}.pt").write_bytes(b"x")
    old = {str(tmp_path / "r0" / "expert_distribution_recorder_100.1.pt"),
           str(tmp_path / "r0" / "gpu_expert_distribution_100.1.pt")}
    pairs = kp.recorded_pairs(tmp_path, old)
    assert [(p[0].split("/")[-2], p[0].rsplit("_", 1)[-1]) for p in pairs] == [
        ("r0", "200.2.pt"), ("r1", "100.1.pt"), ("r1", "200.2.pt")]
    assert all(p[1].replace("gpu_expert_distribution", "expert_distribution_recorder")
               == p[0] for p in pairs)


def _sse(handler_chunks):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True and body["max_tokens"] == 4
        lines = [f"data: {json.dumps(c)}\n\n" for c in handler_chunks] + ["data: [DONE]\n\n"]
        return httpx.Response(200, content="".join(lines).encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def test_one_request_counts_reasoning_and_content_and_takes_usage():
    chunks = [{"choices": [{"delta": {"reasoning_content": "hm"}}]},
              {"choices": [{"delta": {"reasoning_content": " ok"}}]},
              {"choices": [{"delta": {"content": "4"}}]},
              {"choices": [], "usage": {"completion_tokens": 3}}]

    async def go():
        async with httpx.AsyncClient(transport=_sse(chunks)) as c:
            req = kp.Request(domain="math", replica=0, started=0.0)
            await kp._one(c, "http://x/v1", "m", {"messages": [{"role": "user",
                                                               "content": "2+2"}]},
                          4, req)
            return req
    req = asyncio.run(go())
    assert len(req.arrivals) == 3 and req.tokens == 3 and req.error is None
    assert req.ttft is not None and req.ended is not None


def test_example_plan_shape():
    plan = kp.example_plan()
    names = [c["name"] for c in plan["configs"]]
    assert names == ["dp2_uniform", "dp2_frequency", "dp1_frequency", "sglang_tp8_nvfp4"]
    dp2 = plan["configs"][0]["custom"]
    assert dp2["replicas"] == 2 and dp2["tp"] == 4 and dp2["ktransformers_numa_pin"]
    assert plan["configs"][0]["calibrate"] is True
    assert plan["configs"][1]["custom"]["ktransformers_expert_freq_path"] == "@calibration"
    # The placement comparison holds everything but placement fixed.
    a, b = plan["configs"][0]["custom"], plan["configs"][1]["custom"]
    assert plan["configs"][0]["concurrency"] == plan["configs"][1]["concurrency"]
    assert "mixes" not in plan["configs"][0] and "mixes" not in plan["configs"][1]
    assert plan["measure_s"] > 300 and plan["request_timeout_s"] > 600
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"ktransformers_expert_placement", "ktransformers_expert_freq_path"}
