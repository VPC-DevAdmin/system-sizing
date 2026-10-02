import asyncio
import json

import httpx

from simulator.infrastructure.driver import request_one
from simulator.infrastructure.dynamo_runtime import commands
from simulator.infrastructure.spec import ExperimentSpec


def test_reasoning_is_retained_but_not_graded_as_answer():
    expected = [{"record_id": "R1", "quantity": 2}]
    chunks = [
        {
            "choices": [
                {"delta": {"reasoning_content": "Reasoning without JSON"}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {"content": json.dumps(expected)}, "finish_reason": "stop"}]},
    ]
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request):
        assert json.loads(request.content)["reasoning_effort"] == "medium"
        assert "chat_template_kwargs" not in json.loads(request.content)
        return httpx.Response(200, text=body)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await request_one(
                client,
                "http://test/v1",
                "openai/gpt-oss-120b",
                dict(id="test", prompt="test", expected=expected),
                ExperimentSpec(project_id="P01", gpu_ids=[0, 1]),
                "x",
            )

    row = asyncio.run(run())
    assert row["correct"]
    assert row["reasoning_output"] == "Reasoning without JSON"
    assert row["ttft_s"] <= row["final_answer_ttft_s"]
    assert json.loads(row["output"]) == expected


def test_gptoss_sessions_use_same_native_parser_in_both_arms():
    plan = dict(
        port=19200,
        model="/models/gpt-oss-120b",
        served_model_name="m",
        context_limit=131072,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="x",
        gpu_ids=[0],
        sessions=dict(session_cpu_backend="simple_cpu", session_tier="cpu", session_cpu_gb=128),
    )
    for arm in ["baseline", "treatment"]:
        front = commands(dict(plan, arm=arm))[0]
        assert front[2]["DYN_CHAT_PROCESSOR"] == "vllm"
        assert "--harmony-frontend" in front[1]
        assert front[1][front[1].index("--reasoning-parser") + 1] == "openai_gptoss"


def test_p02_kv_precision_override_is_matched_and_explicit():
    from simulator.infrastructure.dynamo import arm_config

    spec = ExperimentSpec(project_id="P02", min_requests=1, session_kv_cache_dtype="bfloat16")
    for arm in ["baseline", "treatment"]:
        assert (
            arm_config({"engine": {"kv_cache_dtype": "fp8"}}, spec, arm)["engine"]["kv_cache_dtype"]
            == "bfloat16"
        )
        assert (
            arm_config(
                {"engine": {"kv_cache_dtype": "fp8"}},
                spec.model_copy(update={"session_kv_cache_dtype": "source"}),
                arm,
            )["engine"]["kv_cache_dtype"]
            == "fp8"
        )
