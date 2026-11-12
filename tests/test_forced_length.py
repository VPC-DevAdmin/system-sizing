from simulator.virtual_user import forced_length_body


def test_most_models_pin_length_with_ignore_eos():
    for model in ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "nvidia/Gemma-4-26B-A4B-NVFP4",
                  "qwen3-30b-a3b-instruct-2507-fp8"):
        assert forced_length_body(model, 300) == {"ignore_eos": True}


def test_gpt_oss_pins_length_with_min_tokens():
    # ignore_eos breaks vLLM's streaming Harmony parser on gpt-oss.
    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "GPT-OSS-20B"):
        body = forced_length_body(model, 300)
        assert body == {"min_tokens": 300}
        assert "ignore_eos" not in body
