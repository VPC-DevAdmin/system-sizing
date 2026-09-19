"""vllm_cuda_multi: whole-box N-replica CUDA engine — command
construction, routing surface, metric aggregation, promote output."""

from __future__ import annotations

import pytest

from simulator.config import EngineConfig
from simulator.engines.vllm_cuda_multi import (
    VllmCudaMultiEngine,
    aggregate_replica_metrics,
    gpus_arg_for,
)


def _cfg(**kw) -> EngineConfig:
    base = dict(type="vllm_cuda_multi", model_id="org/M", port=9100,
                replica_devices=[[0], [4], [1], [5]])
    base.update(kw)
    return EngineConfig(**base)


def test_gpus_arg_quoting() -> None:
    assert gpus_arg_for([3]) == "device=3"
    # Docker parses the value as CSV — multi-device needs embedded quotes.
    assert gpus_arg_for([0, 1]) == '"device=0,1"'


def test_replica_commands_and_urls() -> None:
    eng = VllmCudaMultiEngine(_cfg())
    assert eng.replica_urls == [
        "http://127.0.0.1:9100/v1", "http://127.0.0.1:9101/v1",
        "http://127.0.0.1:9102/v1", "http://127.0.0.1:9103/v1",
    ]
    cmd = eng.build_replica_command(2, [1], "vllm-r2-x")
    joined = " ".join(cmd)
    assert "--gpus device=1" in joined
    assert "--ipc=host" in joined
    assert "--port 9102" in joined                     # port + index
    assert "--tensor-parallel-size 1" in joined
    assert "/root/.cache/huggingface" in joined        # cache mounted

    # A TP2 replica: quoted device pair, tp from the group width.
    eng = VllmCudaMultiEngine(_cfg(replica_devices=[[0, 1], [2, 3]]))
    cmd = eng.build_replica_command(1, [2, 3], "vllm-r1-x")
    assert '"device=2,3"' in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"


def test_launch_refuses_bad_shapes(monkeypatch) -> None:
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/docker")
    # A GPU assigned to two replicas is a silent perf lie — refuse.
    eng = VllmCudaMultiEngine(_cfg(replica_devices=[[0], [0]]))
    with pytest.raises(RuntimeError, match="twice"):
        eng.launch()
    eng = VllmCudaMultiEngine(_cfg(replica_devices=None))
    with pytest.raises(RuntimeError, match="replica_devices"):
        eng.launch()


def test_aggregate_replica_metrics() -> None:
    a = {"num_running": 10, "queue_depth": 2, "kv_cache_used_pct": 40.0,
         "prompt_tokens_total": 1000, "generation_tokens_total": 500,
         "prefix_cache_hits": 80, "prefix_cache_queries": 100,
         "preemptions_total": 1}
    b = {"num_running": 12, "queue_depth": 0, "kv_cache_used_pct": 60.0,
         "prompt_tokens_total": 3000, "generation_tokens_total": 1500,
         "prefix_cache_hits": 40, "prefix_cache_queries": 100}
    agg = aggregate_replica_metrics([a, b])
    assert agg["num_running"] == 22                  # counters summed
    assert agg["prompt_tokens_total"] == 4000        # token rates feed
    assert agg["generation_tokens_total"] == 2000
    assert agg["preemptions_total"] == 1
    assert agg["kv_cache_used_pct"] == 50.0          # gauges averaged
    assert agg["prefix_cache_hit_rate"] == pytest.approx(0.6)
    assert aggregate_replica_metrics([]) == {}


def test_promoted_multi_profile_round_trips(tmp_path) -> None:
    """A dp>1 search winner promotes to a vllm_cuda_multi profile that
    the real config loader and engine registry accept."""
    import textwrap

    from simulator.config import load_config
    from simulator.engines import make_engine
    from simulator.promote import promote_search_winner
    from simulator.search import load_space

    space_path = tmp_path / "space.yaml"
    space_path.write_text(textwrap.dedent("""\
        name: whole-box
        engine: vllm_cuda
        device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]
        model_variants:
          bf16: {model: org/M-30B, served_name: m}
        dimensions:
          tp: [1]
          dp: [8]
          placement: [spread]
    """))
    space = load_space(space_path)
    doc = {
        "space": "whole-box", "space_file": str(space_path),
        "space_hash": space.space_hash(),
        "generated_at": "2026-09-15T21:00:00+00:00",
        "summary": {"best": {"key": "k", "score": 9250.0, "params": {
            "model_variant": "bf16", "tp": 1, "dp": 8,
            "placement": "spread"}}},
    }
    out = promote_search_winner(doc, out_dir=tmp_path / "profiles")
    assert out["warnings"] and "whole-box" in out["warnings"][0]

    cfg = load_config(out["path"])
    assert cfg.engine.type == "vllm_cuda_multi"
    assert len(cfg.engine.replica_devices) == 8
    assert sorted(d for g in cfg.engine.replica_devices for d in g) \
        == list(range(8))
    eng = make_engine(cfg.engine.type, cfg.engine)
    assert isinstance(eng, VllmCudaMultiEngine)
    assert len(eng.replica_urls) == 8


def test_startup_failure_reports_the_engines_own_error(tmp_path) -> None:
    """"see the log" makes every launch failure look alike. The engine's
    last exception line distinguishes an unsupported flag from a
    missing import from an OOM without a round trip."""
    from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine

    e = VllmCudaMultiEngine.__new__(VllmCudaMultiEngine)
    log = tmp_path / "engine.log"
    log.write_text(
        "[r0] INFO 09-17 launching\n"
        '[r0]   File "tokenization_kimi.py", line 19, in <module>\n'
        "[r0] ImportError: cannot import name 'bytes_to_unicode' from "
        "'transformers.models.gpt2.tokenization_gpt2'\n"
    )
    e._log_path = log
    cause = e._startup_cause()
    assert cause.startswith("ImportError:")
    assert "bytes_to_unicode" in cause

    # Later is NOT better when the later line is vLLM's generic
    # wrapper: the specific cause raised earlier is what we want.
    log.write_text(
        "[r0] ValueError: the actual specific cause\n"
        "[r0] RuntimeError: Engine core initialization failed\n"
    )
    assert e._startup_cause().startswith("ValueError:")

    # Degrades without throwing.
    log.write_text("[r0] INFO nothing resembling an exception\n")
    assert "nothing resembling" in e._startup_cause()
    e._log_path = None
    assert "no engine log" in e._startup_cause()
    e._log_path = tmp_path / "missing.log"
    assert "unreadable" in e._startup_cause()


def test_startup_cause_skips_vllms_generic_wrapper(tmp_path) -> None:
    """vLLM's outermost error says only "see root cause above" — the
    real cause is raised earlier, in the engine-core worker."""
    from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine

    e = VllmCudaMultiEngine.__new__(VllmCudaMultiEngine)
    log = tmp_path / "engine.log"
    e._log_path = log

    # The real Kimi-Linear failure: a Triton kernel that does not fit
    # SM120 shared memory, wrapped in a useless RuntimeError.
    log.write_text(
        "[r1] ERROR triton.runtime.errors.OutOfResources: out of resource: "
        "shared memory, Required: 102400, Hardware limit: 101376.\n"
        "[r7] (APIServer pid=1) RuntimeError: Engine core initialization "
        "failed. See root cause above. Failed core proc(s): {}\n"
    )
    cause = e._startup_cause()
    assert "OutOfResources" in cause and "101376" in cause
    assert "root cause above" not in cause

    # An exception whose name is not *Error/*Exception still counts
    # when it lives in an errors module.
    log.write_text("[r0] mylib.errors.Boom: it broke\n")
    assert e._startup_cause().startswith("mylib.errors.Boom:")

    # When the wrapper is genuinely all there is, report it rather
    # than nothing.
    log.write_text(
        "[r0] RuntimeError: Engine core initialization failed. "
        "See root cause above.\n")
    assert "Engine core initialization failed" in e._startup_cause()

    # A log line that merely contains a colon is not an exception.
    log.write_text("[r0] INFO 09-17 20:31:16 model: loaded fine\n")
    assert "INFO" in e._startup_cause()


def test_teardown_noise_does_not_bury_the_real_cause(tmp_path):
    """When init fails, cleanup touches attributes that were never
    created and raises LAST -- winning the 'most recent exception'
    contest and hiding the reason. This cost a wrong diagnosis:
    TensorRT-LLM reported a missing cuda_graph_runner attribute while
    the actual failure, thirty lines earlier, was that its Transformers
    did not recognise the model architecture."""
    from simulator.engines.trtllm import TrtLlmEngine

    log = tmp_path / "engine.log"
    log.write_text(
        "[r4] loading weights\n"
        "[r4] ValueError: The checkpoint you are trying to load has model "
        "type `qwen3_5_moe` but Transformers does not recognize this "
        "architecture.\n"
        "[r4] during cleanup\n"
        "[r4] AttributeError: 'PyTorchModelEngine' object has no attribute "
        "'cuda_graph_runner'\n"
        "[r4] RuntimeError: Executor worker returned error\n")
    eng = TrtLlmEngine(EngineConfig(type="trtllm", model_id="org/M",
                                    replica_devices=[[0]]))
    eng._log_path = log
    cause = eng._startup_cause()
    assert "qwen3_5_moe" in cause
    assert "cuda_graph_runner" not in cause
