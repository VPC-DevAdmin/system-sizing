#!/usr/bin/env python3
"""Write the planner campaign queue files (see planner_campaign.py).

    planner_queues.py xe7740 > queue-xe7740.yaml
    planner_queues.py r770   > queue-r770.yaml

Wave ids follow the campaign plan (G0-G3 on the XE7740, C0-C3 on the
R770). Fixed-shape AIPerf ladders (C1.2, C2, G4, G5) are not capsim
jobs and run from their own scripts.
"""

from __future__ import annotations

import sys

import yaml

CORE6 = ["quick_lookup", "conversational", "document_qa", "long_form_generator",
         "knowledge_search", "agent_task_runner"]
EXTRA10 = ["writer", "code_assist", "meeting_summary", "translation",
           "customer_assistant", "voice_assistant", "coding_agent",
           "reasoning_analysis", "document_extraction", "inbox_triage"]
ALL16 = CORE6 + EXTRA10
MIX6 = ["chat_heavy", "general_knowledge", "analyst_team", "contact_centre",
        "engineering_agents", "finance_ops"]
HISTORY = ["conversational", "code_assist", "agent_task_runner", "customer_assistant"]
HISTORY_SETTINGS = [0, 2, 8]   # full history is the default run
# First arrival rate tried, per GPU (sessions/s). Short single-turn
# work arrives fast; long multi-turn sessions slowly. The search
# doubles from here, so low is safe and only costs a window or two.
GPU_START = {"quick_lookup": 1.0, "inbox_triage": 4.0, "document_extraction": 1.0,
             "voice_assistant": 0.5, "customer_assistant": 0.5,
             "knowledge_search": 0.5, "meeting_summary": 0.25, "translation": 0.5,
             "writer": 0.5}
GPU_START_DEFAULT = 0.25
CPU_START = 0.02

HF = {"/data2/capsim/huggingface": "/root/.cache/huggingface",
      "/data3/capsim/hf-overflow": "/data3/capsim/hf-overflow"}

TUNED_GPU_KEYS = {"qwen30b", "gemma26b", "gptoss120b", "nemosuper", "gptoss20b", "qwen36",
                  "nemonano", "kimilinear", "glm47", "gemma31b", "qwen35", "minimax"}

GPU_MODELS = {
    "qwen30b": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "tp": 1},
    "gemma26b": {"model": "nvidia/Gemma-4-26B-A4B-NVFP4", "tp": 1},
    "gptoss120b": {"model": "openai/gpt-oss-120b", "tp": 1, "reasoning_effort": "low"},
    "nemosuper": {"model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "tp": 2},
    "gptoss20b": {"model": "openai/gpt-oss-20b", "tp": 1, "reasoning_effort": "low"},
    "qwen36": {"model": "nvidia/Qwen3.6-35B-A3B-NVFP4", "tp": 1},
    "nemonano": {"model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "tp": 1},
    "kimilinear": {"model": "Firworks/Kimi-Linear-48B-A3B-Instruct-nvfp4", "tp": 1,
                   "trust_remote_code": True},
    "glm47": {"model": "zai-org/GLM-4.7-Flash", "tp": 1},
    "gemma31b": {"model": "nvidia/Gemma-4-31B-IT-NVFP4", "tp": 1},
    "qwen35": {"model": "nvidia/Qwen3.5-397B-A17B-NVFP4", "tp": 4,
               "engine": "sglang_cuda"},
    "minimax": {"model": "MiniMaxAI/MiniMax-M2.7", "tp": 4},
    "beam": {"model": "reflection/Beam-1.0-501B-A23B-NVFP4", "tp": 8,
             "reasoning_effort": "low",
             "engine_extra": {"gpu_image": "vllm/vllm-openai:reflection-beam",
                              "vllm_extra_flags": ["--kv-cache-dtype", "fp8",
                                                   "--reasoning-parser", "beam"],
                              "docker_extra_env": {"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}}},
    "nemoultra": {"model": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "tp": 8,
                  "engine_extra": {"docker_extra_env": {"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}}},
    "kimik2": {"model": "nvidia/Kimi-K2-Thinking-NVFP4", "tp": 8, "reasoning_effort": "low",
               "trust_remote_code": True},
}


TUNE_SIM = {"open_loop_backfill_factor": None, "open_loop_resolution_pct": 20.0}
TUNE_WORKLOADS = ["conversational", "document_qa"]
GPU_TUNE_FULL = {
    "default": {},
    "seqs512-mbt8k": {"max_num_seqs": 512, "max_num_batched_tokens": 8192},
    "seqs512-mbt32k": {"max_num_seqs": 512, "max_num_batched_tokens": 32768},
    "seqs2048-mbt8k": {"max_num_seqs": 2048, "max_num_batched_tokens": 8192},
    "seqs2048-mbt32k": {"max_num_seqs": 2048, "max_num_batched_tokens": 32768},
    "kvfp8": {"kv_cache_dtype": "fp8"},
    "mem95": {"gpu_memory_utilization": 0.95},
    "combined": {"max_num_seqs": 2048, "max_num_batched_tokens": 16384,
                 "kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.95},
}
GPU_TUNE_LIGHT = {k: GPU_TUNE_FULL[k] for k in ("default", "seqs2048-mbt32k", "kvfp8", "combined")}
GPU_TUNE_TP4 = {"default": {}, "seqs2048-kvfp8": {"max_num_seqs": 2048, "kv_cache_dtype": "fp8"}}


def gtune(key, grid, prefix):
    tp = GPU_MODELS[key]["tp"]
    out = []
    for name, ov in grid.items():
        j = gjob(f"{prefix}-{key}-{tp}gpu-{name}", key, list(TUNE_WORKLOADS), tp,
                 simulation=dict(TUNE_SIM), tune_group=f"{key}-gpu", tune_name=name,
                 tune_overrides=ov, **ov)
        out.append(j)
    out[0]["barrier"] = True
    return out


def gjob(jid, key, workloads, gpus=None, **kw):
    m = dict(GPU_MODELS[key])
    extra = m.pop("engine_extra", {})
    tp = m["tp"]
    gpus = gpus or tp
    wl = workloads if isinstance(workloads, list) else [workloads]
    start = GPU_START.get(wl[0], GPU_START_DEFAULT) * (gpus / tp) * (1.0 if tp == 1 else 0.5)
    ee = dict(extra)
    ee.setdefault("docker_volumes", HF)
    ee = {**ee, **kw.pop("engine_extra", {})}
    j = {"id": jid, **m, "devices": gpus, "workloads": wl,
         "initial_rate_per_s": round(start, 4), "engine_extra": ee}
    if key in TUNED_GPU_KEYS:
        j["tune_key"] = f"{key}-gpu"
    j.update(kw)
    if j.get("tune_group"):
        j.pop("tune_key", None)
    return j


def xe7740() -> dict:
    jobs = []
    q = "qwen30b"
    # G0: shakedown (doubles as backlog item 1's validation), then the
    # lane check: one GPU alone, then all eight lanes at once.
    jobs.append(gjob("G0.4-qwen30b-2gpu-conversational", q, "conversational", 2, barrier=True))
    jobs.append(gjob("G0.3a-qwen30b-1gpu-conversational-solo", q, "conversational", 1, barrier=True))
    for i in range(8):
        jobs.append(gjob(f"G0.3b-qwen30b-1gpu-conversational-lane{i}", q, "conversational", 1,
                         barrier=(i == 0)))
    # T: engine tuning for the Qwen3-30B anchor; G1 waits for the pick.
    jobs += gtune("qwen30b", GPU_TUNE_FULL, "T1")
    # G1: Qwen3-30B anchor.
    for w in ALL16:
        if w == "conversational":
            continue  # G0.3a
        jobs.append(gjob(f"G1.1-qwen30b-1gpu-{w}", q, w, 1))
    for w in HISTORY:
        for h in HISTORY_SETTINGS:
            jobs.append(gjob(f"G1.4-qwen30b-1gpu-{w}-hist{h}", q, w, 1,
                             simulation={"history_window_turns": h}))
    for w in ["quick_lookup", "conversational", "document_qa"]:
        jobs.append(gjob(f"G1.5-qwen30b-1gpu-{w}-kvauto", q, w, 1, kv_cache_dtype="auto",
                         pin=["kv_cache_dtype"]))
    for w in ["document_qa", "agent_task_runner", "meeting_summary"]:
        jobs.append(gjob(f"G1.5-qwen30b-1gpu-{w}-ctx128k", q, w, 1, max_model_len=131072))
    for mix in MIX6:
        jobs.append(gjob(f"G1.6-qwen30b-1gpu-{mix}", q, mix, 1))
    for eng in ["sglang_cuda", "trtllm"]:
        for w in CORE6:
            jobs.append(gjob(f"G1.3-qwen30b-1gpu-{w}-{eng}", q, w, 1, engine=eng))
    for n in (2, 4):
        for w in CORE6:
            jobs.append(gjob(f"G1.2-qwen30b-{n}gpu-{w}", q, w, n))
    for w in CORE6:
        jobs.append(gjob(f"G1.2-qwen30b-8gpu-{w}", q, w, 8))
    for mix in ["general_knowledge", "contact_centre"]:
        jobs.append(gjob(f"G1.6-qwen30b-8gpu-{mix}", q, mix, 8))
    # T: tuning for the other anchors and the remaining models.
    jobs += gtune("gemma26b", GPU_TUNE_FULL, "T2")
    jobs += gtune("gptoss120b", GPU_TUNE_FULL, "T2")
    jobs += gtune("nemosuper", GPU_TUNE_FULL, "T2")
    # G2: other class anchors at their smallest copy.
    for key in ["gemma26b", "gptoss120b", "nemosuper"]:
        tp = GPU_MODELS[key]["tp"]
        for w in ALL16:
            jobs.append(gjob(f"G2-{key}-{tp}gpu-{w}", key, w, tp))
        for w in HISTORY:
            for h in HISTORY_SETTINGS:
                jobs.append(gjob(f"G2-{key}-{tp}gpu-{w}-hist{h}", key, w, tp,
                                 simulation={"history_window_turns": h}))
        for mix in MIX6:
            jobs.append(gjob(f"G2-{key}-{tp}gpu-{mix}", key, mix, tp))
        if key == "gptoss120b":
            for eff in ["medium", "high"]:
                for w in ["conversational", "document_qa", "knowledge_search"]:
                    jobs.append(gjob(f"G2-{key}-{tp}gpu-{w}-reason{eff}", key, w, tp,
                                     reasoning_effort=eff))
    for key in ["gptoss20b", "qwen36", "nemonano", "kimilinear", "glm47", "gemma31b"]:
        jobs += gtune(key, GPU_TUNE_LIGHT, "T3")
    for key in ["qwen35", "minimax"]:
        jobs += gtune(key, GPU_TUNE_TP4, "T3")
    # G3: remaining models, core six at the smallest copy.
    for key in ["gptoss20b", "qwen36", "nemonano", "kimilinear", "glm47", "gemma31b",
                "qwen35", "minimax"]:
        tp = GPU_MODELS[key]["tp"]
        for w in CORE6:
            jobs.append(gjob(f"G3-{key}-{tp}gpu-{w}", key, w, tp))
    # Whole-box runs last so lanes stay full as long as possible.
    for key in ["gemma26b", "gptoss120b", "nemosuper"]:
        for w in CORE6:
            jobs.append(gjob(f"G2-{key}-8gpu-{w}", key, w, 8))
    for key in ["gptoss20b", "qwen36", "nemonano", "kimilinear", "glm47", "gemma31b",
                "qwen35", "minimax"]:
        for w in ["quick_lookup", "document_qa"]:
            jobs.append(gjob(f"G3-{key}-8gpu-{w}", key, w, 8))
    for key in ["beam", "nemoultra", "kimik2"]:
        for w in CORE6:
            jobs.append(gjob(f"G3-{key}-8gpu-{w}", key, w, 8))
    return {
        "box": "xe7740",
        "state_dir": "/data/capsim-planner/state",
        "runs_dir": "/data/capsim-planner/runs",
        "wall_power_csv": "/data/capsim-planner/power/xe7740-wall.csv",
        "defaults": {"engine": "vllm_cuda_multi", "max_model_len": 32768,
                     "gpu_memory_utilization": 0.90},
        "jobs": jobs,
    }


# ── R770 ───────────────────────────────────────────────────────────────

R770_MODELS = {
    # SGLang CPU (the planner's R470 reference engine) unless noted.
    "qwen30b": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
}


def sglang_cpu(model: str, local: str, quant: str | None, max_total_tokens: int = 1048576) -> dict:
    return {
        "type": "sglang", "model_id": model, "model_local_path": local,
        "served_model_name": model.split("/")[-1].lower(),
        "quantization_kind": quant, "tensor_parallel_size": 1,
        "max_model_len": 32768, "max_total_tokens": max_total_tokens,
        "context_length": 32768, "chunked_prefill_size": 4096,
        "mem_fraction_static": 0.85, "disable_overlap_schedule": True,
        "enable_metrics": True, "attention_backend": "intel_amx",
        "host": "127.0.0.1", "startup_timeout_s": 1800,
        "docker_image": "sglang-cpu:xeon-fixed", "docker_shm_size": "64g",
        "docker_network": "host",
        "docker_volumes": {"/mnt/xeonlab-data1/capsim/models": "/models",
                           "/mnt/xeonlab-data1/capsim/huggingface": "/root/.cache/huggingface"},
    }


def vllm_cpu(model: str, local: str) -> dict:
    return {
        "type": "vllm_dual_socket", "model_id": model, "model_local_path": local,
        "vllm_image": "vllm/vllm-openai-cpu:latest",
        "served_model_name": model.split("/")[-1].lower(), "max_model_len": 32768,
        "host": "127.0.0.1", "startup_timeout_s": 1800,
        "replicas": [{"name": "s0", "cpuset_cpus": "", "cpuset_mems": "0", "port": 9100}],
        "docker_volumes": {"/mnt/xeonlab-data1/capsim/models": "/models",
                           "/mnt/xeonlab-data1/capsim/huggingface": "/root/.cache/huggingface"},
    }


def cjob(jid, model, local, quant, workloads, sockets=1, engine="sglang", **kw):
    wl = workloads if isinstance(workloads, list) else [workloads]
    ce = sglang_cpu(model, local, quant) if engine == "sglang" else vllm_cpu(model, local)
    ce.update(kw.pop("cpu_engine_extra", {}))
    j = {"id": jid, "model": model, "devices": sockets, "workloads": wl,
         "initial_rate_per_s": CPU_START * sockets, "cpu_engine": ce,
         "workers_per_device": 4}
    if engine == "sglang" and model == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8":
        j["tune_key"] = "qwen30b-cpu"
    j.update(kw)
    if j.get("tune_group") or j["id"].startswith("C0"):
        j.pop("tune_key", None)
    return j


CPU_TUNE = {
    "default": {},
    "chunk2k": {"cpu_engine": {"chunked_prefill_size": 2048}},
    "chunk8k": {"cpu_engine": {"chunked_prefill_size": 8192}},
    "run64": {"cpu_engine": {"sglang_extra_flags": ["--max-running-requests", "64"]}},
    "run128": {"cpu_engine": {"sglang_extra_flags": ["--max-running-requests", "128"]}},
    "overlap": {"cpu_engine": {"disable_overlap_schedule": False}},
    "cores84": {"socket_cores": 84},
}


def r770() -> dict:
    m, loc, qn = ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                  "/models/Qwen3-30B-A3B-Instruct-2507-FP8", "fp8")
    jobs = []
    jobs.append(cjob("C0.4-qwen30b-1s-quick_lookup", m, loc, qn, "quick_lookup", 1, barrier=True))
    jobs.append(cjob("C0.3a-qwen30b-1s-conversational-solo", m, loc, qn, "conversational", 1,
                     barrier=True))
    for i in range(2):
        jobs.append(cjob(f"C0.3b-qwen30b-1s-conversational-lane{i}", m, loc, qn,
                         "conversational", 1, barrier=(i == 0)))
    # T: SGLang CPU tuning; the SGLang persona jobs wait for the pick.
    for i, (name, ov) in enumerate(CPU_TUNE.items()):
        j = cjob(f"TC1-qwen30b-1s-{name}", m, loc, qn, list(TUNE_WORKLOADS), 1,
                 simulation=dict(TUNE_SIM), tune_group="qwen30b-cpu", tune_name=name,
                 tune_overrides=ov, barrier=(i == 0))
        if "cpu_engine" in ov:
            j["cpu_engine"].update(ov["cpu_engine"])
        if "socket_cores" in ov:
            j["socket_cores"] = ov["socket_cores"]
        jobs.append(j)
    for w in ALL16:
        if w in ("quick_lookup", "conversational"):
            continue
        jobs.append(cjob(f"C1.4-qwen30b-1s-{w}", m, loc, qn, w, 1))
    for w in CORE6:
        jobs.append(cjob(f"C1.1-qwen30b-1s-{w}-vllm", m, "/models/Qwen3-30B-A3B-Instruct-2507",
                         None, w, 1, engine="vllm"))
    for w in HISTORY:
        for h in HISTORY_SETTINGS:
            jobs.append(cjob(f"C1.5-qwen30b-1s-{w}-hist{h}", m, loc, qn, w, 1,
                             simulation={"history_window_turns": h}))
    for w in ["document_qa", "agent_task_runner", "meeting_summary"]:
        jobs.append(cjob(f"C1.6-qwen30b-1s-{w}-ctx128k", m, loc, qn, w, 1,
                         max_model_len=131072,
                         cpu_engine_extra={"context_length": 131072}))
    # Two sockets: one instance per socket (TP2 and SNC layouts run as
    # separate jobs once the per-socket baseline is in).
    for w in ["quick_lookup", "document_qa", "long_form_generator"]:
        jobs.append(cjob(f"C1.3-qwen30b-2s-{w}-tp2", m, loc, qn, w, 2,
                         cpu_engine_extra={"tensor_parallel_size": 2}, barrier=(w == "quick_lookup")))
    for mix in MIX6:
        jobs.append(cjob(f"C1.7-qwen30b-2s-{mix}-tp2", m, loc, qn, mix, 2,
                         cpu_engine_extra={"tensor_parallel_size": 2}))
    return {
        "box": "r770",
        "state_dir": "/mnt/xeonlab-data1/capsim/planner/state",
        "runs_dir": "/mnt/xeonlab-data1/capsim/planner/runs",
        "wall_power_csv": "/mnt/xeonlab-data1/capsim/planner/power/r770-wall.csv",
        "defaults": {},
        "jobs": jobs,
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "xe7740"
    doc = xe7740() if which == "xe7740" else r770()
    yaml.safe_dump(doc, sys.stdout, sort_keys=False, width=120)


if __name__ == "__main__":
    main()
