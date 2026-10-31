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
             "writer": 0.5,
             # Long agent sessions hold huge contexts; one GPU saturates
             # below 2 sessions/min, and a higher start gives a warm-up
             # window that looks stable before the queue builds.
             "coding_agent": 0.02, "agent_task_runner": 0.05}
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
    # Without mem95: with 2,048 sequences at 0.95 the Qwen, Gemma and
    # gpt-oss combined runs all ran out of memory at startup.
    "combined": {"max_num_seqs": 2048, "max_num_batched_tokens": 16384,
                 "kv_cache_dtype": "fp8"},
}
GPU_TUNE_LIGHT = {k: GPU_TUNE_FULL[k] for k in ("default", "seqs2048-mbt32k", "kvfp8", "combined")}
GPU_TUNE_TP4 = {"default": {}, "seqs2048-kvfp8": {"max_num_seqs": 2048, "kv_cache_dtype": "fp8"}}


# Personas whose no-system-prompt controls are run (G1.8, C1.9).
SYS_CONTROL = ["conversational", "knowledge_search", "customer_assistant", "agent_task_runner"]
# Multi-turn work, where a session's own history is worth keeping.
KV_OFFLOAD_WORKLOADS = HISTORY + ["coding_agent", "contact_centre", "engineering_agents"]
# Host memory per GPU for offloaded KV. 8 lanes x 128 GiB fits the
# XE7740's 2 TB beside the page cache.
KV_OFFLOAD_GIB_PER_GPU = 128


def kv_offload(tp: int) -> dict:
    """vLLM flags for native CPU KV offload; the size is the total
    across a replica's TP ranks."""
    return {"vllm_extra_flags": ["--kv-offloading-size", str(KV_OFFLOAD_GIB_PER_GPU * tp),
                                 "--kv-offloading-backend", "native"]}


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
    if j.get("tune_group") or jid.startswith("G0"):
        j.pop("tune_key", None)  # setup runs use engine defaults
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
    # G1.7: prefix cache off on the multi-turn personas -- what reuse
    # of a session's own history is worth (cross-user reuse is
    # prevented by unique prompt text).
    for w in HISTORY:
        jobs.append(gjob(f"G1.7-qwen30b-1gpu-{w}-nocache", q, w, 1,
                         engine_extra={"vllm_extra_flags": ["--no-enable-prefix-caching"]}))
    # G1.8: no shared system prompt -- what the per-persona application
    # prompt (served from cache) is worth.
    for w in SYS_CONTROL:
        jobs.append(gjob(f"G1.8-qwen30b-1gpu-{w}-sys0", q, w, 1,
                         simulation={"system_prompt_tokens": None}))
    for mix in MIX6:
        jobs.append(gjob(f"G1.6-qwen30b-1gpu-{mix}", q, mix, 1))
    # TensorRT-LLM is left out: its MoE GEMMs go through DeepGEMM,
    # which is Hopper-only, so Qwen3-30B-A3B never starts on SM120
    # (b2aed30). SGLang is the second engine.
    for eng in ["sglang_cuda"]:
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
    # K: KV cache offload to host memory (vLLM native connector). Idle
    # sessions' history moves to CPU RAM instead of being evicted, so
    # multi-turn reuse survives near the knee. Same workloads with and
    # without (the G1/G2 runs above) give the uplift.
    for key in ["qwen30b", "gemma26b", "gptoss120b", "nemosuper"]:
        tp = GPU_MODELS[key]["tp"]
        for w in KV_OFFLOAD_WORKLOADS:
            jobs.append(gjob(f"K-{key}-{tp}gpu-{w}-kvoff", key, w, tp,
                             engine_extra=kv_offload(tp)))
    for w in ["conversational", "contact_centre"]:
        jobs.append(gjob(f"K-qwen30b-8gpu-{w}-kvoff", q, w, 8, engine_extra=kv_offload(1)))
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
                           "/mnt/xeonlab-data2/capsim/models": "/models2",
                           "/mnt/xeonlab-data3/capsim/models": "/models3",
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
    start = CPU_SHAPE_START if wl[0].startswith("shape_") else CPU_START
    j = {"id": jid, "model": model, "devices": sockets, "workloads": wl,
         "initial_rate_per_s": start * sockets, "cpu_engine": ce,
         "workers_per_device": 4}
    if engine == "sglang" and model in QWEN30B_CPU:
        j["tune_key"] = "qwen30b-cpu"
    j["simulation"] = {**CPU_SIM, **(kw.pop("simulation", None) or {})}
    j.update(kw)
    if j.get("tune_group") or j["id"].startswith("C0"):
        j.pop("tune_key", None)
    return j


# CPU runs: start low (one socket misses the Standard tier near
# 1 session/min), reach further below the knee, and hold each window
# open until it should contain 30 turns (up to 10 minutes).
CPU_START = 0.01
CPU_SHAPE_START = 0.05
CPU_SIM = {"open_loop_min_window_turns": 30, "open_loop_max_window_s": 600,
           "open_loop_stability_basis": "client_in_flight",
           "open_loop_backfill_floor_ratio": 16.0}
QWEN30B_CPU = ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "Qwen/Qwen3-30B-A3B-Instruct-2507")
SHAPES = ["shape_chat", "shape_rag", "shape_gen"]
CPU_CLOCKS_GHZ = [2.0, 2.4, 2.8, 3.2]
CPU_CORES = [16, 24, 32, 48, 64]

# C3 ladder: (key, hf id, path in the engine container, quantization,
# extra SGLang flags). BF16 unless the model ships a block-FP8 or MXFP4
# checkpoint SGLang's CPU path reads (Nemotron's FP8 is ModelOpt).
GPTOSS_FLAGS = ["--reasoning-parser", "gpt-oss"]
CPU_LADDER = [
    ("qwen4b", "Qwen/Qwen3-4B-Instruct-2507-FP8", "/models2/Qwen3-4B-Instruct-2507-FP8", "fp8", []),
    ("gptoss20b", "openai/gpt-oss-20b", "/models/gpt-oss-20b", None, GPTOSS_FLAGS),
    ("qwen36", "Qwen/Qwen3.6-35B-A3B-FP8", "/models/Qwen3.6-35B-A3B-FP8", "fp8", []),
    ("gemma26b", "google/gemma-4-26B-A4B-it", "/models2/gemma-4-26B-A4B-it", None, []),
    ("nemonano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
     "/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", None, []),
    ("kimilinear", "moonshotai/Kimi-Linear-48B-A3B-Instruct",
     "/models/Kimi-Linear-48B-A3B-Instruct", None, []),
    ("glm47", "zai-org/GLM-4.7-Flash", "/models3/GLM-4.7-Flash", None, []),
    ("gemma31b", "google/gemma-4-31B-it", "/models2/gemma-4-31B-it", None, []),
    ("nemosuper", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
     "/models2/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", None, []),
]
GPTOSS120B_CPU = ("openai/gpt-oss-120b", "/models/gpt-oss-120b", None, GPTOSS_FLAGS)
MINIMAX_CPU = ("MiniMaxAI/MiniMax-M2.7", "/models3/MiniMax-M2.7", "fp8", [])


def clock_settings() -> dict:
    """box_settings for the clock caps (cpupower, both sockets)."""
    reset = ("sudo cpupower frequency-set -u "
             "$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq) >/dev/null")
    return {f"clock{g}": {"apply": f"sudo cpupower frequency-set -u {g}GHz >/dev/null",
                          "reset": reset} for g in CPU_CLOCKS_GHZ}


MEM_RELIABILITY = "BIOS: SysProfile Custom, MemFrequency MaxReliability"


def bios(jid, what, *a, **kw):
    """A run that needs a BIOS change and reboot: queued as manual."""
    return cjob(jid, *a, manual=what, **kw)


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
    for w in HISTORY:
        jobs.append(cjob(f"C1.8-qwen30b-1s-{w}-nocache", m, loc, qn, w, 1,
                         cpu_engine_extra={"sglang_extra_flags": ["--disable-radix-cache"]},
                         pin=["cpu_engine"]))
    # C1.9: no shared system prompt (see G1.8).
    for w in SYS_CONTROL:
        jobs.append(cjob(f"C1.9-qwen30b-1s-{w}-sys0", m, loc, qn, w, 1,
                         simulation={"system_prompt_tokens": None}))
    # Two sockets: one instance per socket (TP2 and SNC layouts run as
    # separate jobs once the per-socket baseline is in).
    for w in ["quick_lookup", "document_qa", "long_form_generator"]:
        jobs.append(cjob(f"C1.3-qwen30b-2s-{w}-tp2", m, loc, qn, w, 2,
                         cpu_engine_extra={"tensor_parallel_size": 2}, barrier=(w == "quick_lookup")))
    for mix in MIX6:
        jobs.append(cjob(f"C1.7-qwen30b-2s-{mix}-tp2", m, loc, qn, mix, 2,
                         cpu_engine_extra={"tensor_parallel_size": 2}))
    # ── C1.2 and C2: fixed shapes for precision and other Xeon SKUs ──
    bf = "/models/Qwen3-30B-A3B-Instruct-2507"
    for w in SHAPES:
        jobs.append(cjob(f"C1.2-qwen30b-1s-{w}-fp8", m, loc, qn, w, 1))
        jobs.append(cjob(f"C1.2-qwen30b-1s-{w}-bf16", QWEN30B_CPU[1], bf, None, w, 1))
    for c in CPU_CORES:  # 86 cores is C1.2 fp8
        for w in SHAPES:
            jobs.append(cjob(f"C2.1-qwen30b-1s-{w}-cores{c}", m, loc, qn, w, 1,
                             socket_cores=c, pin=["socket_cores"]))
    # Hyper-threads used by the engine: every thread of socket 0, the
    # load generator on socket 1 (the job holds both).
    for w in SHAPES:
        jobs.append(cjob(f"C2.5-qwen30b-1s-{w}-ht", m, loc, qn, w, 2, socket_threads=2,
                         initial_rate_per_s=CPU_SHAPE_START))
    # Clock caps last among the runnable C2 work: each cap holds the box.
    for g in CPU_CLOCKS_GHZ:
        for w in SHAPES:
            jobs.append(cjob(f"C2.2-qwen30b-1s-{w}-clock{g}", m, loc, qn, w, 1,
                             box_setting=f"clock{g}"))
        if g == 2.4:  # C2.6 corner: 32 cores at 2.4 GHz, three personas
            for w in ["quick_lookup", "document_qa", "long_form_generator"]:
                jobs.append(cjob(f"C2.6-qwen30b-1s-{w}-cores32-clock2.4", m, loc, qn, w, 1,
                                 socket_cores=32, pin=["socket_cores"], box_setting="clock2.4"))
    # BIOS-dependent runs: queued as manual until the box is reconfigured.
    # The R770 BIOS offers MaxPerf (5,200 MT/s here) or MaxReliability
    # (SysProfile Custom); there is no 4,800 or 4,400 setting.
    for w in SHAPES:
        jobs.append(bios(f"C2.3-qwen30b-1s-{w}-memrel", MEM_RELIABILITY, m, loc, qn, w, 1))
        jobs.append(bios(f"C2.5-qwen30b-1s-{w}-htoff", "BIOS: LogicalProc Disabled",
                         m, loc, qn, w, 2, loadgen_other_socket=True,
                         initial_rate_per_s=CPU_SHAPE_START))
    for w in SHAPES:
        jobs.append(bios(f"C2.4-qwen30b-snc-{w}", "BIOS: SNC on (one instance per SNC domain)",
                         m, loc, qn, w, 2))
        jobs.append(bios(f"C2.5-qwen30b-1s-{w}-amxoff", "Kernel: clearcpuid=amx_tile",
                         m, loc, qn, w, 1, cpu_engine_extra={"attention_backend": "torch_native"}))
    for w in ["quick_lookup", "document_qa", "long_form_generator"]:
        jobs.append(bios(f"C2.6-qwen30b-1s-{w}-memrel", MEM_RELIABILITY, m, loc, qn, w, 1))
    # ── C3: model ladder on one socket ──
    # Each model's first run is its load probe; the rest wait for it.
    for key, hf, path, quant, flags in CPU_LADDER:
        ex = {"sglang_extra_flags": list(flags)} if flags else {}
        reason = {"reasoning_effort": "low"} if flags is GPTOSS_FLAGS else {}
        probe = f"C3.1-{key}-1s-shape_chat"
        jobs.append(cjob(probe, hf, path, quant, "shape_chat", 1, cpu_engine_extra=dict(ex), **reason))
        for w in SHAPES[1:] + CORE6:
            jobs.append(cjob(f"C3.1-{key}-1s-{w}", hf, path, quant, w, 1, after=probe,
                             cpu_engine_extra=dict(ex), **reason))
        if key == "gptoss20b":
            for eff in ["medium", "high"]:
                for w in ["conversational", "document_qa", "knowledge_search"]:
                    jobs.append(cjob(f"C3.3-{key}-1s-{w}-reason{eff}", hf, path, quant, w, 1,
                                     after=probe, cpu_engine_extra=dict(ex), reasoning_effort=eff))
    hf, path, quant, flags = GPTOSS120B_CPU
    probe = "C3.2-gptoss120b-1s-shape_chat"
    jobs.append(cjob(probe, hf, path, quant, "shape_chat", 1,
                     cpu_engine_extra={"sglang_extra_flags": list(flags)}, reasoning_effort="low"))
    for w in SHAPES[1:] + ALL16:
        jobs.append(cjob(f"C3.2-gptoss120b-1s-{w}", hf, path, quant, w, 1, after=probe,
                         cpu_engine_extra={"sglang_extra_flags": list(flags)},
                         reasoning_effort="low"))
    # MiniMax-M2.7 (FP8, 230 GB) across both sockets.
    hf, path, quant, _ = MINIMAX_CPU
    probe = "C3.4-minimax-2s-quick_lookup-tp2"
    jobs.append(cjob(probe, hf, path, quant, "quick_lookup", 2,
                     cpu_engine_extra={"tensor_parallel_size": 2}, barrier=True))
    for w in ["conversational", "document_qa"]:
        jobs.append(cjob(f"C3.4-minimax-2s-{w}-tp2", hf, path, quant, w, 2, after=probe,
                         cpu_engine_extra={"tensor_parallel_size": 2}))
    return {
        "box": "r770",
        "box_settings": clock_settings(),
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
