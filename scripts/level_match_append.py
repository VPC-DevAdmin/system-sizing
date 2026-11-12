"""Append the level-matching tests (T1 and T2) to the live XE7740 queue.

    python scripts/level_match_append.py QUEUE.yaml [--dry-run]

T1 (``R-`` jobs, ``kind: reasoning_probe``) measures how long each variant
reasons on the planner's seven workloads, with real text
(scripts/reasoning_probe_job.py). T2 (``H4-`` jobs) reruns the H wave for the
variants whose reasoning setting moves them to a level: each job takes its
probe's mean reasoning on its workload as the allowance (``reasoning_from``)
and starts only after that probe is done (``after``). A probe that fails
skips its reruns.

Job ids carry the setting last (``H4-glm47-1gpu-h_assistant-think``), like
the existing controls (``-nocache``, ``-kvoff``), so the importer maps the
suffix to a variant and an old thinking-off run never collides with a new one.

Every variant gets an explicit reasoning parser, and models whose template
has a thinking switch get it set on the server (``--default-chat-template-kwargs``):
the Nemotron 3 and GLM-4.7 templates default to thinking on.

Scores (AA index v4.3.2, docs/aa-v432-variant-scores-2026-10-26.json):
GLM-4.7 Flash on 14.87 (Advanced; off 10.56), Nemotron 3 Super on 12.83
(Capable; no off score), Nemotron 3 Ultra on 22.93 (Frontier), MiniMax M2.7
22.76 (always reasons), Beam 1.0 30.33 (AA's single variant; Beam's default
effort is max), sized at both low and max. Gemma 4 26B is left out: off already scores 13.13, Advanced.
gpt-oss-120b low is probed only, to check the 32-token allowance its H1
runs used.

Jobs already in the queue are left alone. The old queue is kept beside it as
QUEUE.pre-level-match.yaml.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import yaml

WORKLOADS = ["h_assistant", "h_document_qa", "h_coding_agent", "h_research_agent",
             "h_service_agent", "h_document_pipeline", "h_deep_analysis"]
VOLUMES = {"/data2/capsim/huggingface": "/root/.cache/huggingface",
           "/data3/capsim/hf-overflow": "/data3/capsim/hf-overflow"}
SLOW_START = {"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}
BEAM_FLAGS = ["--kv-cache-dtype", "fp8", "--reasoning-parser", "beam"]


def thinking(on: bool) -> list[str]:
    return ["--default-chat-template-kwargs", json.dumps({"enable_thinking": on})]


# name -> engine, probe and rerun settings. ``rerun`` False: probe only.
VARIANTS = {
    "minimax-4gpu-think": {
        "model": "MiniMaxAI/MiniMax-M2.7", "tp": 4, "tune_key": "minimax-gpu",
        "flags": ["--reasoning-parser", "minimax_m2"], "template": "H2-minimax-4gpu-{wl}",
        "probe": {"temperature": 1.0, "top_p": 0.95}},
    "nemosuper-2gpu-think": {
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "tp": 2, "tune_key": "nemosuper-gpu",
        "flags": ["--reasoning-parser", "nemotron_v3", *thinking(True)], "startup_timeout_s": 3600,
        "template": "H2-minimax-4gpu-{wl}", "probe": {"temperature": 1.0, "top_p": 1.0}},
    "glm47-1gpu-think": {
        # BF16 weights on one GPU: at 128K context the default 0.90 share
        # leaves no room for the sampler warm-up (OOM, 6 Nov).
        "gpu_memory_utilization": 0.85,
        "model": "zai-org/GLM-4.7-Flash", "tp": 1, "tune_key": "glm47-gpu",
        "flags": ["--reasoning-parser", "glm47", *thinking(True)],
        "template": "H1-gemma26b-1gpu-{wl}", "probe": {"temperature": 1.0, "top_p": 0.95}},
    # GLM-4.7 Flash has never been measured on the H wave, so its off
    # setting gets a baseline too. The probe only confirms it is off.
    "glm47-1gpu-off": {
        # BF16 weights on one GPU: at 128K context the default 0.90 share
        # leaves no room for the sampler warm-up (OOM, 6 Nov).
        "gpu_memory_utilization": 0.85,
        "model": "zai-org/GLM-4.7-Flash", "tp": 1, "tune_key": "glm47-gpu",
        "flags": ["--reasoning-parser", "glm47", *thinking(False)],
        "template": "H1-gemma26b-1gpu-{wl}", "allowance": False,
        "probe": {"temperature": 1.0, "top_p": 0.95, "expect_reasoning": False}},
    "gptoss120b-1gpu-low": {
        "model": "openai/gpt-oss-120b", "tp": 1, "tune_key": "gptoss120b-gpu",
        "flags": ["--reasoning-parser", "openai_gptoss"], "rerun": False,
        "probe": {"temperature": 1.0, "top_p": 1.0, "extra_body": {"reasoning_effort": "low"}}},
    "nemoultra-8gpu-think": {
        "model": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "tp": 8,
        "flags": ["--reasoning-parser", "nemotron_v3", *thinking(True)], "env": SLOW_START,
        "template": "H2-nemoultra-8gpu-{wl}", "probe": {"temperature": 1.0, "top_p": 1.0}},
    # Beam's template applies max effort when none is sent. Both low and
    # max are sized, each its own variant (user, 2026-11-06).
    "beam-8gpu-low": {
        "model": "reflection/Beam-1.0-501B-A23B-NVFP4", "tp": 8, "image": "vllm/vllm-openai:reflection-beam",
        "flags": BEAM_FLAGS, "env": SLOW_START, "template": "H2-beam-8gpu-{wl}",
        "reasoning_effort": "low",
        "probe": {"temperature": 0.6, "top_p": 0.95, "extra_body": {"reasoning_effort": "low"}}},
    "beam-8gpu-max": {
        "model": "reflection/Beam-1.0-501B-A23B-NVFP4", "tp": 8, "image": "vllm/vllm-openai:reflection-beam",
        "flags": BEAM_FLAGS, "env": SLOW_START, "template": "H2-beam-8gpu-{wl}",
        "probe": {"temperature": 0.6, "top_p": 0.95}},
}


def engine_extra(v: dict) -> dict:
    e = {"docker_volumes": dict(VOLUMES), "vllm_extra_flags": list(v["flags"])}
    if v.get("image"):
        e["gpu_image"] = v["image"]
    if v.get("env"):
        e["docker_extra_env"] = dict(v["env"])
    if v.get("startup_timeout_s"):
        e["startup_timeout_s"] = v["startup_timeout_s"]
    return e


def base(v: dict) -> dict:
    j = {"model": v["model"], "tp": v["tp"], "devices": v["tp"], "max_model_len": 131072,
         "engine_extra": engine_extra(v)}
    if v.get("gpu_memory_utilization"):
        j["gpu_memory_utilization"] = v["gpu_memory_utilization"]
    if v.get("tune_key"):
        j["tune_key"] = v["tune_key"]
    return j


def probe_id(name: str) -> str:
    model, gpus, setting = name.rsplit("-", 2)
    return f"R-{model}-{gpus}-{setting}"


def jobs(existing: dict[str, dict]) -> list[dict]:
    probes, reruns = [], []
    for name, v in VARIANTS.items():
        pid = probe_id(name)
        probes.append({"id": pid, "kind": "reasoning_probe", "workloads": [], **base(v),
                       "probe": {"concurrency": 16, "reasoning_room": 32768, **v["probe"]}})
        if v.get("rerun", True) is False:
            continue
        model, gpus, setting = name.rsplit("-", 2)
        for wl in WORKLOADS:
            tmpl = existing[v["template"].format(wl=wl)]
            j = {"id": f"H4-{model}-{gpus}-{wl}-{setting}", **base(v), "workloads": [wl],
                 "initial_rate_per_s": tmpl.get("initial_rate_per_s", 0.25),
                 "simulation": copy.deepcopy(tmpl.get("simulation") or {}), "after": pid}
            if v.get("allowance", True):
                j["reasoning_from"] = pid
            if v.get("reasoning_effort"):
                # Sent with every request, as the probe sent it.
                j["reasoning_effort"] = v["reasoning_effort"]
            reruns.append(j)
    return probes + reruns


# Decision 6: a score for Beam at low effort. HLE (text only, 2,158
# questions, official prompt and judge; scripts/opencompass_job.py) on Beam
# at max and at low. AA's published HLE for Beam is 34.7, so the max run
# checks which setting AA scored. The low-max gap converts to an index gap
# through the HLE-to-index fit. gpt-oss-120b high calibrates the harness
# against AA's published 19.6. A local Qwen3.5 397B judge grades all three.
HLE = {"phase": "infer", "tests": ["hle"], "max_tokens": 100000, "timeout_s": 7200, "reasoning": True}


def hle_jobs() -> tuple[list[dict], list[dict]]:
    """(calibration job to run among the small reruns, Beam and judge jobs for the end)."""
    gptoss = {"id": "OC-hle-gptoss120b-1gpu-high", "kind": "opencompass", "workloads": [],
              "model": "openai/gpt-oss-120b", "tp": 1, "devices": 1, "max_model_len": 131072,
              "tune_key": "gptoss120b-gpu",
              "engine_extra": {"docker_volumes": dict(VOLUMES),
                               "vllm_extra_flags": ["--reasoning-parser", "openai_gptoss"]},
              "opencompass": {**HLE, "temperature": 1.0, "top_p": 1.0,
                              "api_kwargs": {"reasoning_effort": "high"}, "concurrency": 16}}
    beam = VARIANTS["beam-8gpu-max"]
    tail = []
    for setting, kwargs in (("max", {}), ("low", {"reasoning_effort": "low"})):
        tail.append({"id": f"OC-hle-beam-8gpu-{setting}", "kind": "opencompass", "workloads": [],
                     **base(beam),
                     "opencompass": {**HLE, "temperature": 0.6, "top_p": 0.95,
                                     "api_kwargs": kwargs, "concurrency": 64}})
    tail.append({"id": "OC-hle-judge-qwen35-4gpu", "kind": "opencompass", "workloads": [],
                 "barrier": True, "after": "OC-hle-beam-8gpu-low",
                 "model": "nvidia/Qwen3.5-397B-A17B-NVFP4", "tp": 4, "devices": 4, "max_model_len": 131072,
                 "engine": "sglang_cuda",
                 "engine_extra": {"sglang_extra_flags": ["--moe-runner-backend", "flashinfer_cutlass",
                                                         "--reasoning-parser", "qwen3"],
                                  "docker_volumes": dict(VOLUMES)},
                 "opencompass": {"phase": "judge",
                                 "grade": [gptoss["id"], "OC-hle-beam-8gpu-max", "OC-hle-beam-8gpu-low"]}})
    return [gptoss], tail


def main() -> int:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    qpath = Path(args[0])
    q = yaml.safe_load(qpath.read_text())
    existing = {j["id"]: j for j in q["jobs"]}
    new = [j for j in jobs(existing) if j["id"] not in existing]
    calib, tail = hle_jobs()
    calib = [j for j in calib if j["id"] not in existing]
    tail = [j for j in tail if j["id"] not in existing]
    if "--dry-run" in args:
        for j in new + calib + tail:
            print(j["id"], j.get("after", ""), j.get("reasoning_from", ""), j["engine_extra"]["vllm_extra_flags"])
        print(f"{len(new)} jobs")
        return 0
    shutil.copy(qpath, qpath.with_name(qpath.stem + ".pre-level-match.yaml"))
    out = list(q["jobs"]) + new
    # The 1-GPU calibration runs among the small reruns, ahead of the first
    # 8-GPU rerun, so it fills a spare GPU instead of holding up a full-box job.
    at = next((i for i, j in enumerate(out) if j["id"].startswith("H4-nemoultra-")), len(out))
    out[at:at] = calib
    q["jobs"] = out + tail
    new = new + calib + tail
    tmp = qpath.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(q, sort_keys=False, width=120))
    yaml.safe_load(tmp.read_text())
    tmp.replace(qpath)
    print(f"added {len(new)} jobs; queue now {len(q['jobs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
