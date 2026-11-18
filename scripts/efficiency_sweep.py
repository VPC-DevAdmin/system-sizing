#!/usr/bin/env python3
"""Efficiency sweep on the XE7740: many models, one method, wall power per rung.

Plan: docs/efficiency-sweep-plan.md. Builds on inferencex_xe7740.py (fixed
1k/1k and 8k/1k shapes, prefix caching off, GPU power + iDRAC wall power per
rung, p90 interactivity) and the P11 driver underneath it (engine launch in
Docker, AIPerf load, resumable per-rung records).

    efficiency_sweep.py list
    efficiency_sweep.py sweep --workload 1k1k [--resume] <config>...   # all configs when none given
    efficiency_sweep.py sweep --workload 128x128 --ladder 1024,2048 <config>...   # roofline rungs
    efficiency_sweep.py screen --workload 1k1k --concurrency 64 <config>

Results: /data/capsim/effsweep/results/<workload>/<config>/result.json
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferencex_xe7740 as ix  # noqa: E402  (patches the P11 driver: power samplers, prefix cache off)

p = ix.p
# Roofline shape, the Dell/Metrum post's reading: short prompts, short answers, maximum load, no user-speed target.
ix.WORKLOADS["128x128"] = (128, 128)
p.OUT = Path(os.environ.get("EFF_OUT", "/data/capsim/effsweep/results"))

GIANT_ENV = {"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}

# TensorRT-LLM, the engine behind the Dell/Metrum baseline. Launched through the image's own
# entrypoint (an --entrypoint override loses /usr/local/tensorrt/lib, see simulator/engines/trtllm.py),
# weights by hub id from the mounted cache, tokenizer by explicit snapshot path (hub lookup is
# refused offline), prefix reuse off through the options YAML (no CLI flag for it).
TRTLLM = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
TRTLLM_OPTS = "/data3/capsim/hf-overflow/trtllm_effsweep.yaml"  # under the OVERFLOW mount, same path inside
_engine_docker_cmd = p.docker_cmd


def docker_cmd(name, c, gpus, port):
    if c["engine"] != "trtllm":
        return _engine_docker_cmd(name, c, gpus, port)
    snaps = sorted(glob.glob(f"{p.HF_CACHE}/hub/models--{c['model'].replace('/', '--')}/snapshots/*"))
    tokenizer = snaps[-1].replace(p.HF_CACHE, "/root/.cache/huggingface")
    gpu_arg = "device=" + ",".join(map(str, gpus))
    cmd = [
        "docker", "run", "-d", "--name", name, "--gpus", f'"{gpu_arg}"' if len(gpus) > 1 else gpu_arg,
        "--ipc=host", "--network", "host", "--shm-size", "32g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{p.HF_CACHE}:/root/.cache/huggingface", "-v", f"{p.OVERFLOW}:{p.OVERFLOW}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "--label", "capsim.p11=1",
    ]
    for k, v in c["env"].items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        c["image"], "trtllm-serve", "serve", c["model"], "--host", "0.0.0.0", "--port", str(port),
        "--backend", "pytorch", "--tp_size", str(c["tp"]), "--max_seq_len", str(p.MAX_MODEL_LEN),
        "--max_batch_size", str(p.max_seqs(c)), "--max_num_tokens", "16384",
        "--tokenizer", tokenizer, "--extra_llm_api_options", TRTLLM_OPTS,
    ]
    return cmd + c["args"]


p.docker_cmd = docker_cmd
LADDER = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def cfg(model, engine, tp, replicas, max_conc=2048, workloads=("1k1k", "8k1k"), **kw):
    c = p.cfg(model, engine, tp, replicas, **kw)
    c["max_conc"] = max_conc          # ladder cap: one copy over 8 GPUs cannot run 2,048 users
    c["workloads"] = list(workloads)
    return c


QWEN35 = ix.p.CONFIGS["qwen3.5-397b-nvfp4-sglang-tp4"]
QWEN35_MTP = ix.p.CONFIGS["qwen3.5-397b-nvfp4-sglang-tp4-mtp"]
for c in (QWEN35, QWEN35_MTP):
    c["max_conc"], c["workloads"] = 1024, ["1k1k", "8k1k"]

p.CONFIGS = {
    # Reference point: the Dell/Metrum AI post (Nov 2025) ran Llama 3.3 70B FP4 on 8x RTX PRO 6000 with TRT-LLM:
    # 45,527 tok/s and 7.03 tok/s per system watt. llama-3.3-70b-nvfp4 below is the same model on this box with vLLM.
    # Dense
    "llama-3.3-70b-nvfp4": cfg("nvidia/Llama-3.3-70B-Instruct-NVFP4", "vllm", 1, 8),
    "llama-3.3-70b-nvfp4-trtllm": cfg("nvidia/Llama-3.3-70B-Instruct-NVFP4", "trtllm", 1, 8, image=TRTLLM),
    "gemma-4-31b-nvfp4": cfg("nvidia/Gemma-4-31B-IT-NVFP4", "vllm", 1, 8),
    "qwen3-32b-fp8": cfg("Qwen/Qwen3-32B-FP8", "vllm", 1, 8),
    "mistral-small-3.2-24b": cfg("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "vllm", 1, 8),
    # MoE, small and mid
    "gpt-oss-20b": cfg("openai/gpt-oss-20b", "vllm", 1, 8, reasoning="low"),
    "gpt-oss-120b": cfg("openai/gpt-oss-120b", "vllm", 1, 8, reasoning="low"),
    "gemma-4-26b-a4b-nvfp4": cfg("nvidia/Gemma-4-26B-A4B-NVFP4", "vllm", 1, 8),
    "qwen3-30b-a3b-fp8": cfg("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "vllm", 1, 8),
    "qwen3.6-35b-a3b-nvfp4": cfg("nvidia/Qwen3.6-35B-A3B-NVFP4", "vllm", 1, 8),
    "glm-4.7-flash": cfg("zai-org/GLM-4.7-Flash", "vllm", 1, 8),
    "nemotron-3-super-120b-nvfp4": cfg("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "vllm", 2, 4),
    # MoE, large
    "qwen3-235b-a22b-fp8": cfg("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "vllm", 4, 2, max_conc=1024),
    "minimax-m2.7-fp8": cfg("MiniMaxAI/MiniMax-M2.7", "vllm", 4, 2, max_conc=1024),
    "qwen3.5-397b-a17b-nvfp4-sglang": QWEN35,
    "qwen3.5-397b-a17b-nvfp4-sglang-mtp": QWEN35_MTP,
    "beam-1.0-501b-nvfp4": cfg(p.BEAM, "vllm", 8, 1, max_conc=512, image=p.VLLM_BEAM, args=p.BEAM_ARGS,
                               env=GIANT_ENV, reasoning="low"),
    "nemotron-3-ultra-550b-nvfp4": cfg("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "vllm", 8, 1,
                                       max_conc=512, env=GIANT_ENV),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "screen", "sweep"])
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--workload", choices=sorted(ix.WORKLOADS), default="1k1k")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--ladder", help="comma-separated server-wide concurrencies (overrides the cap)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_intermixed_args()
    if a.cmd == "list":
        for k, v in p.CONFIGS.items():
            print(f"{k:38} {v['model']:48} {v['engine']:7} tp{v['tp']} x{v['replicas']}  "
                  f"<= {v['max_conc']} users  {','.join(v['workloads'])}")
        return
    p.ISL, p.OSL = ix.WORKLOADS[a.workload]
    configs = a.configs or [k for k, v in p.CONFIGS.items() if a.workload in v["workloads"]]
    for cid in configs:
        c = p.CONFIGS[cid]
        if a.workload not in c["workloads"] and not a.configs:  # explicit configs run at any shape (roofline pass)
            p.log(f"skip {cid}: not run at {a.workload}")
            continue
        if a.cmd == "screen":
            p.run(cid, [a.concurrency], False, f"{a.workload}-screen", a.keep)
            continue
        ladder = [int(x) for x in a.ladder.split(",")] if a.ladder else [n for n in LADDER if n <= c["max_conc"]]
        p.run(cid, ladder, False, a.workload, a.keep, a.resume)


if __name__ == "__main__":
    main()
