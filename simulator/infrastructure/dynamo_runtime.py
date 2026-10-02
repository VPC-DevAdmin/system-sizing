"""Container-local process supervisor for pinned Dynamo deployments.

Invoked as a file inside the image; it deliberately has no capsim dependencies.
All services share a private file-discovery directory inside this owned container.
"""

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time


def commands(plan):
    base = plan["port"]
    common = [
        "python3",
        "-m",
        "dynamo.vllm",
        "--model",
        plan["model"],
        "--served-model-name",
        plan["served_model_name"],
        "--max-model-len",
        str(plan["context_limit"]),
        "--max-num-seqs",
        str(plan["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(plan["gpu_memory_utilization"]),
        "--tensor-parallel-size",
        "1",
        "--no-enable-prefix-caching",
        "--kv-cache-dtype",
        plan["kv_cache_dtype"],
    ]
    transfer_settings = plan.get("transfer_settings", {})
    if transfer_settings.get("p01_execution") in ("sync", "eager"):
        common += ["--no-async-scheduling"]
    if transfer_settings.get("p01_execution") == "eager":
        common += ["--enforce-eager"]
    if transfer_settings.get("p01_block_size"):
        common += ["--block-size", str(transfer_settings["p01_block_size"])]
    if transfer_settings.get("p01_max_batched_tokens"):
        common += ["--max-num-batched-tokens", str(transfer_settings["p01_max_batched_tokens"])]
    if transfer_settings.get("p01_attention_backend", "auto") != "auto":
        common += ["--attention-backend", transfer_settings["p01_attention_backend"]]
    if plan.get("sessions"):
        common[common.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
        if plan["sessions"].get("session_gpu_gb") is not None:
            common += ["--kv-cache-memory-bytes", str(plan["sessions"]["session_gpu_gb"] * 1024**3)]
    if plan.get("quantization"):
        common += ["--quantization", plan["quantization"]]
    env = {
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_FILE_KV": "/tmp/capsim-dynamo-discovery",
        "DYN_NAMESPACE": plan["namespace"],
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "UCX_RCACHE_MAX_UNRELEASED": "1024",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        # NIXL three-read transfers require DS layout for hybrid SSM models.
        # Keep the same state layout in aggregated and disaggregated arms.
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp",
        "UCX_NET_DEVICES": "lo",
        "UCX_LOG_LEVEL": "info",
    }
    if transfer_settings.get("p01_cuda_launch_blocking"):
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    if transfer_settings.get("p01_transport") == "auto":
        env.update(UCX_TLS="all", UCX_NET_DEVICES="all")
    if transfer_settings.get("p01_cuda_ipc_get") == "enabled":
        env["UCX_CUDA_IPC_ENABLE_GET_ZCOPY"] = "y"
    if transfer_settings.get("p01_memtype_cache") == "disabled":
        env["UCX_MEMTYPE_CACHE"] = "n"
    gptoss_sessions = bool(plan.get("sessions")) and "gpt-oss" in plan["model"].lower()
    rows = []
    pool_count = len(plan.get("frontend_cpus", [[]]))
    for group in range(pool_count):
        pool_env = dict(env, DYN_SYSTEM_PORT=str(base + 9 + group * 40))
        pool_env["DYN_CHAT_PROCESSOR"] = "vllm" if gptoss_sessions else transfer_settings.get("p01_chat_processor", "dynamo")
        if "worker_groups" in plan:
            pool_env.update(
                DYN_NAMESPACE=plan["namespace"] + f"-pool-{group}",
                DYN_FILE_KV=f"/tmp/capsim-dynamo-discovery/pool-{group}",
                CAPSIM_CPU_AFFINITY=json.dumps(plan["frontend_cpus"][group]),
            )
        rows.append(
            (
                "frontend" if pool_count == 1 else f"frontend-{group}",
                [
                    "python3",
                    *([__file__, "--harmony-frontend"] if gptoss_sessions else ["-m", "dynamo.frontend"]),
                    "--http-port",
                    str(base + group * 40),
                    *(["--reasoning-parser", "openai_gptoss"] if gptoss_sessions else []),
                    "--router-mode",
                    "round-robin",
                    *(
                        ["--tokenizer-mode", "mistral", "--config-format", "mistral"]
                        if transfer_settings.get("p01_frontend_format") == "mistral"
                        else []
                    ),
                ],
                pool_env,
            )
        )
    for index in range(len(plan["gpu_ids"])):
        worker_env = dict(
            env,
            CUDA_VISIBLE_DEVICES=(plan["gpu_uuids"][index] if "gpu_uuids" in plan else str(index)),
            DYN_SYSTEM_PORT=str(base + 1 + index),
            VLLM_NIXL_SIDE_CHANNEL_PORT=str(base + 10 + index),
            PROMETHEUS_MULTIPROC_DIR=f"/tmp/capsim-prometheus-worker-{index}",
        )
        if "worker_groups" in plan:
            group = plan["worker_groups"][index]
            worker_env.update(
                DYN_NAMESPACE=plan["namespace"] + f"-pool-{group}",
                DYN_FILE_KV=f"/tmp/capsim-dynamo-discovery/pool-{group}",
                CAPSIM_CPU_AFFINITY=json.dumps(plan["worker_cpus"][index]),
            )
        argv = list(common)
        role = "aggregated"
        if plan.get("sessions") and plan["arm"] == "treatment":
            settings = plan["sessions"]
            if settings.get("session_cpu_backend", "kvbm") == "simple_cpu":
                if settings["session_tier"] != "cpu":
                    raise ValueError("Simple CPU offload supports CPU tier only")
                worker_env["PYTHONPATH"] = "/" + os.pathsep + os.environ.get("PYTHONPATH", "")
                argv += ["--kv-transfer-config", json.dumps({
                    "kv_connector": "ObservedCPUOffloadConnector",
                    "kv_connector_module_path": "capsim_native_cpu",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": settings["session_cpu_gb"] * 1024**3,
                    },
                })]
            else:
                worker_env.update(
                    DYN_KVBM_CPU_CACHE_GB=str(settings["session_cpu_gb"]),
                    DYN_KVBM_METRICS="true",
                    DYN_KVBM_METRICS_PORT=str(base + 20),
                    DYN_KVBM_LEADER_ZMQ_HOST="127.0.0.1",
                    DYN_KVBM_LEADER_ZMQ_PUB_PORT=str(base + 30),
                    DYN_KVBM_LEADER_ZMQ_ACK_PORT=str(base + 31),
                    DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS="5",
                )
                if settings["session_tier"] == "cpu_disk":
                    worker_env.update(
                        DYN_KVBM_DISK_CACHE_GB=str(settings["session_disk_gb"]),
                        DYN_KVBM_DISK_CACHE_DIR="/tmp/capsim-session-cache",
                        DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER="true",
                    )
                argv += [
                    "--kv-transfer-config",
                    json.dumps(
                        {
                            "kv_connector": "DynamoConnector",
                            "kv_connector_module_path": "kvbm.vllm_integration.connector",
                            "kv_role": "kv_both",
                        }
                    ),
                ]
        elif not plan.get("sessions") and plan["arm"] == "treatment":
            role = (
                plan["worker_roles"][index]
                if "worker_roles" in plan
                else "prefill"
                if index < plan["prefill_workers"]
                else "decode"
            )
            argv += [
                "--disaggregation-mode",
                role,
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "NixlConnector",
                        "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
                        "kv_load_failure_policy": "fail",
                        "kv_connector_extra_config": {
                            "kv_lease_duration": transfer_settings.get("p01_kv_lease_s", 30),
                            **(
                                {"enable_cross_layers_blocks": True}
                                if transfer_settings.get("p01_cross_layer_blocks")
                                else {}
                            ),
                        },
                    }
                ),
            ]
        rows.append((f"{role}-{index}", argv, worker_env))
    return rows


def main(plan):
    children, streams = [], []
    process_records = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.makedirs("/evidence", exist_ok=True)
    versions = {}
    for package in ("ai-dynamo", "ai-dynamo-runtime", "vllm", "nixl", "nixl-cu13", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    with open("/evidence/runtime-versions.json", "w") as out:
        json.dump({"python": sys.version, "packages": versions}, out, indent=2)
    with open("/evidence/service-commands.json", "w") as out:
        json.dump(commands(plan), out, indent=2)
    try:
        for name, argv, extra in commands(plan):
            if extra.get("PROMETHEUS_MULTIPROC_DIR"):
                os.makedirs(extra["PROMETHEUS_MULTIPROC_DIR"])
            if extra.get("DYN_FILE_KV"):
                os.makedirs(extra["DYN_FILE_KV"], exist_ok=True)
            cpus = (
                json.loads(extra["CAPSIM_CPU_AFFINITY"])
                if extra.get("CAPSIM_CPU_AFFINITY")
                else None
            )
            stream = open("/evidence/" + name + ".log", "ab", buffering=0)
            streams.append(stream)
            children.append(
                subprocess.Popen(
                    argv,
                    env=dict(os.environ, **extra),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    preexec_fn=(lambda cpus=cpus: os.sched_setaffinity(0, cpus)) if cpus else None,
                )
            )
            child = children[-1]
            if hasattr(child, "pid"):
                process_records.append(
                    dict(
                        name=name,
                        pid=child.pid,
                        initial_cpu_affinity=sorted(os.sched_getaffinity(child.pid)),
                    )
                )
                with open("/evidence/service-processes.json", "w") as out:
                    json.dump(process_records, out, indent=2)
        while not stopping:
            exited = [
                dict(name=commands(plan)[i][0], pid=getattr(p, "pid", None), returncode=p.poll())
                for i, p in enumerate(children)
                if p.poll() is not None
            ]
            if exited:
                with open("/evidence/service-exit.json", "w") as out:
                    json.dump(dict(exited=exited, observed_at=time.time()), out, indent=2)
                raise RuntimeError("Dynamo service exited: " + json.dumps(exited))
            time.sleep(0.5)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 15
        for child in children:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()


def install_harmony_postprocessor():
    """Bridge the pinned Dynamo frontend to vLLM's native GPT-OSS parser.

    No weights, scheduler, cache allocation, or token accounting is changed.
    State belongs to each request/choice; raw token IDs drive channel parsing.
    """
    from dynamo.frontend.prepost import StreamingPostProcessor
    from vllm.parser.harmony import HarmonyParser
    from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

    original = StreamingPostProcessor.process_output

    class GPTOSSHarmonyParser(HarmonyParser):
        reasoning_parser_cls = GptOssReasoningParser

    def process(self, output):
        if not isinstance(self.reasoning_parser, GptOssReasoningParser):
            return original(self, output)
        if self.tool_parser is not None:
            raise RuntimeError("P02 Harmony adapter does not support tool calls")
        if not hasattr(self, "_capsim_harmony"):
            self._capsim_harmony = {}
        if output.index not in self._capsim_harmony:
            self._capsim_harmony[output.index] = GPTOSSHarmonyParser(self.tokenizer)
        parser = self._capsim_harmony[output.index]
        message = parser.parse_delta(
            output.text or "", list(output.token_ids or []),
            self.request_for_sampling, finished=bool(output.finish_reason),
        )
        delta = {}
        if message:
            if message.content:
                delta["content"] = message.content
            if message.reasoning:
                delta["reasoning_content"] = message.reasoning
            if delta:
                delta["role"] = "assistant"
        if output.finish_reason:
            self._capsim_harmony.pop(output.index, None)
        if not delta and not output.finish_reason:
            return None
        return self._build_choice(output, delta)

    StreamingPostProcessor.process_output = process


if __name__ == "__main__":
    if sys.argv[1] == "--harmony-frontend":
        import runpy
        install_harmony_postprocessor()
        sys.argv = ["dynamo.frontend", *sys.argv[2:]]
        runpy.run_module("dynamo.frontend", run_name="__main__")
        sys.exit(0)
    with open(sys.argv[1]) as f:
        main(json.load(f))
