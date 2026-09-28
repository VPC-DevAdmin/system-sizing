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
    if transfer_settings.get("p01_block_size"):
        common += ["--block-size", str(transfer_settings["p01_block_size"])]
    if transfer_settings.get("p01_max_batched_tokens"):
        common += ["--max-num-batched-tokens", str(transfer_settings["p01_max_batched_tokens"])]
    if plan.get("sessions"):
        common[common.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
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
        "UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp",
        "UCX_NET_DEVICES": "lo",
        "UCX_LOG_LEVEL": "info",
    }
    if transfer_settings.get("p01_transport") == "auto":
        env.update(UCX_TLS="all", UCX_NET_DEVICES="all")
    if transfer_settings.get("p01_cuda_ipc_get") == "enabled":
        env["UCX_CUDA_IPC_ENABLE_GET_ZCOPY"] = "y"
    if transfer_settings.get("p01_memtype_cache") == "disabled":
        env["UCX_MEMTYPE_CACHE"] = "n"
    rows = []
    pool_count = len(plan.get("frontend_cpus", [[]]))
    for group in range(pool_count):
        pool_env = dict(env, DYN_SYSTEM_PORT=str(base + 9 + group * 40))
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
                    "-m",
                    "dynamo.frontend",
                    "--http-port",
                    str(base + group * 40),
                    "--router-mode",
                    "round-robin",
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
                        **(
                            {"kv_connector_extra_config": {"enable_cross_layers_blocks": True}}
                            if transfer_settings.get("p01_cross_layer_blocks")
                            else {}
                        ),
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
            exited = [p for p in children if p.poll() is not None]
            if exited:
                raise RuntimeError("A Dynamo service exited; inspect the per-service logs")
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


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        main(json.load(f))
