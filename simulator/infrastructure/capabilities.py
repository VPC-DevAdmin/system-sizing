"""Read-only host evidence. Unknown support is never interpreted as enabled."""

import platform
import shutil
import subprocess
import time
from pathlib import Path


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {
            "argv": argv,
            "returncode": p.returncode,
            "stdout": p.stdout[:65536],
            "stderr": p.stderr[:4096],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": type(exc).__name__}


def inspect_host():
    smi = shutil.which("nvidia-smi")
    evidence = {
        "schema_version": 1,
        "captured_at": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "commands": {},
        "devices": {
            p: Path(p).exists()
            for p in ("/dev/tdx_guest", "/dev/tdx-guest", "/dev/sgx_enclave", "/dev/kvm")
        },
    }
    if smi:
        queries = {
            "gpu": [
                smi,
                "--query-gpu=index,uuid,name,memory.total,driver_version,mig.mode.current,power.limit,power.min_limit,power.max_limit",
                "--format=csv",
            ],
            "topology": [smi, "topo", "-m"],
            "peer_read": [smi, "topo", "-p2p", "r"],
            "peer_write": [smi, "topo", "-p2p", "w"],
            "confidential": [smi, "conf-compute", "-q"],
            "compute_processes": [
                smi,
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            "gpu_admin": ["sudo", "-n", smi, "-L"],
        }
        evidence["commands"] = {k: command(v) for k, v in queries.items()}
    if shutil.which("lscpu"):
        evidence["commands"]["cpu"] = command(["lscpu"])
    if shutil.which("free"):
        evidence["commands"]["host_memory"] = command(["free", "--bytes"])
    from .dynamo import DYNAMO_IMAGE

    if shutil.which("docker"):
        evidence["commands"]["dynamo_image"] = command(
            ["docker", "image", "inspect", DYNAMO_IMAGE, "--format", "{{.Id}}"]
        )
    if shutil.which("lsblk"):
        evidence["commands"]["storage_topology"] = command(
            ["lsblk", "--json", "-o", "NAME,TYPE,SIZE,MODEL,TRAN,MOUNTPOINT"]
        )
    if shutil.which("docker"):
        evidence["commands"]["docker_storage_root"] = command(
            ["docker", "info", "--format", "{{.DockerRootDir}}"]
        )
    for tool in ("docker", "virsh", "ipmitool", "nsys"):
        evidence.setdefault("tools", {})[tool] = shutil.which(tool)
    cc = evidence["commands"].get("confidential", {}).get("stdout", "")
    evidence["summary"] = {
        "dynamo_image_ready": evidence["commands"].get("dynamo_image", {}).get("returncode") == 0,
        "dynamo_note": "Pinned image presence only; SM120 model loading and NIXL transfer still require runtime checks.",
        "gpu_visible": bool(smi and evidence["commands"]["gpu"]["returncode"] == 0),
        "gpu_admin": bool(smi and evidence["commands"]["gpu_admin"]["returncode"] == 0),
        "confidential_mode_evidence": cc.strip() or "Unavailable",
        "confidential_ready": False,
        "confidential_note": "Requires an attested guest, protection-path evidence, and policy tests; host capabilities alone do not qualify this project.",
        "wall_power_ready": False,
        "wall_power_note": "A validated server PSU or PDU sensor has not been configured. GPU power is not wall power.",
    }
    return evidence
