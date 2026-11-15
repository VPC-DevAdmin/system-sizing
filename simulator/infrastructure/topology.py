"""Discover physical GPU locality and build isolated two-socket serving pools."""

import os
import subprocess
from pathlib import Path


def expand_cpus(text):
    result = set()
    for part in text.strip().split(","):
        bounds = [int(n) for n in part.split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _pci_addresses():
    """{GPU index: PCI address as sysfs spells it (dddd:bb:dd.f)}."""
    from ..gpu_vendor import amd_pci_addresses, is_amd
    if is_amd():
        return amd_pci_addresses()
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    out = {}
    for line in raw.splitlines():
        index, address = [v.strip() for v in line.split(",")]
        domain, bus, function = address.lower().split(":")
        out[int(index)] = f"{int(domain, 16):04x}:{bus}:{function}"
    return out


def discover(gpu_ids):
    inventory = {}
    for index, address in _pci_addresses().items():
        if int(index) in gpu_ids:
            node = int((Path("/sys/bus/pci/devices") / address / "numa_node").read_text())
            if node < 0:
                raise ValueError("GPU NUMA membership is unknown; refusing a locality claim")
            cpus = expand_cpus(
                (Path("/sys/devices/system/node") / f"node{node}" / "cpulist").read_text()
            ) & os.sched_getaffinity(0)
            sockets = {
                int(
                    (
                        Path("/sys/devices/system/cpu")
                        / f"cpu{cpu}"
                        / "topology/physical_package_id"
                    ).read_text()
                )
                for cpu in cpus
            }
            if not cpus or len(sockets) != 1:
                raise ValueError("NUMA node does not identify one accessible CPU socket")
            inventory[int(index)] = dict(
                pci_address=address, node=node, socket=sockets.pop(), cpus=sorted(cpus)
            )
    if set(inventory) != set(gpu_ids):
        raise ValueError("Selected GPU missing from physical topology")
    return inventory


def pool_plan(gpu_ids, prefill, arm, placement, inventory):
    nodes = sorted({inventory[g]["node"] for g in gpu_ids})
    sockets = {inventory[g]["socket"] for g in gpu_ids}
    members = [[i for i, g in enumerate(gpu_ids) if inventory[g]["node"] == node] for node in nodes]
    if (
        len(nodes) not in (1, 2)
        or len(sockets) != len(nodes)
        or len({len(m) for m in members}) != 1
    ):
        raise ValueError(
            "Socket placement requires one or two equally sized GPU groups on distinct CPU sockets"
        )
    if placement == "cross_socket" and len(nodes) != 2:
        raise ValueError("Cross-socket control requires two CPU sockets")
    counts = [prefill] if len(nodes) == 1 else [prefill // 2, prefill - prefill // 2]
    if arm == "treatment" and any(not 1 <= p < len(members[i]) for i, p in enumerate(counts)):
        raise ValueError("Each socket pool needs at least one prefill and one decode GPU")
    roles = ["aggregated"] * len(gpu_ids)
    groups = [0] * len(gpu_ids)
    for group, indices in enumerate(members):
        for j, index in enumerate(indices):
            role = (
                "aggregated" if arm == "baseline" else "prefill" if j < counts[group] else "decode"
            )
            roles[index] = role
            groups[index] = 1 - group if placement == "cross_socket" and role == "decode" else group
    allowed = []
    for p, role in enumerate(roles):
        if role != "prefill":
            continue
        for d, other in enumerate(roles):
            if other == "decode" and groups[p] == groups[d]:
                same = inventory[gpu_ids[p]]["socket"] == inventory[gpu_ids[d]]["socket"]
                assert same == (placement == "socket_local")
                allowed.append(
                    dict(prefill_gpu=gpu_ids[p], decode_gpu=gpu_ids[d], same_socket=same)
                )
    return dict(
        placement=placement,
        worker_roles=roles,
        worker_groups=groups,
        worker_cpus=[inventory[g]["cpus"] for g in gpu_ids],
        frontend_cpus=[inventory[gpu_ids[indices[0]]]["cpus"] for indices in members],
        inventory=inventory,
        allowed_kv_pairs=allowed,
        routing="Separate discovery stores and namespaces; least outstanding client requests across pools; round-robin within each pool",
        memory_policy="CPU affinity enforced; host memory binding is not asserted",
    )
