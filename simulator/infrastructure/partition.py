"""MIG transaction for an initially unpartitioned, idle GPU; restore is mandatory."""

import re
import subprocess
import time

from ..benchmarks.storage import read_json, write_json


def command(args, write=False):
    # This driver also requires elevated access for GPU-instance inventory.
    privileged = write or "-lgi" in args
    argv = (["sudo", "-n"] if privileged else []) + ["/usr/bin/nvidia-smi"] + args
    p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise RuntimeError("MIG command failed: " + p.stderr.strip() + " " + p.stdout.strip())
    return p.stdout


def gpu_state(device):
    fields = (
        command(
            [
                "-i",
                str(device),
                "--query-gpu=uuid,mig.mode.current,mig.mode.pending,display_active",
                "--format=csv,noheader",
            ]
        )
        .strip()
        .split(",")
    )
    if len(fields) != 4:
        raise RuntimeError("Unrecognized MIG state response")
    return dict(
        zip(["uuid", "current", "pending", "display"], [x.strip() for x in fields], strict=True)
    )


def devices(uuid):
    result = []
    selected = False
    for line in command(["-L"]).splitlines():
        if line.startswith("GPU "):
            selected = f"UUID: {uuid})" in line
        elif selected and "MIG " in line:
            m = re.search(r"MIG\s+(\S+)\s+Device\s+\d+: \(UUID: (MIG-[^)]+)\)", line)
            if not m:
                raise RuntimeError("Unrecognized MIG device record")
            result.append({"profile": m.group(1), "uuid": m.group(2)})
    return result


class Partition:
    def __init__(self, journal, owner, gpu):
        self.journal = journal
        self.owner = owner
        self.gpu = gpu
        self.record = None

    def create(self):
        initial = gpu_state(self.gpu)
        if (
            initial["current"] != "Disabled"
            or initial["pending"] != "Disabled"
            or initial["display"] != "Disabled"
        ):
            raise RuntimeError("MIG pilot requires an initially disabled, non-display GPU")
        if command(
            ["-i", initial["uuid"], "--query-compute-apps=pid", "--format=csv,noheader"]
        ).strip():
            raise RuntimeError("GPU compute processes are active; refusing MIG changes")
        if devices(initial["uuid"]):
            raise RuntimeError("Existing MIG instances must not be modified")
        profiles = command(["mig", "-i", initial["uuid"], "-lgip"])
        match = re.search(r"MIG 2g\.48gb\s+(\d+)\s+", profiles)
        if not match:
            raise RuntimeError("Two 48 GB MIG instances are unavailable on this GPU")
        self.record = {
            "owner": self.owner,
            "gpu": initial["uuid"],
            "initial": initial,
            "phase": "enabling",
            "profile_id": int(match.group(1)),
            "created": [],
            "events": [],
        }
        write_json(self.journal, self.record)
        self.change(["-i", initial["uuid"], "-mig", "1"])
        if gpu_state(initial["uuid"])["current"] != "Enabled":
            raise RuntimeError(
                "MIG enable requires administrator intervention; no reset or reboot attempted"
            )
        self.record["phase"] = "creating"
        write_json(self.journal, self.record)
        self.change(
            ["mig", "-i", initial["uuid"], "-cgi", f"{match.group(1)},{match.group(1)}", "-C"]
        )
        found = devices(initial["uuid"])
        self.record["created"] = found
        write_json(self.journal, self.record)
        if len(found) != 2 or any(x["profile"] != "2g.48gb" for x in found):
            raise RuntimeError("Unexpected MIG partition result")
        self.record["phase"] = "active"
        write_json(self.journal, self.record)
        return [x["uuid"] for x in found]

    def change(self, args):
        self.record["events"].append({"time": time.time(), "argv": args, "status": "started"})
        write_json(self.journal, self.record)
        output = command(args, write=True)
        self.record["events"][-1].update(status="completed", output=output)
        write_json(self.journal, self.record)

    def restore(self):
        if self.record is None:
            if not self.journal.exists():
                return
            self.record = read_json(self.journal)
        if self.record["owner"] != self.owner or self.record["initial"]["current"] != "Disabled":
            raise RuntimeError("MIG restore ownership is invalid")
        if self.record["phase"] == "restored":
            return
        uuid = self.record["gpu"]
        current = gpu_state(uuid)
        if current["uuid"] != uuid:
            raise RuntimeError("Physical GPU identity changed")
        found = devices(uuid)
        expected = {x["uuid"] for x in self.record["created"]}
        # Partial creation is recoverable only while the original reservation is
        # recorded and all observed instances match this exact bounded recipe.
        if expected and not {x["uuid"] for x in found}.issubset(expected):
            raise RuntimeError("Unowned MIG instance found; refusing destructive cleanup")
        if len(found) > 2 or any(x["profile"] != "2g.48gb" for x in found):
            raise RuntimeError("Unexpected MIG profiles; administrator review required")
        deadline = time.monotonic() + 20
        while command(["-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader"]).strip():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "MIG GPU still has compute processes; refusing partition teardown"
                )
            time.sleep(0.5)
        self.record["phase"] = "restoring"
        write_json(self.journal, self.record)
        if current["current"] == "Enabled" or current["pending"] == "Enabled":
            # This transaction started from zero instances and exclusively reserved
            # the GPU. Explicitly scoped commands cannot touch another GPU.
            instance_text = command(["mig", "-i", uuid, "-lgi"])
            instance_rows = [
                line for line in instance_text.splitlines() if re.search(r"\|\s*\d+\s+MIG ", line)
            ]
            if len(instance_rows) > 2 or any(
                not re.search(r"MIG 2g\.48gb\s+" + str(self.record["profile_id"]) + r"\s+", line)
                for line in instance_rows
            ):
                raise RuntimeError("Unexpected GPU instance profile; refusing cleanup")
            if found:
                self.change(["mig", "-i", uuid, "-dci"])
            if instance_rows:
                self.change(["mig", "-i", uuid, "-dgi"])
            self.change(["-i", uuid, "-mig", "0"])
        final = gpu_state(uuid)
        if final["current"] != "Disabled" or final["pending"] != "Disabled" or devices(uuid):
            raise RuntimeError("Original GPU mode was not restored")
        self.record.update(phase="restored", final=final)
        write_json(self.journal, self.record)
