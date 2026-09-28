"""Explicit interrupted-job recovery, restricted to recorded and labeled resources."""

import json
import re
import subprocess
import threading
import time

from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..config import EngineConfig
from .deployment import OwnedVllm


def recover(path):
    """Operator-triggered recovery; takes every lock the campaign could still hold."""
    with (
        Lease(path / ".execution.lock"),
        Lease(path / ".worker.lock"),
        Lease(hardware_lease_path()),
    ):
        return recover_unlocked(path)


def recover_unlocked(path, *, mark_interrupted=True):
    """Release recorded containers and partitions. Caller owns the campaign locks.

    ``mark_interrupted`` is False when the supervisor calls this right after the
    worker died, so the campaign keeps its failed/cancelled phase.
    """
    state = read_json(path / "state.json")
    if state["phase"] == "complete":
        raise ValueError("Completed campaigns do not need interrupted-job recovery")
    resources = []
    for journal in path.glob("block-*/replica-*/resource.json"):
        if journal.is_symlink():
            raise ValueError("Resource journal must not be a symlink")
        resource = read_json(journal)
        if resource.get("owner") != path.name or not re.fullmatch(
            r"capsim-infra-[a-f0-9]{32}", resource.get("name", "")
        ):
            raise ValueError("Invalid resource ownership record")
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", resource["name"]],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            if "no such" not in result.stderr.lower():
                raise RuntimeError("Cannot inspect recorded container ownership")
        else:
            labels = json.loads(result.stdout)
            if labels.get("capsim.infrastructure") != path.name:
                raise ValueError(
                    "Container label does not match this campaign; refusing cleanup"
                )
            engine = OwnedVllm(EngineConfig(), threading.Event())
            engine._container_id = resource["name"]
            engine.shutdown()
        resources.append({"name": resource["name"], "outcome": "verified_absent"})
    for journal in path.glob("block-*/partition.json"):
        if journal.is_symlink():
            raise ValueError("Partition journal must not be a symlink")
        from .partition import Partition

        partition = Partition(journal, path.name, None)
        partition.restore()
        resources.append({"partition": str(journal.relative_to(path)), "outcome": "restored"})
    record = {
        "time": time.time(),
        "resources": resources,
        "note": "Interrupted resource recovery is not a completed benchmark",
    }
    write_json(path / ("recovery-" + str(time.time_ns()) + ".json"), record)
    state.update(outcome="incomplete", current=None, recovery=record, updated_at=time.time())
    if mark_interrupted:
        state["phase"] = "interrupted"
    write_json(path / "state.json", state)
    return record
