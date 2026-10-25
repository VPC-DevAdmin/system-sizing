"""Evidence-only wrapper for the pinned vLLM SimpleCPUOffloadConnector.

Observe CUDA-confirmed event watermarks. Never change scheduling or cache policy.
Private worker fields are checked explicitly because this adapter is image-pinned.
"""
import json
import os
import threading
import time
import uuid
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)


class ObservedCPUOffloadConnector(SimpleCPUOffloadConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evidence_pending = {"load": {}, "store": {}}
        self._evidence_totals = {"load_blocks": 0, "store_blocks": 0}
        self._evidence_lock = threading.Lock()
        if self.worker_handler is not None:
            for name in ("_load_hwm", "_store_hwm"):
                if not hasattr(self.worker_handler, name):
                    raise RuntimeError("Pinned native CPU telemetry contract changed")
            self._write_evidence()

    def bind_connector_metadata(self, metadata):
        super().bind_connector_metadata(metadata)
        if self.worker_handler is not None:
            for kind, blocks in (("load", metadata.load_cpu_blocks),
                                 ("store", metadata.store_gpu_blocks)):
                event = getattr(metadata, kind + "_event")
                if event >= 0 and blocks:
                    self._evidence_pending[kind].setdefault(event, len(blocks))

    def get_finished(self, finished_req_ids):
        result = super().get_finished(finished_req_ids)
        if self.worker_handler is not None:
            changed = False
            for kind, pending in self._evidence_pending.items():
                watermark = getattr(self.worker_handler, "_" + kind + "_hwm")
                for event in list(pending):
                    if event <= watermark:
                        self._evidence_totals[kind + "_blocks"] += pending.pop(event)
                        changed = True
            if changed:
                self._write_evidence()
        return result

    def _write_evidence(self):
        # Atomic counters only on completions; no GPU synchronization or extra polling.
        path = Path(os.environ.get("CAPSIM_CPU_EVIDENCE_DIR", "/evidence"))
        target = path / f"native-cpu-{os.getpid()}.json"
        record = dict(self._evidence_totals, pid=os.getpid(), timestamp=time.time(),
                      scope="Completed physical cache block copies; not distinct documents",
                      source="Pinned worker CUDA event completion watermarks")
        # Unique temporary per write and a lock: concurrent completions must never race
        # on one temp name, and an evidence write failure must never stop the engine.
        temporary = path / f".native-cpu-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        with self._evidence_lock:
            try:
                temporary.write_text(json.dumps(record))
                temporary.replace(target)
            except OSError as exc:
                print(f"capsim native CPU evidence write failed: {exc}", flush=True)
                try:
                    temporary.unlink()
                except OSError:
                    pass
