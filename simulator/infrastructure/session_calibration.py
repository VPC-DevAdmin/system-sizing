"""Validate an empirical GPU-cache working-set boundary, never bytes-to-users."""

import hashlib
import json
from pathlib import Path


def load_calibration(spec, assets, configuration):
    if not spec.session_capacity_multiples:
        return None
    path = Path(spec.session_capacity_evidence)
    raw = path.read_bytes()
    record = json.loads(raw)
    expected = dict(
        assets=assets,
        gpu_ids=spec.gpu_ids,
        context_limit=spec.context_limit,
        document_records=spec.document_records,
        configuration_sha256=hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
    )
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "GPU cache calibration does not match current assets, allocation or fixture"
        )
    points = record.get("points", [])
    capacity = spec.session_gpu_capacity_documents
    for block in range(2):
        pair = [p for p in points if p.get("block") == block]
        hot = [p for p in pair if p.get("population") == capacity]
        pressure = [p for p in pair if p.get("population", 0) > capacity]
        if not hot or not pressure:
            raise ValueError("Calibration needs repeated hot and pressure observations")
        for point in hot + pressure:
            if (
                point.get("arm") != "baseline"
                or point.get("quality_fraction", 0) < spec.min_quality
                or point.get("measured_returns_per_document", 0) < spec.session_min_returns
                or not point.get("evidence_sha256")
                or len(point["evidence_sha256"]) != 64
                or type(point.get("repeat_gpu_hit_fraction")) not in (int, float)
                or not 0 <= point["repeat_gpu_hit_fraction"] <= 1
            ):
                raise ValueError("Incomplete GPU-cache calibration evidence")
        if not all(p["repeat_gpu_hit_fraction"] >= 0.95 for p in hot) or not any(
            p["repeat_gpu_hit_fraction"] < 0.90 for p in pressure
        ):
            raise ValueError("GPU cache hot/pressure boundary is unresolved")
    return dict(
        sha256=hashlib.sha256(raw).hexdigest(),
        record=record,
        scope="Empirical reusable-prefix boundary for this fixture and allocation, not physical resident-document count",
    )
