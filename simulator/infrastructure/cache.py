"""P03 cache experiments: explicit dtype/backend controls and bounded shape grids.

A passing highest point is a tested lower bound, never an observed capacity
boundary. BF16 cache is a task-quality reference with identical model weights.
"""

import copy


def cache_config(configuration, spec, arm):
    from .worker import arm_config

    cfg = arm_config(configuration, spec, "baseline")
    engine = cfg["engine"]
    flags = engine["vllm_extra_flags"]
    managed = {"--kv-cache-dtype", "--attention-backend", "--calculate-kv-scales", "--dtype"}
    cleaned = []
    i = 0
    while i < len(flags):
        key = flags[i].split("=", 1)[0]
        if key in managed:
            if key != "--calculate-kv-scales" and "=" not in flags[i]:
                if i + 1 >= len(flags) or flags[i + 1].startswith("--"):
                    raise ValueError("Malformed cache override: " + key)
                i += 1
        else:
            cleaned.append(flags[i])
        i += 1
    dtype = {"reference": "bfloat16", "baseline": "fp8", "treatment": "int4_per_token_head"}[arm]
    engine["kv_cache_dtype"] = dtype
    engine["attention_backend"] = "TRITON_ATTN"
    # Explicit flags are authoritative for the current vLLM CUDA launcher.
    engine["vllm_extra_flags"] = cleaned + [
        "--kv-cache-dtype",
        dtype,
        "--attention-backend",
        "TRITON_ATTN",
        "--dtype",
        "bfloat16",
    ]
    return cfg


def cells(spec, arm=None):
    loads = [min(spec.cache_concurrencies)] if arm == "reference" else spec.cache_concurrencies
    return [(records, concurrency) for records in spec.cache_record_counts for concurrency in loads]


def compare_cache(results, spec):
    expected = {
        (block, arm, records, concurrency)
        for block in range(spec.paired_blocks)
        for arm in ("reference", "baseline", "treatment")
        for records, concurrency in cells(spec, arm)
    }
    actual = {}
    for result in results:
        key = (result["block"], result["arm"], result["document_records"], result["concurrency"])
        if key in actual:
            raise ValueError("Duplicate cache measurement cell")
        actual[key] = copy.deepcopy(result["metrics"])
    complete = set(actual) == expected
    reference_by_size = {
        records: complete
        and all(
            (m := actual.get((b, "reference", records, min(spec.cache_concurrencies))))
            and m.get("attempted", 0) >= spec.min_requests
            and m.get("quality_fraction", 0) >= spec.min_quality
            and m.get("error_fraction", 1) <= spec.max_error_fraction
            for b in range(spec.paired_blocks)
        )
        for records in spec.cache_record_counts
    }
    references_passed = all(reference_by_size.values())
    points = []
    for arm in ("baseline", "treatment"):
        for records, concurrency in cells(spec):
            group = [actual.get((b, arm, records, concurrency)) for b in range(spec.paired_blocks)]
            passed = reference_by_size[records] and all(m and m["gates_passed"] for m in group)
            tokens = [
                m["prompt_tokens_min"]
                for m in group
                if m and m.get("prompt_tokens_min") is not None
            ]
            points.append(
                {
                    "arm": arm,
                    "document_records": records,
                    "concurrency": concurrency,
                    "passed": bool(passed and len(tokens) == len(group)),
                    "minimum_measured_input_tokens": min(tokens) if tokens else None,
                }
            )
    boundaries = {}
    for arm in ("baseline", "treatment"):
        acceptable = [p for p in points if p["arm"] == arm and p["passed"]]
        lengths = [
            p["document_records"]
            for p in acceptable
            if p["concurrency"] == min(spec.cache_concurrencies)
        ]
        concurrencies = [
            p["concurrency"]
            for p in acceptable
            if p["document_records"] == spec.cache_record_counts[-1]
        ]
        boundaries[arm] = {
            "largest_tested_qualified_document_records": max(lengths) if lengths else None,
            "largest_tested_qualified_concurrency": max(concurrencies) if concurrencies else None,
        }
    metrics = {}
    for name in (
        "largest_tested_qualified_document_records",
        "largest_tested_qualified_concurrency",
    ):
        a, b = (boundaries[arm][name] for arm in ("baseline", "treatment"))
        if a and b:
            metrics[name] = {
                "baseline_mean": a,
                "treatment_mean": b,
                "change_percent": 100 * (b - a) / a,
                "paired_bootstrap_95_interval": None,
                "unit": "document records"
                if name.endswith("document_records")
                else "concurrent requests",
                "method": "Largest passing tested grid point; not an established capacity maximum",
            }
    return {
        "qualified": False,
        "reason": "Exploratory cache grid; capacity boundaries and broad quality are not established",
        "complete": complete,
        "reference_quality_passed": references_passed,
        "reference_quality_by_document_records": reference_by_size,
        "scope": "Same-backend FP8 versus INT4 cache with a BF16 cache task-quality reference",
        "quality_scope": "Exact synthetic record extraction only",
        "points": points,
        "boundaries": boundaries,
        "metrics": metrics,
    }


def observe_runtime(log, arm):
    """Require runtime evidence of the selected backend and precision, not just argv."""
    import re

    expected = {"reference": "bfloat16", "baseline": "fp8", "treatment": "int4_per_token_head"}[arm]
    backends = set(re.findall(r"Using AttentionBackendEnum\.([A-Z0-9_]+) backend", log))
    dtypes = set(re.findall(r"kv_cache_dtype=([a-z0-9_]+)", log))
    model_dtypes = set(re.findall(r"(?:^|[, ])dtype=torch\.([a-z0-9_]+)", log))
    if backends != {"TRITON_ATTN"} or dtypes != {expected} or model_dtypes != {"bfloat16"}:
        raise ValueError(
            f"Cache runtime selection is unverified: backend={sorted(backends)}, cache={sorted(dtypes)}, model={sorted(model_dtypes)}"
        )
    capacities = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", log)
    return {
        "backend": "TRITON_ATTN",
        "cache_dtype": expected,
        "model_dtype": "bfloat16",
        "reported_cache_token_capacity": int(capacities[-1].replace(",", ""))
        if capacities
        else None,
        "capacity_scope": "Engine-reported allocation; not measured usable context or concurrency",
        "evidence_source": "Owned engine startup log",
    }
