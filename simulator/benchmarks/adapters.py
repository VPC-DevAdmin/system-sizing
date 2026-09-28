"""CLI and schema contracts verified against the pinned upstream releases."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .spec import VERSIONS


class ArtifactError(ValueError):
    pass


def command(
    harness: str,
    binary: str,
    spec: dict,
    endpoint: str,
    model: str,
    tokenizer: str,
    dataset: Path,
    out: Path,
    point: float,
    duration: int,
) -> list[str]:
    extra = {"ignore_eos": spec["ignore_eos"], "temperature": 0}
    if spec.get("_reasoning_effort"):
        extra["reasoning_effort"] = spec["_reasoning_effort"]
    if harness == "aiperf":
        args = [
            binary,
            "profile",
            "--model",
            model,
            "--url",
            endpoint,
            "--endpoint-type",
            "chat",
            "--streaming",
            "--tokenizer",
            tokenizer,
            "--input-file",
            str(dataset),
            "--output-tokens-mean",
            str(spec["output_tokens"]),
            "--benchmark-duration",
            str(duration),
            "--benchmark-grace-period",
            str(spec["timeout_grace_s"]),
            "--random-seed",
            str(spec["seed"]),
            "--artifact-dir",
            str(out),
            "--ui",
            "none",
        ]
        for key, val in extra.items():
            args += ["--extra-inputs", f"{key}:{json.dumps(val)}"]
        args += (
            ["--request-rate", str(point), "--arrival-pattern", "poisson"]
            if spec["recipe"] == "slo"
            else ["--concurrency", str(int(point))]
        )
        return args
    if harness != "guidellm":
        raise ValueError("Unknown harness")
    profile = (
        {"kind": "poisson", "rate": point}
        if spec["recipe"] == "slo"
        else {"kind": "concurrent", "streams": int(point)}
    )
    return [
        str(Path(binary).with_name("python")),
        spec.get("_guidellm_entry", str(Path(__file__).with_name("guidellm_entry.py"))),
        "run",
        "--backend",
        json.dumps(
            {
                "kind": "openai_http",
                "target": endpoint,
                "model": model,
                "request_format": "/v1/chat/completions",
                "extras": {"body": extra},
            }
        ),
        "--profile",
        json.dumps(profile),
        "--constraint",
        f"kind=max_duration,seconds={duration}",
        "--tokenizer",
        json.dumps(
            {
                "kind": "huggingface_auto",
                "model": tokenizer,
                "load_kwargs": {"local_files_only": True},
            }
        ),
        "--data",
        json.dumps({"kind": "capsim_corpus_cycle", "path": str(dataset)}),
        "--data-column-mapper",
        json.dumps(
            {
                "kind": "generative_column_mapper",
                "column_mappings": {
                    "text_column": "text",
                    "prompt_tokens_count_column": "input_tokens_count",
                    "output_tokens_count_column": "output_tokens_count",
                },
            }
        ),
        "--data-loader",
        "kind=pytorch,num_workers=0",
        "--seed",
        f"kind=static,value={spec['seed']}",
        "--output",
        json.dumps({"kind": "json", "path": str(out / "benchmarks.json")}),
    ]


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArtifactError(f"Missing or nonfinite metric: {name}")
    if value < 0:
        raise ArtifactError(f"Negative metric: {name}")
    return value


def parse(harness: str, out: Path) -> dict:
    if harness == "aiperf":
        files = list(out.glob("*aiperf.json"))
        if len(files) != 1:
            raise ArtifactError("Expected one AIPerf summary artifact")
        doc = json.loads(files[0].read_text())
        if doc.get("schema_version") != "1.4" or doc.get("aiperf_version") != VERSIONS[harness]:
            raise ArtifactError("Unsupported AIPerf version or artifact schema")

        def metric(key, stat="avg", optional=False):
            val = doc.get(key)
            if val is None and optional:
                return None
            val = val or {}
            units = {
                "output_token_throughput": {"tokens/sec"},
                "request_throughput": {"requests/sec"},
                "output_token_count": {"tokens"},
                "input_sequence_length": {"tokens"},
                "request_count": {"requests"},
                "error_request_count": {"requests"},
                "time_to_first_token": {"ms"},
                "inter_token_latency": {"ms"},
                "request_latency": {"ms"},
                "benchmark_duration": {"sec"},
            }
            if key in units and val.get("unit") not in units[key]:
                raise ArtifactError(f"Unexpected unit for {key}: {val.get('unit')}")
            return number(val.get(stat), key)

        success = metric("request_count")
        errors = metric("error_request_count", optional=True)
        # AIPerf omits error_request_count on a clean run.
        errors = 0 if errors is None else errors
        result = {
            "output_tok_s": metric("output_token_throughput"),
            "requests_s": metric("request_throughput"),
            "successful": success,
            "output_tokens_mean": metric("output_token_count"),
            "input_tokens_mean": metric("input_sequence_length"),
            "failed": errors,
            "incomplete": None,
            "ttft_p95_ms": metric("time_to_first_token", "p95"),
            "itl_p95_ms": metric("inter_token_latency", "p95", True),
            "tpot_p95_ms": None,
            "latency_p95_ms": metric("request_latency", "p95"),
            "client_delay_ms": None,
            "duration_s": metric("benchmark_duration"),
        }
    else:
        files = [out / "benchmarks.json"]
        doc = json.loads(files[0].read_text())
        meta = doc.get("metadata", {})
        if meta.get("version") != 2 or meta.get("guidellm_version") != VERSIONS[harness]:
            raise ArtifactError("Unsupported GuideLLM version or artifact schema")
        rows = doc.get("benchmarks", [])
        if len(rows) != 1:
            raise ArtifactError("Expected one GuideLLM load point per artifact")
        row = rows[0]
        m = row["metrics"]

        def metric(key, percentile=False, status="successful"):
            stat = m[key][status]
            return number(stat["percentiles"]["p95"] if percentile else stat["mean"], key)

        totals = m["request_totals"]
        incomplete = number(totals["incomplete"], "incomplete")
        cutoff = number(row["end_time"], "measurement end")
        stopped = row.get("scheduler_state", {}).get("end_processing_constraints", {})
        boundary = stopped.get("max_duration", {}).get("metadata", {})
        censored = 0
        # Only attribute cancellation to the duration boundary when native records
        # prove both the scheduler stop reason and each request's timing/status.
        records = row.get("requests", {}).get("incomplete", [])
        window_start = number(row["start_time"], "measurement start")
        # GuideLLM retains some requests outside the metrics window. Match
        # GenerativeRequestsAccumulator.get_within_range(closed=False), which
        # includes partial overlap and treats a missing start as an end event.
        in_window = []
        for request in records:
            end = request.get("request_end_time")
            start = request.get("request_start_time")
            start = end if start is None else start
            if (
                isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and math.isfinite(start)
                and math.isfinite(end)
                and end >= window_start
                and start <= cutoff
            ):
                in_window.append(request)
        if (
            set(stopped) == {"max_duration"}
            and boundary.get("duration_exceeded") is True
            and isinstance(boundary.get("stop_time"), (int, float))
            and abs(boundary["stop_time"] - cutoff) < 0.01
            and len(in_window) == incomplete
        ):
            for request in in_window:
                info = request.get("info", {})
                ended = info.get("timings", {}).get("resolve_end")
                if (
                    info.get("status") == "cancelled"
                    and isinstance(ended, (int, float))
                    and math.isfinite(ended)
                    and ended >= cutoff
                ):
                    censored += 1
        result = {
            "output_tok_s": metric("output_tokens_per_second", status="total"),
            "completed_output_tok_s": metric("output_tokens_per_second"),
            "throughput_scope": "GuideLLM total tokens inside the measurement window, including partial boundary requests",
            "requests_s": metric("requests_per_second"),
            "output_tokens_mean": metric("output_token_count"),
            "input_tokens_mean": metric("prompt_token_count"),
            "successful": number(totals["successful"], "successful"),
            "failed": number(totals["errored"], "errored"),
            "incomplete": incomplete,
            "boundary_cancelled": censored,
            "ttft_p95_ms": metric("time_to_first_token_ms", True),
            "tpot_p95_ms": metric("time_per_output_token_ms", True),
            "itl_p95_ms": metric("inter_token_latency_ms", True),
            "latency_p95_ms": metric("request_latency", True) * 1000,
            "client_delay_ms": number(
                row["scheduler_metrics"]["request_start_delay_avg"], "client start delay"
            )
            * 1000,
            "targeted_start_delay_ms": number(
                row["scheduler_metrics"]["request_targeted_start_delay_avg"], "targeted start delay"
            )
            * 1000,
            "duration_s": number(row["end_time"] - row["start_time"], "duration"),
        }
    if result["successful"] <= 0 or result["duration_s"] <= 0:
        raise ArtifactError("No successful measured requests or positive duration")
    result["source"] = files[0].name
    unexpected = (result["incomplete"] or 0) - result.get("boundary_cancelled", 0)
    result["error_fraction"] = (result["failed"] + unexpected) / (
        result["successful"] + result["failed"] + unexpected
    )
    result["boundary_cancelled_fraction"] = result.get("boundary_cancelled", 0) / (
        result["successful"] + result["failed"] + (result["incomplete"] or 0)
    )
    return result


def verdict(metrics: dict, spec: dict, proxy: dict) -> tuple[str, list[str]]:
    reasons = []
    if metrics["successful"] < spec["min_requests"]:
        reasons.append("Insufficient successful requests for configured evidence threshold")
    if metrics["duration_s"] < spec["duration_s"] * 0.8:
        reasons.append("Measured window shorter than the requested duration")
    # A single startup/scheduling pause is not evidence of sustained client saturation.
    # Use the worst worker's sampled p95 boundary; retain maximum delay in evidence.
    if proxy.get("loop_lag_samples"):
        routing_saturated = (
            proxy.get(
                "max_worker_lag_over_100ms_fraction",
                proxy["loop_lag_over_100ms"] / proxy["loop_lag_samples"],
            )
            > 0.05
        )
    else:
        routing_saturated = proxy.get("max_loop_lag_ms", 0) > 100
    delay_key = "targeted_start_delay_ms" if spec["recipe"] == "slo" else "client_delay_ms"
    if routing_saturated or (metrics.get(delay_key) or 0) > 100:
        reasons.append("Load generator or routing event loop may be saturated")
    if (
        spec["recipe"] == "slo"
        and metrics.get("boundary_cancelled_fraction", 0) > spec["max_error_fraction"]
    ):
        reasons.append("Too many duration-boundary cancellations to establish the latency SLO")
    if any(v == 0 for v in proxy.get("routed", {}).values()):
        reasons.append("At least one intended replica received no requests")
    if proxy.get("active", 0):
        reasons.append("Requests remained in flight when the harness exited")
    for key in ("ttft_p95_ms", "tpot_p95_ms"):
        if spec.get(key) is not None and metrics.get(key) is None:
            reasons.append(f"Harness does not provide required {key}; no metric substitution")
    if (
        spec.get("ignore_eos")
        and metrics.get("output_tokens_mean") is not None
        and abs(metrics["output_tokens_mean"] - spec["output_tokens"])
        > max(1, spec["output_tokens"] * 0.01)
    ):
        reasons.append("Observed mean output length differs from the requested fixed length")
    if spec["recipe"] == "slo" and spec.get("_load_point"):
        expected = spec["_load_point"] * spec["duration_s"]
        received = sum(proxy.get("routed", {}).values())
        if abs(received - expected) > max(expected * 0.2, 3 * math.sqrt(expected)):
            reasons.append(
                "Achieved request arrivals differ materially from the offered Poisson rate"
            )
    if reasons:
        return "inconclusive", reasons
    failures = []
    if metrics["error_fraction"] > spec["max_error_fraction"]:
        failures.append("Error fraction exceeds the configured limit")
    for key in ("ttft_p95_ms", "tpot_p95_ms"):
        if spec.get(key) is not None and metrics[key] > spec[key]:
            failures.append(f"{key} exceeds the configured limit")
    return ("failed_requirements", failures) if failures else ("measured", [])
