"""Export roofline and placement measurements for a sizing tool.

One document per measured configuration, in the sizing tool's shape:
``meta`` (models, engines, engine_config, system) plus ``cohorts`` whose
``curve`` is the concurrency ladder the sweep climbed. The shape is
kept field for field; what the tool's shape has no slot for is ADDED
beside it, never folded into an existing field with a different
meaning:

* Saturation, not SLA. The headline sweep holds streams in flight with
  zero think time and EOS ignored, and enforces no latency target, so
  ``target_status`` is "not_evaluated" and ``target_miss_rate`` null.
  ``status`` is pass when the rung served (>= 90% of finished requests
  answered, the roofline's publishing bar) and the engine held what was
  offered; ``violation_rate`` is the rung's failed-request share.
* Generated vs visible tokens. With EOS ignored every request generates
  exactly ``output_tokens``; for reasoning models part of that is
  chain-of-thought the user does not see. ``generated_tok_per_s`` is
  what was measured; ``visible_output_tok_per_s`` equals it and
  ``meta.output_accounting`` says whether reasoning is inside it.
* Failed configurations are exported too (``final_status: failed``,
  the cause in ``error``, an empty curve): a blank with its reason is a
  sizing fact.

Run on the host that holds the runs:
``python -m simulator.export_sizing --out exports/xe7740``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.1"
MIN_SUCCESS = 0.9

REASONING_MODELS = ("Kimi-K2-Thinking", "gpt-oss", "DeepSeek-V3.1", "DeepSeek-V3.2",
                    "Qwen3-235B-A22B-Thinking", "GLM-5", "MiniMax-M2")


# ── System ────────────────────────────────────────────────────────────

def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_key(model_name: str | None) -> str | None:
    """"Intel(R) Xeon(R) 6787P" -> "Intel:6787P"; AMD EPYC likewise."""
    if not model_name:
        return None
    vendor = "Intel" if "Intel" in model_name else "AMD" if "AMD" in model_name else None
    m = re.search(r"(\d{4,5}[A-Z]{0,2})\b", model_name.replace("(R)", " "))
    if not vendor or not m:
        return None
    return f"{vendor}:{m.group(1)}"


def detect_system() -> dict:
    """What the box is, read from the box: DMI, /proc, sysfs, nvidia-smi."""
    cpu_model = None
    cpuinfo = _read("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    packages, cores = set(), set()
    for d in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
        pkg = _read(str(d / "physical_package_id"))
        core = _read(str(d / "core_id"))
        if pkg is not None:
            packages.add(pkg)
            if core is not None:
                cores.add((pkg, core))
    mem_kb = 0
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            mem_kb = int(line.split()[1])
    numa = sorted(p.name for p in Path("/sys/devices/system/node").glob("node[0-9]*"))
    accel: list[dict] = []
    nvlink = False
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,pcie.link.gen.max,"
                            "pcie.link.width.max", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        rows = [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line]
        by_name: dict[str, dict] = {}
        for name, mem, gen, width in rows:
            e = by_name.setdefault(name, {"vendor": "NVIDIA", "model": name.replace("NVIDIA ", ""),
                                          "count": 0,
                                          "memoryGbEach": round(float(mem) / 1024),
                                          "pcieGen": int(gen), "pcieWidth": int(width)})
            e["count"] += 1
        accel = list(by_name.values())
        nv = subprocess.run(["nvidia-smi", "nvlink", "-s"], capture_output=True,
                            text=True, timeout=20)
        nvlink = "Link" in nv.stdout
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    groups = None
    try:
        from . import arena
        groups = arena.hardware().get("device_groups")
    except Exception:  # noqa: BLE001
        pass
    return {
        "vendor": (_read("/sys/class/dmi/id/sys_vendor") or "").replace(" Inc.", "") or None,
        "platform": _read("/sys/class/dmi/id/product_name"),
        "cpuKey": cpu_key(cpu_model),
        "cpuModel": cpu_model,
        "sockets": len(packages) or None,
        "coresPerSocket": (len(cores) // len(packages)) if packages else None,
        "numaNodes": len(numa) or None,
        "memoryGb": round(mem_kb / 1024 / 1024) if mem_kb else None,
        "accelerators": accel,
        "interconnect": {"nvlink": nvlink,
                         "gpuDomains": groups,
                         "note": "GPUs in a domain share one socket's PCIe root complex; "
                                 "traffic between domains crosses the inter-socket link"},
    }


# ── Rows -> documents ─────────────────────────────────────────────────

def quantization_kind(row: dict, info: dict) -> str | None:
    if row.get("engine") == "llamacpp" or (row.get("engine") == "ktransformers"
                                           and not row.get("kt_native")
                                           and not info.get("kt_native")):
        return "gguf"
    q = info.get("quant")
    if q:
        return str(q).lower()
    name = row.get("model", "").lower()
    for k in ("nvfp4", "mxfp4", "fp8", "int4", "awq", "gptq"):
        if k in name:
            return k
    return None


def rung_served(r: dict) -> bool:
    done = (r.get("samples") or 0) + (r.get("errors") or 0) + (r.get("no_content") or 0)
    return done == 0 or ((r.get("samples") or 0) + (r.get("no_content") or 0)) / done >= MIN_SUCCESS


def rate_confidence(r: dict) -> str:
    """How far a rung's rate can be trusted.

    SGLang and llama.cpp count a request's tokens when it FINISHES; a
    window that saw fewer than two completions per stream read whole
    waves of finishing requests (llama.cpp giants: 67.5 / 135 / 270 on
    one engine; Kimi-K2 NVFP4 on SGLang: 4,121 by counter, 3,270 by
    the engine's own gauge). Rungs measured by the gauge or with ample
    completions are "measured"; the rest are flagged."""
    src = r.get("rate_source") or "counter"
    if src in ("engine_gauge", "client_stream"):
        return "measured"
    inflight = r.get("in_flight") or 0
    done = (r.get("samples") or 0) + (r.get("errors") or 0) + (r.get("no_content") or 0)
    if inflight and done < 2 * inflight:
        return "wave_quantized_suspect"
    return "measured"


def tpot_implied(r: dict) -> float | None:
    """Streams in flight / median time per token: the decode rate the
    per-token timings imply (prefill stalls excluded, so a ceiling)."""
    inflight, tpot = r.get("in_flight"), r.get("tpot_p50_ms")
    if not inflight or not tpot:
        return None
    return round(float(inflight) / (float(tpot) / 1000.0), 1)


def curve_point(r: dict) -> dict:
    samples = r.get("samples") or 0
    errors = r.get("errors") or 0
    no_content = r.get("no_content") or 0
    done = samples + errors + no_content
    held = r.get("held", True)
    served = rung_served(r)
    out = r.get("out_tok_s")
    power = r.get("gpu_power_w")
    return {
        "pool_size": r.get("concurrency"),
        "sample_size": samples,
        "status": "pass" if (served and held) else "fail",
        "target_status": "not_evaluated",
        "violation_rate": round(errors / done, 4) if done else 0.0,
        "target_miss_rate": None,
        "ttft_p50_ms": r.get("ttft_p50_ms"),
        "ttft_p95_ms": r.get("ttft_p95_ms"),
        "tpot_p50_ms": r.get("tpot_p50_ms"),
        "tpot_p95_ms": r.get("tpot_p95_ms"),
        "prompt_tok_per_s": r.get("prompt_tok_s"),
        "visible_output_tok_per_s": out,
        "kv_cache_used_pct": r.get("kv_cache_pct"),
        "measurement_duration_s": r.get("measure_s"),
        # additions
        "generated_tok_per_s": out,
        "total_tok_per_s": r.get("total_tok_s"),
        "in_flight": r.get("in_flight"),
        "queue_depth": r.get("queue_depth"),
        "errors": errors,
        "reasoning_only": no_content,
        "success_rate": round((samples + no_content) / done, 4) if done else None,
        "steady_state": r.get("steady_state"),
        "engine_held_offered": held,
        "rate_source": r.get("rate_source") or "counter",
        "rate_confidence": rate_confidence(r),
        "tpot_implied_tok_per_s": tpot_implied(r),
        "gpu_power_w": power,
        "tokens_per_watt": round(out / power, 3) if out and power else None,
    }


def pool_sizes(curve: list[dict], peak_conc: int | None) -> tuple:
    passing = [c["pool_size"] for c in curve if c["status"] == "pass" and c["pool_size"]]
    target = peak_conc
    soft = max(passing) if passing else None
    fail = min((c["pool_size"] for c in curve if c["status"] == "fail"
                and c["pool_size"] and (target is None or c["pool_size"] > target)), default=None)
    return target, soft, fail


def row_document(row: dict, *, info: dict, sweep: dict | None, system: dict,
                 generated_at: str) -> dict:
    tp = int(row.get("tp") or 1)
    replicas = int(row.get("replicas") or 1)
    reasoning = any(k.lower() in row["model"].lower() for k in REASONING_MODELS)
    engine_config = {
        "quantization_kind": quantization_kind(row, info),
        "tensor_parallel_size": tp,
        "max_model_len": row.get("max_model_len"),
        # additions
        "replicas": replicas,
        "max_num_seqs": row.get("max_num_seqs"),
        "gpu_memory_utilization": row.get("gpu_memory_utilization"),
        "kv_cache_dtype": row.get("kv_cache_dtype"),
        "kv_cache_tokens": row.get("kv_cache_tokens") or (sweep or {}).get("kv_cache_tokens"),
        "placement": row.get("placement") or ("span" if tp > 4 else "pack"),
    }
    for k in ("ktransformers_gpu_experts", "kt_native", "escalated_from",
              "escalated_from_share"):
        if row.get(k) not in (None, ""):
            engine_config[k] = row[k]
    rungs = (sweep or {}).get("rungs") or []
    curve = [curve_point(r) for r in rungs]
    peak = (sweep or {}).get("peak") or {}
    failed = bool(row.get("error"))
    below_gate = False
    if rungs and not failed:
        # Capacity is the best rung that SERVED. Rows measured before
        # the roofline's 90% gate could carry a peak where most requests
        # failed (gpt-oss-20b at 39%); a sizing tool must not read that
        # as capacity.
        passing = [r for r, c in zip(rungs, curve, strict=True) if c["status"] == "pass"]
        if passing:
            settled = [r for r in passing if r.get("steady_state") is not False]
            peak = max(settled or passing, key=lambda r: r.get("out_tok_s") or 0)
        else:
            below_gate = True
    target, soft, fail = pool_sizes(curve, peak.get("concurrency") or row.get("concurrency"))
    success = None
    if peak and rungs:
        pt = curve_point(peak)
        success = pt["success_rate"]
    shape = (sweep or {}).get("shape") or {}
    cohort = {
        "id": (sweep or {}).get("cohort_id") or "headline_generation",
        "category": "persona",
        "final_status": ("failed" if failed else "no_passing_rung" if below_gate else "ok"),
        "target_capacity_pool_size": None if (failed or below_gate) else target,
        "soft_capacity_pool_size": None if (failed or below_gate) else soft,
        "fail_pool_size": fail,
        "capacity_throughput": None if (failed or below_gate) else {
            "pool_size": peak.get("concurrency") or row.get("concurrency"),
            "sample_size": peak.get("samples", row.get("samples")),
            "measurement_duration_s": peak.get("measure_s"),
            "prompt_tok_per_s": peak.get("prompt_tok_s"),
            "visible_output_tok_per_s": peak.get("out_tok_s", row.get("out_tok_s")),
            # additions
            "generated_tok_per_s": peak.get("out_tok_s", row.get("out_tok_s")),
            "in_flight": peak.get("in_flight", row.get("in_flight")),
            "ttft_p95_ms": peak.get("ttft_p95_ms", row.get("ttft_p95_ms")),
            "tpot_p95_ms": peak.get("tpot_p95_ms", row.get("tpot_p95_ms")),
            "success_rate": success if success is not None else row.get("success_rate"),
            "steady_state": peak.get("steady_state", row.get("steady_state")),
            "gpu_power_w": peak.get("gpu_power_w", row.get("gpu_power_w")),
            "rate_confidence": rate_confidence(peak) if peak else None,
            "tpot_implied_tok_per_s": tpot_implied(peak) if peak else None,
            "tokens_per_watt": (round(peak["out_tok_s"] / peak["gpu_power_w"], 3)
                                if peak.get("out_tok_s") and peak.get("gpu_power_w")
                                else row.get("tokens_per_watt")),
        },
        "curve": curve,
        # additions
        "cohort_name": (sweep or {}).get("cohort_name"),
        "stop_reason": (sweep or {}).get("stop_reason"),
        "error": row.get("error"),
    }
    gpus_used = tp * replicas
    return {
        "meta": {
            "generated_at": generated_at,
            "models": [row["model"]],
            "engines": [row["engine"]],
            "source_dir": row.get("run_dir"),
            "engine_config": engine_config,
            "system": {**system, "topology": {"gpusUsed": gpus_used, "replicas": replicas,
                                              "tensorParallelSize": tp,
                                              "placement": engine_config["placement"],
                                              "cpuExperts": row.get("engine") == "ktransformers"}},
            # additions
            "schema_version": SCHEMA_VERSION,
            "run_kind": "roofline_headline",
            "phase": "confirmation" if row.get("confirmed") else "search",
            "workload": {"input_tokens": row.get("input_tokens") or shape.get("input_tokens"),
                         "output_tokens": row.get("output_tokens") or shape.get("output_tokens"),
                         "ignore_eos": shape.get("ignore_eos", True),
                         "think_time_s": 0,
                         "sla_enforced": False},
            "output_accounting": ("generated tokens include reasoning (chain-of-thought)"
                                  if reasoning else "generated tokens are visible output"),
            "model_info": {k: info.get(k) for k in ("vendor", "series", "params_b",
                                                    "approx_size_gb", "quant", "tier",
                                                    "fits_gpu")},
        },
        "cohorts": [cohort],
    }


def placement_documents(results: dict, *, system: dict, generated_at: str) -> list[dict]:
    """The KTransformers hot/cold placement windows: one document per
    configuration, one cohort per prompt mix, the curve over concurrency."""
    by_cfg: dict[str, list[dict]] = {}
    for w in results.get("windows") or []:
        by_cfg.setdefault(w["config"], []).append(w)
    plan_cfgs = {c["name"]: c for c in (results.get("plan") or {}).get("configs", [])}
    docs = []
    for name, windows in by_cfg.items():
        custom = (plan_cfgs.get(name) or {}).get("custom") or {}
        tp = int(windows[0].get("tp") or custom.get("tp") or 1)
        replicas = int(windows[0].get("replicas") or custom.get("replicas") or 1)
        cohorts = []
        for mix in dict.fromkeys(w["mix"] for w in windows):
            pts = []
            for w in sorted((x for x in windows if x["mix"] == mix), key=lambda x: x["concurrency"]):
                eng = w.get("engine") or {}
                ex = w.get("experts") or {}
                pts.append({
                    "pool_size": w["concurrency"], "sample_size": w.get("succeeded"),
                    "status": "pass" if (w.get("success_rate") or 0) >= MIN_SUCCESS else "fail",
                    "target_status": "not_evaluated",
                    "violation_rate": round(1 - w["success_rate"], 4) if w.get("success_rate") is not None else None,
                    "target_miss_rate": None,
                    "ttft_p50_ms": w.get("ttft_p50_ms"), "ttft_p95_ms": w.get("ttft_p95_ms"),
                    "tpot_p50_ms": w.get("tpot_p50_ms"), "tpot_p95_ms": w.get("tpot_p95_ms"),
                    "prompt_tok_per_s": None,
                    "visible_output_tok_per_s": w.get("stream_tok_s"),
                    "kv_cache_used_pct": None,
                    "measurement_duration_s": w.get("measure_s"),
                    "generated_tok_per_s": w.get("stream_tok_s"),
                    "engine_gauge_tok_per_s": eng.get("gauge_tok_s"),
                    "in_flight": eng.get("running"), "queue_depth": eng.get("queue"),
                    "success_rate": w.get("success_rate"),
                    "gpu_power_w": eng.get("gpu_power_w"), "gpu_util_pct": eng.get("gpu_util_pct"),
                    "cpu_busy_by_numa_node": eng.get("cpu_busy_by_node"),
                    "cpu_share_of_expert_activations": ex.get("cpu_share"),
                    "answers_by_domain": w.get("answers_by_domain"),
                    "rate_source": "client_stream",
                })
            best = max(pts, key=lambda p: p["generated_tok_per_s"] or 0)
            passing = [p["pool_size"] for p in pts if p["status"] == "pass"]
            cohorts.append({
                "id": f"mix:{mix}", "category": "prompt_mix", "final_status": "ok",
                "target_capacity_pool_size": best["pool_size"],
                "soft_capacity_pool_size": max(passing) if passing else None,
                "fail_pool_size": None,
                "capacity_throughput": {
                    "pool_size": best["pool_size"], "sample_size": best["sample_size"],
                    "measurement_duration_s": best["measurement_duration_s"],
                    "prompt_tok_per_s": None,
                    "visible_output_tok_per_s": best["visible_output_tok_per_s"],
                    "generated_tok_per_s": best["generated_tok_per_s"],
                    "cpu_share_of_expert_activations": best["cpu_share_of_expert_activations"]},
                "curve": pts,
            })
        docs.append({
            "meta": {
                "generated_at": generated_at,
                "models": [custom.get("model_id") or "moonshotai/Kimi-K2-Thinking"],
                "engines": [custom.get("engine") or "ktransformers"],
                "source_dir": "runs/kt_placement",
                "engine_config": {
                    "quantization_kind": "int4" if custom.get("engine") == "ktransformers" else "nvfp4",
                    "tensor_parallel_size": tp,
                    "max_model_len": custom.get("max_model_len"),
                    "replicas": replicas, "max_num_seqs": custom.get("max_num_seqs"),
                    "gpu_memory_utilization": custom.get("gpu_memory_utilization"),
                    "ktransformers_gpu_experts": windows[0].get("gpu_experts"),
                    "ktransformers_expert_placement": custom.get("ktransformers_expert_placement"),
                    "numa_pinned_replicas": bool(custom.get("ktransformers_numa_pin")),
                },
                "system": {**system, "topology": {"gpusUsed": tp * replicas, "replicas": replicas,
                                                  "tensorParallelSize": tp,
                                                  "placement": "one replica per GPU domain and socket"
                                                  if custom.get("ktransformers_numa_pin") else "pack",
                                                  "cpuExperts": True}},
                "schema_version": SCHEMA_VERSION,
                "run_kind": "kt_expert_placement",
                "config_name": name,
                "workload": {"prompt_set": "six-domain public-dataset set, evaluation half",
                             "max_output_tokens": (results.get("plan") or {}).get("max_tokens"),
                             "ignore_eos": False, "think_time_s": 0, "sla_enforced": False},
                "output_accounting": "generated tokens include reasoning (chain-of-thought)",
                "calibration": {k: v for k, v in (results.get("calibration") or {}).items()
                                if k != "freq_path"},
            },
            "cohorts": cohorts,
        })
    return docs


def collapse_rows(rows: list[dict]) -> list[dict]:
    """The rows a sizing tool wants: every measurement, and for a
    configuration that never produced one, its latest failure. A
    failed attempt later retried or escalated into a success is noise
    -- the giants pass left 282 failures beside 157 measurements, most
    of them superseded."""
    from .roofline import cell_key
    measured = {cell_key(r) for r in rows if not r.get("error")}
    last_failure: dict[str, int] = {}
    for i, r in enumerate(rows):
        if r.get("error"):
            last_failure[cell_key(r)] = i
    return [r for i, r in enumerate(rows)
            if not r.get("error")
            or (cell_key(r) not in measured and last_failure.get(cell_key(r)) == i)]


def build(state: dict, runs_base: Path, *, system: dict,
          placement: dict | None = None, collapse: bool = True) -> list[dict]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info_by_model = (state.get("plan") or {}).get("model_info") or {}
    docs = []
    rows = state.get("results") or []
    if collapse:
        rows = collapse_rows(rows)
    for row in rows:
        sweep = None
        if row.get("run_dir"):
            p = runs_base.parent / row["run_dir"] / "headline_sweep.json"
            if not p.is_file():
                p = runs_base / Path(row["run_dir"]).name / "headline_sweep.json"
            try:
                sweep = json.loads(p.read_text())
            except (OSError, ValueError):
                sweep = None
        docs.append(row_document(row, info=info_by_model.get(row["model"]) or {},
                                 sweep=sweep, system=system, generated_at=generated_at))
    if placement:
        docs += placement_documents(placement, system=system, generated_at=generated_at)
    return docs


def index_row(doc: dict) -> dict:
    m, c = doc["meta"], doc["cohorts"][0]
    cap = c.get("capacity_throughput") or {}
    ec = m["engine_config"]
    return {"model": m["models"][0], "engine": m["engines"][0], "run_kind": m["run_kind"],
            "phase": m.get("phase"), "tp": ec["tensor_parallel_size"],
            "replicas": ec.get("replicas"), "gpus": m["system"]["topology"]["gpusUsed"],
            "quant": ec.get("quantization_kind"),
            "output_tokens": (m.get("workload") or {}).get("output_tokens"),
            "status": c["final_status"], "peak_pool": cap.get("pool_size"),
            "generated_tok_per_s": cap.get("generated_tok_per_s"),
            "success_rate": cap.get("success_rate"),
            "rate_confidence": cap.get("rate_confidence"),
            "tpot_implied_tok_per_s": cap.get("tpot_implied_tok_per_s"),
            "source_dir": m["source_dir"],
            "error": (c.get("error") or "")[:200] or None}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def document_names(docs: list[dict]) -> list[str]:
    """A stable, unique file name per document."""
    seen: dict[str, int] = {}
    names = []
    for d in docs:
        m = d["meta"]
        ec = m["engine_config"]
        base = _slug(f"{m['models'][0].split('/')[-1]}__{m['engines'][0]}__tp{ec['tensor_parallel_size']}"
                     f"x{ec.get('replicas')}__{m.get('config_name') or ''}"
                     f"{Path(m['source_dir'] or 'none').name}")
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return names


def write(docs: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    for name, d in zip(document_names(docs), docs, strict=True):
        (out / "runs" / f"{name}.json").write_text(json.dumps(d, indent=1))
    (out / "all.json").write_text(json.dumps(docs, indent=1))
    (out / "index.json").write_text(json.dumps([index_row(d) for d in docs], indent=1))


def zip_bytes(docs: list[dict]) -> bytes:
    """The same layout as ``write``, as a zip in memory -- what the UI
    downloads."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, d in zip(document_names(docs), docs, strict=True):
            z.writestr(f"runs/{name}.json", json.dumps(d, indent=1))
        z.writestr("all.json", json.dumps(docs, indent=1))
        z.writestr("index.json", json.dumps([index_row(d) for d in docs], indent=1))
    return buf.getvalue()


def summary(docs: list[dict]) -> dict:
    """Counts and the index, for the UI's export panel."""
    idx = [index_row(d) for d in docs]
    return {"documents": len(docs),
            "ok": sum(1 for r in idx if r["status"] == "ok"),
            "failed": sum(1 for r in idx if r["status"] == "failed"),
            "no_passing_rung": sum(1 for r in idx if r["status"] == "no_passing_rung"),
            "placement": sum(1 for r in idx if r["run_kind"] == "kt_expert_placement"),
            "models": len({r["model"] for r in idx}),
            "generated_at": docs[0]["meta"]["generated_at"] if docs else None,
            "system": ({k: v for k, v in docs[0]["meta"]["system"].items() if k != "topology"}
                       if docs else None),
            "index": idx}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m simulator.export_sizing")
    ap.add_argument("--state", default="runs/roofline.json")
    ap.add_argument("--placement", default="runs/kt_placement/kt_placement.json")
    ap.add_argument("--out", default="exports/sizing")
    ap.add_argument("--all-attempts", action="store_true",
                    help="keep failed attempts that a later retry superseded")
    a = ap.parse_args(argv)
    state_path = Path(a.state)
    state = json.loads(state_path.read_text())
    placement = None
    if a.placement and Path(a.placement).is_file():
        placement = json.loads(Path(a.placement).read_text())
    docs = build(state, state_path.parent, system=detect_system(), placement=placement,
                 collapse=not a.all_attempts)
    write(docs, Path(a.out))
    ok = sum(1 for d in docs if d["cohorts"][0]["final_status"] == "ok")
    print(f"{len(docs)} documents ({ok} ok, {len(docs) - ok} failed) -> {a.out}")


if __name__ == "__main__":
    main()
