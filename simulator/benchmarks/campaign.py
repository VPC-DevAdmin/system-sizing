"""Attach durable third-party experiments to exact recorded roofline measurements."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path

from .process import Lease, module_argv, run_process
from .spec import ADAPTER_VERSION, VERSIONS, BenchmarkSpec, digest, resolve_spec, winners
from .storage import campaign_path, read_json, redact, sha256, write_json


def recorded_config(base: Path, row: dict) -> dict:
    """Read the config actually measured, never reconstruct from current UI defaults."""
    if row.get("error") or not row.get("out_tok_s"):
        raise ValueError("Choose a successful recorded measurement")
    root = base.resolve()
    source = Path(row.get("run_dir", "")).resolve()
    if not source.is_relative_to(root) or not (source / "run.db").is_file():
        raise ValueError("Recorded run database unavailable within this runs directory")
    with sqlite3.connect(f"file:{source / 'run.db'}?mode=ro", uri=True) as conn:
        records = conn.execute(
            "SELECT config_json FROM cohort_run WHERE model_id=?", (row["model"],)
        ).fetchall()
    docs = [json.loads(r[0]) for r in records]
    unique = {digest(d): d for d in docs}
    if len(unique) != 1:
        raise ValueError("Recorded run does not resolve to one unambiguous engine configuration")
    doc = next(iter(unique.values()))
    if "resolved_config" in doc:
        doc = doc["resolved_config"]
    elif "engine" not in doc:
        raise ValueError(
            "This legacy result did not save its resolved engine configuration. "
            "Remeasure this roofline cell once to enable faithful third-party benchmarking."
        )
    clean = redact(doc)
    if "[redacted]" in json.dumps(clean.get("engine", {})):
        raise ValueError(
            "Recorded engine configuration contains redacted credentials; use a credential-free local profile"
        )
    if clean != doc:
        # Avoid saving credentials in a new export. A separate auth reference can be
        # added later; authenticated remote endpoints are not the initial target.
        changed = [
            (k, v)
            for k, v in doc.get("engine", {}).items()
            if clean.get("engine", {}).get(k) != v and v not in (None, "", "EMPTY")
        ]
        if changed:
            raise ValueError(
                "Recorded configuration contains credentials; use a credential-free local profile"
            )
    return clean


def create_campaign(
    base: Path, row: dict, spec: BenchmarkSpec, *, source: dict, force: bool = False
) -> Path:
    if not spec.harnesses:
        raise ValueError("Select at least one benchmark harness")
    cfg = recorded_config(base, row)
    summary_path = Path(row["run_dir"]) / "headline_sweep.json"
    if summary_path.is_file():
        shape = read_json(summary_path).get("shape", {})
        for key in ("input_tokens", "output_tokens"):
            if shape.get(key) is not None and shape[key] != row.get(key):
                raise ValueError(
                    f"Recorded result {key} does not match the source measurement; remeasure this roofline cell"
                )
    recipe = resolve_spec(spec, row)
    asset_file = Path(row["run_dir"]) / "resolved-assets.json"
    source_assets = read_json(asset_file) if asset_file.is_file() else None
    manifest = {
        "schema_version": 1,
        "adapter_version": ADAPTER_VERSION,
        "adapter_sources": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "harness_versions": {h: VERSIONS[h] for h in spec.harnesses},
        "selection": source,
        "source_result": row,
        "configuration": cfg,
        "recipe": recipe,
        "source_assets": source_assets,
        "claims": "Harness measurements; no independent certification",
        "limitations": [
            "Quality is not evaluated by this performance suite",
            "Single-turn synthetic corpus; persona capacity requires a separate session run",
            "SLO verdicts use aggregate p95 and error thresholds, not per-request goodput",
            "Selection used each candidate's best recorded roofline shape",
            "Round-robin streaming proxy is included in latency and throughput",
            "GuideLLM uses its public dataset extension to cycle the captured corpus; extension source is included",
        ],
    }
    if not source_assets:
        manifest["limitations"].append(
            "Source asset revisions were not recorded; this is a new measurement of the recorded settings, not proof of identical original assets"
        )
    if force:
        manifest["requested_at"] = time.time()
    path = campaign_path(base, digest(manifest)[:24])
    if not (path / "manifest.json").exists():
        code_dir = path / "adapter-source"
        code_dir.mkdir(parents=True, exist_ok=True)
        for name, expected in manifest["adapter_sources"].items():
            target = code_dir / name
            target.write_bytes((Path(__file__).parent / name).read_bytes())
            if sha256(target) != expected:
                raise ValueError("Adapter source changed during experiment creation; retry")
        write_json(path / "manifest.json", manifest)
        write_json(
            path / "state.json",
            {
                "id": path.name,
                "phase": "queued",
                "outcome": None,
                "created_at": time.time(),
                "updated_at": time.time(),
                "attempts": [],
                "model": row["model"],
                "engine": row["engine"],
                "harnesses": spec.harnesses,
                "current": None,
            },
        )
    elif read_json(path / "manifest.json") != manifest:
        raise ValueError("Experiment identity collision")
    return path


async def execute(path: Path, base: Path) -> dict:
    # Serialize callers before spawning a worker. Otherwise a duplicate caller
    # can mislabel the active campaign when its child loses the worker lease.
    with Lease(path / ".execution.lock"):
        with Lease(path / ".worker.lock"):
            pass  # Also reject an orphaned worker still completing its cleanup.
        return await _execute(path, base)


async def _execute(path: Path, base: Path) -> dict:
    state = read_json(path / "state.json")
    if state["phase"] == "complete" and (path / "evidence.zip").exists():
        for name, expected in read_json(path / "SHA256SUMS.json").items():
            artifact = (path / name).resolve()
            if not artifact.is_relative_to(path.resolve()) or sha256(artifact) != expected:
                raise ValueError("Completed evidence changed; create a fresh experiment")
        return state
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "1")
    env["TOKENIZERS_PARALLELISM"] = "false"
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    manifest = read_json(path / "manifest.json")
    if path.name != digest(manifest)[:24]:
        raise ValueError("Experiment manifest changed; create a fresh experiment")
    try:
        rc = await run_process(
            module_argv(
                package_root,
                "simulator.benchmarks.worker",
                str(path.resolve()),
                str(base.resolve()),
                str(os.getpid()),
            ),
            path / "worker.log",
            timeout=manifest["recipe"]["max_hours"] * 3600 + 30,
            env=env,
            termination_grace=manifest["configuration"]["engine"].get("startup_timeout_s", 600)
            + 60,
        )
        state = read_json(path / "state.json")
        if rc and state["phase"] not in ("failed", "interrupted", "cancelled"):
            state.update(
                phase="interrupted", outcome="incomplete", error="Worker exited unexpectedly"
            )
            write_json(path / "state.json", state)
        return state
    except (asyncio.CancelledError, asyncio.TimeoutError):
        # The worker gets TERM first and saves its attempt in its finally block.
        state = read_json(path / "state.json")
        if state["phase"] != "complete":
            state.update(phase="interrupted", outcome="incomplete", updated_at=time.time())
            write_json(path / "state.json", state)
        raise


async def run_finalists(base: Path, roofline_state, spec: BenchmarkSpec, state_path: Path):
    from ..model_catalog import load_model_catalog
    from ..roofline import save_state

    selected = winners(roofline_state.results, load_model_catalog())
    links = []
    roofline_state.plan["benchmarks"] = spec.model_dump()
    roofline_state.plan["benchmark_campaigns"] = links
    roofline_state.status = "benchmarking"
    save_state(state_path, roofline_state)
    for row in selected:
        roofline_state.current = {**row, "phase": "third_party_benchmarks"}
        link = {"model": row["model"], "status": "incomplete"}
        links.append(link)
        try:
            path = create_campaign(
                base,
                row,
                spec,
                source={
                    "kind": "roofline_family_winner",
                    "roofline_started_at": roofline_state.started_at,
                    "objective": "confirmed_output_tokens_per_second",
                },
            )
            link.update(id=path.name, status="running")
            save_state(state_path, roofline_state)
            result = await execute(path, base)
            link["status"] = result.get("outcome") or result["phase"]
        except asyncio.CancelledError:
            link.update(status="incomplete", error="cancelled")
            save_state(state_path, roofline_state)
            raise
        except Exception as e:  # one bad row must not strand the roofline in "benchmarking"
            link.update(status="incomplete", error=f"{type(e).__name__}: {e}")
        save_state(state_path, roofline_state)
    if not selected:
        links.append(
            {"status": "incomplete", "error": "No confirmed model-family winners available"}
        )
    from .report import export_search

    report_dir = await asyncio.to_thread(export_search, base, links)
    roofline_state.plan["benchmark_report"] = report_dir.name
    roofline_state.current = None
    save_state(state_path, roofline_state)


def list_campaigns(base: Path) -> list[dict]:
    rows = []
    for path in (base / "benchmarks").glob("*/state.json"):
        try:
            d = read_json(path)
            manifest = read_json(path.with_name("manifest.json"))
            running = False
            try:
                with Lease(path.parent / ".worker.lock"):
                    pass
            except RuntimeError:
                running = True
            if d["phase"] not in ("queued", "complete", "failed", "interrupted") and not running:
                d.update(
                    phase="interrupted",
                    outcome="incomplete",
                    error="Worker is no longer running; resume from the last completed attempt",
                )
            if d["phase"] == "complete" and running:
                d["phase"] = "exporting"
            d["recipe"] = manifest["recipe"]
            d["source"] = manifest["selection"]
            d["report_ready"] = not running and all(
                (path.parent / name).is_file() for name in ("report.docx", "evidence.zip")
            )
            rows.append(d)
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)
