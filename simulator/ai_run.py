"""GPU persona runs for the AI capacity planner (AI_RUN_FORMAT).

The planner's GPU category sizes only configurations it has measured:
one XE7740 run per GPU count (1, 2, 4, 8), each the six-persona sweep
the CPU reference runs used, so CPU and GPU capacity compare directly.
This module is the glue around ``capsim sweep`` and ``capsim export``:

* ``configs`` writes one engine config per (model, GPU count) through
  the same builder the roofline and the benchmark form use
  (engines/custom.py): TP 1, one replica per GPU, spread across both
  PCIe/NUMA domains, 8K context, 0.90 memory share -- and a load
  generator sized to the GPU count, so the sweep stops on the SLA and
  never on its own client.
* ``finalize`` takes a run's ``buyer_page_data.json`` and adds what the
  planner needs beside it: ``meta.system`` (hardware identity plus this
  run's topology), ``meta.personas`` (each persona's token and activity
  model -- what turns total users into concurrent users and personas
  into tokens), and a ``--system`` sidecar.
* ``validate`` applies the importer's rules before a file is handed
  over, and names every persona that would import as a floor (no
  tested step above its capacity point).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PERSONAS = ("quick_lookup", "conversational", "writer", "document_qa",
            "code_assist", "long_form_generator")
PLANNER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
GPU_COUNTS = (1, 2, 4, 8)
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
# Workers per GPU for the open-loop generator. The default (16 total)
# was sized for CPU hosts whose knees sit at 16-32 streams; a GPU knee
# is hundreds of streams per card.
WORKERS_PER_GPU = 8
# ...and the generator starts at half of that, so the first windows at
# each rate are not spent catching up.
START_WORKERS_PER_GPU = 4


def system_identity(memory_gb: int = 2048) -> dict:
    """The XE7740's identity as the planner wants it, read from the box
    where it can be (export_sizing.detect_system), with installed RAM
    given explicitly: the kernel reports usable memory (2,015 GiB on a
    2 TB box) and the planner prices what is installed."""
    from .export_sizing import detect_system
    s = detect_system()
    accel = [{k: a[k] for k in ("vendor", "model", "count", "memoryGbEach")}
             for a in s.get("accelerators") or []]
    return {"vendor": s.get("vendor") or "Dell",
            "platform": s.get("platform") or "PowerEdge XE7740",
            "cpuKey": s.get("cpuKey"),
            "sockets": s.get("sockets"),
            "memoryGb": int(memory_gb),
            "accelerators": accel}


def topology(gpus: int) -> dict:
    return {"gpusUsed": int(gpus), "replicas": int(gpus), "tensorParallelSize": 1,
            "placement": "spread"}


def engine_custom(model: str, gpus: int) -> dict:
    # custom_engine takes the vLLM family as vllm_cuda_multi and emits
    # the single-container vllm_cuda type itself for one replica.
    return {"engine": "vllm_cuda_multi",
            "model_id": model, "tp": 1, "replicas": int(gpus), "placement": "spread",
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION}


def write_configs(model: str, out_dir: Path, runs_base: str = "runs_ai") -> list[Path]:
    """One config per GPU count, built by engines/custom.py. Runs land
    in their own base directory, apart from the service's runs/, since
    the persona sweeps run from the CLI (open-loop: the service runs
    sweeps closed-loop, whose single-process generator cannot hold a
    GPU knee's thousands of sessions)."""
    import yaml

    from .engines.custom import config_doc, custom_engine
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    slug = model.split("/")[-1].lower()
    for n in GPU_COUNTS:
        doc = config_doc(custom_engine(engine_custom(model, n)), runs_base)
        doc["simulation"] = {"mode": "open",
                             "open_loop_max_workers": max(16, WORKERS_PER_GPU * n),
                             "open_loop_min_workers": max(4, START_WORKERS_PER_GPU * n)}
        p = out_dir / f"ai-{slug}-{n}gpu.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        paths.append(p)
    return paths


def persona_models() -> dict:
    """Each planner persona's token and activity model, verbatim from
    the persona catalogue the sweep ran with."""
    import yaml
    doc = yaml.safe_load((Path(__file__).parent / "personas_data" / "default.yaml").read_text())
    keep = ("name", "input_tokens", "output_tokens", "turns_per_session",
            "sessions_before_leaving", "inter_session_gap_seconds",
            "read_time_seconds", "active_think_seconds", "sla")
    return {pid: {k: v for k, v in (doc["personas"].get(pid) or {}).items() if k in keep}
            for pid in PERSONAS}


def finalize(export: dict, *, system: dict, gpus: int) -> dict:
    """The planner file: the export with meta.system and meta.personas."""
    out = json.loads(json.dumps(export))
    meta = out.setdefault("meta", {})
    meta["system"] = {**system, "topology": topology(gpus)}
    meta["personas"] = persona_models()
    ec = meta.get("engine_config") or {}
    ec.setdefault("tensor_parallel_size", 1)
    meta["engine_config"] = ec
    return out


def validate(doc: dict, *, require_planner_model: bool = True) -> list[str]:
    """The importer's rules, as problems (empty = importable). Floors
    are reported as ``floor:`` lines: importable, but conservative."""
    problems: list[str] = []
    meta = doc.get("meta") or {}
    models = meta.get("models") or []
    if require_planner_model and models != [PLANNER_MODEL]:
        problems.append(f"models must be exactly [{PLANNER_MODEL}], got {models}")
    sysd = meta.get("system") or {}
    for k in ("platform", "cpuKey", "sockets", "memoryGb", "accelerators", "topology"):
        if not sysd.get(k):
            problems.append(f"meta.system.{k} missing")
    topo = sysd.get("topology") or {}
    if topo and topo.get("replicas", 0) * topo.get("tensorParallelSize", 0) != topo.get("gpusUsed"):
        problems.append("topology: replicas x tensorParallelSize != gpusUsed")
    tp = (meta.get("engine_config") or {}).get("tensor_parallel_size")
    if tp is not None and topo and tp != topo.get("tensorParallelSize"):
        problems.append(f"engine_config.tensor_parallel_size {tp} != topology "
                        f"{topo.get('tensorParallelSize')}")
    personas = {c.get("id"): c for c in doc.get("cohorts") or []
                if c.get("category") == "persona"}
    for pid in PERSONAS:
        c = personas.get(pid)
        if c is None:
            problems.append(f"persona {pid} missing")
            continue
        curve = c.get("curve") or []
        cap = (c.get("capacity_throughput") or {}).get("pool_size")
        if cap is None:
            problems.append(f"floor: {pid} has no passing step (capacity null)")
            continue
        if not any((p.get("pool_size") or 0) > cap for p in curve):
            problems.append(f"floor: {pid} has no tested step above its capacity {cap}")
        elif c.get("fail_pool_size") is None:
            problems.append(f"floor: {pid} never failed above capacity {cap}")
    extra = [c.get("id") for c in doc.get("cohorts") or []
             if c.get("category") == "persona" and c.get("id") not in PERSONAS]
    if extra:
        problems.append(f"unknown persona ids: {extra}")
    return problems


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m simulator.ai_run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs")
    c.add_argument("--model", default=PLANNER_MODEL)
    c.add_argument("--out", default="config/ai_runs")
    f = sub.add_parser("finalize")
    f.add_argument("export")
    f.add_argument("--gpus", type=int, required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--memory-gb", type=int, default=2048)
    f.add_argument("--any-model", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.add_argument("--any-model", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "configs":
        for p in write_configs(a.model, Path(a.out)):
            print(p)
    elif a.cmd == "finalize":
        doc = finalize(json.loads(Path(a.export).read_text()),
                       system=system_identity(a.memory_gb), gpus=a.gpus)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2))
        side = out.with_name(out.stem + ".system.json")
        side.write_text(json.dumps(doc["meta"]["system"], indent=2))
        for line in validate(doc, require_planner_model=not a.any_model) or ["importable"]:
            print(line)
        print(out)
        print(side)
    else:
        doc = json.loads(Path(a.file).read_text())
        for line in validate(doc, require_planner_model=not a.any_model) or ["importable"]:
            print(line)


if __name__ == "__main__":
    main()
