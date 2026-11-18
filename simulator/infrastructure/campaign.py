"""Immutable UUID experiments, duplicate-request protection, owned execution."""

import asyncio
import os
import random
import re
import time
import uuid
from pathlib import Path

from ..benchmarks.campaign import recorded_config
from ..benchmarks.process import Lease, hardware_lease_path, module_argv, run_process
from ..benchmarks.spec import digest
from ..benchmarks.storage import read_json, sha256, write_json
from .catalog import EXECUTABLE, project
from .workload import build_corpus


def campaign_path(base, identity):
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid infrastructure campaign ID")
    root = (Path(base) / "infrastructure").resolve()
    path = root / identity
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Campaign directory must not be a symlink")
    return path


def create(base, row, spec, idempotency_key):
    if spec.project_id not in EXECUTABLE:
        raise ValueError("This project adapter is not implemented yet; no measurement was started")
    if not re.fullmatch(r"[a-f0-9-]{16,64}", idempotency_key):
        raise ValueError("Use a UUID idempotency key; a rerun requires a new key")
    base = Path(base)
    root = base / "infrastructure"
    root.mkdir(parents=True, exist_ok=True)
    identity_input = {"source": row, "spec": spec.model_dump()}
    request_hash = digest(identity_input)
    with Lease(root / ".create.lock"):
        keyfile = root / "requests" / (idempotency_key + ".json")
        if keyfile.exists():
            previous = read_json(keyfile)
            if previous["request_hash"] != request_hash:
                raise ValueError("Idempotency key already used for a different request")
            return campaign_path(base, previous["id"])
        cfg = (
            {"engine": {"type": "diffusion"}}
            if spec.project_id == "P05"
            else recorded_config(base, row)
        )
        if spec.project_id != "P05" and cfg["engine"]["type"] not in (
            "vllm_cuda",
            "vllm_cuda_multi",
        ):
            raise ValueError("This adapter requires a recorded vLLM CUDA configuration")
        if cfg["engine"].get("model_local_path"):
            raise ValueError("Choose a cache-backed model so its revision can be pinned")
        path = campaign_path(base, uuid.uuid4().hex)
        path.mkdir()
        corpus = build_corpus(spec.corpus_size, spec.document_records, spec.seed)
        if spec.p01_goal_seek:
            from .goal_search import shapes

            corpus = {
                shape["id"]: build_corpus(
                    spec.corpus_size, shape["records"], spec.seed, shape["answers"]
                )
                for shape in shapes(spec)
            }
        if spec.project_id == "P03":
            corpus = {
                str(records): build_corpus(spec.corpus_size, records, spec.seed)
                for records in spec.cache_record_counts
            }
        if spec.project_id == "P05":
            from .diffusion import corpus as image_corpus

            corpus = image_corpus(spec)
        write_json(path / "corpus.json", corpus)
        rng = random.Random(spec.seed)
        schedule = []
        for block in range(spec.paired_blocks):
            arms = (
                ["reference", "baseline", "treatment"]
                if spec.project_id == "P03"
                else ["dedicated", "shared", "mig"]
                if spec.project_id == "P06"
                else ["baseline", "treatment"]
            )
            rng.shuffle(arms)
            schedule.extend({"block": block, "arm": arm} for arm in arms)
        if spec.project_id == "P02" and spec.session_mode == "capacity":
            schedule = []
            for block in range(spec.paired_blocks):
                for population in spec.session_population_grid:
                    for rate in spec.session_arrival_rates or spec.session_rate_grid:
                        arms = [
                            (arm, compaction)
                            for arm in (
                                spec.session_cache_arms
                                if spec.session_workload == "knowledge_worker"
                                or spec.session_pacing != "open"
                                else ["baseline", "treatment"]
                            )
                            for compaction in (
                                spec.session_compaction_arms
                                if spec.session_workload == "knowledge_worker"
                                else ["none"]
                            )
                        ]
                        rng.shuffle(arms)
                        for arm, compaction in arms:
                            cell = dict(
                                block=block,
                                arm=arm,
                                population=population,
                                rate=rate if spec.session_arrival_rates else rate * population,
                                per_session_rate=rate / population
                                if spec.session_arrival_rates
                                else rate,
                            )
                            if spec.session_workload == "knowledge_worker":
                                cell["compaction"] = compaction
                            schedule.append(cell)
        if spec.project_id == "P05":
            schedule = []
            for block in range(spec.paired_blocks):
                cells = [
                    {"block": block, "arm": "baseline", "steps": steps}
                    for steps in spec.diffusion_teacher_steps
                ] + [{"block": block, "arm": "treatment", "steps": 4}]
                rng.shuffle(cells)
                schedule.extend(cells)
        if spec.project_id == "P01" and spec.p01_prefill_grid:
            schedule = []
            for block in range(spec.paired_blocks):
                placements = (
                    spec.p01_placement_grid
                    if (spec.p01_goal_seek or spec.p01_token_search)
                    else [spec.p01_placement]
                )
                cells = [
                    {"block": block, "arm": "baseline", "prefill": 0, "placement": placements[0]}
                ] + [
                    {"block": block, "arm": "treatment", "prefill": n, "placement": placement}
                    for n in spec.p01_prefill_grid
                    for placement in placements
                ]
                rng.shuffle(cells)
                schedule.extend(cells)
        if spec.project_id == "P10":
            schedule.insert(0, {"block": 0, "arm": "profile"})
        source_dir = Path(__file__).parent
        sources = {}
        for source in sorted(source_dir.glob("*")):
            if source.suffix in (".py", ".json"):
                dest = path / "adapter-source" / source.name
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(source.read_bytes())
                sources[source.name] = sha256(dest)
        supporting = {}
        package = source_dir.parent
        for source in sorted(package.rglob("*.py")):
            if "__pycache__" in source.parts or source.parent == source_dir:
                continue
            name = str(source.relative_to(package))
            dest = path / "supporting-source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
            supporting[name] = sha256(dest)
        manifest = {
            "supporting_sources": supporting,
            "schema_version": 1,
            "id": path.name,
            "created_at": time.time(),
            "spec": spec.model_dump(),
            "configuration": cfg,
            "source_result": row,
            "project": project(spec.project_id),
            "schedule": schedule,
            "corpus_sha256": sha256(path / "corpus.json"),
            "adapter_sources": sources,
            "harness": "capsim document-extraction driver v1",
            "quality_scope": "Exact JSON extraction from synthetic logistics documents; not broad model quality",
            "cache_policy": "Engine restart per arm; separate warmup using a disjoint corpus; prefix caching disabled",
            "load_policy": "Closed-loop concurrency; measured throughput includes drain time",
            "publication": "Pilot evidence cannot support a publishable performance claim",
        }
        if spec.project_id == "P06" and spec.tenant_workflow == "post_call":
            from .calls import call_corpus

            corpus = call_corpus()
            write_json(path / "corpus.json", corpus)
            manifest["corpus_sha256"] = sha256(path / "corpus.json")
            manifest["harness"] = "capsim speech-to-structured-summary v1"
            manifest["quality_scope"] = (
                "Synthetic support-call audio: normalized word error rate and exact four-field summary; no real-call or broad summarization claim"
            )
            manifest["load_policy"] = (
                "Fixed call arrivals; stages pipeline across calls; queue, errors and final drain included"
            )
            manifest["cache_policy"] = (
                "Fresh models per placement; prefix cache disabled; same audio and prompts across placements"
            )
            for out, title in zip(
                manifest["project"]["outputs"],
                [
                    "Completed audio-hours per hour",
                    "End-to-end call-processing latency",
                    "Two-model GPU consolidation",
                ],
                strict=True,
            ):
                out["title"] = title
        if spec.project_id == "P06" and spec.tenant_workflow == "independent":
            manifest["harness"] = "capsim fixed-arrival independent tenants v1"
            manifest["load_policy"] = (
                "Two endpoints receive equal fixed request rates; then tenant 1 receives the declared noisy-neighbor multiplier. Queueing and admissions count toward latency and failures."
            )
        if spec.project_id == "P05":
            from .diffusion import MODELS

            manifest["diffusion_models"] = MODELS
            manifest["harness"] = "capsim serial diffusion pipeline v1"
            manifest["quality_scope"] = (
                "Technical image integrity; task acceptance requires blind rubric review"
            )
            manifest["cache_policy"] = (
                "Fresh pipeline per arm; disjoint warmup prompt; pinned local model revisions"
            )
            manifest["load_policy"] = (
                "One image at a time; pipeline timing includes text encoding, denoising and VAE decode; image file writing excluded"
            )
        if spec.project_id == "P04" and spec.speculative_concurrencies:
            original = manifest["project"]["outputs"]
            manifest["project"]["outputs"] = [
                dict(
                    o,
                    id=o["id"] + f"-C{c}",
                    title=o["title"] + f" at concurrency {c}",
                    concurrency=c,
                    base_output_id=o["id"],
                )
                for c in spec.speculative_concurrencies
                for o in original
            ]
            manifest["load_policy"] = (
                "Ascending closed-loop load grid within each owned launch; separate warmup and measurement per load; all cells reported"
            )
        if spec.project_id == "P03":
            manifest["harness"] = "capsim matched finite cache cohort v2"
            manifest["load_policy"] = (
                f"Exactly {spec.cache_samples} identical documents per cell; bounded concurrency; all failures and drain included"
            )
        if spec.project_id == "P01":
            from .dynamo import DYNAMO_IMAGE, DYNAMO_SOURCE

            if spec.p01_prefill_grid:
                manifest["load_policy"] = (
                    "Fixed arrival rate grid within every split and aggregate launch; queue and drain included; shared aggregate comparator"
                )
                for out, title in zip(
                    manifest["project"]["outputs"],
                    [
                        "Highest passing tested document rate",
                        "First-token latency at matched arrival rate",
                        "Completed-response latency at matched arrival rate",
                    ],
                    strict=True,
                ):
                    out["title"] = title
            if spec.p01_goal_seek:
                manifest["harness"] = "capsim joint document-shape and GPU-allocation search v1"
                manifest["load_policy"] = (
                    "Per shape and allocation: test rate endpoints, then bisect the passing/failing bracket on the declared rate grid; all measurements retained. Monotonic capacity is a search heuristic, not an asserted property."
                )
                manifest["goal_search"] = {
                    "objective": "largest measured same-shape capacity uplift over aggregate",
                    "rate_grid": spec.p01_request_rates,
                    "rate_resolution": spec.p01_rate_resolution,
                    "placements": spec.p01_placement_grid,
                    "document_records": spec.p01_document_grid,
                    "answer_records": spec.p01_answer_grid,
                    "limit": "Highest measured passing point within declared grid and time budget; no global optimum claim",
                }
                for out, title in zip(
                    manifest["project"]["outputs"],
                    [
                        "Peak document capacity improvement",
                        "Peak absolute document capacity",
                        "Matched workload latency improvement",
                    ],
                    strict=True,
                ):
                    out["title"] = title
            manifest["dynamo"] = {
                "image": DYNAMO_IMAGE,
                "source": DYNAMO_SOURCE,
                "baseline": "One aggregated TP1 worker per GPU",
                "treatment": (
                    f"Search prefill counts {spec.p01_prefill_grid}; remaining GPUs decode"
                    if spec.p01_prefill_grid
                    else f"{spec.prefill_workers} prefill workers; remaining GPUs decode"
                ),
                "transfer": "NIXL; increasing measured byte counters required",
            }
        if spec.project_id == "P02":
            from .dynamo import DYNAMO_IMAGE, DYNAMO_SOURCE

            if getattr(spec, "session_runtime", "dynamo") == "vllm":
                from ..gpu_vendor import engine_image, vendor

                manifest["runtime"] = {
                    "kind": "vllm serve (no Dynamo)",
                    "image": spec.p01_runtime_image
                    or engine_image("vllm", "vllm/vllm-openai:latest"),
                    "gpu_vendor": vendor(),
                }
            else:
                manifest["dynamo"] = {"image": DYNAMO_IMAGE, "source": DYNAMO_SOURCE}
            manifest["harness"] = "capsim finite returning-session replay v2"
            manifest["session_hot_diagnostic"] = True
            manifest["cache_policy"] = (
                "Fresh engine per arm; prime cohort, one immediate GPU-hot diagnostic, disjoint cache pressure, then replay each session once"
            )
            manifest["load_policy"] = (
                "Fixed finite cohort with bounded concurrency; resume throughput excludes prime and pressure phases"
            )
        if spec.project_id == "P02" and spec.session_mode == "capacity":
            manifest.update(
                harness="capsim scheduled document-capacity replay v1",
                session_hot_diagnostic=False,
                cache_policy="Fresh deployment per cell; natural document reuse; GPU-only prefix caching versus CPU offload; automatic GPU KV allocation",
                load_policy="Frozen Poisson arrivals, sticky session routing, all cold turns included, bounded admission and deadline; full declared rate grid",
                quality_scope="Synthetic exact-match document extraction; representative document-analysis validation remains required",
            )
            for output, title in zip(
                manifest["project"]["outputs"],
                [
                    "Document-analysis capacity",
                    "Matched-load latency",
                    "Cache and prefill evidence",
                ],
                strict=True,
            ):
                output["title"] = title
        if spec.p01_token_search:
            manifest.update(
                harness="capsim roofline saturation sweep with persistent Dynamo allocations v1",
                quality_scope="Fixed token-shape generation; no semantic-quality claim",
                load_policy="Closed-loop saturation; native counter rates after convergence; TTFT reported without a latency gate",
            )
            manifest["schedule"].sort(key=lambda c: (c["block"], c["prefill"] != 0))
            for output, title in zip(
                manifest["project"]["outputs"],
                [
                    "Peak output-token throughput uplift",
                    "TTFT at matched concurrency",
                    "Throughput and latency curves",
                ],
                strict=True,
            ):
                output["title"] = title
        if spec.project_id == "P02" and spec.session_prime_documents:
            manifest["load_policy"] = (
                "Scored priming followed by fixed-corpus returns; Poisson times with shuffled round-robin document assignment; bounded open-loop rate sweep"
            )
            manifest["cache_policy"] += "; priming billed separately, no forced eviction"
        write_json(path / "manifest.json", manifest)
        (path / "manifest.sha256").write_text(sha256(path / "manifest.json"))
        write_json(
            path / "state.json",
            {
                "id": path.name,
                "phase": "queued",
                "created_at": time.time(),
                "project_id": spec.project_id,
                "model": row["model"],
                "results": [],
                "attempts": [],
                "current": None,
            },
        )
        write_json(keyfile, {"id": path.name, "request_hash": request_hash})
        return path


def verify(path):
    if sha256(path / "manifest.json") != (path / "manifest.sha256").read_text():
        raise ValueError("Manifest changed; create a new campaign")
    manifest = read_json(path / "manifest.json")
    if sha256(path / "corpus.json") != manifest["corpus_sha256"]:
        raise ValueError("Frozen workload changed")
    for name, checksum in manifest["adapter_sources"].items():
        if sha256(Path(__file__).parent / name) != checksum:
            raise ValueError(
                "Adapter changed; create a new campaign for the updated implementation"
            )
    for name, checksum in manifest.get("supporting_sources", {}).items():
        if sha256(Path(__file__).parent.parent / name) != checksum:
            raise ValueError("Supporting runtime changed; create a fresh campaign")
    return manifest


def listing(base):
    root = Path(base) / "infrastructure"
    result = []
    for path in root.glob("*/state.json"):
        if re.fullmatch(r"[a-f0-9]{32}", path.parent.name) and not path.parent.is_symlink():
            try:
                result.append(read_json(path))
            except (OSError, ValueError):
                continue
    return sorted(result, key=lambda s: s["created_at"], reverse=True)


async def execute(path):
    with Lease(path / ".execution.lock"):
        manifest = verify(path)
        state = read_json(path / "state.json")
        if state["phase"] != "queued":
            # Reruns get a distinct identity. Completed/failed evidence is never overwritten.
            return state
        root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["TOKENIZERS_PARALLELISM"] = "false"
        try:
            rc = await run_process(
                module_argv(root, "simulator.infrastructure.worker", str(path.resolve())),
                path / "worker.log",
                timeout=manifest["spec"]["max_hours"] * 3600 + 120,
                env=env,
                termination_grace=600,
            )
            state = read_json(path / "state.json")
            if rc or state["phase"] not in ("complete", "failed", "cancelled"):
                state.update(
                    phase="failed",
                    error="Worker exited without completing the campaign",
                    current=None,
                )
                write_json(path / "state.json", state)
                await _release_orphans(path)
        except BaseException as exc:
            state = read_json(path / "state.json")
            if state["phase"] not in ("complete", "failed", "cancelled"):
                state.update(
                    phase="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    error=type(exc).__name__,
                    current=None,
                )
                write_json(path / "state.json", state)
            # Containers are dockerd children, not members of the killed process
            # group, so a worker that died mid-teardown can leave GPUs occupied.
            await asyncio.shield(_release_orphans(path))
            raise
        return read_json(path / "state.json")


async def _release_orphans(path):
    """Best-effort resource release once the worker process is gone."""
    from .recovery import recover_unlocked

    try:
        with Lease(path / ".worker.lock"), Lease(hardware_lease_path()):
            record = await asyncio.to_thread(recover_unlocked, path, mark_interrupted=False)
    except Exception as exc:
        record = {"error": f"{type(exc).__name__}: {exc}"}
    state = read_json(path / "state.json")
    state["recovery"] = record
    write_json(path / "state.json", state)
