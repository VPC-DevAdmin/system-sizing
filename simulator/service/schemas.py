"""Request bodies of the control-plane API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class StartRunRequest(BaseModel):
    profile: Optional[str] = None
    config: Optional[str] = None
    # Advanced: benchmark a downloaded model with hand-set engine
    # shape instead of a saved profile. {"model_id", "replicas",
    # "tp", "max_num_seqs"?, "max_num_batched_tokens"?,
    # "kv_cache_dtype"?} — devices are assigned from this machine's
    # detected topology.
    custom: Optional[dict] = None
    # {"kind": "cohort"|"persona", "id": "..."} or {"kind": "sweep",
    # "type": "all"|"personas"|"cohorts"|"a,b,c"}
    workload: dict
    new_run: bool = False
    pool_sizes: Optional[list[int]] = None
    adaptive: bool = False
    # Methodology. "open" (default) — open-loop Poisson session
    # arrivals; capacity is the arrival rate where the engine's queue
    # turns divergent. "closed" — the legacy fixed-pool ramp (kept for
    # comparison runs and for the pool_sizes / adaptive knobs, which
    # only apply there). Sweeps always run closed-loop.
    # None = follow the profile's ``simulation.mode`` (default open).
    mode: Optional[str] = None
    # Headline sweeps only: cap the concurrency ladder (the UI's
    # "max concurrent streams" control). None = the full ladder.
    max_concurrency: Optional[int] = None
    # Shape search only: the pinned prompt length. None = config
    # default (128, the vendor convention).
    input_tokens: Optional[int] = None
    # Joint engine+shape search: which grid to walk. The explicit
    # lists override the preset — a model's KV cost per token decides
    # which shapes are even reachable (Llama-70B is 16x Qwen3.6's, so
    # its grid belongs at short outputs), and that is not something a
    # fixed preset can know.
    preset: Optional[str] = None
    search_max_num_seqs: Optional[list[int]] = None
    search_output_tokens: Optional[list[int]] = None
    # Which servers the joint search covers. None = vLLM only, so an
    # unqualified search costs what it always did; naming both makes
    # the engine a measured dimension rather than an assumption.
    search_engines: Optional[list[str]] = None


class ExportRequest(BaseModel):
    slim: bool = False


class SaveSpecRequest(BaseModel):
    # Either form saves the same catalog entry. The graphical designer
    # sends ``spec`` (structured JSON — the UI has no YAML anywhere);
    # ``yaml`` remains for API users and older clients.
    yaml: Optional[str] = None
    spec: Optional[dict] = None


class ModelDownloadRequest(BaseModel):
    model: str
    # "gguf": stage the model's GGUF companion (KTransformers' weights,
    # from the catalog entry's ``gguf`` block) instead of the HF repo.
    companion: Optional[Literal["gguf"]] = None


class ModelAddRequest(BaseModel):
    model: str                          # HF repo id, org/name
    family: Optional[str] = None        # default: inferred from the id
    quant: Optional[str] = None         # default: inferred from the id
    notes: str = ""
    # Verify the repo exists on the Hub before adding (best-effort:
    # an offline box adds unverified rather than being blocked).
    check_hub: bool = True
    # Rich metadata — the discovery flow fills these from the Hub's
    # own safetensors metadata so an added model arrives fully sized.
    series: Optional[str] = None
    params_b: Optional[float] = None
    moe: Optional[bool] = None
    approx_size_gb: Optional[float] = None
    min_vram_gb: Optional[float] = None
    specialty: Optional[str] = None


class RooflineRequest(BaseModel):
    """Autopilot: stage models, search engines x shapes, confirm, report."""
    # Explicit model list, or None to let the ranker choose.
    models: Optional[list[str]] = None
    # How many the ranker picks, across three tiers: FAST (one model
    # per vendor line -- Qwen3, gpt-oss, GLM, Llama... -- in score
    # order before any line gets a second), LARGE (the biggest models
    # the GPUs hold, up to large_limit) and BEYOND_VRAM (models only
    # KTransformers can serve from host RAM, up to beyond_limit). The
    # fast tier takes what the other two leave. Eight = 3 + 3 + 2.
    model_limit: int = 8
    large_limit: int = 3
    beyond_limit: int = 2
    # false = the FAST tier alone (the pre-spectrum roofline).
    spectrum: bool = True
    # false = the pure KV-cost ranking, precision twins and all.
    diverse: bool = True
    cached_only: bool = False
    # Let tensor parallel span both PCIe/NUMA domains (tp8 on an
    # XE7740). Off by default: cross-domain all-reduce is the wrong
    # answer for throughput. On, it is the only way to hold NVFP4
    # giants (DeepSeek-V3.1 at 413 GB, Kimi-K2 at 594 GB) on the GPUs
    # at all -- the "how big can this box go" question.
    allow_cross_domain_tp: bool = False
    # On resume, forget the FAILED cells of these engines so they run
    # again (after a launcher fix); measured cells are kept.
    retry_engines: Optional[list[str]] = None
    engines: Optional[list[str]] = None      # None = every staged engine
    max_num_seqs: Optional[list[int]] = None
    output_tokens: Optional[list[int]] = None
    input_tokens: int = 128
    # resume: true (default) reuses every cell of the previous run
    # whose full launch shape -- model, engine, batch width, output
    # AND input tokens, memory share, KV precision, levers -- matches.
    # resume: false in the spec, or new_run: true on the enclosing
    # start request, discards the previous state and measures every
    # cell afresh.
    resume: bool = True
    confirm_winners: bool = True


class EnginePullRequest(BaseModel):
    engine: str                         # key in engine_runtimes.RUNTIMES


class StorageRequest(BaseModel):
    hf_cache: str      # absolute directory for model weights


class PromoteRequest(BaseModel):
    # "search" promotes the guided search's best candidate; "registry"
    # promotes the named sweep config (the UI passes its ranked #1).
    source: str
    config_name: Optional[str] = None
    # Promote from an ARCHIVED search (a runs/engine_optimizer/history
    # file name) instead of the live search.json.
    file: Optional[str] = None


class OptimizerStartRequest(BaseModel):
    # mode "arena": guided search over the full feasible space for
    #   this host (optionally narrowed by ``arena``).
    # mode "search": guided coarse-to-fine over a config/search/ YAML.
    # mode "registry": sweep a fixed set of hand-curated configs.
    # The default stays "registry" for wire compatibility (a bare
    # {"profile": ...} body must keep meaning a registry sweep); the
    # UI always sends mode explicitly and defaults to "arena" there.
    mode: str = "registry"
    profile: Optional[str] = None      # registry mode
    space: Optional[str] = None        # search mode: space name or path
    only: Optional[list[str]] = None   # registry mode: config subset
    # arena mode: {"models": [hf ids], "dims": {dim: [values]}} —
    # empty/absent means "everything feasible".
    arena: Optional[dict] = None
    budget: Optional[int] = None       # arena mode: evaluation budget
    new_run: bool = False
