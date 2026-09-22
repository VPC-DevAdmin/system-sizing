"""Roofline autopilot — what is the most this hardware can do?

A capacity benchmark asks "how many users does THIS deployment hold".
A roofline asks the opposite: given the box, which model and which
engine and which shape produce the largest sustained token rate at
all. It is a search over the whole product — models x engines x
launch shapes — and it takes hours, so everything here is built
around that fact rather than apologising for it.

Three consequences, and they shape the whole module:

* **Every step is written to disk before the next begins.** The
  operator will not be watching; they will close a laptop and come
  back over a VPN to a page that must simply be correct. There is no
  in-memory state that matters, so a reconnect is a GET, not a replay.

* **It resumes.** A run that dies at cell 23 of 40 must not repeat the
  first 22, because each of those cost minutes of engine launch. Cells
  are keyed by (model, engine, shape) and completed ones are skipped.

* **It reports a matrix, not a winner.** "Fastest on this box" is one
  number; which engine wins for which model is the finding that
  survives, because it is what tells you what to do with the NEXT
  model. The winner is a cell of that matrix, not a replacement for it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .bus import BUS
from .engines.knobs import GGUF_ENGINES

log = logging.getLogger(__name__)

STATE_NAME = "roofline.json"

# Default shape grid. Short outputs on purpose: a KV-heavy model is
# bound by cache traffic long before it is bound by compute, so long
# generations only move the peak down the concurrency ladder.
DEFAULT_SHAPES = {"max_num_seqs": [1024, 2048], "output_tokens": [128, 256]}
DEFAULT_INPUT_TOKENS = 128


# ── Model selection ───────────────────────────────────────────────────

def kv_bytes_per_token(model_id: str, cache: Path | None = None) -> Optional[int]:
    """KV cache cost of one token, read from the model's own config.

    2 x layers x kv_heads x head_dim x dtype_bytes. This is THE number
    that decides a headline: at large batch the KV traffic dwarfs the
    weight reads, so a model with 16x the KV cost per token cannot be
    rescued by having fewer active parameters.

    None when the model is not staged -- the config has to be on disk
    to be read, and guessing it from parameter count is how a 70B
    dense model gets mistaken for a cheap one.
    """
    from .models import hf_cache_dir
    base = Path(cache) if cache else hf_cache_dir()
    d = base / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return None
    for snap in sorted(d.iterdir()):
        cfg = snap / "config.json"
        if not cfg.is_file():
            continue
        try:
            c = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = c.get("text_config") or c
        layers = text.get("num_hidden_layers")
        heads = text.get("num_attention_heads")
        kv_heads = text.get("num_key_value_heads") or heads
        hidden = text.get("hidden_size")
        head_dim = text.get("head_dim") or (
            hidden // heads if hidden and heads else None)
        if not (layers and kv_heads and head_dim):
            continue
        # One byte per element: every model we rank here runs an fp8 or
        # nvfp4 KV cache. Ranking is by RATIO, so a constant factor
        # cannot change the order.
        return int(2 * layers * kv_heads * head_dim)
    return None


@dataclass
class Candidate:
    """One model, scored for its roofline potential."""
    id: str
    quant: str = ""
    # ``family`` groups the precision variants of one set of weights
    # (qwen3-30b-a3b); ``series`` is the vendor line those weights
    # belong to (Qwen3, gpt-oss, GLM-4.7). Diverse picking round-robins
    # over series so a shortlist spans vendors instead of being three
    # quantisations of the one model the ranker likes best.
    family: str = ""
    series: str = ""
    params_b: Optional[float] = None
    active_b: Optional[float] = None
    moe: bool = False
    size_gb: Optional[float] = None
    kv_bytes: Optional[int] = None
    cached: bool = False
    # Three ways to fit, and the search treats them differently.
    # ``fits_gpu``: the weights load across the box's cards (a GPU
    # engine at some tp). ``fits_ram``: the weights fit host RAM, the
    # CPU-expert budget. ``kt_eligible``: a GGUF companion is
    # catalogued, which is what the GGUF engines (KTransformers,
    # llama.cpp) actually load; ``gguf_engines`` is the companion's
    # allow-list among them. ``fits`` is what the roofline can measure
    # at all: fits_gpu, or a GGUF engine can carry it.
    fits_gpu: bool = True
    # ``kt_native``: a full native checkpoint is staged in a
    # quantisation the KTransformers v0.7 line reads directly (block
    # FP8, compressed-tensors INT4, MXFP4, bf16) -- the road to a 1T
    # model on this engine that needs no GGUF at all.
    kt_native: bool = False
    fits_ram: bool = True
    kt_eligible: bool = False
    gguf_engines: Optional[list] = None
    fits: bool = True
    # Smallest power-of-two tensor parallel that holds a replica, and
    # the replica count that leaves (gpu_count // tp). None when the
    # model does not fit the GPUs at all.
    tp: Optional[int] = None
    replicas: Optional[int] = None
    score: float = 0.0
    measured_kv: bool = False
    why: str = ""
    # Which pass of the series round-robin chose it (1 = best of its
    # series); 0 when it was not picked, or picking was not diverse.
    pick_round: int = 0
    # "fast" | "large" | "beyond_vram" once picked; "" otherwise.
    tier: str = ""

    def info(self) -> dict:
        """The part of a candidate a plan carries per model: enough
        for ``cells`` to choose engines and shape, and for
        ``summarize`` to draw the spectrum."""
        return {"tier": self.tier, "fits_gpu": self.fits_gpu,
                "fits_ram": self.fits_ram, "kt_eligible": self.kt_eligible,
                "kt_native": self.kt_native,
                "gguf_engines": list(self.gguf_engines or GGUF_ENGINES),
                "tp": self.tp, "replicas": self.replicas,
                "approx_size_gb": self.size_gb, "params_b": self.params_b,
                "vendor": vendor_of(self.series or self.family or self.id),
                "series": self.series, "quant": self.quant}


TIERS = ("fast", "large", "beyond_vram")

# How much of host RAM the CPU-resident experts may take. The rest is
# the OS, the KV cache KTransformers keeps on the CPU side, and the
# page cache the GGUF is read through.
RAM_SHARE = 0.85


def max_tp_of(hw: dict) -> Optional[int]:
    """The largest device group in a hardware dict, or None."""
    groups = hw.get("device_groups") or []
    return max((len(g) for g in groups), default=None) or None


def tp_for(need_gb: float, vram_per_gpu_gb: float, gpu_count: int = 8,
           max_tp: Optional[int] = None) -> Optional[int]:
    """Smallest power-of-two tensor parallel with tp x VRAM >= need,
    up to ``max_tp`` (the largest PCIe/NUMA device group -- TP peers
    never span domains, see config/arena.example.yaml) or the box;
    None when even that is not enough."""
    cap = min(int(gpu_count), int(max_tp)) if max_tp else int(gpu_count)
    tp = 1
    while tp <= cap:
        if tp * float(vram_per_gpu_gb) >= float(need_gb):
            return tp
        tp *= 2
    return None


def _native_kt_checkpoint(model_id: str, cache: Path | None = None) -> bool:
    """True when the model's newest cached snapshot holds safetensors
    in a quantisation ``kt_method_for`` names (the v0.7 line's native
    formats). Config-only stagings and GGUF-only models are False."""
    from .engines.ktransformers_v2 import _has_safetensors, kt_method_for
    from .models import _latest_snapshot, _model_dir, hf_cache_dir
    try:
        rev = _latest_snapshot(_model_dir(model_id, cache or hf_cache_dir()))
    except OSError:
        return False
    if rev is None or not _has_safetensors(rev):
        return False
    try:
        doc = json.loads((rev / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return kt_method_for(doc) is not None


def score_models(catalog: list[dict], *, vram_per_gpu_gb: float | None,
                 host_ram_gb: float | None = None, gpu_count: int = 8,
                 max_tp: Optional[int] = None,
                 cache: Path | None = None) -> list[Candidate]:
    """Rank catalog models by how fast they could plausibly go here.

    At saturation the box is reading, per generated token, the KV of
    every live sequence plus a share of the weights. The KV term wins
    by an order of magnitude at the batch sizes a headline runs at, so
    that is the primary key; weight footprint breaks ties because a
    smaller model leaves more VRAM for the cache that is the actual
    constraint.

    This is a PREDICTION and is labelled as one. It orders the search;
    it does not decide the answer.
    """
    from .model_catalog import infer_family
    from .models import model_status

    out: list[Candidate] = []
    for e in catalog:
        mid = str(e.get("id") or "")
        if not mid:
            continue
        st = model_status(mid, cache) if cache else model_status(mid)
        size = e.get("approx_size_gb")
        need = e.get("min_vram_gb") or size
        gpus = max(1, int(gpu_count or 8))
        fits_gpu, tp = True, 1
        cross_domain = False
        if vram_per_gpu_gb and need:
            # Whole-box: a model needing more than one card can still
            # run at tp>1, so this only excludes what will not fit the
            # box at all -- and records the tp that does fit it.
            tp = tp_for(float(need), float(vram_per_gpu_gb), gpus, max_tp)
            fits_gpu = tp is not None
            if tp and max_tp is None and tp > 4 and gpus > 4:
                cross_domain = True
        if e.get("kt_only"):
            # Staged config-only for the GGUF engines: its min_vram_gb
            # is the GPU share attention needs, not a weights
            # footprint. No GPU engine ever runs it.
            fits_gpu, tp = False, None
        # The GGUF engines keep the experts in host RAM and read them
        # from GGUF, so a model beyond VRAM is in reach only when both
        # the companion is catalogued and the weights fit the RAM.
        gguf_spec = e.get("gguf") or {}
        kt_eligible = bool(gguf_spec)
        gguf_engines = (list(gguf_spec.get("engines") or GGUF_ENGINES)
                        if gguf_spec else [])
        if kt_eligible and not e.get("moe") and "ktransformers" in gguf_engines:
            # The KTransformers serving image is an MoE engine (CPU
            # experts, GPU attention); a dense Llama dies at load with
            # KeyError: 'LlamaForCausalLM'. llama.cpp still serves it.
            gguf_engines = [g for g in gguf_engines if g != "ktransformers"]
        # A staged NATIVE checkpoint the v0.7 line reads (Kimi-K2-
        # Thinking's compressed-tensors INT4, DeepSeek's block FP8) is
        # its own road onto KTransformers, whatever the GGUF companion
        # allows: the companion's allow-list is about which engine can
        # read THAT file, not about the engine itself.
        kt_native = bool(e.get("moe")) and _native_kt_checkpoint(mid, cache)
        if kt_native:
            kt_eligible = True
            if "ktransformers" not in gguf_engines:
                gguf_engines = gguf_engines + ["ktransformers"]
        fits_ram = True
        if size:
            fits_ram = (host_ram_gb is not None
                        and float(size) <= float(host_ram_gb) * RAM_SHARE)
        family = str(e.get("family") or infer_family(mid))
        c = Candidate(
            id=mid, quant=str(e.get("quant") or ""),
            family=family,
            series=str(e.get("series") or family.split("-")[0]),
            params_b=e.get("params_b"), moe=bool(e.get("moe")),
            size_gb=size, cached=bool(st.get("cached")),
            kv_bytes=kv_bytes_per_token(mid, cache),
            fits_gpu=fits_gpu, fits_ram=fits_ram, kt_eligible=kt_eligible,
            kt_native=kt_native, gguf_engines=gguf_engines,
            fits=fits_gpu or (kt_eligible and fits_ram),
            tp=tp if fits_gpu else None,
            replicas=(gpus // tp) if fits_gpu else None,
        )
        bits = []
        # Primary: KV bytes per token, when we can read it.
        if c.kv_bytes:
            c.score = 1_000_000.0 / c.kv_bytes
            c.measured_kv = True
            bits.append(f"{c.kv_bytes // 1024} KiB of KV per token, "
                        f"read from the model's own config")
        elif c.params_b:
            # Unstaged: fall back to parameter count, and say so. A
            # dense model's KV scales with its layer count, which
            # tracks size loosely enough to order a shortlist.
            c.score = 200.0 / max(1.0, float(c.params_b)) ** 0.5
            bits.append("KV cost estimated from parameter count "
                        "(weights not staged yet)")
        if c.moe:
            c.score *= 1.35
            bits.append("MoE — fewer active parameters per token")
        if c.quant in ("nvfp4", "fp4", "mxfp4"):
            c.score *= 1.25
            bits.append(f"{c.quant} weights")
        elif c.quant == "fp8":
            c.score *= 1.1
            bits.append("fp8 weights")
        if size and vram_per_gpu_gb and float(size) <= float(vram_per_gpu_gb):
            c.score *= 1.15
            bits.append("fits one GPU — no tensor-parallel all-reduce")
        if cross_domain:
            bits.append(f"tp{tp} spans both PCIe domains — cross-domain "
                        "all-reduce; held on the GPUs, not tuned for speed")
        if not c.fits_gpu:
            # Out of the FAST race either way; the beyond-VRAM tier
            # may still pick it when KTransformers can carry it.
            c.score = 0.0
            if c.fits:
                bits = [f"beyond VRAM — {_gguf_engines_label(c.gguf_engines)} "
                        "only (GGUF companion catalogued, weights fit host "
                        "RAM)"]
            elif c.kt_eligible and host_ram_gb is None:
                bits = ["beyond VRAM and host RAM is unknown here"]
            elif c.kt_eligible:
                bits = [f"does not fit this box — {size:g} GB of weights "
                        f"exceed {RAM_SHARE:.0%} of {host_ram_gb:g} GB of RAM"]
            else:
                bits = ["does not fit this box (no GGUF companion for "
                        "KTransformers or llama.cpp)"]
        c.why = "; ".join(bits)
        out.append(c)
    # Measured beats estimated, always. A staged model whose KV cost
    # was read from its config is a known quantity; an unstaged one is
    # a guess from parameter count that systematically flatters small
    # models -- and acting on it costs a multi-gigabyte download before
    # anyone finds out. An operator who wants an unstaged model can
    # still name it directly.
    return sorted(out, key=lambda x: (not x.measured_kv, -x.score))


_ROUND_WORDS = {1: "best of", 2: "second pick from", 3: "third pick from",
                4: "fourth pick from", 5: "fifth pick from"}


def _round_label(n: int, series: str) -> str:
    word = _ROUND_WORDS.get(n, f"pick {n} from")
    return f"{word} the {series} line" if n == 1 else f"{word} {series}"


def vendor_of(series: str) -> str:
    """The vendor line behind a catalog ``series``: the leading word
    before any version -- "Qwen3" / "Qwen3.6" / "Qwen3 Next" -> "Qwen",
    "GLM-4.7" -> "GLM", "gpt-oss" -> "gpt-oss", "Llama 3.3" -> "Llama"."""
    m = re.match(r"[A-Za-z]+(?:-[A-Za-z]+)*", str(series or ""))
    return m.group(0) if m else str(series or "")


def pick_models(catalog: list[dict], *, vram_per_gpu_gb: float | None,
                host_ram_gb: float | None = None, gpu_count: int = 8,
                max_tp: Optional[int] = None,
                limit: int = 8, cached_only: bool = False,
                cache: Path | None = None,
                diverse: bool = True, spectrum: bool = True,
                large_limit: int = 3, beyond_limit: int = 2
                ) -> list[Candidate]:
    """The shortlist a roofline runs when the operator names no model.

    ``spectrum`` (the default) fills three tiers, in order, until
    ``limit``:

    * FAST -- the vendor round-robin below, over models that fit the
      GPUs. It gets ``limit - large_limit - beyond_limit`` slots first
      (never fewer than one) and whatever the other tiers leave.
    * LARGE -- the largest ``fits_gpu`` models by weight not already
      picked, one per vendor before any vendor's second, at most
      ``large_limit``. The fastest model on a box is rarely the most
      capable one; the operator wants to know what the biggest thing
      the cards can hold does, too.
    * BEYOND_VRAM -- models the GPUs cannot hold but KTransformers can
      (GGUF companion catalogued, weights within host RAM), largest
      first, at most ``beyond_limit``. These are a different kind of
      measurement (CPU-bound experts) and are labelled as such.

    Each pick's ``tier`` and ``why`` say which tier chose it and what
    it will cost (tp, RAM). ``spectrum=False`` is the FAST tier alone.

    ``diverse`` (the default) round-robins over VENDOR (``vendor_of``
    the series): the best candidate of every vendor in score order,
    then every vendor's second-best, and so on until ``limit``. The pure ranking had a
    failure mode the XE7740 hit on its first run: three quantisations
    of Qwen filled the shortlist and the roofline never looked at
    gpt-oss, GLM, Llama or the rest. A roofline is a search over what
    the box can do, and a search that only ever tries one vendor's
    weights has not searched.

    Within a vendor an unmeasured series is taken first, then a
    different ``family`` (different weights), and only last a second
    precision variant of weights already on the list -- the FP8 twin of a model already measured teaches
    less than a model nobody measured. "Measured beats estimated"
    stays the primary sort inside every series, and orders the
    series themselves in the first round.

    ``diverse=False`` is the old behaviour: the top ``limit`` of the
    global ranking, precision twins and all.
    """
    scored = score_models(catalog, vram_per_gpu_gb=vram_per_gpu_gb,
                          host_ram_gb=host_ram_gb, gpu_count=gpu_count,
                          max_tp=max_tp, cache=cache)
    if cached_only:
        scored = [c for c in scored if c.cached]
    ranked = [c for c in scored if c.fits_gpu]
    limit = max(1, limit)
    large_limit = max(0, large_limit) if spectrum else 0
    beyond_limit = max(0, beyond_limit) if spectrum else 0

    if not spectrum:
        return _pick_fast(ranked, limit, diverse)

    # The spectrum's extremes are settled FIRST, so a vendor's largest
    # model is not consumed as its "fastest" pick (GLM-5.3-Flash is the
    # large end of the GLM line; GLM-4.7-Flash is its fast end).
    picks: list[Candidate] = []

    # LARGE: biggest weights the cards hold, one per vendor first.
    large_pool = sorted((c for c in ranked if c.size_gb),
                        key=lambda c: -float(c.size_gb))
    large: list[Candidate] = []
    seen_vendors: set[str] = set()
    for pass_no in (1, 2):
        for c in large_pool:
            if len(large) >= large_limit or len(large) >= limit - 1:
                break
            v = vendor_of(c.series or c.family or c.id)
            if c in large or (pass_no == 1 and v in seen_vendors):
                continue
            seen_vendors.add(v)
            c.tier = "large"
            note = (f"largest that fits the GPUs: {float(c.size_gb):g} GB "
                    f"at tp{c.tp}")
            c.why = f"{note}; {c.why}" if c.why else note
            large.append(c)
    picks += large

    # BEYOND VRAM: what only the GGUF engines can serve, largest first.
    beyond_pool = sorted(
        (c for c in scored if not c.fits_gpu and c.kt_eligible and c.fits_ram
         and c.size_gb),
        key=lambda c: -float(c.size_gb))
    for c in beyond_pool[:beyond_limit]:
        if len(picks) >= limit - 1:
            break
        c.tier = "beyond_vram"
        c.why = (f"beyond VRAM — {_gguf_engines_label(c.gguf_engines)} only, "
                 f"{float(c.size_gb):g} GB of weights in "
                 f"{_ram_label(host_ram_gb)} of RAM")
        picks.append(c)

    # FAST: the vendor round-robin over what is left, filling the rest.
    chosen = {c.id for c in picks}
    fast = _pick_fast([c for c in ranked if c.id not in chosen],
                      max(1, limit - len(picks)), diverse)
    return fast + picks


def _ram_label(gb: float | None) -> str:
    if not gb:
        return "unknown"
    return f"{gb / 1000:g} TB" if gb >= 1000 else f"{gb:g} GB"


def _pick_fast(ranked: list[Candidate], limit: int, diverse: bool,
               already: list[Candidate] | None = None) -> list[Candidate]:
    """The FAST tier: ``limit`` more picks from ``ranked`` (score
    order, fits_gpu only). ``already`` are FAST picks from an earlier
    call, so a second pass keeps the vendor/series/family accounting
    of the first instead of restarting it."""
    if limit <= 0:
        return []
    if not diverse:
        out = ranked[:limit]
        for c in out:
            c.tier = "fast"
        return out

    # Vendors in order of their best candidate; score_models already
    # sorted with measured-first, so first-seen order is that order.
    # The catalog labels Qwen3, Qwen3.6 and Qwen3.8 as separate
    # series (they are different generations), but they are one
    # vendor's weights, and the operator's question -- "what can this
    # box do?" -- is not answered by three of them before one gpt-oss.
    by_vendor: dict[str, list[Candidate]] = {}
    for c in ranked:
        by_vendor.setdefault(vendor_of(c.series or c.family or c.id), []).append(c)

    prior = list(already or [])
    picks: list[Candidate] = []
    rnd = max((p.pick_round for p in prior), default=0)
    while len(picks) < limit and any(by_vendor.values()):
        rnd += 1
        for vendor, pool in by_vendor.items():
            if not pool:
                continue
            if len(picks) >= limit:
                break
            mine = [p for p in prior + picks
                    if vendor_of(p.series or p.family or p.id) == vendor]
            taken_series = {p.series for p in mine}
            taken_families = {p.family for p in mine}
            # Prefer a series nobody measured, then weights nobody
            # measured, then (last) a precision twin.
            fresh_series = [c for c in pool if c.series not in taken_series]
            fresh_family = [c for c in pool if c.family not in taken_families]
            choice = (fresh_series or fresh_family or pool)[0]
            pool.remove(choice)
            choice.pick_round = rnd
            choice.tier = "fast"
            note = _round_label(rnd, vendor)
            if not fresh_series and not fresh_family:
                note += f" (another precision of {choice.family})"
            choice.why = f"{note}; {choice.why}" if choice.why else note
            picks.append(choice)
    return picks


# ── Plan ──────────────────────────────────────────────────────────────

# Everything about a cell that changes what gets launched or offered,
# beyond the four the matrix is drawn over. All of it is in the cell
# key: a roofline restarted with a new prompt length, memory share or
# KV precision measures NEW cells instead of reusing old ones.
SHAPE_KEYS = ("input_tokens", "gpu_memory_utilization", "kv_cache_dtype",
              "replicas", "tp", "max_model_len")
LEVER_PREFIXES = ("trtllm_", "sglang_", "ktransformers_", "llamacpp_")


def _gguf_engines_label(engines: Optional[list]) -> str:
    """"KTransformers / llama.cpp" or whichever of them the companion
    allows -- for the plan's per-model reasons."""
    from .engines.knobs import ENGINE_LABELS
    names = list(engines or GGUF_ENGINES)
    return " / ".join(ENGINE_LABELS.get(e, e) for e in names)


def _is_shape_key(k: str) -> bool:
    return k in SHAPE_KEYS or k.startswith(LEVER_PREFIXES)


def shape_of(custom: dict) -> dict:
    """The launch-shaping fields of a custom request, for the plan."""
    return {k: v for k, v in custom.items()
            if _is_shape_key(k) and v not in (None, "")}


def ladder_for(engine: str, max_num_seqs: int | None,
               search: list[int] | None = None) -> list[int] | None:
    """The concurrency ladder a cell's sweep climbs. GPU engines take
    the caller's ladder (the coarse search rungs, or None for the
    sweep's full default). A GGUF engine serves at most its slot
    count -- KTransformers 4, llama-server 32 -- and a ladder that
    starts at 512 streams measured a queue, not the engine: 32 in
    flight, 480 waiting, readings from 108 to 3,280 tok/s on the same
    model and a second rung that produced nothing (XE7740, every
    llama.cpp cell of the giants pass). Such a cell climbs to its
    slots and one rung past them to show the ceiling.
    """
    if engine not in GGUF_ENGINES:
        return list(search) if search is not None else None
    n = max(1, int(max_num_seqs or 1))
    rungs = sorted({max(1, n // 4), max(1, n // 2), n, n * 2})
    return rungs


def engine_defaults(engine: str) -> dict:
    """Per-engine overrides of the roofline's GPU-engine defaults.

    The defaults (eight replicas, fp8 KV) describe a GPU-resident
    server. KTransformers is not one: its expert path wants every core
    and the whole memory bandwidth of the box, so a second replica
    contends rather than doubles, and it has no KV precision knob at
    all -- fp8 is refused, not ignored (knobs.unsupported). Without
    this every KTransformers cell failed at config time and the matrix
    showed a blank where the only engine able to serve a model larger
    than VRAM should be.

    llama.cpp gets the same shape for the same reasons: one replica
    spread by layer over its whole device group (with the experts
    offloaded, a second replica contends for the same memory
    bandwidth), and ``kv_cache_dtype`` auto because its q8_0 cache is
    a different quantity from the GPU engines' fp8 -- an operator who
    wants it names it.
    """
    if engine in GGUF_ENGINES:
        return {"replicas": 1, "kv_cache_dtype": "auto",
                "max_model_len": KT_MAX_MODEL_LEN}
    return {}


# KTransformers' CPU-side KV cache is sized by max_model_len and its
# whole point is a model that barely fits; 4k is the documented
# serving context and enough for the roofline's short shapes.
# llama-server's pool is max_model_len x slots (engines/llamacpp.py),
# so the same 4k keeps 32 slots to a 128k-token cache.
KT_MAX_MODEL_LEN = 4096


def engines_for(engine: str, info: dict | None) -> bool:
    """Whether a model with plan ``info`` (Candidate.info()) gets a
    cell on ``engine``. No info means the old behaviour: every engine.

    GPU engines need the weights on the cards; the GGUF engines
    (KTransformers, llama.cpp) need the GGUF companion they load from,
    and the companion's ``engines`` allow-list may name only one of
    them. A beyond-VRAM model therefore gets GGUF-engine cells only,
    and a model without a companion gets none of those at all --
    rather than a cell that fails at config time and leaves a blank
    nobody can read.
    """
    if not info:
        return True
    if engine in GGUF_ENGINES:
        return (bool(info.get("kt_eligible"))
                and engine in (info.get("gguf_engines") or GGUF_ENGINES))
    return bool(info.get("fits_gpu", True))


def cells(models: list[str], engines: list[str], shapes: dict, *,
          input_tokens: int = DEFAULT_INPUT_TOKENS,
          engine_shape: dict | None = None,
          model_info: dict[str, dict] | None = None,
          notes: list[str] | None = None) -> list[dict]:
    """Every (model, engine, shape) the run will measure, model-major.

    ``model_info`` (model id -> Candidate.info()) decides which engines
    each model gets (``engines_for``) and the tensor-parallel shape a
    GPU cell launches with: a model whose weights exceed one card runs
    at the smallest power-of-two tp that holds it, with ``gpu_count //
    tp`` replicas, instead of the eight-by-tp1 default. Skipped
    engines are explained in ``notes`` when a list is passed.

    Model-major because switching model costs a full weight load while
    switching engine does not, and because a partial run then holds a
    COMPLETE answer for the models it reached rather than a fragment
    of each.

    Each cell carries its full launch shape (``engine_shape`` plus the
    engine's own defaults), so the cell IS the record of what was
    measured. GGUF-engine cells are clamped to each engine's documented
    batch width (KTransformers 4, llama.cpp 32 slots) and deduplicated:
    two cells that would launch identically are one cell.
    """
    from .engines import ktransformers, llamacpp

    max_batch = {"ktransformers": ktransformers.DOCUMENTED_MAX_BATCH,
                 "llamacpp": llamacpp.DOCUMENTED_MAX_BATCH}
    mns = sorted(shapes.get("max_num_seqs") or DEFAULT_SHAPES["max_num_seqs"])
    outs = sorted(shapes.get("output_tokens") or DEFAULT_SHAPES["output_tokens"])
    out: list[dict] = []
    seen: set[str] = set()
    for m in models:
        info = (model_info or {}).get(m)
        for e in engines:
            if not engines_for(e, info):
                if notes is not None:
                    notes.append(_skip_note(m, e, info))
                continue
            shape = {**shape_of(engine_shape or {}), **engine_defaults(e)}
            if (info and e not in GGUF_ENGINES and info.get("tp")
                    and int(info["tp"]) > 1):
                shape["tp"] = int(info["tp"])
                shape["replicas"] = int(info.get("replicas") or 1)
            for n in mns:
                if e in max_batch:
                    n = min(n, max_batch[e])
                for o in outs:
                    cell = {"model": m, "engine": e, "max_num_seqs": n,
                            "output_tokens": o, "input_tokens": input_tokens,
                            **shape}
                    key = cell_key(cell)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(cell)
    return out


def _skip_note(model: str, engine: str, info: dict) -> str:
    if engine in GGUF_ENGINES:
        if info.get("kt_eligible"):
            allowed = info.get("gguf_engines") or GGUF_ENGINES
            return (f"{model}: no {engine} cells — its GGUF companion is "
                    f"marked for {', '.join(allowed)} only")
        return f"{model}: no {engine} cells — no GGUF companion staged"
    return (f"{model}: no {engine} cells — {info.get('approx_size_gb') or '?'}"
            f" GB of weights exceed the GPUs "
            f"({_gguf_engines_label(info.get('gguf_engines'))} only)")


def estimate_minutes(n_cells: int, *, launch_min: float = 6.0,
                     rungs: int = 4, rung_min: float = 1.5,
                     final_sweep_min: float = 18.0,
                     n_models: int = 1) -> int:
    """Wall clock, shown before anything is committed. Every cell pays
    an engine launch; each model additionally pays one confirmation
    sweep of its winner."""
    return int(n_cells * (launch_min + rungs * rung_min)
               + n_models * final_sweep_min)


# A cell that fails the same way twice is not unlucky, it is
# impossible on this host -- an engine that cannot load a model
# architecture, or a kernel that refuses the GPU. Retrying it on every
# resume costs an engine launch each time and never produces a number.
GIVE_UP_AFTER = 2

# Failures that say nothing about whether the cell CAN succeed: the
# engine was slow to come up, a port was still closing, the smoke
# request hit a server that was not quite ready, the sweep found no
# peak. Two of these in a row are two bad launches, not an
# impossibility, and the cell is retried on the next resume.
TRANSIENT_FAILURE = re.compile(
    r"TimeoutError|not healthy in|did not become healthy|"
    r"EADDRINUSE|address already in use|"
    r"smoke request|cannot serve requests|"
    r"produced no peak|unreadable sweep summary|"
    r"launch cancelled|timed out|Connect(ion|Error|Timeout)|"
    r"No such file or directory",          # staging/mount fault, not the model
    re.I)


def is_transient(err: str) -> bool:
    """A failure that must not write a cell off, however often it
    repeats."""
    return bool(err) and bool(TRANSIENT_FAILURE.search(err))


def error_signature(err: str) -> str:
    """The stable part of a failure, for deciding 'same way twice'.

    Run ids, ports, paths, replica indices and timings change between
    attempts; the exception type and its message do not. The same
    failure reported by replica 3 and by replica 5 is the same
    failure.
    """
    if not err:
        return ""
    # The "(full log: ...)" tail names a per-attempt file and is the
    # variable part by construction, so drop it whole rather than
    # trying to normalise what is inside it.
    s = re.sub(r"\(full log:[^)]*\)", "", err)
    s = re.sub(r"\breplica \d+\b", "replica N", s)
    s = re.sub(r"\[r\d+\]\s*", "", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*s\b", "Ns", s)         # 12.3s, 1800s
    s = re.sub(r"\b\d+(?:\.\d+)?\s*ms\b", "Nms", s)
    s = re.sub(r"run_\d+|:\d{4,5}\b|[0-9a-f]{8,}", "", s)
    return " ".join(s.split())[:160]


# A cell that dies for GPU memory at a fixed shape is not a dead
# model, it is a model that needs more cards per replica. The engine
# said so; the lever is tensor parallelism, and the run pulls it
# itself: the same cell is planned again at twice the tp (half the
# replicas), and again, up to the device group or the box. The model
# card's size is a guess about how an engine holds the weights;
# TensorRT-LLM loading a 63 GB NVFP4 checkpoint to 93 GB on the way
# to the GPU is the case that taught this. Triton's OutOfResources is
# NOT here: that is shared memory per SM, which no tp can buy.
MEMORY_FAILURE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|"
    r"insufficient GPU memory|"
    r"exceeds available|"                    # vLLM: max_num_seqs vs KV blocks
    r"not enough (?:GPU )?memory|"
    r"No available memory for the cache blocks|"
    r"free memory .* less than desired",
    re.I)


def is_memory_failure(err: str) -> bool:
    """A failure the engine attributes to GPU memory at this shape."""
    return bool(err) and bool(MEMORY_FAILURE.search(err))


SHARE_STEP = 0.05
SHARE_FLOOR = 0.80


def escalate_cell(cell: dict, *, gpu_count: int,
                  max_tp: Optional[int] = None,
                  weight_gb: float | None = None,
                  vram_gb: float | None = None) -> Optional[dict]:
    """The next cell to try after this one ran out of GPU memory, or
    None when every lever is spent. GGUF engines have no lever.

    Two levers, chosen by what the memory is going to:

    * **Weight-bound** (the weights per GPU take more than half the
      card, or their size is unknown): twice the tensor parallelism,
      ``gpu_count // tp`` replicas so the box stays full, up to the
      device group (or the box under cross-domain tp). TensorRT-LLM
      loading a 63 GB NVFP4 checkpoint to 93 GB at tp1 is this case.
    * **Not weight-bound** (a 14 GB model on a 96 GB card): the weights
      are not the problem, the 0.95 memory share is -- vLLM sizes the
      KV pool to it and the sampler's first softmax over 2048
      sequences then has nowhere to go (gpt-oss-20b and -120b at 2048
      seqs died this way at tp1, tp2 AND tp4). Step the share down by
      SHARE_STEP, to SHARE_FLOOR, at the same tp.

    When the preferred lever is spent the other one is tried, so a
    cell walks tp1 -> tp2 -> tp4 -> tp4@0.90 -> ... before it is
    given up on.
    """
    if cell.get("engine") in GGUF_ENGINES:
        return None
    tp = int(cell.get("tp") or 1)
    cap = min(int(gpu_count), int(max_tp)) if max_tp else int(gpu_count)
    share = cell.get("gpu_memory_utilization")

    def wider() -> Optional[dict]:
        if tp * 2 > cap:
            return None
        out = dict(cell)
        for k in ("error", "escalated_from", "escalated_from_share", "note"):
            out.pop(k, None)
        out["tp"] = tp * 2
        out["replicas"] = max(1, int(gpu_count) // (tp * 2))
        out["escalated_from"] = tp
        return out

    def leaner() -> Optional[dict]:
        if share in (None, ""):
            return None
        nxt = round(float(share) - SHARE_STEP, 2)
        if nxt < SHARE_FLOOR - 1e-9:
            return None
        out = dict(cell)
        for k in ("error", "escalated_from", "escalated_from_share", "note"):
            out.pop(k, None)
        out["gpu_memory_utilization"] = nxt
        out["escalated_from_share"] = float(share)
        return out

    weight_bound = (weight_gb is None or vram_gb is None
                    or float(weight_gb) / tp > 0.5 * float(vram_gb))
    first, second = (wider, leaner) if weight_bound else (leaner, wider)
    return first() or second()


def escalations(plan_cells: list[dict], results: list[dict], *,
                gpu_count: int, max_tp: Optional[int] = None,
                model_info: dict[str, dict] | None = None,
                vram_gb: float | None = None) -> list[dict]:
    """Cells the plan owes to memory failures already recorded: for
    every failed row that asks for more cards, the next tp step, unless
    the plan already has it. Called on resume so escalations survive a
    restart -- the plan is recomputed from the model list, the results
    are what remember which cells ran out of memory."""
    have = {cell_key(c) for c in plan_cells}
    out: list[dict] = []
    for r in results:
        if not is_memory_failure(r.get("error") or ""):
            continue
        info = (model_info or {}).get(r.get("model")) or {}
        nxt = escalate_cell(r, gpu_count=gpu_count, max_tp=max_tp,
                            weight_gb=info.get("approx_size_gb"),
                            vram_gb=vram_gb)
        if nxt is None:
            continue
        k = cell_key(nxt)
        if k in have:
            continue
        have.add(k)
        out.append({k2: v for k2, v in nxt.items()
                    if k2 in ("model", "engine", "max_num_seqs",
                              "output_tokens", "escalated_from",
                              "escalated_from_share")
                    or _is_shape_key(k2)})
    return out


def permanently_failed(results: list[dict]) -> dict[str, str]:
    """Cells that have failed identically at least GIVE_UP_AFTER times
    for a reason that is not transient, mapped to the reason.
    Reported, not hidden: a blank in the matrix with a cause beside it
    is a finding."""
    seen: dict[str, list[str]] = {}
    for r in results:
        if not r.get("error") or is_transient(r["error"]):
            continue
        seen.setdefault(cell_key(r), []).append(error_signature(r["error"]))
    out = {}
    for key, sigs in seen.items():
        for sig in set(sigs):
            # Out of memory at a fixed shape is deterministic, and the
            # run has already planned the same cell wider; a second
            # launch at this shape would only spend the launch.
            need = 1 if is_memory_failure(sig) else GIVE_UP_AFTER
            if sigs.count(sig) >= need:
                out[key] = sig
                break
    return out


def cell_overrides(cell: dict) -> dict:
    """What a cell tells the config builder: model, engine, batch width
    and every launch-shaping field it carries. The builder merges these
    over the roofline's base defaults, so an engine's own defaults
    (engine_defaults) win over the GPU-engine ones."""
    out = {"model_id": cell["model"], "engine": cell["engine"],
           "max_num_seqs": cell["max_num_seqs"]}
    out.update({k: v for k, v in cell.items()
                if _is_shape_key(k) and k != "input_tokens"
                and v not in (None, "")})
    return out


def cell_key(c: dict) -> str:
    """Identity of a cell for resume: the four matrix axes plus every
    launch-shaping field the cell carries (SHAPE_KEYS and levers).

    Rows written before the shape fields existed lack them and so key
    differently from any cell planned now -- they are re-measured,
    which is the honest outcome: nobody knows what prompt length or
    KV precision they ran with.
    """
    base = (f"{c['model']}|{c['engine']}|{c['max_num_seqs']}"
            f"|{c['output_tokens']}")
    extra = sorted((k, c[k]) for k in c
                   if _is_shape_key(k) and c[k] not in (None, ""))
    if not extra:
        return base
    return base + "|" + "|".join(f"{k}={v}" for k, v in extra)


# ── State (the only thing that matters across a disconnect) ───────────

@dataclass
class State:
    status: str = "planning"
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    input_tokens: int = DEFAULT_INPUT_TOKENS
    plan: dict = field(default_factory=dict)
    estimate_min: int = 0
    models: list = field(default_factory=list)
    results: list = field(default_factory=list)      # completed cells
    current: dict | None = None
    error: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "roofline"
        d["done"] = self.status in ("finished", "failed", "stopped")
        d["summary"] = summarize(
            self.results, model_info=self.plan.get("model_info"),
            models=self.plan.get("models"),
            staging={m.get("id"): m.get("status") for m in self.models})
        d["written_off"] = permanently_failed(self.results)
        return d


def _generation(r: dict) -> float:
    """The rate every ranking uses: GENERATION tokens per second. Total
    (prompt + generation) is carried alongside for the prefill story,
    but it flatters short outputs and prefill-heavy windows -- an
    85k 'total' on a 70B was 38k generated -- and generation is the
    number the vendor convention and this team's history both mean."""
    return float(r.get("out_tok_s") or 0)


def summarize(results: list[dict], model_info: dict | None = None,
              models: list[str] | None = None,
              staging: dict | None = None) -> dict:
    """The matrix, plus the cells that won it -- in two directions.

    Deliberately reports best-per-model and best-per-engine alongside
    the overall winner: a single number tells you what to publish,
    while the matrix tells you what to do with the next model you try.

    A spectrum search has two winners, not one: ``fastest`` (the cell
    with the highest GENERATION token rate; total is reported beside
    it, never ranked on) and ``largest_served`` (the
    biggest model any engine actually produced a peak for), and the
    ``spectrum`` in between -- one row per planned model, by weight,
    with its best cell or the reason it has none. ``model_info`` is
    the plan's per-model record (weights, params, tier); without it
    the spectrum rows carry only what the results say.
    """
    usable = [r for r in results
              if r.get("out_tok_s") and r.get("steady_state", True)]
    by_model: dict[str, dict] = {}
    by_engine: dict[str, dict] = {}
    for r in sorted(usable, key=lambda x: -(x["out_tok_s"] or 0)):
        by_model.setdefault(r["model"], r)
        by_engine.setdefault(r["engine"], r)
    best = max(usable, key=lambda r: r["out_tok_s"], default=None)
    fastest = max(usable, key=_generation, default=None)

    info = model_info or {}
    order: list[str] = list(models or [])
    for r in results:
        if r.get("model") and r["model"] not in order:
            order.append(r["model"])
    failed_models = {r["model"] for r in results if r.get("error")}
    best_total: dict[str, dict] = {}
    for r in sorted(usable, key=_generation, reverse=True):
        best_total.setdefault(r["model"], r)

    def size_key(m: str) -> tuple[float, float]:
        i = info.get(m) or {}
        return (float(i.get("approx_size_gb") or 0),
                float(i.get("params_b") or 0))

    spectrum = []
    for m in order:
        i = info.get(m) or {}
        b = best_total.get(m)
        if b:
            status = "served"
        elif (staging or {}).get(m) == "unavailable":
            status = "unavailable"
        elif m in failed_models:
            status = "failed"
        else:
            status = "pending"
        spectrum.append({
            "model": m, "vendor": i.get("vendor"),
            "params_b": i.get("params_b"),
            "approx_size_gb": i.get("approx_size_gb"),
            "tier": i.get("tier") or "",
            "best_engine": b.get("engine") if b else None,
            "out_tok_s": b.get("out_tok_s") if b else None,
            "total_tok_s": (b.get("total_tok_s") or b.get("out_tok_s"))
            if b else None,
            "concurrency": (b.get("concurrency") or b.get("in_flight"))
            if b else None,
            "kv_capacity_tokens": b.get("kv_cache_tokens") if b else None,
            "ttft_p95_ms": b.get("ttft_p95_ms") if b else None,
            "status": status,
        })
    spectrum.sort(key=lambda row: size_key(row["model"]))

    def largest(among: list[str]) -> dict | None:
        if not among:
            return None
        m = max(among, key=size_key)
        row = next((s for s in spectrum if s["model"] == m), None)
        return dict(row) if row else {"model": m}

    return {
        "best": best,
        "fastest": fastest,
        "largest_served": largest(list(best_total)),
        "largest_attempted": largest(
            [m for m in order if m in best_total or m in failed_models]),
        "spectrum": spectrum,
        "best_per_model": by_model,
        "best_per_engine": by_engine,
        "measured": len(usable),
        "attempted": len(results),
        "failed": [r for r in results if r.get("error")],
    }


def load_state(path: Path) -> Optional[State]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    known = {f for f in State.__dataclass_fields__}
    return State(**{k: v for k, v in d.items() if k in known})


def save_state(path: Path, st: State) -> None:
    """Write atomically. A half-written state read by a reconnecting
    browser is worse than a slightly stale one."""
    st.updated_at = time.time()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st.to_dict(), indent=2))
    tmp.replace(p)


# ── Staging ───────────────────────────────────────────────────────────

async def ensure_staged(models: list[str], st: State, state_path: Path,
                        *, log_dir: Path) -> list[str]:
    """Download anything missing, and report progress while it happens.

    Weights are tens of gigabytes. Doing this inside the run rather
    than demanding it beforehand is the difference between an autopilot
    and a checklist -- but a model that will not download must not take
    the whole run down with it, so failures drop the model and carry
    on with the rest.
    """
    import asyncio
    import subprocess

    from .models import download_command, model_status

    ready: list[str] = []
    for mid in models:
        entry = next((m for m in st.models if m["id"] == mid), None)
        if model_status(mid).get("cached"):
            if entry:
                entry["status"] = "cached"
            ready.append(mid)
            save_state(state_path, st)
            continue
        if entry:
            entry["status"] = "downloading"
        st.note = f"staging weights for {mid}"
        save_state(state_path, st)
        log.info("roofline: downloading %s", mid)
        try:
            argv, env = download_command(mid)
            import os
            log_dir.mkdir(parents=True, exist_ok=True)
            lf = open(log_dir / f"download_{mid.replace('/', '--')}.log", "w")
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=lf, stderr=subprocess.STDOUT,
                env={**os.environ, **env})
            rc = await proc.wait()
        except Exception as e:  # noqa: BLE001
            rc, e_msg = 1, str(e)
            log.warning("roofline: download of %s failed: %s", mid, e)
        else:
            e_msg = f"downloader exited {rc}"
        if rc == 0 and model_status(mid).get("cached"):
            if entry:
                entry["status"] = "cached"
            ready.append(mid)
        else:
            if entry:
                entry["status"] = "unavailable"
                entry["error"] = e_msg
            log.warning("roofline: skipping %s — %s", mid, e_msg)
        save_state(state_path, st)
    return ready


# ── The run ───────────────────────────────────────────────────────────

async def run_roofline(
    *,
    models: list[str],
    engines: list[str],
    shapes: dict | None = None,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    build_config,
    runs_base: Path,
    state_path: Path | None = None,
    resume: bool = True,
    confirm_winners: bool = True,
    engine_shape: dict | None = None,
    model_info: dict[str, dict] | None = None,
    retry_engines: list[str] | None = None,
    redo_engines: list[str] | None = None,
    gpu_count: int = 8,
    max_tp: Optional[int] = None,
    vram_per_gpu_gb: float | None = None,
) -> Path:
    """Stage, search the product, confirm each model's winner, report.

    ``gpu_count`` and ``max_tp`` bound the tp escalation: a cell that
    runs out of GPU memory is planned again at twice its tp (see
    ``escalate_cell``) until the largest device group, or the box when
    cross-domain tp is allowed, is reached.

    ``engine_shape`` is the launch shape shared by every cell (memory
    share, KV precision, replica count, levers); it is recorded on
    each cell and is part of the cell's resume identity.

    ``model_info`` (model id -> Candidate.info()) is the plan's record
    of each model: its tier, whether it fits the GPUs or only
    KTransformers, and the tp that holds it. It decides which engines
    each model gets (``cells``) and is what the spectrum is drawn from.

    ``build_config`` is injected exactly as the joint search does it:
    it takes engine overrides and returns a config path, so this module
    never learns how configs are generated.
    """
    from .headline_shapes import apply_shape_to_generation
    from .persona_loader import USER_CATALOG_DIR

    shapes = shapes or dict(DEFAULT_SHAPES)
    path = Path(state_path or (Path(runs_base) / STATE_NAME))

    st = load_state(path) if resume else None
    done: dict[str, dict] = {}
    hopeless: dict[str, str] = {}
    if st and st.plan.get("cells"):
        if retry_engines:
            # The operator fixed something for these engines (a memory
            # reserve, a port handoff): forget their failures, keep their
            # measurements, and let resume run the cells again.
            before = len(st.results)
            st.results = [r for r in st.results
                          if not (r.get("error") and r.get("engine") in retry_engines)]
            log.info("roofline: retrying %d failed cells on %s",
                     before - len(st.results), ", ".join(retry_engines))
        if redo_engines:
            # A measurement known to be wrong (a ladder that measured
            # a queue) is forgotten whether or not it errored; the
            # cells run again on the next pass.
            before = len(st.results)
            st.results = [r for r in st.results
                          if r.get("engine") not in redo_engines]
            log.info("roofline: redoing %d cells on %s",
                     before - len(st.results), ", ".join(redo_engines))
        done = {cell_key(r): r for r in st.results if not r.get("error")}
        hopeless = permanently_failed(st.results)
        log.info("roofline: resuming with %d cells measured, %d written off",
                 len(done), len(hopeless))
    else:
        st = State()

    notes: list[str] = []
    plan_cells = cells(models, engines, shapes, input_tokens=input_tokens,
                       engine_shape=engine_shape, model_info=model_info,
                       notes=notes)
    owed = escalations(plan_cells, st.results, gpu_count=gpu_count,
                       max_tp=max_tp, model_info=model_info,
                       vram_gb=vram_per_gpu_gb)
    if owed:
        log.info("roofline: %d tp escalations owed to earlier memory "
                 "failures", len(owed))
        plan_cells = plan_cells + owed
    st.status = "staging"
    st.input_tokens = input_tokens
    st.plan = {"models": models, "engines": engines, "shapes": shapes,
               "cells": plan_cells, "model_info": model_info or {},
               "notes": notes}
    st.estimate_min = estimate_minutes(len(plan_cells) - len(done),
                                       n_models=len(models))
    if not st.models:
        st.models = [{"id": m, "status": "pending"} for m in models]
    st.error = None
    save_state(path, st)
    BUS.publish("run", {"event": "started", "mode": "roofline",
                        "models": models, "engines": engines,
                        "cells": len(plan_cells)})

    staged = await ensure_staged(models, st, path,
                                 log_dir=Path(runs_base) / "roofline")
    if not staged:
        st.status = "failed"
        st.error = "no model could be staged"
        save_state(path, st)
        return path

    from .config import load_config
    from .headline_sweep import run_headline_sweep
    from .personas import cohort_from_persona

    st.status = "searching"
    st.note = ""
    save_state(path, st)

    # A worklist rather than a plain loop: a memory failure appends
    # the same cell one tp step wider, right behind the failed one so
    # the model's answer is still complete before the next model.
    work = list(plan_cells)
    i = 0
    while i < len(work):
        cell = work[i]
        i += 1
        if cell["model"] not in staged:
            continue
        key = cell_key(cell)
        if key in done:
            continue
        if key in hopeless:
            # Failed the same way twice already. Retrying costs an
            # engine launch and cannot succeed; the matrix keeps the
            # blank and the reason.
            log.info("roofline: skipping %s — %s", key, hopeless[key][:90])
            continue
        st.current = dict(cell)
        save_state(path, st)
        log.info("roofline: %s / %s mns=%d out=%d", cell["model"],
                 cell["engine"], cell["max_num_seqs"], cell["output_tokens"])
        row = dict(cell)
        try:
            apply_shape_to_generation(USER_CATALOG_DIR, input_tokens,
                                      cell["output_tokens"])
            cfg_path = build_config(cell_overrides(cell))
            sub = load_config(cfg_path)
            sub.output.db_directory = str(runs_base)
            # Coarse ladder while RANKING: the peak sits at the top
            # of the curve, so the low rungs cost minutes and teach
            # nothing. The winner earns the full ladder below.
            from .headline_optimize import SEARCH_LADDER
            summary = await run_headline_sweep(
                sub, cohort_from_persona("headline_generation"),
                new_run=True,
                ladder_override=ladder_for(cell["engine"], cell["max_num_seqs"],
                                           SEARCH_LADDER))
            row.update(_peak_of(summary))
        except Exception as e:  # noqa: BLE001
            # One dead cell must not end a six-hour run.
            row["error"] = f"{type(e).__name__}: {e}"
            log.warning("roofline cell failed: %s", e)
            if is_memory_failure(row["error"]):
                minfo = (model_info or {}).get(cell["model"]) or {}
                nxt = escalate_cell(cell, gpu_count=gpu_count, max_tp=max_tp,
                                    weight_gb=minfo.get("approx_size_gb"),
                                    vram_gb=vram_per_gpu_gb)
                if nxt is not None and cell_key(nxt) not in {
                        cell_key(c) for c in work}:
                    log.info("roofline: %s / %s out of memory at tp%d@%s, "
                             "planning tp%d x %d replicas @%s",
                             cell["model"], cell["engine"],
                             int(cell.get("tp") or 1),
                             cell.get("gpu_memory_utilization"),
                             nxt["tp"], nxt["replicas"],
                             nxt.get("gpu_memory_utilization"))
                    work.insert(i, nxt)
                    st.plan["cells"] = list(st.plan.get("cells") or []) + [nxt]
                elif nxt is None:
                    row["note"] = (f"out of GPU memory at tp{int(cell.get('tp') or 1)}"
                                   f"@{cell.get('gpu_memory_utilization')}; "
                                   f"no wider tp or leaner share left")
        st.results.append(row)
        st.current = None
        save_state(path, st)

    if confirm_winners:
        st.status = "confirming"
        save_state(path, st)
        for mid, best in (summarize(st.results).get("best_per_model")
                          or {}).items():
            # Resume: a winner already confirmed at this exact shape
            # is not swept again -- eleven sweeps cost five hours.
            bk = cell_key(best)
            if any(r.get("confirmed") and not r.get("error")
                   and cell_key(r) == bk for r in st.results):
                continue
            st.current = {**best, "phase": "confirming"}
            save_state(path, st)
            try:
                apply_shape_to_generation(USER_CATALOG_DIR, input_tokens,
                                          best["output_tokens"])
                cfg_path = build_config(cell_overrides(best))
                sub = load_config(cfg_path)
                sub.output.db_directory = str(runs_base)
                final = await run_headline_sweep(
                    sub, cohort_from_persona("headline_generation"),
                    new_run=True,
                    ladder_override=ladder_for(best["engine"],
                                               best["max_num_seqs"]))
                row = {**best, "confirmed": True, **_peak_of(final)}
                st.results.append(row)
            except Exception as e:  # noqa: BLE001
                log.error("roofline confirmation for %s failed: %s", mid, e)
            st.current = None
            save_state(path, st)

    st.status = "finished"
    st.note = ("Search rungs rank candidates; the confirmed rows are the "
               "ones to publish.")
    save_state(path, st)
    BUS.publish("run", {"event": "finished", "mode": "roofline",
                        "final_status": "ok"})
    log.info("roofline done: %s", summarize(st.results).get("best"))
    return path


def _peak_of(summary_path) -> dict:
    """The peak a sweep recorded, flattened into a result row."""
    try:
        doc = json.loads(Path(summary_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"unreadable sweep summary: {e}"}
    pk = doc.get("peak") or {}
    if not pk:
        return {"error": doc.get("stop_reason") or "sweep produced no peak"}
    return {
        "out_tok_s": pk.get("out_tok_s"),
        "total_tok_s": pk.get("total_tok_s"),
        "in_flight": pk.get("in_flight"),
        "concurrency": pk.get("concurrency"),
        "queue_depth": pk.get("queue_depth"),
        "ttft_p95_ms": pk.get("ttft_p95_ms"),
        "tpot_p95_ms": pk.get("tpot_p95_ms"),
        "kv_cache_pct": pk.get("kv_cache_pct"),
        "gpu_power_w": pk.get("gpu_power_w"),
        "steady_state": pk.get("steady_state", True),
        "kv_cache_tokens": doc.get("kv_cache_tokens"),
        "run_dir": str(Path(summary_path).parent),
        "tokens_per_watt": (
            round(pk["out_tok_s"] / pk["gpu_power_w"], 2)
            if pk.get("out_tok_s") and pk.get("gpu_power_w") else None),
    }
