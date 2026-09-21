"""Model catalog — models as data, additions as one line of YAML.

Profiles and search spaces used to be the only places a model could be
named, which made "try a new model" a hand-edit across files. The
catalog makes models first-class: a packaged starter set
(``simulator/models_data/catalog.yaml``) plus local overlays in
``config/models/*.yaml`` (the UI's "Add model" writes
``config/models/local.yaml``). Everything downstream keys off it:

  * ``referenced_models`` lists catalog entries next to profile/space
    references, so the download gate accepts them (the catalog is
    explicit operator data — this does NOT turn the service into a
    general download proxy).
  * Search spaces can pull a whole precision family via
    ``catalog_families``, so a newly added model is searchable without
    editing the space's variant map.
  * ``suggest_quant_siblings`` guesses the conventional quantized
    artifact names for a base model (``-FP8``, ``-AWQ``, RedHatAI
    cross-org quants) and, network permitting, verifies which actually
    exist on the Hub — so adding a model surfaces its quants too.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

PACKAGED_CATALOG = Path(__file__).parent / "models_data" / "catalog.yaml"
USER_CATALOG_DIR = Path("config/models")
LOCAL_CATALOG_NAME = "local.yaml"

# org/name — HF repo ids are exactly two path segments.
_MODEL_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")

KNOWN_QUANTS = ("bf16", "fp8", "awq", "gptq-int4", "gptq-int8",
                "mxfp4", "nvfp4", "int8", "other")

# (regex on the repo NAME, quant) — first match wins.
_QUANT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"fp8", re.I), "fp8"),
    (re.compile(r"awq", re.I), "awq"),
    (re.compile(r"gptq.?int4|w4a16|int4", re.I), "gptq-int4"),
    (re.compile(r"gptq.?int8|w8a16", re.I), "gptq-int8"),
    (re.compile(r"nvfp4", re.I), "nvfp4"),
    (re.compile(r"mxfp4", re.I), "mxfp4"),
    (re.compile(r"int8|w8a8", re.I), "int8"),
]


class CatalogError(ValueError):
    """A catalog file or entry that doesn't validate."""


def _normalize_gguf(raw: object, source: str, model_id: str) -> Optional[dict]:
    """The optional GGUF companion: ``{repo, file, size_gb, engines}``
    or None.

    KTransformers and llama.cpp load weights from GGUF only, so a
    catalog entry that names its companion is what lets Prepare stage
    it and the launchers resolve their GGUF path without the operator
    hand-editing one. ``file`` is one ``.gguf`` (it may carry a
    subdirectory when the repo shards its quants into folders), or a
    DIRECTORY when the quant itself is split into ``-00001-of-0000N``
    shards -- a 400 GB DeepSeek quant is nine files, KTransformers
    reads every .gguf under ``--gguf_path`` and llama-server opens the
    rest from the first shard, so the directory is the unit Prepare
    stages and the launchers mount.

    ``engines`` is the allow-list of GGUF engines that can serve it,
    defaulting to both: an architecture the KTransformers v0.3.2 image
    has no injection rule for (deepseek_v32, kimi_k2, glm_moe_dsa,
    minimax_m2) names ``[llamacpp]`` so the roofline never schedules
    a KTransformers cell that dies at load."""
    if raw in (None, {}, ""):
        return None
    if not isinstance(raw, dict):
        raise CatalogError(f"{source}: '{model_id}' gguf must be a mapping "
                           f"with repo and file")
    repo = str(raw.get("repo") or "")
    file = str(raw.get("file") or "").strip("/")
    if not _MODEL_ID_RE.match(repo):
        raise CatalogError(f"{source}: '{model_id}' gguf.repo '{repo}' is "
                           f"not an org/name HF repo id")
    parts = file.split("/")
    if not file or ".." in parts or "." in parts or any(not x for x in parts):
        raise CatalogError(f"{source}: '{model_id}' gguf.file '{file}' must "
                           f"name a .gguf file or a shard directory inside "
                           f"the repo")
    if "." in parts[-1] and not file.endswith(".gguf"):
        raise CatalogError(f"{source}: '{model_id}' gguf.file '{file}' must "
                           f"name a .gguf file inside the repo")
    size = raw.get("size_gb")
    # Lazy: the engines package pulls in every launcher, and the
    # catalog must stay importable without them.
    from .engines.knobs import GGUF_ENGINES
    engines_raw = raw.get("engines")
    if engines_raw in (None, ""):
        engines = list(GGUF_ENGINES)
    else:
        if isinstance(engines_raw, str):
            engines_raw = [engines_raw]
        if (not isinstance(engines_raw, list) or not engines_raw
                or any(str(e) not in GGUF_ENGINES for e in engines_raw)):
            raise CatalogError(
                f"{source}: '{model_id}' gguf.engines must be a non-empty "
                f"list drawn from {list(GGUF_ENGINES)}, got {engines_raw!r}")
        engines = [e for e in GGUF_ENGINES if e in {str(x) for x in engines_raw}]
    return {"repo": repo, "file": file,
            "size_gb": float(size) if size is not None else None,
            "engines": engines}


def _normalize_host_ram(raw: object, source: str, model_id: str) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise CatalogError(f"{source}: '{model_id}' host_ram_gb must be a "
                           f"number of GB") from None
    if val <= 0:
        raise CatalogError(f"{source}: '{model_id}' host_ram_gb must be > 0")
    return val


def infer_quant(model_id: str) -> str:
    name = model_id.split("/")[-1]
    for pat, quant in _QUANT_PATTERNS:
        if pat.search(name):
            return quant
    return "bf16"


def infer_family(model_id: str) -> str:
    """Family slug from a repo name: strip the org and any quant
    suffix, lowercase. ``Qwen/Qwen3-32B-FP8`` -> ``qwen3-32b``."""
    name = model_id.split("/")[-1]
    for pat, _ in _QUANT_PATTERNS:
        name = pat.sub("", name)
    return re.sub(r"[-_]+", "-", name).strip("-.").lower()


def _normalize_entry(raw: dict, source: str) -> dict:
    if not isinstance(raw, dict) or not raw.get("id"):
        raise CatalogError(f"{source}: every entry needs an 'id'")
    model_id = str(raw["id"])
    if not _MODEL_ID_RE.match(model_id):
        raise CatalogError(
            f"{source}: '{model_id}' is not an org/name HF repo id"
        )
    quant = str(raw.get("quant") or infer_quant(model_id))
    if quant not in KNOWN_QUANTS:
        raise CatalogError(
            f"{source}: '{model_id}' quant '{quant}' unknown — "
            f"known: {KNOWN_QUANTS}"
        )
    name = model_id.split("/")[-1]
    gguf = _normalize_gguf(raw.get("gguf"), source, model_id)
    kt_only = bool(raw.get("kt_only", False))
    if kt_only and not gguf:
        raise CatalogError(f"{source}: '{model_id}' is kt_only but names no "
                           f"gguf companion -- KTransformers has nothing to "
                           f"load")
    return {
        "id": model_id,
        "family": str(raw.get("family") or infer_family(model_id)),
        # Vendor family line as an operator says it ("Qwen3") — the
        # UI's family dropdown groups by this.
        "series": str(raw.get("series") or infer_family(model_id).split("-")[0]),
        "params_b": raw.get("params_b"),
        "specialty": str(raw.get("specialty")
                         or ("coder" if "coder" in name.lower() else "instruct")),
        "quant": quant,
        "approx_size_gb": raw.get("approx_size_gb"),
        "min_vram_gb": raw.get("min_vram_gb"),
        "gated": bool(raw.get("gated", False)),
        "moe": bool(raw.get("moe", False)),
        "engine_args": list(raw.get("engine_args") or []),
        "notes": str(raw.get("notes") or ""),
        "gguf": gguf,
        # The weights exceed the GPUs outright: GPU engines never run
        # this one, only KTransformers does -- which needs the GGUF.
        "kt_only": kt_only,
        "host_ram_gb": _normalize_host_ram(raw.get("host_ram_gb"),
                                           source, model_id),
        "source": source,
    }


def _load_file(path: Path) -> list[dict]:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise CatalogError(f"{path}: {e}") from e
    entries = raw.get("models")
    if not isinstance(entries, list):
        raise CatalogError(f"{path}: top-level 'models' list required")
    return [_normalize_entry(e, str(path)) for e in entries]


def load_model_catalog(
    user_dir: Path | str = USER_CATALOG_DIR,
) -> list[dict]:
    """Merged catalog: packaged defaults, then ``config/models/*.yaml``
    overlays in filename order. Later files win by id, so a local file
    can amend a packaged entry (e.g. add engine_args)."""
    merged: dict[str, dict] = {}
    files = [PACKAGED_CATALOG] if PACKAGED_CATALOG.exists() else []
    d = Path(user_dir)
    if d.exists():
        files += sorted(d.glob("*.yaml"))
    for f in files:
        for entry in _load_file(f):
            merged[entry["id"]] = entry
    return sorted(merged.values(), key=lambda e: (e["family"], e["quant"], e["id"]))


def catalog_families(user_dir: Path | str = USER_CATALOG_DIR) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in load_model_catalog(user_dir):
        out.setdefault(e["family"], []).append(e)
    return out


def add_catalog_model(
    model_id: str,
    *,
    family: Optional[str] = None,
    quant: Optional[str] = None,
    notes: str = "",
    user_dir: Path | str = USER_CATALOG_DIR,
    # Rich metadata (the discovery flow supplies these; manual adds
    # can omit them and edit the overlay file later).
    series: Optional[str] = None,
    params_b: Optional[float] = None,
    moe: Optional[bool] = None,
    approx_size_gb: Optional[float] = None,
    min_vram_gb: Optional[float] = None,
    specialty: Optional[str] = None,
) -> tuple[dict, bool]:
    """Add one model to the local overlay (``config/models/local.yaml``).
    Idempotent: an id already in the merged catalog returns
    ``(existing_entry, False)`` untouched. Returns ``(entry, True)``
    on a new addition. Raises CatalogError on a bad id/quant."""
    rich = {k: v for k, v in {
        "series": series, "params_b": params_b, "moe": moe,
        "approx_size_gb": approx_size_gb, "min_vram_gb": min_vram_gb,
        "specialty": specialty,
    }.items() if v is not None}
    entry = _normalize_entry(
        {"id": model_id, "family": family, "quant": quant,
         "notes": notes, **rich},
        source="add",
    )
    existing = {e["id"]: e for e in load_model_catalog(user_dir)}
    if entry["id"] in existing:
        return existing[entry["id"]], False

    local = Path(user_dir) / LOCAL_CATALOG_NAME
    local.parent.mkdir(parents=True, exist_ok=True)
    doc = {"models": []}
    if local.exists():
        try:
            doc = yaml.safe_load(local.read_text()) or {"models": []}
        except yaml.YAMLError as e:
            raise CatalogError(f"{local} is not valid YAML: {e}") from e
        if not isinstance(doc.get("models"), list):
            doc = {"models": []}
    record = {"id": entry["id"], "family": entry["family"],
              "quant": entry["quant"], **rich}
    if notes:
        record["notes"] = notes
    doc["models"].append(record)
    local.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    entry["source"] = str(local)
    return entry, True


# ── Hub sibling discovery ────────────────────────────────────────────


def hub_model_exists(model_id: str, timeout: float = 5.0) -> Optional[bool]:
    """Does the repo exist on the Hub? None when the check itself
    failed (offline box, proxy) — callers treat that as 'unverified',
    never as 'missing'.

    Status mapping, measured against the live API: gated repos return
    200 with public metadata; NONEXISTENT repos return 401 (not 404)
    to unauthenticated callers, and private repos look identical — so
    401/404 both mean "not usable by this host". The operator's HF
    token rides along when set, so their own private/accepted-gated
    repos verify correctly."""
    import os as _os

    import httpx
    headers = {}
    token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = httpx.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=timeout, follow_redirects=True, headers=headers,
        )
    except httpx.HTTPError:
        return None
    if r.status_code == 200:
        return True
    if r.status_code == 403:
        return True         # exists; this token lacks access
    if r.status_code in (401, 404):
        return False
    return None


def sibling_candidates(model_id: str) -> list[dict]:
    """Conventional quantized-artifact names for a base (bf16) model.
    Pure guessing — pair with ``hub_model_exists`` to verify."""
    if infer_quant(model_id) != "bf16":
        return []          # already a quant artifact
    org, name = model_id.split("/", 1)
    return [
        {"id": f"{org}/{name}-FP8", "quant": "fp8"},
        {"id": f"{org}/{name}-AWQ", "quant": "awq"},
        {"id": f"{org}/{name}-GPTQ-Int4", "quant": "gptq-int4"},
        # RedHatAI (ex-Neural Magic) publishes ungated quants of gated
        # bases under its own org with these naming conventions.
        {"id": f"RedHatAI/{name}-FP8-dynamic", "quant": "fp8"},
        {"id": f"RedHatAI/{name}-quantized.w4a16", "quant": "gptq-int4"},
    ]


def suggest_quant_siblings(
    model_id: str,
    *,
    verify: bool = True,
    user_dir: Path | str = USER_CATALOG_DIR,
) -> list[dict]:
    """Quantized siblings of ``model_id`` worth offering in the UI:
    each candidate carries ``exists`` (True/False/None=unverified) and
    ``in_catalog``. Verified-absent candidates are dropped."""
    known = {e["id"] for e in load_model_catalog(user_dir)}
    out = []
    for cand in sibling_candidates(model_id):
        exists = hub_model_exists(cand["id"]) if verify else None
        if exists is False:
            continue
        out.append({**cand, "exists": exists,
                    "in_catalog": cand["id"] in known})
    return out
