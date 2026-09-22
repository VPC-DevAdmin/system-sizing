"""Model weight staging: HF cache inspection + download commands.

Backs the UI's Models panel: which models do the profiles and search
spaces reference, are their weights already in the HF cache the engine
containers mount, and how do we fetch the missing ones?

Cache resolution (shared by doctor, downloads, engines, optimizer):
``OPTIMIZER_HF_CACHE`` env override > the UI-chosen location persisted
in ``~/.config/capsim/storage.json`` > ``/data/ml/huggingface`` when
that layout exists > ``~/.cache/huggingface``. Every download is
pinned to it via ``HF_HOME`` so weights land where the containers will
look, not wherever the CLI's default happens to be.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml

# Host-local storage selection (written by the UI's Storage step).
# Deliberately OUTSIDE the repo: which disk a lab box uses is per-host
# state, not project configuration.
STORAGE_CONFIG = Path.home() / ".config" / "capsim" / "storage.json"


def _storage_config() -> dict:
    try:
        return json.loads(STORAGE_CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def cache_overflow_roots(cache: Path | None = None) -> list[Path]:
    """Host directories that model entries in the cache SYMLINK into.

    A box outgrows one drive: the XE7740's cache lives on /data2 and
    its 1 TB of NVFP4 giants on /data, linked in as
    ``hub/models--nvidia--Kimi-K2-Thinking-NVFP4 -> /data/capsim/
    hf-overflow/hub/models--...``. The hub library follows the links
    on the host; inside a container they dangle unless the target
    root is mounted at the SAME path. Every engine mounts each root
    this returns beside the cache, so an operator can spread the cache
    over drives with plain symlinks and nothing else.
    """
    cache = cache or hf_cache_dir()
    hub = cache / "hub"
    roots: list[Path] = []
    if not hub.is_dir():
        return roots
    for entry in sorted(hub.iterdir()):
        if not entry.is_symlink():
            continue
        try:
            target = entry.resolve(strict=True)
        except OSError:
            continue
        if cache.resolve() in target.parents:
            continue
        # The overflow's own hub/ directory's parent: mount that root.
        root = target.parent.parent if target.parent.name == "hub" else target.parent
        if root not in roots:
            roots.append(root)
    return roots


CONTAINER_HF_CACHE = "/root/.cache/huggingface"


def container_cache_path(host_path: str | Path) -> Optional[str]:
    """Where ``host_path`` appears inside an engine container WITHOUT a
    mount of its own: under the HF cache mount when it lives in the
    cache, at its own path when it lives in a symlinked overflow root
    (mounted at that path by ``cache_mount_args``), else None.

    Why this exists: a Hub snapshot's files are RELATIVE symlinks into
    the cache's ``blobs/`` directory. Bind-mounting the snapshot's
    GGUF directory alone at ``/gguf`` put every shard outside the
    mount and the server saw "No such file" (XE7740, Qwen3-235B and
    every DeepSeek KTransformers cell). Referencing the same directory
    through the cache mount keeps the links intact.
    """
    host = str(host_path).rstrip("/")
    cache = str(hf_cache_dir()).rstrip("/")
    if host == cache or host.startswith(cache + "/"):
        return CONTAINER_HF_CACHE + host[len(cache):]
    for root in cache_overflow_roots():
        r = str(root).rstrip("/")
        if host == r or host.startswith(r + "/"):
            return host
    return None


def staged_snapshot_in_container(model_id: str | None,
                                 cache: Path | None = None) -> Optional[str]:
    """The staged snapshot directory of ``model_id`` at the path an
    engine container sees it (through the cache mount), or None when
    the model is not a hub id, is not fully cached, or lives outside
    every mounted root.

    Why: ``trtllm-serve`` resolves a hub id's TOKENIZER through the
    hub's model API even when every file is in the cache, and offline
    mode refuses that call -- the server came up with no tokenizer and
    answered every request with HTTP 400 (XE7740, Llama-3.3-70B NVFP4,
    the first TensorRT-LLM launch that survived executor creation).
    Pointing ``--tokenizer`` at the snapshot skips the hub entirely.
    """
    if not model_id or "/" not in model_id or model_id.startswith(("/", ".")):
        return None
    cache = cache or hf_cache_dir()
    try:
        if not model_status(model_id, cache)["cached"]:
            return None
    except OSError:
        return None
    rev = _latest_snapshot(_model_dir(model_id, cache))
    if rev is None:
        return None
    return container_cache_path(rev.resolve() if rev.is_symlink() else rev)


def cache_mount_args(container_cache: str = "/root/.cache/huggingface") -> list[str]:
    """``-v`` arguments for the HF cache plus every overflow root it
    links into (mounted at its own host path so the links resolve)."""
    cache = hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out = ["-v", f"{cache}:{container_cache}"]
    for root in cache_overflow_roots(cache):
        out += ["-v", f"{root}:{root}"]
    return out


def hf_cache_dir() -> Path:
    """Resolved HF cache directory. Precedence: OPTIMIZER_HF_CACHE env
    > the UI-chosen location in ~/.config/capsim/storage.json >
    /data/ml/huggingface when that layout exists > ~/.cache/huggingface.
    """
    return _hf_cache_resolved()[0]


def hf_cache_source() -> str:
    """Where the current resolution came from: env | configured |
    data-layout | default."""
    return _hf_cache_resolved()[1]


def _hf_cache_resolved() -> tuple[Path, str]:
    override = os.environ.get("OPTIMIZER_HF_CACHE")
    if override:
        return Path(override), "env"
    configured = _storage_config().get("hf_cache")
    if configured:
        return Path(configured), "configured"
    data = Path("/data/ml/huggingface")
    if data.exists():
        return data, "data-layout"
    return Path.home() / ".cache" / "huggingface", "default"


def set_hf_cache_dir(path: str | Path) -> Path:
    """Persist the UI-chosen cache location and create it. Raises
    ValueError with a human-readable reason on a bad choice."""
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(f"path must be absolute, got '{path}'")
    if os.environ.get("OPTIMIZER_HF_CACHE"):
        raise ValueError(
            "OPTIMIZER_HF_CACHE is set in the service environment and "
            "overrides any choice made here — unset it first"
        )
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".capsim-write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise ValueError(f"cannot write to {p}: {e}") from e
    STORAGE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = _storage_config()
    cfg["hf_cache"] = str(p)
    STORAGE_CONFIG.write_text(json.dumps(cfg, indent=2))
    return p


def _model_dir(model_id: str, cache: Path) -> Path:
    return cache / "hub" / ("models--" + model_id.replace("/", "--"))


def _dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / 1e9


# The files a KTransformers launch reads from the HF directory: it
# takes weights from the GGUF, so the HF repo contributes config and
# tokenizer only. These are `hf download --include` globs (one
# --include per pattern: the typer-based hf 1.x CLI repeats the flag,
# a second bare pattern would be read as a positional filename); the
# modeling .py files ride along because DeepSeek configs carry an
# auto_map that trust_remote_code resolves against the directory.
CONFIG_ONLY_INCLUDE = ("*.json", "tokenizer*", "*.txt", "*.model", "*.py")


def _include_args(patterns: tuple[str, ...] | list[str]) -> list[str]:
    return [a for pat in patterns for a in ("--include", pat)]
# What "a tokenizer is staged" looks like on disk: the HF fast
# tokenizer, a SentencePiece model, or a tiktoken vocabulary --
# Kimi-K2 ships ``tiktoken.model`` + ``tokenization_kimi.py`` and no
# tokenizer.json, and its config-only staging read as incomplete.
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tiktoken.model")


def _latest_snapshot(d: Path) -> Path | None:
    """The newest non-empty snapshot revision under a cache model dir."""
    snapshots = d / "snapshots"
    if not snapshots.is_dir():
        return None
    revs = [r for r in snapshots.iterdir() if r.is_dir() and any(r.iterdir())]
    if not revs:
        return None
    return max(revs, key=lambda r: r.stat().st_mtime)


def model_arch(model_id: str, cache: Path | None = None) -> str | None:
    """``model_type`` from the cached config.json (``deepseek_v3``,
    ``qwen3_moe``, ...), or None when the config is not staged. The
    KTransformers launcher picks its optimize rule by this; the UI
    shows it."""
    cache = cache or hf_cache_dir()
    rev = _latest_snapshot(_model_dir(model_id, cache))
    if rev is None:
        return None
    try:
        doc = json.loads((rev / "config.json").read_text())
    except (OSError, ValueError):
        return None
    arch = doc.get("model_type") if isinstance(doc, dict) else None
    return str(arch) if arch else None


def _config_staged(rev: Path) -> bool:
    return (rev / "config.json").is_file() and any(
        (rev / t).is_file() for t in _TOKENIZER_FILES)


def model_status(model_id: str, cache: Path | None = None,
                 gguf: dict | None = None,
                 config_only: bool | None = None) -> dict:
    """Cache status for one HF model id. ``cached`` means a snapshot
    revision with files exists and no blob is mid-download. ``gguf``
    is the catalog's companion spec (``{repo, file, size_gb}``) when
    the entry has one; the row then carries its staging status too.

    ``config_only`` is the ``kt_only`` staging rule: the HF directory
    only has to supply config + tokenizer (KTransformers takes the
    weights from the GGUF), so the model counts as cached once
    config.json and a tokenizer are present -- 687 GB of safetensors
    nobody will load are never asked for. None looks the rule up in
    the catalog, so callers that only know the id (the engine's
    offline-mode check, the roofline) get the same answer as Prepare.
    ``arch`` is config.json's model_type once the config is staged."""
    cache = cache or hf_cache_dir()
    if config_only is None:
        entry = catalog_entry(model_id)
        config_only = bool(entry and entry.get("kt_only"))
    d = _model_dir(model_id, cache)
    rev = _latest_snapshot(d)
    has_snapshot = rev is not None
    if config_only:
        has_snapshot = rev is not None and _config_staged(rev)
    incomplete = any((d / "blobs").glob("*.incomplete")) if (d / "blobs").exists() else False
    return {
        "model": model_id,
        "cached": bool(has_snapshot and not incomplete),
        "partial": bool(incomplete or (d.exists() and not has_snapshot)),
        "size_gb": round(_dir_size_gb(d), 1) if d.exists() else 0.0,
        "path": str(d),
        "config_only": bool(config_only),
        "arch": model_arch(model_id, cache) if rev is not None else None,
        "gguf": gguf_status(gguf, cache) if gguf else None,
    }


def catalog_entry(model_id: str) -> dict | None:
    """The merged catalog entry for ``model_id``, or None."""
    from .model_catalog import CatalogError, load_model_catalog
    try:
        for e in load_model_catalog():
            if e["id"] == model_id:
                return e
    except CatalogError:
        pass
    return None


_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")


def _shard_dir_staged(dirpath: Path) -> bool:
    """A sharded quant directory is staged when every shard its names
    promise is present with bytes in it: ``-00003-of-00008`` means
    eight files, and a download killed after five would otherwise
    read as staged and make KTransformers fail at load."""
    try:
        files = [f for f in dirpath.iterdir()
                 if f.is_file() and f.suffix == ".gguf" and f.stat().st_size > 0]
    except OSError:
        return False
    if not files:
        return False
    expected: set[int] = set()
    seen: set[int] = set()
    for f in files:
        m = _SHARD_RE.search(f.name)
        if m:
            seen.add(int(m.group(1)))
            expected.add(int(m.group(2)))
    if not expected:
        return True            # unsharded files in a directory: present
    if len(expected) != 1:
        return False           # two different shard sets mixed together
    n = expected.pop()
    return seen == set(range(1, n + 1))


def gguf_status(entry_or_model_id: dict | str,
                cache: Path | None = None) -> dict | None:
    """Staging status of a model's GGUF companion: ``{repo, file,
    size_gb, sharded, cached, path}``, or None when the model has no
    companion.

    Accepts a catalog entry, a bare companion spec ``{repo, file}``,
    or a model id (looked up in the catalog). ``path`` is the
    DIRECTORY holding the .gguf inside the HF cache snapshot -- what
    KTransformers' ``--gguf_path`` takes -- and is None until the file
    is fully staged. ``hf download <repo> <file>`` only links the
    file into a snapshot once its blob is complete, so a present,
    non-dangling file with bytes in it is the staged signal.

    ``file`` may instead name a shard DIRECTORY (``sharded`` is then
    True): ``path`` is that directory, staged once every
    ``-0000i-of-0000N`` shard is in place.

    KTransformers loads EVERY .gguf under that directory, so the
    catalog names exactly one file (or one shard directory) per
    companion; hand-staging a second quant of the same repo into the
    same snapshot would make the server read both."""
    if isinstance(entry_or_model_id, str):
        entry = catalog_entry(entry_or_model_id)
        spec = entry.get("gguf") if entry else None
    elif "repo" in entry_or_model_id and "file" in entry_or_model_id:
        spec = entry_or_model_id
    else:
        spec = entry_or_model_id.get("gguf")
    if not spec:
        return None
    cache = cache or hf_cache_dir()
    repo, file = str(spec["repo"]), str(spec["file"])
    sharded = not file.endswith(".gguf")
    snapshots = _model_dir(repo, cache) / "snapshots"
    path = None
    if snapshots.is_dir():
        for rev in sorted(snapshots.iterdir(), key=lambda r: r.stat().st_mtime,
                          reverse=True):
            f = rev / file
            try:
                if sharded:
                    if f.is_dir() and _shard_dir_staged(f):
                        path = f
                        break
                elif f.is_file() and f.stat().st_size > 0:
                    path = f.parent
                    break
            except OSError:
                continue
    return {
        "repo": repo,
        "file": file,
        "size_gb": spec.get("size_gb"),
        "sharded": sharded,
        "cached": path is not None,
        "path": str(path) if path else None,
    }


def referenced_models(
    profiles_dirs: tuple[str, ...] = ("config/profiles", "config"),
    search_dir: str = "config/search",
) -> list[dict]:
    """Every HF model id the catalog carries or a GPU profile / search
    space references, with where each came from. Local ``/models/...``
    paths (the CPU pre-staged flow) are skipped — they aren't HF
    downloads."""
    out: dict[str, set[str]] = {}
    meta: dict[str, dict] = {}

    def _add(model, source):
        if not model or str(model).startswith("/"):
            return
        out.setdefault(str(model), set()).add(source)

    for d in profiles_dirs:
        for p in sorted(Path(d).glob("*.yaml")) if Path(d).exists() else []:
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            engine = (raw.get("engine") or {})
            # Only engines that pull from HF at launch (the GPU flow);
            # the CPU flow stages into /models via `capsim ready`.
            if engine.get("type") == "vllm_cuda":
                _add(engine.get("model_id"), f"profile:{p.stem}")
    if Path(search_dir).exists():
        for p in sorted(Path(search_dir).glob("*.yaml")):
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            for vname, v in (raw.get("model_variants") or {}).items():
                if isinstance(v, dict):
                    _add(v.get("model"), f"space:{p.stem}/{vname}")

    from .model_catalog import CatalogError, load_model_catalog
    try:
        for entry in load_model_catalog():
            _add(entry["id"], f"catalog:{entry['family']}")
            meta[entry["id"]] = {
                k: entry[k]
                for k in ("family", "series", "quant", "approx_size_gb",
                          "min_vram_gb", "gated", "moe", "specialty",
                          "notes", "kt_only", "host_ram_gb")
                if k in entry
            }
            if entry.get("gguf"):
                meta[entry["id"]]["gguf"] = entry["gguf"]
    except CatalogError:
        pass          # a broken local overlay must not hide the rest

    cache = hf_cache_dir()
    return [
        {**model_status(m, cache, gguf=meta.get(m, {}).get("gguf"),
                        config_only=bool(meta.get(m, {}).get("kt_only"))),
         "referenced_by": sorted(srcs),
         **{k: v for k, v in meta.get(m, {}).items() if k != "gguf"}}
        for m, srcs in sorted(out.items())
    ]


def download_command(model_id: str,
                     companion: str | None = None) -> tuple[list[str], dict]:
    """(argv, extra_env) for a cache-pinned weight download. Prefers
    the ``hf`` CLI sitting next to this interpreter (the tool venv),
    falling back to PATH.

    A ``kt_only`` catalog entry downloads config-only (``--include``
    config, tokenizer and modeling files): KTransformers reads the
    weights from the GGUF, and the safetensors are the part that does
    not fit the box in the first place.

    ``companion="gguf"`` downloads the model's GGUF companion instead
    (``hf download <repo> <file>``, same cache; a shard directory
    becomes ``--include "<dir>/*.gguf"``) -- ValueError when the
    catalog entry has none."""
    hf_bin = Path(sys.executable).parent / "hf"
    hf = str(hf_bin) if hf_bin.exists() else (shutil.which("hf") or "hf")
    if companion is None:
        argv = [hf, "download", model_id]
        entry = catalog_entry(model_id)
        if entry and entry.get("kt_only"):
            argv += _include_args(CONFIG_ONLY_INCLUDE)
    elif companion == "gguf":
        spec = gguf_status(model_id)
        if spec is None:
            raise ValueError(f"{model_id} has no GGUF companion in the catalog")
        if spec["sharded"]:
            argv = [hf, "download", spec["repo"],
                    *_include_args([f"{spec['file']}/*.gguf"])]
        else:
            argv = [hf, "download", spec["repo"], spec["file"]]
    else:
        raise ValueError(f"unknown companion {companion!r}")
    env = {"HF_HOME": str(hf_cache_dir())}
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            env[var] = os.environ[var]
    return argv, env
