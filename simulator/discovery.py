"""Live model discovery — the Hub, filtered through this box.

Everything the catalog curation did by hand is public API:
``huggingface.co/api/models`` lists an org's models sortable by date
or downloads; each model's detail carries its safetensors parameter
count and dtypes (native FP8 shows as F8_E4M3). This module turns
that into "what new models could THIS box run":

  1. list recent text-generation candidates from the leading orgs,
  2. fetch real parameter counts + dtypes per candidate,
  3. estimate the weights footprint (params × bytes/param + overhead)
     and derive the feasible TP set from the detected GPUs,
  4. tag capabilities readable from name/dtype (MoE via the -A#B
     convention, native FP8, coder/thinking variants, linear/hybrid
     attention lines), age, downloads, gated.

Estimates are planning-grade: min_vram is weights × 1.15 — the same
convention as hand-curated entries — refined by reality the first
time the engine loads the model. Everything degrades gracefully
offline (the endpoint reports the error; nothing blocks).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

HUB = "https://huggingface.co/api/models"

# Orgs whose text-generation lines are worth watching. Operators can
# extend per-call (?orgs=a,b,c).
DEFAULT_ORGS = (
    "Qwen", "zai-org", "deepseek-ai", "meta-llama", "mistralai",
    "openai", "moonshotai", "MiniMaxAI", "google", "ibm-granite",
)

# Name fragments that mean "not a text-generation serving target".
_EXCLUDE = ("gguf", "mlx", "awq", "gptq", "-vl", "vl-", "omni", "audio",
            "image", "video", "embed", "rerank", "guard", "safety",
            "-base", "base-", "bnb", "int4", "int8", "eagle", "draft")

_MOE_RE = re.compile(r"-a\d+(\.\d+)?b", re.I)


def _hub_get(url: str, timeout: float = 15.0) -> Optional[object]:
    import httpx
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            return r.json()
    except Exception as e:  # noqa: BLE001
        log.debug("hub fetch failed %s: %s", url, e)
    return None


def _candidate(model_id: str) -> bool:
    n = model_id.lower()
    if any(x in n for x in _EXCLUDE):
        return False
    # Keep names that look like serving targets: instruct/chat/it
    # variants, MoE-convention names, or versioned flagship lines.
    return any(x in n for x in (
        "instruct", "chat", "-it", "a3b", "a22b", "next", "flash",
        "glm-", "v3.", "-m2", "k2", "kimi", "oss", "thinking",
    )) or bool(_MOE_RE.search(n))


def discover_models(
    orgs: tuple[str, ...] = DEFAULT_ORGS,
    per_org: int = 6,
    vram_per_gpu_gb: Optional[float] = None,
    max_tp: int = 1,
    known_ids: Optional[set] = None,
) -> list[dict]:
    """Recent candidates across ``orgs`` with real sizing. One listing
    call per org + one detail call per kept candidate."""
    known_ids = known_ids or set()
    out: list[dict] = []
    for org in orgs:
        listing = _hub_get(
            f"{HUB}?author={org}&sort=lastModified&direction=-1&limit=40")
        if not isinstance(listing, list):
            continue
        kept = 0
        for m in listing:
            mid = m.get("modelId") or m.get("id") or ""
            if not mid or not _candidate(mid):
                continue
            if kept >= per_org:
                break
            detail = _hub_get(f"{HUB}/{mid}")
            if not isinstance(detail, dict):
                continue
            st = detail.get("safetensors") or {}
            total = st.get("total")
            if not total or total < 6e9:
                continue    # no weights metadata, or below serving size
            kept += 1
            dtypes = set((st.get("parameters") or {}).keys())
            native_fp8 = any("F8" in d for d in dtypes)
            params_b = round(total / 1e9, 1)
            bytes_per = 1.0 if native_fp8 else 2.0
            weights_gb = round(params_b * bytes_per * 1.04, 0)
            min_vram = round(weights_gb * 1.15, 0)
            name = mid.split("/")[-1].lower()
            entry = {
                "id": mid,
                "params_b": params_b,
                "quant": "fp8" if native_fp8 else "bf16",
                "moe": bool(_MOE_RE.search(name)
                            or any(x in name for x in ("a22b", "-m2", "glm-5",
                                                       "v3.", "k2"))),
                "approx_size_gb": weights_gb,
                "min_vram_gb": min_vram,
                "gated": bool(detail.get("gated")),
                "downloads": detail.get("downloads"),
                "last_modified": detail.get("lastModified"),
                "created_at": detail.get("createdAt"),
                "capabilities": [c for c, on in (
                    ("MoE", _MOE_RE.search(name) is not None),
                    ("native-FP8", native_fp8),
                    ("coder", "coder" in name or "code" in name),
                    ("thinking", "thinking" in name or "reasoner" in name),
                    ("linear/hybrid-attn",
                     "next" in name or "linear" in name),
                ) if on],
                "in_catalog": mid in known_ids,
            }
            if vram_per_gpu_gb:
                tps = [t for t in (1, 2, 4, 8, 16) if t <= max_tp
                       and t * vram_per_gpu_gb >= min_vram]
                entry["feasible_tps"] = tps
                entry["feasible"] = bool(tps)
            out.append(entry)
    # Newest first by default; the UI re-sorts client-side.
    out.sort(key=lambda e: e.get("last_modified") or "", reverse=True)
    return out
