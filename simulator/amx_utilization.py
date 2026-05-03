"""Parse oneDNN verbose launch logs into per-run AMX utilisation.

The PMU's ``EXE.AMX_BUSY`` cycles answer "did AMX engage?" but not
"how much of the matmul wall-clock did AMX kernels handle?". The
oneDNN verbose log answers the second question by classifying each
matmul/inner_product dispatch by its kernel implementation tag.

Pair the two: PMU = "AMX was busy for X cycles", oneDNN = "AMX kernels
handled Y% of matmul wall-clock and these were the hot shapes."

Streaming read with a byte budget — verbose logs hit multiple GB on
long runs; we never slurp the whole file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_MAX_LOG_BYTES = 256 * 1024 * 1024  # 256 MiB

_PREFIX_V1 = "onednn_verbose,v1,primitive,exec,cpu,"
_PREFIX_V0 = "onednn_verbose,exec,cpu,"

# AMX tags. Order matters: the AMX regex must run first because the AMX
# tag CONTAINS ``avx512_core`` as a substring.
_AMX_TAG_RE = re.compile(
    r"(?:brg|jit)[^,\s]*:"
    r"(?:avx512_core_amx|avx10_1_512_amx|avx10_2_512_amx)"
    r"(?:_bf16|_int8|_fp16|_fp8)?"
)
_NON_AMX_TAG_RE = re.compile(
    r"(?:brg|jit)[^,\s]*:"
    r"(?:avx512_core_bf16|avx512_core_vnni|avx512_core_fp16|avx512_core"
    r"|avx2_vnni|avx2|sse41|sse42)"
    r"(?!_amx)"
)

_INTERESTING_PRIMITIVES = {"matmul", "inner_product"}
_TOP_SHAPES_LIMIT = 10


@dataclass
class _Dispatch:
    primitive: str
    impl: str
    shape: str
    time_ms: float
    is_amx: bool


@dataclass
class AmxUtilization:
    onednn_matmul_dispatches_amx: int = 0
    onednn_matmul_dispatches_non_amx: int = 0
    onednn_matmul_time_ms_amx: float = 0.0
    onednn_matmul_time_ms_non_amx: float = 0.0
    onednn_amx_time_fraction: float | None = None
    onednn_amx_kernel_types: list[str] = field(default_factory=list)
    onednn_matmul_time_by_shape: list[dict] = field(default_factory=list)

    def is_empty(self) -> bool:
        """True iff the log yielded zero classifiable dispatches.

        Distinct from "all non-AMX": empty means we found no oneDNN
        verbose output (verbose disabled or wrong log format), so no
        verdict should be assigned.
        """
        return (
            self.onednn_matmul_dispatches_amx == 0
            and self.onednn_matmul_dispatches_non_amx == 0
        )

    def to_dict(self) -> dict:
        return {
            "onednn_matmul_dispatches_amx": self.onednn_matmul_dispatches_amx,
            "onednn_matmul_dispatches_non_amx": self.onednn_matmul_dispatches_non_amx,
            "onednn_matmul_time_ms_amx": self.onednn_matmul_time_ms_amx,
            "onednn_matmul_time_ms_non_amx": self.onednn_matmul_time_ms_non_amx,
            "onednn_amx_time_fraction": self.onednn_amx_time_fraction,
            "onednn_amx_kernel_types": sorted(self.onednn_amx_kernel_types),
            "onednn_matmul_time_by_shape": self.onednn_matmul_time_by_shape,
        }


def _parse_line(line: str) -> _Dispatch | None:
    if not line.startswith("onednn_verbose,"):
        return None
    if line.startswith(_PREFIX_V1):
        prim_idx, impl_idx = 5, 6
    elif line.startswith(_PREFIX_V0):
        prim_idx, impl_idx = 3, 4
    else:
        return None
    parts = line.rstrip().split(",")
    if len(parts) < max(prim_idx, impl_idx) + 2:
        return None
    primitive = parts[prim_idx]
    if primitive not in _INTERESTING_PRIMITIVES:
        return None
    impl = parts[impl_idx]
    is_amx = bool(_AMX_TAG_RE.search(impl))
    if not is_amx and not _NON_AMX_TAG_RE.search(impl):
        return None  # unknown tag → skip rather than mis-bucket
    try:
        time_ms = float(parts[-1])
    except ValueError:
        return None
    shape = parts[-2] if len(parts) >= 2 else ""
    return _Dispatch(
        primitive=primitive, impl=impl, shape=shape,
        time_ms=time_ms, is_amx=is_amx,
    )


def _iter_log_lines(path: Path, max_bytes: int) -> Iterator[str]:
    consumed = 0
    try:
        f = path.open("rb")
    except OSError:
        return
    with f:
        for raw in f:
            consumed += len(raw)
            if consumed > max_bytes:
                break
            try:
                yield raw.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                continue


def aggregate(events: Iterable[_Dispatch]) -> AmxUtilization:
    result = AmxUtilization()
    amx_kernels: set[str] = set()
    by_shape: dict[tuple[str, str], tuple[float, int]] = {}
    for ev in events:
        if ev.is_amx:
            result.onednn_matmul_dispatches_amx += 1
            result.onednn_matmul_time_ms_amx += ev.time_ms
            amx_kernels.add(ev.impl)
        else:
            result.onednn_matmul_dispatches_non_amx += 1
            result.onednn_matmul_time_ms_non_amx += ev.time_ms
        prev = by_shape.get((ev.shape, ev.impl), (0.0, 0))
        by_shape[(ev.shape, ev.impl)] = (prev[0] + ev.time_ms, prev[1] + 1)
    total = result.onednn_matmul_time_ms_amx + result.onednn_matmul_time_ms_non_amx
    if total > 0:
        result.onednn_amx_time_fraction = result.onednn_matmul_time_ms_amx / total
    result.onednn_amx_kernel_types = sorted(amx_kernels)
    top = sorted(
        ({"shape": s, "impl": i, "total_ms": round(ms, 4), "calls": c}
         for (s, i), (ms, c) in by_shape.items()),
        key=lambda r: r["total_ms"], reverse=True,
    )[:_TOP_SHAPES_LIMIT]
    result.onednn_matmul_time_by_shape = top
    return result


def parse_amx_utilization(
    log_path: Path | str, *, max_bytes: int = DEFAULT_MAX_LOG_BYTES,
) -> AmxUtilization:
    log_path = Path(log_path)
    events: list[_Dispatch] = []
    for line in _iter_log_lines(log_path, max_bytes):
        ev = _parse_line(line)
        if ev is not None:
            events.append(ev)
    return aggregate(events)
