"""Expand a vLLM-style thread-binding string into a CPU id set.

vLLM's ``VLLM_CPU_OMP_THREADS_BIND`` uses ``|`` to separate per-worker
thread groups. Each group is one or more ranges separated by ``,``::

    "0-15|16-31|32-47|48-63"          # four TP workers, 16 threads each
    "0-7,16-23|8-15,24-31"            # interleaved layout
    "0-31"                            # single binding

Without filtering, ``FrequencyCollector`` averages frequency across
every logical CPU on the host — idle CPUs on the unused socket pull
the mean toward half-nominal. The simulator's bandwidth and PMU work
can be system-wide (the engine is the only meaningful workload on a
benchmark host) but frequency MUST be filtered to the bound set or
it lies.

Implementation note: we tolerate single CPUs (``"0"``), open-ended
ranges, whitespace, and stray empty groups. Parse errors return
``set()`` rather than raising — the collector falls back to "all CPUs".
"""

from __future__ import annotations


def expand_thread_binding(bind_string: str | None) -> set[int]:
    """Parse a vLLM-style thread-binding string to a set of CPU ids.

    Returns an empty set if the input is empty or malformed; the caller
    treats that as "no filter, use all CPUs".
    """
    if not bind_string:
        return set()
    out: set[int] = set()
    for group in bind_string.split("|"):
        group = group.strip()
        if not group:
            continue
        for token in group.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo_s, hi_s = token.split("-", 1)
                try:
                    lo = int(lo_s)
                    hi = int(hi_s)
                except ValueError:
                    continue
                if hi < lo:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                try:
                    out.add(int(token))
                except ValueError:
                    continue
    return out
