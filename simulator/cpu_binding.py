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


def flatten_for_taskset(bind_string: str | None) -> str | None:
    """Convert a vLLM-style group string into a flat ``taskset -c`` arg.

    ``"0-7|8-15|16-23|24-31"`` -> ``"0-31"`` (or a comma list when ranges
    aren't contiguous). Returns ``None`` for empty input so the caller
    knows to skip the wrapper.
    """
    cpus = sorted(expand_thread_binding(bind_string))
    if not cpus:
        return None
    # Compact contiguous runs into ranges for readability; taskset accepts
    # a CSV of ranges either way.
    out: list[str] = []
    run_lo = run_hi = cpus[0]
    for c in cpus[1:]:
        if c == run_hi + 1:
            run_hi = c
        else:
            out.append(f"{run_lo}-{run_hi}" if run_hi > run_lo else f"{run_lo}")
            run_lo = run_hi = c
    out.append(f"{run_lo}-{run_hi}" if run_hi > run_lo else f"{run_lo}")
    return ",".join(out)


def derive_sglang_thread_binding(cpu_bind: str | None, tp: int) -> str:
    """Return a value for ``SGLANG_CPU_OMP_THREADS_BIND`` aligned to ``tp``.

    SGLang-CPU's scheduler asserts ``tp == len(env_var.split('|'))``;
    a TP=4 launch with a 'wrong-shape' bind aborts during init with::

        AssertionError: SGLANG_CPU_OMP_THREADS_BIND setting must be aligned
        with TP size parameter (...). Please double check your settings.

    Three input shapes are accepted:
      1. ``cpu_bind`` already has exactly ``tp`` ``|``-separated groups —
         use as-is. Caller is in control.
      2. ``cpu_bind`` is a single contiguous range ("0-31") and ``tp``
         divides the range cleanly — split evenly.
      3. ``cpu_bind`` is empty/None and ``tp == 1`` — return empty (the
         engine env var will be omitted).

    Anything else raises so the misconfig surfaces at launch time, not
    20 minutes into a model load.
    """
    if not cpu_bind:
        if tp == 1:
            return ""
        raise ValueError(
            f"cpu_bind is required for SGLang TP={tp} (need {tp} groups)"
        )

    groups = [g.strip() for g in cpu_bind.split("|") if g.strip()]
    if len(groups) == tp:
        return "|".join(groups)

    # Single-group case: split into tp evenly.
    if len(groups) == 1:
        cpus = sorted(expand_thread_binding(groups[0]))
        if not cpus:
            raise ValueError(f"cpu_bind {cpu_bind!r} expanded to no CPUs")
        # We require contiguity for an even split — if you have a sparse
        # set, encode the |-grouping yourself in cpu_bind.
        if cpus != list(range(cpus[0], cpus[-1] + 1)):
            raise ValueError(
                f"cpu_bind {cpu_bind!r} is non-contiguous; supply tp={tp} "
                f"explicit groups instead"
            )
        total = len(cpus)
        if total % tp != 0:
            raise ValueError(
                f"cpu_bind {cpu_bind!r} ({total} CPUs) not divisible by "
                f"tp={tp}; pick a tp that divides evenly or pass an "
                f"explicit |-grouped bind"
            )
        per = total // tp
        return "|".join(
            f"{cpus[i*per]}-{cpus[(i+1)*per - 1]}" for i in range(tp)
        )

    raise ValueError(
        f"cpu_bind {cpu_bind!r} has {len(groups)} groups but tp={tp}; "
        f"either supply tp groups or a single range divisible by tp"
    )


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
