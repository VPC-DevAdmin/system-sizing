"""Native speculative activity checks and explicit, unpromoted roofline candidates."""

import math

import yaml

from ..benchmarks.storage import write_json


def counters(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        if name in (
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("Invalid speculative counter")
            values[key] = number
    return values


def verify_activity(directory, replicas):
    evidence = []
    for index in range(replicas):
        before = counters((directory / f"before-replica-{index}.prom").read_text())
        after = counters((directory / f"after-replica-{index}.prom").read_text())
        names = {k.split("{", 1)[0] for k in before}
        if (
            len(names) != 2
            or set(before) != set(after)
            or any(after[k] < before[k] for k in before)
        ):
            raise RuntimeError("Speculative counters missing, changed identity, or reset")
        delta = {k: after[k] - before[k] for k in before}
        drafted = sum(
            v for k, v in delta.items() if k.startswith("vllm:spec_decode_num_draft_tokens_total")
        )
        accepted = sum(
            v
            for k, v in delta.items()
            if k.startswith("vllm:spec_decode_num_accepted_tokens_total")
        )
        if drafted <= 0 or accepted > drafted:
            raise RuntimeError("Speculative activity not verified on every replica")
        evidence.append(
            {
                "replica": index,
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "accepted_fraction": accepted / drafted,
            }
        )
    write_json(
        directory / "speculative-verification.json",
        {
            "replicas": evidence,
            "scope": "Native draft acceptance diagnostic; not a throughput or quality claim",
        },
    )


def export_candidate(path, manifest, comparison):
    """Save a reproducible candidate; never mutate roofline winners from task metrics."""
    configs = sorted(path.glob("block-*-treatment/launch-resolved.yaml"))
    if not configs:
        return
    cfg = yaml.safe_load(configs[0].read_text())
    (path / "roofline-candidate.yaml").write_text(yaml.safe_dump(cfg))
    write_json(
        path / "roofline-candidate.json",
        {
            "status": "requires_matched_roofline_rerun",
            "campaign_id": manifest["id"],
            "source_result": manifest["source_result"],
            "configuration": "roofline-candidate.yaml",
            "task_comparison": comparison,
            "instructions": "Re-run the baseline and this candidate with the same roofline request profile, "
            "GPU count, precision, cache policy, concurrency search and harness versions. "
            "The task result does not update the roofline or prove fixed-token throughput gains.",
            "scope": "Resolved treatment launch from this campaign; resource count is not automatically expanded",
        },
    )
