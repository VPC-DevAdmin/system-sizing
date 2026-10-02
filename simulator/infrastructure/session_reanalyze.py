"""Derive new SLO views from saved evidence without rewriting the original run."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ..benchmarks.storage import read_json, sha256, write_json
from .session_saturation import analyze_requests, compare_saturation
from .spec import ExperimentSpec


def reanalyze(path, thresholds):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    state = read_json(path / "state.json")
    settings = dict(manifest["spec"], session_slo_grid_s=thresholds)
    spec = ExperimentSpec(**settings)
    if spec.project_id != "P02" or not spec.session_arrival_rates:
        raise ValueError("A P02 saturation campaign is required")
    results = copy.deepcopy(state["results"])
    sources = {}
    for row in results:
        source = (path / row["evidence"] / "requests.jsonl").resolve()
        if not source.is_relative_to(path):
            raise ValueError("Evidence outside campaign")
        requests = [json.loads(line) for line in source.read_text().splitlines()]
        row["metrics"].update(analyze_requests(requests, spec))
        sources[str(source.relative_to(path))] = sha256(source)
    output = dict(
        source_campaign=path.name,
        source_state_sha256=sha256(path / "state.json"),
        response_slo_seconds=thresholds,
        request_sources=sources,
        comparison=compare_saturation(results, spec),
    )
    identity = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()[:16]
    target = path / "analyses" / f"slo-{identity}.json"
    write_json(target, output)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign")
    parser.add_argument("--slo-seconds", required=True)
    args = parser.parse_args()
    print(reanalyze(args.campaign, [float(x) for x in args.slo_seconds.split(",")]))
