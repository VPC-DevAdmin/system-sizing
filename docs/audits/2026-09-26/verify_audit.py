"""Read-only reproductions for the benchmark reporting audit.

Run from the repository root with .venv/bin/python docs/audits/2026-09-26/verify_audit.py.
Uses synthetic inputs; launches no engine and changes no benchmark data.
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from simulator.export import _extract_engine_config
from simulator.export_sizing import curve_point
from simulator.search import Objective, rung_sla_ok
from simulator.streaming import consume_with_tiers


class SyntheticStream:
    def __aiter__(self):
        return self.events()

    async def events(self):
        yield NS(choices=[NS(delta=NS(content="one two three four five", reasoning=None))])
        yield NS(choices=[], usage=NS(completion_tokens=5, prompt_tokens=3))

    async def close(self):
        pass


async def create_stream():
    return SyntheticStream()


async def main():
    result = await consume_with_tiers(
        create_stream, pre_ttft_timeout_s=5,
        inter_token_timeout_s=5, hard_timeout_s=10,
    )
    supplied = {
        "type": "vllm_cuda_multi", "model_id": "audit-example",
        "tensor_parallel_size": 1, "replica_devices": [[0], [1], [2], [3]],
        "gpu_memory_utilization": 0.9, "max_num_seqs": 256,
        "kv_cache_dtype": "fp8", "reasoning": True,
    }
    exported = _extract_engine_config({"config_json": json.dumps({"engine": supplied})})
    empty = curve_point({"concurrency": 64, "out_tok_s": 100})
    seconds = 5 * 365 * 24 * 3600
    print(json.dumps({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "synthetic_probes": {
            "stream": {"server_completion_tokens": 5, "stream_content_chunks": 1,
                       "reported_output_tokens": result.output_tokens},
            "missing_latency_is_sla_ok": rung_sla_ok({"throughput_out_tok_s": 100}, Objective()),
            "export_supplied": supplied, "export_retained": exported,
            "empty_outcomes": {k: empty[k] for k in
                               ("status", "sample_size", "violation_rate", "success_rate")},
        },
        "independent_arithmetic": {
            "r770_gpu_only_tok_per_j": {"8B": 5219 / (4 * 151),
                                        "27B": 1921 / (4 * 150),
                                        "35B": 3391 / (4 * 155)},
            "bf16_weight_bytes_billions": {"27B": 27 * 2, "35B": 35 * 2},
            "namd_core_ratio": 240 / 128,
            "namd_per_core_ratio_range": [1.84 / (240 / 128), 1.87 / (240 / 128)],
            "pt_implied_billed_tokens_per_second": 5180423 / 0.90 * 1e6 / seconds,
            "pt_cost_per_million_at_1179_tokens_per_second": 280618 / (1179 * seconds / 1e6),
            "pt_cost_per_million_50pct_from_cost_table": 272032 / (2590212 / 0.90),
            "pt_cost_per_million_100pct_from_cost_table": 280618 / (5180423 / 0.90),
        },
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
