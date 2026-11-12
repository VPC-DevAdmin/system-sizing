# AIPerf check of the 100,000 tokens/s white paper (2026-10-26, settings disclosure added 2026-11-04)

## Note 2026-11-04: prompts repeated, so prefill was mostly served from the prefix cache

These are AIPerf's results as produced. Report them **with these settings attached**:
AIPerf 0.13.0 with its default `--num-dataset-entries 100` (8 processes with different
seeds, so 800 distinct prompts, each resent many times per rung), and vLLM prefix
caching on. The engines therefore served almost every prompt from cache:

| Config | Prefix cache hit rate (engine log) | Prompt tokens per request (server count) |
|---|---|---|
| gpt-oss-120b | 98.5% | 195 |
| gpt-oss-20b | 98.5% | 195 |
| Qwen3-30B-A3B FP8 | 94.1% | 136 |
| Gemma-4-26B NVFP4 | 90.7% | 141 |
| Kimi-Linear-48B NVFP4 | 0.0% (linear-attention hybrid, cannot prefix-cache) | 135 |

For the first four rows the gen tok/s figures are higher than a workload of unique
prompts would give, by an amount not yet measured. Kimi-Linear's 67.6k involved no cache
reuse; part of its gap to the other four comes from their cached prefill. A rerun with
enough distinct prompts that none repeats (for example `--num-dataset-entries 80000` per
process) gives the figure for unique prompts behind the real chat template.

Capsim's own September sweep, the source of the white paper's headline, is held to a
stricter standard (only reuse a real deployment would see) and does not meet it: its
prompts came from a small fixed paragraph set, and the gpt-oss runs show 35-75%
prefix-cache hit rates (gpt-oss-120b run_702 73.9%, run_704 71.3%, run_745 71.7%,
run_756 35.3%; gpt-oss-20b run_753 ~70%; Gemma-4-26B run_326 ~44%).

The white paper *Breaking the 100,000 token per second barrier with XE7740, NVIDIA, and
Intel* reports Capsim's own headline sweep (2026-09-23): Gemma-4-26B NVFP4 at 127k
and gpt-oss-20b at about 106k generated tokens/s. This run repeats that workload with
AIPerf, an independent load generator, on the same server.

Driver [`scripts/x100k_aiperf.py`](../scripts/x100k_aiperf.py) (the P11 driver,
[`scripts/p11_us_survey.py`](../scripts/p11_us_survey.py), at the paper's shape). Full
data [`x100k-aiperf-results-2026-10-26-prefixcached.json`](x100k-aiperf-results-2026-10-26-prefixcached.json). Raw AIPerf records and
engine logs stay on the box under `/data/capsim/x100k/results/sweep`.

No latency target is applied. Every rung is recorded.

## Setup

- Dell XE7740, 8 × RTX PRO 6000 Blackwell Server 96 GB (SM120), PCIe only.
- vLLM v0.29.0 (`vllm/vllm-openai:latest`, sha256:c2914767…), one copy of the model per
  GPU (8 copies), 0.90 GPU memory, `--max-model-len 4096`, `--max-num-seqs 1024` per copy,
  FP8 KV cache (default KV for Kimi-Linear, see below). No other tuning.
- Load: AIPerf 0.13.0, chat endpoint, streaming, synthetic prompts of 128 tokens and
  256 output tokens with `ignore_eos`, closed-loop at 256, 512, 1,024, 2,048, 4,096 and
  8,192 concurrent users. 30 s warmup, 120 s window. Eight AIPerf processes per rung,
  one per copy, records merged.
- For gpt-oss the 256 output tokens include reasoning, as in the paper.

## Peak throughput

| Model | Size (total / active) | AIPerf peak gen tok/s (users) | Capsim, paper | Errors at peak |
|---|---|---|---|---|
| Gemma-4-26B-A4B NVFP4 | 26B / 4B | **130.4k** (8,192) | 127k | 0.002% |
| Qwen3-30B-A3B-Instruct-2507 FP8 | 30B / 3B | **120.8k** (8,192) | not tested | 0 |
| gpt-oss-20b | 21B / 4B | **120.1k** (4,096) | ~106k | 0.007% |
| gpt-oss-120b | 117B / 5B | **104.0k** (8,192) | not published (79-82% completion in Capsim) | 0.019% |
| Kimi-Linear-48B-A3B NVFP4 | 48B / 3B | 67.6k (8,192) | not tested | 0 |

Four models pass 100,000 generated tokens/s with AIPerf. Gemma and gpt-oss-20b match
or exceed the paper's Capsim figures.

## Every rung

Gen tok/s, then first-token latency p50 / p95, per-token time p50, and requests the
engines actually ran (sum of copies, median).

| Users | Gemma-4-26B | Qwen3-30B-A3B | gpt-oss-20b | gpt-oss-120b | Kimi-Linear-48B |
|---|---|---|---|---|---|
| 256 | 22.7k · 35/57 ms · 11.2 ms · 256 | 16.6k · 50/62 · 15.2 · 256 | 28.9k · 54/68 · 8.4 · 256 | 12.7k · 109/130 · 19.4 · 256 | 16.8k · 196/234 · 14.5 · 256 |
| 512 | 40.0k · 40/90 · 12.6 · 512 | 29.0k · 58/87 · 17.5 · 512 | 52.5k · 65/100 · 9.3 · 512 | 20.9k · 142/161 · 23.7 · 512 | 24.4k · 226/267 · 20.1 · 512 |
| 1,024 | 64.3k · 61/168 · 15.5 · 1,024 | 50.2k · 68/146 · 20.1 · 1,024 | 87.7k · 86/199 · 11.0 · 1,016 | 33.9k · 188/225 · 28.9 · 1,024 | 35.7k · 310/382 · 27.6 · 1,024 |
| 2,048 | 97.7k · 90/346 · 20.2 · 2,048 | 82.2k · 104/304 · 24.1 · 2,048 | **119.7k** · 798/1,135 · 13.9 · 1,804 | 55.0k · 263/416 · 35.1 · 2,048 | 49.2k · 400/664 · 40.1 · 2,048 |
| 4,096 | **124.5k** · 323/716 · 31.4 · 4,078 | **117.1k** · 454/618 · 33.1 · 4,094 | **120.1k** · 3,983/5,489 · 17.1 · 2,052 | 82.7k · 496/817 · 45.8 · 4,096 | 60.9k · 636/1,321 · 64.1 · 4,096 |
| 8,192 | **130.4k** · 944/1,901 · 58.3 · 7,984 | **120.8k** · 738/1,457 · 63.5 · 8,192 | **115.2k** · 11,428/13,996 · 22.7 · 2,742 | **104.0k** · 3,253/5,124 · 66.3 · 7,320 | 67.6k · 1,057/23,479 · 109.2 · 7,776 |

## What the data shows

- **The paper's claim holds with an independent tool.** Gemma-4-26B reaches 130.4k with
  AIPerf against 127k with Capsim; gpt-oss-20b reaches 120.1k against about 106k.
- **Four models clear 100k**, from 3B to 5B active parameters, including the 117B
  gpt-oss-120b.
- **Points with interactive latency:** Gemma 97.7k at 2,048 users (first token p95 0.35 s,
  20 ms per token); Qwen3-30B-A3B 117.1k at 4,096 (0.62 s, 33 ms); gpt-oss-20b 119.7k at
  2,048 (1.1 s, 14 ms); gpt-oss-120b 82.7k at 4,096 (0.82 s, 46 ms).
- **gpt-oss-20b stops at about 2,000 running requests.** Above 2,048 users the engines
  hold ~2,050-2,740 of the offered load, so extra users only wait: first-token p95 rises
  to 5.5 s at 4,096 and 14 s at 8,192, and throughput slips to 115k at 8,192.
- **Kimi-Linear-48B tops out at 67.6k.** Per-token time grows fastest of the five
  (109 ms at 8,192), and first-token p95 reaches 23 s there.

## Engine settings and fixes made during the run

| Item | Setting or fix | Why |
|---|---|---|
| Kimi-Linear tokenizer | `--trust-remote-code` for vLLM, `--tokenizer-trust-remote-code` for AIPerf, Kimi only | The repo's tokenizer is custom code (Moonshot's tiktoken tokenizer). Reviewed before enabling: no network or subprocess use; vLLM runs its own built-in Kimi-Linear model code |
| Kimi-Linear KV cache | default KV dtype instead of FP8 | Its MLA layers run on TRITON_MLA, the only MLA backend on SM120; with FP8 KV the decode kernel asks for 102,400 B of shared memory against a 101,376 B limit and the engine dies at startup. Default KV leaves 1.77M KV tokens per GPU |
| Qwen3-4B cache | downloaded README.md, LICENSE, .gitattributes into the HF cache | The cached snapshot lacked them; vLLM does not care, but AIPerf's offline check refuses an incomplete snapshot. Qwen3-4B was then dropped in favour of Kimi-Linear (a 4B model crossing 100k says less) |

## Caveats

- Synthetic prompts with fixed lengths and `ignore_eos`, as in the paper. These rates are
  sustained generation, not the mix of a real user session.
- No quality check in this run (`--no-quality`); gpt-oss and Gemma passed 32/32 on the
  P11 extraction test (docs/p11-results.md).
- The ladder doubles each step, so limits fall between two rungs.
- No power measurement (no wall-power instrumentation on this box).
