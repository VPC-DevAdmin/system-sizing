# MI350P single-card throughput claim: LFM2-24B-A2B at 42,000 output tokens per second

Status: hardening runs recorded 2026-11-16. Section 8 lists what was run and what is still open.

## 1. Summary

One AMD Instinct MI350P in a Dell PowerEdge R7725 served LiquidAI/LFM2-24B-A2B at 42,081 output tokens per second. That is the mean of three 10-minute holds, each on a fresh engine start, with about 8,060 requests in flight at the card's 450 W power cap. The three holds span 42,024 to 42,179, a range of 0.4%.

| Hold | Run | Period | Output tok/s (whole period) | In flight | GPU power | Requests cut by the client |
|---|---|---|---|---|---|---|
| 1 | run_415 | 600 s | 42,040 | 8,060 | 447 W | 108 of 97,458 (0.11%) |
| 2 | run_416 | 600 s | 42,024 | 8,064 | 447 W | 117 of 98,286 (0.12%) |
| 3 | run_417 | 600 s | 42,179 | 8,067 | 448 W | 93 of 98,304 (0.09%) |
| Mean | | | 42,081 | | | |

Each figure is the rise in the two engines' generated-token counters between the start and the end of the hold, divided by the elapsed time.

The figures quoted earlier (45,519, and 46,548 on a repeat) were readings from one 15-second measurement chunk. Throughput at this load is not flat: it holds near 48,000 for about 40 seconds, then drops to 22,000 to 32,000 for a few seconds while a wave of finished requests is replaced and their prompts are processed. A 15-second chunk mostly lands on the high part. The 10-minute mean includes about twelve full waves and replaces those readings.

Recommended position: "42,000 output tokens per second on a single MI350P", or "more than 40,000", with the settings in section 9 stated.

Two outside load tools agree with the harness (section 7.5). vLLM's own benchmark tool measured 41,566 on the engines' counters with a shared prompt prefix and 36,799 with fully unique prompts. AIPerf, with fully unique prompts, measured 36,856. Our own run with reuse off gave 36,918. The shared system prompt is worth about 12% at this shape, and any published figure has to say which case it is.

## 2. What the claim is, and what it is not

It is:

- A saturation throughput figure for one card: the highest output-token rate the card sustained with every stream held in flight.
- Measured from the engine's own generated-token counters.
- On BF16 weights with an FP8 KV cache. The weights are as published; the cache is 8-bit. A scored task set gave the same result with a 16-bit cache, with the 8-bit cache, and with the 8-bit cache under full load (section 7.4).
- Achieved with two vLLM engine processes sharing the one GPU.

It is not:

- A service-level figure. Each stream receives about 5.2 tokens per second (median 191 ms per token), and the 95th-percentile wait for the first token was 4.2 to 4.4 seconds.
- A statement about a typical model. LFM2-24B-A2B has about 2 billion active parameters per token. Section 7 explains why this model is unusually fast.
- A card-versus-card comparison. No other card has been run with this model and method (section 8).

Wording to avoid: the earlier 45,519 and 46,548 figures, which are 15-second readings; "42K tokens per second" without the model name, the request shape, the stream count and the two-engine configuration; any "fastest PCIe card" statement; any per-user or interactive framing.

## 3. Test system

| Item | Value |
|---|---|
| Server | Dell PowerEdge R7725 |
| CPU | 2 x AMD EPYC 9455, 48 cores each, 192 threads |
| Memory | 377 GB |
| GPU | 1 x AMD Instinct MI350P, 128 compute units, 143.7 GB, gfx950 |
| GPU firmware | IFWI 00189939, built 2026-06-12, part 113-350P-01-1K1-000A |
| GPU power cap | 450 W (card maximum 600 W); the card ran at the cap, 448 to 450 W |
| OS and kernel | Ubuntu 24.04.5 LTS, 6.8.0-149-generic |
| GPU driver | amdgpu 7.1.3.31500000 |
| ROCm | 10.0.0 (amd-smi 27.0.0) |
| Container runtime | Docker 29.1.3 |
| Engine | vLLM v0.34.0rc1.dev521+g9bc16ae6a (nightly) |
| AMD kernels library | AITER at commit 95d21128 |
| Engine image | `capsim/vllm-rocm:nightly-9bc16ae6-aiter-95d21128`, image id `sha256:0d41fce41a09…` (built on the box from the two commits above) |
| Benchmark harness | capsim roofline (headline sweep), run from the same host |

Both the engine and AITER are pre-release builds. Any publication has to say so and give the commits.

## 4. Model

| Item | Value |
|---|---|
| Model | LiquidAI/LFM2-24B-A2B, snapshot `20052c41d2a7ccca3d5325331dd54de0c4d9021e` |
| Precision | BF16 weights with FP8 KV cache; weights as published, 44.44 GiB loaded per engine |
| Architecture | 40 layers: 30 convolution, 10 full attention; hidden size 2048 |
| Mixture of experts | 64 experts, 4 active per token |
| Attention | 32 heads, 8 key-value heads |
| KV cache cost | about 10 KB per token (FP8 KV cache) |

The engine log confirms the code paths in use: attention backend `ROCM_ATTN` (chosen automatically), MoE backend `AiterUnquantizedExperts`.

## 5. Implementation

### 5.1 Two engines on one GPU

A single vLLM engine on this model stops at about 28,000 to 29,800 output tokens per second (four runs). At that point the GPU draws about 357 W, well under its cap, and the engine's single scheduling process sits at 100% of one CPU core.

Running two independent engine processes on the same GPU raises throughput, consistent with relieving a host-side bottleneck. Each has its own scheduling process, and the GPU alternates between them. With two engines the card reaches its 450 W cap and both scheduling processes run at about 100% of a core.

capsim implements this as the `replicas_per_gpu` option (`simulator/engines/custom.py`, `simulator/engines/docker_replica.py`, uncommitted at the time of writing):

- The GPU's device group is repeated once per engine.
- `gpu_memory_utilization` is divided by the engine count (0.95 becomes 0.475 each).
- Engines start one at a time, each waiting for the previous one to be ready. vLLM sizes its KV cache from free device memory during load, so two engines loading together leave the second with no cache memory.
- The load generator spreads requests across both engines' URLs.

### 5.2 Engine settings

Each engine was started with these non-default arguments (from the engine log):

```
--model LiquidAI/LFM2-24B-A2B
--trust-remote-code
--max-model-len 2048
--gpu-memory-utilization 0.475
--kv-cache-dtype fp8
--max-num-seqs 4096
--stream-interval 8
--api-server-count 4
```

Environment: `VLLM_ROCM_USE_AITER=1`, `HSA_NO_SCRATCH_RECLAIM=1`.

| Setting | Effect |
|---|---|
| `--max-num-seqs 4096` | Each engine batches up to 4,096 requests, 8,192 across the card |
| `--stream-interval 8` | Tokens are sent to the client in groups of 8, cutting per-token streaming work on the CPU |
| `--api-server-count 4` | Four HTTP front-end processes per engine, so request handling is not limited to one core |
| `--kv-cache-dtype fp8` | Halves KV cache memory; each engine holds about 1.98 million tokens of cache |

Resulting state per engine: 44.44 GiB of weights, 21.0 GiB of KV cache. At peak load the cache was 48% used, so memory was not the limit.

### 5.3 Run request

Each 10-minute hold was started through capsim's API with this request:

```json
{
  "workload": {"kind": "persona", "id": "headline_generation"},
  "shape": {"input_tokens": 128, "output_tokens": 256},
  "ladder": [8192],
  "hold_s": 600,
  "prefix_reuse": "industry",
  "new_run": true,
  "custom": {
    "engine": "vllm_cuda_multi",
    "model_id": "LiquidAI/LFM2-24B-A2B",
    "tp": 1, "replicas": 1, "replicas_per_gpu": 2,
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.95,
    "kv_cache_dtype": "fp8",
    "max_num_seqs": 4096,
    "trust_remote_code": true,
    "vllm_gpu_image": "capsim/vllm-rocm:nightly-9bc16ae6-aiter-95d21128",
    "vllm_extra_env": {"VLLM_ROCM_USE_AITER": "1", "HSA_NO_SCRATCH_RECLAIM": "1"},
    "vllm_extra_flags": ["--stream-interval", "8", "--api-server-count", "4"]
  }
}
```

The other hardening runs change one field each: `batch_requests` in place of `hold_s` (fixed batch), `"prefix_reuse": "off"`, `"kv_cache_dtype": "auto"`, or the shape. All arm definitions are in `/data/capsim/harden/arms/` on the box, generated by `/data/capsim/phase0/make_harden.py`.

## 6. Load and measurement method

### 6.1 Load

- Closed loop at saturation: 8,192 streams are held in flight with no think time. When a request finishes, the next one starts immediately.
- Each request: 128 input tokens, exactly 256 output tokens. End-of-sequence is ignored (`ignore_eos`) so every request generates the full 256.
- The 128 input tokens are a 71-token system prompt shared by all requests, plus 57 tokens unique to each request.
- Load generator: 16 capsim worker processes on the same host, using the streaming chat API.

### 6.2 Prompt reuse

Synthetic shared-prefix workload targeting approximately 54% reuse, motivated by the ideal business-API cache hit rate reported by Wang et al. (USENIX ATC '25); measured reuse was 56.3%. The reuse comes from a system prompt every request shares plus the chat template; the rest of each prompt is unique.

Reuse reduces prompt processing only. Every output token counted was generated. With reuse turned off (no shared prompt, engine prefix cache disabled) the same hold gave 36,918, which is 12% lower (section 7.3).

### 6.3 How the rate is computed

- Once all 8,192 streams are in flight, the harness reads each engine's `generation_tokens_total` counter, waits 600 seconds, and reads it again. The rate is the summed rise divided by the elapsed time. A hold is rejected if either engine's counter is missing or goes backwards.
- The same counters are also read every 15 seconds and once per second; those records show the swing inside the hold (section 7.1) and are not used for the headline.
- The client counts tokens independently from the `usage` record at the end of each response. Section 7.2 compares the two counts on a fixed batch of requests.
- Time per output token is computed from the server's token count for each response, so it is per token and not per 8-token group.

### 6.4 Limits of the method

1. **Request waves are synchronized.** All 8,192 streams start together and each takes about 49 seconds, so they also finish together, and the rate swings between about 48,000 and about 25,000 within each wave. The 10-minute mean covers about twelve waves. Real traffic arrives staggered; the mean under staggered arrivals has not been measured.
2. **The load generator shares the host.** Its 16 processes and the engines run on the same two CPUs.
3. **Engine counters are the engine's own.** The client-side count and the two outside tools in section 7.5 check the load generator and the arithmetic. They do not check vLLM's token counting against a different engine.
4. **About 0.1% of streams stop delivering.** Section 7.2 describes them. Their tokens are in the engine count and not in the client count.

## 7. Results

### 7.1 Ten-minute holds (the headline)

8,192 streams offered, 128 tokens in, 256 out, three fresh engine starts.

| Hold | Run | Output tok/s | In flight | Median second | Slowest second | TTFT p95 | Time per token, median | GPU power | KV cache used |
|---|---|---|---|---|---|---|---|---|---|
| 1 | run_415 | 42,040 | 8,060 | 47,498 | 22,137 | 4.4 s | 191 ms | 447 W | 49% |
| 2 | run_416 | 42,024 | 8,064 | 47,582 | 20,362 | 4.4 s | 191 ms | 447 W | 49% |
| 3 | run_417 | 42,179 | 8,067 | 47,646 | 20,660 | 4.2 s | 190 ms | 448 W | 49% |

Mean 42,081; range 155 tokens per second (0.4%). The pass condition set beforehand was a range within 5%.

The median second is higher than the mean because of the wave pattern described in section 6.4: about four seconds in five run near 48,000, and the seconds when a wave of prompts is processed run at 20,000 to 32,000. No request was preempted and the KV cache stayed under 71% at its per-second peak, so memory was not the limit.

### 7.2 Fixed batch: client count against engine count (run_418)

A fixed batch of 50,000 requests was sent at 8,192 streams and the engines were left to drain. Both sides were then counted.

| Quantity | Value |
|---|---|
| Requests sent | 50,000 |
| Requests answered in full | 49,955 |
| Requests cut by the client | 45 (0.09%) |
| Output tokens counted by the client (from `usage` records) | 12,788,480 |
| Output tokens counted by the engines | 12,799,788 |
| Client over engine | 99.91% |
| Prompt tokens, client and engine | 7,093,610 and 7,100,000 |
| Elapsed, first request to last | 312.9 s |
| Rate over the whole batch, including ramp-up and drain | 40,909 tok/s |

The pass condition was agreement within 1%. The two counts differ by 11,308 tokens, and that difference is the 45 cut requests: the engines generated 12,799,788 of the 12,800,000 tokens the batch asked for, so they produced nearly every token for those 45 requests as well, but the client did not receive them.

The 45 are requests the client recorded as `decode_stalled`: the stream delivered its first tokens and then nothing for 40 seconds, at which point the client closed it. The same thing happened to 0.09% to 0.12% of requests in each 10-minute hold, and to none of 81,920 in the hold with reuse off. No engine error or preemption was logged. Whether the stall is in the engine's HTTP front end or in the load generator has not been established. For the claim it means the rate delivered to clients is about 0.1% below the rate the engines count.

### 7.3 Sensitivity: prompt reuse off, and a longer shape

| Run | Change from the headline | Output tok/s | In flight | TTFT p95 | Requests cut |
|---|---|---|---|---|---|
| run_419 | No shared prompt, engine prefix cache disabled | 36,918 | 7,999 | 10.1 s | 0 of 81,920 |
| run_423 | 1,024 tokens in, 1,024 out, 8,192 streams offered | 20,440 | 4,080 | 87.8 s | 14,011 of 17,963 |
| run_430 | 1,024 tokens in, 1,024 out, 3,840 streams offered | 21,245 | 3,378 | 39.4 s | 3,385 of 12,372 |

Both are 600-second holds measured the same way as the headline.

- Turning reuse off costs 12%. The headline's shared system prompt is therefore worth about 5,200 tokens per second at this shape.
- At the longer shape the engines generate 20,000 to 21,000 tokens per second, but neither run is a clean service figure. At 8,192 streams the load was over-driven: each engine's KV cache holds about 2,040 requests of that length, so 4,080 ran and the rest waited in the queue until the client gave up. At 3,840 streams everything fit, but each wave's prompts take 45 to 50 seconds to process, and while they do the running streams slow to about 8,000 tokens per second in total. 27% of requests went 40 seconds without a chunk and were cut by the client. Tokens delivered to completed requests came to about 15,300 per second. A usable figure for this shape needs staggered arrivals, which has not been run.

### 7.4 Output check: 16-bit cache, 8-bit cache, and under load

Sixty short questions with one right answer each (arithmetic, capitals, general knowledge) were asked at temperature 0 in three conditions.

| Run | Condition | Correct | Seconds per answer |
|---|---|---|---|
| run_420 | BF16 KV cache, idle | 60 of 60 | 0.04 to 1.4 |
| run_421 | FP8 KV cache, idle | 60 of 60 | 0.04 to 1.4 |
| run_422 | FP8 KV cache, both engines asked, 8,192 streams in flight | 60 of 60 | 1.2 to 38 |

Every prompt, reply and verdict is stored in `/data/capsim/harden/quality/`. The set is small and easy. It shows that the engines return correct text with the 8-bit cache and while sharing the GPU under full load; it is not a quality benchmark and would not detect a small accuracy loss.

### 7.5 Outside load tools

#### `vllm bench serve` (vLLM's own load tool)

One client per engine, run inside that engine's container, both at once. Random-token prompts of 128 tokens, 256 output tokens, end-of-sequence ignored, temperature 0.7. Each run used the headline engine configuration, started fresh. The engine figures come from a 2-second sample of both engines' generated-token counters, leaving out the first 30 and last 60 seconds of load.

| Run | Prompts | Streams per engine | Requests (failed) | Client tok/s, whole batch | Engine counters, steady | Time per token, median | Cache hit rate | Nearest capsim figure |
|---|---|---|---|---|---|---|---|---|
| run_426 | 71-token shared prefix + 57 unique | 4,096 | 100,000 (0) | 41,075 | 41,566 | 192 ms | 49% | 42,081 (holds, 56% reuse) |
| run_428 | 128 unique | 4,096 | 100,000 (0) | 36,286 | 36,799 | 218 ms | 3% | 36,918 (reuse off) |
| run_429 | 71-token shared prefix + 57 unique | 2,048 | 60,000 (0) | 36,121 | 36,618 | 109 ms | 50% | 35,900 to 37,500 (earlier ladder, 15-second readings) |

- With a shared prefix the engines' steady rate under the tool's load is within 1.2% of the capsim headline; the tool's prompts gave a lower cache hit rate, which may account for part of that. With unique prompts it is within 0.3% of capsim's reuse-off run.
- "Client tok/s, whole batch" is the two clients' reported output rates added. It includes ramp-up and drain, like capsim's batch run (40,909).
- In each run the engine count exceeds the tool's by about 10,000 tokens. capsim kept one stream running to hold the engines up, and that stream accounts for it.
- Each client process used well under one CPU core; both engine scheduling processes were at 100% of a core.
- The tool has no limit on the gap between chunks, so its zero failures do not show whether any stream paused the way section 7.2 describes.

This is an independent client, so it checks capsim's load generator and arithmetic. It is vLLM's tool measuring vLLM, so it does not check the engine's own token counting.

Two first attempts lost 47,963 of each client's 50,000 requests to the container's open-file limit of 1,024 and are not used. The 2,037 requests per engine that got through in those attempts decoded at 66 to 68 ms per token, which would be about 61,000 tokens per second across both engines. Run_429 tested that at the same stream count under sustained load and measured 109 ms per token and 36,618 tokens per second. The 66 ms was a single wave decoding with no new prompts arriving, so it is the decode-only rate and not a throughput the card sustains.

#### AIPerf 0.13.0

AIPerf was run through capsim's third-party benchmark campaign against the recorded configuration of hold 1 (run_415): the same two engines, started fresh, reached through capsim's round-robin proxy. Settings: 128 tokens in, 256 out, end-of-sequence ignored, temperature 0, streaming chat, a 60-second warm-up, 600 seconds measured, and a set of 250,000 unique prompts so that no prompt repeats within a run.

| Streams | AIPerf output tok/s | Engine counters, same window | Requests | Failed | Time per token, median | TTFT p95 | Prefix-cache hit rate |
|---|---|---|---|---|---|---|---|
| 8,192 | 36,856 | 36,600 to 37,200 | 89,943 | 0 | 216 ms | 9.2 s | 0% |
| 4,096 | 32,936 | 32,700 to 32,900 | 77,824 | 0 | 122 ms | 3.0 s | 0% |

- AIPerf's prompts share nothing but the chat template, so the engines served no prompt tokens from the prefix cache. The matching capsim run is the one with reuse off (section 7.3): 36,918 against AIPerf's 36,856, a difference of 0.2%.
- The headline's 42,081 includes 56% prompt reuse from a shared system prompt. AIPerf was not run with a shared prompt, so it confirms the no-reuse figure directly and the headline only through that comparison.
- The engine-counter range is the rate over three choices of window inside the AIPerf run, read from a 2-second sample of both engines' counters.
- One AIPerf process (its timing manager) ran at 100% of a CPU core at both load points. The engine rate matched capsim's at 8,192 streams, so it does not appear to have limited the result there.
- AIPerf reports this as its own measurement; its figures are given as produced.

### 7.6 Earlier ladder, short windows (run_393)

| Streams offered | In flight | Reported output tok/s | Whole-period mean | Period | TTFT p95 | GPU power | Errors |
|---|---|---|---|---|---|---|---|
| 512 | 511 | 11,010 | 10,879 | 46 s | 0.9 s | 450 W | 0 |
| 2,048 | 2,006 | 27,976 | 27,627 | 31 s | 1.7 s | 450 W | 0 |
| 4,096 | 4,029 | 35,903 | 37,457 | 47 s | 2.4 s | 450 W | 0 |
| 8,192 | 8,174 | 46,548 | 43,010 | 127 s | 6.1 s | 448 W | 35 |

Throughput was still rising at the top of the ladder. 8,192 is the most the two engines accept at this batch setting, so the true peak has not been found. A profile of the fused expert kernel puts its ceiling near 48,000 at 4,096 streams per engine, which matches the rate between waves.

These rows are 15-second readings from before the harness could hold a rung, kept to show how throughput scales with streams. The 8,192 row is superseded by section 7.1. The 35 errors are `decode_stalled` requests of the kind described in section 7.2. Time per output token is omitted here: in this run it was recorded per 8-token group, which the harness has since corrected (section 6.3).

### 7.7 Related runs, short windows

All at 128 tokens in, 256 out, same image, 8,192 streams offered unless noted.

| Configuration | Reported | Whole-period mean | Notes |
|---|---|---|---|
| One engine, batch 4,096 | 28,030 to 29,806 | not extracted | Four runs; engine process at 100% of a core, GPU about 357 W |
| Two engines, batch 2,048 each | 37,653 | not extracted | 4,062 in flight, 4,067 queued |
| Two engines, batch 4,096 each (original) | 45,519 | 45,934 | 31 s period |
| Two engines, batch 4,096 each (repeat) | 46,548 | 43,010 | 127 s period |
| Two engines, AITER flash-attention backend | 40,213 | 42,679 | 112 s; 17 errors; 388 preemptions |
| LFM2.5-24B-A2B FP8, one engine | 28,612 | not extracted | Short-window reading |
| LFM2.5-24B-A2B FP8, two engines | 40,732 | 40,666 | Short-window reading, 31 s period |
| LFM2.5-24B-A2B FP8, three engines | 39,815 | 38,879 | Short-window reading, 99 s period; no gain from a third engine |

Two readings of this table matter for the claim:

- Every two-engine run on this model family lands between 39,000 and 46,000 on the whole-period mean. The result is repeatable in kind, not a one-off.
- A third engine does not help, which is consistent with the GPU at its power cap being the limit once two engines are running.

### 7.8 Why the number is plausible

- About 2 billion parameters are active per token, so the arithmetic per token is close to that of a 2B dense model.
- Only 10 of 40 layers use attention, and the KV cache is about 10 KB per token, so 8,000 streams fit in under half the cache.
- Liquid AI's own figure for this model is 26.8K total tokens per second (input plus output, at 1,024 in / 512 out) on one H100 at 1,024 streams. It is a different quantity from our output-only rate, so it shows the model is fast, not that the two results agree.
- The step from one engine to two is consistent with relieving a host-side bottleneck: with one engine, one CPU core is at 100% and the GPU is under its cap; with two, the GPU is at its cap.

## 8. Hardening status

| # | Run or change | Status | Result |
|---|---|---|---|
| 1 | Hold 8,192 streams for 10 minutes, rate from counters at the two ends | Done | 42,040 (run_415) |
| 2 | Count tokens on the client and compare with the engine counters | Done | 99.91% on a fixed batch of 50,000 requests (run_418); pass limit 1% |
| 3 | Three holds, each on a fresh engine start | Done | 42,040 / 42,024 / 42,179; range 0.4%; pass limit 5% |
| 4 | Output check idle and under load, 16-bit and 8-bit cache | Done | 60 of 60 correct in all three conditions (run_420 to run_422) |
| 5 | Sensitivity: prefix reuse off | Done | 36,918, 12% lower (run_419) |
| 6 | Sensitivity: 1,024 in / 1,024 out | Done, not clean | 20,440 and 21,245 generated (run_423, run_430); 27% of requests cut at the lower load because synchronized prompt waves stall running streams |
| 7 | `vllm bench serve` at the same shape and stream count | Done | 41,566 with a shared prefix (1.2% from the headline); 36,799 with unique prompts (0.3% from reuse off); 36,618 at half the streams |
| 8 | AIPerf at the same shape and stream count | Done | 36,856 with unique prompts and no reuse, within 0.2% of capsim's reuse-off run; 32,936 at 4,096 streams |
| 9 | Staggered request starts | Not run | The synchronized-wave pattern remains (section 6.4); it is also what spoils the 1,024 in / 1,024 out runs |
| 10 | The same model and method on one RTX PRO 6000 Blackwell in the XE7740 | Not run | No card comparison can be made until it is |

Open items:

- The cause of the stalled streams (section 7.2): about 0.1% of requests, none with reuse off.
- Prompt processing costs more than its share of tokens. With no new prompts arriving, two engines at 2,037 streams each decoded at about 61,000 tokens per second; under sustained load at the same stream count they gave 36,618 (section 7.5). How prompts are batched with running streams is the place to look for more throughput, and it has not been explored.
- Extend past 8,192 streams (needs a higher per-engine batch setting) to find the true peak.
- A latency-bounded row: the highest rate at which the first-token wait stays under a stated limit.
- Commit the harness changes and `replicas_per_gpu` so the published configuration maps to a commit. They are uncommitted at the time of writing.

## 9. Disclosures to carry with any published figure

- Model name and precision: LiquidAI/LFM2-24B-A2B, BF16 weights with FP8 KV cache.
- Request shape: 128 tokens in, 256 out, end-of-sequence ignored.
- Streams in flight (about 8,060) and the fact that this is saturation throughput with no latency limit: about 5.2 tokens per second per stream, first-token wait 4.2 to 4.4 seconds at the 95th percentile.
- Two vLLM engine processes on one GPU.
- Engine and library versions, both pre-release, with commits.
- Prompt reuse: 56% of prompt tokens served from the prefix cache through a shared system prompt. Without it the figure is 36,918.
- GPU power cap of 450 W.
- Measurement: mean of three 10-minute holds, from the engines' generated-token counters.
- About 0.1% of requests were cut by the client after a 40-second gap in delivery.

## 10. Evidence

On the R7725 (`ssh amdgpu`). Run directories are under `~/system-sizing/runs/`.

| Figure | Run directory | Summary file |
|---|---|---|
| 42,040 (hold 1) | `run_415` | `/data/capsim/harden/results/r1a_hold600.json` |
| 42,024 (hold 2) | `run_416` | `/data/capsim/harden/results/r1b_hold600.json` |
| 42,179 (hold 3) | `run_417` | `/data/capsim/harden/results/r1c_hold600.json` |
| Fixed batch, 99.91% | `run_418` | `/data/capsim/harden/results/r2_batch50k.json` |
| Reuse off, 36,918 | `run_419` | `/data/capsim/harden/results/r3_hold600_reuse_off.json` |
| Output check, BF16 KV idle | `run_420` | `/data/capsim/harden/quality/r4a_quality_bf16kv_idle.json` |
| Output check, FP8 KV idle | `run_421` | `/data/capsim/harden/quality/r4b_quality_fp8kv_idle.json` |
| Output check, FP8 KV loaded | `run_422` | `/data/capsim/harden/quality/r4c_quality_fp8kv_loaded.json` |
| 1,024 in / 1,024 out, 20,440 | `run_423` | `/data/capsim/harden/results/r5_hold600_1k1k.json` |
| `vllm bench serve`, 41,566 | `run_426` (engines held up by capsim) | `/data/capsim/harden/vbench/v1b_vbench_shared_prefix/` |
| `vllm bench serve`, unique prompts, 36,799 | `run_428` | `/data/capsim/harden/vbench/v2b_vbench_no_prefix/` |
| `vllm bench serve`, 2,048 per engine, 36,618 | `run_429` | `/data/capsim/harden/vbench/v3_vbench_shared_prefix_2048/` |
| 1,024 in / 1,024 out at 3,840 streams, 21,245 | `run_430` | `/data/capsim/harden/results/r5b_hold600_1k1k_3840.json` |
| AIPerf, 36,856 and 32,936 | `benchmarks/fc7ee79ef6dba766bbbeb4b9/attempts/0001-aiperf` and `0002-aiperf` | `profile_export_aiperf.json` in each; engine counter samples in `/data/capsim/harden/aiperf/counters.log` |
| Earlier 15-second readings, 45,519 and 46,548 | `run_378`, `run_393` | `/data/capsim/phase1c/results/d1_vllm_srv_x2_lfm2.json`, `e5_vllm_x2_lfm2_repeat.json` |
| Related short-window runs | `run_391`, `run_394` to `run_396` | `e3`, `e6`, `e7`, `e8` in `/data/capsim/phase1c/results/` |

In each run directory: `headline_sweep.json` (the rung record, with the per-engine counter readings at both ends of the hold), `run.db` table `measurement_telemetry` (per-second record), `engine_vllm_cuda_multi_*.log` (both engines), `resolved-assets.json` (model file fingerprints and image id).

In the `vllm bench serve` directory: `engine0.json` and `engine1.json` (the tool's results), `gen_counters.log` (engine counters every 2 seconds), `counters_start.txt` and `counters_end.txt`, `top.log`.

Arm definitions: `/data/capsim/harden/arms/`. Scripts: `/data/capsim/phase0/make_harden.py`, `quality_probe.py`, `vbench.sh`.

Measurement code: `simulator/headline_sweep.py` (`_hold`, `_batch`, `counter_growth`), `simulator/engines/docker_replica.py` (`get_replica_counters`).

## Appendix A. Context: published figures

These are context, not comparisons. We found no published figure this high for one card; none of the figures below used this model at this stream count, so they do not show that the card is faster than another.

Taken from search summaries on 2026-11-16; the source pages have not been read in full.

| Card | Source | Model and load | Output tok/s per card |
|---|---|---|---|
| MI350P | StorageReview first look | FP8 model, 128 streams | about 11,045 |
| MI350P | AMD MLPerf v6.1, 8 cards | GPT-OSS-120B, offline | about 6,100 |
| MI350P | LIQID | 7B-class, FP4, 128 streams | about 4,653 |
| RTX PRO 6000 Blackwell | CloudRift | Qwen3-Coder-30B AWQ | about 8,425 |
| RTX PRO 6000 Blackwell | Ours, XE7740 | Gemma-4-26B NVFP4 | 15,919 |
| H100 SXM (not PCIe) | Liquid AI | LFM2-24B-A2B, 1,024 streams, 1,024 in / 512 out | 26,800 total tokens (input plus output), not output only |

## 11. Sources for published comparisons

- StorageReview, AMD Instinct MI350P First Look: https://www.storagereview.com/review/amd-instinct-mi350p-first-look
- AMD ROCm blog, MLPerf Inference v6.1 submission: https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-v6.1/README.html
- LIQID, MI350P performance benchmarks: https://www.liqid.com/resource/mi350p-performance-benchmarks
- CloudRift, GPU benchmarks for LLM inference: https://www.cloudrift.ai/gpu-benchmarks
- Liquid AI, LFM2-24B-A2B: https://www.liquid.ai/blog/lfm2-24b-a2b
