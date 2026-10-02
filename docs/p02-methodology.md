# P02: CPU KV reuse, correctness and capacity

The GPU-only arm keeps production prefix caching and automatic GPU cache sizing.
The CPU arm adds the native hybrid-compatible CPU connector. Both use identical
model weights, revision, runtime, precision, GPU allocation and load traces.
A passing run is not automatically a publishable capacity claim.

## Run order

1. Diagnose correctness before any full-server capacity run. Use the controlled
   diagnostic runner below at 8K, 32K, 64K and 96K, first isolated/cache disabled,
   then cache-enabled reuse, then concurrent distinct contexts. Repeat with BF16
   and FP8 KV under matched settings. Target records and questions stay fixed as
   neutral background grows. This isolates context-length effects; it does not
   replace validation of the original 1,600-record workload. Preserve all failures.
2. Calibrate the empirical GPU-only hot working-set boundary for the chosen shape
   and full GPU allocation. Measure two blocks with at least four returns per
   document. Identify C with at least 95% GPU prefix hits and a larger population
   below 90%, with passing quality. C describes a reusable-prefix working set,
   not physical resident documents. Save raw evidence and its hash.
3. Run a fixed-population mechanism experiment at 2C and 4C. Set
   `session_prime_documents=true`, `session_min_returns=4` (prefer 8),
   `session_gpu_capacity_documents=C`, `session_capacity_multiples=[2,4]`, and
   `session_population_grid=[2*C,4*C]`. Supply `session_capacity_evidence`.
   Priming is scored and billed separately. Measurement retains the same corpus,
   with no replacement or deliberate cache flushing. Shuffled round-robin
   assignment gives every document returns while arrival times remain Poisson.
   This is a controlled reuse-distance workload, not a random-user traffic model.
4. Sweep total arrival rate. The minimum rate and duration must supply the return
   budget to every document. Frozen traces are also checked before GPU launch.
   Ensure CPU memory can hold the chosen corpus. Require completed CPU restores,
   inspect native queues, and retain client-limited and failed points. Doubling
   continues to the configured ceiling even with zero refinement rounds. Two
   overload points and a native queue knee are required; a ceiling without them
   is unresolved, not capacity. Timeout and drain costs stay visible.
5. Test growing knowledge-worker sessions separately: one outstanding turn per
   worker and the same completion-relative think time, quality and response SLO
   in both arms. More outstanding queued work alone is not more supported users.
6. Freeze promising rates/populations before confirmation. `mode=confirmation`
   requires at least five independent paired blocks and 600-second windows;
   P02 uses explicit priming instead of generic timed warmup. Adaptive extension
   is disabled in confirmation. Report each block and the paired block interval.
   The interval does not remove selection bias from discovery or establish
   general task quality. Increase the frozen repeat count if pilot variation
   demands it; do not stop when a favorable interval appears.

## Correctness runner

Prepare a dedicated endpoint using the pinned production model/runtime and
capture its resolved launch configuration. Do not modify another workload.
Copy `docs/examples/p02-correctness-endpoint.json` and fill in its identities.
The declared precision/cache state must agree with that captured configuration.
The runner does not configure or verify the serving process; this is an explicit
operator prerequisite, and these results cannot become capacity evidence.

```
python -m simulator.infrastructure.session_diagnostics \
  endpoint-plan.json /path/to/pinned/tokenizer /new/evidence/directory
```

Use a fresh output directory for each of six conditions (BF16/FP8 × isolated,
reuse, concurrent). No retries are automatic. At least ten fixed seed fixtures
are retained alongside exact prompts, expected records, raw reasoning/final
answers and scores. Every context must fit the endpoint's context limit plus
1,024 output tokens. The original 96K workload must pass again before scaling.

## Calibration evidence contract

The JSON contains `assets` from the harness fingerprint, exact `gpu_ids`,
`context_limit`, `document_records`, and `configuration_sha256` (SHA-256 of
`json.dumps(resolved_configuration, sort_keys=True)`, before model pinning).
For each of two blocks include hot and pressure observations in `points`, with
`block`, `population`, `arm="baseline"`, `quality_fraction`,
`measured_returns_per_document`, `repeat_gpu_hit_fraction`, `evidence_sha256`.
Evidence hashes refer to retained measurement bundles, not declarations of GPU
bytes. The runner checks identities, quality, repeated boundary observations and
freezes a copy in the campaign. Calibration must be refreshed after configuration
or fixture changes. This file is operator-supplied empirical evidence; the
validator does not independently reproduce its measurements.

## Accounting and outputs

Native exact metric families supply ordinary prefix query tokens, GPU hits and
CPU hits. Their residual is reported as **uncached prefix tokens**, including
new input. Preempted-query families are not mixed into ordinary counters.
Absent, changed, reset or inconsistent counters cannot become a token-savings
claim. Historical recomputation per correct request and historical recomputed
share remain null until request-level attribution is available. Hybrid-attention
token counters are not GPU FLOPs or cloud-billing savings.

Reports keep sustainable correct throughput, median/p95 TTFT, cache work, quality,
priming and failed loads separate. Single blocks do not produce uplift percentages.
Same-load repeat comparisons use independent blocks, not request-level resampling;
failed or missing blocks suppress the summary rather than improving it by omission.
All reports remain unqualified pending boundary, quality and attribution review.

## Current hardware gate

The original FP8 96K fixture and a BF16 control both failed correctness. The
recent-active-record variant also failed. No 8-GPU capacity result is established.
These code changes do not remove that gate or launch the old held queues.
