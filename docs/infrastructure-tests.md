# Infrastructure Tests

The Infrastructure Tests tab appears in both Classic and Workspace. It contains the ten planned projects and their 31 separately defined base report outputs. Load grids expand applicable outputs into separate reports per tested load.

## Current implementation

P01 runs matched Dynamo aggregated/disaggregated pilots. P02 runs finite document-session replay against CPU or CPU-plus-disk cache tiers. P03 runs exploratory KV precision grids. P04 runs paired speculative decoding on a recorded vLLM CUDA roofline configuration. P05 implements pinned teacher/student image generation and explicit acceptance review. P06 implements dedicated, software-shared and MIG service placements. P10 implements separate CUDA profiling and a matched attention-backend comparison. These seven adapters can launch exploratory tests; real-host validation status and results are recorded separately. P07, P08 and P09 remain unavailable and cannot launch placeholder measurements.

For P04, choose a recorded model configuration, GPU indices, concurrency, document size and run purpose. Both arms use the same resources, generated documents, JSON schema and exact-answer grader. The treatment adds either prompt n-gram speculation or native MTP decoding. MTP requires declared draft layers and indexed MTP tensors in the selected checkpoint. This experiment tests structured extraction from synthetic logistics documents. It does not establish maximum server capacity or broad model quality.

Pilot defaults are short and always produce exploratory results. Confirmation requires at least five paired blocks, ten-minute measurement windows and two-minute warmup windows. Incorrect answers, truncated responses and transport failures remain in the denominator. New campaigns also require at least the configured fraction of attempted requests to be correct and within the response-time target (95% by default). Each pair gets randomized arm order. All settings are frozen for the campaign.

## Evidence

Each campaign has a new UUID. Its directory includes the immutable manifest, generated corpus, source snapshot, model and image fingerprints, environment, engine commands and logs, streaming request records, raw engine counters, measurements, comparisons and output-specific reports. DOCX, Markdown and JSON outputs are exported separately; the ZIP contains shared evidence with SHA-256 checksums.

A pilot, incomplete pair or failed quality gate cannot produce a qualified claim. A confidence interval entirely above zero is required before the output marks a positive improvement as supported. Qualification is specific to the recorded workload and concurrency.

## Recovery and ownership

The runner holds the same hardware lease as the existing engines. Containers have unique names and campaign ownership labels. Cancellation joins startup threads and stops owned containers before releasing resources. The recovery button verifies both the recorded resource and its Docker ownership label before cleanup. It never sweeps unrelated containers.

Remeasurement creates a fresh campaign. Source changes do not rewrite or resume an old measurement under new code. Keep failed campaigns as diagnostic evidence.

## CLI

```bash
capsim infrastructure list
capsim infrastructure preflight
capsim benchmark results
capsim infrastructure run RESULT_ID settings.json
```

P05 ignores the source selection; use `capsim infrastructure run - image-settings.json` with a P05 spec.

CLI execution owns the runs directory. Use the UI/API while the capsim service owns that directory.

Example pilot settings:

```json
{
  "project_id": "P04",
  "mode": "pilot",
  "paired_blocks": 2,
  "duration_s": 60,
  "warmup_s": 15,
  "concurrency": 4,
  "gpu_ids": [0],
  "structured_output": true
}
```

## Execution requirements

P01/P02 require a pinned Dynamo deployment and verified transfer/cache semantics. P03 requires cache capability and quality sweeps. P05 prepares the pinned released teacher/student pair and requires explicit image acceptance review for accepted-image measurements. P06 requires an idle GPU with the supported 48 GB partition profile and passwordless nvidia-smi administration; every run verifies restoration and tenant contracts. P07 requires a supported attested confidential VM and protection path. P08 requires a validated whole-system power source. P09 requires a dated cost ledger and equal-demand accounting. P10 requires profiling, numerical validation and a compatible backend comparison.

GPU board power cannot substitute for whole-system input power. The power-control and cost-calculation utilities are unit tested but are not yet wired into completed P08/P09 experiment adapters.

## Cache grid pilot

The P03 worker implements a bounded document-size/concurrency grid with BF16, FP8 and per-token/head INT4 cache arms. Every arm explicitly selects TRITON_ATTN; inherited cache/backend flags are removed. Frozen corpora, exact input token counts, quality/SLO gates and per-cell engine counters are preserved. Reports describe passing tested points as lower bounds; they do not claim an established maximum or measured KV memory savings.

P03 is available for exploratory pilots after a three-arm Llama smoke test on XE7740. Explicit BF16, FP8 and INT4 runtime settings were verified from engine logs; all eight measured smoke-test requests were correct. This validates execution, not long-context capacity. Other model/backend combinations remain experimental. The pinned vLLM FlashInfer backend gates NVFP4 KV cache to SM100-family devices; that path is not used on the XE7740's SM120 GPUs. Each arm checks the effective backend and cache dtype before measurement, preventing a checkpoint default from silently changing the reference precision. BF16 quality checks use the lowest configured concurrency; FP8 and INT4 cover the full grid.


## P01 Dynamo comparison

P01 now has a pilot adapter for equal-budget aggregated versus disaggregated serving. It pins Dynamo 1.4.2 by OCI digest and records upstream commit 2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a. Install the pinned image from the adapter before launch. Select at least two GPUs and assign at least one to prefill and one to decode. Both arms use Dynamo, TP1 workers, the same checkpoint and cache precision, and disabled prefix caching. Arbitrary original roofline engine flags are replaced with the matched Dynamo settings.

Each arm owns one container with a frontend and all workers, private file discovery, separate logs, and checked auxiliary ports. An exiting child ends the deployment. Existing ownership journals support interrupted cleanup. The treatment must show increasing NIXL byte counters at decode workers during measurement; requested flags alone do not establish transfer.

Reports describe fixed closed-loop concurrency, not fixed offered arrival rate or proven peak capacity. O1 and O2 can reuse the measured requests; O3 remains unavailable until a separate GPU scaling sweep is measured. Runtime and model compatibility must be demonstrated before treating the pilot as functional.

## P02 returning document sessions

P02 uses the pinned Dynamo runtime with GPU-only prefix caching as its baseline and a separately selected CPU or CPU-plus-disk KVBM treatment. Both arms use the same explicit GPU KV allocation. The finite trace primes a cohort, verifies one immediate return with at least 90% native GPU-cache hits, sends disjoint documents to pressure the cache, and resumes every session once. Each phase retains every response and native telemetry. Resume throughput excludes prime and pressure time; their costs are recorded separately. A changed instruction suffix avoids treating the exact original request as the whole session interaction.

The initial four-GiB GPU cache setting is a bounded adapter-validation configuration, not an optimized XE7740 capacity baseline. Use representative GPU/host cache budgets and larger session cohorts before making deployment claims. Only resume TTFT is currently exported as a measured P02 output. Maximum retained sessions, avoided recomputation, multi-turn conversation quality, and sustainable active-user capacity remain separate measurements. CPU/disk reload counters must increase during measured resumes; a tier that starts but never serves a returning session fails verification.

The disk cache lives in a block-backed anonymous Docker volume mounted at the owned container's `/tmp`. The volume is removed during normal cleanup or recovery. A read-only `/run/udev` bind mount provides cuFile discovery metadata for the container. Its free-space preflight uses Docker's actual storage root. It is ephemeral cache, not durable conversation storage. No host mounts or unrelated caches are deleted.


### P03 matched cohort revision

Every measured cache grid cell now replays the same `cache_samples` documents exactly once (default 32). Warmup remains time based and uses a disjoint corpus. Measurement duration does not apply to these finite cells. The BF16 reference gates only the matching document size; all reference failures remain explicit. Raw output and drain time remain in evidence. These small grids report tested lower bounds, not capacity maxima or broad quality equivalence.

### P04 explicit matched loads

Optional `speculative_concurrencies` runs separate warmup and measurement windows at each requested load within the same owned engine launch. Every load produces its own DOCX/JSON/Markdown reports. The grid does not select the best cell or imply a roofline uplift. Native drafted-token counters must increase on every treatment replica; all-rejected drafts remain a valid negative performance result. TTFT is retained alongside completed-answer latency. The resolved treatment configuration is downloadable for a separate matched roofline rerun; no roofline winner is changed by these task measurements.

### P05 diffusion pilot

P05 is independent of LLM roofline selection. It prepares the pinned Qwen-Image teacher and NVIDIA Qwen-Image-Flash student, then runs a randomized teacher-step sweep (20/50 by default) against the four-step student in one isolated Diffusers environment. The student is a released FastGen/DMD2-distilled checkpoint; this adapter does not run a distillation training job. Each request includes text encoding, denoising and VAE decoding; pipeline rate excludes evidence-file writing. Both use 1024×1024 BF16 with frozen prompts and seeds. The student trajectory is verified against the model card.

Generated PNGs, hashes, failed attempts, warmup, runtime settings, model revisions and raw rates are exported. Each claim remains exploratory; accepted-image outputs need a named human review using the fixed rubric. The UI hides arm, steps and timing during review and records every revision. Automated model review cannot be silently promoted to human acceptance. Wall energy, cost and training break-even outputs remain unavailable without their own evidence. Initial preparation needs 150 GiB free; cached reruns need a 20 GiB reserve.


### P06: independent service placements

The pilot compares two dedicated GPUs, two independent processes sharing one GPU, and two 48 GB MIG partitions on the last selected GPU. Use a model that fits a partition with the selected KV allocation; the Qwen NVFP4 winner is the initial candidate. Each tenant receives the same fixed arrival rate. A separate noisy phase increases tenant 1's offered load while retaining tenant 0's rate. Queueing, admission failures and drain time count. Dedicated, software-shared and MIG results remain separate. Density is reported only when every steady tenant passes its quality and latency contract. This is a fixed two-service test, not maximum density or confidential computing.

The MIG transaction requires an initially idle GPU with MIG disabled and no existing partitions. It records ownership before changes, uses only GPU-UUID-scoped nvidia-smi commands, refuses unexpected instances, and restores the original disabled state. No reset or reboot is automated. A failed restoration remains a failed campaign and can be retried through owned recovery after investigating active processes.

### P10: attention backend experiment

A separate engine launch captures a bounded CUDA trace before the A/B measurement. FLASHINFER and TRITON_ATTN are explicit choices with FP8 KV and BF16 model compute held constant. Each launch verifies the actual backend in runtime logs. Performance launches have no profiler. Four matched first-token probes compare top token, top-20 overlap and common-token logprob differences. This limited numerical check supplements task grading; it does not establish full model equivalence.

Latency, throughput and profiling evidence have separate reports. Backend comparisons are not isolated softmax-kernel claims. The diagnostic sets max_iterations=4, and summed kernel durations may overlap. Missing CUDA events, backend fallback, or a failed numerical gate prevents a positive claim.

### P01 PCIe CUDA IPC diagnostic control

The transfer settings expose explicit CUDA IPC GET (`UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y`). UCX otherwise may disable this path on PCIe GPUs without NVLink. Keep the runtime default for existing experiments; enable this option for a new frozen comparison after validating native payload correctness on the host. It applies to both arms, is recorded in the evidence, and does not establish a serving-performance claim by itself. Runtime evidence records both `nixl` and `nixl-cu13` package versions.

P10 exposes matched FP8 or BF16 KV precision in both backends. The runtime verifies cache and compute precision from startup logs. A failed FP8 numerical comparison may be followed by a separate BF16 comparison; it remains separate evidence, and the parity thresholds must not be relaxed to obtain a positive result.


# Infrastructure headline optimization plan

## Objective
Find useful performance and capacity advantages, then confirm the strongest stories. Correctness, matched inputs, service targets and failed results remain visible. Exploratory screening takes priority over repeated confirmation of unpromising configurations.

## P01 — more completed document work on the same GPUs
Compare eight aggregated workers with 2P6D, 3P5D, 4P4D and 5P3D. Use the pinned Dynamo 1.4.2 stack and verified CUDA IPC GET setting. Sweep identical fixed offered arrival rates in every placement; require both first-token and complete-response targets. Record client queue, native worker queue/prefill/decode counters, transfers, failures and drain. Tune batching around promising allocations with the same budget for the aggregate comparator. Zero queueing is not the objective; passing useful demand without sustained backlog is.

Separate outputs: highest passing tested request rate (not an unbounded maximum), first-token latency and completed-response latency at matched offered demand. Every output retains all allocation/load cells; a shared chart shows the curves. A later 8K/1K AIPerf workload is separate from document extraction and cannot be labeled a reproduced NVIDIA result until its recipe matches.

## P06 — transcribe and summarize on one GPU instead of two
Compare two dedicated GPUs (one speech recognition model, one language model), one software-shared GPU, and one GPU with two MIG partitions. Initial models: Whisper large-v3 through faster-whisper and the cached Qwen NVFP4 language model. They perform distinct stages of the same pipeline. Identical models, revisions, precision, beam settings, summary prompt, audio, output limits and KV budget across placements. GPU partitions and all owned containers must be restored after the run.

The first runnable pilot uses frozen synthetic support-call recordings generated by eSpeak, with a known transcript and four-field structured summary: issue, action, owner and deadline. This is a functional and exploratory workflow pilot, not representative contact-center audio. Native transcription is evaluated against the reference with normalized word error rate. The downstream model receives the actual ASR transcript, never the reference. A completed business item must pass both transcription and structured-summary grading and the end-to-end target. Real-call validation and free-form summary acceptance remain separate follow-up work.

Calls arrive at fixed rates. Stages may overlap across different calls in all placements, including dedicated GPUs. Latency includes admission and stage waiting, ASR, HTTP handoff and summary generation. Rejected arrivals and timeouts remain in the denominator; final drain is included. Count audio-hours completed per elapsed hour and preserve every stage result. Warmup uses separate recordings excluded from measured totals.

Separate outputs: qualified audio-hours/hour; p95 end-to-end latency at equal offered demand; passing one-GPU workload versus two-GPU baseline and capacity retained. A 50% GPU-count reduction is available only at demand where both baseline and treatment pass. No power or dollar saving is inferred.

### PCIe interpretation
Do not assume dedicated placement moves large model tensors between GPUs. The standard pipeline passes a small text transcript through the host, including when models share a GPU. MIG does not imply direct inter-partition transfer or zero PCIe traffic. Record transcript bytes and handoff timing; attribute results to measured resource placement and stage behavior. Any PCIe causality claim requires a separate transfer experiment.

## Remaining portfolio priorities
1. P04: tune draft-token counts and batching; challenge the original roofline with matched workloads; produce document-throughput and response-latency outputs separately.
2. P02: increase returning-session population and cache pressure; find passing retention capacity and resume latency. Prefer CPU tier; disk is for a separate cold-session service target.
3. P05: complete blind human image acceptance, then optimize accepted images/hour. Raw generation speed is not usable-image throughput.
4. P03: test INT4 only where FP8 is memory constrained; require passing quality and useful latency before a context-capacity claim.
5. P10: park the current backend candidate. Numerical repeatability diagnosis precedes any new softmax claim.
6. P07–P09: retain documented protection, whole-server telemetry and pricing prerequisites.

## Implementation and execution checklist
- [ ] Add split and fixed-arrival search controls, validation, comparisons and separate reports to P01.
- [ ] Add the post-call workflow option, speech runtime, frozen audio, per-stage grading and reports to P06.
- [ ] Test both UI layouts and regression gates, including rejected arrivals and cleanup.
- [ ] Deploy only while idle; execute GPU campaigns serially and preserve failed attempts.
- [ ] Run the P01 split search and post-call dedicated/shared/MIG pilot; fix execution errors and rerun.
- [ ] Document measured winners, negative cells, setup limits and the next focused optimization.

## Joint document shape and GPU allocation search

Enable **Find the best document shape and GPU split** on P01. Select document
record counts, extracted answer counts, prefill GPU counts, and an increasing
arrival-rate grid with at least three values. The remaining GPUs decode. Every
shape also runs against an equal-GPU aggregate deployment. Shapes are frozen
before execution and reused across placements. Answers terminate naturally and
must match the requested records exactly; measured token ranges accompany results.

For each shape and placement, the driver probes the lowest and highest rates,
then bisects the passing/failing boundary. A failed lowest rate stops that cell;
a passing highest rate is labeled grid ceiling. The declared time limit bounds
the whole campaign including launch and cleanup. Interruption retains partial
measurements and reports incomplete search. Rate bisection assumes a monotonic
boundary; irregular pass islands may be missed, so this is a measured search
within the selected range, not proof of a global maximum.

Three separate reports identify capacity uplift, absolute document capacity, and
matched-rate completion latency. An uplift headline requires a baseline failure
above its passing rate; the conservative comparison uses that failing rate as
the denominator. A still-passing baseline grid ceiling cannot establish an
uplift peak. Search-selected peaks remain exploratory. Absolute request rates
across different document sizes are not interchangeable workloads. All measured
cells, native transfer evidence, decision records and token ranges remain in the
ZIP; CSV and SVG curves can be reused in white papers. The UI displays per-shape
curves and all failures. This mode searches shape, arrival rate and GPU allocation;
it does not independently tune every possible aggregate engine setting or
replace an existing roofline winner automatically.

Goal search refines between the minimum and maximum declared rates at the selected
`p01_rate_resolution` (default 1 request/second), retaining any supplied interior
rate points. It automatically sizes client concurrency for the offered load,
bounded at 512. Client-limited failures cannot establish the aggregate server
capacity boundary; per-cell client concurrency and queue measurements are exported.

### PCIe and CPU placement

Goal search can use isolated socket-local pools, a forced cross-socket control,
or the historical unrestricted pool. The physical PCI address and NUMA/socket
membership of every selected GPU are read from the host; unknown or incompatible
layouts fail preflight. Each isolated pool has a separate frontend, namespace
and file-discovery store. Workers are CPU-affinitized to the socket attached to
their GPU; host memory binding is not claimed. Native transfer checks use the
actual decode-worker map, including noncontiguous worker indices.

On the XE7740 with GPUs 0–3 on one CPU and 4–7 on the other, three prefill GPUs
out of eight can remain local: 1 prefill + 3 decode in one group, and 2 prefill +
2 decode in the other. The client routes to the pool with fewer outstanding
requests rather than forcing equal traffic. The effective pool assignment,
per-request pool, CPU affinity and allowed KV-transfer pairs are retained.
The cross-socket control exchanges decode-pool membership while keeping physical
GPU roles unchanged. Odd splits also change the role ratio inside each control
pool; they are deployment comparisons, not an isolated measurement of link latency.

### Shared A/B points in goal search

Every declared arrival rate is measured for each workload shape and allocation before finer rate-grid bisection. This preserves matched latency comparisons even when aggregate serving passes at the upper limit. Each window drains before the next warmup. A failed lower rate followed by a passing higher rate is marked `non_monotonic`; its capacity boundary cannot generate a capacity headline. The measured rows remain available for investigation.

### Worker failure recovery

Goal-search allocations allow one retry with identical serving settings after all owned processes are cleaned up. Completed measurement rows remain; retries skip them and store new evidence under `retry-01`. Failures remain in `state.attempts` and `failure.json`, with point, error, traceback and evidence location. After a second serving failure, the runner moves to the next allocation and keeps the incomplete search visible. Cleanup failures and cancellation always stop the campaign. The container supervisor records the first exited service and exit code in `service-exit.json`.

Diagnostic `p01_cuda_launch_blocking` enables synchronous CUDA error reporting and suppresses performance peaks. `p01_execution` can request `sync` (disable async scheduling) or `eager` (also disable CUDA graphs); the same setting applies to both arms and remains fixed across retries.

## P01 token throughput search

`p01_token_search` runs the existing roofline `run_headline_sweep` against persistent Dynamo aggregate/disaggregate allocations. It is separate from the document extraction/arrival-rate mode. Use `p01_token_shapes` for input/output content-token pairs, `p01_streams` for the closed-loop concurrency ladder, `p01_engine_max_seqs` for engine batch width, and the existing split/placement grids. The UI exposes token-search controls and reports output tokens/s plus median/p95 TTFT. Prompt template overhead remains part of actual model input; the requested shape uses the roofline persona convention.

Each allocation launches once for all shapes. The shared roofline sweep applies its convergence and plateau rules, uses multiple load-generator processes, and leaves the caller-owned engine running between shapes. Per-shape load generators are stopped before the next shape. The worker raises its own open-file soft limit within the existing hard limit before GPU work; insufficient hard limits fail preflight. Failed serving allocations retry once after owned cleanup, preserving completed shape measurements. Editor-provided orphan cleanup remains intact.

All worker metrics must be scraped; missing workers cannot become a lower throughput result. Aggregate output counters sum all workers; disaggregated output counters sum decode workers so intermediate prefill computation is not counted twice. Prefill counters, transfer evidence and all-worker activity are retained separately. Uplift ranking requires convergence, at least 30 finished requests, at most 1% errors, and no discarded scrape chunks. It is the best observed result in the tested grid, not proof of global optimality. Same-concurrency TTFT is explicitly not same-arrival-rate TTFT. The controller records historical roofline entries as context unless their full measurement contract permits reuse.

Three editable output reports and a checksummed evidence ZIP retain the measured curves and methods. The first production batch uses Gemma, Mistral Small 3.2, GPT-OSS-120B and Qwen3.6. Legacy-only sources require a one-cell roofline refresh to record resolved configuration before campaign creation. Model-specific compatibility and kernel failures remain visible; a large MoE model is not a dense-model substitute.

### Token-search recovery and runtime compatibility

P01 records optional immutable runtime images, attention backend, chat processor,
and NIXL KV lease duration in the experiment specification. Apply each runtime
choice to both aggregate and disaggregated arms. A changed runtime requires a
fresh matched comparison; prior measurements remain historical evidence.

Token-search connection and timeout failures receive the same bounded allocation
retry as serving failures. Cleanup finishes before relaunch, and completed shapes
are retained. A failed aggregate baseline stops the campaign; missing treatment
shapes mark the search failed rather than complete. Cancellation and cleanup
failures remain terminal. Workers must drain before the next shape begins.

A faster aggregate rung that has not converged blocks a peak-uplift claim until
that baseline is resolved. Short recovery grids are exploratory; extend the load
ladder if throughput is still increasing before calling the result a roofline.


## P02 document-analysis capacity mode

In Infrastructure tests, select **P02 → Document-analysis capacity**. Select a cached roofline model that fits one GPU; this mode uses one TP1 replica per selected GPU. A single-GPU run is a feasibility test. Select all eight XE7740 GPUs to measure the server directly; do not multiply a single-GPU result by eight.

Baseline is GPU-only prefix caching. Treatment adds Dynamo CPU KV offload. Both use automatic GPU KV allocation and the same pinned model, image, precision, context, GPU memory utilization, sequence limit and sticky document routing. It does not disable baseline caching or reduce its GPU memory to force an advantage. CPU cache size is **per replica**; host and NUMA-node free-memory checks reserve additional headroom.

### Demand and scoring

- Choose a grid of document populations and mean seconds between questions per document. Both arms receive the same seeded Poisson trace at each point. Total offered rate is population divided by the mean interval.
- Each document has its own unique prefix. Independent questions reuse that document until the configured question count replaces it. Questions may overlap; this is document Q&A, not sequential multi-turn conversation.
- All first touches and replacements count. No artificial pressure/eviction phase is inserted. Readiness probing uses unrelated content; there is no corpus warmup.
- Initial fixtures ask for exact records from synthetic documents and require valid, correct JSON. This is an objective harness fixture, **not evidence of general legal, clinical or code-review quality**.
- Documents are tokenized locally with the pinned model before launch; context/output-budget violations stop the run. Structured-output support must be available in the chosen runtime/model.

### Reading results

Every declared load is measured in randomized paired order. There is no local-plateau early stop. The final half of each arrival window measures sustained correct requests/s, output tokens/s and outstanding-queue slope. All-window totals include the drain. TTFT median/p95 and service-target capacity are separate from unconstrained stable capacity.

The capacity metric is the largest **tested active document population at an equal per-document question cadence** that passes correctness/error gates without growing backlog. It is not the number of simultaneous decoding streams or the number of documents that fit in memory. A passing highest population means the grid needs extension. Client scheduling lag, admission rejection or persistent client queue saturation cannot establish a GPU capacity boundary.

Three separate DOCX/claim outputs share an evidence ZIP:

1. Document-analysis capacity and passing populations.
2. Matched-load TTFT and latency-target results.
3. Cache/prefill evidence and queue behavior.

CPU reloads must be observed to attribute a benefit to CPU memory. Native cache counters and raw per-GPU telemetry are retained; logical prompt-token counts must not be described as recomputed tokens. Review cache occupancy, prefix hits, CPU reloads, GPU activity, queue slopes and preemptions together. An absence of reloads is a legitimate negative/control result.

### Recommended execution sequence

1. One-GPU feasibility using a supported model, normal GPU memory settings, 300-second windows, document populations 32/64/128/256, mean question intervals 60/30 seconds and eight questions per document. These are discovery settings, not a promised winning workload.
2. Check actual token lengths, scored-answer correctness, CPU reloads and client headroom. Adjust document size or population using realistic document-use assumptions. Preserve failed and non-winning measurements.
3. Tune the GPU-only baseline and give both arms comparable tuning effort; the selected source supplies model/memory settings, but its arbitrary extra engine flags are not replayed by this Dynamo adapter.
4. Run all eight GPUs directly. Keep per-replica CPU allocations within each socket's available memory. Compare identical populations and cadence; increase client concurrency if it limits the trace.
5. Extend passing upper boundaries, refine between passing/failing populations, and independently repeat the selected comparison and a no-benefit control. Longer windows should include multiple document replacements.
6. Validate representative document tasks before publishing the intended headline. The current exporter deliberately marks synthetic discovery results exploratory; it does not automatically approve marketing claims.

API uses `session_mode: "capacity"`, `session_gpu_gb: null`, `session_tier: "cpu"`, `session_population_grid` and `session_rate_grid` (**questions/second per document**, not total server rate). Set `warmup_s: 0`; `duration_s` must allow at least four expected questions per document. `session_turns`, `session_answer_records`, `session_max_pending`, `session_cpu_gb`, generic concurrency, timeouts and latency targets are editable. The default UI is a single-GPU discovery run. Existing `session_mode: "resume"` preserves the earlier latency pilot and optional disk-tier diagnostic.

No GPU campaign is started by installing these changes. Earlier stopped queues and automation remain stopped.


### Knowledge-worker density ladder (UI preset)

In **P02 → Document-analysis capacity → Test shape**, choose **Knowledge-worker density ladder**. It is the configuration behind the Intel Xeon KV-offload white paper: all eight GPUs, a fixed population of users, each primed once with a 16-record reference document padded with neutral working notes to a stated context size (default 96,000 tokens), no document replacement, open-loop Poisson arrivals over the population, and the total rate doubled from the initial grid until both arms show two failing windows, then one knee refinement. Sets `session_prime_documents`, `session_document_padding`, `session_context_tokens`, `session_min_returns`, `session_refinement_rounds`, `session_client_lag_s`, `session_engine_max_seqs`, the native CPU backend and a 48 GB host tier per replica. The lowest arrival rate must give every user at least the minimum returns inside the window (rate ≥ users × returns ÷ duration); the spec rejects grids that do not. Measured 2 October 2026 at 96 users (campaign 2ce1867b), 80 users and, as separate plans, 56 users, 48K-token contexts at 192 users and 288 users with a 128 GB tier.

### P02 saturation refinement (current default)

The capacity mode now accepts `session_arrival_rates` as total server requests/second for each fixed document working set. The legacy per-document cadence grid remains available when this list is empty. The full initial rate grid is followed by bounded doubling up to `session_max_arrival_rate`, then paired midpoint refinement (`session_refinement_rounds`). Both arms must show two successive overloaded points; the native queue-time curve must support the knee. Reaching the rate ceiling, client saturation, missing metrics or invalid answers leaves the saturation conclusion unresolved.

The headline concurrency is the exact time-weighted outstanding-request count in the final measurement half, integrated from request lifetimes. Dispatched concurrency and client waiting are separate. A larger backlog alone does not qualify. Both completed throughput and the same response-time SLO must pass. SLO scoring uses the final-half **arrival cohort**, including failed and timed-out requests after drain, rather than selecting only responses that happened to finish inside the measurement window.

`session_slo_grid_s` records multiple response-time views. `request_timeout_s` is the measurement deadline; thresholds beyond it are rejected because censored responses cannot be recovered. Native vLLM queue histograms and running/waiting gauges are recorded alongside cache telemetry. Native p95 queue values are histogram bucket upper bounds. TTFT is never substituted for queue time.

The same evidence also supports a separate reusable-document-population curve at common per-document cadence. The selected population is a workload count, not a verified count of contexts physically resident in cache. The current connector exposes block counters and transfers, not unique document residency; `resident_document_count` therefore remains null. No conversion from cache bytes to resident users is made.

Recalculate response SLOs after the run without changing original evidence:

```sh
python -m simulator.infrastructure.session_reanalyze /path/to/campaign --slo-seconds 5,10,20,60,120
```

This writes a new analysis JSON with hashes of source evidence under the campaign's `analyses` directory. Original reports, manifest, results and request records stay unchanged.

Prepared XE7740 discovery profile: prior working P02 Llama-3.3-70B-Instruct-NVFP4 source; all eight GPUs; 128 and 512 distinct document-session slots; 192-record documents, two-record answers; eight questions per document; 128 GB CPU cache per GPU; 300-second arrival windows plus bounded drain; 600-second measurement deadline; 8192 client streams and 256 engine sequences per GPU; initial total rates 0.5/1/2/4/8/16 requests/s extending up to 128; one exploratory paired block, followed by two refinement rounds. Independent confirmation remains a separate follow-up before publishing a claim. This profile is bounded to 24 hours.

Deferred launch uses a separate immutable source release, so the current service and unrelated tests need no restart. Readiness requires the existing named GPU workload containers to stop, zero GPU compute processes, low GPU memory/utilization, no active capsim run, an available hardware lease, and two idle observations at least 120 seconds apart. The worker checks GPU idleness again before every deployment and holds the hardware lease. A `hold` file beside the deferred plan prevents launch. The controller never removes other workloads and never restarts a failed or terminal campaign automatically.
