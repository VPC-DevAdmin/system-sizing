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
