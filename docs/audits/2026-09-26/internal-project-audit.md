# Internal audit: benchmark evidence and publication controls

**26 September 2026 | system-sizing / capsim | For Chris**

## Decision

Keep the project's separation of saturation throughput, SLA capacity, optimization and telemetry. Before the next customer report, fix token accounting and missing-data acceptance, preserve the full executed configuration, and require every headline to reconcile with its source measurements.

The biggest risk is publishing a precise-looking result whose meaning changes between collection, export and presentation. Four small, read-only reproductions demonstrate that risk in the current code. This audit proposes fixes; it does not claim they have been implemented.

Review base: commit `7ab7b48f9af4ce72d3372ad7bf5d08b6362423c4` plus the working tree present on 26 September. Existing uncommitted application/UI changes were left untouched. Findings below were checked against current code rather than copied from the older improvement plan, which records many already-completed fixes.

Related deliverable: [review of the five Dell reports](docs/audits/2026-09-26/dell-report-review.md).

## Verified strengths to retain

- **Separate capacity definitions:** independently arriving sessions, stable-rate and SLA-rate boundaries, and a distinct fixed-concurrency saturation mode. [Algorithm](docs/algorithm.md:5)
- **Load-generator checks:** arrival tardiness, worker-death handling and settling logic help distinguish a client limit from a server limit. [Open-loop implementation](simulator/open_loop.py)
- **Explicit tuning space:** engines, model variants, TP, replicas, placement, memory and batching are actual search dimensions. [Search](simulator/search.py:44)
- **Confirmation measurement:** roofline winners are remeasured separately from search observations. [Confirmation](simulator/roofline.py:1662)
- **Useful telemetry:** per-device measurements and collector availability preserve important qualifications. [Metric definitions](docs/metrics.md)
- **Calibration/evaluation separation in placement work:** domain prompt sets distinguish learning a placement from evaluating it. [Prompt sets](simulator/prompt_sets.py:1)
- **Power scope is partly explained already:** the classic results view warns when only components are measured. [Power rendering](simulator/ui/results.js:715)

These are practical advantages over many of the reviewed publications. They do not eliminate the need for the release controls below.

## Priority register

Suggested owners are roles, not assignments to named people. “Confirmed” means demonstrated in code or a synthetic probe; “control gap” identifies a publication requirement that is not established by the reviewed path.

| ID | Severity | Status | Required change | Suggested owner |
|---|---|---|---|---|
| I1 | Critical | Confirmed | Count actual tokens and reconcile server/client counters | Measurement |
| I2 | Critical | Confirmed | Missing measurements must never become passing results | Measurement/export |
| I3 | High | Confirmed | Preserve complete launch settings in the standard export | Export |
| I4 | High | Confirmed semantic risk | Separate generated, reasoning and answer tokens; gate answer latency | Measurement/reporting |
| I5 | High | Control gap | Attach immutable run identity and hardware to evidence at measurement time | Run orchestration |
| I6 | High | Control gap | Make tuning and publication acceptance separate decisions | Optimization |
| I7 | High | Control gap | Link task-quality evidence to the exact deployment being recommended | Evaluation |
| I8 | High | Confirmed workload limitation | Validate synthetic traffic against representative tasks and cache behavior | Workloads |
| I9 | High | Control gap | Quantify repeatability and protect minority personas | Statistics |
| I10 | High | Confirmed scope risk | Give power/energy metrics explicit boundaries and coverage | Telemetry/export |
| I11 | High when making cost claims | Control gap | Build costs from auditable token and service denominators | Reporting |
| I12 | Medium | Control gap | Generate figures, captions and claims from one approved dataset | Reporting |
| I13 | Medium | Control gap | Separate bottleneck evidence from causal proof | Performance engineering |
| I14 | Medium | Control gap | Bound capacity claims by tested traffic, duration and recovery behavior | Methodology |

## I1. Critical — Stream chunks are counted as tokens

**Evidence.** [streaming.py:244](simulator/streaming.py:244) increments `reasoning_tokens` or `output_tokens` once per nonempty delta. It does not use the final usage count. Those counts enter TPOT and history accounting in [virtual_user.py:519](simulator/virtual_user.py:519), and token-rate exports in [export.py:990](simulator/export.py:990).

**Reproduction:** a synthetic stream with one content chunk and server-reported `completion_tokens=5` produces `output_tokens=1`. This proves a transport-chunk assumption; it does not prove every deployed engine batches its tokens or every historical rate is wrong. Engine-side counter rates require separate tracing.

**Impact:** chunk batching can lower reported output token counts and inflate TPOT. Different transport behavior can alter cross-engine rankings without any change in model performance. History size and downstream cost accounting can also drift.

**Fix:** request and retain authoritative usage when supported. For answer/reasoning splits, use validated backend-specific counters or the pinned tokenizer over complete decoded text, with explicit accounting scope. Keep SSE chunk arrival timestamps as chunk timestamps unless actual token boundaries are available. Record count source and coverage; do not silently substitute an estimate.

**Acceptance:** the same token sequence delivered in one, several or one-token-per-event chunks yields identical total counts and compatible latency definitions. Cover multibyte text, reasoning, empty deltas, missing usage, partial failures, EOS and speculative decoding. Reconcile request totals with server generation counters over aligned windows; explain discrepancies. Identify affected published runs before reuse.

**External lesson:** R2/R3 and T2 show how an ambiguous numerator contaminates efficiency and economics.

## I2. Critical — Unknown evidence can become a pass

**Evidence and reproductions:**

1. [search.py:655](simulator/search.py:655): `rung_sla_ok()` returns `True` when both latency fields are absent.
2. [export_sizing.py:177](simulator/export_sizing.py:177): `curve_point()` with a throughput value but no outcomes yields `status="pass"`, zero samples, zero violation rate and null success rate. `rung_served()` treats zero completed outcomes as served; `held` defaults to true.
3. [roofline.py:1253](simulator/roofline.py:1253) defaults missing steady-state evidence to true and treats missing success rate favorably in ranking. If every candidate is below the success threshold, ranking still selects a best observation. That is not necessarily a valid publishable winner.

These are verified function-level acceptance paths, not a claim that every normal run supplies incomplete data.

**Fix:** make `unknown`, `invalid`, `measured_pass` and `measured_fail` distinct. Keep incomplete historical data inspectable, but require a publication gate with a minimum sample count, duration, outcome completeness, required percentiles, measured steady state and a declared error budget. Search score and report approval must be separate fields.

**Acceptance:** missing/null/nonfinite latencies, zero samples, missing success counts and absent steady-state evidence cannot produce an SLA pass or a publishable result. A search with no qualifying candidates returns “no qualifying configuration,” while retaining diagnostic rankings. Test all-failing as well as mixed candidate sets.

**External lesson:** B4, R4 and T3 require an explicit service acceptance rule rather than a favorable throughput number.

## I3. High — Standard export loses material settings

**Evidence.** The whitelist in [export.py:1450](simulator/export.py:1450) omits fields such as `replica_devices`, `gpu_memory_utilization`, `max_num_seqs`, `kv_cache_dtype` and `reasoning`. The reproduction supplies those settings and confirms that only engine type, model ID and TP survive. The source database's full `config_json` is valuable; its presence does not make a standalone customer export complete.

The standard exporter also uses the first encountered configuration as run-wide metadata. The documented same-config assumption should be enforced, not relied on when resuming or appending measurements. [export.py:1402](simulator/export.py:1402)

**Fix:** export a complete, redacted resolved configuration per configuration hash, with cohort/cell references. Include actual command line, effective defaults, GPU grouping, CPU binding, memory policy, weight and KV precision, batching, prompt/decode options and relevant engine-specific levers. Reject mixed-config merges unless explicitly modeled.

**Acceptance:** an export can reconstruct every executed launch without local defaults. Changing any material setting changes the identity. A heterogeneous run retains separate identities. Secrets remain redacted. Diffing two configurations explains all departures from a matched comparison.

**External lesson:** B1/B2, P6 and R1.

## I4. High — Answer service and generated work need distinct semantics

**Evidence.** The standard path gates TTFT on the first reasoning or content delta; first-content latency is diagnostic. [streaming.py:91](simulator/streaming.py:91), [virtual_user.py:519](simulator/virtual_user.py:519). The sizing exporter explicitly aliases generated tokens to `visible_output_tok_per_s`; reasoning-only completions can count as served. [export_sizing.py:1](simulator/export_sizing.py:1), [curve output](simulator/export_sizing.py:177)

**Impact:** an application may hide reasoning, so the first reasoning delta need not mean the user sees an answer. A completed generation request can still fail to produce usable content. A global “visible” field is unsafe across products.

**Fix:** retain separate prompt, cached-prompt, generated, reasoning and answer-token fields, with observed/estimated/unknown status. Declare whether a metric measures engine work or useful answers. Allow an answer SLA covering first content, end-to-end completion and a valid response. Never substitute first-reasoning time for absent first-content time in an answer-quality claim.

**Acceptance:** a fast-reasoning/slow-answer fixture fails a strict answer SLA; a reasoning-only response remains valid compute work but fails an answer requirement. Consumers cannot accidentally use generation throughput as answer throughput or total billable tokens.

## I5. High — Evidence identity must describe the measured machine

**Evidence.** The sizing export calls `detect_system()` when exporting. [export_sizing.py:559](simulator/export_sizing.py:559). Its CLI documentation tells users to run on the measurement host, which reduces risk, but rerunning after hardware changes or on another host can attach the wrong system metadata. The reviewed standard export lacks a complete immutable dependency manifest.

**Fix:** capture the hardware and software snapshot at run start. Retain server/CPU/GPU identifiers, topology, populated memory and actual speed, BIOS/firmware, OS/kernel/driver, power caps, package versions, container digest, model/tokenizer revision, harness commit plus dirty-tree patch hash, dataset/seed and timestamps. Treat exporter-machine metadata separately.

**Acceptance:** exporting the same evidence from another machine preserves the original system. An independent reader can detect any change of weights, image, harness, firmware or topology. Include startup logs establishing the actual precision/kernel/profile, plus a model-fit check and observed memory allocation.

**External lesson:** R1, B2 and T8.

## I6. High — Tune fairly, then confirm independently

**Evidence.** Search supports a soft latency penalty, and roofline has a distinct confirmation pass. Both are useful. Neither is a complete publication protocol. [Objective](simulator/search.py:95), [confirmation](simulator/roofline.py:1662)

**Fix:** freeze two tracks before testing:

- **Matched comparison:** same task content, checkpoint/quality requirement, precision where feasible, software support policy and acceptance criteria. Explicitly enumerate unavoidable differences.
- **Best deployment:** equal tuning effort and stopping rules; platform-specific settings allowed, but the same service and quality target applies.

Use calibration workloads to tune and separate workloads/seeds to confirm. Give baselines the same tuning opportunity. Retain failed candidates and anomalous repeats; record exclusion reasons. Report “best observed within this search” unless exhaustive optimality is established. A soft penalty is suitable for search direction, but not for accepting a final SLA violation.

**Acceptance:** publish the candidate space, resource/time budget, objective, exclusions and all confirmation repeats. Confirmation failure produces no winner until a qualifying configuration is demonstrated. New settings must reach the actual engine, verified from effective launch logs.

**External lesson:** B1, P6 and T5.

## I7. High — Join performance to task quality for the exact deployed variant

**Control gap.** Capacity scoring uses latency/failure evidence. It is not a semantic evaluator. The source reviewed does not establish a mandatory join between each published capacity result and a matching quality result.

**Fix:** link the completed task-quality/third-party evidence by checkpoint, quantization, tokenizer, reasoning setting, prompt template and decoding policy. Define domain-specific acceptance thresholds and audit the grader. Evaluate quantized variants against the relevant reference; do not substitute an unrelated public leaderboard number.

**Acceptance:** each deployment recommendation carries a quality result or explicitly limits itself to load performance. Faster but less accurate configurations cannot win a quality-constrained comparison. Agentic claims include successful task completion and valid tool use; RAG claims include retrieval and grounding when the full pipeline is in scope.

**External lesson:** S5/S7 and R5. “Premium quality” in latency-oriented UI prose should be renamed “premium latency” or otherwise clearly defined.

## I8. High — Synthetic traffic needs a representativeness and cache audit

**Evidence.** Persona requests use generated filler text with temperature 0.7. [virtual_user.py:367](simulator/virtual_user.py:367), [tokenizer_corpus.py](simulator/tokenizer_corpus.py). Corpus fallback can estimate token counts when a tokenizer is unavailable. Multi-domain placement prompts are a separate, stronger content source; they do not automatically validate every persona run.

**Fix:** keep synthetic traffic for controlled stress tests, and add sanitized representative task replays. Document shared prefixes, history growth, cache state, language, input/output length distributions, EOS policy and reasoning budget. Measure realized tokenizer counts including chat-template overhead. Mark estimates explicitly. Pair common-document tests with native-token-length tests when comparing models.

**Acceptance:** every published workload has a content/version manifest and measured length histogram. Compare cold, warm and realistic shared-prefix conditions. Quantify whether ranking changes with representative content, especially MoE routing. A token-length proxy is never labeled an accuracy or application benchmark.

**External lesson:** B5, S3, S5 and R6.

## I9. High — Repeatability and per-persona guarantees need stronger statistics

**Evidence.** The open-loop aggregate applies a Wilson interval to pooled turn violations. [open_loop.py:535](simulator/open_loop.py:535). Turns within a session and windows from the same run can be correlated; a cohort-wide pass can mask failure in a small persona subgroup. Queue autocorrelation handling already exists and should be retained.

**Fix:** report per-persona acceptance in addition to the aggregate. Confirm boundaries across independent runs/seeds and counterbalanced platform order. Use a session/run-aware uncertainty method where dependence matters. Predeclare a precision target and minimum observations; three independent repeats can be an initial floor, not a universal claim of statistical sufficiency.

**Acceptance:** publish repeats, sample counts, uncertainty, the last passing/first failing rate and censoring rules. A rare persona with insufficient evidence is unknown rather than presumed passing. A high-volume easy persona cannot hide unacceptable service to another group. Small wins whose uncertainty overlaps should be described accordingly.

**External lesson:** S1/S2, B4 and T3.

## I10. High — Power scope must survive export and missing samples

**Evidence.** Classic results prefer chassis power and otherwise sum available components, with an explanatory caveat. However, `hasSys` is computed across all plotted points, so mixed coverage can label a component fallback as chassis in the summary. [results.js:715](simulator/ui/results.js:715). The sizing export uses GPU power for the generic `tokens_per_watt` field. [export_sizing.py:217](simulator/export_sizing.py:217)

**Fix:** attach `power_scope`, sensor source, sample coverage and measurement interval to each result. Label GPU-only efficiency `generated_tok_per_j_gpu`; use a distinct chassis metric. Use time-integrated energy over the same window as the token numerator. Document iDRAC sensor cadence and validate it against an appropriate external meter for report-grade energy claims.

**Acceptance:** four devices at 150 W sum to 600 W. CPU/GPU components are not added to a chassis reading. Sensor loss produces a qualified or absent efficiency result, not a silent scope switch. Watt, joule, kWh and per-million-token conversions reconcile. Cost claims require the energy boundary they actually price.

**External lesson:** R2/R3 and T6.

## I11. High for economics — Cost must use the same useful work

**Control gap.** Token and saturation exports are not, by themselves, a reproducible ownership-cost analysis.

**Fix:** publish an executable cost model with purchase date/price, service life, idle/load power, utilization schedule, availability headroom, facility and operating assumptions, and explicit exclusions. Keep cloud input/output/cache/reasoning tariffs separate and date the region/model/tier. Compare the same successful tasks and SLA, then provide token economics as a secondary view.

**Acceptance:** independently reproduce every total from counts and prices. Test low, medium and high utilization and show break-even conditions. Distinguish fraction of time active from fraction of peak throughput. Higher utilization should reduce amortized fixed cost per token when other assumptions are held constant. The cost workload and performance workload must share evidence IDs.

**External lesson:** T1/T2/T7 and R7.

## I12. Medium — Make report QA a data check and a visual check

**Fix:** one approved results table generates the report's numbers, charts, labels, accessible descriptions and claims. Add automatic checks for comparator direction, percent versus percentage points, extrema ordering, unit conversions, arithmetic, model naming, memory feasibility and dates. Ship executable scripts as files rather than relying on copy/paste from PDF typography.

**Acceptance:** tables and figures reconcile exactly before rounding; every number has a source-cell ID. Comparable bar charts share a baseline or clearly state otherwise. Inspect exported pages at normal reading size; check clipping, contrast, label length, captions and accessibility text. A technical reviewer and an editorial reviewer approve separately.

**External lesson:** P5/P9, S6, B7 and T1. A reversed chart label has High severity even though its fix is typographical.

## I13. Medium — Attribute bottlenecks conservatively

**Control gap.** Telemetry supports explanations, but utilization alone does not prove the cause of a limit.

**Fix:** separate observed facts from likely causes. Confirm consequential recommendations with one-variable interventions: topology, replicas, batch width, queue limit, power cap, CPU binding, task mix or cache policy. Use correct memory terms for the actual GPU, and name missing collectors.

**Acceptance:** “GPU busy” is not automatically “GPU compute limited.” A recommendation to buy more GPUs includes evidence that CPU/database/tool stages will not dominate. Record how much the controlled intervention changes service capacity and uncertainty.

**External lesson:** T5/T8, B3 and P4.

## I14. Medium — Bound the capacity claim and include recovery

**Control gap.** A finite steady-state window and a Poisson workload are useful models, not proof of stability under every production arrival pattern.

**Fix:** publish the tested duration, rate bracket, workload mix and operational headroom. Add burst, overload/recovery, long-session and mixed-tenant checks for deployments where those matter. Describe the roofline feature as an empirical throughput search in customer prose unless a formal compute/bandwidth roofline model is actually presented.

**Acceptance:** distinguish “highest tested stable rate” from a universal physical maximum. Report backlog recovery, rejection policy and latency during overload. Do not substitute concurrent session estimates for in-flight requests or licensed users. Retain a clear legend for all three.

**External lesson:** B4, R4 and T4.

## Deeper review: concrete controls for engine, topology and workflow claims

The expanded paper review adds the following acceptance work to the existing priorities. These are publication controls and proposed verification work, not implemented fixes.

### 1. An executed run and a confirmed run still need a publication decision — I2/I6

Inspection of the retained [XE7740 index](exports/xe7740_sizing/index.json) found **18 of 439 records** labeled `status="ok"` with recorded success below 95%; six are labeled `phase="confirmation"`. For example, run 327 is a confirmation observation with recorded success 0.357. These are valuable diagnostic observations, but their status labels cannot stand in for acceptance. This does not establish that those observations were published or that any particular final winner was selected from them. The 95% filter is a diagnostic threshold, not a retroactive workload SLA.

**Required action:** distinguish execution completion, measurement validity, transport success, task correctness, steady state and SLA acceptance in the export and report. Rank only qualifying candidates for customer recommendations. Keep rejected candidates visible with reasons. Add an all-candidates-fail outcome; a confirmation run that fails acceptance is not a confirmed recommendation.

**Acceptance:** a report built from the retained index cannot promote a low-success observation merely because it is `ok` or in the confirmation phase. Preserve the observed rate and failure breakdown. The inspection and arithmetic are retained in [deeper-review-evidence.json](docs/audits/2026-09-26/deeper-review-evidence.json).

### 2. Engine support must be specific to the executed combination — I3/I5/I6

The [engine notes](simulator/engine_notes.py) record a TensorRT-LLM 1.2.1 FP8 block-scale MoE failure involving a kernel that rejects SM120, and an attempted backend override that did not change that selection. They also record that copying a vLLM memory fraction directly into SGLang exhausted workspace memory. Neither observation supports a universal verdict about an engine family.

**Required action:** qualify exact model revision, quantization recipe, architecture, GPU compute capability, engine image and selected kernel/backend. Run a short correctness-and-memory preflight, then stress the intended context and batch range. Record effective settings after engine defaults and overrides. Translate memory knobs by what they allocate; verify allocated weights, cache, graph buffers and free workspace.

**Acceptance:** an engine starts, serves representative prompts correctly and survives the intended load before it competes in the search. An unsupported combination becomes a compatibility result, not a zero-performance score or a silent fallback to a different experiment.

### 3. Every tuning claim needs a traceable intervention — I6/I13

The notes record large observed effects from chunked prefill and detokenization workers, but the human-readable entries do not themselves provide the complete paired run manifests. They also explicitly correct an earlier CUDA-graph attribution because two knobs had changed together.

**Required action:** attach both run IDs, immutable launch manifests, counters, workload and repetitions to every retained tuning conclusion. Change one factor for causal attribution, or describe a multi-factor change as a package. Include the selected setting's latency, quality and failure effects, not just output rate.

**Acceptance:** a reviewer can reproduce the claimed comparison from the evidence bundle. Do not present the recorded roughly twofold observation as an expected gain for other models, engines or the reviewed Dell papers. Revisit the module-level prose in `engine_notes.py`, which still broadly describes three harmful knobs even though the CUDA-graph entry now correctly says its effect was not isolated.

### 4. Physical fit and topology need runtime proof — I3/I5/I6

**Required action:** reconcile model bytes, quantization overhead, per-device residency, state/cache capacity and runtime workspace. Retain GPU UUIDs, PCIe/NUMA map, replica membership, CPU thread affinity and memory policy. For the same GPU budget, compare feasible independent replicas with larger TP groups. Record offload explicitly, including CPU and PCIe cost. Distinguish whole-server throughput, per-replica throughput and per-device telemetry.

**Acceptance:** no customer configuration exceeds its memory budget without a disclosed mechanism. Requested placement agrees with observed worker placement. A cross-domain TP group is a measured choice with a reason, rather than an unexplained default.

### 5. Protect workload equivalence across engines and models — I1/I4/I7/I8

**Required action:** freeze task content, tokenizer policy, chat template, reasoning budget, EOS/forced-length behavior, cache policy and input/output mix. Publish both matched-document and matched-token tests where those answer different questions. Distinguish cold ingestion, repeated-prefix service and multi-turn cache reuse. Record first generated token, first answer content and complete usable answer separately where reasoning is present.

**Acceptance:** an engine or model cannot win merely by generating fewer useful answers, stopping early, reusing more cached input or hiding reasoning time. The completed third-party quality results join to the exact served variant and settings. Forced-length synthetic saturation results remain explicitly labeled and are not represented as task productivity.

### 6. Add an explicit contract for every “agent capacity” claim — I7/I8/I9/I14

**Required action:** define whether the unit is a model request, conversation, deterministic workflow or adaptive agent. For workflows retain tool-call counts, branching/loop distributions, model calls, retries, stage timing and outcome grading. Time test-fixture creation separately from genuine production data preparation. Use fixed-concurrency and independent-arrival tests for their respective questions; vary task order and production mix.

**Acceptance:** publish correct tasks/minute and latency per workflow, with offered/admitted/completed/failed/abandoned counts and backlog. Include startup, steady-state and drain boundaries. A conversation persona cannot be relabeled an autonomous agent without executing and evaluating the associated tools and control flow.

### 7. Diagnose delivery and non-GPU limits before recommending hardware — I13/I14

**Required action:** correlate generator scheduling lag, server admission, active sequences, cache occupancy, preemption, token delivery, CPU utilization and GPU activity. For application workloads add database/tool/orchestration timers. Confirm suspected limits by changing one resource or setting. Record the load generator's separate CPU budget so an engine comparison is not accidentally a client comparison.

**Acceptance:** claims of spare capacity or GPU limitation include evidence beyond average GPU utilization. Recommendations for more GPUs show that accepted useful work increases, with no hidden loss in latency or task quality. Keep saturation throughput separate from SLA capacity even when a high-throughput run has perfect transport success.

## Release sequence

### Gate 1 — Measurement correctness

Complete I1, I2 and I4. Recompute affected metrics where raw evidence permits it; otherwise rerun. Do not apply a guessed global correction factor. Preserve old exports with a superseded status and a reason.

### Gate 2 — Reproducible deployment and units

Complete I3, I5 and I10. The bundle must identify the executed machine, model, software and settings, and reconcile token and energy scopes. A third party should be able to explain every chart from the bundle alone.

### Gate 3 — Fair experiment and useful service

Apply I6–I9 and I14 to the next report protocol. Freeze the matched/tuned tracks, quality join, independent confirmation, workload validity and statistical acceptance before looking at final winners.

### Gate 4 — Customer publication

Apply I11–I13. Reconcile economics, qualify causal claims and verify the final presentation. Reject unsupported hardware, quality or TCO conclusions even if individual throughput numbers are valid.

## Minimum publication evidence package

1. `manifest.json`: run/configuration IDs, hashes, provenance, hardware/software and exact measurement scopes.
2. Resolved launch configuration, redacted commands and startup logs showing effective settings.
3. Workload definitions, prompt/task dataset versions, seeds and quality results for the selected variant.
4. Per-request or per-task outcomes, timestamps, actual token accounting and raw telemetry with availability flags.
5. All repetitions, tuning candidates, failures, exclusions and the separate confirmation results.
6. One generated claims table: source IDs, calculation, unit, uncertainty and acceptance state for every published statement.
7. Reproduction instructions and executable scripts; cost inputs when economics is claimed.
8. Final technical/editorial sign-off and a dated change log.

## Validation performed for this audit

Four synthetic code probes and independent report arithmetic were executed locally. They require no GPUs, network access or benchmark-data changes. They demonstrate specific function behavior; they are not substitute hardware acceptance tests. No production code was changed and no full test-suite result is claimed.

- [Reproduction script](docs/audits/2026-09-26/verify_audit.py)
- [Captured results](docs/audits/2026-09-26/verification.json)

Run from `/Users/chrisbranch/Projects/system-sizing`:

```sh
.venv/bin/python docs/audits/2026-09-26/verify_audit.py
```

The four observed outputs are: five advertised completion tokens counted as one; missing latency accepted by the search SLA helper; material configuration fields omitted from standard export; and an empty outcome set exported as passing. These are the first corrections to make before improving report appearance.
