# Third-party benchmarks

AIPerf **0.13.0** and GuideLLM **0.7.4** run through Roofline. Each campaign records the selected engine configuration, fixed recipe, source result, harness versions, dataset, exact commands, raw outputs, the exact adapter source, telemetry, recovery attempts, an editable DOCX and a checksummed evidence ZIP. These are named-harness measurements, not certification or model-quality evaluations.

## Use the UI

1. In **Lab → Roofline → Setup → Third-party benchmarks**, select either or both tools. **Prepare selected tools** installs them in separate virtual environments. Python 3.11–3.13 is required; set `CAPSIM_BENCHMARK_PYTHON` if it is not on PATH.
2. Leave internal winner confirmation enabled. A self-search benchmarks one confirmed throughput winner per catalog model family and parameter size, across quantizations and engines. The candidate's best recorded token shape defines the selection objective. Precision alternatives remain available for manual runs.
3. Choose a recipe, durations, repetitions, workload and evidence thresholds. The estimated time excludes tool installation, model launches, drain time and retries.
4. For a manual measurement, select a recorded result in **Roofline Workspace → Third-party benchmark evidence**, or use the selected pairing's **Run Third Party Benchmarks** action. Set the recipe in Setup, then run it. The selected engine and precision are preserved.
5. The evidence panel shows each harness/load/repetition outcome, reasons in the outcome tooltip, methodology, resume controls, DOCX and ZIP downloads. Select **New experiment** to repeat a completed campaign; otherwise an identical request reuses its existing evidence.
6. Automatic finalist campaigns also produce a combined search DOCX and ZIP. The search report includes unsuccessful models as well as completed measurements.

AIPerf and GuideLLM run sequentially. A healthy engine launch is shared under the warm-cache policy; repetitions alternate harness order. The restart policy relaunches before each attempt. Warmup runs are separate and excluded from the reported measurement. A round-robin streaming proxy visits every replica and is included in timing. At high concurrency it uses separate routing processes (up to 32, sized to the load and available CPUs). Per-worker snapshots and warmup routing evidence are retained; a dead worker invalidates the attempt. The supervisor raises its open-file soft limit within the existing host hard limit and records the effective values. An insufficient hard limit is a preflight failure.

## Recipes and interpretation

- **Reproduce:** use the recorded prompt/output lengths and concurrency. The corpus is synthetic and shared between harnesses; this reproduces the shape, not the original prompt text.
- **Matched serving sweep:** explicit concurrency points; optional prompt/output length overrides.
- **SLO request-rate sweep:** Poisson arrivals with explicit rates and p95 TTFT and/or TPOT limits. Insufficient achieved arrivals, short windows, undersampling, missing metrics and possible client saturation are inconclusive. Routing saturation means more than 5% of the 250 ms event-loop samples during active requests on any worker exceed 100 ms of delay; the maximum delay is also retained. GuideLLM mean request-start delay above 100 ms is a separate client check; Poisson SLO runs use targeted-start delay, which also includes scheduler queuing. It assesses aggregate latency/error thresholds; it does not claim per-request goodput, queue stability, or concurrent-user capacity.
- **Sustained confirmation:** a single load point with a user-selected long duration. Duration is explicit; the recipe does not silently lengthen a run.

The default is 300 measured seconds, 30 warmup seconds and three repetitions, with a minimum of 100 successful requests. These are editable starting values, not universal publication standards. Each repetition remains separate; percentiles are never averaged across harnesses. AIPerf's ITL is not substituted for TPOT: a TPOT gate is inconclusive for that adapter. GuideLLM stops at the measurement boundary. Cancellations proven by its native stop reason and per-request timestamps are reported separately from request errors; unproven or early cancellations still count as failures. SLO runs with a boundary-cancelled fraction above the configured error limit are inconclusive. GuideLLM output throughput uses its native total in-window token rate, including partial boundary requests; completed-request throughput is also retained. AIPerf uses its native completed-request throughput and drain duration. These rates and their timing denominators are not interchangeable; native artifacts remain authoritative.

Prompt lengths exclude the serving engine's chat-template overhead. Both tools receive temperature 0 and the same output limit; `ignore_eos` defaults to true. The recorded reasoning effort is forwarded when enabled. AIPerf may reuse the corpus; GuideLLM uses its public data-deserializer extension to cycle the same captured corpus for duration-based runs. The small extension's source is included in the ZIP; upstream scheduling and measurement remain unchanged. Warm-cache reuse can benefit prefix caching and is disclosed.

## Fidelity and recovery

New headline/roofline measurements save the complete configuration and local asset fingerprint. GPU containers use resolved image IDs and cached model snapshots during external measurement. A model/image change from a recorded source is rejected. Original results without complete configuration require one new internal measurement; the integration does not reconstruct missing settings from current UI defaults. Custom in-container model paths currently require an explicit host snapshot mapping and are rejected. Tokenizers must be locally available and load without executing remote code.

The service run lock, per-campaign lease and per-user host hardware lease prevent overlapping capsim jobs. Before managed GPU launch, the worker refuses an occupied GPU; it does not remove another job's containers. This does not coordinate unrelated schedulers or other host users. Stop those workloads before measurement.

A campaign has a time budget, disk budget, minimum-free-space check and bounded attempts. Each harness process has a separate artifact-processing allowance (900 seconds by default), in addition to the requested windows and drain grace; large streaming runs can spend minutes compiling native statistics. This changes the process deadline, not the measured interval. Crashes preserve the failed attempt, shut down the owned engine, then retry the same settings. No OOM repair reduces batch size, precision, load or context silently. Resume skips settled attempts and reruns interrupted windows from the beginning. A browser disconnect does not cancel a run. Owner death cancels the worker; after machine failure, orphaned campaigns become resumable in the UI. Completed evidence is checked before reuse.

Tool installs pin the harness version, capture all resolved dependency versions and reuse that lock for repair. A broken or drifted environment is retained with a `.broken-*` suffix and rebuilt once. Review/delete old retained environments outside active runs when reclaiming disk space.

## CLI and API

The service owns its runs directory while running. Use the UI/API for that directory, or stop the service before using the CLI.

```sh
capsim benchmark prepare aiperf --runs-dir runs
capsim benchmark prepare guidellm --runs-dir runs
capsim benchmark results --runs-dir runs
capsim benchmark run RESULT_ID recipe.json --runs-dir runs
capsim benchmark resume CAMPAIGN_ID --runs-dir runs
capsim benchmark export CAMPAIGN_ID --runs-dir runs
```

Example `recipe.json`:

```json
{"harnesses":["aiperf","guidellm"],"recipe":"reproduce","duration_s":300,"warmup_s":30,"repetitions":3}
```

API prefix: `/api/roofline/benchmarks`. GET lists campaigns; `/results` returns immutable result IDs. POST the prefix with `{"result_id":"...","spec":{...},"fresh":false}`. POST `/prepare` with the recipe, or `/{id}/resume`. GET `/{id}` for manifest/state and `/{id}/export/docx` or `/zip` for artifacts. The normal `/api/runs/stop` cancels the active campaign. Automatic selection is configured through the existing roofline request's optional `benchmarks` object.

Evidence lives under `runs/benchmarks/ID/`. `engine.yaml` records the source configuration; `launch-resolved.yaml` records the pinned launch. Per-attempt commands and native outputs describe exact tool behavior. `SHA256SUMS.json` covers ZIP contents. The editable report is a presentation of that evidence; edits to it do not change the original measurement.

## Deferred work

- **MLPerf Inference:** separate rules/version-specific workflow and submission evidence. No MLPerf-compliance claims are made by these integrations.
- Model quality evaluation, production persona/session replay, external load-generator hosts and formal whole-system power measurement.
- Verified custom model-mount mappings and additional tokenizer/model families as Linux GPU pilots demonstrate support.
