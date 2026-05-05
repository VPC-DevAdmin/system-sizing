# `run.db` schema

One SQLite file per `runs/run_NN/` directory. Every cohort/persona invocation
against that run dir appends rows to the same file; isolation is by
`cohort_run_id`. Schema lives in [simulator/database.py](../simulator/database.py)
and is created idempotently on `Database` open via `CREATE TABLE IF NOT EXISTS`.

Connection settings: `journal_mode=WAL`, `synchronous=NORMAL`,
`isolation_level=None` (autocommit), `check_same_thread=False`. A single
process-wide `threading.Lock` wraps every cursor — multiple async tasks
share one connection.

## Entity diagram

```
cohort_run                          (one row per `run_cohort` invocation)
   │  cohort_run_id (PK)
   │
   ├──< cohort_measurements          (one row per ramp step; window-level
   │     │  measurement_id (PK)       PMU/BW/power/AMX/freq totals are
   │     │                            inlined as nullable columns)
   │     │
   │     ├──< turn_events              (one row per LLM turn captured in window)
   │     └──< measurement_telemetry    (per-second samples within window)
   │
   ├──< simulation_snapshots         (per-second pool/in-flight tick)
   └──< virtual_users                (one row per simulated user lifecycle)
```

## `cohort_run` — one row per `run_cohort()` call

| column                    | type | notes |
|---|---|---|
| `cohort_run_id`           | TEXT PK | `uuid.uuid4().hex` minted in [`runner.run_cohort`](../simulator/runner.py). |
| `started_at`              | TEXT NOT NULL | ISO-8601 UTC at run start. |
| `completed_at`            | TEXT | Set on `finalise_run`; NULL while in flight. |
| `engine_type`             | TEXT NOT NULL | `vllm`, `sglang`, `vllm_dual_socket`. |
| `model_id`                | TEXT NOT NULL | HF model id (e.g. `Qwen/Qwen3-30B-A3B-Instruct-2507`). |
| `cohort_id`               | TEXT NOT NULL | `chat_heavy`, `engineering_heavy`, … or a persona id when `cohort_from_persona` was used. |
| `cohort_definition_json`  | TEXT NOT NULL | JSON of `{id, name, description, category, persona_weights}`. |
| `config_json`             | TEXT NOT NULL | Full `Config` dataclass as JSON — engine + simulation + telemetry + output sections. Captures the YAML knobs that produced this measurement. |
| `final_status`            | TEXT | One of: `ok`, `interrupted`, `time_limit`, `no_samples`, NULL (in progress). Resume only skips `'ok'`. |
| `prefix_cache_engine_hits` | INTEGER | Cumulative engine-reported prefix-cache hits at end-of-run (sum across replicas for `vllm_dual_socket`). |
| `prefix_cache_engine_queries` | INTEGER | Cumulative engine-reported prefix-cache queries — denominator for the engine hit rate. |
| `prefix_cache_engine_hit_rate` | REAL | Convenience: `hits / queries` (or whatever the engine returns directly when it exposes a hit-rate gauge). Surfaced in export under `cohort.prefix_cache.engine_hit_rate`. |

Resume is gated entirely by `final_status='ok'` — see
[`runner.find_completed_runs`](../simulator/runner.py).

## `cohort_measurements` — one row per ramp step

Inserted **up front** (with placeholder zeros and `capacity_status='pending'`)
so telemetry rows can attach by `measurement_id`. Updated to its final values
once the ramp step finishes.

| column                       | type | notes |
|---|---|---|
| `measurement_id`             | INTEGER PK AUTOINCREMENT | Foreign key target for events / telemetry / aggregate. |
| `cohort_run_id`              | TEXT NOT NULL | → `cohort_run.cohort_run_id`. |
| `step_index`                 | INTEGER NOT NULL | 0-based ordinal within the ramp. |
| `target_pool_size`           | INTEGER NOT NULL | Concurrency target the adaptive stepper picked. |
| `measured_avg_pool_size`     | REAL NOT NULL | Currently mirrors `target_pool_size` (closed-loop pool tracks target). |
| `measured_avg_in_flight`     | REAL NOT NULL | Mean of `_in_flight_sampler` ticks during the measurement window. |
| `measurement_started_at`     | TEXT NOT NULL | ISO-8601 UTC. |
| `measurement_duration_s`     | INTEGER NOT NULL | Wall-clock seconds the window ran. |
| `sample_size`                | INTEGER NOT NULL | Number of `turn_events` captured in this window. |
| `ttft_violation_rate`        | REAL NOT NULL | Fraction of turns whose TTFT exceeded the persona's `ttft_floor_seconds`. |
| `tpot_violation_rate`        | REAL NOT NULL | Same for TPOT vs `tpot_floor_ms`. |
| `combined_violation_rate`    | REAL NOT NULL | Fraction of turns violating *either* SLA — drives `capacity_status` and the adaptive bisection. |
| `violation_rate_ci_lower`    | REAL NOT NULL | Wilson-score 95% CI lower bound on the combined rate. |
| `violation_rate_ci_upper`    | REAL NOT NULL | Wilson upper bound. |
| `ttft_p50_ms` / `ttft_p75_ms` / `ttft_p95_ms` | REAL | Linear-interpolated percentiles over `sample_size` turns. |
| `tpot_p50_ms` / `tpot_p75_ms` / `tpot_p95_ms` | REAL | Same. |
| `avg_kv_cache_pct`           | REAL | Mean of `measurement_telemetry.kv_cache_used_pct` for this window. |
| `estimated_prefix_hit_rate`  | REAL | Best-effort delta of engine-reported `prefix_cache_hits` across the window. NULL when the engine doesn't expose the counter. |
| `capacity_status`            | TEXT NOT NULL | `pass` (`combined<5%`) / `marginal` (`5–30%`) / `fail` (`≥30%`) / `pending` (pre-update) / `unstable` / `no_samples`. The 30% boundary aligns with `knee_zone_threshold` so visual status and bisection trigger agree. |
| `pmu_cycles` / `pmu_instructions` / `pmu_ipc` | REAL | `perf stat` totals over the window. |
| `pmu_stalls_mem_any` / `pmu_stalls_l3_miss` / `pmu_stall_mem_ratio` | REAL | Memory-stall fractions for the bottleneck heuristic. |
| `pmu_amx_ops`                | REAL | Raw AMX event counter when the kernel exposes it. |
| `amx_perf_event_name`        | TEXT | Which raw event the collector fell back to (Intel GNR-specific). |
| `pmu_llc_reference` / `pmu_llc_miss` | REAL | LLC pressure indicators. |
| `mem_local_fraction` / `mem_remote_fraction` | REAL | NUMA-local vs remote DRAM access shares from `mem_load_l3_miss_retired.*_dram`. |
| `memory_bw_read_gb_s_avg` / `_peak` | REAL | IMC uncore counters (per-controller summed). |
| `memory_bw_write_gb_s_avg` / `_peak` | REAL | Same. |
| `bandwidth_status`           | TEXT | `ok` / `degraded` / `unsupported` — collector self-report. |
| `power_w_avg` / `power_w_peak` / `power_status` | REAL/TEXT | RAPL package energy delta / window seconds. |
| `effective_freq_ghz_mean` / `_stddev` / `_min` | REAL | Window-level rollups of the per-second `measurement_telemetry.freq_*` samples. |
| `onednn_amx_time_fraction`   | REAL | Parsed from `ONEDNN_VERBOSE` in the engine log post-run; `amx_time / total_matmul_time`. Run-level signal — attached to the **last** measurement so the bottleneck attributor sees it. |
| `onednn_matmul_dispatches_amx` / `_non_amx` | INTEGER | Dispatch counts from the same log parse. |

Index: `idx_measurements_run` on `cohort_run_id`.

Aggregate columns are populated via two write paths, both calling
`Database.update_measurement(measurement_id, row)`:

1. The post-window UPDATE that fills percentiles, violation rates, and
   `capacity_status` (driven by [measurement.run_measurement_step](../simulator/measurement.py)).
2. The per-window aggregate rollup (PMU / BW / power / freq) at end of
   the same step, plus a post-run oneDNN-AMX attach against the last
   measurement (in [runner.run_cohort](../simulator/runner.py)).

Earlier schema kept these aggregate columns in a separate
`measurement_aggregate` table (1:1 with `cohort_measurements`); they
were collapsed in. Legacy DBs are migrated transparently on first open
by `Database._migrate_legacy_aggregate` — `ALTER TABLE ADD COLUMN` for
any missing destination, copy values via subquery `UPDATE`, then
`DROP TABLE measurement_aggregate`.

## `turn_events` — one row per LLM turn captured in a measurement window

The granular sample population that drives the percentile + violation columns
above. `_event_to_row` in [measurement.py](../simulator/measurement.py).

| column                 | type | notes |
|---|---|---|
| `event_id`             | INTEGER PK AUTOINCREMENT | |
| `measurement_id`       | INTEGER NOT NULL | → `cohort_measurements.measurement_id`. |
| `persona_id`           | TEXT NOT NULL | The user-archetype id that issued the turn — needed because cohorts mix personas. |
| `user_id`              | TEXT NOT NULL | Stable for the simulated user's whole lifecycle. |
| `session_id`           | TEXT NOT NULL | Fresh UUID per multi-turn session — distinguishes turn 0 (cold) from turns 1..N (warm cache). |
| `turn_index`           | INTEGER NOT NULL | 0-based within session. |
| `submitted_at_ms`      | INTEGER NOT NULL | Wall-clock ms when the request was sent to the engine. |
| `ttft_ms`              | REAL NOT NULL | First-token latency. |
| `completed_at_ms`      | INTEGER NOT NULL | Wall-clock ms at last token. |
| `input_tokens`         | INTEGER NOT NULL | Prompt tokens (this turn only). |
| `history_tokens`       | INTEGER NOT NULL | Tokens of prior conversation re-sent — sized via the persona's lognormal sampler. Drives prefix-cache analysis (`turn_index>0` rows with non-zero history are the cache-hit candidates). |
| `output_tokens`        | INTEGER NOT NULL | Generated tokens. |
| `tpot_ms`              | REAL NOT NULL | Per-output-token latency = (completed-first_token)/output_tokens. |
| `end_to_end_ms`        | REAL NOT NULL | `completed_at_ms - submitted_at_ms`. |
| `in_flight_at_submit`  | INTEGER NOT NULL | `state.in_flight` snapshot when the turn was issued. |
| `in_flight_avg_during` | REAL | Mean during the turn's lifetime. NULL if not captured. |
| `in_flight_peak_during`| INTEGER | Peak during the turn's lifetime. |
| `sla_ttft_violation`   | INTEGER NOT NULL | 0/1 — `e.ttft_violation()`. |
| `sla_tpot_violation`   | INTEGER NOT NULL | 0/1 — `e.tpot_violation()`. |
| `token_timestamps_json`| TEXT | Tier-3 opt-in: JSON `[[elapsed_ms, cum_tokens], …]`, one entry per streamed chunk. NULL when `enable_token_timestamps=False`. |

Index: `idx_events_measurement` on `measurement_id`.

## `simulation_snapshots` — per-second pool/in-flight tick

Cheap heartbeat written by `SnapshotRecorder` on a fixed cadence (default 1 s)
so the dashboard has something to render between measurement-window updates.

| column                | type | notes |
|---|---|---|
| `snapshot_id`         | INTEGER PK AUTOINCREMENT | |
| `cohort_run_id`       | TEXT NOT NULL | |
| `snapshot_at_ms`      | INTEGER NOT NULL | |
| `phase`               | TEXT NOT NULL | `idle` / `ramp` / `warmup` / `measuring`. |
| `pool_size`           | INTEGER NOT NULL | `pool.target_size` at sample time. |
| `in_flight`           | INTEGER NOT NULL | `state.in_flight`. |
| `requests_completed`  | INTEGER NOT NULL | `state.completed` (cumulative). |
| `errors`              | INTEGER NOT NULL | `state.errors` (cumulative). |
| `step_samples`        | INTEGER | Live count of turn events landed in the in-flight measurement buffer. 0 outside a measurement window. |
| `step_target_samples` | INTEGER | Target the buffer is filling toward (`SimulationConfig.target_samples_per_step`). 0 outside a measurement window — dashboard renders "—". |

Index: `idx_snapshots_run_time` on `(cohort_run_id, snapshot_at_ms)`.

## `measurement_telemetry` — per-second samples within a measurement window

Driven by `MeasurementTelemetry.start(measurement_id)` /  `.stop()`. Captures
the engine `/metrics` scrape plus host-side per-second observables. Aggregated
into `cohort_measurements.avg_kv_cache_pct` and consumed by the buyer-page
export's per-step "telemetry" section.

| column               | type | notes |
|---|---|---|
| `telemetry_id`       | INTEGER PK AUTOINCREMENT | |
| `measurement_id`     | INTEGER NOT NULL | → `cohort_measurements`. |
| `sampled_at_ms`      | INTEGER NOT NULL | |
| `kv_cache_used_pct`  | REAL | From `vllm:cpu_cache_usage_perc` or `sglang:token_usage`, normalised to 0-100. |
| `queue_depth`        | INTEGER | `vllm:num_requests_waiting` / `sglang:num_waiting_reqs`. |
| `prefix_cache_hits`  | INTEGER | Cumulative engine counter; deltas drive `estimated_prefix_hit_rate`. |
| `prefix_cache_misses`| INTEGER | Reserved column — populated when the engine exposes a separate miss counter. |
| `cpu_util_avg`       | REAL | Average over `bound_cpus` (engine pinning set). |
| `memory_used_gb`     | REAL | Host RSS. |
| `engine_rss_gb`      | REAL | Engine process RSS — separated from host because dual-socket spawns multiple engine procs. |
| `freq_mhz_mean`      | REAL | Effective frequency over `bound_cpus` (three-tier read in [frequency.py](../simulator/frequency.py)). |
| `freq_mhz_stddev`    | REAL | Spread across bound cores — non-zero stddev flags an unbalanced workload. |
| `freq_mhz_min`       | REAL | Worst-core frequency — surfaces droop the mean would hide. |

Index: `idx_telemetry_measurement` on `measurement_id`.

## `virtual_users` — one row per simulated user lifecycle

Upserted (PK = `user_id`) so a user's row gets its `terminated_at_ms` and final
session counts when the pool manager replaces it.

| column                | type | notes |
|---|---|---|
| `user_id`             | TEXT PK | UUID minted at spawn. |
| `cohort_run_id`       | TEXT NOT NULL | |
| `persona_id`          | TEXT NOT NULL | Which archetype the user was assigned. |
| `spawned_at_ms`       | INTEGER NOT NULL | |
| `terminated_at_ms`    | INTEGER | NULL while alive. |
| `sessions_target`     | INTEGER NOT NULL | Sampled at spawn from the persona's session-count distribution. |
| `sessions_completed`  | INTEGER NOT NULL | |
| `turns_total`         | INTEGER NOT NULL | Across all sessions. |
| `pool_size_at_spawn`  | INTEGER NOT NULL | The ramp's `target_pool_size` at the moment this user was spawned. |
| `replaced_user_id`    | TEXT | When pool manager replaces a finished user, the new user records who it took over for — useful for traceback when latency spikes correlate with replacements. |

## Capacity-status state machine

The capacity-status column (`pass` / `marginal` / `fail` / `pending` /
`unstable` / `no_samples`) tells the operator + the export layer one thing:
"could the system handle this concurrency?" Bands are deliberately tuned for
sample-noise tolerance at `n=100` — see the comment in
[`measurement.run_measurement_step`](../simulator/measurement.py).

## What's *not* persisted

- LLM completion text — only token counts and timing.
- Per-replica routing maps for `vllm_dual_socket` — surfaced in the run log
  via `pool.routing_summary()` and aggregated `prefix_cache_hit_rate`, but not
  schematised. The buyer-page export reads engine-aggregated metrics instead.
- `perf stat` per-second CSVs — written to `runs/run_NN/perf_m*.csv` for
  post-hoc inspection; only the window-level rollup lands in
  `measurement_aggregate`.
