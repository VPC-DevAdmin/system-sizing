# Telemetry reference — every metric, its source, and its limits

Everything the benchmark records per second, where it comes from, and
— just as important — what this stack **cannot** honestly measure and
why. Collectors self-report status (`ok / no_data / not_available /
disabled / skipped_remote_target`) into `cohort_run.collectors_json`,
so every export states which evidence backs its conclusions.

## Simulator truth (closed-loop session state)

The virtual-user pool is ground truth for session behavior — no
inference needed. Per second, on `simulation_snapshots`:

| Metric | Meaning |
|---|---|
| `pool_size` | target concurrent sessions |
| `in_flight` | requests at the engine right now |
| `prefill_in_flight` | submitted, no first token yet (live, from the streaming consumer's first-chunk hook) |
| `decode_in_flight` | streaming tokens |
| `sessions_warm` | mid-conversation think: their next turn replays the conversation prefix, so their KV is the prefix cache's **hot set** |
| `sessions_cold` | fresh / pre-first-turn sessions — nothing cached |
| `requests_completed`, `errors`, `step_samples/target` | progress |

**"Resident sessions by HBM vs DDR"** — the honest version: vLLM does
not expose per-request KV residency, and in our current configs KV
lives only in HBM (no CPU-offload connector). `sessions_warm` × mean
history tokens ≈ the hot KV working set; the engine-side
`kv_cache_used_pct` says how much of the HBM pool it occupies, and
`prefix_cache_hit_rate` says whether warm sessions actually found
their prefix still resident (a falling hit rate under load = the hot
set no longer fits — eviction pressure). If we later enable KV
offload, DDR-resident bytes become a real engine metric to scrape.

## Engine (vLLM `/metrics`, scraped 1 Hz)

| Metric | Meaning |
|---|---|
| `kv_cache_used_pct` | engine's own KV pool gauge |
| `num_running` / `queue_depth` | engine-side running / waiting requests |
| `prefill_tok_s` | Δ`prompt_tokens_total` — prompt ingest rate (compute-bound phase) |
| `decode_tok_s` | Δ`generation_tokens_total` — generation rate (memory-bandwidth-bound phase) |
| `preemptions` | scheduler evictions under KV pressure — nonzero rate precedes latency cliffs |
| `prefix_cache_hits/queries/hit_rate` | prefix cache effectiveness |

**Expert activations (MoE)** — *not obtainable today*: vLLM does not
export per-expert routing counts on `/metrics`. What we DO capture:
whether expert parallelism is on (config), and its aggregate effects
(per-GPU balance below, throughput, latency). Getting true expert
activation histograms would need a patched engine or EPLB debug hooks
— worth revisiting when vLLM lands expert-load metrics upstream.

## GPU (NVML; nvidia-smi fallback) — per device + aggregate

| Metric | Meaning |
|---|---|
| `sm_util_pct` | SM busy % |
| `mem_util_pct` | **DRAM-controller busy %** — the bandwidth-pressure signal; pegged mem with idle SM is the classic decode signature |
| `vram_used_gb` / total | HBM occupancy (weights + KV pool; vLLM preallocates to gmu) |
| `power_w`, `temperature_c`, `sm_clock_mhz` | power/thermal state |
| `throttled` | active slowdown reasons (power cap, thermal, HW brake) |
| `pcie_tx/rx_mb_s` | PCIe traffic (NVML only) — the TP-over-PCIe tax made visible; near-zero for pure-DP shapes |

Per-device rows land in `gpu_devices_json` every second — imbalance
across a TP group or between DP replicas is visible directly
("GPU distributions"). Aggregates (util/clock mean, VRAM/power sum)
stay in columns for trend queries.

## Host CPU / memory / storage / network (`host_json`, 1 Hz)

| Metric | Meaning |
|---|---|
| `cores_util_pct[]` | **per-core** utilization — a single pegged tokenizer thread is invisible in any average |
| `cpu_breakdown_pct` | what cycles went to: user / system / iowait / irq / steal |
| `cpu_util_bound_avg` | engine-cpuset utilization (existing column) |
| `load1`, `ctx_switches_s`, `interrupts_s` | scheduler pressure |
| `mem` | available / page-cache / dirty / swap (plus existing `memory_used_gb`, `engine_rss_gb`) |
| `disk.<dev>` | per-NVMe read/write MB/s, IOPS, device busy% — **storage pressure** |
| `net` | NIC rx/tx MB/s (loopback excluded) |
| `system_power_w` | chassis power via iDRAC IPMI (`ipmitool dcmi power reading`) when granted — see below |

**CPU package power (RAPL)** — *unavailable on this fleet*: kernel
5.15 lacks Granite Rapids TPMI/RAPL support (needs ~6.5; HWE kernel
is the fix — DKMS must rebuild the NVIDIA module for the new kernel
BEFORE rebooting). Until then, chassis-level IPMI power + per-GPU NVML
power bracket the picture. To grant IPMI without root:
`sudo apt install ipmitool` and add the service user to the group
owning `/dev/ipmi0` (or a narrow sudoers entry).

**Memory bandwidth (IMC counters)** — the existing per-IMC PMU
collector; on this kernel/CPU combination it may report
`not_supported` (uncore event support is also newer-kernel). GPU-side
bandwidth pressure is covered by `mem_util_pct` regardless.

## What each phase's bottleneck looks like

- **Prefill-bound**: `prefill_tok_s` spikes, GPU `sm_util_pct` high,
  `mem_util_pct` moderate, TTFT p95 rising, `queue_depth` growing.
- **Decode-bound**: `mem_util_pct` pegged with SM well below 100,
  TPOT rising as batches widen.
- **KV-bound**: `kv_cache_used_pct` ≥ ~95, `preemptions` climbing,
  prefix hit rate falling — capacity knee imminent.
- **Host-bound** (tokenize/schedule/network): one pegged core in
  `cores_util_pct` or high `system`/`irq` share while GPUs idle.
- **Thermal/power-bound**: `throttled` true, `sm_clock_mhz` sagging,
  chassis watts flat at its cap.
