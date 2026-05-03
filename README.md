# Persona Capacity Simulator

Drives a single, long-running LLM inference engine (vLLM or SGLang) with realistic persona-based workloads to find the per-cohort capacity knee, with telemetry-attributed bottleneck analysis.

## Quick start

```bash
make setup
make run-cohort  ENGINE=vllm  MODEL=Qwen/Qwen2.5-7B-Instruct  COHORT=chat_heavy
make dashboard                                      # in another terminal
make run-sweep   ENGINE=vllm  MODEL=Qwen/Qwen2.5-7B-Instruct
make export                                         # buyer_page_data.json
```

## Make targets

| Target | What it does |
|---|---|
| `make setup` | Install package + deps |
| `make launch-engine` | Launch engine subprocess only (manual testing) |
| `make run-cohort` | Run a single cohort end-to-end |
| `make run-sweep` | Run all cohorts back-to-back against one engine |
| `make dashboard` | Live `rich`-based progress view of the latest run |
| `make export` | Build `buyer_page_data.json` from `runs/*.db` |
| `make clean` / `clean-runs` | Tidy caches / wipe run databases |

Variables: `ENGINE` (vllm|sglang), `MODEL`, `COHORT`, `CONFIG`, `RUN_DIR`.

## Layout

```
simulator/
  cli.py              # typer entry point used by Make targets
  runner.py           # cohort run orchestration
  config.py           # YAML + CLI config
  personas.py         # 6 personas, 5 cohorts
  distributions.py    # LogNormal / Discrete samplers
  tokenizer_corpus.py # filler text targeted at token counts
  engines/
    base.py           # subprocess + Prometheus metric parser
    vllm.py           # Xeon CPU tuning recipe
    sglang.py         # CPU launcher
  virtual_user.py     # one async task per simulated user
  pool_manager.py     # spawn/replace virtual users at target pool size
  adaptive.py         # next-pool-size selection logic
  measurement.py      # ramp -> stabilize -> measure
  telemetry.py        # snapshots, perf-stat PMU, engine metrics
  database.py         # SQLite schema + capture
  dashboard.py        # rich live view
  export.py           # buyer-page JSON
config/default.yaml
Makefile
```

Each cohort run produces one SQLite file in `runs/`.
