# Deploying capsim on a benchmark host

The landing flow for both supported host classes — **Xeon CPU-only**
and **Xeon + NVIDIA GPU** — plus the remote-endpoint and laptop/mock
cases. Prerequisites on the target: Python 3.10+, Docker, git.
Everything else (uv, the capsim tool, models, engine images) is
installed by the flow itself, without sudo.

## 1. Install

```bash
git clone <repo> && cd system-sizing
./install.sh
```

`install.sh` installs [uv](https://docs.astral.sh/uv/) per-user if
missing, then installs capsim as an isolated uv tool from the
checkout. Idempotent — re-run to upgrade. Uninstall with
`uv tool uninstall capsim`. Run the later commands from the repo root
so `config/` profiles resolve.

## 2. Validate the host

```bash
capsim doctor
```

Prints a pass/warn/fail table and writes `doctor.json` (exit code 1 on
any fail — fit for scripting):

| Check | Fail means | Warn means |
|---|---|---|
| `cpu` / `numa` | — (skip on non-Linux) | — |
| `docker` | daemon missing/unreachable — engines can't launch | — |
| `gpu` + `gpu_container_toolkit` | GPU present but Docker has no nvidia runtime → install nvidia-container-toolkit, restart daemon | nvidia-smi flaky |
| `disk` | < 30 GB free — models alone need 30-60 GB | < 150 GB free |
| `perf_pmu` / `rapl_power` | — | telemetry collectors will be blocked; the run still completes with NULLs. Fixes are printed. |
| `hf_reachability` / `hf_token` | — | downloads need network or pre-staged weights |

Doctor ends by recommending candidate **profiles** for the detected
hardware (`capsim list-profiles` shows all of them).

## 3. Prove the pipeline (before any big download)

```bash
capsim smoke --profile <recommended>
```

Launches the profile's real engine with a ~1.5 GB stand-in model
(Qwen3-0.6B), runs 2 virtual users through a short measured window in
an isolated `runs/smoke/` dir, exports, and validates the export
against the schema contract. Gates: engine + run, streaming samples,
export contract; telemetry collector statuses are reported
informationally. ~10 minutes on a fresh box, a couple of minutes
re-run. If smoke passes, the whole pipeline works on this host.

## 4. Full preparation and first benchmark

```bash
capsim ready --profile <recommended>       # engine image + full model
capsim serve                               # web UI on 127.0.0.1:8321
# or headless:
make run-sweep CONFIG=<profile path>       # nohup'd, SSH-safe
```

For the UI over SSH, tunnel the port: `ssh -L 8321:localhost:8321 <host>`
— the service deliberately binds localhost and has no auth.

## Host-class notes

**Xeon CPU-only** — the SGLang path builds a Docker image from source
on first `ready` (~15-20 min); the vLLM-CPU path pulls the upstream
image. Telemetry wants `kernel.perf_event_paranoid=-1` and readable
RAPL for the full evidence set (doctor prints the exact commands).

**Xeon + NVIDIA GPU** — needs the NVIDIA driver and
nvidia-container-toolkit (doctor verifies the runtime registration,
not just nvidia-smi). Profile `xeon-gpu-qwen3-30b` targets one
80 GB-class device at TP=1; its comments describe the TP=2 variant for
2× 40-48 GB hosts. The GPU collector (NVML, nvidia-smi fallback) needs
no extra permissions.

**Remote endpoint** — nothing to install on the system under test;
copy `config/profiles/remote-endpoint.yaml`, set `endpoint_url`, and
run from any client box near it. Host telemetry is skipped by design
and recorded as such in the export.

**Laptop / demo** — `capsim serve`, pick the `mock` profile in the UI.
No Docker, no models, real pipeline.

## Upgrading

```bash
git pull && ./install.sh
```

Run DBs migrate forward automatically on next open (schema version is
stamped in each `run.db`; newer-than-code DBs are refused rather than
corrupted). Export JSON carries `schema_version` — downstream
consumers should pin against it.
