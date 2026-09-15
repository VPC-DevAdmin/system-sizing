# Deploying capsim on a benchmark host

The landing flow for both supported host classes — **Xeon CPU-only**
and **Xeon + NVIDIA GPU** — plus the remote-endpoint and laptop/mock
cases. Prerequisites on the target: Python 3.10+, Docker, git.
Everything else (uv, the capsim tool, models, engine images) is
installed by the flow itself, without sudo.

## 0. Bare OS → prerequisites (one-time, needs sudo)

Fresh box with nothing but the OS (Ubuntu 22.04/24.04 shown; RHEL
notes below). This is the only sudo-requiring stage.

```bash
# Basics — git, python3 (Ubuntu ships 3.10+; uv manages its own
# Python for capsim anyway), and perf for the PMU/bandwidth collectors.
sudo apt-get update
sudo apt-get install -y git python3 curl linux-tools-common linux-tools-$(uname -r)

# Docker (engine runtime), and let your user run it without sudo.
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER && newgrp docker

# GPU hosts only: NVIDIA driver. Headers for the RUNNING kernel first
# so DKMS builds against it (and avoid apt full-upgrade here — a new
# kernel is what forces a reboot). Use the -open variant: it supports
# everything Turing+ and is REQUIRED for Blackwell (see field notes).
sudo apt-get install -y linux-headers-$(uname -r) nvidia-driver-580-server-open
sudo reboot                                        # then verify: nvidia-smi

# No-reboot alternative (labs where reboots are costly): if
# `lsmod | grep nouveau` is empty — common on headless GPU servers —
# just load the modules by hand and skip the reboot:
#   sudo modprobe nvidia nvidia_uvm && nvidia-smi
# If nouveau IS loaded: blacklist it (modprobe.d + update-initramfs),
# then try `sudo rmmod nouveau` before the modprobe. If rmmod refuses
# ("in use" — it's holding the console framebuffer), reboot is the
# clean path. Everything below driver install never needs a reboot.
```

Driver field notes (learned the hard way on an XE7740 with 8× RTX PRO
6000 Blackwell):

* **Blackwell-generation GPUs require the `-open` driver variant**
  (`nvidia-driver-<ver>-server-open`). The proprietary module loads
  cleanly and then reports "No devices were found"; `dmesg | grep -i
  nvrm` says outright "requires use of the NVIDIA open kernel
  modules". Verify which module is loaded with
  `modinfo nvidia | grep -m1 license` — the open module reports
  `Dual MIT/GPL`, the proprietary one `NVIDIA`.
* **"Driver/library version mismatch" from nvidia-smi** means the
  loaded kernel module and the userspace libs come from different
  driver generations — typical on lab machines with layered installs.
  Check `cat /proc/driver/nvidia/version` vs `dpkg -l | grep nvidia`;
  cleanest fix is to purge every nvidia package and reinstall ONE
  series (the apt resolver usually drags the tangle out in one purge).
* **DKMS won't replace a same-version module.** When switching
  proprietary → open within the same series, the precompiled
  proprietary packages (`linux-modules/objects-nvidia-<ver>-server-*`,
  no `-open`) still own the .ko files and shadow the open build.
  Purge them, then `sudo dkms install nvidia/<version> --force &&
  sudo depmod -a`, then rmmod/modprobe.

```bash

# … and nvidia-container-toolkit, so Docker can hand containers GPUs.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Telemetry permissions (warn-only if skipped — runs complete without
# them, but bottleneck evidence is much thinner):
echo 'kernel.perf_event_paranoid=-1' | sudo tee /etc/sysctl.d/99-capsim.conf
sudo sysctl --system
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null || true
```

RHEL/Rocky: `dnf install git python3 perf docker` (or docker-ce from
Docker's repo), driver via the NVIDIA CUDA repo or precompiled
modules, container toolkit from the same NVIDIA repo. Everything from
step 1 down is identical.

`capsim doctor` (step 2) verifies every one of these — including the
container-toolkit runtime registration, which nvidia-smi alone does
not prove — so run it after this stage rather than trusting the
install went cleanly.

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

Find the best engine launch shape BEFORE the first sweep — capacity
numbers are only as good as the launch config they were measured on.
From the UI's **Optimizer** tab (or `make optimize-engine
PROFILE=nvidia_qwen3` headless): it sweeps KV-pool sizing, batch
width, chunked prefill, and TP=2 vs data-parallel replicas against
representative latency/throughput cells, ranks the outcomes, and
persists/resumes at `runs/engine_optimizer/run.json`. Encode the
winner in the profile you sweep with (e.g. add the winning
`--max-num-seqs` to `vllm_extra_flags`, or switch to TP=2).

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
