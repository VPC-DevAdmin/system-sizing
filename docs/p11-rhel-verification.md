# P11 RHEL verification: side-by-side install and reduced run (plan, 2026-10-26)

Status: planned. Nothing installed yet.

## Why

The P11 survey ([p11-results.md](p11-results.md)) ran on Ubuntu 22.04. Canonical is a
UK company, so the host OS does not fit the all-US story. We re-verify on Red Hat
Enterprise Linux (Red Hat, US) without reinstalling or touching the Ubuntu system.

## Approach

Install RHEL on the spare disk `nvme1n1` and boot either OS from the iDRAC boot menu.
Ubuntu on `nvme0n1` stays as it is and remains the default boot. The three data disks
(`/data`, `/data2`, `/data3`, ext4, about 1 TB of model weights) are mounted by RHEL at
the same paths, so nothing is downloaded again.

Box facts this relies on (checked 2026-10-26):

| Item | Value |
|---|---|
| Ubuntu disk | `nvme0n1` (ESP, /boot, LVM root). Must not be touched. |
| RHEL target | `nvme1n1`, 3.2 TB, old unimported ZFS pool. Approved for wiping. |
| Data disks | `nvme2n1` /data (UUID dcfd91e6-6eb0-4e40-bb70-f42e76044968), `nvme4n1` /data2 (LABEL capsim-data), `nvme3n1` /data3 (LABEL capsim-data3). Must not be formatted. |
| Secure Boot | Disabled, so the NVIDIA kernel module needs no MOK enrollment |
| Network | Only link up is a USB Realtek RTL8153 (`r8152` driver), 100.67.151.209/24. The onboard Broadcom 25G ports are down. |
| User | `user`, UID/GID 1000, owner of the files on the data disks |
| GPU driver on Ubuntu | 580.178.04 |
| UEFI boot | Order 0005 (ubuntu), 0002 (PXE); timeout 0 |

## Impact on other users

The box also runs services owned by others: the agentic stack (LiteLLM, Langfuse,
Postgres, Redis, MinIO, ClickHouse, two CPU vLLM servers), dell-agent-quickstart,
polyglot and kgrag containers. They are all down while the box is booted into RHEL.
Schedule the window with their owners. Rebooting back into Ubuntu brings them back
(Docker restart policies), but this should be checked after the first switch back.

## Step 1: before rebooting (on Ubuntu, over ssh, no sudo needed)

1. Export the exact engine images so RHEL runs bit-identical containers:
   `docker save` of `vllm/vllm-openai:latest` (v0.29.0), `lmsysorg/sglang:latest`
   (v0.5.20) and `vllm/vllm-openai:reflection-beam` to `/data/p11-images/`
   (about 135 GB; /data has 259 GB free). Record the image IDs.
2. Note the IP setup (static or DHCP) for the RHEL installer.

## Step 2: RHEL install (needs a person with iDRAC access, about 1 hour)

1. Get an entitlement. The free Red Hat Developer subscription covers this, or use
   Dell's existing one. Download the latest RHEL 9 DVD ISO. RHEL 9 has the widest
   NVIDIA driver and Red Hat AI Inference Server (RHAIIS) support; check RHAIIS's
   supported host list at download time.
2. Mount the ISO through iDRAC virtual media and use one-time boot to start it.
3. In the installer:
   - **Installation Destination: select only `nvme1n1`.** Untick every other disk.
     Use automatic partitioning and "delete all" on `nvme1n1` only. This creates its own
     EFI partition on `nvme1n1` and leaves Ubuntu's alone.
   - Network: enable the Realtek USB adapter, with the same address as Ubuntu or a new
     one. If the installer doesn't see the adapter, cable one onboard 25G port instead.
   - Hostname `xe7740-rhel`.
   - Create user `user` with **UID/GID 1000**, administrator ticked.
   - Software selection: Server, no GUI.
4. After the install, boot order: put `ubuntu` back first so a normal reboot returns to
   Ubuntu, and boot RHEL with iDRAC "next boot" or `efibootmgr --bootnext`. (Optional:
   RHEL as default during the test window.)
5. First boot into RHEL:
   - `subscription-manager register`
   - Add the Mac's ssh key to `~user/.ssh/authorized_keys`
   - Give `user` passwordless sudo for the setup (`/etc/sudoers.d/p11`). It can be
     removed afterwards.
   - Update the `xe7740` ssh alias, or add `xe7740-rhel`, if the IP changes.

## Step 3: RHEL setup (Claude over ssh, about 1 hour)

1. `dnf update`, reboot.
2. Mount the data disks at the same paths. Add `/etc/fstab` entries by UUID/LABEL with
   `nofail`, matching Ubuntu's. The HF cache symlinks `/data2/... -> /data3/...` then
   resolve unchanged.
3. NVIDIA driver: the **580 branch** (same as Ubuntu, so only the OS changes) from
   NVIDIA's RHEL 9 CUDA repository, open kernel modules. Confirm with `nvidia-smi`
   (8 GPUs, 580.x).
4. Containers: Podman (Red Hat) plus `nvidia-container-toolkit`, with a CDI spec
   (`nvidia-ctk cdi generate`). Podman storage goes on `/data/podman`, because the
   100 GB-class root disk isn't the place for 135 GB of images. Then `podman load` the
   three saved images and check the IDs match Ubuntu's.
5. Pull the Red Hat AI Inference Server vLLM image (`registry.redhat.io/rhaiis/vllm-cuda-rhel9`,
   current version) after `podman login registry.redhat.io`. Record its vLLM version.
6. Rebuild the AIPerf 0.13.0 environment with `uv` (the Ubuntu venv points at
   Ubuntu's Python). Same version as the survey.
7. Smoke test: one small model (gpt-oss-20b) on one GPU, 32 users for 60 s.

## Step 4: driver changes (in the repo, before the run)

`scripts/p11_us_survey.py` needs four small changes:

1. **Container runtime switch** (`P11_RUNTIME=podman`). Use `--device nvidia.com/gpu=N`
   per GPU instead of `--gpus`. Add `--security-opt label=disable` so SELinux doesn't
   force a relabel of the 1 TB data mounts.
2. **`--ladder` option** to run only selected rungs.
3. **RHAIIS configs**: the same models with the RHAIIS image.
4. **Results root per OS** (`P11_OUT=/data/capsim/p11/results-rhel`), so Ubuntu data is
   never overwritten.

Everything else stays the same: workload, four AIPerf shards, quality corpus and grader,
and the engine settings and fixes listed in p11-results.md.

## Step 5: reduced verification run (about 5 hours)

Two arms, each recording every rung with no target applied (as in the survey):

**Arm A: same containers, RHEL host.** It isolates the host OS: same images (bit-identical),
same driver branch, same GPUs, same settings. Each model on the engine that peaked on
Ubuntu:

| Config | Rungs (users) | Window |
|---|---|---|
| gpt-oss-20b, vLLM, 1 × 8 | 32, 512, 2,048 | 120 s |
| Gemma-4-26B NVFP4, vLLM, 1 × 8 | 32, 512, 4,096 | 120 s |
| gpt-oss-120b, vLLM, 1 × 8 | 32, 512, 4,096 | 120 s |
| Gemma-4-31B NVFP4, vLLM, 1 × 8 | 32, 512, 2,048 | 120 s |
| Nemotron-3-Super NVFP4, vLLM, 2 × 4 | 32, 512, 2,048 | 120 s |
| Beam 1.0 NVFP4, vLLM (reflection-beam image), 8 × 1 | 32, 512, 1,024 | 120 s |
| Nemotron-3-Ultra NVFP4, SGLang, 8 × 1 | 32, 128, 1,024 | 120 s; 600 s for 1,024 |

Each config also reruns the quality check (32 + 32 documents).

The rungs are a low point, a mid point and each model's Ubuntu peak, all of which already
exist in the Ubuntu data for a direct comparison.

**Arm B: Red Hat AI Inference Server.** This is the fully Red Hat stack: RHEL host and
Red Hat's vLLM build. The same rungs and quality check for every model RHAIIS's vLLM
version can load. Beam probably needs the reflection-beam image, because Beam isn't in a
vLLM release yet; if RHAIIS can't load a model, that is recorded, not dropped.

**Reporting.** For each config and rung: Ubuntu vs RHEL (Arm A) and vs RHAIIS (Arm B)
gen tok/s, TTFT and ITL p50/p95, engine running requests, errors, and quality, with the
percentage difference. No pass/fail threshold. As a reference for run-to-run noise, the
Ubuntu data already has the same config at the same load measured twice (screen and
ladder at 1,024 users), and those differ by about 1–3%.

Output: `docs/p11-rhel-results.md` plus the JSON, in the same format as the survey.

## Step 6: switch back

Reboot into Ubuntu (default boot entry). Check the other users' containers came back.
Remove the temporary sudoers file on RHEL. Leave RHEL installed on `nvme1n1` for later
reruns unless the disk is needed.

## Risks

| Risk | Mitigation |
|---|---|
| The installer touches `nvme0n1` or a data disk | Select only `nvme1n1` in Installation Destination; check the summary screen before "Begin Installation" |
| RHEL installer has no driver for the USB Realtek adapter | Cable an onboard 25G port; RHEL ships Broadcom `bnxt_en` |
| NVIDIA 580 driver unavailable for the installed RHEL minor | Use the nearest 580.x NVIDIA publishes for that minor, and record it |
| SELinux blocks container access to the data disks | `--security-opt label=disable` on the P11 containers only |
| RHAIIS needs a newer RHEL or a different driver | Record it, run Arm A anyway; Arm B uses whatever RHAIIS supports |
| Other services are down during the window | Agree the window with their owners first |
