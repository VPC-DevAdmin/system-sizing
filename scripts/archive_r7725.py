#!/usr/bin/env python3
"""Export what is needed to reproduce the R7725 / MI350P results after the
server is wiped: measured results, run configs, launch settings, build files,
the hardware and software inventory, and model revisions.

Runs on the box with the standard library only. Reads; never writes outside
the output folder. Raw prompt sets, model weights, media frames and tool
environments are left out (see MANIFEST.md in the output).

    python3 scripts/archive_r7725.py [--out DIR] [--pip-freeze]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("CAPSIM_REPO") or Path(__file__).resolve().parent.parent)
RUNS = REPO / "runs"
CAPSIM = Path("/data/capsim")
HUB = CAPSIM / "huggingface" / "hub"
MAX_COPY = 5 * 1024 * 1024  # skip any single file above 5 MB
SKIP_DIRS = {"archive", "huggingface", "aiperf_models", "vbench_models", "__pycache__",
             "tokenizer-cache", "adapter-source", ".git", "frames", "media"}
SKIP_EXT = {".ppm", ".mov", ".jpg", ".jpeg", ".png", ".mp4", ".mkv", ".h264", ".hevc", ".264",
            ".265", ".pt", ".bin", ".safetensors", ".zip", ".pyc", ".so",
            ".docx", ".whl", ".tar", ".gz"}
SKIP_NAMES = {"inputs.json", "prompts.jsonl", "profile_export.jsonl"}
ENGINE_LOG_HEAD = 250  # lines: the launch command and resolved engine config


def sh(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")).strip()
    except Exception as e:  # noqa: BLE001
        return f"[failed: {e}]"


def copy_tree(src: Path, dst: Path, skipped: list[str]) -> int:
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(root) / f
            if p.is_symlink() or not p.is_file():
                continue
            if f in SKIP_NAMES or p.suffix.lower() in SKIP_EXT:
                continue
            if p.stat().st_size > MAX_COPY:
                skipped.append(f"{p} ({p.stat().st_size // 1_000_000} MB)")
                continue
            out = dst / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    return n


def export_results(out: Path) -> dict:
    """One CSV per table across every run.db, with the run name prepended."""
    out.mkdir(parents=True, exist_ok=True)
    tables = {"cohort_run": None, "cohort_measurements": None,
              "measurement_telemetry": None}
    writers, handles, counts = {}, {}, {t: 0 for t in tables}
    dbs = sorted(glob.glob(str(RUNS / "run_*" / "run.db")),
                 key=lambda p: int(Path(p).parent.name.split("_")[1]))
    for db in dbs:
        run = Path(db).parent.name
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            continue
        for t in tables:
            try:
                cur = c.execute(f'select * from "{t}"')
            except sqlite3.Error:
                continue
            cols = ["run"] + [d[0] for d in cur.description]
            if t not in writers:
                name = f"{t}.csv.gz" if t == "measurement_telemetry" else f"{t}.csv"
                h = (gzip.open(out / name, "wt", newline="") if name.endswith(".gz")
                     else open(out / name, "w", newline=""))
                handles[t] = h
                writers[t] = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
                writers[t].writeheader()
            for row in cur:
                writers[t].writerow(dict(zip(cols, (run, *row))))
                counts[t] += 1
        c.close()
    for h in handles.values():
        h.close()
    counts["run_databases"] = len(dbs)
    return counts


def export_run_files(out: Path) -> int:
    """Small per-run files plus the head of each engine log (launch settings)."""
    n = 0
    for d in sorted(RUNS.glob("run_*")):
        dst = out / d.name
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("engine_") and p.suffix == ".log":
                dst.mkdir(parents=True, exist_ok=True)
                with open(p, errors="replace") as f, open(dst / (p.stem + ".head.log"), "w") as o:
                    for i, line in enumerate(f):
                        if i >= ENGINE_LOG_HEAD:
                            break
                        o.write(line)
                n += 1
            elif p.suffix in {".json", ".yaml", ".yml"} and p.stat().st_size < 1_000_000:
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / p.name)
                n += 1
    return n


def export_models(out: Path) -> int:
    rows = []
    for m in sorted(HUB.glob("models--*")):
        repo = m.name[len("models--"):].replace("--", "/", 1)
        snaps = sorted((m / "snapshots").glob("*")) if (m / "snapshots").exists() else []
        size = sh(f"du -sLb '{m}' 2>/dev/null | cut -f1", 120)
        for s in snaps or [None]:
            cfg = {}
            if s and (s / "config.json").exists():
                try:
                    cfg = json.loads((s / "config.json").read_text())
                except Exception:  # noqa: BLE001
                    cfg = {}
            rows.append({
                "repo": repo,
                "revision": s.name if s else "",
                "bytes_on_disk": size if size.isdigit() else "",
                "architectures": ",".join(cfg.get("architectures", []) or []),
                "torch_dtype": cfg.get("torch_dtype", cfg.get("dtype", "")),
                "quantization": json.dumps(cfg.get("quantization_config", ""))[:400],
            })
    with open(out / "models.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["repo"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


SYSTEM_CMDS = {
    "os-release.txt": "cat /etc/os-release",
    "uname.txt": "uname -a",
    "kernel-cmdline.txt": "cat /proc/cmdline",
    "lscpu.txt": "lscpu",
    "numa.txt": "numactl -H",
    "memory.txt": "free -h; echo; grep -E 'MemTotal|Huge' /proc/meminfo; echo; cat /sys/kernel/mm/transparent_hugepage/enabled",
    "storage.txt": "lsblk -o NAME,SIZE,TYPE,MODEL,MOUNTPOINT; echo; df -h",
    "pci-gpu.txt": "lspci -nn | grep -iE 'amd|display|accelerat|vga'; echo; lspci -vv -d 1002: 2>/dev/null | grep -E '^[0-9a-f]|LnkSta:|LnkCap:'",
    "cpu-governor.txt": "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1",
    "sysctl.txt": "sysctl -a 2>/dev/null | grep -E '^(vm\\.|kernel\\.numa|kernel\\.sched|net\\.core\\.somaxconn|fs\\.file-max)'",
    "limits.txt": "ulimit -a",
    "amd-smi-version.txt": "amd-smi version",
    "amd-smi-static.txt": "amd-smi static",
    "amd-smi-list.txt": "amd-smi list",
    "amd-smi-partition.txt": "amd-smi partition",
    "amd-smi-metric.txt": "amd-smi metric",
    "rocminfo.txt": "rocminfo",
    "rocm-version.txt": "cat /opt/rocm/.info/version* 2>&1; ls -d /opt/rocm* 2>&1",
    "rocm-packages.txt": "dpkg -l | grep -iE 'rocm|amdgpu|hip|hsa|mivisionx|rocdecode|rocjpeg|amd-smi'",
    "dkms.txt": "dkms status 2>&1",
    "amdgpu-module.txt": "modinfo amdgpu 2>&1 | head -20; echo; cat /sys/module/amdgpu/version 2>&1",
    "docker-version.txt": "docker version",
    "docker-images.txt": "docker images --digests --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Digest}}\t{{.Size}}\t{{.CreatedAt}}'",
    "docker-ps.txt": "docker ps -a --no-trunc --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}'",
    "python.txt": "python3 --version; pip --version 2>&1",
    "env-capsim.txt": "env | grep -iE '^(CAPSIM|VLLM|SGLANG|HSA|ROCM|HIP|HF_|AITER)' | sed -E 's/(TOKEN|KEY|SECRET)=.*/\\1=[redacted]/I'",
    "ssh-note.txt": "hostname; date -u",
}


def export_system(out: Path, pip_freeze: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, cmd in SYSTEM_CMDS.items():
        (out / name).write_text(sh(cmd) + "\n")
    images = [l.split("\t")[0] for l in (out / "docker-images.txt").read_text().splitlines() if l.strip()]
    insp = out / "docker-inspect"
    insp.mkdir(exist_ok=True)
    for img in images:
        safe = img.replace("/", "_").replace(":", "__")
        (insp / f"{safe}.json").write_text(sh(f"docker image inspect '{img}'") + "\n")
        (insp / f"{safe}.history.txt").write_text(sh(f"docker history --no-trunc '{img}'") + "\n")
        if pip_freeze and ("vllm" in img or "sgl" in img):
            (insp / f"{safe}.pip-freeze.txt").write_text(
                sh(f"docker run --rm --entrypoint sh '{img}' -c "
                   "'pip freeze 2>/dev/null || python3 -m pip freeze'", 300) + "\n")


def export_code(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "git-head.txt").write_text(sh(f"git -C {REPO} log -1 --format='%H %ci %s'") + "\n")
    (out / "git-status.txt").write_text(sh(f"git -C {REPO} status --short") + "\n")
    (out / "uncommitted.patch").write_text(sh(f"git -C {REPO} diff", 120) + "\n")
    untracked = sh(f"git -C {REPO} ls-files --others --exclude-standard").splitlines()
    for rel in untracked:
        p = REPO / rel
        if p.is_file() and p.stat().st_size < MAX_COPY and p.suffix.lower() not in SKIP_EXT:
            dst = out / "untracked" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    ap.add_argument("--out", default=f"/data/capsim/archive/r7725-mi350p-{stamp}")
    ap.add_argument("--pip-freeze", action="store_true",
                    help="start each engine image briefly (no GPU) to list its Python packages")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []

    counts = export_results(out / "results")
    n_run_files = export_run_files(out / "run-files")
    n_models = export_models(out)
    export_system(out / "system", a.pip_freeze)
    export_code(out / "code")

    n_capsim = copy_tree(CAPSIM, out / "capsim", skipped)
    top = out / "runs-top"
    top.mkdir(exist_ok=True)
    for p in RUNS.iterdir():
        if p.is_file() and p.stat().st_size < MAX_COPY:
            shutil.copy2(p, top / p.name)
    if (RUNS / "roofline").exists():
        copy_tree(RUNS / "roofline", top / "roofline", skipped)
    n_bench = copy_tree(RUNS / "benchmarks", out / "aiperf-benchmarks", skipped) \
        if (RUNS / "benchmarks").exists() else 0

    (out / "MANIFEST.md").write_text(f"""# R7725 / MI350P archive

Exported {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by
`scripts/archive_r7725.py` from `{sh('hostname')}`.

| Folder | Contents |
|---|---|
| `results/` | Every row of every `run.db`: runs with their full config ({counts.get('cohort_run', 0)}), measured load steps ({counts.get('cohort_measurements', 0)}), telemetry samples ({counts.get('measurement_telemetry', 0)}), from {counts.get('run_databases', 0)} databases |
| `run-files/` | Per-run JSON and the first {ENGINE_LOG_HEAD} lines of each engine log, which hold the launch command ({n_run_files} files) |
| `runs-top/` | Queue and roofline definition files from `runs/` |
| `aiperf-benchmarks/` | AIPerf cross-check configs, manifests and reports ({n_bench} files; prompt sets left out) |
| `capsim/` | `/data/capsim` scripts, Dockerfiles, phase results, hardening, partition, profile and probe outputs ({n_capsim} files) |
| `models.csv` | Every cached model with its exact revision and size ({n_models} rows) |
| `system/` | CPU, memory, storage, GPU, ROCm, driver, kernel settings, container images with digests, build history |
| `code/` | Repo commit on the box, uncommitted changes as a patch, untracked files |

## Left out on purpose

Model weights (re-download by revision from `models.csv`), AIPerf prompt
sets and raw request exports, media probe frames and video, tool
environments, compiled files, full engine logs, and any file over
{MAX_COPY // 1_000_000} MB.

## Not captured

- BIOS settings, firmware versions and DIMM layout (need root: `dmidecode`,
  iDRAC export).
- The container images themselves (37 to 70 GB each). The Dockerfiles and
  base image digests are here; a rebuild depends on the upstream commits
  still being available.

## Files skipped for size

""" + ("\n".join(f"- `{s}`" for s in skipped) or "None.") + "\n")
    print(json.dumps({"out": str(out), **counts, "run_files": n_run_files,
                      "models": n_models, "capsim_files": n_capsim,
                      "bench_files": n_bench, "skipped": len(skipped)}, indent=1))


if __name__ == "__main__":
    main()
