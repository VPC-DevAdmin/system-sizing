"""Host validation for the one-command landing flow (roadmap 0.4).

``capsim doctor`` extends the config-scoped preflight into a full host
report: CPU capabilities, GPU stack, Docker, disk, telemetry
permissions, and Hugging Face reachability — each as a pass/warn/fail/
skip row. Telemetry-permission problems are warn-only (collectors
degrade gracefully and the run completes with NULLs); missing Docker
or disk space are hard fails for the local-docker target.

The report renders as a table and lands as ``doctor.json`` for
scripting; the CLI exits non-zero when any check fails.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .preflight import HardwareInfo, detect_hardware

# Statuses in severity order. "skip" means the check doesn't apply on
# this host (e.g. GPU checks on a CPU-only box) — never an error.
OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    hardware: dict = field(default_factory=dict)
    gpus: list[dict] = field(default_factory=list)
    recommended_configs: list[str] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "checks": [asdict(c) for c in self.checks],
            "hardware": self.hardware,
            "gpus": self.gpus,
            "recommended_configs": self.recommended_configs,
            "failed": self.failed,
        }


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, (p.stdout or p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timed out"


# ── Individual checks ─────────────────────────────────────────────────


def _check_cpu(report: DoctorReport) -> HardwareInfo:
    info = detect_hardware()
    report.hardware = {
        "vendor": info.vendor,
        "cpu_model": info.cpu_model,
        "physical_cores": info.physical_cores,
        "sockets": info.sockets,
        "detection_status": info.detection_status,
    }
    if info.detection_status != "ok":
        report.add(
            "cpu", SKIP,
            "no /proc/cpuinfo — non-Linux host; run doctor on the "
            "target benchmark box",
        )
        return info
    interesting = sorted(
        info.flags & {"amx_tile", "amx_bf16", "avx512f", "avx512_bf16", "avx512_vnni"}
    )
    report.add(
        "cpu", OK,
        f"{info.vendor} — {info.cpu_model} "
        f"({info.physical_cores} physical cores, {info.sockets} socket(s); "
        f"isa: {', '.join(interesting) or 'no AMX/AVX-512 flags'})",
    )
    return info


def _check_numa(report: DoctorReport) -> None:
    nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*")) \
        if Path("/sys/devices/system/node").exists() else []
    if not nodes:
        report.add("numa", SKIP, "no NUMA topology exposed")
        return
    report.add("numa", OK, f"{len(nodes)} NUMA node(s)")


def _check_docker(report: DoctorReport) -> bool:
    rc, out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if rc == 127:
        report.add(
            "docker", FAIL,
            "docker not installed — required to launch engines "
            "(local-docker target)",
        )
        return False
    if rc != 0:
        report.add(
            "docker", FAIL,
            f"docker daemon unreachable: {out.splitlines()[-1] if out else 'unknown error'}",
        )
        return False
    report.add("docker", OK, f"daemon reachable (server {out})")
    return True


def _check_gpu(report: DoctorReport, docker_ok: bool) -> bool:
    rc, out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    if rc == 127:
        report.add("gpu", SKIP, "no nvidia-smi — CPU-only host")
        return False
    if rc != 0:
        report.add(
            "gpu", WARN,
            f"nvidia-smi present but failed: {out.splitlines()[-1] if out else rc}",
        )
        return False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            report.gpus.append({
                "name": parts[0], "memory": parts[1], "driver": parts[2],
            })
    if report.gpus:
        detail = (
            f"{len(report.gpus)} GPU(s): "
            + "; ".join(f"{g['name']} ({g['memory']})" for g in report.gpus)
            + f" — driver {report.gpus[0]['driver']}"
        )
    else:
        detail = "nvidia-smi responded but reported no GPUs"
    report.add("gpu", OK if report.gpus else WARN, detail)
    if not report.gpus:
        return False
    # Container toolkit: engines run inside Docker, so the nvidia
    # runtime must be registered for --gpus to work.
    if docker_ok:
        rc, out = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
        if rc == 0 and "nvidia" in out:
            report.add("gpu_container_toolkit", OK, "nvidia runtime registered with Docker")
        else:
            report.add(
                "gpu_container_toolkit", FAIL,
                "GPU present but Docker has no nvidia runtime — install "
                "nvidia-container-toolkit and restart the daemon",
            )
    return True


def _check_disk(report: DoctorReport) -> None:
    # Models are 30-60 GB plus Docker images: warn under 150 GB free,
    # fail under 30 GB. Checks the largest-free of the common roots.
    candidates = [Path("/data"), Path.home(), Path("/var/lib/docker")]
    best: tuple[Path, int] | None = None
    for c in candidates:
        if c.exists():
            free = shutil.disk_usage(c).free
            if best is None or free > best[1]:
                best = (c, free)
    if best is None:
        report.add("disk", WARN, "could not stat any known volume")
        return
    path, free = best
    free_gb = free / 1e9
    if free_gb < 30:
        report.add("disk", FAIL, f"only {free_gb:.0f} GB free at {path} — models alone need 30-60 GB")
    elif free_gb < 150:
        report.add("disk", WARN, f"{free_gb:.0f} GB free at {path} — enough for one model, tight for several")
    else:
        report.add("disk", OK, f"{free_gb:.0f} GB free at {path}")


def _check_perf(report: DoctorReport) -> None:
    p = Path("/proc/sys/kernel/perf_event_paranoid")
    if not p.exists():
        report.add("perf_pmu", SKIP, "no perf_event_paranoid — non-Linux host")
        return
    try:
        val = int(p.read_text().strip())
    except (OSError, ValueError):
        report.add("perf_pmu", WARN, "could not read perf_event_paranoid")
        return
    if val <= 0:
        report.add("perf_pmu", OK, f"perf_event_paranoid={val} — PMU + uncore collectors available")
    else:
        report.add(
            "perf_pmu", WARN,
            f"perf_event_paranoid={val} — system-wide PMU / IMC-bandwidth "
            f"collectors will be blocked (run completes without them); "
            f"fix: sudo sysctl kernel.perf_event_paranoid=-1",
        )


def _check_rapl(report: DoctorReport) -> None:
    rapl = Path("/sys/class/powercap")
    domains = sorted(rapl.glob("intel-rapl:*")) if rapl.exists() else []
    if not domains:
        report.add("rapl_power", SKIP, "no RAPL powercap domains")
        return
    probe = domains[0] / "energy_uj"
    try:
        probe.read_text()
        report.add("rapl_power", OK, f"{len(domains)} RAPL domain(s) readable")
    except (OSError, PermissionError):
        report.add(
            "rapl_power", WARN,
            "RAPL present but energy_uj not readable — package-power "
            "telemetry will be missing; fix: sudo chmod a+r "
            "/sys/class/powercap/intel-rapl:*/energy_uj",
        )


def _check_hf(report: DoctorReport) -> None:
    try:
        import httpx
        r = httpx.head("https://huggingface.co", timeout=5, follow_redirects=True)
        if r.status_code < 500:
            report.add("hf_reachability", OK, "huggingface.co reachable")
        else:
            report.add("hf_reachability", WARN, f"huggingface.co returned {r.status_code}")
    except Exception as e:  # noqa: BLE001
        report.add(
            "hf_reachability", WARN,
            f"huggingface.co unreachable ({type(e).__name__}) — model "
            f"downloads need network or a pre-staged model dir",
        )
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if os.environ.get("HF_TOKEN") or token_file.exists():
        report.add("hf_token", OK, "HF token present (gated models downloadable)")
    else:
        report.add("hf_token", SKIP, "no HF token — fine unless the model is gated")


def _recommend_configs(
    report: DoctorReport, info: HardwareInfo, gpu: bool,
    config_dir: Path,
) -> None:
    """Map detected hardware to candidate profiles (roadmap 1.3).

    Naming convention does the matching: profile/config stems carry
    their host class ("-gpu-" / "xeon" / "r7735"). GPU beats CPU when
    both apply — the GPU is why the box exists."""
    from .config import list_profiles

    profiles = list_profiles()   # name -> path; profiles/ wins over config/
    if not profiles:
        return

    def _stems(pred) -> list[str]:
        return sorted(n for n in profiles if pred(n))

    picks: list[str] = []
    if gpu:
        picks = _stems(lambda n: "-gpu-" in n or n.startswith("gpu-"))
    if not picks and info.vendor == "intel":
        picks = _stems(lambda n: n.startswith("xeon"))
    if not picks and info.vendor == "amd":
        picks = _stems(lambda n: n.startswith("r7735"))
    report.recommended_configs = picks
    if picks:
        report.add(
            "profile", OK,
            "candidate profiles for this host (--profile <name>): "
            + ", ".join(picks),
        )
    elif gpu or info.vendor:
        report.add(
            "profile", WARN,
            f"no bundled profile matches this host "
            f"(vendor={info.vendor}, gpu={gpu}) — see capsim list-profiles",
        )


def run_doctor(config_dir: Path = Path("config")) -> DoctorReport:
    report = DoctorReport()
    info = _check_cpu(report)
    _check_numa(report)
    docker_ok = _check_docker(report)
    gpu = _check_gpu(report, docker_ok)
    _check_disk(report)
    _check_perf(report)
    _check_rapl(report)
    _check_hf(report)
    _recommend_configs(report, info, gpu, config_dir)
    return report
