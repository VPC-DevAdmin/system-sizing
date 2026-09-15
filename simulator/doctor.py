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

import json
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


def _nearest_existing(path: Path) -> Path:
    """Walk up until a path exists — free space of the filesystem a
    not-yet-created directory WILL land on."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def parse_lsblk_unmounted(doc: dict, min_gb: float = 400.0) -> list[tuple[str, float]]:
    """Large block devices with no mountpoint anywhere in their tree —
    the 'this lab box has a data NVMe nobody mounted' case. Pure
    parser over ``lsblk -J -b -o NAME,SIZE,TYPE,MOUNTPOINT`` output."""

    def mounted(node: dict) -> bool:
        if node.get("mountpoint") or node.get("mountpoints", [None]) != [None] \
                and any(node.get("mountpoints") or []):
            return True
        return any(mounted(c) for c in node.get("children") or [])

    out: list[tuple[str, float]] = []
    for dev in doc.get("blockdevices") or []:
        if dev.get("type") != "disk":
            continue
        size_gb = float(dev.get("size") or 0) / 1e9
        if size_gb >= min_gb and not mounted(dev):
            out.append((str(dev.get("name")), size_gb))
    return out


def _check_disk(report: DoctorReport, docker_ok: bool = False) -> None:
    """Free space where the tool will ACTUALLY write — Docker's real
    storage root (asked of the daemon, honoring data-root moves), the
    HF cache the engine containers mount, and /data/ml when staged
    models live there — not just whichever common path is largest.
    Then two hints boot-disk-only checks can't give: a bigger mounted
    data filesystem worth pointing the caches at, and large unmounted
    disks. Models are 30-60 GB each plus ~20 GB of engine images:
    fail under 30 GB on any consumer, warn under 150 GB.
    """
    from .models import hf_cache_dir

    consumers: dict[str, Path] = {}
    if docker_ok:
        rc, out = _run(["docker", "info", "--format", "{{.DockerRootDir}}"])
        if rc == 0 and out.strip():
            consumers["docker"] = Path(out.strip().splitlines()[-1])
    consumers["hf-cache"] = hf_cache_dir()
    if Path("/data/ml").exists():
        consumers["/data/ml"] = Path("/data/ml")

    details: list[str] = []
    status = OK
    consumer_free: list[float] = []
    seen_dev: set = set()
    for name, path in consumers.items():
        probe = _nearest_existing(path)
        try:
            usage = shutil.disk_usage(probe)
            dev = probe.stat().st_dev
        except OSError:
            continue
        free_gb = usage.free / 1e9
        consumer_free.append(free_gb)
        same = " (same filesystem)" if dev in seen_dev else ""
        seen_dev.add(dev)
        details.append(f"{name}: {free_gb:.0f} GB free at {probe}{same}")
        if free_gb < 30:
            status = FAIL
        elif free_gb < 150 and status != FAIL:
            status = WARN
    if not details:
        report.add("disk", WARN, "could not stat any storage consumer path")
        return

    hints: list[str] = []
    # A mounted filesystem much bigger than where the caches sit.
    try:
        import psutil
        best: tuple[str, float] | None = None
        for part in psutil.disk_partitions(all=False):
            mp = part.mountpoint
            if part.fstype in ("tmpfs", "devtmpfs", "squashfs", "overlay", "vfat") \
                    or mp.startswith(("/boot", "/snap", "/System", "/private/var/vm")):
                continue
            try:
                free_gb = shutil.disk_usage(mp).free / 1e9
            except OSError:
                continue
            if best is None or free_gb > best[1]:
                best = (mp, free_gb)
        if best and consumer_free and best[1] > 2 * max(consumer_free) \
                and best[1] >= 200:
            hints.append(
                f"{best[0]} has {best[1]:.0f} GB free — point the caches "
                f"there (OPTIMIZER_HF_CACHE=<dir>; docker daemon "
                f"'data-root'; or mount/symlink /data/ml)"
            )
    except ImportError:
        pass
    # Unmounted large disks (Linux only; best effort).
    rc, out = _run(["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT"])
    if rc == 0 and out.strip():
        try:
            unmounted = parse_lsblk_unmounted(json.loads(out))
        except (ValueError, KeyError):
            unmounted = []
        if unmounted:
            devs = ", ".join(f"{n} ({g / 1000:.1f} TB)" if g >= 1000
                             else f"{n} ({g:.0f} GB)" for n, g in unmounted)
            hints.append(
                f"UNMOUNTED disk(s) present: {devs} — format and mount "
                f"(e.g. at /data), then point the caches there"
            )
            if status == OK:
                status = WARN
    report.add("disk", status, "; ".join(details + hints))


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
    _check_disk(report, docker_ok)
    _check_perf(report)
    _check_rapl(report)
    _check_hf(report)
    _recommend_configs(report, info, gpu, config_dir)
    return report
