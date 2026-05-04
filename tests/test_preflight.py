"""Hardware preflight tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.preflight import (
    HardwareInfo,
    HardwareRequirements,
    PreflightError,
    check_requirements,
    detect_hardware,
    preflight_check,
)
from simulator.config import load_config


# ── Detection ─────────────────────────────────────────────────────────


_INTEL_GNR_CPUINFO = """\
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model name	: Intel(R) Xeon(R) 6761P
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req hfi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
cpu cores	: 64
physical id	: 0

processor	: 64
vendor_id	: GenuineIntel
cpu cores	: 64
physical id	: 1
"""

_AMD_EPYC_CPUINFO = """\
processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 25
model name	: AMD EPYC 9354 32-Core Processor
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl avx512_bf16 vnmi avx512vbmi umip pku ospke vaes vpclmulqdq avx512_vbmi2 gfni avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d
cpu cores	: 32
physical id	: 0

processor	: 32
cpu cores	: 32
physical id	: 1
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_detect_hardware_intel(tmp_path: Path) -> None:
    cpuinfo = _write(tmp_path, "intel.cpuinfo", _INTEL_GNR_CPUINFO)
    info = detect_hardware(str(cpuinfo))
    assert info.detection_status == "ok"
    assert info.vendor == "intel"
    assert "Xeon" in (info.cpu_model or "")
    # Critical AMX flags present on Granite Rapids
    assert "amx_tile" in info.flags
    assert "amx_bf16" in info.flags
    assert info.physical_cores == 128   # 64 cores × 2 sockets
    assert info.sockets == 2


def test_detect_hardware_amd(tmp_path: Path) -> None:
    cpuinfo = _write(tmp_path, "amd.cpuinfo", _AMD_EPYC_CPUINFO)
    info = detect_hardware(str(cpuinfo))
    assert info.detection_status == "ok"
    assert info.vendor == "amd"
    assert "EPYC" in (info.cpu_model or "")
    # Specifically confirm AMX is NOT present on EPYC (the whole point)
    assert "amx_tile" not in info.flags
    assert "amx_bf16" not in info.flags
    # AVX-512 IS on Zen4 EPYC though
    assert "avx512f" in info.flags
    assert info.physical_cores == 64    # 32 cores × 2 sockets
    assert info.sockets == 2


def test_detect_hardware_no_proc_cpuinfo() -> None:
    """Non-Linux dev hosts return a sentinel; the validator soft-skips."""
    info = detect_hardware("/nonexistent/cpuinfo")
    assert info.detection_status == "no_proc_cpuinfo"
    assert info.vendor is None
    assert info.flags == set()


# ── Requirements check ────────────────────────────────────────────────


def _intel_info() -> HardwareInfo:
    return HardwareInfo(
        vendor="intel", cpu_model="Xeon 6761P",
        flags={"avx512f", "avx512_bf16", "amx_tile", "amx_bf16", "amx_int8"},
        physical_cores=128, sockets=2, detection_status="ok",
    )


def _amd_info() -> HardwareInfo:
    return HardwareInfo(
        vendor="amd", cpu_model="EPYC 9354",
        flags={"avx512f", "avx512_bf16"},
        physical_cores=64, sockets=2, detection_status="ok",
    )


def test_empty_requirements_always_pass() -> None:
    """Configs without hardware_requirements (the BF16 default) skip
    validation — no surprise gates for users who don't opt in."""
    assert check_requirements(_amd_info(), HardwareRequirements()) == []
    assert check_requirements(_intel_info(), HardwareRequirements()) == []


def test_intel_only_requirement_blocks_amd() -> None:
    """The headline use case: SGLang FP8 on AMD must fail preflight."""
    reqs = HardwareRequirements(
        cpu_vendor="intel",
        cpu_features=["amx_tile", "amx_bf16"],
    )
    failures = check_requirements(_amd_info(), reqs)
    assert failures, "AMD must not satisfy Intel+AMX requirements"
    msg = "\n".join(failures)
    assert "intel" in msg
    assert "amd" in msg
    assert "amx_tile" in msg


def test_intel_with_amx_passes() -> None:
    reqs = HardwareRequirements(
        cpu_vendor="intel",
        cpu_features=["amx_tile", "amx_bf16"],
    )
    assert check_requirements(_intel_info(), reqs) == []


def test_min_physical_cores_check() -> None:
    reqs = HardwareRequirements(min_physical_cores=128)
    assert check_requirements(_intel_info(), reqs) == []
    failures = check_requirements(_amd_info(), reqs)
    assert any("min_physical_cores" in f for f in failures)


def test_skips_when_detection_unavailable() -> None:
    """Non-Linux: detector reports no_proc_cpuinfo → no failures
    regardless of constraints (we can't check, so we don't gate)."""
    info = HardwareInfo(
        vendor=None, cpu_model=None, flags=set(),
        physical_cores=None, sockets=None,
        detection_status="no_proc_cpuinfo",
    )
    reqs = HardwareRequirements(cpu_vendor="intel", cpu_features=["amx_tile"])
    assert check_requirements(info, reqs) == []


# ── End-to-end: load real configs ─────────────────────────────────────


def test_fp8_config_carries_intel_amx_requirement() -> None:
    """The FP8 config MUST set the Intel-AMX gate — if someone removes
    this in a refactor, the preflight stops protecting AMD users."""
    cfg = load_config("config/xeon_sglang_qwen3_30b_a3b_fp8.yaml")
    reqs = cfg.engine.hardware_requirements
    assert reqs.cpu_vendor == "intel"
    assert "amx_tile" in reqs.cpu_features
    assert "amx_bf16" in reqs.cpu_features


def test_amd_dual_socket_config_requires_amd_with_avx512() -> None:
    """The R7735 dual-socket config bakes in 'this is AMD with AVX-512'.
    If someone refactors the preflight to allow Intel here, the box
    would silently drop into the wrong OMP-pinning shape."""
    cfg = load_config("config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml")
    reqs = cfg.engine.hardware_requirements
    assert reqs.cpu_vendor == "amd"
    assert "avx512f" in reqs.cpu_features
    assert "avx512_bf16" in reqs.cpu_features
    assert reqs.min_sockets == 2


# ── preflight_check end-to-end ────────────────────────────────────────


def test_preflight_check_raises_with_helpful_message(monkeypatch) -> None:
    """The error message must include the failing constraint, the
    detected vendor, and the cpu_model — enough to act on without
    consulting the source."""
    import simulator.preflight as pf
    monkeypatch.setattr(pf, "detect_hardware", _amd_info)
    reqs = HardwareRequirements(
        cpu_vendor="intel",
        cpu_features=["amx_tile"],
        notes="SGLang CPU FP8 needs AMX",
    )
    with pytest.raises(PreflightError) as ei:
        preflight_check(reqs)
    msg = str(ei.value)
    assert "FAILED" in msg
    assert "intel" in msg
    assert "amd" in msg
    assert "EPYC 9354" in msg
    assert "SGLang CPU FP8 needs AMX" in msg


def test_preflight_check_passes_silently_when_satisfied(monkeypatch) -> None:
    import simulator.preflight as pf
    monkeypatch.setattr(pf, "detect_hardware", _intel_info)
    reqs = HardwareRequirements(cpu_vendor="intel", cpu_features=["amx_tile"])
    # Should return True without raising
    assert preflight_check(reqs) is True
