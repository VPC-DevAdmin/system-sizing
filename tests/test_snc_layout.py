import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import planner_campaign as pc  # noqa: E402

JOB = {"id": "snc", "cpu_engine": {"type": "sglang", "tensor_parallel_size": 1}}


def test_snc_domains_cover_each_socket_once():
    cores = [set(map(int, pc.R770_SNC_CPUS[d].split(","))) for d in range(4)]
    assert all(len(c) == 43 for c in cores)
    assert cores[0] | cores[1] == set(map(int, pc.R770_SOCKET_CPUS[0].split(",")))
    assert cores[2] | cores[3] == set(map(int, pc.R770_SOCKET_CPUS[1].split(",")))
    assert not (cores[0] & cores[1]) and not (cores[2] & cores[3])


def test_snc_engine_pins_domain_cores_and_memory(tmp_path):
    doc = pc.cpu_config(JOB, [3], "s3", tmp_path, snc=True)
    eng = doc["engine"]
    assert eng["cpu_bind"] == pc.R770_SNC_CPUS[3]
    assert eng["docker_extra_args"][-2:] == ["--cpuset-mems", "3"]
    assert eng["port"] == pc.BASE_PORT + 3 * pc.LANE_PORTS


def test_socket_layout_unchanged_without_snc(tmp_path):
    eng = pc.cpu_config(JOB, [1], "s1", tmp_path)["engine"]
    assert eng["cpu_bind"] == pc.R770_SOCKET_CPUS[1]
    assert "--cpuset-mems" not in (eng.get("docker_extra_args") or [])


def test_numa_guard(tmp_path):
    q = pc.Queue.__new__(pc.Queue)
    q.box, q.doc = "r770", {"r770_snc": True}
    pc.check_numa(q, nodes=4)
    assert q.total_devices() == [0, 1, 2, 3]
    with pytest.raises(SystemExit):
        pc.check_numa(q, nodes=2)
    q.doc = {}
    pc.check_numa(q, nodes=2)
    with pytest.raises(SystemExit):
        pc.check_numa(q, nodes=4)
