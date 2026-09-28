import pytest

from simulator.infrastructure.dynamo_runtime import commands
from simulator.infrastructure.topology import pool_plan


def inventory():
    return {
        g: dict(node=g // 4, socket=g // 4, cpus=[g // 4, g // 4 + 2], pci_address=str(g))
        for g in range(8)
    }


def test_three_of_eight_can_remain_local_without_equal_pool_capacity():
    plan = pool_plan([0, 4, 1, 5, 2, 6, 3, 7], 3, "treatment", "socket_local", inventory())
    assert plan["worker_roles"].count("prefill") == 3
    assert all(p["same_socket"] for p in plan["allowed_kv_pairs"])
    counts = [
        (
            sum(
                r == "prefill" and g == pool
                for r, g in zip(plan["worker_roles"], plan["worker_groups"], strict=True)
            ),
            sum(
                r == "decode" and g == pool
                for r, g in zip(plan["worker_roles"], plan["worker_groups"], strict=True)
            ),
        )
        for pool in (0, 1)
    ]
    assert counts == [(1, 3), (2, 2)]
    cross = pool_plan([0, 4, 1, 5, 2, 6, 3, 7], 3, "treatment", "cross_socket", inventory())
    assert cross["worker_roles"] == plan["worker_roles"]
    assert all(not p["same_socket"] for p in cross["allowed_kv_pairs"])


def test_pool_isolation_and_affinity_are_in_effective_commands():
    topology = pool_plan(list(range(8)), 4, "treatment", "socket_local", inventory())
    plan = dict(
        topology,
        port=19200,
        model="/model",
        served_model_name="m",
        context_limit=32768,
        max_num_seqs=96,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="test",
        gpu_ids=list(range(8)),
        prefill_workers=4,
        arm="treatment",
    )
    rows = commands(plan)
    assert len(rows) == 10
    front = rows[:2]
    assert front[0][2]["DYN_FILE_KV"] != front[1][2]["DYN_FILE_KV"]
    for i, (_, argv, env) in enumerate(rows[2:]):
        group = topology["worker_groups"][i]
        assert env["DYN_FILE_KV"] == front[group][2]["DYN_FILE_KV"]
        assert env["DYN_NAMESPACE"] == front[group][2]["DYN_NAMESPACE"]
        assert env["CAPSIM_CPU_AFFINITY"]
        assert argv[argv.index("--disaggregation-mode") + 1] == topology["worker_roles"][i]


def test_invalid_groups_fail_instead_of_silently_crossing_sockets():
    for selected, prefill in [([0, 4], 1), ([0, 1, 4, 5], 1), ([0, 1, 4], 2)]:
        with pytest.raises(ValueError):
            pool_plan(selected, prefill, "treatment", "socket_local", inventory())
    single = pool_plan([0, 1, 2, 3], 1, "treatment", "socket_local", inventory())
    assert len(single["frontend_cpus"]) == 1 and len(single["allowed_kv_pairs"]) == 3
    with pytest.raises(ValueError):
        pool_plan([0, 1, 2, 3], 1, "treatment", "cross_socket", inventory())


def test_transfer_checks_actual_decode_workers_after_topology_assignment(tmp_path):
    from types import SimpleNamespace

    from simulator.infrastructure.dynamo import verify_transfer

    metric = 'vllm:nixl_bytes_transferred_sum{model_name="m"}'
    for worker in (1, 3):
        (tmp_path / f"before-worker-{worker}.prom").write_text(metric + " 1")
        (tmp_path / f"after-worker-{worker}.prom").write_text(metric + " 101")
    verify_transfer(
        tmp_path,
        SimpleNamespace(prefill_workers=2, gpu_ids=list(range(4))),
        ["prefill", "decode", "prefill", "decode"],
    )
