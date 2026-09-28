import asyncio
import json
from types import SimpleNamespace

from simulator.infrastructure import call_worker


def test_call_arrivals_include_rejections_and_end_to_end_queue(monkeypatch, tmp_path):
    async def process(*args):
        await asyncio.sleep(0.08)
        return dict(
            transport_ok=True,
            correct=True,
            elapsed_s=0.08,
            ttft_s=None,
            usage=None,
            audio_seconds=10,
            summary_correct=True,
        )

    monkeypatch.setattr(call_worker, "process_call", process)
    spec = SimpleNamespace(
        duration_s=0.15,
        concurrency=1,
        request_timeout_s=1,
        response_slo_s=0.1,
        min_requests=1,
        min_quality=0.95,
        max_error_fraction=0.01,
        min_slo_fraction=0.95,
    )
    result = asyncio.run(
        call_worker.measure_calls(None, [{}], spec, tmp_path / "requests.jsonl", 80)
    )
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert result["attempted"] == result["offered_requests"] == 12
    assert any(r.get("error") == "admission_limit" for r in rows)
    assert any(r.get("queue_s", 0) > 0.05 for r in rows)
    assert not result["gates_passed"]
    assert (
        result["qualified_audio_hours_per_hour"]
        == sum(r.get("audio_seconds", 0) for r in rows if r["correct"] and r["elapsed_s"] <= 0.1)
        / result["elapsed_s_including_drain"]
    )


def test_stop_during_cleanup_still_removes_both_engines_and_restores_partition():
    import threading
    import time

    from simulator.infrastructure.deployment import finish_owned_cleanup

    seen = []
    started = threading.Event()

    class Engine:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            started.set()
            time.sleep(0.04)
            seen.append(self.name)

    class Partition:
        def restore(self):
            seen.append("restored")

    async def run():
        task = asyncio.create_task(
            finish_owned_cleanup([Engine("speech"), Engine("summary")], partition=Partition())
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation must propagate after cleanup")

    asyncio.run(run())
    assert seen == ["speech", "summary", "restored"]


def test_pipeline_deadline_is_counted_as_a_failed_call(monkeypatch):
    async def stalled(*args):
        await asyncio.sleep(1)

    monkeypatch.setattr(call_worker, "_process_call", stalled)
    row = asyncio.run(
        call_worker.process_call(
            None, None, {"id": "timeout"}, SimpleNamespace(request_timeout_s=0.01)
        )
    )
    assert not row["transport_ok"] and not row["correct"]
    assert row["elapsed_s"] >= 0.01 and row["error"] == "Pipeline deadline exceeded"
