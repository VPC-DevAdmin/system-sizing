"""Owned speech + language pipeline compared on two GPUs, sharing and MIG."""

import asyncio
import copy
import json
import math
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import yaml

from ..benchmarks.assets import fingerprint, pin_model
from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import assert_gpu_idle, capture_environment, telemetry
from ..config import load_config
from .calls import grade_summary, summary_prompt, word_error_rate
from .deployment import OwnedVllm, finish_owned_cleanup, launch_owned
from .driver import summarize
from .partition import Partition


async def _process_call(client, llm, row, spec):
    start = time.monotonic()
    result = dict(task_id=row["id"], transport_ok=False, correct=False, ttft_s=None, usage=None)
    try:
        response = await client.post("http://127.0.0.1:19390/transcribe", json={"id": row["id"]})
        response.raise_for_status()
        speech = response.json()
        asr_done = time.monotonic()
        transcript = speech["text"]
        body = dict(
            model=llm.api_model_name,
            messages=[{"role": "user", "content": summary_prompt(transcript)}],
            temperature=0,
            seed=spec.seed,
            max_tokens=256,
            chat_template_kwargs={"enable_thinking": False},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "call_summary",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            k: {"type": "string"} for k in ["issue", "action", "owner", "deadline"]
                        },
                        "required": ["issue", "action", "owner", "deadline"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        submitted = time.monotonic()
        reply = await client.post(
            f"http://{llm.cfg.host}:{llm.cfg.port}/v1".rstrip("/") + "/chat/completions",
            json=body,
            headers={"Authorization": "Bearer " + llm.api_key},
        )
        reply.raise_for_status()
        data = reply.json()
        content = data["choices"][0]["message"]["content"]
        wer = word_error_rate(row["reference"], transcript)
        parsed = json.loads(content)
        summary_ok = grade_summary(parsed, row["expected"])
        result.update(
            transport_ok=True,
            correct=wer <= spec.call_max_wer and summary_ok,
            wer=wer,
            summary_correct=summary_ok,
            transcript=transcript,
            summary=parsed,
            audio_seconds=speech["audio_seconds"],
            asr_s=asr_done - start,
            asr_server_s=speech["server_s"],
            asr_queue_s=speech["asr_queue_s"],
            transcript_bytes=len(transcript.encode()),
            handoff_client_s=submitted - asr_done,
            summary_s=time.monotonic() - submitted,
            usage=data.get("usage"),
            finish_reason=data["choices"][0].get("finish_reason"),
        )
        if result["finish_reason"] != "stop":
            result["correct"] = False
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["elapsed_s"] = time.monotonic() - start
    return result


async def process_call(client, llm, row, spec):
    start = time.monotonic()
    try:
        return await asyncio.wait_for(_process_call(client, llm, row, spec), spec.request_timeout_s)
    except asyncio.TimeoutError:
        return dict(
            task_id=row["id"],
            transport_ok=False,
            correct=False,
            ttft_s=None,
            usage=None,
            elapsed_s=time.monotonic() - start,
            error="Pipeline deadline exceeded",
        )


async def measure_calls(llm, corpus, spec, path, rate):
    start = time.monotonic()
    rows = []
    tasks = set()
    all_tasks = []
    slots = asyncio.Semaphore(spec.concurrency)
    count = math.ceil(spec.duration_s * rate)
    with path.open("w") as out:

        def save(row):
            rows.append(row)
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()

        async with httpx.AsyncClient(
            timeout=spec.request_timeout_s,
            limits=httpx.Limits(max_connections=spec.concurrency * 2),
        ) as client:

            async def send(i, arrival):
                async with slots:
                    queue = time.monotonic() - arrival
                    r = await process_call(client, llm, corpus[i % len(corpus)], spec)
                    r.update(request_id=i, queue_s=queue, scheduled_offset_s=i / rate)
                    r["elapsed_s"] += queue
                    save(r)

            try:
                for i in range(count):
                    arrival = start + i / rate
                    await asyncio.sleep(max(0, arrival - time.monotonic()))
                    if len(tasks) >= spec.concurrency * 4:
                        save(
                            dict(
                                request_id=i,
                                transport_ok=False,
                                correct=False,
                                ttft_s=None,
                                elapsed_s=0,
                                error="admission_limit",
                            )
                        )
                        continue
                    task = asyncio.create_task(send(i, arrival))
                    tasks.add(task)
                    all_tasks.append(task)
                    task.add_done_callback(tasks.discard)
                await asyncio.gather(*all_tasks)
                await asyncio.sleep(max(0, start + spec.duration_s - time.monotonic()))
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - start
    result = summarize(rows, elapsed, spec)
    result.update(
        offered_requests=count,
        offered_calls_per_second=rate,
        qualified_audio_hours_per_hour=sum(
            r.get("audio_seconds", 0)
            for r in rows
            if r["correct"] and r["elapsed_s"] <= spec.response_slo_s
        )
        / elapsed,
        summary_correct=sum(r.get("summary_correct", False) for r in rows),
    )
    result["gates_passed"] &= len(rows) == count
    return result


async def run(path, manifest, spec):
    from .call_report import compare_calls
    from .call_runtime import OwnedSpeech, prepare
    from .report import export
    from .tenant_worker import config

    state = read_json(path / "state.json")
    comparison = {"qualified": False, "metrics": {}, "reason": "Incomplete post-call comparison"}

    def update(phase, **kw):
        state.update(phase=phase, updated_at=time.time(), **kw)
        write_json(path / "state.json", state)

    try:
        with Lease(hardware_lease_path()):
            await asyncio.to_thread(
                assert_gpu_idle, SimpleNamespace(cfg=SimpleNamespace(type="vllm_cuda"))
            )
            corpus = read_json(path / "corpus.json")
            cfgfile = path / "prepare-config.yaml"
            cfgfile.write_text(
                yaml.safe_dump(config(manifest["configuration"], spec, "dedicated", spec.gpu_ids))
            )
            cfg = load_config(cfgfile)
            assets = await asyncio.to_thread(fingerprint, cfg.engine)
            if not assets.get("snapshot") or not assets.get("image"):
                raise ValueError("Summary model must be cached")
            write_json(path / "summary-assets.json", assets)
            write_json(path / "environment.json", await asyncio.to_thread(capture_environment, cfg))
            update("preparing_speech")
            # Preparation is joined on cancellation before releasing the hardware lease.
            prep = asyncio.create_task(
                asyncio.to_thread(prepare, path, corpus, assets["image"]["id"])
            )
            try:
                image, model = await asyncio.shield(prep)
            except BaseException:
                await prep
                raise
            for cell in manifest["schedule"]:
                directory = path / f"block-{cell['block']:02d}-{cell['arm']}"
                directory.mkdir()
                llm = speech = partition = sampler = None
                try:
                    selected = (
                        spec.gpu_ids if cell["arm"] == "dedicated" else [spec.gpu_ids[-1]] * 2
                    )
                    if cell["arm"] == "mig":
                        update("partitioning", current=cell)
                        partition = Partition(
                            directory / "partition.json", path.name, spec.gpu_ids[-1]
                        )
                        job = asyncio.create_task(asyncio.to_thread(partition.create))
                        try:
                            selected = await asyncio.shield(job)
                        except BaseException:
                            await job
                            raise
                    doc = config(manifest["configuration"], spec, cell["arm"], [selected[1]])
                    (directory / "config.yaml").write_text(yaml.safe_dump(doc))
                    enginecfg = load_config(directory / "config.yaml").engine
                    enginecfg.gpu_image = assets["image"]["id"]
                    pin_model(enginecfg, assets["snapshot"])
                    (directory / "launch-resolved.yaml").write_text(
                        yaml.safe_dump(asdict(enginecfg))
                    )
                    llm = OwnedVllm(enginecfg, threading.Event(), path.name)
                    update("launching", current=cell)
                    await launch_owned(llm, directory / "replica-summary")
                    speechcfg = copy.deepcopy(enginecfg)
                    speechcfg.port = 19390
                    speech = OwnedSpeech(speechcfg, path.name, image, model, path, selected[0])
                    job = asyncio.create_task(
                        asyncio.to_thread(speech.launch, directory / "replica-speech")
                    )
                    try:
                        await asyncio.shield(job)
                    except BaseException:
                        speech.stopping.set()
                        try:
                            await job
                        except Exception:
                            pass
                        raise
                    write_json(
                        directory / "placement.json",
                        dict(
                            speech_device=selected[0],
                            summary_device=selected[1],
                            physical_gpus=2 if cell["arm"] == "dedicated" else 1,
                        ),
                    )
                    (directory / "speech-effective.json").write_bytes(
                        (path / "speech-effective.json").read_bytes()
                    )
                    sampler = asyncio.create_task(
                        telemetry(directory, llm, 4 * 1024**3, time.monotonic())
                    )
                    warm = [r for r in corpus if r["warmup"]]
                    measured = [r for r in corpus if not r["warmup"]]
                    async with httpx.AsyncClient(timeout=spec.request_timeout_s) as client:
                        warm_results = [
                            await process_call(client, llm, warm[0], spec) for _ in range(2)
                        ]
                        write_json(directory / "warmup.json", warm_results)
                        if not all(r["transport_ok"] for r in warm_results):
                            raise RuntimeError("Post-call warmup transport failed")
                    for rate in spec.call_request_rates:
                        update("measuring", current=dict(cell, rate=rate))
                        task = asyncio.create_task(
                            measure_calls(
                                llm, measured, spec, directory / f"rate-{rate:g}.jsonl", rate
                            )
                        )
                        try:
                            done, _ = await asyncio.wait(
                                [task, sampler], return_when=asyncio.FIRST_COMPLETED
                            )
                            if sampler in done:
                                await sampler
                                raise RuntimeError("Telemetry stopped")
                            metrics = await task
                        finally:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        state["results"].append(dict(cell, rate=rate, metrics=metrics))
                        update("measuring", current=dict(cell, rate=rate))
                finally:
                    update("cleaning", current=cell)
                    await finish_owned_cleanup([speech, llm], sampler, partition)
            comparison = compare_calls(state["results"], spec)
            write_json(path / "comparison.json", comparison)
            update("complete", current=None, outcome="needs_review")
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", current=None, error=str(exc), outcome="incomplete")
    finally:
        try:
            await asyncio.to_thread(export, path, comparison)
        except Exception as exc:
            update("failed", error="Evidence export failed: " + str(exc), outcome="incomplete")
