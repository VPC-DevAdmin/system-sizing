"""Shared machinery for whole-box, N-replica Docker engines.

``vllm_cuda_multi`` proved the shape: N independent GPU-pinned
containers, one per replica, sticky-routed by the pool so multi-turn
conversations keep prefix-cache locality. TensorRT-LLM needs exactly
the same lifecycle — launch N containers, wait for each to answer,
aggregate counters across them, stream every log into one file — and
differs only in the server binary it runs and the dialect its
metrics endpoint speaks.

Keeping that lifecycle in one place is not tidiness. The launch path
accumulated hard-won corrections (embedded quoting for multi-device
``--gpus``, the stale-container sweep, reading the engine's OWN
exception out of the log instead of "see the log"), and a second copy
would silently miss every one of them.

Subclasses supply:
  * ``ENGINE_NAME``      — log filename and container-name prefix
  * ``build_replica_command(index, devices, name)`` — the docker argv
  * ``_ready_url(port)`` / ``_metrics_url(port)``   — optional
  * ``parse_metrics(text)``                        — optional
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from .base import Engine, redact_argv

log = logging.getLogger(__name__)

# Container-name prefixes capsim owns exclusively. Anything matching
# is fair game for the pre-launch sweep.
# One per engine: a leftover container of ANY engine holds port 9100
# and answers health checks for the wrong server, so every prefix must
# be swept before every launch.
# Per-launch HTTP port windows. Each launch takes the next window of
# PORT_WINDOW ports above cfg.port, cycling through PORT_WINDOWS
# windows, so a port is reused only after PORT_WINDOWS launches --
# long past the ~60 s TIME_WAIT the previous cell's connections leave
# behind. With the defaults the range is cfg.port .. cfg.port + 127.
PORT_WINDOW = 16          # replicas per launch never exceed 8
PORT_WINDOWS = 8
_launch_counter = itertools.count(int(time.time()))


def next_launch_number() -> int:
    """Monotonic within this process, seeded from the clock so two
    processes started seconds apart (a serve restart mid-roofline) do
    not replay the same windows."""
    return next(_launch_counter)


def replica_port(base_port: int, launch_no: int, index: int) -> int:
    if not 0 <= index < PORT_WINDOW:
        raise ValueError(f"replica index {index} exceeds the port window")
    return int(base_port) + (int(launch_no) % PORT_WINDOWS) * PORT_WINDOW + int(index)


CAPSIM_CONTAINER_PREFIXES = ("vllm-", "trtllm-", "sglang-",
                             "ktransformers-", "llamacpp-")


def gpus_arg_for(device_ids: list[int]) -> str:
    """Docker ``--gpus`` value for a device list. Docker parses the
    value as CSV, so multi-device lists need EMBEDDED quotes
    (`"device=0,1"`, quote characters included) or the daemon reads
    device=0 + count=1 and refuses."""
    arg = "device=" + ",".join(str(i) for i in device_ids)
    return f'"{arg}"' if len(device_ids) > 1 else arg


def container_name_filter(prefixes=CAPSIM_CONTAINER_PREFIXES) -> list[str]:
    """``docker ps --filter`` values that match capsim's containers and
    ONLY capsim's.

    Docker's ``name=`` filter is an unanchored regular expression, so
    ``name=vllm-`` also matched a user's ``my-vllm-dev`` and the sweep
    removed it. Anchored to the start of the name; Docker reports
    names with a leading slash and matches the filter against that
    form in some versions, so the anchor tolerates an optional one.
    """
    # The prefixes are letters and a hyphen, literal in a Go regexp
    # outside a character class -- no escaping, which would only add
    # a "\-" for the daemon to puzzle over.
    return [f"name=^/?{p}" for p in prefixes]


def remove_stale_engine_containers() -> None:
    """rm -f any leftover capsim engine container before launching.

    A hard-killed serve leaves its engine containers RUNNING — the
    next launch then dies with "address already in use" while the
    health check happily gets 200s from the OLD engine on the same
    port, and every request 404s against the wrong model. These name
    prefixes are exclusively capsim-owned, and benchmark launches are
    mutually exclusive with the optimizer, so removal here is safe.
    """
    for prefix, flt in zip(CAPSIM_CONTAINER_PREFIXES,
                           container_name_filter(), strict=True):
        try:
            res = subprocess.run(
                ["docker", "ps", "-aq", "--filter", flt],
                capture_output=True, text=True, timeout=30,
            )
            cids = res.stdout.split()
            if cids:
                log.warning(
                    "removing %d stale %s* container(s) from a previous "
                    "run before launch", len(cids), prefix,
                )
                subprocess.run(["docker", "rm", "-f", *cids],
                               capture_output=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            log.debug("stale-container sweep for %s failed: %s", prefix, e)


def aggregate_replica_metrics(per_replica: list[dict]) -> dict[str, float]:
    """Whole-box view of N replicas' metrics: counters summed
    (including the token counters the prefill/decode rates derive
    from), gauges averaged, hit rate recomputed from the sums."""
    if not per_replica:
        return {}
    agg: dict[str, float] = {}
    for k in ("num_running", "queue_depth", "prefix_cache_hits",
              "prefix_cache_queries", "prompt_tokens_total",
              "generation_tokens_total", "preemptions_total",
              # Pool capacity is per replica; the box holds their sum.
              "kv_cache_tokens"):
        vals = [m[k] for m in per_replica if k in m]
        if vals:
            agg[k] = sum(vals)
    # Gauges: averaged. A rate that arrives ready-made (TensorRT-LLM's
    # cacheHitRate, SGLang's cache_hit_rate) would otherwise be dropped
    # on the floor for any multi-replica engine; when the underlying
    # counters are also present the recompute below overrides it.
    for k in ("kv_cache_used_pct", "prefix_cache_hit_rate"):
        vals = [m[k] for m in per_replica if k in m]
        if vals:
            agg[k] = sum(vals) / len(vals)
    if agg.get("prefix_cache_queries"):
        agg["prefix_cache_hit_rate"] = (
            agg.get("prefix_cache_hits", 0) / agg["prefix_cache_queries"]
        )
    return agg


class DockerReplicaEngine(Engine):
    """N GPU-pinned server containers, sticky-routed by the pool."""

    # Subclasses override: names the log file and the containers.
    ENGINE_NAME = "docker"

    def __init__(self, engine_config):
        super().__init__(engine_config)
        # (index, devices, port, container_id, streamer) per replica
        self._replicas: list[tuple[int, list[int], int, str,
                                   Optional[subprocess.Popen]]] = []
        self._run_id: str = ""
        # HTTP ports rotate per launch (replica_port): the previous
        # cell's servers leave their ports in TIME_WAIT for a minute
        # after teardown, and trtllm-serve's bind check (a plain socket
        # without SO_REUSEADDR) fails on them -- "Address already in
        # use" on replicas 4-7 right after a stop, XE7740 2026-09-21.
        self._launch_no = next_launch_number()
        # Cancel safety. launch() runs in a worker thread
        # (asyncio.to_thread) and cancelling the awaiting task does
        # NOT stop the thread: it goes on creating containers while
        # the caller's shutdown() sees only what was appended so far.
        # So shutdown() raises a flag, every step of the launch checks
        # it, and the append is atomic with that check -- a container
        # created after the flag is removed on the spot.
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # ── Subclass interface ────────────────────────────────────────────

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        raise NotImplementedError

    def _ready_url(self, port: int) -> str:
        return f"http://{self.cfg.host}:{port}/v1/models"

    def _metrics_url(self, port: int) -> str:
        return f"http://{self.cfg.host}:{port}/metrics"

    def parse_metrics(self, text: str) -> dict[str, float]:
        return self._parse_prometheus(text)

    # ── Shape ─────────────────────────────────────────────────────────

    def _device_groups(self) -> list[list[int]]:
        groups = getattr(self.cfg, "replica_devices", None)
        if not groups or not all(isinstance(g, (list, tuple)) and g
                                 for g in groups):
            raise RuntimeError(
                f"{self.cfg.type} needs engine.replica_devices — a list "
                "of GPU-id lists, one per replica, e.g. [[0],[1],[2]]"
            )
        return [[int(d) for d in g] for g in groups]

    def _port(self, index: int) -> int:
        return replica_port(self.cfg.port, self._launch_no, index)

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self._port(0)}/v1"

    @property
    def replica_urls(self) -> list[str]:
        n = len(self._device_groups())
        return [f"http://{self.cfg.host}:{self._port(i)}/v1"
                for i in range(n)]

    # ── Docker argv helpers (shared by every subclass) ────────────────

    def _mount_args(self) -> list[str]:
        """Volume, token and env arguments common to every engine
        image: the HF cache the weights live in, plus any hub token."""
        cfg = self.cfg
        out: list[str] = []
        mounted = set()
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            if not Path(host_path).exists():
                continue
            out += ["-v", f"{host_path}:{container_path}"]
            mounted.add(container_path)
        if "/root/.cache/huggingface" not in mounted:
            from ..models import cache_mount_args
            out += cache_mount_args()
        from .base import hub_env_args
        out += hub_env_args(cfg.model_id)
        for k, v in (cfg.docker_extra_env or {}).items():
            out += ["-e", f"{k}={v}"]
        return out

    # ── Lifecycle ─────────────────────────────────────────────────────

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._replicas:
            raise RuntimeError("Engine already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                f"docker not found on PATH; {self.cfg.type} runs its "
                "image in Docker (needs nvidia-container-toolkit)."
            )
        groups = self._device_groups()
        flat = [d for g in groups for d in g]
        if len(set(flat)) != len(flat):
            raise RuntimeError(
                f"replica_devices assigns a GPU twice: {groups}"
            )
        remove_stale_engine_containers()

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        # Exposed so a subclass can derive per-LAUNCH resources from it.
        # Fixed ports are safe within one launch and unsafe across
        # consecutive ones: a sweep tears down eight replicas and
        # immediately starts eight more, and the old sockets have not
        # finished closing.
        self._run_id = run_id
        self._log_path = log_dir / f"engine_{self.ENGINE_NAME}_{run_id}.log"
        # A shutdown() that arrives BEFORE this point (the awaiting task
        # cancelled before the thread ran) is the one case not covered:
        # the launch proceeds and the caller's shutdown() has already
        # returned. Every later arrival is.
        self._stopping.clear()

        try:
            log.info("Starting %d %s replicas on devices %s",
                     len(groups), self.ENGINE_NAME, groups)
            for i, devices in enumerate(groups):
                self._launch_replica(i, devices, run_id)
            for i, _devices, port, cid, _ in list(self._replicas):
                self._wait_for_replica_ready(i, port, cid)
            log.info("All replicas ready: %s", ", ".join(self.replica_urls))
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        # Flag first, then take the list under the lock: a launch
        # thread past the flag check but before its append will see
        # the flag at the append and remove its own container.
        self._stopping.set()
        with self._lock:
            replicas, self._replicas = list(self._replicas), []
        for i, _devices, _port, cid, streamer in replicas:
            log.info("Stopping replica %d (%s)", i, cid[:12])
            self._stop_container(cid)
            self._stop_streamer(streamer)

    @staticmethod
    def _stop_container(cid: str) -> None:
        try:
            subprocess.run(["docker", "stop", "-t", "30", cid],
                           capture_output=True, timeout=45)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", cid],
                           capture_output=True)

    @staticmethod
    def _stop_streamer(streamer: Optional[subprocess.Popen]) -> None:
        if streamer is None:
            return
        try:
            streamer.terminate()
            try:
                streamer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamer.kill()
        except Exception:  # noqa: BLE001
            pass

    def _cancelled(self) -> RuntimeError:
        return RuntimeError(
            f"{self.ENGINE_NAME} launch cancelled: shutdown() was called "
            f"while replicas were still starting")

    def _launch_replica(self, index: int, devices: list[int],
                        run_id: str) -> None:
        if self._stopping.is_set():
            raise self._cancelled()
        # The name is fixed BEFORE docker run so a run that hangs (the
        # daemon stalls pulling an image, say) can still be cleaned up
        # by name -- there is no id to remove it by until it returns.
        name = f"{self.ENGINE_NAME.split('_')[0]}-r{index}-{run_id}"
        cmd = self.build_replica_command(index, devices, name)
        log.info("docker run r%d: %s", index, redact_argv(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            hint = ""
            if "could not select device driver" in stderr:
                hint = (" — nvidia-container-toolkit missing or the "
                        "daemon not restarted (capsim doctor checks this)")
            raise RuntimeError(
                f"docker run for replica {index} failed "
                f"(rc={e.returncode}): {stderr}{hint}"
            ) from e
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=60)
            raise RuntimeError(
                f"docker run for replica {index} did not return in 120s; "
                f"removed container {name} by name"
            ) from e
        cid = result.stdout.strip()
        streamer = self._spawn_log_streamer(cid, prefix=f"[r{index}] ")
        with self._lock:
            if self._stopping.is_set():
                # shutdown() ran between docker run returning and this
                # append; it never saw this container, so remove it.
                self._stop_streamer(streamer)
                subprocess.run(["docker", "rm", "-f", cid],
                               capture_output=True, timeout=60)
                raise self._cancelled()
            self._replicas.append((index, devices, self._port(index), cid,
                                   streamer))

    def _startup_cause(self, tail_bytes: int = 200_000) -> str:
        """The engine's OWN last error line, lifted out of the log.

        "see the log" makes every launch failure look alike and costs
        a round trip to diagnose; the exception line distinguishes an
        unsupported flag from a missing import from an OOM at a
        glance."""
        if self._log_path is None:
            return "no engine log was captured"
        try:
            with open(self._log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - tail_bytes))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return "engine log unreadable"
        # Python names its exceptions "SomeError: message". The LAST
        # one is usually the API server's generic wrapper ("Engine
        # core initialization failed. See root cause above"), which
        # says nothing — the real cause is raised earlier, in the
        # engine-core worker. Prefer the last NON-generic line.
        pat = re.compile(
            r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*)"
            r"\s*:\s+(\S.*)")
        looks_like_exc = re.compile(
            r"(Error|Exception)$|\.(errors?|exceptions?)\.", re.I)
        # Errors that are TEARDOWN NOISE, not causes. When init fails,
        # cleanup touches attributes that were never created, and the
        # resulting AttributeError is raised LAST -- so it wins the
        # "most recent exception" contest and buries the real reason.
        # This cost a wrong diagnosis once: TensorRT-LLM reported
        # "'PyTorchModelEngine' object has no attribute
        # 'cuda_graph_runner'" while the actual failure, 30 lines
        # earlier, was that its Transformers did not recognise the
        # model architecture at all.
        generic = re.compile(
            r"see root cause above|engine core initialization failed|"
            r"engine process failed to start|see stack trace|"
            r"object has no attribute 'cuda_graph_runner'|"
            r"executor worker returned error",
            re.I)
        fallback = None
        for ln in reversed(lines):
            m = pat.search(ln)
            if not m or not looks_like_exc.search(m.group(1)):
                continue
            msg = f"{m.group(1)}: {m.group(2).strip()[:300]}"
            if generic.search(m.group(2)):
                fallback = fallback or msg
                continue
            return msg
        if fallback:
            return fallback
        for ln in reversed(lines):
            if ln.strip():
                return ln.strip()[:300]
        return "the engine log is empty"

    # Lines that mean the launch is over even though the container is
    # still running. Some servers keep their API process alive after
    # the model-loading worker dies (KTransformers' balance_serve
    # forks the engine; the parent lingers), so "container exited" is
    # never raised and the health wait runs the full timeout -- 30
    # minutes per cell on the XE7740 for a DeepSeek load that died in
    # its first CUDA kernel. The failure is then filed as a transient
    # timeout and retried on every resume. Matched against the
    # engine's own log; deliberately narrow, since engines log
    # recoverable errors too.
    FATAL_LOG_PATTERNS = re.compile(
        r"no kernel image is available for execution on the device|"
        r"CUDA error: (?:an illegal memory access|device-side assert|"
        r"invalid device function|out of memory)|"
        r"torch\.OutOfMemoryError|"
        r"Executor creation failed due to insufficient GPU memory",
        re.I)

    def _fatal_in_log(self, tail_bytes: int | None = 65536) -> Optional[str]:
        """The first fatal line in the engine log, or None. ``tail_bytes``
        bounds the read for the once-a-few-seconds startup scan; None
        reads the whole file -- an engine that died mid-sweep has by
        then buried its traceback under a hundred thousand lines of
        request errors (DeepSeek-V3.1-NVFP4 at tp8: the OOM sat 180k
        lines back and the death was reported without its cause)."""
        if self._log_path is None:
            return None
        try:
            with open(self._log_path, "rb") as f:
                if tail_bytes is not None:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - tail_bytes))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return None
        m = self.FATAL_LOG_PATTERNS.search(text)
        if not m:
            return None
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end if line_end >= 0 else None].strip()
        return re.sub(r"^\[r\d+\]\s*", "", line)[:300]

    # A launch that is still visibly loading when the timeout lands is
    # not stuck. Nemotron-3-Super at tp2 x 4 replicas on the XE7740
    # finished its weight load and CUDA-graph capture at 30:00 -- the
    # very second the 1800 s wait gave up -- and was filed as a
    # transient timeout. While the engine log keeps growing the wait
    # continues, up to PROGRESS_GRACE_FACTOR x the timeout; a log that
    # has been silent for PROGRESS_SILENCE_S is a hang and the timeout
    # stands.
    PROGRESS_GRACE_FACTOR = 2.0
    PROGRESS_SILENCE_S = 300.0

    def _log_size(self) -> int:
        try:
            return self._log_path.stat().st_size if self._log_path else 0
        except OSError:
            return 0

    def _wait_for_replica_ready(self, index: int, port: int,
                                container_id: str) -> None:
        start = time.time()
        backoff = 1.0
        last_scan = 0.0
        timeout = float(self.cfg.startup_timeout_s)
        hard_cap = timeout * self.PROGRESS_GRACE_FACTOR
        size = self._log_size()
        last_growth = start
        extended = False
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= timeout:
                cur = self._log_size()
                if cur > size:
                    size, last_growth = cur, now
                if (elapsed >= hard_cap
                        or now - last_growth > self.PROGRESS_SILENCE_S):
                    break
                if not extended:
                    extended = True
                    log.info("replica %d not ready at %.0fs but its log "
                             "is still growing; waiting up to %.0fs",
                             index, timeout, hard_cap)
            if self._stopping.is_set():
                raise self._cancelled()
            if time.time() - last_scan >= 5.0:
                last_scan = time.time()
                fatal = self._fatal_in_log()
                if fatal:
                    raise RuntimeError(
                        f"replica {index} engine died during startup "
                        f"(container still running): {fatal} (full log: "
                        f"{self._log_path})")
            try:
                r = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}",
                     container_id],
                    capture_output=True, text=True, timeout=5,
                )
                if r.stdout.strip() != "true":
                    raise RuntimeError(
                        f"replica {index} container exited during startup: "
                        f"{self._startup_cause()} (full log: "
                        f"{self._log_path})"
                    )
            except subprocess.TimeoutExpired:
                pass
            try:
                resp = httpx.get(self._ready_url(port), timeout=2.0)
                if resp.status_code == 200:
                    log.info("replica %d ready after %.1fs",
                             index, time.time() - start)
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.2)
        raise TimeoutError(
            f"replica {index} not healthy in {int(time.time() - start)}s "
            f"(timeout {self.cfg.startup_timeout_s}s"
            f"{', extended while the log grew' if extended else ''}) "
            f"— see {self._log_path}"
        )

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def pid(self) -> Optional[int]:
        pids = self.pids
        return pids[0] if pids else None

    @property
    def pids(self) -> list[int]:
        """Host PID of every replica container — the engine-RSS
        rollup sums the whole set, not just replica 0."""
        out: list[int] = []
        for _i, _d, _p, cid, _s in self._replicas:
            try:
                r = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Pid}}", cid],
                    capture_output=True, text=True, timeout=5,
                )
                v = r.stdout.strip()
                if v and v != "0":
                    out.append(int(v))
            except Exception:  # noqa: BLE001
                continue
        return out

    def health_check(self) -> bool:
        if not self._replicas:
            return False
        for _i, _d, port, _cid, _s in self._replicas:
            try:
                r = httpx.get(self._ready_url(port), timeout=2.0)
                if r.status_code != 200:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    def get_metrics(self) -> dict[str, float]:
        """Whole-box counters, plus how many replicas answered.

        A replica whose scrape fails is left out of the sums. Because
        every rate is a difference of cumulative counters between two
        scrapes, a missing replica at one boundary and not the other
        makes its whole history look like new generation: the XE7740's
        gpt-oss telemetry showed cache-hit counters going backwards
        and decode rates of 895k and 1.08M tok/s. The sweep reads
        ``replicas_scraped`` against ``replicas_total`` and discards a
        chunk that was not whole.
        """
        per_replica = []
        for i, _d, port, _cid, _s in self._replicas:
            try:
                # Ten seconds, not two: a 1T model's /metrics at 4k
                # streams answered slowly and every boundary scrape
                # that timed out cost the chunk.
                r = httpx.get(self._metrics_url(port), timeout=10.0)
                if r.status_code == 200:
                    per_replica.append(self._parse_replica(i, r.text))
            except Exception:  # noqa: BLE001
                continue
        agg = aggregate_replica_metrics(per_replica)
        agg["replicas_scraped"] = float(len(per_replica))
        agg["replicas_total"] = float(len(self._replicas))
        return agg

    def _parse_replica(self, index: int, text: str) -> dict[str, float]:
        """Hook so stateful parsers can key accumulators per replica."""
        return self.parse_metrics(text)

    def _build_command(self) -> list[str]:
        return []          # launch() is overridden entirely

    def _build_env(self) -> dict[str, str]:
        return dict(os.environ)

    def _spawn_log_streamer(self, container_id: str, *,
                            prefix: str) -> Optional[subprocess.Popen]:
        if self._log_path is None:
            return None
        try:
            log_file = open(self._log_path, "ab")
            # sed -u: line-buffered. Into a file sed block-buffers by
            # default, so the engine's last lines -- the ones a startup
            # failure is diagnosed from -- sat in its buffer while
            # _startup_cause read an empty log.
            shell_cmd = (f"docker logs -f {container_id} 2>&1 | "
                         f"sed -u 's/^/{prefix}/'")
            return subprocess.Popen(shell_cmd, shell=True, stdout=log_file,
                                    stderr=subprocess.STDOUT)
        except Exception as e:  # noqa: BLE001
            log.warning("log streamer for %s failed: %s",
                        container_id[:12], e)
            return None
