"""Engine abstraction.

The simulator only ever talks to an OpenAI-compatible endpoint. Engine
implementations own their tuning recipe and metrics endpoint.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

import httpx

log = logging.getLogger(__name__)

# Environment variable names whose VALUES must never reach a log line,
# an exception message or a persisted failure reason. Matched as a
# case-insensitive substring of the name, so HF_TOKEN,
# HUGGING_FACE_HUB_TOKEN, OPENAI_API_KEY and AWS_SECRET_ACCESS_KEY are
# all covered without enumerating them.
SECRET_ENV_PATTERN = re.compile(r"TOKEN|SECRET|KEY|PASSWORD", re.I)
_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)


def _mask_assignment(token: str) -> str:
    m = _ENV_ASSIGNMENT.match(token)
    if m and m.group(2) and SECRET_ENV_PATTERN.search(m.group(1)):
        return f"{m.group(1)}=***"
    return token


def hub_env_args(model_id: str | None) -> list[str]:
    """``-e`` arguments every engine container gets for the Hugging
    Face hub: the token passthrough for gated models, and
    ``HF_HUB_OFFLINE=1`` when the weights are already staged in the
    cache the container mounts.

    Engines call the hub at startup even when every file is cached
    (vLLM lists the repo, tokenizers re-check revisions). On a
    benchmark box without outbound DNS that call fails and the launch
    dies with "Temporary failure in name resolution" -- the exact case
    capsim's Prepare flow stages weights to avoid. Offline mode makes
    the hub library serve the cache without the round-trip; it is set
    only when the snapshot is complete, so a model that still needs
    downloading keeps its network. An explicit ``HF_HUB_OFFLINE`` in
    the environment wins either way.
    """
    out: list[str] = []
    from ..models import hf_token
    tok = hf_token()
    if tok:
        out += ["-e", f"HF_TOKEN={tok}"]
    explicit = os.environ.get("HF_HUB_OFFLINE")
    if explicit is not None:
        out += ["-e", f"HF_HUB_OFFLINE={explicit}"]
    elif model_id and "/" in model_id:
        from ..models import model_status
        try:
            cached = bool(model_status(model_id)["cached"])
        except OSError:
            cached = False
        if cached:
            out += ["-e", "HF_HUB_OFFLINE=1"]
    return out


def redact_argv(cmd: Sequence[str]) -> str:
    """``" ".join(cmd)`` with secret env values masked.

    Every launcher passes the HF token to its container as
    ``-e HF_TOKEN=<value>`` and then logs the whole argv, which put the
    token in every engine log and -- via the optimizer's failure
    reason -- in ``run.json``. The NAME is kept (that a token was passed
    is diagnostic); the value is replaced by ``***``. Handles
    ``-e NAME=VALUE`` / ``--env NAME=VALUE``, the joined forms
    ``-eNAME=VALUE`` / ``--env=NAME=VALUE``, and a bare ``NAME=VALUE``
    token anywhere in the command (``env HF_TOKEN=... cmd``).
    """
    out: list[str] = []
    for tok in cmd:
        tok = str(tok)
        if tok.startswith("--env="):
            out.append("--env=" + _mask_assignment(tok[len("--env="):]))
        elif tok.startswith("-e") and not tok.startswith("--") and "=" in tok:
            out.append("-e" + _mask_assignment(tok[2:]))
        else:
            out.append(_mask_assignment(tok))
    return " ".join(out)


class Engine:
    """Base class for engine launchers."""

    def __init__(self, engine_config):
        self.cfg = engine_config
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self._log_path: Optional[Path] = None

    # -- Subclass interface ----------------------------------------------------

    def _build_command(self) -> list[str]:
        raise NotImplementedError

    def _build_env(self) -> dict[str, str]:
        raise NotImplementedError

    def _health_path(self) -> str:
        return "/health"

    def _metrics_path(self) -> str:
        return "/metrics"

    # -- Public API ------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self.cfg.base_url

    @property
    def replica_urls(self) -> list[str]:
        """Per-replica OpenAI-compatible base URLs.

        Single-backend engines (vLLM direct, SGLang) return a one-element
        list ``[self.base_url]``. Multi-backend engines (e.g. dual-socket
        NUMA-pinned vLLM) override to return one URL per replica; the
        simulator's pool manager hash-routes each virtual user to a
        specific replica so multi-turn conversations preserve prefix-
        cache locality on one backend.
        """
        return [self.base_url]

    @property
    def api_key(self) -> str:
        """OpenAI-compatible API key the simulator uses to talk to this
        engine. Direct vLLM / SGLang accept any non-empty value; the
        ``vllm_dual_socket`` engine overrides to return the configured
        LiteLLM master key. Default ``"EMPTY"`` works for engines with
        no auth."""
        return self.cfg.api_key if hasattr(self.cfg, "api_key") and self.cfg.api_key else "EMPTY"

    @property
    def model_id(self) -> str:
        """Canonical HF model id. Used for the tokenizer corpus and run
        metadata. NOT necessarily the name the engine serves under —
        see ``api_model_name`` for that."""
        return self.cfg.model_id

    @property
    def api_model_name(self) -> str:
        """Model name to send in OpenAI-compatible API requests.

        When ``served_model_name`` is set in config (we pass it via
        ``--served-model-name`` on the engine command line), the engine
        registers the model under that name and rejects requests using
        the canonical HF id. Vanilla vLLM / SGLang configs leave this
        unset and serve under the HF id directly, so the fallback to
        ``model_id`` is correct.
        """
        return getattr(self.cfg, "served_model_name", None) or self.cfg.model_id

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_path

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._proc is not None:
            raise RuntimeError("Engine already launched")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"engine_{self.cfg.type}_{int(time.time())}.log"
        self._log_path = log_path
        self._log_file = open(log_path, "w")

        cmd = self._build_command()
        env = self._build_env()
        log.info("Launching %s: %s", self.cfg.type, redact_argv(cmd))
        log.info("Engine logs -> %s", log_path)

        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        try:
            self._wait_for_health(self.cfg.startup_timeout_s)
        except Exception:
            # A server that came up but never answered is still
            # running. Left alone it holds the port and the CPU cores
            # into the next launch, which then fails for a reason that
            # has nothing to do with its own config.
            self.shutdown()
            raise

    def _signal_group(self, sig: int) -> None:
        """Signal the engine's whole process group; a group that has
        already gone is not an error."""
        if self._proc is None:
            return
        if os.name == "nt":
            if sig == signal.SIGKILL:
                self._proc.kill()
            else:
                self._proc.terminate()
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except ProcessLookupError:
            pass

    def shutdown(self) -> None:
        if self._proc is None:
            return
        log.info("Shutting down %s engine (pid=%s)", self.cfg.type, self._proc.pid)
        try:
            self._signal_group(signal.SIGTERM)
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("Engine did not stop gracefully; killing")
            self._signal_group(signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=10)
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            self._proc = None

    def health_check(self) -> bool:
        try:
            url = f"http://{self.cfg.host}:{self.cfg.port}{self._health_path()}"
            r = httpx.get(url, timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def get_metrics(self) -> dict[str, float]:
        """Return parsed Prometheus metrics from the engine.

        Returns the most relevant signals: kv cache utilisation, queue depth,
        prefix cache hits/misses. Falls back to an empty dict if not exposed.
        """
        url = f"http://{self.cfg.host}:{self.cfg.port}{self._metrics_path()}"
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code != 200:
                return {}
            return self._parse_prometheus(r.text)
        except Exception:
            return {}

    # -- Helpers ---------------------------------------------------------------

    def _wait_for_health(self, timeout_s: int) -> None:
        start = time.time()
        backoff = 1.0
        while time.time() - start < timeout_s:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(
                    f"Engine process exited with code {self._proc.returncode} "
                    f"during startup. See log for details."
                )
            if self.health_check():
                log.info("Engine healthy after %.1fs", time.time() - start)
                return
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.2)
        raise TimeoutError(f"Engine did not become healthy within {timeout_s}s")

    # Prometheus metric-name patterns that mean "fraction of KV cache
    # in use." vLLM has shipped this metric under at least three names
    # across recent CPU image versions:
    #   * ``vllm:gpu_cache_usage_perc``  (legacy, kept on CPU images for
    #     compatibility — depending on version may or may not emit)
    #   * ``vllm:cpu_cache_usage_perc``  (some intermediate releases)
    #   * ``vllm:kv_cache_usage_perc``   (newer / unified naming)
    # Match on substring so any of these (and any future variant
    # following the same convention) is caught.
    _KV_USAGE_PATTERNS = (
        "_cache_usage_perc",
        "kv_cache_usage",
    )

    @staticmethod
    def _label(line: str, key: str) -> Optional[str]:
        """One label value out of a Prometheus exposition line."""
        import re
        m = re.search(rf'{re.escape(key)}="([^"]*)"', line)
        return m.group(1) if m else None

    @staticmethod
    def _parse_prometheus(text: str) -> dict[str, float]:
        """Extract a small set of metric values from Prometheus exposition.

        Returns a normalised dict using canonical keys:
        kv_cache_used_pct, queue_depth, prefix_cache_hits,
        prefix_cache_queries, prefix_cache_hit_rate, num_running.

        Any unknown ``vllm:`` metric whose name suggests KV cache usage
        is also captured under ``kv_cache_used_pct`` — defensive against
        upstream renames in the vllm-openai-cpu image.
        """
        wanted = {
            "vllm:num_requests_running": "num_running",
            "vllm:num_requests_waiting": "queue_depth",
            "vllm:prefix_cache_hits_total": "prefix_cache_hits",
            "vllm:prefix_cache_queries_total": "prefix_cache_queries",
            # Monotonic token counters — the telemetry loop turns their
            # deltas into prefill tok/s and decode tok/s.
            "vllm:prompt_tokens_total": "prompt_tokens_total",
            "vllm:generation_tokens_total": "generation_tokens_total",
            # Scheduler evictions under KV pressure: a nonzero rate
            # means requests are being restarted — latency cliffs
            # follow. Distinct signal from queue depth.
            "vllm:num_preemptions_total": "preemptions_total",
            # SGLang. Names verified against the observability
            # collector in lmsysorg/sglang, not guessed: the queue
            # gauge is num_queue_reqs (there is no num_waiting_reqs),
            # and without the two token counters below the headline
            # sweep -- which measures throughput as a delta of
            # generation_tokens_total -- reads a flat zero.
            "sglang:num_running_reqs": "num_running",
            "sglang:num_queue_reqs": "queue_depth",
            "sglang:cache_hit_rate": "prefix_cache_hit_rate",
            "sglang:token_usage": "kv_cache_used_pct",
            "sglang:prompt_tokens_total": "prompt_tokens_total",
            "sglang:generation_tokens_total": "generation_tokens_total",
            # SGLang retracts requests under KV pressure; same signal
            # as vLLM preemptions -- latency cliffs follow.
            "sglang:num_retracted_reqs": "preemptions_total",
            # KV pool capacity, in tokens. Not a performance number --
            # it is how the one-memory-knob translation gets CHECKED.
            # Two engines given "the same" share of VRAM should hold a
            # comparable number of KV tokens; if they do not, a
            # throughput comparison between them is measuring
            # allocation rather than engine quality.
            "sglang:max_total_num_tokens": "kv_cache_tokens",
        }
        out: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            # vLLM publishes its KV pool size as LABELS on an Info
            # metric whose value is a constant 1.0, so the usual
            # name->value path cannot see it.
            if line.startswith("vllm:cache_config_info"):
                blocks = Engine._label(line, "num_gpu_blocks")
                size = Engine._label(line, "block_size")
                if blocks and size:
                    try:
                        out["kv_cache_tokens"] = float(blocks) * float(size)
                    except ValueError:
                        pass
                continue
            # name{labels} value  OR  name value
            try:
                name_part, _, value_str = line.rpartition(" ")
                name = name_part.split("{", 1)[0].strip()
            except Exception:
                continue
            if name in wanted:
                try:
                    out[wanted[name]] = float(value_str)
                except ValueError:
                    pass
                continue
            # Fallback: any vllm: metric whose name implies KV cache
            # utilisation. Don't overwrite an explicit hit, but pick up
            # variants the upstream image may have introduced.
            if name.startswith("vllm:") and "kv_cache_used_pct" not in out:
                if any(p in name for p in Engine._KV_USAGE_PATTERNS):
                    try:
                        out["kv_cache_used_pct"] = float(value_str)
                    except ValueError:
                        pass
        # Compute hit rate if components present
        if "prefix_cache_hits" in out and "prefix_cache_queries" in out:
            q = out["prefix_cache_queries"]
            if q > 0:
                out["prefix_cache_hit_rate"] = out["prefix_cache_hits"] / q
        # Convert kv usage from fraction to percent if it looks like a fraction
        if "kv_cache_used_pct" in out and out["kv_cache_used_pct"] <= 1.0:
            out["kv_cache_used_pct"] *= 100.0
        return out

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
