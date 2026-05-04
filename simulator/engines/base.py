"""Engine abstraction.

The simulator only ever talks to an OpenAI-compatible endpoint. Engine
implementations own their tuning recipe and metrics endpoint.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)


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
        log.info("Launching %s: %s", self.cfg.type, " ".join(cmd))
        log.info("Engine logs -> %s", log_path)

        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        self._wait_for_health(self.cfg.startup_timeout_s)

    def shutdown(self) -> None:
        if self._proc is None:
            return
        log.info("Shutting down %s engine (pid=%s)", self.cfg.type, self._proc.pid)
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.terminate()
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("Engine did not stop gracefully; killing")
            if os.name != "nt":
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            else:
                self._proc.kill()
        finally:
            if self._log_file is not None:
                self._log_file.close()
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

    @staticmethod
    def _parse_prometheus(text: str) -> dict[str, float]:
        """Extract a small set of metric values from Prometheus exposition.

        Looks for vLLM and SGLang names; returns a normalised dict using
        canonical keys: kv_cache_used_pct, queue_depth, prefix_cache_hits,
        prefix_cache_misses, num_running, num_waiting.
        """
        wanted = {
            "vllm:gpu_cache_usage_perc": "kv_cache_used_pct",
            "vllm:cpu_cache_usage_perc": "kv_cache_used_pct",
            "vllm:num_requests_running": "num_running",
            "vllm:num_requests_waiting": "queue_depth",
            "vllm:prefix_cache_hits_total": "prefix_cache_hits",
            "vllm:prefix_cache_queries_total": "prefix_cache_queries",
            # SGLang naming is in flux; best-effort matches.
            "sglang:num_running_reqs": "num_running",
            "sglang:num_waiting_reqs": "queue_depth",
            "sglang:cache_hit_rate": "prefix_cache_hit_rate",
            "sglang:token_usage": "kv_cache_used_pct",
        }
        out: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            # name{labels} value  OR  name value
            try:
                name_part, _, value_str = line.rpartition(" ")
                name = name_part.split("{", 1)[0].strip()
                if name in wanted:
                    out[wanted[name]] = float(value_str)
            except Exception:
                continue
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
