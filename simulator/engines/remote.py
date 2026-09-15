"""Remote-endpoint target (roadmap 1.1).

Benchmarks an OpenAI-compatible endpoint the simulator does NOT launch
or own — a vLLM/SGLang box elsewhere, a gateway, or a hosted API.
``launch()`` only verifies the endpoint answers; ``shutdown()`` is a
no-op (never stop something we didn't start).

What changes versus a local engine:
  * No host telemetry — the runner passes ``host_telemetry=False`` so
    PMU/bandwidth/power/frequency/CPU/GPU collectors are skipped (they
    would measure the client box). Client-observed latencies and the
    endpoint's own /metrics (when ``endpoint_metrics_url`` is set)
    remain, and the export's ``collectors`` block records the skips —
    bottleneck attribution degrades to prefill/decode classification.
  * ``base_url`` comes from ``engine.endpoint_url`` verbatim (include
    the ``/v1`` suffix); auth from ``engine.endpoint_api_key``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from .base import Engine

log = logging.getLogger(__name__)


class RemoteEngine(Engine):
    """Attach-only engine for endpoints owned by someone else."""

    def launch(self, log_dir: str | Path = "runs") -> None:
        """No process to start — just gate on the endpoint answering."""
        start = time.time()
        deadline = start + min(self.cfg.startup_timeout_s, 60)
        last_err: Optional[str] = None
        while time.time() < deadline:
            if self.health_check():
                log.info(
                    "Remote endpoint %s answering after %.1fs",
                    self.base_url, time.time() - start,
                )
                return
            last_err = "no 200 from /models yet"
            time.sleep(2.0)
        raise RuntimeError(
            f"Remote endpoint {self.base_url} not answering ({last_err}). "
            f"Check engine.endpoint_url (must include /v1) and network."
        )

    def shutdown(self) -> None:
        """Never stop an endpoint we didn't start."""

    @property
    def pid(self) -> Optional[int]:
        return None

    def health_check(self) -> bool:
        try:
            headers = {}
            if self.cfg.endpoint_api_key:
                headers["Authorization"] = f"Bearer {self.cfg.endpoint_api_key}"
            r = httpx.get(f"{self.base_url}/models", headers=headers, timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def get_metrics(self) -> dict[str, float]:
        """Scrape the endpoint's Prometheus metrics when a URL is
        configured — these describe the system under test, unlike the
        host collectors. Empty dict otherwise."""
        url = self.cfg.endpoint_metrics_url
        if not url:
            return {}
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code != 200:
                return {}
            return self._parse_prometheus(r.text)
        except Exception:
            return {}

    # Base abstract API — never used (launch is overridden).
    def _build_command(self) -> list[str]:
        raise NotImplementedError("RemoteEngine does not launch a process")

    def _build_env(self) -> dict[str, str]:
        raise NotImplementedError("RemoteEngine does not launch a process")
