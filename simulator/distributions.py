"""Sampling distributions used by personas."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


class Distribution:
    """Base class for samplers. Subclasses implement sample()."""

    def sample(self, rng: random.Random) -> float:
        raise NotImplementedError

    def sample_int(self, rng: random.Random) -> int:
        return max(1, int(round(self.sample(rng))))


@dataclass
class LogNormal(Distribution):
    """Log-normal: parameterised by mean of underlying normal and sigma.

    Use ``LogNormal.from_median(median, sigma)`` for an intuitive constructor —
    the median of a log-normal equals exp(mu).
    """

    mu: float
    sigma: float
    min_value: float = 0.0
    max_value: float = float("inf")

    @classmethod
    def from_median(cls, median: float, sigma: float, **kwargs) -> "LogNormal":
        return cls(mu=math.log(median), sigma=sigma, **kwargs)

    def sample(self, rng: random.Random) -> float:
        v = rng.lognormvariate(self.mu, self.sigma)
        return max(self.min_value, min(self.max_value, v))


@dataclass
class Discrete(Distribution):
    """Discrete weighted choice over numeric values."""

    weights: dict  # {value: weight}

    def sample(self, rng: random.Random) -> float:
        items = list(self.weights.items())
        values = [v for v, _ in items]
        weights = [w for _, w in items]
        return float(rng.choices(values, weights=weights, k=1)[0])


@dataclass
class Constant(Distribution):
    value: float

    def sample(self, rng: random.Random) -> float:
        return self.value


def summarize(dist: Distribution) -> dict:
    """Analytic {median, mean, p90} for a distribution — the numbers a
    HUMAN needs to understand what a persona means ("~400-token
    questions, occasionally 1200") without reading YAML or sampling.

    LogNormal: median = e^mu, mean = e^(mu+sigma^2/2),
    p90 = e^(mu + 1.2816*sigma) (clamped to the distribution's caps).
    Discrete: weighted median / mean / p90 over the pmf.
    """
    if isinstance(dist, LogNormal):
        clamp = lambda v: max(dist.min_value, min(dist.max_value, v))  # noqa: E731
        return {
            "median": clamp(math.exp(dist.mu)),
            "mean": clamp(math.exp(dist.mu + dist.sigma ** 2 / 2)),
            "p90": clamp(math.exp(dist.mu + 1.2816 * dist.sigma)),
        }
    if isinstance(dist, Discrete):
        items = sorted(dist.weights.items())
        total = sum(w for _, w in items) or 1.0
        mean = sum(v * w for v, w in items) / total
        def _quantile(q: float) -> float:
            acc = 0.0
            for v, w in items:
                acc += w / total
                if acc >= q - 1e-9:      # float-sum tolerance
                    return float(v)
            return float(items[-1][0])
        return {"median": _quantile(0.5), "mean": mean,
                "p90": _quantile(0.9)}
    if isinstance(dist, Constant):
        return {"median": dist.value, "mean": dist.value,
                "p90": dist.value}
    return {"median": None, "mean": None, "p90": None}
