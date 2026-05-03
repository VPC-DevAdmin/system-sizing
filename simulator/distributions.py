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
