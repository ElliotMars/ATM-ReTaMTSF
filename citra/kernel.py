"""RBF kernel and an independent full-sum oracle (never semantic Top-K)."""

from dataclasses import dataclass
import math
import numpy as np


def vector(value, size=None):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or (size is not None and arr.size != size):
        raise ValueError("expected a vector of the configured dimension")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vectors must be finite")
    return arr


def immutable(value):
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite arithmetic; rescale inputs")
    # Backed by immutable bytes: callers cannot re-enable writes.
    return np.frombuffer(arr.tobytes(), dtype=np.float64).reshape(arr.shape)


@dataclass(frozen=True)
class RBFKernel:
    sigma: float = 1.0

    def __post_init__(self):
        squared = self.sigma * self.sigma
        if (not math.isfinite(self.sigma) or self.sigma <= 0
                or not math.isfinite(squared) or squared <= 0
                or not math.isfinite(1.0 / squared)):
            raise ValueError("sigma must have a finite positive squared inverse")

    @property
    def hessian_bound(self):
        """Global spectral norm bound, valid along every Taylor segment."""
        return 1.0 / (self.sigma * self.sigma)

    def value(self, query, embedding):
        q = vector(query)
        z = vector(embedding, q.size)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = (q - z) / self.sigma
            distance = float(delta @ delta)
        return math.exp(-0.5 * distance)

    def gradient_z(self, query, embedding):
        q = vector(query)
        z = vector(embedding, q.size)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return immutable(self.value(q, z) * ((q - z) / self.sigma) / self.sigma)


def exact_aggregate(documents, query, kernel, impact_dim):
    """Full legal-document sum with compensated componentwise reduction."""
    q = vector(query)
    rows = [kernel.value(q, d.embedding) * vector(d.impact_vector, impact_dim)
            for d in documents]
    return immutable([math.fsum(float(row[j]) for row in rows)
                      for j in range(impact_dim)])
