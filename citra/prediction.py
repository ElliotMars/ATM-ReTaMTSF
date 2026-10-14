"""Fixed NumPy linear forecaster for output-certificate correctness tests.

This is not the trainable PyTorch integration planned for Milestone 3.
"""

import math
import numpy as np

from .kernel import immutable


class FixedLinearForecaster:
    def __init__(self, seq_len, channels, horizon, impact_dim, seed=0,
                 max_lipschitz=1.0):
        if any(not isinstance(x, int) or x <= 0
               for x in (seq_len, channels, horizon, impact_dim)):
            raise ValueError("model dimensions must be positive integers")
        if not math.isfinite(max_lipschitz) or max_lipschitz < 0:
            raise ValueError("invalid Lipschitz limit")
        self.seq_len, self.channels, self.horizon = seq_len, channels, horizon
        self.impact_dim = impact_dim
        rng = np.random.default_rng(seed)
        self.backbone = immutable(rng.normal(size=(horizon, seq_len)) / seq_len)
        weight = rng.normal(size=(channels * horizon, impact_dim))
        weight *= max_lipschitz / max(float(np.linalg.norm(weight)), 1.0)
        self.weight = immutable(weight)

    @property
    def lipschitz_bound(self):
        # Frobenius >= spectral norm; fixed weights cannot change after query.
        return float(np.linalg.norm(self.weight))

    def predict(self, history, aggregate):
        x = np.asarray(history, dtype=np.float64)
        a = np.asarray(aggregate, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.seq_len):
            raise ValueError("history shape must be [B,C,L]")
        if a.shape != (x.shape[0], self.impact_dim):
            raise ValueError("aggregate shape must be [B,m]")
        if not np.isfinite(x).all() or not np.isfinite(a).all():
            raise ValueError("prediction inputs must be finite")
        numerical = x @ self.backbone.T
        correction = (a @ self.weight.T).reshape(x.shape[0], self.channels, self.horizon)
        return numerical + correction
