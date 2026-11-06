import numpy as np
import torch
from scipy.stats import rankdata


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask shapes differ")
    if not mask.any():
        return prediction.sum() * 0
    selected = prediction[mask].float() - target[mask].float()
    if not torch.isfinite(selected).all():
        raise ValueError("Nonfinite prediction/valid target")
    return selected.square().mean()


def metrics(prediction, target, mask, stocknet=False):
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask, dtype=bool)
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("NaN/Inf in predictions or valid labels")
    delta = prediction[mask] - target[mask]
    result = {"mse": float(np.mean(delta**2)) if len(delta) else None,
              "mae": float(np.mean(np.abs(delta))) if len(delta) else None,
              "valid_label_count": int(mask.sum()), "label_coverage": float(mask.mean()), "channels": []}
    for channel in range(target.shape[-1]):
        valid = mask[..., channel]
        error = prediction[..., channel][valid] - target[..., channel][valid]
        result["channels"].append({"channel": channel, "mse": float(np.mean(error**2)) if len(error) else None,
                                   "mae": float(np.mean(abs(error))) if len(error) else None, "valid_count": int(valid.sum())})
    if stocknet:
        result["directional_accuracy"] = float((np.sign(prediction[mask]) == np.sign(target[mask])).mean()) if mask.any() else None
        correlations = []
        for p, y, valid in zip(prediction.reshape(-1, prediction.shape[-1]), target.reshape(-1, target.shape[-1]), mask.reshape(-1, mask.shape[-1])):
            if valid.sum() >= 5 and np.std(p[valid]) > 0 and np.std(y[valid]) > 0:
                correlations.append(float(np.corrcoef(rankdata(p[valid]), rankdata(y[valid]))[0, 1]))
        result["rank_ic"] = float(np.mean(correlations)) if correlations else None
        result["rank_ic_cross_sections"] = len(correlations)
        result["not_a_trading_return"] = True
    return result


def block_bootstrap(values, block_size, seed=2024, draws=1000):
    """Resample contiguous ORIGIN blocks; caller groups/averages horizons first."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    block_size = max(1, min(int(block_size), len(values)))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        starts = rng.integers(0, len(values) - block_size + 1, size=int(np.ceil(len(values) / block_size)))
        sampled = np.concatenate([values[start:start + block_size] for start in starts])[:len(values)]
        estimates.append(float(sampled.mean()))
    return {"mean": float(values.mean()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "block_size_origins": block_size, "draws": draws, "seed": seed}


class Accumulator:
    def __init__(self, channels, stocknet=False):
        self.squared, self.absolute, self.counts = np.zeros(channels), np.zeros(channels), np.zeros(channels, dtype=np.int64)
        self.total_cells, self.direction, self.rank_ics = 0, 0, []
        self.stocknet = stocknet
        self.origin_mse = []

    def update(self, prediction, truth, mask):
        prediction, truth = prediction.astype(np.float64), truth.astype(np.float64)
        if not np.isfinite(prediction).all() or not np.isfinite(truth[mask]).all():
            raise ValueError("Nonfinite valid evaluation values")
        error = np.where(mask, prediction - truth, 0)
        self.squared += (error**2).sum((0, 1))
        self.absolute += abs(error).sum((0, 1))
        self.counts += mask.sum((0, 1))
        self.total_cells += mask.size
        for row, valid in zip(error, mask):
            self.origin_mse.append(float((row**2).sum() / max(1, valid.sum())))
        if self.stocknet:
            self.direction += int((np.sign(prediction[mask]) == np.sign(truth[mask])).sum())
            for p, y, valid in zip(prediction.reshape(-1, prediction.shape[-1]), truth.reshape(-1, truth.shape[-1]), mask.reshape(-1, mask.shape[-1])):
                if valid.sum() >= 5 and np.std(p[valid]) > 0 and np.std(y[valid]) > 0:
                    self.rank_ics.append(float(np.corrcoef(rankdata(p[valid]), rankdata(y[valid]))[0, 1]))

    def result(self):
        total = int(self.counts.sum())
        out = {"mse": float(self.squared.sum() / total) if total else None,
               "mae": float(self.absolute.sum() / total) if total else None, "valid_label_count": total,
               "label_coverage": total / self.total_cells if self.total_cells else None,
               "channels": [{"channel": i, "valid_count": int(n), "mse": float(self.squared[i] / n) if n else None,
                             "mae": float(self.absolute[i] / n) if n else None} for i, n in enumerate(self.counts)]}
        if self.stocknet:
            out.update(directional_accuracy=self.direction / total if total else None,
                       rank_ic=float(np.mean(self.rank_ics)) if self.rank_ics else None,
                       rank_ic_cross_sections=len(self.rank_ics), not_a_trading_return=True)
        return out
