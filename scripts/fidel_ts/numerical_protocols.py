"""Independent N1/N2 numerical preparation, masks and window accounting.

No PyTorch Dataset, model, fitting on validation/test, or label imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.fidel_ts.common import AuditError, write_json
from scripts.fidel_ts.numerical_diagnostics import (
    duplicate_analysis, iso, resolved_rows, segment_ids, split_ids,
)
from scripts.fidel_ts.stage1_5_io import SPLITS

PROTOCOL_DIRS = {"n1": "n1_official_compatible", "n2": "n2_strict_contiguous"}


class ProtocolBlocked(AuditError):
    pass


def fit_masked_scaler(values, valid, channels):
    counts = valid.sum(axis=0).astype(np.int64)
    if (counts == 0).any():
        raise ProtocolBlocked("No valid native training samples for: " + ", ".join(
            name for name, count in zip(channels, counts) if count == 0))
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=0) / counts
    centered = np.where(valid, safe - mean, 0.0)
    variance = (centered ** 2).sum(axis=0) / counts
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ProtocolBlocked("Non-finite masked scaler statistics")
    return {"channels": channels, "mean": mean.tolist(), "variance": variance.tolist(),
            "scale": scale.tolist(), "fit_sample_counts": counts.tolist(), "ddof": 0,
            "fit_scope": "native_training_after_duplicate_resolution_valid_cells_only",
            "constant_channel_scale": 1.0}


def causal_fill_history(values, valid, training_mean):
    """Input-representation helper: the supplied slice must contain history only.

    Never call this on labels. No carry from validation/test or later timesteps
    is searched. The original validity mask is returned unchanged.
    """
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise AuditError("History and x_mask must have the same [time,channel] shape")
    if not np.isfinite(values[valid]).all():
        raise AuditError("x_mask marks non-finite observations as valid")
    last = np.asarray(training_mean, dtype=float).copy()
    if last.shape != (values.shape[1],) or not np.isfinite(last).all():
        raise AuditError("A finite native training mean is required for every channel")
    output = np.empty_like(values)
    for i in range(len(values)):
        last[valid[i]] = values[i, valid[i]]
        output[i] = last
    return output, valid.copy()


def masked_error_sums(prediction, truth, y_mask):
    """Reference mask semantics for future losses/metrics, not a training loop."""
    prediction, truth, mask = np.asarray(prediction), np.asarray(truth), np.asarray(y_mask, dtype=bool)
    if prediction.shape != truth.shape or mask.shape != truth.shape:
        raise AuditError("MIMO predictions, labels and mask must have identical shapes")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(truth[mask]).all():
        raise AuditError("Non-finite values in valid metric positions")
    delta = prediction[mask] - truth[mask]
    return {"squared_error_sum": float(np.square(delta).sum()),
            "absolute_error_sum": float(np.abs(delta).sum()), "valid_count": int(mask.sum())}


def build_protocol(data, protocol, config):
    if protocol not in PROTOCOL_DIRS:
        raise AuditError("Protocol must be n1 or n2")
    if data.times.isna().any():
        raise ProtocolBlocked("Invalid timestamps cannot enter either frozen protocol")
    original_splits = split_ids(data.times, config)
    train_original = np.flatnonzero(original_splits == 0)
    if not len(train_original):
        raise ProtocolBlocked("Empty native training split")
    native_segments = segment_ids(data.ns, data.expected_seconds)
    factor = config["hour_seconds"] // data.expected_seconds
    if factor * data.expected_seconds != config["hour_seconds"]:
        raise ProtocolBlocked("Native frequency does not divide the frozen hour")
    duplicates = duplicate_analysis(data, config)
    anchor = None
    if protocol == "n1":
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise AuditError("N1 needs scikit-learn; see requirements-stage1_5.txt") from exc
        # The audited official loader casts to float32 BEFORE fit/transform.
        published = data.values.astype(np.float32)
        estimator = StandardScaler()
        try:
            estimator.fit(published[train_original])
        except ValueError as exc:
            raise ProtocolBlocked(f"Official StandardScaler rejects native training data: {exc}") from exc
        if not np.isfinite(estimator.mean_).all() or not np.isfinite(estimator.scale_).all():
            raise ProtocolBlocked("Official scaler has an undefined training statistic; not silently cleaned")
        # Transform native rows before sampling, as the official loader does.
        # In particular an Inf on an unsampled validation/test row must still
        # fail instead of disappearing through the hourly selection.
        try:
            official_scaled = estimator.transform(published).astype(np.float32)
        except ValueError as exc:
            raise ProtocolBlocked(f"Official transform rejects native data: {exc}") from exc
        counts = np.broadcast_to(np.asarray(estimator.n_samples_seen_), (len(data.channels),))
        scaler = {"channels": data.channels, "mean": estimator.mean_.tolist(),
                  "variance": estimator.var_.tolist(), "scale": estimator.scale_.tolist(),
                  "fit_sample_counts": counts.astype(np.int64).tolist(), "ddof": 0,
                  "fit_count_semantics": "official finite values including -9999; NaNs ignored by sklearn",
                  "scientifically_valid_native_training_counts": data.valid[train_original].sum(axis=0).tolist(),
                  "fit_scope": "official_native_train_float32_before_split_local_row_subsampling",
                  "class": "sklearn.preprocessing.StandardScaler"}
        selected = {name: np.flatnonzero(original_splits == s)[::factor] for s, name in enumerate(SPLITS)}
        scientific_valid = data.valid
    else:
        retained = resolved_rows(data, duplicates)
        ambiguous_ns = np.array(sorted(pd.Timestamp(group["timestamp"]).value for group in duplicates["groups"]
                                        if not group["all_channels_exact_equal"]), dtype=np.int64)
        after_resolution_gaps = []
        for i in np.flatnonzero(np.diff(data.ns[retained]) != data.expected_seconds * 10**9) + 1:
            left, right = int(data.ns[retained[i - 1]]), int(data.ns[retained[i]])
            excluded = ambiguous_ns[(ambiguous_ns > left) & (ambiguous_ns < right)]
            delta = right - left
            expected = data.expected_seconds * 10**9
            after_resolution_gaps.append({"left": iso(data.times[retained[i - 1]]), "right": iso(data.times[retained[i]]),
                                          "seconds": delta / 10**9,
                                          "theoretical_missing_points": delta // expected - 1 if delta > expected and delta % expected == 0 else None,
                                          "new_or_widened_by_ambiguous_exclusion": bool(len(excluded)),
                                          "excluded_ambiguous_timestamps": [pd.Timestamp(int(value)).isoformat() for value in excluded]})
        duplicates["n2_after_resolution_gaps"] = after_resolution_gaps
        retained_splits = split_ids(data.times[retained], config)
        training = retained[retained_splits == 0]
        if not len(training):
            raise ProtocolBlocked("No retained training timestamp")
        scaler = fit_masked_scaler(data.values[training], data.valid[training], data.channels)
        candidates = training[data.valid[training].any(axis=1)]
        if not len(candidates):
            raise ProtocolBlocked("No valid training observation can define the hourly phase")
        anchor_ns = int(data.ns[candidates[0]])
        hour_ns = config["hour_seconds"] * 10**9
        anchor = {"timestamp": iso(data.times[candidates[0]]), "original_row": int(candidates[0]),
                  "phase_ns_mod_hour": anchor_ns % hour_ns,
                  "selection": "earliest_retained_training_time_with_at_least_one_valid_channel",
                  "fixed_across_splits": True}
        at_phase = retained[(data.ns[retained] - anchor_ns) % hour_ns == 0]
        selected = {name: at_phase[original_splits[at_phase] == s] for s, name in enumerate(SPLITS)}
        strict_segments = segment_ids(data.ns[retained], data.expected_seconds)
        native_segments = np.full(len(data.times), -1, dtype=np.int64)
        native_segments[retained] = strict_segments
        scientific_valid = data.valid
        published = np.where(scientific_valid, data.values, np.nan)
    splits = {}
    for name, rows in selected.items():
        if protocol == "n1":
            scaled = official_scaled[rows]
        else:
            scaled = (published[rows] - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
        splits[name] = {"times": data.times[rows], "values": published[rows], "scaled": scaled,
                        "valid": scientific_valid[rows], "source_rows": rows,
                        "native_segment": native_segments[rows]}
    return {"protocol": protocol, "channels": data.channels, "scaler": scaler, "splits": splits,
            "hour_anchor": anchor, "factor": factor,
            "duplicates": duplicates, "native_training_rows_used": len(train_original) if protocol == "n1" else len(training),
            "official_execution_label_policy": "unmasked_published_values_including_finite_sentinels" if protocol == "n1" else None,
            "scientific_mask_policy": "Jena -9999 and all nonfinite values invalid; N1 reports mask without changing official values"}


def window_accounting(times, valid, native_segment, lookback, horizon, strict):
    n, channels = valid.shape
    count = max(0, n - lookback - horizon + 1)
    starts = np.arange(count, dtype=np.int64)
    end = starts + lookback + horizon
    y_start = starts + lookback
    ns = times.to_numpy(dtype="datetime64[ns]").view("i8")
    bad = (np.diff(ns) != 3600 * 10**9) | (np.diff(native_segment) != 0)
    prefix_bad = np.r_[0, np.cumsum(bad, dtype=np.int64)]
    continuous = (prefix_bad[end - 1] - prefix_bad[starts] == 0) if count else np.array([], dtype=bool)
    prefix_valid = np.vstack([np.zeros((1, channels), dtype=np.int64), np.cumsum(valid, axis=0, dtype=np.int64)])
    label_counts = prefix_valid[end] - prefix_valid[y_start]
    any_label = label_counts.sum(axis=1) > 0
    full = (label_counts == horizon).all(axis=1)
    eligible = continuous & any_label if strict else np.ones(count, dtype=bool)
    summary = {"lookback": lookback, "horizon": horizon, "theoretical_windows": count,
               "continuous_windows": int(continuous.sum()),
               "full_valid_label_windows_among_continuous": int((continuous & full).sum()),
               "masked_label_eligible_windows": int((continuous & any_label).sum()),
               "published_window_indices": int(eligible.sum()),
               "valid_label_occurrences_by_channel": label_counts[continuous & any_label].sum(axis=0).tolist(),
               "valid_label_occurrences_total": int(label_counts[continuous & any_label].sum()),
               "label_count_interpretation": "overlapping window-cell occurrences, not independent observations",
               "official_windows_filtered_for_quality": False if not strict else None,
               "masked_label_rule": "all channels share time windows; individual invalid y cells are excluded"}
    selected = starts[eligible]
    indices = pd.DataFrame({"input_start_row": selected, "label_start_row": selected + lookback,
                            "label_end_row_exclusive": selected + lookback + horizon,
                            "forecast_origin": times[selected + lookback] if len(selected) else pd.DatetimeIndex([]),
                            "last_history_time": times[selected + lookback - 1] if len(selected) else pd.DatetimeIndex([]),
                            "continuous": continuous[eligible], "full_valid_labels": full[eligible]})
    return summary, indices


def protocol_report(result, config):
    report, windows, indices = {}, {}, {}
    for name, split in result["splits"].items():
        times, valid = split["times"], split["valid"]
        ns = times.to_numpy(dtype="datetime64[ns]").view("i8")
        report[name] = {"points": len(times), "time_start": iso(times.min()), "time_end": iso(times.max()),
                        "phase_ns_mod_hour_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(ns % (3600 * 10**9), return_counts=True))},
                        "duplicate_timestamp_extras": int(times.duplicated().sum()),
                        "hour_interval_anomalies": int((np.diff(ns) != 3600 * 10**9).sum()),
                        "native_segment_transitions": int((np.diff(split["native_segment"]) != 0).sum()),
                        "invalid_cells_by_channel": (~valid).sum(axis=0).tolist(),
                        "invalid_label_cell_ratio": float((~valid).mean()) if valid.size else None,
                        "scientifically_valid_cells_by_channel": valid.sum(axis=0).tolist()}
        windows[name] = {}
        for horizon in config["horizons"]:
            info, index = window_accounting(times, valid, split["native_segment"], config["lookback"],
                                            horizon, result["protocol"] == "n2")
            windows[name][str(horizon)] = {**info, "channels": result["channels"]}
            indices[(name, horizon)] = index
    blocked = []
    if result["protocol"] == "n2":
        for name in SPLITS:
            for horizon in config["horizons"]:
                if windows[name][str(horizon)]["published_window_indices"] < config["n2"]["minimum_eligible_windows_per_split_horizon"]:
                    blocked.append(f"{name}/H{horizon}: insufficient continuous windows with at least one valid label")
    return {"protocol": result["protocol"], "channels": result["channels"], "splits": report,
            "hour_anchor": result["hour_anchor"], "scaler_fit_counts": result["scaler"]["fit_sample_counts"],
            "upstream_future_interpolation_risk_eliminated": False,
            "blocked_reasons": blocked}, windows, indices


def write_prepared(output, result, config, binding, prepare=True):
    quality, windows, indices = protocol_report(result, config)
    write_json(output / "manifest.json", {**binding, "protocol": result["protocol"],
                                          "mode": "prepare" if prepare else "report-only",
                                          "fixed_split_boundaries": config["split_boundaries"],
                                          "borrow_previous_split_history": False,
                                          "forecast_origin_definition": "first predicted timestamp; history is strictly earlier in N2",
                                          "hour_anchor": result["hour_anchor"],
                                          "official_execution_label_policy": result["official_execution_label_policy"],
                                          "scientific_mask_policy": result["scientific_mask_policy"]})
    write_json(output / "channels.json", {"channels": result["channels"], "order": "fixed_source_configuration"})
    write_json(output / "splits.json", {name: {"start_inclusive": None if i == 0 else config["split_boundaries"][i - 1],
                                               "end_exclusive": config["split_boundaries"][i] if i < 2 else None}
                                        for i, name in enumerate(SPLITS)})
    write_json(output / "scaler.json", result["scaler"])
    write_json(output / "quality_summary.json", quality)
    write_json(output / "window_eligibility.json", windows)
    write_json(output / "duplicate_resolution.json", result["duplicates"])
    if prepare:
        for name, split in result["splits"].items():
            directory = output / name
            directory.mkdir()
            pd.DataFrame(split["values"], columns=result["channels"]).to_parquet(directory / "values.parquet", index=False)
            pd.DataFrame(split["scaled"], columns=result["channels"]).to_parquet(directory / "scaled.parquet", index=False)
            pd.DataFrame(split["valid"], columns=result["channels"]).to_parquet(directory / "validity_mask.parquet", index=False)
            pd.DataFrame({"timestamp": split["times"], "original_row": split["source_rows"],
                          "native_segment_id": split["native_segment"]}).to_parquet(directory / "timestamps.parquet", index=False)
        (output / "window_indices").mkdir()
        for (split, horizon), index in indices.items():
            index.to_parquet(output / "window_indices" / f"{split}_H{horizon}.parquet", index=False)
    return quality["blocked_reasons"]
