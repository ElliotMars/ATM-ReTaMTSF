"""Native-frequency diagnostics and explicit duplicate resolution; no I/O writes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from scripts.fidel_ts.common import AuditError
from scripts.fidel_ts.stage1_5_io import SPLITS


@dataclass
class NativeData:
    times: pd.DatetimeIndex
    values: np.ndarray
    arrow_null: np.ndarray
    channels: list
    dataset: str
    expected_seconds: int

    @cached_property
    def ns(self):
        return self.times.to_numpy(dtype="datetime64[ns]").view("i8")

    @cached_property
    def valid(self):
        valid = np.isfinite(self.values) & ~self.arrow_null
        if self.dataset == "jena":
            valid &= self.values != -9999
        return valid


def native_from_frame(frame, dataset, spec, arrow_null=None):
    column = spec["time_column"]
    channels = list(spec["expected_channels"])
    actual = [name for name in frame.columns if name != column]
    if column not in frame or actual != channels:
        raise AuditError("All configured 20/21 channels are required; no target/channel selection is allowed")
    # Keep original physical row order. Selection of a named column order is explicit.
    times = pd.DatetimeIndex(pd.to_datetime(frame[column], errors="coerce"))
    if times.tz is not None:
        raise AuditError("Frozen boundaries are timezone-naive; an aware source requires an explicit reviewed protocol")
    values = frame[channels].to_numpy(dtype=np.float64, na_value=np.nan)
    nulls = np.zeros(values.shape, dtype=bool) if arrow_null is None else arrow_null
    return NativeData(times, values, nulls, channels, dataset, spec["expected_interval_seconds"])


def load_native(path, dataset, spec):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    nulls = np.column_stack([table.column(name).is_null().to_numpy() for name in spec["expected_channels"]])
    return native_from_frame(table.to_pandas(), dataset, spec, nulls)


def split_ids(times, config):
    boundaries = [pd.Timestamp(value) for value in config["split_boundaries"]]
    result = np.full(len(times), -1, dtype=np.int8)
    good = ~times.isna()
    result[good & (times < boundaries[0])] = 0
    result[good & (times >= boundaries[0]) & (times < boundaries[1])] = 1
    result[good & (times >= boundaries[1])] = 2
    return result


def iso(value):
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()


def scalar(value):
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+Inf"
    if np.isneginf(value):
        return "-Inf"
    return float(value)


def duplicate_analysis(data, config):
    mask = data.times.duplicated(keep=False) & ~data.times.isna()
    ids = np.flatnonzero(mask)
    groups = []
    splits = split_ids(data.times, config)
    all_valid = data.valid
    if len(ids):
        by_time = pd.Series(ids, index=data.times[ids])
        for timestamp, group in by_time.groupby(level=0, sort=True):
            rows = group.to_numpy(dtype=np.int64)
            values = data.values[rows]
            valid = all_valid[rows]
            equal_columns = ((values == values[0]) | (np.isnan(values) & np.isnan(values[0]))).all(axis=0)
            conflicting = [name for j, name in enumerate(data.channels)
                           if len(np.unique(values[valid[:, j], j])) > 1]
            exact = bool(equal_columns.all())
            if exact:
                classification = "identical" if valid.all() else "identical_with_invalid"
            elif conflicting:
                classification = "conflicting"
            else:
                classification = "indeterminate_invalid"
            different = [name for name, same in zip(data.channels, equal_columns) if not same]
            examples = {name: [scalar(x) for x in values[:, data.channels.index(name)][:8]]
                        for name in different}
            groups.append({"timestamp": iso(timestamp), "original_rows": rows.tolist(),
                           "records": len(rows), "extra_records": len(rows) - 1,
                           "split": SPLITS[int(splits[rows[0]])],
                           "all_channels_exact_equal": exact,
                           "classification": classification, "conflicting_valid_channels": conflicting,
                           "different_raw_channels": different, "value_examples": examples,
                           "contains_invalid": bool((~valid).any()),
                           "contains_arrow_null": bool(data.arrow_null[rows].any()),
                           "contains_sentinel": bool((values == -9999).any()),
                           "contains_nonfinite": bool((~np.isfinite(values)).any()),
                           "n2_action": "retain_one_with_original_mask" if exact else "exclude_ambiguous_timestamp",
                           "unresolved_timezone_issue": True})
    return {"groups": groups, "duplicate_groups": len(groups),
            "extra_duplicate_records": sum(group["extra_records"] for group in groups),
            "equality_rule": "exact stored values, paired NaNs equal; invalid equality is structural, not verified observation agreement",
            "classification_counts": {name: sum(g["classification"] == name for g in groups)
                                      for name in ("identical", "identical_with_invalid", "conflicting", "indeterminate_invalid")},
            "timezone_cause": "unknown; duplicates are not attributed to DST"}


def resolved_rows(data, duplicates):
    keep = ~data.times.isna()
    for group in duplicates["groups"]:
        rows = group["original_rows"]
        if group["all_channels_exact_equal"]:
            keep[rows[1:]] = False
        else:
            keep[rows] = False
    indices = np.flatnonzero(keep)
    return indices[np.argsort(data.ns[indices], kind="stable")]


def segment_ids(ns, expected_seconds):
    if not len(ns):
        return np.array([], dtype=np.int64)
    return np.cumsum(np.r_[False, np.diff(ns) != expected_seconds * 1_000_000_000]).astype(np.int64)


def time_gap_analysis(data, config):
    expected_ns = data.expected_seconds * 1_000_000_000
    splits = split_ids(data.times, config)
    boundaries = [pd.Timestamp(t).value for t in config["split_boundaries"]]

    def event(left, right, original_rows=None):
        delta_ns = int(right.value - left.value)
        delta = delta_ns / 1_000_000_000
        left_split = int(split_ids(pd.DatetimeIndex([left]), config)[0])
        right_split = int(split_ids(pd.DatetimeIndex([right]), config)[0])
        aligned = delta_ns > 0 and delta_ns % expected_ns == 0
        span_lo, span_hi = sorted((left.value, right.value))
        return {"left": iso(left), "right": iso(right), "seconds": delta,
                "kind": ("clock_rollback" if delta < 0 else "duplicate" if delta == 0 else
                         "gap" if delta > data.expected_seconds else "short_offgrid_interval"),
                "theoretical_missing_points": int(delta_ns // expected_ns - 1) if aligned else None,
                "original_rows": original_rows,
                "splits": [SPLITS[left_split], SPLITS[right_split]],
                "crosses_split_boundary": left_split != right_split,
                "near_split_boundary": any(span_lo - config["boundary_near_seconds"] * 10**9 <= b
                                             <= span_hi + config["boundary_near_seconds"] * 10**9 for b in boundaries),
                "cause": "unknown_not_attributed_to_timezone_or_DST"}

    original = []
    for i in np.flatnonzero(np.diff(data.ns) != expected_ns) + 1:
        if not pd.isna(data.times[i - 1]) and not pd.isna(data.times[i]):
            original.append(event(data.times[i - 1], data.times[i], [int(i - 1), int(i)]))
    good = np.flatnonzero(~data.times.isna())
    ordered = good[np.argsort(data.ns[good], kind="stable")]
    sorted_events = [event(data.times[ordered[i - 1]], data.times[ordered[i]],
                           [int(ordered[i - 1]), int(ordered[i])])
                     for i in np.flatnonzero(np.diff(data.ns[ordered]) != expected_ns) + 1]
    unique = data.times[ordered].drop_duplicates()
    unique_ns = unique.to_numpy(dtype="datetime64[ns]").view("i8")
    unique_anomalies = [event(unique[i - 1], unique[i]) for i in np.flatnonzero(np.diff(unique_ns) != expected_ns) + 1]
    gaps = [item for item in unique_anomalies if item["seconds"] > data.expected_seconds]

    def distribution(ns):
        values, counts = np.unique(np.diff(ns), return_counts=True)
        return {str(int(value) / 10**9): int(count) for value, count in zip(values, counts)}

    original_delta = np.diff(data.ns)
    valid_pairs = ~data.times.isna()[:-1] & ~data.times.isna()[1:]
    delta_values, delta_counts = np.unique(original_delta[valid_pairs], return_counts=True)
    return {"original_order_anomalies": original, "stable_sorted_anomalies": sorted_events,
            "unique_timestamp_gaps": gaps, "unique_timestamp_anomalies": unique_anomalies,
            "original_irregular_count": len(original),
            "clock_rollbacks": sum(e["seconds"] < 0 for e in original),
            "original_interval_counts_seconds": {str(int(v) / 10**9): int(n) for v, n in zip(delta_values, delta_counts)},
            "stable_sorted_interval_counts_seconds": distribution(data.ns[ordered]),
            "unique_interval_counts_seconds": distribution(unique_ns),
            "unique_gap_count": len(gaps), "unassigned_invalid_time_rows": int((splits < 0).sum())}


def invalid_intervals(data, invalid, rows):
    positions = rows[invalid[rows]]
    if not len(positions):
        return []
    breaks = np.r_[True, (np.diff(positions) != 1)
                    | (np.diff(data.ns[positions]) != data.expected_seconds * 10**9)]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(positions)]
    return [{"first_original_row": int(positions[a]), "last_original_row": int(positions[b - 1]),
             "start": iso(data.times[positions[a]]), "end": iso(data.times[positions[b - 1]]),
             "points": int(b - a)} for a, b in zip(starts, ends)]


def missing_value_analysis(data, config):
    ids = split_ids(data.times, config)
    valid = data.valid
    output = {}
    for s, name in enumerate(SPLITS):
        rows = np.flatnonzero(ids == s)
        channels = []
        for j, channel in enumerate(data.channels):
            values = data.values[rows, j]
            usable = values[valid[rows, j]]
            channels.append({"channel": channel, "arrow_null": int(data.arrow_null[rows, j].sum()),
                             "nan_including_arrow_null": int(np.isnan(values).sum()),
                             "inf": int(np.isinf(values).sum()), "minus_9999": int((values == -9999).sum()),
                             "valid_observations": int(valid[rows, j].sum()),
                             "valid_negative_observations": int((usable < 0).sum()),
                             "valid_min": float(usable.min()) if len(usable) else None,
                             "valid_max": float(usable.max()) if len(usable) else None,
                             "invalid_intervals": invalid_intervals(data, ~valid[:, j], rows)})
        simultaneous = rows[(~valid[rows]).sum(axis=1) > 1]
        output[name] = {"rows": len(rows), "channels": channels,
                        "multi_channel_invalid_rows": [{"original_row": int(row), "timestamp": iso(data.times[row]),
                                                         "channels": [c for c, ok in zip(data.channels, valid[row]) if not ok]}
                                                        for row in simultaneous]}
    return {"splits": output,
            "invalid_rule": "NaN/Inf/Arrow null; additionally -9999 for Jena; negative values otherwise retained",
            "category_overlap": "Arrow null is included in NaN count; invalid mask is the union, not category sum"}


def diagnose(data, config):
    duplicates = duplicate_analysis(data, config)
    gaps = time_gap_analysis(data, config)
    return {"numerical_diagnostics.json": {
                "rows": len(data.times), "channels": data.channels, "numeric_channel_count": len(data.channels),
                "time_start": iso(data.times.min()), "time_end": iso(data.times.max()),
                "invalid_timestamp_rows": np.flatnonzero(data.times.isna()).tolist(),
                "extra_duplicate_records": duplicates["extra_duplicate_records"],
                "duplicate_groups": duplicates["duplicate_groups"], "unique_gap_count": gaps["unique_gap_count"],
                "unresolved_timezone_issue": bool(duplicates["groups"] or gaps["clock_rollbacks"]),
                "timezone": None, "modifies_raw_data": False},
            "duplicate_analysis.json": duplicates, "time_gap_analysis.json": gaps,
            "missing_value_analysis.json": missing_value_analysis(data, config)}
