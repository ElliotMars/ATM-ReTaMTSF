"""Read-only numerical/text diagnostics. No cleaning, resampling or retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

from scripts.fidel_ts.common import AuditError, data_path, json_digest


class JSONObject(list):
    """Preserve JSON object pairs, including duplicate keys, for auditing."""


def load_pairs(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=JSONObject)


def channel_mapping(columns, static, spec):
    info = static.get("channel_info", {})
    candidates = [key for key in spec["static_subject_candidates"] if isinstance(info.get(key), dict)]
    subject = candidates[0] if candidates else None
    descriptions = info.get(subject, {})
    aliases = spec.get("channel_aliases", {})
    mapping, used = [], Counter()
    for raw in columns:
        name = raw if raw in descriptions else aliases.get(raw, raw)
        exists = isinstance(descriptions.get(name), str) and bool(descriptions[name].strip())
        if exists:
            used[name] += 1
        mapping.append({"raw_channel_id": raw, "description_channel_id": name if exists else None,
                        "description_subject_id": subject,
                        "match": ("exact" if name == raw else "explicit_alias") if exists else "missing",
                        "description": descriptions.get(name) if exists else None})
    return {"subject_id": spec["subject_id"], "static_subject_selected": subject,
            "static_subject_candidates_found": candidates,
            "mapping": mapping, "unmapped_channels": [m["raw_channel_id"] for m in mapping if m["match"] == "missing"],
            "unused_descriptions": sorted(set(descriptions) - set(used)),
            "ambiguous_description_targets": sorted(key for key, count in used.items() if count > 1)}


def numerical_summary(path, static, spec):
    """Full local-file scan, invoked only by the user's inspect command."""
    try:
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AuditError("Numerical inspection requires numpy, pandas and pyarrow") from exc
    table = pq.read_table(path)
    frame = table.to_pandas()
    column = spec["time_column"]
    if column not in frame or frame.columns.duplicated().any():
        raise AuditError(f"Missing time column or duplicate column names in {path}")
    raw_time = frame[column]
    try:
        timestamps = pd.to_datetime(raw_time, errors="coerce")
        ordered = timestamps.dropna().sort_values().drop_duplicates()
        source_deltas = timestamps.diff().dt.total_seconds()
        sorted_deltas = ordered.diff().dt.total_seconds().dropna()
    except (ValueError, TypeError, AttributeError) as exc:
        raise AuditError(f"Cannot interpret time axis without a timezone/format assumption: {exc}") from exc
    expected = spec["expected_interval_seconds"]
    valid = timestamps.dropna()
    numeric = [name for name in frame.columns if name != column and pd.api.types.is_numeric_dtype(frame[name])]
    channels = []
    for name in numeric:
        values = frame[name].to_numpy(dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        nulls = table.column(name).null_count
        nan_total = int(np.isnan(values).sum())
        sentinel = int((values == -9999).sum())
        channels.append({"name": name, "dtype": str(frame[name].dtype),
                         "arrow_null_count": int(nulls), "missing_including_nan": nan_total,
                         "ieee_nan_excluding_arrow_null": max(0, nan_total - nulls),
                         "positive_inf": int(np.isposinf(values).sum()),
                         "negative_inf": int(np.isneginf(values).sum()),
                         "sentinel_minus_9999": sentinel,
                         "sentinel_row_examples": np.flatnonzero(values == -9999)[:20].tolist(),
                         "negative_finite_values": int((finite < 0).sum()),
                         "negative_finite_row_examples": np.flatnonzero(np.isfinite(values) & (values < 0))[:20].tolist(),
                         "non_finite_row_examples": np.flatnonzero(~np.isfinite(values))[:20].tolist(),
                         "negative_value_interpretation": "reported_only_not_automatically_anomalous",
                         "finite_min": float(finite.min()) if finite.size else None,
                         "finite_max": float(finite.max()) if finite.size else None,
                         "finite_range_excluding_minus_9999": {
                             "min": float(finite[finite != -9999].min()) if (finite != -9999).any() else None,
                             "max": float(finite[finite != -9999].max()) if (finite != -9999).any() else None}})
    expected_channels = set(spec["expected_channels"])
    mapping = channel_mapping(numeric, static, spec)
    irregular = source_deltas.notna() & source_deltas.ne(expected)
    gaps = sorted_deltas[sorted_deltas.gt(expected)]
    return {
        "rows": len(frame), "subject_id": spec["subject_id"], "time_column": column,
        "time_dtype": str(raw_time.dtype),
        "timezone": str(timestamps.dt.tz) if timestamps.dt.tz is not None else None,
        "timezone_assumed": False,
        "time_start": valid.min().isoformat() if len(valid) else None,
        "time_end": valid.max().isoformat() if len(valid) else None,
        "null_or_unparseable_timestamps": int(timestamps.isna().sum()),
        "unparseable_non_null_timestamps": int((timestamps.isna() & raw_time.notna()).sum()),
        "duplicate_timestamp_rows_beyond_first": int(valid.duplicated().sum()),
        "all_rows_in_duplicate_timestamp_groups": int(valid.duplicated(keep=False).sum()),
        "expected_interval_seconds": expected,
        "original_adjacent_interval_counts_seconds": {str(k): int(v) for k, v in source_deltas.dropna().value_counts().sort_index().items()},
        "original_nonincreasing_intervals": int(source_deltas.le(0).sum()),
        "original_irregular_intervals": int(irregular.sum()),
        "irregular_interval_row_examples": [int(i) for i in frame.index[irregular][:20]],
        "sorted_unique_interval_counts_seconds": {str(k): int(v) for k, v in sorted_deltas.value_counts().sort_index().items()},
        "sorted_unique_gap_intervals": len(gaps),
        "grid_aligned_missing_points_within_gaps": sum(int(v // expected - 1) for v in gaps if v % expected == 0),
        "gaps_not_divisible_by_expected_interval": int(sum(v % expected != 0 for v in gaps)),
        "numeric_channel_count": len(numeric), "channels": channels,
        "missing_expected_channels": sorted(expected_channels - set(numeric)),
        "unexpected_numeric_channels": sorted(set(numeric) - expected_channels),
        "non_numeric_non_time_columns": [name for name in frame if name != column and name not in numeric],
        "exact_duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "repeated_numeric_vectors_beyond_first": int(frame[numeric].duplicated().sum()) if numeric else None,
        "repeated_vector_interpretation": "may_be_legitimate_constant_readings_not_deduplicated",
        "static_mapping": mapping,
        "modifications_performed": [],
    }


def caiso_alignment(directory, allowed_paths=None):
    """Compare the three published imputed input time axes without joining them."""
    names = [f"raw_data/all_channel_imputed/{name}.parquet" for name in ("co2", "demand", "fuelsource")]
    names.append("time_series/all_except_battery.parquet")
    if allowed_paths is not None and not set(names).issubset(allowed_paths):
        return {"status": "not_checked_not_selected_by_profile", "required_files": names}
    missing = [name for name in names if not data_path(directory, name).is_file()]
    if missing:
        return {"status": "not_checked_missing_provenance_inputs", "missing_files": missing}
    import pandas as pd
    series, counts = [], {}
    for name in names:
        frame = pd.read_parquet(data_path(directory, name), columns=["Time"])
        counts[name] = len(frame)
        series.append(pd.to_datetime(frame["Time"], errors="raise").reset_index(drop=True))
    return {"status": "checked", "rows": counts,
            "time_axes_exactly_equal_including_order": all(series[0].equals(item) for item in series[1:]),
            "limitation": "time-axis agreement alone does not establish causal interpolation or data vintage"}


TIME_KEYS = {
    "issued_at": ("issued_at", "issue_time", "forecast_issued_at", "publication_time"),
    "available_at": ("available_at", "availability_time"),
    "valid_start": ("valid_start", "forecast_valid_start"),
    "valid_end": ("valid_end", "forecast_valid_end"),
}
ALL_TIME_KEYS = {name for values in TIME_KEYS.values() for name in values}
EXPECTED_TEXT_FIELDS = {
    "formal_report": {"Time of Day", "Current overall Weather Condition", "Weather Trend",
                      "Temperature Trend", "Wind Condition", "Pressure", "Humidity"},
    "general_report": {"daily", "Early Morning", "Morning", "Afternoon", "Evening"},
}


def availability_assessment(available_at, evidence=None):
    """No evidence is supplied by the CLI: mere field presence stays unknown.

    This helper defines the future provenance contract; it is not a retriever.
    A protocol assumption never grants verified as-of eligibility.
    """
    evidence = evidence or {}
    status = evidence.get("status", "unknown")
    reference = evidence.get("reference")
    aware = False
    if isinstance(available_at, str):
        try:
            aware = datetime.fromisoformat(available_at.replace("Z", "+00:00")).utcoffset() is not None
        except ValueError:
            pass
    if status == "verified" and reference and aware:
        return {"status": "verified", "evidence_reference": reference,
                "verified_asof_eligibility": True}
    if status == "protocol_assumption" and reference:
        return {"status": "protocol_assumption", "evidence_reference": reference,
                "verified_asof_eligibility": False}
    return {"status": "unknown", "evidence_reference": None, "verified_asof_eligibility": False}


def source_time_key_iso(key):
    """Parse only an observed key. Never assign it to issue/availability/validity."""
    if not isinstance(key, str):
        return None
    for length, fmt in ((8, "%Y%m%d"), (12, "%Y%m%d%H%M"), (14, "%Y%m%d%H%M%S")):
        if len(key) == length and key.isdigit():
            try:
                return datetime.strptime(key, fmt).isoformat()
            except ValueError:
                return None
    try:
        return datetime.fromisoformat(key.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def text_source_identity(dataset, source_file, spec):
    parts = Path(source_file).parts
    if "weather_report" not in parts:
        raise AuditError(f"Not an original weather report: {source_file}")
    at = parts.index("weather_report")
    report_type = parts[at + 1] if at + 1 < len(parts) - 1 else "unknown"
    location = parts[1] if dataset == "caiso" and len(parts) > 2 else "jena"
    version_match = re.search(r"_(v\d+)$", Path(source_file).stem)
    return {"dataset": dataset, "subject_id": spec["subject_id"], "location_id": location,
            "report_type": report_type, "source_file": source_file,
            "version_id": version_match.group(1) if version_match else "unversioned",
            "version_semantics": "file_variant_not_verified_release_sequence"}


def text_leaves(value, pointer="", skip_time_fields=True):
    if isinstance(value, str):
        yield pointer or "/text", value
    elif isinstance(value, JSONObject):
        for key, child in value:
            if skip_time_fields and str(key).lower() in ALL_TIME_KEYS:
                continue
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from text_leaves(child, pointer + "/" + escaped, skip_time_fields)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from text_leaves(child, pointer + f"/{i}", skip_time_fields)


def explicit_times(value):
    pairs = value if isinstance(value, JSONObject) else []
    result, evidence = {}, {}
    for field, names in TIME_KEYS.items():
        matches = [(key, child) for key, child in pairs
                   if str(key).lower() in names and isinstance(child, str) and child.strip()]
        # Ambiguous duplicate time fields must not be silently resolved.
        result[field] = matches[0][1] if len(matches) == 1 else None
        evidence[field] = {"matching_fields": [str(key) for key, _ in matches],
                           "ambiguous": len(matches) > 1}
    return result, evidence


def normalize_documents(dataset, spec, source_file, source_key, value, occurrence=0):
    source = text_source_identity(dataset, source_file, spec)
    times, time_evidence = explicit_times(value)
    leaves_seen = Counter()
    for pointer, text in text_leaves(value):
        leaf_occurrence = leaves_seen[pointer]
        leaves_seen[pointer] += 1
        doc_id = json_digest([dataset, spec["revision"], source_file, source_key,
                              occurrence, pointer, leaf_occurrence])
        yield {"document_id": doc_id, **source, "revision": spec["revision"],
               "source_time_key": source_key, "parsed_source_time_key": source_time_key_iso(source_key),
               "source_key_occurrence": occurrence, "field_path": pointer,
               "field_occurrence": leaf_occurrence, "text": text,
               "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               **times, "time_field_evidence": time_evidence,
               "availability": availability_assessment(times["available_at"])}


def text_summary(directory, paths, dataset, spec, documents_handle):
    """Process one JSON file at a time; preserve duplicates, stream document ledger."""
    counts = Counter({"documents": 0, "records": 0, "duplicate_time_keys_within_files": 0})
    lengths, hashes, fields = Counter(), Counter(), Counter()
    slots, time_slots, slot_locations = Counter(), Counter(), defaultdict(set)
    variants = defaultdict(dict)
    file_reports = []
    time_counts = Counter()
    time_examples = {field: set() for field in TIME_KEYS}
    for source_file in sorted(paths):
        root = load_pairs(data_path(directory, source_file))
        source = text_source_identity(dataset, source_file, spec)
        if not isinstance(root, JSONObject):
            raise AuditError(f"Weather report root must be a time-key object: {source_file}")
        seen, key_shapes = Counter(), Counter()
        local = Counter()
        key_min = key_max = None
        root_field_names = Counter()
        missing_fields = Counter()
        for key, value in root:
            key = str(key)
            occurrence = seen[key]
            seen[key] += 1
            parsed = source_time_key_iso(key)
            key_shapes[str(len(key)) + (":digits" if key.isdigit() else ":other")] += 1
            if parsed:
                key_min = min(key_min, parsed) if key_min else parsed
                key_max = max(key_max, parsed) if key_max else parsed
            else:
                local["unparsed_source_time_keys"] += 1
            if isinstance(value, JSONObject):
                root_field_names.update(str(k) for k, _ in value)
                missing_fields.update(EXPECTED_TEXT_FIELDS.get(source["report_type"], set())
                                      - {str(k) for k, _ in value})
                if len({k for k, _ in value}) != len(value):
                    local["records_with_duplicate_fields"] += 1
            else:
                local["non_object_records"] += 1
            docs = list(normalize_documents(dataset, spec, source_file, key, value, occurrence))
            if not docs:
                local["records_without_text"] += 1
            for doc in docs:
                documents_handle.write(json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n")
                local["documents"] += 1
                counts["documents"] += 1
                lengths[len(doc["text"])] += 1
                hashes[doc["text_sha256"]] += 1
                fields[doc["field_path"]] += 1
                for field in TIME_KEYS:
                    if doc[field] is not None:
                        time_counts[field] += 1
                        if len(time_examples[field]) < 20:
                            time_examples[field].add(doc[field])
                slot = (source["location_id"], key, source["report_type"],
                        source["version_id"], doc["field_path"])
                slots[slot] += 1
                time_slots[slot[:-1]] += 1
                slot_locations[key].add(source["location_id"])
            if source["version_id"] != "unversioned":
                group = (source["location_id"], source["report_type"])
                variant = variants[group].setdefault(source["version_id"], {})
                variant[(key, occurrence)] = json_digest(sorted(
                    (d["field_path"], d["field_occurrence"], d["text"]) for d in docs))
            local["records"] += 1
        counts["records"] += local["records"]
        counts["duplicate_time_keys_within_files"] += sum(v - 1 for v in seen.values())
        file_reports.append({**source, "record_count": local["records"],
                             "document_count": local["documents"], "unique_time_keys": len(seen),
                             "duplicate_time_keys_beyond_first": sum(v - 1 for v in seen.values()),
                             "time_key_shapes": dict(key_shapes),
                             "parsed_key_min": key_min, "parsed_key_max": key_max,
                             "record_field_counts": dict(root_field_names),
                             "missing_expected_field_record_counts": dict(missing_fields),
                             "diagnostics": dict(local),
                             "time_coverage_interpretation": "source_key_range_only_not_forecast_validity"})
    differences = []
    for (location, report_type), versions in sorted(variants.items()):
        for a, b in combinations(sorted(versions), 2):
            common = versions[a].keys() & versions[b].keys()
            differences.append({"location_id": location, "report_type": report_type,
                                "file_versions": [a, b], "shared_records": len(common),
                                "different_record_texts": sum(versions[a][key] != versions[b][key] for key in common),
                                "only_first_version_records": len(versions[a].keys() - versions[b].keys()),
                                "only_second_version_records": len(versions[b].keys() - versions[a].keys()),
                                "chronological_release_order_verified": False})
    count = counts["documents"]
    length_summary = {"unit": "unicode_characters", "count": count,
                      "min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None,
                      "mean": sum(k * v for k, v in lengths.items()) / count if count else None}
    summary = {"files_checked": len(file_reports), **dict(counts), "files": file_reports,
               "field_path_counts": dict(fields), "text_lengths": length_summary,
               "unique_text_hashes": len(hashes),
               "repeated_texts_beyond_first": sum(v - 1 for v in hashes.values()),
               "same_facet_slot_additional_texts": sum(v - 1 for v in slots.values()),
               "time_slots_with_multiple_texts": sum(v > 1 for v in time_slots.values()),
               "documents_per_time_slot_counts": {str(k): v for k, v in sorted(Counter(time_slots.values()).items())},
               "source_time_keys_with_multiple_locations": sum(len(v) > 1 for v in slot_locations.values()),
               "version_differences": differences,
               "explicit_time_field_document_counts": {field: time_counts[field] for field in TIME_KEYS},
               "explicit_time_field_value_examples_unverified": {
                   field: sorted(values) for field, values in time_examples.items()},
               "forecast_coverage": "unknown_except_explicit_unverified_fields_in_documents_ledger"}
    availability = {"documents": count, "status_counts": {"verified": 0, "protocol_assumption": 0, "unknown": count},
                    "documents_with_explicit_available_at": time_counts["available_at"],
                    "documents_missing_available_at": count - time_counts["available_at"],
                    "formal_multimodal_training_allowed": False,
                    "verified_asof_default": "reject_unknown_available_at_or_unverified_provenance",
                    "reason": "Stage 1 does not attest historical issuance; source keys never fill available_at",
                    "no_text_files": not file_reports}
    return summary, availability
