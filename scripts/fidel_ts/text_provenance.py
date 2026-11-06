"""Report-level provenance, stratified retrospective diagnostics and T1/T2/T3 rules."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from itertools import combinations
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd

from scripts.fidel_ts.common import AuditError, data_path, json_digest
from scripts.fidel_ts.inspection import (
    explicit_times, load_pairs, normalize_documents, source_time_key_iso,
    text_leaves, text_source_identity,
)
from scripts.fidel_ts.numerical_diagnostics import NativeData, iso, load_native, split_ids
from scripts.fidel_ts.stage1_5_io import SPLITS
from scripts.fidel_ts.weather_evidence import inspect_local_sources


REPORT_FIELDS = ("dataset", "location_id", "source_file", "report_type", "source_time_key",
                 "source_key_occurrence", "version_id")


def report_identity(document):
    return json_digest([document["revision"], *[document[key] for key in REPORT_FIELDS]])


def reconstruct_reports(ledger, dataset, spec, manifest):
    reports = {}
    with Path(ledger).open(encoding="utf-8") as handle:
        for line in handle:
            document = json.loads(line)
            if (document["dataset"], document["revision"]) != (dataset, spec["revision"]):
                raise AuditError("Stage 1 ledger dataset/revision mismatch")
            if document["subject_id"] != spec["subject_id"] or document["source_file"] not in manifest["files"]:
                raise AuditError("Stage 1 ledger has an unbound Subject or source file")
            expected_id = json_digest([dataset, spec["revision"], document["source_file"], document["source_time_key"],
                                       document["source_key_occurrence"], document["field_path"], document["field_occurrence"]])
            if document["document_id"] != expected_id:
                raise AuditError("Stage 1 document identity differs from the preserved schema")
            report_id = report_identity(document)
            if report_id not in reports:
                reports[report_id] = {key: document[key] for key in REPORT_FIELDS}
                reports[report_id].update(report_id=report_id, revision=spec["revision"], subject_id=spec["subject_id"],
                                          facets=[], issued_at=document.get("issued_at"), available_at=document.get("available_at"),
                                          valid_start=document.get("valid_start"), valid_end=document.get("valid_end"),
                                          forecast_origin=None, availability_status="unknown",
                                          time_semantics="source key is not issuance or verified forecast validity")
            report = reports[report_id]
            if any(f["document_id"] == document["document_id"] for f in report["facets"]):
                raise AuditError("Duplicate facet identity in ledger")
            report["facets"].append({"facet_id": json_digest([report_id, document["field_path"], document["field_occurrence"]]),
                                     "document_id": document["document_id"], "field_path": document["field_path"],
                                     "field_occurrence": document["field_occurrence"], "text": document["text"]})
    return list(reports.values())


def include_empty_source_reports(reports, directory, dataset, spec, source_files):
    """An empty report has no Stage 1 facet, but still counts as one report."""
    result = {report["report_id"]: report for report in reports}
    found = set()
    for source in sorted(source_files):
        metadata = text_source_identity(dataset, source, spec)
        occurrences = Counter()
        for key, value in load_pairs(data_path(directory, source)):
            key = str(key)
            occurrence = occurrences[key]
            occurrences[key] += 1
            candidate = {**metadata, "revision": spec["revision"], "source_time_key": key,
                         "source_key_occurrence": occurrence}
            report_id = report_identity(candidate)
            found.add(report_id)
            if report_id not in result:
                if next(text_leaves(value), None) is not None:
                    raise AuditError("Original nonempty report is missing from the Stage 1 ledger")
                times, _ = explicit_times(value)
                result[report_id] = {**candidate, **times, "report_id": report_id, "facets": [],
                                     "forecast_origin": None, "availability_status": "unknown",
                                     "time_semantics": "source key is not issuance or forecast validity"}
    if not set(result).issubset(found):
        raise AuditError("Ledger contains a report not found in the audited originals")
    return list(result.values())


def access_decision(report, forecast_origin, forecast_end=None):
    """Diagnostic eligibility only, never a production retriever or fake issue time."""
    key = source_time_key_iso(report["source_time_key"])
    try:
        time = pd.Timestamp(key) if key else None
        origin = pd.Timestamp(forecast_origin)
        past = bool(time is not None and time < origin)
        aligned = bool(time is not None and forecast_end is not None and origin <= time < pd.Timestamp(forecast_end))
    except (ValueError, TypeError):
        past = aligned = False
    return {"T1": "blocked_not_verifiable", "T2_key_only_eligible": past,
            "T2": "auxiliary_archived_past_context_not_verified_asof",
            "T3_key_aligned": aligned, "T3": "retrospective_oracle_risk"}


def stratified_samples(reports, config, time_bounds=None):
    rng = random.Random(config["sampling"]["seed"])
    counts, reservoirs = Counter(), defaultdict(list)
    for report in sorted(reports, key=lambda value: value["report_id"]):
        parsed = source_time_key_iso(report["source_time_key"])
        if not parsed:
            continue
        timestamp = pd.Timestamp(parsed)
        if timestamp.tz is not None:
            continue  # Unknown cross-clock alignment is not silently coerced.
        if time_bounds is not None and not time_bounds[0] <= timestamp <= time_bounds[1]:
            continue
        split = SPLITS[int(split_ids(pd.DatetimeIndex([timestamp]), config)[0])]
        season = ("DJF", "MAM", "JJA", "SON")[(timestamp.month % 12) // 3]
        stratum = (split, timestamp.year, season, report["location_id"], report["report_type"], report["version_id"])
        counts[stratum] += 1
        pool = reservoirs[stratum]
        limit = config["sampling"]["reports_per_stratum"]
        if len(pool) < limit:
            pool.append(report)
        else:
            selected = rng.randrange(counts[stratum])
            if selected < limit:
                pool[selected] = report
    return [(key, report) for key in sorted(reservoirs) for report in reservoirs[key]], counts


def numeric_context(data, timestamp, hours):
    delta = pd.Timedelta(hours=hours)
    output = {}
    for name, mask in (("before", (data.times >= timestamp - delta) & (data.times < timestamp)),
                       ("after", (data.times >= timestamp) & (data.times < timestamp + delta))):
        selected = np.flatnonzero(mask)
        summary = {}
        if len(selected):
            selected = selected[np.argsort(data.ns[selected], kind="stable")]
        for j, channel in enumerate(data.channels):
            values = data.values[selected, j]
            values = values[data.valid[selected, j]]
            summary[channel] = {"valid_count": len(values), "min": float(values.min()) if len(values) else None,
                                "max": float(values.max()) if len(values) else None,
                                "mean": float(values.mean()) if len(values) else None,
                                "first": float(values[0]) if len(values) else None,
                                "last": float(values[-1]) if len(values) else None}
        output[name] = {"original_rows_in_interval": len(selected), "channels": summary}
    output["reference_time"] = iso(timestamp)
    output["interpretation"] = "retrospective diagnostic around source key, not an asserted forecast origin"
    return output


def extract_content_cues(report):
    text = "\n".join(facet["text"] for facet in report["facets"])
    number = r"(-?\d+(?:\.\d+)?)"
    ranges = re.findall(number + r"\s*(?:to|–|—|-)\s*" + number + r"\s*(?:degrees?\s*(?:Celsius|C)\b|°C)", text, re.I)
    winds = re.findall(number + r"\s*(km/h|kmh|m/s)\b", text, re.I)
    rain = re.findall(number + r"\s*mm\b", text, re.I)
    humidity = re.findall(number + r"\s*%", text)
    return {"temperature_ranges_degc": [[float(a), float(b)] for a, b in ranges],
            "wind_values": [{"value": float(v), "unit": unit} for v, unit in winds],
            "rainfall_mm_mentions_unverified_role": [float(v) for v in rain],
            "percentage_mentions_unverified_role": [float(v) for v in humidity],
            "trend_terms": sorted({word.lower() for word in re.findall(
                r"\b(?:increase|increasing|rise|rising|drop|dropping|fall|falling|decrease|decreasing|steady|stable)\b", text, re.I)}),
            "role_and_time_coverage_inference": "not_verified"}


def report_structure(reports, config, time_bounds=None):
    slots = Counter((r["location_id"], r["source_time_key"], r["report_type"], r["version_id"]) for r in reports)
    samples, strata = stratified_samples(reports, config, time_bounds)
    periods = Counter()
    for r in reports:
        key = source_time_key_iso(r["source_time_key"])
        if key:
            t = pd.Timestamp(key)
            if t.tz is None:
                periods[f"{SPLITS[int(split_ids(pd.DatetimeIndex([t]), config)[0])]}/{t.year}"] += 1
    return {"reports": len(reports), "facets": sum(len(r["facets"]) for r in reports),
            "by_location": dict(Counter(r["location_id"] for r in reports)),
            "by_report_type": dict(Counter(r["report_type"] for r in reports)),
            "by_file_variant": dict(Counter(r["version_id"] for r in reports)),
            "reports_by_split_year": dict(periods),
            "additional_reports_in_same_location_type_variant_time_slot": sum(n - 1 for n in slots.values()),
            "report_count_is_not_issuance_count": True,
            "sampling_numerical_time_bounds": [iso(value) for value in time_bounds] if time_bounds else None,
            "sampling_scope": "naive parsable source keys within primary numerical range; inventory counts still include all reports",
            "sampling_strata": [{"stratum": list(k), "population": v} for k, v in sorted(strata.items())],
            "sampled_reports": len(samples)}, samples


def version_comparison(reports):
    groups = defaultdict(lambda: defaultdict(dict))
    for report in reports:
        if report["version_id"] == "unversioned":
            continue
        key = (report["source_time_key"], report["source_key_occurrence"])
        signature = json_digest(sorted((f["field_path"], f["field_occurrence"], f["text"]) for f in report["facets"]))
        groups[(report["location_id"], report["report_type"])][report["version_id"]][key] = signature
    comparisons = []
    for group, versions in groups.items():
        for a, b in combinations(sorted(versions), 2):
            shared = versions[a].keys() & versions[b].keys()
            comparisons.append({"location_id": group[0], "report_type": group[1], "variants": [a, b],
                                "shared_time_records": len(shared),
                                "different_text_records": sum(versions[a][key] != versions[b][key] for key in shared),
                                "only_first": len(versions[a].keys() - versions[b].keys()),
                                "only_second": len(versions[b].keys() - versions[a].keys()),
                                "historical_release_order": "unknown", "version_generation_time": None})
    return comparisons


def sampled_raw_reports(directory, samples, spec):
    """Reuse Stage 1 parser and normalization, check selected original report text."""
    wanted = defaultdict(dict)
    for _, report in samples:
        wanted[report["source_file"]][(report["source_time_key"], report["source_key_occurrence"])] = report
    checked = set()
    for source, wanted_keys in wanted.items():
        occurrences = Counter()
        for key, value in load_pairs(data_path(directory, source)):
            occurrence = occurrences[str(key)]
            occurrences[str(key)] += 1
            report = wanted_keys.get((str(key), occurrence))
            if report is None:
                continue
            docs = list(normalize_documents(report["dataset"], spec, source, str(key), value, occurrence))
            original = {(d["document_id"], d["text"]) for d in docs}
            preserved = {(f["document_id"], f["text"]) for f in report["facets"]}
            if original != preserved:
                raise AuditError("Stage 1 facets disagree with the selected original weather report")
            checked.add(report["report_id"])
    if checked != {r["report_id"] for _, r in samples}:
        raise AuditError("Sampled original report identities were not all found")


def analyse_text(directory, audit_dir, spec, dataset, manifest, config, output):
    reports = reconstruct_reports(audit_dir / "documents.jsonl", dataset, spec, manifest)
    audit_manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    sources = [entry["path"] for entry in audit_manifest["files"]
               if "/weather_report/" in entry["path"] and entry["path"].endswith(".json")
               and Path(entry["path"]).name != "static_info.json"]
    reports = include_empty_source_reports(reports, directory, dataset, spec, sources)
    data = load_native(data_path(directory, spec["primary_file"]), dataset, spec)
    bounds = ((data.times.min(), data.times.max())
              if config["sampling"]["sample_only_source_keys_within_primary_numerical_range"] else None)
    structure, samples = report_structure(reports, config, bounds)
    structure["report_count_scope"] = "all original report records including those with zero textual facets"
    sampled_raw_reports(directory, samples, spec)
    source_analysis, observations = inspect_local_sources(directory, manifest, dataset)
    source_analysis["variant_comparison"] = version_comparison(reports)
    observations_by_location = defaultdict(list)
    for observation in observations:
        observed = pd.Timestamp(observation["clock"]["start"])
        if observed.tz is None:
            observations_by_location[observation["location_id"]].append((observed, observation))
    raw_numerical = []
    if dataset == "jena":
        import pyarrow.parquet as pq
        for item in source_analysis["sources"]:
            if item.get("format") != "parquet":
                continue
            frame = pq.read_table(data_path(directory, item["source_file"])).to_pandas()
            if spec["time_column"] not in frame:
                item["context_status"] = "missing_known_time_column"
                continue
            columns = [c for c in frame if c != spec["time_column"] and pd.api.types.is_numeric_dtype(frame[c])]
            clock = frame[spec["time_column"]]
            if not pd.api.types.is_datetime64_any_dtype(clock):
                clock = clock.map(lambda value: source_time_key_iso(value) if isinstance(value, str) else None)
            times = pd.DatetimeIndex(pd.to_datetime(clock, errors="coerce"))
            item["unparsed_observation_timestamps"] = int(times.isna().sum())
            if times.isna().all():
                item["context_status"] = "raw_timestamp_format_requires_evidence_no_fuzzy_parsing"
                continue
            if times.tz is not None:
                item["context_status"] = "timezone_alignment_unverified"
                continue
            values = frame[columns].to_numpy(dtype=float, na_value=np.nan)
            raw_numerical.append((item["source_file"], NativeData(times, values, np.zeros(values.shape, dtype=bool), columns, dataset, spec["expected_interval_seconds"])))
    with (output / "reports.jsonl").open("x", encoding="utf-8") as ledger:
        for report in reports:
            ledger.write(json.dumps(report, ensure_ascii=False) + "\n")
    with (output / "text_leakage_risk_samples.jsonl").open("x", encoding="utf-8") as handle:
        for stratum, report in samples:
            timestamp = pd.Timestamp(source_time_key_iso(report["source_time_key"]))
            context = numeric_context(data, timestamp, config["sampling"]["context_hours"])
            cues = extract_content_cues(report)
            matches = []
            for observed, observation in observations_by_location[report["location_id"]]:
                if abs(observed - timestamp) <= pd.Timedelta(hours=config["sampling"]["context_hours"]):
                    matches.append(observation)
            risks = ["unknown_historical_issuance", "retrospective_comparison_is_not_leakage_proof"]
            temperature = context["after"]["channels"].get("T (degC)", {})
            before_temperature = context["before"]["channels"].get("T (degC)", {})
            temperature_delta = (temperature["mean"] - before_temperature["mean"]
                                 if temperature.get("mean") is not None and before_temperature.get("mean") is not None else None)
            for low, high in cues["temperature_ranges_degc"]:
                if temperature.get("mean") is not None and min(low, high) - config["sampling"]["temperature_tolerance_degc"] <= temperature["mean"] <= max(low, high) + config["sampling"]["temperature_tolerance_degc"]:
                    risks.append("text_temperature_range_compatible_with_post_key_observed_mean_not_proof")
            sample = {"stratum": list(stratum), **report,
                      "analysis_reference_time": iso(timestamp), "forecast_origin": None,
                      "published_numerical_context": context, "content_cues": cues,
                      "temperature_after_minus_before_mean": temperature_delta,
                      "trend_interpretation": "compare stated trend terms with this diagnostic delta manually; no leakage verdict",
                      "raw_weather_observation_matches": matches[:20], "raw_weather_matching_rows": len(matches),
                      "raw_weather_clock_alignment": "protocol_assumption_only_not_verified_timezone",
                      "raw_numerical_context": [{"source_file": path, "summary": numeric_context(raw, timestamp, config["sampling"]["context_hours"])}
                                                for path, raw in raw_numerical if ((raw.times >= timestamp - pd.Timedelta(hours=config["sampling"]["context_hours"])) & (raw.times < timestamp + pd.Timedelta(hours=config["sampling"]["context_hours"]))).any()],
                      "risk_indicators": sorted(set(risks)),
                      "evidence_references": [report["source_file"], spec["primary_file"]]}
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    temporal = {"reports": len(reports), "facets": structure["facets"],
                "availability_status_counts": {"verified": 0, "protocol_assumption": 0, "unknown": len(reports)},
                "explicit_fields_in_ledger": {field: sum(r.get(field) is not None for r in reports)
                                              for field in ("issued_at", "available_at", "valid_start", "valid_end", "forecast_origin")},
                "T1": "blocked_not_verifiable", "T2": "auxiliary_archived_past_context_only",
                "T3": "retrospective_oracle_risk", "forecast_validity_scoring_allowed": False,
                "default_temporal_scoring_alternative": "exploratory_source_key_distance_only_separately_named",
                "predictive_utility_may_override_availability": False,
                "no_author_reply_required": True}
    return structure, source_analysis, temporal
