"""Inspect local source schemas/code and retain observation-clock evidence.

Schema resemblance and content agreement are labelled inferences, not proof of
historical issuance. Downloaded Python files are parsed, never executed.
"""

import ast
from collections import Counter
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path
import re

from scripts.fidel_ts.common import ROOT, data_path, read_json, sha256_file
from scripts.fidel_ts.inspection import JSONObject, load_pairs, source_time_key_iso


def plain(value):
    if isinstance(value, JSONObject):
        return {str(k): plain(v) for k, v in value}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonstandard_json_numeric_token": str(value)}
    return value


def observed_clock(record):
    """Only explicit calendar labels; no conversion of undocumented epoch units."""
    date = record.get("Date", record.get("date"))
    time = record.get("Time", record.get("time"))
    candidates = []
    if isinstance(date, str) and isinstance(time, str):
        candidates.append(("Date+Time", date + "T" + time))
    candidates.extend((name, record.get(name)) for name in ("Date Time", "datetime", "timestamp", "Time", "time", "Date", "date"))
    for field, value in candidates:
        if isinstance(value, str):
            parsed = source_time_key_iso(value)
            if parsed:
                return {"start": parsed, "end": None, "field": field,
                        "role": "observed_calendar_label_not_issuance", "timezone_verified": False}
    ds = record.get("ds")
    if isinstance(ds, str):
        match = re.search(r"(\d{1,2} [A-Za-z]+ \d{4}),?\s+(\d{1,2}:\d{2})\s*[—–-]\s*(\d{1,2}:\d{2})", ds)
        if match:
            try:
                day, month, year = match[1].split()
                months = {name.lower(): i + 1 for i, name in enumerate((
                    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"))}
                begin = datetime(int(year), months[month.lower()], int(day), *map(int, match[2].split(":")))
                end = datetime(int(year), months[month.lower()], int(day), *map(int, match[3].split(":")))
                if end < begin:
                    end += timedelta(days=1)
                return {"start": begin.isoformat(), "end": end.isoformat(), "field": "ds",
                        "role": "observation_period_label_not_forecast_validity", "timezone_verified": False}
            except (ValueError, KeyError):
                pass
    return None


def code_evidence(path, relative):
    text = path.read_text(encoding="utf-8-sig")
    result = {"source_file": relative, "executed": False, "calls": []}
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        result["parse_error"] = str(exc)
        return result
    watched = {"interpolate", "resample", "encode", "read_parquet", "read_csv", "from_pretrained"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in watched:
            result["calls"].append({"line": node.lineno, "call": node.func.attr,
                                    "expression": (ast.get_source_segment(text, node) or "")[:600],
                                    "evidence_reference": f"{relative}:L{node.lineno}"})
    result["text_generation_from_future_observations_proven"] = False
    result["limitation"] = "Static call list does not establish text generation inputs or historical release time"
    return result


def inspect_local_sources(directory, manifest, dataset):
    inventory = {entry["path"] for entry in manifest["inventory"]["files"]}
    downloaded = {name for name, entry in manifest["files"].items() if entry["status"] == "complete"}
    wanted = sorted(name for name in inventory if (
        dataset == "caiso" and name.startswith("weather/") and "/raw_data/" in name)
        or (dataset == "jena" and name.startswith("raw_data/") and name.endswith(".parquet")))
    report = {"missing_provenance_files": sorted(set(wanted) - downloaded), "sources": [], "code_evidence": [],
              "historical_issuance_evidence": "unknown", "generation_mechanism": "unknown",
              "strong_future_observation_generation_evidence": [],
              "author_contact_required": False}
    observations = []
    reference = read_json(ROOT / "docs/fidel_ts/00_audit_evidence.json")
    report["official_pipeline_reference"] = {
        "commit": reference["official_code"]["sha"],
        "prior_audit": "docs/fidel_ts/00_repository_audit.md#4",
        "matching_limitation": "source-key backward/nearest matching is not verification of historical availability",
        "local_cached_code": []}
    for source in reference["source_files"]:
        if source["dataset"] != "official_code" or source["path"] not in {
            "data_provider/data_loader.py", "data_provider/data_helper.py"}:
            continue
        cached = Path("/tmp/fidel_ts_stage0/official") / source["path"]
        evidence = {"source_file": source["path"], "expected_sha256": source["sha256"],
                    "cached_path": str(cached), "cache_status": "unavailable"}
        if cached.is_file():
            evidence["cache_status"] = "hash_matched" if sha256_file(cached) == source["sha256"] else "hash_mismatch_not_used"
            if evidence["cache_status"] == "hash_matched":
                evidence["code"] = code_evidence(cached, str(cached))
        report["official_pipeline_reference"]["local_cached_code"].append(evidence)
    for name in sorted(downloaded):
        path = data_path(directory, name)
        if name.startswith("scripts/") and name.endswith(".py"):
            report["code_evidence"].append(code_evidence(path, name))
    for name in wanted:
        if name not in downloaded:
            continue
        item = {"source_file": name, "sha256": manifest["files"][name]["sha256"]}
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            metadata = pq.ParquetFile(data_path(directory, name))
            item.update(format="parquet", rows=metadata.metadata.num_rows,
                        columns=metadata.schema_arrow.names,
                        issuance_interpretation="numerical observations do not prove text issuance")
            report["sources"].append(item)
            continue
        if name.endswith(".json"):
            root = load_pairs(data_path(directory, name))
            if isinstance(root, JSONObject):
                # Retain source keys as metadata; do not call them available_at.
                records = [(str(key), plain(value)) for key, value in root]
            elif isinstance(root, list):
                records = [(str(i), plain(value)) for i, value in enumerate(root)]
            else:
                records = []
        elif name.endswith(".csv"):
            with data_path(directory, name).open(encoding="utf-8-sig", newline="") as handle:
                records = [(str(i), row) for i, row in enumerate(csv.DictReader(handle))]
        else:
            continue
        fields = Counter()
        clocks = 0
        location = Path(name).parts[1] if dataset == "caiso" else "jena"
        examples = []
        for key, record in records:
            if not isinstance(record, dict):
                continue
            fields.update(str(field) for field in record.keys())
            if len(examples) < 2:
                examples.append({"record_pointer": key, "fields": record})
            clock = observed_clock(record)
            if clock:
                clocks += 1
                observations.append({"source_file": name, "record_pointer": key,
                                     "location_id": location, "clock": clock, "raw_values": record})
        issue_candidates = [key for key in fields if key.lower() in {
            "issued_at", "available_at", "forecast_origin", "issue_time", "run_time", "model_run"}]
        forecast_candidates = [key for key in fields if key.lower() in {
            "valid_start", "valid_end", "forecast_valid_start", "forecast_valid_end"}]
        observation_like = bool(set(key.lower() for key in fields) & {"temp", "humidity", "hum", "baro", "desc", "weather"})
        item.update(format=Path(name).suffix, records=len(records), field_counts=dict(fields),
                    explicit_issue_or_run_field_candidates=issue_candidates,
                    explicit_forecast_validity_field_candidates=forecast_candidates,
                    records_with_calendar_clock=clocks, examples=examples,
                    observed_schema_inference="observation_archive_like" if observation_like else "unknown",
                    inference_is_not_generation_proof=True,
                    availability_status="unknown")
        report["sources"].append(item)
    return report, observations
