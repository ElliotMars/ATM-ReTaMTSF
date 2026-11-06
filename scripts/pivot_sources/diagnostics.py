"""Read-only task/clock audits and explicitly scoped candidate coverage."""

from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import re

from scripts.pivot_sources.asof import availability, eligible_records, utc
from scripts.pivot_sources.common import ROOT, SourceError, evidence_ok, read_json, safe_path, sha256
from scripts.pivot_sources import gfs_archive, nws_iem, stocknet


def clock_evidence_scan(dataset):
    directory = ROOT / "dataset/Fidel-TS/raw" / ("California_ISO" if dataset == "caiso" else "Jena_Atmospheric_Physics")
    candidates = [directory / "README.md", *sorted((directory / "scripts").glob("*.py"))]
    result = []
    for path in candidates:
        if not path.is_file():
            continue
        matches = [{"line": i, "text": line[:500]} for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
                   if re.search(r"\b(?:UTC|GMT|PST|PDT|CET|CEST)\b|timezone|time.zone|tz_localize|tz_convert", line, re.I)]
        result.append({"file": str(path), "sha256": sha256(path), "candidate_lines": matches,
                       "interpretation": "keyword occurrence is not evidence that the numerical Time column uses that clock"})
    return result


def n2_origins(spec, clock_evidence=None):
    """Recompute only new-H window statistics; never edit Stage 1.5 artifacts."""
    import pandas as pd
    from scripts.fidel_ts.common import assert_identity, load_source
    from scripts.fidel_ts.numerical_protocols import window_accounting
    from scripts.fidel_ts.stage1_5_io import read_bundle
    directory = ROOT / spec["n2_directory"]
    binding = read_bundle(directory)
    manifest = read_json(directory / "manifest.json")
    source = load_source(spec["dataset"])
    assert_identity(binding, spec["dataset"], source)
    assert_identity(manifest, spec["dataset"], source)
    if manifest.get("protocol") != "n2" or manifest.get("mode") != "prepare":
        raise SourceError("Only successful prepared N2 data can be inspected")
    if manifest["primary_sha256"] != binding["primary_sha256"]:
        raise SourceError("N2 primary hash binding is inconsistent")
    expected_channels = source["expected_channels"]
    if read_json(directory / "channels.json")["channels"] != expected_channels or len(expected_channels) != spec["channels"]:
        raise SourceError("Full configured 20/21 channel universe is required")
    clock_verified = (evidence_ok(clock_evidence, ("numerical_clock",)) and
                      clock_evidence.get("bound_primary_sha256") == binding["primary_sha256"] and
                      isinstance(clock_evidence.get("timezone"), str))
    result = {"n2_manifest": str(directory / "manifest.json"), "n2_success_sha256": sha256(directory / "_SUCCESS.json"),
              "original_primary_sha256": binding["primary_sha256"], "channel_count": len(expected_channels),
              "clock_status": "reviewed_evidence_bound_to_primary" if clock_verified else "blocked_unknown_numerical_timezone",
              "clock_evidence": clock_evidence, "splits": {}, "origins": [], "new_H_statistics_only": True,
              "timezone_candidates": clock_evidence_scan(spec["dataset"])}
    for name in ("train", "validation", "test"):
        timestamps = pd.read_parquet(directory / name / "timestamps.parquet")
        validity = pd.read_parquet(directory / name / "validity_mask.parquet")
        if list(validity.columns) != expected_channels:
            raise SourceError("N2 mask channel order changed")
        times = pd.DatetimeIndex(timestamps["timestamp"])
        converted = None
        if clock_verified:
            try:
                converted = times.tz_localize(clock_evidence["timezone"], ambiguous="raise", nonexistent="raise").tz_convert("UTC")
            except Exception as exc:
                # Timezone packages expose different DST exception classes.
                result["clock_status"] = "blocked_ambiguous_or_nonexistent_local_clock"
                result["clock_error"] = type(exc).__name__ + ": " + str(exc)
        result["splits"][name] = {}
        for horizon in spec["horizons"]:
            summary, indices = window_accounting(times, validity.to_numpy(dtype=bool), timestamps["native_segment_id"].to_numpy(),
                                                 spec["lookback"], horizon, True)
            if spec.get("archive_start_candidate"):
                summary["forecast_origins_before_candidate_archive_calendar"] = int(
                    (indices["forecast_origin"] < pd.Timestamp(spec["archive_start_candidate"])).sum())
                summary["archive_impact_time_basis"] = "original_naive_calendar_labels_not_verified_UTC"
            result["splits"][name][str(horizon)] = summary
            if converted is not None:
                for position in indices["label_start_row"]:
                    origin = converted[int(position)]
                    if origin.date().isoformat() in spec["probe_dates"]:
                        result["origins"].append({"split": name, "horizon": horizon, "forecast_origin": origin.isoformat()})
    if result["clock_status"].startswith("blocked"):
        result["origins"] = []  # No partially localized multi-split sequence.
    return result


def policy_audit(records, source, lags):
    scenarios = [("strict", 0), *(("run_lag" if source == "gfs" else "source_time", lag) for lag in lags)]
    return [{"policy": policy, "lag_hours": lag, "status_counts": dict(Counter(
        availability(record, policy, lag)["availability_status"] for record in records)),
             "strict_verified_records": sum(availability(record, policy, lag)["strict_verified"] for record in records),
             "records_with_available_at": sum(availability(record, policy, lag)["available_at"] is not None for record in records),
             "lag_status": "none" if policy == "strict" else "experiment_assumption_not_observed_latency"}
            for policy, lag in scenarios]


def nws_coverage(records, origins, spec):
    records = list({record["document_id"]: record for record in records}.values())
    rows = []
    for office in spec["afos_ids"]:
        selected = [record for record in records if record.get("afos_id") == office]
        for policy, lag in [("strict", 0), *(("source_time", lag) for lag in spec["lag_hours"])]:
            policy_verifiable = any(availability(record, policy, lag)["available_at"] is not None for record in selected)
            for split in ("train", "validation", "test"):
                for horizon in spec["horizons"]:
                    at_times = [utc(row["forecast_origin"]) for row in origins if row["split"] == split and row["horizon"] == horizon]
                    recent, no_candidates, ages, issued_ages = Counter(), 0, [], []
                    for origin in at_times:
                        pool = eligible_records(selected, origin, policy, lag)
                        no_candidates += not bool(pool)
                        if pool:
                            age = min((origin - utc(record["available_at"])).total_seconds() / 3600 for record in pool)
                            ages.append(age)
                            if any(record.get("issued_at") for record in pool):
                                issued_ages.append(min((origin - utc(record["issued_at"])).total_seconds() / 3600
                                                       for record in pool if record.get("issued_at")))
                            for window in (24, 72, 168):
                                recent[window] += age <= window
                    rows.append({"afos_id": office, "split": split, "horizon": horizon, "policy": policy, "lag_hours": lag,
                                 "policy_status": "sample_only" if policy_verifiable else "not_verifiable",
                                 "sampled_forecast_origins": len(at_times), "no_observed_candidate_windows": no_candidates if policy_verifiable else None,
                                 "observed_candidate_within_hours": {str(window): {"count": recent[window] if policy_verifiable else None,
                                     "fraction": recent[window] / len(at_times) if at_times and policy_verifiable else None} for window in (24, 72, 168)},
                                 "latest_observed_availability_age_hours": {"min": min(ages) if ages else None, "max": max(ages) if ages else None},
                                 "latest_observed_issuance_age_hours": {"min": min(issued_ages) if issued_ages else None, "max": max(issued_ages) if issued_ages else None},
                                 "forecast_validity_coverage": "unknown_AFD_valid_intervals_not_invented",
                                 "denominator_scope": "N2 origins on configured probe dates only; sampled announcements may be incomplete"})
    return rows


def build_calendar(spec, provided=None, provider="none"):
    if provided:
        return stocknet.calendar_rows(provided), {"provider": "local_csv", "file": str(provided), "sha256": sha256(provided)}
    if provider != "exchange_calendars":
        return [], {"status": "not_checked", "reason": "supply an official calendar CSV or explicitly select provisional exchange_calendars"}
    try:
        import exchange_calendars
        calendar = exchange_calendars.get_calendar(spec["market"], start="2013-11-01", end="2015-12-31")
    except ImportError as exc:
        raise SourceError("exchange_calendars missing; installation is a separate user action") from exc
    rows = []
    for session, row in calendar.schedule.iterrows():
        opened = row.get("open", row.get("market_open"))
        closed = row.get("close", row.get("market_close"))
        rows.append({"session": session.strftime("%Y-%m-%d"), "open": utc(opened.to_pydatetime()).isoformat(),
                     "close": utc(closed.to_pydatetime()).isoformat()})
    return rows, {"provider": "exchange_calendars", "version": exchange_calendars.__version__, "market": spec["market"],
                  "status": "provisional_derived_calendar_not_independent_official_exchange_attestation",
                  "handles": ["DST", "weekends", "holidays", "early_closes"], "official_review_required": True}


def stock_audit(probe, probe_dir, spec, args):
    if "listing" not in probe:
        return {"task": None, "text": None, "calendar_officially_reviewed": False,
                "status": "not_checked_incomplete_repository_probe", "probe_error": probe.get("probe_error")}
    universe = probe["listing"]["fixed_universe"]
    prices, files = {}, []
    for entry in probe.get("raw_files", []):
        if entry["path"].startswith("price/raw/") and entry["entity_id"]:
            path = safe_path(probe_dir, entry["response"]["raw_file"])
            try:
                prices[entry["entity_id"]] = stocknet.parse_prices(path.read_bytes())
            except SourceError as exc:
                files.append({"file": entry["path"], "error": str(exc)})
    if args.local_stocknet_root:
        from scripts.pivot_sources.probe import git_blob_sha
        for entry in probe["listing"]["files"]:
            if not entry["path"].startswith("price/raw/") or not entry["entity_id"]:
                continue
            path = safe_path(args.local_stocknet_root, entry["path"])
            if path.is_file():
                body = path.read_bytes()
                if git_blob_sha(body) != entry["git_blob_sha1"]:
                    raise SourceError("Local StockNet price differs from pinned inventory")
                prices[entry["entity_id"]] = stocknet.parse_prices(body)
                files.append({"file": str(path), "sha256": sha256(path)})
    calendar, calendar_info = build_calendar(spec, args.calendar_file, args.calendar_provider)
    official_evidence = read_json(args.calendar_evidence) if args.calendar_evidence else None
    official_calendar = bool(args.calendar_file and evidence_ok(official_evidence, ("official_exchange_calendar",))
                             and official_evidence.get("calendar_sha256") == sha256(args.calendar_file))
    tasks = stocknet.price_task(prices, universe, calendar, spec) if calendar else None
    text = stocknet.tweet_audit(probe.get("records", []), universe, calendar, spec)
    return {"price_schema_issues": files, "calendar": calendar_info, "calendar_official_evidence": official_evidence,
            "calendar_officially_reviewed": official_calendar, "task": tasks, "text": text,
            "created_at_is_not_verified_historical_receipt": True, "adjusted_close_risk": "not_cleared",
            "return_definitions": ["close_to_close", "log_return_candidate", "adjusted_close_return_diagnostic_only"],
            "origin_policy": spec["origin_policy"]}


def required_estimate(source, probe, spec):
    result = {"scope": "planning_only", "full_download_authorized": False, "known_bytes": None, "unknown_size_files": None,
              "authentication": "unknown_requires_service_documentation_review", "fees_or_quota": "unknown",
              "do_not_assume_anonymous_access": True, "full_coverage_status": "not_yet_full_coverage_verified"}
    if source == "stocknet":
        listing = probe.get("listing", {})
        result.update(known_bytes=listing.get("total_known_file_bytes"), unknown_size_files=listing.get("files_with_unknown_size"),
                      metadata_listing_truncated=listing.get("tree_truncated"),
                      extent="only files in actually visited directories, not a whole-repository estimate", git_clone_used=False,
                      inventory_complete=listing.get("inventory_complete", False), full_archive_bytes=None,
                      unvisited_directory_sizes="unknown")
    if source == "gfs":
        result["observed_file_sizes"] = [{"url": row["request"]["url"], "bytes": row["response"].get("advertised_bytes")}
                                         for row in probe.get("request_results", [])]
        result["planning_formula"] = "documented runs/day * actual covered days * selected leads * evidenced subset bytes; unknown factors remain null"
        result["global_GRIB_download_default"] = "forbidden"
    if source == "nws":
        result["planning_formula"] = "full metadata record count * measured raw text sizes; a few probe days cannot establish full count"
    return result
