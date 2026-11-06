"""Shared Stage 1.6 CLI. Only mode=probe plus --allow-network can use HTTP."""

import argparse
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys

from scripts.pivot_sources.asof import availability, utc
from scripts.pivot_sources.common import (
    CONFIG, ROOT, SourceError, digest, load_config, load_plan, output_run, read_bundle,
    read_json, read_jsonl, readiness, safe_path, seal, sha256, write_json, write_jsonl,
)
from scripts.pivot_sources.diagnostics import (
    n2_origins, nws_coverage, policy_audit, required_estimate, stock_audit,
)
from scripts.pivot_sources.http import BoundedHTTP
from scripts.pivot_sources import gfs_archive, nws_iem
from scripts.pivot_sources.probe import planned_probe, stock_probe, validate_period


def parser_for(source):
    parser = argparse.ArgumentParser(description=f"Stage 1.6 {source}: explicit probe or offline audit/coverage")
    parser.add_argument("--mode", choices=("dry-run", "probe", "audit", "coverage"), default="dry-run")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, help="Previous sealed probe/audit bundle")
    parser.add_argument("--request-plan", type=Path, help="Evidence-bound exact historical request URLs and response contract")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Only interrupted, unsealed HTTP probe, with identical binding")
    for option in ("max-requests", "max-file-bytes", "max-total-bytes", "max-records", "retries"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--min-interval-seconds", type=float)
    parser.add_argument("--start", help="Timezone-aware inclusive source-time bound, at most two days")
    parser.add_argument("--end", help="Timezone-aware exclusive bound")
    parser.add_argument("--afos-id", action="append", choices=("AFDMTR", "AFDSGX"))
    parser.add_argument("--origin", action="append", default=[], help="Explicit aware probe origin; never establishes N2 clock alignment")
    parser.add_argument("--timezone-evidence", type=Path)
    parser.add_argument("--allow-small-sample", action="store_true")
    parser.add_argument("--grib-inspect", action="store_true")
    parser.add_argument("--revision", default="HEAD", help="StockNet HEAD resolution or fixed commit SHA")
    parser.add_argument("--max-price-files", type=int, default=2)
    parser.add_argument("--max-tweet-files", type=int, default=4)
    parser.add_argument("--max-tweet-directories", type=int, default=2)
    parser.add_argument("--tweet-time-evidence", type=Path)
    parser.add_argument("--calendar-file", type=Path)
    parser.add_argument("--calendar-evidence", type=Path)
    parser.add_argument("--calendar-provider", choices=("none", "exchange_calendars"), default="none")
    parser.add_argument("--local-stocknet-root", type=Path, help="Optional existing pinned raw prices; never downloads full data")
    return parser


def baseline_payload(source, spec, probe):
    records = probe.get("records", [])
    # All gates start unverified; reference-page access is not data access.
    return {
        "source_inventory.json": {"source": source, "dataset": spec["dataset"], "probe": probe,
                                   "full_coverage_status": "not_yet_full_coverage_verified"},
        "sample_records.jsonl": records,
        "timestamp_semantics.json": {"issued_at": "source issuance/creation if evidenced",
            "received_at": "independent historical receipt only, never current fetch time",
            "run_time": "model initialization, not public availability",
            "file_publication_time": "historical file publication only if supported by independent evidence",
            "valid_start_end": "forecast target time support, not issue/receipt",
            "retrieved_at_now": "current request time, not historical availability",
            "source_time_key": "source index, never automatic available_at",
            "naive_numerical_clock_policy": "blocked_until_evidence_review",
            "availability_statuses": ["verified_source_time", "verified_received_time", "latency_assumption", "unknown"]},
        "asof_policy_audit.json": {"policies": policy_audit(records, source, spec["lag_hours"]),
                                   "filter_before_retrieval_and_similarity": True},
        "availability_scenarios.json": {"source": source, "lag_hours": spec["lag_hours"],
                                         "lag_semantics": "prespecified_experiment_assumptions_not_verified_delays",
                                         "origin_results": [], "status": "not_checked"},
        "coverage_probe.json": {"status": "not_checked", "full_coverage_status": "not_yet_full_coverage_verified",
                                "probe_dates": spec["probe_dates"], "population_rates": None},
        "required_download_estimate.json": required_estimate(source, probe, spec),
        "known_risks.json": {"risks": ["probe samples cannot verify all-period coverage",
            "source timestamp does not imply verified historical receipt",
            "human semantic evidence review still required"], "parse_issues": probe.get("parse_issues", [])},
        "source_readiness.json": readiness(evidence=["source_inventory.json", "timestamp_semantics.json"],
                                             issues=["probe scope only; task and temporal alignment not yet accepted"])}


def update_decision(payload, source, spec, probe, task=None):
    records = probe.get("records", [])
    actual = probe.get("actual_data_access_checked", False)
    source_gate = "blocked" if probe.get("probe_error") else "conditional" if actual else "not_checked"
    issues = []
    if probe.get("probe_error"):
        issues.append("probe incomplete; access/contract/budget error: " + probe["probe_error"])
    policies = payload["asof_policy_audit.json"]["policies"]
    strict_count = next((row["strict_verified_records"] for row in policies if row["policy"] == "strict"), 0)
    conditional_count = max((row["records_with_available_at"] for row in policies if row["policy"] != "strict"), default=0)
    temporal = "conditional" if strict_count or conditional_count else "not_checked"
    task_gate = "not_checked"
    if task is not None and source in {"nws", "gfs"}:
        blocked = task["clock_status"].startswith("blocked")
        task_gate = "blocked" if blocked else "conditional"
        if blocked:
            issues.append("numerical timezone/UTC correspondence unverified; strict numeric-text alignment blocked")
        empty_windows = [f"{split}/H{horizon}" for split, horizons in task["splits"].items() for horizon, result in horizons.items()
                         if not result["published_window_indices"]]
        if empty_windows:
            task_gate = "blocked"
            issues.append("no eligible N2 windows: " + ", ".join(empty_windows))
    if source == "stocknet":
        listing = probe.get("listing", {})
        if listing.get("tree_truncated"):
            issues.append("Git tree truncated; file/universe counts incomplete")
        if listing.get("universe_count") != 88:
            issues.append("fixed 88-stock universe not established; no automatic target reduction")
        if task:
            numeric = task.get("task")
            task_gate = "conditional" if numeric and numeric["all_88_price_series_loaded"] else "not_checked"
            if numeric and numeric["all_88_price_series_loaded"]:
                empty = [f"{split}/H{h}" for split, row in numeric["per_split_windows"].items()
                         for h, stats in row["horizons"].items() if stats["full_universe_valid_label_windows"] == 0]
                if empty:
                    task_gate = "blocked"
                    issues.append("full 88-stock labels have no complete windows: " + ", ".join(empty))
            else:
                issues.append("only sampled price files; common 88-stock trading days remain unverified")
            if not task.get("calendar_officially_reviewed"):
                issues.append("calendar needs independent official exchange evidence; library-derived schedule is provisional")
            issues.append("historical price adjustment/corporate-action mechanism not cleared")
    if not records:
        issues.append("no valid historical records observed; documentation-only result is not data validation")
    if not strict_count:
        issues.append("Strict Verified historical availability not established")
    issues.append("full-period coverage/index not verified; no unrestricted large download approval")
    # No bounded probe alone can promote all gates to pass. A future full-index
    # acceptance step is deliberately outside this Stage 1.6 implementation.
    payload["source_readiness.json"] = readiness(source_gate, temporal, task_gate,
        evidence=["source_inventory.json", "asof_policy_audit.json", "coverage_probe.json"], issues=issues,
        conditional=bool(conditional_count and task_gate == "conditional"))


def audit_payload(source, spec, prior, prior_dir, args):
    probe = prior["probe"]
    payload = baseline_payload(source, spec, probe)
    task = None
    if source in {"nws", "gfs"}:
        clock = read_json(args.timezone_evidence) if args.timezone_evidence else None
        task = n2_origins(spec, clock)
        payload["timestamp_semantics.json"]["numerical_alignment_audit"] = task
        if source == "nws":
            payload["timestamp_semantics.json"]["bulletin_audit"] = nws_iem.bulletin_audit(probe.get("records", []))
        if source == "gfs":
            payload["timestamp_semantics.json"]["gfs_metadata_probe"] = gfs_archive.metadata_matrix(probe.get("records", []), spec)
            payload["known_risks.json"]["risks"].extend([
                "2014 Jena history precedes candidate archive start; affected windows must be counted",
                "run initialization is not publication", "3/12-hour forecast points do not become hourly observations",
                "Jena station coordinates and forecast grid-point displacement require evidence",
                "same as-of GFS runs and variables must be shared by numerical/text baselines"])
            # Count exclusion under the declared candidate start without claiming
            # that the archive actually covers every later day.
            payload["coverage_probe.json"]["pre_archive_training_impact"] = {
                "candidate_archive_start": spec["archive_start_candidate"],
                "status": "requires_timezone_and_actual_archive_metadata",
                "native_calendar_window_counts_by_horizon": {
                    horizon: {"pre_archive_origins": stats["forecast_origins_before_candidate_archive_calendar"],
                              "total_eligible_origins": stats["published_window_indices"]}
                    for horizon, stats in task["splits"]["train"].items()},
                "policy": "retain 2014 numeric history; no fabricated GFS text; report no-external-text mask"}
            if args.grib_inspect:
                payload["timestamp_semantics.json"]["tiny_grib_samples"] = [
                    {"raw_file": name, "metadata": gfs_archive.grib_metadata(safe_path(prior_dir, name))}
                    for name in probe.get("raw_samples", [])]
    else:
        task = stock_audit(probe, prior_dir, spec, args)
        payload["timestamp_semantics.json"]["stocknet_audit"] = task
    update_decision(payload, source, spec, probe, task)
    return payload, task


def run(source, argv=None):
    parser = parser_for(source)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        spec = deepcopy(config["sources"][source])
        limits = dict(config["limits"])
        for key in limits:
            value = getattr(args, key, None)
            if value is not None:
                limits[key] = value
        args.max_records = limits["max_records"]
        if args.max_records < 1 or min(args.max_price_files, args.max_tweet_files) < 0:
            raise SourceError("Record caps must be positive and file counts nonnegative")
        if limits["max_requests"] > 64 or limits["max_file_bytes"] > 8 * 1024**2 or limits["max_total_bytes"] > 32 * 1024**2 or args.max_records > 1000:
            raise SourceError("Stage 1.6 hard probe ceilings: 64 attempts, 8 MiB/file, 32 MiB total, 1000 records")
        if args.max_price_files > 8 or args.max_tweet_files > 16:
            raise SourceError("StockNet probe may fetch at most 8 price and 16 tweet files, never a full clone")
        if not 0 <= args.max_tweet_directories <= 4:
            raise SourceError("StockNet directory probe is limited to 0..4 tweet entity directories")
        if bool(args.start) != bool(args.end):
            raise SourceError("Supply both --start and --end")
        if args.start:
            if source == "stocknet":
                raise SourceError("StockNet samples use discovered file dates; use the fixed candidate split")
            start, end = validate_period(args.start, args.end, spec)
            days = []
            day = start
            while day < end:
                days.append(day.date().isoformat())
                day += timedelta(days=1)
            spec["probe_dates"] = days
        for origin in args.origin:
            utc(origin)
        if args.mode == "dry-run":
            print(json.dumps({"source": source, "dry_run": True, "network": False, "config": spec,
                              "limits": limits, "output": str(args.output_dir),
                              "historical_interface": "NWS/GFS require saved-doc evidence request plan; StockNet resolves/pins GitHub commit",
                              "data_or_source_validation_executed": False}, indent=2, ensure_ascii=False))
            return 0
        if args.output_dir is None:
            raise SourceError("--output-dir is required")
        if args.allow_network and args.mode != "probe":
            raise SourceError("Network may be enabled only for mode=probe")
        if args.resume and args.mode != "probe":
            raise SourceError("Only interrupted probes may resume")
        if args.mode == "probe" and not args.allow_network:
            raise SourceError("Probe requires explicit --allow-network; use --mode dry-run offline")
        plan = load_plan(args.request_plan, source) if args.mode == "probe" else None
        binding = {"schema_version": 1, "stage": "1.6", "source": source, "mode": args.mode,
                   "config_digest": digest(config), "effective_spec": spec, "limits": limits,
                   "request_plan_sha256": sha256(args.request_plan) if args.request_plan else None,
                   "evidence_input_hashes": {name: sha256(path) for name, path in (
                       ("timezone_evidence", args.timezone_evidence), ("calendar_file", args.calendar_file),
                       ("calendar_evidence", args.calendar_evidence), ("tweet_time_evidence", args.tweet_time_evidence)) if path},
                   "input_bundle_sha256": None, "cli_options": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items() if key not in {"resume", "output_dir", "allow_network"}}}
        prior = None
        if args.mode in {"audit", "coverage"}:
            if args.input_dir is None:
                raise SourceError("Audit/coverage requires --input-dir")
            marker = read_bundle(args.input_dir)
            if marker.get("source") != source or marker.get("config_digest") != digest(config):
                raise SourceError("Input source/config mismatch")
            binding["input_bundle_sha256"] = digest(marker)
            prior = read_json(args.input_dir / "source_inventory.json")
            if not args.start:
                spec["probe_dates"] = marker["effective_spec"]["probe_dates"]
                binding["effective_spec"] = spec
            # Raw paths belong to the original probe, never the derived audit.
            raw_root = Path(prior.get("raw_bundle_directory", str(args.input_dir))).resolve()
            read_bundle(raw_root)
        else:
            raw_root = args.output_dir.resolve()
        inputs = [path for path in (args.input_dir, args.local_stocknet_root) if path]
        with output_run(args.output_dir, binding, args.resume, inputs) as output:
            incomplete = False
            if args.mode == "probe":
                client = BoundedHTTP(output, spec["allowed_hosts"], limits, allow_network=True)
                try:
                    probe = stock_probe(spec, client, args) if source == "stocknet" else planned_probe(source, spec, client, plan, args)
                except (SourceError, ValueError, KeyError, TypeError) as exc:
                    incomplete = True
                    probe = {"probe_error": str(exc), "records": [], "parse_issues": [],
                             "actual_data_access_checked": False, "contract_status": "not_checked_incomplete_probe",
                             "note": "raw successful responses remain in HTTP manifest; partial records are not invented"}
                    with (output / "failures.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"phase": "probe", "error": str(exc)}) + "\n")
                payload = baseline_payload(source, spec, probe)
                payload["source_inventory.json"]["http_usage"] = {key: client.state[key] for key in ("requests_used", "bytes_received", "bytes_budget_charged")}
                update_decision(payload, source, spec, probe)
            else:
                payload, task = audit_payload(source, spec, prior, raw_root, args)
                probe = prior["probe"]
                if args.mode == "coverage":
                    if source in {"nws", "gfs"}:
                        origins = task["origins"]
                        payload["coverage_probe.json"].update(status="probe_only" if origins else "blocked_clock_alignment",
                            numerical_window_counts=task["splits"], clock_status=task["clock_status"],
                            probe_forecast_origins=len(origins), population_rates=None)
                        if source == "nws":
                            if not probe.get("records"):
                                payload["coverage_probe.json"]["status"] = "not_checked_no_historical_records"
                            payload["coverage_probe.json"]["office_horizon_policies"] = nws_coverage(probe.get("records", []), origins, spec)
                        else:
                            times = sorted({row["forecast_origin"] for row in origins} | set(args.origin))
                            payload["availability_scenarios.json"].update(status="probe_only" if times else "not_checked_no_aligned_origins",
                                origin_results=gfs_archive.run_scenarios(probe.get("records", []), times, spec["horizons"], spec["lag_hours"]),
                                explicit_origins_do_not_verify_numerical_clock=bool(args.origin))
                    else:
                        payload["coverage_probe.json"].update(status="not_checked_incomplete_probe" if probe.get("probe_error") else "sample_only", stocknet=task,
                                                              population_rates=None)
            payload["source_inventory.json"]["raw_bundle_directory"] = str(raw_root)
            for name, value in payload.items():
                if name.endswith(".jsonl"):
                    write_jsonl(output / name, value)
                else:
                    write_json(output / name, value)
            seal(output, binding, incomplete=incomplete)
        print(json.dumps({"source": source, "output": str(args.output_dir), **payload["source_readiness.json"]}, indent=2, ensure_ascii=False))
        return 2 if incomplete else 0
    except (SourceError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
