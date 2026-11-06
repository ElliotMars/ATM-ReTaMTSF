"""Forecast-run metadata and deterministic forecast-only template design."""

from collections import defaultdict
from datetime import timedelta
import json
import math

from scripts.pivot_sources.asof import availability, eligible_records, utc
from scripts.pivot_sources.common import SourceError, basic_record, dotted, evidence_ok


def normalize(item, contract, url, raw_file):
    fields = contract["fields"]
    if contract.get("forecast_lead_unit") != "hours":
        raise SourceError("GFS lead units must be documented explicitly as hours")
    run = dotted(item, fields["run_time"])
    lead = dotted(item, fields["forecast_lead_hours"])
    if run is None or not isinstance(lead, (int, float)) or isinstance(lead, bool) or not math.isfinite(lead) or lead < 0:
        raise SourceError("Explicit aware run_time and a finite nonnegative hourly lead required")
    run = utc(run)
    valid = run + timedelta(hours=lead)
    declared_valid = dotted(item, fields.get("valid_time", "__missing__"))
    if declared_valid is not None and utc(declared_valid) != valid:
        raise SourceError("GFS valid time disagrees with initialization + lead")
    version = dotted(item, fields.get("model_version", "__missing__"))
    source_id = dotted(item, fields["source_record_id"])
    if not source_id:
        raise SourceError("GFS metadata record identity missing")
    publication = dotted(item, fields.get("file_publication_time", "__missing__"))
    if publication:
        utc(publication)
    initialization = dotted(item, fields.get("model_initialization_time", "__missing__"))
    init_derivation = "explicit_source_field" if initialization is not None else "unknown"
    if initialization is not None:
        initialization = utc(initialization).isoformat()
        if utc(initialization) != run:
            raise SourceError("Run and initialization differ; an explicit reviewed lead-time basis is required")
    elif evidence_ok(contract.get("time_evidence", {}).get("run_time"), ("model_initialization",)):
        initialization = run.isoformat()
        init_derivation = "reviewed_run_field_means_model_initialization"
    raw = json.dumps(item, ensure_ascii=False, sort_keys=True).encode()
    record = basic_record("jena", "gfs_archive", source_id, None, url, raw_file, raw,
                          run_time=run.isoformat(), source_time_key=run.isoformat(), timezone="UTC", location_id="jena",
                          valid_start=valid.isoformat(), valid_end=None, forecast_valid_time=valid.isoformat(),
                          forecast_lead_hours=lead, file_publication_time=publication,
                          model_initialization_time=initialization, model_version=version,
                          availability_evidence=contract.get("time_evidence", {}))
    record.update(origin_kind="model_forecast", run_id=f"{run.isoformat()}|{version or 'unverified_version_record:' + str(source_id)}",
                  model_initialization_time_derivation=init_derivation,
                  forecast_valid_time_derivation="run_time_plus_explicit_hourly_lead_checked_against_source_valid_field_if_present",
                  grid_resolution=dotted(item, fields.get("grid_resolution", "__missing__")),
                  variables=dotted(item, fields.get("variables", "__missing__")),
                  forecast_time_support="instant_point_not_hourly_observation",
                  access_metadata=dotted(item, fields.get("access_metadata", "__missing__")))
    return record


def run_scenarios(records, origins, horizons, lags=(3, 6, 12)):
    output = []
    for origin in origins:
        at = utc(origin)
        for lag in lags:
            verifiable = sum(availability(record, "run_lag", lag)["available_at"] is not None for record in records)
            pool = eligible_records(records, at, "run_lag", lag)
            runs = defaultdict(list)
            for record in pool:
                runs[record["run_id"]].append(record)
            # Each model/version is kept distinct; do not arbitrarily choose
            # between simultaneous variants or combine leads across versions.
            latest = max((utc(items[0]["run_time"]) for items in runs.values()), default=None)
            latest_runs = {key: items for key, items in runs.items() if latest is not None and utc(items[0]["run_time"]) == latest}
            for horizon in horizons:
                needed = {at + timedelta(hours=i) for i in range(horizon)}
                per_run = []
                for run_id, items in sorted(latest_runs.items()):
                    valid_points = {utc(item["forecast_valid_time"]) for item in items}
                    present = valid_points & needed
                    ordered = sorted(valid_points)
                    steps = sorted({(b - a).total_seconds() / 3600 for a, b in zip(ordered, ordered[1:])})
                    per_run.append({"run_id": run_id, "run_time": latest.isoformat(),
                                    "availability_policies": sorted({item["availability_policy"] for item in items}),
                                    "actual_publication_or_receipt_preferred": any(item.get("actual_publication_or_receipt_preferred", False) for item in items),
                                    "run_age_hours": (at - latest).total_seconds() / 3600,
                                    "raw_point_interval_hours": steps,
                                    "contains_legal_future_forecast": any(t >= at for t in valid_points),
                                    "covers_target_span": bool(ordered and ordered[0] <= at and ordered[-1] >= max(needed)),
                                    "covers_all_target_hours_exactly": present == needed,
                                    "matched_target_hours": len(present), "required_target_hours": horizon,
                                    "point_interpolation_performed": False})
                output.append({"forecast_origin": at.isoformat(), "lag_hours": lag,
                               "policy_status": "probe_only" if verifiable else "not_verifiable",
                               "records_with_evidenced_policy_time": verifiable,
                               "lag_status": "experiment_assumption_not_verified_publication_delay",
                               "horizon": horizon, "latest_available_run_variants": per_run,
                               "no_available_run": not bool(latest_runs),
                               "coverage_scope": "only actually probed lead metadata; absent leads are unverified"})
    return output


def metadata_matrix(records, spec):
    observed = {(utc(row["run_time"]).date().isoformat(), utc(row["run_time"]).hour, row["forecast_lead_hours"])
                for row in records}
    return {"requested_run_lead_matrix": [
        {"date": day, "cycle_utc_hour": cycle, "forecast_lead_hours": lead,
         "status": "observed_metadata_only" if (day, cycle, lead) in observed else "not_observed_in_probe_not_proven_missing"}
        for day in spec["probe_dates"] for cycle in spec["cycles_utc"] for lead in spec["leads_hours"]],
        "record_metadata": [{"source_record_id": row["source_record_id"], "run_id": row["run_id"],
                             "grid_resolution": row.get("grid_resolution"), "variables_and_levels": row.get("variables"),
                             "file_publication_time": row.get("file_publication_time"), "source_url": row["source_url"]} for row in records],
        "site": spec["site"], "site_to_grid_displacement": "unknown_until_station_and_actual_grid_coordinates_reviewed",
        "all_years_access_verified": False}


def forecast_text(points):
    """Template helper for one documented, extracted grid-point forecast.

    Does not access Jena truth or infer weather descriptions. Called only by
    future user tooling, never as part of the default metadata probe.
    """
    if not points:
        raise SourceError("No forecast points")
    keys = ("run_id", "run_time", "model_version", "grid_latitude", "grid_longitude")
    first = points[0]
    if any(point.get("origin_kind") != "model_forecast" or not point.get("raw_file_sha256")
           or any(point.get(key) != first.get(key) for key in keys) for point in points):
        raise SourceError("Template accepts only provenance-bound forecasts from the same run/version/grid point")
    for key in keys[:2]:
        if not first.get(key):
            raise SourceError("Run provenance missing")
    if any(not isinstance(first.get(key), (int, float)) or not math.isfinite(first[key]) for key in ("grid_latitude", "grid_longitude")):
        raise SourceError("Actual forecast grid coordinates required")
    sentences = []
    supported = {"temperature": {"K", "degC"}, "relative_humidity": {"%"}, "wind_speed": {"m/s"},
                 "surface_pressure": {"Pa", "hPa"}, "precipitation": {"m", "mm", "kg/m2"}}
    for point in sorted(points, key=lambda value: (utc(value["valid_time"]), value["variable"])):
        variable, unit, value = point["variable"], point["unit"], point["value"]
        if point.get("level") is None or not point.get("level_type"):
            raise SourceError("Forecast variable must retain its actual vertical level")
        if variable not in supported or unit not in supported[variable] or not math.isfinite(value):
            raise SourceError("Unsupported variable/unit/value; no invented weather description")
        run, valid = utc(point["run_time"]), utc(point["valid_time"])
        if run + timedelta(hours=point["forecast_lead_hours"]) != valid:
            raise SourceError("Template lead/time mismatch")
        if variable == "temperature" and unit == "K":
            value, unit = value - 273.15, "degC"
        if variable == "surface_pressure" and unit == "Pa":
            value, unit = value / 100, "hPa"
        if variable == "precipitation":
            if not point.get("accumulation_start") or not point.get("accumulation_end"):
                raise SourceError("Precipitation requires its actual accumulation period")
            if unit == "m":
                value, unit = value * 1000, "mm"
        sentences.append(f"Forecast valid {valid.isoformat()}, lead {point['forecast_lead_hours']} h: {variable} {value:.3f} {unit}, level {point['level']} {point['level_type']}."
                         + (f" Accumulation {point['accumulation_start']} to {point['accumulation_end']}." if variable == "precipitation" else ""))
    return {"text": " ".join(sentences), "run_id": first["run_id"],
            "raw_forecast_provenance": points, "uses_future_station_observations": False,
            "time_support": "listed forecast instants/accumulation intervals only; no hourly observation claim"}


def grib_metadata(path, max_messages=32):
    """Optional local eccodes inspection of an explicitly allowed tiny sample.

    No grid arrays or full-file expansion are read. Neither GRIB names nor
    HTTP Last-Modified establish historical publication availability.
    """
    try:
        import eccodes
    except ImportError as exc:
        raise SourceError("Optional eccodes package required for local GRIB inspection; no automatic installation") from exc
    rows = []
    names = ("dataDate", "dataTime", "forecastTime", "stepUnits", "startStep", "endStep", "validityDate", "validityTime",
             "shortName", "name", "typeOfLevel", "level", "units", "gridType", "Ni", "Nj",
             "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees")
    with open(path, "rb") as handle:
        for _ in range(max_messages):
            message = eccodes.codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                row = {}
                for name in names:
                    try:
                        row[name] = eccodes.codes_get(message, name)
                    except eccodes.CodesInternalError:
                        row[name] = None
                rows.append(row)
            finally:
                eccodes.codes_release(message)
    return {"messages": rows, "scan_limit": max_messages, "full_file_scanned": len(rows) < max_messages,
            "historical_publication_time": None, "site_extraction_performed": False,
            "variable_presence_scope": "inspected messages only; not an archive-wide claim"}
