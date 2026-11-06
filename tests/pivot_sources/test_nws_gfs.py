import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.pivot_sources.common import SourceError, load_config
from scripts.pivot_sources.gfs_archive import forecast_text, normalize as normalize_gfs, run_scenarios
from scripts.pivot_sources.nws_iem import bulletin_audit, normalize, product_id_parts
from scripts.pivot_sources.probe import validate_period
from scripts.pivot_sources import diagnostics
from scripts.pivot_sources.common import sha256, write_json
from scripts.fidel_ts.common import identity, load_source


def nws_row(evidence_factory, text="FXUS66 KMTR 151200 CCA\nAFDMTR\nDiscussion."):
    contract = {"fields": {"source_record_id": "id", "text": "text", "received_at": "received"},
                "time_evidence": {"product_id_time": evidence_factory("issuance"), "received_at": evidence_factory("historical_reception")}}
    item = {"id": "201807151200-KMTR-FXUS66-AFDMTR", "text": text, "received": "2018-07-15T12:04:00Z"}
    return normalize(item, contract, "https://example.org/nws", "raw/fixture.raw", ["AFDMTR", "AFDSGX"],
                     "2018-07-15T00:00:00Z", "2018-07-16T00:00:00Z")


def test_product_id_utc_syntax_has_no_automatic_semantic_verification():
    result = product_id_parts("202201011200-KMTR-FXUS66-AFDMTR")
    assert result["timestamp_candidate_utc"] == "2022-01-01T12:00:00+00:00"
    assert result["semantic_status"].startswith("syntax_only")


def test_nws_duplicates_updates_and_independent_receipt(evidence_factory):
    first = nws_row(evidence_factory)
    updated = nws_row(evidence_factory, "FXUS66 KMTR 151200 CCB\nAFDMTR\nChanged discussion.")
    assert first["document_id"] != updated["document_id"]
    report = bulletin_audit([first, first, updated])
    assert report["duplicate_product_id_occurrences"] == 2
    assert report["same_id_distinct_payloads"] == [first["source_record_id"]]
    assert len(report["updates_or_corrections"]) == 3
    assert first["issued_at"] != first["received_at"]
    assert first["available_at"] is None  # record has no selected protocol yet


def test_nws_different_offices_are_not_mixed(evidence_factory):
    first = nws_row(evidence_factory)
    second = {**first, "location_id": "KSGX", "afos_id": "AFDSGX"}
    assert bulletin_audit([first, second])["by_office"] == {"KMTR": 1, "KSGX": 1}


def test_nws_missing_text_and_out_of_range_rejected(evidence_factory):
    with pytest.raises(SourceError, match="text missing"):
        nws_row(evidence_factory, "")
    with pytest.raises(SourceError, match="outside"):
        validate_period("2014-01-01T00:00:00Z", "2014-01-02T00:00:00Z", load_config()["sources"]["nws"])


def test_nws_archive_fetch_time_cannot_supply_historical_receipt(evidence_factory):
    row = nws_row(evidence_factory)
    row["retrieved_at_now"] = "2026-11-05T00:00:00Z"
    assert row["received_at"] == "2018-07-15T12:04:00Z"
    assert not bulletin_audit([row])["historical_receipt_inferred_from_download_time"]


def gfs_row(evidence_factory, run="2021-01-01T00:00:00Z", lead=24, version="fixture-v1"):
    contract = {"fields": {"source_record_id": "id", "run_time": "run", "forecast_lead_hours": "lead", "model_version": "version"},
                "forecast_lead_unit": "hours", "time_evidence": {"run_time": evidence_factory("model_initialization")}}
    return normalize_gfs({"id": f"{run}/{lead}/{version}", "run": run, "lead": lead, "version": version}, contract,
                         "https://example.org/gfs", "raw/fixture.raw")


def test_gfs_run_lead_and_valid_time(evidence_factory):
    row = gfs_row(evidence_factory, lead=72)
    assert row["run_time"] == "2021-01-01T00:00:00+00:00"
    assert row["forecast_valid_time"] == "2021-01-04T00:00:00+00:00"
    assert row["available_at"] is None and row["file_publication_time"] is None


@pytest.mark.parametrize("lag,available", [(3, True), (6, True), (12, False)])
def test_gfs_prespecified_publication_lag_scenarios(evidence_factory, lag, available):
    report = run_scenarios([gfs_row(evidence_factory)], ["2021-01-01T06:00:00Z"], [24], [lag])[0]
    assert report["no_available_run"] is not available
    assert report["lag_status"].startswith("experiment_assumption")


def test_future_runs_excluded_and_latest_past_run_selected(evidence_factory):
    rows = [gfs_row(evidence_factory), gfs_row(evidence_factory, "2021-01-01T06:00:00Z"),
            gfs_row(evidence_factory, "2021-01-01T18:00:00Z")]
    report = run_scenarios(rows, ["2021-01-01T10:00:00Z"], [24], [3])[0]
    assert report["latest_available_run_variants"][0]["run_time"] == "2021-01-01T06:00:00+00:00"


def test_gfs_short_or_missing_forecast_does_not_cover_hourly_targets(evidence_factory):
    row = run_scenarios([gfs_row(evidence_factory)], ["2021-01-01T06:00:00Z"], [72], [3])[0]
    assert not row["latest_available_run_variants"][0]["covers_all_target_hours_exactly"]
    assert run_scenarios([], ["2021-01-01T06:00:00Z"], [24], [3])[0]["no_available_run"]


def test_gfs_versions_not_merged(evidence_factory):
    rows = [gfs_row(evidence_factory, version="v1"), gfs_row(evidence_factory, version="v2")]
    report = run_scenarios(rows, ["2021-01-01T06:00:00Z"], [24], [3])[0]
    assert len(report["latest_available_run_variants"]) == 2


def test_gfs_verified_publication_takes_priority_over_optimistic_lag(evidence_factory):
    row = gfs_row(evidence_factory)
    row["file_publication_time"] = "2021-01-01T10:00:00Z"
    row["availability_evidence"]["file_publication_time"] = evidence_factory("historical_publication")
    assert run_scenarios([row], ["2021-01-01T06:00:00Z"], [24], [3])[0]["no_available_run"]


def test_forecast_template_rejects_observations_and_keeps_units_and_leads():
    point = {"origin_kind": "model_forecast", "raw_file_sha256": "a" * 64, "run_id": "fixture",
             "run_time": "2021-01-01T00:00:00Z", "model_version": "v1", "grid_latitude": 51., "grid_longitude": 12.,
             "valid_time": "2021-01-02T00:00:00Z", "forecast_lead_hours": 24, "variable": "temperature", "unit": "K", "value": 273.15}
    point.update(level=2, level_type="heightAboveGround")
    text = forecast_text([point])
    assert "0.000 degC" in text["text"] and not text["uses_future_station_observations"]
    with pytest.raises(SourceError, match="only provenance-bound forecasts"):
        forecast_text([{**point, "origin_kind": "future_station_observation"}])


def synthetic_n2(tmp_path, monkeypatch, validation_start="2021-01-01"):
    source = load_source("caiso")
    directory = tmp_path / "processed"
    directory.mkdir()
    meta = identity("caiso", source)
    write_json(directory / "manifest.json", {**meta, "protocol": "n2", "mode": "prepare", "primary_sha256": "a" * 64})
    write_json(directory / "channels.json", {"channels": source["expected_channels"]})
    for split, start in (("train", "2020-01-01"), ("validation", validation_start), ("test", "2022-01-01")):
        path = directory / split
        path.mkdir()
        pd.DataFrame({"timestamp": pd.date_range(start, periods=6, freq="h"), "native_segment_id": [0] * 6}).to_parquet(path / "timestamps.parquet", index=False)
        pd.DataFrame(np.ones((6, 20), dtype=bool), columns=source["expected_channels"]).to_parquet(path / "validity_mask.parquet", index=False)
    artifacts = {path.relative_to(directory).as_posix(): sha256(path) for path in directory.rglob("*") if path.is_file()}
    write_json(directory / "_SUCCESS.json", {**meta, "primary_sha256": "a" * 64, "artifacts": artifacts})
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    spec = copy.deepcopy(load_config()["sources"]["nws"])
    spec.update(n2_directory="processed", lookback=2, horizons=[2], probe_dates=[validation_start])
    return spec, directory


def test_naive_n2_is_blocked_until_primary_bound_clock_evidence(tmp_path, monkeypatch, evidence_factory):
    spec, directory = synthetic_n2(tmp_path, monkeypatch)
    original = sha256(directory / "validation/timestamps.parquet")
    blocked = diagnostics.n2_origins(spec)
    assert blocked["clock_status"].startswith("blocked") and blocked["origins"] == []
    evidence = {**evidence_factory("numerical_clock"), "timezone": "UTC", "bound_primary_sha256": "a" * 64}
    checked = diagnostics.n2_origins(spec, evidence)
    assert len(checked["origins"]) == 3
    assert checked["channel_count"] == 20
    assert sha256(directory / "validation/timestamps.parquet") == original


def test_dst_ambiguity_cannot_be_inferred_away(tmp_path, monkeypatch, evidence_factory):
    spec, _ = synthetic_n2(tmp_path, monkeypatch, "2021-11-07")
    evidence = {**evidence_factory("numerical_clock"), "timezone": "America/New_York", "bound_primary_sha256": "a" * 64}
    result = diagnostics.n2_origins(spec, evidence)
    assert result["clock_status"] == "blocked_ambiguous_or_nonexistent_local_clock"
    assert result["origins"] == []
