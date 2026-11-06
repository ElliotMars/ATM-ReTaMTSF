import copy
import json
from pathlib import Path

import pandas as pd

from scripts.fidel_ts.common import load_source, read_json
from scripts.fidel_ts.inspection import JSONObject, normalize_documents
from scripts.fidel_ts.stage1_5_io import PROTOCOL_CONFIG
from scripts.fidel_ts.text_provenance import (
    access_decision, extract_content_cues, include_empty_source_reports,
    reconstruct_reports, report_identity, stratified_samples, version_comparison,
)
from scripts.fidel_ts.weather_evidence import code_evidence, observed_clock


def write_ledger(tmp_path, spec, dataset, records):
    documents = []
    files = {}
    for source, key, text in records:
        files[source] = {"status": "complete"}
        documents.extend(normalize_documents(dataset, spec, source, key, JSONObject(text), 0))
    ledger = tmp_path / "documents.jsonl"
    ledger.write_text("".join(json.dumps(document) + "\n" for document in documents))
    return ledger, {"files": files}


def test_multiple_facets_share_one_report_id(spec, tmp_path):
    source = "weather/weather_report/formal_report/wm_messages_v1.json"
    ledger, manifest = write_ledger(tmp_path, spec, "jena", [(source, "202001010000", [("Time of Day", "Morning"), ("Weather Trend", "Steady")])])
    reports = reconstruct_reports(ledger, "jena", spec, manifest)
    assert len(reports) == 1 and len(reports[0]["facets"]) == 2
    assert len({f["facet_id"] for f in reports[0]["facets"]}) == 2
    assert reports[0]["forecast_origin"] is None


def test_caiso_locations_never_mix_report_identities(tmp_path):
    spec = load_source("caiso")
    records = [(f"weather/{city}/weather_report/general_report/general_weather_forecast_2020.json",
                "20200101", [("daily", "Clear")]) for city in ("san-diego", "san-francisco")]
    ledger, manifest = write_ledger(tmp_path, spec, "caiso", records)
    reports = reconstruct_reports(ledger, "caiso", spec, manifest)
    assert len({report["report_id"] for report in reports}) == 2
    assert {r["location_id"] for r in reports} == {"san-diego", "san-francisco"}


def test_jena_variants_have_no_historical_release_order(spec, tmp_path):
    records = [(f"weather/weather_report/formal_report/wm_messages_{version}.json", "202001010000",
                [("Weather Trend", text)]) for version, text in (("v1", "Clear"), ("v2", "Cloudy"), ("v3", "Clear"))]
    ledger, manifest = write_ledger(tmp_path, spec, "jena", records)
    reports = reconstruct_reports(ledger, "jena", spec, manifest)
    assert len({report["report_id"] for report in reports}) == 3
    comparisons = version_comparison(reports)
    assert len(comparisons) == 3
    assert all(item["historical_release_order"] == "unknown" for item in comparisons)
    assert all(item["version_generation_time"] is None for item in comparisons)


def test_source_key_never_becomes_available_at_or_forecast_origin(spec, tmp_path):
    source = "weather/weather_report/formal_report/wm_messages_v1.json"
    ledger, manifest = write_ledger(tmp_path, spec, "jena", [(source, "202001010000", [("Weather Trend", "Clear")])])
    report = reconstruct_reports(ledger, "jena", spec, manifest)[0]
    assert report["available_at"] is None and report["issued_at"] is None
    assert report["valid_start"] is None and report["valid_end"] is None
    decision = access_decision(report, "2020-01-01T01:00:00", "2020-01-02T01:00:00")
    assert decision["T1"] == "blocked_not_verifiable"
    assert decision["T2_key_only_eligible"]
    future = access_decision(report, "2019-12-31T23:00:00", "2020-01-01T01:00:00")
    assert not future["T2_key_only_eligible"] and future["T3_key_aligned"]
    assert future["T3"] == "retrospective_oracle_risk"


def test_same_key_different_occurrences_and_empty_reports_are_distinct(spec, tmp_path):
    source = "weather/weather_report/formal_report/wm_messages_v1.json"
    ledger, manifest = write_ledger(tmp_path, spec, "jena", [(source, "202001010000", [("Weather Trend", "Clear")])])
    reports = reconstruct_reports(ledger, "jena", spec, manifest)
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    path.write_text('{"202001010000":{"Weather Trend":"Clear"},"202001010000":null}')
    expanded = include_empty_source_reports(reports, tmp_path, "jena", spec, [source])
    assert len(expanded) == 2
    assert len({r["report_id"] for r in expanded}) == 2
    assert sum(len(r["facets"]) for r in expanded) == 1


def test_seeded_strata_cover_train_validation_test_and_seasons(spec, tmp_path):
    source = "weather/weather_report/formal_report/wm_messages_v1.json"
    records = [(source, key, [("Weather Trend", "Clear")]) for key in (
        "202001010000", "202004010000", "202007010000", "202010010000", "202101010000", "202201010000")]
    ledger, manifest = write_ledger(tmp_path, spec, "jena", records)
    reports = reconstruct_reports(ledger, "jena", spec, manifest)
    config = read_json(PROTOCOL_CONFIG)
    first, strata = stratified_samples(reports, config)
    second, _ = stratified_samples(list(reversed(reports)), config)
    assert [r["report_id"] for _, r in first] == [r["report_id"] for _, r in second]
    assert {key[0] for key in strata} == {"train", "validation", "test"}
    assert {key[2] for key in strata} == {"DJF", "MAM", "JJA", "SON"}


def test_observation_period_is_not_forecast_issue_time():
    clock = observed_clock({"ds": "Monday, 1 January 2018, 00:00 — 06:00", "temp": 12})
    assert clock["start"] == "2018-01-01T00:00:00"
    assert clock["role"] == "observation_period_label_not_forecast_validity"
    assert not clock["timezone_verified"]
    assert observed_clock({"date": 1514764800000.0, "temp": 12}) is None


def test_script_inspection_does_not_execute_code(tmp_path):
    script = tmp_path / "producer.py"
    side_effect = tmp_path / "must_not_exist"
    script.write_text(f"raise RuntimeError('do not execute')\nopen({str(side_effect)!r}, 'w')\nx.interpolate(method='linear')\n")
    evidence = code_evidence(script, "producer.py")
    assert not side_effect.exists()
    assert evidence["calls"][0]["call"] == "interpolate"
    assert evidence["text_generation_from_future_observations_proven"] is False


def test_weather_cues_are_not_leakage_verdicts():
    report = {"facets": [{"text": "Temperature 12-18 degrees Celsius, wind 3 m/s, humidity 80%, rising."}]}
    cues = extract_content_cues(report)
    assert cues["temperature_ranges_degc"] == [[12., 18.]]
    assert cues["wind_values"] == [{"value": 3., "unit": "m/s"}]
    assert cues["percentage_mentions_unverified_role"] == [80.]
    assert cues["role_and_time_coverage_inference"] == "not_verified"
