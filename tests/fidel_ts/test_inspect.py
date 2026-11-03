import io
import json

import numpy as np
import pandas as pd
import pytest

from scripts.fidel_ts.common import load_source, sha256_file
from scripts.fidel_ts.inspection import (
    JSONObject, availability_assessment, caiso_alignment, channel_mapping, normalize_documents,
    numerical_summary, text_summary,
)
from scripts.fidel_ts.inspect import inspect_dataset


def test_duplicate_times_sentinel_nan_inf_and_channel_mapping(spec, tmp_path):
    frame = pd.DataFrame({
        "Date Time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:10", "2020-01-01 00:10", "2020-01-01 00:40"]),
        "wv (m/s)": [-9999.0, np.nan, np.inf, 2.0],
        "rho (g/m**3)": [1., 2., 2., 4.],
    })
    path = tmp_path / "tiny.parquet"
    frame.to_parquet(path, index=False)
    original = sha256_file(path)
    static = {"channel_info": {"weather_large": {"wv (m/s)": "Wind", "rho (g/m³)": "Density"}}}
    result = numerical_summary(path, static, spec)
    assert result["duplicate_timestamp_rows_beyond_first"] == 1
    assert result["sorted_unique_gap_intervals"] == 1
    assert result["grid_aligned_missing_points_within_gaps"] == 2
    wind = result["channels"][0]
    assert wind["sentinel_minus_9999"] == 1
    assert wind["missing_including_nan"] == 1
    assert wind["positive_inf"] == 1
    assert result["static_mapping"]["mapping"][1]["match"] == "explicit_alias"
    assert sha256_file(path) == original
    # The summary must be serializable without NaN or NumPy scalars.
    json.dumps(result, allow_nan=False)


def test_unknown_channel_not_fuzzily_matched(spec):
    result = channel_mapping(["mystery", "SWDR (W/m�)"], {
        "channel_info": {"weather_large": {"SWDR (W/m²)": "Radiation"}}}, spec)
    assert result["unmapped_channels"] == ["mystery"]
    assert result["mapping"][1]["description_channel_id"] == "SWDR (W/m²)"


def test_caiso_twenty_channels_and_negative_values_are_only_reported(tmp_path):
    spec = load_source("caiso")
    frame = pd.DataFrame({"Time": pd.date_range("2020-01-01", periods=3, freq="5min"),
                          **{name: [0., 1., 2.] for name in spec["expected_channels"]}})
    frame["Current demand"] = [-1., 2., 3.]
    path = tmp_path / "caiso.parquet"
    frame.to_parquet(path, index=False)
    result = numerical_summary(path, {"channel_info": {"all": {
        name: name for name in spec["expected_channels"]}}}, spec)
    assert result["numeric_channel_count"] == 20
    assert result["missing_expected_channels"] == []
    demand = next(item for item in result["channels"] if item["name"] == "Current demand")
    assert demand["negative_finite_values"] == 1
    assert demand["negative_finite_row_examples"] == [0]
    assert pd.read_parquet(path)["Current demand"].iloc[0] == -1.


def test_caiso_alignment_checks_primary_and_selected_profile(tmp_path):
    names = [f"raw_data/all_channel_imputed/{name}.parquet" for name in ("co2", "demand", "fuelsource")]
    names.append("time_series/all_except_battery.parquet")
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Time": pd.date_range("2020-01-01", periods=3, freq="5min")}).to_parquet(path, index=False)
    assert caiso_alignment(tmp_path, names)["time_axes_exactly_equal_including_order"]
    primary = tmp_path / names[-1]
    pd.DataFrame({"Time": pd.date_range("2020-01-02", periods=3, freq="5min")}).to_parquet(primary, index=False)
    assert not caiso_alignment(tmp_path, names)["time_axes_exactly_equal_including_order"]
    assert caiso_alignment(tmp_path, [names[-1]])["status"] == "not_checked_not_selected_by_profile"


def test_same_date_multiple_locations_keep_distinct_ids(tmp_path):
    spec = load_source("caiso")
    paths = []
    for city in ("san-diego", "san-francisco"):
        name = f"weather/{city}/weather_report/general_report/general_weather_forecast_2020.json"
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text('{"20200101":{"daily":"Cloudy"}}')
        paths.append(name)
    handle = io.StringIO()
    summary, availability = text_summary(tmp_path, paths, "caiso", spec, handle)
    docs = [json.loads(line) for line in handle.getvalue().splitlines()]
    assert len({doc["document_id"] for doc in docs}) == 2
    assert {doc["location_id"] for doc in docs} == {"san-diego", "san-francisco"}
    assert summary["source_time_keys_with_multiple_locations"] == 1
    assert availability["documents_missing_available_at"] == 2
    assert all(doc["available_at"] is None and doc["valid_start"] is None for doc in docs)


def test_duplicate_json_keys_and_missing_fields_are_preserved(spec, tmp_path):
    name = "weather/weather_report/formal_report/wm_messages_v1.json"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text('{"202001010000":{"Weather Trend":"A"},"202001010000":{"Weather Trend":"B"},"202001010600":null}')
    handle = io.StringIO()
    summary, _ = text_summary(tmp_path, [name], "jena", spec, handle)
    docs = [json.loads(line) for line in handle.getvalue().splitlines()]
    assert len(docs) == 2 and docs[0]["document_id"] != docs[1]["document_id"]
    assert summary["duplicate_time_keys_within_files"] == 1
    assert summary["files"][0]["diagnostics"]["records_without_text"] == 1
    assert summary["files"][0]["missing_expected_field_record_counts"]["Time of Day"] == 2


def test_jena_variants_are_not_release_vintages(spec, tmp_path):
    paths = []
    for version, text in (("v1", "Calm"), ("v2", "Windy"), ("v3", "Calm")):
        name = f"weather/weather_report/formal_report/wm_messages_{version}.json"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"202001010000": {"Wind Condition": text}}))
        paths.append(name)
    ledger = io.StringIO()
    summary, availability = text_summary(tmp_path, paths, "jena", spec, ledger)
    comparisons = summary["version_differences"]
    assert len(comparisons) == 3
    assert all(item["chronological_release_order_verified"] is False for item in comparisons)
    assert any(item["different_record_texts"] for item in comparisons)
    assert availability["status_counts"] == {"verified": 0, "protocol_assumption": 0, "unknown": 3}


def test_explicit_available_at_does_not_automatically_attest_vintage(spec):
    value = JSONObject([("text", "A"), ("available_at", "2020-01-01T00:00:00Z")])
    doc = list(normalize_documents("jena", spec,
               "weather/weather_report/formal_report/wm_messages_v1.json", "202001010600", value))[0]
    assert doc["available_at"] == "2020-01-01T00:00:00Z"
    assert doc["issued_at"] is None
    assert doc["availability"]["status"] == "unknown"
    assert not doc["availability"]["verified_asof_eligibility"]


def test_availability_states_need_evidence():
    assert availability_assessment(None)["status"] == "unknown"
    evidence = {"status": "verified", "reference": "synthetic-original-release-log:1"}
    assert availability_assessment(None, evidence)["status"] == "unknown"
    assert availability_assessment("2020-01-01T00:00:00", evidence)["status"] == "unknown"
    assert availability_assessment("2020-01-01T00:00:00Z", evidence)["status"] == "verified"
    assumption = availability_assessment("2020-01-01T00:00:00Z", {
        "status": "protocol_assumption", "reference": "explicit synthetic convention"})
    assert assumption["status"] == "protocol_assumption"
    assert not assumption["verified_asof_eligibility"]


def test_audit_artifacts_and_input_immutability(spec, snapshot_factory, tmp_path):
    parquet = tmp_path / "fixture.parquet"
    pd.DataFrame({"Date Time": pd.date_range("2020-01-01", periods=3, freq="10min"),
                  "T (degC)": [1., 2., 3.]}).to_parquet(parquet, index=False)
    name = "weather/weather_report/formal_report/wm_messages_v1.json"
    payloads = {spec["primary_file"]: parquet.read_bytes(),
                spec["static_info_file"]: json.dumps({"channel_info": {"weather_large": {"T (degC)": "Temperature"}}}).encode(),
                name: b'{"202001010000":{"Weather Trend":"Calm"}}'}
    directory, _, _ = snapshot_factory(spec, payloads)
    before = {name: sha256_file(directory / name) for name in payloads}
    output = tmp_path / "audit"
    result = inspect_dataset("jena", spec, directory, output)
    for name in ("manifest.json", "numerical_summary.json", "text_summary.json",
                 "availability_summary.json", "documents.jsonl", "_SUCCESS.json"):
        assert (output / name).is_file()
    assert result["formal_multimodal_training_allowed"] is False
    assert before == {name: sha256_file(directory / name) for name in payloads}
    with pytest.raises(FileExistsError):
        inspect_dataset("jena", spec, directory, output)
