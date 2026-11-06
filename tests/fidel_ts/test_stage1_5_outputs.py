"""Output integrity, non-overwrite, comparison and readiness tests; offline only."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.fidel_ts.common import AuditError, MANIFEST, read_json, sha256_file, write_json
from scripts.fidel_ts.numerical_diagnostics import NativeData
from scripts.fidel_ts.numerical_protocols import build_protocol, write_prepared
from scripts.fidel_ts.stage1_5_io import (
    PROTOCOL_CONFIG, finish_output, load_protocol, new_output, read_bundle, recheck_input,
)
from scripts.fidel_ts.summarize_stage1_5 import compare_prepared, pytest_status, readiness_decision
from scripts.fidel_ts import numerical_cli, stage1_5_io


def test_original_file_hash_unchanged_after_preparation_helpers(tmp_path):
    config = copy.deepcopy(read_json(PROTOCOL_CONFIG))
    config["lookback"], config["horizons"] = 1, [1]
    times = pd.date_range("2020-01-01", periods=18, freq="10min")
    values = np.column_stack([np.arange(18.), np.arange(18.) + 1])
    path = tmp_path / "original.parquet"
    pd.DataFrame({"time": times, "a": values[:, 0], "b": values[:, 1]}).to_parquet(path, index=False)
    original = sha256_file(path)
    data = NativeData(times, values, np.zeros_like(values, dtype=bool), ["a", "b"], "jena", 600)
    result = build_protocol(data, "n2", config)
    output = tmp_path / "processed"
    output.mkdir()
    write_prepared(output, result, config, {}, prepare=True)
    assert sha256_file(path) == original
    assert (output / "train/validity_mask.parquet").is_file()
    assert (output / "window_indices/train_H1.parquet").is_file()


def test_success_and_failure_outputs_cannot_be_overwritten(tmp_path):
    output = tmp_path / "run"
    with new_output(output):
        write_json(output / "data.json", {"value": 1})
        finish_output(output, {"scope": "fixture"})
    assert read_bundle(output)["status"] == "complete"
    with pytest.raises(FileExistsError):
        with new_output(output):
            pass
    write_json(output / "data.json", {"value": 2})
    with pytest.raises(AuditError, match="hash"):
        read_bundle(output)


def test_failure_cannot_masquerade_as_success(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        with new_output(output):
            write_json(output / "partial.json", {"partial": True})
            raise RuntimeError("synthetic interruption")
    assert (output / "failure.json").is_file()
    assert not (output / "_SUCCESS.json").exists()


def test_blocked_protocol_is_not_published_as_success(tmp_path):
    output = tmp_path / "blocked"
    output.mkdir()
    write_json(output / "quality_summary.json", {"blocked_reasons": ["no valid training observations"]})
    finish_output(output, {"scope": "fixture"}, ["no valid training observations"])
    assert not (output / "_SUCCESS.json").exists()
    assert read_bundle(output, allow_blocked=True)["status"] == "blocked"
    with pytest.raises(FileNotFoundError):
        read_bundle(output)


def test_raw_snapshot_recheck_detects_changes(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "primary.parquet"
    path.write_bytes(b"original")
    digest = sha256_file(path)
    monkeypatch.setattr(stage1_5_io, "verify_manifest", lambda *args: {"ok": True})
    spec = {"primary_file": "primary.parquet"}
    recheck_input("jena", spec, raw, digest)
    path.write_bytes(b"changed")
    with pytest.raises(AuditError, match="changed"):
        recheck_input("jena", spec, raw, digest)


def test_snapshot_manifest_cannot_change_during_a_run(tmp_path, monkeypatch):
    path = tmp_path / "primary.parquet"
    path.write_bytes(b"original")
    manifest = tmp_path / MANIFEST
    manifest.write_text("{}")
    digest = sha256_file(manifest)
    monkeypatch.setattr(stage1_5_io, "verify_manifest", lambda *args: {"ok": True})
    manifest.write_text('{"changed": true}')
    with pytest.raises(AuditError, match="manifest changed"):
        recheck_input("jena", {"primary_file": path.name}, tmp_path, sha256_file(path), digest)


def test_dry_run_does_not_read_numerical_data_or_write_output(tmp_path, monkeypatch, capsys):
    called = []
    spec = {"schema_version": 1, "repository": "fixture", "revision": "fixed", "primary_file": "primary.parquet"}
    manifest = {"files": {"primary.parquet": {"size": 42}}}

    def snapshot(*args):
        called.append(args[-1])
        return spec, tmp_path / "raw", manifest

    def forbidden(*args):
        raise AssertionError("Dry run must not scan raw numerical data")

    monkeypatch.setattr(numerical_cli, "snapshot", snapshot)
    monkeypatch.setattr(numerical_cli, "load_native", forbidden)
    output = tmp_path / "absent"
    assert numerical_cli.run_numerical("prepare", ["--dataset", "jena", "--protocol", "n2", "--dry-run", "--output-dir", str(output)]) == 0
    assert called == [False]
    assert not output.exists()
    assert not json.loads(capsys.readouterr().out)["data_read_or_checked"]


def test_frozen_protocol_parameters_cannot_be_silently_ignored(tmp_path):
    config = read_json(PROTOCOL_CONFIG)
    config["n1"]["native_fit_dtype"] = "float64"
    path = tmp_path / "bad.json"
    write_json(path, config)
    with pytest.raises(AuditError, match="Unsupported n1"):
        load_protocol(path)


def test_compare_prepared_protocols_records_timestamp_sets_and_all_targets(tmp_path):
    config = copy.deepcopy(read_json(PROTOCOL_CONFIG))
    config["lookback"], config["horizons"] = 1, [1]
    times = pd.DatetimeIndex([])
    for start in ("2020-01-01", "2021-01-01", "2022-01-01"):
        times = times.append(pd.date_range(start, periods=18, freq="10min"))
    values = np.column_stack([np.arange(54.), np.arange(54.) + 2])
    data = NativeData(times, values, np.zeros_like(values, dtype=bool), ["a", "b"], "jena", 600)
    n1, n2 = tmp_path / "n1", tmp_path / "n2"
    for protocol, directory in (("n1", n1), ("n2", n2)):
        directory.mkdir()
        write_prepared(directory, build_protocol(data, protocol, config), config, {}, prepare=True)
    output = tmp_path / "comparison"
    output.mkdir()
    comparison = compare_prepared(n1, n2, output, config)
    assert comparison["status"] == "compared"
    assert comparison["channels"] == ["a", "b"]
    assert comparison["splits"]["train"]["shared_sampled_timestamps"] == 3
    assert comparison["splits"]["test"]["horizons"]["1"]["n2"]["published_window_indices"] == 2
    assert (output / "downsampled_timestamp_sets.parquet").is_file()


def test_junit_report_must_cover_all_new_test_modules(tmp_path):
    path = tmp_path / "tests.xml"
    cases = "".join(f'<testcase classname="tests.fidel_ts.test_stage1_5_{name}" name="test_fixture"/>'
                    for name in ("numerical", "provenance", "outputs"))
    path.write_text('<testsuites><testsuite errors="0">' + cases + '</testsuite></testsuites>')
    assert pytest_status(path)["status"] == "reported_pass"
    path.write_text('<testsuites><testsuite errors="1"/></testsuites>')
    assert pytest_status(path)["status"] == "blocked_or_incomplete"


def test_readiness_is_separate_and_asof_remains_blocked_without_issuance():
    decision = readiness_decision({"status": "compared", "n2_blocked_reasons": []}, {"reports": 10, "T1": "blocked_not_verifiable"}, {"status": "reported_pass"})
    assert decision["numerical_mimo_readiness"]["status"] == "candidate_for_stage2_review"
    assert decision["archived_text_research_readiness"]["status"] == "auxiliary_only"
    assert decision["verified_asof_multimodal_readiness"]["status"] == "blocked_not_verifiable"
    blocked = readiness_decision({"status": "blocked"}, {"reports": 0, "T1": "blocked_not_verifiable"}, {"status": "unverified"})
    assert blocked["numerical_mimo_readiness"]["status"] == "blocked"
    assert "ready" not in blocked
