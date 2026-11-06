"""Synthetic numerical protocol tests. Written for manual offline execution."""

import copy

import numpy as np
import pandas as pd
import pytest

from scripts.fidel_ts.common import AuditError, read_json
from scripts.fidel_ts.numerical_diagnostics import (
    NativeData, diagnose, duplicate_analysis, missing_value_analysis, native_from_frame,
    resolved_rows, segment_ids, split_ids, time_gap_analysis,
)
from scripts.fidel_ts.numerical_protocols import (
    ProtocolBlocked, build_protocol, causal_fill_history, fit_masked_scaler,
    masked_error_sums, window_accounting,
)
from scripts.fidel_ts.stage1_5_io import PROTOCOL_CONFIG


@pytest.fixture
def config15():
    return copy.deepcopy(read_json(PROTOCOL_CONFIG))


def native(times, values, dataset="jena"):
    values = np.asarray(values, dtype=float)
    return NativeData(pd.DatetimeIndex(times), values, np.zeros(values.shape, dtype=bool),
                      [f"channel_{i}" for i in range(values.shape[1])], dataset, 600)


def test_identical_duplicates_are_collapsed_with_lineage(config15):
    data = native(pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01 00:10"], format="mixed"),
                  [[1., 2.], [1., 2.], [2., 3.]])
    report = duplicate_analysis(data, config15)
    assert report["extra_duplicate_records"] == 1 and report["duplicate_groups"] == 1
    assert report["groups"][0]["original_rows"] == [0, 1]
    assert report["groups"][0]["classification"] == "identical"
    assert resolved_rows(data, report).tolist() == [0, 2]


def test_conflicting_duplicates_are_excluded_not_averaged(config15):
    data = native(pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01 00:10"], format="mixed"),
                  [[1., 2.], [9., 2.], [2., 3.]])
    report = duplicate_analysis(data, config15)
    assert report["groups"][0]["classification"] == "conflicting"
    assert report["groups"][0]["conflicting_valid_channels"] == ["channel_0"]
    assert resolved_rows(data, report).tolist() == [2]
    assert report["groups"][0]["unresolved_timezone_issue"]


def test_duplicate_invalid_patterns_are_not_merged(config15):
    times = pd.to_datetime(["2020-01-01", "2020-01-01"])
    data = native(times, [[1., np.nan], [1., 3.]])
    report = duplicate_analysis(data, config15)
    assert report["groups"][0]["classification"] == "indeterminate_invalid"
    assert len(resolved_rows(data, report)) == 0
    paired = native(times, [[1., np.nan], [1., np.nan]])
    result = duplicate_analysis(paired, config15)
    assert result["groups"][0]["classification"] == "identical_with_invalid"
    assert resolved_rows(paired, result).tolist() == [0]
    assert not paired.valid[0, 1]


def test_gap_and_clock_rollback_are_distinct(config15):
    data = native(pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 00:10"]),
                  [[1.], [2.], [3.]])
    result = time_gap_analysis(data, config15)
    assert result["clock_rollbacks"] == 1
    assert result["unique_gap_count"] == 1
    assert result["unique_timestamp_gaps"][0]["theoretical_missing_points"] == 1
    assert result["unique_timestamp_gaps"][0]["cause"].startswith("unknown")
    assert diagnose(data, config15)["numerical_diagnostics.json"]["unresolved_timezone_issue"]


def test_invalid_values_and_valid_negative_temperatures(config15):
    data = native(pd.date_range("2020-01-01", periods=5, freq="10min"),
                  [[-5., 1.], [-9999., np.nan], [np.inf, 2.], [-np.inf, 3.], [4., 4.]])
    data.arrow_null[1, 1] = True
    assert data.valid[0, 0]  # A legitimate negative observation is not missing.
    assert not data.valid[1:4, 0].any()
    report = missing_value_analysis(data, config15)["splits"]["train"]
    assert report["channels"][0]["minus_9999"] == 1
    assert report["channels"][0]["inf"] == 2
    assert report["channels"][1]["arrow_null"] == 1
    assert report["channels"][1]["nan_including_arrow_null"] == 1
    caiso = native(data.times, data.values, "caiso")
    assert caiso.valid[1, 0]  # No invented CAISO sentinel rule.


def test_split_boundaries_are_inclusive_on_right(config15):
    times = pd.to_datetime(["2020-12-31 23:50", "2021-01-01 00:00", "2021-12-31 23:50", "2022-01-01 00:00"])
    assert split_ids(times, config15).tolist() == [0, 1, 1, 2]


def phase_fixture():
    times = pd.DatetimeIndex([])
    for start in ("2020-12-30 00:10", "2021-01-01 00:00", "2022-01-01 00:00"):
        times = times.append(pd.date_range(start, periods=7, freq="10min"))
    values = np.column_stack([np.arange(21, dtype=float) ** 2, np.arange(21, dtype=float) + 1])
    return native(times, values)


def test_n1_anchors_each_split_n2_uses_one_training_phase(config15):
    data = phase_fixture()
    n1 = build_protocol(data, "n1", config15)
    n2 = build_protocol(data, "n2", config15)
    assert n1["splits"]["train"]["source_rows"].tolist() == [0, 6]
    assert n1["splits"]["validation"]["source_rows"].tolist() == [7, 13]
    assert n1["splits"]["test"]["source_rows"].tolist() == [14, 20]
    assert n2["hour_anchor"]["phase_ns_mod_hour"] == 600 * 10**9
    assert n2["splits"]["validation"]["source_rows"].tolist() == [8]
    assert n2["splits"]["test"]["source_rows"].tolist() == [15]
    assert all(t.minute == 10 for split in n2["splits"].values() for t in split["times"])


def test_native_training_scaler_never_sees_validation_or_test(config15):
    data = phase_fixture()
    result = build_protocol(data, "n2", config15)
    assert result["scaler"]["fit_sample_counts"] == [7, 7]
    assert result["scaler"]["mean"][0] == pytest.approx(np.mean(np.arange(7) ** 2))
    changed_values = data.values.copy()
    changed_values[7:] = 1.e8
    changed = native(data.times, changed_values)
    assert build_protocol(changed, "n2", config15)["scaler"] == result["scaler"]
    assert build_protocol(data, "n1", config15)["scaler"] == build_protocol(changed, "n1", config15)["scaler"]


def test_n1_keeps_sentinel_in_scaler_n2_rejects_no_valid_train(config15):
    data = phase_fixture()
    values = data.values.copy()
    values[:7, 0] = -9999
    changed = native(data.times, values)
    result = build_protocol(changed, "n1", config15)
    assert result["scaler"]["mean"][0] == -9999
    assert result["splits"]["train"]["values"][0, 0] == -9999
    assert not result["splits"]["train"]["valid"][:, 0].any()
    with pytest.raises(ProtocolBlocked, match="No valid"):
        build_protocol(changed, "n2", config15)


def test_n1_transforms_native_split_before_subsampling(config15):
    data = phase_fixture()
    values = data.values.copy()
    values[8, 0] = np.inf  # Validation selects rows 7 and 13, not row 8.
    with pytest.raises(ProtocolBlocked, match="transform rejects native data"):
        build_protocol(native(data.times, values), "n1", config15)


def test_windows_cannot_cross_hour_or_native_gaps():
    times = pd.date_range("2020-01-01", periods=8, freq="h")
    valid = np.ones((8, 2), dtype=bool)
    summary, _ = window_accounting(times, valid, np.array([0, 0, 0, 0, 1, 1, 1, 1]), 3, 2, True)
    assert summary["theoretical_windows"] == 4
    assert summary["continuous_windows"] == 0
    missing_hour = times.delete(4)
    result, _ = window_accounting(missing_hour, valid[:7], np.zeros(7, dtype=int), 3, 2, True)
    assert result["continuous_windows"] == 0


def test_h720_mimo_label_mask_counts_every_channel():
    times = pd.date_range("2020-01-01", periods=1080, freq="h")
    valid = np.ones((1080, 21), dtype=bool)
    valid[500, 0] = False
    valid[600, 20] = False
    info, index = window_accounting(times, valid, np.zeros(1080, dtype=int), 360, 720, True)
    assert info["theoretical_windows"] == 1
    assert info["continuous_windows"] == 1
    assert info["masked_label_eligible_windows"] == 1
    assert info["full_valid_label_windows_among_continuous"] == 0
    assert len(info["valid_label_occurrences_by_channel"]) == 21
    assert info["valid_label_occurrences_total"] == 720 * 21 - 2
    assert index["label_start_row"].iloc[0] == 360
    assert index["label_end_row_exclusive"].iloc[0] == 1080


def test_input_forward_fill_uses_only_past_values_and_training_fallback():
    values = np.array([[np.nan, -5.], [2., np.nan], [np.nan, 4.], [9., 8.]])
    valid = np.isfinite(values)
    filled, mask = causal_fill_history(values, valid, [3., 6.])
    assert filled[:, 0].tolist() == [3., 2., 2., 9.]
    assert filled[:, 1].tolist() == [-5., -5., 4., 8.]
    changed = values.copy()
    changed[-1] = 1.e9
    future_changed, _ = causal_fill_history(changed, valid, [3., 6.])
    np.testing.assert_equal(filled[:-1], future_changed[:-1])
    np.testing.assert_equal(mask, valid)


def test_invalid_labels_never_enter_metric_or_loss_sums():
    truth = np.array([[1., np.nan], [np.inf, 3.]])
    mask = np.array([[True, False], [False, True]])
    prediction = np.array([[2., 1.e9], [-1.e9, 5.]])
    result = masked_error_sums(prediction, truth, mask)
    assert result == {"squared_error_sum": 5., "absolute_error_sum": 3., "valid_count": 2}
    with pytest.raises(AuditError):
        masked_error_sums(prediction[:, :1], truth, mask)


def test_channel_selection_or_order_changes_are_rejected(spec):
    frame = pd.DataFrame({"Date Time": pd.date_range("2020-01-01", periods=2, freq="10min"), "T (degC)": [1., 2.]})
    with pytest.raises(AuditError, match="All configured"):
        native_from_frame(frame, "jena", spec)
