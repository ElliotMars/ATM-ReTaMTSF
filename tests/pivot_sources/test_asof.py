"""Source time, actual receipt and lag scenarios must stay distinct."""
import pytest

from scripts.pivot_sources.asof import availability, eligible_records, utc
from scripts.pivot_sources.common import SourceError


def record(evidence_factory):
    return {"source_record_id": "fixture", "issued_at": "2021-01-01T12:00:00Z", "received_at": "2021-01-01T12:05:00Z",
            "valid_start": "2021-01-02T00:00:00Z", "available_at": None,
            "availability_evidence": {"issued_at": evidence_factory("issuance"), "received_at": evidence_factory("historical_reception")}}


def test_aware_timezones_and_equality(evidence_factory):
    row = record(evidence_factory)
    assert len(eligible_records([row], "2021-01-01T07:05:00-05:00")) == 1
    assert not eligible_records([row], "2021-01-01T12:04:59Z")
    result = availability(row)
    assert result["available_at"] == "2021-01-01T12:05:00+00:00"
    assert result["availability_status"] == "verified_received_time"
    assert result["strict_verified"]


def test_future_issuance_blocked_but_future_forecast_validity_allowed(evidence_factory):
    row = record(evidence_factory)
    assert not eligible_records([row], "2021-01-01T11:00:00Z", "source_time")
    assert eligible_records([row], "2021-01-01T13:00:00Z", "source_time")


def test_missing_available_time_and_unreviewed_time_are_blocked(evidence_factory):
    row = record(evidence_factory)
    row["availability_evidence"] = {}
    row["available_at"] = "1900-01-01T00:00:00Z"  # unsupported pre-filled value must be ignored
    assert availability(row)["available_at"] is None
    assert not eligible_records([row], "2025-01-01T00:00:00Z")


def test_source_time_plus_lag_is_conditional(evidence_factory):
    row = record(evidence_factory)
    assert not eligible_records([row], "2021-01-01T14:59:59Z", "source_time", 3)
    assert eligible_records([row], "2021-01-01T15:00:00Z", "source_time", 3)
    result = availability(row, "source_time", 3)
    assert result["availability_status"] == "latency_assumption" and not result["strict_verified"]
    assert result["tier"] == "Source-time + Conservative Lag"


def test_run_time_is_never_implicitly_available(evidence_factory):
    row = {"run_time": "2021-01-01T00:00:00Z", "availability_evidence": {"run_time": evidence_factory("model_initialization")}}
    assert availability(row)["available_at"] is None
    assert availability(row, "run_lag", 6)["available_at"] == "2021-01-01T06:00:00+00:00"


def test_inconsistent_receipt_is_not_verified(evidence_factory):
    row = record(evidence_factory)
    row["received_at"] = "2021-01-01T11:59:00Z"
    assert not availability(row)["strict_verified"]
    assert availability(row)["availability_status"] == "unknown"


@pytest.mark.parametrize("value", [None, "2021-01-01T12:00:00", "nonsense"])
def test_uncertain_or_naive_clock_rejected(value):
    with pytest.raises(SourceError):
        utc(value)


def test_asof_pool_is_filtered_before_any_ranking(evidence_factory):
    legal = record(evidence_factory)
    future = {**legal, "received_at": "2022-01-01T00:00:00Z", "source_record_id": "future"}
    pool = eligible_records([future, legal], "2021-01-01T13:00:00Z")
    assert [item["source_record_id"] for item in pool] == ["fixture"]
