"""As-of filtering MUST precede retrieval/ranking; no implicit clock localization."""

from datetime import datetime, timedelta, timezone
from scripts.pivot_sources.common import SourceError, evidence_ok


def utc(value):
    if not isinstance(value, (str, datetime)):
        raise SourceError("An explicit timezone-aware timestamp is required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except ValueError as exc:
        raise SourceError("Invalid ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise SourceError("Naive timestamps cannot be silently localized")
    return result.astimezone(timezone.utc)


def availability(record, policy="strict", lag_hours=0):
    if lag_hours < 0:
        raise SourceError("Availability lag cannot be negative")
    if policy == "run_lag":
        actual = availability(record, "strict", 0)
        if actual["strict_verified"]:
            return {**actual, "requested_scenario": "run_lag", "scenario_lag_hours": lag_hours,
                    "actual_publication_or_receipt_preferred": True}
    result = {**record, "available_at": None, "availability_status": "unknown",
              "availability_policy": policy, "strict_verified": False,
              "lag_hours": lag_hours, "lag_is_experimental_assumption": policy != "strict"}
    evidence = record.get("availability_evidence") or {}
    field = None
    if policy == "strict":
        if lag_hours:
            raise SourceError("Strict receipt/publication policy cannot add an assumed lag")
        for candidate, role in (("received_at", "historical_reception"), ("file_publication_time", "historical_publication")):
            if record.get(candidate) and evidence_ok(evidence.get(candidate), (role,)):
                field = candidate
                result["strict_verified"] = True
                result["availability_status"] = "verified_received_time" if candidate == "received_at" else "verified_source_time"
                break
    elif policy in ("source_time", "run_lag"):
        field = "issued_at" if policy == "source_time" else "run_time"
        roles = ("issuance", "creation", "historical_publication") if policy == "source_time" else ("model_initialization",)
        if not record.get(field) or not evidence_ok(evidence.get(field), roles):
            field = None
        else:
            result["availability_status"] = "latency_assumption" if lag_hours or policy == "run_lag" else "verified_source_time"
    else:
        raise SourceError("Unknown availability policy")
    if field:
        try:
            chosen = utc(record[field])
            # Inconsistent source clocks cannot be promoted to verified receipt.
            if field in ("received_at", "file_publication_time") and record.get("issued_at") and chosen < utc(record["issued_at"]):
                raise SourceError("Receipt/publication precedes issuance")
            if field == "file_publication_time" and record.get("run_time") and chosen < utc(record["run_time"]):
                raise SourceError("Publication precedes model initialization")
            result["available_at"] = (chosen + timedelta(hours=lag_hours)).isoformat()
        except SourceError as exc:
            result.update(available_at=None, strict_verified=False, availability_status="unknown", time_error=str(exc))
    result["tier"] = "Strict Verified" if result["strict_verified"] else "Source-time + Conservative Lag" if result["available_at"] else "Not Verifiable"
    return result


def eligible_records(records, forecast_origin, policy="strict", lag_hours=0):
    """Returns the legal candidate pool. A caller can rank ONLY this result."""
    origin = utc(forecast_origin)
    pool = []
    for record in records:
        item = availability(record, policy, lag_hours)
        if item["available_at"] is not None and utc(item["available_at"]) <= origin:
            pool.append(item)
    return pool
