"""NWS/IEM AFD parser. Endpoint/field mappings require saved formal docs."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import re

from scripts.pivot_sources.asof import utc
from scripts.pivot_sources.common import SourceError, basic_record, dotted, evidence_ok

PRODUCT_ID = re.compile(r"^(\d{12}|\d{14})-([A-Z0-9]{4})-([A-Z0-9]{6})-(AFD[A-Z0-9]{3})(?:-(.+))?$")
HEADER = re.compile(r"^([A-Z]{4}\d{2})\s+([A-Z0-9]{4})\s+(\d{6})(?:\s+([A-Z]{3}))?\s*$", re.M)


def product_id_parts(value):
    match = PRODUCT_ID.fullmatch(str(value))
    if not match:
        raise SourceError("Unknown Product ID syntax; do not infer a release date")
    stamp, center, wmo, afos, suffix = match.groups()
    fmt = "%Y%m%d%H%M" if len(stamp) == 12 else "%Y%m%d%H%M%S"
    timestamp = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
    return {"timestamp_candidate_utc": timestamp.isoformat(), "center": center,
            "wmo": wmo, "afos_id": afos, "id_suffix": suffix,
            "semantic_status": "syntax_only_requires_documented_timestamp_meaning"}


def bulletin_header(text, reference_time):
    match = HEADER.search(text or "")
    if not match:
        return {"status": "missing_header"}
    wmo, center, clock, correction = match.groups()
    reference = utc(reference_time)
    candidates = []
    for offset in (-1, 0, 1):
        month_index = reference.year * 12 + reference.month - 1 + offset
        year, month = divmod(month_index, 12)
        try:
            candidate = datetime(year, month + 1, int(clock[:2]), int(clock[2:4]), int(clock[4:]), tzinfo=timezone.utc)
            if abs(candidate - reference) <= timedelta(days=2):
                candidates.append(candidate)
        except ValueError:
            pass
    return {"wmo": wmo, "center": center, "day_hour_minute": clock, "correction_code": correction,
            "issued_at_candidate": candidates[0].isoformat() if len(candidates) == 1 else None,
            "status": "unique_month_resolved_against_full_product_id" if len(candidates) == 1 else "ambiguous",
            "semantic_note": "WMO header interpreted as UTC only under reviewed source documentation"}


def normalize(item, contract, url, raw_file, allowed_afos, start, end):
    fields = contract["fields"]
    source_id = dotted(item, fields["source_record_id"])
    parts = product_id_parts(source_id)
    if parts["afos_id"] not in allowed_afos:
        raise SourceError("Response contains an unrequested AFOS office/product")
    if parts["center"] != {"AFDMTR": "KMTR", "AFDSGX": "KSGX"}[parts["afos_id"]]:
        raise SourceError("AFOS ID and source center contradict each other")
    key_time = utc(parts["timestamp_candidate_utc"])
    if not utc(start) <= key_time < utc(end):
        raise SourceError("Out-of-range NWS record rejected")
    text = dotted(item, fields["text"])
    if not isinstance(text, str) or not text.strip():
        raise SourceError("NWS bulletin text missing")
    time_evidence = contract.get("time_evidence", {})
    issue = dotted(item, fields.get("issued_at", "__missing__"))
    if issue is None and evidence_ok(time_evidence.get("product_id_time"), ("issuance",)):
        issue = key_time.isoformat()
        time_evidence = {**time_evidence, "issued_at": time_evidence["product_id_time"]}
    received = dotted(item, fields.get("received_at", "__missing__"))
    for value in (issue, received):
        if value is not None:
            utc(value)  # Metadata timestamps must be explicitly aware.
    header = bulletin_header(text, key_time)
    disagreements = []
    if issue and utc(issue) != key_time:
        disagreements.append("metadata_issuance_differs_from_product_id")
    if header.get("issued_at_candidate") and utc(header["issued_at_candidate"]) != key_time:
        disagreements.append("header_time_differs_from_product_id")
    if header.get("center") and (header["center"] != parts["center"] or header["wmo"] != parts["wmo"]):
        disagreements.append("header_center_or_WMO_differs_from_product_id")
    if received and issue and utc(received) < utc(issue):
        disagreements.append("receipt_precedes_issuance")
    if disagreements:
        time_evidence = {}  # Keep fields, but prevent automatic eligibility.
    raw = json.dumps(item, sort_keys=True, ensure_ascii=False).encode()
    result = basic_record("caiso", "nws_iem", source_id, text, url, raw_file, raw,
                          source_time_key=source_id, issued_at=issue, received_at=received,
                          timezone="UTC", location_id=parts["center"], afos_id=parts["afos_id"],
                          bulletin_type="AFD", source_center=parts["center"],
                          availability_evidence=time_evidence)
    result.update(product_id_metadata=parts, bulletin_header=header, timestamp_disagreements=disagreements,
                  valid_interval_status="unknown_no_forecast_validity_inferred_from_AFD_text")
    return result


def bulletin_audit(records):
    identities = Counter(record["source_record_id"] for record in records)
    versions = {}
    for record in records:
        versions.setdefault(record["source_record_id"], set()).add(record["raw_record_sha256"])
    return {"records": len(records), "by_office": dict(Counter(r["location_id"] for r in records)),
            "duplicate_product_id_occurrences": sum(n - 1 for n in identities.values()),
            "same_id_distinct_payloads": [key for key, values in versions.items() if len(values) > 1],
            "updates_or_corrections": [r["document_id"] for r in records if r.get("bulletin_header", {}).get("correction_code")],
            "timestamp_disagreements": [r["document_id"] for r in records if r.get("timestamp_disagreements")],
            "received_at_fields": sum(r.get("received_at") is not None for r in records),
            "historical_receipt_inferred_from_download_time": False}
