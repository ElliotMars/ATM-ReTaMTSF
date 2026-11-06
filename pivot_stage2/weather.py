"""Documented IEM range queries and source-time records; never assume N2 UTC."""
from datetime import timedelta
import hashlib
import io
import re
import time
from urllib.parse import urlencode
import zipfile

import pandas as pd

from .common import DATA, ARTIFACTS, digest, read_json, sha, write_json
from .fetch import Downloader

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"


def iem_url(pil, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pil not in {"AFDMTR", "AFDSGX"} or start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError("Explicit office and aware increasing UTC range required")
    if end - start > pd.Timedelta(days=14):
        raise ValueError("AFD requests are chunked at <=14 days")
    return ENDPOINT + "?" + urlencode({"pil": pil, "sdate": start.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "edate": end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ"), "fmt": "zip", "limit": 9999, "order": "asc"})


def parse_zip(path, url, start, end, lag_hours=1):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    records, rejected = [], []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            body = archive.read(name)
            text = body.decode("utf-8", errors="replace")
            match = re.search(r"(?m)^([A-Z]{4}\d{2})\s+(K[A-Z]{3})\s+(\d{6})(?:\s+([A-Z]{3}))?\s*$", text)
            if not match:
                rejected.append({"file": name, "reason": "missing_WMO_header"})
                continue
            wmo, center, clock, correction = match.groups()
            pil = next((item for item in ("AFDMTR", "AFDSGX") if re.search(r"(?m)^" + item + r"\s*$", text)), None)
            if not pil or center != {"AFDMTR": "KMTR", "AFDSGX": "KSGX"}[pil]:
                rejected.append({"file": name, "reason": "office_or_product_mismatch"})
                continue
            candidates = []
            for month in pd.period_range(start.tz_localize(None).to_period("M"), end.tz_localize(None).to_period("M"), freq="M"):
                try:
                    stamp = pd.Timestamp(year=month.year, month=month.month, day=int(clock[:2]), hour=int(clock[2:4]), minute=int(clock[4:]), tz="UTC")
                    if start <= stamp < end:
                        candidates.append(stamp)
                except ValueError:
                    pass
            if len(candidates) != 1:
                rejected.append({"file": name, "reason": "ambiguous_month_or_out_of_range"})
                continue
            issued = candidates[0]
            full_id = f"{issued.strftime('%Y%m%d%H%M')}-{center}-{wmo}-{pil}" + (f"-{correction}" if correction else "")
            records.append({"document_id": digest([full_id, hashlib.sha256(body).hexdigest()]), "source_record_id": full_id,
                            "text": text[:16000], "issued_at": issued.isoformat(), "available_at": (issued + pd.Timedelta(hours=lag_hours)).isoformat(),
                            "received_at": None, "availability_policy": f"documented_product_time_plus_{lag_hours}h_assumption",
                            "strict_verified_receipt": False, "office": center, "pil": pil, "revision_code": correction,
                            "source_url": url, "source_file": name, "raw_record_sha256": hashlib.sha256(body).hexdigest()})
    return records, rejected


def fetch_iem_range(client, pil, start, end):
    """Fully automatic range adapter, used only when numerical clock is known."""
    current, end = pd.Timestamp(start), pd.Timestamp(end)
    result = []
    while current < end:
        stop = min(end, current + pd.Timedelta(days=14))
        url = iem_url(pil, current, stop)
        time.sleep(1)
        path, info = client.get(url, "nws", 16 * 1024**2)
        records, issues = parse_zip(path, url, current, stop)
        if len(records) >= 9999:
            raise ValueError("Server record cap reached; do not claim complete chunk coverage")
        result.extend(records)
        current = stop
    return result


def source_decisions():
    acquisition = read_json(DATA / "sources/weather_acquisition.json")
    result = {}
    nws = acquisition.get("nws", {})
    parsed, rejected = [], []
    for batch in nws.get("actual_batches", []):
        try:
            records, errors = parse_zip(batch["path"], batch["url"], "2021-07-15T00:00:00Z", "2021-07-16T00:00:00Z")
            parsed.extend(records)
            rejected.extend(errors)
        except Exception as exc:
            rejected.append({"error": str(exc)})
    output = ARTIFACTS / "sources"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "nws_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in parsed:
            handle.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
    result["nws"] = {"real_data_acquired": bool(parsed), "representative_records": len(parsed), "rejected": rejected,
                     "interface": {"documentation": nws.get("documentation"), "sha256": nws.get("sha256"),
                                   "semantic_review": "official_GET_query_fields_and_UTC_bounds_read_by_agent; distinct_from_request_success",
                                   "query_parameters": ["pil", "sdate", "edate", "fmt", "limit", "order"]},
                     "full_period_download": False, "full_period_not_downloaded_reason": "CAISO_N2_clock_unresolved; avoid downloading unusable bulk text",
                     "availability_scenarios_hours": [1, 3], "historical_receipt": "unknown",
                     "strict_crossmodal_alignment": False, "decision": "NO-GO_multimodal_unknown_CAISO_physical_clock"}
    gfs = acquisition.get("gfs", {})
    official_path = ARTIFACTS / "sources/final_official_evidence.json"
    access_evidence = read_json(official_path).get("gfs_access", {}) if official_path.exists() else {}
    result["gfs"] = {"real_forecast_values_acquired": False, "official_metadata_acquired": "documentation" in gfs,
                     "documentation": gfs.get("documentation"), "sha256": gfs.get("sha256"),
                     "actual_historical_publication_time": None, "candidate_lag_hours": [3, 6, 12],
                     "decision": "NO-GO_no_budget_compatible_historical_subset_and_unresolved_Jena_clock",
                     "reason": "Official data-access page marks Get a Subset as login required; user stop rule applied without login, registration or global downloads",
                     "access_evidence": access_evidence,
                     "archive_start_in_official_metadata": "2015-01-15", "pre_2015_text_fabricated": False,
                     "no_global_GRIB_download": True}
    write_json(output / "source_decisions.json", result)
    return result
