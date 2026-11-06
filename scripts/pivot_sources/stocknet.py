"""Pinned StockNet file inventory, aware tweet times and calendar-based returns."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import PurePosixPath
import re
from zoneinfo import ZoneInfo

from scripts.pivot_sources.asof import availability, utc
from scripts.pivot_sources.common import SourceError, basic_record, digest, evidence_ok


def inventory(tree, revision, expected_universe=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SourceError("StockNet must be pinned to a resolved 40-character commit")
    entries = []
    prices, tweets = set(), set()
    for item in tree.get("tree", []):
        path = item.get("path", "")
        parts = PurePosixPath(path).parts
        if item.get("type") != "blob" or not parts or parts[0] not in {"price", "tweet"}:
            continue
        entity = None
        if len(parts) == 3 and parts[:2] == ("price", "raw") and PurePosixPath(parts[2]).suffix.lower() == ".csv":
            entity = PurePosixPath(parts[2]).stem
            prices.add(entity)
        elif len(parts) >= 4 and parts[:2] == ("tweet", "raw"):
            entity = parts[2]
            tweets.add(entity)
        entries.append({"path": path, "git_blob_sha1": item.get("sha"), "size": item.get("size"), "entity_id": entity})
    universe = list(expected_universe) if expected_universe is not None else sorted(prices)
    if len(universe) != len(set(universe)):
        raise SourceError("Universe contains duplicate entities")
    return {"revision": revision, "tree_truncated": bool(tree.get("truncated", True)), "files": entries,
            "fixed_universe": universe, "universe_sha256": digest(universe),
            "universe_derivation": "explicit_config" if expected_universe is not None else "all_discovered_raw_price_entities_sorted",
            "expected_channel_count": 88, "universe_count": len(universe),
            "entities_with_price_files": sorted(prices), "entities_with_tweet_files": sorted(tweets),
            "entities_missing_price_files": sorted(set(universe) - prices),
            "entities_missing_tweet_files": sorted(set(universe) - tweets),
            "tweet_entities_outside_fixed_universe": sorted(tweets - set(universe)),
            "total_known_file_bytes": sum(entry["size"] for entry in entries if isinstance(entry["size"], int)),
            "files_with_unknown_size": sum(not isinstance(entry["size"], int) for entry in entries),
            "coverage_status": "not_yet_full_coverage_verified"}


def parse_created_at(value):
    if not isinstance(value, str):
        raise SourceError("Twitter created_at missing")
    try:
        return utc(value).isoformat()
    except SourceError:
        try:
            match = re.fullmatch(r"[A-Za-z]{3} ([A-Za-z]{3}) (\d{1,2}) (\d{2}):(\d{2}):(\d{2}) ([+-]\d{4}) (\d{4})", value)
            if match is None:
                raise ValueError("not a Twitter calendar label")
            month, day, hour, minute, second, offset, year = match.groups()
            months = {name: i + 1 for i, name in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}
            sign = 1 if offset[0] == "+" else -1
            zone = timezone(sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5])))
            timestamp = datetime(int(year), months[month], int(day), int(hour), int(minute), int(second), tzinfo=zone)
            return utc(timestamp).isoformat()
        except (ValueError, KeyError) as exc:
            raise SourceError("Unsupported/naive Twitter created_at") from exc


def tweet_records(body):
    text = body.decode("utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SourceError("Unsupported tweet payload: expected JSON object/array/JSONL")
    return value


def normalize_tweet(item, entity, url, source_file, time_evidence=None):
    identifier = item.get("id_str", item.get("id"))
    text = item.get("full_text", item.get("text"))
    if identifier is None or not isinstance(text, str) or not text.strip():
        raise SourceError("Tweet ID or original text missing; preprocessed tokens are not silently substituted")
    created = parse_created_at(item.get("created_at"))
    raw = json.dumps(item, sort_keys=True, ensure_ascii=False).encode()
    record = basic_record("stocknet", "stocknet_twitter", identifier, text, url, source_file, raw,
                          source_time_key=item.get("created_at"), issued_at=created, timezone="UTC", entity_id=entity,
                          availability_evidence={"issued_at": time_evidence} if time_evidence else {})
    nested = []
    for key in ("retweeted_status", "quoted_status"):
        child = item.get(key)
        if isinstance(child, dict):
            try:
                child_time = parse_created_at(child.get("created_at"))
            except SourceError:
                child_time = None
            nested.append({"field": key, "source_record_id": child.get("id_str", child.get("id")), "created_at": child_time,
                           "time_role": "embedded_content_creation_not_outer_post_availability"})
    record.update(embedded_content=nested, original_post_created_at=created,
                  historical_receipt_time=None, creation_time_is_not_verified_collection_time=True)
    return record


def parse_prices(body):
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    headers = reader.fieldnames or []
    if "Date" not in headers or "Close" not in headers:
        raise SourceError("Raw price CSV must explicitly expose Date and Close; no guessed positional schema")
    result = {}
    duplicate_dates = []
    invalid, zero_volume = [], []
    for row in reader:
        try:
            date = datetime.strptime(row["Date"], "%Y-%m-%d").date().isoformat()
        except (ValueError, TypeError) as exc:
            raise SourceError("Invalid price trading date") from exc
        values = {}
        for key in ("Close", "Adj Close"):
            try:
                value = float(row.get(key, ""))
                values[key] = value if math.isfinite(value) and value > 0 else None
            except (ValueError, TypeError):
                values[key] = None
        if values["Close"] is None:
            invalid.append(date)
        try:
            if float(row.get("Volume", "")) == 0:
                zero_volume.append(date)
        except (ValueError, TypeError):
            pass
        if date in result:
            duplicate_dates.append(date)
            result[date] = {"Close": None, "Adj Close": None}  # ambiguity, never choose/average
        else:
            result[date] = values
    return {"prices": result, "duplicate_dates": duplicate_dates, "invalid_close_dates": invalid,
            "zero_volume_dates": zero_volume, "halt_or_delisting_status": "unknown_do_not_infer_from_missing_rows_or_zero_volume",
            "adjusted_close_present": "Adj Close" in headers,
            "adjustment_mechanism": "unknown", "future_adjustment_risk": "requires_source_and_corporate_action_review"}


def calendar_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen, previous = set(), None
    for row in rows:
        day = datetime.strptime(row["session"], "%Y-%m-%d").date()
        opened, closed = utc(row["open"]), utc(row["close"])
        if row["session"] in seen or opened >= closed or previous is not None and day <= previous:
            raise SourceError("Calendar must have unique sorted sessions and valid aware open/close timestamps")
        if closed.astimezone(ZoneInfo("America/New_York")).date() != day:
            raise SourceError("Calendar session label disagrees with New York close date")
        seen.add(row["session"])
        previous = day
    return rows


def price_task(prices, universe, calendar, spec):
    """Session-index returns, never compress a missing trading session away."""
    dates = [row["session"] for row in calendar if spec["start_date"] <= row["session"] < spec["split_boundaries"][-1]]
    missing = {entity: [day for day in dates if prices.get(entity, {}).get("prices", {}).get(day, {}).get("Close") is None]
               for entity in universe}
    common = [day for day in dates if all(day not in missing[entity] for entity in universe)]
    values = {}
    for entity in universe:
        entries = prices.get(entity, {}).get("prices", {})
        rows = []
        for i, day in enumerate(dates):
            before, after = entries.get(dates[i - 1], {}) if i else {}, entries.get(day, {})
            previous, current = before.get("Close"), after.get("Close")
            adjusted_previous, adjusted_current = before.get("Adj Close"), after.get("Adj Close")
            rows.append({"session": day,
                         "close_to_close": current / previous - 1 if current and previous else None,
                         "log_return": math.log(current / previous) if current and previous else None,
                         "adjusted_close_return": adjusted_current / adjusted_previous - 1 if adjusted_current and adjusted_previous else None})
        values[entity] = rows
    window_info = {}
    for split, begin, end in (("train", spec["start_date"], spec["split_boundaries"][0]),
                              ("validation", spec["split_boundaries"][0], spec["split_boundaries"][1]),
                              ("test", spec["split_boundaries"][1], spec["split_boundaries"][2])):
        indices = [i for i, day in enumerate(dates) if begin <= day < end]
        window_info[split] = {"sessions": len(indices), "horizons": {}}
        for horizon in spec["horizons"]:
            count = max(0, len(indices) - spec["lookback"] - horizon + 1)
            valid_windows, valid_labels, total_labels = 0, 0, count * horizon * len(universe)
            valid_input_windows, eligible_masked_windows = 0, 0
            for start in range(count):
                inputs = indices[start:start + spec["lookback"]]
                input_count = sum(values[entity][i]["log_return"] is not None for entity in universe for i in inputs)
                labels = indices[start + spec["lookback"]:start + spec["lookback"] + horizon]
                nvalid = sum(values[entity][i]["log_return"] is not None for entity in universe for i in labels)
                valid_labels += nvalid
                valid_windows += int(input_count > 0 and nvalid == horizon * len(universe))
                valid_input_windows += int(input_count > 0)
                eligible_masked_windows += int(input_count > 0 and nvalid > 0)
            window_info[split]["horizons"][str(horizon)] = {"theoretical_windows": count, "full_universe_valid_label_windows": valid_windows,
                                                          "valid_input_windows": valid_input_windows, "eligible_masked_windows": eligible_masked_windows,
                                                          "valid_label_occurrences": valid_labels,
                                                          "invalid_label_ratio": 1 - valid_labels / total_labels if total_labels else None}
    all_loaded = len(universe) == 88 and all(entity in prices for entity in universe)
    return {"fixed_universe": universe, "universe_sha256": digest(universe), "loaded_price_entities": sorted(prices),
            "all_88_price_series_loaded": all_loaded, "common_valid_trading_days": len(common) if all_loaded else None,
            "calendar_session_count": len(dates), "missing_sessions_by_entity": missing,
            "per_split_windows": window_info, "counts_scope": "full_universe_local_files" if all_loaded else "incomplete_price_probe_not_full_MIMO_verification",
            "returns": values, "no_future_fill": True,
            "adjusted_close_default_allowed": False, "classification_labels_used": False,
            "note": "time-split rows do not borrow previous split history; first return needs its immediately preceding official session"}


def tweet_audit(records, universe, calendar, spec):
    by_id, by_text = Counter(r["source_record_id"] for r in records), Counter(r["text"] for r in records)
    by_entity = Counter(r["entity_id"] for r in records)
    results = []
    applied = {f"{policy}+{lag}h": [availability(record, policy, lag) for record in records]
               for policy, lag in [("strict", 0), *(("source_time", lag) for lag in spec["lag_hours"]) ]}
    for index, row in enumerate(calendar):
        if not spec["start_date"] <= row["session"] < spec["split_boundaries"][-1]:
            continue
        origin = utc(row["close"]) + timedelta(minutes=spec["origin_policy"]["minutes_after_actual_session_close"])
        split = "train" if row["session"] < spec["split_boundaries"][0] else "validation" if row["session"] < spec["split_boundaries"][1] else "test"
        for lookback in (1, 3, 5, 20):
            if index < lookback:
                continue
            # Previous actual session's close boundary includes weekends/holidays
            # that lie between trading sessions; no synthetic weekend sessions.
            begin = utc(calendar[index - lookback]["close"])
            counts = Counter(r["entity_id"] for r in records if begin < utc(r["issued_at"]) <= origin)
            policies = {}
            for policy, items in applied.items():
                verifiable = any(item["available_at"] is not None for item in items)
                eligible = Counter(item["entity_id"] for item in items if item["available_at"] is not None
                                   and utc(item["available_at"]) <= origin and begin < utc(item["issued_at"]) <= origin)
                policies[policy] = {"status": "sample_only" if verifiable else "not_verifiable",
                                    "tweet_counts_by_entity": dict(eligible) if verifiable else None,
                                    "entities_without_observed_eligible_text": [entity for entity in universe if not eligible[entity]] if verifiable else None}
            results.append({"session": row["session"], "split": split, "origin": origin.isoformat(),
                            "lookback_trading_days": lookback, "tweet_counts_by_entity": {entity: counts[entity] for entity in universe},
                            "entities_without_observed_text": [entity for entity in universe if not counts[entity]],
                            "asof_policy_candidates": policies,
                            "scope": "sampled_raw_tweet_files_only_not_full_text_coverage"})
    return {"by_entity": dict(by_entity), "universe_entities_without_observed_text": [e for e in universe if not by_entity[e]],
            "duplicate_id_occurrences": sum(n - 1 for n in by_id.values()),
            "duplicate_text_occurrences": sum(n - 1 for n in by_text.values()),
            "created_at_start": min((r["issued_at"] for r in records), default=None),
            "created_at_end": max((r["issued_at"] for r in records), default=None), "session_coverage": results,
            "text_missing_mask_semantics": "no sampled tweet is not proof of no tweet in the full archive",
            "full_coverage_status": "not_yet_full_coverage_verified"}
