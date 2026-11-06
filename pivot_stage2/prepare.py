"""One-shot real data preparation; no training or changes to accepted N2."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import io
import hashlib
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from .common import ROOT, ARTIFACTS, DATA, SPLITS, TASKS, digest, new_directory, read_json, seal, sha, verify, versions, write_json
from .data import valid_windows
from .encoding import cache_embeddings
from .fetch import Downloader, acquire_stocknet, acquire_weather


def fit_scaler(values, mask, train, allow_empty=False):
    valid = mask[train]
    count = valid.sum(0)
    if (count == 0).all() or ((count == 0).any() and not allow_empty):
        raise ValueError("Zero valid training observations for a fixed channel")
    v = np.where(valid, values[train], 0)
    mean = v.sum(0) / np.maximum(count, 1)
    variance = np.where(valid, values[train] - mean, 0) ** 2
    scale = np.sqrt(variance.sum(0) / np.maximum(count, 1))
    scale[scale == 0] = 1
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_counts": count.tolist(), "fit_scope": "train_valid_cells_only",
            "unidentified_channels": np.flatnonzero(count == 0).tolist(),
            "zero_training_observation_policy": "fixed mean=0 scale=1 representation constants, never fitted from val/test; original masks retained"}


def write_numeric(directory, dataset, channels, values, mask, times, origins, split_ids, segments, scaler, source, multimodal=False):
    lookback, horizons, expected_channels = TASKS[dataset]
    if len(channels) != expected_channels or len(set(channels)) != expected_channels:
        raise ValueError("Fixed channel universe mismatch")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Nonfinite normalized valid values")
    report = {}
    target_times = {}
    for split_index, split in enumerate(SPLITS):
        select = np.flatnonzero(split_ids == split_index)
        v, m, t, s = values[select].astype(np.float32), mask[select], times[select], segments[select]
        np.savez(directory / f"{split}.npz", values=v, mask=m, times=t, origins=origins[select], segments=s)
        windows = {}
        for horizon in horizons:
            starts = valid_windows(t, s, m, lookback, horizon, None if dataset == "stocknet" else 3600 * 10**9)
            np.save(directory / f"{split}_H{horizon}_starts.npy", starts)
            # All row starts were checked for complete L+H time continuity and
            # at least one true observation in BOTH history and target.
            windows[str(horizon)] = {"windows": len(starts), "lookback": lookback, "horizon": horizon,
                                    "all_L_and_H_dates_checked": True, "full_channel_count": expected_channels}
        target_times[split] = set(t.tolist())
        report[split] = {"points": len(select), "start": pd.Timestamp(t[0]).isoformat() if len(t) else None,
                         "end": pd.Timestamp(t[-1]).isoformat() if len(t) else None,
                         "valid_cells": int(m.sum()), "missing_cells": int((~m).sum()),
                         "all_missing_channels": [name for name, count in zip(channels, m.sum(0)) if count == 0], "horizons": windows}
    if any(target_times[a] & target_times[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Split target date overlap")
    manifest = {"dataset": dataset, "lookback": lookback, "horizons": horizons, "channels": channels,
                "channel_order_sha256": digest(channels), "scaler": scaler, "source": source,
                "multimodal_enabled": multimodal, "numerical_clock": "UTC_session_close_plus_30min" if dataset == "stocknet" else "unresolved_naive_original_labels",
                "splits": report, "all_windows_validated": True,
                "input_fill": "window-local past-only forward fill; normalized training mean fallback 0",
                "label_policy": "never fill true labels; safe placeholders excluded by shared y_mask"}
    write_json(directory / "manifest.json", manifest)
    return manifest


def fresh(dataset):
    directory = DATA / dataset
    if directory.exists() and not (directory / "_SUCCESS.json").exists():
        suffix = 1
        while directory.with_name(f"{dataset}_failed_{suffix}").exists():
            suffix += 1
        directory.rename(directory.with_name(f"{dataset}_failed_{suffix}"))
    return directory


def prepare_n2(dataset):
    directory = fresh(dataset)
    if (directory / "_SUCCESS.json").exists():
        verify(directory)
        return read_json(directory / "manifest.json")
    from scripts.fidel_ts.stage1_5_io import read_bundle
    from scripts.fidel_ts.common import load_source, dataset_directory, sha256_file
    from scripts.fidel_ts.numerical_diagnostics import load_native
    from scripts.fidel_ts.numerical_protocols import build_protocol
    source = ROOT / f"dataset/Fidel-TS/processed/{dataset}/n2_strict_contiguous"
    old_binding = read_bundle(source)
    spec = load_source(dataset)
    channels = read_json(source / "channels.json")["channels"]
    if channels != spec["expected_channels"]:
        raise ValueError("N2 channel order changed")
    scaler = read_json(source / "scaler.json")
    protocol = read_json(ROOT / "configs/fidel_ts/protocols_stage1_5.json")
    raw_path = dataset_directory(ROOT / "dataset/Fidel-TS/raw", spec) / spec["primary_file"]
    before = sha(raw_path)
    if before != old_binding["primary_sha256"]:
        raise ValueError("Original raw data changed")
    recomputed = build_protocol(load_native(raw_path, dataset, spec), "n2", protocol)["scaler"]
    scaler_ok = all(np.allclose(recomputed[key], scaler[key], rtol=1e-10, atol=1e-10) for key in ("mean", "scale", "fit_sample_counts"))
    if not scaler_ok or sha(raw_path) != before:
        raise ValueError("N2 train-only scaler or source hash mismatch")
    arrays = []
    for index, split in enumerate(SPLITS):
        table = pd.read_parquet(source / split / "timestamps.parquet")
        values = pd.read_parquet(source / split / "scaled.parquet")
        mask = pd.read_parquet(source / split / "validity_mask.parquet")
        if list(values) != channels or list(mask) != channels:
            raise ValueError("Fixed all-channel order mismatch")
        arrays.append((values.to_numpy(), mask.to_numpy(dtype=bool), pd.DatetimeIndex(table["timestamp"]).asi8,
                       table["native_segment_id"].to_numpy(), np.full(len(table), index)))
    with new_directory(directory) as output:
        manifest = write_numeric(output, dataset, channels, np.concatenate([a[0] for a in arrays]), np.concatenate([a[1] for a in arrays]),
                                 np.concatenate([a[2] for a in arrays]), np.full(sum(len(a[0]) for a in arrays), -1, dtype=np.int64),
                                 np.concatenate([a[4] for a in arrays]), np.concatenate([a[3] for a in arrays]),
                                 {"mean": scaler["mean"], "scale": scaler["scale"], "fit_counts": scaler["fit_sample_counts"],
                                  "fit_scope": scaler["fit_scope"], "independently_recomputed_train_native_scaler": True},
                                 {"N2_directory": str(source), "N2_success_sha256": sha(source / "_SUCCESS.json"),
                                  "raw_file": str(raw_path), "raw_sha256_before": before, "raw_sha256_after": sha(raw_path),
                                  "upstream_interpolation_risk_remaining": True})
        write_json(output / "quality.json", {"train_scaler_verified": scaler_ok, "raw_unchanged": before == sha(raw_path),
                                            "clock_alignment": "blocked_unknown_physical_timezone", "no_external_text_fabricated": True})
        seal(output, {"dataset": dataset, "kind": "numerical_only", "source_binding": old_binding["primary_sha256"]})
    return manifest


def load_raw_tweets(path):
    text = path.decode("utf-8") if isinstance(path, bytes) else path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def prepare_stocknet(client):
    directory = fresh("stocknet")
    if (directory / "_SUCCESS.json").exists():
        verify(directory)
        return read_json(directory / "manifest.json")
    record = acquire_stocknet(client)
    root = Path(record["root"])
    archive = zipfile.ZipFile(io.BytesIO(Path(record["archive"]).read_bytes()))
    members = {"/".join(name.split("/")[1:]): name for name in archive.namelist() if not name.endswith("/")}

    def source_bytes(name):
        body = archive.read(members[name])
        if hashlib.sha256(body).hexdigest() != record["files"][name]:
            raise ValueError("Original source member hash mismatch")
        return body
    table = list(csv.DictReader(io.StringIO(source_bytes("StockTable").decode()), delimiter="\t"))
    channels = [row["Symbol"].lstrip("$").strip() for row in table]
    if len(channels) != 88 or len(set(channels)) != 88:
        raise ValueError("Original StockTable does not define exactly 88 unique stocks")
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS", start="2013-12-01", end="2015-12-31")
    sessions = calendar.sessions
    days = [session.strftime("%Y-%m-%d") for session in sessions]
    closes = pd.DatetimeIndex([calendar.session_close(session) for session in sessions])
    if closes.tz is None:
        raise ValueError("Calendar must expose aware actual close times")
    origins = (closes + pd.Timedelta(minutes=30)).asi8
    prices = np.full((len(days), 88), np.nan)
    issues, source_price_dates = [], set()
    for c, channel in enumerate(channels):
        name = f"price/raw/{channel}.csv"
        if name not in members:
            issues.append({"entity": channel, "issue": "missing_price_file_no_target_removed"})
            continue
        frame = pd.read_csv(io.BytesIO(source_bytes(name)))
        if not {"Date", "Close"}.issubset(frame):
            raise ValueError(f"Unsupported explicit price schema: {channel}")
        grouped = frame.groupby("Date")["Close"].agg(lambda values: values.iloc[0] if values.nunique(dropna=False) == 1 else np.nan)
        mapped = grouped.reindex(days).to_numpy(dtype=float)
        prices[:, c] = np.where(np.isfinite(mapped) & (mapped > 0), mapped, np.nan)
        source_price_dates.update(date for date in frame["Date"].tolist() if "2014-01-01" <= date < "2016-01-01")
    returns = np.full_like(prices, np.nan)
    valid_prices = np.isfinite(prices)
    valid = valid_prices[1:] & valid_prices[:-1]
    ratio = np.divide(prices[1:], prices[:-1], out=np.ones_like(prices[1:]), where=valid)
    returns[1:] = np.where(valid, np.log(ratio), np.nan)
    selected = np.array(["2014-01-01" <= day < "2016-01-01" for day in days])
    times = pd.to_datetime(np.array(days)[selected]).asi8
    origins = origins[selected]
    values = returns[selected]
    mask = np.isfinite(values)
    dates = np.array(days)[selected]
    splits = np.where(dates < "2015-07-01", 0, np.where(dates < "2015-10-01", 1, 2))
    scaler = fit_scaler(values, mask, splits == 0, allow_empty=True)
    scaled = (values - np.array(scaler["mean"])) / np.array(scaler["scale"])
    # Preserve source original text. No price/label values enter aggregation.
    from scripts.pivot_sources.stocknet import parse_created_at
    unique = {}
    versions_by_id = defaultdict(set)
    text_issues = Counter()
    entities_with_raw_tweets = set()
    lower, upper = pd.Timestamp("2013-12-01", tz="UTC").value, pd.Timestamp("2016-02-01", tz="UTC").value
    for c, entity in enumerate(channels):
        for name in sorted(name for name in record["files"] if name.startswith(f"tweet/raw/{entity}/")):
            try:
                tweets = load_raw_tweets(source_bytes(name))
            except (ValueError, UnicodeError):
                text_issues["invalid_raw_json_files"] += 1
                continue
            for item in tweets:
                if not isinstance(item, dict):
                    text_issues["invalid_record_shape"] += 1
                    continue
                text = item.get("full_text") or item.get("text")
                identifier = str(item.get("id_str") or item.get("id") or "")
                if not isinstance(text, str) or not text.strip() or not identifier:
                    text_issues["missing_text_or_id"] += 1
                    continue
                try:
                    created = pd.Timestamp(parse_created_at(item.get("created_at"))).value
                except ValueError:
                    text_issues["invalid_created_at"] += 1
                    continue
                if not lower <= created < upper:
                    continue
                text_hash = digest(text)
                key = (entity, identifier, text_hash)
                if key in unique:
                    text_issues["duplicate_entity_tweet_text"] += 1
                    continue
                versions_by_id[(entity, identifier)].add(text_hash)
                unique[key] = {"entity_id": entity, "id": identifier, "created_ns": int(created), "text": text,
                               "source_file": name, "text_sha256": text_hash}
                entities_with_raw_tweets.add(entity)
        print(f"StockNet source parse {c + 1}/88", flush=True)
    groups = defaultdict(list)
    for tweet in unique.values():
        if len(versions_by_id[(tweet["entity_id"], tweet["id"])]) > 1:
            text_issues["ambiguous_same_id_different_text_quarantined"] += 1
            continue
        day = pd.Timestamp(tweet["created_ns"], tz="UTC").strftime("%Y-%m-%d")
        groups[(tweet["entity_id"], day)].append(tweet)
    documents = []
    for (entity, day), members in sorted(groups.items()):
        members.sort(key=lambda row: (row["created_ns"], row["id"]))
        latest = max(member["created_ns"] for member in members)
        identifiers = [(member["id"], member["text_sha256"]) for member in members]
        documents.append({"document_id": digest([record["commit"], entity, day, identifiers]), "source": "stocknet_original_twitter",
                          "entity_id": entity, "source_time_key": day, "text": "\n".join(member["text"] for member in members)[:2048],
                          "issued_at_ns": latest, "received_at_ns": None, "available_at_ns": latest + 3600 * 10**9,
                          "availability_policy": "creation_time_plus_1h_experiment_assumption_not_verified_receipt",
                          "strict_verified_asof": False, "aggregation": "entity_UTC_day_latest_member_time_conservative",
                          "member_count": len(members), "member_ids_and_text_hashes": identifiers,
                          "source_files": sorted({member["source_file"] for member in members}), "source_commit": record["commit"]})
    extremes = np.argwhere(mask & (abs(values) > 0.2))
    with new_directory(directory) as output:
        manifest = write_numeric(output, "stocknet", channels, scaled, mask, times, origins, splits, np.zeros(len(times), int), scaler,
            {"commit": record["commit"], "archive_sha256": record["archive_sha256"], "source_manifest": str(root / "manifest.json"),
             "source_manifest_sha256": sha(root / "manifest.json"), "raw_Close_semantics": "published_Yahoo_Close_not_a_verified_point_in_time_vintage",
             "corporate_action_and_back_adjustment_risk": "unresolved_disclosed_no_adjusted_close_used"}, bool(documents))
        with (output / "documents.jsonl").open("x", encoding="utf-8") as handle:
            for doc in documents:
                handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        pd.DataFrame({"session": dates, "origin_UTC": pd.to_datetime(origins, utc=True).astype(str)}).to_csv(output / "calendar.csv", index=False)
        encoder_report = cache_embeddings(output / "text", documents, channels, pd.Timestamp("2015-07-01", tz="UTC").value - 1, client)
        quality = {"price_issues": issues, "text_issues": dict(text_issues), "parsed_unique_tweet_entity_records": len(unique),
                   "aggregate_documents": len(documents), "entities_with_observed_text": len(entities_with_raw_tweets),
                   "calendar": {"market": "XNYS", "package_version": xcals.__version__, "actual_close_plus_minutes": 30,
                                "source_price_dates_not_in_calendar": sorted(source_price_dates - set(dates)),
                                "calendar_sessions_without_any_valid_price": dates[~valid_prices[selected].any(1)].tolist()},
                   "return_abs_above_20pct": [{"date": str(dates[i]), "channel": channels[c], "log_return": float(values[i, c])} for i, c in extremes],
                   "return_anomalies_removed": False, "adjusted_close_used": False, "embedding": encoder_report,
                   "structurally_unsupervised_channels": [channels[i] for i in scaler["unidentified_channels"]],
                   "effective_supervised_channel_count": int((np.asarray(scaler["fit_counts"]) > 0).sum()),
                   "no_future_prices_or_returns_used_to_generate_text": True, "daily_aggregation_may_delay_earlier_tweets": True}
        write_json(output / "quality.json", quality)
        seal(output, {"dataset": "stocknet", "source_commit": record["commit"], "kind": "conditional_creation_time_multimodal"})
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", *TASKS), default="all")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    client = Downloader()
    results = {}
    targets = [args.dataset] if args.dataset != "all" else ["stocknet", "caiso", "jena"]
    for dataset in targets:
        start = time.perf_counter()
        try:
            result = prepare_stocknet(client) if dataset == "stocknet" else prepare_n2(dataset)
            results[dataset] = {"status": "prepared", "seconds": time.perf_counter() - start, "splits": result["splits"]}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[dataset] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(dataset, results[dataset]["status"], flush=True)
    write_json(ARTIFACTS / "preparation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
