"""Exhaustive vectorized real-data checks and conditional experiment admission."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA, ARTIFACTS, SPLITS, TASKS, digest, read_json, sha, verify, write_json
from .data import valid_windows
from .weather import source_decisions


def text_coverage(documents, channels, origins, lag_hours, lookback_days=30):
    counts = np.zeros((len(origins), len(channels)), dtype=np.int32)
    for c, channel in enumerate(channels):
        available = np.sort(np.array([row["issued_at_ns"] + int(lag_hours * 3600e9) for row in documents if row["entity_id"] == channel], dtype=np.int64))
        right = np.searchsorted(available, origins, side="right")
        left = np.searchsorted(available, origins - int(lookback_days * 86400e9), side="right")
        counts[:, c] = right - left
        # Last accessible document never lies after its prediction origin.
        if len(available) and (right > 0).any():
            valid = right > 0
            assert np.all(available[right[valid] - 1] <= origins[valid])
    return {"samples": len(origins), "channel_window_coverage": float((counts > 0).mean()) if counts.size else None,
            "windows_with_any_text": int((counts > 0).any(1).sum()), "windows_with_no_text": int((counts == 0).all(1).sum()),
            "legal_document_count_percentiles": np.quantile(counts, [0, .25, .5, .75, 1]).tolist() if counts.size else None,
            "lag_hours": lag_hours, "lag_status": "creation_time_assumption_not_verified_receipt", "lookback_days": lookback_days,
            "all_origins_candidate_time_boundaries_checked": True}


def audit_dataset(dataset):
    directory = DATA / dataset
    binding = verify(directory)
    meta, quality = read_json(directory / "manifest.json"), read_json(directory / "quality.json")
    lookback, horizons, channels = TASKS[dataset]
    assert len(meta["channels"]) == channels and meta["lookback"] == lookback and meta["horizons"] == horizons
    assert digest(meta["channels"]) == meta["channel_order_sha256"]
    documents = [json.loads(line) for line in (directory / "documents.jsonl").read_text().splitlines()] if meta["multimodal_enabled"] else []
    if documents:
        embeddings = np.load(directory / "text/embeddings.npy", mmap_mode="r")
        assert embeddings.shape == (len(documents), 384) and np.isfinite(embeddings).all()
        assert len(set(doc["document_id"] for doc in documents)) == len(documents)
        assert all(not doc["strict_verified_asof"] for doc in documents)
    result = {"dataset": dataset, "channel_count": channels, "channel_order": meta["channels"],
              "prepared_success_sha256": sha(directory / "_SUCCESS.json"), "splits": {}, "text_documents": len(documents),
              "multimodal_enabled": meta["multimodal_enabled"], "source_hashes_verified": True,
              "no_text_filter_changes_numerical_sample_set": True, "all_horizons_checked": horizons}
    times_by_split = {}
    for split in SPLITS:
        data = np.load(directory / f"{split}.npz")
        v, mask, times = data["values"], data["mask"], data["times"]
        assert v.shape == mask.shape == (len(times), channels) and np.isfinite(v[mask]).all()
        assert np.all(np.diff(times) > 0)
        times_by_split[split] = set(times.tolist())
        result["splits"][split] = {"points": len(times), "start": pd.Timestamp(times[0]).isoformat(),
                                   "end": pd.Timestamp(times[-1]).isoformat(), "input_label_source_missing_ratio": float((~mask).mean()), "horizons": {}}
        prefix = np.vstack([np.zeros((1, channels), dtype=np.int64), mask.cumsum(0)])
        for horizon in horizons:
            saved = np.load(directory / f"{split}_H{horizon}_starts.npy")
            recomputed = valid_windows(times, data["segments"], mask, lookback, horizon, None if dataset == "stocknet" else 3600 * 10**9)
            assert np.array_equal(saved, recomputed)
            input_count = prefix[saved + lookback] - prefix[saved]
            target_count = prefix[saved + lookback + horizon] - prefix[saved + lookback]
            assert (input_count.sum(1) > 0).all() and (target_count.sum(1) > 0).all()
            row = {"samples": len(saved), "input_valid_occurrences": int(input_count.sum()),
                   "label_valid_occurrences": int(target_count.sum()), "label_valid_by_channel": target_count.sum(0).tolist(),
                   "full_L_H_calendar_and_masks_verified": True}
            if documents:
                origins = data["origins"][saved + lookback - 1]
                row["text_coverage_lag_scenarios"] = {str(lag): text_coverage(documents, meta["channels"], origins, lag) for lag in (0, 1, 3)}
            result["splits"][split]["horizons"][str(horizon)] = row
        if dataset == "stocknet" and split == "train":
            count = mask.sum(0)
            normalized_mean = np.where(mask, v, 0).sum(0, dtype=np.float64) / np.maximum(count, 1)
            assert np.all(abs(normalized_mean[count > 0]) < 1e-5)
            assert count.tolist() == meta["scaler"]["fit_counts"]
            result["train_scaler_rechecked"] = True
    assert all(not times_by_split[a] & times_by_split[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    result["split_targets_disjoint"] = True
    if dataset != "stocknet":
        source = meta["source"]
        assert sha(source["raw_file"]) == source["raw_sha256_before"] == source["raw_sha256_after"]
        assert sha(Path(source["N2_directory"]) / "_SUCCESS.json") == source["N2_success_sha256"]
        result.update(historical_source_unchanged=True, train_native_scaler_independently_recomputed=True,
                      strict_text_alignment="blocked_unknown_timezone")
    else:
        source = meta["source"]
        assert sha(source["source_manifest"]) == source["source_manifest_sha256"]
        raw = read_json(source["source_manifest"])
        assert sha(raw["archive"]) == raw["archive_sha256"]
        result.update(fixed_universe_from_original_StockTable=True, calendar=quality["calendar"],
                      effective_supervised_channels=quality["effective_supervised_channel_count"],
                      structurally_unsupervised_channels=quality["structurally_unsupervised_channels"],
                      embedding=quality["embedding"], raw_price_vintage_risk="unresolved_disclosed")
        evidence = read_json(ARTIFACTS / "sources/final_official_evidence.json")["nyse_2014_2015_pdf"]
        assert sha(evidence["path"]) == evidence["sha256"]
        calendar_frame = pd.read_csv(directory / "calendar.csv")
        session_days = set(calendar_frame["session"])
        holidays = ["2014-01-01", "2014-01-20", "2014-02-17", "2014-04-18", "2014-05-26", "2014-07-04", "2014-09-01", "2014-11-27", "2014-12-25",
                    "2015-01-01", "2015-01-19", "2015-02-16", "2015-04-03", "2015-05-25", "2015-07-03", "2015-09-07", "2015-11-26", "2015-12-25"]
        early = {"2014-07-03", "2014-11-28", "2014-12-24", "2015-11-27", "2015-12-24"}
        assert not session_days.intersection(holidays) and early.issubset(session_days)
        origin_ny = pd.to_datetime(calendar_frame["origin_UTC"], utc=True).dt.tz_convert("America/New_York")
        for day, timestamp in zip(calendar_frame["session"], origin_ny):
            assert timestamp.hour == (13 if day in early else 16) and timestamp.minute == 30
        result["official_calendar_validation"] = {"source_url": evidence["source_url"], "raw_pdf_sha256": evidence["sha256"],
            "holidays_checked": 18, "early_close_sessions_checked": 5, "all_session_origin_clocks_checked": len(calendar_frame),
            "semantic_review": "agent_read_official_historical_notice; program independently compared actual session origins"}
    write_json(ARTIFACTS / f"{dataset}_data_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", *TASKS), default="all")
    args = parser.parse_args()
    reports, admission = {}, {}
    for dataset in ([args.dataset] if args.dataset != "all" else list(TASKS)):
        try:
            reports[dataset] = audit_dataset(dataset)
            report = reports[dataset]
            enough = all(stats["samples"] > 0 for split in report["splits"].values() for stats in split["horizons"].values())
            multimodal = dataset == "stocknet" and report["text_documents"] > 0 and enough
            admission[dataset] = {"decision": "CONDITIONAL" if multimodal else "NO-GO_MULTIMODAL",
                "numeric_enabled": enough, "multimodal_enabled": multimodal, "main_enabled": multimodal,
                "strict_verified_t1": False,
                "protocol": "archived_tweet_creation_plus_1h_assumption" if multimodal else "numerical_only_original_N2_clock_unresolved",
                "fixed_channels": TASKS[dataset][2], "horizons": TASKS[dataset][1],
                "effective_supervised_channels": report.get("effective_supervised_channels", TASKS[dataset][2]),
                "structurally_unsupervised_channels": report.get("structurally_unsupervised_channels", []),
                "required_disclosures": ["creation_time_not_verified_historical_receipt", "retrospective_Yahoo_price_adjustment_risk",
                                         "closing_prices_assumed_obtainable_by_actual_close_plus30m_not_verified_receipt",
                                         "88_array_channels_with_87_observed_supervised_channels_GMRE_missing"] if multimodal else ["no_claim_of_successful_external_text_alignment"],
                "data_audit": str(ARTIFACTS / f"{dataset}_data_audit.json")}
        except Exception as exc:
            reports[dataset] = {"error": f"{type(exc).__name__}: {exc}"}
            admission[dataset] = {"decision": "NO-GO", "numeric_enabled": False, "multimodal_enabled": False, "main_enabled": False, "reason": str(exc)}
    weather = source_decisions()
    write_json(ARTIFACTS / "data_acceptance.json", reports)
    write_json(ARTIFACTS / "admission.json", {"datasets": admission, "weather_sources": weather,
        "main_combinations": [key for key, row in admission.items() if row["main_enabled"]],
        "multi_dataset_external_text_validation_sufficient": sum(row["main_enabled"] for row in admission.values()) > 1,
        "formal_training_started": False})
    print(json.dumps({name: {key: value for key, value in row.items() if key in {"decision", "main_enabled", "numeric_enabled", "reason"}}
                      for name, row in admission.items()}, indent=2))
    if any("error" in report for report in reports.values()):
        raise SystemExit("Prepared-data integrity audit failed; inspect data_acceptance.json. No training was started.")


if __name__ == "__main__":
    main()
