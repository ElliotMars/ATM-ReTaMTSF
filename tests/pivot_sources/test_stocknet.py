import csv
from datetime import datetime, timedelta
import math
import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import pytest

from scripts.pivot_sources.common import SourceError, load_config
from scripts.pivot_sources.stocknet import (
    calendar_rows, inventory, normalize_tweet, parse_created_at, parse_prices, price_task, tweet_audit,
)
from scripts.pivot_sources.probe import git_blob_sha, stock_probe


def test_created_at_is_aware_utc_and_creation_is_not_receipt():
    assert parse_created_at("Mon Jul 20 12:34:56 +0000 2015") == "2015-07-20T12:34:56+00:00"
    row = normalize_tweet({"id_str": "1", "text": "fixture", "created_at": "2015-07-20T12:34:56Z"}, "A", "https://example.org/t", "raw/a")
    assert row["received_at"] is None and row["available_at"] is None
    with pytest.raises(SourceError):
        parse_created_at("2015-07-20T12:34:56")


def test_actual_session_close_tracks_dst_holidays_and_early_close(tmp_path):
    path = tmp_path / "calendar.csv"
    path.write_text("session,open,close\n2015-03-06,2015-03-06T14:30:00Z,2015-03-06T21:00:00Z\n"
                    "2015-03-09,2015-03-09T13:30:00Z,2015-03-09T20:00:00Z\n"
                    "2015-11-25,2015-11-25T14:30:00Z,2015-11-25T21:00:00Z\n"
                    "2015-11-27,2015-11-27T14:30:00Z,2015-11-27T18:00:00Z\n")
    rows = calendar_rows(path)
    closes = [datetime.fromisoformat(row["close"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York")) for row in rows]
    assert [value.hour for value in closes] == [16, 16, 16, 13]
    assert "2015-03-07" not in [row["session"] for row in rows]  # weekend
    assert "2015-11-26" not in [row["session"] for row in rows]  # holiday absent from supplied calendar


def test_pinned_inventory_retains_no_tweet_stocks_and_flags_truncation():
    tree = {"truncated": False, "tree": [{"type": "blob", "path": "price/raw/A.csv", "size": 10, "sha": "b" * 40},
                                               {"type": "blob", "path": "price/raw/B.csv", "size": 20, "sha": "c" * 40},
                                               {"type": "blob", "path": "tweet/raw/A/2015-01-01", "size": 30, "sha": "d" * 40}]}
    result = inventory(tree, "a" * 40)
    assert result["fixed_universe"] == ["A", "B"]
    assert result["entities_missing_tweet_files"] == ["B"] and result["universe_count"] == 2
    assert result["total_known_file_bytes"] == 60
    assert inventory({**tree, "truncated": True}, "a" * 40)["tree_truncated"]


def test_close_returns_do_not_bridge_a_missing_official_trading_day():
    spec = load_config()["sources"]["stocknet"].copy()
    spec["lookback"], spec["horizons"] = 1, [1]
    calendar = [{"session": f"2015-06-{day:02d}"} for day in (1, 2, 3, 4)]
    a = parse_prices(b"Date,Close,Adj Close\n2015-06-01,100,50\n2015-06-02,110,55\n2015-06-04,121,60.5\n")
    report = price_task({"A": a}, ["A", "B"], calendar, spec)
    values = report["returns"]["A"]
    assert values[1]["log_return"] == pytest.approx(math.log(1.1))
    assert values[2]["log_return"] is None and values[3]["log_return"] is None
    assert "B" in report["fixed_universe"] and report["common_valid_trading_days"] is None
    assert not report["adjusted_close_default_allowed"]
    assert report["missing_sessions_by_entity"]["B"] == [row["session"] for row in calendar]


def test_stocknet_candidate_split_boundaries():
    spec = load_config()["sources"]["stocknet"].copy()
    spec["lookback"], spec["horizons"] = 1, [1]
    calendar = [{"session": day} for day in ("2015-06-29", "2015-06-30", "2015-07-01", "2015-09-30", "2015-10-01", "2015-12-31")]
    result = price_task({}, ["A"], calendar, spec)
    assert [result["per_split_windows"][split]["sessions"] for split in ("train", "validation", "test")] == [2, 2, 2]


def test_duplicate_prices_are_ambiguous_and_adjustment_is_not_cleared():
    result = parse_prices(b"Date,Close,Adj Close\n2015-01-02,100,50\n2015-01-02,110,55\n")
    assert result["prices"]["2015-01-02"]["Close"] is None
    assert result["future_adjustment_risk"].startswith("requires")


def test_duplicate_tweets_missing_text_and_embedded_older_creation():
    item = {"id": 1, "text": "fixture", "created_at": "2015-07-20T12:00:00Z",
            "quoted_status": {"id": 2, "created_at": "2015-07-19T12:00:00Z"}}
    record = normalize_tweet(item, "A", "https://example.org/t", "raw/t")
    report = tweet_audit([record, record], ["A", "B"], [], load_config()["sources"]["stocknet"])
    assert report["duplicate_id_occurrences"] == 1
    assert report["universe_entities_without_observed_text"] == ["B"]
    assert record["embedded_content"][0]["created_at"] < record["issued_at"]
    with pytest.raises(SourceError, match="text missing"):
        normalize_tweet({**item, "text": ""}, "A", "https://example.org/t", "raw/t")


def test_shared_tweet_keeps_distinct_entity_associations_and_asof_lag(evidence_factory):
    spec = load_config()["sources"]["stocknet"]
    item = {"id": 1, "text": "fixture", "created_at": "2015-07-20T20:00:00Z"}
    a = normalize_tweet(item, "A", "https://example.org/t", "raw/a", evidence_factory("creation"))
    b = normalize_tweet(item, "B", "https://example.org/t", "raw/b", evidence_factory("creation"))
    assert a["document_id"] != b["document_id"]
    calendar = [{"session": "2015-07-17", "close": "2015-07-17T20:00:00Z"},
                {"session": "2015-07-20", "close": "2015-07-20T20:00:00Z"}]
    row = tweet_audit([a, b], ["A", "B"], calendar, spec)["session_coverage"][0]
    assert row["asof_policy_candidates"]["source_time+0h"]["tweet_counts_by_entity"] == {"A": 1, "B": 1}
    assert row["asof_policy_candidates"]["source_time+1h"]["tweet_counts_by_entity"] == {}
    assert row["asof_policy_candidates"]["strict+0h"]["tweet_counts_by_entity"] is None


def test_stock_probe_uses_bounded_directory_walk_and_pinned_file_urls():
    spec = load_config()["sources"]["stocknet"]
    repository = spec["repository"]
    commit, trees = "a" * 40, [str(i) * 40 for i in range(1, 7)]
    price = b"Date,Close\n2015-07-15,100\n"
    tweet = json.dumps({"id": 1, "created_at": "2015-07-15T12:00:00Z", "text": "fixture"}).encode()
    tree_nodes = [
        [{"path": "price", "type": "tree", "sha": trees[1]}, {"path": "tweet", "type": "tree", "sha": trees[3]}],
        [{"path": "raw", "type": "tree", "sha": trees[2]}],
        [{"path": name + ".csv", "type": "blob", "sha": git_blob_sha(price), "size": len(price)} for name in ("A", "B")],
        [{"path": "raw", "type": "tree", "sha": trees[4]}],
        [{"path": "A", "type": "tree", "sha": trees[5]}],
        [{"path": "2015-07-15", "type": "blob", "sha": git_blob_sha(tweet), "size": len(tweet)}]]
    responses = {f"https://api.github.com/repos/{repository}/commits/HEAD":
                 json.dumps({"sha": commit, "commit": {"tree": {"sha": trees[0]}}}).encode()}
    for sha, nodes in zip(trees, tree_nodes):
        responses[f"https://api.github.com/repos/{repository}/git/trees/{sha}"] = json.dumps({"sha": sha, "tree": nodes, "truncated": False}).encode()
    responses[f"https://raw.githubusercontent.com/{repository}/{commit}/price/raw/A.csv"] = price
    responses[f"https://raw.githubusercontent.com/{repository}/{commit}/tweet/raw/A/2015-07-15"] = tweet

    class LocalClient:
        calls = []

        def fetch(self, url):
            self.calls.append(url)
            body = responses[url]
            return body, {"raw_file": f"raw/{len(self.calls)}", "sha256": "b" * 64, "received_at_now": "2026-11-05T00:00:00Z"}

    args = SimpleNamespace(revision="HEAD", max_tweet_directories=1, max_price_files=1, max_tweet_files=1,
                           max_records=2, tweet_time_evidence=None)
    client = LocalClient()
    result = stock_probe(spec, client, args)
    assert len(client.calls) == 9 and all("recursive=" not in url for url in client.calls)
    assert result["listing"]["fixed_universe"] == ["A", "B"]
    assert result["listing"]["revision"] == commit and not result["listing"]["inventory_complete"]
    assert result["records"][0]["entity_id"] == "A"
