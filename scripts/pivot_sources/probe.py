"""Explicit small probes. No unverified historical API endpoint is guessed."""

from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from scripts.pivot_sources.asof import utc
from scripts.pivot_sources.common import (
    SourceError, discover_links, evidence_ok, mapped_records, read_json,
)
from scripts.pivot_sources.http import BoundedHTTP, validate_url
from scripts.pivot_sources import gfs_archive, nws_iem, stocknet


def validate_period(start, end, spec):
    start_time, end_time = utc(start), utc(end)
    bounds = spec["numerical_calendar_range"]
    if not start_time < end_time or end_time - start_time > timedelta(days=2):
        raise SourceError("A probe request must cover at most two days with start < end")
    if start_time.date().isoformat() < bounds[0] or end_time.date().isoformat() > bounds[1]:
        raise SourceError("Probe request lies outside the configured numerical calendar range")
    return start_time, end_time


def documentation_probe(client, spec):
    documents = []
    for url in spec["documentation_urls"]:
        body, response = client.fetch(url)
        documents.append({"url": url, "response": response, "links": discover_links(body, url),
                          "interpretation": "saved reference page; does not establish historical record accessibility"})
    return documents


def planned_probe(source, spec, client, plan, args):
    documents = documentation_probe(client, spec)
    result = {"documentation": documents, "records": [], "request_results": [], "parse_issues": [],
              "contract_status": "missing_documented_request_plan", "records_truncated": False,
              "actual_data_access_checked": False, "raw_samples": []}
    if plan is None:
        return result
    validate_url(plan["interface_evidence"]["source_url"], spec["allowed_hosts"])
    contract = plan.get("response_contract", {})
    result["response_contract"] = contract
    result["interface_evidence"] = plan["interface_evidence"]
    small_count = 0
    record_requests = max(1, sum(request.get("kind", "records") in {"records", "metadata"} for request in plan["requests"]))
    for request in plan.get("requests", []):
        if len(result["records"]) >= args.max_records:
            result["records_truncated"] = True
            break
        kind = request.get("kind", "records")
        method = request.get("method", "GET")
        if source == "nws" and kind == "records":
            start, end = request.get("start"), request.get("end")
            validate_period(start, end, spec)
            if args.start and (utc(start) < utc(args.start) or utc(end) > utc(args.end)):
                raise SourceError("Plan request lies outside the explicit CLI date range")
            afos = request.get("afos_ids", [])
            if not afos or not set(afos).issubset(set(args.afos_id or spec["afos_ids"])):
                raise SourceError("Every NWS request must bind requested AFOS IDs")
        if source == "gfs":
            if kind not in {"catalog", "metadata", "records", "grib_index", "small_sample", "file_head"}:
                raise SourceError("Unsupported GFS probe kind")
            looks_global = bool(re.search(r"(?:grib2?|grb2?|pgrb2)", request["url"], re.I))
            if kind == "file_head" and method != "HEAD":
                raise SourceError("Global file-size inspection must use HEAD")
            if (looks_global or request.get("binary")) and method != "HEAD" and kind != "small_sample" and kind != "grib_index":
                raise SourceError("Global GRIB download refused; metadata/index or reviewed tiny sample only")
            if kind == "small_sample":
                if not args.allow_small_sample or small_count or not evidence_ok(request.get("small_sample_evidence"), ("bounded_forecast_sample",)):
                    raise SourceError("At most one explicitly allowed, documentation-bound tiny forecast sample")
                if not isinstance(request.get("expected_bytes"), int) or not 0 < request["expected_bytes"] <= min(2 * 1024**2, client.limits["max_file_bytes"]):
                    raise SourceError("Tiny forecast sample must have an evidenced size within the byte cap")
                small_count += 1
        body, response = client.fetch(request["url"], method, allow_binary=kind == "small_sample")
        result["request_results"].append({"request": request, "response": response})
        if kind == "small_sample":
            if len(body) != request["expected_bytes"]:
                raise SourceError("Actual forecast sample size differs from reviewed size")
            result["raw_samples"].append(response["raw_file"])
        if kind not in {"records", "metadata"} or method == "HEAD":
            continue
        if source == "nws" and contract.get("format") == "nws_text":
            if not request.get("product_id"):
                raise SourceError("A raw bulletin request must bind its full documented Product ID")
            values = [{"product_id": request["product_id"], "text": body.decode("utf-8")}]
        else:
            values = mapped_records(body, contract)
        remaining = min(args.max_records - len(result["records"]), math.ceil(args.max_records / record_requests))
        result["records_truncated"] |= len(values) > remaining
        for index, item in enumerate(values[:remaining]):
            try:
                if source == "nws":
                    record = nws_iem.normalize(item, contract, request["url"], response["raw_file"], afos, start, end)
                else:
                    record = gfs_archive.normalize(item, contract, request["url"], response["raw_file"])
                    run = utc(record["run_time"])
                    validate_period(run.isoformat(), (run + timedelta(hours=1)).isoformat(), spec)
                    if run.hour not in spec["cycles_utc"] or run.minute or run.second:
                        raise SourceError("Unexpected GFS cycle; retain as a failed metadata check")
                    if args.start and not utc(args.start) <= run < utc(args.end):
                        raise SourceError("GFS run lies outside CLI date range")
                record["retrieved_at_now"] = response["received_at_now"]
                record["raw_file_sha256"] = response["sha256"]
                result["records"].append(record)
            except (SourceError, TypeError, KeyError, ValueError) as exc:
                result["parse_issues"].append({"raw_file": response["raw_file"], "record_index": index,
                                               "error": str(exc), "record_rejected": True})
        result["actual_data_access_checked"] |= bool(values)
    result["contract_status"] = "documentation_bound_plan_used_not_full_archive_verification"
    return result


def git_blob_sha(body):
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()


def stock_probe(spec, client, args):
    repository = spec["repository"]
    if repository != "yumoxu/stocknet-dataset":
        raise SourceError("Unexpected StockNet repository")
    revision = args.revision
    if revision != "HEAD" and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SourceError("Use HEAD to resolve once or an immutable 40-character commit")
    # GitHub's commits/tree API is used only by this explicit probe. The exact
    # returned commit/tree SHA is pinned before any raw-file URL is constructed.
    commit_url = f"https://api.github.com/repos/{repository}/commits/{revision}"
    body, commit_response = client.fetch(commit_url)
    commit = json.loads(body)
    resolved = commit.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", resolved) or revision != "HEAD" and resolved != revision:
        raise SourceError("GitHub did not return the asserted immutable commit")
    tree_sha = commit["commit"]["tree"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        raise SourceError("Invalid pinned Git tree SHA")
    visited = []
    flattened = []

    def read_tree(identifier, prefix=""):
        url = f"https://api.github.com/repos/{repository}/git/trees/{identifier}"
        data, response = client.fetch(url)
        value = json.loads(data)
        if value.get("sha") != identifier:
            raise SourceError("Pinned Git tree SHA mismatch")
        visited.append({"prefix": prefix, "response": response, "truncated": value.get("truncated")})
        for entry in value.get("tree", []):
            flattened.append({**entry, "path": prefix + entry["path"]})
        return value.get("tree", [])

    def child(nodes, name):
        match = next((entry for entry in nodes if entry.get("path") == name and entry.get("type") == "tree"), None)
        if match is None:
            raise SourceError(f"Expected directory {name} not found in actual Git tree; inspect saved metadata")
        return match["sha"]

    # Avoid the potentially enormous recursive tree. Inspect small exact
    # directories, then only a bounded number of real tweet-entity directories.
    root_nodes = read_tree(tree_sha)
    price_nodes = read_tree(child(root_nodes, "price"), "price/")
    read_tree(child(price_nodes, "raw"), "price/raw/")
    tweet_nodes = read_tree(child(root_nodes, "tweet"), "tweet/")
    tweet_raw_nodes = read_tree(child(tweet_nodes, "raw"), "tweet/raw/")
    tweet_entities = sorted(entry["path"] for entry in tweet_raw_nodes if entry.get("type") == "tree")
    selected_entities = tweet_entities[:args.max_tweet_directories]
    for entity in selected_entities:
        read_tree(child(tweet_raw_nodes, entity), f"tweet/raw/{entity}/")
    tree = {"tree": flattened, "truncated": any(item["truncated"] is not False for item in visited)}
    listing = stocknet.inventory(tree, resolved, spec.get("universe"))
    listing.update(commit_response=commit_response, directory_responses=visited,
                   directory_presence={"price/raw": True, "tweet/raw": True,
                       "price/preprocessed": any(entry.get("path") == "preprocessed" and entry.get("type") == "tree" for entry in price_nodes),
                       "tweet/preprocessed": any(entry.get("path") == "preprocessed" and entry.get("type") == "tree" for entry in tweet_nodes)},
                   metadata_scope="bounded_directory_walk_not_full_recursive_repository_tree",
                   inventory_complete=False, tweet_entity_directories=tweet_entities,
                   inspected_tweet_entity_directories=selected_entities,
                   access_interface="GitHub REST commit/tree and immutable raw files; runtime response not yet checked by agent")
    listing["entities_without_tweet_directory"] = sorted(set(listing["fixed_universe"]) - set(tweet_entities))
    listing["entities_missing_tweet_files"] = None  # unvisited directories are unknown, not empty
    candidates = listing["files"]
    prices = sorted((entry for entry in candidates if entry["path"].startswith("price/raw/") and entry["entity_id"]), key=lambda entry: entry["path"])
    tweets = sorted((entry for entry in candidates if entry["path"].startswith("tweet/raw/")), key=lambda entry: entry["path"])
    chosen_tweets = []
    # Deterministic cross-split samples; files are selected only from the actual
    # tree, with no assumption that these paths/dates exist.
    for index, wanted in enumerate(spec["probe_dates"]):
        matches = [entry for entry in tweets if PurePosixPath(entry["path"]).name.startswith(wanted)]
        if matches:
            selected = matches[index % len(matches)]
            if selected not in chosen_tweets:
                chosen_tweets.append(selected)
    chosen_tweets.extend(entry for entry in tweets if entry not in chosen_tweets)
    picked = prices[:args.max_price_files] + chosen_tweets[:args.max_tweet_files]
    raw_files, records, issues = [], [], []
    truncated = False
    tweet_time_evidence = read_json(args.tweet_time_evidence) if args.tweet_time_evidence else None
    # README is discovered from the actual tree rather than assuming its case.
    readme = next((entry for entry in root_nodes if entry.get("type") == "blob" and entry["path"].lower() == "readme.md"), None)
    if readme:
        picked = [{"path": readme["path"], "git_blob_sha1": readme["sha"], "entity_id": None}] + picked
    for entry in picked:
        if entry["path"].startswith("tweet/") and len(records) >= args.max_records:
            break
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or any(not re.fullmatch(r"[A-Za-z0-9._^=-]+", part) for part in path.parts):
            raise SourceError("Unsafe/unexpected Git tree path")
        url = f"https://raw.githubusercontent.com/{repository}/{resolved}/{path.as_posix()}"
        body, response = client.fetch(url)
        if git_blob_sha(body) != entry["git_blob_sha1"]:
            raise SourceError("Pinned Git blob SHA mismatch")
        saved = {**entry, "response": response, "source_url": url}
        raw_files.append(saved)
        if not entry["path"].startswith("tweet/raw/"):
            continue
        try:
            items = stocknet.tweet_records(body)
        except (SourceError, ValueError) as exc:
            issues.append({"source_file": entry["path"], "error": str(exc)})
            continue
        per_file = min(args.max_records - len(records), math.ceil(args.max_records / max(1, args.max_tweet_files)))
        truncated |= len(items) > per_file
        for index, item in enumerate(items[:per_file]):
            if len(records) >= args.max_records:
                break
            try:
                record = stocknet.normalize_tweet(item, entry["entity_id"], url, response["raw_file"], tweet_time_evidence)
                record.update(repository_revision=resolved, original_repository_file=entry["path"], raw_file_sha256=response["sha256"],
                              retrieved_at_now=response["received_at_now"])
                records.append(record)
            except (SourceError, ValueError, TypeError) as exc:
                issues.append({"source_file": entry["path"], "record_index": index, "error": str(exc), "record_rejected": True})
    return {"listing": listing, "raw_files": raw_files, "records": records, "parse_issues": issues,
            "records_truncated": truncated, "actual_data_access_checked": bool(raw_files),
            "contract_status": "raw_schema_checked_only_on_downloaded_samples"}
