import json

import pytest

from scripts.fidel_ts.common import AuditError, load_source, read_json, write_json
from scripts.fidel_ts.download import download_dataset, main, verify_manifest


def test_revision_is_pinned(spec):
    assert spec["revision"] == "c8da24175c7aabe5c3b02b503e4c5afe44a605fd"
    with pytest.raises(AuditError, match="fixed source revision"):
        load_source("jena", revision="main")


def test_profiles_do_not_select_pickle_or_merged_weather():
    from scripts.fidel_ts.common import selected_files
    for dataset in ("caiso", "jena"):
        source = load_source(dataset)
        inventory = {"files": [{"path": source["primary_file"]},
                               {"path": "weather/merged_general_report/merged_general_weather_forecast.json"},
                               {"path": "weather/report_embedding/formal_report/wm_messages_v1.pkl"}]}
        for profile in source["profiles"]:
            assert selected_files(inventory, source, profile) == [{"path": source["primary_file"]}]


def test_existing_verified_file_is_checked_not_downloaded(spec, snapshot_factory, tmp_path):
    payloads = {spec["primary_file"]: b"synthetic numerical bytes"}
    directory, inventory, _ = snapshot_factory(spec, payloads, profile="numerical")

    def forbidden(*args):
        raise AssertionError("Existing valid file should not download")

    result = download_dataset("jena", spec, inventory, "numerical", directory, tmp_path / "cache",
                              retries=0, fetch_file=forbidden)
    assert result["ok"]
    assert read_json(directory / "manifest.json")["files"][spec["primary_file"]]["last_action"] == "checked_existing"


def test_existing_corruption_is_not_overwritten(spec, snapshot_factory, tmp_path):
    directory, inventory, fetch = snapshot_factory(spec, {spec["primary_file"]: b"abc"}, profile="numerical")
    target = directory / spec["primary_file"]
    target.write_bytes(b"xyz")
    with pytest.raises(AuditError, match="SHA-256"):
        download_dataset("jena", spec, inventory, "numerical", directory, tmp_path / "cache",
                         retries=0, fetch_file=fetch)
    assert target.read_bytes() == b"xyz"


def test_different_revision_cannot_mix(spec, snapshot_factory, tmp_path):
    directory, inventory, fetch = snapshot_factory(spec, {spec["primary_file"]: b"abc"}, profile="numerical")
    marker = read_json(directory / ".fidel_ts_identity.json")
    marker["revision"] = "0" * 40
    write_json(directory / ".fidel_ts_identity.json", marker)
    with pytest.raises(AuditError, match="identity mismatch"):
        download_dataset("jena", spec, inventory, "numerical", directory, tmp_path / "cache",
                         retries=0, fetch_file=fetch)


def test_partial_transfer_is_not_published_and_can_retry(spec, inventory_factory, tmp_path):
    body = b"complete synthetic payload"
    inventory = inventory_factory(spec, {spec["primary_file"]: body})
    directory = tmp_path / "raw" / spec["directory"]
    cache = tmp_path / "cache_file"

    def partial(*args):
        cache.write_bytes(body[:4])
        return cache

    with pytest.raises(AuditError, match="Download failed"):
        download_dataset("jena", spec, inventory, "numerical", directory, tmp_path,
                         retries=0, fetch_file=partial)
    assert not (directory / spec["primary_file"]).exists()
    record = read_json(directory / "manifest.json")["files"][spec["primary_file"]]
    assert record["status"] == "failed"
    assert record["attempts"][0]["status"] == "failed"
    assert not verify_manifest("jena", spec, directory, "numerical")["ok"]

    def complete(*args):
        cache.write_bytes(body)
        return cache

    result = download_dataset("jena", spec, inventory, "numerical", directory, tmp_path,
                              retries=0, fetch_file=complete)
    assert result["ok"]
    assert len(read_json(directory / "manifest.json")["files"][spec["primary_file"]]["attempts"]) == 2


def test_dry_run_is_offline_with_inventory_and_writes_nothing(spec, inventory_factory, tmp_path, capsys):
    inventory = inventory_factory(spec, {spec["primary_file"]: b"abc"})
    source = tmp_path / "inventory.json"
    write_json(source, inventory)
    destination = tmp_path / "nonexistent"
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert main(["dry-run", "--dataset", "jena", "--profile", "numerical",
                 "--inventory", str(source), "--output-root", str(destination)]) == 0
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert before == after
    assert json.loads(capsys.readouterr().out)["estimated_total_bytes"] == 3


def test_stage0_listing_keeps_unknown_sizes_and_never_downloads(tmp_path, capsys):
    assert main(["list", "--dataset", "caiso", "--stage0", "--output-root", str(tmp_path / "absent")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["unknown_size_files"] > 0
    assert result["estimated_total_bytes"] is None
    assert not (tmp_path / "absent").exists()


def test_unknown_size_prevents_download(spec, inventory_factory, tmp_path):
    inventory = inventory_factory(spec, {spec["primary_file"]: b"abc"})
    inventory["files"][0]["size"] = None
    with pytest.raises(AuditError, match="sizes are unknown"):
        download_dataset("jena", spec, inventory, "numerical", tmp_path / "raw", tmp_path)
    assert not (tmp_path / "raw").exists()


def test_nonempty_unmanaged_directory_is_rejected(spec, inventory_factory, tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "other_dataset.txt").write_text("preserve me")
    inventory = inventory_factory(spec, {spec["primary_file"]: b"abc"})
    with pytest.raises(AuditError, match="unmanaged"):
        download_dataset("jena", spec, inventory, "numerical", directory, tmp_path)
    assert (directory / "other_dataset.txt").read_text() == "preserve me"
