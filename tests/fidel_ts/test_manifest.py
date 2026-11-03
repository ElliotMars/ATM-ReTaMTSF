import copy

import pytest

from scripts.fidel_ts.common import AuditError, data_path, normalize_inventory, read_json, write_json
from scripts.fidel_ts.download import verify_manifest


def test_manifest_hash_size_revision_and_completeness(spec, snapshot_factory):
    directory, _, _ = snapshot_factory(spec, {spec["primary_file"]: b"abc"}, profile="numerical")
    path = directory / "manifest.json"
    original = read_json(path)
    assert verify_manifest("jena", spec, directory, "numerical")["ok"]
    for mutation in ("sha256", "size", "revision", "missing_entry", "inventory"):
        value = copy.deepcopy(original)
        if mutation == "revision":
            value["revision"] = "0" * 40
        elif mutation == "missing_entry":
            value["files"].pop(spec["primary_file"])
        elif mutation == "inventory":
            value["inventory"]["files"][0]["size"] += 1
        else:
            value["files"][spec["primary_file"]][mutation] = "0" * 64 if mutation == "sha256" else 999
        write_json(path, value)
        assert not verify_manifest("jena", spec, directory, "numerical")["ok"], mutation
    write_json(path, original)
    (directory / "unexpected.json").write_text("{}")
    assert any("Unexpected/unrecorded" in e for e in verify_manifest("jena", spec, directory)["errors"])


def test_duplicate_inventory_entries_rejected(spec, inventory_factory):
    inventory = inventory_factory(spec, {spec["primary_file"]: b"abc"})
    inventory["files"].append(copy.deepcopy(inventory["files"][0]))
    with pytest.raises(AuditError, match="Duplicate"):
        normalize_inventory(inventory, "jena", spec)


def test_repository_listing_must_be_complete(spec, inventory_factory, monkeypatch):
    from scripts.fidel_ts import common
    inventory = inventory_factory(spec, {spec["primary_file"]: b"abc"})
    monkeypatch.setattr(common, "pinned_repository_paths", lambda dataset: {spec["primary_file"], "missing.json"})
    with pytest.raises(AuditError, match="Incomplete/changed"):
        normalize_inventory(inventory, "jena", spec)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../../escape", "a\\b", "a//b", "C:/drive"])
def test_path_traversal_rejected(tmp_path, path):
    with pytest.raises(AuditError):
        data_path(tmp_path, path)


def test_symlink_data_path_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(AuditError, match="symlink"):
        data_path(tmp_path, "link/data.json")
