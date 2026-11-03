"""Local synthetic fixtures only. Nothing downloads real Fidel-TS files."""

import hashlib
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fidel_ts.common import identity, load_source
from scripts.fidel_ts import common
from scripts.fidel_ts.download import download_dataset


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Stage 1 unit tests must not access the network")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    # Unit fixtures deliberately contain a tiny synthetic repository rather than
    # all 32/147 real paths. Production always checks the Stage 0 reference tree.
    monkeypatch.setattr(common, "pinned_repository_paths", lambda dataset: None)


@pytest.fixture
def spec():
    return load_source("jena")


@pytest.fixture
def inventory_factory():
    def make(spec, payloads, dataset="jena"):
        return {**identity(dataset, spec), "origin": "synthetic_test_fixture",
                "complete_repository_listing": True,
                "files": [{"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                           "git_blob_sha1": None} for name, body in sorted(payloads.items())]}
    return make


@pytest.fixture
def snapshot_factory(tmp_path, inventory_factory):
    def make(spec, payloads, dataset="jena", profile="standard"):
        inventory = inventory_factory(spec, payloads, dataset)
        directory = tmp_path / "raw" / spec["directory"]
        cache = tmp_path / "fixture_cache"
        cache.mkdir(exist_ok=True)

        def fetch_file(unused_spec, entry, unused_cache):
            target = cache / (hashlib.sha256(entry["path"].encode()).hexdigest() + ".bin")
            target.write_bytes(payloads[entry["path"]])
            return target

        result = download_dataset(dataset, spec, inventory, profile, directory, cache,
                                  retries=0, fetch_file=fetch_file)
        assert result["ok"], result
        return directory, inventory, fetch_file
    return make
