"""Synthetic fixtures only. Network remains forbidden even for HTTP tests."""
from pathlib import Path
import socket
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pivot_sources.common import sha256


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests cannot use real network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def evidence_factory(tmp_path):
    def make(role):
        path = tmp_path / (role + ".txt")
        text = "Synthetic fixture: documented field has role " + role
        path.write_text(text, encoding="utf-8")
        return {"reviewed": True, "role": role, "source_url": "https://example.org/fixture",
                "local_file": str(path), "sha256": sha256(path), "quote": text}
    return make
