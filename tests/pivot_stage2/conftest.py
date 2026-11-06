from pathlib import Path
import json
import socket
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pivot_stage2.common import seal, write_json
from pivot_stage2.data import valid_windows


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit tests cannot access network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    torch.set_num_threads(2)


@pytest.fixture
def bundle_factory(tmp_path):
    def make(channels=3, lookback=8, horizons=(1, 3, 5), documents=True):
        directory = tmp_path / f"synthetic_{channels}_{lookback}_{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        names = [f"entity{i}" for i in range(channels)]
        rng = np.random.default_rng(12)
        n = lookback + max(horizons) + 7
        for j, split in enumerate(("train", "validation", "test")):
            values = rng.normal(size=(n, channels)).astype(np.float32)
            mask = np.ones_like(values, dtype=bool)
            mask[1, -1] = False
            values[~mask] = np.nan
            times = np.arange(n, dtype=np.int64) * 3600 * 10**9 + (j + 1) * 1000 * 3600 * 10**9
            segments = np.zeros(n, dtype=int)
            np.savez(directory / f"{split}.npz", values=values, mask=mask, times=times, origins=times, segments=segments)
            for horizon in horizons:
                np.save(directory / f"{split}_H{horizon}_starts.npy", valid_windows(times, segments, mask, lookback, horizon, 3600 * 10**9))
        manifest = {"dataset": "synthetic", "channels": names, "lookback": lookback, "horizons": list(horizons),
                    "scaler": {"mean": [0.] * channels, "scale": [1.] * channels}, "multimodal_enabled": documents}
        write_json(directory / "manifest.json", manifest)
        if documents:
            docs = [{"document_id": f"d{i}", "source": "synthetic_fixture_only", "entity_id": name,
                     "issued_at_ns": 900 * 3600 * 10**9, "available_at_ns": 901 * 3600 * 10**9, "received_at_ns": None,
                     "availability_policy": "synthetic_source_time_assumption", "text": f"{name} market rising"} for i, name in enumerate(names)]
            (directory / "documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in docs))
            text_dir = directory / "text"
            text_dir.mkdir()
            np.save(text_dir / "embeddings.npy", rng.normal(size=(channels, 384)).astype(np.float32))
            np.save(text_dir / "query_vectors.npy", rng.normal(size=(channels, 6, 384)).astype(np.float32))
        seal(directory, {"kind": "synthetic_fixture_only"})
        return directory
    return make
