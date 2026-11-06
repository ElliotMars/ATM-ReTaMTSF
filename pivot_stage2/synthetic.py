"""Explicit synthetic end-to-end fallback. Never part of research admission."""
import argparse
import json
import time

import numpy as np
import pandas as pd

from .common import DATA, ARTIFACTS, SPLITS, digest, new_directory, seal, verify, write_json
from .data import valid_windows
from .encoding import cache_embeddings
from .prepare import fit_scaler
from .train import run


def prepare(channels=3, lookback=20, horizon=5):
    directory = DATA / f"synthetic_C{channels}_L{lookback}_H{horizon}"
    if (directory / "_SUCCESS.json").exists():
        verify(directory)
        return directory
    rng = np.random.default_rng(2024)
    n = lookback + horizon + 8
    raw = rng.normal(0, .02, size=(3, n, channels))
    for t in range(1, n):
        raw[:, t] += 0.4 * raw[:, t - 1]
    masks = rng.random(raw.shape) > .08
    masks[:, :, -1] = False  # exercise a fully missing numerical channel
    scaler = fit_scaler(raw[0], masks[0], np.ones(n, bool), allow_empty=True)
    values = (raw - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    names = [f"synthetic_entity{i}" for i in range(channels)]
    documents = []
    with new_directory(directory) as output:
        train_end = None
        for index, split in enumerate(SPLITS):
            base = pd.Timestamp("2020-01-01", tz="UTC").value + index * (n + 100) * 3600 * 10**9
            times = base + np.arange(n, dtype=np.int64) * 3600 * 10**9
            if index == 0:
                train_end = int(times[-1])
            numeric = np.where(masks[index], values[index], np.nan).astype(np.float32)
            np.savez(output / f"{split}.npz", values=numeric, mask=masks[index], times=times, origins=times, segments=np.zeros(n, int))
            np.save(output / f"{split}_H{horizon}_starts.npy", valid_windows(times, np.zeros(n), masks[index], lookback, horizon, 3600 * 10**9))
            for t in range(4, n, 4):
                for c, name in enumerate(names):
                    # ONLY past simulated observations; no future target cells.
                    observed = raw[index, t - 4:t, c][masks[index, t - 4:t, c]]
                    mean = float(observed.mean()) if len(observed) else 0.
                    text = f"{name} historical values {'rising' if mean >= 0 else 'falling'}; synthetic fixture only"
                    documents.append({"document_id": digest([split, t, c, text]), "source": "synthetic_only_not_real_evidence",
                                      "entity_id": name, "text": text, "issued_at_ns": int(times[t]), "received_at_ns": None,
                                      "available_at_ns": int(times[t] + 3600 * 10**9), "availability_policy": "synthetic_past_only_plus_1h"})
        (output / "documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in documents))
        manifest = {"dataset": "synthetic", "channels": names, "lookback": lookback, "horizons": [horizon],
                    "scaler": scaler, "multimodal_enabled": True, "research_admitted": False}
        write_json(output / "manifest.json", manifest)
        cache_embeddings(output / "text", documents, names, train_end)
        seal(output, {"kind": "synthetic_fixture_not_research_data"})
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    directory = prepare(args.channels, args.lookback, args.horizon)
    config = {"dataset": "synthetic", "data_directory": str(directory), "seed": 2024, "horizon": args.horizon,
              "device": "cpu", "model": {"name": "pivot", "dim": 16, "dropout": 0.},
              "retrieval": {"strategy": "semantic_temporal", "top_k": 2, "lag_hours": 1}}
    output = ARTIFACTS / "synthetic_smoke" / f"C{args.channels}_L{args.lookback}_H{args.horizon}_{time.time_ns()}"
    print(json.dumps(run(config, output, smoke=True), indent=2))


if __name__ == "__main__":
    main()
