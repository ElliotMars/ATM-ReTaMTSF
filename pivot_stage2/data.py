from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .common import read_json, verify, digest
from .retrieval import Retriever


def valid_windows(times, segment, mask, lookback, horizon, expected_step=None):
    """Check ALL L history dates AND H target dates, including their join."""
    n = len(times)
    count = max(0, n - lookback - horizon + 1)
    starts = np.arange(count, dtype=np.int64)
    if not count:
        return starts
    edge_bad = np.diff(segment) != 0
    if expected_step is not None:
        edge_bad |= np.diff(times) != expected_step
    prefix = np.r_[0, np.cumsum(edge_bad)]
    continuous = prefix[starts + lookback + horizon - 1] == prefix[starts]
    # Missing cells/channels are allowed, but empty history/target windows are not.
    cells = mask.sum(1)
    p = np.r_[0, np.cumsum(cells)]
    input_valid = p[starts + lookback] > p[starts]
    target_valid = p[starts + lookback + horizon] > p[starts + lookback]
    return starts[continuous & input_valid & target_valid]


def causal_fill(values, mask):
    output = np.asarray(values, dtype=np.float32).copy()
    carried = np.zeros(values.shape[1], dtype=np.float32)  # normalized train mean
    for index in range(len(output)):
        carried[mask[index]] = output[index, mask[index]]
        output[index] = carried
    return output


class MIMODataset(Dataset):
    def __init__(self, directory, split, horizon, strategy="none", top_k=4, temporal_weight=0.2, lag_hours=1, seed=2024, check_hashes=True):
        self.directory = Path(directory)
        self.bundle = verify(directory) if check_hashes else read_json(self.directory / "_SUCCESS.json")
        self.meta = read_json(self.directory / "manifest.json")
        if split not in {"train", "validation", "test"} or horizon not in self.meta["horizons"]:
            raise ValueError("Unknown split/horizon")
        self.split, self.horizon, self.lookback = split, horizon, self.meta["lookback"]
        self.channels = self.meta["channels"]
        arrays = np.load(self.directory / f"{split}.npz", allow_pickle=False)
        self.values, self.mask = arrays["values"], arrays["mask"]
        self.times, self.origins = arrays["times"], arrays["origins"]
        self.starts = np.load(self.directory / f"{split}_H{horizon}_starts.npy")
        self.sample_binding = digest([self.bundle["artifacts"], split, horizon])
        self.retriever = None
        if strategy not in {"none", "missing"} and self.meta.get("multimodal_enabled"):
            documents = [json.loads(line) for line in (self.directory / "documents.jsonl").read_text().splitlines()]
            embeddings = np.load(self.directory / "text/embeddings.npy", mmap_mode="r")
            queries = np.load(self.directory / "text/query_vectors.npy")
            self.retriever = Retriever(documents, embeddings, queries, self.channels, strategy, top_k, temporal_weight, lag_hours, seed)
        self.top_k = top_k

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        start = int(self.starts[item])
        stop = start + self.lookback
        raw_x, xmask = self.values[start:stop], self.mask[start:stop]
        ymask = self.mask[stop:stop + self.horizon]
        y = np.where(ymask, self.values[stop:stop + self.horizon], 0).astype(np.float32)
        # Stocks forecast AFTER the last history close; weather origin is first
        # predicted native timestamp, retained as naive if clock is unresolved.
        origin_position = stop - 1 if self.meta["dataset"] == "stocknet" else stop
        origin_ns = int(self.origins[origin_position])
        numeric_time = int(self.times[origin_position])
        if self.retriever:
            # Query summaries use raw historical return units (not labels).
            query_history = raw_x * np.asarray(self.meta["scaler"]["scale"]) + np.asarray(self.meta["scaler"]["mean"])
            vectors, tmask, trace, counts = self.retriever.tensors_and_trace(query_history, xmask, origin_ns)
        else:
            vectors = np.zeros((len(self.channels), self.top_k, 384), np.float32)
            tmask, trace, counts = np.zeros(vectors.shape[:-1], bool), [[] for _ in self.channels], np.zeros(len(self.channels), int)
        return {"x_num": torch.from_numpy(causal_fill(raw_x, xmask)), "x_mask": torch.from_numpy(xmask.copy()),
                "y": torch.from_numpy(y), "y_mask": torch.from_numpy(ymask.copy()),
                "origin_time": origin_ns if origin_ns >= 0 else pd.Timestamp(numeric_time).isoformat(),
                "origin_utc_ns": origin_ns, "origin_timezone": "UTC" if origin_ns >= 0 else None, "numeric_time": numeric_time,
                "channel_ids": torch.arange(len(self.channels)), "text_embeddings": torch.from_numpy(vectors),
                "text_mask": torch.from_numpy(tmask), "text_metadata": trace,
                "eligible_document_counts": torch.from_numpy(counts.copy()),
                "sample_id": digest([self.sample_binding, start])}


def collate(samples):
    output = {}
    for key in samples[0]:
        if isinstance(samples[0][key], torch.Tensor):
            output[key] = torch.stack([sample[key] for sample in samples])
        else:
            output[key] = [sample[key] for sample in samples]
    return output


class HorizonSampler(Sampler):
    def __init__(self, dataset, seed=2024, shuffle=False):
        self.dataset, self.seed, self.shuffle = dataset, seed, shuffle

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        ids = np.arange(len(self))
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(ids)
        return iter(ids.tolist())


def loader(dataset, batch_size=8, shuffle=False, seed=2024):
    return DataLoader(dataset, batch_size=batch_size, sampler=HorizonSampler(dataset, seed, shuffle),
                      num_workers=0, collate_fn=collate, drop_last=False)
