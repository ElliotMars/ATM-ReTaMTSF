"""Filter as-of and entity BEFORE cosine scoring. No labels enter this module."""
from functools import lru_cache
import hashlib

import numpy as np

from .common import digest, read_json


class Retriever:
    def __init__(self, documents, embeddings, queries, channels, strategy="semantic_temporal", top_k=4,
                 temporal_weight=0.2, lag_hours=1, seed=2024, window_days=30):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if lag_hours < 0 or window_days <= 0:
            raise ValueError("Availability lag must be nonnegative and candidate window positive")
        if strategy not in {"none", "missing", "latest", "random", "shuffled", "mean", "semantic", "semantic_temporal"}:
            raise ValueError("Unknown retrieval strategy")
        for doc in documents:
            received = doc.get("received_at_ns")
            if received is not None and (not doc.get("strict_verified_asof") or received < doc["issued_at_ns"]):
                raise ValueError("Historical reception requires explicit verified provenance and cannot precede issuance")
        self.documents, self.embeddings, self.queries = documents, embeddings, queries
        self.channels = channels
        self.strategy, self.top_k, self.temporal_weight = strategy, top_k, temporal_weight
        self.lag_hours = lag_hours
        self.lag_ns, self.seed, self.window_ns = int(lag_hours * 3600 * 1e9), seed, int(window_days * 86400 * 1e9)
        self.created = np.array([doc["issued_at_ns"] for doc in documents], dtype=np.int64)
        self.available = np.array([doc.get("received_at_ns") or doc["issued_at_ns"] + self.lag_ns for doc in documents], dtype=np.int64)
        self.groups = {channel: np.array([i for i, doc in enumerate(documents) if doc.get("entity_id") in (channel, None)], dtype=np.int64)
                       for channel in channels}
        self.binding = digest({"document_ids": [doc["document_id"] for doc in documents], "strategy": strategy,
            "top_k": top_k, "temporal_weight": temporal_weight, "lag_hours": lag_hours, "seed": seed})
        self.cache = {}
        self.mean_cache = {}

    def eligible_ids(self, origin_ns, channel):
        if origin_ns < 0:
            return np.array([], dtype=np.int64)
        ids = self.groups[channel]
        # No semantic score, Top-K or random sampling is computed before this.
        return ids[(self.available[ids] <= origin_ns) & (self.available[ids] > origin_ns - self.window_ns)]

    def retrieve(self, history, mask, origin_ns):
        history = np.where(mask, history, 0)
        count = np.maximum(mask.sum(0), 1)
        means = history.sum(0) / count
        variance = (np.where(mask, history - means, 0) ** 2).sum(0) / count
        trend = np.where(means > 1e-4, 2, np.where(means < -1e-4, 0, 1))
        bins = trend * 2 + (np.sqrt(variance) >= 0.02)
        key = (int(origin_ns), tuple(bins.tolist()))
        if key in self.cache:
            return self.cache[key]
        ids_out = np.full((len(self.channels), self.top_k), -1, dtype=np.int64)
        scores_out = np.zeros(ids_out.shape, dtype=np.float32)
        eligible_counts = []
        for c, channel in enumerate(self.channels):
            legal = self.eligible_ids(origin_ns, channel)
            eligible_counts.append(len(legal))
            if not len(legal) or self.strategy in {"none", "missing"}:
                continue
            ages = (origin_ns - self.available[legal]) / (86400 * 1e9)
            if self.strategy == "latest":
                scores = -ages
            elif self.strategy == "random":
                local_seed = int(hashlib.sha256(f"{self.seed}/{origin_ns}/{channel}".encode()).hexdigest()[:16], 16)
                scores = np.random.default_rng(local_seed).random(len(legal))
            elif self.strategy == "mean":
                # Deterministic bounded pool; no semantic ranking.
                scores = -ages
            else:
                scores = self.embeddings[legal] @ self.queries[c, bins[c]]
                if self.strategy in {"semantic_temporal", "shuffled"}:
                    scores = scores + self.temporal_weight * np.exp(-ages / 3)
            # Ties are deterministic by document index.
            order = np.lexsort((legal, -scores))[:self.top_k]
            chosen = legal[order]
            ids_out[c, :len(chosen)] = chosen
            scores_out[c, :len(chosen)] = scores[order]
        if self.strategy == "shuffled":
            # Exchange channel associations only at the SAME prediction origin.
            # Every exchanged document remains legal for that origin.
            rng = np.random.default_rng((self.seed + int(origin_ns // 10**9)) % (2**32))
            permutation = rng.permutation(len(self.channels))
            ids_out, scores_out = ids_out[permutation], scores_out[permutation]
        result = ids_out, scores_out, np.asarray(eligible_counts)
        self.cache[key] = result
        return result

    def tensors_and_trace(self, history, mask, origin_ns):
        if self.strategy == "mean":
            if origin_ns in self.mean_cache:
                return self.mean_cache[origin_ns]
            vectors = np.zeros((len(self.channels), self.top_k, self.embeddings.shape[1]), np.float32)
            valid = np.zeros(vectors.shape[:-1], bool)
            trace, counts = [], []
            for c, channel in enumerate(self.channels):
                legal = self.eligible_ids(origin_ns, channel)
                counts.append(len(legal))
                if len(legal):
                    vectors[c, 0] = self.embeddings[legal].mean(0)
                    valid[c, 0] = True
                    members = [self.documents[i]["document_id"] for i in legal]
                    trace.append([{"document_id": "pooled_" + digest(members), "member_document_ids": members,
                                   "source": "eligible_mean_pool_derived_representation", "available_at_ns": int(self.available[legal].max()),
                                   "issued_at_ns": int(self.created[legal].max()), "score": 1.0 / len(legal),
                                   "origin_ns": int(origin_ns), "asof_legal": True, "policy": f"creation_time_plus_{self.lag_hours}h_assumption"}])
                else:
                    trace.append([])
            result = vectors, valid, trace, np.asarray(counts)
            self.mean_cache[origin_ns] = result
            return result
        ids, scores, counts = self.retrieve(history, mask, int(origin_ns))
        valid = ids >= 0
        vectors = np.zeros((*ids.shape, self.embeddings.shape[1]), dtype=np.float32)
        vectors[valid] = self.embeddings[ids[valid]]
        trace = []
        for c in range(len(self.channels)):
            row = []
            for k in np.flatnonzero(valid[c]):
                i = ids[c, k]
                assert self.available[i] <= origin_ns
                row.append({"document_id": self.documents[i]["document_id"], "source": self.documents[i]["source"],
                            "available_at_ns": int(self.available[i]), "issued_at_ns": int(self.created[i]),
                            "origin_ns": int(origin_ns), "score": float(scores[c, k]), "asof_legal": True,
                            "entity_id": self.documents[i].get("entity_id"),
                            "policy": "historical_received_time" if self.documents[i].get("received_at_ns") is not None else f"creation_time_plus_{self.lag_hours}h_assumption",
                            "document_default_policy": self.documents[i]["availability_policy"],
                            "query_source": "only_raw_historical_returns; no_y",
                            "query_history_mean": float(np.mean(history[:, c][mask[:, c]])) if mask[:, c].any() else None,
                            "query_history_std": float(np.std(history[:, c][mask[:, c]])) if mask[:, c].any() else None})
            trace.append(row)
        return vectors, valid, trace, counts
