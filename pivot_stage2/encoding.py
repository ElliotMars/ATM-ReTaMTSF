"""Frozen MiniLM, with a train-only TF-IDF fallback and a common interface."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import DATA, digest, read_json, sha, write_json


class TextEncoder:
    dim = 384

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = read_json(self.directory / "encoder.json")
        self.kind = self.meta["kind"]
        self.model = self.tokenizer = None
        self.vectorizer = joblib.load(self.directory / "tfidf.joblib") if self.kind == "train_only_tfidf" else None

    def encode(self, texts):
        texts = [text[:2048] for text in texts]
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self.vectorizer is not None:
            sparse = self.vectorizer.transform(texts).astype(np.float32)
            output = np.zeros((len(texts), self.dim), dtype=np.float32)
            output[:, :sparse.shape[1]] = sparse.toarray()
            return output
        import torch
        if self.model is None:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.meta["model_directory"], local_files_only=True)
            self.model = AutoModel.from_pretrained(self.meta["model_directory"], local_files_only=True).eval()
            self.model.requires_grad_(False)
        rows = []
        with torch.inference_mode():
            for i in range(0, len(texts), 64):
                tokens = self.tokenizer(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                rows.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)


def build_encoder(directory, documents, train_end_ns, client=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "encoder.json").exists():
        return TextEncoder(directory)
    fallback_reason = None
    if client is not None:
        try:
            import transformers
            from .fetch import Downloader
            metadata, _ = client.get("https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2", "weights", 2 * 1024**2)
            commit = read_json(metadata)["sha"]
            model_dir = directory / "minilm"
            model_dir.mkdir(exist_ok=True)
            hashes = {}
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.txt", "model.safetensors"):
                path, info = client.get(f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{commit}/{name}", "weights", 110 * 1024**2)
                target = model_dir / name
                if not target.exists():
                    target.write_bytes(path.read_bytes())
                hashes[name] = sha(target)
            write_json(directory / "encoder.json", {"kind": "frozen_all_MiniLM_L6_v2", "dimension": 384,
                "repository": "sentence-transformers/all-MiniLM-L6-v2", "commit": commit, "files": hashes,
                "model_directory": str(model_dir.resolve()), "pooling": "attention_mask_mean_then_L2_normalize", "max_tokens": 128})
            encoder = TextEncoder(directory)
            encoder.encode(["startup verification"])
            return encoder
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
    train_texts = [doc["text"] for doc in documents if doc["available_at_ns"] <= train_end_ns]
    vectorizer = TfidfVectorizer(max_features=384, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)
    vectorizer.fit(train_texts or ["synthetic placeholder vocabulary"])
    joblib.dump(vectorizer, directory / "tfidf.joblib")
    write_json(directory / "encoder.json", {"kind": "train_only_tfidf", "dimension": 384,
        "fit_document_count": len(train_texts), "fit_latest_available_ns": max((doc["available_at_ns"] for doc in documents if doc["available_at_ns"] <= train_end_ns), default=None),
        "train_end_ns": int(train_end_ns), "fallback_reason": fallback_reason or "offline synthetic test",
        "vocabulary_sha256": digest({key: int(value) for key, value in vectorizer.vocabulary_.items()}), "not_equivalent_to_pretrained_semantic_encoder": True})
    return TextEncoder(directory)


def query_text(entity, trend, volatility):
    return f"Company {entity} financial stock price returns {trend}, {volatility} volatility market news"


def cache_embeddings(directory, documents, channels, train_end_ns, client=None):
    start = time.perf_counter()
    encoder = build_encoder(directory, documents, train_end_ns, client)
    embeddings = encoder.encode([doc["text"] for doc in documents])
    np.save(Path(directory) / "embeddings.npy", embeddings)
    queries = [query_text(channel, trend, vol) for channel in channels
               for trend in ("falling", "stable", "rising") for vol in ("low", "high")]
    query_vectors = encoder.encode(queries).reshape(len(channels), 6, 384)
    np.save(Path(directory) / "query_vectors.npy", query_vectors)
    return {"encoder": encoder.meta, "seconds": time.perf_counter() - start,
            "embedding_sha256": sha(Path(directory) / "embeddings.npy"), "documents": len(documents)}
