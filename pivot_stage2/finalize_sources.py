"""Bounded final evidence capture and optional immutable MiniLM upgrade."""
import json
from pathlib import Path
import shutil
from pypdf import PdfReader

import numpy as np

from .common import DATA, ARTIFACTS, new_directory, read_json, seal, sha, verify, write_json
from .encoding import build_encoder, cache_embeddings
from .fetch import Downloader


def capture_evidence(client):
    destination = ARTIFACTS / "sources/final_official_evidence.json"
    evidence = read_json(destination) if destination.exists() else {}
    for name, url in {
        "gfs_access": "https://gdex.ucar.edu/datasets/d084001/dataaccess/",
        "nyse_2014_2015": "https://ir.theice.com/press/news-details/2013/NYSE-Euronext-Announces-2015-Holiday-Calendar-and-Early-Closings/default.aspx",
        "nyse_2014_2015_pdf": "https://s2.q4cdn.com/154085107/files/doc_news/archive/b97d77df-0ecb-4db0-9ced-75bdb9b218ba.pdf",
    }.items():
        prior = evidence.get(name, {})
        if prior.get("path"):
            if sha(prior["path"]) != prior["sha256"]:
                raise ValueError(f"Official evidence hash mismatch: {name}")
            continue
        # A denied HTML page is not retried indefinitely. The public PDF
        # contains the original historical notice and is fetched separately.
        if prior.get("error"):
            continue
        try:
            path, info = client.get(url, "docs", 3 * 1024**2)
            evidence[name] = {"path": str(path), "source_url": url, "sha256": info["sha256"]}
            if name == "gfs_access":
                html = path.read_text()
                evidence[name]["subset_login_required_observed"] = "login required" in html.lower()
        except Exception as exc:
            evidence[name] = {"error": str(exc), "source_url": url}
    pdf = evidence.get("nyse_2014_2015_pdf", {})
    if pdf.get("path"):
        text_path = ARTIFACTS / "sources/nyse_2014_2015_official.txt"
        if pdf.get("extracted_text"):
            if sha(pdf["extracted_text"]) != pdf["extracted_text_sha256"]:
                raise ValueError("Historical calendar extracted text hash mismatch")
        else:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text("\n".join(page.extract_text() or "" for page in PdfReader(pdf["path"]).pages), encoding="utf-8")
            pdf.update(extracted_text=str(text_path), extracted_text_sha256=sha(text_path))
    write_json(destination, evidence)
    return evidence


def main():
    client = Downloader()
    capture_evidence(client)
    directory = DATA / "stocknet"
    if not (directory / "_SUCCESS.json").exists():
        return
    verify(directory)
    meta = read_json(directory / "text/encoder.json")
    if meta["kind"] != "train_only_tfidf":
        return
    documents = [json.loads(line) for line in (directory / "documents.jsonl").read_text().splitlines()]
    manifest = read_json(directory / "manifest.json")
    candidate = DATA / "minilm_candidate_v1"
    encoder = build_encoder(candidate, documents, 1435708799999999999, client)
    write_json(ARTIFACTS / "encoder_attempt.json", encoder.meta)
    if encoder.kind == "train_only_tfidf":
        print("Preferred encoder unavailable; verified train-only TF-IDF retained", flush=True)
        return
    old = DATA / "stocknet_tfidf_v1"
    if old.exists():
        raise ValueError("Previous TF-IDF version already exists; refusing overwrite")
    directory.rename(old)
    try:
        with new_directory(directory) as output:
            for path in old.iterdir():
                if path.is_file() and path.name not in {"_SUCCESS.json", "quality.json"}:
                    shutil.copy2(path, output / path.name)
            shutil.copytree(candidate, output / "text")
            encoder_meta = read_json(output / "text/encoder.json")
            encoder_meta["model_directory"] = str((output / "text/minilm").resolve())
            write_json(output / "text/encoder.json", encoder_meta)
            report = cache_embeddings(output / "text", documents, manifest["channels"], 1435708799999999999)
            quality = read_json(old / "quality.json")
            quality["embedding"] = report
            quality["previous_tfidf_version_preserved"] = str(old)
            write_json(output / "quality.json", quality)
            seal(output, {"dataset": "stocknet", "kind": "conditional_creation_time_multimodal", "encoder": encoder_meta["kind"]})
        print("Frozen MiniLM upgrade complete", flush=True)
    except BaseException:
        if directory.exists():
            failed = DATA / "stocknet_minilm_failed_v1"
            if not failed.exists():
                directory.rename(failed)
        if not directory.exists():
            old.rename(directory)
        raise


if __name__ == "__main__":
    main()
