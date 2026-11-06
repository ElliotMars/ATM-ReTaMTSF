"""Bounded, resumable public downloads, saved separately from old snapshots."""
import argparse
from datetime import timedelta
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import zipfile

import requests

from .common import DATA, ARTIFACTS, digest, read_json, sha, utc_now, write_json

CAPS = {"stocknet": 1024**3, "nws": 500 * 1024**2, "gfs": 250 * 1024**2,
        "weights": 200 * 1024**2, "docs": 20 * 1024**2}


class Downloader:
    def __init__(self, directory=DATA / "sources"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = self.directory / "downloads.json"
        self.state = read_json(self.journal) if self.journal.exists() else {"records": {}, "bytes": {}, "attempts": []}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PIVOT-research-data-preparation/2.0", "Accept-Encoding": "identity"})

    def save(self):
        write_json(self.journal, self.state)

    def get(self, url, group="docs", limit=8 * 1024**2, timeout=30):
        if not url.startswith("https://"):
            raise ValueError("HTTPS sources only")
        key = digest(url)
        path = self.directory / (key + ".raw")
        previous = self.state["records"].get(key)
        if previous and previous.get("complete"):
            if sha(path) != previous["sha256"]:
                raise ValueError("Cached download hash mismatch")
            return path, previous
        part = path.with_suffix(".part")
        for retry in range(3):
            used = self.state["bytes"].get(group, 0)
            # Reserve 400 MiB for dependency wheels within the global 2 GiB cap.
            total_remaining = 2 * 1024**3 - 400 * 1024**2 - sum(self.state["bytes"].values())
            cap = min(limit, CAPS[group] - used, total_remaining)
            if cap <= 0:
                raise ValueError(f"{group} download budget exhausted")
            offset = part.stat().st_size if part.exists() else 0
            headers = {}
            if offset and previous and previous.get("etag"):
                headers = {"Range": f"bytes={offset}-", "If-Range": previous["etag"]}
            attempt = {"url": url, "at": utc_now(), "group": group, "status": "pending"}
            self.state["attempts"].append(attempt)
            self.save()
            try:
                with self.session.get(url, stream=True, timeout=(10, timeout), headers=headers) as response:
                    attempt["http_status"] = response.status_code
                    response.raise_for_status()
                    length = response.headers.get("Content-Length")
                    if length and int(length) > cap:
                        raise ValueError(f"{group} declared size {length} exceeds remaining budget {cap}")
                    resume = response.status_code == 206 and bool(headers)
                    if not resume and part.exists():
                        # Preserve the failed attempt rather than overwrite it.
                        part.rename(part.with_name(part.name + f".abandoned_{len(self.state['attempts'])}"))
                    previous = {"url": url, "group": group, "etag": response.headers.get("ETag"), "complete": False}
                    self.state["records"][key] = previous
                    received = 0
                    self.save()
                    with part.open("ab" if resume else "xb") as stream:
                        for chunk in response.iter_content(64 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            self.state["bytes"][group] = self.state["bytes"].get(group, 0) + len(chunk)
                            self.save()
                            if received > cap:
                                raise ValueError("Streaming body exceeded budget")
                            stream.write(chunk)
                    if length and received != int(length):
                        raise ValueError("Incomplete HTTP response")
                    part.rename(path)
                    previous.update(complete=True, bytes=path.stat().st_size, sha256=sha(path), fetched_at=utc_now(),
                                    final_url=response.url, last_modified=response.headers.get("Last-Modified"))
                    attempt["status"] = "complete"
                    self.save()
                    return path, previous
            except (requests.RequestException, OSError, ValueError) as exc:
                attempt.update(status="failed", error=str(exc))
                self.save()
                if isinstance(exc, ValueError) or retry == 2 or attempt.get("http_status") in {401, 403, 404, 429}:
                    raise
                time.sleep(2 ** retry)
        raise RuntimeError("Download failed")


def acquire_stocknet(client):
    endpoint = "https://api.github.com/repos/yumoxu/stocknet-dataset/commits/HEAD"
    try:
        if any(row.get("url") == endpoint and row.get("http_status") == 403 for row in client.state["attempts"]):
            raise ValueError("REST quota unavailable; use the public repository page ref")
        metadata_path, _ = client.get(endpoint, "stocknet")
        commit = read_json(metadata_path)["sha"]
    except (requests.RequestException, ValueError):
        metadata_path, _ = client.get("https://github.com/yumoxu/stocknet-dataset", "stocknet")
        html = metadata_path.read_text(encoding="utf-8")
        found = re.search(r'"currentOid":"([0-9a-f]{40})"', html)
        if found is None:
            found = re.search(r'/yumoxu/stocknet-dataset/commit/([0-9a-f]{40})', html)
        if found is None:
            raise ValueError("Public page did not expose an immutable commit; refusing an unpinned archive")
        commit = found[1]
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Invalid source commit")
    archive, info = client.get(f"https://codeload.github.com/yumoxu/stocknet-dataset/zip/{commit}", "stocknet", CAPS["stocknet"], timeout=60)
    extracted = client.directory / f"stocknet_{commit}"
    manifest = extracted / "manifest.json"
    if manifest.exists():
        record = read_json(manifest)
        if sha(archive) != record["archive_sha256"]:
            raise ValueError("Original ZIP hash changed")
        # Individual members are checked against this manifest during parsing.
        # Previously extracted copies are not input to Stage 2 preparation.
        return record
    extracted.mkdir(exist_ok=True)
    files = {}
    uncompressed = 0
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as handle:
        for member in handle.infolist():
            parts = PurePosixPath(member.filename).parts
            if member.is_dir() or len(parts) < 2 or ".." in parts:
                continue
            relative = PurePosixPath(*parts[1:])
            name = relative.as_posix()
            if not (name.startswith(("price/raw/", "tweet/raw/")) or name in {"README.md", "LICENSE", "StockTable"}):
                continue
            uncompressed += member.file_size
            if uncompressed > 2 * 1024**3:
                raise ValueError("Uncompressed selected source exceeds safety limit")
            body = handle.read(member)
            if not name.startswith(("price/raw/", "tweet/raw/")):
                target = extracted / name
                if target.exists():
                    if target.read_bytes() != body:
                        raise ValueError("Source extraction conflict")
                else:
                    target.write_bytes(body)
            import hashlib
            files[name] = hashlib.sha256(body).hexdigest()
    result = {"repository": "https://github.com/yumoxu/stocknet-dataset", "commit": commit,
              "archive": str(archive), "archive_sha256": info["sha256"], "root": str(extracted), "files": files,
              "storage": "immutable_zip", "no_per_record_filesystem_extraction_required": True}
    write_json(manifest, result)
    return result


def acquire_weather(client):
    results = {}
    urls = {"nws": "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=",
            "gfs": "https://gdex.ucar.edu/datasets/d084001/"}
    for source, url in urls.items():
        try:
            path, record = client.get(url, "docs")
            results[source] = {"documentation": str(path), "sha256": record["sha256"], "source_url": url,
                               "semantic_review": "separate_from_HTTP_success"}
        except Exception as exc:
            results[source] = {"error": str(exc)}
    # Only a small representative AFD batch is useful until CAISO clock
    # evidence is established. Do not download years of unusable text.
    if "error" not in results["nws"]:
        for pil in ("AFDMTR", "AFDSGX"):
            url = ("https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?"
                   f"pil={pil}&sdate=2021-07-15T00:00Z&edate=2021-07-16T00:00Z&fmt=zip&limit=100&order=asc")
            try:
                time.sleep(1)
                path, record = client.get(url, "nws", 8 * 1024**2)
                results["nws"].setdefault("actual_batches", []).append({"pil": pil, "path": str(path), **record})
            except Exception as exc:
                results["nws"].setdefault("batch_errors", []).append({"pil": pil, "error": str(exc)})
    write_json(client.directory / "weather_acquisition.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("all", "stocknet", "weather"), default="all")
    args = parser.parse_args()
    client = Downloader()
    results = {}
    if args.source in {"all", "stocknet"}:
        try:
            results["stocknet"] = acquire_stocknet(client)
        except Exception as exc:
            results["stocknet"] = {"error": str(exc)}
    if args.source in {"all", "weather"}:
        results["weather"] = acquire_weather(client)
    write_json(ARTIFACTS / "acquisition.json", results)
    print(json.dumps({key: ({"commit": value["commit"], "files": len(value["files"])} if "commit" in value else value)
                      for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
