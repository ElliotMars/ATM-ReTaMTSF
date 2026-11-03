"""Small, standard-library helpers shared by the Stage 1 CLIs."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/fidel_ts/sources.json"
PINS = {
    "caiso": ("fidel-ts/California_ISO", "75f74f556edd5fbdd080c4db6d4b7547fc331f39"),
    "jena": ("fidel-ts/Jena_Atmospheric_Physics", "c8da24175c7aabe5c3b02b503e4c5afe44a605fd"),
}
SCHEMA_VERSION = 1
IDENTITY = ".fidel_ts_identity.json"
MANIFEST = "manifest.json"


class AuditError(ValueError):
    """An actionable provenance, integrity or input error."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise AuditError(f"Refusing JSON symlink: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    """Atomic replacement of a tool-owned metadata file, never a data file."""
    path = Path(path)
    if path.is_symlink():
        raise AuditError(f"Refusing metadata symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AuditError(f"Invalid relative path: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or PurePosixPath(value).is_absolute():
        raise AuditError(f"Unsafe relative path: {value!r}")
    if ":" in parts[0]:
        raise AuditError(f"Invalid drive-like path: {value!r}")
    return value


def data_path(directory, relative):
    """Reject path traversal and symlinks instead of following data elsewhere."""
    relative_path(relative)
    directory = Path(directory)
    current = directory
    for part in ("", *PurePosixPath(relative).parts):
        if part:
            current = current / part
        if current.is_symlink():
            raise AuditError(f"Refusing symlink in data path: {current}")
    base = directory.resolve()
    if not current.resolve().is_relative_to(base):
        raise AuditError(f"Path escapes dataset directory: {relative}")
    return current


def load_source(dataset, config=DEFAULT_CONFIG, revision=None):
    document = read_json(config)
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION or dataset not in PINS:
        raise AuditError("Unsupported source configuration or dataset")
    spec = document["datasets"][dataset]
    if (spec.get("repository"), spec.get("revision")) != PINS[dataset]:
        raise AuditError(f"{dataset}: configuration must retain the Stage 0 repository and revision")
    if revision is not None and revision != spec["revision"]:
        raise AuditError("Requested revision differs from the fixed source revision")
    if spec.get("directory") != spec["repository"].split("/")[-1]:
        raise AuditError("Dataset directory must match its repository name")
    relative_path(spec["primary_file"])
    relative_path(spec["static_info_file"])
    return spec


def dataset_directory(output_root, spec):
    return Path(output_root).expanduser().resolve() / spec["directory"]


def identity(dataset, spec):
    return {"schema_version": SCHEMA_VERSION, "dataset": dataset,
            "repository": spec["repository"], "revision": spec["revision"]}


def assert_identity(value, dataset, spec):
    expected = identity(dataset, spec)
    if not isinstance(value, dict) or any(value.get(key) != val for key, val in expected.items()):
        raise AuditError("Dataset/repository/revision identity mismatch; use a separate empty directory")


def pinned_repository_paths(dataset):
    """Anchor inventory completeness in the immutable Stage 0 listing."""
    evidence = read_json(ROOT / "docs/fidel_ts/00_audit_evidence.json")
    source = evidence["datasets"][dataset]
    if (source["id"], source["revision"]) != PINS[dataset]:
        raise AuditError("Stage 0 reference listing identity mismatch")
    return set(source["files"])


def normalize_inventory(value, dataset, spec):
    assert_identity(value, dataset, spec)
    entries = value.get("files")
    if not isinstance(entries, list) or not entries:
        raise AuditError("Inventory must contain a nonempty files list")
    files, seen = [], set()
    for item in entries:
        if not isinstance(item, dict):
            raise AuditError("Inventory entries must be objects")
        path = relative_path(item.get("path"))
        if path in seen or path in (MANIFEST, IDENTITY) or path.startswith(".fidel_ts"):
            raise AuditError(f"Duplicate or reserved inventory path: {path}")
        seen.add(path)
        size = item.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise AuditError(f"Invalid file size for {path}")
        entry = {"path": path, "size": size,
                 "sha256": item.get("sha256"), "git_blob_sha1": item.get("git_blob_sha1")}
        for field, length in (("sha256", 64), ("git_blob_sha1", 40)):
            if entry[field] is not None and not re.fullmatch(f"[0-9a-f]{{{length}}}", entry[field]):
                raise AuditError(f"Invalid {field} for {path}")
        files.append(entry)
    reference = pinned_repository_paths(dataset)
    if reference is not None and seen != reference:
        raise AuditError(f"Incomplete/changed pinned repository listing; missing={sorted(reference - seen)}, "
                         f"unexpected={sorted(seen - reference)}")
    result = dict(value)
    result["files"] = sorted(files, key=lambda entry: entry["path"])
    return result


def selected_files(inventory, spec, profile):
    if profile not in spec["profiles"]:
        raise AuditError(f"Unknown profile: {profile}")
    patterns = spec["profiles"][profile]
    files = [entry for entry in inventory["files"]
             if any(fnmatch.fnmatchcase(entry["path"], pat) for pat in patterns)
             and not entry["path"].endswith((".pkl", ".pickle"))]
    if spec["primary_file"] not in {item["path"] for item in files}:
        raise AuditError("Inventory/profile does not contain the preferred numerical file")
    return files


def stage0_inventory(path, dataset, spec):
    """The old evidence lists names, but many sizes are genuinely unknown."""
    evidence = read_json(path)
    source = evidence["datasets"][dataset]
    if (source["id"], source["revision"]) != PINS[dataset]:
        raise AuditError("Stage 0 evidence revision mismatch")
    known = {item["path"]: {"size": item["bytes"], "sha256": item["sha256"]}
             for item in evidence.get("source_files", []) if item["dataset"] == dataset}
    for item in evidence.get("parquet_metadata", []):
        if item["dataset"] == dataset:
            known.setdefault(item["path"], {})["size"] = item["size_bytes"]
    value = {**identity(dataset, spec), "origin": "stage0_local_evidence",
             "complete_repository_listing": True,
             "evidence_sha256": sha256_file(path), "files": [
                 {"path": name, **known.get(name, {})} for name in source["files"]]}
    return normalize_inventory(value, dataset, spec)
