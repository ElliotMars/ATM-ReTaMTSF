#!/usr/bin/env python3
"""Fixed-revision Fidel-TS inventory/download/verify CLI. Run manually only."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# Keep both `python scripts/.../download.py` and package imports usable.
if __package__ in (None, ""):
    script_directory = Path(__file__).resolve().parent
    # A neighbouring inspect.py must never shadow Python's stdlib inspect.
    sys.path[:] = [str(script_directory.parents[1])] + [
        item for item in sys.path if Path(item or ".").resolve() != script_directory]

from scripts.fidel_ts.common import (
    AuditError, DEFAULT_CONFIG, IDENTITY, MANIFEST, ROOT, assert_identity,
    data_path, dataset_directory, identity, json_digest, load_source,
    normalize_inventory, read_json, selected_files, sha256_file,
    stage0_inventory, utc_now, write_json,
)


def fetch_inventory(dataset, spec):
    """Fetch metadata only. Never resolve a branch such as main."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise AuditError("Install huggingface_hub to fetch an inventory; --stage0 is offline") from exc
    files = []
    for item in HfApi().list_repo_tree(spec["repository"], repo_type="dataset",
                                      revision=spec["revision"], recursive=True):
        if not hasattr(item, "size"):
            continue
        lfs = getattr(item, "lfs", None)
        checksum = (lfs.get("sha256") if isinstance(lfs, dict)
                    else getattr(lfs, "sha256", None))
        files.append({"path": item.path, "size": item.size, "sha256": checksum,
                      "git_blob_sha1": None if lfs else getattr(item, "blob_id", None)})
    return normalize_inventory({**identity(dataset, spec), "origin": "huggingface_hub",
                                "retrieved_at": utc_now(), "complete_repository_listing": True,
                                "files": files}, dataset, spec)


def check_file(path, expected, recorded_sha256=None):
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"Missing file or disallowed symlink: {path}")
    size = path.stat().st_size
    if expected["size"] is None or size != expected["size"]:
        raise AuditError(f"Size mismatch: {path}; expected={expected['size']}, actual={size}")
    checksum = sha256_file(path)
    if recorded_sha256 is not None and checksum != recorded_sha256:
        raise AuditError(f"Recorded SHA-256 mismatch: {path}")
    if expected.get("sha256") and checksum != expected["sha256"]:
        raise AuditError(f"Source SHA-256 mismatch: {path}")
    if expected.get("git_blob_sha1"):
        digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["git_blob_sha1"]:
            raise AuditError(f"Source Git blob hash mismatch: {path}")
    return size, checksum


def verify_manifest(dataset, spec, directory, profile=None):
    """Offline, read-only verification. A completed manifest is required."""
    directory = Path(directory)
    errors = []
    try:
        assert_identity(read_json(data_path(directory, IDENTITY)), dataset, spec)
        manifest = read_json(data_path(directory, MANIFEST))
        assert_identity(manifest, dataset, spec)
        inventory = normalize_inventory(manifest["inventory"], dataset, spec)
        if manifest.get("inventory_digest") != json_digest(inventory["files"]):
            raise AuditError("Inventory digest mismatch")
        if not inventory.get("complete_repository_listing"):
            raise AuditError("Inventory does not claim a complete repository listing")
        profiles = manifest.get("profiles", [])
        if not profiles or len(profiles) != len(set(profiles)):
            raise AuditError("Missing or duplicated manifest profiles")
        required = {entry["path"]: entry for name in profiles
                    for entry in selected_files(inventory, spec, name)}
        if profile:
            requested = {entry["path"] for entry in selected_files(inventory, spec, profile)}
            if not requested.issubset(required):
                errors.append(f"Profile {profile} is not fully covered by this manifest")
        records = manifest.get("files", {})
        if not isinstance(records, dict) or set(records) != set(required):
            raise AuditError("Manifest file set is incomplete or contains unexpected entries")
        for name, entry in required.items():
            record = records[name]
            try:
                if not isinstance(record, dict):
                    raise AuditError(f"Malformed manifest record: {name}")
                if record.get("expected") != entry or record.get("status") != "complete":
                    raise AuditError(f"Incomplete or changed manifest entry: {name}")
                if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
                    raise AuditError(f"Missing recorded SHA-256: {name}")
                size, _ = check_file(data_path(directory, name), entry, record["sha256"])
                if record.get("size") != size:
                    raise AuditError(f"Recorded size mismatch: {name}")
            except (AuditError, OSError) as exc:
                errors.append(str(exc))
        controls = {IDENTITY, MANIFEST, ".fidel_ts_download.lock"}
        for path in directory.rglob("*"):
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                errors.append(f"Unexpected symlink: {relative}")
            elif path.is_file() and relative not in controls and relative not in required:
                if not relative.startswith(".fidel_ts_staging/"):
                    errors.append(f"Unexpected/unrecorded file: {relative}")
        return {"ok": not errors, "dataset": dataset, "revision": spec["revision"],
                "directory": str(directory), "profiles": profiles,
                "checked_files": len(required), "errors": errors}
    except (AuditError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "dataset": dataset, "directory": str(directory),
                "errors": [str(exc)]}


@contextmanager
def download_lock(directory):
    """Avoid concurrent writers; retain no background process."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = data_path(directory, ".fidel_ts_download.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AuditError(f"Download lock exists: {lock}; check for another writer before removing it") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def cached_download(spec, entry, cache_directory):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AuditError("Install huggingface_hub before downloading") from exc
    # The normal Hub cache retains incomplete transfers and resumes them.
    # Download into that cache, never directly into the published data directory.
    return Path(hf_hub_download(repo_id=spec["repository"], repo_type="dataset",
                                filename=entry["path"], revision=spec["revision"],
                                cache_dir=str(cache_directory)))


def publish_file(cached, destination, entry, staging_directory):
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir(exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="copy-", suffix=".part", dir=staging_directory)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as target, Path(cached).open("rb") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        size, checksum = check_file(temporary, entry)
        # Same-filesystem hard link is atomic and fails rather than overwriting.
        os.link(temporary, destination)
        return size, checksum
    finally:
        temporary.unlink(missing_ok=True)


def download_dataset(dataset, spec, inventory, profile, directory, cache_directory,
                     retries=2, fetch_file=cached_download):
    inventory = normalize_inventory(inventory, dataset, spec)
    if inventory.get("origin") == "stage0_local_evidence":
        raise AuditError("Download requires a Stage 1 inventory, not the partial Stage 0 size evidence")
    selection = selected_files(inventory, spec, profile)
    if not inventory.get("complete_repository_listing"):
        raise AuditError("Download requires a complete repository inventory")
    if any(item["size"] is None for item in selection):
        raise AuditError("Selected sizes are unknown; fetch and save a live inventory first")
    directory = Path(directory)
    if directory.is_symlink():
        raise AuditError("Dataset directory cannot be a symlink")
    if directory.exists() and not (directory / IDENTITY).exists():
        if any(directory.iterdir()):
            raise AuditError("Nonempty unmanaged directory; refusing to adopt or overwrite data")
    with download_lock(directory):
        marker = data_path(directory, IDENTITY)
        manifest_path = data_path(directory, MANIFEST)
        if marker.exists():
            assert_identity(read_json(marker), dataset, spec)
            if not manifest_path.exists():
                raise AuditError("Managed directory has no manifest; inspect before retrying")
        else:
            write_json(marker, identity(dataset, spec))
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            assert_identity(manifest, dataset, spec)
            previous = normalize_inventory(manifest["inventory"], dataset, spec)
            if (json_digest(previous["files"]) != manifest.get("inventory_digest")
                    or previous["files"] != inventory["files"]):
                raise AuditError("Inventory changed for an existing managed snapshot")
        else:
            manifest = {**identity(dataset, spec), "created_at": utc_now(),
                        "inventory": inventory, "inventory_digest": json_digest(inventory["files"]),
                        "profiles": [], "files": {}}
        if profile not in manifest["profiles"]:
            manifest["profiles"].append(profile)
        for item in selection:
            manifest["files"].setdefault(item["path"],
                                         {"expected": item, "status": "pending", "attempts": []})
        write_json(manifest_path, manifest)
        staging = data_path(directory, ".fidel_ts_staging")
        for entry in selection:
            name = entry["path"]
            record = manifest["files"][name]
            destination = data_path(directory, name)
            if destination.exists():
                # An interrupted metadata write can leave an orphan. Only recover
                # it if an independent pinned-source content hash authenticates it.
                if (record.get("status") != "complete" and not entry.get("sha256")
                        and not entry.get("git_blob_sha1")):
                    raise AuditError(f"Unrecorded existing file without source hash: {name}")
                size, checksum = check_file(destination, entry, record.get("sha256"))
                record.update(status="complete", size=size, sha256=checksum, last_action="checked_existing")
                write_json(manifest_path, manifest)
                continue
            if record.get("status") == "complete":
                raise AuditError(f"Previously completed file disappeared: {name}; investigate before retrying")
            for attempt in range(retries + 1):
                event = {"at": utc_now(), "attempt_in_invocation": attempt + 1, "status": "started"}
                record["attempts"].append(event)
                record["status"] = "downloading"
                write_json(manifest_path, manifest)
                try:
                    cached = fetch_file(spec, entry, cache_directory)
                    size, checksum = publish_file(cached, destination, entry, staging)
                    record.update(status="complete", size=size, sha256=checksum, last_action="downloaded")
                    event["status"] = "complete"
                    write_json(manifest_path, manifest)
                    break
                except Exception as exc:
                    event.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    record["status"] = "failed"
                    write_json(manifest_path, manifest)
                    if attempt == retries:
                        raise AuditError(f"Download failed for {name}; manifest retains failure/retry details: {exc}") from exc
                    time.sleep(min(2 ** attempt, 8))
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
    return verify_manifest(dataset, spec, directory, profile)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("list", "dry-run", "download", "verify"))
    parser.add_argument("--dataset", required=True, choices=("caiso", "jena"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--revision", help="Optional assertion; must equal the fixed SHA")
    parser.add_argument("--profile", choices=("numerical", "standard", "provenance"), default="standard")
    parser.add_argument("--output-root", type=Path, default=ROOT / "dataset/Fidel-TS/raw")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--inventory", type=Path, help="Use a saved Stage 1 inventory; no metadata network request")
    source.add_argument("--stage0", action="store_true", help="Use local Stage 0 evidence; unknown sizes stay unknown")
    parser.add_argument("--save-inventory", type=Path, help="For list only; save metadata outside the raw directory")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "dataset/Fidel-TS/.hf-cache")
    parser.add_argument("--retries", type=int, default=2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.retries < 0:
            raise AuditError("--retries must be nonnegative")
        if args.save_inventory and args.operation != "list":
            raise AuditError("--save-inventory is allowed only with list; dry-run never writes files")
        if args.stage0 and args.operation == "download":
            raise AuditError("Stage 0 evidence is for list/dry-run; download using a saved Stage 1 inventory")
        spec = load_source(args.dataset, args.config, args.revision)
        directory = dataset_directory(args.output_root, spec)
        if args.operation == "verify":
            if args.inventory or args.stage0:
                raise AuditError("verify uses the managed local manifest, without remote metadata")
            result = verify_manifest(args.dataset, spec, directory, args.profile)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 3
        if args.stage0:
            inventory = stage0_inventory(ROOT / "docs/fidel_ts/00_audit_evidence.json", args.dataset, spec)
        elif args.inventory:
            inventory = normalize_inventory(read_json(args.inventory), args.dataset, spec)
        else:
            inventory = fetch_inventory(args.dataset, spec)
        selection = selected_files(inventory, spec, args.profile)
        if args.operation == "download":
            result = download_dataset(args.dataset, spec, inventory, args.profile,
                                      directory, args.cache_dir, args.retries)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 3
        chosen = {item["path"] for item in selection}
        rows = [{"dataset": args.dataset, "repository": spec["repository"],
                 "revision": spec["revision"], "profile": args.profile,
                 "path": item["path"], "size": item["size"],
                 "selected": item["path"] in chosen,
                 "local_exists": data_path(directory, item["path"]).is_file(),
                 "destination": str(data_path(directory, item["path"]))}
                for item in (inventory["files"] if args.operation == "list" else selection)]
        unknown = sum(item["size"] is None for item in selection)
        print(json.dumps({"operation": args.operation, "directory": str(directory),
                          "selected_files": len(selection), "unknown_size_files": unknown,
                          "known_selected_bytes": sum(item["size"] or 0 for item in selection),
                          "estimated_total_bytes": None if unknown else sum(item["size"] for item in selection),
                          "files": rows}, ensure_ascii=False, indent=2))
        if args.save_inventory:
            if args.save_inventory.exists():
                raise AuditError("Inventory output already exists; preserve it or choose a new path")
            if args.save_inventory.resolve().is_relative_to(directory.resolve()):
                raise AuditError("Save inventories outside the managed raw directory")
            write_json(args.save_inventory, inventory)
        return 0
    except (AuditError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
