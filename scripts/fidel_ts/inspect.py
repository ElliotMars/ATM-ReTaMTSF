#!/usr/bin/env python3
"""Audit local CAISO/Jena snapshots without modifying source data. Run manually."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    script_directory = Path(__file__).resolve().parent
    # Prevent this filename from shadowing stdlib inspect in pandas/NumPy.
    sys.path[:] = [str(script_directory.parents[1])] + [
        item for item in sys.path if Path(item or ".").resolve() != script_directory]

from scripts.fidel_ts.common import (
    AuditError, DEFAULT_CONFIG, MANIFEST, ROOT, data_path, dataset_directory,
    identity, load_source, read_json, selected_files, sha256_file, utc_now, write_json,
)
from scripts.fidel_ts.download import verify_manifest
from scripts.fidel_ts.inspection import caiso_alignment, numerical_summary, text_summary


def inspect_dataset(dataset, spec, directory, output_dir, profile="standard", config_path=DEFAULT_CONFIG):
    directory, output_dir = Path(directory), Path(output_dir)
    if (output_dir.resolve().is_relative_to(directory.resolve())
            or directory.resolve().is_relative_to(output_dir.resolve())):
        raise AuditError("Audit output must be separate from the input dataset directory")
    verified = verify_manifest(dataset, spec, directory, profile)
    if not verified["ok"]:
        raise AuditError("Verify the local snapshot first: " + "; ".join(verified["errors"]))
    manifest_path = data_path(directory, MANIFEST)
    initial_manifest_hash = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    selection = selected_files(manifest["inventory"], spec, profile)
    names = [item["path"] for item in selection]
    required = (spec["primary_file"], spec["static_info_file"])
    if any(name not in names for name in required):
        raise AuditError("Selected profile lacks the primary numerical or static metadata file")
    # An existing output is deliberately never replaced, including failed runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        static = read_json(data_path(directory, spec["static_info_file"]))
        numbers = numerical_summary(data_path(directory, spec["primary_file"]), static, spec)
        numbers["input_alignment"] = (caiso_alignment(directory, names) if dataset == "caiso" else {
            "status": "not_attested", "reason": "Jena numerical preprocessing provenance remains unresolved"})
        weather_paths = [name for name in names if "/weather_report/" in name
                         and name.endswith(".json") and Path(name).name != "static_info.json"]
        with (output_dir / "documents.jsonl").open("x", encoding="utf-8") as handle:
            texts, availability = text_summary(directory, weather_paths, dataset, spec, handle)
        # Detect changes while auditing rather than certifying a moving input.
        final_verified = verify_manifest(dataset, spec, directory, profile)
        if not final_verified["ok"] or sha256_file(manifest_path) != initial_manifest_hash:
            raise AuditError("Input snapshot changed during inspection; keep this failed run and retry separately")
        meta = {**identity(dataset, spec), "audit_at": utc_now(), "profile": profile,
                "input_directory": str(directory.resolve()), "source_config_sha256": sha256_file(config_path),
                "download_manifest_sha256": initial_manifest_hash, "snapshot_verification": verified,
                "files": [{"path": name, "size": manifest["files"][name]["size"],
                           "sha256": manifest["files"][name]["sha256"]} for name in names],
                "original_data_modified": False,
                "scope": "local_data_diagnostics_not_training_approval",
                "availability_evidence_policy": "no release evidence is attested by this Stage 1 CLI"}
        for name, value in (("manifest.json", meta), ("numerical_summary.json", numbers),
                            ("text_summary.json", texts), ("availability_summary.json", availability)):
            write_json(output_dir / name, value)
        write_json(output_dir / "_SUCCESS.json", {
            **identity(dataset, spec), "completed_at": utc_now(),
            "meaning": "audit_finished_not_data_quality_or_causality_approval",
            "artifacts": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()}})
        return {"output_directory": str(output_dir), "rows": numbers["rows"],
                "numeric_channels": numbers["numeric_channel_count"], "text_documents": availability["documents"],
                "formal_multimodal_training_allowed": False}
    except Exception as exc:
        write_json(output_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                                                   "at": utc_now(), "complete": False})
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("caiso", "jena"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--revision", help="Optional assertion against the fixed SHA")
    parser.add_argument("--profile", choices=("numerical", "standard", "provenance"), default="standard")
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset/Fidel-TS/raw")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, nonexisting audit directory")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Explicit read-only-data mode; this CLI never cleans or resamples even without this flag")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        spec = load_source(args.dataset, args.config, args.revision)
        result = inspect_dataset(args.dataset, spec, dataset_directory(args.input_root, spec),
                                 args.output_dir, args.profile, args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (AuditError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
