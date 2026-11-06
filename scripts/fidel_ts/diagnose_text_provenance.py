#!/usr/bin/env python3
"""Offline report/source diagnostics, never a declaration of historical issuance."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    directory = Path(__file__).resolve().parent
    sys.path[:] = [str(directory.parents[1])] + [p for p in sys.path if Path(p or ".").resolve() != directory]

import argparse
import json

from scripts.fidel_ts.common import AuditError, DEFAULT_CONFIG, ROOT, assert_identity, read_json, write_json
from scripts.fidel_ts.stage1_5_io import (
    PROTOCOL_CONFIG, finish_output, input_binding, load_protocol, new_output,
    preview, read_bundle, recheck_input, runtime_versions, snapshot,
)
from scripts.fidel_ts.text_provenance import analyse_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("caiso", "jena"), required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset/Fidel-TS/raw")
    parser.add_argument("--audit-dir", type=Path, help="Successful Stage 1 audit directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROTOCOL_CONFIG)
    parser.add_argument("--sources-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_protocol(args.config)
        spec, directory, manifest = snapshot(args.dataset, args.input_root, args.sources_config, not args.dry_run)
        audit_dir = args.audit_dir or ROOT / "artifacts/fidel_ts/data_audit" / args.dataset
        if args.dry_run:
            result = preview(args.dataset, spec, directory, manifest, args.output_dir, "text_provenance", config)
            result["stage1_audit_directory"] = str(audit_dir)
            print(json.dumps(result, indent=2))
            return 0
        prior = read_bundle(audit_dir)
        assert_identity(prior, args.dataset, spec)
        audit = read_json(audit_dir / "manifest.json")
        assert_identity(audit, args.dataset, spec)
        for entry in audit["files"]:
            current = manifest["files"].get(entry["path"], {})
            if current.get("sha256") != entry["sha256"] or current.get("status") != "complete":
                raise AuditError("Stage 1 audit is not bound to the current verified source files")
        binding = {**input_binding(args.dataset, spec, directory, manifest, config),
                   "stage1_audit_manifest_sha256": prior["artifacts"]["manifest.json"],
                   "stage1_ledger_sha256": prior["artifacts"]["documents.jsonl"],
                   "runtime_versions": runtime_versions()}
        with new_output(args.output_dir, (args.input_root, directory, audit_dir, ROOT / "artifacts/fidel_ts/data_audit")) as output:
            structure, sources, temporal = analyse_text(directory, audit_dir, spec, args.dataset, manifest, config, output)
            for name, value in (("manifest.json", binding), ("text_report_structure.json", structure),
                                ("text_source_analysis.json", sources), ("text_temporal_analysis.json", temporal)):
                write_json(output / name, value)
            recheck_input(args.dataset, spec, directory, binding["primary_sha256"], binding["snapshot_manifest_sha256"])
            finish_output(output, binding)
        print(json.dumps({"dataset": args.dataset, "output": str(args.output_dir),
                          "reports": structure["reports"], "facets": structure["facets"],
                          "T1": temporal["T1"], "missing_provenance_files": len(sources["missing_provenance_files"])}, indent=2))
        return 0
    except (AuditError, OSError, KeyError, TypeError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
