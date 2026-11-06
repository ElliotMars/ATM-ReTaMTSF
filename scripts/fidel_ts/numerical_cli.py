"""Shared argument handling for offline Stage 1.5 numerical commands."""

import argparse
import json
from pathlib import Path
import sys

from scripts.fidel_ts.common import AuditError, DEFAULT_CONFIG, ROOT, data_path, write_json
from scripts.fidel_ts.numerical_diagnostics import diagnose, load_native
from scripts.fidel_ts.numerical_protocols import ProtocolBlocked, build_protocol, write_prepared
from scripts.fidel_ts.stage1_5_io import (
    PROTOCOL_CONFIG, finish_output, input_binding, load_protocol, new_output,
    preview, recheck_input, runtime_versions, snapshot,
)


def run_numerical(operation, argv=None):
    parser = argparse.ArgumentParser(description=f"Offline Stage 1.5 numerical {operation}; run manually")
    parser.add_argument("--dataset", choices=("caiso", "jena"), required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset/Fidel-TS/raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROTOCOL_CONFIG)
    parser.add_argument("--sources-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    if operation == "prepare":
        parser.add_argument("--protocol", choices=("n1", "n2"), required=True)
        parser.add_argument("--mode", choices=("report-only", "prepare"), default="report-only")
    args = parser.parse_args(argv)
    try:
        config = load_protocol(args.config)
        spec, directory, manifest = snapshot(args.dataset, args.input_root, args.sources_config, not args.dry_run)
        if args.dry_run:
            print(json.dumps(preview(args.dataset, spec, directory, manifest, args.output_dir, operation, config), indent=2))
            return 0
        binding = {**input_binding(args.dataset, spec, directory, manifest, config),
                   "runtime_versions": runtime_versions()}
        with new_output(args.output_dir, (args.input_root, directory, ROOT / "artifacts/fidel_ts/data_audit")) as output:
            data = load_native(data_path(directory, spec["primary_file"]), args.dataset, spec)
            blocked = []
            if operation == "diagnose":
                for name, value in diagnose(data, config).items():
                    write_json(output / name, value)
                write_json(output / "manifest.json", binding)
            else:
                try:
                    result = build_protocol(data, args.protocol, config)
                    blocked = write_prepared(output, result, config, binding, args.mode == "prepare")
                except ProtocolBlocked as exc:
                    blocked = [str(exc)]
                    write_json(output / "quality_summary.json", {"protocol": args.protocol, "blocked_reasons": blocked})
                    write_json(output / "manifest.json", {**binding, "protocol": args.protocol, "mode": args.mode})
            recheck_input(args.dataset, spec, directory, binding["primary_sha256"], binding["snapshot_manifest_sha256"])
            finish_output(output, binding, blocked)
        print(json.dumps({"dataset": args.dataset, "operation": operation, "output": str(args.output_dir),
                          "status": "blocked" if blocked else "complete", "blocked_reasons": blocked}, indent=2))
        return 3 if blocked else 0
    except (AuditError, OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
