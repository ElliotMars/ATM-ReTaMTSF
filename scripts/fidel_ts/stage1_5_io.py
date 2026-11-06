"""Offline snapshot binding and non-overwriting output transactions."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import platform
from importlib.metadata import PackageNotFoundError, version

from scripts.fidel_ts.common import (
    AuditError, DEFAULT_CONFIG, MANIFEST, ROOT, assert_identity, data_path,
    dataset_directory, identity, json_digest, load_source, read_json,
    sha256_file, utc_now, write_json,
)
from scripts.fidel_ts.download import verify_manifest

PROTOCOL_CONFIG = ROOT / "configs/fidel_ts/protocols_stage1_5.json"
SPLITS = ("train", "validation", "test")


def load_protocol(path=PROTOCOL_CONFIG):
    config = read_json(path)
    if config.get("schema_version") != 1 or len(config.get("split_boundaries", [])) != 2:
        raise AuditError("Invalid Stage 1.5 configuration")
    if config.get("lookback") != 360 or config.get("horizons") != [24, 168, 336, 720]:
        raise AuditError("Stage 1.5 freezes L=360 and all four official horizons")
    if config.get("hour_seconds") != 3600:
        raise AuditError("Only the frozen hourly protocol is supported")
    if config["n2"].get("anchor_validity") != "at_least_one_valid_channel":
        raise AuditError("Unsupported anchor validity policy")
    if config["split_boundaries"] != ["2021-01-01T00:00:00", "2022-01-01T00:00:00"]:
        raise AuditError("Production split boundaries must retain 2021/2022")
    frozen = {
        "n1": {"sort": False, "native_fit_dtype": "float32", "sample_anchor": "first_original_row_in_each_split",
               "borrow_previous_split_history": False, "scaler": "sklearn.preprocessing.StandardScaler",
               "sentinels_retained_in_values_and_scaler": True},
        "n2": {"duplicate_policy": "collapse_exact_equal_exclude_conflict_or_indeterminate",
               "scaler_fit_scope": "valid_native_training_observations_after_duplicate_resolution",
               "hour_anchor": "earliest_retained_training_timestamp_with_valid_observation",
               "continuity": "hourly_steps_and_same_native_contiguous_segment",
               "borrow_previous_split_history": False,
               "label_policy": "shared_all_channel_validity_mask_never_fill_labels"}}
    for protocol, fields in frozen.items():
        if any(config[protocol].get(key) != value for key, value in fields.items()):
            raise AuditError(f"Unsupported {protocol} configuration; do not silently change the frozen algorithm")
    if config["sampling"]["reports_per_stratum"] < 1 or config["sampling"]["context_hours"] <= 0:
        raise AuditError("Sampling count and context duration must be positive")
    if not 0 <= config["n2"]["significant_window_retention_warning_below"] <= 1:
        raise AuditError("Retention warning threshold must be within [0,1]")
    if config["n2"]["minimum_eligible_windows_per_split_horizon"] < 1:
        raise AuditError("At least one eligible window is required in every split/horizon")
    return config


def snapshot(dataset, input_root, sources=DEFAULT_CONFIG, verify=True):
    spec = load_source(dataset, sources)
    directory = dataset_directory(input_root, spec)
    manifest = read_json(data_path(directory, MANIFEST))
    assert_identity(manifest, dataset, spec)
    if verify:
        result = verify_manifest(dataset, spec, directory, "numerical")
        if not result["ok"]:
            raise AuditError("Snapshot Verify failed: " + "; ".join(result["errors"]))
    return spec, directory, manifest


def preview(dataset, spec, directory, manifest, output_dir, operation, config):
    return {**identity(dataset, spec), "operation": operation, "dry_run": True,
            "input_directory": str(directory), "primary_file": spec["primary_file"],
            "primary_expected_bytes": manifest["files"][spec["primary_file"]]["size"],
            "output_directory": str(output_dir), "protocol_config_digest": json_digest(config),
            "data_read_or_checked": False, "network": False,
            "note": "metadata preview only; actual file checks and computations require a non-dry run"}


def input_binding(dataset, spec, directory, manifest, config):
    return {**identity(dataset, spec), "input_directory": str(directory.resolve()),
            "source_config_digest": json_digest(spec),
            "primary_file": spec["primary_file"],
            "primary_sha256": manifest["files"][spec["primary_file"]]["sha256"],
            "snapshot_manifest_sha256": sha256_file(data_path(directory, MANIFEST)),
            "protocol_config_digest": json_digest(config), "protocol_config": config,
            "created_at": utc_now()}


def recheck_input(dataset, spec, directory, primary_sha256, manifest_sha256=None):
    result = verify_manifest(dataset, spec, directory, "numerical")
    if not result["ok"] or sha256_file(data_path(directory, spec["primary_file"])) != primary_sha256:
        raise AuditError("Raw snapshot changed while processing; output cannot be published as successful")
    if manifest_sha256 is not None and sha256_file(data_path(directory, MANIFEST)) != manifest_sha256:
        raise AuditError("Raw snapshot manifest changed during this run; retry into a new output directory")


@contextmanager
def new_output(path, forbidden=()):
    path = Path(path)
    resolved = path.resolve()
    for other in forbidden:
        other = Path(other).resolve()
        if resolved.is_relative_to(other) or other.is_relative_to(resolved):
            raise AuditError(f"Output must be separate from protected input: {other}")
    # Never replace an old successful OR failed run.
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    except Exception as exc:
        write_json(path / "failure.json", {"complete": False, "at": utc_now(),
                                          "error_type": type(exc).__name__, "error": str(exc)})
        raise


def finish_output(path, binding, blocked_reasons=()):
    marker = "_BLOCKED.json" if blocked_reasons else "_SUCCESS.json"
    hashes = {p.relative_to(path).as_posix(): sha256_file(p)
              for p in sorted(path.rglob("*")) if p.is_file()}
    write_json(path / marker, {**binding, "status": "blocked" if blocked_reasons else "complete",
                              "blocked_reasons": list(blocked_reasons), "artifacts": hashes,
                              "meaning": "artifact_integrity_not_model_or_causality_approval"})


def read_bundle(path, allow_blocked=False):
    path = Path(path)
    marker = path / "_SUCCESS.json"
    if not marker.is_file() and allow_blocked:
        marker = path / "_BLOCKED.json"
    record = read_json(marker)
    if record.get("status") not in (None, "complete", "blocked"):
        raise AuditError(f"Invalid completion marker: {marker}")
    for name, digest in record["artifacts"].items():
        if sha256_file(data_path(path, name)) != digest:
            raise AuditError(f"Artifact hash mismatch: {path / name}")
    return record


def runtime_versions():
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
