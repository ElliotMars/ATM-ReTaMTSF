"""Aggregate completed formal runs; never mix smoke scores into research tables."""
from collections import defaultdict
from datetime import datetime
import csv
from pathlib import Path
import uuid

import numpy as np
import yaml

from .common import ARTIFACTS, read_json, verify, new_directory, write_json, seal
from .metrics import block_bootstrap


def collect(root=None):
    root = Path(root or ARTIFACTS / "experiments")
    records, seen = [], set()
    groups, fingerprints = defaultdict(list), {}
    for path in sorted(root.rglob("summary.json")) if root.exists() else []:
        if not (path.parent / "_SUCCESS.json").exists():
            continue
        verify(path.parent)
        summary = read_json(path)
        if summary["smoke_only"]:
            continue
        config = yaml.safe_load((path.parent / "config.yaml").read_text())
        key = (summary["dataset"], summary["horizon"], config["variant"])
        unique = (*key, summary["seed"])
        if unique in seen:
            raise ValueError("Duplicate dataset/horizon/variant/seed; no selecting favorable repeats")
        seen.add(unique)
        comparable = key[:2]
        fingerprint = summary["test"]["sample_ids_sha256"]
        if comparable in fingerprints and fingerprints[comparable] != fingerprint:
            raise ValueError("Methods were evaluated on different samples")
        fingerprints[comparable] = fingerprint
        row = {"dataset": key[0], "horizon": key[1], "variant": key[2], "seed": summary["seed"],
               **{name: summary["test"].get(name) for name in ("mse_standardized", "mae_standardized", "mse", "mae", "directional_accuracy", "rank_ic", "label_coverage")},
               "parameters": summary["parameters"], "wall_seconds": summary["wall_seconds"], "run": str(path.parent)}
        records.append(row)
        groups[key].append(row)
    aggregate = []
    for key, rows in groups.items():
        item = {"dataset": key[0], "horizon": key[1], "variant": key[2], "seeds": sorted(row["seed"] for row in rows), "statistics": {}}
        curves, origins = [], None
        for row in rows:
            errors = np.load(Path(row["run"]) / "test/origin_errors.npz")
            if origins is not None and not np.array_equal(origins, errors["origins"]):
                raise ValueError("Bootstrap origins differ")
            origins = errors["origins"]
            curves.append(errors["mse_standardized"])
        for metric in ("mse_standardized", "mae_standardized", "mse", "mae", "directional_accuracy", "rank_ic"):
            values = [row[metric] for row in rows if row[metric] is not None]
            item["statistics"][metric] = {"mean": float(np.mean(values)) if values else None,
                                          "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        item["seed_mean_temporal_block_ci"] = block_bootstrap(np.mean(curves, axis=0), key[1])
        baseline = groups.get((key[0], key[1], "backbone"), [])
        paired = []
        by_seed = {row["seed"]: row for row in baseline}
        for row in rows:
            if row["seed"] in by_seed:
                a = np.load(Path(row["run"]) / "test/origin_errors.npz")["mse_standardized"]
                b = np.load(Path(by_seed[row["seed"]]["run"]) / "test/origin_errors.npz")["mse_standardized"]
                paired.append(a - b)
        item["paired_mse_difference_vs_backbone_ci"] = block_bootstrap(np.mean(paired, axis=0), key[1]) if paired else None
        aggregate.append(item)
    output = ARTIFACTS / "collections" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    with new_directory(output):
        write_json(output / "results.json", {"status": "formal_results_available" if records else "no_formal_results_yet",
            "runs": records, "aggregates": aggregate, "smoke_results_excluded": True,
            "bootstrap_scope": "seed-averaged errors with contiguous origin blocks >= horizon; seed variance reported separately"})
        fields = list(records[0]) if records else ["dataset", "horizon", "variant", "seed", "mse_standardized", "mae_standardized", "mse", "mae"]
        with (output / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        seal(output, {"kind": "formal_result_collection", "runs": len(records)})
    write_json(ARTIFACTS / "latest_collection.json", {"directory": str(output)})
    return str(output)


if __name__ == "__main__":
    print(collect())
