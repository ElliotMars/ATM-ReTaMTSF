"""Explicit experiment matrix. No data selection based on model results."""
from copy import deepcopy
from pathlib import Path
import yaml

from .common import ROOT, ARTIFACTS, DATA, TASKS, read_json

MAIN = ["naive", "dlinear", "patchtst", "backbone", "text_mean", "latest", "semantic", "semantic_temporal", "full_pivot"]
ABLATIONS = ["without_ctsr", "without_str", "without_temporal", "without_fusion", "random_eligible", "shuffled_eligible",
             "text_missing", "topk1", "topk8", "lag0", "lag3"]


def configuration(dataset, horizon, variant="full_pivot", seed=2024):
    _, horizons, channels = TASKS[dataset]
    if horizon not in horizons or variant not in MAIN + ABLATIONS:
        raise ValueError("Unknown frozen experiment")
    numerical = variant in {"naive", "dlinear", "patchtst", "backbone", "text_missing", "without_fusion"}
    model = {"name": "pivot", "dim": 32, "layers": 1, "heads": 4, "patch": 8 if dataset == "stocknet" else 16,
             "stride": 4 if dataset == "stocknet" else 8, "dropout": 0.1, "use_ctsr": True,
             "channel_top_k": 4, "fusion": "cross_attention"}
    retrieval = {"strategy": "semantic_temporal", "top_k": 4, "temporal_weight": 0.2, "lag_hours": 1}
    if variant == "naive":
        model["name"] = "zero" if dataset == "stocknet" else "persistence"
    elif variant in {"dlinear", "patchtst"}:
        model["name"] = variant
    if numerical:
        model["fusion"], retrieval["strategy"] = "none", "none"
    if variant in {"text_mean", "latest", "semantic", "semantic_temporal", "without_ctsr"}:
        model["use_ctsr"] = False
    if variant in {"text_mean", "without_str"}:
        retrieval["strategy"] = "mean"
    if variant == "text_mean":
        model["fusion"] = "mean"
    if variant in {"latest", "semantic"}:
        retrieval["strategy"] = variant
    if variant == "without_temporal":
        retrieval["strategy"] = "semantic"
    if variant == "random_eligible":
        retrieval["strategy"] = "random"
    if variant == "shuffled_eligible":
        retrieval["strategy"] = "shuffled"
    if variant in {"topk1", "topk8"}:
        retrieval["top_k"] = int(variant[-1])
    if variant in {"lag0", "lag3"}:
        retrieval["lag_hours"] = int(variant[-1])
    enabled = dataset == "stocknet" or numerical
    return {"dataset": dataset, "data_directory": str(DATA / dataset), "variant": variant, "enabled": enabled,
            "disabled_reason": None if enabled else "external_text_alignment_not_admitted; use numerical controls only",
            "horizon": horizon, "seed": seed, "formal_seeds": [2024, 2025, 2026], "device": "auto", "cpu_threads": 2,
            "batch_size": 8, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 30,
            "patience": 5, "amp": True, "selection_metric": "mse_standardized", "model": model, "retrieval": retrieval,
            "comparison_rule": "same prepared windows/scaler/y_mask; text absence never removes a sample",
            "information_permission": "creation_time_plus_lag_conditional_not_verified_receipt" if dataset == "stocknet" else "numerical_only_no_clock_guessing"}


def generate():
    for dataset, (_, horizons, _) in TASKS.items():
        directory = ROOT / "configs/pivot_stage2" / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for horizon in horizons:
            for variant in MAIN + ABLATIONS:
                path = directory / f"H{horizon}_{variant}.yaml"
                config = configuration(dataset, horizon, variant)
                text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
                if path.exists() and path.read_text() != text:
                    raise ValueError(f"Refusing to replace changed experiment config: {path}")
                path.write_text(text)


if __name__ == "__main__":
    generate()
