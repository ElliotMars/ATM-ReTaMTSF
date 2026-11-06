"""Training/evaluation with immutable runs; smoke mode is exactly one step."""
import argparse
from copy import deepcopy
import csv
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .common import ART as ARTIFACTS, DATA, digest, new_directory, read_json, seal, sha, versions, write_json
from .data import MIMODataset, loader
from pivot_stage2.metrics import Accumulator, masked_mse
from .models import build_model

FEATURES = ("x_num", "x_mask", "text_embeddings", "text_mask", "gfs_num", "gfs_mask")
CODE_BINDING = {"v2/"+name: sha(Path(__file__).parent / name) for name in
                ("data.py", "models.py", "prepare.py", "timezone.py", "train.py")}
CODE_BINDING.update({"v1/"+name: sha(Path(__file__).parent.parent / "pivot_stage2" / name)
                     for name in ("models.py", "metrics.py", "data.py")})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def features(batch, device):
    return {name: batch[name].to(device) for name in FEATURES}


def evaluate(model, batches, device, meta, output=None, max_batches=None):
    model.eval()
    accumulator = Accumulator(len(meta["channels"]), meta["dataset"] == "stocknet")
    standardized = Accumulator(len(meta["channels"]))
    mean, scale = np.asarray(meta["scaler"]["mean"]), np.asarray(meta["scaler"]["scale"])
    ids, origins = [], []
    arrays = None
    if output:
        output = Path(output)
        output.mkdir()
        size = len(batches.dataset) if max_batches is None else min(len(batches.dataset), max_batches * batches.batch_size)
        shape = (size, batches.dataset.horizon, len(meta["channels"]))
        arrays = [np.lib.format.open_memmap(output / name, mode="w+", dtype=dtype, shape=shape)
                  for name, dtype in (("prediction.npy", np.float64), ("truth.npy", np.float64), ("mask.npy", bool))]
    cursor = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for number, batch in enumerate(batches):
            if max_batches is not None and number >= max_batches:
                break
            p_standardized = model(**features(batch, device)).cpu().numpy().astype(np.float64)
            p = p_standardized * scale + mean
            if getattr(model, "zero", False):
                p.fill(0)  # Exact raw zero return, not rounded normalized train mean.
            y = batch["y"].numpy().astype(np.float64) * scale + mean
            mask = batch["y_mask"].numpy()
            standardized.update(p_standardized, batch["y"].numpy(), mask)
            y = np.where(mask, y, np.nan)
            accumulator.update(p, y, mask)
            if arrays:
                for array, value in zip(arrays, (p, y, mask)):
                    array[cursor:cursor + len(p)] = value
            cursor += len(p)
            ids.extend(batch["sample_id"])
            origins.extend(batch["numeric_time"])
    result = accumulator.result()
    standardized_result = standardized.result()
    result.update(mse_standardized=standardized_result["mse"], mae_standardized=standardized_result["mae"])
    result.update(samples=cursor, inference_seconds=time.perf_counter() - started,
                  sample_ids_sha256=digest(ids), raw_unit_metrics=True)
    if output:
        for array in arrays:
            array.flush()
        write_json(output / "sample_ids.json", ids)
        np.savez(output / "origin_errors.npz", origins=np.asarray(origins, dtype=np.int64), mse=np.asarray(accumulator.origin_mse),
                 mse_standardized=np.asarray(standardized.origin_mse))
        write_json(output / "metrics.json", result)
    return result


def run(config, output, smoke=False, resume=None):
    config = deepcopy(config)
    if not config.get("enabled", True):
        raise ValueError("This experimental combination is disabled by data admission")
    set_seed(config.get("seed", 2024))
    torch.set_num_threads(config.get("cpu_threads", 2))
    requested_device = config.get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu" if requested_device == "auto" else requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use the prepared CPU environment or a compatible GPU runtime")
    dataset_name, horizon = config["dataset"], config["horizon"]
    data_dir = Path(config.get("data_directory", DATA / dataset_name))
    retrieval = config.get("retrieval", {})
    # Verify sources ONCE per run; shared prepared directory remains immutable.
    datasets = {split: MIMODataset(data_dir, split, horizon, check_hashes=split == "train", seed=config["seed"], **retrieval)
                for split in ("train", "validation", "test")}
    if smoke:
        # One real sample per split, explicitly spanning winter, summer and DST.
        wanted = {"train": "2022-01-15 12:10", "validation": "2024-07-15 12:10", "test": "2025-03-30 12:10"}
        import pandas as pd
        for split, ds in datasets.items():
            if len(ds):
                origin_naive = ds.naive_times[ds.starts + ds.lookback - 1]
                selected = int(np.argmin(abs(origin_naive - pd.Timestamp(wanted[split]).value)))
                ds.starts = ds.starts[selected:selected + 1]
    if any(not len(value) for value in datasets.values()):
        raise ValueError("Each split needs eligible windows")
    meta = datasets["train"].meta
    batch_size = 1 if smoke else config.get("batch_size", 8)
    batches = {split: loader(value, batch_size, split == "train", config["seed"]) for split, value in datasets.items()}
    model_config = {**config.get("model", {}), "lookback": meta["lookback"], "channels": len(meta["channels"]), "horizon": horizon,
                    "zero_values": (-np.array(meta["scaler"]["mean"]) / np.array(meta["scaler"]["scale"])).tolist()}
    model = build_model(model_config).to(device)
    params = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-3), weight_decay=config.get("weight_decay", 1e-4)) if params else None
    amp = device.type == "cuda" and config.get("amp", True)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best, best_state, first_epoch, patience = [], float("inf"), deepcopy(model.state_dict()), 0, 0
    binding = {"prepared_data_sha256": sha(data_dir / "_SUCCESS.json"), "gfs_data_sha256": sha(DATA / "gfs/_SUCCESS.json"), "model": model_config, "retrieval": retrieval,
               "seed": config["seed"], "horizon": horizon, "code_sha256": CODE_BINDING,
               "training": {key: config.get(key) for key in ("batch_size", "learning_rate", "weight_decay", "selection_metric", "patience")}}
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved["binding"] != binding:
            raise ValueError("Checkpoint data/model/retrieval/seed binding mismatch")
        model.load_state_dict(saved["model"])
        if optimizer:
            optimizer.load_state_dict(saved["optimizer"])
        best, best_state, first_epoch = saved["best"], saved["best_model"], saved["epoch"] + 1
        history, patience = saved["history"], saved["patience"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if saved.get("amp_scaler"):
            grad_scaler.load_state_dict(saved["amp_scaler"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with new_directory(output) as directory:
        (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(directory / "binding.json", binding)
        epochs = 1 if smoke else config.get("epochs", 30)
        optimizer_steps = 0
        for epoch in range(first_epoch, first_epoch + 1 if smoke else epochs):
            model.train()
            batches["train"].sampler.seed = config["seed"] + epoch
            losses, train_started = [], time.perf_counter()
            if optimizer:
                for number, batch in enumerate(batches["train"]):
                    if smoke and number >= 1:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        prediction = model(**features(batch, device))
                        loss = masked_mse(prediction, batch["y"].to(device), batch["y_mask"].to(device))
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer_steps += 1
                    losses.append(float(loss.detach().cpu()))
            train_seconds = time.perf_counter() - train_started
            validation = evaluate(model, batches["validation"], device, meta, max_batches=1 if smoke else None)
            selection_value = validation[config.get("selection_metric", "mse_standardized")]
            improved = selection_value < best
            if improved:
                best, best_state, patience = selection_value, deepcopy(model.state_dict()), 0
            else:
                patience += 1
            history.append({"epoch": epoch, "train_mse_standardized": float(np.mean(losses)) if losses else None,
                            "validation_mse_raw": validation["mse"], "train_seconds_including_loading": train_seconds,
                            "validation_seconds": validation["inference_seconds"], "seconds": time.perf_counter() - train_started})
            state = {"binding": binding, "model": model.state_dict(), "best_model": best_state,
                     "optimizer": optimizer.state_dict() if optimizer else None, "best": best, "epoch": epoch, "patience": patience,
                     "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "amp_scaler": grad_scaler.state_dict()}
            torch.save(state, directory / "latest.pt")
            if improved:
                torch.save(state, directory / "best.pt")
            if not optimizer or patience >= config.get("patience", 5):
                break
        if not (directory / "best.pt").exists():
            torch.save({"binding": binding, "best_model": best_state}, directory / "best.pt")
        model.load_state_dict(best_state)
        saved = torch.load(directory / "best.pt", map_location=device, weights_only=False)
        clone = build_model(model_config).to(device).eval()
        clone.load_state_dict(saved["best_model"])
        example = next(iter(batches["test"]))
        model.eval()
        with torch.inference_mode():
            expected = model(**features(example, device))
            actual = clone(**features(example, device))
            if not torch.allclose(expected, actual, rtol=1e-6, atol=1e-6):
                raise ValueError("Checkpoint roundtrip prediction mismatch")
            no_text = features(example, device)
            no_text["text_mask"] = torch.zeros_like(no_text["text_mask"])
            no_text_pred = model(**no_text)
            if not torch.isfinite(no_text_pred).all():
                raise ValueError("No-text inference failed")
        result = evaluate(model, batches["test"], device, meta, directory / "test", 1 if smoke else None)
        write_json(directory / "history.json", history)
        with (directory / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        trace = {"sample_ids": example["sample_id"], "documents": example["text_metadata"],
                 "origin_time": example["origin_time"], "history_end_time": example["history_end_time"], "first_target_time": example["first_target_time"], "gfs_pool_ids": example["gfs_pool_ids"], "no_future_labels_in_model_features": sorted(FEATURES)}
        write_json(directory / "retrieval_trace.json", trace)
        summary = {"dataset": dataset_name, "horizon": horizon, "seed": config["seed"], "parameters": params,
                   "device": str(device), "amp": amp, "optimizer_steps": optimizer_steps,
                   "wall_seconds": time.perf_counter() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                   "checkpoint_roundtrip": True, "no_text_finite": True, "smoke_only": smoke,
                   "not_a_research_result": smoke, "test": result, "versions": versions()}
        write_json(directory / "summary.json", summary)
        seal(directory, {"kind": "smoke" if smoke else "experiment", "config_digest": digest(config), "binding": binding})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(run(config, args.output_dir, args.smoke, args.resume), indent=2))


if __name__ == "__main__":
    main()
