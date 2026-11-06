"""Only --execute starts work. Formal experiments are never run by preparation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

from .common import ARTIFACTS, DATA, ROOT, TASKS, digest, read_json, verify, write_json
from .configurations import MAIN, ABLATIONS, configuration, generate


def plans(action, include_numeric=False, dataset=None):
    if action not in {"main", "ablations", "pilot"}:
        return [{"action": action, "formal_training": False}]
    admission_path = ARTIFACTS / "admission.json"
    admission = read_json(admission_path)["datasets"] if admission_path.exists() else {}
    selected = []
    for name in TASKS:
        if dataset and name != dataset:
            continue
        allowed = admission.get(name, {}).get("main_enabled", False)
        numeric = admission.get(name, {}).get("numeric_enabled", False)
        if not allowed and not (numeric and (include_numeric or action == "pilot")):
            continue
        variants = ABLATIONS if action == "ablations" else MAIN
        if not allowed:
            variants = [value for value in variants if value in {"naive", "dlinear", "patchtst", "backbone", "text_missing", "without_fusion"}]
        seeds = [2024] if action == "pilot" else [2024, 2025, 2026]
        for horizon in TASKS[name][1]:
            for variant in variants:
                for seed in seeds:
                    selected.append({"dataset": name, "horizon": horizon, "variant": variant, "seed": seed,
                                     "smoke": action == "pilot", "formal_training": action != "pilot"})
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "pilot", "main", "ablations", "collect"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-numerical-controls", action="store_true")
    parser.add_argument("--dataset", choices=list(TASKS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    selected = plans(args.action, args.include_numerical_controls, args.dataset)
    if not args.execute:
        print(json.dumps({"dry_run": True, "action": args.action, "network": False, "jobs": selected,
                          "formal_jobs": sum(item.get("formal_training", False) for item in selected)}, indent=2))
        return
    if args.action == "prepare":
        for module in ("fetch", "prepare", "finalize_sources", "audit", "configurations"):
            subprocess.run([sys.executable, "-m", "pivot_stage2." + module], cwd=ROOT, check=True)
        return
    if args.action == "validate":
        subprocess.run([sys.executable, "-m", "pivot_stage2.audit"], cwd=ROOT, check=True)
        log = ARTIFACTS / "logs/validation_tests.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/fidel_ts", "tests/pivot_sources", "tests/pivot_stage2",
                            "--junitxml=" + str(ARTIFACTS / "logs/validation_tests.xml")], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(str(log))
        return
    if args.action == "collect":
        from .collect import collect
        print(collect())
        return
    from .train import CODE_BINDING, run
    if not selected:
        raise SystemExit("No admitted experiment combinations; inspect admission.json. No jobs started.")
    summaries = []
    for item in selected:
        path = ROOT / "configs/pivot_stage2" / item["dataset"] / f"H{item['horizon']}_{item['variant']}.yaml"
        config = yaml.safe_load(path.read_text()) if path.exists() else configuration(item["dataset"], item["horizon"], item["variant"], item["seed"])
        config["seed"] = item["seed"]
        config["device"] = args.device
        directory = ARTIFACTS / ("smoke" if args.action == "pilot" else "experiments") / item["dataset"] / f"H{item['horizon']}" / item["variant"] / f"seed{item['seed']}"
        if (directory / "_SUCCESS.json").exists():
            record = verify(directory)
            if record.get("binding", {}).get("code_sha256") == CODE_BINDING and record["config_digest"] == digest(config):
                summaries.append(read_json(directory / "summary.json"))
                continue
            if args.action != "pilot":
                raise ValueError("Successful formal run uses different code/config; preserve it and select a new experiment version")
            retired = ARTIFACTS / "retired_smoke" / str(time.time_ns()) / item["dataset"] / f"H{item['horizon']}" / item["variant"]
            retired.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(retired)
        resume = None
        if directory.exists():
            number = 1
            while directory.with_name(directory.name + f"_failed_{number}").exists():
                number += 1
            backup = directory.with_name(directory.name + f"_failed_{number}")
            directory.rename(backup)
            if (backup / "latest.pt").exists():
                resume = backup / "latest.pt"
        print("Running", item, flush=True)
        summaries.append(run(config, directory, smoke=item["smoke"], resume=resume))
    if args.action == "pilot":
        write_json(ARTIFACTS / "smoke_summary.json", {"runs": summaries, "formal_training": False,
            "each_trainable_run_optimizer_steps": 1, "seeds": [2024], "not_research_results": True})


if __name__ == "__main__":
    main()
