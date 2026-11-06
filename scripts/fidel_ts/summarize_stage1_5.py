#!/usr/bin/env python3
"""Compare N1/N2 and produce distinct numerical/archived/as-of readiness decisions."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    directory = Path(__file__).resolve().parent
    sys.path[:] = [str(directory.parents[1])] + [p for p in sys.path if Path(p or ".").resolve() != directory]

import argparse
from collections import Counter
import json
import xml.etree.ElementTree as ET

import pandas as pd

from scripts.fidel_ts.common import AuditError, ROOT, assert_identity, json_digest, load_source, read_json, sha256_file, write_json
from scripts.fidel_ts.numerical_protocols import PROTOCOL_DIRS
from scripts.fidel_ts.stage1_5_io import (
    PROTOCOL_CONFIG, SPLITS, finish_output, load_protocol, new_output, read_bundle,
)


def pytest_status(path):
    if not path.is_file():
        return {"status": "unverified", "reason": "JUnit report not provided"}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    required = ("test_stage1_5_numerical", "test_stage1_5_provenance", "test_stage1_5_outputs")
    covered = [name for name in required if any(name in case.get("classname", "") for case in cases)]
    failures = sum(len(list(case.iter("failure"))) for case in cases)
    case_errors = sum(len(list(case.iter("error"))) for case in cases)
    skipped = sum(len(list(case.iter("skipped"))) for case in cases)
    # Collection/import errors can exist without a testcase.
    suite_errors = sum(int(suite.get("errors", 0)) for suite in root.iter("testsuite"))
    errors = max(case_errors, suite_errors)
    passed = bool(cases) and not failures and not skipped and not errors and len(covered) == len(required)
    return {"status": "reported_pass" if passed else "blocked_or_incomplete", "testcases": len(cases),
            "failures_or_errors": failures + errors, "skipped": skipped,
            "stage1_5_test_modules_seen": covered, "junit_sha256": sha256_file(path),
            "limitation": "user-supplied execution record; not an independent attestation of the executed source tree"}


def compare_prepared(n1_dir, n2_dir, output, config):
    q1 = read_json(n1_dir / "quality_summary.json")
    q2 = read_json(n2_dir / "quality_summary.json")
    if "splits" not in q1 or "splits" not in q2:
        return {"status": "blocked", "reason": "A numerical protocol could not be prepared",
                "n1_blocked_reasons": q1.get("blocked_reasons", []), "n2_blocked_reasons": q2.get("blocked_reasons", [])}
    w1, w2 = read_json(n1_dir / "window_eligibility.json"), read_json(n2_dir / "window_eligibility.json")
    comparisons, timestamp_rows = {}, []
    for split in SPLITS:
        p1, p2 = n1_dir / split / "timestamps.parquet", n2_dir / split / "timestamps.parquet"
        if not p1.is_file() or not p2.is_file():
            raise AuditError("Comparison requires --mode prepare outputs, not report-only outputs")
        t1, t2 = pd.read_parquet(p1), pd.read_parquet(p2)
        counts1, counts2 = Counter(t1["timestamp"].tolist()), Counter(t2["timestamp"].tolist())
        keys1, keys2 = set(counts1), set(counts2)
        for timestamp in sorted(keys1 | keys2):
            timestamp_rows.append({"split": split, "timestamp": timestamp,
                                   "n1_occurrences": counts1[timestamp], "n2_occurrences": counts2[timestamp]})
        horizons = {}
        for horizon in config["horizons"]:
            a, b = w1[split][str(horizon)], w2[split][str(horizon)]
            i1 = pd.read_parquet(n1_dir / "window_indices" / f"{split}_H{horizon}.parquet")
            i2 = pd.read_parquet(n2_dir / "window_indices" / f"{split}_H{horizon}.parquet")
            origins1, origins2 = set(i1["forecast_origin"]), set(i2["forecast_origin"])
            continuous1 = set(i1.loc[i1["continuous"], "forecast_origin"])
            ratio = b["published_window_indices"] / a["theoretical_windows"] if a["theoretical_windows"] else None
            horizons[str(horizon)] = {
                "n1": a, "n2": b, "n2_to_n1_theoretical_window_retention": ratio,
                "significant_reduction_warning": ratio is not None and ratio < config["n2"]["significant_window_retention_warning_below"],
                "shared_origins": len(origins1 & origins2), "n1_only_origins": len(origins1 - origins2),
                "n2_only_origins": len(origins2 - origins1),
                "shared_continuous_hourly_window_origins": len(continuous1 & origins2),
                "origin_overlap_caveat": "unfiltered shared origins alone do not imply identical N1 histories/labels"}
        comparisons[split] = {"n1": q1["splits"][split], "n2": q2["splits"][split],
                              "shared_sampled_timestamps": len(keys1 & keys2),
                              "n1_only_timestamps": len(keys1 - keys2), "n2_only_timestamps": len(keys2 - keys1),
                              "horizons": horizons}
    pd.DataFrame(timestamp_rows).to_parquet(output / "downsampled_timestamp_sets.parquet", index=False)
    return {"status": "compared", "channels": q2["channels"], "splits": comparisons,
            "n1_native_scaler_counts": q1["scaler_fit_counts"], "n2_native_valid_scaler_counts": q2["scaler_fit_counts"],
            "n2_hour_anchor": q2["hour_anchor"], "n2_blocked_reasons": q2["blocked_reasons"],
            "n1_compatibility": "audited operation order implemented, runtime versions recorded; no claim of reproduced paper metrics",
            "time_processing_differences": "stable sort, duplicate exclusion, one training phase, no native-gap crossing",
            "value_processing_differences": "invalid sentinels masked in N2; N1 retains them in its scaler and values",
            "n2_undoes_upstream_interpolation": False}


def readiness_decision(comparison, temporal, tests):
    reasons = list(comparison.get("n2_blocked_reasons", []))
    if comparison.get("status") != "compared":
        reasons.append("Numerical protocol comparison unavailable")
    if tests["status"] != "reported_pass":
        reasons.append("Stage 1.5 offline tests are not reported fully passing")
    return {
        "numerical_mimo_readiness": {"status": "blocked" if reasons else "candidate_for_stage2_review",
                                     "reasons": reasons, "default_protocol": "N2",
                                     "N1_role": "official_compatible_comparability",
                                     "scope": "numerical artifacts only, no MIMO Dataset or model implemented"},
        "archived_text_research_readiness": {"status": "auxiliary_only" if temporal.get("facets", temporal["reports"]) else "blocked_no_textual_facets",
                                             "T2": "past_source_key_only_not_verified_availability",
                                             "T3": "retrospective_oracle_risk_diagnostics_only"},
        "verified_asof_multimodal_readiness": {"status": "blocked_not_verifiable", "T1": temporal["T1"],
                                               "author_contact_required": False,
                                               "next_source_requirement": "reconstruct an external text archive with trustworthy issuance and version records if needed"},
        "remaining_scientific_limitations": ["no verified historical text issuance or forecast validity",
                                             "naive clocks and unresolved timezone semantics",
                                             "published upstream interpolation cannot be undone by N2",
                                             "extreme finite CAISO values are retained and require scientific review, not blanket negative-value removal",
                                             "N1 and N2 scores must be reported as different protocols",
                                             "all methods must share all channels, window identities, scaler and y_mask"]}


def render_decision(results, tests):
    lines = ["# PIVOT Stage 1.5 Dataset Readiness Decision", "",
             "Generated from user-executed, hash-checked artifacts. This report does not approve model training.", "",
             "## A. 可验证事实", "",
             "来源为各数据集的 Stage 1.5 manifest、numerical diagnostics、N1/N2 quality/window reports 与 text diagnostics；未联系作者。", ""]
    for dataset, record in results.items():
        facts = record["numerical_diagnostics"]
        lines.extend([f"### {dataset}", "",
                      f"原始行数：{facts['rows']}；通道数：{facts['numeric_channel_count']}；额外重复记录：{facts['extra_duplicate_records']}；重复组：{facts['duplicate_groups']}；唯一时间轴缺口：{facts['unique_gap_count']}。", "",
                      f"报告数：{record['text_structure']['reports']}；Facet 数：{record['text_structure']['facets']}；T1：{record['text_temporal']['T1']}。", ""])
    lines.extend(["## B. 基于证据的推断", "",
                  "观测式字段、趋势文字与实况接近只提示调查风险，不能单独证明事后生成或不存在泄漏。N2 与 N1 的相位/窗口差异见 downsampling_comparison.json。", "",
                  "## C. 无法验证", "",
                  "每条报告的历史发行时间、未保存的预报更新版本、缺少生成脚本的文本生成方式，以及无时区字段的真实物理时间轴，不能由人工假设补齐。", "",
                  "## D. 默认决策", "",
                  "N1 仅用于官方可比性；N2 是全通道候选主数值协议。T2 仅用于历史档案辅助研究；T3 仅用于回溯对齐与潜在泄漏诊断；T1 在缺少发行/版本证据时阻断。", "",
                  "Temporal Relevance 不得伪造 valid 区间；报告时间键距离只能另行命名为探索性评分。Utility 不能绕过可用性要求。", "",
                  "## 分项 Readiness", "", "```json",
                  json.dumps({name: record["decision"] for name, record in results.items()}, ensure_ascii=False, indent=2),
                  "```", "", "## 测试执行记录", "", "```json", json.dumps(tests, ensure_ascii=False, indent=2), "```", "",
                  "Stage 2 不自动执行；数值产物进入 review 不代表已实现或验证 MIMO 模型。没有笼统的 ready:true。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", type=Path, default=ROOT / "artifacts/fidel_ts/stage1_5")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "dataset/Fidel-TS/processed")
    parser.add_argument("--caiso-text-dir", type=Path)
    parser.add_argument("--jena-text-dir", type=Path)
    parser.add_argument("--pytest-report", type=Path, default=ROOT / "artifacts/fidel_ts/stage1_5/tests.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-report", type=Path, help="Optional new Markdown path; existing files are never overwritten")
    parser.add_argument("--config", type=Path, default=PROTOCOL_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_protocol(args.config)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "diagnostics_root": str(args.diagnostics_root),
                              "processed_root": str(args.processed_root), "output": str(args.output_dir),
                              "decision_report": str(args.decision_report), "data_checks_executed": False}, indent=2))
            return 0
        if args.decision_report and args.decision_report.exists():
            raise AuditError("Decision report already exists; choose a new report path")
        tests = pytest_status(args.pytest_report)
        results = {}
        input_dirs = [args.processed_root, ROOT / "dataset/Fidel-TS/raw", ROOT / "artifacts/fidel_ts/data_audit"]
        for name in ("caiso", "jena"):
            input_dirs.extend([args.diagnostics_root / name / "numerical",
                               getattr(args, f"{name}_text_dir") or args.diagnostics_root / name / "text"])
        if args.decision_report and any(args.decision_report.resolve().is_relative_to(Path(path).resolve()) for path in input_dirs):
            raise AuditError("Decision report must be outside raw data, processed data and existing diagnostic bundles")
        with new_output(args.output_dir, input_dirs) as output:
            for dataset in ("caiso", "jena"):
                spec = load_source(dataset)
                base = args.diagnostics_root / dataset
                numerical = base / "numerical"
                text = getattr(args, f"{dataset}_text_dir") or base / "text"
                n1 = args.processed_root / dataset / PROTOCOL_DIRS["n1"]
                n2 = args.processed_root / dataset / PROTOCOL_DIRS["n2"]
                bindings = [read_bundle(path, allow_blocked=True) for path in (numerical, text, n1, n2)]
                for binding in bindings:
                    assert_identity(binding, dataset, spec)
                    if binding["protocol_config_digest"] != json_digest(config):
                        raise AuditError("Protocol config changed between runs")
                    if binding["source_config_digest"] != json_digest(spec):
                        raise AuditError("Source/channel configuration changed between runs")
                    if binding["primary_sha256"] != bindings[0]["primary_sha256"]:
                        raise AuditError("Inconsistent primary source across diagnostic/preparation runs")
                target = output / dataset
                target.mkdir()
                comparison = compare_prepared(n1, n2, target, config)
                temporal = read_json(text / "text_temporal_analysis.json")
                decision = readiness_decision(comparison, temporal, tests)
                results[dataset] = {"numerical_diagnostics": read_json(numerical / "numerical_diagnostics.json"),
                                    "text_structure": read_json(text / "text_report_structure.json"),
                                    "text_temporal": temporal, "comparison": comparison, "decision": decision,
                                    "input_artifact_directories": [str(p) for p in (numerical, text, n1, n2)]}
                write_json(target / "downsampling_comparison.json", comparison)
                write_json(target / "dataset_readiness.json", decision)
            compact = {name: record["decision"] for name, record in results.items()}
            write_json(output / "dataset_readiness.json", {"datasets": compact, "tests": tests})
            write_json(output / "brief_summary.json", {name: {
                "numerical_mimo": record["decision"]["numerical_mimo_readiness"]["status"],
                "archived_text": record["decision"]["archived_text_research_readiness"]["status"],
                "verified_asof": record["decision"]["verified_asof_multimodal_readiness"]["status"],
                "channels": record["numerical_diagnostics"]["numeric_channel_count"]} for name, record in results.items()})
            markdown = render_decision(results, tests)
            (output / "dataset_readiness_decision.md").write_text(markdown, encoding="utf-8")
            if args.decision_report:
                args.decision_report.parent.mkdir(parents=True, exist_ok=True)
                with args.decision_report.open("x", encoding="utf-8") as handle:
                    handle.write(markdown)
            finish_output(output, {"schema_version": 1, "scope": "stage1_5_decision", "protocol_config_digest": json_digest(config)})
        print(json.dumps({"output": str(args.output_dir), "decision_report": str(args.decision_report),
                          "verified_asof_multimodal": "blocked_not_verifiable"}, indent=2))
        return 0
    except (AuditError, OSError, KeyError, TypeError, ValueError, ImportError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
