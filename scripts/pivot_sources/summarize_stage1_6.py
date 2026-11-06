#!/usr/bin/env python3
"""Generate a review report ONLY from user-produced, hash-checked bundles."""

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import xml.etree.ElementTree as ET

from scripts.pivot_sources.common import (
    CONFIG, ROOT, SourceError, digest, load_config, output_run, protected_output,
    read_bundle, read_json, seal, sha256, write_json,
)


def test_report(path):
    if path is None or not path.is_file():
        return {"status": "not_checked"}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    modules = ("test_asof", "test_nws_gfs", "test_stocknet", "test_http_outputs")
    seen = [name for name in modules if any(name in case.get("classname", "") for case in cases)]
    bad = sum(1 for case in cases if list(case.iter("failure")) or list(case.iter("error")) or list(case.iter("skipped")))
    errors = sum(int(suite.get("errors", 0)) for suite in root.iter("testsuite"))
    return {"status": "user_reported_pass" if cases and len(seen) == len(modules) and not bad and not errors else "blocked_or_incomplete",
            "case_count": len(cases), "modules_seen": seen, "bad_cases": bad, "suite_errors": errors,
            "sha256": sha256(path), "limitation": "user-supplied execution record, not source-tree attestation"}


def render(results, tests):
    lines = ["# PIVOT Stage 1.6 External Source Readiness", "",
             "由用户实际运行产物生成；原始返回内容、请求日志和解析产物可追溯。此报告不启动 Stage 2 或批准无限制下载。", "",
             "## 可核查事实", ""]
    for source, row in results.items():
        lines.extend([f"### {source}", "", f"输入目录：`{row['input_directory']}`；样本记录数：{row['sample_record_count']}。", "",
                      "```json", json.dumps(row["readiness"], ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## 推断和未知", "",
        "服务页面可访问不证明历史数据可用。来源时间加延迟属于实验假设，不能升级为逐条真实接收时间。小样本不能证明全时期覆盖。", "",
        "本地 quote/hash 只能检查用户审阅证据的字节绑定，不能代替对官方字段语义的人工审查。", "",
        "## 默认实验约束", "",
        "CAISO 固定 20 通道，Jena 固定 21 通道；原 N2 产物不变。新 Horizon 的统计不等于 Stage 2 Dataset。", "",
        "StockNet 固定 88 股票，不按推文数或预测效果删除目标；Log Return 只是候选，事后复权机制尚需审查。", "",
        "GFS 外部数值 Baseline 与文本 Baseline 必须共享完全相同的合法 Run、变量、格点和时间支持。", "",
        "## 测试记录", "", "```json", json.dumps(tests, ensure_ascii=False, indent=2), "```", "",
        "全时期状态：`not_yet_full_coverage_verified`。下一步只能按逐来源 Gate 结论补证据，不得把 not_checked 当 pass。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--nws-dir", type=Path, required=True)
    parser.add_argument("--gfs-dir", type=Path, required=True)
    parser.add_argument("--stocknet-dir", type=Path, required=True)
    parser.add_argument("--pytest-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        directories = {name: getattr(args, f"{name}_dir") for name in ("nws", "gfs", "stocknet")}
        if args.dry_run:
            print(json.dumps({"dry_run": True, "network": False, "inputs": {key: str(value) for key, value in directories.items()},
                              "output": str(args.output_dir), "checks_executed": False}))
            return 0
        if args.decision_report:
            protected_output(args.decision_report, directories.values())
            if args.decision_report.exists():
                raise SourceError("Decision report exists; choose a new path")
        tests = test_report(args.pytest_report)
        binding = {"stage": "1.6", "mode": "summary", "config_digest": digest(config)}
        results = {}
        with output_run(args.output_dir, binding, inputs=directories.values()) as output:
            for name, directory in directories.items():
                record = read_bundle(directory)
                if record.get("source") != name or record.get("config_digest") != digest(config):
                    raise SourceError("Source/config mismatch")
                inventory = read_json(directory / "source_inventory.json")
                decision = read_json(directory / "source_readiness.json")
                if decision["full_coverage_status"] != "not_yet_full_coverage_verified":
                    raise SourceError("A bounded probe cannot attest full coverage")
                if tests["status"] != "user_reported_pass":
                    decision["verified_t1_feasible"] = False
                    decision["conditional_t1_feasible"] = False
                    if decision["decision"] == "GO":
                        decision["decision"] = "CONDITIONAL"
                    decision["blocking_issues"].append("Stage 1.6 tests not reported passing")
                results[name] = {"input_directory": str(directory), "bundle_digest": digest(record),
                                 "sample_record_count": len(inventory["probe"].get("records", [])), "readiness": decision}
            write_json(output / "source_readiness.json", results)
            write_json(output / "stage1_6_brief_summary.json", {
                "sources": {name: row["readiness"] for name, row in results.items()}, "tests": tests,
                "full_coverage_status": "not_yet_full_coverage_verified", "stage2_started": False})
            markdown = render(results, tests)
            (output / "01_6_source_readiness_decision.md").write_text(markdown, encoding="utf-8")
            if args.decision_report:
                args.decision_report.parent.mkdir(parents=True, exist_ok=True)
                with args.decision_report.open("x", encoding="utf-8") as handle:
                    handle.write(markdown)
            seal(output, binding)
        print(json.dumps({"output": str(args.output_dir), "source_decisions": {name: row["readiness"]["decision"] for name, row in results.items()}}, indent=2))
        return 0
    except (SourceError, OSError, ValueError, KeyError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
