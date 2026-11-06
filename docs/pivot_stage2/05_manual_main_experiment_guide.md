# 用户最终操作手册

数据、embedding、依赖、YAML 与 smoke 已完成。本工作区的脚本默认使用 `.venv-pivot/bin/python`，无需再填写审核 JSON、手工下载或预处理。开始前请阅读 [数据决定及限制](00_final_dataset_decision.md)：默认 StockNet 是 88 维数组、87 通道有监督的 CONDITIONAL 方案，采用明确的历史可用性假设；不应声称严格历史接收时间已验证。

在仓库根目录运行。下面环境检查已经由本轮执行；用户可再次执行确认当前目录产物完整，然后自行启动正式训练：

```bash
cd /opt/data/private/ATM-ReTaMTSF
bash scripts/experiments/validate_ready.sh --execute
bash scripts/experiments/run_main.sh --execute --device cpu
bash scripts/experiments/run_ablations.sh --execute --device cpu
bash scripts/experiments/collect_results.sh --execute
```

主实验 81 次、消融 99 次，按固定 3 个种子串行运行，验证集早停。当前已验证的是 CPU 环境，以上命令可直接使用。没有 GPU 训练时长保证。若将来在另行准备并核实的 CUDA Python 环境运行，可通过 PIVOT_PYTHON 指定解释器并选择 --device cuda；当前交付未实际验证该环境，不把切换 CUDA 作为开始前置任务。

六个脚本缺省都是 dry-run。显式预览示例：

```bash
bash scripts/experiments/run_main.sh --dry-run
bash scripts/experiments/run_ablations.sh --dry-run
```

默认正式清单只含 StockNet，来源于 `artifacts/pivot_stage2/v1/admission.json`。若希望做保留的天气**数值**控制组，可显式执行（本轮未启动）：

```bash
bash scripts/experiments/run_main.sh --execute --include-numerical-controls --dataset caiso --device cpu
bash scripts/experiments/run_main.sh --execute --include-numerical-controls --dataset jena --device cpu
```

这些开关不会启用被关闭的 NWS/GFS 文本。天气评估数组较大，采用磁盘 memmap 保存原单位预测和真值，建议先确认存储容量；它们不是默认主实验必需步骤。

`prepare_all.sh --execute` 是自动复现入口，本轮前置工作已经完成，不需要用户再运行。它按缓存、不可变数据 Manifest 和源哈希复用完成产物，自动获取所需公开证据，不要求 reviewed.json。`run_pilot.sh --execute --device cpu` 是已执行过的单步工程验收，不是正式实验。

正式结果保存在 `artifacts/pivot_stage2/v1/experiments/<dataset>/H<h>/<variant>/seed<seed>/`，含 config.yaml、latest.pt、best.pt、history.json/csv、summary.json、retrieval_trace.json、test/{prediction,truth,mask}.npy、test/origin_errors.npz、样本来源与 _SUCCESS 哈希。汇总只读 experiments，排除 smoke/retired_smoke。

中断后重跑同一命令：成功且配置/代码哈希一致的运行会跳过；未完成目录保留为 failed 版本并从 latest checkpoint 恢复。成功结果与不同配置/代码发生冲突时拒绝覆盖，需保留旧版本并使用新实验版本。不得通过选择性重跑 seed 或修改测试样本来改善主表。

每次 collect 生成不可变 collections/<时间_随机后缀>/results.csv 和 results.json，`latest_collection.json` 指向最新表；尚无正式结果时明确标记 no_formal_results_yet。表中报告种子均值/标准差及时间块区间，不能将本轮 smoke 的误差或耗时解释为正式预测性能。

最终交付不包含任何自动 push、merge 或旧实验结果修改。上述正式主表和消融必须由用户手动启动；本轮未启动它们。
