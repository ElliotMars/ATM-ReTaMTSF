# 实际验收结果

以下是本轮真实执行的证据；正式实验结果数为 0，smoke 数字不能进入论文性能表。

## 测试

最终通过 **225 项，0 failure/error**，日志 [validation_tests.log](../../artifacts/pivot_stage2/v1/logs/validation_tests.log)，JUnit [validation_tests.xml](../../artifacts/pivot_stage2/v1/logs/validation_tests.xml)。其中既有测试 112 项、新增 Stage 2 测试 113 项，最终一次耗时约 10.20 秒。

覆盖 Dataset、不同 H 的 Sampler、全 L+H 窗口、仅训练 scaler、未来文档不可先评分、负延迟拒绝、随机合法文本、CTSR 自身排除/并列/缺失通道、Fusion 全 Mask 恒等、有效标签损失、NaN/Inf、零收益、Rank IC、块 Bootstrap、checkpoint 恢复和矩阵准入。参数化前向/反向覆盖 C={3,20,21,88} 与 H={1,3,5,24,72,168,336}、batch=1。测试禁止真实网络依赖。

首次 Stage 2 测试的 3 个失败来自测试文档时间超出 30 天候选窗口，已修正夹具并增加非空文本断言；原失败日志 stage2_tests_initial.* 保留。环境中原 PyTorch 导入缺少 iJIT_NotifyEvent，已在独立 .venv-pivot 中安装 CPU PyTorch，原环境未卸载；原始安装失败和修复日志均保留，没有把早期失败隐去。

## 真实数据审计与 smoke

已实际核查全部真实窗口、全部计划 H、固定通道顺序、训练 scaler、分割边界、源哈希与历史 N2 数据未改写。见 [data_acceptance.json](../../artifacts/pivot_stage2/v1/data_acceptance.json) 和 final_validation.log。日历 18 个休市日、5 个提前收盘日与 504 个真实 origin 已按官方通知核对。

最终 real pilot 共 **55 个配置**：StockNet 27（9 方法×3H）、CAISO 12（4 数值方法×3H）、Jena 16（4 数值方法×4H）。每个配置仅一个真实训练样本、一个验证样本、一个测试样本；45 个可训练模型各一次 optimizer step，10 个无参数 naive 不更新。全部使用 seed=2024，没有正式多种子训练。

每个配置已做 forward、可训练模型 backward/step、验证、最佳 checkpoint 保存/加载预测一致性、测试与强制无文本有限值推理。日志 [final_real_smoke.log](../../artifacts/pivot_stage2/v1/logs/final_real_smoke.log)，逐运行数据及参数、耗时、预测、Mask、文本追踪位于 artifacts/pivot_stage2/v1/smoke/，汇总 [smoke_summary.json](../../artifacts/pivot_stage2/v1/smoke_summary.json)。修复期间旧 smoke 版本保留于 retired_smoke，不参与任何正式汇总。

独立合成端到端 CLI 另实际执行 C=3/L=20/H=5 与 C=20/L=360/H=72，均一次 step、checkpoint roundtrip、无文本推理通过，见 synthetic_c3.log、synthetic_c20.log。合成文本只来自各自历史窗口，不能代替真实源验收。GFS numeric control 仅合成接口测试通过，真实 GFS 数据验收未通过。

## 入口与结果汇总

六个入口都已实际执行 bash -n 和 --dry-run，见 [script_checks.json](../../artifacts/pivot_stage2/v1/logs/script_checks.json) 与各 *_dry_run.log。main dry-run=81，ablations dry-run=99，pilot dry-run=55；dry-run 不创建训练任务、不联网。

实际执行 prepare_all --execute 的完整缓存重放、validate_ready --execute、run_pilot --execute、源/数据准备、collect_results --execute（空正式结果）。完整准备入口日志为 prepare_all_cached_replay.log。空汇总状态为 no_formal_results_yet，smoke_results_excluded=true，路径见 latest_collection.json。run_main/run_ablations 只执行过 dry-run，**没有执行 --execute**。

环境为 Python 3.10.0、Torch 2.5.1+cpu、Transformers 4.36.2；详见 environment.json 与 environment_freeze.txt。本轮 CUDA 不可用，仅 CPU 被实际测试；CUDA/AMP 分支已实现，未宣称 GPU 验收通过。用户现有环境不需要再安装依赖，脚本默认调用已准备好的 .venv-pivot。
