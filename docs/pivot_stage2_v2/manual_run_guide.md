# 用户最终手动命令

当前工作区的数据、UTC映射、真实GFS缓存、模板、MiniLM、配置、依赖与单步验收已完成。请先接受 [时区解释](jena_timezone_decision.md)和 [可用性条件](data_and_asof_report.md)。用户不需要再下载、预处理、填写reviewed.json或安装依赖。

在仓库根目录执行：

```bash
cd /opt/data/private/ATM-ReTaMTSF
bash scripts/experiments_v2/validate_ready.sh --execute
bash scripts/experiments_v2/run_pilot.sh --execute --device cpu
bash scripts/experiments_v2/run_main.sh --execute --device cpu
bash scripts/experiments_v2/run_ablations.sh --execute --device cpu
bash scripts/experiments_v2/collect_results.sh --execute
```

Pilot已由本轮实际执行。再次执行时，代码、配置及数值/GFS数据绑定一致的成功验收会跳过；不是重复正式训练。主表默认63次、消融108次，正式种子2024/2025/2026。主表和消融须由用户自行启动，本轮没有启动。

脚本缺省为dry-run；预览不会联网或训练：

```bash
bash scripts/experiments_v2/run_main.sh --dry-run
bash scripts/experiments_v2/run_ablations.sh --dry-run
```

H=336已准备数据、覆盖审计及模型形状测试，未进入本轮核心真实pilot；如决定开展可选扩展，可在对应pilot/main/ablations命令追加 --include-h336。其12h策略通常只能覆盖前155个目标小时，不能描述成336小时完整外部预报覆盖。

默认使用已准备好的.venv-pivot CPU环境。CUDA分支未在当前机器验证；未来若自行提供兼容CUDA解释器，可通过PIVOT_PYTHON指定并选--device cuda，但不把这个切换作为开始实验的前置任务。

正式结果路径：artifacts/pivot_stage2/v2/experiments/H<h>/<variant>/seed<seed>/，含config.yaml、binding.json、best.pt、latest.pt、history.csv/json、retrieval_trace.json、summary.json、test下prediction/truth/mask.npy和origin_errors.npz。每次汇总创建独立collections/<version>/results.csv/json，latest_collection.json指向最新；无正式结果时状态为no_formal_results_yet。

中断后重跑原命令：成功且绑定一致的任务跳过；失败目录保留并从latest checkpoint恢复。若成功结果对应不同代码/数据/配置，拒绝覆盖，应保留原结果并另开实验版本，不择优重跑seed。准入清单绑定数值与GFS数据，发生源变更须重新验证，避免沿用旧通过状态。

prepare_all.sh --execute是完整自动复现入口，本轮已验证，无需再次执行。所有v2数据和结果在独立stage2_v2/v2目录，v1保持不变。更多模型与公平性细节见model_and_fairness.md，完整样本数/覆盖率见data_and_asof_report.md。
