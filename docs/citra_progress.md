# 进度与交付边界

本轮执行 prompt.md 最后明确要求的 Milestone 1。Milestone 2/3/4 尚未完成。

## 已实现

新增独立 citra 包（temporal/kernel/index/prediction/synthetic）、23 项 unittest 测试、CPU 合成入口、
本地数据元数据检查脚本及六份研究文档。旧文件没有改写，用户 prompt.md 保留。
支持未知时间排除、UTC cutoff、半开有效区间、未来/迟到文档、历史版本、append-only 修订、
合法全集 oracle、V/M/S 摘要、RBF 梯度和保守 Hessian 界、最大界优先细化、零预算精确执行、
空库零向量，以及固定线性头的输出证书和 `[B,C,H]` 形状检查。
更新采用缓存失效和重建，保证正确性；不是高效增量索引算法。

## 实际验证与环境

- `/root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v`：23 项通过，首次运行 2.888 秒；
  test_certified_bound 含 1296 组参数对照。梯度同时用独立 Dual forward-mode 自动微分和有限差分核对。
- `/root/miniconda3/envs/text4ts/bin/python run_citra_synthetic.py --documents 256 --seed 0 --output docs/citra_smoke_results.json`：
  3 个 cutoff × 4 个 epsilon，共 12 组 CPU 查询；预算超限 0，输出偏差超过界加 1e-10 的情况 0。
  `python scripts/audit_citra_data.py --output docs/citra_local_data_audit.json`：四域均为 raw_data_missing。
- `python -m py_compile citra/*.py tests/test_citra.py scripts/audit_citra_data.py run_citra_synthetic.py`：语法检查通过。
- 默认 python 导入 NumPy 失败；现有 text4ts 为 Python 3.10.0 / NumPy 2.2.5。
- PyTorch 导入失败：`libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`。
  未修改环境、未安装新依赖；CPU NumPy 原型不受影响。PyTorch/GPU 集成未运行。
- 执行沙箱初始化报 `bwrap: Failed to make / slave: Permission denied`；命令使用批准的沙箱外执行。
- 本地 dataset/Time-MMD 为空，四域都没有实际可用性元数据可核实。

seed=0、cutoff=30 天、256 个可见文档的实际 smoke 结果如下（单位毫秒；暖查询默认重复 5 次）。
同一快照 filtered exact P50=6.642 ms。只代表本次机器上的小规模运行，不能宣称统计显著的加速。

|epsilon|暖查询精确扫描|返回输出界|实际输出偏差|冷查询延迟|暖查询 P50|
|---|---|---|---|---|---|
|0|256|0|0|21.913|6.687|
|0.01|256|0|0|25.507|10.199|
|0.1|224|0.0663833|0.00527677|24.949|9.217|
|1|0|0.974833|0.0545680|17.977|2.164|

严格预算下认证查询比简单精确求和更慢；宽松预算下这组暖查询减少了扫描，但冷构建仍有明显成本。
此结果支持继续研究成本权衡，尚不能证明算法在真实数据上有优势。

## 数学范围与研究判断

解析证书约束的是近似相对于完整合法文本的输出差异，仅在有效固定 Lipschitz 上界下成立。
float64 实现以预先声明的数值容差核对，没有形式化 IEEE 舍入误差证明。
大半径/抵消影响会产生很松的界，可能全扫描；新 cutoff 重建可能让端到端执行更慢。
目前具备可复核的研究原型，尚无实质性能/预测增量证据。经典泰勒展开不属于原创贡献。

## 下一阶段最高优先级

先修复/确认可用 PyTorch 环境，同时设计可跨 cutoff 复用且正确处理版本与迟到数据的时间索引，
为真实增量摘要维护增加对照测试。统一四种系统基线和成本统计，测不同 cutoff 的端到端冷/暖性能。
随后新增独立 trainable models/CITRA.py 和公平预测训练；最后核实并导入真实历史文本与观测。
真实数据下载/导入、真实预测 MSE/MAE、GPU、持久化和大规模评估均未完成。
