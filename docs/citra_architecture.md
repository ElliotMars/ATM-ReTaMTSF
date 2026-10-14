# 模块与数据流

新增代码与旧 ReTaMTSF 分离，旧 README、运行入口和模型未修改。

```
原始文档 + 有证据的时间字段 + 固定 embedding/impact
                    |
        Document -> TemporalDocumentStore (append-only)
                    |
              snapshot(cutoff)
                 /       \
     independent oracle  CITRAIndex (legal snapshot only)
                              |
             moments + best-first refinement + analytic bound
                              |
              A_approx, aggregate_bound, L*bound, stats
                              |
             FixedLinearForecaster -> [B,C,H] (验证用)
```

- `citra/temporal.py`：带来源标记、UTC 时间、不可写向量的 Document；插入、修订、历史快照、精确/认证查询。
- `citra/kernel.py`：RBF、文档坐标梯度、Hessian 上界、独立逐文档全求和 oracle。
- `citra/index.py`：中位数二叉空间分割、V/M/S 摘要、最大界优先细化、执行计数。
- `citra/prediction.py`：固定参数 NumPy 线性骨干和受控线性影响头；验证输出形状和误差放大。
- `citra/synthetic.py`：可复现实验种子、明确模拟日期、未来/迟到/修订和正负影响。
- `tests/test_citra.py`：时间语义、数学正确性、查询与输出证书测试。
- `run_citra_synthetic.py`：CPU 实验入口，oracle/filtered exact/无认证根摘要/CITRA 数值对照和统计。
- `scripts/audit_citra_data.py`：不加载预测依赖的本地 CSV 元数据检查。

依赖只新增对现有 NumPy 的使用，测试框架使用标准库 unittest；未安装包。
当前可用 `/root/miniconda3/envs/text4ts/bin/python` 的 NumPy 2.2.5；默认 Python 缺少 NumPy。

缓存只存一个 `(cutoff,generation)` 的索引。同一 cutoff 暖查询省去重建；更新或新 cutoff 使缓存失效。
索引构建包含向量读取和统计量计算，查询 stats 在冷查询中计入 `index_document_reads` 和构建耗时。
`metadata_records_checked` 记录时间快照筛选的全部版本，`documents_scanned` 记录查询阶段精确文档扫描，
两者不同；不能把暖查询扫描比例宣称为端到端数据库访问比例。
`summary_kernel_evaluations` 记录摘要中心核计算（当前每个非叶节点两次），exact_kernel_evaluations
记录逐文档核计算，summary_bytes 只含数值摘要，不含 Python 对象、元数据和文档存储。

当前是内存最小正确性原型，没有磁盘页统计、实体分区查询、持续增量矩维护、跨 cutoff 时态索引、
编码器版本持久化或 trainable PyTorch 模型。后续应在不改变算子语义的前提下补齐这些功能。
