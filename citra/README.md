# CITRA 最小正确性原型

仅依赖 NumPy 和 Python 标准库，旧 ReTaMTSF 路径保持独立。

从仓库根目录运行：

```bash
/root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v
/root/miniconda3/envs/text4ts/bin/python run_citra_synthetic.py --documents 256 --seed 0
python scripts/audit_citra_data.py
```

其他环境中用已安装 NumPy 的 Python 替代上述绝对路径即可；不需要模型下载、GPU 或真实数据。

```python
from datetime import datetime, timezone
from citra import Document, TemporalDocumentStore

cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
store = TemporalDocumentStore(embedding_dim=2, impact_dim=2)
store.insert_document(Document(
    doc_id="example", version_id="v1", source="example", entity_id="series",
    issued_at=cutoff, available_at=cutoff, valid_from=cutoff, valid_to=None,
    event_start=None, event_end=None, embedding=[0.0, 0.1], impact_vector=[1.0, -1.0],
    availability_provenance="verified",  # 真实导入须保存并核实来源证据
))
exact = store.aggregate_exact([0.0, 0.0], cutoff)
result = store.query_certified([0.0, 0.0], cutoff, epsilon=0.1)
print(result.aggregate, result.aggregate_bound, result.stats)
```

默认 lipschitz=1 只认证聚合误差。预测输出证书须提供固定修正头的有效 Lipschitz 上界。
模拟可用时间必须标为 simulated 并显式设置 allow_simulated=True；未知时间默认排除。
同一 doc_id 修订使用 append_revision，新 cutoff 或修订会重建合法快照索引并计入冷查询成本。
当前 float64 实现提供解析截断误差界及明确数值容差验证，不提供形式化机器舍入认证。
固定 NumPy 预测头只用于证书/形状测试；PyTorch 训练集成尚未实现。

详细推导、数据审计、限制与真实运行结果见仓库 `docs/citra_*.md`。
