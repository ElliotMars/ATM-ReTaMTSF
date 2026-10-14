# 评估协议与可运行实验

## 当前 CPU 实验

```bash
/root/miniconda3/envs/text4ts/bin/python -m unittest discover -s tests -v
/root/miniconda3/envs/text4ts/bin/python run_citra_synthetic.py --documents 256 --seed 0 --output docs/citra_smoke_results.json
python scripts/audit_citra_data.py --output docs/citra_local_data_audit.json
```

合成测试证书矩阵：4 个种子，embedding 维度 1/3/16，leaf_size 1/4/16，radius 1e-6/0.05/100，
cutoff 天数 0/10/30，epsilon 0/1e-5/0.05/100，总计 1296 组。
影响维度 4，包含负影响、修订、迟到和未来文档；另有重复文本/抵消影响/零半径测试。
FixedLinearForecaster 参数固定，Full Text 和 CITRA 使用同一模型。相对于 oracle 的聚合偏差、
预测偏差、理论界有效性分别验证，不能互相替代。

实验入口对以下执行方式提供原型对照：全版本时间过滤+完整求和（Full Scan）、已筛选快照精确求和、
无认证根摘要、认证自适应查询。前两者合法文本语义相同。无认证根摘要仅作为粗精度参照，
当前没有完整四基线吞吐量评估，也不宣称实现了可比的统一系统 benchmark。
全扫描计时包含快照筛选；filtered exact 的暖计时不含准备过程，准备成本另列。
认证查询分别报告冷启动（含快照和索引构建）、同一 cutoff 暖查询的耗时与读计数。
offline oracle 对照在查询计时外单独执行，绝不用于在线提前终止或修正界。

JSON 包含预测偏差/界、tightness（error/bound，零界为 null）、P50/P95、扫描比例、
摘要字节数、构建时间、文档读取量和插入总耗时。5 次默认重复只用于 smoke，不能支持稳定性能结论。
摘要字节数不是完整索引内存；插入计时不是动态索引节点维护性能。当前不报告 MSE/MAE，因为没有
训练、预测目标或真实观测标签；禁止将近似误差指标改名为预测准确率。

## 后续实验矩阵（尚未运行）

1. 系统：统一时间过滤/缓存/冷暖执行协议；Exact Full Scan、Temporal Filter+Exact、
   无认证分层近似、CITRA；逐 cutoff 查询而非只重复同一历史快照。
2. 扩展：文档数量、维度、时间选择率、修订率、半径、正负抵消强度、epsilon；
   多随机种子、多次计时与 P50/P95，完整索引内存、构建和真实节点更新开销。
3. 预测：TS-only、Semantic Top-K、Full As-of、CITRA；相同骨干、训练区间、预算与种子。
   Full As-of/CITRA 共享固定参数；Top-K 只是有限候选近似，不能作为 full oracle。
4. 输出分别报告未来标签 MSE/MAE、Full/CITRA 输出偏差、证书有效率、tightness 和系统性能。
5. 真实数据：核实 NWS AFD 历史版本和首次可用时间，ASOS 观测发布延迟，MOS 可选；
   按时间连续划分 train/validation/test，预处理仅拟合 train，保留版本和来源 manifest。

没有本地真实数据；尚未实现下载/真实数据导入适配器。FAISS 不作为必需依赖，后续比较时必须说明
Top-K 与完整核求和的语义差异。GPU/PyTorch 集成目前受环境错误影响，未运行，不能记为通过。
