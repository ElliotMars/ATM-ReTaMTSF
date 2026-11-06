# 实际执行的验收

最终pytest **324 passed，0 failure/error**，JUnit时间 15.41s。既有225项和新增99项均通过。证据：logs/validation_tests.log、validation_tests.xml、final_acceptance.log。

56个参数化转换例覆盖2019—2025冬/夏/三月/十月边界，额外测试Berlin不明时刻；真实来源映射另有53例。覆盖全部H（含可选336）、C=21、batch=1、全数值缺失通道、无文本、缺GFS、未来站点y变化不影响查询或外部值、3/6/12h权限、名义day7晚端Mask、候选全集相同、Raw缺预报恒等、模型forward/backward、单step和checkpoint恢复。特别构造位于真实origin与first_target之间才可用的预报，验证它不会因错误截点被放行。

## 真实数据

全部实际窗口、数值与Mask对应、训练scaler、原生连续段、UTC映射、跨Split标签不重叠、原始响应与产物哈希已穷尽检查。原始全期映射604,311行；默认N2全期选定行UTC单调唯一。原始异常保留为诊断，不伪造修复。

| Split/季节 | history_end UTC | forecast_origin UTC | first_target UTC | 合法事实 | 合法目标小时(H168) |
|---|---|---|---|---|---|
| train | 2022-01-15T11:10:00+00:00 | 2022-01-15T11:20:00+00:00 | 2022-01-15T12:10:00+00:00 | 581 | 155 |
| validation | 2024-07-15T11:10:00+00:00 | 2024-07-15T11:20:00+00:00 | 2024-07-15T12:10:00+00:00 | 581 | 155 |
| test | 2025-03-30T11:10:00+00:00 | 2025-03-30T11:20:00+00:00 | 2025-03-30T12:10:00+00:00 | 581 | 155 |

这三个真实起点分别承担训练、验证和测试的单样本验收。真实缺GFS验证样本也保留并完成有限值推理，见real_no_GFS_inference.json。三季节文档例见gfs_document_examples.jsonl，来自真实API预测而非未来Jena真值。

当前主表pilot **21个配置**（7方法×3H），仅seed2024，每个恰好1 optimizer step、1个训练样本、1个验证样本、1个测试样本。检查点回读预测一致、强制无文本推理全部通过。seasonal_model_probes.json另记录同一已训练checkpoint在冬季、夏季、DST附近三个真实起点的实际前向结果与物理时刻。日志real_pilot_accepted.log，summary与预测在smoke/；旧修复期间版本保留retired_smoke/。smoke误差不构成正式性能表。

六个Shell入口均通过bash -n与--dry-run：main=63、ablations=108、pilot=21；没有对main/ablations使用--execute。prepare_all --execute完整缓存重放已通过；collect --execute空正式结果已验证排除smoke。正式结果数0，无多种子正式训练、无消融训练。

v1交付的238个源码/配置/测试/文档文件哈希逐一保持一致，StockNet、CAISO、Jena三个v1数据Bundle哈希通过。N2原值/Mask完全一致，既有历史结果未修改。没有由本代理执行git commit/push/merge。

当前Python 3.10.0、Torch 2.5.1+cpu，只实际验证CPU；CUDA/AMP已实现但未做GPU验收。不需要用户安装依赖或重做数据准备。环境明细environment.json、environment_freeze.txt；最终绑定delivery_verification.json与delivery_manifest.json。
