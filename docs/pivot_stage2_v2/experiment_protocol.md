# 冻结实验方案

默认 Jena 21 通道，L=360 小时，核心 H={24,72,168}，H=336仅通过 --include-h336 可选加入。按站点固定CET日历，Train=2021-04-01至2023-12-31，Val=2024全年，Test=2025-01-01至2025-06-30。分割边界先按选定物理时区解释再转UTC，例如Val从2023-12-31T23:00Z开始，不将本地日期字符串直接当UTC。

| Split | 点数 | 原标签首 | 原标签尾 | 实际UTC首 | 实际UTC尾 |
|---|---|---|---|---|---|
| train | 24106 | 2021-04-01 00:10:00 | 2023-12-31 23:10:00 | 2021-03-31 23:10:00+00:00 | 2023-12-31 22:10:00+00:00 |
| validation | 8784 | 2024-01-01 00:10:00 | 2024-12-31 23:10:00 | 2023-12-31 23:10:00+00:00 | 2024-12-31 22:10:00+00:00 |
| test | 4344 | 2025-01-01 00:10:00 | 2025-06-30 23:10:00 | 2024-12-31 23:10:00+00:00 | 2025-06-30 22:10:00+00:00 |

每个Split独立提供360小时预热，不借用前一Split历史。原N2数值和Mask逐格保留；新的回归scaler仅拟合本轮Train有效观测。窗口必须完整覆盖L+H、1小时步长、同一原生连续段，且有有效历史和标签；缺失通道由Mask处理，不减少维度。标签不填补。

主表7种：Numerical Only、GFS Raw Numerical、Text Mean Fusion、Latest Eligible、Semantic、Semantic+Temporal、Full PIVOT。Full使用CTSR；简单文本基线关闭CTSR以明确区分通道模块；Numerical Only与Raw控制保留数值骨干CTSR，without_ctsr另隔离该因素。

12个消融配置：without_ctsr、without_str（全池均值）、without_temporal、random_eligible、text_missing、topk1、topk8、timezone_cet_common、timezone_berlin、timezone_utc、lag3、lag6。默认latency=12h作为对照。时区敏感性三组只使用三种解释都有效的同一原始行窗口交集，保持数值输入/目标完全相同；与全量主表分别汇总，不能混合不同样本指纹。Berlin不明时刻及跨DST步长变化窗口排除并报告，不silent infer。时区敏感性只用于稳健性报告，不以测试MSE重选主时区；without_temporal只关闭检索时间分数，仍保留合法目标时刻/提前量字段。

固定正式种子2024、2025、2026；默认主表63次，消融108次。76个YAML覆盖4个H×19个方法/消融，默认脚本只运行3个核心H。预算AdamW、lr=0.001、weight_decay=0.0001、batch=8、最多30epochs、patience=5、dim=32、1层4头、patch=16/stride=8、dropout=0.1。没有本轮超参搜索。

验证标准化MSE选择best checkpoint；模型选择结束后才完整评估Test。报告标准化和原单位MSE/MAE、每通道误差、有效标签覆盖、参数量、训练/推理时长及峰值显存；21个通道量纲不同，标准化指标是主比较。保存预测、真值、Mask、起点误差、配置、模型/优化器/AMP/RNG恢复状态、源与代码哈希。

汇总只读正式experiments目录，排除smoke/retired_smoke，拒绝重复dataset/H/scope/method/seed和不同样本指纹。报告种子均值与样本标准差，使用块长至少H的预测起点顺序块Bootstrap；时区敏感性按原始naive历史行配对，UTC外部检索时刻另存。时间块区间是有限样本近似，跨被排除窗口的物理跨度可能更长，应结合原点序列检查；不将随机种子方差与时间相关性视为同一来源。

本轮仅运行单batch/单step验收，没有启动上述正式预算、正式多种子或消融训练。整体准入仍为CONDITIONAL：时区有强来源支持，但GFS固定提前量参考+延迟和观测end+10min是明确假设，未验证历史实际送达。所有科研效果待用户手动正式实验后判断。
