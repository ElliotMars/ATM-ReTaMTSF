# 冻结实验方案

本方案在正式结果产生前固定。默认正式任务只读取机器准入清单中的 StockNet，CAISO/Jena 仅通过显式数值控制组选项加入。不得通过某个方法的测试表现改变样本或准入。

## 数值任务

| 数据 | L | H | C | 分割 | 默认状态 |
|---|---|---|---|---|---|
| StockNet | 20 个交易日 | 1、3、5 | 88，实际 87 个可监督 | Train 2014-01—2015-06；Val 2015-07—09；Test 2015-10—12 | 有条件主实验 |
| CAISO N2 | 360 小时 | 24、72、168 | 20，含全部原始预测类通道 | Train <2021；Val 2021；Test ≥2022 | 可选数值控制 |
| Jena N2 | 360 小时 | 24、72、168、336 | 21 | Train <2021；Val 2021；Test ≥2022 | 可选数值控制 |

所有方法对同一数据和 H 使用相同的 starts、x、y 和 y_mask。文本缺失不得删样本。每个 Split 内检查完整 L+H 的时间连续性，Jena 不能跨原生断段；不以填补后的标签替代真实标签。数值输入只在当前历史窗口内向前填充，开头缺失用训练均值表示，原 Mask 同时输入模型。数值 scaler 仅由训练有效观测拟合；N2 复用且独立重算核实其原生训练 scaler。

## 基线与消融

主表 9 种方法：Zero Return / Persistence、DLinear、PatchTST 风格编码器、PIVOT numerical backbone、Text Mean、Latest Eligible、Semantic、Semantic+Temporal、Full PIVOT。

消融 11 种：without_ctsr、without_str（所有合法候选均值池化）、without_temporal、without_fusion、random_eligible、shuffled_eligible、text_missing、topk1、topk8、lag0、lag3。所有随机/打乱操作都在相同预测起点合法候选中完成；打乱用于破坏通道关联，不改变可用时间合法性。without_fusion/text_missing 与主表数值控制可能是结构等价检查，不应当作独立方法证据重复宣传。

默认文本 Top-K=4、CTSR 通道 Top-K=4；文本候选时间范围 30 自然日。Top-K 消融改变文本条数，CTSR 数保持不变。默认可用性延迟 1 小时，额外 0/3 小时使用同一数值样本集合。

正式种子固定 2024、2025、2026。默认主表 9×3H×3seeds=81 次，消融 11×3H×3seeds=99 次。200 个 YAML 覆盖所有候选数据集与方法；天气文本配置标为 disabled，默认不会被脚本执行。可选天气主表仅含 naive、dlinear、patchtst、backbone。

## 预算、模型选择与指标

AdamW，lr=0.001，weight_decay=0.0001，batch_size=8，最多 30 epochs，patience=5；默认 dim=32、Transformer 1 层、4 heads、dropout=0.1。StockNet patch/stride=8/4，天气=16/8。固定设置，无本轮调参或超参搜索。

以验证集 **标准化 MSE** 选择最佳 checkpoint；测试集仅在选择结束后评估。模型输入键显式限定为 x_num、x_mask、text_embeddings、text_mask，y 不会传入模型。恢复包含模型、最佳权重、优化器、AMP scaler、随机状态、数据与代码绑定；已有成功结果不会被不同配置覆盖。

报告有效标签上的原单位和标准化 MSE/MAE；跨单位天气数据以标准化指标为主要比较，原单位指标供核查。StockNet 附加三值 sign（负/零/正）Directional Accuracy、每个 origin/lead 至少 5 个有效股票且有可计算方差时的横截面 Spearman Rank IC、标签覆盖率。常数预测的 Rank IC 可为 null。Directional Accuracy 不是交易收益。

统计汇总报告三个种子均值与样本标准差；先对种子求起点误差均值，再对连续预测起点进行块 Bootstrap，块长至少 H，提供区间和相对于 backbone 的配对 MSE 差异区间。小样本区间应谨慎解释，尤其 StockNet 测试窗口仅 40—44 个。汇总拒绝不同评估样本指纹、重复的 dataset/H/method/seed，不择优保留重跑结果。

参数量、训练/推理耗时、峰值显存、预处理和 embedding 时间均记录。参数量为模型注册参数总数；禁用组件可能保留注册权重，因此不能把该计数直接当作有效计算量。不同模型不保证参数量完全一致；文本方法尽量共享骨干与训练预算。所有预处理/索引成本必须与训练成本分开报告。

本轮只执行 batch=1 的真实 smoke，每个可训练配置恰好 1 optimizer step，单批验证和测试；没有运行以上正式预算与三个种子的实验。
