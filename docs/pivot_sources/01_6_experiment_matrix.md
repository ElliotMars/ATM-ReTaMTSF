# PIVOT Stage 1.6 实验约束矩阵

Stage 1.6 的代码、测试与协议文档已经编写，但全部单元测试、外部来源 Probe 和真实数据验证均尚未执行。

本文件冻结候选研究任务，不表示来源已被接纳。Stage 2 未实施。

| 组合 | 数值输入/目标 | L | H | 划分 | 文本候选 |
|---|---|---|---|---|---|
| CAISO + NWS/IEM | 原 N2，小时级，全 20 通道 | 360 小时 | 24/72/168 小时 | <2021 / 2021 / >=2022 | AFDMTR、AFDSGX，办公室独立 |
| Jena + GFS | 原 N2，小时级，全 21 通道 | 360 小时 | 24/72/168/336 小时 | <2021 / 2021 / >=2022 | 经审计合法历史 Run 的确定性预报文字 |
| StockNet + Twitter | 固定 88 股票，交易日逐日收益率回归 | 20 个交易日 | 1/3/5 个交易日 | 2014-01 至 2015-06 / 2015-07 至 09 / 2015-10 至 12 | 原始推文创建时间、实体、版本及实际可用性证据 |

StockNet 划分是本研究候选方案，不称为原论文官方划分。Log Return 为候选主目标，价格来源、公司行为和调整机制未审清前不能自动接纳。Close-to-Close 和 Adjusted Close Return 只用于定义比较。分类标签不作回归标签。

所有任务固定通道顺序。CAISO 的 Day ahead forecast、Hour ahead forecast 保留；没有 18 通道实验。没有推文的股票保留，并报告文本缺失状态。共同交易日不足时阻断或保持条件接纳，不删股票、不用未来值回填收益率。

新 H=72 的诊断读取既有 N2 时间戳、原生连续段和全通道 Mask，单独计算窗口统计，不修改 Stage 1.5 配置或产物。窗口不跨 Split 借历史，不跨缺口，不实施正式 Dataset。

## 统一信息权限

1. 先按 `available_at <= forecast_origin` 形成候选池，再检索、计算相似度和排序。
2. Strict Verified 需要经审阅的真实历史接收/公开发布时间证据。
3. Source-time + Conservative Lag 单独报告假设，不能宣传为逐条实际接收时间。
4. GFS 使用 run+3/6/12 小时预设场景；可信实际公开/接收时间存在时优先使用。
5. NWS 另外报告 issued-based、issued+lag 和真实 received-based 情景。零延迟也是未证明瞬时可获得性的实验假设。
6. StockNet 预测起点初设为**当天实际交易收盘后 30 分钟**，这是实验配置。提前收盘日随真实 calendar close 调整，不固定 UTC 小时。

## GFS 后续必需对照

| 对照 | 输入权限 |
|---|---|
| Jena Numerical Only | 同一 Jena 历史与标签 |
| Jena + GFS Raw Numerical | 同一合法历史 GFS Run、变量、格点、Lead 与有效时刻 |
| Jena + GFS Text Simple Fusion | 同一外部信息转换为固定模板文字 |
| Jena + GFS Text Basic Retrieval | 同一合法文字候选池 |
| PIVOT | 同一合法文字候选池；Utility 不得绕过可用性 |

不能将 GFS 强预报系统本身带来的外部信息收益，全部归因为文本表示或检索方法。所有 GFS 方法共享 As-of Run 集、空间提取、原始时间分辨率、缺失策略和有效标签 Mask。

2014 年 Jena 数值历史保留。候选 GFS 档案始于 2015 的信息来自任务要求，实际起始目录仍待 Probe。审计按每个 H 输出原始日历标签早于候选起点的窗口数量；它不是已证明 UTC 对齐后的精确损失，也不表示 2015 年之后档案完整。

## 确定性 GFS 文字设计

仅接受 `origin_kind=model_forecast`、同一 Run/版本/格点、带原始文件 SHA-256 的预测点。禁止 Jena 未来实况。模板逐项写明有效时刻、Lead、变量、值和单位，不调用 LLM，不添加“晴朗”等无直接数值支持的描述。

| 变量 | 接受的原始单位 | 转换/限制 |
|---|---|---|
| Temperature | K / degC | K−273.15；需核查 Level |
| Relative Humidity | % | 保留单位；需核查 Level |
| Wind Speed | m/s | 若以后由 U/V 推导，应保留两分量同 Level/时刻和 hypot 规则；当前 helper 不自动提取 |
| Surface Pressure | Pa / hPa | Pa÷100 |
| Precipitation | m / mm / kg/m2 | m×1000 转 mm；必须保留实际累计起止，不改写成逐小时雨量 |

这是可支持变量的模板设计，**不表示所选历史 GFS 产品已实证包含这些变量**。变量/Level/分辨率只能由实际元数据或小样本确认。3/12 小时预报点不改写成真实逐小时观测；预报跨度覆盖和所有目标小时有直接预测点分别统计。Jena 数值相位为 :10 时也不会自动把整点 GFS 移位或插值。

## 准入解释

Gate A/B/C 分别判断来源完整性、历史可用性和任务兼容性，状态为 pass/conditional/blocked/not_checked。任何 blocked 导致当前 NO-GO；全 pass 才能 GO；其余 CONDITIONAL。有限 Probe 不会自动把全时期状态从 `not_yet_full_coverage_verified` 改成通过。当前代码只实现受限 Probe，因此最多提供带证据的条件可行性，完整接纳需后续独立审核。
