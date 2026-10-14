# 任务：基于当前 ReTaMTSF 仓库开发 CITRA 原型

你是一名具有 ICDE/SIGMOD/VLDB 研究经验的数据库系统研究员，同时熟悉多模态时间序列预测、PyTorch、向量检索、时态数据库和近似查询处理。

请在**当前仓库的实际代码基础上**开展研究级重构，开发新方法：

**CITRA: Certified Influence-aware Temporal Retrieval and Aggregation**

目标论文暂定为：

*Certified Temporal Evidence Aggregation for Text-Augmented Time Series Forecasting*

研究目标是设计一个支持历史时刻文本可用性、版本化文档、预测条件化聚合和可认证预测输出误差的数据库查询算子及其索引算法，并验证其对多变量时间序列预测和查询效率的贡献。

注意：这是尚待实验验证的研究假设。不要预设 CITRA 已具备 ICDE 录用级创新性，不要编造理论保证、运行结果或预测收益。

## 0. 仓库检查与安全要求

当前仓库应为 `ElliotMars/ATM-ReTaMTSF`，原代码包括：

- `models/ReTaMTSF.py`
- `layers/ReTaMTSFLayers.py`
- `data_provider/data_loader.py`
- `data_provider/data_factory.py`
- `utils/Coherence_Compute.py`
- `pre_process/Time-MMD/`
- `exp/`
- `run_ReTaMTSF_*.py`

以上是已有 GitHub 版本中核对过的路径。但本地工作树可能有所变化，因此必须先运行 `git status`，读取实际目录、README、训练入口和核心模块，不得假定本地内容与远端完全一致。

要求：

1. 不覆盖、不删除原 ReTaMTSF 的代码、权重、实验日志和数据。
2. 不自动提交到远端，不执行强制推送、硬重置或清除用户改动。
3. 在工作树安全的情况下，使用新分支 `feature/citra-prototype`；如果已有未提交改动，先报告状态并采用不会覆盖工作的处理方式。
4. 尽可能复用已有数据格式、配置及训练工具，但 CITRA 模型与原模型保持独立。
5. 新增可运行的模块、单元测试、实验入口和研究文档，不要只生成框架空文件。
6. 优先兼容当前环境已有 Python/PyTorch 版本；新增依赖需列明且尽量轻量。

## 1. 第一优先级：审计旧仓库是否存在未来文本泄漏

重点检查 `data_provider/data_loader.py` 中各数据集的 `__getitem__`、文本检索以及文本 embedding 的生成逻辑。

已知 Energy 数据加载器存在以下值得审计的设计：

- `h_begin = lbw_end`
- `h_end = lbw_end + pred_len`
- `time_span_sample = self.time_span[h_begin:h_end]`
- retrieval 模式随后根据未来时间区间筛选 `News-日期.npy`。

请确认实际代码中这些日期代表发布时间、事件时间、目标时间还是其他含义。

要求：

- 区分 `event_time`、`issued_at`、`available_at`、`ingested_at` 和文档版本有效时间。
- 不要在缺少证据时断言历史实验必然发生泄漏。
- 只要没有可靠的历史可用时间，就不能把对应文本作为严格回测中的可见证据。
- 对旧 Time-MMD 数据集输出审计结果；保留其原始实验路径。
- 新建 Point-in-Time 文档访问接口及单元测试。

在 `docs/citra_data_audit.md` 记录每个数据字段的真实含义、来源、已验证事项、假设和潜在问题。

## 2. 定义 CITRA 的核心查询问题

预测查询为：

`q = (X_history, target, horizon, cutoff, epsilon)`

其中 `cutoff` 是预测起点，`epsilon` 是相对于完整合法文本聚合结果的允许预测输出偏差。

历史快照中的文档必须在 `cutoff` 时刻可访问，并且使用当时可访问的正确版本。

CITRA 不返回普通 Top-K 新闻列表作为主要结果，而返回用于预测的近似文本影响摘要及其认证误差上界。

设文档表示为 `z_d ∈ R^p`，影响向量为 `v_d ∈ R^m`，预测查询表示为 `q_z`。

初始版本使用可解析的平滑核：

`k(q_z,z_d) = exp(-||q_z-z_d||² / (2*sigma²))`

完整聚合定义：

`A_full(q, cutoff) = Σ_{d in visible(cutoff)} k(q_z,z_d) * v_d`

首先实现该可认证的求和聚合，不允许将语义 Top-K 的结果错误地称为全量参考值。

预测输出采用：

`Y_full = F_TS(X_history) + G(X_history, A_full)`

`Y_fast = F_TS(X_history) + G(X_history, A_approx)`

第一版预测修正头 G 必须具有可计算、有效的 Lipschitz 上界，例如采用范数可控制的线性投影或有明确界的门控结构。

最终要求：

`||Y_fast - Y_full||₂ <= certified_bound <= epsilon`

注意：

- 误差界仅约束近似模型与完整合法文本模型的输出差异。
- 不保证相对于未知未来真实值的 MSE。
- 不允许将一般 Transformer 或未经约束的非线性网络默认视为具有已知 Lipschitz 常数。
- 不允许使用未来真值或测试集标签构造在线聚合权重。

## 3. 实现 CITRA-Index：分层时间有效的影响聚合索引

新建独立模块，例如 `citra/`，具体结构可在检查现有工程后合理调整。

每个索引节点 P 至少维护：

- 文档数量。
- 文档 embedding 中心 `c_P`。
- 最大 embedding 半径。
- 聚合影响向量 `V_P = Σ v_d`。
- 一阶矩 `M_P = Σ (z_d-c_P) v_dᵀ`。
- 二阶加权残差 `S_P = Σ ||z_d-c_P||² ||v_d||`。
- 文档可用时间和版本元数据。
- 子节点及文档引用。

查询时，节点的一阶泰勒近似为：

`A_approx(P) = k(q,c_P) * V_P + M_Pᵀ * grad_z k(q,c_P)`

对于具有有效 Hessian 范数上界 `H_P(q)` 的节点：

`||A_exact(P)-A_approx(P)||₂ <= 0.5 * H_P(q) * S_P`

对于 RBF 核，可以先使用保守但严格成立的 Hessian 上界，然后再研究更紧的计算方式。

重要要求：

- 实现时检查矩阵维度和梯度定义，不要直接照抄伪代码。
- 用数值微分和自动微分对照检查解析梯度。
- 误差摘要必须按照数学定义正确计算。
- 多节点聚合必须无遗漏、无重复计数。
- `epsilon=0` 时应允许精确遍历文档，而不是返回虚假的零上界。
- 空文档库必须返回零影响和合法的零误差界。
- 浮点比较需采用明确数值容差，不能通过事后修改误差界掩盖违反理论的情况。

## 4. 设计自适应查询算法 CITRA-Query

采用 coarse-to-fine 查询：

1. 获取预测起点对应的合法历史快照。
2. 使用索引粗粒度节点计算聚合近似。
3. 为每个节点计算严格有效的误差贡献上界。
4. 汇总为预测输出误差上界。
5. 如果满足 `epsilon`，立即返回。
6. 否则细化一个或多个具有较高误差贡献的节点。
7. 重复直到满足约束，或回退到精确执行。

实现一个独立的 exact aggregation oracle，保证可随时进行正确性对照。

首先使用最大误差贡献优先的 best-first 策略，不必一开始就实现复杂的学习型查询优化器。

必须统计：

- 访问的索引节点数量。
- 实际扫描的文档数。
- 精确核函数计算次数。
- 查询耗时。
- 返回的误差上界。
- 实际相对于 exact oracle 的预测输出偏差。
- 是否需要回退到精确执行。

索引的中心研究问题是：在相同输出误差阈值下，CITRA 是否能减少实际数据库访问与查询计算。

## 5. 历史时间与版本管理

建立独立的 `TemporalDocumentStore`。

文档至少包含：

`doc_id, version_id, source, entity_id, issued_at, available_at, valid_from, valid_to, event_start, event_end, embedding, impact_vector`

实现：

- `insert_document`
- `append_revision`
- `snapshot(cutoff)`
- `visible_documents(cutoff)`
- `aggregate_exact(query, cutoff)`
- `query_certified(query, cutoff, epsilon)`

对历史版本采用 append-only 或其他有清晰语义的设计。

禁止将历史时刻的文档悄悄替换为最新版本。

如果需要实现增量更新，应保证受影响索引节点的统计量被正确维护。对于混合了合法与非法版本的节点，不得直接使用整节点聚合摘要；必须继续细分，或采用明确证明正确的时间前缀摘要机制。

当原始数据不提供真实摄入时间时，必须明确标记为未知或模拟，不允许伪造历史摄入时间。

## 6. 与 ReTaMTSF 的集成

不要直接把原来的 `TextCrossAttention` 替换掉后声称两者等价。

新增独立预测模型 `models/CITRA.py` 或等效模块。

建议组成：

- 一个数值预测 Backbone。
- 一个文本 embedding / impact encoder。
- CITRA-Index 与 CITRA-Query。
- 一个具备可计算误差放大上界的预测修正头。
- 最终多变量多步预测输出。

第一阶段可使用小型 Linear/DLinear 风格时序骨干；之后再接入 iTransformer 或其他强基线。

必须区分：

1. 文本是否能改善实际预测误差。
2. 近似聚合是否接近全量文本聚合结果。
3. CITRA 是否比现有执行方式更快。

这三个问题不能互相替代。

为了公平比较，TS-only、Full Text、CITRA 必须使用可比的训练配置。CITRA 近似与 Full Text 认证实验应共享同一组已固定模型参数。

## 7. 第一批数据与实验

先用完全可控的合成数据验证数学正确性，不依赖真实数据下载。

生成具有明确时间戳、版本、正负文本影响及可重现实验种子的合成文档。涵盖：

- 过去文档与未来文档。
- 文档修订和迟到文档。
- 高维与低维 embedding。
- 不同时间选择率。
- 不同 epsilon。
- 重复文档和相互抵消的影响。
- 极小和极大的节点半径。

然后提供真实数据适配器接口，优先考虑：

- NWS AFD 历史预报文本。
- ASOS 历史气象观测。
- 可选的 MOS 历史数值天气预报。

如果本地数据不存在，不要臆造真实实验结果。实现下载或导入脚本、数据检查脚本和 README，并明确报告未完成的数据准备工作。

真实数据必须使用按时间顺序划分的训练、验证和测试区间。

## 8. 必须实现的基线

系统实验：

- Exact Full Scan。
- Temporal Filter + Exact Aggregation。
- 无认证的分层近似聚合。
- CITRA Certified Aggregation。

能够合法比较时，再增加 FAISS 等近似检索方案。注意 FAISS Top-K 不一定与全量加权求和具有相同语义，比较时必须明确：它只是有限候选近似方案，而非严格等价的查询算子。

预测实验：

- TS-only。
- Semantic Top-K + Forecast。
- Full As-of Text Aggregation + Forecast。
- CITRA Certified Aggregation + Forecast。

报告：

- MSE / MAE。
- Full Text 与 CITRA 的预测偏差。
- Certified Bound 是否始终有效。
- Bound Tightness。
- P50 / P95 查询延迟。
- 扫描比例、索引大小、构建时间。
- 增量更新开销。
- 误差阈值与性能的权衡。

## 9. 测试要求

至少新增以下单元测试：

- `test_temporal_visibility`
- `test_historical_revision`
- `test_no_future_document`
- `test_exact_aggregation`
- `test_taylor_approximation`
- `test_certified_bound`
- `test_epsilon_monotonicity`
- `test_empty_database`
- `test_incremental_update`
- `test_prediction_shape`

其中 `test_certified_bound` 必须在不同随机种子、embedding 维度、节点大小和 epsilon 下，对照精确计算结果验证误差上界。

请注意：查询运行时间不一定随 epsilon 严格单调，数值算法也可能出现边界行为，因此 `test_epsilon_monotonicity` 应首先验证合理的算法工作量与精度性质，不能假定所有性能指标都数学单调。

先运行 CPU smoke test，再尝试 GPU 集成。不允许因为缺少 GPU 或大规模真实数据，就将未运行的实验标记为通过。

## 10. 研究文档

新增：

- `docs/citra_problem.md`：查询定义与目标。
- `docs/citra_theory.md`：完整推导、假设与理论限制。
- `docs/citra_data_audit.md`：原 ReTaMTSF 时间泄漏风险审计。
- `docs/citra_architecture.md`：模块及数据流。
- `docs/citra_experiments.md`：实验矩阵、评估协议。
- `docs/citra_progress.md`：已完成工作、测试结果、未解决问题。

特别要求在理论文档中讨论：

1. 泰勒展开及 Hessian 上界如何计算。
2. 多节点误差如何累积。
3. 预测修正头的 Lipschitz 常数如何确定。
4. 历史版本过滤对聚合摘要正确性的影响。
5. 为什么误差认证不等于真实未来预测准确率保证。
6. 哪些机制属于已有技术，哪些部分才可能构成新的数据库算法贡献。

不允许以经典泰勒定理本身作为论文原创性主张。

## 11. 实施优先级和执行方式

请真实修改代码，不要只写一个研究计划。

按照以下顺序逐步执行：

**Milestone 1：原仓库审计与最小正确性原型**

完成代码审计、TemporalDocumentStore、Exact Aggregation、基本 CITRA-Index、误差界计算和合成数据单元测试。这是最重要的里程碑。

**Milestone 2：完整的自适应索引查询**

实现 coarse-to-fine 算法、历史版本更新、查询性能统计，并与简单基线对比。

**Milestone 3：预测模型集成**

新建 CITRA 模型，保留 ReTaMTSF 旧基线，跑通端到端训练和评估。

**Milestone 4：真实数据与大规模评估**

接入真实历史文本，开展时间有效性、预测效果与数据库性能实验。

如果单次上下文不足，请优先完成 Milestone 1 的可运行代码、测试和文档，再总结后续步骤，不要为了覆盖全部里程碑留下大量未经验证的伪实现。

遇到数学保证无法满足的情况，请停止声称 Certified，明确报告原因并提出修正。不得通过使用未来答案、遍历全集却不计入耗时、或者返回极松的无限上界来伪造算法优势。

## 12. 最终交付格式

每完成一个里程碑，请报告：

1. 修改或新增的文件。
2. 已实际实现的功能。
3. 运行过的测试命令和真实结果。
4. 未通过的测试和环境问题。
5. 数学假设及潜在反例。
6. 与现有方法相比，目前是否存在实质研究增量。
7. 下一阶段最高优先级任务。

现在先阅读当前仓库并执行 Milestone 1。不要只回复设计说明；在保证现有工作安全的前提下，直接开始实现、运行测试并提供可核查的结果。