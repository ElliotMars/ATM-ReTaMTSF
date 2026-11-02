# ReTaMTSF → Fidel-TS：分阶段代码重构与实验执行

你现在是一名熟悉 PyTorch、时间序列预测、多模态学习与科研实验规范的资深研究工程师。

请对我的 ReTaMTSF 项目进行分阶段重构，将主要 Benchmark 从 Time-MMD 迁移至 Fidel-TS，以支持后续向 Knowledge-Based Systems（KBS）期刊投稿。

## 一、项目与数据来源

### 原始项目

GitHub：

https://github.com/ElliotMars/ATM-ReTaMTSF

请以当前工作区的实际代码为准，不要假设仓库结构与你的记忆完全一致。

重点检查：

- `models/ReTaMTSF.py`
- `layers/ReTaMTSFLayers.py`
- `data_provider/data_loader.py`
- `data_provider/data_factory.py`
- `exp/exp_ReTaMTSF_*.py`
- `run_ReTaMTSF_*.py`
- `utils/Coherence_Compute.py`
- `pre_process/Time-MMD/`
- 原有 Baseline、训练与评测脚本

### 新 Benchmark

Fidel-TS 论文：

https://arxiv.org/abs/2509.24789

Fidel-TS 数据集：

https://huggingface.co/fidel-ts

优先适配：

1. `fidel-ts/California_ISO`
2. `fidel-ts/Jena_Atmospheric_Physics`

后续扩展：

3. `fidel-ts/Bear_room`
4. 其他具有合适多变量数值序列及文本上下文的 Fidel-TS 子集

不要一次性下载全部六个数据集。

先检查数据集 README、真实文件树、时间序列 Parquet、天气报告 JSON、文本 Embedding、Subject/Channel 元信息以及版本结构。

不要凭论文中的概括推测具体文件格式。

---

## 二、重构的总体原则

本次不是设计一个全新的神经网络，而是迁移 Benchmark、补强数据正确性和科研验证。

### 必须保留的内容

尽可能保留原有：

1. CTSR：Coherence-based Time Series Retrieval。
2. Content Synthesis：时间序列内容合成模块。
3. STR：Semantics-driven Text Retrieval。
4. Text Cross-Attention。
5. Cross-modal Fusion。
6. 原有预测头与基本损失函数。

不主动增加复杂 Gate、Contrastive Loss、LLM Backbone、新型 Attention 或其他大型模块。

如果发现模型代码存在阻碍正确训练的 Bug，可以做最小修复，但必须记录：

- 原代码行为。
- Bug 的具体影响。
- 修改后的行为。
- 是否改变原模型数学定义。
- 是否需要重新训练和验证。

所有不可避免的模型结构变化都必须有独立的理由和测试。

### 必须避免

- 不将 CITRA 的 AVL、泰勒近似或 Certified Aggregation 引入 ReTaMTSF。
- 不以修改模型结构来掩盖 Benchmark 适配问题。
- 不覆盖原 Time-MMD 代码、实验结果或 Checkpoint。
- 不使用测试集选择超参数。
- 不人为构造有利于 ReTaMTSF 的数据划分。
- 不在没有证据的情况下声称 Fidel-TS 已保证所有特征无泄漏。
- 不将未来真实观测当作已知天气预报。
- 不为了获得更好结果而自动调整测试协议。
- 不自动推送远端 GitHub。
- 不修改与当前任务无关的文件。

建议使用独立分支，例如 `feature/fidel-ts-adaptation`，并独立保存新实验产物。

---

# 三、非常重要：执行方式与手动交接协议

**必须严格采用阶段式工作流。每完成一个阶段，先汇报结果并停止，等待我明确回复“继续下一阶段”。**

你可以自主完成：

- 阅读代码和文档。
- 检查少量文件元数据。
- 编写和修改代码。
- 静态检查。
- 单元测试。
- 使用极少量本地样本进行 Smoke Test。
- 不需要长时间运行的小型 CPU 验证。
- 汇总我手动完成的实验产物。

但以下工作必须交给我手动执行：

1. 大规模数据集下载。
2. 全量数据预处理。
3. 全量文本 Embedding 生成。
4. 需要大量 GPU 计算的训练。
5. 多随机种子、多 Horizon 的实验。
6. 大规模参数搜索。
7. 完整 Baseline 训练。
8. 长时间运行的评测和性能 Benchmark。

默认规则：

**预计执行超过约 60 秒、可能占用大量 GPU/CPU/内存/磁盘、或需要长时间外部下载的命令，不得由你主动运行。**

不确定成本时，先做 Dry Run、显示数据规模或生成执行命令。

你必须：

- 编写对应脚本。
- 运行必要的小样本测试。
- 给出我能够直接复制的 Shell 命令。
- 说明输入位置和输出位置。
- 说明预期生成哪些文件。
- 提供检查实验是否成功的命令。
- 明确说明可能的资源占用和异常处理方式。
- 停止，等待我执行并反馈。

不要在等待我时自行进行下一阶段。

不要使用后台进程、`nohup`、自动创建长期运行的 `tmux` 任务等方式绕过交接机制。

如果我反馈训练日志、错误信息或实验结果，你应先检查、修复和核验当前阶段，再决定是否进入下一阶段。

---

# 四、阶段 0：原仓库与 Fidel-TS 结构审计

**本次开始执行时，只执行阶段 0，不要直接进入后续阶段。**

## 0.1 原仓库审计

阅读实际代码并确定：

- 数据加载和训练入口。
- 时间序列输入输出形状。
- 文本 Embedding 输入形状。
- CTSR 参考通道的组织方式。
- STR 的 Top-K 候选文本构建方式。
- Cross-Attention 对序列长度和维度的要求。
- 原有训练、验证、测试划分。
- 归一化操作及拟合范围。
- 模型支持单目标还是多目标预测。
- 原来的 Aligned、Retrieval、Random、Ablation 模式如何工作。

特别检查：

原 `data_loader.py` 存在随机选择数据起始位置的逻辑，必须确认它是否影响可复现性、连续时间窗口或训练测试划分。

原 Text Retrieval 使用预测区间对应的文本日期。需要明确这与新 Benchmark 的预测起点可用性要求存在什么差别。

原 Time-MMD Dataloader 不应直接复用于 Fidel-TS。

## 0.2 Fidel-TS 数据审计

优先查看 California ISO 和 Jena 的：

- 真实目录结构。
- 数值数据列与时间戳。
- Subject ID。
- Channel ID。
- 数值采样频率。
- 天气报告及其内容结构。
- 天气预报发行时间。
- 预报覆盖时间。
- 多个预报版本。
- 静态 Channel Description。
- 已提供的文本 Embedding。
- 数据缺失及已有预处理方式。

特别检查 `time_series/`、`weather/weather_report/`、`weather/report_embedding/`、`static_info.json` 等真实路径。

注意：这些是可能存在的路径类别，必须以下载到的实际数据文件为准。

检查官方有没有推荐的：

- Train/Validation/Test 时间划分。
- Lookback 长度。
- Prediction Horizon。
- 评价指标。
- Subject 与 Channel 组织方式。
- Baseline 配置。
- 数据预处理协议。

如存在多个 Dataset 版本，记录具体 Revision / Commit SHA。

如果没有明确提供官方划分，不得声称自行设计的划分是官方设置。

## 0.3 形成适配设计报告

新增：

`docs/fidel_ts/00_repository_audit.md`

报告包括：

1. 原模型的完整数据流。
2. Fidel-TS 的实际数据结构。
3. 两者的数据接口差异。
4. 哪些文件必须新增。
5. 哪些原文件需要最小修改。
6. 哪些模型模块可以保持完全不变。
7. 关键数据泄漏风险。
8. CAISO/Jena 的适配难度。
9. 后续各阶段的具体执行计划。

在这个阶段，只读检查与必要的轻量元数据脚本是允许的。

**阶段 0 完成后停止，并向我汇报，不要下载完整数据或启动训练。**

---

# 五、阶段 1：Fidel-TS 下载及数据检查工具

在我批准阶段 0 后开始。

## 1.1 实现可复现下载脚本

为 CAISO、Jena 编写独立下载入口。

要求支持：

- 指定数据集。
- 指定下载目录。
- 指定 Revision。
- 可选仅下载需要的文件类型。
- 下载断点续传。
- 已存在数据校验。
- 不覆盖其他数据集。
- 记录来源、Revision、文件列表及哈希。
- 支持只列出文件、不实际下载的模式。

建议目录：

`dataset/Fidel-TS/raw/California_ISO/`

`dataset/Fidel-TS/raw/Jena_Atmospheric_Physics/`

不要假设 Hugging Face 的 `load_dataset()` 可以直接解析所有嵌套元数据。

优先根据实际文件组织使用 `huggingface_hub` 下载和 Parquet/JSON 解析。

对于 `.pkl` 等可能包含序列化对象的文件，不要默认执行不可信的 Pickle 反序列化。先确认来源和内容格式，必要时优先使用原始文本重新生成 Embedding。

## 1.2 数据检查脚本

输出：

- 数值时间范围。
- 采样频率。
- 各 Subject/Channel 数量。
- 缺失率。
- 重复时间戳。
- 文本文件数量。
- 文本版本数量。
- 文本发布时间覆盖率。
- 文本预测覆盖时间。
- 数值与文本时间重叠情况。
- 可用于训练的时间范围。

提供 `--inspect-only` 选项。

这个阶段不要执行完整下载，由我手动运行命令。

下载完成后，我会把输出日志提供给你，再进入下一阶段。

---

# 六、阶段 2：独立的 Fidel-TS 数据适配器

本阶段是最关键的工程工作。

新增独立模块，例如：

- `data_provider/fidel_ts_dataset.py`
- `data_provider/fidel_ts_schema.py`
- `data_provider/fidel_ts_collate.py`
- `configs/fidel_ts/`

可以根据代码规范调整命名，但不要破坏原始 Time-MMD Dataset。

## 2.1 统一样本接口

目标是在不修改原模型主要结构的情况下，提供：

- `target_series_x`: `[B, C_target, L]`
- `target_series_y`: `[B, C_target, H]`
- `TS_database`: `[B, C_reference, L]`
- `query_text_embeddings`: `[B, C_target, H, D]`
- `channel_description_embeddings`: `[B, C_target, 1, D]`
- `candidate_text_embeddings`: 兼容原 STR 的候选文本张量
- `candidate_text_mask`: 有效候选标记
- `forecast_cutoff`: 预测起点
- `sample_metadata`: Source / Subject / Channel / 版本等

不必强制返回 Tuple，可使用结构化 Batch，只要模型调用接口正确即可。

文本维度优先延续原模型的 384 维配置。

如果使用官方预计算 Embedding，必须核对其模型和维度；不一致时通过明确的投影或重新编码处理，不允许偷偷截断向量。

## 2.2 Subject 与 Channel 处理

必须保留 Fidel-TS 对 Subject 和 Channel 的区别。

优先实现单 Subject 内的多变量预测：

- 选定目标 Channel。
- 其他可用 Channel 作为 CTSR 候选。
- 不默认把不同 Subject 的数据当作同一个连续时间序列。
- 不混合具有不同单位、时间戳或语义的通道。
- 防止目标 Channel 的未来数据通过参考通道泄漏。

保持原 ReTaMTSF 的单目标、多参考通道设定。

如果某个数据子集没有合适的多变量参考通道，应报告该限制，不人为复制目标通道来构造 CTSR。

## 2.3 严格时间划分

优先遵循 Fidel-TS 官方论文与代码的设置。

窗口必须满足：

- 历史数值不晚于预测起点。
- 标签严格属于预测区间。
- 训练期 Scaler 只能使用训练数据拟合。
- 验证、测试不重新拟合 Scaler。
- 训练样本不得跨越不允许的划分边界。
- 测试结果不得用于模型选择。
- 对不同时区进行一致转换。
- 缺失值处理不得读取未来测试信息。

特别审查 Fidel-TS 提供的已插值数值文件是否使用了双向插值。

如果存在可能使用未来观测的预处理，应区分官方 Benchmark 设置与更加严格的因果设置，并记录限制，不能默默修改后仍与官方数字直接比较。

## 2.4 单元测试

构造小规模合成数据，测试：

- 预测区间正确。
- Subject/Channel 映射正确。
- 不发生跨 Subject 窗口拼接。
- Scaler 不接触验证或测试数据。
- 缺失与重复时间戳处理。
- 所有返回张量形状正确。
- CPU Dataloader 正常工作。

本阶段只运行单元测试与少量样本加载，不启动完整训练。

---

# 七、阶段 3：文本检索与时间可用性适配

保留原 STR 的语义 Top-K 思路，但使候选池符合 Fidel-TS 的数据特点。

## 3.1 明确时间语义

对每条文本建立内部记录：

- `document_id`
- `subject_id`
- `location_id`
- `issued_at`
- `available_at`
- `forecast_valid_start`
- `forecast_valid_end`
- `version_id`
- `text`
- `embedding`

字段名称可以根据实际数据调整。

必须区分：

**发行时间、系统可用时间和预报覆盖时间。**

如果某项时间信息未提供，不得捏造。

对于预测起点 \(t\)，只能检索满足已定义可用性规则的文本。

如果多个版本对应相同预报区间，选择在截止时刻可用的版本，而不是离线档案中最终的版本。

已发布的未来天气预报可以作为未来已知外生信息使用。

但预测起点之后才生成的天气预报、未来真实天气观测及未来目标值均不得使用。

## 3.2 适配原 STR

保留原有：

- Query Text Embedding。
- Channel Description Embedding。
- 相似度 Top-K。
- Text Cross-Attention。

新增：

- 合法候选过滤。
- 有效文本 Mask。
- 可复现的 Top-K 排序。
- 文本为空时的处理。
- 版本选择。
- 多候选文本的 Batch Padding。

不要因为候选文本不足 K 条而重复相同文本并将其视为新的独立证据。

必要时允许最小修改 STR，使 Padding 不参与相似度排序和注意力。

如果需要添加 Mask，应保持原网络的数学含义尽可能不变。

## 3.3 文本 Embedding

优先统一使用 ReTaMTSF 原来的 `paraphrase-MiniLM-L6-v2`，输出 384 维。

建议离线完成天气报告编码、Channel Description 编码和 Query Text 编码。

记录：

- Embedding 模型名称。
- 模型 Revision。
- 文本清洗规则。
- 截断长度。
- 文本模板。
- 缓存文件哈希。
- 文本与版本 ID 映射。

若原报告较长，先进行文本长度分布统计，不要直接通过随意截取头部导致重要信息丢失。

提供可复现的编码策略，并记录信息截断情况。

全量编码由我手动执行。

## 3.4 必须增加的文本控制

支持以下模式：

- `real_text`
- `date_only`
- `content_only`
- `random_legal`
- `shuffled_content`
- `no_text`

其中：

`date_only` 只保留时间描述，不输入真实天气内容。

`content_only` 去掉人工增加的日期前缀，但不强行删除天气预报中原有的时间和季节描述。

`random_legal` 只能从预测时刻合法的候选集合中采样。

`shuffled_content` 保留原时槽的发行时间和版本结构，仅替换合法文本内容。

`no_text` 不读取实际天气文本信息。

所有控制必须使用相同的数值任务、预测划分及公平的训练预算。

## 3.5 本阶段验收

必须通过以下测试：

1. `issued_at > cutoff` 的文本无法被选择。
2. 多版本文本不会选择截止时刻之后发布的版本。
3. 原始天气预报覆盖未来时间，不会因此被误删。
4. 不同 Subject 不会错误混用文本。
5. 无候选文本时不会产生 NaN。
6. 所有候选 Embedding 与原文身份正确绑定。
7. Random/Shuffled 不会读取未来文本池。
8. 改变无关未来文本不会改变当前 cutoff 的候选集合。

---

# 八、阶段 4：接入原 ReTaMTSF 模型

在数据与文本适配完成后，接入原有网络。

优先新增：

- `exp/exp_ReTaMTSF_FidelTS.py`
- `run_ReTaMTSF_FidelTS.py`
- `configs/fidel_ts/caiso.yaml`
- `configs/fidel_ts/jena.yaml`

可根据仓库已有代码抽象公共逻辑，但不要为了代码美观而进行大规模无关重构。

## 4.1 最小侵入式接入

尽量使原有 `models/ReTaMTSF.py` 与主要网络层保持不变。

如果必须修改：

- 清楚标注修改位置。
- 确保旧 Time-MMD 模型调用依然兼容。
- 增加形状与数值回归测试。
- 不改动已有检索评分公式，除非发现明确 Bug。
- 不引入新的训练损失。

## 4.2 Smoke Test

使用合成数据或极小本地样本验证：

- CPU Forward。
- GPU Forward（环境可用且无需长时间运行时）。
- Backward。
- 一次 Optimizer Step。
- Checkpoint 保存与恢复。
- 文本关闭/开启模式。
- CTSR-only。
- STR-only。
- Full。
- 不同 Batch Size。
- 不同预测长度。

检查全部张量形状、有限值和梯度。

只运行少量 Batch，不做完整训练。

---

# 九、阶段 5：先导实验

本阶段由你编写脚本，我手动运行完整训练。

## 5.1 数据集

优先：

- California ISO。
- Jena Atmospheric Physics。

先选择官方协议中适合原模型的任务及预测长度，不根据测试效果挑选最容易的任务。

## 5.2 第一批实验方法

至少包含：

1. Persistence / Last Value。
2. DLinear。
3. ReTaMTSF TS-only。
4. ReTaMTSF Full。
5. ReTaMTSF Date-only。
6. ReTaMTSF Random Legal Text。
7. ReTaMTSF No Text（作为必要的无文本控制）。

如资源允许再补：

8. Content-only。
9. Shuffled Content。

所有方法共享：

- 相同训练验证测试切分。
- 相同预测目标和长度。
- 相同数值历史。
- 相同标准化规则。
- 明确可比较的模型训练预算。
- 对应的文本访问权限。

第一轮不要进行大规模参数搜索。

至少固定三个随机种子；可以先单种子验证流程，再手动跑完整三种子。

## 5.3 自动生成实验命令

请提供：

- 单模型训练命令。
- 单模型测试命令。
- 三种子顺序运行脚本。
- 可选择 GPU 的脚本。
- 可跳过已完成实验的脚本。
- 实验汇总脚本。
- 运行失败时不覆盖旧结果的机制。

训练日志应保存：

- Dataset。
- Target。
- Seed。
- Lookback。
- Horizon。
- Epoch。
- 学习率。
- Batch Size。
- 最佳验证 Epoch。
- Train/Val/Test MSE、MAE。
- Peak GPU Memory。
- 训练与推理耗时。
- Git Commit。
- 数据与配置哈希。

最终汇总为 CSV 和 Markdown 表格。

**本阶段不允许你主动启动完整训练。必须交给我运行。**

我提供结果后，你先分析原因，再提出下一阶段方案。

---

# 十、阶段 6：完整消融与鲁棒性实验

只有先导实验的技术流程验证完成后再执行。

本阶段复用原仓库已经实现的检索方案，尽量避免增加网络复杂度。

## 6.1 双路检索消融

比较：

- No Retrieval。
- CTSR-only。
- STR-only。
- Full ReTaMTSF。

并记录参数量及实际输入信息差异。

## 6.2 CTSR 检索对照

根据现有实现比较：

- Coherence。
- Lagged Correlation。
- Partial Correlation。
- Granger-style Score。
- Random Channel。

可以增加 Pearson Correlation，但不是强制。

需要说明每种方法的计算代价与输入条件。

## 6.3 文本信息消融

比较：

- Full Text。
- Date-only。
- Content-only。
- Random Legal。
- Shuffled Content。
- No Text。

重点验证真正的文本内容是否贡献了额外的预测信息。

不得仅凭 Full 优于 TS-only 就声称文本语义有效。

## 6.4 参数敏感性

保留原有：

- TS Top-K。
- Text Top-K。
- Lookback。
- Prediction Horizon。

根据验证集决定少量合理取值，不做无依据的巨大网格搜索。

## 6.5 多随机种子统计

报告：

- MSE/MAE 均值。
- 标准差。
- 逐目标指标。
- 配对预测损失差。
- 时间块 Bootstrap 置信区间。

如果时间序列相关性较强，应使用合理的块长度，并说明统计限制。

不要把同一时间样本在不同 Seed 下的重复预测当作相互独立的观测。

## 6.6 失败案例

识别：

- 文本改善预测的情形。
- 文本损害预测的情形。
- Coherence 选择效果不佳的情形。
- 日期与语义检索发生分歧的情形。

保留全部负结果。

---

# 十一、阶段 7：Baseline、资源测试与论文实验报告

本阶段以 KBS 投稿为目标。

## 7.1 Baseline

至少适配：

- DLinear。
- PatchTST。
- iTransformer。

如能获取 Fidel-TS 官方评测协议，再评估适合比较的多模态模型。

其中 ReTime、TGTSF 等与原 ReTaMTSF 直接相关的方法，需要优先核查代码适配难度。

如果某个 Baseline 无法适配，不要编造结果；说明其输入假设与任务不兼容的原因。

所有 Baseline 使用相同的任务设置和公平的超参数选择协议。

不允许将不同数据集、不同历史窗口或不同文本输入设置下的论文数字直接拼在一张主结果表中。

## 7.2 效率实验

报告：

- 模型参数量。
- 文本离线编码时间。
- 在线文本检索成本。
- 模型 Forward 时间。
- 完整预测延迟。
- GPU 峰值显存。
- CPU 内存。
- 与 TS-only 和强数值 Baseline 的成本差异。

区分离线/在线成本。

不要将预先计算的文本 Embedding 描述为零成本。

## 7.3 自动形成最终实验报告

新增：

`docs/fidel_ts/final_experiment_report.md`

内容包括：

1. Fidel-TS 数据处理协议。
2. 正式训练/验证/测试设置。
3. 主预测结果。
4. 文本内容消融。
5. CTSR/STR 消融。
6. 参数敏感性。
7. 时间可用性审计。
8. 资源成本。
9. 失败案例。
10. 数据和实现限制。
11. 对原 ReTaMTSF 论文贡献的影响。
12. KBS 投稿前需要修改的论文章节。

所有图表应通过保存的原始预测、配置和评测产物自动生成。

如果 Full ReTaMTSF 没有超过强数值 Baseline，必须如实说明。

如果真实文本不优于 Date-only 或 Random Legal Text，必须说明独立语义收益证据不足。

不得通过只汇报有利的 Dataset、Horizon 或 Seed 来包装结果。

---

# 十二、项目文件组织建议

尽量保持原仓库不变，新功能使用独立目录：

`data_provider/fidel_ts_*.py`

`pre_process/Fidel-TS/`

`configs/fidel_ts/`

`exp/exp_ReTaMTSF_FidelTS.py`

`run_ReTaMTSF_FidelTS.py`

`scripts/fidel_ts/`

`tests/fidel_ts/`

`docs/fidel_ts/`

`artifacts/fidel_ts/`

`dataset/Fidel-TS/`

实际目录可以根据原仓库规范微调。

将大型数据、缓存、Checkpoint 和模型权重正确加入 `.gitignore`。

不要将原始 Time-MMD 实验产物覆盖为 Fidel-TS 结果。

---

# 十三、每个阶段必须使用的回复格式

每完成一个阶段，请严格按照以下格式回复我：

## 当前阶段

阶段编号、阶段名称。

## 1. 已完成的内容

说明实际完成了什么。

## 2. 修改的文件

逐项列出新建或修改的文件及其功能。

## 3. 已验证的测试

列出实际执行的测试命令、结果和未通过项。

## 4. 需要我手动执行的命令

提供可以直接复制运行的完整 Bash 命令。

命令必须包含清楚的路径、配置、输出位置和必要环境变量。

如存在多组命令，按正确执行顺序给出。

不得把未实际运行的训练命令描述为已完成。

## 5. 实验产物

说明预期生成的文件、日志和结果表。

## 6. 验收方式

提供我可以手动运行的检查命令，以及判断成功或失败的标准。

## 7. 已发现的问题

区分：

- 已解决。
- 尚未解决。
- 会影响正确性的问题。
- 只影响性能的问题。

## 8. 下一阶段

说明下一阶段拟开展什么。

然后**停止执行**。

等待我明确回复“继续下一阶段”或提供实验结果。

---

# 十四、科研质量与停止条件

本项目不是为了证明 ReTaMTSF 一定比所有 Baseline 更好，而是判断其原有方法能否在更可靠的数据集上成立。

在任何阶段，如果发现：

- 数据时间字段不足以支持正确回放；
- 原 Benchmark 提供的文本可能包含预测目标答案；
- CTSR 候选通道与 Subject/Channel 结构不兼容；
- 主要模型存在影响预测有效性的 Bug；
- 实际文本输入与训练任务不匹配；

应优先停止、报告并提出最小修正。

如果先导实验发现真实文本收益非常有限，不要立即大幅修改网络。

先完成相同条件下的 TS-only、Date-only、Random/Shuffle 分析，区分问题来自数据、检索或融合模型。

如果最终新 Benchmark 不支持原有核心研究主张，应坦诚形成负结论，并提出论文定位调整建议。

---

## 现在开始

**请现在只执行阶段 0：仓库审计与 Fidel-TS 数据结构调查。**

不要直接进入阶段 1–7。

不要启动大型下载、完整文本编码或模型训练。

阶段 0 完成后输出 `docs/fidel_ts/00_repository_audit.md`，展示适配方案、风险、预计改动文件、测试计划，以及下一阶段需要我手动执行的命令。

随后停止，等待我确认。