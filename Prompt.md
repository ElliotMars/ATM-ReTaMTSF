# PIVOT Stage 1.6 → Stage 2：一站式完成数据准备、实验方案确定与主实验代码交付

你是一名熟悉多变量时间序列预测、文本增强预测、信息检索、PyTorch、数据工程与科研实验设计的高级研究工程师。

当前项目：PIVOT，目标期刊 Information Sciences。

仓库：`ElliotMars/ATM-ReTaMTSF`

**本轮任务是一次性交付任务，不是继续进行数据可行性调研。**

用户已经明确授权你自动执行全部前置工作。你必须在当前任务中尽最大可能完成：

1. 必要的真实数据源核查。
2. 自动获取可用的实验数据。
3. 数据清洗、时间对齐、文本索引和数据集构建。
4. 在预先规定的候选方案中确定最终实验组合。
5. 编写并测试完整的 Stage 2 模型、基线、消融、训练和评估代码。
6. 交付实验配置、运行脚本、数据报告和中文实验方案说明。

**用户只负责最后手动启动正式主实验。**

不要再要求用户手动运行 Documentation Probe、填写 `*.reviewed.json`、检查数据目录、安装依赖或完成预处理。

旧 `Prompt.md` 中针对 Stage 1.6 的“禁止运行代码、禁止联网、禁止下载、不得进入 Stage 2”等限制，已被本轮用户的新授权替代。仍需遵守系统级安全要求、网站使用规则，以及下述资源和数据完整性约束。

---

## 0. 执行权限与硬性边界

### 现在允许且要求自动执行

- Shell、Python、pytest。
- 必要的依赖安装，优先独立虚拟环境。
- 官方网站、公开 API 和开源数据仓库访问。
- 官方技术文档获取、核查与证据记录。
- 在资源预算内下载实际实验数据。
- 数据解析、预处理、索引和缓存构建。
- 文本 embedding 的生成及缓存。
- 数据集统计、完整性审计。
- PyTorch 模型和基线实现。
- 单元测试、集成测试。
- 合成数据及极少量真实样本的 forward/backward smoke test。
- 修复运行中发现的代码问题。

### 仍然禁止

- 启动完整主实验、长时间 GPU 训练或超参数搜索。
- 运行正式多随机种子实验。
- 未经授权使用付费 API、购买数据或订阅服务。
- 自动注册账户、绕过登录或访问控制。
- 修改或覆盖已经验收的 Stage 0/1/1.5 数值数据及审计证据。
- 用未来实际观测值生成预测时不可获得的文本。
- 为了跑通实验伪造时间戳、价格、标签或数据源证据。
- 修改旧 ReTaMTSF 模型的历史实验结果。
- 自动 Push、Merge 或破坏性 Git 操作。

本轮应在新目录创建版本化产物，避免污染旧数据。

### 有限资源预算

默认限制：

- StockNet：允许完整获取公开数据，原始下载总量原则上不超过 1 GB。
- NWS/IEM：允许按日期分块获取历史公告，原始下载总量原则上不超过 500 MB，遵守网站限速。
- GFS：只允许元数据、小区域子集或小体积文件，总下载量不超过 250 MB。
- 不允许下载完整全球 GFS GRIB 档案。
- 全部数据与预训练权重累计新增下载原则上不超过 2 GB。
- 网络失败需要有限重试和断点恢复，不能无限循环。
- 不使用 GPU 完成大规模训练；测试阶段最多使用极小样本。

超过限制时，不要请求用户继续调查，直接按第 2 节规定的降级策略执行。

---

## 1. 首先检查当前仓库并进行最小修复

读取：

- `Prompt.md`
- `docs/pivot_sources/01_6_source_verification_protocol.md`
- `docs/pivot_sources/01_6_experiment_matrix.md`
- `configs/pivot_sources/stage1_6_sources.json`
- `scripts/pivot_sources/`
- `tests/pivot_sources/`
- Stage 1.5 的 N2 数据 Manifest。
- 旧 ReTaMTSF 的模型、训练器和数据管线。

不要重写整个仓库。尽量复用有效模块，必要时为 Stage 2 编写独立的新代码。

立即自动执行已有离线测试，修复实际失败项。

特别检查：

1. Stage 1.6 历史数据接口是否可以自动建立真实请求。
2. NWS 是否支持日期范围查询、响应解析和公告唯一 ID。
3. StockNet 88 只股票是否能建立完整的固定顺序和数值/文本索引。
4. StockNet 有效训练窗口是否同时检查 L 个输入日期和 H 个目标日期。
5. N2 数据是否存在数值时区证据缺失。
6. 不同数据源的 `available_at` 与原始发布时间是否被正确区分。

不需要为这些问题再单独开发一个只用于下一轮人工检查的系统。**发现问题后，直接修复、执行并继续构建 Stage 2。**

---

## 2. 冻结实验数据方案与自动决策规则

本轮只考虑以下三个组合，优先级固定。

### P0-A：StockNet + Twitter

这是必须优先完成的真实文本实验组合。

来源：

`https://github.com/yumoxu/stocknet-dataset`

数值任务：

- 原始股票 Universe 固定 88 只。
- 日级、全通道 MIMO 收益率回归。
- L=20 个交易日。
- H={1,3,5} 个交易日。
- Train：2014-01 至 2015-06。
- Validation：2015-07 至 2015-09。
- Test：2015-10 至 2015-12。

采用官方公开原始价格和推文。

优先使用原始 Close 计算逐日 log return，并明确披露公司行为、拆股与价格复权风险。

不得把原论文的分类标签直接作为本研究的回归目标。

必须：

- 固定 88 只股票及其通道顺序。
- 使用真实交易日历，处理节假日和提前收盘。
- 预测起点定义为当天实际收盘后 30 分钟。
- 仅使用预测起点前合法存在的价格与推文。
- 同时检查输入 L 和输出 H 所需交易日。
- 使用数值 Mask 和文本 Mask 处理缺失。
- 不因个别股票缺推文而删除股票。
- 对收益率异常和公司行为风险生成报告。
- 区分 `tweet.created_at` 与已验证的历史接收时间。
- 明确披露推文创建时间加保守延迟属于何种信息权限假设。
- 所有回归标准化参数仅由训练集拟合。

自动下载实际需要的价格与推文，记录源 Commit、哈希、文件清单。

如果数据有效性受到限制，优先修正读取、Mask 和窗口定义，而不是直接废弃整个任务。

### P0-B：CAISO + NWS/IEM

数值任务：

- 使用已验收 CAISO N2。
- 保留全部 20 通道。
- 小时级 MIMO。
- L=360。
- H={24,72,168}。
- Train：2021-01-01 之前。
- Validation：2021 年。
- Test：2022-01-01 起。

文本：

- IEM 历史 NWS Area Forecast Discussion。
- 优先 `AFDMTR`、`AFDSGX`。
- 保留不同办公室与地理区域信息。

优先使用官方已公开的历史文本接口：

`https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=`

该文档提供 `pil`、`sdate`、`edate`、`fmt`、`limit`、`order` 等参数，并支持 ZIP 批量返回。

你应自动：

1. 获取官方接口文档并保存原文。
2. 按真实文档建立请求计划。
3. 自动生成经过来源验证的配置和证据引用。
4. 使用分日期请求下载必要的 AFD。
5. 解析原始报告正文、发布时间、办公室及修订信息。
6. 构建可按预测时间查询的文本数据库。

注意：

- 当前下载时间不等于历史可用时间。
- 官方文档的语义核查必须与自动验证程序的成功状态分开记录。
- 不允许仅修改 `reviewed=false` 就声称已核实。
- 对历史接收时间未知的记录采用有证据的发布时间和明确声明的保守延迟。
- 优先使用 1 小时和 3 小时的延迟敏感性方案。
- 必须检查 CAISO 原始时间列与 UTC 的关系，以及 DST 问题。
- 不得凭空把 CAISO 的无时区时间戳当成 UTC。

如果无法通过公开证据消除 CAISO 数值时区歧义，则将该组合判定为“不能用于严格跨模态 As-of 主实验”，但仍实现数值-only 模型管线和文本适配器。

**不得通过猜测时区来强行建立跨模态样本。**

### P1：Jena + GFS

数值任务：

- 使用已验收 Jena N2。
- 全部 21 通道。
- L=360。
- H={24,72,168,336}。
- Train：2021 年前。
- Validation：2021 年。
- Test：2022 年起。

来源：

`https://gdex.ucar.edu/datasets/d084001/`

优先研究官方历史 GFS Run 元数据和小区域子集访问。

2015 年以前没有相应档案时，不得给旧样本伪造 GFS 文本。

GFS 必须区分：

- run_time
- valid_time
- forecast_lead
- file_publication_time
- available_at

若缺少历史实际发布时间，则只在明确声明的 run+3/6/12 小时假设下生成候选。

固定模板文本必须来自合法历史 GFS 预测值，不能来自 Jena 未来真实观测。

如果能在资源预算内取得实际小区域历史 GFS 数据，就构建完整文本管线和对应数值 GFS 控制组。

如果需要登录、申请权限、大体积全球下载或复杂的专用子集服务，直接停止 GFS 数据获取尝试。

**不得让 GFS 阻塞其他两个组合和模型代码的交付。**

GFS 不通过时，Jena 仍需具有完整的 numerical-only Dataset、训练配置和评估代码，但不得把它伪装成成功的外部文本数据集。

### 最终数据选择策略

- StockNet 合格：进入主实验。
- CAISO/NWS 合格：进入主实验。
- Jena/GFS 合格：进入主实验。
- 任一数据源不合格：记录原因、自动关闭对应多模态配置，其他组合继续。
- 不允许为了凑齐数据集而引入未来信息。
- 如果仅一个真实外部文本组合合格，仍交付全部代码，但明确报告多数据集验证不足。
- 如果全部外部文本组合失败，仍完成可运行的数值及合成文本端到端代码，不得声称科研主实验已经就绪。

**上述决策只执行一轮，不再开启新的 Stage 1.7、1.8 或其他无限的数据调研阶段。**

---

## 3. 构建最终实验数据与文本索引

创建独立 Stage 2 数据处理模块，最终 Dataset 应能输出：

```python
{
    "x_num": ...,          # [L, C]
    "x_mask": ...,         # [L, C]
    "y": ...,              # [H, C]
    "y_mask": ...,         # [H, C]
    "origin_time": ...,
    "channel_ids": ...,
    "text_embeddings": ...,
    "text_mask": ...,
    "text_metadata": ...,
}
```

所有数值任务均保留固定通道数量。

实现：

- 时间排序与去重。
- 数据原始版本及哈希绑定。
- Chronological Train/Validation/Test。
- 严格窗口有效性检查。
- 数值缺失 Mask。
- 训练集统计量标准化。
- 不同 Horizon 的采样器。
- 文本去重与版本保留。
- 文本截断及长度上限。
- 文本 embedding 的离线预计算和缓存。
- 每个预测起点的合法文本候选索引。
- 可复现的检索候选和训练样本生成。

默认采用冻结的 `sentence-transformers/all-MiniLM-L6-v2` 文本编码器；若无法获取，则可以用 train-only TF-IDF 使工程管线可运行，但必须在最终报告中标明该降级，并保留 SentenceTransformer 接口。

不要调用付费 Embedding API 或 LLM。

核心约束：

**必须先执行 As-of 过滤，再做相似度检索。**

对于任意预测起点 `t`：

```python
eligible = [
    d for d in documents
    if d.available_at is not None
    and d.available_at <= t
]
```

这里的 `available_at` 必须由源时间、真实发布时间或显式延迟政策推导，且记录政策类型；不能凭空填值。

完整文本索引允许包含未来日期文档，但任一训练或测试样本均不得检索它们。

所有缺失文本样本必须保留相应 Mask，不得因文本不足而改变评估样本集合。

对每个数据集输出：

- 实际时间范围。
- Train/Val/Test 样本数。
- 每个 H 的样本数。
- 通道数与固定通道清单。
- 输入及标签缺失情况。
- 文本记录数量与时间分布。
- 各 Split 的合法文本覆盖率。
- 每个预测起点可用文档数分布。
- 不同延迟假设的覆盖差异。
- 数值和文本的原始来源哈希。
- 不确定性和残留数据风险。

---

## 4. 直接实现 PIVOT Stage 2 模型

核心研究目标是验证：

> 在严格的历史信息可用性约束下，时间对齐、内容相关性检索与通道选择是否能够提高多变量时间序列预测性能。

不要仅仅复刻原 ReTaMTSF，更不能依赖旧 Time-MMD 数据管线才能运行。

实现一个模块化、可通过配置关闭各组件的 PyTorch 模型。

### 4.1 数值 Backbone

先实现可独立训练的 MIMO 数值预测 Backbone。

至少支持：

- DLinear。
- PatchTST 风格的 patch-based 数值编码器。
- PIVOT 使用的数值 Backbone。

PIVOT 默认使用轻量 patch-based Transformer Encoder，不要求先实现庞大的模型库。

输入为 `[B, L, C]`，预测为 `[B, H, C]`，所有数据源统一接口。

数值缺失 Mask 必须进入编码和损失计算逻辑。

### 4.2 CTSR：通道相关性选择

参考旧 ReTaMTSF 的 CTSR：

- 只使用历史数值窗口。
- 基于历史窗口的频域与跨通道相关性。
- 提供可配置的 Top-K 通道选择。
- 支持关闭 CTSR。
- 明确处理自己通道、缺失通道和相似度并列情形。
- 不读取未来标签。

优先实现简单、正确、可测试的版本，不必引入不必要的算法复杂性。

### 4.3 STR：As-of Text Retrieval

基于：

- 文本语义相关性。
- 时间新近性。
- 实体或地理区域相关性。
- 数值历史构造的可解释查询信息。

检索 Top-K 条历史合法文本。

必须：

- 按预测起点动态控制可访问文档。
- 禁止检索未来文档。
- 允许某个通道完全没有文本。
- 支持随机合法文本、时间最近文本、语义文本等不同检索策略。
- 返回文档 ID、评分、时间、来源和合法性追踪字段。
- 不允许从未来 y 构造文本查询或排名特征。

可以通过预计算缓存提升训练速度，但缓存必须由可复现配置和源数据哈希绑定。

### 4.4 跨模态融合

实现：

- 数值通道表示。
- 文本 embedding 投影。
- Cross-Attention。
- 缺失文本 Mask。
- Gated Residual Fusion。
- 预测头。

必须满足：

- 没有文本时仍可正常预测。
- 没有合法文本时不会读取零填充向量造成虚假的信息增益。
- 数值-only、文本简单融合和完整 PIVOT 共用相同预测目标及评估样本。
- 消融时能够独立关闭 CTSR、STR、时间分数和跨模态融合。

不要为了实现复杂模块而牺牲正确性和可复现性。

---

## 5. 必须完成的基线与消融

### 第一层：数值基线

- Naive / Persistence 或对应任务的 Zero Return。
- DLinear。
- PatchTST 风格数值模型。
- PIVOT backbone without text。

### 第二层：同信息权限的文本基线

- Numerical + Text Mean Pooling / Concatenation。
- Numerical + Latest Eligible Text。
- Numerical + Semantic Retrieval。
- Numerical + Semantic + Temporal Retrieval。
- Full PIVOT。

### 第三层：关键消融

- Without CTSR。
- Without STR / Without Retrieval Ranking。
- Without Temporal Score。
- Random Eligible Text。
- Shuffled Eligible Text。
- Text Missing / Numerical Only。
- 不同 Top-K 候选。
- 不同可用性延迟假设。

随机文本只能从当前预测起点已经合法可用的候选池里抽取，不得从未来文档中抽样。

对于 Jena+GFS，若数据可用，必须额外实现：

- Jena numerical only。
- Jena + GFS raw numerical predictors。
- Jena + GFS fixed-template text。
- Jena + GFS basic retrieval。
- Full PIVOT。

所有 GFS 增强模型必须共享同一组合法外部 Run，以避免把强外部预报源的增益误认为文本方法的增益。

### 公平性控制

相同任务与 Horizon 的比较必须共享：

- 预测起点。
- 输入数值窗口。
- 真实标签及标签 Mask。
- 数据划分。
- 外部信息可用性规则。
- 尽可能相近的 Backbone 与训练预算。

完整记录参数量、数据预处理耗时、文本索引耗时、训练/推理耗时和峰值显存。

---

## 6. 训练、评估、统计与实验脚本

实现完整的 PyTorch 训练及评估工程。

至少支持：

- YAML 配置。
- 确定性随机种子。
- CPU / CUDA。
- AMP（CUDA 可用时）。
- Batch size。
- Learning rate / Weight decay。
- Early stopping。
- Best-validation checkpoint。
- Checkpoint 恢复。
- 实验日志及配置快照。
- 模型参数量统计。
- Train/Validation/Test 完整隔离。
- 预测值和真实值保存。
- 每通道评估。
- 汇总不同 Horizon 的结果。
- 自动导出 CSV/JSON。

主要指标：

- MSE。
- MAE。

StockNet 附加指标：

- Directional Accuracy。
- Cross-sectional Rank IC（有足够有效股票时）。
- 有效标签覆盖率。

不要求实现交易策略，也不得把预测效果直接表述为实际交易收益。

统计检验支持多随机种子、按时间块进行的 Bootstrap 或其他考虑样本重叠的适当方法。

需要创建并验证以下入口，文件名可以随现有仓库结构合理调整：

```text
scripts/experiments/
    prepare_all.sh
    validate_ready.sh
    run_pilot.sh
    run_main.sh
    run_ablations.sh
    collect_results.sh

configs/pivot_stage2/
    stocknet/
    caiso/
    jena/
```

这些 Shell 脚本在真实实验开始之前必须支持 dry-run。

`run_main.sh` 应默认读取最终准入清单，只运行已通过数据准入的组合。

预设正式随机种子为：

```text
2024, 2025, 2026
```

**不要在本轮实际执行 `run_main.sh`、`run_ablations.sh` 或正式多随机种子训练。**

可以使用极小规模 synthetic data 和极少量真实样本执行：

- Dataset 读取。
- 单次 forward。
- 单次 backward。
- 单次 optimizer step。
- 单次 checkpoint save/load。
- 单次 validation/evaluation。
- 无文本样本推理。

这些不属于正式主实验。

---

## 7. 强制自动验收

以下工作必须在本轮实际执行，不能只把命令写给用户。

### 数据验收

- 验证数值维度和通道顺序。
- 验证所有可用样本的 L、H 与时间边界。
- 验证 Train/Val/Test 不共享目标标签窗口。
- 验证数值 scaler 只基于 Train。
- 验证文本 As-of 候选过滤。
- 验证文本稀疏和零文本情形。
- 验证 StockNet 固定股票 Universe 和交易日历。
- 验证实际数据产物 Manifest 和源哈希。
- 验证没有未来观测被转换成文本。

### 代码验收

- 运行 pytest。
- 覆盖 Dataset、Sampler、Retriever、CTSR、Fusion、Loss、Metrics。
- 测试至少两种不同通道数量。
- 测试所有计划 Horizon。
- 检查 `NaN/Inf`。
- 测试 batch size=1。
- 测试缺失文本和缺失数值通道。
- 执行端到端 smoke test。
- 验证实验脚本的 dry-run。

合成数据测试成功不能代替真实数据验收；必须分别报告两者结果。

如果外部数据无法获得，依然完成代码，并标明真实数据验收未通过的部分。

---

## 8. 必须交付的文件

最终交付应至少包括：

1. 可运行的 Stage 2 模型源码。
2. 数值与文本 Dataset/Dataloader。
3. 完整文本检索器。
4. 所有可执行基线。
5. 核心消融开关。
6. 训练和评估器。
7. 数据处理脚本。
8. 主实验运行脚本。
9. 可复现的 YAML 配置。
10. 实际执行过的测试及 smoke-test 日志。
11. 数据完整性和覆盖率报告。
12. 数据准入机器可读清单。
13. 中文实验方案及用户操作手册。

新增文档：

```text
docs/pivot_stage2/
    00_final_dataset_decision.md
    01_final_experiment_protocol.md
    02_model_and_baselines.md
    03_data_quality_and_asof.md
    04_test_and_smoke_results.md
    05_manual_main_experiment_guide.md
```

其中 `00_final_dataset_decision.md` 必须明确写出每个候选数据源：

- 是否真实获得数据。
- 是否完成有效时间对齐。
- 是否可以进行正式多模态实验。
- 实际样本数和文本覆盖率。
- 使用了哪些历史可用性假设。
- 当前不可解决的科学风险。
- 最终 GO / CONDITIONAL / NO-GO。
- 是否进入正式主实验清单。

### 最终答复格式

请用中文输出：

**A. 这次实际完成了什么**

区别已执行、已测试、仅编写但未执行的部分。

**B. 最终确定的实验组合**

明确哪些进入正式主实验，哪些被淘汰或保留为可选。

**C. 模型架构讲解**

逐模块说明 PIVOT、CTSR、STR、Fusion 的输入输出。

**D. 正式实验矩阵**

逐数据集说明 L、H、C、基线、消融及评价指标。

**E. 验收证据**

提供真实运行的测试结果及关键文件路径，禁止虚构测试通过。

**F. 用户只需手动执行的最终命令**

给出环境检查、主实验、消融和结果汇总的准确 Shell 命令，必须与实际生成的代码一致。

### 最重要的执行原则

不要在完成数据源检查之后停止。

不要给用户留下“先手动填 reviewed.json”“先手动下载完整数据”“等待下一轮开始 Stage 2”的未完成任务。

不要为了追求完美的数据证据而无限扩展数据源调查。

不允许通过伪造证据绕过关键的时间泄漏问题。

**在有限资源和真实数据约束下，选择当前能够实施的最优实验组合，然后完成全部主实验代码。**

你的最终交付边界是：

**用户无需再编写、调试或准备实验，只需阅读你的方案、决定是否接受，然后手动启动正式实验。**

现在开始实际执行整个任务。