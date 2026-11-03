# PIVOT 项目重构 — Stage 1：Fidel-TS 数据准备与来源审计工具

你是一名具有时间序列预测、多模态学习、PyTorch 工程和科研可复现性经验的研究工程师。

请基于当前本地仓库，逐步将原 ReTaMTSF 项目重构为新的研究项目 PIVOT，主要目标期刊为 Information Sciences。

仓库：
https://github.com/ElliotMars/ATM-ReTaMTSF

**本次只实施 Stage 1。完成代码修改之后必须停止，不允许自行进入下一阶段。**

## 一、最新研究目标

原模型名称 ReTaMTSF 不再用作新论文方法名，新方法暂定为 PIVOT。

新版方法计划包括：

1. 保留 Coherence-based Numerical Retrieval。
2. 保留数值 Content Synthesis。
3. 在原 STR 基础上增加 Temporal Relevance。
4. 增加 Channel- and Horizon-conditioned Predictive Utility Reranking。
5. 保留主要 Cross-Attention 融合机制。
6. 将原单目标预测改成多变量输入、多变量输出（MIMO）。
7. 以 Fidel-TS 为主要 Benchmark。
8. 所有新 Baseline 必须在统一协议下实际运行。

这只是后续研究目标。

**本阶段禁止实现 MIMO、Temporal Scorer、Utility Scorer 或训练代码重构。**

## 二、最重要的执行限制

我本人负责执行所有需要运行的任务，你主要负责修改代码。

### 你可以做

- 阅读本地文件。
- 使用只读命令查看文件、Git 状态与代码结构。
- 编写 Python 模块。
- 修改必要的配置文件。
- 编写测试代码。
- 编写 Shell 脚本。
- 编写数据检查工具。
- 编写实验说明和文档。
- 静态检查代码逻辑，但不执行代码。

### 你不可以做

- 不运行任何 Python 脚本。
- 不运行 pytest、unittest 或 Smoke Test。
- 不运行任何模型 Forward/Backward。
- 不下载数据或预训练模型。
- 不安装 Python 包。
- 不执行数据预处理。
- 不运行训练和评测。
- 不进行任何 GPU 计算。
- 不访问远程 API 下载文件或元数据。
- 不自动启动后台任务。
- 不自动 Commit、Push 或 Merge。
- 不自动运行生成的检查命令。

即使某个测试预计只需几秒，也必须交给我手动执行。

**你的任务是准备好代码及准确的命令，而不是替我运行。**

如果本地缺少依赖，直接记录依赖，不要尝试安装。

以上规则优先于仓库现有 `Prompt.md` 中允许自动执行轻量测试的旧规定。

## 三、首先阅读已有审计结果

在修改代码前阅读：

- `docs/fidel_ts/00_repository_audit.md`
- `docs/fidel_ts/00_audit_evidence.json`
- `Prompt.md`
- `README.md`
- `data_provider/data_loader.py`
- `data_provider/data_factory.py`
- `models/ReTaMTSF.py`
- `layers/ReTaMTSFLayers.py`

重点理解阶段 0 的以下结论：

1. CAISO 和 Jena 都可以构建多变量预测任务。
2. 原模型目前实际上只支持单目标输出。
3. 原模型的部分 Linear 在 Forward 中临时创建。
4. 原始数据加载存在条件性 Scaler 列顺序问题。
5. Jena 包含 -9999 哨兵值和可能的时间网格缺口。
6. CAISO 数值数据可能存在需要核验的插值和时间对齐问题。
7. Fidel-TS 天气文本缺少独立的历史发行/可用时间证据。
8. Jena v1/v2/v3 不能直接认定为按照时间发布的三个历史版本。

不得重新从零开始完成 Stage 0。

如果 Stage 0 的某些信息与当前代码冲突，以实际仓库为准，记录差异。

不要修改已有 Stage 0 证据文件。

## 四、Fidel-TS 固定数据来源

本阶段只考虑两个数据集。

### California ISO

Hugging Face：

`fidel-ts/California_ISO`

固定 Revision：

`75f74f556edd5fbdd080c4db6d4b7547fc331f39`

首选数值数据：

`time_series/all_except_battery.parquet`

需要支持获取：

- 数值 Parquet。
- `id_info.json`。
- `static_info.json` 等相关元数据。
- San Diego 原始 weather_report。
- San Francisco 原始 weather_report。
- 必要的来源审计文件。

不要默认使用已经合并的天气文本，因为 Stage 0 发现 merged general report 可能丢失地点身份。

### Jena Atmospheric Physics

Hugging Face：

`fidel-ts/Jena_Atmospheric_Physics`

固定 Revision：

`c8da24175c7aabe5c3b02b503e4c5afe44a605fd`

首选数值数据：

`time_series/weather_large.parquet`

需要支持获取：

- 数值 Parquet。
- `id_info.json`。
- `static_info.json`。
- weather_report 的 formal/general 文本。
- 必要的来源审计文件。

优先获取 JSON 原文，不默认下载 `.pkl` Embedding。

两组数据的正式历史时间范围、文件路径和数据规模，以已固定的 Stage 0 证据及下载后校验结果为准。

不得根据文件名虚构数据的实际时间覆盖。

## 五、任务 A：实现数据下载工具

建议新增：

`scripts/fidel_ts/download.py`

`configs/fidel_ts/sources.json`

下载工具至少提供四种操作：

### 1. List

获取或使用本地 Manifest 展示固定 Revision 下的数据文件列表。

输出：

- Dataset。
- Repository。
- Revision。
- 文件相对路径。
- 文件大小。
- 选定下载 Profile。
- 本地是否存在。

### 2. Dry Run

展示将下载哪些文件、预计总字节数、最终保存位置。

不修改文件，不实际下载。

### 3. Download

由用户手动运行。

要求：

- 支持指定 Dataset。
- 支持指定输出目录。
- 使用固定 Revision。
- 保留原相对路径。
- 支持已有文件检查。
- 尽可能支持断点续传。
- 下载失败不能把不完整文件当作成功。
- 不允许覆盖其他数据集。
- 不允许把不同 Revision 混入同一个数据目录。
- 保存实际文件大小和 SHA-256。
- 记录失败和重试信息。

### 4. Verify

由用户手动运行。

校验：

- 文件是否存在。
- 文件大小是否正确。
- 已记录的哈希是否匹配。
- Revision 是否与配置一致。
- Manifest 是否完整。
- 是否存在意外覆盖或混合版本。

不需要设计复杂的新下载框架。

优先使用 `huggingface_hub` 等常规工具，并确保错误信息清楚。

不要反序列化未核验的 Pickle 文件。

## 六、任务 B：实现数据结构检查工具

建议新增：

`scripts/fidel_ts/inspect.py`

工具需要支持 CAISO/Jena。

### 数值数据检查

检查以下内容：

- 数值行数。
- 时间起点和终点。
- 时间戳重复。
- 采样间隔分布。
- 不规则时间间隔。
- 数值通道数量。
- 各通道名称。
- 缺失值数量。
- NaN/Inf。
- Jena 的 -9999 哨兵值。
- 不同通道数值范围。
- CAISO 相关输入的异常负值。
- 可能的重复数据。
- 数值与静态通道描述之间的映射。

对于 CAISO，重点核查正式配置使用的 20 个通道。

对于 Jena，重点处理 Stage 0 已记录的 Parquet 列名与静态描述不一致问题。

**本阶段只报告问题，不自动修正异常值，不执行插值，不重采样。**

### 文本数据检查

读取天气报告 JSON，统计：

- 文档数量。
- 时间键分布。
- 记录中的字段名称。
- 文本长度。
- 重复时间键。
- 同时间键多条文本。
- Location。
- Subject。
- Report Type。
- File Version。
- 可能的预测覆盖区间。
- 是否存在明确发行时间字段。
- 是否存在明确可用时间字段。
- 不同文本版本之间的差异。

每条文本应具有确定的来源身份。

建议使用如下内部字段：

`document_id`

`dataset`

`subject_id`

`location_id`

`report_type`

`source_file`

`source_time_key`

`text`

`issued_at`

`available_at`

`valid_start`

`valid_end`

`version_id`

允许 `issued_at`、`available_at`、`valid_start`、`valid_end` 为 `None`。

**没有证据的时间字段必须保持未知，绝不允许用 source_time_key 自动填充 available_at。**

Jena v1/v2/v3 在没有独立发行日志之前，应视为不同文本文件版本，而不是已验证的在线版本序列。

### 时间可用性审计

新增字段或报告状态：

- `verified`
- `protocol_assumption`
- `unknown`

这里的 verified 必须有明确的原始证据。

不能仅根据官方文章声称严格因果，就把每条发布文本标为 verified。

如果只能获得时间键和天气描述，状态应保持 unknown。

当正式 As-of 模式需要 `available_at` 而该值未知时，后续适配器应默认拒绝将其作为已验证可用文本。

本阶段只定义该规则和检查报告，不实现正式检索。

## 七、任务 C：生成机器可读审计报告

建议输出：

`artifacts/fidel_ts/data_audit/`

包含：

- `manifest.json`
- `numerical_summary.json`
- `text_summary.json`
- `availability_summary.json`

同时新增：

`docs/fidel_ts/01_data_download_and_provenance.md`

报告必须区分：

1. 已经由 Stage 0 验证的事实。
2. 本阶段代码准备支持的检查。
3. 等待用户下载后才可验证的内容。
4. 尚无证据的时间来源问题。
5. 会阻止正式多模态训练的问题。

不得在没有运行数据检查工具的情况下填写虚构的统计结果。

## 八、任务 D：编写测试代码，但不要执行

新增适当的测试文件，例如：

`tests/fidel_ts/test_download.py`

`tests/fidel_ts/test_inspect.py`

`tests/fidel_ts/test_manifest.py`

使用少量代码内构造的 Mock 数据或 Fixture。

至少覆盖以下情形：

1. 固定 Revision。
2. 拒绝不同 Revision 混合。
3. 文件已存在时的处理。
4. 下载不完整时的处理。
5. Manifest 和 SHA-256 校验。
6. 重复时间戳。
7. Jena -9999 哨兵值。
8. 不一致的 Channel 名称。
9. 同一个日期多个 Location。
10. 文本没有 available_at。
11. Jena v1/v2/v3 不被误判为时间修订记录。
12. JSON 记录字段缺失。

如果需要外部依赖，记录安装命令，但不能执行。

测试代码必须保证默认使用本地 Fixture，不主动访问网络，也不下载真实数据。

你只编写测试，不运行。

## 九、文件修改边界

本阶段优先新增：

- `scripts/fidel_ts/download.py`
- `scripts/fidel_ts/inspect.py`
- `configs/fidel_ts/sources.json`
- `tests/fidel_ts/`
- `docs/fidel_ts/01_data_download_and_provenance.md`

可以新增必要的公共辅助模块。

允许对 `.gitignore` 做最小修改，避免提交大型数据、下载缓存和运行产物。

本阶段不要修改：

- `models/ReTaMTSF.py`
- `layers/ReTaMTSFLayers.py`
- 原 Time-MMD Dataloader。
- 原 Time-MMD 实验脚本。
- 原有 Baseline。
- 旧 Checkpoint。
- 原始实验结果。

不要整体重命名仓库，不要批量替换 ReTaMTSF 字符串。

新方法名称 PIVOT 的正式代码迁移留到后续阶段。

## 十、交给我手动执行的命令

完成代码后，提供一组真实、可直接复制的命令。

至少包含：

1. 所需依赖的安装命令（仅在需要时）。
2. CAISO 文件清单命令。
3. Jena 文件清单命令。
4. CAISO Dry Run。
5. Jena Dry Run。
6. CAISO 下载。
7. Jena 下载。
8. 数据文件校验。
9. 数据结构审计。
10. 单元测试。
11. Git 修改检查。

所有命令必须和你实际实现的 CLI 参数一致。

不能提供尚未实现的参数。

说明建议执行顺序、输入目录、输出目录、预期产物和成功标准。

不允许自行执行其中任何命令。

## 十一、完成后的回复格式

请按以下顺序回复：

### 1. 本阶段修改概述

说明本次只完成了哪些代码工作。

### 2. 修改文件清单

逐项说明所有新增、修改文件。

### 3. 核心实现

简述下载、数据检查、Manifest 和时间来源状态如何处理。

### 4. 测试状态

明确写：

“测试代码已编写，但按照用户要求未执行。”

不要将未运行的测试标注为 PASS。

### 5. 用户手动命令

给出完整、准确且可以复制的命令。

### 6. 预期结果和验收标准

说明每个命令执行成功后应出现什么。

### 7. 未解决问题

尤其说明：

- 天气发行时间是否仍然未知。
- 数值数据质量哪些问题尚未由真实文件核验。
- 目前是否允许进入正式多模态训练。

### 8. 下一阶段建议

下一阶段预期是 Fidel-TS 多变量数据加载器，但只描述计划，不实施。

## 十二、现在开始

现在只执行 Stage 1。

先根据已有 Stage 0 审计了解仓库，随后编写下载工具、数据审计工具、配置、测试与文档。

不得下载任何文件，不得运行任何测试，不得执行训练，不得安装依赖，不得推送远端。

**完成修改后停止，等待我手动运行命令并提供结果。**