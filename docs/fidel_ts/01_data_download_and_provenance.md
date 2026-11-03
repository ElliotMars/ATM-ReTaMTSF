# PIVOT Stage 1：Fidel-TS 数据准备与来源审计工具

准备日期：2026-11-02。当前工作区基准提交为 `6e2a9443e9d224a15e426b389d11b27fba1bae43`。

**本阶段只编写工具、配置、测试和本文档。没有运行 Python、安装依赖、测试、下载、远程 API 请求、数据处理、训练或 GPU 计算。测试代码已编写，但按照用户要求未执行。以下统计示例均不是本阶段运行结果。**

本项目后续研究名称为 PIVOT、目标期刊为 Information Sciences。Stage 0 保留其历史名称和结论；不改写历史证据。MIMO、Temporal Relevance、Predictive Utility Reranking、模型修复和训练重构均不在本阶段实施。

## 1. 证据分层与未解决事项

| 层级 | 内容 | 当前状态 |
| --- | --- | --- |
| Stage 0 已核实 | 两个固定 Revision 的实际文件清单；首选数值文件的页脚行数、列名、范围；部分天气前缀和官方预处理源码 | 仅引用已有 [审计报告](00_repository_audit.md) 与 [证据 JSON](00_audit_evidence.json)，没有重新联网调查 |
| 本次准备的检查 | 文件身份/大小/哈希；完整本地数值扫描；重复键、地点、文件版本、字段、文本长度、时间来源审计 | 代码已编写，尚未运行，不能声明通过 |
| 等待真实文件核验 | 缺失/哨兵比例、重复和不规则时间戳、负值位置、通道映射、文本覆盖和各版本差异 | 必须由用户下载后运行工具 |
| 仍无证据 | 历史 forecast 的 issued_at、available_at、valid 区间、时区及 vintage 生成来源 | 不用文件名、source_time_key、下载时间或论文概括补齐 |
| 正式多模态训练 | 历史可用性证据不足，以及旧模型/旧加载器已记录的正确性问题 | **当前不能放行**；审计 `_SUCCESS.json` 不表示训练协议通过 |

当前源码仍存在 Stage 0 记录的单目标输出、forward 局部 Linear 和条件性 Scaler 列顺序问题；本阶段没有改动这些文件。Stage 0 所建议的单目标首轮方案属于旧阶段设计，后续数据加载器应按本次 Prompt 的新 MIMO 研究方向另行设计，而不是声称模型已支持 MIMO。

固定来源：

| Dataset 参数 | Repository | Revision | 首选数值文件 / Subject |
| --- | --- | --- | --- |
| `caiso` | `fidel-ts/California_ISO` | `75f74f556edd5fbdd080c4db6d4b7547fc331f39` | `time_series/all_except_battery.parquet` / `all_except_battery` |
| `jena` | `fidel-ts/Jena_Atmospheric_Physics` | `c8da24175c7aabe5c3b02b503e4c5afe44a605fd` | `time_series/weather_large.parquet` / `weather_large` |

Stage 0 页脚记录：CAISO 首选文件 22,264,375 字节、724,896 行、20 个数值通道；Jena 23,421,608 字节、604,311 行、21 个数值通道。这里只是先前证据，不是本阶段全量检查结果。不能从年度文件名推断实际时间覆盖。

## 2. 文件组织与依赖

新增配置 `configs/fidel_ts/sources.json` 定义两个来源、固定 SHA、首选文件、时间列、原始采样间隔、静态 Subject 候选、Jena 显式列名别名及三个 Profile：

| Profile | 范围 | 用途 |
| --- | --- | --- |
| `numerical` | 首选 Parquet、README、ID 和静态 JSON | 先做数值准备；不能假装已经审计天气 |
| `standard`（默认） | numerical + 原始 formal/general weather_report JSON + 来源脚本 | 本次推荐的数据准备范围 |
| `provenance` | standard + 原始/已插值数值输入；CAISO 两地 raw weather JSON/CSV | 深入追溯来源与时间轴；先 dry-run 查看额外大小 |

所有 Profile 都不选择 Pickle，也不默认选 CAISO merged weather report/merged Embedding。下载的来源脚本只保存为证据，工具不会 import 或执行它们。Jena `weather_report/general_report/static_info.json` 作为元数据保留，但不会被误计为天气记录。

工具只需 Python >=3.10、huggingface_hub、NumPy、pandas、PyArrow；测试另需 pytest。依赖边界见 `scripts/fidel_ts/requirements-stage1.txt`。没有 PyTorch、LLM 或文本编码依赖。当前依赖是否可用本阶段未探测。

目录约定：

```text
dataset/Fidel-TS/
  .hf-cache/                         # Hub 缓存，保存可续传的下载
  raw/
    California_ISO/
      .fidel_ts_identity.json        # Dataset/Repository/Revision 身份
      manifest.json                 # 下载状态、选集、预期大小、实测哈希和重试记录
      time_series/...               # 原相对路径保留
      weather/san-diego/...
      weather/san-francisco/...
    Jena_Atmospheric_Physics/
      .fidel_ts_identity.json
      manifest.json
      time_series/...
      weather/weather_report/...
artifacts/fidel_ts/
  inventories/{caiso,jena}.json       # 用户手动 List 保存的远程元数据
  data_audit/{caiso,jena}/            # 每次检查必须指定全新的目录
    manifest.json
    numerical_summary.json
    text_summary.json
    availability_summary.json
    documents.jsonl
    _SUCCESS.json                   # 只说明审计完成
```

原始数据、缓存、运行产物、专用虚拟环境和新模块的 Python 缓存已加入 `.gitignore`。本阶段不会创建上述数据或运行产物目录。

## 3. 下载工具的行为

入口：`python scripts/fidel_ts/download.py OPERATION --dataset caiso|jena ...`。

- `list`：默认通过 `huggingface_hub.HfApi.list_repo_tree` 读取固定 SHA 的完整元数据，展示所有文件的 Dataset、Repository、Revision、路径、大小、Profile、是否选中和本地存在情况。只有用户运行才请求网络。`--save-inventory` 保存可复用的清单，不下载数据；不覆盖已有清单。
- `--stage0`：让 list/dry-run 从本地 Stage 0 证据读取清单。没有记录的大小为 null，不虚构预计总量；Download 要求使用 Stage 1 清单，不接受此选项。工具不读取 Stage 0 的片段 hash 作为整个大文件的 hash。
- `--inventory PATH`：使用已保存的 Stage 1 清单。List 和 Dry Run 不请求远程元数据；Download 仍可能为实际文件使用网络。
- `dry-run`：只输出选中文件、已知字节/未知大小计数、预计总量和最终位置，不创建目录或修改文件。禁止与 `--save-inventory` 同用。
- `download`：仅接受固定来源 SHA；不支持随意换 main。依赖 Hub 缓存的常规续传机制，取到缓存文件后复制到目标同文件系统的 `.part`，校验大小及可用的源 hash，再用不覆盖的原子 hard link 发布。发布前不会标记 complete。不支持 hard link 的文件系统会报错并保留失败状态，不退化为非原子覆盖。
- `verify`：完全离线，检查身份、清单摘要、选定 Profile 是否被覆盖、每个文件状态/大小/实测 SHA-256、可用源 hash，以及意外文件/符号链接。返回 `ok:true` 仅表示 Manifest 所声明选集完整且当前文件匹配。

对于 LFS，记录并校验源 SHA-256；普通 Git 文件核对 Git blob SHA-1，同时始终记录本地 SHA-256。若本地清单没有独立源 hash，实测 hash 只用于后续完整性检查，不是对来源真实性的新证明。inventory 的文件集合还必须与固定 Stage 0 文件树一致，不能删去文件后仍称完整清单。选集、Profile、完整 inventory 及其摘要保存在下载 Manifest；不把仅有部分文件的失败下载当成成功。

已有文件规则：完整且匹配的文件只校验、不覆盖；大小或 hash 不匹配则停止。非空无身份目录拒绝接管，错误 Dataset/Revision 拒绝混入。若文件已经原子发布但上一次 Manifest 写入被中断，仅在固定来源的独立内容 hash 可验证时恢复登记；否则要求人工调查。

同一快照可从 numerical 扩展到 standard/provenance，Manifest 追踪 Profile 的并集；相同 Revision 的不同 inventory 内容不会静默合并。每次请求失败的时间、错误、尝试号记录在对应文件的 attempts 中；默认额外重试两次。并发写入通过排他锁拒绝。缓存之外的原始文件不会自动删除或修复。

## 4. 数据检查工具的行为

入口：`python scripts/fidel_ts/inspect.py --dataset ... --input-root ... --output-dir ... --inspect-only`。`--inspect-only` 为显式只读数据声明；即使不传，工具也没有清洗、插值或重采样路径。

工具先离线校验快照与所需 Profile，再执行本地扫描；结束前再次校验，避免把变化中的输入当作固定快照。输出目录必须不存在、且与数据目录分离；不会覆盖旧成功/失败结果。失败后尽可能保存 `failure.json`，没有 `_SUCCESS.json` 的目录不能视为完整审计。

### 数值

扫描首选 Parquet 的真实行数、时间范围、原始顺序相邻间隔分布、非递增和不规则间隔、重复时间戳，以及排序去重时间轴上的 gap；后者仅为诊断视图，不改原表。不推断未提供的时区。

逐通道记录 Arrow null、包含 null 的 NaN、非 null IEEE NaN、正负 Inf、-9999 计数、有限值范围、排除 -9999 后的描述性范围、负值计数。CAISO 的负值只标记为需核查，不把所有负值认定为错误或截零。记录完全重复行与重复数值向量，并说明常量读数可能合法。

配置列出 CAISO 正式 20 通道及 Jena 21 通道的期望名称，报告缺失/额外列。Jena 的四个历史名称差异仅用于**描述映射**：保留 raw_channel_id，不自动重命名 Parquet。缺失或多对一描述映射会显式报告，不模糊猜测。

CAISO provenance 输入齐全时，额外比较三张 `all_channel_imputed` 表与首选数值文件的 Time 轴行数与顺序是否严格相等；若本次只选 standard，则报告未检查及需要哪些文件，不偷偷读取选集之外的输入。即使时间轴相等，也不代表已证明插值因果性。Jena 数值清洗出处仍明确标为未证实。

### 文本与来源身份

按文件逐个解析 JSON，使用保留对象 pairs 的解析方式保留重复时间键和重复 facet 字段。根不是时间键对象时报告错误，不偷偷套格式。无文本、非对象记录、缺失预期字段、无法解析的时间键都写入诊断。

每个字符串 facet 为一条审计 document，写入 `documents.jsonl`，不做编码或检索。document_id 由 Dataset、Revision、相对文件、原始时间键、同键出现次数、字段路径及同字段出现次数确定；文本另有 SHA-256。因此两个地点同日不会冲突，重复键也不会被覆盖，文本变更可与来源身份分开追踪。

保留字段：`document_id`、`dataset`、`subject_id`、`location_id`、`report_type`、`source_file`、`source_time_key`、`text`、`issued_at`、`available_at`、`valid_start`、`valid_end`、`version_id`；另有 facet JSON pointer、来源 Revision 和字段证据位置。统计 record 数和 document 数时明确区分，不能把一条七 facet 的报告误当作七次独立发行。

只从记录中的明确字段提取时间字符串；没有字段保持 null。存在多个冲突的同类字段时也保持未知并记录歧义。原时间键可以单独解析为日期诊断，**绝不复制成 available_at、issued_at 或 valid 区间**。正文中的明早、月份、日期和文件年度不自动转换为预测覆盖区间；只列出明确字段及未证实示例。

Jena v1/v2/v3 作为文件变体比较相同来源时槽的内容 hash、共同/独有记录数；`chronological_release_order_verified` 始终为 false。年度文件不是自动的发行版本。

### 时间可用性状态

状态定义为 `verified`、`protocol_assumption`、`unknown`。本次 CLI 不接受自动放行的时间假设或未经复核的发行证明：**即使记录出现 available_at，也只保留原字段，来源可回放性状态仍为 unknown**。

公共 helper 定义后续证据接口：verified 至少需明确的原始证据引用和带时区的 available_at；protocol_assumption 必须有可追踪约定且不能获得 verified-as-of 资格。本阶段不接入该证明流程、不检索候选，不声称仅有引用就已自动核验历史发行真实性。

所有实际 CLI 文本目前统计为 unknown，`formal_multimodal_training_allowed:false`。无文本文件也不能因 unknown 数为 0 而视作通过。正式 As-of 适配器后续必须默认拒绝未知可用时间或未证实来源；合法提前发布的未来预报不应仅因覆盖未来被拒绝。

## 5. 用户手动执行命令

**以下命令全部未由本代理执行。** 建议顺序：安装专用依赖→本地单元测试→两份远程 List→两份 Dry Run→人工查看总量→下载→Verify→Inspect→Git 修改检查。所有操作只针对两个指定子集。

### 5.1 专用环境与依赖（已有合适环境可直接激活）

```bash
cd /opt/data/private/ATM-ReTaMTSF
python3 -m venv .venv-fidel-ts
source .venv-fidel-ts/bin/activate
python -m pip install -r scripts/fidel_ts/requirements-stage1.txt
python -m pip freeze > /tmp/pivot-stage1-requirements-installed.txt
```

如系统 Python 缺少 venv，使用你自己的 Python>=3.10 环境；本阶段不为你安装系统包。没有 GPU 需求。库版本区间只限定 API 兼容范围；保存实际 freeze 文件以记录重现环境。

### 5.2 本地 Fixture 单元测试

```bash
cd /opt/data/private/ATM-ReTaMTSF
source .venv-fidel-ts/bin/activate
HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/fidel_ts
```

测试仅使用 tmp_path 内合成内容，socket 连接被测试 fixture 拒绝，不拉取数据/模型。应实际观察全部测试是否通过；本阶段没有 PASS 结果。失败时先提交 traceback，不直接跳过失败测试进入真实数据下载。

### 5.3 可选：先用 Stage 0 本地清单了解选集（不联网、不写文件）

```bash
python scripts/fidel_ts/download.py list --dataset caiso --stage0 --profile standard
python scripts/fidel_ts/download.py list --dataset jena --stage0 --profile standard
```

这时总量可能为 null，因为旧证据未记录所有 JSON 大小；不是空文件或零下载成本。下载前仍需下面的完整大小清单。

### 5.4 两个固定版本的完整清单（只下载元数据）

```bash
python scripts/fidel_ts/download.py list --dataset caiso --profile standard \
  --save-inventory artifacts/fidel_ts/inventories/caiso.json
python scripts/fidel_ts/download.py list --dataset jena --profile standard \
  --save-inventory artifacts/fidel_ts/inventories/jena.json
```

输出终端 JSON 和对应 inventory；清单文件已存在时不会覆盖，可复用 `--inventory` 或换新文件名。默认连接公开 Hugging Face，身份凭据如有需要由你配置 `HF_TOKEN` 环境变量，不写入配置/日志。失败时不要换成 main 或另一个未经说明的 Revision。

### 5.5 Dry Run：查看 Profile、准确大小和目标位置

```bash
python scripts/fidel_ts/download.py dry-run --dataset caiso --profile standard \
  --inventory artifacts/fidel_ts/inventories/caiso.json \
  --output-root dataset/Fidel-TS/raw
python scripts/fidel_ts/download.py dry-run --dataset jena --profile standard \
  --inventory artifacts/fidel_ts/inventories/jena.json \
  --output-root dataset/Fidel-TS/raw
```

应输出非 null 的 `estimated_total_bytes`、`unknown_size_files:0`、每个选中文件的最终路径；不会写文件。检查 CAISO 两地 formal/general 都在选集内、没有 merged 天气和 `.pkl`。

### 5.6 下载（由用户运行；可能长时间联网）

```bash
python scripts/fidel_ts/download.py download --dataset caiso --profile standard \
  --inventory artifacts/fidel_ts/inventories/caiso.json \
  --output-root dataset/Fidel-TS/raw --cache-dir dataset/Fidel-TS/.hf-cache --retries 2
python scripts/fidel_ts/download.py download --dataset jena --profile standard \
  --inventory artifacts/fidel_ts/inventories/jena.json \
  --output-root dataset/Fidel-TS/raw --cache-dir dataset/Fidel-TS/.hf-cache --retries 2
```

下载结果分别保存到 `raw/California_ISO/`、`raw/Jena_Atmospheric_Physics/`。网络中断后可重跑同一命令，缓存尽可能续传，已完成文件重验；不要先清空缓存。没有写入 complete 的 `.part` 或失败 Manifest 都不是成功数据。

若需追溯 CAISO 的三张原表或 Jena 数值原始分段，先将 Dry Run 的 `--profile standard` 改为 `--profile provenance` 查看额外字节数，再对同一个 inventory/数据目录运行 provenance Download。Profile 扩展保持相同 Revision，但可能明显增加磁盘占用。不要未经查看规模下载整个仓库。

### 5.7 离线 Verify

```bash
python scripts/fidel_ts/download.py verify --dataset caiso --profile standard --output-root dataset/Fidel-TS/raw
python scripts/fidel_ts/download.py verify --dataset jena --profile standard --output-root dataset/Fidel-TS/raw
```

退出码 0 且 `ok:true` 才表示本次声明的选集匹配。退出码 3 表示校验失败；其他输入/下载错误通常为 2。校验会完整读文件算 hash，不能将仅有文件名视作成功。发现已有文件损坏时先保留 Manifest/错误日志，将有问题的快照移到单独备份位置后再重新准备，不自动覆盖或静默“修复”。

### 5.8 真实本地数据结构审计

```bash
python scripts/fidel_ts/inspect.py --dataset caiso --profile standard \
  --input-root dataset/Fidel-TS/raw --inspect-only \
  --output-dir artifacts/fidel_ts/data_audit/caiso
python scripts/fidel_ts/inspect.py --dataset jena --profile standard \
  --input-root dataset/Fidel-TS/raw --inspect-only \
  --output-dir artifacts/fidel_ts/data_audit/jena
```

输出目录已存在时会停止；重跑请换新目录，如 `caiso_retry_01`，不覆盖旧结果。已下载 provenance 时，可将 profile 改成 provenance，以将可用原始文件加入审计输入清单。没有必要为 standard 先伪造 provenance 检查结果。

### 5.9 产物与 Git 检查

```bash
python -m json.tool artifacts/fidel_ts/data_audit/caiso/numerical_summary.json
python -m json.tool artifacts/fidel_ts/data_audit/jena/text_summary.json
python -m json.tool artifacts/fidel_ts/data_audit/caiso/availability_summary.json
python -m json.tool artifacts/fidel_ts/data_audit/jena/availability_summary.json
git diff --check
git status --short
git diff --stat
git diff -- models/ReTaMTSF.py layers/ReTaMTSFLayers.py data_provider/data_loader.py data_provider/data_factory.py docs/fidel_ts/00_repository_audit.md docs/fidel_ts/00_audit_evidence.json
```

最后一条应没有本阶段对受保护文件的修改。`Prompt.md` 在本次开始前已有用户修改，本阶段未编辑，不应将它混入代理新增代码。未跟踪文件不会出现在普通 `git diff --stat`，请结合 `git status` 查看新增模块/测试/配置。

## 6. 验收标准、资源与异常处理

| 操作 | 成功标准 | 不代表什么 |
| --- | --- | --- |
| Fixture 测试 | 用户实际运行后全部通过 | 不是完整数据或模型测试 |
| List | 固定 SHA、完整清单、选择标记与文件大小；保存 inventory | 不是已经下载数据 |
| Dry Run | 选集、总大小、目标目录可核对，无文件变化 | 不是校验本地内容 |
| Download / Verify | 所需 Profile 被 Manifest 覆盖、所有文件 complete、大小/hash 匹配、退出码0 | 不是对所有天气历史发行的证明 |
| Inspect | 四份 JSON + documents.jsonl + `_SUCCESS.json`，输入前后不变 | 不是无缺失/无泄漏/准许训练 |
| Availability | 应诚实显示 unknown/缺失字段，正式训练放行字段为 false | unknown 不能被当作 verified；0份文本不构成通过 |

资源：metadata 清单较小；准确下载体积以用户 Dry Run 为准。缓存和原始数据各保留一份，另有同文件系统暂存，建议可用磁盘至少为选集总量的 2–3 倍，documents.jsonl 还会增加审计磁盘占用。完整数值检查及逐文件 JSON 解析会使用 CPU/内存；预计可能数百 MB 到数 GB，实际取决于版本/记录数，不声称已经测过时间或内存。每次只运行一个 Inspect，重试时保留原产物。测试只需小型本地 Fixture。

持续中断的下载请回传 Manifest 内 attempts；不要将下载时间当作历史 available_at。进程被强制终止可能留下锁，在确认没有写入进程后由你手动处理。失败审计保留 failure.json/部分 ledger；修改输入后必须重新运行 Verify，并使用新输出目录。

## 7. 测试范围与下一阶段

测试代码覆盖固定 SHA、混合版本拒绝、已有文件复核/损坏不覆盖、部分下载不发布与重试、Dry Run 无写入、Stage 0 未知大小、Manifest/hash/大小/选集完整性、路径与符号链接、时间重复/缺口、-9999、NaN/Inf、Jena 列名映射、同日多地点、重复 JSON 键、缺字段、无 available_at、v1/v2/v3 非发行版本、审计产物与输入不变。

**测试代码已编写，但按照用户要求未执行。** 所有 CLI、依赖版本及真实数据统计仍等待用户运行后确认；代码只经过人工静态阅读，没有执行语法编译或 Smoke Test。

下一阶段可准备独立 Fidel-TS 多变量数据加载器，明确 Subject/Channel、MIMO Batch、固定时间划分、train-only Scaler、连续窗口与缺失策略。它不等于直接修补旧 Time-MMD Loader，也不等于已能正式训练。文本来源和时间证据、数值质量、原模型数学接口问题需要依据用户手动产物逐项处理。

Stage 1 编码工作到此停止，等待用户手动命令的日志、测试结果与真实审计产物。
