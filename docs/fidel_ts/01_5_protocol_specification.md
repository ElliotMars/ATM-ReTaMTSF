# PIVOT Stage 1.5：协议冻结、来源诊断与手动交接

准备日期：2026-11-03；工作区基准提交 `86ff6be172690e029ce9f3e2dd72c65915eb3d6b`。

**Stage 1.5 代码已编写，但所有测试、数值处理和文本来源检查均未执行。** 本文记录实现的规则及已有证据，不把代码准备当作运行通过。没有 Python 执行、依赖安装、联网、下载、真实数据处理、训练、Embedding 或 GPU 计算；没有联系作者，也没有编写联系作者的邮件。

## 1. 已有证据与当前默认决策

下列数字来自用户已完成的 Stage 1 产物，不是本次重新计算。来源：`artifacts/fidel_ts/data_audit/{caiso,jena}/{numerical_summary,text_summary,availability_summary}.json` 及成功标记。

| Dataset | 原始行数 / 通道 | 额外重复记录 | 原始间隔异常 | 唯一时间轴缺口 | Facet | 已验证 available_at |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CAISO | 724896 / 20 | 0 | 0 | 0 | 172826 | 0 |
| Jena | 604311 / 21 | 334 | 15 | 8 | 373596 | 0 |

Jena 的 334 是额外重复记录数；Stage 1 报告 668 行参与重复，**本阶段未据此虚构重复组内的数值一致性**。逐组判断由新诊断工具运行后给出。既有 -9999 计数：wv=19、max.wv=20、SWDR=1、max.PAR=191、CO2=1029。时区原因未证实，不能将重复/缺口归因于 DST。

已读取的本地固定版本官方代码确认：timestamp splitter 使用 2021/2022 边界，不借用前一 Split 历史；loader 在 **float32 原生训练值**上 fit `StandardScaler`，transform 后再逐 Split 做 `[::12]` 或 `[::6]`。缓存来源见 Stage 0 的 `source_files` 哈希清单和 `/tmp/fidel_ts_stage0/official/`；该临时缓存若不存在，工具不会下载替代副本。

立即采用的默认决定：

- N1 用于官方流程可比性；不能宣称它具备严格实时因果性，也不等于复现论文指标。
- N2 为全 20/21 通道候选主数值协议；只有诊断、产物与测试核验后才能进入 Stage 2 数值接口设计。
- T1 当前为 `blocked_not_verifiable`；不等待作者回复，不人工填发行时间。
- T2 仅为历史档案辅助协议；T3 仅为 `retrospective_oracle_risk` 档案对齐诊断。
- 现有数据若不能支持正式动态文本研究，应另行构建可信历史发行/版本记录的外部文本源。联系作者不是前置条件。

## 2. 新增代码与原文件边界

协议配置：`configs/fidel_ts/protocols_stage1_5.json`。四个用户入口：

- `scripts/fidel_ts/diagnose_numerical.py`
- `scripts/fidel_ts/prepare_numerical.py`
- `scripts/fidel_ts/diagnose_text_provenance.py`
- `scripts/fidel_ts/summarize_stage1_5.py`

辅助模块：`stage1_5_io.py`、`numerical_diagnostics.py`、`numerical_protocols.py`、`numerical_cli.py`、`text_provenance.py`、`weather_evidence.py`。依赖文件：`requirements-stage1_5.txt`；在 Stage 1 依赖上增加 N1 所用的 scikit-learn。

测试：`tests/fidel_ts/test_stage1_5_{numerical,provenance,outputs}.py`。

本阶段没有修改 Stage 1 公共模块、历史文档/证据、模型、层、旧 Loader/factory、Time-MMD、Baseline、原始数值/天气和旧权重。没有实现正式 MIMO Dataset、Temporal Scorer、Utility Scorer 或训练循环。

## 3. 数值诊断规则

输入只能是通过已有 Manifest/Verify 检查的固定 Revision 快照：

- CAISO：`75f74f556edd5fbdd080c4db6d4b7547fc331f39`，`time_series/all_except_battery.parquet`，20 通道。
- Jena：`c8da24175c7aabe5c3b02b503e4c5afe44a605fd`，`time_series/weather_large.parquet`，21 通道。

列名与顺序必须与固定来源配置相同，不提供自动选择 target/channel 的参数。全部真实运行前后再次校验快照，比较首选原文件 SHA-256。

### 重复组

每组输出时间、原始物理行号、额外记录数、Split、所有列相等情况、冲突列和值例、Arrow Null/非有限/哨兵状态。

- 数值精确比较，成对 NaN 按结构相等处理，不采用容差平均。
- 有效值完全一致、无无效值：`identical`。
- 原始结构完全一致但含无效值：`identical_with_invalid`。保留一条，同时保留无效 Mask；不声称无效观测内容已得到验证。
- 至少一个通道存在不同的有效数值：`conflicting`，整个时间点排除。
- 原始结构不同且由于无效值不能判断一致：`indeterminate_invalid`，整个时间点排除，不拼接互补通道。

去重保留的第一条仅用于**已证明结构相同**的重复记录，并保留完整行号追溯；不是对冲突记录选择第一条。标记 `unresolved_timezone_issue`，不声称重建了真实物理时间轴。

### 时间与无效值

区分原始行顺序、稳定排序、唯一时间轴。每个异常/缺口提供端点、原始行号（适用时）、间隔、理论缺失点、Split/跨界情况和距离边界一天内的标记。非网格整数倍的间隔不编造缺失点数量；短于预期的 off-grid 间隔不冒充正向缺口。

逐通道、逐 Split 记录 Null、NaN、Inf、-9999、有效计数、有效范围和负值计数、原顺序下无效连续段、多通道同时无效位置。Null 已计入 NaN 时不重复相加。仅 Jena 将 -9999 视作无效；正常负温度和 CAISO 负值不会被 blanket removal。

## 4. N1 / N2 的独立定义

| 规则 | N1 Official-Compatible | N2 Strict-Contiguous |
| --- | --- | --- |
| 数据顺序 | 发布行顺序不改 | 时间戳稳定排序 |
| 重复值 | 忠实保留 | 相同结构折叠；冲突/无法判定时间点排除 |
| Split | `<2021`、2021 年、`>=2022` | 相同边界 |
| 历史跨 Split | 不借用 | 本次也冻结为不借用，便于比较 |
| Scaler | 官方 float32 原生 train 上 fit StandardScaler | 去重后原生 train 的逐通道有效观测，float64、ddof=0 |
| 抽样 | 每个 Split 的首个原始行开始 `[::12/6]` | 固定训练相位，以真实时间戳整小时差选点 |
| 哨兵 | 保留在数值与 Scaler 中 | 无效，数值表示为 NaN，原有效性 Mask 为 false |
| 输入填充 | 不新增清洗改变官方流程 | 准备阶段不填；另有仅针对历史切片的因果前填 helper |
| 标签 | 官方数值保留；科学 Mask 仅作质量追踪 | 不填标签，所有方法共享 y_mask |
| 窗口 | 索引忠实保留全部理论官方窗口，另报质量 | 必须连续小时、同一原生连续区段，且至少一个有效标签 |

N2 锚点选择：去重后的训练序列中，最早一个**至少一个通道有效**的时间戳。其纳秒模一小时相位固定用于三个 Split；不要求所有通道在同一时刻均有效，不为多保留样本调整相位，不做均值聚合、重采样或最近邻。

N2 采用比仅检查小时差更保守的连续性：原生频率缺口和排除 ambiguous 点造成的缺口都会切断 segment，即使缺口两侧恰有连续的小时抽样点，也不跨 segment 生成窗口。`duplicate_resolution.json` 记录去重后缺口及被排除的 ambiguous 时间点。

N2 Scaler 的每个通道都必须有有效训练样本；零样本明确 blocked。常量通道 scale=1。不会在验证/测试重新拟合，不用未来实况填历史。`causal_fill_history` 只接收一个预测起点之前的历史切片：先用训练均值回退，之后只向前携带已见有效值，并返回未改变的 x_mask。该 helper 不会应用到完整标签表。

N1 输出的 scientific validity mask 也将 Jena -9999 标为 false，但这不改变 N1 原始值/标准化值和官方窗口。如果后续在 N1 上额外套用新的 masked loss，那必须另行命名，不能当作完全原样的官方评测。N2 下所有新 Baseline/PIVOT 必须共享相同 Mask 与窗口。

两协议都是新目录中的派生产物。N2 **不能消除发布 Parquet 上游已经发生的未来插值**。CAISO 的极端有限值也保留，仍需科学审查，不以修改测试协议掩盖。

## 5. 全通道窗口与对比

冻结 L=360，H=24/168/336/720，每个 Split 单独计数，使用全部 20/21 个通道。

统计理论窗口、连续窗口、连续且标签全有效的窗口、连续且至少一个标签有效的窗口、有效标签 cell 次数及逐通道次数。重叠窗口中的同一标签反复出现时计为 occurrence，不能当作独立统计观测。

实际索引记录 input 起点、label 起点/终点、首预测时间 `forecast_origin`、最后历史时间以及完整标签/连续性标记。N2 不跨缺口；默认允许部分标签无效，但以 y_mask 排除相应 loss/metric，不能用插值、训练均值或前填伪造标签。

对比工具保存两个协议的完整抽样时间戳集合及 occurrence 次数，比较每个 H 的 origin 集合和连续窗口交集，并提供保留率。仅有 origin 相同不意味着不连续的 N1 历史相同，报告明确区分。

默认每个 Split/H 至少一个可用窗口是技术门槛，不是统计功效保证。N2 与 N1 理论窗口比低于 0.8 会警告显著减少，供人工复核；不会为提高数量重选相位。缺少可用窗口或全通道训练统计会 blocked。

## 6. 文本来源诊断与使用决定

复用已成功且 hash 匹配的 Stage 1 `documents.jsonl` 与现有 JSON parser。先绑定 Stage 1 文件哈希和当前快照，再重建报告；Facet 共享 report_id，位置、文件、版本、同键 occurrence 都进入身份。空报告也从原 JSON 计入报告数。报告数不等于独立发行数。

抽样使用固定 seed=2025，按 Split、年份、季节、Location、Report Type、文件版本分层，每层一份。为了能做数值对应，只从 primary 数值 min/max 范围内可解析的原时间键中抽样；报告清单仍计入所有原始报告。抽样不足或未覆盖的时段不伪造。

每份样本保存报告全文/Facet 身份、原始 source key、前后各六小时的全通道有效观测摘要，以及可获得的原始天气/原始 Jena 数值上下文。显式识别摄氏温度范围、带单位风速、mm/百分比提及和趋势词；比较温度范围与后续实况只形成风险提示，不能凭一个分数宣告泄漏或无泄漏。

CAISO 两地 raw JSON/CSV 分开检查字段。没有证据时不会把未说明单位的 epoch 数字转换成发行时间；原文 `ds` 的完整日历区段可作为观测时段标签，但不是预报 validity。CSV 中明确的日期时间只作为未验证时区的观测时钟进行辅助匹配。匹配字段与本地文件/record pointer 都保留。

Jena v1/v2/v3 比较同键的内容与覆盖差异，不把版本名当发行顺序或生成时间。可获得的原始数值分段仅作为观察对照，不能证明文本来自预报。

对下载的 `.py` 仅做 AST 阅读，记录读文件、插值、编码等具体调用与行号，不执行它们。若缺少生成脚本，不会凭编码/合并脚本宣称已经证明未来观测生成链。原始结构似实况、文字与实况很接近，都只属于合理推断；强来源证据需要能够追溯真实生成输入。

所有无独立历史发行证据的报告维持 availability_status=unknown。`source_time_key`、`issued_at`、`available_at`、`valid_start/end`、`forecast_origin` 分开：诊断的 source-key reference time 不是模型 forecast_origin。

默认 T 协议：

- **T1：blocked_not_verifiable**。只有可信历史发行和版本证据支持的 available_at 才可进入正式 as-of 判断；不假定提前一天、六小时或零延迟。
- **T2：辅助历史档案**。仅 source key < forecast_origin，仍未证明历史发行可用性。
- **T3：retrospective_oracle_risk**。仅用于预测区间 key 对齐的回溯诊断，不能支撑无泄漏主结论。

没有 verified valid_start/end 就不能称 Forecast Validity Scoring；source-key 距离探索需另行命名。Utility 不能让当时拿不到的文本进入正式池。本阶段只定义决策与诊断，不实现检索、模型或 scorer。

## 7. 用户手动命令

所有下列命令都未由代理执行。在你已用于 Stage 1 的 Python 环境中执行；如缺少新增依赖，先由你安装：

```bash
cd /opt/data/private/ATM-ReTaMTSF
python -m pip install -r scripts/fidel_ts/requirements-stage1_5.txt
```

### Step 1：离线单元测试

```bash
test ! -e artifacts/fidel_ts/stage1_5/tests.xml && \
HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/fidel_ts \
  --junitxml=artifacts/fidel_ts/stage1_5/tests.xml
```

测试只使用本地合成 Fixture。保留 JUnit 文件；文件已经存在时，上述命令会停止，重跑应换新 JUnit 路径，并同步修改 Step 8 的 `--pytest-report`。汇总工具检查三个新测试模块是否出现，以及错误/失败/跳过情况。旧 Stage 1 单独通过的报告不能冒充 Stage 1.5 验证。

### Step 2：CAISO 数值诊断

```bash
python scripts/fidel_ts/diagnose_numerical.py --dataset caiso \
  --input-root dataset/Fidel-TS/raw --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir artifacts/fidel_ts/stage1_5/caiso/numerical
```

### Step 3：Jena 数值诊断

```bash
python scripts/fidel_ts/diagnose_numerical.py --dataset jena \
  --input-root dataset/Fidel-TS/raw --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir artifacts/fidel_ts/stage1_5/jena/numerical
```

### Step 4：CAISO 文本来源检查

```bash
python scripts/fidel_ts/diagnose_text_provenance.py --dataset caiso \
  --input-root dataset/Fidel-TS/raw --audit-dir artifacts/fidel_ts/data_audit/caiso \
  --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir artifacts/fidel_ts/stage1_5/caiso/text
```

### Step 5：Jena 文本来源检查

```bash
python scripts/fidel_ts/diagnose_text_provenance.py --dataset jena \
  --input-root dataset/Fidel-TS/raw --audit-dir artifacts/fidel_ts/data_audit/jena \
  --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir artifacts/fidel_ts/stage1_5/jena/text
```

### Step 6：按需补 provenance 来源（先看规模，再由用户下载）

当前本地 CAISO 两地目录只见 weather_report；standard 不含 raw weather。工具会给出 `missing_provenance_files`，不会自动下载。

```bash
python scripts/fidel_ts/download.py dry-run --dataset caiso --profile provenance \
  --inventory artifacts/fidel_ts/inventories/caiso.json --output-root dataset/Fidel-TS/raw
python scripts/fidel_ts/download.py dry-run --dataset jena --profile provenance \
  --inventory artifacts/fidel_ts/inventories/jena.json --output-root dataset/Fidel-TS/raw
```

需要时，在确认大小后执行：

```bash
python scripts/fidel_ts/download.py download --dataset caiso --profile provenance \
  --inventory artifacts/fidel_ts/inventories/caiso.json --output-root dataset/Fidel-TS/raw \
  --cache-dir dataset/Fidel-TS/.hf-cache --retries 2
python scripts/fidel_ts/download.py download --dataset jena --profile provenance \
  --inventory artifacts/fidel_ts/inventories/jena.json --output-root dataset/Fidel-TS/raw \
  --cache-dir dataset/Fidel-TS/.hf-cache --retries 2
python scripts/fidel_ts/download.py verify --dataset caiso --profile provenance --output-root dataset/Fidel-TS/raw
python scripts/fidel_ts/download.py verify --dataset jena --profile provenance --output-root dataset/Fidel-TS/raw
```

补下载后若要利用新增来源，重跑 Step 4/5，分别使用全新目录 `.../caiso/text_with_provenance`、`.../jena/text_with_provenance`。Stage 1 老审计仍保留，工具逐文件匹配原哈希，而不是要求扩展 Profile 后下载 Manifest 整体哈希仍相同。不要覆盖旧成功文本审计。

```bash
python scripts/fidel_ts/diagnose_text_provenance.py --dataset caiso \
  --input-root dataset/Fidel-TS/raw --audit-dir artifacts/fidel_ts/data_audit/caiso \
  --output-dir artifacts/fidel_ts/stage1_5/caiso/text_with_provenance
python scripts/fidel_ts/diagnose_text_provenance.py --dataset jena \
  --input-root dataset/Fidel-TS/raw --audit-dir artifacts/fidel_ts/data_audit/jena \
  --output-dir artifacts/fidel_ts/stage1_5/jena/text_with_provenance
```

### Step 7：独立准备 N1 / N2

以下示例是真实准备。若只查看输入和路径，先在同一命令末尾加 `--dry-run`；只算统计、不写序列表时将 mode 改为 `report-only`，并使用另一输出目录。后续正式对比需要 prepare 产物。

```bash
python scripts/fidel_ts/prepare_numerical.py --dataset caiso --protocol n1 --mode prepare \
  --input-root dataset/Fidel-TS/raw --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir dataset/Fidel-TS/processed/caiso/n1_official_compatible
python scripts/fidel_ts/prepare_numerical.py --dataset caiso --protocol n2 --mode prepare \
  --input-root dataset/Fidel-TS/raw --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir dataset/Fidel-TS/processed/caiso/n2_strict_contiguous
python scripts/fidel_ts/prepare_numerical.py --dataset jena --protocol n1 --mode prepare \
  --input-root dataset/Fidel-TS/raw --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir dataset/Fidel-TS/processed/jena/n1_official_compatible
python scripts/fidel_ts/prepare_numerical.py --dataset jena --protocol n2 --mode prepare \
  --input-root dataset/Fidel-TS/raw --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir dataset/Fidel-TS/processed/jena/n2_strict_contiguous
```

数值协议被科学条件阻断时返回 3，并写 `_BLOCKED.json`，不能将其当作通过。输入/依赖/处理错误通常返回 2，没有成功标记。不要为了消除 blocked 擅自换目标、Split、H 或小时锚点。

### Step 8：对比和生成 Dataset Readiness 报告

```bash
python scripts/fidel_ts/summarize_stage1_5.py \
  --diagnostics-root artifacts/fidel_ts/stage1_5 \
  --processed-root dataset/Fidel-TS/processed \
  --pytest-report artifacts/fidel_ts/stage1_5/tests.xml \
  --config configs/fidel_ts/protocols_stage1_5.json \
  --output-dir artifacts/fidel_ts/stage1_5/readiness \
  --decision-report docs/fidel_ts/01_5_dataset_readiness_decision.md
```

如果 Step 6 后创建了两份新版文本诊断，改用完整命令：

```bash
python scripts/fidel_ts/summarize_stage1_5.py \
  --diagnostics-root artifacts/fidel_ts/stage1_5 --processed-root dataset/Fidel-TS/processed \
  --pytest-report artifacts/fidel_ts/stage1_5/tests.xml \
  --caiso-text-dir artifacts/fidel_ts/stage1_5/caiso/text_with_provenance \
  --jena-text-dir artifacts/fidel_ts/stage1_5/jena/text_with_provenance \
  --output-dir artifacts/fidel_ts/stage1_5/readiness_with_provenance \
  --decision-report docs/fidel_ts/01_5_dataset_readiness_decision_with_provenance.md
```

也可以只指定一个数据集的新诊断目录；不按修改时间自动挑“最新”结果。使用此变体时，Step 9 的摘要路径相应改为 `readiness_with_provenance`。

目标 `docs/fidel_ts/01_5_dataset_readiness_decision.md` 由此步骤根据真实产物生成，**当前未提前填入未运行的 Stage 1.5 数字**。文件已存在时拒绝覆盖，应换新报告路径。

### Step 9：供人工审核的精简摘要

```bash
python -m json.tool artifacts/fidel_ts/stage1_5/readiness/brief_summary.json
python -m json.tool artifacts/fidel_ts/stage1_5/readiness/caiso/dataset_readiness.json
python -m json.tool artifacts/fidel_ts/stage1_5/readiness/jena/dataset_readiness.json
git diff --check
git status --short
```

所有 CLI 都支持 `--dry-run`，且不默认联网、不自动下载；只有 Step 6 明确调用已有 Download 入口才会由用户触发网络。Dry Run 仅读元数据/配置，不扫描 Parquet 或运行真实诊断，也不创建输出目录。

## 8. 准确产物路径与验收

`artifacts/fidel_ts/stage1_5/{dataset}/numerical/`：

- numerical_diagnostics.json
- duplicate_analysis.json
- time_gap_analysis.json
- missing_value_analysis.json
- manifest.json、_SUCCESS.json（成功运行时）

`artifacts/fidel_ts/stage1_5/{dataset}/text/`（或指定新目录）：

- text_report_structure.json、reports.jsonl
- text_source_analysis.json
- text_temporal_analysis.json
- text_leakage_risk_samples.jsonl
- manifest.json、_SUCCESS.json

每套 `dataset/Fidel-TS/processed/{dataset}/{protocol_dir}/`：

- manifest.json、channels.json、splits.json、scaler.json
- quality_summary.json、duplicate_resolution.json、window_eligibility.json
- `{train,validation,test}/{values,scaled,validity_mask,timestamps}.parquet`
- `window_indices/{split}_H{24,168,336,720}.parquet`
- _SUCCESS.json 或 _BLOCKED.json，不能同时视为成功

`artifacts/fidel_ts/stage1_5/readiness/`：

- `{caiso,jena}/downsampling_comparison.json`
- `{caiso,jena}/downsampled_timestamp_sets.parquet`
- `{caiso,jena}/dataset_readiness.json`
- dataset_readiness.json、brief_summary.json、dataset_readiness_decision.md、_SUCCESS.json

比较前若某协议因 Scaler 等问题完全 blocked，则相应对比明确返回 blocked，不能伪造时间集合。报告区分 Numerical MIMO Readiness、Archived Text Research Readiness、Verified As-of Multimodal Readiness 和 Remaining Scientific Limitations，没有笼统 ready:true。

验收：离线测试应由用户实际运行并全部通过；有效产物需成功标记及其列出的全部 hash 一致；N2 不能含重复时间戳，训练统计必须只来自有效 native train，所有通道顺序固定，所有索引窗口满足双重连续性；原始 Parquet SHA-256 前后不变；未知 available_at 仍为 null/unknown，T1 仍 blocked。成功生成报告不代表已解决时区或上游插值问题。

资源：真实扫描、哈希、Parquet 输出和文本 ledger 处理可能超过 60 秒，全部由用户运行。内存可能为数百 MB 至数 GB，取决于原始分段、报告与 Facet 数；输出还包括多套表、索引和报告 ledger。先 Dry Run，逐数据集顺序执行；未实测资源和耗时，不写成已验证数字。失败目录保留 failure.json 或部分文件；无成功标记时不得接入后续流程，重试用新目录。

## 9. 停止点

所有后续工具调用只由用户执行。Stage 2 不以作者回复为条件，但数值方向仍需 N2 产物和离线测试核验；正式 T1 动态文本多模态仍因历史发行证据不足受阻。T2/T3 可用于清楚标注的辅助研究，不能包装成可部署的文本预测收益。

本次到 Stage 1.5 代码准备为止，等待用户执行日志、JUnit、数值对比与来源诊断结果；不进入 Stage 2。
