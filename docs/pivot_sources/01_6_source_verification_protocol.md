# PIVOT Stage 1.6：外部来源验证与手动交接

Stage 1.6 的代码、测试与协议文档已经编写，但全部单元测试、外部来源 Probe 和真实数据验证均尚未执行。

本阶段不访问网络、不运行 Python/pytest、不安装依赖、不处理真实数据。所有下列命令供用户手动运行。没有完整数据下载、Embedding、模型、Stage 2 Dataset 或联系作者的要求。

## 1. 已阅读的本地事实

读取了 `Prompt.md`、Stage 1.5 决策文档、`artifacts/fidel_ts/stage1_5/readiness_with_provenance/brief_summary.json`、两套 N2 `quality_summary.json`、来源配置、既有公共代码和本地 README/脚本线索。既有报告显示两套 numerical artifacts 均为 candidate_for_stage2_review，原天气文本 T1 均 blocked。上述是用户此前运行结果，本阶段没有重新验证其哈希或计算统计。

现有 N2 全通道列表包含 CAISO 的 Day ahead forecast、Hour ahead forecast。数值时间轴未赋可信时区；Jena 小时相位为 :10。Stage 1.6 不改变这些既有产物。

当前没有本地 NWS/IEM 或 GDEX 的正式接口文档副本，且用户禁止代理联网。因此没有把记忆中的接口、查询参数、认证方式或返回字段伪装成已核查事实。

**NWS/GFS 是文档驱动的适配器**：不带 request plan 的 probe 只保存任务给定的官方参考页及页面实际链接，不能获取或宣称验证了历史公告/Run。实际数据请求必须通过一个引用已保存正式文档的、人工审阅的 exact-URL 请求清单。下载文档、填写请求清单、再次探测是明确的人工交接步骤，不是等待作者回复。

StockNet 使用 GitHub commit/tree 资源读取元数据，先将 HEAD 解析为实际 Commit SHA，再据真实 tree 路径选择少量文件。默认按层读取 price/raw 清单、tweet/raw 实体目录和最多两个实体叶目录，避免递归整库 tree 的巨大返回。它也尚未运行，不保证当前服务可访问、目录未变或无需认证。

## 2. 文件和接口

配置：`configs/pivot_sources/stage1_6_sources.json`。三个不可直接执行的证据/请求模板：`evidence.template.json`、`nws_request_plan.template.json`、`gfs_request_plan.template.json`。

`scripts/pivot_sources/`：

- `common.py`：文件哈希、证据引用、受保护输出、bundle 完整性、Gate 决策。
- `asof.py`：有时区时间、来源/接收/运行延迟策略、检索之前的候选过滤。
- `http.py`：显式联网、持久化预算、重试、限速、原始字节保存、去重和恢复。
- `nws_iem.py`：Product ID 候选格式、WMO 时间候选、AFOS/办公室、原文、修订及冲突检查。
- `gfs_archive.py`：Run/Lead/Valid 时间、分版本运行情景、固定模板 helper、可选本地小 GRIB 元数据。
- `stocknet.py`：固定 Universe、原始 Tweet、交易日历、三种收益率、价格与文本覆盖。
- `probe.py`：正式文档探测、证据绑定请求清单、GitHub 元数据与小样本流程。
- `diagnostics.py`：N2 只读审计、时区线索、Probe 范围覆盖、StockNet 交易日审计。
- `cli.py`：公共 CLI；三个 `verify_*.py` 入口只指定 source。
- `summarize_stage1_6.py`：根据真实 bundle/JUnit 生成最终决策。
- `requirements-stage1_6.txt`：由用户安装的依赖清单。

三个来源 CLI 共有 `--mode {dry-run,probe,audit,coverage}`、`--config`、`--output-dir`、`--input-dir`、HTTP 限额参数。只有 `probe --allow-network` 可以联网；模块导入、dry-run、audit、coverage 全部离线。

NWS/GFS 支持 `--request-plan`、有时区 `--start/--end`（单次最多两天）；NWS 支持重复 `--afos-id`。GFS 支持 `--origin`、`--allow-small-sample`、`--grib-inspect`。StockNet 支持 Commit `--revision`、价格/推文文件上限、`--tweet-time-evidence`、`--calendar-file`、`--calendar-evidence`、`--calendar-provider exchange_calendars` 和可选现有本地价格目录 `--local-stocknet-root`。

## 3. 证据和请求清单

所有证据必须含 `reviewed=true`、`role`、`source_url`、`local_file`、`sha256`、`quote`。代码检查 quote 实际出现在相同哈希的本地文件；语义是否足以支持声明仍需人工审阅，不能只改一个布尔值就当验证成功。模板 reviewed=false，缺失字段为 null。

证据角色：

| role | 意义 |
|---|---|
| documented_interface | 官方文档说明接口/查询/返回结构 |
| issuance / creation | 原始发行/创建时间字段的真实含义 |
| historical_reception | 独立的历史接收时间 |
| historical_publication | 历史公开发布时间；不是当前 HTTP Last-Modified |
| model_initialization | 模型初始化时刻，不代表可下载时刻 |
| numerical_clock | 数值时间列与物理时区/UTC 的关系 |
| official_exchange_calendar | 实际交易日及开闭市时间 |
| bounded_forecast_sample | 一个 <=2 MiB 小预测文件/服务端裁剪样本的范围及大小 |

`--timezone-evidence` 的 evidence 还必须给出 IANA `timezone` 和 `bound_primary_sha256`，后者必须匹配 N2 manifest 的原始主表哈希。只有此后才尝试本地化；重复/不存在的 DST 时间直接 blocked，不采用 infer、shift 或猜测 UTC/洛杉矶/欧洲时区。引用代码里出现 timezone 单词只算调查线索。

日历证据还需 `calendar_sha256` 绑定所用 CSV。CSV 列为 `session,open,close`；日期为 YYYY-MM-DD，open/close 为有时区的 ISO 时间。必须包含实际交易日、假日和提前收盘；不能用一般工作日序列代替。exchange_calendars 可提供显式选择的**临时候选日历**，保留版本，仍需独立官方核查。

从正式页面选用文档后，将 request plan 模板复制为一个新的配置文件，例如 `configs/pivot_sources/nws_request_plan.reviewed.json`。不要修改模板来冒充预先核查结果。

请求清单顶层必须有 source、schema_version=1、interface_evidence、requests 和 response_contract。每条 request 包括实际文档支持的完整 url、GET/HEAD method、kind 和 documentation_reference。代码不生成未知参数。

NWS 每条 records 请求还需要 start/end 和 afos_ids；必须落在配置中的候选数值日历范围内。JSON 返回通过 records_path 和 fields 的实际字段名映射，不推测别名。支持根数组（records_path=""）或嵌套对象数组。纯正文接口用 format=nws_text、fields={source_record_id:product_id,text:text}，并在 request 中提供完整 product_id。其 UTC/发行含义必须由 time_evidence.product_id_time 证明，否则只保留时间候选。

GFS kind=catalog/grib_index 只保存原始目录/索引及引用，不擅自解释未知格式；file_head 必须 HEAD，只读取 advertised bytes；JSON records/metadata 可用显式 fields 映射 run_time、forecast_lead_hours、source_record_id、model_version、grid_resolution、variables、valid_time、file_publication_time。forecast_lead_unit 必须为 hours 且有文档依据。缺失的变量/Level/空间格点/发布时刻保持 unknown，不能由文件名猜测后升级为 verified。

GFS 元数据适配器目前不自动解析任意 HTML 目录或站点私有索引格式，也不提交认证/裁剪任务。若服务仅提供这类格式，原始响应可保存审阅，但结构化 Run/Lead 检查保持 not_checked，需根据真实正式文档补充对应解析适配。没有现成匿名 JSON API 的假设。

GFS 唯一 tiny sample 需要 kind=small_sample、expected_bytes、bounded_forecast_sample 证据及 `--allow-small-sample`。样本最多 2 MiB，默认不允许全球 GRIB GET。`--grib-inspect` 仅在 audit 时读取这一小文件的最多 32 个 GRIB message header，不安装 eccodes、不读取全球格点数组、不生成实际文本。解析出的字段仍不证明历史发布。

正式文档可能指向额外官方主机；当前 allowlist 不包含的主机会被拒绝。应先审阅实际来源归属，再在单独配置中显式增加，并在新目录重跑。不同配置哈希的产物不能混合汇总。不会自动跟随跨站跳转、注册账号或记录密钥。

## 4. 时间和科学判定

每条规范记录保留任务要求的全部字段，未知为 None。document_id 包含 source ID 与 canonical raw-record hash；同一 source ID 的不同正文不会被合并。raw_record_sha256 是单条解析记录的稳定 JSON 编码哈希；原始返回的完整字节另有 raw_file_sha256/HTTP manifest 哈希。raw 文件和解析文件分开。

Strict Verified 仅接受经审阅的历史接收或实际公开时间；发布时间可信但接收未知时，Source-time + Conservative Lag 单列。NWS issued-based 的 lag=0 也是诊断场景，不宣称所有用户即时获得。GFS run+3/6/12 小时仅是假设；有可靠实际公开/接收时间时优先使用。下载时间和 HTTP Last-Modified 不自动填入历史字段。

没有可核查字段则 Not Verifiable。预报有效时间晚于 origin 是合法未来预测的必要属性之一，不是泄漏证据；发行/可用时间晚于 origin 的记录不能进入候选池。当前只有候选过滤 helper，不含检索器、模型或 Dataloader。

CAISO 与 NWS UTC 对齐若无证据，task Gate blocked。Jena 同样不能未经证据本地化。AFD 不凭正文推造 forecast_valid 区间。GFS 统计所探测 Run 在不同 lag 下的合法性、最新 Run 年龄、版本及点间隔，分别报告跨度和逐目标小时直接覆盖。缺失 Lead 不用未来 Run 替代。未验证版本的不同记录不拼成一个“已知同版本”运行。

StockNet created_at 转 UTC 只证明解析后的创建时间候选；需 Twitter 原始字段语义证据才能启用 creation-based policy，仍不证明历史收集/接收。保留 retweet/quote 内嵌时间，外层可用性不由内层更早时间决定。无推文股票不移除。标签依赖官方相邻交易 session；缺价格后下一个 session 的收益率仍无效，不能跳过缺日计算跨日收益伪装成逐日收益。

数值价格只读。若已在本地拥有固定 Commit 的完整价格文件，可显式指定 root，逐文件 Git blob SHA 校验。默认仅下载少量价格文件，因此不会声称 88 股票共同交易日已验证。复权价格保持风险未清除；不使用原论文分类标签。

## 5. 网络和输出安全

默认 HTTP 上限：12 attempts（重试也计数）、2 MiB/文件、8 MiB 总量、超时 30 秒、重试 2 次、请求间隔 1 秒、最多 100 条解析记录。CLI 可显式调整，硬上限为 64 attempts、8 MiB/文件、32 MiB 总量、1000 记录；GFS tiny file 独立上限仍为 2 MiB。StockNet 最多选择 8 个价格、16 个推文文件，不 clone。

限额只描述下载/解析规模，不代表服务配额、费用或实际档案大小。StockNet 按层查目录，默认最多两个、硬上限四个推文实体叶目录，不使用 recursive=1 整库 tree。未访问目录的文件数量和总大小保持 unknown，已列文件大小只是局部下界。Content-Length 超限在读取正文之前拒绝；未知长度读到 cap 时留下部分文件，绝不作为成功响应。响应失败、重试、下载字节、请求时间存 http_manifest.json；当前获取时间字段明确命名 received_at_now，历史 received_at 仍为 None。突然中断的未确认读操作单列为预算保守预留，不冒充实际收到的字节数。

不跟随自动重定向；遇到登录、403、401、配额、跳转时保留失败日志并停止，不绕过访问控制。User-Agent 明确标识研究 Probe。URL 中常见密钥/签名参数被拒绝，计划文件和日志不得放敏感凭据。

仅 run-local HTTP checkpoint/journal 可在未封存运行中更新；raw attempt 文件以 xb 新建，旧原始文件和成功 bundle 不覆盖。`--resume` 仅恢复中断、未封存且绑定一致的 probe；请求数/字节数预算继续累积，已有 URL 响应校验 hash 后复用。预算已耗尽时用新目录和明确的新计划重跑，不借 resume 重置预算。解析产物已部分写出时拒绝原目录恢复，避免混合运行。

成功封存 `_SUCCESS.json` 表示产物完整，不表示来源通过 T1。已捕获的 HTTP/请求契约错误写 failures.jsonl、未验证的九类结果和 `_INCOMPLETE.json`，可继续离线审计并输出当前 NO-GO；它表示 Probe 未完成，不能解读为来源永远不可用。此类封存失败目录重跑应使用新目录。意外中断且尚未封存的 HTTP Probe 才能 `--resume`。返回码 0 是工具完成；科学 GO/CONDITIONAL/NO-GO 必须读取 source_readiness.json。错误返回 2。

## 6. 用户手动执行顺序

下列命令没有由代理执行。在已有 Stage 1.5 Python 环境执行；确有缺依赖时，由用户自行安装：

```bash
cd /opt/data/private/ATM-ReTaMTSF
python -m pip install -r scripts/pivot_sources/requirements-stage1_6.txt
```

### Step 1：全部离线测试

```bash
test ! -e artifacts/pivot_sources/stage1_6/tests.xml && \
HF_HUB_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/fidel_ts tests/pivot_sources \
  --junitxml=artifacts/pivot_sources/stage1_6/tests.xml
```

测试只使用本地合成 Fixture 和注入的假 HTTP 响应；socket 在测试中被禁用。已有 JUnit 不覆盖，重跑请换路径并同步修改 Step 10。

### Step 2：三个纯本地配置 Dry Run

```bash
python scripts/pivot_sources/verify_nws_iem.py --mode dry-run
python scripts/pivot_sources/verify_gfs_archive.py --mode dry-run
python scripts/pivot_sources/verify_stocknet.py --mode dry-run
```

Dry Run 不读取数值表、不发请求、不创建输出目录。它不等于接口/文件完整性验证。

### Step 3：NWS 文档与历史公告小 Probe

先由用户明确联网，只保存官方文档和实际链接：

```bash
python scripts/pivot_sources/verify_nws_iem.py --mode probe --allow-network \
  --max-requests 4 --max-file-bytes 2097152 --max-total-bytes 4194304 \
  --output-dir artifacts/pivot_sources/stage1_6/nws/documentation
```

这一步 **不会**查询历史 AFD。审阅输出 documentation/raw 文件后，填写 reviewed request plan。plan 应按配置的 2018/2020/2021/2022/2024 年各少量日期覆盖 AFDMTR/AFDSGX。然后执行：

```bash
python scripts/pivot_sources/verify_nws_iem.py --mode probe --allow-network \
  --request-plan configs/pivot_sources/nws_request_plan.reviewed.json \
  --afos-id AFDMTR --afos-id AFDSGX --max-requests 16 --max-records 100 \
  --max-file-bytes 2097152 --max-total-bytes 8388608 \
  --output-dir artifacts/pivot_sources/stage1_6/nws/probe
```

也可指定单个有时区时间范围 `--start 2021-07-15T00:00:00Z --end 2021-07-16T00:00:00Z`；plan 请求必须落在其中。不要把 template 文件直接作为已审阅 plan 使用。

若暂不能建立正式接口证据，可将第一条命令的 output-dir 设为 `.../nws/probe`，继续离线审计以生成明确未验证结论；不可因此声称已探测历史公告。

### Step 4：NWS 时间与 CAISO 时区审计

```bash
python scripts/pivot_sources/verify_nws_iem.py --mode audit \
  --input-dir artifacts/pivot_sources/stage1_6/nws/probe \
  --output-dir artifacts/pivot_sources/stage1_6/nws/audit
```

默认会输出 blocked_unknown_numerical_timezone。只有取得真实证据后，在新目录重新运行并增加 `--timezone-evidence configs/pivot_sources/caiso_clock.reviewed.json`。不要将 null 填成猜测时区。

### Step 5：GFS 文档/目录元数据 Probe

```bash
python scripts/pivot_sources/verify_gfs_archive.py --mode probe --allow-network \
  --max-requests 3 --max-file-bytes 2097152 --max-total-bytes 4194304 \
  --output-dir artifacts/pivot_sources/stage1_6/gfs/documentation
```

这里只保存给定 GDEX 页面及其真实链接，不声称已验证 2015–2025 档案。审阅访问权限、目录/元数据接口和时间字段后填写 reviewed plan，再运行：

```bash
python scripts/pivot_sources/verify_gfs_archive.py --mode probe --allow-network \
  --request-plan configs/pivot_sources/gfs_request_plan.reviewed.json \
  --max-requests 16 --max-records 100 --max-file-bytes 2097152 \
  --max-total-bytes 8388608 \
  --output-dir artifacts/pivot_sources/stage1_6/gfs/probe
```

plan 应记录指定六年、四 Cycle、四 Lead 的实际查询范围。有限请求未观察到的组合标为未核查，不能称为实际文件不存在。若接口不支持小区域或需要登录，记录实际证据及规模，保持 conditional/blocked，不自动提交下载任务。没有 reviewed plan 时可让纯文档 Probe 输出到 `.../gfs/probe`，供后续生成未验证报告。

### Step 6：GFS Run/Lead 与延迟审计

```bash
python scripts/pivot_sources/verify_gfs_archive.py --mode audit \
  --input-dir artifacts/pivot_sources/stage1_6/gfs/probe \
  --output-dir artifacts/pivot_sources/stage1_6/gfs/audit
```

若已有明确允许的小样本，audit 可加 `--grib-inspect`；用户需另行准备 eccodes/native library。没有小样本就不需要安装。主工具不会安装或下载。

### Step 7：StockNet 有限文件清单与样本

```bash
python scripts/pivot_sources/verify_stocknet.py --mode probe --allow-network \
  --revision HEAD --max-requests 20 --max-price-files 2 --max-tweet-files 4 \
  --max-records 100 --max-file-bytes 2097152 --max-total-bytes 8388608 \
  --output-dir artifacts/pivot_sources/stage1_6/stocknet/probe
```

HEAD 只用于首次解析；返回的 Commit 固定后再请求 tree/实际文件，产物保留 Commit 和 Git blob/本地 SHA-256。复现时用该 40 位 SHA 替换 HEAD。原始 Twitter created_at 的审阅证据可在 probe 中通过 `--tweet-time-evidence <reviewed.json>` 传入；缺失时 creation-based eligibility 保持不可验证。

### Step 8：StockNet 推文/价格/交易日审计

以下选择显式的临时规则日历，不能自动升级为官方日历已核查：

```bash
python scripts/pivot_sources/verify_stocknet.py --mode audit \
  --input-dir artifacts/pivot_sources/stage1_6/stocknet/probe \
  --calendar-provider exchange_calendars \
  --output-dir artifacts/pivot_sources/stage1_6/stocknet/audit
```

若有已审阅官方 CSV，改用 `--calendar-file <calendar.csv> --calendar-evidence <reviewed.json>`。若本地已有固定 Commit 的全价格文件，可另加 `--local-stocknet-root <root>`，其中应存在 price/raw，工具核对 Git blob，不下载。使用任何新输入后都换新输出目录。

### Step 9：三个组合的 As-of 覆盖诊断

```bash
python scripts/pivot_sources/verify_nws_iem.py --mode coverage \
  --input-dir artifacts/pivot_sources/stage1_6/nws/audit \
  --output-dir artifacts/pivot_sources/stage1_6/nws/coverage

python scripts/pivot_sources/verify_gfs_archive.py --mode coverage \
  --input-dir artifacts/pivot_sources/stage1_6/gfs/audit \
  --origin 2021-07-15T12:00:00Z \
  --output-dir artifacts/pivot_sources/stage1_6/gfs/coverage

python scripts/pivot_sources/verify_stocknet.py --mode coverage \
  --input-dir artifacts/pivot_sources/stage1_6/stocknet/audit \
  --calendar-provider exchange_calendars \
  --output-dir artifacts/pivot_sources/stage1_6/stocknet/coverage
```

GFS 的显式 origin 只用于 UTC Run/lag 小样本情景，不验证 Jena 数值时区，不解除 task Gate。取得 clock/calendar/local-file 证据后，coverage 中应重复传入相应参数，保持和 audit 同一证据策略；换新目录保存。NWS 未确定时区时不会计算伪造的 UTC 覆盖率，返回 blocked。所有率只限 Probe 日期和实际观察到的记录，population_rates 始终 null。

### Step 10：最终 Source Readiness 报告

```bash
python scripts/pivot_sources/summarize_stage1_6.py \
  --nws-dir artifacts/pivot_sources/stage1_6/nws/coverage \
  --gfs-dir artifacts/pivot_sources/stage1_6/gfs/coverage \
  --stocknet-dir artifacts/pivot_sources/stage1_6/stocknet/coverage \
  --pytest-report artifacts/pivot_sources/stage1_6/tests.xml \
  --output-dir artifacts/pivot_sources/stage1_6/readiness \
  --decision-report docs/pivot_sources/01_6_source_readiness_decision.md

python -m json.tool artifacts/pivot_sources/stage1_6/readiness/stage1_6_brief_summary.json
```

最终决策 Markdown **仅由此步骤实际生成**，当前没有该文件，也未填写虚假的通过结论。三个新测试模块之外还检查工程安全模块，共四个新模块的 JUnit 必须出现且无失败/错误/跳过；这是用户提供的运行记录，不是独立执行证明。

## 7. 输出与验收

`artifacts/pivot_sources/stage1_6/{nws,gfs,stocknet}/{probe,audit,coverage}/` 每个成功运行包含：

- source_inventory.json
- sample_records.jsonl
- timestamp_semantics.json
- asof_policy_audit.json
- availability_scenarios.json
- coverage_probe.json
- required_download_estimate.json
- known_risks.json
- source_readiness.json
- run_binding.json、_SUCCESS.json

probe 另外包含 raw/*.raw 和 http_manifest.json；派生 audit/coverage 引用原 Probe 的绝对目录并验证其封存哈希。资料页 Probe 也输出同样字段，但 actual_data_access_checked=false、记录为空，不算历史数据已查。

汇总目录包含 stage1_6_brief_summary.json、source_readiness.json、01_6_source_readiness_decision.md 和哈希标记。默认所有全时期覆盖状态均为 not_yet_full_coverage_verified。

正式大规模下载前需独立审核：

- Gate A：来源、版本、实际访问方式、固定文件标识、真实规模/权限/配额；网页访问不能代替数据访问。
- Gate B：来源/接收/运行/有效时间的语义和证据；明确条件延迟；数值物理时间可对齐；没有未来实况文字。
- Gate C：全 20/21/88 通道、完整候选 L/H、足够窗口和共同交易日、文本覆盖及公平外部信息约束。

任一 blocked 为当前 NO-GO；其余未完整核查则 CONDITIONAL。当前受限 Probe 不会自动产生全 pass，不能拿代码完成、测试完成或少量样本替代全时期验收。下一步是用户运行并返回证据和摘要；不启动 Stage 2，不联系作者。
