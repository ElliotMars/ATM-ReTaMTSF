# PIVOT — Stage 1.6：T1 External Text Source Verification

你是一名熟悉时间序列预测、多模态学习、数据工程、历史数据可用性审计、气象预报档案和金融时间序列的高级研究工程师。

当前仓库：

https://github.com/ElliotMars/ATM-ReTaMTSF

研究项目：PIVOT

目标期刊：Information Sciences。

当前工作进度：

- Stage 0：仓库与来源审计，已完成。
- Stage 1：Fidel-TS 数据下载与完整性验证，已完成。
- Stage 1.5：CAISO/Jena 数值数据清洗、N1/N2 协议及文本 provenance 审计，已完成主要验收。
- Stage 2：尚未开始。

本次只实现 **Stage 1.6：三个外部文本来源的 T1 可行性验证工具**。

注意：本阶段不进行大规模数据下载、不构建完整文本库、不训练模型，也不实施 Stage 2。

---

# 1. 执行权限：必须严格遵守

用户负责运行所有命令。

你只能：

- 阅读现有文件与代码。
- 检查 Stage 1.5 的输出。
- 编写 Python 检查工具。
- 编写数据源适配器。
- 编写配置和合成测试。
- 编写来源说明和验收文档。
- 提供用户手动执行的命令。

禁止：

- 执行 Python 或 pytest。
- 执行任何 Shell 命令来运行代码。
- 下载任何数据。
- 访问 NWS/IEM、NCAR、NOAA、GitHub 或其他网络 API。
- 安装依赖。
- 执行数据预处理。
- 生成 Embedding。
- 调用 LLM。
- 使用 GPU。
- 训练或评估模型。
- 修改已验收的 N1/N2 数据。
- 修改旧 ReTaMTSF 模型。
- 修改 Stage 0/1/1.5 历史审计结果。
- 自动 Commit、Push、Merge。
- 进入 Stage 2。

即使检查可以很快完成，也不能替用户运行。

**所有网络请求功能必须是显式 CLI，由用户手动触发；导入模块或运行离线测试时不得隐式联网。**

不需要联系数据集作者。

---

# 2. Stage 1.6 的核心目标

我们已经决定使用三个数据组合：

A. CAISO + NWS/IEM 历史天气预报文本。

B. Jena + NOAA GFS 历史预报数据生成的文本。

C. StockNet + 历史 Twitter 文本。

本阶段需要在下载完整数据集之前确定：

1. 数据源是否真实存在。
2. 原始记录是否含有可核查的历史时间字段。
3. 该字段代表发布时间、模型运行时间、预测有效时间还是观测时间。
4. 是否具有真实 As-of 可用性证据。
5. 能否与对应数值数据时间对齐。
6. 是否覆盖 Train/Validation/Test。
7. 是否能够提供足够的历史文本候选。
8. 是否存在无法解决的泄漏风险。
9. 完整下载的规模与成本。
10. 是否应当允许该来源进入下一阶段。

**不能仅因为 API 可访问、存在时间戳或文本与数值高度相关，就自动判定 T1 验证通过。**

最终每个数据组合必须独立输出 GO / CONDITIONAL / NO-GO 决策及证据。

---

# 3. 预设实验任务

## 3.1 CAISO + NWS

数值数据：

Stage 1.5 已准备的 CAISO N2。

- 20 个数值通道。
- 全通道 MIMO。
- 频率：1 小时。
- L=360。
- H={24,72,168}。

必须保留全部 20 个通道。

不得增加 18 通道实验或删除 Day ahead forecast、Hour ahead forecast。

训练划分：

- Train：2021-01-01 之前。
- Validation：2021 年。
- Test：2022-01-01 及之后。

文本：

- 美国国家气象局 NWS。
- Iowa Environmental Mesonet（IEM）历史 NWS 文本档案。
- 优先 Area Forecast Discussion（AFD）。
- 优先 AFDMTR、AFDSGX。
- 保留预报办公室和地理区域信息。

## 3.2 Jena + GFS

数值数据：

Stage 1.5 已准备的 Jena N2。

- 21 个数值通道。
- 全通道 MIMO。
- 频率：1 小时。
- L=360。
- H={24,72,168,336}。

数值历史仍使用原有 N2 数据。

外部天气预测采用 NOAA GFS 历史运行档案。

优先调查：

https://gdex.ucar.edu/datasets/d084001/

GFS 具有历史运行批次和预报步长，但要区分：

- run_time
- model_initialization_time
- forecast_lead
- valid_time
- file_publication_time
- available_at

不要将 run_time 直接视为 available_at。

根据官方档案，2015 年起有历史运行，00/06/12/18 UTC 每天四次，预报最长约 384 小时。

2014 年 Jena 历史数据早于该档案覆盖，不能假设具有 GFS 文本。

必须审计对训练集样本量的影响。

## 3.3 StockNet

来源：

https://github.com/yumoxu/stocknet-dataset

候选数值：

- 88 只股票。
- 日级数值。
- MIMO 收益率回归。

候选任务：

- L=20 个交易日。
- H={1,3,5} 个交易日。

候选日期划分：

- Train：2014-01 至 2015-06。
- Validation：2015-07 至 2015-09。
- Test：2015-10 至 2015-12。

这是计划采用的划分，不代表原论文官方划分。

保留股票顺序和固定股票 Universe。

如果部分股票缺少推文，不允许自动删除该股票；应记录文本覆盖情况并支持缺失文本 Mask。

必须检查所有股票是否具备一致的交易日历和足够的数值观测。

若 88 只股票的共同交易日不足，或需要大范围伪造标签，必须标记为 CONDITIONAL / NO-GO，而不是偷偷减少目标通道。

---

# 4. 统一的 T1 As-of 定义

所有数据集使用统一原则：

对于预测起点 t，只有预测时已经真实可用的信息才能参与检索。

候选文本集合：

D(t) = { d_i | available_at(d_i) <= t }

每份文本至少保留：

- dataset
- source
- source_record_id
- document_id
- text
- source_url
- source_file
- source_time_key
- issued_at
- received_at
- run_time
- available_at
- valid_start
- valid_end
- timezone
- location_id / entity_id
- availability_status
- availability_evidence
- availability_policy
- raw_record_sha256

缺失字段使用 None，不允许自行填造。

必须区分：

- verified_source_time：源文件提供可靠的发行/创建时间。
- verified_received_time：存在可核查的接收时间。
- latency_assumption：使用明确声明的时间延迟假设。
- unknown：无法证实可用时间。

若使用保守延迟：

available_at = issued_at + latency

或者：

available_at = run_time + latency

必须标明 latency 是证实的事实还是实验假设。

两者不能混为一谈。

对于严格 T1，提供至少三种判定：

1. Strict Verified：已核查真实可用时间。
2. Source-time + Conservative Lag：有可信发行/创建时间，另加明确延迟假设。
3. Not Verifiable：无法支持该协议。

对于第二种情况，必须明确披露延迟假设，不能标记为逐条真实接收时间已验证。

必须在算法检索和相似度计算之前执行 As-of 过滤。

不能先从整个文本库检索 Top-K，再事后剔除未来报告。

---

# Part A：NWS/IEM 数据源验证

# 5. NWS/IEM 来源适配器

建议新增：

scripts/pivot_sources/nws_iem.py

以及：

scripts/pivot_sources/verify_nws_iem.py

仅编写代码，不执行。

参考来源：

https://mesonet.agron.iastate.edu/nws/text.php

https://mesonet.agron.iastate.edu/api/

需要检查：

- IEM 正式文档支持的查询方式。
- AFOS Product ID。
- AFDMTR。
- AFDSGX。
- NWS 产品时间戳。
- UTC 时间规则。
- 公告原文。
- 预报办公室。
- Product ID 唯一性。
- 归档记录的接收时间是否独立存在。

优先使用有正式文档支持的接口。

不要凭空猜测 API Endpoint、请求参数、响应字段。

如果所选接口不支持某个必须字段，应明确记录并提供可替代接口的调查说明。

不要使用网页爬取作为默认方案，除非官方公开接口确实无法满足需求。

## 5.1 NWS 小样本验证

实现可由用户手动运行的 Probe。

要求支持：

- 不联网的 Dry Run。
- 指定日期范围。
- 指定 AFOS ID。
- 指定最大请求数。
- 指定最大记录数。
- 指定输出目录。
- 显式允许网络的参数。
- 失败重试与限速。
- 请求超时。
- HTTP 错误处理。

小样本检查建议覆盖：

- 2018 年训练期。
- 2020 年训练期。
- 2021 年验证期。
- 2022 年测试期。
- 2024 年测试期。

每个时间段选择少量代表性日期。

这些是待验证日期，不得在未查询前宣称数据一定完整。

输出每份公告：

- 原文。
- Product ID。
- 时间戳。
- 来源中心。
- 所属预报办公室。
- 公告类型。
- 原始来源 URL。
- 请求获取时间。
- 正文哈希。
- 元数据。

注意：

当前下载时间不是历史文本的发布时间。

## 5.2 NWS 时间审计

确认：

- 公告标识中的 UTC 时间含义。
- 公告正文中的发行时间。
- IEM 产品元数据时间。
- 若存在接收时间，如何与发行时间区分。
- 同一公告修订与更正记录。
- 相同时间的多个版本。
- 是否存在产品重复。

对于无法核对历史接收时间的公告，不能假定它在发行时间瞬间就被所有用户获取。

至少提供：

- issued_at-based eligibility。
- issued_at + conservative lag eligibility。
- verified_received_at eligibility（仅在确有相应字段时）。

延迟场景可以配置，但不应把某个小时数声明为真实发行延迟。

## 5.3 CAISO 时区核查

Stage 1.5 的 CAISO 数值时间戳没有自动指定时区。

必须调查：

- CAISO 发布数据的时间语义。
- 原始数据源是否存在时区字段。
- Time 列与 UTC 的关系。
- DST 转换是否影响时间匹配。
- 与 NWS UTC 公告的实际对应关系。

不得直接：

timestamp.tz_localize("UTC")

也不得未经核实直接指定 America/Los_Angeles。

如果无法确立时间语义，应当将严格 T1 数值对齐标记为 blocked，并生成需要进一步核对的证据清单。

## 5.4 NWS 文本覆盖

针对 CAISO 的：

L=360

H={24,72,168}

检查：

- 各 Split 有多少预测起点。
- 过去 24 小时存在公告的比例。
- 过去 72 小时存在公告的比例。
- 过去 168 小时存在公告的比例。
- 最近一份公告的时间距离。
- 两个预报办公室分别的覆盖率。
- 没有历史文本候选的窗口数量。
- 可用文本随预测 Horizon 的分布。

本阶段仅对 Probe 样本进行覆盖估计。

不得将有限日期 Probe 的覆盖率冒充全时期覆盖率。

全时期精确覆盖率必须在完整元数据索引到位后才能报告。

---

# Part B：Jena + GFS 数据源验证

# 6. GFS 来源适配器

建议新增：

scripts/pivot_sources/gfs_archive.py

以及：

scripts/pivot_sources/verify_gfs_archive.py

优先以官方 NCAR GDEX、NOAA 公共 GFS 历史档案及其公开元数据为依据。

需要检查：

- 2015–2025 年历史数据能否实际获得。
- 各年份具体产品目录。
- 预报运行批次。
- 文件名中的 Cycle。
- Forecast Lead。
- 数据格点分辨率。
- Jena 经纬度对应格点。
- GRIB2 变量与 Level。
- 是否支持小区域裁剪或按变量提取。
- 可否避免下载整个全球 GRIB2。
- 历史文件的可用时间证据。
- 访问方式、权限、配额与可能的下载限制。

不要假设数据一定可以通过匿名 HTTP 直接下载。

如果需要认证、注册或存在费用，必须明确告知用户，不得自行注册或提交请求。

## 6.1 先做元数据 Probe

只准备查询目录和少量历史运行样本的功能。

代表性年份：

- 2015。
- 2018。
- 2020。
- 2021。
- 2022。
- 2024。

代表性运行时刻：

- 00 UTC。
- 06 UTC。
- 12 UTC。
- 18 UTC。

预测步长重点检查：

- 24 小时。
- 72 小时。
- 168 小时。
- 336 小时。

不能下载完整全球数据来完成这一步。

可以在用户显式允许时下载一个非常小的实际样本，或经过服务端裁剪的单个 Jena 区域样本，以核查 GRIB2 元数据。

如果无法支持小体积 Probe，就输出所需下载量和阻断原因。

## 6.2 GFS 真正的 As-of 检查

必须区分：

run_time

forecast_valid_time

file_publication_time

available_at

GFS 的运行批次开始时间不是完成时间，也不自动代表可以公开下载的时间。

实现延迟敏感性分析：

- 假设延迟 3 小时。
- 假设延迟 6 小时。
- 假设延迟 12 小时。

这三个值只作为预先指定的候选场景。

不能将其称为实际已验证发布延迟。

如果官方资料提供更可靠的实际发布信息，应优先使用真实证据，并记录来源。

输出：

- 在每个延迟场景下，哪些历史运行对预测起点可用。
- 每个 Horizon 的预报时间覆盖情况。
- 最新可用运行距离预测起点的时间。
- 是否包含合法的未来预报。
- 预报是否覆盖所有目标小时。
- 原始预报分辨率。

不得用未来运行替代历史时刻缺失的运行。

## 6.3 GFS 数值转文本的可行性

本阶段只设计确定性文字生成方案，不需要实际完成大规模转换。

使用固定模板，不调用 LLM。

建议支持：

- Temperature。
- Relative Humidity。
- Wind Speed。
- Surface Pressure。
- Precipitation（若变量真实可得）。

必须明确：

- 哪些变量在 GFS 历史文件中可得。
- 单位及换算规则。
- 预报格点与 Jena 实际站点的空间差异。
- 时间间隔。
- 预测有效时间。
- 生成文本的原始 Run ID。
- 对应 Forecast Lead。

不能使用 Jena 的未来真实天气值生成文本。

不能将 3/12 小时间隔的预报值描述为真实逐小时观测。

对没有直接数值支持的天气条件，不得添加主观描述。

要保留从原始预报文件到生成文本的完整 Provenance。

## 6.4 GFS 公平性要求

这是 PIVOT 论文必须保留的实验约束。

由于 GFS 本身就是强大的外部天气预测系统，不能只比较：

Jena History

vs

Jena History + GFS-generated Text。

后续必须有：

1. Jena 数值 Only。
2. Jena + 同一历史 GFS Run 的原始数值。
3. Jena + GFS 文本的简单融合。
4. Jena + GFS 文本的基础检索。
5. 完整 PIVOT。

所有使用 GFS 的模型必须共享相同的 As-of Run 集合及外部信息权限。

本阶段不实现这些 Baseline，只将其写入实验协议。

---

# Part C：StockNet 数据源验证

# 7. StockNet 来源适配器

建议新增：

scripts/pivot_sources/stocknet.py

scripts/pivot_sources/verify_stocknet.py

原始来源：

https://github.com/yumoxu/stocknet-dataset

本阶段首先准备文件清单检查与小样本下载流程。

优先检验：

- price/raw。
- price/preprocessed。
- tweet/raw。
- tweet/preprocessed。
- 股票清单。
- 文件格式。
- 实际日期范围。
- Twitter created_at 字段。
- 推文文本字段。
- 是否存在重复或无效文本。
- 股票代码与推文目录关系。
- 价格是否包含复权处理。

必须保留来源 Commit 或 Release 标识以及文件哈希。

不要在本阶段克隆完整 StockNet 仓库。

如果需要下载小样本，必须由用户显式运行带有限制的命令。

## 7.1 StockNet 的 MIMO 任务可行性

候选设置：

L=20 trading days。

H={1,3,5} trading days。

目标：

对固定股票 Universe 的未来逐日收益率进行多变量回归。

必须检查：

- 实际有价格数据的股票数量。
- 实际有文本数据的股票数量。
- 股票代码列表。
- 交易日期是否一致。
- 缺失交易日。
- 股票停牌/退市影响。
- 有效的共同交易日数量。
- 每个 Split 的样本数量。
- 每个 Horizon 的可用窗口数量。
- 无效标签比例。
- 是否发生个股目标集合的无意改变。

使用交易日索引，不把周末和节假日伪装成正常交易日。

没有推文的股票可以保留数值通道，并提供文本缺失 Mask。

不得自动删除没有推文的股票。

对于交易日之间的数值缺失，不能通过未来值回填来伪造收益率。

## 7.2 收益率定义与未来信息风险

本阶段需要比较以下候选定义：

- Close-to-Close Return。
- Log Return。
- Adjusted Close Return。

推荐使用逐日 Log Return 作为候选主目标，但必须先检查原始数据来源和价格调整机制。

特别注意：

事后计算的 Adjusted Close 可能引入历史时点尚未知晓的公司行为调整信息。

不得在没有审核来源的情况下，直接使用全时期事后复权价格作为严格 T1 输入。

如果使用调整后价格，应记录公司行为、调整规则，以及其对历史可用性的影响。

默认不把 StockNet 原论文的涨跌分类标签当作本研究的 MIMO 回归目标。

## 7.3 StockNet 时间对齐

Twitter created_at 必须转换为有时区的 UTC 时间戳。

交易日与收盘时间采用官方交易日历。

不能假设每天都有交易。

必须考虑美股夏令时变化。

建议初始预测起点定义为：

当日美国东部市场收盘后的固定、保守时间。

但这个时间必须是明确的实验配置，并检查对应日期是否为提前收盘日或其他特殊交易日。

只允许已创建且在预测起点前可获得的推文进入候选池。

对于已知发布时间之外的历史可用性，需明确使用 creation-time-based policy 或保守延迟假设。

推文的发布时间不能与股票交易日期混淆。

不得使用未来交易日的推文预测此前股票收益率。

## 7.4 StockNet 文本覆盖

按股票、日期、Split 统计：

- 推文数量。
- 无文本的股票。
- 无文本的交易日。
- 过去 1/3/5/20 交易日的文本覆盖。
- 推文创建时间范围。
- 同一推文重复出现。
- 推文 ID 重复。
- 文本内容缺失。
- 时间戳解析失败。
- 被引用/转发文本是否含有更早时间内容。

不要根据股票收益率挑选文本。

不能只保留推文很多或预测较容易的股票。

需要给出在固定 88 股票 Universe 下训练 MIMO 预测是否现实的判断。

---

# Part D：统一数据源准入决策

# 8. 建立三层准入标准

每个数据组合都需要独立检查。

## Gate A：Source Integrity

- 官方来源明确。
- 数据可实际访问。
- 来源版本可记录。
- 原始文件可校验。
- 文件规模与访问成本可接受。

## Gate B：Temporal Availability

- 时间字段语义明确。
- 能判断哪些信息在预测时刻之前可用。
- 历史可用性证据可追溯。
- 不把观测时间当作发行时间。
- 不把模型初始化时间当作数据可获取时间。
- 不依靠未来实况生成外部文本。
- 无法核实处明确记录假设或阻断。

## Gate C：Forecast Task Compatibility

- 数值数据有足够样本。
- 全通道目标定义合理。
- L/H 可用。
- 时间划分可复现。
- 文本候选覆盖可统计。
- 没有强制使用未来文本。
- 所有 Baseline 可以采用相同信息约束。

三个 Gate 分别输出：

- pass
- conditional
- blocked
- not_checked

严格禁止将 not_checked 当作 pass。

## 8.1 最终状态

每个数据集需要输出：

- source_status
- temporal_availability_status
- task_compatibility_status
- verified_t1_feasible
- conditional_t1_feasible
- blocking_issues
- evidence
- next_required_actions

注意：

“工具写好”不代表数据集已通过验证。

“少量样本通过”不代表全部历史年份都通过验证。

“日期覆盖”不代表真实文本完整覆盖。

“具备历史 Run Time”不代表具备真实发布时间。

---

# 9. 文件布局

建议新增以下目录：

configs/pivot_sources/

scripts/pivot_sources/

tests/pivot_sources/

docs/pivot_sources/

artifacts/pivot_sources/stage1_6/

具体文件可根据现有项目结构优化，但不能破坏 Stage 1.5。

至少包含：

- 来源配置文件。
- 公共 As-of 时间判定模块。
- NWS/IEM 适配器。
- GFS 适配器。
- StockNet 适配器。
- Probe CLI。
- 离线合成测试。
- 完整数据源准入文档。
- 结果汇总 CLI。

建议配置：

configs/pivot_sources/stage1_6_sources.json

建议文档：

docs/pivot_sources/01_6_source_verification_protocol.md

docs/pivot_sources/01_6_experiment_matrix.md

docs/pivot_sources/01_6_source_readiness_decision.md

其中最终 Readiness 文档应由真实 Probe 结果生成，不得提前填写虚假的通过结论。

---

# 10. 网络请求和数据安全

所有网络代码默认离线。

仅当用户显式使用允许联网的 CLI 选项时才能发起请求。

实现：

- 最大 HTTP 请求数。
- 最大单文件字节数。
- 最大总下载字节数。
- 超时。
- 重试。
- 指数退避。
- 访问限速。
- 清晰的 User-Agent。
- HTTP 响应状态检查。
- 下载文件哈希。
- 中断后可恢复的 Manifest。
- 不覆盖已有文件。
- 原始返回内容与解析结果分开保存。

如果外部服务需要登录或授权：

- 不自动注册。
- 不绕过认证。
- 不把密钥写入配置或日志。
- 提供人工认证说明。

Stage 1.6 的 Probe 不允许触发整套 GFS 全球网格下载。

StockNet 不允许默认完整 Git Clone。

NWS 不允许默认请求全部年份的完整公告。

---

# 11. 离线测试

新增合成 Fixture 测试。

至少覆盖：

### As-of 通用逻辑

- 不同时间时区比较。
- 发布时刻等于预测起点。
- 发行时间晚于预测起点。
- 接收时间晚于发行时间。
- 缺失 available_at。
- 带保守延迟的判定。
- 有效预报时间在未来但发布在过去。
- 不确定时间字段被阻断。

### NWS

- UTC Product ID 解析。
- 重复公告。
- 更新公告。
- 文本缺失。
- 不同预报办公室。
- 发布时间与接收时间不一致。
- 归档时间不等于发行时间。
- 数据范围外请求被拒绝。

### GFS

- Run Time 与 Valid Time。
- 不同 Forecast Lead。
- 3/6/12 小时发布延迟场景。
- 预测时最新可用 Run。
- 未来运行不得参与过去预测。
- 预报时效不足。
- 历史文件缺失。
- 多模型运行版本区分。
- 不得从未来实况生成文本。

### StockNet

- created_at 解析。
- 纽约交易时间。
- 夏令时。
- 周末与假日。
- 提前收盘。
- 多股票交易日对齐。
- 股票没有推文。
- 股票缺少价格。
- 交易日收益率。
- 未来复权数据风险标记。
- 训练/验证/测试边界。

### 工程安全

- 离线模式不联网。
- 请求数量限制。
- 下载体积限制。
- 不覆盖原始文件。
- 重复请求去重。
- 失败响应不写入成功 Manifest。
- 产物哈希一致性。

只编写测试，不执行。

---

# 12. Stage 1.6 的运行顺序

代码完成后，给用户提供可以直接复制的实际 CLI 命令。

按以下顺序：

Step 1：全部离线单元测试。

Step 2：三个来源的纯本地配置 Dry Run。

Step 3：NWS 少量历史公告 Probe。

Step 4：NWS 发布时间/接收时间/CAISO 时区检查。

Step 5：GFS 历史文件目录元数据 Probe。

Step 6：GFS 少量 Run/Lead 和发布延迟场景检查。

Step 7：StockNet 小样本文件清单与元数据 Probe。

Step 8：StockNet 推文时间与价格交易日检查。

Step 9：三个数据组合的 As-of 覆盖诊断。

Step 10：生成最终 Source Readiness 报告。

对于需要全时期数据才能得到的统计，只能输出：

not_yet_full_coverage_verified

不能根据 Probe 估计直接声明正式全时期覆盖率。

所有命令必须与真实实现的 CLI 参数一致。

**你不能执行其中任何一步。**

---

# 13. 本阶段的输出

三个数据组合分别生成：

- source_inventory.json
- sample_records.jsonl
- timestamp_semantics.json
- asof_policy_audit.json
- availability_scenarios.json
- coverage_probe.json
- required_download_estimate.json
- known_risks.json
- source_readiness.json

汇总：

- stage1_6_brief_summary.json
- 01_6_source_readiness_decision.md

每个结论必须携带来源证据。

缺失证据时应明确为 unknown / conditional / blocked。

不得生成虚假的覆盖数量、发行时间或文件大小。

---

# 14. 与后续 Stage 2 的关系

本阶段不要实现 Dataloader。

Stage 2 仍负责正式 MIMO 数据加载。

完成 Stage 1.6 后，我们应该能够明确：

### CAISO

- NWS/IEM 是否可以支持 T1。
- 哪些真实发布时间字段可以使用。
- 是否存在完整或部分历史覆盖。
- CAISO 时区语义是否能够对齐。
- 20 通道 N2 是否可以与 NWS 组合。

### Jena

- GFS 历史运行档案是否实际可访问。
- 运行时间和有效时间是否可核查。
- 可用延迟假设如何定义。
- 各个 Horizon 是否有足够预报覆盖。
- GFS 转文本是否科学合理。
- 是否需要将 T1 状态限定为 Conditional As-of。

### StockNet

- 88 只股票是否能建立统一 MIMO 任务。
- Tweet created_at 是否可以作为历史发表时间依据。
- 文本与交易日能否严格匹配。
- 是否存在复权价格或交易日对齐风险。
- L=20、H={1,3,5} 是否能获得足够样本。
- 是否需要根据数据实际情况调整任务定义。

---

# 15. 最终回复要求

完成代码后按以下顺序汇报：

1. 已完成的 Stage 1.6 文件修改。
2. 文件清单。
3. NWS 适配器功能。
4. GFS 适配器功能。
5. StockNet 适配器功能。
6. 三个来源的 As-of 约束。
7. 已知的科学风险。
8. 尚未通过实际数据确认的问题。
9. 离线测试命令。
10. 三套 Probe 的手动运行命令。
11. 全部输出目录。
12. 正式大规模下载前的 Go/No-Go 条件。
13. 下一阶段建议。

必须明确：

“Stage 1.6 的代码、测试与协议文档已经编写，但全部单元测试、外部来源 Probe 和真实数据验证均尚未执行。”

不得宣称尚未运行的检查通过。

不得提前宣称三个数据组合均已通过严格 T1。

不得下载完整数据集。

不得进入 Stage 2。

完成后立即停止，等待用户手动运行和返回结果。