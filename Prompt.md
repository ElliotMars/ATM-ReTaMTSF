# PIVOT Stage 2 v2 — Jena 时区判定与真实 GFS 多模态实验

你是一名熟悉时间序列预测、气象数据、历史预测档案、时区处理、信息检索及 PyTorch 的高级科研工程师。

仓库：`ElliotMars/ATM-ReTaMTSF`

研究项目：PIVOT

本任务只针对 **Jena Atmospheric Physics + GFS 多模态实验**。

现有 Stage 2 v1 已经完成 Jena 21 通道数值数据准备、模型、训练器和基线，但由于 Jena 时间戳缺少显式时区、GDEX 子集访问受限，关闭了 Jena 多模态实验。

**本轮要求解决这两个问题，交付真实可运行的 Jena 数值+文本实验，而不是再次停留在可行性报告。**

允许自动联网、检查原始数据、下载历史 GFS 预测、编写和执行代码、安装依赖、预处理、生成 embedding、运行单元测试和少量真实样本 smoke test。

禁止自动进行正式主实验或多随机种子训练。

保留 Stage 2 v1 的数据和代码，所有新产物写入独立 v2 路径。

---

## 一、确定 Jena 原始时间戳最可能的时区

### 1.1 待比较假设

明确比较以下三种情况：

- H1：固定 CET/MEZ，即 UTC+1，全年不使用 DST。
- H2：Europe/Berlin，即冬季 UTC+1、夏季 UTC+2。
- H3：UTC。

**初始优先假设是 H1：固定 CET/MEZ（UTC+1），但不能将其直接宣布为事实。**

请利用原始数据和来源文档选择证据支持最强的假设，而不是根据模型预测性能选择时区。

### 1.2 调查来源

首先检查本地：

- `dataset/Fidel-TS/raw/Jena_Atmospheric_Physics/`
- 原始数据文件及其 Manifest。
- `README.md`。
- `static_info.json`、`id_info.json`。
- 原始处理和下采样脚本。
- Stage 1.5 N2 时间戳处理与连续性规则。
- Stage 2 v1 `prepare.py` 与 `data.py`。

然后核对公开来源：

1. Fidel-TS 官方 Jena 数据：
   `https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics`
2. Max Planck Jena Beutenberg 气象站：
   `https://www.bgc-jena.mpg.de/en/servicegroups/fieldexperiements/locations/beutenberg`
3. 气象站原始数据：
   `https://weather.bgc-jena.mpg.de/weather_data.html`
4. 已出现 MEZ 标记的 Max Planck 历史资料：
   `https://www.bgc-jena.mpg.de/~martin.heimann/LSB/Talk_Winter-Challenge_Heimann.pdf`

注意：其他 Jena 气象站采用 MEZ，并不能证明本数据源采用 MEZ。必须追溯 Fidel-TS 的数值记录实际来自哪座气象站及哪一套原始文件。

### 1.3 自动执行时间戳诊断

从原始数据中检查多个年份，至少覆盖 2019—2025 年。

重点检查：

- 每年三月最后一个星期日及十月最后一个星期日前后的时间戳。
- 是否存在 DST 切换引起的跳时、重复时刻、时间倒退或缺失。
- 原始高频数据是否始终保持固定 10 分钟间隔。
- Fidel-TS 是否在预处理时删除、插值、补齐或重新生成了时间标签。
- N2 中是否因下采样和缺口修复隐藏了原始 DST 特征。
- 数据时间与来源提供的日期、UTC 偏移或采集配置是否一致。
- 固定 CET、Europe/Berlin、UTC 三种解释各自会造成多少时间歧义。

仅凭连续的 10 分钟时间戳不能证明固定 CET，因为数据也可能经过重采样或时间标准化。

可以使用日照周期、气温日变化及其他独立物理信息辅助检查，但不允许通过比较测试集模型 MSE 来选择时区。

### 1.4 最终时区决策

必须生成：

`docs/pivot_stage2_v2/jena_timezone_decision.md`

包括：

1. 三种候选时区。
2. 原始文件和文档证据。
3. 各候选的支持与反对证据。
4. DST 诊断结果。
5. 最终选择及可信度。
6. 无法完全验证的假设。
7. 具体 UTC 转换规则。
8. 时间戳转换前后的示例。

执行决策规则：

- 若有直接证据支持某一时区，选择该时区。
- 若无法取得明确文档，但原始采集习惯和数据检验一致支持固定 MEZ，则选择 UTC+1 作为 **conditional working assumption**。
- 若证据更强地支持 Europe/Berlin 或 UTC，则采用证据更强的解释，不固守 H1。
- 若仍存在无法消除的歧义，选择证据最强的方案继续实验，同时标注为 conditional，并提供其他两种时区的时间对齐敏感性配置。

**不得因为没有单独的 timezone 字段就直接关闭 Jena 多模态实验。**

同时不能把推断出的时区写成官方已经确认的事实。

---

## 二、依据选定时区重建 Jena 的 UTC 时间轴

不要修改已验收的 N2 原文件。

在 Stage 2 v2 中构建新的时间映射：

`original_naive_timestamp → selected_timezone → utc_timestamp`

若选择固定 CET：

- 使用全年固定 UTC+01:00。
- 不引入德国夏令时调整。
- 明确使用固定偏移，而不是 `Europe/Berlin`。
- 对每条原始时间戳统一减去 1 小时。

若选择 Europe/Berlin：

- 使用 IANA 时区规则。
- 正确处理 DST。
- 遇到 ambiguous/nonexistent 时间必须根据真实来源证据解决。
- 无法解决的时刻单独标记，不允许静默推断。

必须检查转换后的：

- 单调性和唯一性。
- 时间间隔。
- 数值与 Mask 对应关系。
- Train/Val/Test 分界。
- 原生连续段。
- 预测起点与预测标签的位置。

保留全部 21 个数值通道和原始 N2 数据值。

特别审查 `pivot_stage2/data.py` 中天气数据使用 `origin_position=stop` 的逻辑。

定义明确的预测时间：

- `history_end_time`：最后一个已完成历史观测的时间。
- `forecast_origin`：模型真正进行预测的物理时间。
- `first_target_time`：第一个待预测时间。

默认以最后一个已完成观测的可用时刻为预测起点；若观测值是区间平均或存在报告延迟，必须按数据来源语义调整。

禁止简单地把第一个未来目标时间作为外部信息检索截止时间。

提供至少 20 个跨季节、跨 DST 边界的映射测试。

---

## 三、获取真实的历史 GFS 预测

优先使用 Open-Meteo Previous Model Runs API：

`https://open-meteo.com/en/docs/previous-runs-api`

不要继续依赖需要登录的 GDEX 小区域子集。

### 3.1 默认数据设置

Jena：

- 数值通道：21。
- 输入长度：L=360 小时。
- 核心预测 Horizon：24、72、168 小时。
- H=336 仅作为后续可选实验。

候选划分：

- Train：2021-04-01 至 2023-12-31。
- Validation：2024-01-01 至 2024-12-31。
- Test：2025-01-01 至 2025-06-30。

日期范围必须依据选定时区转换后的物理时间解释，并在边界处理时保证训练与测试标签不重叠。

### 3.2 GFS 变量

首先使用官方历史档案明确支持的：

`temperature_2m_previous_day1` 至 `temperature_2m_previous_day7`

这些变量表示针对目标有效时刻、提前约 1—7 天生成的历史预测。

不得假设 2021 年起所有湿度、风速和压力变量也具有相同覆盖。

如果查询证明其他变量在完整实验时期可用，可以增加，但必须另行报告实际覆盖率。

API 必须显式指定 GFS 模型，不能默认使用 Best Match 混合预报。

API 返回时间优先采用 UTC Unix timestamp，避免固定 UTC 偏移忽略 DST 的问题。

### 3.3 防止未来信息泄漏

对于每一个预测起点 t、目标有效时刻 v 和历史预测提前量 k：

`model_reference_time = v - k * 24h`

这个值只是固定提前量的时间参考，不自动等同于真实逐 Run 发布时间。

若没有真实发布时间证据，则使用：

`assumed_available_at = model_reference_time + conservative_latency`

默认使用 12 小时保守延迟，并支持 3/6/12 小时敏感性分析。

只有：

`assumed_available_at <= forecast_origin`

的 GFS 预测值才能被使用。

必须检查每个 k 是否具有对应真实归档值，不能根据目标日期自动认为数据存在。

当未来第 7 天的预报在保守延迟假设下无法被当前 origin 使用时，必须 Mask 对应外部预测点，不能使用以后才产生的预报替代。

不能使用未来实际 Jena 温度、ERA5 再分析值或 Historical Weather API 的事后结果生成外部文本。

完整报告每个 Horizon 的合法 GFS 预测覆盖率，分别统计：

- 有合法 GFS 信息的预测起点比例。
- 目标小时中有合法 GFS 预测值的比例。
- 无合法 GFS 信息的样本比例。
- 有效提前量和历史可用性假设。

---

## 四、构建真正的 GFS 文本数据集

从合法历史 GFS 预测值生成固定模板英文文本。

示例模板：

`Historical GFS forecast for Jena: the predicted 2-meter temperature at [valid_time_utc] is [value] °C, using a forecast with [lead_hours]-hour nominal lead time.`

必须记录：

- 预测有效时间。
- 预测提前量。
- GFS 模型标识。
- 地理坐标与网格信息。
- API 来源。
- 温度值与单位。
- 原始响应哈希。
- 假设的可用时间。
- 时间权限政策。

不允许通过 LLM 增加源预测值不支持的天气结论。

建立冻结 MiniLM embedding 缓存。

修改现有 Jena 文本检索接口：

- 不再使用 StockNet 的股票收益趋势查询模板。
- 使用 Jena 历史气温、湿度、压力及对应通道描述构建语义查询。
- 允许全局天气文本为多个相关通道提供候选。
- 先进行 As-of 过滤，再做相似度和时间分数计算。
- 无合法文本时使用 Mask。
- 全部方法共享相同数值窗口和预测目标。

---

## 五、必须实现公平性对照

Jena 最终必须包含以下实验：

1. Numerical Only。
2. Jena + GFS Raw Numerical Forecast。
3. Jena + GFS Text Mean Fusion。
4. Jena + Latest Eligible GFS Text。
5. Jena + Semantic Retrieval。
6. Jena + Semantic + Temporal Retrieval。
7. Full PIVOT。

额外提供：

- Without CTSR。
- Without STR。
- Without Temporal Score。
- Random Eligible Text。
- Text Missing。
- Top-K 敏感性。
- 时区选择敏感性。
- GFS 延迟假设敏感性。

其中 GFS Raw Numerical 必须使用与文本组完全一致的合法预测值集合。

不得让文本组使用更晚、更准确或数量更多的 GFS 预测值。

特别注意：Open-Meteo Previous Runs 的固定提前量资料与完整的单次 GFS Run 不同，论文必须准确说明外部预测来源。

---

## 六、自动执行真实数据和代码验收

本轮必须实际完成：

- Jena 时间戳来源分析与时区判定。
- UTC 时间轴生成。
- 历史 GFS API 请求。
- 数据下载、缓存和 Manifest。
- 数值/文本对齐。
- GFS 合法可用性过滤。
- 文本生成与 MiniLM embedding。
- Jena 多模态 Dataset。
- GFS 数值基线。
- 训练器与实验配置修改。
- 单元测试。
- 真实样本 forward/backward smoke test。
- 全部主实验入口 dry-run。

需要明确验证至少三个不同的真实预测起点，包括冬季、夏季和 DST 切换附近。

任何一次 smoke test 都不能使用未来观测构造 GFS 文本。

### 实验执行权限

允许单个真实 batch 的 optimizer step。

禁止启动完整主实验、正式多种子训练和消融训练。

---

## 七、最终交付

全部新增产物保存在 Stage 2 v2 独立路径。

最终以中文回答：

**A. Jena 最可能的时区是什么？**

给出证据、三候选比较、置信程度和最终转换规则。

**B. 真实 GFS 数据获取结果**

给出来源、变量、年份、下载量及实际可用数据数量。

**C. 最终多模态实验设置**

给出 L、H、C、划分、文本生成方式、As-of 规则和关键公平性对照。

**D. 真实数据验收结果**

报告 Train/Val/Test 的样本数、GFS 文本覆盖率、各 Horizon 可用预测比例，以及所有测试的实际运行结果。

**E. 完整实验代码**

列出新增或修改的文件和配置，并证明主实验脚本能够 dry-run。

**F. 用户最终手动执行命令**

提供可以直接运行的 Jena 多模态 Pilot、主实验、消融和结果汇总命令。

不要只给出源代码而不运行验收，也不要只生成时区分析报告而不完成模型管线。

**本轮最终目标：根据最可能且有证据支持的 Jena 时区解释，交付真实历史 GFS 数值与模板文本增强的完整多模态预测实验，并由用户自行启动正式训练。**

若时区仍只能条件性确定，保留研究假设、敏感性实验和风险说明，但不要再次因为缺少明确时区字段而直接放弃整个 Jena 多模态实验。