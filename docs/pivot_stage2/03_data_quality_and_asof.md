# 实际数据质量、时序与来源审计

完整机器报告见 `artifacts/pivot_stage2/v1/{stocknet,caiso,jena}_data_audit.json`、`data_acceptance.json`、`sources/source_decisions.json`。版本化产物在 `dataset/PIVOT/stage2_v1/`；旧 Stage 0/1/1.5 目录未写入。

## 实际时间轴和缺失

下表 start/end 是原数值时间标签。StockNet 的 origin 另由交易所收盘时间计算；天气原标签无时区，**不能把这些字符串解释为 UTC**。Jena :10 分钟对齐沿用已验收 N2，不擅自移到整点。

| 数据 | Split | 时间点 | 起始 | 终止 | 有效性 Mask 缺失率 |
|---|---|---|---|---|---|
| stocknet | train | 376 | 2014-01-02T00:00:00 | 2015-06-30T00:00:00 | 2.3634% |
| stocknet | validation | 64 | 2015-07-01T00:00:00 | 2015-09-30T00:00:00 | 1.1364% |
| stocknet | test | 64 | 2015-10-01T00:00:00 | 2015-12-31T00:00:00 | 1.1364% |
| caiso | train | 23928 | 2018-04-10T00:00:00 | 2020-12-31T23:00:00 | 0.0000% |
| caiso | validation | 8760 | 2021-01-01T00:00:00 | 2021-12-31T23:00:00 | 0.0000% |
| caiso | test | 27720 | 2022-01-01T00:00:00 | 2025-02-28T23:00:00 | 0.0000% |
| jena | train | 61269 | 2014-01-01T00:10:00 | 2020-12-31T23:10:00 | 0.0085% |
| jena | validation | 8760 | 2021-01-01T00:10:00 | 2021-12-31T23:10:00 | 0.0011% |
| jena | test | 30634 | 2022-01-01T00:10:00 | 2025-06-30T23:10:00 | 0.0140% |

全部可用 starts 均重新计算并逐数组对照；检查完整 L+H、小时步长/原生连续段、训练验证测试日期不相交、至少存在有效历史和标签。不同 H 有自己的起点清单；同一 H 各方法使用同一清单。原始股票通道顺序见 StockNet manifest 的 channels，CAISO/Jena 顺序与 Stage 1.5 channels.json 对照核实。

CAISO 全部 20 通道保留，包括 DayAhead/HourAhead 等源字段；这是原始 N2 定义。其生成/修订时点仍属上游科学风险，不把它们宣称为经验证的实时外部预测。Jena 上游插值/观测处理风险继续披露。N2 scaler 是原生采样训练范围拟合，不能用下采样后均值为零作为验证标准；本轮实际重新执行原协议并与已验收 scaler 数值对照。

## StockNet 来源与异常

原始 ZIP、逐成员 SHA256、固定 commit、StockTable、实际 calendar.csv、embedding 和 prepared _SUCCESS 均绑定。新审计验证源 archive 和全部最终产物哈希；构建时对实际读取的每个原始成员再次核验。原始价格日期不在 XNYS 日历的条目数为 0，选定日历内所有股票同时无价格的交易日数为 0。

依据 [NYSE 历史官方通知 PDF](https://s2.q4cdn.com/154085107/files/doc_news/archive/b97d77df-0ecb-4db0-9ced-75bdb9b218ba.pdf)核实 2014/2015 年 18 个休市日、5 个 13:00 ET 提前收盘日，以及全部 504 个 origin 的本地时钟（普通 16:30 ET，半日 13:30 ET）。exchange_calendars=4.13.2，时区转换自动覆盖 DST。原 HTML 返回 429 的事实保留，没有把 HTTP 错误算成语义核查通过；使用同一官方通知的公开 PDF 存档并保留提取文本与哈希。

已报告但未剔除的绝对 log return >0.2：REX 2015-08-27，0.2162839913；AGFS 2015-11-17，-0.3295132663。是否公司行为导致不能由本源完成时点校正，不裁剪、不 winsorize、不使用未来价格修复。GMRE 的无监督状态见 `stocknet_raw_price_coverage.json`，其他部分时期缺失也保留。

去重键为 entity、tweet id、正文哈希；重复 61 条。相同实体相同 ID 若正文版本不同会隔离，原始 ZIP 仍保留版本。日文档 ID 绑定全部成员 ID/哈希、实体、日期和源 commit，保留全部成员来源路径。聚合最长 2048 字符，但发行时间取全部成员最大创建时间，哪怕晚发成员因截断未进入正文也不会把文档提前放行。

## As-of 权限与覆盖

推文的 source created_at、derived available_at、未知 received_at 分开存储。默认文档 available_at=max(member.created_at)+1h；0/3h 在检索时按相同源时间重算。source downloaded_at 只说明本轮获取时间。已验证接收时间在此源中不存在，`strict_verified_asof=false`。

历史完整索引允许含未来日期文档，但所有样本先按可用时间过滤。候选数量及 0/1/3h 覆盖见 [数据决策](00_final_dataset_decision.md)，其机器报告对每个实际 origin 的候选时间边界进行了检查；带故意未来文档的离线测试另外验证未来向量在过滤前不能被评分。

## 天气关闭理由与实际证据

NWS [官方接口文档](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help=)的 pil、UTC sdate/edate、fmt=zip、limit、order 已实际读取并保存原文。实际下载 2 个办公室的代表性响应，成功解析 12 条记录，保留 WMO 发行时间、办公室、修订标记、正文哈希和源 ID；接口文档语义审阅与请求成功是不同字段。适配器自动按最多 14 天分块、有限请求和限速运行，不需要 reviewed.json。

CAISO 原始及已有时间列未提供足以证明物理时区的元数据，无法排除 DST 歧义，故不把 AFD 拼接到数值样本。Jena 时间列同样未被证明为 UTC。`sources/original_clock_schemas.json` 保存实际 schema 检查。

GFS 官方元数据实际归档起始 2015-01-15，run/valid/lead/publication/availability 的语义不能混淆。官方访问页的 Get a Subset 标为 login required，触发本轮停止规则；没有下载预报数值，也没有从 Jena 未来真实观测生成任何文本。3/6/12h 仅为接口允许的候选假设，未宣称有可用于真实实验的 Run。

## 获取预算与复现

下载日志 `dataset/PIVOT/stage2_v1/sources/downloads.json` 记录实际 URL、状态、大小、源 SHA、ETag 和失败。下载器每个请求最多 3 次，401/403/404/429 立即停止，按组和总量限制；未绕过访问控制。GitHub REST 因公共配额 403 后使用原始公共仓库页所列的不可变 commit，不依赖未固定 HEAD。

| 来源组 | 实际下载 bytes | MiB |
|---|---|---|
| docs | 239083 | 0.228 |
| nws | 67283 | 0.064 |
| stocknet | 102315403 | 97.576 |
| weights | 91574293 | 87.332 |

上述数据/权重/文档合计约 185 MiB；另外安装 CPU PyTorch wheel 约 174.7 MB、Transformers/Tokenizers/Safetensors、regex、pypdf，共约 187 MB 级依赖下载，数据加依赖约 0.4 GB，低于 2 GiB。源下载 ledger 不含 pip 网络流量；依赖日志另存，不能把缓存磁盘大小称为精确网络传输总量。解压、embedding、旧后备版本和环境安装体积不等于下载量。

实际 MiniLM embedding 索引耗时 329.208 秒（CPU）。首次 N2 准备耗时在 preparation.log 中约 CAISO 1.670 秒、Jena 1.780 秒；StockNet 首个成功 TF-IDF 准备 25.242 秒，MiniLM 升级成本单列。下载/早期解压等待不包含在这些计算耗时中。每次训练另写 wall_seconds、epoch/validation/inference 时间和 peak_cuda_bytes，不能混同端到端获取时间。
