# 旧 Time-MMD 时间有效性审计

审计对象为本地 origin `https://github.com/ElliotMars/ATM-ReTaMTSF.git`，起始分支 main。
初始 `git status --short` 仅有用户提供的未跟踪 prompt.md；已保留，并新建 feature/citra-prototype。
没有修改原模型、加载器、预处理、训练入口、权重或数据。

## 本地代码证据

|数据集|目标区间文本访问证据（data_loader.py）|日期到 embedding 的本地来源|结论|
|---|---|---|---|
|Energy|__getitem__ 99；h_begin=lbw_end 102；未来区间 110；retrieval 152–168|Energy/TextPackage.py 将 start_date 重命名为 Date；NewsDatabasePre_embed.py 24、43 保存 News-Date.npy|不能证明 cutoff 可用|
|Health_US|__getitem__ 279；未来区间 290；retrieval 332–348|Health_US 中同类 TextPackage、NewsDatabasePre_embed|相同风险，未获可用时间证据|
|Environment|__getitem__ 458；未来区间 469；retrieval 511–527|Environment 中同类 TextPackage、NewsDatabasePre_embed|相同风险，未获可用时间证据|
|Climate|__getitem__ 636；未来区间 647；retrieval 689–705|Climate 中同类 TextPackage、NewsDatabasePre_embed|相同风险，未获可用时间证据|

四个类的 aligned 模式也按未来目标日期加载对应文件；random 模式从全目录抽样，没有 cutoff
可见性过滤。retrieval 模式将日期限制在目标预测区间，未检查实际发布时间、可用时间或版本。
训练/测试流程在 exp/exp_ReTaMTSF_*.py 中直接把这些文本传给原模型；
models/ReTaMTSF.py 和 layers/ReTaMTSFLayers.py 中的文本注意力没有补充时间可见性验证。
data_factory.py 保留三种模式，run_ReTaMTSF_*.py 默认为 retrieval。

## 字段真实含义、来源和未知项

|字段|已从本地代码验证的来源/用途|尚未验证的含义或问题|
|---|---|---|
|数值 start_date/end_date|NumericalPackage.py 保留/重排原 CSV 区间字段，按 start_date 排序；加载器使用第一列作为 time_span|区间起止来自原 CSV，不是文本发布时间；底层观测发布延迟未知|
|文本 start_date/end_date|*_report.csv、*_search.csv；TextPackage 将区间拼入正文|看起来是内容描述/对齐区间，缺少原始记录与采集说明，不能直接断言为 event_time|
|fact|TextPackage 拼入正文|可能描述同期或事后事实；真实生成和首次发布时刻未知|
|preds|原字段被 drop；注释中的 fact+preds 路径未启用|不能据此推断当前 embedding 包含预测文本|
|Date|TextPackage 将 start_date 重命名，并将同一天 report/search 的正文合并|是派生的对齐键，不能当作 issued_at 或 available_at；合并后丢失单文档时间与版本边界|
|News-YYYY-MM-DD.npy|NewsDatabasePre_embed 将 Date 格式化为日期后保存归一化 embedding|不是发布时间证据；文件 mtime 也不能证明历史首次可用|
|查询文本 embedding|Text_preembedding.py、querytext_preembedding.py 由数值目标 start/end_date 产生 `From ... to ...`|包含目标区间日期但未读取目标数值；预知规则日历不自动意味着标签泄漏；不规则区间是否可预知需另审计|
|通道描述 embedding|ChannelDescription_Pack/QueryText_Pack 固定描述；ForecastingPoint=0 编码|人工描述来源、修改时间与历史可用性未记录|
|issued_at|原路径未保留独立字段|未知，不能以 start_date 代替|
|available_at|原路径未保留独立字段|未知；严格回测必须排除未经证明的文本|
|ingested_at|原路径没有历史系统摄入日志|未知；不得用本次导入时刻伪造过去摄入时间|
|valid_from/valid_to、version_id|没有历史文档版本管理|同日合并和保存单文件无法恢复当时版本|

common/newsdatabase_pre_embed_multi.py 沿用 Date 命名；其中 openai_pca 路径在不存在 PCA 文件时
对输入全文 embedding 拟合 PCA，没有时间划分。若该可选路径使用跨测试期全文，另有表示拟合泄漏风险，
但当前没有证据表明历史实验启用了该路径。当前不改动它，也未调用任何外部 API。

## 数据与结论边界

本地 dataset/Time-MMD 为空，没有原 CSV、parquet、embedding 或历史可用时间日志。
审计只确认代码的数据流，不能核实每条记录的发布时间，也不能复算或裁定历史实验结果。
预测未来事件的文档完全可能在 cutoff 前发布，所以按目标区间选文件本身不足以证明每条文本泄漏。
但旧路径不具备严格 point-in-time 回测资格。保留旧实验行为并明确其风险；CITRA 默认拒绝未知可用性。

运行标准库检查：`python scripts/audit_citra_data.py --output docs/citra_local_data_audit.json`。
检查器输出本地列名、记录数、时间字段缺失和文件存在性；发现时间列也不自动视为 verified。
真实数据下一步必须保留原始文本、来源 URI、首次公开时间证据、实际/未知摄入时间、版本及哈希。
NWS AFD/ASOS/MOS 的导入与来源时间语义核验属于后续里程碑，目前没有下载或真实数据结果。
