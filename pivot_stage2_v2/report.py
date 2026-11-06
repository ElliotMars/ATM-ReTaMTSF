"""Generate Chinese delivery documents directly from completed real evidence."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import pandas as pd
from .common import ROOT,DATA,ART,HOUR,read_json,write_json,sha

def table(headers,rows):
    return '| '+' | '.join(map(str,headers))+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def main():
    docs=ROOT/'docs/pivot_stage2_v2';docs.mkdir(parents=True,exist_ok=True)
    decision=read_json(ART/'timezone_decision.json');diag=read_json(ART/'timezone_diagnostics.json');audit=read_json(ART/'data_acceptance.json')
    gf=read_json(DATA/'gfs/manifest.json');acq=read_json(ART/'gfs_acquisition.json');env=read_json(ART/'environment.json')
    native=read_json(DATA/'native_time_mapping/manifest.json');examples=read_json(DATA/'native_time_mapping/real_mapping_examples.json')
    smoke=read_json(ART/'smoke_summary.json');suite=next(ET.parse(ART/'logs/validation_tests.xml').getroot().iter('testsuite'))
    tests=int(suite.attrib['tests']);assert int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
    rows=[]
    for split,r in audit['timezone_hypotheses']['fixed_cet']['splits'].items():
        for h,x in r['horizons'].items():
            c=x['gfs_coverage']['12']
            rows.append([split,h,x['samples'],f"{c['origins_with_any_ratio']:.4%}",f"{c['target_hour_coverage']:.4%}",c['origins_without_GFS'],c['legal_fact_count']])
    sensitivity=[]
    for split,r in audit['timezone_hypotheses']['fixed_cet']['splits'].items():
        for h,x in r['horizons'].items():
            sensitivity.append([split,h,*[f"{x['gfs_coverage'][str(l)]['target_hour_coverage']:.4%}" for l in (3,6,12)]])
    years={}
    for b in acq['batches']:
        r=read_json(b['path']);ts=pd.to_datetime(r['hourly']['time'],unit='s',utc=True)
        for y in sorted(set(ts.year)):
            selected=ts.year==y
            years.setdefault(int(y),{'hours':0,'valid_by_k':np.zeros(7,np.int64)})
            years[int(y)]['hours']+=int(selected.sum())
            for k in range(1,8):years[int(y)]['valid_by_k'][k-1]+=sum(r['hourly'][f'temperature_2m_previous_day{k}'][i] is not None for i in np.flatnonzero(selected))
    write_json(ART/'GFS_yearly_coverage.json',{str(y):{'hours':v['hours'],'nonnull_by_day1_to_day7':v['valid_by_k'].tolist()} for y,v in years.items()})
    year_table=table(['UTC年份','原始有效小时','day1…day7 实际非空值数'],[[y,v['hours'],v['valid_by_k'].tolist()] for y,v in years.items()])
    dst_table=table(['年','DST名义日期','原始10分钟点','重复/倒退/异常间隔','Berlin未解决点'],[[r['year'],r['transition_day'],r['raw_points'],f"{r['duplicates']}/{r['backward_edges']}/{r['non_10minute_edges']}",r['berlin_unresolved_local_times']] for r in diag['dst_windows']])
    mapping_table=table(['原始naive','固定CET→UTC','Berlin→UTC（不明为None）','UTC假设'],[[e['original_naive_timestamp'],e['fixed_CET_UTC'],e['Europe_Berlin_UTC_or_NaT'],e['UTC_hypothesis']] for e in examples if e['original_naive_timestamp'].startswith(('2024-','2025-'))])
    source_checks=table(['官方原始文件','重叠时刻','比较数值','完全相同','最大差异'],[[r['archive'],r['overlap_timestamps'],r['finite_cells'],r['exact_equal_cells'],r['max_absolute_difference']] for r in decision['numeric_station_trace']])
    (docs/'jena_timezone_decision.md').write_text('''# Jena 原始时区的最终选择

**选择 H1：全年固定 CET/MEZ，即 UTC+01:00，不使用 DST。** 这是有直接站点文档和数值溯源支持的高可信解释；不能将所有历史年份的时钟配置、网络接收时间都声称为独立核实。GFS 可用时间仍为假设，因此整个实验准入为 CONDITIONAL。

## 三个候选及证据

| 假设 | 支持 | 反对或未解决 | 决定 |
|---|---|---|---|
| H1 固定 CET/MEZ | 本站官方说明明确使用 MEZ；本站 CSV 与 Fidel 数值指纹一致；DST 附近保持原有10分钟序列 | 未取得每一年独立 logger 配置和历史接收日志 | 默认采用 |
| H2 Europe/Berlin | 德国日常民用时间可能使用夏令时 | 本站标记 MEZ；春季02时段有记录、秋季未显示对应双份小时；强行 localize 会产生不存在/不明确时刻 | 只作敏感性，不静默推断 |
| H3 UTC | 连续数据同样可由 UTC 产生，且无 DST 歧义 | 与本站文档的 MEZ 定义不符，未发现源数据已转 UTC 的证据 | 只作敏感性 |

没有比较任何验证/测试 MSE 来选择时区。

## 来源追溯

Fidel README 给出的 timeanddate 链接不是可信的数值站点溯源，不能据此定时区。已实际下载 [Beutenberg 屋顶气象站说明](https://weather.bgc-jena.mpg.de/Wetterstation.pdf)：2025-02-04 版本第6页定义 Date Time 为 MEZ，且是平均区间的结束时间；前页说明每10秒采集、每10分钟形成统计并轮询下载。此处引用的是实际匹配的屋顶站说明，并非另一座 Jena 站的惯例。

下载 [2019上半年屋顶原始 CSV](https://weather.bgc-jena.mpg.de/data/mpi_roof_2019a.zip)和 [2024年屋顶原始 CSV](https://weather.bgc-jena.mpg.de/data/mpi_roof_2024.zip)，按同一 naive 时间比对气压、温度、湿度和 CO₂：

'''+source_checks+'''
共 315,072 个数值完全相同，建立 Fidel 数值记录到 MPI Beutenberg 屋顶源文件的实际联系。其他 Heimann 历史资料仅作背景，未将其他站点的 MEZ 当成本数据的证据。官方站点及下载页、德/英文站点 PDF、原始 CSV、Fidel 页面均已保存原文与 SHA256，见 source_evidence.json 和 timezone_decision.json。

## 原始时间诊断与预处理核查

'''+dst_table+f'''
2019—2025 已有数据覆盖的 13 个 DST 窗口均有 432 个原始点，三日内没有重复、倒退或非10分钟边；2025年10月超出原始数据终止日，明确记为没有来源数据。仅凭这项连续性不能证明 H1，决定还依赖上面的站点说明与指纹。

全期原始数据 {native['original_rows']:,} 行，并非处处规整：共有 {native['raw_duplicate_timestamp_extras']} 个重复额外时刻、{native['raw_clock_backwards']} 次倒退、15 个非10分钟相邻边，均不在预期 DST 切换日。例如 2022-06-12 倒退30分钟，2023-10-21 回退至前一天；不将这些异常臆测成 DST。完整事件见 native_time_mapping/anomalous_original_edges.json。

Fidel raw_data 三文件拼接与 time_series/weather_large.parquet 的全部 {diag['raw_vs_processed']['raw_total']:,} 个时间标签逐序相同；21通道数值逐格相同，未发现这两层之间删点、重新生成标签或缺失转有效值。不能据此证明更上游 raw_data 从未处理。N2 已执行重复值冲突处理、固定 :10 小时相位抽样和连续段保留，可能隐藏原始异常；本轮直接诊断原始高频文件，未只看 N2。

H2 对全期原始标签产生 {diag['raw_vs_processed']['raw_berlin_unresolved_all_years']} 个 ambiguous/nonexistent 时刻，2019—2025已有DST窗口内为78个。H1/H3 没有时区解释歧义，但原有重复/倒退仍是源问题，不能靠时区平移修复。

## 转换及预测时刻

`UTC = original_naive_timestamp - 1 hour`，使用 Python 固定 offset timezone(+01:00)，不是 Europe/Berlin。原始全期三个候选映射已保存在独立 native_mapping.parquet；训练使用的 v2 N2 UTC 时间严格单调唯一、保留原生断段与数值 Mask。Berlin 的不明时刻显式为 NaT，涉及这些时刻或非1小时间隔的窗口排除；不自动猜测折叠方向。

`history_end_time` 是最后一个10分钟统计区间的结束时间。默认 `forecast_origin = history_end_time + 10min`，10分钟是基于轮询周期的明确接收延迟假设，不是经验证的历史送达时间；必须早于 `first_target_time = history_end_time + 1h`。修复仅在 v2 Dataset，v1 的 weather origin_position=stop 逻辑没有改写，也没有沿用于外部信息截点。

实测提供 {len(examples)} 个跨季节/切换日映射例，另有56项公式映射测试。部分例子：

'''+mapping_table+'''
可信度：H1 有直接来源支持且明显强于另外两者；每个历史年份的独立时钟配置仍未完全验证。保留 H2/H3 的共同数值窗口敏感性配置，且不把推断置信度与历史发布时间验证混为一谈。
''',encoding='utf-8')
    (docs/'data_and_asof_report.md').write_text(f'''# 真实 Jena/GFS 数据与可用性报告

GFS 来自 [Open-Meteo Previous Model Runs](https://open-meteo.com/en/docs/previous-runs-api)，请求显式 `models=gfs_global`、`timezone=UTC`、`timeformat=unixtime`，仅获取官方长期历史覆盖支持的 temperature_2m_previous_day1…7；没有引入 Best Match、ERA5、Historical Weather API、未来 Jena 真值或额外湿度/风压变量。

实际完成 {len(acq['batches'])} 个按月/边界日请求，UTC有效时刻 2021-03-30 至 2025-07-01。末尾一天仅提供 UTC 时区敏感性的相位插值边界，不增加 Jena 标签或延长 Test。GFS 响应共 {sum(b['bytes'] for b in acq['batches']):,} bytes（{sum(b['bytes'] for b in acq['batches'])/2**20:.3f} MiB）；包括站点文档/CSV/PDF等已缓存并记账的 v2 来源下载 {env['source_download_bytes']:,} bytes（{env['source_download_bytes']/2**20:.3f} MiB）。该记账不含浏览工具与初次连通性探测的传输开销。没有新下载预训练权重、全球 GRIB 或付费数据。

原始小时有效时刻 {gf['raw_hourly_points']:,} 个；各 k 实际非空数 {gf['raw_nonnull_values_by_k']}。对齐后真实非空预报事实 {gf['forecast_fact_count']:,} 个，各 k 为 {gf['aligned_nonnull_values_by_k']}。模板正文去重后 {gf['unique_text_templates']:,} 条，冻结 MiniLM 缓存维度384；日期、坐标、提前量、来源哈希通过列式事实索引保留，每个事实有独立文档 ID，不把模板数当成样本事实数。

{year_table}
## 站点、网格和插值

请求坐标来自 MPI Beutenberg 官方页面：50.91N、11.56E、220m。API返回网格示例：`{acq['batches'][0]['response_grid']}`。每个响应都保留实际网格/高程信息和完整请求参数，不能把请求点称为精确原始GFS格点。该产品包含 Open-Meteo 的小时插值及高程处理；站点屋顶传感器与模型2米温度、站点10分钟统计与预报瞬时值存在观测口径差异。

N2 保持每小时 :10 的原相位。GFS 整点预报用前后两点线性插值到 :10（权重5/6、1/6），再共同按0.1°C精度舍入，**原数值控制和文本使用完全相同的派生预报值**。不是插值 Jena 未来真值。某端点缺失时整个外部点保持 Mask，不反向填充。此预报有效时刻对应目标区间结束，作为预测特征，未宣称等于10分钟平均真值。

## 信息权限

固定提前量的 `reference(v,k)=v-k*24h` 只是名义时间参考，不是已验证的某次GFS初始化时间。对插值两端点，采用更保守的

`assumed_available_at = max(reference(left,k), reference(right,k)) + latency`

默认 latency=12h，敏感性3/6h。只有 available <= 最后历史观测可用时刻 的事实才进入候选；先过滤，后访问候选向量和评分。某个目标日/提前量没有真实归档值时保持缺失，不按日期自动推定存在。verified_run_time/publication_time 均为 null，所有方法都不能声称使用了完整独立 Single Run。API归档在本轮响应哈希上冻结，不能证明逐条历史发布和归档修订版本。

## 全部真实窗口覆盖

比例由所有实际起点穷尽统计，含没有外部信息的样本；H=336 为可选。起点覆盖与目标小时覆盖是不同指标，末端不可用小时照常预测，外部 Mask 为假。

'''+table(['Split','H','数值窗口','起点有合法GFS','目标小时有合法GFS','零GFS起点','合法预报事实使用次数'],rows)+'''
最后一列是滑动窗口内的重复使用次数，不是独立下载记录数。默认 H=168 最多155个目标小时具有合法预报，晚端13小时屏蔽；H=336后半段也不伪造长提前量。

延迟敏感性下，目标小时覆盖：

'''+table(['Split','H','3h','6h','12h'],sensitivity)+f'''
数据值、目标 Mask 和起点不会因文本稀疏而删减。已实际选择一个真实无GFS验证样本进行推理，见 real_no_GFS_inference.json。

## 产物和成本

`dataset/PIVOT/stage2_v2/jena_{{fixed_cet,berlin,utc}}/` 保存新的分割、UTC数组、原始数值、Mask、scaler与全H starts；`gfs/forecasts.npz` 保存预报值及两端点来源索引，text_templates.jsonl/embeddings.npy/query_templates.json/queries.npy 保存完整语义缓存。36条跨季节真实文档例在 artifacts/pivot_stage2/v2/gfs_document_examples.jsonl，包含完整英文文本、有效时刻、温度/单位、模型/坐标、原始响应哈希、源API、名义参考和假设可用时间。

最终 GFS 准备耗时 {gf['preparation_seconds']:.3f} 秒（CPU，含解析、模板去重、冻结编码与查询缓存）；环境为 Torch {env['versions']['torch']}，只实际验证CPU。旧边界版本和旧smoke保留，未污染 v1；最终 source/data/code 绑定由 delivery_verification.json 核查。
''',encoding='utf-8')
    (docs/'model_and_fairness.md').write_text('''# v2 模型、文本与公平性

v2 独立模块复用 v1 有效的 PatchEncoder、CTSR、Fusion 和 Mask 指标，未改写 v1。完整训练器位于 pivot_stage2_v2/train.py，Dataset 和外部数值控制位于 data.py/models.py。

```mermaid
flowchart LR
 X[历史360小时×21通道与Mask] --> N[Patch Transformer与可选CTSR]
 X --> Q[历史温度/湿度/气压与通道描述]
 G[真实GFS固定提前量事实] --> A[按原点过滤两端点可用性]
 A --> R[合法候选全集]
 R --> V[原数值控制 H×7及Mask]
 R --> T[Mean或Latest或语义/时间TopK]
 Q --> T
 T --> E[冻结MiniLM语义向量与显式时间特征]
 E --> F[Cross Attention与门控残差]
 N --> F
 F --> P[H×21预测]
 V --> P
```

原数值控制和文本增强是不同模型分支，不把它们同时提供给 Full PIVOT。Dataset 统一输出 x_num/x_mask、y/y_mask、UTC origin/history_end/first_target、文本向量/Mask/来源，以及 gfs_num/gfs_mask。模型特征显式白名单不含 y。

数值控制从合法预报全集构造 [H,7] 的温度矩阵和 Mask，按 Jena 训练温度 scaler 归一化，经线性投影加到同一数值骨干预测。它使用的是共同的 API 派生预报值，不声称是未经处理的全球 GRIB。没有合法预报时投影输入全零，严格退化为数值骨干。

模板英文正文只陈述源温度和名义提前量；实际 UTC 有效时刻加在完整文档及结构化 metadata 中，不生成源数值不支持的天气结论。冻结 MiniLM 对去重后的天气事实正文编码，绝对日期单独存储和过滤。相同事实正文在不同时刻共享向量，所有文档仍有不同的时间/ID/来源。融合另学习三个显式时间特征：相对 origin 的 valid hours、可用时间的新近性、名义提前量（均/168），避免依赖语言模型理解日期字符串。这一投影参与训练，MiniLM保持冻结。

检索查询不使用股票收益模板。21个原始通道均按源别名解析到对应气象描述，结合历史窗口有效温度均值（<5/5—20/>20°C）、湿度（<60/≥60%）与气压（<1000/≥1000mbar）构成12种解释性天气查询；只使用历史原始单位，不读取未来目标。全球天气候选可被全部相关通道访问，由语义排名区分，保留21个输出通道。

时间项为0.2×exp(-age_hours/24)，默认Top-K=4。latest按可用时间，random用seed/origin/channel确定的合法池局部随机数，mean池化所有合法事实。CTSR只用历史中心化相关度/频域幅度，排除自身、缺失通道，稳定处理并列。文本投影、Cross-Attention和门控残差接入同一数值表示；全无文本的行严格为数值表示，填充区和时间特征均受Mask控制。

所有方法共享同一H的数值起点、输入、真实标签、Mask和外部信息权限。Raw 和文本可访问的候选全集完全相同；检索只从此全集取K，mean使用全集，文本组没有更多或更晚的源预报。检索减少事实数量是算法的既定差异，未为文本组单独换更准确的Run或提供未来观测。

MiniLM的发布时间晚于历史任务，属于所有文本方法共享的现代离线先验，不是2014/2021年真实部署还原；未知预训练语料重合风险需披露。参数数目报告注册参数，关闭组件可仍保留权重，不代表活跃计算量完全匹配。模板语义、显式时序特征与强外部预报源的贡献必须分别由原数值控制、文本基线和消融判断，目前没有正式性能证据。
''',encoding='utf-8')
    bounds=[]
    nm=read_json(DATA/'jena_fixed_cet/manifest.json')
    for s,r in nm['splits'].items():bounds.append([s,r['points'],r['naive_start'],r['naive_end'],r['utc_first_resolved'],r['utc_last_resolved']])
    (docs/'experiment_protocol.md').write_text('''# 冻结实验方案

默认 Jena 21 通道，L=360 小时，核心 H={24,72,168}，H=336仅通过 --include-h336 可选加入。按站点固定CET日历，Train=2021-04-01至2023-12-31，Val=2024全年，Test=2025-01-01至2025-06-30。分割边界先按选定物理时区解释再转UTC，例如Val从2023-12-31T23:00Z开始，不将本地日期字符串直接当UTC。

'''+table(['Split','点数','原标签首','原标签尾','实际UTC首','实际UTC尾'],bounds)+'''
每个Split独立提供360小时预热，不借用前一Split历史。原N2数值和Mask逐格保留；新的回归scaler仅拟合本轮Train有效观测。窗口必须完整覆盖L+H、1小时步长、同一原生连续段，且有有效历史和标签；缺失通道由Mask处理，不减少维度。标签不填补。

主表7种：Numerical Only、GFS Raw Numerical、Text Mean Fusion、Latest Eligible、Semantic、Semantic+Temporal、Full PIVOT。Full使用CTSR；简单文本基线关闭CTSR以明确区分通道模块；Numerical Only与Raw控制保留数值骨干CTSR，without_ctsr另隔离该因素。

12个消融配置：without_ctsr、without_str（全池均值）、without_temporal、random_eligible、text_missing、topk1、topk8、timezone_cet_common、timezone_berlin、timezone_utc、lag3、lag6。默认latency=12h作为对照。时区敏感性三组只使用三种解释都有效的同一原始行窗口交集，保持数值输入/目标完全相同；与全量主表分别汇总，不能混合不同样本指纹。Berlin不明时刻及跨DST步长变化窗口排除并报告，不silent infer。时区敏感性只用于稳健性报告，不以测试MSE重选主时区；without_temporal只关闭检索时间分数，仍保留合法目标时刻/提前量字段。

固定正式种子2024、2025、2026；默认主表63次，消融108次。76个YAML覆盖4个H×19个方法/消融，默认脚本只运行3个核心H。预算AdamW、lr=0.001、weight_decay=0.0001、batch=8、最多30epochs、patience=5、dim=32、1层4头、patch=16/stride=8、dropout=0.1。没有本轮超参搜索。

验证标准化MSE选择best checkpoint；模型选择结束后才完整评估Test。报告标准化和原单位MSE/MAE、每通道误差、有效标签覆盖、参数量、训练/推理时长及峰值显存；21个通道量纲不同，标准化指标是主比较。保存预测、真值、Mask、起点误差、配置、模型/优化器/AMP/RNG恢复状态、源与代码哈希。

汇总只读正式experiments目录，排除smoke/retired_smoke，拒绝重复dataset/H/scope/method/seed和不同样本指纹。报告种子均值与样本标准差，使用块长至少H的预测起点顺序块Bootstrap；时区敏感性按原始naive历史行配对，UTC外部检索时刻另存。时间块区间是有限样本近似，跨被排除窗口的物理跨度可能更长，应结合原点序列检查；不将随机种子方差与时间相关性视为同一来源。

本轮仅运行单batch/单step验收，没有启动上述正式预算、正式多种子或消融训练。整体准入仍为CONDITIONAL：时区有强来源支持，但GFS固定提前量参考+延迟和观测end+10min是明确假设，未验证历史实际送达。所有科研效果待用户手动正式实验后判断。
''',encoding='utf-8')
    seasons=table(['Split/季节','history_end UTC','forecast_origin UTC','first_target UTC','合法事实','合法目标小时(H168)'],[[r['split'],r['actual_history_end_utc'],r['forecast_origin_utc'],r['first_target_utc'],r['candidate_facts'],r['legal_target_hours']] for r in audit['seasonal_real_samples']])
    preservation=read_json(ART/'v1_preservation.json')
    (docs/'test_and_smoke_results.md').write_text(f'''# 实际执行的验收

最终pytest **{tests} passed，0 failure/error**，JUnit时间 {float(suite.attrib['time']):.2f}s。既有225项和新增99项均通过。证据：logs/validation_tests.log、validation_tests.xml、final_acceptance.log。

56个参数化转换例覆盖2019—2025冬/夏/三月/十月边界，额外测试Berlin不明时刻；真实来源映射另有{len(examples)}例。覆盖全部H（含可选336）、C=21、batch=1、全数值缺失通道、无文本、缺GFS、未来站点y变化不影响查询或外部值、3/6/12h权限、名义day7晚端Mask、候选全集相同、Raw缺预报恒等、模型forward/backward、单step和checkpoint恢复。特别构造位于真实origin与first_target之间才可用的预报，验证它不会因错误截点被放行。

## 真实数据

全部实际窗口、数值与Mask对应、训练scaler、原生连续段、UTC映射、跨Split标签不重叠、原始响应与产物哈希已穷尽检查。原始全期映射{native['original_rows']:,}行；默认N2全期选定行UTC单调唯一。原始异常保留为诊断，不伪造修复。

{seasons}
这三个真实起点分别承担训练、验证和测试的单样本验收。真实缺GFS验证样本也保留并完成有限值推理，见real_no_GFS_inference.json。三季节文档例见gfs_document_examples.jsonl，来自真实API预测而非未来Jena真值。

当前主表pilot **{len(smoke['runs'])}个配置**（7方法×3H），仅seed2024，每个恰好1 optimizer step、1个训练样本、1个验证样本、1个测试样本。检查点回读预测一致、强制无文本推理全部通过。seasonal_model_probes.json另记录同一已训练checkpoint在冬季、夏季、DST附近三个真实起点的实际前向结果与物理时刻。日志real_pilot_accepted.log，summary与预测在smoke/；旧修复期间版本保留retired_smoke/。smoke误差不构成正式性能表。

六个Shell入口均通过bash -n与--dry-run：main=63、ablations=108、pilot=21；没有对main/ablations使用--execute。prepare_all --execute完整缓存重放已通过；collect --execute空正式结果已验证排除smoke。正式结果数0，无多种子正式训练、无消融训练。

v1交付的{preservation['v1_source_config_test_doc_files_verified']}个源码/配置/测试/文档文件哈希逐一保持一致，StockNet、CAISO、Jena三个v1数据Bundle哈希通过。N2原值/Mask完全一致，既有历史结果未修改。没有由本代理执行git commit/push/merge。

当前Python {env['versions']['python']}、Torch {env['versions']['torch']}，只实际验证CPU；CUDA/AMP已实现但未做GPU验收。不需要用户安装依赖或重做数据准备。环境明细environment.json、environment_freeze.txt；最终绑定delivery_verification.json与delivery_manifest.json。
''',encoding='utf-8')
    (docs/'manual_run_guide.md').write_text('''# 用户最终手动命令

当前工作区的数据、UTC映射、真实GFS缓存、模板、MiniLM、配置、依赖与单步验收已完成。请先接受 [时区解释](jena_timezone_decision.md)和 [可用性条件](data_and_asof_report.md)。用户不需要再下载、预处理、填写reviewed.json或安装依赖。

在仓库根目录执行：

```bash
cd /opt/data/private/ATM-ReTaMTSF
bash scripts/experiments_v2/validate_ready.sh --execute
bash scripts/experiments_v2/run_pilot.sh --execute --device cpu
bash scripts/experiments_v2/run_main.sh --execute --device cpu
bash scripts/experiments_v2/run_ablations.sh --execute --device cpu
bash scripts/experiments_v2/collect_results.sh --execute
```

Pilot已由本轮实际执行。再次执行时，代码、配置及数值/GFS数据绑定一致的成功验收会跳过；不是重复正式训练。主表默认63次、消融108次，正式种子2024/2025/2026。主表和消融须由用户自行启动，本轮没有启动。

脚本缺省为dry-run；预览不会联网或训练：

```bash
bash scripts/experiments_v2/run_main.sh --dry-run
bash scripts/experiments_v2/run_ablations.sh --dry-run
```

H=336已准备数据、覆盖审计及模型形状测试，未进入本轮核心真实pilot；如决定开展可选扩展，可在对应pilot/main/ablations命令追加 --include-h336。其12h策略通常只能覆盖前155个目标小时，不能描述成336小时完整外部预报覆盖。

默认使用已准备好的.venv-pivot CPU环境。CUDA分支未在当前机器验证；未来若自行提供兼容CUDA解释器，可通过PIVOT_PYTHON指定并选--device cuda，但不把这个切换作为开始实验的前置任务。

正式结果路径：artifacts/pivot_stage2/v2/experiments/H<h>/<variant>/seed<seed>/，含config.yaml、binding.json、best.pt、latest.pt、history.csv/json、retrieval_trace.json、summary.json、test下prediction/truth/mask.npy和origin_errors.npz。每次汇总创建独立collections/<version>/results.csv/json，latest_collection.json指向最新；无正式结果时状态为no_formal_results_yet。

中断后重跑原命令：成功且绑定一致的任务跳过；失败目录保留并从latest checkpoint恢复。若成功结果对应不同代码/数据/配置，拒绝覆盖，应保留原结果并另开实验版本，不择优重跑seed。准入清单绑定数值与GFS数据，发生源变更须重新验证，避免沿用旧通过状态。

prepare_all.sh --execute是完整自动复现入口，本轮已验证，无需再次执行。所有v2数据和结果在独立stage2_v2/v2目录，v1保持不变。更多模型与公平性细节见model_and_fairness.md，完整样本数/覆盖率见data_and_asof_report.md。
''',encoding='utf-8')
    print('Generated 6 Chinese v2 documents from real evidence',flush=True)
if __name__=='__main__':main()
