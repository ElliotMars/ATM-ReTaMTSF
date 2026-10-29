# 阶段 0：ReTaMTSF 仓库审计与 Fidel-TS 适配设计

审计日期：2026-10-29。任务说明实际文件名为 `Prmopt.md`；已完整阅读。本次只执行阶段 0。原模型、Time-MMD 加载器、实验脚本、结果、Checkpoint 均未修改，没有启动训练、全量预处理、完整数据下载或文本编码。

**结论：CAISO、Jena 都有适合单 Subject、多 Channel 的数值结构，但目前不能放行严格实时多模态实验。** 天气样本缺少独立发行/可用时间，不能根据时间键或 `forecast` 文件名证明预测起点可用。原网络还存在影响训练有效性的条件性问题。下一阶段只能在用户明确批准后先实现下载与检查工具；在时间可用性和模型问题解决前，不进入正式多模态训练。

## 1. 审计范围、版本与证据边界

| 对象 | 本次固定版本 | 实际核查方式 |
| --- | --- | --- |
| 本工作区 | `a650d4367c29b52b01a1fe87085d590332609a08`，分支 `main` | 实际源码、AST、极小标准库样例 |
| California_ISO | `75f74f556edd5fbdd080c4db6d4b7547fc331f39` | HF API 文件清单；README、JSON 元信息、官方预处理源码；两个 Parquet 尾部各 64 KiB；五个天气 JSON 前缀各 64 KiB |
| Jena_Atmospheric_Physics | `c8da24175c7aabe5c3b02b503e4c5afe44a605fd` | 同上；一个 Parquet 尾部；四个天气 JSON 前缀 |
| Fidel-TS 官方代码 | `81043864342e550ed9ffdaa7d9ae6d8420460c6e`，本次默认分支为 `Reasoning_baselines` | 官方仓库页面、固定 SHA 的数据代码/YAML/训练脚本 |
| 论文 | `arXiv:2509.24789v5`，2026-09-18 修订 | HTML 正文、附录 C.4 |

来源：[CAISO 固定版本文件树](https://huggingface.co/datasets/fidel-ts/California_ISO/tree/75f74f556edd5fbdd080c4db6d4b7547fc331f39)、[Jena 固定版本文件树](https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics/tree/c8da24175c7aabe5c3b02b503e4c5afe44a605fd)、[官方代码固定版本](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/tree/81043864342e550ed9ffdaa7d9ae6d8420460c6e)、[论文 v5](https://arxiv.org/html/2509.24789v5)。HF API 返回的最后修改时间分别为 2026-05-03 16:10:43 UTC 和 16:24:49 UTC。

机器可读证据保存在 [`00_audit_evidence.json`](00_audit_evidence.json)：包含完整文件名清单、固定 Revision、来源链接、所读源码 SHA-256、Parquet 页脚摘要、局部 JSON 样本统计及本地源码哈希。天气原文未整份复制进仓库。

边界说明：

- 两个 HF 仓库分别列出 147、32 个文件；这只是文件清单，没有下载整库。没有访问其余四个子数据集的数据。
- 12 次数据 Range 读取总计 786,432 字节，其中 Parquet 196,608 字节，JSON 589,824 字节；另仅读取小型文档、源码和 API 元数据。均验证 HTTP 206；没有加载任何 Pickle。
- 当前 Python 缺少 `torch/numpy/pandas/pyarrow/huggingface_hub`；旧脚本指定的 `/opt/data/private/envs/ReTaMTSF/bin/python`、`/opt/conda/bin/python` 也不存在。本次没有安装环境。
- Parquet 使用临时标准库 Compact Protocol 解码器读取页脚，核对 `PAR1`、footer 长度、完整字段消费、schema 与行数。**这是元数据审计，不是全量数据检查**；阶段 1 应用 PyArrow 独立复核。
- 重复时间戳、准确缺口位置、哨兵比例、全量 NaN/Inf、天气全时段覆盖率均未确认。下文明确区分事实、源码行为和待验证假设。
- GitHub API 曾返回 403 限流，随后通过官方仓库页面和固定 SHA 的 raw 文件完成读取。本地默认沙箱初始化失败，必要的只读命令通过权限审核后执行。
- 审计开始时已有未跟踪文件 `Prmopt.md`，保持原样。`dataset/Time-MMD` 是 gitlink，指向 `00281e2d86058286d5548b15a7670e8eda57ef62`，当前目录未提供实际数据。本阶段未切换分支；后续新增适配器宜使用独立分支。

## 2. 原仓库完整数据流

### 2.1 入口与训练评测

四套入口为 `run_ReTaMTSF_{Energy,Health_US,Environment,Climate}.py`，分别调用同名 `exp/exp_ReTaMTSF_*.py`，经 `data_provider/data_factory.py` 创建四个 `Dataset_ReTaMTSF_*` 和对应 collate。四套实验文件主体相同，主要差异为 provider 与可视化/调试输出频率。

流程如下：

```mermaid
flowchart LR
    A[Time-MMD CSV] --> B[数值与文本预处理]
    B --> C[Parquet 数值与 NPY 文本缓存]
    C --> D[Dataset 与 Collate]
    D --> E[CTSR 历史参考通道 Top-K]
    E --> F[Content Synthesis]
    D --> G[STR 文本 Top-K]
    G --> H[Text Cross-Attention]
    F --> I[Cross-modal Fusion 与输出头]
    H --> I
    I --> J[MSE / Adam / 验证集早停]
```

训练使用 MSE、Adam、weight decay 和验证集 EarlyStopping。每个 epoch 也计算并打印 test loss（`exp_ReTaMTSF_Environment.py:175`），但早停实际只用 validation loss，**不能据此断言代码自动使用测试集选模型**；持续查看测试结果仍有人工调参风险。新入口应直到验证选择结束才访问正式测试集。

创建了 OneCycleLR，但 `scheduler.step()` 被注释；当前默认 `lradj=type3` 实际走 `utils/tools.py:48` 的按 epoch 指数衰减，不能将原训练描述成已执行 OneCycle 调度。

测试重新构建 Dataset，通过 `load_latest_checkpoint()` 按模糊 setting、目录修改时间选择权重，而非显式 run ID。setting 不包含 seed、文本模式和检索类型等关键信息，单独测试的 setting 还少了训练 setting 中的 batch size。新入口必须使用明确的 checkpoint 路径与配置哈希。

测试在标准化空间计算 MSE/MAE 等指标，保存 `*_pred.npy`、`*_trues.npy`、`*_x.npy` 和 `result.txt`；没有调用 inverse transform，也没有保存足够的时间戳、版本、Scaler 和数据来源信息。`MAPE/MSPE` 直接除以标签，零标签会产生非有限值，主指标宜沿用 MSE/MAE。

### 2.2 数值加载、切分、归一化与随机起点

四套 `data_provider/data_loader.py` 实现基本一致（Energy 40–97 行，其余对应 220、400、578 行起）：

1. 读一张以 `start_date/end_date` 开头的宽表。
2. 令 `n=int(N*num_data)`，计算 `max_start=N-stride*(n-1)-1`，然后随机抽起点。
3. **实际取行是 `[start_index+i for i in range(n)]`，没有乘 stride。** 因此抽取的是一段连续行，`stride` 只影响可选起点和报错条件，不实现下采样或滑窗步幅。连续行也不等于时间戳等间距，加载器未检查缺口/重复。
4. 当前所选子段按 70% train、约 10% val、20% test 顺序切分。val/test 在左边借用 `L` 个历史点，其标签不越过各自划分区间；这种借用历史本身不是泄漏。
5. target 为 `target_ids` 指定列；其余所有数值列作为参考数据库，排除 target 和两列日期。每个样本只读取历史部分的参考通道，没有把参考通道未来值直接传入模型。
6. `StandardScaler` 在当前 Dataset 自己选出的子段训练部分拟合；随后 transform target+references。

必须区分的随机性：

- 默认 `num_data=1, stride=1` 时 `max_start=0`，起点固定，**默认完整数据不会发生随机裁剪**。命令行 `num_data` 类型是 int，不能直接传 0.5。
- 若通过代码传分数或后续放开参数，train/val/test 分别实例化并各自调用 NumPy RNG，不共享裁剪起点。即使有全局 seed，划分还依赖调用顺序，可能重叠且 Scaler 拟合不同时间段。
- 标准库极小样例：N=1000、子段 n=500，train 起点 400、test 起点 0 都是合法起点；train 的训练段覆盖 400–749，test 标签覆盖 400–499，出现 100 个重叠标签。这是可触发风险，不是声称当前默认配置已发生。
- 新适配器应使用持久化的时间边界与窗口索引，禁止随机改变数据划分；随机性只用于训练 shuffle、初始化和合法池内的实验控制。

**Scaler 列顺序 Bug：** fit 使用 `df_raw.columns[2:]`，transform 却使用 `[targets, remaining_columns]`；target 不位于原先相同位置时均值/标准差错配。原预设目标常在首列，不能据此认为所有调用均受影响。极小算例中原列 a=[0,2]、b=[100,104]，把 b 移到第一列后旧方式将 b[0] 标成 99，正确结果是 -1。新适配器必须保存列名顺序，并按列名 transform/inverse transform。旧加载器不直接复用，也不在本阶段修改。

### 2.3 Batch 形状与模型约束

记 B 为 batch，L 为 lookback，H 为 horizon，C_t 为目标数，C_r 为历史参考候选数，K_t=`nref`，K_n=`nref_text`，D=384。

| 输入/中间量 | 实际形状 | 约束与行为 |
| --- | --- | --- |
| target history / label | `[B,C_t,L]` / `[B,C_t,H]` | Dataset 内先是时间在前，collate 转置 |
| TS database | `[B,C_r,L]` | 同时间历史通道，目标已排除 |
| query / description | `[B,C_t,H,384]` / `[B,C_t,1,384]` | 查询通常只编码预测区间日期，描述编码通道语义 |
| news database | `[B,N,1,384]` | 单候选一个向量；现实现要求 M=1 |
| CTSR 输出 | `[B,C_t*K_t,L]` | 按每个目标检索历史通道 |
| Content Synthesis | `[B,C_t*(K_t+1),L,384]` | 标量 Linear→位置编码→LayerNorm→时间/通道聚合 |
| STR 与 Text Cross-Attention | `[B,C_t*K_n,H,384]` | description→选中文本，再 query→结果，再 self decoder |
| Fusion | 借用历史最后 `label_len` 点，再拼接 H 个文本 token | query 长 `label_len+H`，memory 长 L；不要求 L=H |
| 最终预测 | **实际 `[B,1,H]`** | `DimensionReducer.forward()` 把 `C_T=1` 写死 |

虽然前半段接受多个目标，当前端到端模型仅应作为**单目标、多参考通道**使用。多目标标签可能与 `[B,1,H]` 广播计算损失而未报错，不能宣称已实现独立多目标预测。

`ContentSynthesis` 默认 d_model=384、4 heads；Text/Fusion 默认 8 heads、3 层 decoder；所有模态维度必须相容。CTSR 的 coherence 需要 `2 <= nperseg <= L`，且应保证有足够分段以避免单段评分退化。新配置还应验证 `0<label_len<=L`、K 的范围、张量有限值与同一时间网格。

CTSR 正常在 `[B,C_t,C_r]` 分数上 Top-K：coherence 为 Hann 窗、50% overlap、rFFT 后平均平方相干性；另已实现 `lagged_corr`、`partial_corr`、`granger`。Granger 是回归残差方差比评分，不是显著性检验。当前没有 random-channel 模式。若 K_t>=候选数，则直接使用全部参考并补零，不再按评分排序。阶段 6 的 random 对照需另加可复现的选择逻辑，不改已有评分公式。

### 2.4 STR 与四种原模式

原查询缓存由 `pre_process/Time-MMD/*/Text_preembedding.py` 等生成，模板为日期区间；默认模型 `paraphrase-MiniLM-L6-v2`，384 维，向量 L2 归一化。新闻来自 `TextPackage.py` 将同日 report/search 的 `fact` 拼接，`preds` 未使用；缓存按日期写入 `News-YYYY-MM-DD.npy`。没有保留独立发布时间或文档版本。

| 原模式 | Dataset 行为 | 模型行为 | 新协议的差别 |
| --- | --- | --- | --- |
| `retrieval` | 从预测标签日期首尾区间内挑新闻 | 每个预测步用 query 与候选做点积 Top-K | 未验证文本在 cutoff 前已可用 |
| `aligned` | 每个预测日期读同日新闻，缺失补零 | 扩展为 K_n 份，绕过语义排序 | 仍可能读到预测起点后产生的事实 |
| `random` | 从整个目录随机抽 H 份，文件列表未排序 | 与 aligned 相同，重复 K_n 份 | 可抽未来文本、验证/测试期文本 |
| `ablation=1` | **Dataset 仍读取 query/description/news** | 跳过 TextCrossAttention；融合走 TS-only 分支 | 不是严格“不读取实际文本”的 no_text 控制 |
| `ablation_syn=1` | 数据不变 | 跳过聚合层，仍保留 target/ref 的编码与拼接 | 不是关闭 CTSR |

实际 `TextCrossAttention.retrival()` 只使用 query 相似度；`des_emb` 不参与评分，`qt_ref_text` 赋值后未用于选择。通道描述参与后面的注意力，不能把当前实现描述成 query/description 双评分检索。

候选不足 K_n 时原实现重复已选候选；N=0 会出现除零/空张量问题。Environment/Health/Climate collate 只 padding、不提供 mask；Energy 直接 stack 要求各样本候选数一致。Padding 会参与 Top-K 和注意力，尤其全零可能胜过负相似度候选。新适配器必须返回有效 mask，补齐位置不代表额外证据。

### 2.5 原模型正确性问题与最小修复边界

**本阶段全部只记录，没有修复，也没有重训。** 下表“数学影响/重训”是后续若实施建议的要求。

| 编号 | 原行为与影响 | 最小处理建议 | 是否改变数学定义 / 是否重训 |
| --- | --- | --- | --- |
| M1 | `CrossandOutput` 380/384 行，通道数不等时在 forward 内新建局部 `nn.Linear`，每次随机初始化，不进入 optimizer/state_dict；即使 eval 也受新权重随机性影响 | 首先在公平预设中固定 K_n=K_t+1 使通道相等。若敏感性必须取不等组合，再在 init 注册持久化投影并加独立测试 | 相等通道路径不变；修复不等路径参数生命周期，原不等路径结果不可复用，需重训验证 |
| M2 | `DimensionReducer` 438 行硬编码 C_T=1，多目标被混合 | Fidel-TS 首轮限定一个 target；对多目标配置明确报错 | 限制接口不改变单目标数学；扩展多目标需另行理由与测试 |
| M3 | TS-only 分支取历史最后 H 个 token；H>L 时只输出 L 个，且计算的 `label` self-attention 未用于返回结果 | 首轮可采用官方 H=24、168、336，L=360；H=720 的 TS-only 暂不支持。后续若修复必须单独说明历史→horizon 映射与公平性 | 增加长度投影会改变 TS-only 的时间映射，不能暗中改；需独立测试和重训。不得只延长 TS-only 的输入而仍称同一任务 |
| M4 | N=0、不足 K 重复候选、无 padding mask | 新合法候选适配加 mask、空池确定性无文本路径，不复制证据 | 改变原先错误/未定义边界；对全部有效且 N>=K 的路径保持评分与输出回归一致；新设置需重训 |
| D1 | Scaler fit/transform 列顺序错位 | 独立新数据适配器按命名列固定顺序，保存 Scaler | 不是网络变化，但改变数值输入，需重新训练 |
| D2 | 可选随机裁剪各划分独立，stride 未实现 | 新数据集采用固定时间切分与连续窗口索引 | 不是网络变化；旧权重和新协议结果不能直接拼接 |

M1 在原 `run_Environment.sh` 的 K_t=2、K_n=4..9 等配置中会触发；原 Python 默认参数 K_n=K_t+1 则避开此分支。这两种情况必须分别说明。

`length_reducer` 虽在 init 定义，但没有用于 forward，不能认为它已解决 M3。CTSR-only 对应保留 nref、关闭文本；STR-only 需 nref=0、保留目标历史与文本；No Retrieval 需 nref=0 且关闭文本。当前不存在完全独立、已验证的这四组公平入口。

### 2.6 Baseline 与旧脚本

`baseline/ReTime_real_data.py:564` 先在全表拟合 Scaler，再生成重叠的 L+H 窗口，之后才按窗口数做 7:1:2 划分。既有全量缩放风险，也有划分边界标签重叠风险；其检索候选是随机训练窗口、输出任务也与 ReTaMTSF 单目标历史通道检索不完全相同，不能直接视作公平的 Fidel-TS Baseline。

现有 `run_text_modes.sh`、`run_text_model_experiments.sh`、`run_ts_retrieval_experiments.sh` 和资源 benchmark 含旧环境路径；汇总使用“最新 test_results 目录”，并覆盖固定名字 summary/log。新实验需独立 run ID、显式产物路径、成功标记与失败不覆盖机制。已有 `.gitignore` 有 7,411 行，主要逐文件条目，尚不能保证新 Fidel-TS 缓存都被忽略；阶段 1 再添加新目录规则，本阶段不改。

## 3. Fidel-TS 实际文件与数据结构

### 3.1 California ISO

固定 Revision 下的关键实际路径如下，`YYYY` 为 2018–2025；年份范围来自文件名，不等于已核实每年文件的完整覆盖。

```text
California_ISO/
  README.md
  id_info.json / id_info_except_battery.json
  static_info.json / static_info_demand.json / static_info_except_battery.json
  static_info_embeddings_demand.pkl / static_info_embeddings_except_battery.pkl
  time_series/
    all.parquet
    all_except_battery.parquet
    id_info.json / id_info_except_battery.json
  raw_data/
    all_channel_raw/raw_{co2,demand,fuelsource}.parquet
    all_channel_imputed/{co2,demand,fuelsource}.parquet
  weather/
    san-diego/ 与 san-francisco/
      raw_data/ daily_weather_YYYY.csv、daily_weather_raw_YYYY.json、hourly_weather_raw_YYYY.csv
      weather_report/formal_report/fast_general_formal_forecast_YYYY.json
      weather_report/general_report/general_weather_forecast_YYYY.json
      report_embedding/formal_report/fast_general_formal_embeddings_YYYY.pkl
      report_embedding/general_report/general_weather_embeddings_YYYY.pkl
    merged_general_report/merged_general_weather_forecast.json
    merged_report_embedding/fast_general_formal_embeddings_YYYY.pkl
    merged_report_embedding/static_info_embeddings_except_battery.pkl
  scripts/ clean_and_impute.py、merge_all_channel.py、embed.py、embed_static.py、merge_embed.py、merge_general_report.py
```

文件树来源为[固定版本](https://huggingface.co/datasets/fidel-ts/California_ISO/tree/75f74f556edd5fbdd080c4db6d4b7547fc331f39)，并非 README 的示意目录。

| 文件 | 文件字节数 | 页脚行数 | Channel / Subject | 时间范围，均为无时区字段 |
| --- | ---: | ---: | --- | --- |
| `time_series/all.parquet` | 23,494,657 | 724,896 | 21 个数值列，Subject `all` | 2018-04-10 00:00 至 2025-02-28 23:55 |
| `time_series/all_except_battery.parquet` | 22,264,375 | 724,896 | 20 个数值列，Subject `all_except_battery` | 同上 |

两个文件是同一电网 Subject 的两种通道变体，不能当成两个独立系统重复训练/统计。正式官方配置选择 `all_except_battery`，不是 21 通道的 `all`。

`Time` 的逻辑类型为 timestamp ns、`isAdjustedToUTC=false`，其余列为 double。5 分钟频率与元信息、源码、时间跨度/行数一致，但没有据此证明所有相邻行均为 5 分钟。

`all` 的实际列顺序：

```text
Time,
Biogas CO2, Biomass CO2, Natural Gas CO2, Coal CO2, Imports CO2, Geothermal CO2,
Day ahead forecast, Hour ahead forecast, Current demand,
SOLAR, WIND, GEOTHERMAL, BIOMASS, BIOGAS, SMALL HYDRO, COAL, NUCLEAR,
NATURAL GAS, LARGE HYDRO, BATTERIES, IMPORTS
```

不同列有 MW 与 mtCO2/hr 等不同单位；它们是同一系统的不同 Channel，可以分别标准化后作为历史协变量，但不能当成同一物理量拼成连续序列。通道说明见[官方 static_info](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/static_info.json)。

页脚所有列 null_count=0，但存在很大的负值，如 `Natural Gas CO2` min=-206547、`NATURAL GAS` min=-280718、`Current demand` min=-2651。不能仅凭零 null 宣称数据已正确清洗；不同单位、真实负值与异常需分开检查，不能一律截零。

[`id_info.json`](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/id_info.json) 记录长度 724896 和六段 downtime。源码 [`clean_and_impute.py`](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/scripts/clean_and_impute.py) 将 5 分钟网格中长度>=24 的 NaN 段填零，其余在整列做线性插值。内部缺口可能读取右侧未来观测；“默认 forward 填充方向”不意味着线性插值只用过去。生成长段标签也依赖知道整段长度，不应把未来 downtime 结束时间当作当时已知。

[`merge_all_channel.py`](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/scripts/merge_all_channel.py) 按 DataFrame 行索引横向 concat，不按 `Time` join，时间列取最后一张表。是否三种原始表逐行严格对齐仍需下载后检查；本次未断言发布文件已错位。

`Day ahead forecast`、`Hour ahead forecast` 是电力需求预测通道，但宽表没有它们的发行/修订元信息。首轮只使用其历史区间，不读取其未来区间；这些历史点若有事后修订，仍需记录数据 vintage 限制。不能将列名含 forecast 等价为任意未来值都已知。

### 3.2 Jena Atmospheric Physics

```text
Jena_Atmospheric_Physics/
  README.md / id_info.json / static_info.json / static_info_embeddings.pkl
  raw_data/weather_2014-18.parquet、weather_2019-22.parquet、weather_2023-25.parquet
  time_series/weather_large.parquet、id_info.json
  weather/
    weather_report/formal_report/wm_messages_v1.json、wm_messages_v2.json、wm_messages_v3.json
    weather_report/general_report/fast_general_weather_forecast_2014.json ... 2025.json
    weather_report/general_report/static_info.json
    report_embedding/formal_report/wm_messages_v1.pkl、wm_messages_v2.pkl、wm_messages_v3.pkl
    report_embedding/formal_report/static_info_embeddings.pkl
  scripts/embed.py、embed_static.py
```

来源：[固定文件树](https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics/tree/c8da24175c7aabe5c3b02b503e4c5afe44a605fd)。

`time_series/weather_large.parquet` 为 23,421,608 字节，604,311 行，一个 Subject `weather_large`，21 个 double 数值通道。时间列实际名为 **`Date Time`**，timestamp ns、无时区，页脚范围 2014-01-01 00:10 至 2025-06-30 23:50。

实际数值列如下（`�` 是文件中已存在的 U+FFFD，不是本次展示解码替换）：

```text
p (mbar), T (degC), Tpot (K), Tdew (degC), rh (%),
VPmax (mbar), VPact (mbar), VPdef (mbar), sh (g/kg), H2OC (mmol/mol),
rho (g/m**3), wv (m/s), max. wv (m/s), wd (deg), rain (mm), raining (s),
SWDR (W/m�), PAR (�mol/m�/s), max. PAR (�mol/m�/s), Tlog (degC), CO2 (ppm)
```

10 分钟频率来自官方配置。按页脚 min/max 推算完整 10 分钟网格应有 604,655 行，实际少 344 行。**差值不是已确认恰有 344 个缺失时间戳**：还可能同时含重复或不规则点。滑窗必须检查实际相邻差值；不能默认连续拼接，也不能把 `iloc[::6]` 称为已保证小时对齐。

页脚所有列 null_count=0，但 `wv`、`max. wv`、`SWDR`、`max. PAR`、`CO2` 的最小值均为 **-9999**。这说明存在需要进一步核查的缺失/异常哨兵，不能把零 null 当作无缺失。其比例、位置和官方如何处理尚未核实。

[`static_info.json`](https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics/blob/c8da24175c7aabe5c3b02b503e4c5afe44a605fd/static_info.json) 与实际 Parquet 有四项列名不一致，必须用显式、一一对应、可审计的别名表；保留 raw_channel_id：

| 实际 Parquet Channel | 描述 JSON Channel |
| --- | --- |
| `rho (g/m**3)` | `rho (g/m³)` |
| `SWDR (W/m�)` | `SWDR (W/m²)` |
| `PAR (�mol/m�/s)` | `PAR (μmol/m²/s)` |
| `max. PAR (�mol/m�/s)` | `max. PAR (μmol/m²/s)` |

[`id_info.json`](https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics/blob/c8da24175c7aabe5c3b02b503e4c5afe44a605fd/id_info.json) 只给空的 sensor_downtime，没有 len。这不代表数值无缺口/哨兵。发布树中没有 Jena 的数值清洗/插值脚本，不能套用 CAISO 的脚本来声称已经核实 Jena 的插值方式。Jena README 的 Data source 链接指向 timeanddate 天气页，数值数据溯源和时区证据仍需补充。

### 3.3 天气记录、时间语义与版本：当前的首要阻断项

正式报告的抽样来源：[CAISO San Diego](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/weather/san-diego/weather_report/formal_report/fast_general_formal_forecast_2018.json)、[San Francisco](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/weather/san-francisco/weather_report/formal_report/fast_general_formal_forecast_2018.json)、[Jena v1](https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics/blob/c8da24175c7aabe5c3b02b503e4c5afe44a605fd/weather/weather_report/formal_report/wm_messages_v1.json)。

| 类别 | 已看到的实际结构 | 可以确认 / 不能确认 |
| --- | --- | --- |
| formal_report | `{YYYYMMDDHHMM: {七类文本字段}}`；CAISO 首键 `201801010000`，Jena 首键 `201401010000` | 前缀内相邻键间隔为 6 小时；不能认定此键就是发行时间 |
| 七类字段 | Time of Day、Current overall Weather Condition、Weather Trend、Temperature Trend、Wind Condition、Pressure、Humidity | 是七个文本 facet，不能直接当成七个发布时间版本 |
| general_report | `{YYYYMMDD: {daily, Early Morning, Morning, Afternoon, Evening}}` | 前缀内日间隔；没有独立发行/覆盖区间字段 |
| Jena v1/v2/v3 | 同一首时间键、相同字段名，不同措辞，部分内容也有差异 | 是三个文件版本；没有证据证明它们是逐次发布的预报 vintage，不能把 v3 视作 cutoff 前“最新版本” |
| CAISO raw weather | JSON 数组，含 `date` 毫秒值、`ts`、`ds` 时段、温度/湿度/风/气压等 | 抽样未见明确的 forecast issue/available_at；不能把真实天气重写成将来时后当作已知预报 |

抽样证据：CAISO 两地 formal 前缀分别含 114、112 条完整记录，Jena 三版本前缀分别含 112、116、112 条；每个文件只取 64 KiB。一般报告及合并报告抽样另外统计在 evidence JSON。Jena 2014 general 文件的首个键实际为 `20140201`，不是根据年份推断出的 `20140101`，说明不能靠文件名认定全年覆盖。全量发布时间覆盖率和预报有效覆盖率**尚无法计算**。

目前只能把时间键保留为 `source_time_key`。`issued_at`、`available_at`、`forecast_valid_start/end`、timezone 必须维持未知，除非取得正式文档、生成脚本或发行日志。不能虚构提前一天发行、固定 6 小时有效或零延迟可用，也不能把版本文件名当发布时间。

**可证明的是字段不足，不是已证明所有文本泄漏。** 但这已经足以阻止宣称“严格实时无泄漏”的实验。在 Jena，天气文本与温度等目标高度相关，若由同区间真实观测生成，会直接威胁目标答案泄漏；需要追踪生成来源，而非只做最近时间匹配。

CAISO 合并文本还有独立的问题：[`merge_general_report.py`](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/scripts/merge_general_report.py) 以 `path_parts[-3]` 取 region，对于实际路径两地都会得到 `weather_report`，同时间槽被覆盖。抽查发布的 [`merged_general_weather_forecast.json`](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/weather/merged_general_report/merged_general_weather_forecast.json) 首条也只含 `weather_report` 键，不能还原两地身份。新适配器应从两地独立原文建记录，保留 location_id；不默认采用已合并 JSON。已合并 Embedding 也要核对地点与 facet 的行序，不能只凭形状绑定原文。

**后续最小选择：** 阶段 1 可以制作文件与字段检查工具；阶段 2 的数值单元测试仍可设计，但正式 real_text 训练必须等待时间/来源证据。若最终拿不到，则只能诚实标注为离线档案条件实验，或更换具有可回放发行记录的文本来源；二者都需用户决定并记录。数值 TS-only 可作为独立任务，但不能替代原多模态主张的验证。不能用网络改造修补数据来源问题。

### 3.4 已有 Embedding

生产源码 [`CAISO embed.py`](https://huggingface.co/datasets/fidel-ts/California_ISO/blob/75f74f556edd5fbdd080c4db6d4b7547fc331f39/scripts/embed.py)、[`Jena embed.py`](https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics/blob/c8da24175c7aabe5c3b02b503e4c5afe44a605fd/scripts/embed.py) 指定 `jinaai/jina-embeddings-v3`、`truncate_dim=256`，使用 joblib `.pkl`，未固定模型 revision。static 编码脚本也指定 256 维。**这是源码预期，尚未通过反序列化核实发布数组**。

按代码，每个时间键可能对应多个 facet 向量，而不是 `[1,D]`；CAISO 合并源码将两地数组纵向堆叠。静态通道向量按 JSON 插入顺序构造，不等于 Parquet 列顺序；CAISO 两种顺序实际不同。官方脚本还含旧路径、glob 与硬编码代理设置，不能直接复制运行。

适配优先重新编码合法原文为原模型的 MiniLM 384 维，固定 revision、模板、截断策略和文档身份。若选择复用 256 维 embedding，只能显式投影并记录新增参数/重训影响；不能偷偷裁剪、补零或冒充 384 维语义空间。本阶段未下载权重、未运行官方 trust_remote_code、未反序列化 Pickle。

## 4. 官方协议：已确认内容与复现差别

[论文附录 C.4](https://arxiv.org/html/2509.24789v5#A3.SS4) 给出 CAISO/Jena 小时任务：L=360，H∈{24,168,336,720}；训练在 2021-01-01 前，验证在 2021 年，测试从 2022-01-01 开始，主指标 MSE/MAE。不能把旧 7:1:2 直接称为新 Benchmark 的正式协议。

固定版本的 [CAISO 小时 YAML](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/data_configs/California_ISO/fullCAISO_H.yaml) 和 [Jena 小时 YAML](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml) 一致指定 timestamp split 2021/2022、target=all，CAISO `all_except_battery`、downsample=12；Jena `weather_large`、downsample=6。

需要明确的代码细节：

- 官方 [`Universal_Dataset`](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/data_provider/data_loader.py) 在 native 训练数据上 fit Scaler，transform 后才 `data[::downsample]`；这是逐行抽样，不是小时均值聚合。新实验若改为小时数据上拟合或小时均值，应标成协议变体，不能默默混用。
- [`timestamp_spliter`](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/data_provider/data_helper.py) 不给 val/test 追加前一划分的 L 点历史。若采用允许预测起点前历史的滚动评测，必须记录由此增加的窗口数量和与官方数字的差别，不能只复制时间边界就称完全复现。
- 当前官方 loader 有可选连续窗口检查，但需要配置采样间隔，不能据此认为发布数据已无缺口。Jena 按行每六点抽样，在缺口后相位可能改变，应对照实际时间戳和官方窗口清单验证。
- [CAISO 文本 YAML](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/data_configs/California_ISO/fullCAISO_hetero_TGTSF_H.yaml) 使用合并 formal Embedding、matching=backward；[Jena 文本 YAML](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF_H.yaml) 指定 v1。它们并未实现独立发行时间字段；backward 匹配时间键不构成发布时间证明。
- 官方 `time_matcher` 的 nearest 可能选后面的键，random 从整池选；backward 用 searchsorted-1，在恰好同键时选上一条，首条前又会夹到首条。新实现不能直接复用这些规则来满足 cutoff 可用性。

正式数值 Baseline 的代码入口确实存在：DLinear、PatchTST、iTransformer；对应 CAISO/Jena 脚本列出小时 H=24/168/336/720、L=360。可从官方实现适配到统一 Dataset，不直接搬用旧 ReTime 数据协议。

不同脚本预算并不统一，例如 [CAISO DLinear](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/scripts/California_ISO/dlinear.sh) 显式 30 epoch、batch 256；[Jena DLinear](https://github.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline/blob/81043864342e550ed9ffdaa7d9ae6d8420460c6e/scripts/Jena_Atmospheric_Physics/dlinear.sh) batch 512、epoch 走默认。TGTSF 官方两数据集脚本预算、horizon 列表也不同，仍需阶段 7 检查模型配置和输入假设。不得声称官方 Baseline 已在本仓库运行。

**原模型是单目标、官方 YAML 是全部目标。** 迁移方案应先建立可审计的固定目标任务，所有方法共享它。若要直接与官方多目标数字比较，应完成逐目标实验并明确聚合协议/计算预算，或另行验证多目标支持；不能把单个容易目标的分数与官方全通道均值并列为主结果。

## 5. 数据接口差异与适配设计

### 5.1 统一 Batch

| 新字段 | 目标形状/内容 | 适配原则 |
| --- | --- | --- |
| `target_series_x/y` | `[B,1,L]` / `[B,1,H]` | 首轮一个目标 Channel；标签严格落在划分内 |
| `TS_database` | `[B,C_reference,L]` | 同 Subject、同历史网格；排除目标，不读未来 |
| `query_text_embeddings` | `[B,1,H,384]` | 由已知日历和明确模板生成，保存模板 hash |
| `channel_description_embeddings` | `[B,1,1,384]` | 显式 raw ID→描述别名，不按位置猜测 |
| `candidate_text_embeddings` | `[B,Nmax,1,384]` | 每候选一个确定文档/facet向量；合法性先过滤 |
| `candidate_text_mask` | `[B,Nmax]` bool | Padding 不参与 Top-K/attention；不足 K 不复制证据 |
| `forecast_cutoff` | 带明确定义的时间点 | 区分最后历史时间与首预测时间，记录索引约定 |
| `sample_metadata` | 数据来源、Revision、Subject、原始 Channel、location、document/facet/version ID、时间字段与有效性状态 | 支持逐预测回溯 |

数值与文本时区要分别核实后转换，不能因地名就直接强行 localize；两地均有 DST，须处理重复/不存在的本地时间。未知时区是显式审计项，不能用服务器 Asia/Shanghai 解释数据。

文本合法池必须以真实 `available_at<=cutoff` 为依据；已提前发布、覆盖未来的预报可以使用。覆盖未来不是拒绝条件，发行晚于 cutoff 才是。没有发行证据时禁止假造值来让测试通过。

### 5.2 计划新增及最小修改文件（本阶段未创建实现）

| 阶段 | 计划文件 | 目的 |
| --- | --- | --- |
| 1 | `scripts/fidel_ts/download.py`、`inspect.py`，`configs/fidel_ts/sources.json` | pin revision、list/dry-run、选择性下载、恢复/校验、manifest/hash；只检查或字段统计 |
| 2 | `data_provider/fidel_ts_schema.py`、`fidel_ts_dataset.py`、`fidel_ts_collate.py` | 独立 Subject/Channel、固定切分、Scaler、连续窗口、结构化 Batch |
| 2 | `configs/fidel_ts/{caiso,jena}.yaml`，`tests/fidel_ts/test_dataset.py` | 明确 native/hourly、目标、切分、别名、缺失策略、合成数据测试 |
| 3 | `data_provider/fidel_ts_text.py`，`pre_process/Fidel-TS/encode_text.py`，`tests/fidel_ts/test_text_availability.py` | 可用性与版本过滤、稳定 Top-K、离线 384 维编码、六种文本控制 |
| 4 | `exp/exp_ReTaMTSF_FidelTS.py`、`run_ReTaMTSF_FidelTS.py`、`tests/fidel_ts/test_model_smoke.py` | 明确 checkpoint/run ID，新入口，不继承旧测试窥视与最新目录选择 |
| 5–7 | `scripts/fidel_ts/run_pilot.sh`、`summarize.py`、`evaluate.py`、`benchmark.py`，消融/绘图脚本 | 生成手动命令，保留预测/配置/计时/资源/seed/版本与负结果 |
| 7 | `docs/fidel_ts/final_experiment_report.md` | 由保存的原始产物生成报告，不能填入未运行数字 |

原文件必要修改候选：

- `models/ReTaMTSF.py`：仅为新候选 mask/真正不读文本的分支增加向后兼容的可选参数（若无法在包装层实现）。原五参数调用保留。
- `layers/ReTaMTSFLayers.py`：最小 mask/空池处理；不等通道投影若必须支持，移到 init 并单独记录。不得顺手改评分公式；M3 需单独设计确认，未预先承诺更换预测头。
- `.gitignore`：后续新增 `dataset/Fidel-TS/`、`artifacts/fidel_ts/`、新编码缓存规则，保证审计文档可跟踪。
- 旧 `data_loader.py`、Time-MMD 预处理、四套原训练入口和已有结果可全部保持不变；新 factory 可独立实现，不强求改原 factory。

在输入合法、单目标、通道数相等时，可完全保留 coherence/correlation 评分、Content Synthesis、TextCrossAttention 的有效 token 运算、Cross-modal Fusion、DimensionReducer 的单目标运算、MLP 输出头、MSE 和 Adam。无需 Gate、对比损失、LLM Backbone 或新 Attention。

## 6. 关键风险、修正和放行条件

| 风险 | 影响类型 | 最小控制及放行证据 |
| --- | --- | --- |
| 天气发行/可用时间未知、可能是历史观测生成文本 | 科研正确性，当前阻断正式多模态训练 | 取得生成脚本/发行档案/时间语义；保留未知，不能假设 |
| v1–v3 无发行顺序依据 | 时间回放错误 | 固定文本变体；有真实发行证据后才做 cutoff 版本选择 |
| CAISO merged general 地点身份丢失 | 文本身份/覆盖错误 | 使用两地原文；按 `(location,time,facet,version)` 绑定文档与 Embedding |
| 原 Time-MMD 预测区间事实检索、全池 random | 未来信息访问 | 独立合法池；对无关未来文本的添加/修改保持当前候选不变 |
| 短缺口插值与事后 downtime 事件 | 数值/文本未来观测污染 | 官方插值协议与严格因果协议分开命名；不能拿严格变体直接对比官方数字 |
| Jena -9999、缺口、列名失配 | 数值任务错误 | 原始与插值数据核对、显式 sentinel mask、连续窗口检查、唯一别名表 |
| CAISO 按行合并而非按时间 join | 历史通道可能错位 | 对三张原表时间轴作一致性检查；不凭源码推断发布文件已损坏 |
| Scaler 列顺序/fit 范围、各划分独立随机起点 | 数值和划分错误 | 固定时间索引，train-only fit，保存列顺序和统计量 |
| forward 新建投影、多目标混合、H>L 的 TS-only | 训练/评测有效性 | 限制合法配置或最小修复；确定性、参数注册、形状和 checkpoint 回归测试 |
| 文本不足 K、Padding 被选、空池 NaN | 证据重复/数值不稳定 | mask、稳定 tie-break、空池路径；全有效输入数值回归 |
| 官方多目标与新单目标任务不同 | 主结果不可比 | 所有 Baseline 同目标同窗口；明确逐目标汇总与权重 |
| attention 成本随候选数/H 放大 | 性能和资源 | 先按张量尺寸估算，少量 smoke；完整资源 benchmark 手动执行 |

两数据集适配难度：

- **CAISO：数值接口中等、文本审计高。** 一个 Subject、20 个正式通道适合 CTSR；但存在多地点合并、需求预测 vintage、缺失插值和原表行对齐问题。不能因为天气与负荷相关就预期模型会获益。
- **Jena：数值接口中等偏高、文本审计高。** 一个 Subject、21 通道适合 CTSR，但有时间缺口、-9999 与列名编码问题，数值清洗出处未齐；文本可能直接描述气象目标，更需要来源验证。
- 首轮固定目标可考虑 CAISO `Current demand`、Jena `T (degC)`，这只是语义合理的候选，**尚未定案、不是官方唯一目标**。应在看测试结果前根据论文任务与数据质量确定目标清单，并最终报告逐目标结果。

## 7. 后续阶段执行计划与测试计划

所有阶段均遵守：完成后汇报并停止，等待用户明确“继续下一阶段”。预计超过约 60 秒或成本不明确的下载/预处理/编码/训练/大评测交给用户；先 dry-run 和列规模。没有后台任务。

| 阶段 | 具体工作和最低验收 | 手动交接 |
| --- | --- | --- |
| 1 | 实现 pin Revision、目录隔离、list-only、文件过滤、断点恢复与已有文件/hash校验；检查所有时间/文本字段。以小型 fixture 测试下载异常、混用 revision 拒绝、无覆盖 | 输出真实脚本命令与预计字节数，用户按需下载 CAISO/Jena；不要先下载其余四个集合。文本来源审计仍是阻断项 |
| 2 | 合成双 Subject 多 Channel 数据；验证不跨 Subject、不跨 gap、标签区间、重复/缺失处理、train-only Scaler 与列顺序；CPU collate/B=1和多 B形状 | 全量处理脚本用户执行，保存窗口 manifest 与协议 hash |
| 3 | 只有获得可用性证据的文本才进 real_text。测试晚发拒绝、多版本 cutoff、合法未来预报保留、Subject/地点隔离、空池无 NaN、ID绑定、随机/打乱不读未来、未来不变性 | 先统计长度；用户全量 MiniLM 编码。保存模型 revision、模板、token截断统计与哈希 |
| 4 | 仅做小批 CPU forward/backward/一次 optimizer step、checkpoint往返；环境允许时少量 GPU。覆盖 text on/off、CTSR-only、STR-only、Full、B与H变化，参数/梯度/有限值检查 | M1/M3 如需修复先独立记录数学影响；不得借此启动完整训练 |
| 5 | Persistence、DLinear、TS-only、Full、Date-only、Random Legal、No Text；六种文本控制中补 Content-only/Shuffle。先官方 H=24 做流程验证，再按资源安排；不依测试挑目标 | 用户单 seed 先导→固定至少三 seed 的顺序脚本；严格同任务预算、显式 GPU和失败隔离 |
| 6 | 四种双路检索消融，已有 coherence/lagged/partial/Granger与新增random，K/L/H敏感性；配对损失与时间块Bootstrap，不将同时间不同seed当独立样本 | 用户多种子、多horizon训练/长期评测；报告所有负结果。H720若原 TS-only 不支持必须先解决或公开限制 |
| 7 | 接入 PatchTST、iTransformer、核查 TGTSF/ReTime；公平预算；离线编码与在线检索/forward/端到端耗时分开 | 用户完整训练与性能测量；用原始预测与配置生成最终报告，不编造结果 |

六种文本控制需语义明确：`real_text`、`date_only`、`content_only`、`random_legal`、`shuffled_content`、`no_text`。content_only 只移除人为日期前缀；random_legal 只在合法池抽样；shuffle 保留时槽和版本，只用当时合法内容；no_text 不加载真实天气文本。TS-only 与 no_text 的参数和运算路径差别必须明确报告，不能把二者命名当成已相同。

长 horizon 的资源风险：DimensionReducer 将通道×H 展成 attention 序列，成本随其平方增长；候选构造还大量 repeat。H=720、多 K、高 batch 不能直接沿用旧 32/64 batch。先输出尺寸估算、小 batch 检查，资源允许与否不能通过测试集分数决定。

## 8. 本阶段验证与产物

实际完成的轻量验证：

1. 对模型、层、四个入口/实验、加载器、相关检索工具、Time-MMD 全部 Python 预处理和 Baseline/benchmark 共 **46 个文件**执行 `ast.parse`，全部通过。这是语法检查，不是依赖导入或模型正确性验证。
2. AST 确认 forward 内两处局部 Linear（380/384 行）、单目标硬编码（438 行）、四套加载器连续 indices（56/236/415/593 行）。
3. 直接执行原 AST 的 indices 表达式：stride=3 仍为 `[5,6,7,8]`。用极小标准库算例确认独立裁剪可重叠、Scaler 列错位、H720/L360 的 TS-only 切片长度、官方两地 region 键相撞。
4. 验证 12 个 Range 请求为 HTTP 206，各 65536 字节；解析三个 Parquet footer 的 schema/行数/统计与八个天气字典前缀，另检查一个 raw-weather 数组前缀；没有反序列化 Pickle。
5. 读取固定官方 SHA 的 YAML、切分函数、文本匹配和 Baseline 脚本，与论文附录核对。

未执行：PyTorch forward/backward、GPU可用性测试、全量数据完整性、官方 Embedding 实际维度、训练/评测。缺少依赖不等于这些测试已通过。本阶段未修复代码，所以没有“修复后训练正确”的结论。

新增产物仅两项：

- `docs/fidel_ts/00_repository_audit.md`：本报告。
- `docs/fidel_ts/00_audit_evidence.json`：固定来源、哈希与元数据摘要。

没有实验指标、预测文件、模型权重或实验结果表。

## 9. 手动验收与下一阶段命令

### 9.1 当前可运行：离线核验报告和审计快照

以下命令不联网、不读完整数据、不使用 GPU，输出到终端，通常不足 1 秒：

```bash
cd /opt/data/private/ATM-ReTaMTSF
python - <<'PY'
import ast, hashlib, json
from pathlib import Path
report = Path('docs/fidel_ts/00_repository_audit.md')
evidence = Path('docs/fidel_ts/00_audit_evidence.json')
assert report.is_file() and report.stat().st_size > 0
x = json.loads(evidence.read_text())
assert x['scope'] == 'stage_0_only'
assert len(x['datasets']['caiso']['files']) == 147
assert len(x['datasets']['jena']['files']) == 32
for name, digest in x['local_source_sha256'].items():
    p = Path(name)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
    ast.parse(p.read_text(), filename=name)
assert [p['rows_footer'] for p in x['parquet_metadata']] == [724896, 724896, 604311]
assert not x['checks']['training_run'] and not x['checks']['pickle_loaded']
print('PASS: audit files, pinned inventory, source hashes and metadata summary')
PY
git diff --check
git status --short
```

成功标准：打印 PASS，`git diff --check` 无错误，源码无改动；新产物只有上述两个文档（另有用户原先的 `Prmopt.md`）。后续阶段有意修改源码后，本地 hash 检查会提示旧审计快照已过期，这是预期行为，应保留版本出处而不是随意更新哈希掩盖变化。

### 9.2 阶段 1 获批后可手动运行：固定版本文件清单，仍不下载数据

**现在无需运行，也不要据此提前开始下一阶段。** 下面是为交接提供的可复制元数据命令；它只保存清单，正式下载入口尚未实现，不把将来脚本冒充本阶段成果。

```bash
cd /opt/data/private/ATM-ReTaMTSF
python - <<'PY'
import json, os, urllib.request
from pathlib import Path
repos = {
    'California_ISO': '75f74f556edd5fbdd080c4db6d4b7547fc331f39',
    'Jena_Atmospheric_Physics': 'c8da24175c7aabe5c3b02b503e4c5afe44a605fd',
}
out = Path('artifacts/fidel_ts/stage1_inventory')
out.mkdir(parents=True, exist_ok=True)
for name, revision in repos.items():
    url = f'https://huggingface.co/api/datasets/fidel-ts/{name}/tree/{revision}?recursive=true&limit=1000'
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read(2_000_001)
        assert len(data) <= 2_000_000, 'metadata size cap exceeded'
        assert 'rel="next"' not in response.headers.get('Link', ''), 'pagination required'
    entries = json.loads(data)
    payload = {'repo': f'fidel-ts/{name}', 'revision': revision, 'url': url, 'entries': entries}
    path = out / f'{name}_{revision}_files.json'
    with path.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(name, 'files=', sum(e['type'] == 'file' for e in entries),
          'repository_bytes=', sum(e.get('size', 0) for e in entries if e['type'] == 'file'),
          'manifest=', path)
PY
```

输入是上述两个公开仓库的固定 Revision，输出为 `artifacts/fidel_ts/stage1_inventory/*_files.json`。成功标准：两个清单分别含 147、32 个文件；数值字段显示的是仓库总字节估算，不是已下载字节。低 CPU、无需 GPU，最多两份 2 MB 元数据；两次网络请求各设 20 秒超时。网络/限流失败时保存错误日志并停止，不换成未固定 main；文件已存在会因 `open('x')` 报错，先核对现有清单，不自动覆盖。

该命令在阶段 0 **未作为阶段 1 工作运行**。正式阶段 1 会实现并小样本验证选择性下载、断点/校验工具，再给出 CAISO/Jena 独立的真实下载与检查命令、输出列表和资源估算；全量下载仍交给用户。

## 10. 当前问题状态与停止点

- **已解决的调查问题：** 定位实际任务文件；原模型真实数据流、随机起点默认/非默认行为、单目标限制、具体 HF Revision、真实列名与版本树、官方切分与窗口已查明。
- **尚未修复且影响正确性：** 天气发行/来源/时区证据不足，CAISO 地点合并与通道对齐待核实，Jena 哨兵/缺口/描述映射，原 Scaler 与网络 M1–M4。不得把本报告当成已解决这些问题。
- **性能问题：** 长序列 attention、重复磁盘读取、repeat、调试 CSV/PDF、旧资源脚本路径与计时；只在协议正确后做必要优化。
- **下一阶段：** 等待用户明确回复“继续下一阶段”，仅实施阶段 1 下载及检查工具。正式多模态实验的放行还需要时间可用性证据与模型最小修复核验。

**阶段 0 到此停止。**
