# PIVOT Stage 1.5 Dataset Readiness Decision

Generated from user-executed, hash-checked artifacts. This report does not approve model training.

## A. 可验证事实

来源为各数据集的 Stage 1.5 manifest、numerical diagnostics、N1/N2 quality/window reports 与 text diagnostics；未联系作者。

### caiso

原始行数：724896；通道数：20；额外重复记录：0；重复组：0；唯一时间轴缺口：0。

报告数：26186；Facet 数：172826；T1：blocked_not_verifiable。

### jena

原始行数：604311；通道数：21；额外重复记录：334；重复组：334；唯一时间轴缺口：8。

报告数：54564；Facet 数：373596；T1：blocked_not_verifiable。

## B. 基于证据的推断

观测式字段、趋势文字与实况接近只提示调查风险，不能单独证明事后生成或不存在泄漏。N2 与 N1 的相位/窗口差异见 downsampling_comparison.json。

## C. 无法验证

每条报告的历史发行时间、未保存的预报更新版本、缺少生成脚本的文本生成方式，以及无时区字段的真实物理时间轴，不能由人工假设补齐。

## D. 默认决策

N1 仅用于官方可比性；N2 是全通道候选主数值协议。T2 仅用于历史档案辅助研究；T3 仅用于回溯对齐与潜在泄漏诊断；T1 在缺少发行/版本证据时阻断。

Temporal Relevance 不得伪造 valid 区间；报告时间键距离只能另行命名为探索性评分。Utility 不能绕过可用性要求。

## 分项 Readiness

```json
{
  "caiso": {
    "numerical_mimo_readiness": {
      "status": "candidate_for_stage2_review",
      "reasons": [],
      "default_protocol": "N2",
      "N1_role": "official_compatible_comparability",
      "scope": "numerical artifacts only, no MIMO Dataset or model implemented"
    },
    "archived_text_research_readiness": {
      "status": "auxiliary_only",
      "T2": "past_source_key_only_not_verified_availability",
      "T3": "retrospective_oracle_risk_diagnostics_only"
    },
    "verified_asof_multimodal_readiness": {
      "status": "blocked_not_verifiable",
      "T1": "blocked_not_verifiable",
      "author_contact_required": false,
      "next_source_requirement": "reconstruct an external text archive with trustworthy issuance and version records if needed"
    },
    "remaining_scientific_limitations": [
      "no verified historical text issuance or forecast validity",
      "naive clocks and unresolved timezone semantics",
      "published upstream interpolation cannot be undone by N2",
      "extreme finite CAISO values are retained and require scientific review, not blanket negative-value removal",
      "N1 and N2 scores must be reported as different protocols",
      "all methods must share all channels, window identities, scaler and y_mask"
    ]
  },
  "jena": {
    "numerical_mimo_readiness": {
      "status": "candidate_for_stage2_review",
      "reasons": [],
      "default_protocol": "N2",
      "N1_role": "official_compatible_comparability",
      "scope": "numerical artifacts only, no MIMO Dataset or model implemented"
    },
    "archived_text_research_readiness": {
      "status": "auxiliary_only",
      "T2": "past_source_key_only_not_verified_availability",
      "T3": "retrospective_oracle_risk_diagnostics_only"
    },
    "verified_asof_multimodal_readiness": {
      "status": "blocked_not_verifiable",
      "T1": "blocked_not_verifiable",
      "author_contact_required": false,
      "next_source_requirement": "reconstruct an external text archive with trustworthy issuance and version records if needed"
    },
    "remaining_scientific_limitations": [
      "no verified historical text issuance or forecast validity",
      "naive clocks and unresolved timezone semantics",
      "published upstream interpolation cannot be undone by N2",
      "extreme finite CAISO values are retained and require scientific review, not blanket negative-value removal",
      "N1 and N2 scores must be reported as different protocols",
      "all methods must share all channels, window identities, scaler and y_mask"
    ]
  }
}
```

## 测试执行记录

```json
{
  "status": "reported_pass",
  "testcases": 65,
  "failures_or_errors": 0,
  "skipped": 0,
  "stage1_5_test_modules_seen": [
    "test_stage1_5_numerical",
    "test_stage1_5_provenance",
    "test_stage1_5_outputs"
  ],
  "junit_sha256": "eedc60225f9092b0860ec11c11ea41d8249b5f7158b579451eb6973844733c45",
  "limitation": "user-supplied execution record; not an independent attestation of the executed source tree"
}
```

Stage 2 不自动执行；数值产物进入 review 不代表已实现或验证 MIMO 模型。没有笼统的 ready:true。
