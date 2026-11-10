# 可解释视觉状态与风险分层的图像分类测试诊断：六类实证研究

本稿更新主次结果与适用范围。旧20261004论文保留；新增来源只有执行完成后才报告。研究不要求覆盖—F1曲线单调，不提出训练增强或全面优势。

## 摘要

本研究在固定六类分类任务、1800训练目标、八架构及三个训练种子下，用线上/本地VLM描述可见状态并构造类别内视觉风险层。COCO新池的24次运行均出现高风险层F1低于低风险层；在共同风险配额和逐类别等错误数控制下，Coverage相对Random提供混淆类型诊断收益。统一表征错误簇和强基线比较呈现条件性结果，不能概括为普遍优势。自动标签不是人工真值，风险评分不代表所有困难因素。

## 问题与贡献

RQ1：预定义风险层能否区分固定模型的测得性能？RQ2：旧池冻结薄弱状态是否在新池/新来源复现？RQ3：控制预定义风险、预算及错误数后，Coverage是否增加独立混淆类型/错误簇广度？
贡献是可复算的可解释状态测试组织协议、同风险等错误数诊断对照，以及跨池复现和条件性负结果的证据。没有文献核验支持“首创”；不把显著性当新颖性。

## 数据、协议与统计

原COCO A为发现池、COCO B为2400独立样本确认池，各类400；B的弱状态清单在预测前由A冻结，外部来源沿用清单不重选。分类权重、预处理、五维H、unknown规则和支持阈值固定。状态标注三视图且类别可见，分类器只用固定目标框裁剪。
VOC2012因完整历史候选排除后dog仅47张停止，未开展新标注/分类预测。改用VOC2007 trainval+test合并，六类100/class，预算10–50/class，各层20候选选10。COCO B选样比例25%、VOC50%；各来源内部方法预算相同，不能把不同池原始覆盖分母直接合并。
全部24运行来自8架构×3种子。主要风险6项Holm、主要随机诊断4项Holm、冻结状态迁移2项Holm；强基线12项、删字段4项及统一ViT和全20敏感性不省略。12/20族存在精确架构检验分辨率限制，未通过不能当零效应；固定面板图像bootstrap不能代表新增架构或重新选样。

## 主要风险确认

| 数据池 | 来源 | Q5−Q1 Macro-F1 (pp) | 95%区间(pp) | 六项Holm p |
|---|---|---:|---|---:|
| sixclass_independent_confirmation_20261004 | local | -25.50 | [-27.17,-23.94] | 0.04688 |
| sixclass_independent_confirmation_20261004 | online | -21.30 | [-22.59,-19.86] | 0.04688 |
| sixclass_voc_external_confirmation_20261005 | local | -16.57 | [-19.47,-13.52] | 0.04688 |
| sixclass_voc_external_confirmation_20261005 | online | -14.44 | [-16.65,-11.97] | 0.04688 |

## 外部来源确认状态

VOC2007已完成；以下与COCO B分开报告。

## 全部方法结果与负结果

| 池 | 表征 | 来源 | 对照 | 指标 | AUC差 | 家族Holm p | 全20 p |
|---|---|---|---|---|---:|---:|---:|
| sixclass_independent_confirmation_20261004 | classifier | local | random | confusion | +0.0158 | 0.03125 | 0.15625 |
| sixclass_independent_confirmation_20261004 | classifier | local | random | clusters | +0.0103 | 0.03125 | 0.23438 |
| sixclass_independent_confirmation_20261004 | classifier | local | deepgini | confusion | +0.0197 | 0.14062 | 0.23438 |
| sixclass_independent_confirmation_20261004 | classifier | local | deepgini | clusters | +0.0159 | 0.09375 | 0.15625 |
| sixclass_independent_confirmation_20261004 | classifier | local | dsa | confusion | +0.0205 | 0.14062 | 0.23438 |
| sixclass_independent_confirmation_20261004 | classifier | local | dsa | clusters | +0.0122 | 0.16406 | 0.28125 |
| sixclass_independent_confirmation_20261004 | classifier | local | kcenter | confusion | +0.0075 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | local | kcenter | clusters | +0.0071 | 0.84375 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | local | coverage_without_H | confusion | -0.0060 | 0.86719 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | local | coverage_without_H | clusters | +0.0037 | 0.86719 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | online | random | confusion | +0.0224 | 0.03125 | 0.15625 |
| sixclass_independent_confirmation_20261004 | classifier | online | random | clusters | +0.0070 | 0.04688 | 0.51562 |
| sixclass_independent_confirmation_20261004 | classifier | online | deepgini | confusion | +0.0250 | 0.09375 | 0.15625 |
| sixclass_independent_confirmation_20261004 | classifier | online | deepgini | clusters | +0.0060 | 0.84375 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | online | dsa | confusion | +0.0249 | 0.09375 | 0.15625 |
| sixclass_independent_confirmation_20261004 | classifier | online | dsa | clusters | +0.0041 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | online | kcenter | confusion | +0.0064 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | online | kcenter | clusters | -0.0043 | 0.96875 | 1.00000 |
| sixclass_independent_confirmation_20261004 | classifier | online | coverage_without_H | confusion | +0.0086 | 0.28125 | 0.70312 |
| sixclass_independent_confirmation_20261004 | classifier | online | coverage_without_H | clusters | -0.0033 | 0.86719 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | random | confusion | +0.0158 | 0.03125 | 0.15625 |
| sixclass_independent_confirmation_20261004 | vit | local | random | clusters | +0.0038 | 0.25781 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | deepgini | confusion | +0.0197 | 0.15625 | 0.25000 |
| sixclass_independent_confirmation_20261004 | vit | local | deepgini | clusters | +0.0063 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | dsa | confusion | +0.0205 | 0.15625 | 0.25000 |
| sixclass_independent_confirmation_20261004 | vit | local | dsa | clusters | +0.0047 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | kcenter | confusion | +0.0075 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | kcenter | clusters | -0.0084 | 0.82031 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | coverage_without_H | confusion | -0.0060 | 0.86719 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | local | coverage_without_H | clusters | -0.0004 | 0.87500 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | online | random | confusion | +0.0224 | 0.03125 | 0.15625 |
| sixclass_independent_confirmation_20261004 | vit | online | random | clusters | +0.0078 | 0.10938 | 0.60156 |
| sixclass_independent_confirmation_20261004 | vit | online | deepgini | confusion | +0.0250 | 0.09375 | 0.15625 |
| sixclass_independent_confirmation_20261004 | vit | online | deepgini | clusters | +0.0194 | 0.15625 | 0.28125 |
| sixclass_independent_confirmation_20261004 | vit | online | dsa | confusion | +0.0249 | 0.09375 | 0.15625 |
| sixclass_independent_confirmation_20261004 | vit | online | dsa | clusters | +0.0172 | 0.15625 | 0.25000 |
| sixclass_independent_confirmation_20261004 | vit | online | kcenter | confusion | +0.0064 | 1.00000 | 1.00000 |
| sixclass_independent_confirmation_20261004 | vit | online | kcenter | clusters | +0.0127 | 0.15625 | 0.25000 |
| sixclass_independent_confirmation_20261004 | vit | online | coverage_without_H | confusion | +0.0086 | 0.28125 | 0.70312 |
| sixclass_independent_confirmation_20261004 | vit | online | coverage_without_H | clusters | -0.0041 | 0.86719 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | random | confusion | +0.0018 | 0.98438 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | random | clusters | +0.0010 | 0.98438 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | deepgini | confusion | +0.0039 | 0.78125 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | deepgini | clusters | +0.0069 | 0.76562 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | dsa | confusion | +0.0043 | 0.78125 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | dsa | clusters | +0.0061 | 0.78125 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | kcenter | confusion | -0.0072 | 0.76562 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | kcenter | clusters | -0.0089 | 0.78125 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | coverage_without_H | confusion | -0.0015 | 0.73438 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | local | coverage_without_H | clusters | +0.0023 | 0.63281 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | random | confusion | +0.0185 | 0.03125 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | random | clusters | +0.0164 | 0.03125 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | deepgini | confusion | +0.0238 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | deepgini | clusters | +0.0195 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | dsa | confusion | +0.0197 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | dsa | clusters | +0.0178 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | kcenter | confusion | +0.0191 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | kcenter | clusters | +0.0119 | 0.78125 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | coverage_without_H | confusion | +0.0127 | 0.06250 | 0.20312 |
| sixclass_voc_external_confirmation_20261005 | classifier | online | coverage_without_H | clusters | +0.0088 | 0.63281 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | random | confusion | +0.0018 | 0.98438 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | random | clusters | -0.0003 | 0.98438 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | deepgini | confusion | +0.0039 | 0.65625 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | deepgini | clusters | +0.0073 | 0.65625 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | dsa | confusion | +0.0043 | 0.65625 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | dsa | clusters | +0.0067 | 0.65625 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | kcenter | confusion | -0.0072 | 0.65625 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | kcenter | clusters | -0.0153 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | vit | local | coverage_without_H | confusion | -0.0015 | 0.89062 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | local | coverage_without_H | clusters | +0.0040 | 0.82031 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | online | random | confusion | +0.0185 | 0.03125 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | vit | online | random | clusters | +0.0072 | 0.09375 | 0.43750 |
| sixclass_voc_external_confirmation_20261005 | vit | online | deepgini | confusion | +0.0238 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | vit | online | deepgini | clusters | +0.0100 | 0.25000 | 0.43750 |
| sixclass_voc_external_confirmation_20261005 | vit | online | dsa | confusion | +0.0197 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | vit | online | dsa | clusters | +0.0088 | 0.43750 | 0.75000 |
| sixclass_voc_external_confirmation_20261005 | vit | online | kcenter | confusion | +0.0191 | 0.09375 | 0.15625 |
| sixclass_voc_external_confirmation_20261005 | vit | online | kcenter | clusters | -0.0080 | 0.65625 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | vit | online | coverage_without_H | confusion | +0.0127 | 0.06250 | 0.23438 |
| sixclass_voc_external_confirmation_20261005 | vit | online | coverage_without_H | clusters | +0.0027 | 0.89062 | 1.00000 |

COCO B分类器表征下Random主四项支持，统一ViT错误簇收益未通过主家族校正。强基线与删字段优势没有稳定确认；不能只展示正向平均值。全部区间和逐预算/运行结果见full_outcome_index.csv及原始表。

## 冻结状态迁移

| 池 | 来源 | 状态减类别Recall(pp) | 两项Holm p |
|---|---|---:|---:|
| sixclass_independent_confirmation_20261004 | local | -21.73 | 0.01562 |
| sixclass_independent_confirmation_20261004 | online | -20.77 | 0.01562 |
| sixclass_voc_external_confirmation_20261005 | online | +1.62 | 1.00000 |
| sixclass_voc_external_confirmation_20261005 | local | -23.83 | 1.00000 |

## 成本与实际案例

成本明细见cost_summary.csv、cost_raw_response_ledger.csv与stage_durations.csv；未提供实际账单或历史费率，金额未知。已知token包括有usage的失败/重试响应；缺失usage不能当0。局部服务器时长、成功请求时长、并行墙钟与GPU活跃时间不能混用。cached_policy_costs.csv为ResNet18首种子的两次代表性CPU缓存选样实测，不是全部模型均值；Random1000次与单次固定策略分别标明，不直接作速度排名。
diagnostic_cases.html同时展示原A冻结状态在新池的复现/未复现、同状态正确样本、高风险成功和VLM分歧。全状态支持、三种子差异和风险匹配差见state_case_metrics.csv。案例经结果筛选用于解释，不是独立假设检验或失败根因证明。

## 有效性边界与结论

相对于当前五维风险评分的额外信息不等于控制全部难度；自动状态的可解释性不等于语义准确率。有限类别、8架构、数据来源与单目标分类限制泛化；旧VOC发现过程、预训练数据可能重叠均说明。
研究支持在所测固定模型与候选池中用视觉风险组织能力测试，并用状态诊断定位可复现的薄弱条件。Coverage相对风险匹配Random的混淆类型广度提供条件性证据；对强基线、统一表征错误簇的普遍优势仍未确立。

## 与旧完整论文的关系

本稿为增补后的证据与口径稿，旧完整论文保留文献和详细方法：/home/hcf/.codex/worktrees/2fec/QE/docs/visual_state_coverage_difficulty_stratified_paper_20261004.md
未增加未经核验的新文献、作者、资助或发表承诺。投稿前仍需把两稿合并、验证引用、补作者信息及目标期刊模板。
