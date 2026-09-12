# 图片难度评分：声明与证据边界

所有数值来自本地原始输出、审核记录及可复算转换；本轮没有引入新的外部文献主张。

| 声明 | 证据 | 状态与允许表述 |
| --- | --- | --- |
| 已对齐 1,000 张原图的三个模型输出 | `artifacts/difficulty/baseline_20260907/prepared/inventory.json`，`predictions.json`，`issues.csv` | supported：3,000 条原图预测有效；不包含蜕变评分 |
| 原图字节、来源及配对可追溯 | 同目录 `samples.csv`、`lineage.csv`、`provenance.json` | partial：当前字节和精确同源已绑定，历史推理字节及近似重复未清除 |
| 已知 35 张训练重合，其余 965 张未知 | 同目录 `inventory.json` 和 `samples.csv` 的来源证据字段；既有 `data/source_audit/old_pilot_audit.csv` 与历史/当前字节核对 | supported：仅限已有审计覆盖，不声称其余无重合 |
| 助手视觉检查 80 张，认可 55，存疑 24，范围外 1 | `reviews/baseline_v1_visual_notes.json`、`reviews/baseline_v1_visual_reviews.csv`，审核图表 `artifacts/difficulty/baseline_v1_review_sheets/` | partial：真实执行的助手视觉判断，不是人类认证或全体标签正确性证明 |
| A 的继承标签诊断有 987 张同为 0 | `artifacts/difficulty/baseline_20260907/inherited_diagnostic_r2/report.json` 的 `analyses.unassigned.diagnostic_a_inherited.discrimination` | supported：98.7% 同分，不能称为 987 张正式“简单”图 |
| 审核后 A 为 49 张 0、6 张 0.5 | `artifacts/difficulty/baseline_20260907/reviewed_r2/scores.csv` 与 `report.json` | supported：仅 55 张审核候选，不外推到总体 |
| CLIP 错误率在两个 A 区间为 7/49、1/6 | 同目录 `report.json` 的 `analyses.unassigned.score_a.independent_models.clip.bands` | partial：带样本量和 Wilson 区间的描述性观察；不证明单调预测价值 |
| A 排序对模型组合敏感 | 同目录 `report.json` 的 `analyses.unassigned.stability_a` | partial：移除各模型的相关 0.904、0.389，仅适用于已审核集合 |
| 类别内尚无充分区分证据 | 同目录 `report.json` 的 `macro_class_auc`、`by_class` | partial：4 类有定义的 CLIP 宏平均 AUC 0.496；其余未定义；未验证跨类有效 |
| B 的真实校准条件不足 | 同目录 `calibration.json`、`report.json.reviewed_availability_by_class`；全部继承标签错误数来自准备目录 `inventory.json` | supported：真实 B 为空，待验证；实现测试不等于方法验证 |
| 工程自动检查通过且历史文件未覆盖 | 同目录 `checks.json`、`run.json`，准备目录 `provenance.json`；`tests/test_difficulty.py` 的复现测试命令见运行记录 | supported：在本轮运行及测试范围内成立 |
| 当前方法可以正式划分易中难 | 上述证据不足 | unsupported：明确写作“当前评分尚不适合正式分组” |

结果方向：A/B 越高表示参考模型识别越困难；独立模型错误率/AUC 越高表示更能预测其错误；NLL、Brier、ECE 越低表示相应校准指标更好；排序相关越高表示该敏感性检查中越稳定。

结论审核：结果可追溯通过；外部引用不适用；声明范围限制为探索性诊断。研究有效性没有通过，不能以工程完成代替。
