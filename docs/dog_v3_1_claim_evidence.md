# dog_v3.1 论文主张证据表

| Claim ID | 论文主张 | 证据 | 支持边界 |
|---|---|---|---|
| C1 | 900张图片的 v3.1 特征响应严格成功率为100% | `artifacts/dog500_features_v3_1_20260913/report.json`；`artifacts/dog_candidate400_features_v3_1_20260913/report.json` | 仅适用于当前图片、模型、提示词和协议 |
| C2 | 缺口驱动选择100张后，清洁596张覆盖指数为84.17% | `artifacts/dog_v3_1_combined_supplement_20260913_final/report.json` | 覆盖指数不是识别准确率；清洁集排除4张非真实犬目标 |
| C3 | 补图相对初始500张提升5.53个百分点 | 同上；指标计算脚本 `scripts/evaluate_dog_v3_1_supplement.py` | 单一候选池和预算，未与随机策展做公平对照 |
| C7 | 目标达成率由63.41%提高至74.80%（提升11.39个百分点） | 初始与清洁集 `feature_state_distribution*.csv`；按初始计数冻结阈值后计算 | 这是预设状态数量目标，不是识别准确率；阈值和状态集合不得事后调整 |
| C4 | 视觉属性可作为结构化图像标注对象 | Krishna et al., Visual Genome, 2017 | 支持相关工作背景，不证明本文流程优于该工作 |
| C5 | 覆盖、主动覆盖和稀有切片选择已有研究基础 | Kaushal et al. 2019；Jiang & Rostamizadeh 2021；Kothawade et al. 2022 | 本文的差异是犬类VLM状态缺口和可审计数据闭环，仍需对照实验 |
| C6 | 数据策展需要统一预算和基线比较 | Feuer et al., SELECT, NeurIPS 2024 | 当前论文明确把随机/分层随机对照列为未完成工作 |

## 未支持的强主张

- 不能声称该方法首次提出或没有前人工作；
- 不能声称覆盖指数等于识别准确率；
- 不能声称补图提高了识别模型性能；
- 不能声称596张的全部字段都已人工验证；
- 不能声称 dog 案例证明跨类别泛化。
