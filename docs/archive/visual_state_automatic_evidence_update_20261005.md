# 无人工复核补强：执行结果、限制与后续入口

日期：2026-10-05。当前状态：可独立执行的已有证据补强已完成；新增D/E数据实验未执行。没有新增人工复核，没有重新训练模型，没有发起本轮新VLM标注请求。

## 1 实际完成了什么

| 工作 | 实际结果 | 证据 |
|---|---|---|
| 输入及程序尺度核对 | A/B/C共5400目标通过；未发现框范围、面积、文件或保存哈希不一致 | automatic_audit/audit.json |
| 官方属性对照 | 六类×两来源×两个字段，共24条汇总记录 | automatic_audit/official_attribute_reference.csv |
| 双VLM状态与风险对照 | 共270条类别/字段记录；与真实分类预测的分歧组错误率分开保存 | automatic_audit/dual_vlm_agreement.csv |
| 风险敏感性整理 | B/C原有24项风险变体结果完整汇总，没有重复训练或选取最有利规则 | automatic_audit/existing_risk_sensitivity.csv |
| 混合模型执行 | 两种来源分别使用3000目标×24训练run=72000条唯一图像/run记录；未收敛，不进入确认结论 | statistics/status.json |
| 诊断AUC混合模型 | 执行完成，但有奇异/边界警告，不替代原主检验 | statistics/auc_fit_audits.json |
| 回归测试交付 | 31张唯一图像、99个图像/模型/种子断言，重新推理99/99与冻结预测一致 | regression_replay/audit.json |
| 图表 | 官方参考一致性、双VLM字段一致性、2×3成功/失败/未复现实例面板 | figures/ |

三种子与重复图像具有相关性，72000行不是72000个独立样本。回归测试清单来自已有探索性案例，99/99表示同权重输入的重执行一致，不表示99张都分类正确。

## 2 官方属性对照发现

参考来自历史A中2289个Open Images目标，排除没有同格式参考的111个VOC目标。unknown从可核对分母中单独排除，并报告数量。none归为0、light/moderate/heavy归为1；该规则不核对严重程度等级。

| VLM | 字段 | 已知目标 | 二值参考一致率 | 正类一致率 | 负类一致率 | unknown |
|---|---|---:|---:|---:|---:|---:|
| 本地 | 遮挡 | 2289 | 61.16% | 82.43% | 41.93% | 0 |
| 线上 | 遮挡 | 2284 | 66.07% | 82.10% | 51.58% | 5 |
| 本地 | 截断 | 2289 | 67.50% | 88.12% | 56.41% | 0 |
| 线上 | 截断 | 2287 | 79.62% | 83.25% | 77.67% | 2 |

这里的正类一致率为在官方标记为1的目标中VLM也标记为1的比例；负类一致率为官方0目标中VLM也标记为0的比例。它们不是部署分类任务的Precision/Recall，也不等于全部视觉状态准确率。

结果说明来源、字段与正负状态存在不同程度的分歧。尤其遮挡的负类一致率较低，提示VLM更频繁地给出“存在遮挡”；该现象可能同时涉及字段语义、目标边界解释和自动标签偏差，当前不能确定分歧的全部原因。

论文应将这些值作为构念有效性限制和协议对照证据，不写成“自动标签准确率很高”。官方属性本身的定义与标注来源也不是本研究所有状态的完整真值。[Open Images官方框格式](https://storage.googleapis.com/openimages/web/download_v7.html)

## 3 主结论是否受到影响

分类Accuracy/F1仍由官方类别与模型预测计算；这次自动核对不会改变已保存的预测及原分类指标。受到限制的是“自动状态在多大程度上代表准确视觉条件”的解释。

已有风险敏感性结果显示：删除尺度、可见性、遮挡、截断、朝向中的任一维，或使用预定等权归一化风险后，B/C各来源的六项变体均保持Q5−Q1 F1负向，且各变体24个运行均为负向。变体平均差值范围为：

| 池/标注来源 | 六变体平均Q5−Q1 F1范围（百分点） |
|---|---:|
| B/本地 | −26.95至−21.55 |
| B/线上 | −23.55至−11.50 |
| C/本地 | −17.44至−12.79 |
| C/线上 | −17.27至−9.54 |

这些是既有敏感性结果的整理，不是新外部确认。相应12项Holm校正p均为0.09375，未通过0.05；方向稳定不能改写为所有变体都得到校正后的显著支持，也不能证明自动层级就是人工难度真值。

目前最合适的论点仍是：**自动可见状态协议形成的风险层在已评估的固定模型中呈现稳定的条件性能差异；其语义可靠性具有字段与来源限制。** Coverage方法的额外诊断收益继续按原基线、数据池和来源分别解释。

## 4 统计补充为何没有进入论文的确认结论

当前环境无R，采用statsmodels的变分贝叶斯二项混合模型，对本地/线上风险分别拟合。架构固定，架构×种子run及图像为交叉随机截距；采用稀疏随机设计，避免将图像随机效应构造成巨大稠密矩阵。[实现作者文档](https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.html)

先用60张分层目标、1440条图像/run记录验证了执行接口；小规模拟合通过不代表完整拟合一定可靠。完整两个拟合均达到事先保存的300次迭代上限而未收敛。原始参数与诊断保留，但其优势比、边际差和近似后验区间不作为可信论文结果引用。

AUC的线性混合模型有奇异/边界警告；保留系数与警告，未升级为确认检验。没有为获得更小p值继续换规格、增加迭代后删除原失败，或用补充模型替代八架构主检验。

未收敛估计图单独放在`statistics/unconfirmed_figures/`作为执行诊断，不进入本轮可用论文图表。

## 5 新D/E为何未执行

全历史扫描覆盖1884份相关清单，提取13341个Open Images图像ID。沿用Confidence=1、单目标框资格、非群体框、非depiction、非inside、面积至少0.001等过滤，得到以下sheep数量：

| 已缓存split | 合格唯一目标图像 | 不在任何历史清单中的图像 |
|---|---:|---:|
| train | 939 | 0 |
| validation | 38 | 0 |
| test | 115 | 0 |

因此无法按要求建立六类各400个未用目标的新池。没有改用旧图、放宽条件或删除sheep来凑数量。car初始元数据流完整性未验证，不能把其初步保留数量当全量可用数；后续代码增加了部分HTTP响应检查。

下一步需在新预测之前确定新的数据来源、审计六类映射和未用目标数量，再形成来源修订。Objects365可作为候选审计对象，但官方获取入口要求注册，当前尚未核实六类的实际可下载合格数量，不能承诺已经具备2400张。[官方下载说明](https://www.objects365.org/download.html)

## 6 查看结果与复现

完整产物目录：`/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/`。

- [本轮结果报告](/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/report_zh.md)
- [官方属性参考图](/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/figures/official_reference_agreement.svg)
- [双VLM字段一致性图](/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/figures/dual_vlm_field_agreement.svg)
- [成功、失败与未复现实例](/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/figures/diagnostic_success_failure_counterexamples.svg)
- [回归测试清单](/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/regression_suite.csv)
- [统计诊断状态](/home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/statistics/status.json)

查看后台历史日志：

```bash
tail -n 80 -f /home/hcf/.codex/worktrees/2fec/QE/artifacts/automatic_strengthening_20261005/pipeline.log
```

当前后台已结束本轮可执行阶段，状态是`partial_complete`，不是D/E已完成。后续同权重复评：

```bash
/home/hcf/project/QE/.venv/bin/python /home/hcf/.codex/worktrees/2fec/QE/scripts/replay_visual_regression_suite.py
```

该命令检查固定输入/权重的哈希并重新推理；它不修改权重。所有结果继续保持无本研究人工复核的口径。
