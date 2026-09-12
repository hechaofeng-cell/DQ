# 图片难度模块执行记录

日期：2026-09-07。范围：接入现有基准原图与 GPU 预测，完成审核流程、候选评分及证据检查，不运行新模型推理、分布调整或蜕变测试。

## 实施前决策

待检验命题：固定参考模型错误比例能否为原图提供可解释、可区分并能预测不参与评分模型错误的候选难度。
固定条件：历史十类闭集、历史权重/预处理、基准目录、ResNet-50 与 Swin-T 评分、CLIP 辅助验证、随机种子和每类 5 张随机审核、全部错误/分歧定向审核。
主要检查：标签清晰性、同分数量、辅助模型分区错误率与区间、移除模型的排序相关、类别内表现。B 必须先有独立的温度拟合和校准检查。
停止正式方法推进的条件：来源风险无法排除、标签不充分、错误数不足、校准检查失败或独立验证缺失。届时保留可复算诊断，结论不得升级为正式分组方法。

## 输入

历史数据位于 `data/full_gpu_20260905/`，来源证据位于 `data/source_audit/`。原图直接从历史协议指定 ZIP 读取。模块对历史 `data/` 文件与归档记录运行前后 SHA-256，产物都位于独立的 `artifacts/`。
配置为 `configs/difficulty_baseline_v1.json`。确切权重标识、预处理、类别顺序与提示词见准备产物 `config.json`；Python、代码 SHA-256、运行起止时间和输入哈希见各结果的 `run.json`。

## 已执行命令

下面输出均为本轮真实运行；再次执行需要使用新的输出目录和新的审核输出文件名。

```bash
.venv/bin/python -m qe_quality.difficulty prepare --config configs/difficulty_baseline_v1.json --output artifacts/difficulty/baseline_20260907/prepared
.venv/bin/python scripts/materialize_difficulty_reviews.py --prepared artifacts/difficulty/baseline_20260907/prepared --notes reviews/baseline_v1_visual_notes.json --output reviews/baseline_v1_visual_reviews.csv
.venv/bin/python -m qe_quality.difficulty evaluate --prepared artifacts/difficulty/baseline_20260907/prepared --output artifacts/difficulty/baseline_20260907/inherited_diagnostic_r2
.venv/bin/python -m qe_quality.difficulty evaluate --prepared artifacts/difficulty/baseline_20260907/prepared --reviews reviews/baseline_v1_visual_reviews.csv --output artifacts/difficulty/baseline_20260907/reviewed_r2
.venv/bin/python -m unittest discover -s tests -v
```

视觉审核使用先期同一输入集合的 `artifacts/difficulty/baseline_v1_prepared` 生成图表：

```bash
.venv/bin/python -m qe_quality.difficulty review-sheets --prepared artifacts/difficulty/baseline_v1_prepared --output artifacts/difficulty/baseline_v1_review_sheets
```

已查看其中 sheet_01.jpg 至 sheet_07.jpg 全部 80 项，逐条记录判断和存疑原因。审核 CSV 按最终准备结果的原图 SHA-256 绑定；审核集合必须与队列完全一致。图表只作检查预览，不作为模型输入或新增数据集图片。

## 失败与修复

第一次带部分审核记录的真实输出 `artifacts/difficulty/baseline_20260907/reviewed/` 在 CSV 写入时因未审核行缺少 evidence/reviewed_at 字段失败。修复为所有行初始化相同列，增加 `test_partial_review_overlay_keeps_uniform_csv_schema`，再写入全新 reviewed_r2 目录。
失败目录保留作执行记录，没有完成哈希清单，不是有效结果。先期 `baseline_v1_prepared`、review_sheets 及成功但较早的 inherited_diagnostic 也保留；正式交付引用上述最终 prepared 和两个 r2 结果。

## 验证与决定

19 项测试通过，包含真实历史文件格式的合成端到端流程、已知校准概率恢复、独立校准/验证、冻结后禁止改标签或划分、固定分母、异常数据、标签修正、部分审核、同源冲突、文件保护。
真实数据的自动检查通过，所有 1,000 张均有三模型有效输出；本轮 A 审核候选 55 张，继承标签诊断 920 张，24 张存疑及 1 张范围外无分数；B 为 0 张。

没有在真实数据上执行 freeze 或 validation：样本用途仍为 unassigned，因为没有充足的审核标签、错误数和清楚的来源证据。测试中的冻结/校准使用明确标注的合成数据，不伪装为真实实验。
没有通过研究验收，不开展分组、分布评价和第二阶段。具体证据解释见 `difficulty_validation_report.md`，逐条允许声明见 `difficulty_claim_evidence.md`。
