# 自动类别与特征流水线

`qe_quality.auto_feature` 将目标标注、类别判断、schema 生成和特征识别串成一个可恢复流程。首版策略适用于 `person`，但候选类别和 schema 元规则允许后续扩展到其他目标类别。

## 输入

使用 `target_manifest.csv`，每行一个已有目标实例：

```text
image_id,target_instance_id,image_path,bbox,marked_image_path,crop_image_path,target_class
```

`target_class` 只用于评估 `classification_correct`，不会进入类别判断或 schema 生成提示词。

## 运行

```bash
.venv/bin/python scripts/run_auto_feature_pipeline.py \
  --manifest artifacts/person/target_manifest.csv \
  --output artifacts/person/auto_feature_v1
```

使用 `--resume` 会复用已经通过校验的类别结果、冻结 schema 和特征结果。每次新协议或新模型使用新的输出目录。

## 产物

- `category_results.json` / `category_summary.json`：独立类别判断和整批一致性结果；
- `schema_proposal.json`：模型原始 schema 提议；
- `schema_validation.json`：校验状态和重生成次数；
- `schema_frozen.json` / `schema_fingerprint.txt`：整批复用的冻结 schema；
- `raw/`、`parsed/`：每次 VLM 响应和严格解析结果；
- `results.csv`：类别、特征、证据引用的 fingerprint 和程序计算几何字段；
- `errors.json`、`pipeline_status.json`、`report.json`：失败记录、运行状态和汇总。

schema 生成失败、类别不一致或特征字段非法时，流程不会把错误静默转换成 `unknown`。
