# PASCAL VOC 2012 独立dog状态验证候选池

日期：2026-09-20

## 目的

本候选池用于独立验证在 Open Images V7 中发现的dog风险状态，不参与风险状态发现、模型训练或覆盖排序学习。

预注册的首轮候选状态为：

- `mouth_visibility=not_visible`
- `head_visibility=partial`
- `muzzle_visibility=partial`

在状态标签冻结前，不读取24个分类模型在这些图片上的预测、置信度或错误结果。

## 数据来源与目标规则

- 来源：PASCAL VOC 2012 `trainval`；
- 官方归档：`VOCtrainval_11-May-2012.tar`；
- 归档MD5：`6cd6e144f989b92b3379bac3b3de84fd`；
- 只读取官方 `ImageSets/Main/dog_trainval.txt` 中标记为 `+1` 的图片；
- 每张图片固定面积最大的非困难dog实例作为A目标；
- 不使用分类模型输出选择目标或排除图片。

官方正样本清单包含1,286张dog图片，共有1,515个非困难dog实例。冻结候选池为每图一个目标，共1,286个目标。

## 独立性审计

参考集合包含现有COCO与Open Images原图12,266张。对PASCAL候选执行：

1. SHA-256精确重复检查；
2. 全图64位difference hash近重复检查；
3. dHash汉明距离不大于3的图片进入待复核区，不直接进入候选池。

本次结果：

- 精确重复：0；
- 阈值内近重复：0；
- 图片ID唯一：1,286/1,286；
- SHA-256唯一：1,286/1,286；
- 原图、A框图和裁剪图完整：1,286/1,286。

这里的去重结论限于SHA-256和dHash阈值；正式论文中不应表述为排除了所有语义相似图片。

## 关键文件

- 数据准备脚本：`scripts/prepare_pascal_voc2012_dog_validation.py`
- 冻结特征输入清单：`data/pascal_voc2012_dog_validation_v1/manifests/dog_feature_candidate_manifest.csv`
- 全部目标审计：`data/pascal_voc2012_dog_validation_v1/manifests/all_dog_targets_audit.csv`
- 去重参考索引：`data/pascal_voc2012_dog_validation_v1/audit/reference_image_hashes.csv`
- 构建报告：`data/pascal_voc2012_dog_validation_v1/report.json`
- 数据来源记录：`data/pascal_voc2012_dog_validation_v1/archive_provenance.json`

## 可复现命令

```bash
.venv/bin/python -u scripts/prepare_pascal_voc2012_dog_validation.py all
```

本地VLM冒烟测试已完成1张，结果为`feature_ok=1`、`errors=0`。全池特征标注命令为：

```bash
.venv/bin/python -u scripts/run_dog_local_pipeline.py \
  --input data/pascal_voc2012_dog_validation_v1/manifests/dog_feature_candidate_manifest.csv \
  --schema configs/dog_feature_schema_v3_1.json \
  --feature-prompt prompts/dog_feature_annotation_v3_1.txt \
  --output artifacts/pascal_voc2012_dog_validation_v1/features_auto \
  --stages features \
  --resume
```

## 后续冻结步骤

1. 对1,286张候选运行dog_v3.1；
2. 人工复核三个预注册状态及三项均清晰的对照状态；
3. 在不读取分类错误的条件下构造目标状态组和匹配对照组；
4. 冻结约300张独立验证集；
5. 使用现有24个固定模型推理，不重新训练；
6. 计算风险差、置信区间、跨架构方向一致性和混合效应模型。

