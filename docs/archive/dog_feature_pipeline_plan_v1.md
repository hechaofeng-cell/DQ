# 犬类视觉特征标注与独立识别方案 v1

本方案用于 `data/coco2017/dog500` 的 500 张犬类图片。每张图固定选择面积最大的 `dog` 框作为 A 目标，生成原图、A 框标记图和未标记原图裁剪放大图。COCO/YOLO 类别编号 16 为 dog；真实标签只用于目标选择和程序验收，不发送给独立分类模型。

## 可行性评价

方案可行，但必须区分三个实验：特征填写模型知道目标类别为 dog；独立分类 VLM 不知道真实类别；程序使用 COCO 目标类别计算正确性。第一版不应把特征覆盖率解释成识别准确率，也不应把可见性分数直接称为识别难度。当前仓库已有的数据目录历史清单标注为 `COCO val2017`，但 URL/方案文本出现 `train2017`，正式实验前必须以实际下载来源重新核对并固化，不可混写。

## 已固化协议

- 特征字典：`configs/dog_feature_schema_v1.json`，20 个字段，人工审核后冻结。
- 特征提示词：`prompts/dog_feature_annotation_v1.txt`。
- 独立分类提示词：`prompts/dog_independent_classifier_v1.txt`。
- 数据准备脚本：`scripts/prepare_dog_feature_pipeline.py`。
- 输出目录：`artifacts/dog_feature_pipeline_v1/`。
- 固定划分：calibration 100、development 200、validation 200，随机种子 20260911。

## 运行

```bash
.venv/bin/python scripts/prepare_dog_feature_pipeline.py
```

脚本会严格检查 500 个唯一图像 ID和每张图至少一个 dog 框，选择面积最大框，输出 `target_manifest.csv`、`split_manifest.csv`、`marked/`、`crops/` 和 `prepare_report.json`。它拒绝覆盖已有输出；历史实验必须使用新目录。

## 后续模型阶段

特征 VLM 的输入为三张图和冻结字典，输出必须通过 JSON、字段完整性、枚举值和 ID 校验；格式错误、接口失败、unknown、人工未审核分别记录。独立 VLM 只接收三张图与候选类别列表，分类正确性由程序执行 `predicted_class == target_class`。

## 可见性与统计

可见性评分为 `100 * (1 - O - T)`，O 为其他物体遮挡比例，T 为边界截断比例；无法区分时保存 `visibility_score=null` 和 `visibility_status=unknown`。视角、模糊、对比度和关键部位可见性单独统计。报告必须包含每个特征的覆盖率、unknown 率、人工一致率、各状态分类错误率、拒答/接口/格式错误和验证集结果。

人工审核至少覆盖随机 30 张、目标大小、轮廓/遮挡状态、多狗、边界目标和模型分歧样本；原始审核表不得覆盖。验证集规则冻结后不得继续调参。

## 当前限制

本次代码先完成可复现的数据协议和输入物料生成；特征字典及提示词已固化，但尚未自动调用 LLM/VLM。在线模型配置、API Key 和批量结果落盘模块就绪后才能产生特征覆盖率和独立识别率，不能用准备阶段结果冒充模型实验结果。
