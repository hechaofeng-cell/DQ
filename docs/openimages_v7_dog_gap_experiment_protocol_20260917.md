# Open Images V7 犬类覆盖补图与多模型验证方案

版本：v1.0  
日期：2026-09-17  
状态：数据准备完成；本地 VLM 特征标注运行中，完成后自动执行选样及四模型实验；正式模型结果尚未产生

## 1. 研究问题

在一个独立于现有 COCO 实验、犬类目标更充足的数据源上，检验：

> 在相同新增犬类样本预算下，`dog_v3.1` 缺口驱动补图是否比随机补图带来更高的视觉状态覆盖，并在多个预先固定的分类模型上带来更好的识别性能？

本轮不再根据结果筛选模型。模型、数据规模、随机种子、评价指标和固定测试集都在结果产生前冻结。

## 2. 数据源选择

采用 Open Images V7 的目标框子集。选择理由：

- 官方提供 600 类目标框，训练框覆盖约 174 万张图片；
- 同时包含 dog、cat、horse、sheep、person，可复用当前五分类任务；
- 训练、验证、测试为官方独立 split；
- 框标注包含遮挡、截断、群体、图像化目标等属性；
- 可按图片 ID 下载所需子集，不需要下载完整 9M 图片。

本实验只使用真实目标框，排除 `IsGroupOf=1`、`IsDepiction=1` 和 `IsInside=1`。所有类别使用完全相同的裁剪规则。

## 3. 冻结数据对象

| 对象 | 数量 | 作用 |
|---|---:|---|
| `D_base_train` | 6,800 | dog/cat/horse/person 各 1,500，sheep 800 |
| `C_dog_train` | 3,000 | 与基线互斥的犬类候选池 |
| `D_val` | 150 | 官方 validation，每类 30，选择最佳 epoch |
| `D_test` | 500 | 官方 test，每类 100，只做最终评价 |
| `D_difficulty` | 100 | 与主测试集互斥的 dog 困难附加集 |
| `S_random` | 300 | 从候选池按固定种子随机选取 |
| `S_gap` | 300 | 从同一候选池按 `dog_v3.1` 缺口选择 |

三个训练条件为：

```text
base      = D_base_train
random300 = D_base_train union S_random
gap300    = D_base_train union S_gap
```

`random300` 和 `gap300` 的新增数量完全一致，非犬数据、验证集和测试集完全一致。候选池与基线、验证集、测试集按 `image_id` 互斥。

`D_difficulty` 不参与总体 Accuracy 或 Macro-F1。它优先纳入 Open Images test 中面积小于 3% 的 dog 目标，再补足遮挡、截断和普通 dog，用于报告小目标、极小目标、遮挡和截断 Recall。它与主测试集按 `image_id` 互斥，也不参与任何选样或调参。

Open Images 在排除群体框、图像化目标和极小框后，validation 仅有 38 个、train 仅有 939 个可用 sheep 图片目标，因此不能强行采用每类 1,500/200 的统一数量。训练阶段固定使用类别均衡有放回采样，每轮采样 6,800 个目标；验证和测试保持类别平衡。

## 4. 候选选择流程

1. 从官方 train split 生成 1,500 个基线 dog 目标和 3,000 个候选 dog 目标；
2. 为 4,500 个 dog 目标生成原图、A 框图和 A 框裁剪放大图；
3. 使用冻结的 `dog_v3.1` schema 和提示词进行特征标注；
4. 本轮采用自动选样，不进行人工清理；依赖官方 `IsDepiction` 等框过滤和结构化响应校验，保留可能的标注噪声作为研究限制；
5. 仅用 1,500 个基线 dog 计算 `C_before` 和缺口清单；
6. 从同一批 3,000 候选中独立冻结 `S_random` 与 `S_gap`；
7. 计算 `C_random`、`C_gap`，但不查看测试集模型结果来修改选样；
8. 若 `C_gap < 90%`，只扩充训练候选池并记录新版本，不改变 `D_val`、`D_test`、特征协议或模型配置。

## 5. 固定模型

| 模型 | 角色 |
|---|---|
| ResNet-18 | 成熟主基线 |
| ConvNeXt-Tiny | 现代卷积网络 |
| MaxVit-T | 卷积与注意力混合架构 |
| EfficientNetV2-S | 高效缩放卷积网络 |

全部使用 ImageNet-1K 预训练权重，输入为固定目标框裁剪。每个模型在 `base`、`random300`、`gap300` 三个条件上使用相同训练配置与三个固定训练种子。

## 6. 两条指标线

视觉覆盖线：

```text
C_before, C_random, C_gap
Delta C_gap-base
Delta C_gap-random
```

模型性能线：

```text
Accuracy, Macro-Precision, Macro-Recall, Macro-F1
dog Precision, dog Recall, dog F1
遮挡 dog Recall, 截断 dog Recall, 小目标 dog Recall
```

主要比较是 `gap300 - random300`，`gap300 - base` 只说明增加数据后的总变化。`Delta C > 0` 不自动等于 `Delta M > 0`。

## 7. 判定规则

主要指标预先固定为 Macro-F1 和 dog F1。每个模型报告三个训练种子的均值、标准差和逐种子配对差值。

支持方法有效的最低证据标准为：

1. `C_gap > C_random`；
2. 至少三个预注册模型的 Macro-F1 或 dog F1 均值满足 `gap300 > random300`；
3. 提升方向在逐种子结果中具有可重复性；
4. 不隐去下降的模型或指标。

若只出现覆盖率提升而模型没有提升，结论应写为“视觉覆盖改善未转化为当前分类器的稳定性能收益”。

## 8. 执行顺序

```bash
.venv/bin/python scripts/prepare_openimages_v7_experiment.py metadata
.venv/bin/python scripts/prepare_openimages_v7_experiment.py plan
.venv/bin/python scripts/prepare_openimages_v7_experiment.py download --workers 16
.venv/bin/python scripts/prepare_openimages_v7_experiment.py materialize
```

随后运行 `dog_v3.1`、冻结 `S_random/S_gap`，最后开始四模型训练。原始标注、清单、逐样本预测、模型权重、训练日志和校验值均需保留。

完整特征标注与选样可用以下可恢复长任务运行：

```bash
bash scripts/run_openimages_dog_v3_1.sh
```

脚本先标注 1,500 个基线 dog，再标注 3,000 个候选 dog。本轮不进行人工审核；本地 VLM 完成后由续跑脚本自动冻结选样、训练和生成审计报告：

```bash
bash scripts/continue_openimages_v7_experiment.sh
```

如需手动重新执行选样，可运行：

```bash
.venv/bin/python scripts/select_openimages_dog_supplements.py \
  --base-features artifacts/openimages_v7_dog_base_features_v1/results.csv \
  --candidate-features artifacts/openimages_v7_dog_candidate_features_v1/results.csv
```

该命令生成 `base/random300/gap300` 三组训练清单。自动选样可用于本轮探索性验证，但不得表述为已经人工校验的数据集。

冻结三组清单后运行：

```bash
.venv/bin/python scripts/run_openimages_v7_four_model_experiment.py
```

## 9. 结论边界

- 这是已知目标框条件下的五分类区域分类，不是端到端目标检测；
- Open Images 图像许可需按图片逐项保留来源信息并复核，标注许可为 CC BY 4.0；
- 本轮可以检验跨数据源可重复性，但不能证明对所有类别和所有模型普遍有效；
- 不能为了得到“全都提升”的结果继续更换模型或删除负结果。

## 10. 2026-09-17 执行状态

已完成：

- 筛选官方框标注：train 1,095,385 行、validation 19,618 行、test 61,940 行；
- 下载并完整解码 10,550 张唯一原图；
- 物化基线 6,800、候选 dog 3,000、验证 150、主测试 500、困难 dog 测试 100；
- 生成 1,500 条基线 dog 和 3,000 条候选 dog 三视图特征清单；
- 检查所有分区的 `image_id`、原图 SHA-256 和裁剪 SHA-256，跨分区精确重复均为 0；
- `dog_v3.1` 新数据试跑 10/10 成功，无结构化响应错误，平均约 6.04 秒/图；基线 1,500/1,500 已完成。

困难 dog 测试集包含：极小目标 10、小目标（含极小）42、遮挡 36、截断 32；这些属性可重叠。

尚未完成：

- 候选 3,000 个 dog 目标的 `dog_v3.1` 标注；
- 本轮不进行人工清洁；此限制须与结果共同报告；
- `random300/gap300` 正式冻结与覆盖率计算；
- 四模型三条件训练和结果表。
