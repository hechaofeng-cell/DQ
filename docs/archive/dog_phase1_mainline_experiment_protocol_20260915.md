# 第一阶段主流程实验方案：总数据集中的 dog 缺口补图与模型前后评价

版本：v1.0  
日期：2026-09-15  
状态：方案冻结前审阅稿，尚未生成 `D0_train / D0_val / D_test` 正式清单，尚未运行训练实验

## 1. 结论

方案总体可行。第一阶段可以按“一条主流程、两条指标线、一次前后对比”推进：

```text
总数据集 D0
  -> 固定 D0_train / D0_val / D_test
  -> 训练基线模型 M_before
  -> 从 D0_train 提取 Ddog_train
  -> 计算 dog 视觉覆盖 C_before
  -> 从独立训练候选池选择缺口补图 S_gap
  -> 构建 Ddog_train+ 和 D1_train
  -> 用相同配置训练 M_after
  -> 在同一 D_test 上比较 Delta M
```

两条结果线分别为：

```text
视觉覆盖线：C_before -> C_after -> Delta C
模型性能线：M_before -> M_after -> Delta M
```

第一阶段不加入随机补图对照，也不同时比较多种模型。这适合先验证数据、训练、补图和评价链路是否完整可运行。但是，它只能形成补图前后的描述性比较，不能识别“缺口驱动选择相对于随机选择”的因果优势。

## 2. 审阅后必须修正的口径

### 2.1 候选池必须在 D0 之外

若候选 dog 图片已经属于 `D0_train`，把它们再次“合并回总数据集”不会产生新增训练数据。因此必须定义一个与 D0 互斥的训练候选池：

```text
C_train：允许被选择的训练候选池
S_gap subset C_train
S_gap intersect D0 = empty
D1_train = D0_train union S_gap
```

候选池不得包含 `D0_val` 或 `D_test` 中的图片、目标实例及其近重复版本。

### 2.2 D0_val 也必须保持不变

前后模型不仅要共用 `D_test`，还必须共用同一个 `D0_val`。模型选择、early stopping、学习率调度和决策阈值只能使用这个固定验证集。否则 `M_before` 与 `M_after` 可能因为模型选择规则不同而不可比。

### 2.3 D0 应定义为“目标实例数据集”

第一阶段任务是 COCO 目标框裁剪后的多类别区域分类。因此一个样本应由目标实例而不是仅由图片定义：

```text
sample_id = image_id + annotation_id/target_instance_id
input     = 按固定规则裁剪的目标区域
label     = 该目标框的 COCO 类别
group_id  = image_id
```

同一图片中的所有目标实例必须进入同一个 split，禁止把同一原图的不同框拆到训练集和测试集。

### 2.4 这是 oracle 目标框区域分类

测试时使用真实 COCO 框裁剪，意味着模型无需自行定位目标。实验评价的是：

> 在目标位置已知的条件下，区域分类器对目标类别的识别性能。

不能把结果写成端到端目标检测性能。检测任务应作为第二阶段独立实验。

### 2.5 历史覆盖率不能自动代入新主实验

现有 `78.64% -> 84.17%` 来自历史500张初始集与清洁596张发布集：

- 初始覆盖率来源：`artifacts/dog500_features_v3_1_20260913/distribution/report.json`；
- 补图后覆盖率来源：`artifacts/dog_v3_1_combined_supplement_20260913_final/report.json`；
- 初始500张包含后来排除的4张非真实犬，而最终596张已排除它们；
- 两批历史数据实际来自 COCO `train2017`。

因此这组数字可继续作为历史案例结果报告，但新主实验必须在最终冻结的、同一清洁规则下的 `Ddog_train` 与 `Ddog_train+` 上重新计算：

```text
C_before = coverage(Ddog_train)
C_after  = coverage(Ddog_train union S_gap)
Delta C  = C_after - C_before
```

只有当新冻结清单与历史集合完全相同，才可复用历史数值。

### 2.6 测试集可做困难状态标注，但不得反馈到选择过程

小目标可由 COCO 框面积直接计算；遮挡、截断、复杂背景等切片需要额外状态标签。允许在 `D_test` 冻结后为分组评价生成这些标签，但必须满足：

- 不用于计算训练集缺口；
- 不用于候选选图；
- 不用于修改 `dog_v3.1`；
- 不用于调整训练超参数；
- 论文中的困难状态结论优先使用人工复核后的测试标签。

## 3. 冻结的数据对象

正式执行前应生成以下不可覆盖清单：

| 对象 | 定义 | 是否变化 |
|---|---|---|
| `D0_train` | 原始总训练目标实例 | 冻结 |
| `D0_val` | 模型选择与early stopping数据 | 全程不变 |
| `D_test` | 最终评价目标实例 | 全程不变 |
| `Ddog_train` | `D0_train`中标签为dog的目标实例 | 由冻结清单确定 |
| `C_train` | 与D0互斥的候选训练图片/目标 | 选择前冻结 |
| `S_gap` | 从候选池选出的dog补图实例 | 选择后冻结 |
| `Ddog_train+` | `Ddog_train union S_gap` | 补图后版本 |
| `D1_train` | `D0_train union S_gap` | 补图后总训练集 |

每张目标实例至少保留：

```text
sample_id,image_id,annotation_id,target_instance_id,class_id,class_name,
source_split,bbox_xywh,bbox_area_ratio,crop_path,image_sha256,
source_image_sha256,split,dataset_version
```

## 4. 数据划分和泄漏检查

### 4.1 推荐来源

- `D0_train / D0_val`：从 COCO train2017 构建并按 `image_id` 分组划分；
- `D_test`：优先使用从未参与字段设计、补图和调参的 COCO val2017 目标；
- `C_train`：COCO train2017 中未进入 D0 的候选目标。

若采用其他总数据集，仍须满足相同的分组和互斥规则。

### 4.2 必须通过的检查

1. `sample_id` 在每个版本内唯一；
2. `image_id` 不跨 train、val、test；
3. 原图 SHA-256 不跨 train、val、test；
4. 裁剪图 SHA-256 不跨 train、val、test；
5. 使用感知哈希检查近重复图片，并人工裁决命中项；
6. `S_gap` 与 D0、D0_val、D_test 严格互斥；
7. 标签、框坐标和图像尺寸可回查原始 COCO 标注；
8. 修正历史 `dog500/manifest.csv` 中 `source` 写为 val2017、但URL实际指向 train2017 的元数据不一致。

任一检查失败时不得开始正式训练。

## 5. 第一阶段模型任务

### 5.1 任务定义

```text
输入：固定规则生成的COCO目标框裁剪图
输出：一个COCO目标类别
任务：封闭集多类别区域分类
```

所有类别必须使用相同裁剪协议。建议固定：

- 框坐标来源；
- 是否增加上下文padding及其比例；
- 越界处理；
- resize尺寸；
- 插值方式；
- 颜色空间和归一化参数。

dog样本不能使用一种裁剪规则、非dog样本使用另一种规则。

### 5.2 固定模型

第一阶段固定：

```text
MobileNetV3-Large + ImageNet预训练权重
```

选择它的原因是训练成本较低，适合先打通完整闭环。第一阶段不同时运行ResNet50、ConvNeXt或ViT。

### 5.3 固定训练协议

配置文件必须记录并冻结：

- 权重文件及其校验值；
- 输入尺寸；
- 分类头结构；
- 冻结主干和全量微调的阶段；
- epoch和early stopping规则；
- 优化器、学习率和调度器；
- batch size；
- 数据增强；
- 类别权重或采样策略；
- 随机种子；
- 验证指标和最佳权重选择规则；
- 软件、CUDA、GPU与依赖版本。

`M_before`和`M_after`唯一允许不同的是训练数据清单。新增dog数据会改变类别分布，因此类别权重或采样器必须按预先写明的固定规则计算，不能看见结果后修改。

## 6. 一条主流程

### 阶段A：冻结D0并训练基线

1. 生成 `D0_train.csv`、`D0_val.csv`、`D_test.csv`；
2. 运行泄漏和近重复检查；
3. 冻结模型配置；
4. 训练 `M_before`；
5. 保存最佳权重、逐样本预测、训练日志和混淆矩阵；
6. 只在固定 `D_test` 上生成最终基线指标。

阶段A回答：

> 原始总训练集得到的区域分类模型，在固定测试集上的表现如何？

### 阶段B：提取dog训练子集并冻结C_before

1. 从 `D0_train` 按真实类别提取 `Ddog_train`；
2. 对每张dog目标生成原图、A框图和裁剪图；
3. 使用冻结的 `dog_v3.1` schema和提示词运行特征标注；
4. 严格验证JSON、枚举、多选和依赖规则；
5. 在人工清洁后计算状态分布和 `C_before`；
6. 冻结缺失、稀疏、偏少状态清单。

冻结的历史协议文件及SHA-256：

```text
configs/dog_feature_schema_v3_1.json
41256cbdcbf3322d531ce8a758ae2666afa33d9e6e7c387b5e027e47b0498bee

prompts/dog_feature_annotation_v3_1.txt
3363ec6ff61934002746c91e6155b8deb05c2e087ce91c426b466a0e9914a30e
```

若正式实验继续使用该协议，不得在看到候选池或测试结果后修改协议内容。

### 阶段C：缺口驱动补图

1. 冻结与D0互斥的 `C_train`；
2. 仅根据 `Ddog_train` 的缺口和候选训练图片信息选图；
3. 保存每张候选的选择分数、命中状态和选择原因；
4. 人工排除非真实犬、错误框、损坏图和明显错误标签；
5. 对 `S_gap` 做精确重复和近重复检查；
6. 冻结最终 `S_gap.csv`；
7. 计算 `C_after` 和 `Delta C`。

阶段C只回答：

> 缺口补图后，dog训练子集的既定视觉状态覆盖指标是否提高？

### 阶段D：合并总训练集并复训

```text
Ddog_train+ = Ddog_train union S_gap
D1_train    = D0_train union S_gap
D0_val      = 不变
D_test      = 不变
```

使用与阶段A完全相同的模型配置训练 `M_after`。保存与基线完全对应的权重、逐样本预测、日志和混淆矩阵。

### 阶段E：前后比较

在同一个 `D_test` 上比较 `M_before` 与 `M_after`。比较必须基于同一批 `sample_id`，不能因模型失败或无预测而改变分母。

## 7. 两条指标线

### 7.1 视觉覆盖线

沿用冻结定义：

```text
C(D) = mean(min(n_state / 30, 1))
```

计算时排除 `unknown` 和 `other`，状态集合和分母固定。报告：

| 指标 | 补图前 | 补图后 | 变化 |
|---|---:|---:|---:|
| dog视觉状态覆盖指数 | `C_before` | `C_after` | `Delta C` |
| 完全缺失状态数 | 待计算 | 待计算 | 待计算 |
| 稀疏状态数 | 待计算 | 待计算 | 待计算 |
| 偏少状态数 | 待计算 | 待计算 | 待计算 |

历史案例的78.64%、84.17%和+5.53个百分点应单列为“既有案例结果”，在正式清单复算完成前不要预填到新主实验表。

### 7.2 模型性能线

主指标冻结为：

```text
Primary endpoint = Macro-F1 on D_test
```

关键次指标为dog Recall和困难dog状态Recall。完整结果表：

| 指标 | M_before | M_after | Delta M |
|---|---:|---:|---:|
| Top-1 Accuracy | 待运行 | 待运行 | 待计算 |
| Macro-F1 | 待运行 | 待运行 | 待计算 |
| Macro-Precision | 待运行 | 待运行 | 待计算 |
| Macro-Recall | 待运行 | 待运行 | 待计算 |
| dog Recall | 待运行 | 待运行 | 待计算 |
| dog F1（one-vs-rest） | 待运行 | 待运行 | 待计算 |
| 非dog类别平均Recall | 待运行 | 待运行 | 待计算 |
| 小目标dog Recall | 待运行 | 待运行 | 待计算 |
| 遮挡dog Recall | 待运行 | 待运行 | 待计算 |
| 截断dog Recall | 待运行 | 待运行 | 待计算 |
| 复杂背景dog Recall | 待运行 | 待运行 | 待计算 |

每个指标同时报告分子、分母和95%置信区间。困难切片样本量过小时只报告描述性结果，不作显著性结论。

## 8. 一次前后对比的解释边界

第一阶段的可检验假设为：

```text
H1a: Delta C > 0
H1b: M_after在固定D_test上的Macro-F1或dog困难状态Recall高于M_before
```

可能结果及允许表述：

| 结果 | 允许表述 |
|---|---|
| `Delta C > 0`，`Delta M > 0` | 补图后覆盖与模型性能同时改善；二者在本次前后实验中伴随出现 |
| `Delta C > 0`，总体近似不变，困难dog Recall提高 | 覆盖改善主要伴随困难dog状态收益，整体收益有限 |
| `Delta C > 0`，`Delta M`近似为0 | 覆盖改善未自动转化为当前模型的可测性能收益 |
| `Delta C > 0`，`Delta M < 0` | 补图改变了训练分布但损害当前评价性能，需要检查类别不平衡和灾难性偏移 |

无论结果如何，第一阶段都不能写：

- 缺口驱动补图优于随机补图；
- 覆盖提升导致了模型性能提升；
- 当前区域分类结果代表目标检测性能；
- dog实验已经证明跨类别泛化。

没有随机对照时，`M_after - M_before` 同时包含新增样本数量、类别比例、具体图像内容和覆盖变化的共同影响，无法单独归因于覆盖驱动策略。

## 9. 运行稳定性与最小实验规模

为了先打通主线，可以按以下两步执行：

1. 工程冒烟：一个固定种子，完成 `M_before -> M_after` 全流程；
2. 论文结果：在配置不变后至少运行3个训练种子，报告均值、标准差，并在相同测试样本上做配对bootstrap。

多个训练种子仍然是同一个模型和同一主流程，不属于多模型比较。若第一阶段资源只允许一个种子，结果必须标为流程验证或初步结果，不能声称稳定增益。

## 10. 必须保存的产物

建议使用不覆盖的版本目录：

```text
artifacts/dog_mainline_phase1_<date>/
  protocol.json
  data/
    D0_train.csv
    D0_val.csv
    D_test.csv
    Ddog_train.csv
    candidate_train.csv
    S_gap.csv
    D1_train.csv
    overlap_audit.json
  coverage_before/
    results.csv
    feature_state_distribution.csv
    gaps.csv
    report.json
  coverage_after/
    results.csv
    feature_state_distribution.csv
    gaps.csv
    report.json
  model_before/
    config.json
    best.pt
    train_log.csv
    predictions_test.csv
    metrics.json
  model_after/
    config.json
    best.pt
    train_log.csv
    predictions_test.csv
    metrics.json
  comparison/
    metric_delta.csv
    slice_metrics.csv
    confusion_matrices/
    report.md
  checksums.csv
  FROZEN.json
```

`protocol.json` 至少记录：代码提交、数据清单SHA-256、schema/prompt SHA-256、模型权重标识、种子、训练配置、指标定义、失败处理和时间戳。

## 11. 阶段门槛

### Gate 1：允许训练M_before

- D0的任务、类别集合和来源已明确；
- train/val/test清单已冻结；
- 泄漏与近重复检查通过；
- 所有类别使用同一目标裁剪协议；
- 模型和训练配置已冻结。

### Gate 2：允许进行缺口补图

- `Ddog_train`来自冻结的 `D0_train`；
- 非真实犬和错误目标已先清洁；
- `C_before`、状态集合和缺口清单已冻结；
- `C_train`与D0、val、test互斥；
- 测试集未参与缺口或选图。

### Gate 3：允许训练M_after

- `S_gap`已人工复核并冻结；
- `D1_train = D0_train union S_gap`可复算；
- val/test未变化；
- 配置diff确认除训练清单外没有实验因素改变。

### Gate 4：允许形成论文结果

- `M_before`和`M_after`对同一测试样本均有预测；
- 所有指标可从逐样本预测复算；
- `Delta C`和`Delta M`均有原始证据路径；
- 结果表包含样本量、分母、不确定性和失败情况；
- 结论未超出前后比较的证据范围。

## 12. 当前状态审计

已存在：

- `dog_v3.1` schema与提示词；
- 历史500张初始dog标注；
- 历史400张候选标注；
- 历史100张缺口驱动补图；
- 历史清洁596张覆盖结果；
- 在线VLM的dog正样本召回对照。

尚不存在或尚未冻结：

- 作为本实验对象的总数据集D0定义；
- `D0_train / D0_val / D_test`正式目标实例清单；
- 总数据集的类别集合和类别数量；
- 与D0互斥的正式候选池；
- 全类别统一目标框裁剪协议；
- MobileNetV3-Large训练配置；
- `M_before`和`M_after`权重及逐样本预测；
- 测试集困难dog状态的独立复核标签。

因此，当前不能直接从“训练M_before”开始。必须先完成D0任务定义和三份冻结清单。

## 13. 下一步执行顺序

第一批只做以下三项：

1. 明确D0的数据来源、类别集合和每类目标实例数量；
2. 按 `image_id` 分组生成并冻结 `D0_train / D0_val / D_test`；
3. 生成泄漏、哈希和近重复审计报告。

上述三项通过 Gate 1 后，再实现统一目标裁剪和MobileNetV3-Large基线训练。不要在数据定义尚未冻结时继续补图或调整 `dog_v3.1`。

## 14. 第一阶段论文表述模板

在仅完成主流程时，可使用：

> 在固定总数据集划分、固定目标区域分类任务和固定dog视觉特征协议下，本研究从总训练集中提取dog目标，依据训练子集的视觉状态缺口补充dog训练图片，并将其合并回总训练集。研究分别报告dog训练子集的覆盖变化，以及同一模型在固定测试集上的前后性能变化。该实验用于验证覆盖驱动补图闭环及观察伴随的模型变化，不用于证明该策略优于随机补图。

若 `Delta C > 0` 但模型结果未提高，应直接报告负面或混合结果，不更换指标、测试集或训练设置来追求正向结论。

