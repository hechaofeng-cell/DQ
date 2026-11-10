# 面向视觉状态覆盖的犬类图像数据集构建与缺口驱动补图

**阶段性论文初稿（dog v3.1）**

## 摘要

现有目标检测数据集通常以类别数量、检测框和总体规模描述数据质量，难以回答一个类别内部是否覆盖了足够丰富的视觉状态。本文提出一套面向视觉状态覆盖的数据集构建流程：首先使用 COCO 犬类图像和程序固定的 A 目标，利用本地视觉语言模型（VLM）输出有限枚举的犬类视觉特征；随后通过严格的 JSON、枚举、多选数组和字段依赖校验，统计每个视觉状态的分布；最后根据低频状态从候选池中选择补图，并冻结清洁版本。我们在500张初始犬类图片和400张候选图片上运行 `dog_v3.1` 协议，900张特征响应均通过严格格式校验。基于45个低频状态，选择100张候选图片后，清洁596张数据集的状态覆盖指数由78.64%提高到84.17%，提升5.53个百分点。该结果支持“缺口驱动选择可以改善当前候选池内的特征状态覆盖”，但不等同于识别准确率提升，也不证明对其他类别具有泛化性。

**关键词：** 数据集构建；数据策展；视觉语言模型；属性标注；长尾状态；主动覆盖；犬类视觉

## 1 引言

COCO 为犬类目标提供了类别和边界框，但同一类别内部仍可能存在颜色、毛发、姿态、朝向、遮挡和场景等长尾状态。仅统计“dog”类别的图片数量，无法判断这些状态是否被充分覆盖。Visual Genome 已经证明，区域级对象、属性和关系可以作为结构化视觉数据的一等标注对象[Visual Genome](https://link.springer.com/article/10.1007/s11263-016-0981-7)。另一方面，数据子集选择和主动覆盖研究表明，coverage、diversity 和稀有切片可以用于数据获取[Learning From Less Data](https://arxiv.org/abs/1901.01151)；Active Covering 将覆盖目标形式化为查询策略[Active Covering](https://proceedings.mlr.press/v139/jiang21i.html)，TALISMAN 则针对稀有数据切片进行目标化采样[TALISMAN](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4820_ECCV_2022_paper.php)。SELECT 进一步表明，数据策展策略本身需要在统一预算和基线下比较[SELECT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f6de5ed486b1802874fe9c70b50277e4-Abstract.html)。

本文关注一个更具体的数据构建问题：在固定类别、固定目标和固定补图预算下，如何使用 VLM 生成的结构化视觉状态发现数据缺口，并从候选池中选择互补图片。本文的贡献是一个可审计的工程—评价闭环，而不是新的视觉编码器或主动学习理论。

本文贡献如下：

1. 定义了一个固定目标、三视图输入和23项有限枚举特征的 `dog_v3.1` 协议；
2. 实现了包含原始响应、严格校验、unknown/非法值分离、哈希和版本冻结的数据流水线；
3. 提出以低频视觉状态为目标的候选选择流程，并在犬类案例中报告补图前后的状态覆盖变化；
4. 明确区分覆盖指数、格式成功率与识别性能，给出当前证据边界和未完成验证。

## 2 相关工作

### 2.1 结构化视觉属性标注

Visual Genome 收集对象、属性、关系和区域描述，并将视觉概念绑定到图像区域，为结构化场景理解提供数据基础[Visual Genome](https://link.springer.com/article/10.1007/s11263-016-0981-7)。本文沿用“对象属性需要显式结构化”的思想，但将对象范围固定为一个 COCO dog A 目标，并把属性集合冻结为有限枚举，以支持状态计数和缺口分析。

### 2.2 数据子集选择与稀有切片

已有数据子集选择方法使用多样性、覆盖和代表性减少冗余[Learning From Less Data](https://arxiv.org/abs/1901.01151)。Active Covering 研究在未标注池中选择查询样本以覆盖正例[Active Covering](https://proceedings.mlr.press/v139/jiang21i.html)。TALISMAN 专门针对稀有类别或稀有切片，并使用区域特征进行目标化获取[TALISMAN](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4820_ECCV_2022_paper.php)。本文的差异在于：选择目标不是类别正例、嵌入空间多样性或检测不确定性，而是 VLM 特征字典中的低频状态。

### 2.3 数据策展基准

SELECT 系统比较多种图像数据策展策略，并强调需要统一预算和可比较基线[SELECT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f6de5ed486b1802874fe9c70b50277e4-Abstract.html)。本文当前实验只展示了一个缺口驱动策略及其案例结果，尚未完成随机、分层随机和贪心方法的公平对照，因此不声称该选择策略普遍优于其他策展方法。

## 3 方法

### 3.1 问题定义

设初始数据集为 $D_0$，候选池为 $C$，每张图片由 COCO 标注固定一个目标实例 $A$。对每个目标输出23项特征 $f=(f_1,ldots,f_{23})$。对特征 $j$ 的状态 $s$，记计数为 $n_{j,s}$。本文的覆盖指数定义为：

$$
I(D)=\frac{1}{|\mathcal S|}\sum_{(j,s)\in\mathcal S}\min\left(\frac{n_{j,s}}{30},1\right),
$$

其中 $\mathcal S$ 排除 `unknown` 和 `other` 状态。状态数量少于10、10至29和至少30分别记为 sparse、weak 和 covered；数量为0记为 missing。

### 3.2 固定目标与三视图输入

每张图片由程序读取 COCO 标注，选择面积最大的 dog 框作为 A 目标。模型接收原图、A框标记图和未标记裁剪放大图。目标框、面积比例、中心位置和边界截断由程序计算，模型不负责重新选择目标。

### 3.3 dog_v3.1 特征协议

特征分为五组：毛发3项、头部和面部8项、身体6项、遮挡与截断5项、场景1项。特征包含毛发颜色/图案/质地，头部、耳朵、口鼻、嘴部、眼睛、舌头和头部朝向，身体外形/朝向/腿部可见数量/尾巴/姿态动作，遮挡与截断程度及部位，犬体可识别性和场景。

眼睛状态取值包括双眼睁开、双眼闭合、单眼睁开、单眼闭合、状态不一致、不可见和未知。尾巴不可观察时，尾巴姿态必须为未知。遮挡部位和截断部位是互斥约束的数组。姿态动作状态只输出一个值，并使用预先冻结的动作优先级。

### 3.4 严格校验与版本治理

模型响应必须满足 schema、字段完整性、枚举值、多选数组、证据一一对应、目标ID和依赖规则。非法枚举不会自动转换为 `unknown`；程序保留原始响应并将该图片标记为失败。`dog_v3.1` 是独立协议，不能与 v2 结果无条件混合。

### 3.5 缺口驱动候选选择

首先在 $D_0$ 上计算缺口。对候选图片，根据其可覆盖的低频状态计算贡献分数，在固定100张预算下选择图片，并保持 tiny_target、small_target、multiple_dogs、boundary_target、person_cooccurrence 五类分层每类15至30张。已知非真实犬表现形式从正式清洁集排除。

## 4 实验设置

### 4.1 数据和实现

初始集来自 COCO dog500，候选池为独立的400张候选图片。模型为本地 `qwen3-vl:30b-a3b-instruct`，通过 Ollama API、GPU后台运行，温度为0，上下文长度为8192，最大输出长度为2200。三种输入图像处理协议在两批数据中保持一致。

### 4.2 评价指标

- **严格特征成功率：** `feature_status=ok` 的图片数除以图片总数；
- **覆盖指数：** 上述状态覆盖公式；
- **unknown率：** 某字段 unknown 数除以图片数；
- **缺口数：** 未达到预设状态目标的状态数。

严格特征成功率是格式和协议指标，不是识别准确率。本文没有将新增补图的独立分类结果纳入评价。

## 5 实验结果

### 5.1 v3.1标注成功率

| 数据集 | 图片数 | 严格成功 | 失败 | 严格成功率 |
|---|---:|---:|---:|---:|
| 初始数据集 | 500 | 500 | 0 | 100% |
| 候选池 | 400 | 400 | 0 | 100% |
| 合计 | 900 | 900 | 0 | 100% |

v3.1 初始23张格式失败在规则和提示词修订后重新运行，最终全部通过。该结果证明协议在当前模型、输入和数据范围内可执行，不证明所有特征值都是真实标签。

### 5.2 缺口与补图

500张初始集包含123个可评价状态，覆盖指数为78.64%，其中45个状态低于30张：9个缺失、10个稀疏、26个偏少。400张候选池中排除4张已知非真实犬后得到396张有效候选，按缺口贡献和分层约束选择100张。

| 指标 | 初始500张 | 补图后清洁596张 |
|---|---:|---:|
| 图片数量 | 500 | 596 |
| 严格特征成功率 | 100% | 100% |
| 覆盖指数 | 78.64% | 84.17% |
| 低频状态 | 45 | 33 |
| 完全缺失状态 | 9 | 7 |
| 稀疏状态 | 10 | 10 |
| 偏少状态 | 26 | 16 |

补图使覆盖指数提高5.53个百分点，关闭14个低频状态。剩余33个状态在当前候选池和预算下未达到目标，包含红色毛发、条纹或花斑图案、混合眼睛状态、夹尾、吠叫、多场景等长尾状态。

### 5.3 场景扩展结果

v3.1 将场景细分为 `indoor_other`、`outdoor_other` 和 `outdoor_water`，避免把明确的室外水域或一般室外场景混为不确定。初始500张中，`outdoor_other` 为53张、`indoor_other` 为47张、`outdoor_water` 为36张。这些值表示已知场景但无法归入更具体类别，不等同于 `unknown`。

## 6 讨论

结果支持一个有限结论：在固定犬类数据、固定 v3.1 特征字典和当前400张候选池内，缺口驱动选择能够提高视觉状态覆盖。该结论与已有覆盖和稀有切片选择工作的方向一致，但本文尚未通过随机基线证明选择策略的相对优势。

覆盖指数的提升不能解释为 dog 识别率提升。当前实验未对新增100张运行独立分类VLM，也没有训练同一识别模型比较补图前后的性能。因此，本文把补图结果定位为数据覆盖改善，而不是模型性能改善。

## 7 局限性与可复现性

1. 23项特征由本地VLM产生，人工审核只覆盖部分样本和部分字段，不能视为全量人工真值；
2. `tail_posture`、`mouth_state` 和 `tongue_visibility` 的 unknown率受可见性影响，unknown不一定表示模型失败；
3. 选择算法只在一个候选池和一个预算下测试，尚未与随机、分层随机和其他贪心方法比较；
4. 500张初始数据参与了字段和缺口设计，不能将其原始 validation 划分描述为完全未使用的独立测试集；
5. 当前案例只有 dog，不能据此证明跨类别泛化；
6. COCO原图的许可和再分发边界需要按照数据集原始条款单独核对。

代码、schema、提示词和结果目录已保存在项目中：

- `configs/dog_feature_schema_v3_1.json`；
- `prompts/dog_feature_annotation_v3_1.txt`；
- `artifacts/dog500_features_v3_1_20260913`；
- `artifacts/dog_candidate400_features_v3_1_20260913`；
- `artifacts/dog_v3_1_supplement_selection_20260913`；
- `artifacts/dog_v3_1_combined_supplement_20260913_final`。

整体流程图见：

![dog v3.1流程图](/home/hcf/project/QE/output/imagegen/dog_v3_1_workflow_zh_polished.png)

## 8 结论

本文构建了一个面向视觉状态覆盖的犬类图像数据集流程。通过固定A目标、三视图VLM输入、23项有限枚举特征、严格校验和缺口驱动选图，900张图片均产生了通过协议校验的 v3.1 特征响应。选择100张补图后，清洁596张数据集的覆盖指数从78.64%提升到84.17%。

该结果表明，缺口驱动补图可以作为犬类数据集构建中的可审计步骤。当前证据仍限于覆盖改善，尚不足以支持识别性能提升、全量标签正确或跨类别泛化等更强结论。下一步应进行随机策展对照、双人一致性评估和新增图片的独立分类验证。

## 参考文献

1. Krishna, R. et al. **Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations.** International Journal of Computer Vision, 2017. [论文链接](https://link.springer.com/article/10.1007/s11263-016-0981-7)
2. Kaushal, V. et al. **Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision.** arXiv:1901.01151, 2019. [论文链接](https://arxiv.org/abs/1901.01151)
3. Jiang, H. and Rostamizadeh, A. **Active Covering.** ICML, PMLR 139, 2021. [论文链接](https://proceedings.mlr.press/v139/jiang21i.html)
4. Kothawade, S. et al. **TALISMAN: Targeted Active Learning for Object Detection with Rare Classes and Slices Using Submodular Mutual Information.** ECCV, 2022. [论文链接](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4820_ECCV_2022_paper.php)
5. Feuer, B. et al. **SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Classification.** NeurIPS 2024 Datasets and Benchmarks Track. [论文链接](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f6de5ed486b1802874fe9c70b50277e4-Abstract.html)
