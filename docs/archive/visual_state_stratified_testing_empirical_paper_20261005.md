# 基于视觉状态覆盖与风险分层的多类别图像分类模型测试：一项实证研究

Visual-State Coverage and Risk-Stratified Testing of Multiclass Image Classifiers: An Empirical Study

中文论文初稿，2026年10月5日。本文以已完成的确认实验为主要证据；作者、单位及资助信息由作者后续填写。

## 摘要

图像分类模型的总体性能指标不能直接说明测试集包含哪些类别内部视觉条件，也不能定位模型在不同条件下的能力差异。本文从模型测试与诊断角度，研究预定义视觉状态、类别内风险分层和测试选样之间的关系。我们构建有限枚举的可见视觉状态协议，使用本地与线上视觉语言模型分别标注固定目标，并将状态覆盖与视觉风险作为两个不同维度组织测试。在固定六类分类任务、1800个训练目标、八种预训练模型架构及三个训练种子的条件下，冻结全部24份分类器权重，在2400个COCO新目标和600个VOC外部来源目标上进行确认实验。类别内风险评分仅使用目标尺度、身体可见性、遮挡、截断和朝向，不读取被测分类器的预测结果；每个类别划分为五个等数量风险层。

完整风险层评测显示，两种标注来源下的24次运行均出现最高风险层Macro-F1低于最低风险层。在COCO确认池上，本地与线上分层的平均降幅分别为25.50和21.30个百分点；在VOC外部确认池上分别为16.57和14.44个百分点，均通过预定风险检验族的多重检验校正。中间层及累计性能并不普遍单调。在同类别预算、共同风险配额和逐类别等错误数控制下，覆盖选样相对随机选样的诊断收益得到部分支持，但该收益依赖标注来源、数据池、表征和控制方式。更严格的精确风险匹配在VOC上未获得校正后的显著支持，对强基线的普遍优势亦未确立。历史薄弱状态在COCO新样本上具有复现证据，其跨来源复现则受状态支持不足限制。研究结果支持将视觉状态与风险分层用于呈现模型能力差异，并为总体性能评价补充可追踪的状态诊断信息。

**关键词：** 深度学习测试；视觉状态覆盖；风险分层；多类别图像分类；模型能力诊断；视觉语言模型；实证研究

## Abstract

Aggregate classification scores do not directly describe the within-class visual conditions represented in a test set or locate condition-specific weaknesses. We study visual-state coverage and risk-stratified testing using a finite state protocol and two automatic vision-language annotators. Twenty-four classifiers, comprising eight ImageNet-pretrained architectures and three training seeds, are trained on a fixed, balanced six-class dataset and remain frozen throughout testing. Confirmatory evaluations use 2,400 previously unused COCO targets and 600 targets from the separate VOC2007 source. A predefined risk score uses object scale, visibility, occlusion, truncation, and orientation without consulting classifier predictions. Each class is divided into five equally sized risk strata. Under both annotators, all 24 runs exhibit lower Macro-F1 in the highest-risk stratum than in the lowest-risk stratum. Mean decreases are 25.50 and 21.30 percentage points on COCO, and 16.57 and 14.44 points on VOC, with significance retained in the prespecified risk-test family. Intermediate and cumulative curves are not uniformly monotonic. Coverage-guided selection provides conditional diagnostic gains over random selection under common risk quotas and class-specific equal-error control. These gains vary with annotation source, data pool, representation, and matching protocol; exact risk matching does not yield corrected significant gains on VOC. Previously frozen weak states replicate on new COCO samples, whereas external-source state validation is limited by insufficient support. The findings support visual-state risk stratification as a testing instrument for exposing model capability differences and supplying condition-specific diagnostics alongside aggregate performance metrics.

## 1 引言

一个分类器在测试集上取得较高F1，说明它在该测试集合的样本构成下具有较好的平均表现，却不直接回答它是否见过背向目标、严重模糊、局部可见或小尺度目标。当这些条件在测试集中比例很低时，总体指标可能同时混合典型条件下的成功与少数条件下的失败。模型测试因此需要进一步回答：已经测试了哪些视觉条件，不同条件下的性能差异有多大，以及哪些薄弱条件能够在新样本上复现。

深度学习测试已提出网络内部覆盖、基于训练行为差异的充分性度量，以及输入多样性选样等方法。DeepXplore和DeepGauge从内部活动描述测试覆盖，Surprise Adequacy从测试输入与训练行为的差异构造充分性依据。相关实证研究也表明，覆盖指标的增长不能自动等价为更有效的错误揭示。因此，覆盖指标需要与实际测试结果分别评价。[DeepXplore](https://arxiv.org/abs/1705.06640)、[DeepGauge](https://arxiv.org/abs/1803.07519)、[Surprise Adequacy](https://arxiv.org/abs/1808.08444)、[神经元覆盖实证研究](https://fabrice.harel-canada.com/publication/neuron-coverage-eval/neuron-coverage-eval.pdf)。

本文关注能够通过图片直接描述的类别内部状态。我们将“测试条件是否充分出现”与“测试条件是否困难”分开：视觉状态覆盖记录状态出现及证据数量；视觉风险评分用于组织相对困难的测试层。较高覆盖可以来自新增常见条件，也可以来自新增困难条件，因而不能要求覆盖提高时F1必然降低。另一方面，测试集改变后测得F1的下降也不意味着模型本身退化，因为被测权重始终保持不变。

据此，本文的核心论点是：基于预定义可见视觉状态的风险分层测试，可以揭示固定分类器在不同视觉条件下的能力差异，并为总体性能指标提供更细致的诊断信息。我们进一步检验覆盖选样能否在控制样本预算、风险构成与错误数量后提供额外诊断收益，而不将全面优越性作为研究成立的前提。

本文围绕四个研究问题展开。

1. **RQ1：预定义视觉风险分层能否区分固定分类器的测得性能，并在新样本及独立数据来源上复现？**
2. **RQ2：逐步扩展测试集时，状态覆盖、独立新增批次性能和累计性能呈现何种关系？**
3. **RQ3：覆盖驱动选样在共同风险配额、等错误数及精确风险匹配控制下，是否增加失败类型的诊断广度？**
4. **RQ4：模型在高风险条件下的能力和具体薄弱状态有何差异，状态诊断能否在新样本上复现？**

本文的贡献包括三个方面。首先，给出固定目标、共享视觉状态、频次覆盖和类别内风险分层组成的测试组织协议，使测试条件可以明确记录和复算。其次，在固定模型、类别平衡和双自动标注器条件下，提供风险层性能差异及跨来源确认的实证证据。最后，通过共同风险配额、逐类别等错误数和状态迁移分析，区分覆盖选样的条件性诊断收益与尚未获得支持的泛化结论。本文的贡献定位于测试协议与实证诊断，不提出新的分类器或训练增强算法。

## 2 相关工作

### 2.1 覆盖指标与测试充分性

DeepXplore将神经元覆盖与模型差分行为结合用于测试生成；DeepGauge提出多个粒度的测试覆盖标准。两者主要从网络内部活动描述测试执行情况。本文采用外部可见状态，关注测试者能够理解的遮挡、可见性、朝向与场景等条件，而不将外部状态覆盖解释为网络内部逻辑覆盖。[DeepXplore](https://arxiv.org/abs/1705.06640)、[DeepGauge](https://arxiv.org/abs/1803.07519)。

Harel-Canada等在其研究设置下发现，提高神经元覆盖并没有带来更有效的测试集，且可能伴随更少的缺陷揭示和更强的预测偏向。该结果不能直接外推到本文的视觉状态指标，但说明“覆盖提高即测试更好”需要独立验证。本文分别报告状态证据、分类性能和诊断广度，而不以覆盖数值代替有效性评价。[作者原文](https://fabrice.harel-canada.com/publication/neuron-coverage-eval/neuron-coverage-eval.pdf)。

Input Distribution Coverage使用输入分布的潜在表征与特征交互衡量充分性。本文与其共享输入条件覆盖这一问题方向，但当前只统计明确枚举的单字段状态边际覆盖，不计算完整的组合交互覆盖。因此，本文没有证明全部状态组合已经被充分测试。[论文DOI](https://doi.org/10.1145/3576040)、[作者学位论文第5章中的方法说明](https://www.cs.virginia.edu/~soffa/research/dissertations/swaroopa_thesis.pdf)。

### 2.2 测试优先级与输入多样性

Surprise Adequacy依据测试输入相对训练行为的差异组织测试；DeepGini通过分类输出概率的不纯度优先选择可能误分类的样本。Core-set研究从表征几何角度选择数据子集。本文分别使用距离型Surprise Adequacy、DeepGini和冻结ViT表征上的贪心k-center作为对照，比较它们与可解释状态选样的诊断表现。[Surprise Adequacy](https://arxiv.org/abs/1808.08444)、[DeepGini](https://arxiv.org/abs/1903.00661)、[Core-set](https://arxiv.org/abs/1708.00489)。

黑盒输入多样性研究已经评价输入表征多样性与错误揭示的关系。DeepGD进一步同时考虑预测不确定性与不同失败的多样性。因此，发现不同失败类型并非本文首次提出的目标。本文的具体研究边界在于可解释视觉状态、类别内风险组织、双标注来源，以及逐真实类别等错误数的诊断控制。DeepGD尚未纳入本文的直接实验，本文不对其作性能比较。[黑盒多样性研究](https://arxiv.org/abs/2112.12591)、[DeepGD](https://arxiv.org/abs/2303.04878)。

### 2.3 本文的实证定位

本文不以“高风险条件更难”这一预期现象单独作为算法创新。我们检验的是：一组无需分类器输出定义的可见风险条件，是否在多个被测模型和另一数据来源上形成一致的性能差异；在这种风险组织已经存在时，状态覆盖选样还能增加多少诊断信息。通过同时保留正结果、负结果和支持不足的状态，本文提供的是方法适用性的经验依据，而不是通用最优选样结论。

## 3 方法

### 3.1 问题定义与信息边界

被测任务为固定目标区域的六类别分类。类别集合为dog、cat、horse、sheep、person和car。一个样本由源图、预先确定的目标框及官方目标类别构成；分类器输入为目标框裁剪图，输出六类之一。对于新候选图像，目标框由标注程序确定，不由VLM或分类器自行选择。

训练完成后冻结分类器参数。测试阶段改变的是测试目标及其组织方式，不增加训练样本，也不重新训练分类器。对同一图片，线上与本地状态标签改变的是风险层和选样结果，分类器预测保持相同。

流程可概括为：固定目标与模型 → 双VLM状态标注 → 建立状态宇宙与风险层 → 按预算选样 → 分类器评测 → 状态诊断与新池确认。分类真值来自数据集类别标注；VLM生成的是视觉状态描述，不作为分类真值。

### 3.2 双VLM视觉状态标注

本地标注器为`qwen3-vl:30b-a3b-instruct`，线上标注器为`sensenova-6.8-flash-lite`。每个标注请求提供原图、A框标记图及A框放大裁剪图。状态标注提示中包含目标类别，因而属于类别条件下的状态描述，并非隐藏类别的识别实验。

类别原生协议分别保留19项通用对象字段和18项car字段；跨类别主分析映射到14项共享字段，另由程序计算目标尺度。共享字段如下。

| 状态组 | 主分析字段 |
|---|---|
| 可见性与方向 | 身体可见性、身体朝向 |
| 遮挡与截断 | 遮挡程度、遮挡物类型、截断程度 |
| 群体与交互 | 群体关系、交互关系 |
| 场景与成像 | 场景、背景复杂度、模糊程度、照明、目标背景对比 |
| 目标呈现 | 可识别性、视觉呈现形式 |
| 程序字段 | 目标框面积占原图比例对应的尺度状态 |

状态采用有限枚举；不可判断时保留`unknown`。字段、枚举与值类型通过结构校验后进入分析。严格有效响应只证明格式符合协议，不证明语义标签完全正确。本文未取得人工状态真值，两种自动标注结果分别分析，用于检查来源敏感性。

后续自动补强对Open Images框级遮挡/截断属性进行了预定二值归并对照，同时核对输入与程序尺度；这些检查只提供部分字段的参考一致性，不作为全部状态的语义准确率。执行结果及未完成的新池确认见[无人工复核补强记录](/home/hcf/.codex/worktrees/2fec/QE/docs/visual_state_automatic_evidence_update_20261005.md)。

目标尺度来自原始图像中的目标框面积比，不来自放大后的裁剪尺寸。它描述原图的目标尺度条件，不能独立代表裁剪中保留的纹理信息或分类器实际输入的信息量。

### 3.3 视觉状态频次覆盖

一个状态记为“字段—枚举值”对；多值字段允许一个目标激活多个状态。对于类别$c$，以该候选池两种标注来源观测到的非`unknown`状态并集构成状态宇宙$U_c$。在整个预算序列内保持$U_c$不变。

记所选集合$S$中状态$s$的支持数为$n_s(S)$，本文的频次覆盖定义为

$$
C_c(S)=\frac{1}{|U_c|}\sum_{s\in U_c}\min\left(\frac{n_s(S)}{\tau},1\right),\qquad
C(S)=\frac{1}{6}\sum_{c=1}^{6}C_c(S),\qquad \tau=20.
$$

一个状态达到20个目标的支持后，对该状态继续增加样本不再提高其分数。频次覆盖不同于“出现至少一次”的存在覆盖，后者另作为辅助指标。`unknown`不计为已覆盖的状态证据。

该指标相对于当前有限候选池定义，不是现实视觉世界的绝对覆盖率。状态宇宙由全池自动标签确定，属于候选池条件下的测试组织；不代表在完全未知数据流中无需预标注即可选样。不同候选池的可行状态分母不同，本文不把它们的绝对覆盖百分比直接合并比较。

在同一状态宇宙、同一标注来源及累积不删除样本的条件下，该覆盖按定义不会下降。这个性质说明状态证据增加，不足以单独证明新增目标更困难或测试策略更有效。

### 3.4 类别内视觉风险评分与五层划分

风险评分$H$由五个可见维度相加，范围为0–13。评分不读取被测分类器的预测类别、置信度、错误标签或中间表征，也不使用VLM的“可识别性”字段。固定映射如下。

| 风险维度 | 状态到分数的映射 |
|---|---|
| 目标框面积比$a$ | $a\geq0.10$：0；$0.03\leq a<0.10$：1；$0.01\leq a<0.03$：2；$a<0.01$：3 |
| 身体可见性 | full/most：0；partial：2；minimal：3 |
| 遮挡程度 | none：0；light：1；moderate：2；heavy：3 |
| 截断程度 | none：0；light：0.5；moderate：1；heavy：2 |
| 身体朝向 | front/left_side/right_side/three_quarter：0；back：1；overhead_or_unusual：2 |

对于未知维度，将其分数保留为$[0,h_{max}]$区间；主分析采用总风险中点，敏感性分析分别按总风险下界和上界重新划层。每种标注来源在每个类别内按风险分数排序，同分时按样本ID排序，划分为五个等数量层Q1–Q5。

Q5代表当前类别内相对高风险，不是跨类别统一的绝对难度阈值。同分目标可能跨越层边界；某些风险维度也存在相关性。因此，五层用于组织可比较的测试条件，不保证每个更高层目标都比每个较低层目标困难。

### 3.5 覆盖驱动选样与两类风险控制

Coverage在当前可选目标中贪心增加稀缺状态证据。设目标$x$激活的状态集合为$A(x)$，其选样增益为

$$
g(x\mid S)=\sum_{s\in A(x),\,n_s(S)<20}
\left[\log(n_s(S)+2)-\log(n_s(S)+1)\right].
$$

每次选择满足约束且增益最大的目标，随后更新计数；计数跨风险层累计。该策略直接优化凹状态证据增益，不等同于直接最大化错误率，也不是上述线性截断覆盖公式的严格最优化。

本文采用两个不同的控制协议，两者的结果分别报告。

**共同风险配额协议。** 在每个“标注来源—类别—风险层”内部，以$(H中点,未知维数)$组成风险格，按完整层中各格的频数比例通过最大余数法分配选样名额。Coverage、Random、DeepGini、DSA、k-center和删字段Coverage共享相同配额。这是主要方法比较；风险格配额及模型无关排序在确认池分类预测前冻结。

**Coverage所选风险直方图的精确匹配协议。** 先在每层进行不受共同配额限制的Coverage选样，再让随机组精确匹配该选择的$(H中点,未知维数)$联合频数。该协议检验特定Coverage序列相对其自身风险匹配随机组的收益，属于补充分析。它和共同配额协议同时改变了Coverage允许选中的集合，不能把两者差异简单解释为增加控制后的单一因果效应。

随机组均在满足约束的格内无放回抽样，重复1000次。所有协议保持真实类别预算一致。匹配风险分数与未知维数不能控制全部视觉难度，亦不能替代对所有状态组合的匹配。

### 3.6 对照方法

| 方法 | 排序依据 | 所需信息 |
|---|---|---|
| Coverage | 稀缺可见状态的凹增益 | 自动视觉状态 |
| Random | 约束内无放回随机抽样，1000次 | 类别及配额 |
| DeepGini | $1-\sum_k p_k(x)^2$，降序 | 被测分类器输出概率 |
| DSA | 测试表征到预测类别最近训练目标的距离，相对该训练目标到其他类别最近训练目标距离的比值 | 被测分类器表征、训练目标及类别 |
| ViT k-center | 归一化冻结ViT-B/16表征中的贪心最远点 | 独立固定嵌入 |
| Coverage_without_H | 删除五个风险字段及可识别性字段后的Coverage | 剩余九项共享字段 |

k-center跨层保留已有目标距离信息，首点从当前层中心附近的可行目标选取。该实现是几何多样性基线，并非完整重现Core-set主动学习的再训练流程。删字段组用于检查直接风险字段的影响，但剩余字段仍可能关联困难程度，不能称为完全去除难度。

### 3.7 性能与诊断指标

分类性能报告Accuracy、Macro-Precision、Macro-Recall、Macro-F1及每类Recall/F1。对类别$c$，$F1_c=2P_cR_c/(P_c+R_c)$；Macro-F1为六个类别F1的算术平均，不能以平均Precision和平均Recall的调和平均代替。在真实类别等数量的测试集合中，Accuracy与Macro-Recall相等。

独立诊断采用两个指标。**混淆类型广度**记录每个真实类别被误判为哪些其他类别；**错误簇广度**在冻结表征中按真实类别聚类错误目标，作为失败样本表征多样性的代理。对于有$m$个错误的类别，非单例簇数为$\max(2,\min(8,\operatorname{round}(\sqrt m)))$；零错误和单错误分别按既有特例处理。主分析使用各分类器的固定表征，敏感性分析改用所有模型共用的冻结ViT表征。错误簇并不等于独立缺陷根因。

为了避免“错误更多，所以类型更多”的数量解释，两种方法在每个真实类别上匹配错误数，取两组错误数的较小值$q$。若一组含$N$个错误，其中某类型出现$r$次，则在$q$个无放回错误样本中发现该类型的精确期望概率为

$$
1-\frac{\binom{N-r}{q}}{\binom{N}{q}}.
$$

对可行混淆类型或错误簇归一化后进行六类宏平均，再计算预算曲线的归一化梯形面积。面积计算包含零预算原点，按该池最大预算归一化；它不是ROC-AUC，也不是F1提升。等错误数匹配是事后评价控制，不是测试前已经知道错误的在线选择机制。

### 3.8 薄弱状态发现与冻结验证

在历史发现池A中，对于每个“标注来源—架构—类别”，以三个种子的平均Recall筛选支持至少20个目标、状态Recall较类别Recall低至少10个百分点的状态；每类最多保留三项。最终得到175条冻结“状态—模型”组合，本地87条、线上88条。

该清单在COCO确认池B预测前冻结，VOC确认池C直接沿用。新池支持不足20的组合标记为不可评价，不重新挑选更有利状态。主统计以可评价状态的Recall减类别Recall为差值；同时保留逐状态支持量、方向复现及可匹配风险参考。状态关联用于定位薄弱条件，不证明该字段是错误的因果根源。

## 4 实验设置

### 4.1 数据用途与独立性

本文保留训练、验证与测试的用途边界。研究重点是冻结模型后的测试组织，不意味着可以省略训练数据与评测数据的区分。训练集每类300个目标，共1800个，来自COCO及既有COCO dog初始清单；固定验证集90个目标，基础测试集300个目标。

历史发现池A包含2400个目标，主要来自Open Images，并保留111个VOC2012 sheep目标，用于早期探索及薄弱状态发现。本文主要确认结果来自新构建的B和C，避免将旧池的事后分层结果误写为本轮确认结果。

| 数据用途 | 来源与角色 | 每类目标 | 总目标 | 完整单风险层 | 每层新增/类 | 累计测试规模 |
|---|---|---:|---:|---:|---:|---|
| 固定训练集 | COCO训练目标 | 300 | 1800 | — | — | — |
| 固定验证集 | 固定COCO清单 | 15 | 90 | — | — | — |
| 基础测试集 | 固定COCO清单 | 50 | 300 | — | — | — |
| 发现池A | 历史混合来源 | 400 | 2400 | 480 | — | 用于发现与规则形成 |
| 确认池B | 未用COCO train2017目标 | 400 | 2400 | 480 | 20 | 120、240、360、480、600 |
| 外部确认池C | 未用VOC2007 trainval+test目标 | 100 | 600 | 120 | 10 | 60、120、180、240、300 |

COCO提供具有实例标注的场景图像，VOC2007提供其原生目标标注。本文从官方目标标注构建固定框六分类任务，并非复现官方检测或多标签分类评测；VOC数值不能当作官方排行榜结果。[COCO原论文](https://arxiv.org/abs/1405.0312)、[VOC2007官方页面](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html)。

新池按预定哈希次序选择图像，一个源图只保留一个目标，类间图像互斥；同类存在多个框时固定最大目标框。采样不读取分类错误，也不依据实验收益富集困难样本。历史图像ID、图像与裁剪哈希均参与排除；近重复检测采用既有pHash与冻结ViT联合阈值。B属于同来源的新样本确认，C属于不同数据来源确认，二者不能混称。自动重复检查不保证排除全部近重复，也不能排除ImageNet预训练阶段的重叠。

VOC来源由VOC2012改为VOC2007，是因为完整历史排除后原计划的dog数量不足；来源调整发生在新标注及分类预测之前，不依据性能决定。六类统一保持100个正式目标。本轮B和C各自完成全池双标注，严格有效响应分别为每来源2400/2400和600/600；C没有替换正式目标。该成功率仅指协议有效性。

B的最终选样比例为25%，C为50%，两池预算与可行状态数量不同。所有方法在各池内部同预算比较；本文不以跨池绝对覆盖值判断哪一个池更充分。

### 4.2 被测模型与固定训练

八种架构为ResNet-18、ResNet-50、MobileNetV3-Small、MobileNetV3-Large、DenseNet-121、EfficientNet-B0、ConvNeXt-Tiny和Swin-T。每种架构训练三个种子：20260916、20260917和20260918，共24份权重。模型使用ImageNet预训练初始化，替换六分类头并微调整个网络。

统一训练20轮，每轮类别均衡加权抽样3000次，batch size为64，AdamW学习率与weight decay均为$10^{-4}$，采用余弦学习率调度。训练输入包含Resize(256)、224像素随机裁剪、水平翻转、轻度ColorJitter及ImageNet归一化；评测使用相同5%边距目标裁剪、Resize(256)、CenterCrop(224)与归一化。以固定验证集Macro-F1选择checkpoint。

统一配置用于控制实验流程，不代表每种架构已经取得各自最佳调参表现。本轮确认仅加载既有权重，未重新训练，也未根据新池结果调整风险权重、模型配置或状态协议。

### 4.3 三种测试视角

**完整风险层**评测该层全部目标，B每层480个、C每层120个，用于回答风险层是否区分模型表现；它不受层内Coverage选择影响。

**独立新增批次**仅评测该层实际选中的新目标，B每批120个、C每批60个，观察加入该层时的新样本表现。

**累计测试集合**合并此前所有批次，再从完整混淆矩阵重新计算性能。累计F1不是批次F1的算术平均。三个视角分开报告，避免低风险目标在累计集合中的占比掩盖新增高风险批次的能力差异。

### 4.4 统计分析与证据层次

每个架构先对三个种子取平均，再以八个架构为单位枚举符号翻转，进行双侧精确检验。主要置信区间采用5000次架构/种子层级bootstrap，条件于当前候选池与已执行配置。风险差值另报告固定24模型面板下、类别和风险层内部的源图重采样区间，作为样本不确定性的补充；两类区间不代表同一个变动来源。

主要风险检验族包含两种标注来源×三种unknown处理的六项Q5−Q1比较；主要方法族包含两种来源×两个诊断指标的四项Coverage相对共同配额Random比较；冻结状态迁移为两项。强基线比较为12项次要族，删字段为四项补充族；全部20项方法比较统一校正、统一ViT表征、来源交叉及风险变体分别保留。各池分别检验，不将某一来源的显著结果当成另一来源也成立。

八架构双侧精确检验最小p值为$2/2^8=0.0078125$。12项或20项Holm族的第一项门槛小于该值，按当前设计无法启动拒绝。这限制了强基线比较的统计判定能力；未通过校正不能直接解释为不存在效应。固定面板的图像bootstrap区间作为补充估计，不替换预定主检验，也不把三个种子视为新增独立架构。

研究问题源于历史探索，最初风险规则包含事后形成过程；B和C沿用冻结规则、清单和权重执行后续确认。代码及输入哈希审计用于验证执行一致性，不能将整项研究追溯称为完全盲测。

## 5 实验结果

### 5.1 RQ1：视觉风险层能够区分固定模型表现

表1报告完整风险层Macro-F1，数值为24次运行平均。两种标注来源分别定义风险层，对应的图片构成可能不同。

**表1 完整风险层Macro-F1（%）。**

| 风险层 | B本地 | B线上 | C本地 | C线上 |
|---|---:|---:|---:|---:|
| Q1 | 95.11 | 93.95 | 92.45 | 93.21 |
| Q2 | 91.64 | 92.27 | 91.72 | 91.92 |
| Q3 | 89.37 | 86.17 | 93.92 | 89.87 |
| Q4 | 84.21 | 85.02 | 88.98 | 89.08 |
| Q5 | 69.62 | 72.65 | 75.88 | 78.78 |

**表2 最高风险层减最低风险层的Macro-F1差。区间为架构/种子层级bootstrap，单位为百分点。**

| 确认池 | 标注来源 | Q5−Q1 | 95%区间 | 负方向运行 | 六项Holm p |
|---|---|---:|---|---:|---:|
| B | 本地 | −25.50 | [−27.17, −23.94] | 24/24 | 0.04688 |
| B | 线上 | −21.30 | [−22.59, −19.86] | 24/24 | 0.04688 |
| C | 本地 | −16.57 | [−19.47, −13.52] | 24/24 | 0.04688 |
| C | 线上 | −14.44 | [−16.65, −11.97] | 24/24 | 0.04688 |

两池、两来源的Q5−Q1均为负，并通过预定风险族校正。unknown中点、下界与上界分析亦保留该方向。固定模型面板的源图重采样95%区间也均低于零，C本地为[−23.43, −10.69]个百分点，C线上为[−20.37, −8.47]个百分点，支持该差异并非仅依赖于架构聚合方式。

中间层并不普遍单调。C本地Q3平均F1高于Q1与Q2，说明风险评分没有穷尽类别构成、图像质量和模型决策所受到的全部影响。最稳定的确认结果是最高风险层与最低风险层之间的差异，而非每级风险提升都产生固定幅度的下降。

![图1 COCO确认池B完整风险层上的模型Macro-F1。每层480个目标，各模型为三个种子的均值；两面板分别采用本地和线上状态分层。](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/figures/full_stratum_model_f1.png)

![图2 VOC外部确认池C完整风险层上的模型Macro-F1。每层120个目标，模型权重和风险规则沿用B。](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/figures/full_stratum_model_f1.png)

图1和图2中的圆点为实际评测的五个风险层，平滑连接仅用于视觉引导，不表示增加了实验测量点。

### 5.2 RQ2：覆盖增长与性能响应必须分开解释

表3采用层内Coverage与其风险直方图精确匹配随机协议对应的Coverage序列，分别报告累计覆盖与累计Macro-F1。它不是后续共同配额六方法比较中的同一个Coverage集合。

**表3 分层Coverage逐步扩展的累计结果（覆盖与F1均为%）。**

| 池 | 阶段 | 累计目标 | 本地覆盖 | 本地F1 | 线上覆盖 | 线上F1 |
|---|---|---:|---:|---:|---:|---:|
| B | Q1 | 120 | 21.46 | 91.26 | 21.65 | 88.31 |
| B | Q2 | 240 | 37.53 | 88.80 | 38.04 | 86.42 |
| B | Q3 | 360 | 48.80 | 87.01 | 49.92 | 83.97 |
| B | Q4 | 480 | 58.86 | 84.62 | 59.53 | 82.89 |
| B | Q5 | 600 | 68.64 | 78.34 | 68.90 | 78.48 |
| C | Q1 | 60 | 11.48 | 92.11 | 11.56 | 92.15 |
| C | Q2 | 120 | 22.99 | 90.10 | 23.17 | 91.05 |
| C | Q3 | 180 | 32.28 | 90.62 | 32.57 | 89.57 |
| C | Q4 | 240 | 39.43 | 89.71 | 40.43 | 88.08 |
| C | Q5 | 300 | 46.88 | 84.44 | 48.07 | 85.16 |

覆盖在各自固定分母下持续上升，累计F1从起点到终点总体下降。B本地与线上累计F1分别下降12.92和9.82个百分点；C分别下降7.67和6.99个百分点。由于模型未重新训练，这些变化表明测试集合变得不同，不能解释为模型随测试而学习或退化。

独立新增批次更直接显示新增条件的表现。在B中，本地Q1与Q5批次F1分别为91.26%和51.84%，线上为88.31%和60.71%；在C中，本地分别为92.11%和62.98%，线上为92.15%和73.06%。累计集合仍包含早期相对低风险目标，因此终点累计F1高于最后独立批次F1。

局部回升同样存在。例如C本地累计F1从Q2的90.10%升至Q3的90.62%，与此同时覆盖从22.99%升至32.28%。这一观察直接说明覆盖增长与F1下降不是一一对应规律，也没有必要人为要求曲线单调或先增后减。

![图3 VOC确认池C的独立新增批次、累计Macro-F1和累计状态覆盖。Coverage与其精确风险匹配随机对照分别展示；跨池绝对覆盖值不作等价比较。](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/figures/batch_cumulative_and_coverage.png)

### 5.3 RQ3：共同风险配额下存在条件性诊断收益

表4报告Coverage相对Random的归一化预算AUC差，已经逐真实类别匹配错误数。正值表示相同错误预算下发现的混淆类型或错误簇更广，不表示分类器F1提高。

**表4 共同风险配额下Coverage相对Random的等错误数诊断比较。**

| 池 | 来源 | 诊断指标 | AUC差 | 95%区间 | 主四项Holm p | 全20项Holm p |
|---|---|---|---:|---|---:|---:|
| B | 本地 | 混淆类型 | +0.0158 | [+0.0081, +0.0235] | 0.03125 | 0.15625 |
| B | 本地 | 错误簇 | +0.0103 | [+0.0039, +0.0170] | 0.03125 | 0.23438 |
| B | 线上 | 混淆类型 | +0.0224 | [+0.0152, +0.0305] | 0.03125 | 0.15625 |
| B | 线上 | 错误簇 | +0.0070 | [+0.0006, +0.0132] | 0.04688 | 0.51562 |
| C | 本地 | 混淆类型 | +0.0018 | [−0.0051, +0.0081] | 0.98438 | 1.00000 |
| C | 本地 | 错误簇 | +0.0010 | [−0.0063, +0.0076] | 0.98438 | 1.00000 |
| C | 线上 | 混淆类型 | +0.0185 | [+0.0133, +0.0244] | 0.03125 | 0.15625 |
| C | 线上 | 错误簇 | +0.0164 | [+0.0100, +0.0227] | 0.03125 | 0.15625 |

B中四项预定主要比较均获得支持；C中只有线上标注的两项获得支持。本地C的平均收益接近零，区间跨零。该结果说明，额外诊断收益具有标注来源与数据池依赖，不能只根据B结果将其普遍化。

全20项统一校正未通过。这里同时存在检验族范围改变和八架构精确p值分辨率限制；正文保留主要家族及统一敏感性结果，不以较小家族的显著性声明全部方法比较占优。

### 5.4 精确风险匹配与强基线结果限定了收益范围

**表5 Coverage所选风险直方图精确匹配后的随机对照。AUC差仍按逐类别等错误数计算。**

| 池 | 来源 | 指标 | AUC差 | 四项Holm p |
|---|---|---|---:|---:|
| B | 本地 | 混淆类型 | +0.0097 | 0.07031 |
| B | 本地 | 错误簇 | +0.0109 | 0.03125 |
| B | 线上 | 混淆类型 | +0.0181 | 0.03125 |
| B | 线上 | 错误簇 | +0.0108 | 0.03125 |
| C | 本地 | 混淆类型 | +0.0047 | 0.34375 |
| C | 本地 | 错误簇 | +0.0044 | 0.15625 |
| C | 线上 | 混淆类型 | +0.0071 | 0.18750 |
| C | 线上 | 错误簇 | +0.0004 | 0.88281 |

在C上，这四项均未通过校正，因此“Coverage提供稳定独立于风险的诊断收益”尚未得到跨来源确认。部分区间或点估计为正，也不替代预定校正检验。

对DeepGini、DSA和k-center的比较呈现混合方向，未建立普遍优势。例如C本地Coverage相对k-center的混淆类型AUC差为−0.0072，错误簇为−0.0089；C线上相对DeepGini分别为+0.0238和+0.0195，但12项强基线族的校正p均未达到0.05。它们分别表明观察方向与统计支持必须分开，不能以正向均值概括所有设置。

统一ViT表征后，Coverage相对Random的错误簇结果亦未通过主要四项校正：B本地/线上p为0.25781/0.10938，C为0.98438/0.09375。混淆类型不依赖聚类表征，而错误簇结果随表征变化，提示“不同错误”这一代理定义对结论有实质影响。完整强基线、删字段和表征敏感性表均保留在附录材料。

### 5.5 RQ4：高风险能力存在模型差异

表6报告每个架构在Q5的平均F1，先平均三个训练种子。所有架构均展示，不根据结果仅保留有利模型。

**表6 八架构的完整最高风险层Macro-F1（%）。**

| 架构 | B本地 | B线上 | C本地 | C线上 |
|---|---:|---:|---:|---:|
| ResNet-18 | 67.97 | 69.67 | 71.37 | 75.85 |
| ResNet-50 | 70.02 | 72.48 | 77.35 | 80.32 |
| MobileNetV3-Small | 64.67 | 67.95 | 69.09 | 72.26 |
| MobileNetV3-Large | 68.77 | 71.84 | 76.05 | 79.90 |
| DenseNet-121 | 66.13 | 70.21 | 73.25 | 74.60 |
| EfficientNet-B0 | 70.66 | 73.57 | 73.58 | 78.17 |
| ConvNeXt-Tiny | 74.97 | 78.36 | 84.46 | 85.50 |
| Swin-T | 73.73 | 77.14 | 81.89 | 83.61 |

ConvNeXt-Tiny在四个“池—标注来源”设置的Q5平均F1中均最高，Swin-T均居第二。这是当前权重和统一训练配置下的描述性排序，不是模型总体优劣或两两统计优势的结论。

能力评价还应同时考虑高风险性能水平与Q5−Q1降幅。例如C本地ConvNeXt-Tiny的Q5 F1为84.46%，下降8.81个百分点，而ResNet-18为71.37%，下降21.89个百分点。在B本地，Swin-T降幅22.66个百分点略小于ConvNeXt-Tiny的22.73个百分点，但Q5性能仍较低。只按下降幅度排序可能混淆低起点与高风险能力。

### 5.6 冻结薄弱状态的新样本复现与跨来源限制

**表7 原发现池冻结薄弱状态在新池的验证。差值为状态Recall减类别Recall，主比较按运行与架构聚合。**

| 池 | 来源 | 冻结组合 | 支持足够的组合 | 平均Recall差（百分点） | 两项Holm p |
|---|---|---:|---:|---:|---:|
| B | 本地 | 87 | 78 | −21.73 | 0.01562 |
| B | 线上 | 88 | 75 | −20.77 | 0.01562 |
| C | 本地 | 87 | 1 | −23.83 | 1.00000 |
| C | 线上 | 88 | 3 | +1.62 | 1.00000 |

B中，在不重选状态的条件下，冻结薄弱状态具有显著的平均Recall落差，支持状态诊断不仅是旧池上的结果描述。C的主要问题是支持不足：仅四条组合达到至少20个目标门槛。线上三条的平均方向也未保持负向。由此，跨来源状态可迁移性的证据不足；本地单条负向差值不能代替整体复现。

这里的平均差不是将所有状态目标合并后的单一Recall。状态可能重叠，本文按既定“状态—模型”清单及架构聚合规则计算，不将重叠目标当成更多独立样本。

### 5.7 状态诊断提供具体薄弱条件，而非因果根源

表8展示冻结清单在B中的部分验证案例，并包含未复现方向的反例。数值为三个种子的平均Recall，状态支持量是唯一目标数，不能乘以三个种子。

**表8 COCO确认池B中的探索性状态诊断案例。**

| 模型 | 类别与状态 | 标注来源 | 支持量 | 类别Recall（%） | 状态Recall（%） | 差（百分点） |
|---|---|---|---:|---:|---:|---:|
| ResNet-50 | dog，身体背向 | 本地 | 46 | 81.92 | 57.25 | −24.67 |
| ResNet-50 | cat，原图目标尺度small | 程序尺度/本地分析 | 23 | 82.67 | 39.13 | −43.54 |
| EfficientNet-B0 | horse，身体背向 | 本地 | 47 | 82.58 | 60.28 | −22.30 |
| ResNet-50 | sheep，严重模糊 | 线上 | 44 | 88.42 | 57.58 | −30.84 |
| DenseNet-121 | dog，原图目标尺度medium | 程序尺度/本地分析 | 79 | 73.42 | 81.01 | +7.60 |
| EfficientNet-B0 | car，原图目标尺度medium | 程序尺度/本地分析 | 75 | 97.25 | 97.33 | +0.08 |

这些案例说明类别总体Recall未直接呈现的状态差异。例如ResNet-50的dog类别Recall为81.92%，但背向状态只有57.25%；该差异能够为后续测试指定明确条件。dog中等尺度状态在新池却高于类别平均，car中等尺度状态接近类别平均，说明旧薄弱条件并不总是复现。

背向目标可能减少可见头部与面部线索，严重模糊可能减少纹理信息，但这些是与观察相容的解释，不是已被实验隔离的失败原因。状态与遮挡、尺度、背景等条件可能共现，同状态中也存在正确目标。案例集同时保存同状态正确样本、高风险成功样本和双VLM分歧，避免只展示失败图片形成选择性解释。

![图4 COCO确认池B的六类别风险层Recall。该图补充总体Macro-F1，呈现不同类别的风险响应。](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/figures/class_risk_recall.png)

### 5.8 标注成本与复用

**表9 状态标注的请求成本记录。token为已保存响应中可取得usage的累计量，包含有usage的重试响应。**

| 池 | 标注器 | 成功目标 | 已保存响应数 | 输入token | 输出token | 成功请求中位耗时（秒） | 成功请求P95（秒） |
|---|---|---:|---:|---:|---:|---:|---:|
| B | 本地 | 2400 | 2404 | 8,621,403 | 1,485,051 | 4.00 | 4.30 |
| B | 线上 | 2400 | 2758 | 8,894,234 | 1,406,487 | 11.41 | 14.62 |
| C | 本地 | 600 | 600 | 2,141,410 | 363,204 | 3.94 | 4.30 |
| C | 线上 | 600 | 654 | 1,949,263 | 324,786 | 9.92 | 12.31 |

VLM标注是候选池级成本，完成后可在多个冻结模型和选样预算中复用。C的独立请求账本记录本地600次、线上660次尝试；线上已保存响应数为654，不能把响应文件数与尝试数等同。服务并行执行，因此请求时长之和不等于墙钟时间；本地服务报告时间亦不是实测GPU活跃时间。

当前没有实际账单或可核实历史计费单价，货币费用保留为未知。缓存选样微基准仅测量代表性模型的CPU选样过程，不包含全池状态标注、嵌入提取与分类推理，不据此宣称端到端成本优势。

## 6 讨论

### 6.1 研究支持什么样的测试结论

最充分的证据来自完整风险层比较：在所评估六类、八种模型及两个确认来源中，最高风险层的F1均低于最低风险层。由于风险层在不读取分类器输出的情况下定义，这个差异不是直接按错误结果排序产生的。另一数据来源保留了差异方向，使这一结论比单池探索更有支撑。

其测试意义在于将模型评价从一个总体分数展开为条件响应。总体分数仍然有用，但测试者可以同时看到高风险层表现、类别响应及具体状态落差，从而判断现有测试证据是否包含关注的条件。本文没有证明风险评分已经刻画所有难度，也没有提出一个可以替代全部测试充分性的单一分数。

### 6.2 覆盖、难度与性能为何不应合并为一个指标

覆盖描述状态证据的广度与频次，风险描述一组预定义困难条件，性能描述固定模型在当前样本构成下的正确性。三者相关但不相同。向测试集加入更多容易识别的新状态，可能同时提高覆盖与F1；加入更多困难状态，可能提高覆盖并降低F1。Precision还会受到类别间误报流向影响，不能预设它必须与Recall同向。

因此，本文没有将F1下降设为测试策略有效性的充分条件。一个策略即使挑出很少几种重复困难样本，也可能取得较高错误率；另一个策略可能错误较少但揭示更广的失败类型。等错误数诊断对照正是为了区分这两种情况。

### 6.3 Coverage的附加价值是有条件的

共同配额协议表明，覆盖选样在部分设置下增加混淆类型和错误簇的广度。VOC的本地来源未复现该收益，精确风险匹配结果也未获得跨来源显著支持；统一ViT表征改变了错误簇结论。因此，现有证据支持把Coverage作为测试组织与解释工具，并报告它何时提供额外信息，而不足以将其称为独立于所有难度、表征和数据来源的诊断方法。

双标注器差异可能来自状态枚举理解、未知处理或同一图像的视觉描述差异，当前缺乏人工状态真值，无法确定哪一种标注更正确。线上结果更有利于某项比较，也不能据此选择性删除本地结果。

### 6.4 模型能力比较应结合水平、响应与条件

高风险层性能回答“模型在较困难条件下能识别多少”，风险差回答“从较低风险到较高风险变化有多大”，状态诊断回答“哪些具体条件需要关注”。三者提供互补信息。当前ConvNeXt-Tiny和Swin-T在高风险层平均性能中较靠前，但模型选择仍取决于测试任务、训练配置及计算约束，不能只依据本研究排序给出通用最佳模型。

### 6.5 状态复现需要足够的条件证据

状态诊断的可复现性依赖候选池中相应状态是否有足够样本。VOC确认池统一较小，沿用至少20个目标的支持门槛后，仅少数冻结组合可评价。降低门槛或事后重挑状态可以增加表面可评价数量，却会改变原确认问题。本文保留原门槛，并将跨来源状态复现列为证据不足，而不使用单个负向案例代替确认。

## 7 有效性威胁与适用范围

**构念有效性。** 可见视觉状态是有限描述，频次覆盖不是完整状态组合覆盖，也不代表真实场景全部条件。风险五维存在相关性，固定权重未独立校准，分位层具有相对性。错误簇是表征代理而非缺陷根因，统一表征敏感性结果表明其定义影响结论。

**内部有效性。** VLM标签经过结构校验及部分字段的官方二值属性参考对照，未验证本研究人工语义准确率。该对照显示字段和来源间的分歧，不能证明全部严重程度等级或其他状态准确。分类器使用裁剪图，VLM同时使用原图、标记图和放大图，可见信息并不相同。遮挡、场景和目标尺度也可能共现；匹配标量风险不能隔离全部混杂因素。重复检测及历史清单排除不保证消除全部近重复，预训练重叠未被排除。

**统计结论有效性。** 八架构限制了精确p值分辨率和强基线校正能力。三个种子描述同一架构训练变动，不是24个独立架构。固定池下层级bootstrap与固定模型面板下图像bootstrap覆盖不同不确定性；非线性等错误数诊断的重复权重重采样亦可能引入偏差。本文保留效应方向、区间、预定家族与完整敏感性结果，不以某个区间或p值替代全部证据。

后续逐图二项混合模型在预定迭代上限内未收敛，AUC混合模型亦有奇异/边界警告；它们保留为执行诊断，不提供新的确认结论，也不替代正文主检验。拟增加的未用Open Images池因sheep历史排重后数量不足未执行，不计入外部确认结果。

**外部有效性。** 实验涉及六类对象及两个确认数据来源，类别包含多种动物、person及car，但仍不能代表全部视觉任务。person是目标类别，不等于独立人脸识别任务；本文没有将历史人脸实验混入六类主统计。研究采用固定框分类，未验证目标检测、分割、开放集识别或实际部署中的目标定位错误。

**流程与选择偏差。** 风险规则最初来自探索，后续确认沿用冻结规则。全候选池预标注和按类别配额选样是已标注测试池设置，不覆盖完全未知流式输入。VOC2007合并原生trainval与test以构建本研究评测池，不能作为官方划分的标准模型比较。A中的历史技术替换与C的来源调整均记录，后者在新预测前完成；这些记录不能消除所有来源选择偏差。

## 8 结论

本文研究了基于视觉状态覆盖与风险分层的多类别图像分类模型测试。在固定训练数据与模型参数条件下，通过双自动标注器、类别平衡风险层、预算梯度及诊断对照，将测试条件覆盖、风险构成与测得性能分开评价。

在本研究评估范围内，预定义视觉风险层稳定区分了模型表现：最高风险层Macro-F1在全部运行中低于最低风险层，该现象在新样本和另一数据来源上复现。覆盖驱动选样在部分控制条件下增加了诊断广度，但收益受标注来源、数据池与表征影响，尚未形成跨设置的普遍优势。冻结薄弱状态在新COCO样本上获得复现支持，其跨来源验证受状态支持不足限制。

研究表明，视觉状态与风险分层能够为总体性能评价补充条件化的能力证据和状态诊断。实践中应共同报告测试条件、风险层表现、状态支持量及失败类型，并保留局部回升、未复现状态和不显著对照，避免以覆盖率或F1单独概括模型测试质量。

## 参考文献

[1] Pei K, Cao Y, Yang J, Jana S. DeepXplore: Automated Whitebox Testing of Deep Learning Systems. SOSP, 2017. [原文](https://arxiv.org/abs/1705.06640).

[2] Ma L, et al. DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems. ASE, 2018. [原文](https://arxiv.org/abs/1803.07519).

[3] Kim J, Feldt R, Yoo S. Guiding Deep Learning System Testing using Surprise Adequacy. ICSE, 2019. [原文](https://arxiv.org/abs/1808.08444).

[4] Harel-Canada F, Wang L, Gulzar M A, Gu Q, Kim M. Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks? ESEC/FSE, 2020. DOI: 10.1145/3368089.3409754. [作者原文](https://fabrice.harel-canada.com/publication/neuron-coverage-eval/neuron-coverage-eval.pdf).

[5] Dola S, Dwyer M B, Soffa M L. Input Distribution Coverage: Measuring Feature Interaction Adequacy in Neural Network Testing. ACM Transactions on Software Engineering and Methodology, 32(3), 2023. DOI: 10.1145/3576040. [DOI](https://doi.org/10.1145/3576040)；[作者学位论文第5章中的对应方法](https://www.cs.virginia.edu/~soffa/research/dissertations/swaroopa_thesis.pdf)。

[6] Feng Y, Shi Q, Gao X, Wan J, Fang C, Chen Z. DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks. ISSTA, 2020. [原文](https://arxiv.org/abs/1903.00661).

[7] Sener O, Savarese S. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR, 2018. [原文](https://arxiv.org/abs/1708.00489).

[8] Aghababaeyan Z, Abdellatif M, Briand L, Ramesh S, Bagherzadeh M. Black-Box Testing of Deep Neural Networks Through Test Case Diversity. IEEE Transactions on Software Engineering, 2023. DOI: 10.1109/TSE.2023.3243522. [原文](https://arxiv.org/abs/2112.12591).

[9] Aghababaeyan Z, Abdellatif M, Dadkhah M, Briand L. DeepGD: A Multi-Objective Black-Box Test Selection Approach for Deep Neural Networks. 2024修订版，DOI: 10.1145/3644388. [原文](https://arxiv.org/abs/2303.04878).

[10] Lin T Y, et al. Microsoft COCO: Common Objects in Context. arXiv:1405.0312. [原文](https://arxiv.org/abs/1405.0312).

[11] PASCAL VOC. The PASCAL Visual Object Classes Challenge 2007. [官方数据与任务说明](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html).

## 附录A：结果追踪与复算材料

本附录提供本地审阅链接，不表示实验已作为公开复现包发布。原始图片的许可与下载方式沿用数据来源要求。

| 正文证据 | 原始材料 |
|---|---|
| 数据、冻结规则及预算 | [COCO确认协议](/home/hcf/.codex/worktrees/2fec/QE/docs/sixclass_independent_confirmation_protocol_20261004.md)、[VOC外部确认协议](/home/hcf/.codex/worktrees/2fec/QE/docs/sixclass_voc_external_confirmation_protocol_20261005.md) |
| 表1、表2：B完整层与差值 | [完整层指标](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/tables/full_stratum_metrics.csv)、[风险差值](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/tables/difficulty_contrasts.csv) |
| 表1、表2：C完整层与差值 | [完整层指标](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/tables/full_stratum_metrics.csv)、[风险差值](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/tables/difficulty_contrasts.csv) |
| 表3：累计曲线 | [B累计指标](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/tables/cumulative_metrics.csv)、[C累计指标](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/tables/cumulative_metrics.csv) |
| 表4与强基线 | [B全部方法比较](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/comparison/tables/comparisons.csv)、[C全部方法比较](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/comparison/tables/comparisons.csv) |
| 表5：精确风险匹配 | [B比较](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/tables/matched_equal_error_comparisons.csv)、[C比较](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/tables/matched_equal_error_comparisons.csv) |
| 表6：模型高风险能力 | [B模型响应](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/confirmation/tables/model_robustness.csv)、[C模型响应](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/confirmation/tables/model_robustness.csv) |
| 表7：冻结状态迁移 | [B迁移汇总](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_independent_confirmation_20261004/transfer/summary.json)、[C迁移汇总](/home/hcf/.codex/worktrees/2fec/QE/artifacts/sixclass_voc_external_confirmation_20261005/transfer/summary.json) |
| 表8：状态案例 | [全350条状态验证记录](/home/hcf/.codex/worktrees/2fec/QE/artifacts/empirical_evidence_strengthening_20261005/state_case_metrics.csv)、[所选案例](/home/hcf/.codex/worktrees/2fec/QE/artifacts/empirical_evidence_strengthening_20261005/selected_state_cases.csv)、[含正确样本与反例的图片案例](/home/hcf/.codex/worktrees/2fec/QE/artifacts/empirical_evidence_strengthening_20261005/diagnostic_cases.html) |
| 表9：标注成本 | [成本汇总](/home/hcf/.codex/worktrees/2fec/QE/artifacts/empirical_evidence_strengthening_20261005/cost_summary.csv)、[阶段耗时](/home/hcf/.codex/worktrees/2fec/QE/artifacts/empirical_evidence_strengthening_20261005/stage_durations.csv) |
| 全部正负结果与表征敏感性 | [80项完整结果索引](/home/hcf/.codex/worktrees/2fec/QE/artifacts/empirical_evidence_strengthening_20261005/full_outcome_index.csv) |

## 附录B：主要结论与支持等级

| 结论 | 当前支持程度 | 表述边界 |
|---|---|---|
| 预定义风险分层区分固定分类器表现 | 在被测设置内支持，另一来源复现 | 主要是Q5与Q1差异；不要求逐层单调 |
| 覆盖与性能提供不同测试信息 | 数据与指标定义支持 | 不能把覆盖增长直接解释为F1下降的原因 |
| Coverage相对Random增加诊断广度 | 条件性支持 | 依赖来源、控制协议及表征；VOC本地未获支持 |
| Coverage全面优于强基线 | 未确立 | 不作为本文主要论点 |
| 冻结薄弱状态在新样本中复现 | B中支持 | 不代表每条状态均复现 |
| 冻结薄弱状态跨来源稳定迁移 | 证据不足 | C仅四条达到支持门槛，不能泛化 |
| 状态标签准确或状态是失败根因 | 未验证 | 自动结构校验与关联诊断不构成人工真值或因果证明 |
