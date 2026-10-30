# 视觉状态覆盖与难度分层的多类别模型测试：一项双标注器实证研究

Visual-State Coverage and Difficulty-Stratified Testing of Multiclass Image Classifiers: An Empirical Study with Two Automatic Annotators

## 摘要

常规图像分类评测通常报告固定测试集上的总体性能，却较少说明测试集覆盖了哪些类别内部视觉状态，以及模型在不同视觉风险层上的能力差异。增加测试样本或提高状态覆盖并不必然降低测得性能，因此，单独观察F1与覆盖率的相关曲线不足以判断测试充分性。本文研究可解释视觉状态覆盖、类别内风险分层与错误诊断之间的关系。我们固定六类图像分类任务、1800个训练目标、8种ImageNet预训练架构及3个训练种子，在包含dog、cat、horse、sheep、person、car各400个目标的修订候选池上开展测试。线上与本地视觉语言模型分别完成全部2400个目标的状态标注，主覆盖指标采用14项共享字段及程序计算的目标尺度。未分层实验比较Coverage、随机、DeepGini、距离型Surprise Adequacy和冻结ViT嵌入k-center；随后追加不使用分类器输出定义的五维视觉风险，将每类分成五个等数量层，并设置同预算、同层及精确风险分数匹配的随机对照。

完整难度层结果显示，本地标签下平均Macro-F1从Q1的94.90%降至Q5的82.67%，线上标签下从93.68%降至84.35%；两来源的24次运行均出现Q5低于Q1，平均降幅分别为12.22和9.33个百分点。未知状态上下界敏感性分析保留该方向，但中间层并不普遍单调。分层Coverage的累计测试集由120个扩至600个目标时，两来源覆盖率分别由21.51%和21.63%增至66.56%和67.29%，Macro-F1分别由93.25%和91.95%降至85.06%和85.28%。在原未分层实验中，逐类别匹配错误数后，Coverage相对随机的混淆类型广度和错误簇广度均得到预定主检验支持；进一步精确匹配风险后，只有线上标签下错误簇广度仍通过补充检验族校正，归一化预算AUC差为+0.0100，其余三项未获支持。结果表明，难度分层能够揭示总体指标掩盖的能力差异，而Coverage的额外诊断价值依赖标注来源、评价指标和控制条件。本文提供可审计的测试组织与诊断分析流程，不将覆盖增长解释为模型训练改善，也不声称覆盖选样全面优于其他测试方法。

关键词：深度学习测试；视觉状态覆盖；难度分层；多类别图像分类；错误诊断；视觉语言模型；实证研究

## Abstract

Aggregate classification scores reveal limited information about within-class visual conditions and model robustness. We study visual-state coverage, risk-stratified evaluation, and diagnostic breadth using six classes, 2,400 candidate targets, eight ImageNet-pretrained architectures, and three training seeds. Training data and classifier weights remain fixed throughout testing. A local and an online vision-language model independently annotate every target. Coverage uses fourteen shared visual fields and a program-derived scale field. The original selection experiment compares coverage-guided selection with random selection, DeepGini, distance-based Surprise Adequacy, and frozen-ViT k-center. A post-hoc supplement constructs five equal-sized, within-class risk strata without classifier outputs and introduces random controls that exactly match the selected risk-score and unknown-dimension histograms.

Across all 24 runs, highest-risk Macro-F1 is below lowest-risk Macro-F1 under both annotation sources, with mean decreases of 12.22 and 9.33 percentage points. Intermediate strata are not uniformly monotonic. Coverage-stratified cumulative tests increase state coverage while decreasing measured F1, but these changes describe test composition rather than model deterioration. After exact risk matching and class-specific equal-error control, only online-annotation error-cluster breadth retains a corrected significant gain over random selection: a normalized budget-AUC difference of 0.0100. The findings support risk stratification as an interpretable testing instrument and identify conditional, metric-dependent diagnostic value for coverage-guided selection. They do not establish universal superiority or a causal relationship between coverage and classification error.

## 1 引言

图像分类模型可能在多数常见样本上表现良好，却在遮挡、局部可见、截断、异常朝向等条件下集中失败。若测试集主要包含典型目标，一个较高的总体F1既不能说明困难状态已被测试，也不能说明不同模型的能力边界相同。对于软件测试，除了“模型平均答对多少”，还需要回答“测试了哪些条件”“哪些条件下更容易失败”以及“在有限预算内发现的失败是否具有足够广度”。

深度学习测试已提出神经元覆盖、Surprise Adequacy和输入分布覆盖等方法[1–3,9]，也已有研究质疑某些覆盖指标与错误揭示能力的关系[6]。黑盒输入多样性和多目标选样进一步说明，发现错误数量与发现不同失败并非相同目标[7–8]。因此，本研究不以“覆盖增加必然使F1下降”作为方法有效性的前提，也不以某一种覆盖率代替所有测试质量。

本文使用有限枚举的可见视觉状态描述类别内部变化，并将覆盖与视觉风险分开操作。覆盖表示测试集提供了多少状态证据；风险是由目标尺度、可见性、遮挡、截断和朝向定义的测试分层代理。两者可能相关，却不是同一变量。原实验只优化状态覆盖时，累计F1曾随预算增加而上升。这并不表示模型在测试过程中变强，而是说明覆盖排序没有保证后加入样本更难。为更清楚地解释模型能力，本文追加类别平衡的风险分层，并分别报告完整层、独立新增批次和累计集合。

研究问题如下。

1. **RQ1：视觉风险分层能否稳定区分固定分类器的性能？** 重点比较同等类别构成下的Q1与Q5，并检查未知状态处理的敏感性。
2. **RQ2：分层逐步扩展测试集时，状态覆盖与分类性能如何共同变化？** 区分批次和累计结果，检查是否存在中间阶段回升。
3. **RQ3：Coverage是否提供超过随机选样的诊断广度，其收益是否仍能在匹配错误数和风险后保留？** 使用不依赖VLM状态真值的混淆类型及错误簇指标。
4. **RQ4：不同模型的风险响应和薄弱状态有何差异，结论对标注来源是否敏感？** 报告全部模型及类别，不根据结果选择有利模型。

本文贡献定位为实证证据和可复算的测试流程：第一，给出共享视觉状态覆盖与类别内风险分层的组合方法；第二，通过固定训练、类别平衡、双标注器及风险匹配对照区分覆盖、难度和错误数量；第三，提供完整风险响应、类别状态诊断和条件性负结果。本文不提出新的分类器，也不将“困难测试降低平均性能”本身作为独立算法创新。

## 2 相关工作

### 2.1 深度学习覆盖与测试充分性

DeepXplore将神经元覆盖与差分行为用于测试生成[1]，DeepGauge提出多个粒度的内部覆盖标准[2]。这些方法关注网络内部活动，与本文的外部可见视觉状态不同。Harel-Canada等对神经元覆盖的实证研究表明，在其研究设置下，提高覆盖不能保证更有效的错误揭示[6]。本文据此分别评价输入状态证据、测得性能和错误诊断，不把覆盖率单独视为充分性结论。

Input Distribution Coverage通过潜在输入表示及特征交互度量测试充分性[9]。它与本文共享“输入条件覆盖”这一研究方向，但其潜在空间交互与本文有限可见枚举不同。本文当前指标只统计单字段状态的边际覆盖，尚未构建完整组合交互覆盖，也不能宣称已覆盖所有视觉条件。

### 2.2 测试选样与多样性

Surprise Adequacy衡量测试输入相对训练行为的差异[3]；DeepGini通过预测概率的分散程度优先选择潜在错误[4]。Core-set工作从表征几何角度讨论样本选择[5]。本文实现冻结ViT表征上的贪心k-center作为几何多样性对照，属于研究中的操作化基线，不声称逐项复现原主动学习训练流程。

黑盒测试多样性研究已经比较输入表征多样性与覆盖指标的错误揭示关系[7]；DeepGD同时考虑不确定性和不同失败的多样性[8]。因此，“关注不同失败而不仅是错误数量”已有研究基础。本文与其区别在于可解释视觉状态、双自动标注器、类别内风险匹配，以及逐类别等错误数的诊断控制。当前并未实现DeepGD基线，故不能声称优于该方法。

### 2.3 实证流程与本文范围

近期ADEPT整合多种测试充分性技术，强调一致执行、缓存和结构化结果[10]。本文的可审计工程流程有助于复算，但工程完整性本身不构成“首个通用测试框架”的证据。本文聚焦于一个具体问题：在固定模型上，可解释视觉风险如何影响测试结果，以及状态Coverage在更严格控制后还保留多少诊断收益。

## 3 方法

### 3.1 固定目标分类与视觉状态标注

每个样本由一个预先确定的目标框定义，类别标签来自数据集标注。分类器输入为该目标的裁剪图，输出六类之一。VLM状态标注输入为原图、A框标记图及放大裁剪图，固定目标由程序指定，VLM不自行寻找测试对象。状态标注提示包含目标类别信息，因此该阶段是类别条件下的状态描述，不是盲分类实验。VLM响应不参与修改分类器权重。

使用本地`qwen3-vl:30b-a3b-instruct`与线上`sensenova-6.8-flash-lite`，分别生成结构化标签。主覆盖使用14项共享字段：身体可见性、身体朝向、遮挡程度、遮挡物类型、截断程度、群体关系、交互关系、场景、背景复杂度、模糊程度、照明、目标背景对比、可识别性和视觉呈现形式。另由原图目标框面积比计算尺度字段。类别专属字段保留用于解释，不进入六类主覆盖分母。

结构校验检查字段、枚举和值类型；100%严格有效响应只表示格式符合协议，不代表语义标签100%正确。unknown保存为不确定状态，不将其计为已覆盖的视觉证据。本文没有进行人工状态真值验证，双标注器结果用于分析来源敏感性，不能代替人工准确率估计。

### 3.2 频次覆盖与Coverage选样

对类别c，在完整修订候选池中，两来源已观测到的非unknown状态并集构成固定状态宇宙U_c。一个状态为“字段—枚举值”对，多值字段可同时激活多个状态。对所选测试集S，记状态s的出现次数为n_s(S)。频次覆盖定义为：

$$
C_c(S)=\frac{1}{|U_c|}\sum_{s\in U_c}\min\left(\frac{n_s(S)}{20},1\right),\qquad C(S)=\frac{1}{6}\sum_c C_c(S).
$$

阈值20表示状态证据达到20个目标后不再增加该状态得分。该指标与“至少出现一次”的存在覆盖不同；少数状态虽已出现，但频次证据不足仍会拉低频次覆盖。分母基于当前有限候选池，因而结果是候选池条件下的覆盖，不是现实视觉世界的覆盖。状态宇宙沿用完整候选池，不随预算或难度层重新缩小，否则不同点不能直接比较。

Coverage贪心选样采用饱和后的凹增益：对于每个候选，累加其激活状态在当前计数下的log(n+2)−log(n+1)，仅对当前n<20的状态加分。每次选择最大增益目标并更新计数。它直接优化的是稀缺状态证据，不等同于直接最大化分类错误，也不是直接最大化上式线性截断覆盖。在未分层主实验中，它对每类全池排序；补充分层实验中，它仅在当前风险层选样，计数跨层连续累计。补充实验并列增益按sample_id升序解开，旧主实验的既有次序保持原样。

### 3.3 不依赖分类器输出的风险分层

视觉风险H由五维分数相加，范围为0–13。分数不使用分类器是否答对、预测置信度或特征表征，也不使用VLM的target_recognizability字段，降低直接以模型失败定义“困难”的循环性。

| 风险维度 | 固定映射 |
|---|---|
| 原图框面积比a | a≥0.10：0；0.03≤a<0.10：1；0.01≤a<0.03：2；a<0.01：3 |
| 身体可见性 | full/most：0；partial：2；minimal：3 |
| 遮挡程度 | none：0；light：1；moderate：2；heavy：3 |
| 截断程度 | none：0；light：0.5；moderate：1；heavy：2 |
| 身体朝向 | front/left_side/right_side/three_quarter：0；back：1；overhead_or_unusual：2 |

未知维度保存从0到该维度最大值的区间，H同时保存下界、上界及中点。主分析采用中点排序，下界与上界分别重新划层作为敏感性分析，不把unknown固定解释为低风险。本地有83个目标、线上有112个目标在这四项VLM风险维度中至少一项未知，分别占3.46%和4.67%。

每种标注来源分别在每类内部按H、sample_id升序排序，划为Q1–Q5，每层80个目标。六类同层合计480个目标，类别数量相同。Q5表示该类别内相对高风险，不是统一的绝对难度门槛；不同类别及标注来源的分数分布不同。同分目标可跨越层边界，例如本地dog的Q1上限和Q2下限同为0.5。因此，五层是一种平衡的分位划分，不表示每张Q2图都严格难于每张Q1图。

这个评分是一种可见风险代理。身体可见性、遮挡和截断可能相关，当前权重也没有独立学习或校准。目标尺度来自原图，而分类器输入是放大后的裁剪图，二者不可混同。因此，结果解释为风险层与性能的关联，不解释为各风险因素的因果效应。

### 3.4 三种测试视角

完整层评测使用某一Q层全部480个目标，直接观察固定模型对风险层的响应。它不受层内Coverage选样影响，是判断风险代理是否有用的主要补充证据。

独立新增批次在每类每层通过Coverage选20个目标，共120个；五批互不重复。它回答“新加入这一层样本时，模型表现如何”。累计测试集将前k批合并，规模为120、240、360、480、600，每类对应20、40、60、80、100个目标。累计Macro-F1从整体混淆矩阵重新计算，不是各批F1的简单平均。

主分层覆盖分母对所有批次固定。原未分层主实验与新分层序列具有不同选样约束，因而不能将两者的端点差直接解释为某方法经过优化后变强。

### 3.5 同预算及精确风险匹配对照

原未分层实验包含四个对照：全池内同类别预算随机选样1000次；DeepGini按1−Σp_k²降序；DSA采用分类器表征中到预测类别最近训练目标的距离与该训练目标到其他类别最近训练目标距离的比值；k-center在冻结ViT-B/16表征中归一化后使用余弦距离贪心最远点选样，首点为最接近类别表征中心的目标。各方法每类预算相同。

新分层随机对照比仅匹配层更严格。对每个来源、类别和风险层，统计Coverage选择的20个目标的(H中点，未知维数)联合频数，再在该层无放回随机抽取相同联合频数的样本。重复1000次，累计集合也保留此前层的同规模约束。所有匹配频数均通过程序审计。这个对照控制标量风险及未知维数，但没有匹配全部视觉状态组合，也不能控制VLM未观测的风险。

新随机对照只用于分层Coverage。本文没有将DeepGini、DSA、k-center重新约束到同样的风险层，因此强基线结论只来自原未分层比较，不能由新分层结果扩大。

### 3.6 性能与独立诊断指标

分类性能报告Accuracy、Macro-Precision、Macro-Recall、Macro-F1及每类Recall。每类F1为该类Precision与Recall的调和平均，Macro-F1为六个类别F1的算术平均，不是dog F1，也不是先平均Precision、Recall后计算F1。由于测试集各真实类别数量相同，Accuracy与Macro-Recall在本设置中相等。

主要诊断指标与VLM标签无关。混淆类型广度统计每个真实类别被误判到的不同其他类别；错误簇广度在冻结分类器表征中，对完整候选池的错误样本按真实类别聚类。非单例错误类别的簇数为max(2,min(8,round(√m)))，m为该类全池错误数；0错误时按不可评价并记覆盖0，1错误时为单例簇。错误簇是表征多样性的操作化代理，不表示每簇具有独立软件缺陷根因。

为降低“错误多所以类型多”的解释，两个方法逐真实类别取相同错误数m，即该类别两组错误数的较小值，再通过无放回稀释的精确期望计算各类型被发现的概率。对一个组中N个错误里出现r次的类型，m个错误样本发现它的概率为1−组合数(N−r,m)/组合数(N,m)。随后对可行类型数归一化、六类宏平均，并计算每类预算20–100曲线的归一化梯形AUC。该AUC不是ROC-AUC，差值也不是分类F1的提升。

匹配错误数是统计控制，不是测试前知道哪些目标会错的在线算法。完整池错误结构同样用于事后诊断评价，不输入Coverage排序。

## 4 实验设置与证据层次

### 4.1 数据及来源

实验含六类固定目标。训练目标主要来自COCO及既有dog初始集合，类别和框依据官方标注[11]；验证和基础测试来自COCO val2017。候选池主要来自Open Images[12]，并保留历史扩展中111个VOC2012 sheep目标。该混合来源构成是实验清单中的实际情况，不将修订池误称为单一Open Images官方测试划分。

| 数据用途 | 每类目标数 | 总目标数 | 使用范围 |
|---|---:|---:|---|
| 固定训练集 | 300 | 1800 | 训练六分类器 |
| 固定验证集 | 15 | 90 | 选择checkpoint |
| 独立基础测试 | 50 | 300 | 基础参考，不参与当前候选排序 |
| 修订候选池 | 400 | 2400 | 当前所有测试选样与风险分析 |
| 完整Q层 | 80 | 480 | 同层全部目标评测 |
| 新增Q层批次 | 20 | 120 | 分层Coverage新增目标 |
| 累计最终测试 | 100 | 600 | Q1至Q5所选目标合并 |

候选池的具体来源为：dog，Open Images train/test各300/100；cat各200/200；horse train/test/validation为254/136/10；person为274/120/6；sheep为163/116/10，另有VOC2012目标111个；car为Open Images test400个。数据集原split仅用于追踪来源，候选目标没有用于本研究分类器训练。类别与来源分布并非完全交叉平衡，尤其VOC补充只存在于sheep，因此不能完全分离类别和数据源的影响。

原候选池中horse和person各一个目标被线上服务多次拒绝。按同类别、来源、官方遮挡/截断标志及尺度区间的登记规则替换，其他2398个目标保持不变。替换不依据分类器错误或Coverage收益，保留拒绝记录和对应映射。新两个目标补做全部24份冻结权重预测；其余2398条预测和表征核对为逐元素不变。线上、本地均在修订池完成2400条严格有效标签。接口拒绝不是分类错误，不能计入分类错误率。

训练/评测清单进行ID、哈希及既有近重复阈值检查；新替换目标的审计未发现达到pHash与ViT相似度联合阈值的近重复。自动阈值检查不保证所有近重复均被排除，也不排除ImageNet预训练阶段潜在重叠。

### 4.2 分类器与固定训练

架构为ResNet-18、ResNet-50[13]，MobileNetV3-Small、MobileNetV3-Large[14]，DenseNet-121[15]，EfficientNet-B0[16]，ConvNeXt-Tiny[17]和Swin-T[18]。每个架构使用三个种子20260916、20260917、20260918，共24次独立训练运行；架构相关统计以8个架构为单位，种子不视为24个独立架构。

模型使用torchvision的ImageNet预训练权重，更换六分类头并微调整个网络。统一训练20轮，每轮类别均衡加权抽样3000次，batch为64，AdamW学习率1×10⁻⁴、weight decay为1×10⁻⁴，余弦学习率调度。输入采用Resize(256)、224像素裁剪、ImageNet归一化；训练加入随机裁剪、水平翻转及轻度ColorJitter，评测使用中心裁剪。以固定验证集Macro-F1选择checkpoint。架构采用统一配置，便于控制流程，但不等于每个架构都进行了最佳超参数搜索。

测试阶段权重固定。双VLM来源只改变状态标签、风险划分和选样次序，不改变同一图片上的分类器输出。当前论文没有进行新增训练样本后的再训练实验，F1变化表示测得性能变化，不表示模型权重改善或退化。

### 4.3 统计分析与事后性质

置信区间采用5000次架构/种子层级bootstrap；置换检验以8种架构平均差为单位，枚举符号翻转。原未分层四项Coverage相对随机主检验、16项强基线敏感性族及新补充检验族分别报告，不混合解释。新分析包含两来源×三种unknown处理的6项Q5−Q1检验，Holm校正；精确风险匹配后的两来源×两诊断指标为另一个4项补充检验族。

上述置信区间描述固定候选池下架构与种子变动，不包含重新抽取整个数据池的样本不确定性。1000次随机对照用于估计对照选择的均值和分布，不等于1000个独立数据集。只有8个架构时，置换p值分辨率有限，因而不以bootstrap区间未跨零代替家族校正检验。

六类风险分层是在看过未分层主实验结果后追加的补充分析。分析代码、风险规则和选样次序在本轮分层指标计算前保存哈希，但该操作不使研究成为独立盲测。正文同时保留主实验和补充结果，不能将所有新结果重新称为事前假设验证。

## 5 实验结果

### 5.1 RQ1：完整风险层揭示总体性能差异

完整层使用每类80个、共480个目标，表中为24次运行的平均值。它评价所有层内目标，避免把层内Coverage选择效应误当作风险层效应。

| 风险层 | 本地Macro-F1 | 本地Accuracy/Recall | 本地Macro-Precision | 线上Macro-F1 | 线上Accuracy/Recall | 线上Macro-Precision |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 94.90% | 94.90% | 95.17% | 93.68% | 93.73% | 94.08% |
| Q2 | 92.65% | 92.71% | 93.10% | 91.28% | 91.38% | 91.67% |
| Q3 | 91.88% | 91.90% | 92.32% | 91.19% | 91.19% | 91.63% |
| Q4 | 90.15% | 90.18% | 90.52% | 91.75% | 91.81% | 92.10% |
| Q5 | 82.67% | 82.83% | 83.84% | 84.35% | 84.42% | 85.50% |

Q5相对Q1的平均Macro-F1差，本地为−12.22个百分点，95% CI为[−13.34,−10.92]；线上为−9.33个百分点，95% CI为[−10.40,−8.18]。两来源24次运行均为负，六项补充族的Holm p均为0.046875。本地下界/上界处理的平均差分别为−12.00/−12.29个百分点；线上分别为−9.08/−9.33个百分点，方向保持一致。

因此，在当前数据和模型配置内，高风险层稳定低于低风险层，但不能扩写为“F1随每一级风险都必然下降”。线上Q4平均F1高于Q3，类别级响应也有回升。该非单调性说明五维风险只解释部分分类难度。

![图1 完整风险层上的八模型Macro-F1。每层480目标，每条线为三个种子的均值。圆点为真实Q层，平滑线仅为PCHIP视觉引导，不增加实验点。](../artifacts/sixclass_difficulty_supplement_20261004/figures/full_stratum_model_f1.png)

### 5.2 RQ2：批次、累计性能和覆盖增长

每批120个目标。新增Q5批次的F1明显低于Q1，但累计结果仍保留先前低风险样本，因而累计降幅较小。

| 层/阶段 | 总累计目标 | 本地新增批次F1 | 本地累计F1 | 本地累计覆盖 | 线上新增批次F1 | 线上累计F1 | 线上累计覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 120 | 93.25% | 93.25% | 21.51% | 91.95% | 91.95% | 21.63% |
| Q2 | 240 | 91.57% | 92.44% | 36.41% | 89.94% | 90.95% | 36.54% |
| Q3 | 360 | 88.71% | 91.21% | 47.27% | 85.25% | 89.06% | 47.68% |
| Q4 | 480 | 81.82% | 88.86% | 56.98% | 88.03% | 88.81% | 56.73% |
| Q5 | 600 | 69.27% | 85.06% | 66.56% | 70.66% | 85.28% | 67.29% |

本地累计F1下降8.20个百分点，线上下降6.67个百分点；最后累计错误率分别为14.88%与14.64%。这些是固定权重面对改变测试集合的结果，不是模型在新增图片后变差。线上Q4批次从Q3的85.25%回升至88.03%，累计F1仍略降，进一步说明批次与累计的方向可以不同。

相同风险分数和未知维数的随机对照，最终本地覆盖59.41%、F1为88.64%；线上覆盖59.15%、F1为88.77%。对应Coverage最终覆盖较高，但F1较低约3.58/3.48个百分点。这是描述性对比：标量风险相同并不意味着所有视觉条件相同，Coverage倾向选择不同的状态构成，后续诊断检验才评价它是否提供额外信息。

![图2 分层Coverage的新增批次、累计Macro-F1与累计频次覆盖。实线为Coverage，虚线为精确风险匹配随机均值；每点先按运行计算，再平均24次运行，随机对照另先平均1000次选样。](../artifacts/sixclass_difficulty_supplement_20261004/figures/batch_cumulative_and_coverage.png)

频次覆盖随累计加入非重复样本不下降，主要来自固定分母下计数只增不减的定义。它不能单独证明新增批次更困难。本文分层序列呈现覆盖上升与F1下降，是风险约束和状态选样共同作用下的观察，不是“覆盖导致分类错误”的因果证明。

### 5.3 RQ3：先报告未分层主实验，再增加风险控制

原未分层实验在每类100个、总600目标时，表现如下。它与分层序列选中的图片不同，表中结果不能直接与上一节作模型退化差值。

| 未分层方法 | 频次覆盖 | Accuracy | Macro-F1 | 错误率 |
|---|---:|---:|---:|---:|
| 本地状态Coverage | 69.27% | 82.97% | 82.88% | 17.03% |
| 线上状态Coverage | 71.33% | 83.69% | 83.63% | 16.31% |
| Random | 本地56.96%/线上56.90% | 90.49% | 90.43% | 9.51% |
| DeepGini | 本地60.15%/线上60.99% | 72.08% | 71.86% | 27.92% |
| DSA | 本地61.01%/线上62.34% | 74.15% | 74.05% | 25.85% |
| ViT k-center | 本地60.12%/线上61.06% | 81.76% | 81.81% | 18.24% |

DeepGini和DSA发现的原始错误比例高于Coverage，故本文不能以最高错误率声称Coverage占优。原四项主结果考察的是逐类别等错误数之后的诊断广度。

| 未分层Coverage来源 | 等错误数指标 | 相对Random的AUC差 | 95% CI | 正向运行 | 四项主族Holm p |
|---|---|---:|---|---:|---:|
| 本地 | 混淆类型 | +0.0258 | [+0.0153,+0.0349] | 22/24 | 0.03125 |
| 本地 | 错误簇 | +0.0152 | [+0.0093,+0.0214] | 23/24 | 0.03125 |
| 线上 | 混淆类型 | +0.0132 | [+0.0049,+0.0217] | 18/24 | 0.03125 |
| 线上 | 错误簇 | +0.0128 | [+0.0062,+0.0187] | 18/24 | 0.03125 |

四项预定主检验通过，但两来源×四基线×两指标共16项统一Holm敏感性校正后没有一项达到0.05。这两种统计范围需要同时呈现，不能只保留较小检验族结果来宣称全面优越。

新分层补充实验进一步匹配风险。其结果如下，差值仍是逐类别等错误数后的归一化预算AUC。

| 分层Coverage来源 | 指标 | 相对精确风险Random的AUC差 | 95% CI | 正向运行 | 四项补充族Holm p |
|---|---|---:|---|---:|---:|
| 本地 | 混淆类型 | −0.0044 | [−0.0120,+0.0034] | 9/24 | 0.546875 |
| 本地 | 错误簇 | +0.0033 | [−0.0027,+0.0094] | 13/24 | 0.546875 |
| 线上 | 混淆类型 | +0.0065 | [−0.0034,+0.0168] | 15/24 | 0.515625 |
| 线上 | 错误簇 | +0.0100 | [+0.0019,+0.0176] | 18/24 | 0.03125 |

只有线上错误簇指标保留校正后的正向证据。本地两个指标以及线上混淆类型未获支持。不能把不显著解释为已经证明零效应；同样，不能将单项通过写成覆盖具有普遍独立价值。

风险匹配同时改变了对照约束和Coverage可选择范围，因此旧、新AUC差不能作正式中介效应分解。允许的解释是：在这个更严格且不同的分层设置中，额外诊断收益变得更有条件性；部分原随机对照收益可能与风险分布或层内其他构成有关，需要进一步独立验证。

![图3 精确风险匹配及逐类别等错误数之后的诊断收益。误差线为架构/种子层级bootstrap 95% CI，p值为四项补充检验族Holm校正；区间和检验均以当前固定候选池为条件。](../artifacts/sixclass_difficulty_supplement_20261004/figures/risk_matched_independent_diagnostics.png)

### 5.4 RQ4：模型能力和类别响应并不一致

模型排名应说明评价准则。完整Q1–Q5 Macro-F1曲线的归一化面积评价风险范围内的平均表现；Q5 F1则更直接反映最高相对风险层的能力。前者不是ROC-AUC，也不保证与Q5单点排序相同。

| 模型 | 本地Q1 F1 | 本地Q5 F1 | 本地Q5−Q1 | 线上Q1 F1 | 线上Q5 F1 | 线上Q5−Q1 |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 97.70% | 88.74% | −8.96 pp | 96.72% | 89.80% | −6.92 pp |
| Swin-T | 96.87% | 86.50% | −10.38 pp | 95.51% | 88.05% | −7.46 pp |
| ResNet-50 | 95.95% | 82.82% | −13.13 pp | 94.95% | 84.62% | −10.33 pp |
| EfficientNet-B0 | 95.96% | 82.43% | −13.53 pp | 94.33% | 84.76% | −9.57 pp |
| DenseNet-121 | 94.79% | 82.68% | −12.10 pp | 94.14% | 83.14% | −11.00 pp |
| MobileNetV3-Large | 94.92% | 81.70% | −13.22 pp | 93.00% | 84.46% | −8.54 pp |
| ResNet-18 | 92.69% | 79.52% | −13.16 pp | 92.07% | 81.56% | −10.50 pp |
| MobileNetV3-Small | 90.30% | 77.00% | −13.30 pp | 88.69% | 78.40% | −10.30 pp |

按完整风险曲线面积，两来源的排序均为ConvNeXt-Tiny、Swin-T、ResNet-50、EfficientNet-B0、DenseNet-121、MobileNetV3-Large、ResNet-18、MobileNetV3-Small。ConvNeXt-Tiny在本地/线上面积分别为0.9502/0.9502，Swin-T为0.9344/0.9341；最高风险层也均以前两者领先。第三名则依指标改变：线上Q5的EfficientNet-B0略高于ResNet-50。所有排名均为三个种子的描述性均值，未进行全部架构成对显著性检验，不能声称每一个相邻名次都有统计意义。

类别层面，完整层Q1到Q5的Recall变化如下。

| 类别 | 本地Q1→Q5 Recall | 本地变化 | 线上Q1→Q5 Recall | 线上变化 |
|---|---|---:|---|---:|
| dog | 92.76%→86.41% | −6.35 pp | 92.29%→90.31% | −1.98 pp |
| cat | 96.04%→90.47% | −5.57 pp | 94.90%→88.59% | −6.30 pp |
| horse | 95.31%→79.32% | −15.99 pp | 96.77%→77.24% | −19.53 pp |
| sheep | 98.23%→83.54% | −14.69 pp | 96.51%→86.88% | −9.64 pp |
| person | 87.08%→63.80% | −23.28 pp | 81.93%→69.17% | −12.76 pp |
| car | 100.00%→93.44% | −6.56 pp | 100.00%→94.32% | −5.68 pp |

person与horse的高风险层表现较低，car在最高风险层仍保持较高Recall。不同类别对统一风险代理的响应不同；不能因为六类数量平衡，就认为六类具有相同难度或相同视觉状态分布。

![图4 六类完整风险层Recall。每格为24次运行均值，每类每层80目标；颜色统一按60%–100%映射。](../artifacts/sixclass_difficulty_supplement_20261004/figures/class_risk_recall.png)

### 5.5 薄弱状态与标注来源敏感性

在完整候选池上，对状态支持数至少20的条件计算每类Recall。下表为有明确支持数的person示例，不是独立样本组，也不是自动根因证明。

| 条件状态 | 本地支持数/Recall | 线上支持数/Recall |
|---|---|---|
| severe模糊 | 49 / 50.94% | 53 / 52.12% |
| heavy遮挡 | 38 / 52.63% | 34 / 56.00% |
| 程序尺度tiny | 41 / 55.89% | 41 / 55.89% |
| 背面朝向 | 77 / 71.27% | 58 / 73.56% |

严重模糊与重度遮挡条件下的低Recall在两来源中都出现，支持将这些条件列为优先诊断对象。模糊字段没有参与五维风险评分，提供了该评分之外的解释线索；但这些条件可重叠，例如同一目标可能同时小、模糊且遮挡。当前分析不能分别归因各因素，也不能从“背面更易错”推出某种单一机制。

两来源14共享字段含unknown的精确一致率为65.24%。遮挡物类型45.38%、截断程度49.71%、朝向50.92%，视觉呈现形式98.75%。这说明字段级差异明显，一致率不能称标注准确率。原未分层最终选样在每类的两来源共同目标比例为45%–56%，说明即便主结果方向一致，两来源也没有构造同一张测试清单。

原六类car次要结果只有本地两项诊断指标获校正支持，线上两项未通过；五组状态轮换留出的10项差值虽均为正，Holm p均为0.078125，未通过。正文不能将类别专属和留出分析写成全面复现。

### 5.6 历史跨源负结果

此前另有VOC2012五类、每类100个、总500目标的确认实验，使用当时五分类器而不是当前六分类器。其逐类别等错误数混淆AUC差为+0.0008，95% CI[−0.0110,+0.0124]；错误簇差为−0.0008，95% CI[−0.0140,+0.0122]，校正p均为1.0000。该历史设置没有确认独立诊断收益。

这一结果作为跨源可复现性的负证据保留，但不与当前六类效应直接合并，也不声称当前六类风险分层已完成独立数据源复现。当前池本身包含VOC羊类，不能以其混合来源构成代替完整的新六类外部验证。

## 6 讨论

### 6.1 何种结论适合测试方向

当前证据较强地支持一个有限结论：固定模型在类别平衡的高风险测试层上表现低于低风险层，且不同模型、类别和状态的下降幅度不同。其测试价值是将一个总体F1拆解为可解释的能力响应，而不是以降低模型得分为目的。

“覆盖率上升”与“F1下降”共同出现，说明在这个分层扩展设置中测试集提供了更多状态证据，同时包含更具挑战性的目标；两者是测试构成变化的结果。原未分层Coverage中F1曾随预算上升，所以不能把这条负关系称为普遍规律。若以后某风险层中F1回升，应保留并分析其类别或状态构成，而不应通过删除样本制造单调下降。

### 6.2 Coverage的独立诊断价值有多强

原四项等错误数主检验支持Coverage相对普通随机的诊断广度。新风险匹配只在线上错误簇指标保留支持，更合理的表述是“存在部分、条件性的额外诊断信息”，而不是“覆盖完全独立于难度”。另外，完整16项强基线校正未通过，DeepGini与DSA发现更多原始错误，这说明Coverage与基于输出不确定性的选样具有不同取舍。

若实际测试目标是快速发现尽可能多的错误，应参考错误率、不确定性基线及预算；若目标是组织可解释状态或比较高风险能力，分层视觉状态有更直接的解释作用；若目标是发现不同失败，需要继续看等错误数广度。测试工具的选择应由诊断目标决定，不能只比较一个覆盖百分比。

### 6.3 模型选择应报告性能水平和下降幅度

仅比较F1下降幅度可能偏向原本基线较低的模型，下降小不等于困难状态表现高。本文同时报告Q1、Q5及风险曲线面积。在当前配置中，ConvNeXt-Tiny与Swin-T具有较高Q5 F1及较小降幅，可作为后续目标任务的候选，而不是宣称对任意图像任务最优。模型参数规模、算力开销和独立部署数据表现仍可能改变选择。

### 6.4 方法的可解释性与不足

有限枚举状态比抽象表征更容易对应“重遮挡”“背面”“局部可见”等测试描述，但自动标注存在来源差异，手工风险权重也限制了跨类别等价性。对规则未知状态保留区间、使用全层结果及风险匹配对照，比单独绘制覆盖—F1折线更能说明研究边界。

本文主线已经从早期“dog补图后训练性能是否提升”转为“固定多类分类器的测试充分性与能力诊断”。训练增强效果不在本文结论中；后台执行、缓存和多来源标注属于可复算工程过程，不作为已经验证的分布式测试新算法贡献。

## 7 有效性威胁与适用边界

**构念效度。** H是视觉风险代理，不是经过人类一致性验证的绝对难度。多个分量相关且权重固定；原图尺度在裁剪分类中并不等于输入像素尺度。覆盖是单状态边际证据，并未覆盖状态组合交互；簇不是独立根因。

**标注效度。** 全部状态来自自动VLM，没有人工状态真值。两来源一致、unknown敏感性和跨源评价可暴露部分不确定性，却不能证明标签正确。结构成功率不能当语义准确率，online与local也不可视为可互换真值。

**内部效度。** 类别数量平衡控制类别比例，但候选来源和视觉状态并不均匀；sheep的VOC补充具有来源混杂。风险精确匹配只控制H与未知维数，不控制全部风险组合。两张拒绝替换保持流程可审计，但不能称原2400目标原样完成。

**统计结论效度。** 24次运行仅来自8种架构，不能提供24个独立架构的证据；bootstrap没有覆盖候选池重采样不确定性。不同检验族须并列报告。本文没有对全部模型对、全部类别或全部状态作显著性结论。

**选择与时间效度。** 分层分析事后追加，即使本轮运算前冻结规则，也可能受此前研究认识影响。VLM别名、接口响应和约束解码版本随时间变化；历史缓存与当前调用不构成同时点的独立标注重复。未来应在未查看结果的新池上冻结风险规则再确认。

**外部效度。** 六类包含动物、人和刚性物体，但没有覆盖全部物体类型；当前person不是专门人脸识别，不报告身份、种族或其他人口属性识别。训练与测试主要使用目标区域分类，不能直接推广到目标检测、整图多标签识别或生成式VLM分类。历史VOC五类确认未获支持，当前六类风险结果仍需新的独立候选池复现。

## 8 结论

本文在固定训练与权重条件下，用双自动状态标注器开展六类、八架构、三种子的视觉测试实证研究。类别内风险分层稳定区分低、高风险层：两来源的24次运行均出现Q5低于Q1，平均Macro-F1降低12.22和9.33个百分点；类别级及中间层仍有明显差异。分层累计测试扩展同时提高状态证据并降低测得F1，但它反映测试构成，不表示模型在测试过程中退化。

Coverage的诊断优势具有条件性：原未分层相对普通随机的四项等错误数主检验通过，而精确风险匹配后只有线上错误簇广度保留校正支持；强基线全族比较及历史跨源确认没有支持全面优势。由此，视觉状态覆盖适合作为可解释测试组织与诊断线索，风险分层适合作为固定模型能力响应的分析工具。两者应与错误数量、独立诊断广度及来源敏感性共同报告，而不能用覆盖百分比或F1下降单独证明测试方法有效。

## 参考文献

[1] Pei K, Cao Y, Yang J, Jana S. DeepXplore: Automated Whitebox Testing of Deep Learning Systems. SOSP, 2017. [原文](https://arxiv.org/abs/1705.06640).

[2] Ma L, et al. DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems. ASE, 2018. [原文](https://arxiv.org/abs/1803.07519).

[3] Kim J, Feldt R, Yoo S. Guiding Deep Learning System Testing using Surprise Adequacy. ICSE, 2019. [原文](https://arxiv.org/abs/1808.08444).

[4] Feng Y, Shi Q, Gao X, Wan J, Fang C, Chen Z. DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks. ISSTA, 2020. [原文](https://arxiv.org/abs/1903.00661).

[5] Sener O, Savarese S. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR, 2018. [原文](https://arxiv.org/abs/1708.00489).

[6] Harel-Canada F, Wang L, Gulzar M A, Gu Q, Kim M. Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks? ESEC/FSE, 2020. DOI: 10.1145/3368089.3409754. [作者原文](https://fabrice.harel-canada.com/publication/neuron-coverage-eval/neuron-coverage-eval.pdf).

[7] Aghababaeyan Z, Abdellatif M, Briand L, Ramesh S, Bagherzadeh M. Black-Box Testing of Deep Neural Networks Through Test Case Diversity. IEEE Transactions on Software Engineering, 2023. DOI: 10.1109/TSE.2023.3243522. [原文](https://arxiv.org/abs/2112.12591).

[8] Aghababaeyan Z, Abdellatif M, Dadkhah M, Briand L. DeepGD: A Multi-Objective Black-Box Test Selection Approach for Deep Neural Networks. ACM Transactions on Software Engineering and Methodology, 2024. DOI: 10.1145/3644388. [原文](https://arxiv.org/abs/2303.04878).

[9] Dola S, Dwyer M B, Soffa M L. Input Distribution Coverage: Measuring Feature Interaction Adequacy in Neural Network Testing. ACM Transactions on Software Engineering and Methodology, 32(3), Article 81, 2023. DOI: 10.1145/3576040. [出版方](https://doi.org/10.1145/3576040).

[10] Kao Y, Burnham S, Fahy T R, Ghanbari A. ADEPT: A Unified Framework for Deep Learning Test Adequacy. ISSTA, 2026. DOI: 10.1145/3837729.3840488. [原文](https://arxiv.org/abs/2608.12144).

[11] Lin T Y, et al. Microsoft COCO: Common Objects in Context. ECCV, 2014. [原文](https://arxiv.org/abs/1405.0312).

[12] Kuznetsova A, et al. The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale. International Journal of Computer Vision, 2020. [原文](https://arxiv.org/abs/1811.00982). 本研究使用Open Images V7清单；本条为数据集系列论文，并不将V4论文称为V7发布论文。

[13] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016. [原文](https://arxiv.org/abs/1512.03385).

[14] Howard A, et al. Searching for MobileNetV3. ICCV, 2019. [原文](https://arxiv.org/abs/1905.02244).

[15] Huang G, Liu Z, van der Maaten L, Weinberger K Q. Densely Connected Convolutional Networks. CVPR, 2017. [原文](https://arxiv.org/abs/1608.06993).

[16] Tan M, Le Q V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019. [原文](https://arxiv.org/abs/1905.11946).

[17] Liu Z, et al. A ConvNet for the 2020s. CVPR, 2022. [原文](https://arxiv.org/abs/2201.03545).

[18] Liu Z, et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021. [原文](https://arxiv.org/abs/2103.14030).

## 附录A：结果与复算材料

本文版本日期为2026-10-04。以下路径相对于项目根目录；Markdown图引用与Word内嵌图片均来自实际结果，未手工修改数据点。

| 正文内容 | 原始结果或复算入口 |
|---|---|
| 数据来源和数量 | artifacts/sixclass_dual_vlm_replaced_20261004/manifests/ |
| 原未分层主结果与全16项校正 | artifacts/sixclass_dual_vlm_replaced_20261004/analysis/tables/independent_comparisons.csv |
| 未分层覆盖与性能 | 同目录cross_fixed_metrics.csv、cross_random_summary.csv |
| 标注一致性与选样交集 | 同目录annotation_agreement.csv、selection_overlap.csv |
| car与留出边界 | 同目录category_independent_comparisons.csv、cross_source_holdout_comparisons.csv |
| 完整层、批次、累计结果 | artifacts/sixclass_difficulty_supplement_20261004/tables/full_stratum_metrics.csv、selected_batch_metrics.csv、cumulative_metrics.csv |
| 风险及同分边界 | 同目录difficulty_scores.csv、stratum_profiles.csv |
| 精确风险随机对照 | 同目录matched_random_summary.csv、matched_equal_error_metrics.csv；上级random_orders.npz |
| 新增补充检验 | 同目录difficulty_contrasts.csv、matched_equal_error_comparisons.csv |
| 模型与状态诊断 | 同目录model_robustness.csv、state_recall.csv |
| 历史VOC负结果 | artifacts/voc2012_confirmation_benchmark_20260929/report_zh.md |
| 新分析执行 | scripts/analyze_sixclass_difficulty.py |
| 图表执行与可编辑SVG | scripts/plot_sixclass_difficulty.py；artifacts/sixclass_difficulty_supplement_20261004/figures/ |
| 输入冻结与程序审计 | artifacts/sixclass_difficulty_supplement_20261004/input_code_order_freeze.sha256.json、final_audit.json |

完整层结果720行包含两来源、三种unknown规则、24运行和五层；主中点下的批次、累计、随机摘要及严格错误控制各240行。最终审计确认2400目标、24运行、1000随机重复、每层类别数和风险匹配频数正确，源输入及原选样文件未改动。这一程序审计证明执行口径一致，不证明状态语义或论文结论普适。

## 附录B：投稿前表述边界

本文是实证研究稿，尚未套用具体期刊模板。作者、单位、资助、利益冲突及数据代码公开链接应由作者在选择期刊后填写；本文没有虚构这些信息。数据公开应以原始数据集ID、目标框、下载脚本和授权允许的内容为主，不能将第三方图像自动视为可再分发。

允许的结论是“当前平衡六类固定模型设置中，风险分层揭示了能力差异；覆盖选样的独立诊断收益在更严格控制后呈现部分支持”。不允许改写为“覆盖使F1必然下降”“Coverage全面优于所有基线”“线上标签经过人工验证”“当前模型排行榜普遍成立”或“六类外部验证已成功复现”。若后续加入独立六类候选池，应冻结本次风险规则与指标后再运行，新的证据再用于扩大结论。
