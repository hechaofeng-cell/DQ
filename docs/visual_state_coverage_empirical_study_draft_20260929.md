# 视觉状态覆盖能否提高深度图像分类测试充分性？一项跨类别、多模型的受控实证研究

> 完整稿版本：2026-09-29  
> 作者：待补充  
> 目标类型：软件测试/机器学习系统测试方向实证研究论文  
> 当前状态：实验、统计分析、图表与结论已冻结；作者、单位、基金信息及目标刊物格式待补充

## 摘要

深度图像分类模型通常以Accuracy、Precision、Recall和F1等聚合指标评价，但随机测试集可能重复常见视觉状态，难以说明有限预算是否覆盖小目标、遮挡、截断、非常规姿态及复杂成像条件。本文研究在模型和训练数据固定时，基于可见视觉状态的测试选择能否构造更具诊断性的测试集。主实验构建dog、cat、horse、sheep和person五类均衡候选池，每类400张，共2000张；使用19项共享视觉特征和目标尺度描述类别内部状态，并在8种ImageNet预训练分类模型、3个训练种子和五档预算下比较Coverage、Random、DeepGini、DSA和k-center。Coverage相对随机在留出失败状态、混淆类型和错误簇的预算曲线下面积（AUC）上分别提高0.1587、0.2232和0.2540。进一步逐类别匹配错误数后，混淆类型和错误簇优势缩小为0.0222和0.0115，其中仅混淆类型通过预定义两项家族内Holm校正。75%强基线加25% Coverage相对随机补充的原始优势在相同控制后未获得统计支持。线上`sensenova-6.8-flash-lite`对200张困难样本的19字段微平均精确一致率为59.61%；替换这些标签后，逐类别等错误数的混淆类型和错误簇AUC优势为0.0247和0.0167，两项Holm校正均为0.0156，且相对本地排序的变化区间均跨0。最后，在PASCAL VOC 2012构建的500张确认池中，Coverage原始混淆类型和错误簇AUC仍高于随机0.0995和0.1058，但逐类别等错误数差异为0.0008和-0.0008，均未获得统计支持。结果表明，视觉状态Coverage能够稳定改变测试分布并集中暴露困难样本；其超越随机的独立诊断价值只在主池获得部分且对10%困难标签替换稳定的支持，尚不能推广至同时改变数据来源、标注器、候选池规模和预算范围的VOC配置。

**关键词：** 深度学习测试；测试充分性；视觉状态覆盖；测试选择；失败模式；实证研究

## Abstract

Aggregate metrics such as accuracy, precision, recall, and F1 do not reveal whether a bounded image-classification test suite exercises uncommon visual states, including small targets, occlusion, truncation, unusual poses, and adverse imaging conditions. This study evaluates whether selecting tests by observable visual-state coverage produces a more diagnostic test distribution while holding the trained models fixed. The main pool contains 2,000 class-balanced candidates from dog, cat, horse, sheep, and person. Each target is represented by 19 shared visual features and target scale. We compare Coverage, Random, DeepGini, DSA, and k-center across eight ImageNet-pretrained classifiers, three training seeds, and five per-class budgets. Relative to Random, Coverage increases the area under the budget curve for held-out failed states, confusion types, and error clusters by 0.1587, 0.2232, and 0.2540. After matching the number of errors separately within each true class, the gains shrink to 0.0222 and 0.0115 for confusion types and error clusters; only the former passes Holm correction within the prespecified two-metric family. Reannotation of 200 difficult samples with the online `sensenova-6.8-flash-lite` model yields 59.61% micro exact agreement across 19 fields. Replacing these labels gives class-stratified equal-error effects of 0.0247 and 0.0167, both with Holm-adjusted p=0.0156, while their changes from the local-label ordering have confidence intervals crossing zero. In a 500-sample PASCAL VOC 2012 confirmation configuration, the raw confusion-type and error-cluster gains remain positive, but the class-stratified equal-error effects are 0.0008 and -0.0008 and are not statistically supported. Visual-state Coverage is therefore useful as an interpretable, model-independent pressure-test distribution constructor, while its diagnostic diversity per observed error remains configuration-dependent.

**Keywords:** deep learning testing; test adequacy; visual-state coverage; test selection; failure modes; empirical study

## 1 引言

深度神经网络已广泛用于视觉分类系统，但高平均准确率并不自动意味着测试充分。若测试集主要由大目标、清晰图像和常见姿态组成，即使模型取得较高F1，也可能没有充分暴露小目标、遮挡、截断或非自然视觉呈现下的缺陷。传统软件测试通过结构覆盖讨论程序的哪些部分被执行；深度学习测试研究则先后提出神经元覆盖、多粒度神经元覆盖、Surprise Adequacy等指标，用于衡量测试输入对模型内部行为的探索程度[1-3]。然而，已有研究也指出，更高的神经元覆盖不一定对应更强的缺陷发现能力[6-7]。因此，“覆盖了什么”以及“覆盖是否与失效行为相关”仍需在受控实验中直接验证。

本文从输入空间而非神经元激活空间出发，将测试图像表示为有限枚举的可见视觉状态，例如目标尺度、身体和头部可见性、朝向、动作、遮挡、截断、场景、背景复杂度、模糊、光照和视觉呈现形式。其基本假设是：在相同测试预算下，覆盖更多类别内部视觉状态，可能使测试集触达更多模型能力边界。这里的目标不是提高被测模型的F1，也不是训练新的识别模型，而是构造更有诊断性的测试集。因此，Coverage所选测试集更难、F1更低并不代表训练退化；关键问题是它是否发现了更广且不同的失败模式。

仅使用同一状态体系进行选样和评价会产生循环论证风险：某方法按状态覆盖选样，自然可能在同一状态指标上取得优势。为降低该风险，本文设计了三层评价。第一层评价视觉状态Coverage对失败状态广度的直接贡献；第二层每次屏蔽一组状态，仅使用其余四组选样，再在被屏蔽状态组上评价；第三层完全脱离视觉状态定义，评价类别混淆类型和模型表征空间错误簇。我们还将Coverage与随机、DeepGini[4]、DSA[3]及k-center[5]置于同预算条件下比较，并在8种架构、3个训练种子和5个类别上重复实验。

本文的主要贡献如下：

1. 提出并实现一种面向完整目标图像分类测试的类别内部视觉状态表示，将19项共享可见特征与目标尺度组合，用于描述测试集而非训练模型内部结构。
2. 构造类别均衡、预算连续的测试选择协议，并以失败状态触达、混淆类型和错误簇为诊断指标，区分普通分类性能与测试充分性。
3. 引入状态组留出、合并及逐类别等错误数稀释，区分同源覆盖优势、错误数量效应与错误类别构成效应。
4. 设计固定预算75/25互补实验，并用逐类别等错误数重新检验“Coverage比随机补充更有信息”与“Coverage超过完整强基线”两个主张。
5. 使用第二个线上VLM复标困难状态样本，并在500张PASCAL VOC候选池上执行预先冻结的确认实验；正结果与未复现结果均纳入结论边界。

## 2 背景与相关工作

### 2.1 深度学习测试覆盖

DeepXplore将神经元覆盖引入深度学习系统测试，并结合差分测试寻找错误行为[1]。DeepGauge进一步提出多粒度神经元级与层级测试准则[2]。这类方法借鉴传统代码覆盖思想，但神经网络内部激活并不等价于离散程序分支。Harel-Canada等人的实证研究发现，提高神经元覆盖可能伴随更少的缺陷、更不自然的输入和更强的预测偏置[6]；后续NLC从层输出分布角度重新讨论覆盖准则[7]。这些研究说明，覆盖指标必须通过缺陷发现或测试质量结果进行验证，不能仅报告覆盖数值本身。

本文与神经元覆盖的区别在于：所定义状态是可见、可解释的输入属性，而非模型内部激活。它不要求访问模型梯度或网络层，因而可跨模型复用；代价是状态依赖外部标注协议，并可能受到标注误差与特征定义边界影响。

### 2.2 测试优先级与样本选择

DeepGini根据输出概率分布的不纯度对可能误分类的测试进行排序[4]；基于距离的Surprise Adequacy（DSA）通过输入相对训练行为的“惊讶程度”刻画测试充分性[3]。二者均与被测模型相关，通常擅长将错误集中到较小预算中。k-center则在表征空间内最大化样本与已选中心的距离，广泛用于core-set和主动学习选择[5]。这些方法分别对应不确定性、训练分布偏离和几何多样性。

输入多样性已被用于黑盒测试选择，并被发现与缺陷发现相关[17]；DeepGD进一步联合不确定性与错误样本多样性，以兼顾误分类数量和不同故障的发现[18]。因此，简单地把覆盖和不确定性放在一起并不足以构成新的方法贡献。本文关注的差异是：有限枚举、可直接解释且可跨被测模型复用的视觉语义状态，能否在错误数量受控以及强基线固定预算替换条件下提供可测量的增量信息。

视觉状态Coverage的目标与上述方法并不完全相同。它不直接寻找置信度最低或嵌入距离最远的样本，而是平衡类别内部可见状态的证据数量。本文因而不预设Coverage必须超过所有模型相关方法，而是检验其能否提供独立、可解释且跨模型稳定的诊断信号。

### 2.3 本文研究空白

Input Distribution Coverage已使用生成表征和组合交互覆盖度量DNN输入测试充分性[19]。这说明“输入特征覆盖”及“状态交互”本身并非空白。本文不主张首次提出输入覆盖，而是把研究边界收窄到类别内部可见状态：在严格同预算、多模型条件下构造连续测试选择曲线，并同时接受状态留出、状态无关错误指标、等错误数控制和固定预算替换实验。本文回答的不是“覆盖越高是否必然越好”，而是视觉状态Coverage在何种意义上有效、其增量信息能否独立于错误数量、相对强基线能保留什么，以及在哪些条件下失效。

## 3 研究问题

本文设置五个研究问题：

- **RQ1：** 在同预算条件下，视觉状态Coverage是否比随机选择发现更广的失败状态？
- **RQ2：** 当评价状态不参与选样、评价改为状态无关错误指标并控制错误数量后，Coverage相对随机的诊断广度是否仍然存在？
- **RQ3：** 在固定预算中，Coverage是比随机更有信息的补充信号吗；它能否稳定超过完整的DeepGini或k-center？
- **RQ4：** 主池结论对第二VLM标注以及PASCAL VOC确认配置是否稳定？
- **RQ5：** 哪些单一或组合视觉状态与跨模型高错误率相关，不同模型在高覆盖压力测试上的表现有何差异？

## 4 研究方法

### 4.1 研究对象与数据来源

被测任务为五分类目标区域识别，类别为dog、cat、horse、sheep和person。模型输入是数据集标注框对应的目标裁剪图，而不是完整场景图。冻结训练集共1500张，每类300张；候选测试池共2000张，每类400张。候选主要来自Open Images，sheep类别另包含111张PASCAL VOC图像。COCO、Open Images和PASCAL VOC均是常用目标识别数据来源[8-10]。

候选池按类别严格均衡，以避免某一类别数量随预算增长而主导总体指标。每个类别使用相同预算 \(b\in\{20,40,60,80,100\}\)，因此总测试规模依次为100、200、300、400和500张。候选池包含2000个唯一源图。进一步的SHA-256、64位pHash和ConvNeXt-Tiny嵌入联合审计未发现候选池内部或候选池—训练集之间满足预设阈值的重复对。

### 4.2 视觉状态协议

状态协议包含19项类别共享特征：身体可见性、头部可见性、身体朝向、头部朝向、姿态动作、腿部可见性、遮挡程度、遮挡物类型、截断程度、截断部位、群体上下文、交互上下文、场景环境、背景复杂度、模糊程度、光照、目标—背景对比度、目标可识别性和视觉呈现形式。目标尺度由标注框面积占图像面积的比例派生：

\[
\operatorname{scale}(x)=
\begin{cases}
\text{tiny}, & r<0.01,\\
\text{small}, & 0.01\le r<0.03,\\
\text{medium}, & 0.03\le r<0.10,\\
\text{large}, & r\ge0.10.
\end{cases}
\]

19项状态由本地视觉语言模型（vision-language model，VLM）`qwen3-vl:30b-a3b-instruct`结构化标注。模型接收原图、标记框图和裁剪图，并被要求只描述可见事实，不推断品种、身份、年龄、性别、种族、健康或性格。2000张均通过枚举值、字段完整性和样本ID一致性的程序校验。不同类别的可行状态数因候选池实际出现状态而异：dog为91、cat为88、horse为90、sheep为92、person为94。这里的“可行”只表示在当前候选池出现过，不代表自然世界状态全集。

### 4.3 覆盖度与Coverage选样

对类别 \(c\) 的候选测试子集 \(S_c\)，设可行状态集合为 \(V_c\)，状态 \(v\) 在子集中的支持度为 \(n_v(S_c)\)。本文采用带上限的频数覆盖：

\[
C_c(S_c)=\frac{1}{|V_c|}\sum_{v\in V_c}\min\left(\frac{n_v(S_c)}{\tau},1\right),
\]

其中饱和阈值 \(\tau=20\)。总体覆盖为五个类别覆盖的宏平均。该指标不仅要求状态出现，还要求获得一定重复证据，同时通过饱和上限降低常见状态继续累积的收益。

Coverage使用确定性贪心策略构造完整排序。在当前已选集合的计数为 \(n_v\) 时，候选样本 \(x\) 的增益为：

\[
g(x\mid S)=\sum_{v\in x,\,n_v<\tau}
\left[\log(n_v+2)-\log(n_v+1)\right].
\]

每轮选择增益最大的样本，并以样本ID稳定打破并列。对数边际收益使稀缺状态优先，同时允许对同一状态逐步积累证据。

### 4.4 对照方法

实验比较五种同预算选择方法：

| 方法 | 选择信号 | 是否依赖被测模型 |
|---|---|---|
| Coverage | 可见视觉状态的凹收益覆盖 | 否 |
| Random | 类别内均匀随机，1000次重复 | 否 |
| DeepGini | 输出概率的Gini不纯度 | 是 |
| DSA | 候选相对训练表征的距离比 | 是 |
| k-center | ImageNet预训练ViT-B/16嵌入的最远优先 | 否（但依赖外部表征） |

DeepGini和DSA分别代表错误集中与训练分布惊讶度；k-center代表强表征多样性基线。所有方法均在每个类别内部独立选择相同数量样本。

### 4.5 被测模型与冻结协议

实验使用ResNet-18、ResNet-50、MobileNetV3-Small、MobileNetV3-Large、DenseNet-121、EfficientNet-B0、ConvNeXt-Tiny和Swin-T，覆盖经典残差网络、移动端网络、密集连接网络、复合缩放网络、现代卷积网络和层次化Transformer[11-16]。所有模型使用ImageNet预训练权重，在相同五分类训练协议下训练：20轮、每轮类别平衡抽样3000个样本、batch size 64、AdamW优化器、学习率 \(10^{-4}\)、权重衰减 \(10^{-4}\)，并使用余弦退火。每种架构运行三个随机种子：20260916、20260917和20260918。

本研究的测试选择阶段不重新训练模型。训练数据、权重、预处理、候选池、标签及模型预测全部冻结，仅改变进入测试套件的样本及其预算。因此不同选择方法下的F1变化表示测试套件难度变化，不表示模型性能被选择方法“提高”或“降低”。

### 4.6 评价指标

#### 4.6.1 分类指标

报告Accuracy、Macro-Precision、Macro-Recall、Macro-F1和类别指标。它们用于描述所选测试套件上测得的性能，但不是本文主要测试充分性指标。

#### 4.6.2 失败状态广度

若状态 \(v\) 在当前类别测试子集中至少出现3次，且这些样本中至少有一次分类错误，则记为已触达失败状态。实际程序以当前类别候选池中出现过的固定状态全集 \(V_c\) 为分母，类别失败状态广度定义为：

\[
F_c(S_c)=\frac{|\{v:n_v(S_c)\ge3\land e_v(S_c)>0\}|}
{|V_c|}.
\]

五类宏平均作为主要指标。当前子集中支持度不足3的状态对分子贡献为0；可评价状态数量另行报告，但不作为分母。因此该指标同时反映测试套件对候选状态空间的触达与错误暴露，理论上可能低于100%。与错误率相比，它关注错误分布的广度，而非仅关注错误数量。

#### 4.6.3 状态留出迁移

将20项分析特征划分为五组：尺度与可见性、朝向与姿态、遮挡与截断、上下文与场景、图像质量与视觉呈现。每一折中，Coverage仅能访问其中四组，第五组完全不参与选样；随后在第五组上计算失败状态广度。五组轮换后取平均。

#### 4.6.4 状态无关错误指标

混淆类型覆盖统计每个真实类别所暴露的不同错误预测类别比例，例如dog→cat与dog→person是两个不同混淆类型。错误簇覆盖则先在完整候选池中收集每个模型的错误样本，再使用模型倒数第二层表征进行KMeans聚类，最后统计测试套件触达的错误簇比例。主实验的簇数依据错误量设置，上限为每类8簇；敏感性分析固定每类 \(K\in\{4,6,8,10\}\)。聚类在方法评价前完成，且不使用VLM状态。

#### 4.6.5 等错误数稀释

为排除“Coverage只是发现了更多错误，所以自然触达更多错误类型”的解释，本文对每个模型、种子、预算及基线进行成对错误数匹配。五类错误合并计算，但错误标签保留真实类别，因此dog→cat与horse→cat属于不同混淆类型。设某测试套件发现 \(N\) 个错误，错误类型 \(j\) 出现 \(m_j\) 次，从中无放回抽取 \(k\) 个错误时，期望触达类型比例为：

\[
D(E,k)=\frac{1}{|T|}\sum_{j\in T}\left[1-\frac{\binom{N-m_j}{k}}{\binom{N}{k}}\right].
\]

每次比较取 \(k=\min(N_{Coverage},N_{baseline})\)。该式直接计算随机稀释的期望，不依赖额外Monte Carlo抽样；混淆类型与错误簇分别计算。随机基线仍对1000个类别均衡排序取平均。

主实验完成后进一步冻结逐类别等错误数控制。对真实类别 \(c\) 分别取 \(k_c=\min(N_{Coverage,c},N_{baseline,c})\)，在类内按该式计算期望覆盖，再对五类宏平均。若 \(k_c=0\)，该类双方均按0计入，且另行报告可评价类别比例。该控制同时排除总体错误数和错误类别构成不同这两种替代解释。

#### 4.6.6 固定预算互补组

互补实验保持每类总预算不变，将75%预算分配给DeepGini或k-center，再分别以Coverage和随机排序补足25%，并在类别内去重。比较包括：Coverage25与Random25，用于判断视觉状态是否比任意替换更有信息；Coverage25与100%强基线，用于判断替换是否带来净收益。75/25比例、五档预算和主要状态无关指标在运行前固定，不依据结果调节。两组比较随后按4.6.5节的逐类别等错误数规则复算，以检验原始差异是否仅由各类别错误数量不同产生。

#### 4.6.7 状态组合

预先限定三组二元组合：小目标×中重度遮挡、非常规姿态×中重度截断、模糊×低对比度。对分类错误拟合二项广义估计方程，包含类别、模型架构和训练种子固定效应，并以图片ID作为重复观测组。交互优势比仅用于判断联合状态与乘法主效应是否存在额外关联，不作因果解释。若四个状态单元不完整，则该交互不可估计。

#### 4.6.8 第二标注器与VOC确认配置

为量化状态协议对标注模型的依赖，从主池冻结的Coverage顺序中每类取前40张，共200张困难状态样本，使用线上`sensenova-6.8-flash-lite`按同一`object_state_v1.0`协议和相同三视图重新标注。enum字段报告精确一致率与Cohen's kappa，set字段报告精确一致率与Jaccard；随后用线上结果替换这200张标签，重新生成Coverage顺序，并复算原始及逐类别等错误数状态无关指标。该设计不把任一VLM视为人工真值，只检验困难子集上的标注器一致性，以及Coverage—Random比较对10%标签替换的敏感性。

确认池从PASCAL VOC 2012 trainval构建，五类各100张，共500张，类别之间不复用源图，并排除与主池原图及训练裁剪的SHA-256精确重复。状态全部由上述线上VLM标注。冻结原8种架构、3个训练种子和分类权重，仅对新池推理；每类预算预先设为10、20、40、60和80。确认主指标为逐类别等错误数混淆类型与错误簇AUC。若某类全池只有1个错误，则定义为单例错误簇；若没有错误，则该类簇覆盖按0计并报告不可评价比例。该配置同时改变了数据来源、状态标注器、候选池规模和预算范围，因而用于检验结论能否在整套外部配置中复现，不能单独识别其中某一变化的影响。

### 4.7 统计分析

主池曲线使用从0到每类100张预算的归一化梯形AUC，VOC确认使用0到每类80张。比较单位为24个“模型架构—训练种子”运行。效应量是成对AUC差。95%置信区间使用分层bootstrap：先重采样8种架构，再在每种架构内重采样3个种子，共5000次。显著性检验对8个架构均值执行精确配对符号翻转检验。直接实验在四个基线内进行Holm校正；解耦实验同时报告指标族内校正和全部12项比较的全局校正。合并及逐类别等错误数的主家族均包含Coverage相对Random的两项状态无关指标；互补实验主家族包含两种强基线上的四项Coverage25相对Random25比较；VOC确认家族包含两项逐类别等错误数指标。随机基线均使用1000条类别均衡排序。

## 5 实验结果

### 5.1 RQ1：Coverage是否比随机选择发现更广的失败状态？

在每类100张、总计500张的终点预算下，Coverage的状态覆盖为71.62%，高于随机选择的60.27%；失败状态广度为82.88%，高于随机的61.56%。与此同时，Coverage测试套件错误率为18.67%，高于随机的9.93%，Macro-F1为81.39%，低于随机的90.07%。这不是模型退化，而是Coverage选出了更具挑战性的测试样本。

| 方法 | 状态覆盖 | 失败状态广度 | 错误率 | Macro-F1 |
|---|---:|---:|---:|---:|
| Coverage | 71.62% | 82.88% | 18.67% | 81.39% |
| Random | 60.27% | 61.56% | 9.93% | 90.07% |
| DeepGini | 63.48% | 80.42% | 30.36% | 69.71% |
| DSA | 64.19% | 80.53% | 28.34% | 71.55% |
| k-center | 63.19% | 77.94% | 19.17% | 80.93% |

在完整状态体系既参与选样又参与评价时，Coverage相对随机的失败状态广度AUC提高0.2396，95% CI为[0.2267, 0.2514]，24/24个运行均为正向，Holm校正后 \(p=0.0312\)。在dog、cat、horse、sheep和person五个类别上，Coverage相对随机的AUC差分别为0.2349、0.2631、0.2486、0.2481和0.2034，均为24/24运行正向。

该结果支持RQ1，但由于同一状态体系同时参与选样和评价，只能视为直接有效性证据，不能单独证明独立诊断价值。

### 5.2 RQ2：Coverage是否具有独立诊断价值？

在五折状态组留出条件下，Coverage相对随机的留出失败状态广度AUC提高0.1587，95% CI为[0.1422, 0.1749]；混淆类型覆盖AUC提高0.2232，95% CI为[0.1777, 0.2693]；错误簇覆盖AUC提高0.2540，95% CI为[0.2243, 0.2838]。三项指标均在24/24个运行中为正向，指标族内Holm校正 \(p=0.0312\)。

| 独立评价指标 | Coverage−Random AUC | 95% CI | 正向运行 | 指标族Holm p |
|---|---:|---:|---:|---:|
| 留出失败状态广度 | +0.1587 | [0.1422, 0.1749] | 24/24 | 0.0312 |
| 混淆类型覆盖 | +0.2232 | [0.1777, 0.2693] | 24/24 | 0.0312 |
| 错误簇覆盖 | +0.2540 | [0.2243, 0.2838] | 24/24 | 0.0312 |

五个留出组均复现Coverage相对随机的优势：尺度与可见性为+0.1912，朝向与姿态为+0.1064，遮挡与截断为+0.2181，上下文与场景为+0.1811，图像质量与视觉呈现为+0.0965；每组均为24/24运行正向，跨五组Holm校正 \(p=0.0391\)。这表明Coverage并非只会提高它直接优化的状态计数，还能迁移到未参与选样的状态组，并触达更多独立定义的混淆类型和错误簇。

需要注意，若将三项指标与四个基线构成的12项比较进行一次全局Holm校正，最小校正值为0.0938，未低于0.05。因此，本文将“相对随机具有独立诊断价值”表述为跨运行稳定且在预定义指标族内显著的证据，而不将其表述为通过所有多重比较口径的最终确认。

#### 5.2.1 等错误数控制

状态无关指标的原始优势仍可能来自Coverage测试套件包含更多错误。为此，本文在每个模型、种子和预算内匹配Coverage与基线的错误数量，再计算期望混淆类型和错误簇覆盖。相对随机基线，Coverage的等错误数混淆类型AUC高0.0371，95% CI为[0.0246, 0.0484]，23/24个运行为正；错误簇AUC高0.0191，95% CI为[0.0076, 0.0296]，20/24个运行为正。两项主要比较的Holm校正值均为0.0156。

| 等错误数指标 | Coverage−Random AUC | 95% CI | 正向运行 | Holm p |
|---|---:|---:|---:|---:|
| 混淆类型覆盖 | +0.0371 | [0.0246, 0.0484] | 23/24 | 0.0156 |
| 错误簇覆盖 | +0.0191 | [0.0076, 0.0296] | 20/24 | 0.0156 |

在每类100张的终点，相对随机比较时两者平均匹配49.7个错误；Coverage的期望混淆覆盖为82.06%，随机为77.82%，期望错误簇覆盖为78.17%和75.67%。相对DeepGini、DSA和k-center的终点点估计也均为正，但完整曲线结果并不一致：Coverage相对DSA的两项95% CI下界为正，相对DeepGini和k-center则跨0。由于这些强基线比较属于次要分析，本文不据此声称Coverage具有统一优势。

合并五类进行错误数匹配仍可能受到错误类别构成影响。逐类别分别匹配后，Coverage相对随机的混淆类型AUC优势降至0.0222，95% CI为[0.0135, 0.0301]，21/24个运行为正，家族内Holm校正 \(p=0.0156\)；错误簇优势降至0.0115，95% CI为[0.0008, 0.0215]，18/24个运行为正，但Holm校正 \(p=0.0547\)。在同时校正Coverage对四个基线的8项逐类别比较时，两项全局Holm值分别为0.0625和0.2734。

| 逐类别等错误数指标 | Coverage−Random AUC | 95% CI | 正向运行 | 家族Holm p | 全局Holm p |
|---|---:|---:|---:|---:|---:|
| 混淆类型覆盖 | +0.0222 | [0.0135, 0.0301] | 21/24 | 0.0156 | 0.0625 |
| 错误簇覆盖 | +0.0115 | [0.0008, 0.0215] | 18/24 | 0.0547 | 0.2734 |

因此，RQ2只获得部分支持：在预定义的两项主指标家族内，主候选池的混淆类型结果未被总体错误数及其类别构成完全解释；错误簇证据在逐类别和多重比较控制后处于边界，不能写成两项均稳定独立。扩大到全部8项逐类别比较时，混淆类型的全局Holm值为0.0625，进一步限制了该结论的强度。

![解耦诊断曲线](../artifacts/cross_state_decoupled_diagnostics_20260928/figures/decoupled_diagnostic_curves.png)

### 5.3 RQ3（强基线比较）：Coverage相对强基线的作用边界

解耦评价改变了对Coverage的判断。在同源状态指标上，Coverage相对k-center的失败状态广度AUC高0.0609；但在状态组留出后，Coverage低0.0195，95% CI为[-0.0288, -0.0102]。对于混淆类型覆盖和错误簇覆盖，Coverage相对k-center的AUC分别为-0.0161和-0.0268。Coverage也低于DeepGini和DSA的三项独立诊断指标。

| 指标 | Coverage−k-center AUC | Coverage−DeepGini AUC | Coverage−DSA AUC |
|---|---:|---:|---:|
| 留出失败状态广度 | -0.0195 | -0.0701 | -0.0670 |
| 混淆类型覆盖 | -0.0161 | -0.0915 | -0.0674 |
| 错误簇覆盖 | -0.0268 | -0.1166 | -0.0999 |

终点预算也呈现相同关系。每类100张时，Coverage的留出失败状态广度、混淆覆盖和错误簇覆盖分别为76.45%、95.03%和94.48%；k-center分别为78.39%、95.76%和95.78%；DeepGini分别为80.74%、98.33%和98.82%。DeepGini通过30.36%的高错误率集中发现错误，Coverage的错误率为18.01%，与k-center的19.17%接近。

因此，Coverage的合理定位不是“优于所有测试选择算法”，而是一个可解释、与被测模型无关、在主池原始口径下明显高于随机的输入语义维度。模型相关不确定性方法更适合在低预算下集中错误，表征多样性方法更适合覆盖模型行为几何。三类信号是否具有可操作的互补性由5.5节进一步检验。

### 5.4 错误簇数量敏感性

固定候选池、预测、选样顺序和预算，仅将每类错误簇数量设为4、6、8和10。Coverage相对Random的错误簇覆盖AUC优势分别为0.2261、0.2480、0.2509和0.2405，四个95%置信区间下界均为正，且每个设置均为24/24运行正向。Coverage相对k-center的点估计在四个设置下均为负，其中 \(K=4\) 和 \(K=10\) 的95%置信区间完全低于0， \(K=6\) 和 \(K=8\) 跨越0。

| 每类错误簇K | Coverage−Random AUC | 95% CI | Coverage−k-center AUC | 95% CI |
|---:|---:|---:|---:|---:|
| 4 | +0.2261 | [0.1928, 0.2605] | -0.0196 | [-0.0394, -0.0012] |
| 6 | +0.2480 | [0.2186, 0.2777] | -0.0185 | [-0.0401, 0.0020] |
| 8 | +0.2509 | [0.2359, 0.2675] | -0.0223 | [-0.0484, 0.0023] |
| 10 | +0.2405 | [0.2188, 0.2596] | -0.0343 | [-0.0543, -0.0120] |

该结果说明“Coverage优于随机、但没有超过k-center”的方向不依赖单一错误簇数量；但不能声称Coverage在每一个K上都显著弱于k-center。

![错误簇数量敏感性](../artifacts/error_cluster_sensitivity_20260929/figures/error_cluster_sensitivity.png)

### 5.5 RQ3（互补检验）：Coverage能否作为强基线的增量信号？

固定每类预算后，Coverage25表示75%强基线加25% Coverage，Random25表示75%强基线加25%随机样本。相对Random25，Coverage25在DeepGini组合中的混淆类型和错误簇AUC分别提高0.0113和0.0142，在k-center组合中分别提高0.0359和0.0360；四项95% CI下界均为正，家族内Holm校正值均为0.0312。

| 强基线 | 指标 | Coverage25−Random25 AUC | 95% CI | 正向运行 | Holm p |
|---|---|---:|---:|---:|---:|
| DeepGini | 混淆类型 | +0.0113 | [0.0045, 0.0206] | 16/24 | 0.0312 |
| DeepGini | 错误簇 | +0.0142 | [0.0072, 0.0218] | 20/24 | 0.0312 |
| k-center | 混淆类型 | +0.0359 | [0.0161, 0.0569] | 21/24 | 0.0312 |
| k-center | 错误簇 | +0.0360 | [0.0248, 0.0478] | 22/24 | 0.0312 |

在每类100张时，DeepGini、DeepGini+Coverage25和DeepGini+Random25的错误簇覆盖分别为98.82%、99.67%和98.61%；k-center三组分别为95.78%、95.54%和94.67%。Coverage25还将状态覆盖从DeepGini的63.48%提高到66.54%，从k-center的63.19%提高到66.38%。按原始错误覆盖口径，Coverage补充优于随机补充。

但逐类别匹配错误数后，四项Coverage25−Random25差异均未达到显著：DeepGini组合的混淆类型和错误簇AUC差为0.0045和0.0060，k-center组合为0.0027和-0.0024；四项95% CI均跨0，家族内Holm校正值为0.2812至0.7656。

| 强基线 | 逐类别等错误数指标 | AUC差值 | 95% CI | 正向运行 | Holm p |
|---|---|---:|---:|---:|---:|
| DeepGini | 混淆类型 | +0.0045 | [-0.0009, 0.0102] | 14/24 | 0.3281 |
| DeepGini | 错误簇 | +0.0060 | [-0.0003, 0.0124] | 17/24 | 0.2812 |
| k-center | 混淆类型 | +0.0027 | [-0.0101, 0.0159] | 14/24 | 0.7656 |
| k-center | 错误簇 | -0.0024 | [-0.0091, 0.0048] | 11/24 | 0.7656 |

相对100%强基线，Coverage25的四项状态无关指标比较在家族内Holm校正后同样均未达到0.05；其中DeepGini组合的错误簇AUC点估计为-0.0098，未校正95% CI为[-0.0158, -0.0033]。终点的新发现—丢失分析显示，Coverage25平均为DeepGini新增0.33个、丢失0.12个错误簇，而对k-center新增和丢失均为0.50个。综合严格控制，RQ3不获支持：原始覆盖差异表明语义补充改变了测试组成，但当前实验未获得其在相同类别错误数下优于随机补充或超过完整强基线的证据。这里的未显著结果不表示方法等效。

![等错误数与固定预算互补结果](../artifacts/coverage_incremental_value_20260929/figures/incremental_value_summary.png)

### 5.6 RQ4：结论对标注器和VOC确认配置是否稳定？

#### 5.6.1 第二VLM复标

线上`sensenova-6.8-flash-lite`对每类40张、本地Coverage排序前列的困难样本完成200/200严格结构化复标。19个字段共3800个样本—字段实例的微平均精确一致率为59.61%；17个enum字段的宏平均精确一致率为62.82%，宏平均Cohen's kappa为0.444；两个set字段的宏平均精确一致率为32.25%，宏平均Jaccard为0.432。字段差异较大：视觉呈现形式一致率为96.0%，而截断部位、遮挡物类型和截断程度分别为28.5%、36.0%和42.5%。这说明严格JSON成功不等于状态语义稳定，特别是多标签部位和遮挡解释具有明显标注器依赖。

用线上结果替换完整2000池中的这200张标签并重新生成Coverage顺序后，前20/40/60/80/100张的类别宏平均集合Jaccard分别为21.67%、36.11%、52.75%、62.70%和71.02%，完整排序平均Kendall \(\tau=0.815\)。虽然低预算具体样本变化明显，Coverage相对随机的混淆类型和错误簇AUC仍为+0.2343和+0.2598，均24/24运行正向；相对原本地VLM排序的AUC变化为+0.0111和+0.0058，两项95% CI均跨0。由此可见，具体状态标签和早期排序对VLM敏感，但主池相对随机的原始独立指标方向对10%困难标签替换较稳定。

逐类别分别匹配错误数量后，线上合并Coverage相对随机的混淆类型AUC优势为+0.0247，95% CI为[0.0156, 0.0325]，22/24个运行正向；错误簇优势为+0.0167，95% CI为[0.0066, 0.0271]，19/24个运行正向。两项家族内Holm校正值均为0.0156。相对本地VLM排序，两项AUC变化分别为+0.0025和+0.0053，95% CI分别为[-0.0074, 0.0106]和[-0.0022, 0.0130]。因此，主池逐类别等错误数优势对这次10%困难标签替换保持稳定；变化区间跨0表示未观察到稳定差异，不表示两种标注等效。

| 逐类别等错误数敏感性 | AUC差值 | 95% CI | 正向运行 | Holm p |
|---|---:|---:|---:|---:|
| 线上合并Coverage−Random：混淆类型 | +0.0247 | [0.0156, 0.0325] | 22/24 | 0.0156 |
| 线上合并Coverage−Random：错误簇 | +0.0167 | [0.0066, 0.0271] | 19/24 | 0.0156 |
| 线上合并−本地Coverage：混淆类型 | +0.0025 | [-0.0074, 0.0106] | 14/24 | 未作主家族校正 |
| 线上合并−本地Coverage：错误簇 | +0.0053 | [-0.0022, 0.0130] | 17/24 | 未作主家族校正 |

![线上与本地VLM一致性](../artifacts/online_vlm_reannotation_20260929/analysis/online_vlm_agreement.png)

![逐类别等错误数的VLM标签敏感性](../artifacts/online_vlm_reannotation_20260929/analysis/online_class_equal_error_sensitivity.png)

#### 5.6.2 PASCAL VOC确认配置

跨源池包含PASCAL VOC 2012的500个唯一目标，每类100张；构建阶段排除了46个与现有候选或训练裁剪精确重复的候选，最终原图、裁剪和源图ID均为500/500唯一。线上VLM完成500/500状态标注。冻结分类器在每类80张终点时，Coverage错误率为15.96%，随机为14.59%，Macro-F1分别为84.17%和85.52%。Coverage相对随机的原始混淆类型和错误簇AUC高0.0995和0.1058，两项均24/24运行正向。

然而，逐类别匹配错误数后，混淆类型AUC差仅为+0.0008，95% CI为[-0.0110, 0.0124]，15/24运行正向；错误簇AUC差为-0.0008，95% CI为[-0.0140, 0.0122]，11/24运行正向。两项家族内Holm校正值均为1.0。

| VOC确认指标 | Coverage−Random AUC | 95% CI | 正向运行 | Holm p |
|---|---:|---:|---:|---:|
| 原始混淆类型覆盖 | +0.0995 | [0.0789, 0.1216] | 24/24 | 未作为确认主家族 |
| 原始错误簇覆盖 | +0.1058 | [0.0807, 0.1330] | 24/24 | 未作为确认主家族 |
| 逐类别等错误数混淆类型 | +0.0008 | [-0.0110, 0.0124] | 15/24 | 1.0000 |
| 逐类别等错误数错误簇 | -0.0008 | [-0.0140, 0.0122] | 11/24 | 1.0000 |

该确认配置不支持“Coverage在相同类别错误数量下仍发现更多错误类型”的可泛化主张。它支持的更窄结论是：Coverage在两个池中都选择了更易暴露错误和原始错误模式的样本；但在VOC配置中，匹配各类别错误数量后没有观察到可确认的剩余类型广度优势。置信区间跨0表示未复现，而不是证明Coverage与Random等效，也不能证明原始优势全部由错误数量导致。

![跨源确认曲线](../artifacts/voc2012_confirmation_benchmark_20260929/figures/confirmation_curves.png)

### 5.7 RQ5：模型排序、失败状态与状态组合

在Coverage每类100张、总计500张的同一高覆盖测试集上，ConvNeXt-Tiny的三种子平均Macro-F1最高，为89.22%；其后依次为Swin-T、ResNet-50、EfficientNet-B0、DenseNet-121、MobileNetV3-Large、ResNet-18和MobileNetV3-Small。该排序仅描述当前训练协议和候选池下的视觉状态鲁棒性，不是通用模型排行榜。

| 排名 | 模型 | Macro-F1 | 错误率 | 失败状态广度 |
|---:|---|---:|---:|---:|
| 1 | ConvNeXt-Tiny | 89.22% ± 1.14% | 10.80% | 72.64% |
| 2 | Swin-T | 85.87% ± 1.28% | 14.13% | 77.45% |
| 3 | ResNet-50 | 83.64% ± 0.71% | 16.53% | 83.61% |
| 4 | EfficientNet-B0 | 81.95% ± 1.33% | 18.00% | 82.23% |
| 5 | DenseNet-121 | 79.62% ± 1.78% | 20.47% | 85.94% |
| 6 | MobileNetV3-Large | 79.25% ± 2.24% | 20.73% | 84.90% |
| 7 | ResNet-18 | 76.05% ± 2.45% | 23.93% | 87.93% |
| 8 | MobileNetV3-Small | 75.52% ± 1.09% | 24.73% | 88.38% |

对完整2000张候选池聚合24组预测后，总体错误率为9.91%。小目标组错误率为19.84%，关键部位不可见为14.02%，中重度遮挡为11.89%，中重度截断为11.27%，动态或非常规姿态为10.50%，复杂场景或不利成像质量为11.93%。这些组允许重叠，不能直接相加。

![挑战组错误率](../artifacts/multiclass_failure_atlas_20260929/figures/challenge_group_error_rates.png)

支持度不少于20张的状态中，错误率最高的是低可识别性31.28%、严重模糊30.04%、重度遮挡28.15%、雕像或玩具26.45%、头部不可见24.16%和极小目标23.54%。候选池混淆矩阵显示person的聚合召回最低，为82.9%，主要被预测为dog和horse；案例图进一步显示，骑乘、人与动物重叠以及只包含人体局部的裁剪会造成稳定混淆。

![聚合混淆矩阵](../artifacts/multiclass_failure_atlas_20260929/figures/aggregate_confusion_matrix.png)

![典型失败案例](../artifacts/multiclass_failure_atlas_20260929/figures/failure_case_atlas.png)

预定义状态组合显示，小目标且中重度遮挡样本的聚合错误率为23.07%，高于仅小目标的17.76%、仅中重度遮挡的9.59%和两者均无的6.46%；但在控制类别、模型架构和种子后，交互优势比为0.808，95% CI为[0.519, 1.258]，未显示超出乘法主效应的额外关联。非常规姿态×中重度截断的交互优势比为1.392，95% CI为[0.839, 2.310]，同样未达到显著。模糊且低对比度的8张图片聚合错误率为51.56%，但候选池不存在“清晰且低对比度”样本，交互项不可估计。状态组合可以描述困难样本，却没有在当前数据中形成稳定的新交互规律。

![状态组合与错误率](../artifacts/coverage_incremental_value_20260929/figures/state_interaction_error_rates.png)

## 6 讨论

### 6.1 为什么覆盖提高时F1可能下降

随机测试集更接近候选池的高频区域，因此容易重复抽到清晰、大目标和常见姿态。Coverage主动提高低频视觉状态的证据数量，可能引入更多小目标、遮挡、截断、模糊和非自然视觉呈现样本。在主池的同预算比较中，冻结模型在Coverage套件上的F1更低、错误率更高；这描述的是所选测试套件更具挑战性，而不是模型参数变差。

反过来，覆盖提高时Precision或Recall局部上升也不矛盾。加入样本会同时改变真实类别构成、各类预测数量和错误方向；当新增样本纠正某类预测分母或带来较易识别的新状态时，Precision可能上升。测试充分性曲线因此不应被预设为单调下降或“先升后降”，其形状是模型、类别、状态组合和预算共同作用的经验结果。

### 6.2 Coverage的独立价值与局限

Coverage最稳定的证据是它能在固定预算下改变测试组成、提高状态覆盖并选出更多困难样本。主池的留出状态、混淆类型和错误簇均优于随机；逐类别匹配错误数后，混淆类型仍得到家族内支持，而错误簇处于边界。这说明视觉状态并非完全脱离模型行为的装饰性标签，但“独立诊断多样性”只获得部分而非全面支持。

VOC确认配置进一步限定了机制：Coverage仍比随机触达更多原始混淆和错误簇，但逐类别匹配错误数后差异不再获得统计支持。因此，视觉状态Coverage目前最可靠的作用是构造可解释的压力测试分布；现有证据不足以说明它在不同配置下都能提高每个错误所携带的诊断信息。DeepGini和DSA更贴近具体模型的错误边界，k-center以高维表征捕获协议之外的外观差异；Coverage的价值是模型无关和可解释，但当前证据不支持它取代这些强基线，也未支持75/25混合组在严格控制后优于随机补充。

### 6.3 对模型测试实践的启示

第一，仅报告总体F1不足以描述测试集质量。测试报告应同时给出类别均衡性、视觉状态分布、失败状态广度和典型错误图谱。第二，测试选择方法必须与同预算随机和至少一种强多样性/不确定性基线比较；否则更高覆盖可能只是确定性选样相对随机波动的结果。第三，Coverage评价需防止同源循环，至少采用状态组留出或状态无关错误指标。第四，模型排名取决于压力测试分布：在当前高覆盖测试集上，ConvNeXt-Tiny排名第一，但排名不能外推为所有视觉任务的通用结论。

第五，当一个方法选择出更多错误时，应按真实类别分别进行等错误数稀释，避免总体匹配掩盖错误类别构成差异。第六，“原始错误覆盖更高”不等于“单位错误的信息量更高”，二者需要分别报告。第七，自动状态协议至少应进行第二标注器一致性和标签替换敏感性分析；结构校验成功不能替代语义稳定性证据。

## 7 有效性威胁

**构念效度。** 视觉状态体系是有限枚举，不能覆盖纹理、细粒度外观和所有跨状态交互。失败状态广度以固定候选状态全集为分母，将支持度至少3且出现一次错误作为触达，能够衡量广度但不衡量严重程度；支持度阈值3和频数饱和阈值20也属于设计选择。本文通过独立混淆、错误簇与等错误数控制降低但不能消除该威胁。

**内部效度。** 所有模型、训练数据、预测、预算和候选池均冻结；类别内等预算减少了类别比例混杂。候选池与训练集的精确重复和近重复审计未发现泄漏。不过，错误簇由被测模型表征生成，不能视为客观语义真值；不同模型的错误簇含义也可能不同。

**标注效度。** 主池视觉状态由本地VLM生成，VOC池由线上`sensenova-6.8-flash-lite`生成，两者均完成结构与枚举校验。困难子集的字段微平均一致率仅59.61%、enum宏平均kappa为0.444，表明状态边界具有明显标注器依赖；另一方面，替换这10%标签后，主池相对随机的原始指标方向保持，逐类别等错误数的混淆和错误簇优势也均通过预定义家族校正。该一致率不表示任一VLM的准确率，困难子集也不能代表全池平均一致性；敏感性结论仅适用于10%困难样本替换，不能证明全池或任意标注器下稳定。目标类别继承公开数据集标注。

**外部效度。** 研究覆盖Open Images为主的2000张池和PASCAL VOC 2012的500张确认池，但仅涉及五类完整目标、目标框裁剪分类和八种ImageNet预训练架构。VOC配置同时改变了数据来源、VLM标注器、池规模和预算范围，逐类别等错误数优势未复现；因此不能把差异归因于某一个因素，也不能把主池的独立诊断价值外推到其他配置。face属于局部目标，未混入主实验；研究尚未覆盖检测、分割、开放集识别、视频或独立机构数据。

**统计结论效度。** 24个运行来自8种架构和3个种子，不是24个完全独立架构。本文使用架构分层bootstrap与架构级符号翻转检验，并同时报告效应量、置信区间和方向一致性。主池解耦实验在各指标族内显著，但全局12项校正的最小值为0.0938；逐类别等错误数下仅混淆类型通过两项家族校正，错误簇与四项互补比较未通过。VOC确认配置的两项主指标均未通过。三组状态交互中两组区间较宽，另一组因缺少状态单元不可估计。

## 8 结论

本文以五类完整目标、八种模型、三个训练种子、两个候选池和五档类别均衡预算，检验输入视觉状态Coverage在深度图像分类测试中的作用。主池中，Coverage相对随机稳定提高留出失败状态、混淆类型和错误簇广度；逐类别等错误数后，仅混淆类型优势通过本地标签下的预定义家族校正。75/25互补组在同一严格控制后未获得优于随机补充或完整强基线的证据。第二VLM显示状态标签一致性有限，但替换10%困难标签后，逐类别等错误数的混淆和错误簇优势均通过家族校正，且相对本地排序的变化区间跨0。VOC配置中，Coverage仍选择出更多错误和原始错误模式，但逐类别等错误数后的类型广度优势未复现。

本文的核心结论是：**视觉状态Coverage是一种模型无关、可解释的压力测试分布构造方法，能够在当前两个候选池中增加困难状态和原始失效暴露；它在主池对随机的独立混淆多样性获得部分支持，但该优势未在同时改变数据来源、标注器、池规模和预算范围的VOC配置中复现。** 覆盖率升高既不表示模型性能提高，也不要求测得的F1单调下降。正结果与未复现结果共同界定了视觉状态Coverage在图像分类测试中的用途和证据边界。

## 9 后续工作

后续研究可采用因子化设计逐一控制数据来源、标注器、池规模和预算范围，以解释VOC配置未复现的来源；第三个独立数据源应在运行前冻结这些因素和判定标准。若扩展标注可靠性，可采用小规模独立仲裁金标准或多VLM共识。对于状态组合，应补足缺失单元和低支持联合状态，而不是依据当前结果删除不利组合。

## 10 数据、代码与模型使用说明

本文实验的候选清单、状态标注、模型预测、选样顺序、中间统计表、原始线上VLM响应、分析脚本和图表均保存在附录A列出的项目目录中。统一审计脚本核对了9229条补强结果记录，并对正文关键数值、两批线上VLM完成数量、在线标签替换敏感性和确认池唯一性进行检查。投稿冻结包另保存运行环境、核心依赖版本、关键文件SHA-256和复现命令。公开发布时应提供数据集图片ID、目标框、下载与预处理脚本、状态协议、随机种子和环境清单；受原始数据集许可约束的图像不应直接重新分发。

线上VLM仅用于生成可见状态标签，没有接收被测分类器预测或本文统计结果。调用过程保存原始响应与解析结果；因图像大小导致HTTP 413时，仅缩放三视图传输尺寸后重试，不修改状态值。VLM输出属于自动标注而非人工真值，相关一致性限制已在第7节报告。

## 11 利益冲突、资助与致谢

**利益冲突：** 待作者在投稿前声明。  
**资助信息：** 待补充。  
**致谢：** 待补充。

## 参考文献

[1] Pei, K., Cao, Y., Yang, J., & Jana, S. DeepXplore: Automated Whitebox Testing of Deep Learning Systems. SOSP, 2017. [arXiv](https://arxiv.org/abs/1705.06640)

[2] Ma, L., et al. DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems. ASE, 2018. [arXiv](https://arxiv.org/abs/1803.07519)

[3] Kim, J., Feldt, R., & Yoo, S. Guiding Deep Learning System Testing using Surprise Adequacy. ICSE, 2019. [arXiv](https://arxiv.org/abs/1808.08444)

[4] Feng, Y., Shi, Q., Gao, X., Wan, J., Fang, C., & Chen, Z. DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks. ISSTA, 2020. [arXiv](https://arxiv.org/abs/1903.00661)

[5] Sener, O., & Savarese, S. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR, 2018. [OpenReview](https://openreview.net/pdf?id=H1aIuk-RW)

[6] Harel-Canada, F., Wang, L., Gulzar, M. A., Gu, Q., & Kim, M. Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks? ESEC/FSE, 2020. [DOI](https://doi.org/10.1145/3368089.3409754)

[7] Yuan, Y., Pang, Q., & Wang, S. Revisiting Neuron Coverage for DNN Testing: A Layer-Wise and Distribution-Aware Criterion. ICSE, 2023. [arXiv](https://arxiv.org/abs/2112.01955)

[8] Lin, T.-Y., et al. Microsoft COCO: Common Objects in Context. ECCV, 2014. [arXiv](https://arxiv.org/abs/1405.0312)

[9] Kuznetsova, A., et al. The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale. IJCV, 2020. [arXiv](https://arxiv.org/abs/1811.00982)

[10] Everingham, M., Eslami, S. M. A., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. The Pascal Visual Object Classes Challenge: A Retrospective. IJCV, 2015. [DOI](https://doi.org/10.1007/s11263-014-0733-5)

[11] He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016. [arXiv](https://arxiv.org/abs/1512.03385)

[12] Howard, A., et al. Searching for MobileNetV3. ICCV, 2019. [arXiv](https://arxiv.org/abs/1905.02244)

[13] Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. Densely Connected Convolutional Networks. CVPR, 2017. [arXiv](https://arxiv.org/abs/1608.06993)

[14] Tan, M., & Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019. [arXiv](https://arxiv.org/abs/1905.11946)

[15] Liu, Z., et al. A ConvNet for the 2020s. CVPR, 2022. [arXiv](https://arxiv.org/abs/2201.03545)

[16] Liu, Z., et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021. [arXiv](https://arxiv.org/abs/2103.14030)

[17] Aghababaeyan, Z., Abdellatif, M., Briand, L., Ramesh, S., & Bagherzadeh, M. Black-Box Testing of Deep Neural Networks Through Test Case Diversity. IEEE Transactions on Software Engineering, 2023. [arXiv](https://arxiv.org/abs/2112.12591)

[18] Aghababaeyan, Z., Abdellatif, M., Dadkhah, M., & Briand, L. DeepGD: A Multi-Objective Black-Box Test Selection Approach for Deep Neural Networks. ACM Transactions on Software Engineering and Methodology, 2024. [arXiv](https://arxiv.org/abs/2303.04878)

[19] Dola, S., Dwyer, M. B., & Soffa, M. L. Input Distribution Coverage: Measuring Feature Interaction Adequacy in Neural Network Testing. ACM Transactions on Software Engineering and Methodology, 32(3), 2023. [DOI](https://doi.org/10.1145/3576040)

## 附录A：可复现材料索引

- 候选池与状态协议：`artifacts/multiclass_object_pool_v1_20260928/`
- 五方法主实验：`artifacts/multiclass_object_selection_benchmark_20260928/`
- 状态留出与独立错误指标：`artifacts/cross_state_decoupled_diagnostics_20260928/`
- 错误簇数量敏感性：`artifacts/error_cluster_sensitivity_20260929/`
- 重复与近重复审计：`artifacts/near_duplicate_audit_20260929/`
- 失败模式与案例图谱：`artifacts/multiclass_failure_atlas_20260929/`
- 等错误数、固定预算互补与状态组合：`artifacts/coverage_incremental_value_20260929/`
- 逐类别等错误数与互补控制：`artifacts/class_stratified_error_controls_20260929/`
- 线上VLM困难样本复标与敏感性：`artifacts/online_vlm_reannotation_20260929/`
- PASCAL VOC跨源候选池：`artifacts/voc2012_confirmation_pool_20260929/`
- PASCAL VOC跨源确认实验：`artifacts/voc2012_confirmation_benchmark_20260929/`
- 投稿冻结包、环境与文件哈希：`artifacts/submission_freeze_20260929/`
- 核心执行脚本：
  - `scripts/run_multiclass_object_selection_benchmark.py`
  - `scripts/run_cross_state_decoupled_diagnostics.py`
  - `scripts/run_error_cluster_sensitivity.py`
  - `scripts/audit_multiclass_near_duplicates.py`
  - `scripts/build_multiclass_failure_atlas.py`
  - `scripts/run_coverage_incremental_value.py`
  - `scripts/summarize_coverage_incremental_value.py`
  - `scripts/run_class_stratified_error_controls.py`
  - `scripts/prepare_voc2012_confirmation_pool.py`
  - `scripts/prepare_online_reannotation_subset.py`
  - `scripts/annotate_multiclass_online_sensenova.py`
  - `scripts/analyze_online_vlm_agreement.py`
  - `scripts/run_voc2012_confirmation_benchmark.py`
