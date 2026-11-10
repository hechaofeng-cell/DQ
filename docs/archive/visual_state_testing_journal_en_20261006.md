# Visual-State Risk Stratification for Testing Multiclass Image Classifiers: An Empirical Study

## Abstract

Aggregate classification scores summarize a test set but provide limited information about the visual conditions associated with failures. We investigate an automated testing protocol that separates visual-state coverage, risk composition, and diagnostic breadth. Twenty-four frozen classifiers, comprising eight architectures and three training seeds, are evaluated on 5,400 confirmation targets from COCO, PASCAL VOC2007, and Objects365. Two vision-language models independently annotate predefined states; category-relative risk strata are constructed without classifier outputs. Across all three pools and both annotation sources, every classifier run obtains lower macro-averaged F1 in the highest-risk stratum than in the lowest. The mean differences range from 14.44 to 41.80 percentage points and exceed the fluctuations observed in 10,000 class-balanced random stratifications. With test size and class proportions fixed, increasing the highest-risk share across 21 measured points reduces mean F1 by 12.29–34.77 percentage points between the endpoints. Frozen weak-state descriptions identify reproducible conditional deficits when external support is sufficient. Coverage selection also provides conditional diagnostic gains against random selection, but stronger baselines, stricter controls, and alternative representations limit this result. Official attribute comparisons and disagreement analyses expose substantial annotation uncertainty. The findings support visual-state stratification as a practical way to qualify model evaluations and locate testable weaknesses in the evaluated setting, rather than treating increased coverage or error counts as evidence of universally better testing.

**Keywords:** Deep learning testing; Visual-state coverage; Risk stratification; Multiclass classification; Failure diagnosis; Empirical study

## 1 Introduction

A classifier can achieve a strong aggregate score while failing frequently on a small subset of visual conditions. A test set containing few back-facing, partially visible, or blurred targets may conceal those weaknesses. Software testing therefore needs evidence about the conditions exercised by a test suite, as well as its aggregate accuracy. For learned classifiers, changing that suite can substantially change the measured score even when the software and model weights remain fixed.

Coverage criteria, surprise-based measures, and diversity-based selection address different aspects of this problem. DeepXplore and DeepGauge describe internal network activity (Pei et al., 2017; Ma et al., 2018). Surprise Adequacy compares test behavior with training behavior (Kim et al., 2019). Input-distribution and black-box diversity studies examine input-space adequacy and failure revelation (Dola et al., 2023; Aghababaeyan et al., 2023). These lines of research establish that test adequacy and diversity are existing research problems. They also motivate evaluating the usefulness of a coverage measure independently of its numerical growth. Evidence concerning neuron coverage illustrates why a larger coverage score cannot, by itself, establish better fault detection (Harel-Canada et al., 2020).

We study predefined, externally visible conditions that a tester can interpret and use to specify further tests. Two vision-language models (VLMs) annotate fixed target regions. A five-dimensional score organizes targets into category-relative risk strata without consulting the classifiers being tested. A separate frequency-based coverage measure records the amount of state evidence in a selected suite. These quantities serve different purposes: risk organizes test conditions, coverage describes state evidence, and classification outcomes measure observed model behavior. Their separation prevents a falling F1 score from being mistaken for a trained model becoming worse, or an increasing coverage score from being mistaken for superior testing.

Our central question is whether automated visual-state organization provides useful conditional evidence beyond an aggregate score. The evaluation uses six classes, eight architectures, and three seeds. A discovery pool supplies weak-state hypotheses; three subsequent pools provide confirmation targets, including a larger Objects365 pool that addresses the limited state support in VOC2007. Models remain frozen throughout confirmation. We investigate three research questions:

1. **RQ1:** Does predefined visual-risk stratification distinguish classifier performance beyond class-balanced random stratification, and how sensitive is that distinction to the score definition?
2. **RQ2:** How do measured performance and model rankings respond when risk composition changes under a fixed test budget and fixed class proportions?
3. **RQ3:** What reproducible conditional weaknesses and additional diagnostic breadth does visual-state analysis provide, and where does its usefulness stop?

The study makes three contributions. First, it specifies an auditable protocol separating visible-state evidence, category-relative risk, and classifier outcomes. Second, it provides controlled empirical evidence on risk-dependent performance and ranking responses, including random-stratification and dimension-removal checks. Third, it evaluates diagnostic usefulness through frozen-state confirmation and budget-, risk-, and error-count-controlled selection comparisons. The contribution is a testing protocol and an empirical account of its applicability; it is not a new recognition architecture, a proof of universal coverage superiority, or a model-repair algorithm.

## 2 Related work

### 2.1 Coverage and test adequacy

DeepXplore combines internal coverage with differential model behavior for test generation, while DeepGauge defines several coverage granularities (Pei et al., 2017; Ma et al., 2018). Our external states describe conditions such as visibility and occlusion; they do not approximate a network's control-flow or neuron coverage. Harel-Canada et al. (2020) question neuron coverage as an effectiveness measure in their evaluated settings. That finding does not establish that every coverage metric is ineffective, but reinforces the need for separate outcome validation.

Input Distribution Coverage measures latent input features and their interactions (Dola et al., 2023). Our measure instead counts finite, interpretable field–value states. It is marginal coverage: observing visibility and orientation separately does not establish adequate coverage of their combinations. Accordingly, we do not describe a high score as comprehensive coverage of real-world visual situations.

### 2.2 Prioritization and diversity

DeepGini ranks tests using output-probability impurity (Feng et al., 2020). Surprise Adequacy measures departure from training behavior (Kim et al., 2019). Core-set selection motivates geometric diversity in representation space (Sener and Savarese, 2018). We compare state selection with DeepGini, distance-based Surprise Adequacy (DSA), and a frozen-embedding greedy k-center baseline. The latter is a geometric selection baseline, not a reproduction of the full active-learning and retraining procedure.

Black-box diversity testing and DeepGD already study revealing varied failures through input diversity and uncertainty (Aghababaeyan et al., 2023, 2024). Thus, diverse failure discovery is not a new objective introduced here. Our empirical focus is interpretable state organization, two annotation sources, frozen condition confirmation, and class-specific error-count control. DeepGD itself is not implemented in this study, so we make no direct performance claim against it.

## 3 Testing protocol

### 3.1 Task and information boundaries

The task is six-class target-region classification: dog, cat, horse, sheep, person, and car. Each target has a source image, an official bounding box, and an official category label. The classifier receives a crop of that predetermined box and predicts one of the six classes. The person category denotes annotated person targets, not a face-recognition task. We evaluate recognition of supplied regions, not object localization or open-world classification.

VLM requests contain the original image, an image marking target A, and an enlarged crop of A. The target category is included in the state prompt. VLMs therefore perform category-conditioned state description, not an independent blind category-identification test. They do not provide classification ground truth. Classification predictions are cached once per target and model; changing annotators changes test organization, not those predictions. Figure 1 shows the separation of these information sources.

![Figure 1. Testing workflow. Official labels define target regions and classification truth. VLM states organize test conditions. Classifier predictions are held fixed and used only for outcome evaluation.](../artifacts/journal_revision_20261006/figures/01_workflow.png)

Local annotations use `qwen3-vl:30b-a3b-instruct`; online annotations use `sensenova-6.8-flash-lite`. Native protocols contain 19 general-object fields and 18 car fields. Both map to 14 shared fields: body visibility, orientation, occlusion severity, occluder type, truncation severity, group relation, interaction, scene, background complexity, blur, lighting, target–background contrast, recognizability, and presentation form. Original-image target scale is computed from box area, giving 15 analysis fields including this programmatic field. Enumerations and normalization rules are retained in Supplementary Information S1. Unobservable values remain `unknown`. Successful structural validation does not establish semantic label accuracy.

### 3.2 Coverage and selection

A state is a field–value pair; multivalued fields can activate several states. For class c, the state universe U_c is the union of known states observed by either annotator in that candidate pool. It is fixed across all budgets in the pool. If n_s(S) is the support for state s in suite S, frequency coverage is

$$
C_c(S)=\frac{1}{|U_c|}\sum_{s\in U_c}\min\left(\frac{n_s(S)}{20},1\right),\qquad C(S)=\frac{1}{6}\sum_{c=1}^{6}C_c(S).
$$

Unknown values do not count as coverage evidence. The threshold of 20 caps each state's contribution. This pool-conditioned score is not an absolute percentage of all possible visual conditions, and scores from different state universes are not directly comparable. In a nested suite with fixed labels and denominator, coverage cannot decrease by construction. That monotonicity is a descriptive property, not an effectiveness result.

Coverage selection greedily favors rare states. For candidate x activating states A(x), its marginal gain is

$$
g(x\mid S)=\sum_{s\in A(x),\,n_s(S)<20}\left[\log(n_s(S)+2)-\log(n_s(S)+1)\right].
$$

Counts persist across strata. This concave state-evidence objective is not exactly the clipped linear coverage objective and does not optimize classifier errors. Before comparing strategies, we impose shared risk-cell quotas within each category and stratum. A cell is defined by the midpoint risk score and the number of unknown risk dimensions; proportional quotas use largest-remainder allocation. Coverage, Random, DeepGini, DSA, k-center, and a reduced-field Coverage variant share these quotas. Random selection is repeated 1,000 times without replacement within feasible cells.

A supplementary protocol first selects Coverage without the common quotas, then exactly matches its risk-cell histogram with random suites. It evaluates a different feasible Coverage sequence. Differences between the two protocols are not a clean causal estimate of adding one control. The reduced-field variant removes the five risk fields and recognizability, leaving nine shared fields; those remaining fields can still correlate with difficulty.

### 3.3 Risk strata and controls

The risk score sums five visible dimensions (Table 1), ranges from 0 to 13, and excludes classifier probabilities, embeddings, errors, and the VLM recognizability field. Unknown dimensions contribute half their maximum in the primary analysis; lower- and upper-bound assignments form sensitivity analyses. Targets are sorted within each category by score, with target ID resolving ties, and divided into equal quintiles Q1–Q5. Q5 is relatively high risk within a category and pool, not an absolute difficulty threshold shared by all classes.

**Table 1. Predefined risk mapping. Scale uses box area divided by original-image area.**

| Dimension | Mapping to score |
|---|---|
| Area ratio a | a ≥ 0.10: 0; 0.03 ≤ a < 0.10: 1; 0.01 ≤ a < 0.03: 2; a < 0.01: 3 |
| Body visibility | full/most: 0; partial: 2; minimal: 3 |
| Occlusion | none: 0; light: 1; moderate: 2; heavy: 3 |
| Truncation | none: 0; light: 0.5; moderate: 1; heavy: 2 |
| Orientation | front/side/three-quarter: 0; back: 1; overhead/unusual: 2 |

Three controls distinguish the roles of grouping, score definition, and test composition. First, 10,000 category-wise random permutations per pool produce equal-sized random quintiles. All classifier predictions are reused, yielding a conditional null distribution for the panel-average Q5−Q1 difference. Second, each dimension is removed in turn and the remaining score is used to reassign strata; an equally normalized dimension variant is also evaluated. Removing a dimension changes stratum membership and does not isolate its causal effect. Third, fixed-budget suites contain 120 targets, 20 per class. The Q5 share varies from 0% to 100% in 5-point increments; the remaining targets come from Q1–Q4. Each of the 21 measured points uses 1,000 repeated draws with paired sampling orders. B and D additionally permit a 480-target sensitivity budget.

### 3.4 Outcomes and frozen weakness confirmation

We report accuracy, macro-averaged precision, recall and F1, and class-specific metrics. Macro-F1 is the arithmetic mean of the six class F1 scores, not the harmonic mean of macro-precision and macro-recall. With equal true-class counts, accuracy equals macro-recall. Performance is interpreted under the stated suite composition.

Raw failed-state breadth counts state evidence associated with mistakes but depends on the same automated state labels used in selection. Two additional measures avoid that direct dependence: confusion-type breadth and representation-cluster breadth among erroneous targets. Error clusters are proxies for representation diversity, not verified independent fault causes. Clustering uses frozen classifier features in the main analysis and a common frozen ViT-B/16 representation in sensitivity analysis. Per-class cluster counts follow the fixed square-root rule capped at eight, with explicit zero- and single-error cases.

To separate failure breadth from error quantity, competing suites are rarefied to the smaller error count in each true class. For N errors, r members of one failure type, and a matched sample of q errors, its expected discovery probability is

$$
1-\frac{\binom{N-r}{q}}{\binom{N}{q}}.
$$

Normalized class-wise breadth is macro-averaged, then summarized by normalized trapezoidal area over budget, including the zero-budget origin. This budget AUC is neither ROC-AUC nor an F1 gain. Equal-error comparison is a retrospective diagnostic control, not an online selection mechanism with advance knowledge of failures.

The discovery pool supplies a frozen list of 175 state–architecture combinations: 87 local and 88 online. A state must have at least 20 targets and mean recall at least 10 percentage points below its class recall; at most three states per class and architecture are retained. The list is fixed before B predictions and carried to C and D without replacement by more favorable conditions. New-pool support below 20 is marked not evaluable. Overlapping states are not treated as independent samples.

## 4 Empirical design

### 4.1 Data and model panel

Table 2 distinguishes discovery and confirmation. Six-class counts are balanced in each pool. The 5,400 confirmation targets come from B, C, and D; the 2,400 discovery targets are additional. A is mostly Open Images with 111 VOC2012 sheep targets. C merges permitted VOC2007 source splits for this study and is not an official benchmark-score submission. D uses Objects365 v2 targets from fixed image-package prefixes after a recorded pre-prediction amendment addressing insufficient sheep targets in the initial validation source. Pool construction details, category mapping, manifests, exclusions, and amendments are retained in the supplement.

The source datasets and attribute definitions are documented in their original publications and official resources (Lin et al., 2014; PASCAL VOC, 2007; Objects365 Consortium, n.d.; Open Images, n.d.).

**Table 2. Data roles. Every pool contains dog, cat, horse, sheep, person, and car in equal proportions.**

| Set | Source | Targets | Per class | Role |
|---|---|---:|---:|---|
| Training | COCO | 1,800 | 300 | Classifier fitting |
| Validation | COCO | 90 | 15 | Checkpoint selection |
| Base test | COCO | 300 | 50 | Historical reference |
| A | Open Images/VOC2012 | 2,400 | 400 | Weak-state discovery |
| B | COCO | 2,400 | 400 | New-target confirmation |
| C | VOC2007 | 600 | 100 | External-source confirmation |
| D | Objects365 v2 | 2,400 | 400 | Larger external confirmation |

Cross-set checks use official IDs, image and crop hashes, perceptual hashes, and fixed ViT similarity. The recorded near-duplicate rule excludes pairs with perceptual-hash distance at most four and cosine similarity at least 0.980, or distance at most eight and cosine similarity at least 0.995. These checks do not prove absence of every near duplicate or of overlap with large-scale pretraining data.

The panel comprises ResNet-18, ResNet-50, MobileNetV3-Small, MobileNetV3-Large, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny, and Swin-T. Each uses ImageNet-pretrained initialization and a six-class head, with seeds 20260916, 20260917, and 20260918. Training uses 20 epochs, class-balanced weighted sampling of 3,000 examples per epoch, batch size 64, AdamW, learning rate and weight decay both 10^-4, and cosine scheduling. A five-percent box-margin crop is resized to 256 pixels; training uses random 224-pixel cropping, flipping and color jitter, while evaluation uses a center crop. The validation macro-F1 selects checkpoints. Common settings support controlled comparison but are not architecture-specific optimal tuning. All 24 checkpoints are subsequently frozen.

### 4.2 Inference and analysis units

The architecture is the primary unit for paired comparisons: three seeds are averaged within each of eight architectures. Two-sided exhaustive sign-flip tests use 2^8 assignments. Architecture/seed hierarchical bootstraps use 5,000 repetitions, conditional on the available pool. Separate source-image bootstraps address target sampling under the fixed model panel. These intervals answer different questions and are not interchangeable.

Within each pool, the original risk family contains six source/unknown-rule contrasts. The primary selection family contains four source-by-diagnostic-metric contrasts; 12 comparisons with stronger baselines are secondary, and four reduced-field comparisons are supplementary. A joint 20-comparison correction is also reported. With eight paired architecture units, the minimum two-sided sign-flip p value is 0.0078125. A 12-test Holm family therefore cannot pass its first 0.05/12 threshold under this design. This limits inference even when effects have consistent directions; it does not justify ignoring correction or counting seeds as independent architectures.

The later random-stratification control uses a Monte Carlo two-sided absolute-gap test with a plus-one correction and a six-comparison Holm family. Its random ranges describe random grouping of the fixed pool, not confidence intervals for deployment performance. Fixed-budget mixture ranges are percentiles of paired repeated-draw differences, not confidence intervals of a mean. Dimension-removal comparisons are supplementary; their 12-test within-pool Holm correction is retained. Later cached-prediction analyses were designed after earlier outcomes were available. We distinguish those checks from hypotheses frozen before confirmation and do not describe the whole project as prospectively preregistered.

### 4.3 Annotation and computational evidence

Official classification labels supply category truth. No human ground truth is available for the complete state protocol. Open Images box-level occlusion and truncation attributes provide partial reference checks on A, reported in Section 7. Those binary attributes and VLM severity labels are not equivalent constructs. Dual annotation, unknown-value sensitivity and dimension ablation check consequences of uncertainty; they do not establish semantic accuracy of every field.

State annotation is reused across the 24 classifiers and selection analyses. Successful-request median/P95 latency is 4.00/4.30 seconds locally and 11.41/14.62 seconds online for B, and 3.94/4.30 and 9.92/12.31 seconds for C. These are request latencies, not end-to-end runtime comparisons. D logs contain all attempts, including retries, and are reported separately in the supplement. No verified invoices or historical unit prices support a monetary-cost estimate.

VLM prompting is an experimental component documented above. Codex also assisted analysis-code preparation and substantive manuscript drafting. The final scientific interpretation, reference verification and submission remain the authors' responsibility; this disclosure does not treat AI tools as authors.

## 5 Results

### 5.1 RQ1: Risk strata separate performance beyond random grouping

Every classifier run has a negative full-quintile Q5−Q1 difference in all six pool–annotator settings. Panel-average differences range from −14.44 to −41.80 percentage points (Table 3). The original six-contrast risk families pass Holm correction within each confirmation pool. Intermediate quintiles are not uniformly monotonic: for example, local VOC2007 Q3 exceeds Q1. The result concerns a reproducible endpoint distinction, not a guarantee of a strictly ordered curve.

**Table 3. Observed full-quintile F1 differences versus 10,000 random class-balanced stratifications. Values are percentage points, except p. Random ranges are null percentiles.**

| Pool | Annotator | Observed Q5−Q1 | Random mean | Random 95% range | Six-test Holm p |
|---|---|---:|---:|---|---:|
| B | Local | −25.50 | −0.01 | [−3.24, +3.22] | 0.00060 |
| B | Online | −21.30 | −0.01 | [−3.24, +3.22] | 0.00060 |
| C | Local | −16.57 | +0.02 | [−5.61, +5.76] | 0.00060 |
| C | Online | −14.44 | +0.02 | [−5.61, +5.76] | 0.00060 |
| D | Local | −41.80 | +0.01 | [−4.24, +4.23] | 0.00060 |
| D | Online | −41.25 | +0.01 | [−4.24, +4.23] | 0.00060 |

![Figure 2. Risk-defined F1 gaps compared with random-grouping ranges on a common scale. Diamonds mark observed differences; gray bars show the central 95% of random-stratification differences.](../artifacts/journal_revision_20261006/figures/02_risk_random.png)

No random repetition reaches the absolute observed panel gap in any setting. This rejects random grouping as an adequate account of the observed endpoint separation, conditional on these targets and classifiers. It does not establish superiority over every meaningful alternative stratification or prove that a particular state causes errors.

Deleting any one dimension retains negative endpoint differences for every run. Table 4 reports full, scale-removed and occlusion-removed variants; all variants and stratum-membership changes appear in Supplementary Information S3. Removing scale produces the largest attenuation among dimension removals in D: 9.43 points locally and 13.26 online. Removing occlusion retains gaps of −11.31 to −43.11 points across the settings. Consequently, the separation does not disappear when the least reliable reference-checked field is removed. However, correlated dimensions remain, membership changes, and the corrected ablation family does not pass (minimum adjusted p = 0.09375).

**Table 4. Score sensitivity: mean Q5−Q1 F1 differences, in percentage points. Ablation directions are descriptive; corrected tests do not establish dimension-specific effects.**

| Pool | Annotator | Full score | Without scale | Without occlusion |
|---|---|---:|---:|---:|
| B | Local | −25.50 | −21.55 | −25.76 |
| B | Online | −21.30 | −11.50 | −21.06 |
| C | Local | −16.57 | −12.79 | −13.27 |
| C | Online | −14.44 | −9.54 | −11.31 |
| D | Local | −41.80 | −32.37 | −43.11 |
| D | Online | −41.25 | −27.99 | −41.75 |

**RQ1 finding.** The predefined score organizes a repeatable endpoint performance contrast beyond random stratification. Dimension deletion supports directional persistence, while annotation semantics and the contribution of individual dimensions remain uncertain.

### 5.2 RQ2: Fixed-budget composition exposes different capability profiles

With 120 targets and equal class proportions, the mean F1 difference between zero-percent and all-Q5 suites is negative in every pool–annotator setting (Table 5). All 48 architecture-by-setting endpoint comparisons are negative. Figure 3 uses all 21 measured points; lines connect empirical means without fitted smoothing or invented observations. The experiment varies risk composition directly. It is not an intervention on coverage alone.

**Table 5. Fixed-budget mixture endpoints. F1 values are percentages; differences and repeated-draw ranges are percentage points. The zero-percent endpoint samples Q1–Q4.**

| Pool | Annotator | 0% Q5 | 100% Q5 | Difference | Paired 95% sampling range |
|---|---|---:|---:|---:|---|
| B | Local | 90.18 | 69.40 | −20.78 | [−27.06, −14.21] |
| B | Online | 89.34 | 72.72 | −16.62 | [−22.82, −10.45] |
| C | Local | 91.80 | 75.88 | −15.92 | [−18.76, −13.05] |
| C | Online | 91.07 | 78.78 | −12.29 | [−15.23, −8.98] |
| D | Local | 79.81 | 45.04 | −34.77 | [−42.12, −27.58] |
| D | Online | 79.33 | 47.49 | −31.84 | [−39.65, −24.08] |

![Figure 3. Change in panel-average F1 relative to the zero-percent Q5 mixture, with test size and class proportions fixed. Each line contains 21 measured proportions and 1,000 draws per proportion; markers are shown at selected points for readability.](../artifacts/journal_revision_20261006/figures/03_mixture_response.png)

The 480-target sensitivity budget preserves endpoint directions in B and D. Nevertheless, class recall is not universally decreasing. Local-B car recall increases from 95.18% to 97.20%, with a repeated-draw difference range crossing zero. A class-relative risk score can capture different recognition challenges across categories, and precision may rise if cross-class false positives change. Aggregate F1 therefore cannot be inferred from a single class's recall or precision.

Table 6 compares complete Q5 suites, separately from the 120-target mixtures. ConvNeXt-Tiny and Swin-T occupy the first two mean positions in all six Q5 settings and at all 126 fixed-budget pool–source–proportion settings. Some middle ranks change by as much as four positions. Stable mean positions do not establish uncertainty-free superiority: image-bootstrap top-two frequency for Swin-T is substantially lower in D than in B, and some pairwise score intervals overlap. These rankings apply to the stated pretrained models, training recipe and task.

**Table 6. Complete Q5 macro-F1, averaged over three seeds per architecture (%). B and D contain 480 Q5 targets; C contains 120.**

| Architecture | B local | B online | C local | C online | D local | D online |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 67.97 | 69.67 | 71.37 | 75.85 | 45.30 | 48.95 |
| ResNet-50 | 70.02 | 72.48 | 77.35 | 80.32 | 44.59 | 46.57 |
| MobileNetV3-Small | 64.67 | 67.95 | 69.09 | 72.26 | 42.49 | 44.14 |
| MobileNetV3-Large | 68.77 | 71.84 | 76.05 | 79.90 | 45.57 | 48.04 |
| DenseNet-121 | 66.13 | 70.21 | 73.25 | 74.60 | 39.64 | 42.86 |
| EfficientNet-B0 | 70.66 | 73.57 | 73.58 | 78.17 | 46.24 | 48.87 |
| ConvNeXt-Tiny | 74.97 | 78.36 | 84.46 | 85.50 | 51.60 | 54.03 |
| Swin-T | 73.73 | 77.14 | 81.89 | 83.61 | 46.36 | 49.16 |

Earlier nested Coverage sequences increase state evidence while changing both suite size and composition. For example, local D coverage increases from 21.22% to 69.16%, while F1 first rises from 78.32% to 80.01% and then falls to 65.91%. Those sequences do not use the primary common-risk quotas. Their nonmonotonic response reinforces the separation of coverage and risk rather than providing a causal coverage effect.

**RQ2 finding.** Fixed-budget risk mixtures reveal large capability differences while distinguishing stable leading mean ranks from composition-sensitive middle ranks. Neither increased coverage nor increased risk entails a universally monotonic response for every category and metric.

### 5.3 RQ3: State diagnosis is reproducible, but selection gains are conditional

Table 7 tests Coverage against Random under common risk quotas and class-specific error-count matching. B and D show positive diagnostic effects under the four-contrast primary correction; C supports only the online source. The effects are small normalized breadth-AUC differences, not percentage-point F1 improvements. The joint 20-test correction supports none of these contrasts. Complete stronger-baseline results are retained in Supplementary Information S4: examples include negative local-C differences against k-center and a negative online-D confusion-breadth difference against DeepGini (−0.0031). We do not conclude that Coverage outperforms the stronger baselines generally.

**Table 7. Coverage minus Random diagnostic breadth AUC under common risk quotas and equal-error control. Reported p values use the four-contrast primary family within each pool.**

| Pool | Annotator | Confusion AUC difference | Holm p | Cluster AUC difference | Holm p |
|---|---|---:|---:|---:|---:|
| B | Local | +0.0158 | 0.03125 | +0.0103 | 0.03125 |
| B | Online | +0.0224 | 0.03125 | +0.0070 | 0.04688 |
| C | Local | +0.0018 | 0.98438 | +0.0010 | 0.98438 |
| C | Online | +0.0185 | 0.03125 | +0.0164 | 0.03125 |
| D | Local | +0.0377 | 0.03125 | +0.0262 | 0.03125 |
| D | Online | +0.0168 | 0.03125 | +0.0123 | 0.03125 |

Exact matching to Coverage's own risk histogram narrows support: D retains a corrected local confusion-type effect, but not its other three comparisons. Using a common ViT error-cluster representation supports D's two contrasts but not B's or C's. These results qualify the independence of diagnostic gains from risk and representation; they do not remove all residual difficulty confounding.

The fixed-budget mixtures provide a complementary negative result. Raw failed-state breadth rises toward the all-Q5 endpoint. After matching error counts against a 20%-Q5 reference, confusion-type and cluster breadth do not consistently increase. A suite revealing more mistakes can therefore reveal more labeled failure states without expanding distinct failure evidence to the same extent.

Frozen state confirmation supplies more directly interpretable evidence (Table 8). Most states in B and D have adequate support and lower recall than their corresponding class average. VOC2007 supports only one local and three online state–architecture combinations, so it cannot establish transfer of the full shortlist. D supports 160 of 175 combinations; 142 supported combinations retain a negative direction after averaging seeds. These are overlapping conditional descriptions, not 160 independent defects or 142 proven root causes.

**Table 8. Confirmation of the frozen discovery shortlist. Recall differences use the prescribed architecture/run aggregation, not a direct average of all overlapping combinations.**

| Pool | Annotator | Frozen combinations | Evaluable | Mean recall difference (pp) | Two-test Holm p |
|---|---|---:|---:|---:|---:|
| B | Local | 87 | 78 | −21.73 | 0.01562 |
| B | Online | 88 | 75 | −20.77 | 0.01562 |
| C | Local | 87 | 1 | −23.83 | 1.00000 |
| C | Online | 88 | 3 | +1.62 | 1.00000 |
| D | Local | 87 | 79 | −22.89 | 0.01562 |
| D | Online | 88 | 81 | −20.43 | 0.01562 |

Table 9 and Figure 4 illustrate how a tester can turn these outputs into concrete follow-up conditions. ResNet-50's dog recall in B is 81.92%, whereas back-facing dog recall is 57.25%. The state specifies a test condition that the category average alone does not communicate. Conversely, DenseNet-121's medium-scale dog condition exceeds its category average in B, showing that an initially weak condition need not remain weak in new data. No repair experiment is performed, and these associations do not identify causal error mechanisms.

**Table 9. Conditional diagnosis examples in B, including a non-replicating direction. Recall is averaged over three seeds; support counts unique targets.**

| Architecture and condition | Source | Support | Class recall (%) | State recall (%) | Difference (pp) |
|---|---|---:|---:|---:|---:|
| ResNet-50, back-facing dog | Local | 46 | 81.92 | 57.25 | −24.67 |
| ResNet-50, small-area cat | Program/local | 23 | 82.67 | 39.13 | −43.54 |
| ResNet-50, heavily blurred sheep | Online | 44 | 88.42 | 57.58 | −30.84 |
| DenseNet-121, medium-area dog | Program/local | 79 | 73.42 | 81.01 | +7.60 |

![Figure 4. Category-average recall versus conditional recall in B. Common 0–100% axes preserve magnitude. The fourth example retains a positive difference rather than suppressing a non-replicating condition.](../artifacts/journal_revision_20261006/figures/04_condition_cases.png)

**RQ3 finding.** Frozen visual-state descriptions identify reproducible, actionable test conditions where support is adequate. Additional selection breadth is conditional on source, controls and representation; it is not a universal effectiveness advantage.

## 6 Implications for model testing

### 6.1 Report test composition with performance

The strongest evidence concerns test organization: large endpoint gaps persist across the fixed model panel, exceed random-stratification fluctuations, and recur in external-source targets. A useful evaluation report should therefore include class proportions, risk composition, state support and unknown rates alongside aggregate scores. The evaluated suites measure capability under specified visual conditions; they do not estimate deployment performance without a deployment distribution.

Separating suite expansion from fixed-budget mixture testing is important. Expansion answers what additional state evidence has been exercised. Mixture testing answers how measured capability responds when a fixed amount of testing is redistributed across risk conditions. A falling score under the latter can expose a capability limit without showing any degradation of the software under test.

### 6.2 Use diagnosis to specify follow-up tests

Conditional reports supply practical test requirements, such as collecting further back-facing dog targets to check a particular model. The frozen-list protocol reduces the temptation to replace unsupported conditions with newly favorable ones after seeing a confirmation result. Adequate support and held-out confirmation are needed before using a state description as evidence of a persistent weakness. A diagnosis can guide future testing without an accompanying retraining experiment; repair effectiveness remains a separate question.

### 6.3 Treat coverage as evidence accounting

Coverage records what has been exercised. It need not be a universally best prioritization objective to be useful in test reporting. Nevertheless, state-selection usefulness requires explicit comparison: matching risk, counting errors separately from their diversity, and checking representation sensitivity. Strong negative results here delimit that usefulness. Combining confidence or geometric diversity with state reporting is a reasonable future direction, but its effectiveness is not established by this study.

## 7 Threats to validity

**Construct validity.** Automated labels are the main uncertainty. Table 10 reports partial reference checks against official binary Open Images attributes, after the fixed binary mapping and exclusion of unknown predictions. Occlusion agreement is weaker than truncation agreement. These values do not validate the full semantic protocol, and their F1 values refer to attribute agreement, not the six-class classifier. Removing occlusion preserves risk-gap directions, but does not make remaining labels accurate. Frequency coverage counts marginal states and has a threshold and pool-conditioned denominator; error-cluster breadth depends on representation and clustering rules. Risk dimensions are correlated, and scale is measured in the original image even though classification uses crops.

**Table 10. Partial reference agreement for official binary attributes in A. Unknown predictions are excluded; n denotes valid comparisons. Kappa is Cohen's agreement coefficient.**

| Attribute | Annotator | n | Agreement (%) | Attribute F1 (%) | Kappa |
|---|---|---:|---:|---:|---:|
| Occlusion | Local | 2,289 | 61.16 | 66.84 | 0.238 |
| Occlusion | Online | 2,284 | 66.07 | 69.67 | 0.331 |
| Truncation | Local | 2,289 | 67.50 | 65.46 | 0.384 |
| Truncation | Online | 2,287 | 79.62 | 74.08 | 0.576 |

**Internal validity.** Model-output-independent risk definitions avoid selecting strata directly on classifier mistakes, but do not eliminate all confounding. Common quotas control specified risk cells rather than every visual condition; equal-error comparisons occur after evaluation. Condition-confirmation estimates remain associations. The project includes exploratory stages and later cached analyses, not a fully prospective preregistration. Recorded source changes, technical replacements and protocol amendments must remain auditable.

Strict cross-VLM agreement is too restrictive to certify the entire result (Table 11). The four annotated risk fields exclude programmatic scale; agreement requires all four to match and be known. Each pool retains only five Q5 targets across all classes, below the planned class-wise comparison support. Therefore, strictly agreement-only high/low comparisons and ranks are not estimable under that rule. Field-wise agreement checks retain many B and D conditional deficits, but are a less stringent, selected analysis and cannot replace full-protocol validation.

**Table 11. Support for strict cross-annotator agreement. Q5 support is the total retained across six classes.**

| Pool | Targets | Four-risk-field agreement | Retained (%) | All-14-field agreement | Retained Q5 targets |
|---|---:|---:|---:|---:|---:|
| B | 2,400 | 166 | 6.92 | 4 | 5 |
| C | 600 | 56 | 9.33 | 0 | 5 |
| D | 2,400 | 187 | 7.79 | 8 | 5 |

**Statistical validity.** Eight architectures provide limited independent units and a coarse exact-test resolution. Thousands of repeated suites do not create thousands of independent architectures or new candidate targets. We retain corrected, uncorrected and effect-size outputs rather than treating power limitations as evidence of equivalence. Ranking uncertainty, sparse state support and shared targets across labels further constrain interpretation. A nonconverged mixed-effects attempt does not supply inferential evidence and is not used for claims.

**External validity.** Six box-annotated categories and three confirmation pools do not represent every visual task, object family or image source. B shares the COCO source with training, whereas C and D provide source changes. The model panel uses conventional ImageNet-pretrained architectures under a common fine-tuning recipe; results do not rank all contemporary foundation models. Crops remove localization demands, full-suite state labeling assumes advance access to the pool, and pretrained-data overlap cannot be excluded comprehensively. Dataset licenses and source-image rights govern any artifact release.

## 8 Conclusion

Across the evaluated six-class task, 24 frozen classifiers and three confirmation pools, predefined visual-risk strata expose substantial endpoint performance differences beyond random grouping. Fixed-budget mixtures show how risk composition changes measured capability while separating stable leading mean ranks from uncertain and composition-sensitive rankings. Frozen state descriptions identify reproducible conditional weaknesses when external support is adequate. These findings support reporting visual-state conditions alongside aggregate metrics and using confirmed conditions to specify further tests. They do not establish semantic accuracy of all automated labels, causal error explanations, or universally superior Coverage selection. Coverage, risk and outcome evidence should be reported separately to make model testing more interpretable and its claims better bounded.

## Statements and Declarations

**Funding:** AUTHOR TO COMPLETE: list verified funding sources and grant identifiers, or confirm that no specific funding was received.

**Competing interests:** AUTHOR TO COMPLETE: provide the actual financial and non-financial conflict statement for all authors.

**Author contributions:** AUTHOR TO COMPLETE: assign actual contributions to named authors. No AI tool is listed as an author.

**Ethics and permissions:** The study uses existing object-dataset images and category annotations; it does not infer personal identity, ethnicity or sensitive human attributes. AUTHOR TO COMPLETE: confirm applicable institutional ethics requirements and source permissions. No unverified exemption or permission is asserted.

**Data availability:** The local reproducibility materials contain target manifests, protocol versions, frozen selections, per-model prediction caches, analysis scripts, result tables and audits. They have not yet been deposited in a public archive. Source images remain subject to their original dataset licenses. AUTHOR TO COMPLETE: insert the actual repository or persistent identifier and verified access conditions; provide reconstruction instructions where redistribution is restricted. The Supplementary Information preserves extended results and an evidence index but does not itself contain all prediction caches.

**AI assistance:** Experimental state annotation and analysis/drafting assistance are described in Sections 3 and 4.3. Authors must approve the final text and take responsibility for the evidence and interpretation.

## References

Aghababaeyan, Z., Abdellatif, M., Briand, L., Ramesh, S., & Bagherzadeh, M. (2023). Black-box testing of deep neural networks through test case diversity. *IEEE Transactions on Software Engineering*. https://doi.org/10.1109/TSE.2023.3243522

Aghababaeyan, Z., Abdellatif, M., Dadkhah, M., & Briand, L. (2024). DeepGD: A multi-objective black-box test selection approach for deep neural networks. *ACM Transactions on Software Engineering and Methodology*. https://doi.org/10.1145/3644388

Dola, S., Dwyer, M. B., & Soffa, M. L. (2023). Input distribution coverage: Measuring feature interaction adequacy in neural network testing. *ACM Transactions on Software Engineering and Methodology, 32*(3). https://doi.org/10.1145/3576040

Feng, Y., Shi, Q., Gao, X., Wan, J., Fang, C., & Chen, Z. (2020). DeepGini: Prioritizing massive tests to enhance the robustness of deep neural networks. In *Proceedings of ISSTA*. https://doi.org/10.1145/3395363.3397357

Harel-Canada, F., Wang, L., Gulzar, M. A., Gu, Q., & Kim, M. (2020). Is neuron coverage a meaningful measure for testing deep neural networks? In *Proceedings of ESEC/FSE*. https://doi.org/10.1145/3368089.3409754

Kim, J., Feldt, R., & Yoo, S. (2019). Guiding deep learning system testing using surprise adequacy. In *Proceedings of ICSE*. https://arxiv.org/abs/1808.08444

Lin, T.-Y., et al. (2014). Microsoft COCO: Common objects in context. In *Proceedings of ECCV*. https://arxiv.org/abs/1405.0312

Ma, L., Juefei-Xu, F., Zhang, F., Sun, J., Xue, M., Li, B., Chen, C., Su, T., Li, L., Liu, Y., Zhao, J., & Wang, Y. (2018). DeepGauge: Multi-granularity testing criteria for deep learning systems. In *Proceedings of ASE*. https://doi.org/10.1145/3238147.3238202

Objects365 Consortium. (n.d.). *Objects365 dataset download and annotation resources*. https://www.objects365.org/download.html

Open Images. (n.d.). *Open Images V7 download: Bounding-box attributes and data formats*. https://storage.googleapis.com/openimages/web/download_v7.html

PASCAL VOC. (2007). *The PASCAL Visual Object Classes Challenge 2007*. https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html

Pei, K., Cao, Y., Yang, J., & Jana, S. (2017). DeepXplore: Automated whitebox testing of deep learning systems. In *Proceedings of SOSP*. https://doi.org/10.1145/3132747.3132785

Sener, O., & Savarese, S. (2018). Active learning for convolutional neural networks: A core-set approach. In *Proceedings of ICLR*. https://arxiv.org/abs/1708.00489
