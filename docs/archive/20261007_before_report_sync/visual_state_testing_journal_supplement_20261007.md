# Supplementary Information: Visual-State Risk Stratification

## S1 Protocol, schema and reproducibility scope

The supplement accompanies the revised empirical manuscript. Numbers below are retained results, not new model runs. The evidence bundle includes CSV tables, protocol files and an index; the new portable bundle includes target identifiers, states and all classification logits/predictions for B/C/D. It is a local review package, not a public deposit. Full source images, training checkpoints and original feature caches are not bundled.

The generic native object schema and the car prompt are included. The shared mapping is implemented in `scripts/dual_vlm_common.py`; source-selection and confirmation logic is preserved by the accompanying protocol documents. Unknown states remain unknown. Original-image scale is computed, not inferred by a VLM.

**Table S1. Generic native fields and allowed states. Shared fields retain identical names and enumerations across the two native protocols; protocol-specific fields are excluded. The bundled shared schema specifies the 14 retained fields.**

| Field | Type | Allowed states |
|---|---|---|
| body_visibility | enum | full, most, partial, minimal, unknown |
| head_visibility | enum | clear, partial, not_visible, unknown |
| body_orientation | enum | front, back, left_side, right_side, three_quarter, overhead_or_unusual, unknown |
| head_orientation | enum | front, back, left, right, up, down, unknown |
| pose_action | enum | standing, sitting, lying, walking, running, jumping, riding_or_being_ridden, interacting, other, unknown |
| leg_visibility | enum | all_expected, some, one, none, unknown |
| occlusion_level | enum | none, light, moderate, heavy, unknown |
| occluder_type | set | none, person, same_class_object, other_object, vegetation, frame_or_structure, image_boundary, multiple, unknown |
| truncation_level | enum | none, light, moderate, heavy, unknown |
| truncation_parts | set | none, head, torso, legs, tail_or_rear, multiple, unknown |
| group_context | enum | single, multiple_separated, multiple_overlapping, crowd_or_group, unknown |
| interaction_context | enum | none, with_person, with_same_class, with_other_animal, with_object, multiple, unknown |
| scene_environment | enum | indoor, outdoor_urban, outdoor_natural, farm_or_field, road_or_vehicle, water_or_beach, snow, other, unknown |
| background_complexity | enum | simple, moderate, complex, unknown |
| blur_level | enum | clear, mild, severe, unknown |
| illumination | enum | normal, low_light, backlit, strong_shadow, overexposed, mixed_or_colored, unknown |
| target_background_contrast | enum | high, medium, low, unknown |
| target_recognizability | enum | high, medium, low, unrecognizable, unknown |
| visual_representation | enum | natural_photo, screen_or_print, illustration_or_artwork, statue_or_toy, unknown |

## S2 Analysis families and interval interpretation

Architecture-level paired inference first averages the three training seeds. The eight-architecture exact two-sided sign-flip test has minimum p = 2/256 = 0.0078125. Pool-wise Holm families remain separate, and the all-20 sensitivity correction remains available. The 12-test strong-baseline and ablation families cannot reject their first null at 0.05 with this test resolution. A positive effect is not a corrected rejection, and a failure to reject is not evidence of equivalence.

Random-stratification central ranges describe a finite-pool grouping null. Fixed-budget mixture ranges describe paired repeated sampling, rather than the precision of the mean. Hierarchical model/seed confidence intervals condition on the available targets; source-image bootstraps condition on the fixed classifiers. Failed mixed-model fits supply no inference. No additional successful statistical model is invented.

The historical discovery and later cached analyses are not all prospective. Conditions and source-selection rules for confirmation were frozen before new predictions where documented. The random-stratification and dimension-ablation analyses are explicitly supplementary checks added after earlier results.

## S3 All score variants

Each row is the panel-average Q5−Q1 difference under a redefined score, not a same-target causal contrast. Negative/positive run counts are descriptive. Confidence intervals condition on the retained targets. Membership rank correlations and Q5 overlaps are available in `membership_summary.csv`.

**Table S2. Full and six alternative score definitions across all pool–annotator settings. Differences and confidence intervals are in F1 units (multiply by 100 for percentage points).**

| pool | source | variant | mean_delta_auc | ci95_low | ci95_high | negative_runs | holm_twelve_p |
|---|---|---|---|---|---|---|---|
| B | local | full | -0.254954 | -0.271853 | -0.239513 | 24 | — |
| B | local | drop_size | -0.215491 | -0.233114 | -0.197555 | 24 | 0.093750 |
| B | local | drop_body_visibility | -0.259904 | -0.281982 | -0.240606 | 24 | 0.093750 |
| B | local | drop_occlusion_level | -0.257618 | -0.273451 | -0.241620 | 24 | 0.093750 |
| B | local | drop_truncation_level | -0.269451 | -0.290459 | -0.251536 | 24 | 0.093750 |
| B | local | drop_body_orientation | -0.241225 | -0.260652 | -0.224249 | 24 | 0.093750 |
| B | local | equal_normalized | -0.260569 | -0.276956 | -0.244779 | 24 | 0.093750 |
| B | online | full | -0.212959 | -0.226347 | -0.198767 | 24 | — |
| B | online | drop_size | -0.114966 | -0.124784 | -0.103608 | 24 | 0.093750 |
| B | online | drop_body_visibility | -0.235493 | -0.251729 | -0.219668 | 24 | 0.093750 |
| B | online | drop_occlusion_level | -0.210552 | -0.223544 | -0.197940 | 24 | 0.093750 |
| B | online | drop_truncation_level | -0.228240 | -0.242014 | -0.212228 | 24 | 0.093750 |
| B | online | drop_body_orientation | -0.211326 | -0.223810 | -0.198088 | 24 | 0.093750 |
| B | online | equal_normalized | -0.204401 | -0.216748 | -0.190993 | 24 | 0.093750 |
| C | local | full | -0.165747 | -0.193947 | -0.134695 | 24 | — |
| C | local | drop_size | -0.127947 | -0.153643 | -0.102327 | 24 | 0.093750 |
| C | local | drop_body_visibility | -0.174427 | -0.198620 | -0.148677 | 24 | 0.093750 |
| C | local | drop_occlusion_level | -0.132691 | -0.158418 | -0.106074 | 24 | 0.093750 |
| C | local | drop_truncation_level | -0.170824 | -0.194692 | -0.145219 | 24 | 0.093750 |
| C | local | drop_body_orientation | -0.142758 | -0.169575 | -0.113630 | 24 | 0.093750 |
| C | local | equal_normalized | -0.167715 | -0.196784 | -0.136133 | 24 | 0.093750 |
| C | online | full | -0.144354 | -0.167132 | -0.120280 | 24 | — |
| C | online | drop_size | -0.095371 | -0.121969 | -0.066814 | 24 | 0.093750 |
| C | online | drop_body_visibility | -0.172666 | -0.195780 | -0.147774 | 24 | 0.093750 |
| C | online | drop_occlusion_level | -0.113079 | -0.129184 | -0.096508 | 24 | 0.093750 |
| C | online | drop_truncation_level | -0.155154 | -0.177722 | -0.130631 | 24 | 0.093750 |
| C | online | drop_body_orientation | -0.124969 | -0.147962 | -0.101718 | 24 | 0.093750 |
| C | online | equal_normalized | -0.134110 | -0.156797 | -0.109221 | 24 | 0.093750 |
| D | local | full | -0.418015 | -0.440158 | -0.398923 | 24 | — |
| D | local | drop_size | -0.323730 | -0.336486 | -0.313259 | 24 | 0.093750 |
| D | local | drop_body_visibility | -0.405195 | -0.427551 | -0.385939 | 24 | 0.093750 |
| D | local | drop_occlusion_level | -0.431122 | -0.456789 | -0.409785 | 24 | 0.093750 |
| D | local | drop_truncation_level | -0.443481 | -0.467957 | -0.421209 | 24 | 0.093750 |
| D | local | drop_body_orientation | -0.404501 | -0.427709 | -0.383741 | 24 | 0.093750 |
| D | local | equal_normalized | -0.408989 | -0.430019 | -0.390766 | 24 | 0.093750 |
| D | online | full | -0.412499 | -0.436091 | -0.393137 | 24 | — |
| D | online | drop_size | -0.279910 | -0.286207 | -0.273301 | 24 | 0.093750 |
| D | online | drop_body_visibility | -0.430381 | -0.455040 | -0.409099 | 24 | 0.093750 |
| D | online | drop_occlusion_level | -0.417486 | -0.438826 | -0.398510 | 24 | 0.093750 |
| D | online | drop_truncation_level | -0.437953 | -0.460305 | -0.418179 | 24 | 0.093750 |
| D | online | drop_body_orientation | -0.406260 | -0.427776 | -0.387644 | 24 | 0.093750 |
| D | online | equal_normalized | -0.410006 | -0.431934 | -0.390371 | 24 | 0.093750 |

## S4 All selection contrasts, including negative results

The main comparison imposes the same category/stratum risk-cell quotas for all strategies. Error quantities are matched within each true category by exact rarefaction. AUC is normalized trapezoidal area over budget and includes the zero-budget origin. It is not ROC-AUC. A classifier-representation cluster is a diversity proxy, not a confirmed fault root cause. DeepGD is discussed as related work but is not evaluated directly.

**Table S3. Complete selection contrasts in COCO; primary, secondary and joint corrections are retained.**

| source | baseline | metric | mean_delta_auc | ci95_low | ci95_high | holm_primary4_p | holm_strong12_p | holm_ablation4_p | holm_all20_p |
|---|---|---|---|---|---|---|---|---|---|
| local | random | confusion | 0.015767 | 0.008119 | 0.023499 | 0.031250 | — | — | 0.156250 |
| local | random | clusters | 0.010260 | 0.003895 | 0.016955 | 0.031250 | — | — | 0.234375 |
| local | deepgini | confusion | 0.019681 | 0.007655 | 0.032873 | — | 0.140625 | — | 0.234375 |
| local | deepgini | clusters | 0.015902 | 0.006043 | 0.025997 | — | 0.093750 | — | 0.156250 |
| local | dsa | confusion | 0.020539 | 0.008621 | 0.032707 | — | 0.140625 | — | 0.234375 |
| local | dsa | clusters | 0.012191 | 0.003040 | 0.021795 | — | 0.164062 | — | 0.281250 |
| local | kcenter | confusion | 0.007498 | -0.011690 | 0.025521 | — | 1.000000 | — | 1.000000 |
| local | kcenter | clusters | 0.007112 | -0.002278 | 0.017711 | — | 0.843750 | — | 1.000000 |
| local | coverage_without_H | confusion | -0.006047 | -0.017491 | 0.003996 | — | — | 0.867188 | 1.000000 |
| local | coverage_without_H | clusters | 0.003748 | -0.002666 | 0.010891 | — | — | 0.867188 | 1.000000 |
| online | random | confusion | 0.022426 | 0.015171 | 0.030498 | 0.031250 | — | — | 0.156250 |
| online | random | clusters | 0.007021 | 0.000565 | 0.013161 | 0.046875 | — | — | 0.515625 |
| online | deepgini | confusion | 0.025044 | 0.013627 | 0.038469 | — | 0.093750 | — | 0.156250 |
| online | deepgini | clusters | 0.006035 | -0.003550 | 0.016231 | — | 0.843750 | — | 1.000000 |
| online | dsa | confusion | 0.024869 | 0.016251 | 0.035043 | — | 0.093750 | — | 0.156250 |
| online | dsa | clusters | 0.004075 | -0.009497 | 0.017448 | — | 1.000000 | — | 1.000000 |
| online | kcenter | confusion | 0.006423 | -0.006328 | 0.020949 | — | 1.000000 | — | 1.000000 |
| online | kcenter | clusters | -0.004344 | -0.013047 | 0.004483 | — | 0.968750 | — | 1.000000 |
| online | coverage_without_H | confusion | 0.008563 | -0.000076 | 0.017393 | — | — | 0.281250 | 0.703125 |
| online | coverage_without_H | clusters | -0.003315 | -0.011348 | 0.005038 | — | — | 0.867188 | 1.000000 |

**Table S4. Complete selection contrasts in VOC2007; primary, secondary and joint corrections are retained.**

| source | baseline | metric | mean_delta_auc | ci95_low | ci95_high | holm_primary4_p | holm_strong12_p | holm_ablation4_p | holm_all20_p |
|---|---|---|---|---|---|---|---|---|---|
| local | random | confusion | 0.001850 | -0.005078 | 0.008072 | 0.984375 | — | — | 1.000000 |
| local | random | clusters | 0.001017 | -0.006310 | 0.007646 | 0.984375 | — | — | 1.000000 |
| local | deepgini | confusion | 0.003875 | -0.003650 | 0.010280 | — | 0.781250 | — | 1.000000 |
| local | deepgini | clusters | 0.006900 | -0.002095 | 0.015250 | — | 0.765625 | — | 1.000000 |
| local | dsa | confusion | 0.004326 | -0.003122 | 0.010542 | — | 0.781250 | — | 1.000000 |
| local | dsa | clusters | 0.006054 | -0.003797 | 0.015090 | — | 0.781250 | — | 1.000000 |
| local | kcenter | confusion | -0.007189 | -0.016672 | 0.001765 | — | 0.765625 | — | 1.000000 |
| local | kcenter | clusters | -0.008946 | -0.019773 | 0.002501 | — | 0.781250 | — | 1.000000 |
| local | coverage_without_H | confusion | -0.001460 | -0.009866 | 0.005271 | — | — | 0.734375 | 1.000000 |
| local | coverage_without_H | clusters | 0.002283 | -0.004152 | 0.008680 | — | — | 0.632812 | 1.000000 |
| online | random | confusion | 0.018487 | 0.013296 | 0.024372 | 0.031250 | — | — | 0.156250 |
| online | random | clusters | 0.016377 | 0.009978 | 0.022734 | 0.031250 | — | — | 0.156250 |
| online | deepgini | confusion | 0.023782 | 0.018342 | 0.029212 | — | 0.093750 | — | 0.156250 |
| online | deepgini | clusters | 0.019489 | 0.010451 | 0.029235 | — | 0.093750 | — | 0.156250 |
| online | dsa | confusion | 0.019747 | 0.014295 | 0.025094 | — | 0.093750 | — | 0.156250 |
| online | dsa | clusters | 0.017812 | 0.009261 | 0.027013 | — | 0.093750 | — | 0.156250 |
| online | kcenter | confusion | 0.019069 | 0.009016 | 0.028910 | — | 0.093750 | — | 0.156250 |
| online | kcenter | clusters | 0.011902 | -0.003506 | 0.026497 | — | 0.781250 | — | 1.000000 |
| online | coverage_without_H | confusion | 0.012746 | 0.004974 | 0.020654 | — | — | 0.062500 | 0.203125 |
| online | coverage_without_H | clusters | 0.008773 | -0.003350 | 0.021343 | — | — | 0.632812 | 1.000000 |

**Table S5. Complete selection contrasts in Objects365; primary, secondary and joint corrections are retained.**

| source | baseline | metric | mean_delta_auc | ci95_low | ci95_high | holm_primary4_p | holm_strong12_p | holm_ablation4_p | holm_all20_p |
|---|---|---|---|---|---|---|---|---|---|
| local | random | confusion | 0.037697 | 0.026298 | 0.050029 | 0.031250 | — | — | 0.156250 |
| local | random | clusters | 0.026202 | 0.020512 | 0.032083 | 0.031250 | — | — | 0.156250 |
| local | deepgini | confusion | 0.029785 | 0.015247 | 0.045201 | — | 0.093750 | — | 0.156250 |
| local | deepgini | clusters | 0.036619 | 0.029518 | 0.043462 | — | 0.093750 | — | 0.156250 |
| local | dsa | confusion | 0.027917 | 0.012223 | 0.043111 | — | 0.164062 | — | 0.257812 |
| local | dsa | clusters | 0.036871 | 0.028516 | 0.044380 | — | 0.093750 | — | 0.156250 |
| local | kcenter | confusion | 0.004801 | -0.008276 | 0.017576 | — | 1.000000 | — | 1.000000 |
| local | kcenter | clusters | 0.012896 | 0.002233 | 0.022943 | — | 0.312500 | — | 0.437500 |
| local | coverage_without_H | confusion | 0.002956 | -0.010917 | 0.018114 | — | — | 0.679688 | 1.000000 |
| local | coverage_without_H | clusters | 0.012259 | 0.003589 | 0.020541 | — | — | 0.093750 | 0.257812 |
| online | random | confusion | 0.016755 | 0.005241 | 0.028918 | 0.031250 | — | — | 0.156250 |
| online | random | clusters | 0.012321 | 0.005553 | 0.018040 | 0.031250 | — | — | 0.187500 |
| online | deepgini | confusion | -0.003139 | -0.017131 | 0.011643 | — | 1.000000 | — | 1.000000 |
| online | deepgini | clusters | 0.014716 | 0.005235 | 0.024740 | — | 0.093750 | — | 0.156250 |
| online | dsa | confusion | 0.004328 | -0.009572 | 0.019316 | — | 1.000000 | — | 1.000000 |
| online | dsa | clusters | 0.017086 | 0.008123 | 0.026571 | — | 0.093750 | — | 0.156250 |
| online | kcenter | confusion | 0.003257 | -0.014422 | 0.020207 | — | 1.000000 | — | 1.000000 |
| online | kcenter | clusters | 0.019004 | 0.006196 | 0.030827 | — | 0.164062 | — | 0.257812 |
| online | coverage_without_H | confusion | 0.006590 | -0.003426 | 0.016621 | — | — | 0.421875 | 1.000000 |
| online | coverage_without_H | clusters | 0.008227 | 0.001288 | 0.015492 | — | — | 0.093750 | 0.257812 |

## S5 Risk matching and common-representation sensitivity

The exact-histogram control first changes Coverage's feasible set, then matches Random to that Coverage sequence. It is a different protocol from the common quotas. Separate tables for all pools are provided in the evidence bundle. For D, only the local confusion-type contrast passes the original four-test correction under exact matching. The common ViT cluster sensitivity supports D but not B or C. These limits prevent a universal risk-independent or representation-independent selection claim.

## S6 Frozen-state support and transfer

A supplies 87 local and 88 online state–architecture hypotheses. A state requires at least 20 unique targets and a recall deficit of at least ten percentage points at discovery. Confirmation never replaces unsupported hypotheses with favorable new ones. B supports 78/75 combinations; C supports 1/3; D supports 79/81. D retains 69/73 negative directions after averaging seeds, but the primary mean uses the prescribed architecture/run aggregation. Overlapping state descriptions do not create independent defects. Risk-matched residual recall deficits in D (−8.33/−7.80 points) remain descriptive associations.

## S7 Annotation disagreement and reference checks

Strict agreement requires all four annotated risk dimensions to be known and equal. B/C/D retain 166/56/187 targets and only five Q5 targets in each pool. This is insufficient for the planned six-class agreement-only endpoint comparison. Agreement on all 14 shared fields retains 4/0/8 targets. Fieldwise agreement-only confirmation retains many B/D deficit directions, but uses a weaker selected subset and does not validate the entire protocol.

The reference-attribute CSV contains the full precision, recall, specificity, F1 and kappa outputs for official binary Open Images occlusion and truncation. Different official and VLM definitions limit interpretation. Programmatic scale is recalculated on all 7,800 pool targets; agreement of computed scale is not validation of semantic VLM labels.

## S8 Mixtures, rankings and raw failure breadth

The primary mixture budget is 120 (20 per category), with 21 actual risk-share points and 1,000 paired draws per point. No observations are inserted by smoothing. The zero-share suite samples Q1–Q4; it is not a pure Q1 suite. B/D additionally permit 480 targets. Complete panel, class and architecture tables are included. ConvNeXt-Tiny and Swin-T lead mean rankings at all 126 primary settings; image-bootstrap uncertainty remains material, especially on D. Intermediate ranks can move four places.

Raw failed-state breadth increases, whereas class-specific equal-error confusion and cluster changes against the 20%-Q5 reference do not show consistent increases. These controls prevent interpreting more mistakes as automatically broader distinct failure evidence.

## S9 Cost and implementation provenance

B successful-request median/P95 seconds: local 4.00/4.30, online 11.41/14.62. C: local 3.94/4.30, online 9.92/12.31. D all-attempt median/P95 seconds: local 4.00/4.34, online 10.13/13.84; attempt counts are 2,409/2,826. The success-only and all-attempt latency populations are not comparable. No invoice-backed monetary cost is available.

The D source amendment precedes new classification predictions: initial validation sheep supply was insufficient, so fixed train-package prefixes were added and the documented download budget increased from 40 to 80 GiB. The realized fixed-prefix download was 67.76 GiB. Annotation-service and technical replacement records are preserved; selection was not optimized on classifier outcomes.

## S10 Evidence inventory and local reconstruction

The bundle provides source-relative paths and SHA-256 checksums in `evidence_index.json`. Main Table 3 maps to random_summary.csv; Table 4 to ablation_summary.csv; Tables 7–8 to distribution/panel tables and full-stratum NPZs; Table 9 to each pool comparison table; Table 10 to frozen transfer tables; Tables 11–13 to the audited historical manuscript and attribute/disagreement outputs. The new scale tables 5–6 map to summary.csv and area_condition_summary.csv. These indices identify local review evidence, not a public repository.

The full Chinese evidence manuscript is separately preserved as a working archive; its 28 numbered tables, appendix and 17 figures have not been deleted. It is not advertised as an English journal supplement. Figure sources are editable SVG with English labels, plus PDF/EPS/PNG exports. The LaTeX source uses the official Springer class and contains actual author/declaration placeholders; authors must complete those facts and actual data-access conditions before submitting.


## S11 Continuous-area control and residual condition diagnosis

The supplementary cached analysis was specified on 2026-10-07 after earlier outcomes were known. Area-only ranking uses negative continuous official box-area ratio; full risk and area are sorted within class with sample-ID ties and equal quintiles. G is Q1−Q5 macro-F1. The primary paired contrast is G_full−G_area, with eight architecture units, three seeds per unit, 5,000 hierarchical bootstrap draws and a separate six-test Holm family. Random tie priorities are shared between schemes, changed only within exact-score ties, and repeated 1,000 times; their central ranges measure grouping variation, not model-population uncertainty.

**Table S6. Complete paired scale-control outputs, in F1 units.**

| pool | source | targets | layer_total | full_gap | area_gap | increment_mean | ci_low | ci_high | p | positive_runs | negative_runs | q5_retention | q5_jaccard | tie_repeats | tie_increment_mean | tie_increment_low | tie_increment_high | holm_six_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | local | 2400 | 480 | 0.254954 | 0.233778 | 0.021176 | 0.006034 | 0.037515 | 0.039062 | 18 | 6 | 0.500000 | 0.333333 | 1000 | 0.025380 | 0.017434 | 0.033261 | 0.093750 |
| B | online | 2400 | 480 | 0.212959 | 0.233778 | -0.020819 | -0.033875 | -0.006906 | 0.023438 | 5 | 19 | 0.437500 | 0.280000 | 1000 | -0.019773 | -0.028851 | -0.011430 | 0.093750 |
| C | local | 600 | 120 | 0.165747 | 0.109513 | 0.056234 | 0.023546 | 0.088847 | 0.023438 | 20 | 4 | 0.375000 | 0.230769 | 1000 | 0.054411 | 0.037364 | 0.070655 | 0.093750 |
| C | online | 600 | 120 | 0.144354 | 0.109513 | 0.034841 | 0.007936 | 0.062136 | 0.062500 | 17 | 7 | 0.391667 | 0.243523 | 1000 | 0.034178 | 0.016794 | 0.050573 | 0.093750 |
| D | local | 2400 | 480 | 0.418015 | 0.438897 | -0.020883 | -0.032758 | -0.009365 | 0.007812 | 6 | 18 | 0.625000 | 0.454545 | 1000 | -0.013247 | -0.024134 | -0.002897 | 0.046875 |
| D | online | 2400 | 480 | 0.412499 | 0.438897 | -0.026399 | -0.038360 | -0.015486 | 0.007812 | 2 | 22 | 0.518750 | 0.350211 | 1000 | -0.022714 | -0.033289 | -0.011969 | 0.046875 |

Non-scale diagnostic hypotheses retain the discovery-A freeze. Within each class and continuous-area quintile, a state and the same-field known non-state reference must each have at least two images. A hypothesis is evaluable only with at least 20 state images overall, 20 matched state and 20 matched reference images, and at least 50% state retention. Recall differences are weighted by matched state counts. Matching controls quintile membership, not exact continuous area or all confounders. Overlapping hypotheses are not independent defects. All non-scale hypotheses, including nonnegative and insufficient-support rows, appear in area_condition_summary.csv.

## S12 Unknown rates and stratum support

unknown_rates.csv provides every class-specific native field denominator, including protocol-specific fields. The table below aggregates native fields by pool and source; a field absent from the car protocol does not receive a false zero count. stratum_profiles.csv and memberships.csv retain every class-wise layer score/area range and support, for both full-risk and continuous-area schemes.

**Table S7. Unknown fractions by native field; absent fields are excluded from that field denominator.**

| pool | source | field | targets | unknown | unknown_fraction |
|---|---|---|---|---|---|
| B | local | background_complexity | 2400 | 4 | 0.001667 |
| B | local | blur_level | 2400 | 0 | 0.000000 |
| B | local | body_orientation | 2400 | 135 | 0.056250 |
| B | local | body_visibility | 2400 | 29 | 0.012083 |
| B | local | car_color | 400 | 42 | 0.105000 |
| B | local | group_context | 2400 | 9 | 0.003750 |
| B | local | head_orientation | 2000 | 316 | 0.158000 |
| B | local | head_visibility | 2000 | 18 | 0.009000 |
| B | local | illumination | 2400 | 3 | 0.001250 |
| B | local | interaction_context | 2400 | 34 | 0.014167 |
| B | local | leg_visibility | 2000 | 26 | 0.013000 |
| B | local | occluder_type | 2400 | 1 | 0.000417 |
| B | local | occlusion_level | 2400 | 4 | 0.001667 |
| B | local | pose_action | 2000 | 43 | 0.021500 |
| B | local | reflective_surface | 400 | 62 | 0.155000 |
| B | local | scene_environment | 2400 | 7 | 0.002917 |
| B | local | target_background_contrast | 2400 | 4 | 0.001667 |
| B | local | target_recognizability | 2400 | 0 | 0.000000 |
| B | local | truncation_level | 2400 | 4 | 0.001667 |
| B | local | truncation_parts | 2000 | 2 | 0.001000 |
| B | local | visible_vehicle_parts | 400 | 20 | 0.050000 |
| B | local | visual_representation | 2400 | 4 | 0.001667 |
| B | local | wheel_visibility | 400 | 1 | 0.002500 |
| B | online | background_complexity | 2400 | 9 | 0.003750 |
| B | online | blur_level | 2400 | 1 | 0.000417 |
| B | online | body_orientation | 2400 | 172 | 0.071667 |
| B | online | body_visibility | 2400 | 22 | 0.009167 |
| B | online | car_color | 400 | 99 | 0.247500 |
| B | online | group_context | 2400 | 65 | 0.027083 |
| B | online | head_orientation | 2000 | 119 | 0.059500 |
| B | online | head_visibility | 2000 | 8 | 0.004000 |
| B | online | illumination | 2400 | 9 | 0.003750 |
| B | online | interaction_context | 2400 | 43 | 0.017917 |
| B | online | leg_visibility | 2000 | 67 | 0.033500 |
| B | online | occluder_type | 2400 | 20 | 0.008333 |
| B | online | occlusion_level | 2400 | 18 | 0.007500 |
| B | online | pose_action | 2000 | 29 | 0.014500 |
| B | online | reflective_surface | 400 | 139 | 0.347500 |
| B | online | scene_environment | 2400 | 9 | 0.003750 |
| B | online | target_background_contrast | 2400 | 17 | 0.007083 |
| B | online | target_recognizability | 2400 | 8 | 0.003333 |
| B | online | truncation_level | 2400 | 22 | 0.009167 |
| B | online | truncation_parts | 2000 | 5 | 0.002500 |
| B | online | visible_vehicle_parts | 400 | 67 | 0.167500 |
| B | online | visual_representation | 2400 | 5 | 0.002083 |
| B | online | wheel_visibility | 400 | 89 | 0.222500 |
| C | local | background_complexity | 600 | 0 | 0.000000 |
| C | local | blur_level | 600 | 0 | 0.000000 |
| C | local | body_orientation | 600 | 10 | 0.016667 |
| C | local | body_visibility | 600 | 0 | 0.000000 |
| C | local | car_color | 100 | 0 | 0.000000 |
| C | local | group_context | 600 | 0 | 0.000000 |
| C | local | head_orientation | 500 | 36 | 0.072000 |
| C | local | head_visibility | 500 | 0 | 0.000000 |
| C | local | illumination | 600 | 0 | 0.000000 |
| C | local | interaction_context | 600 | 0 | 0.000000 |
| C | local | leg_visibility | 500 | 0 | 0.000000 |
| C | local | occluder_type | 600 | 0 | 0.000000 |
| C | local | occlusion_level | 600 | 0 | 0.000000 |
| C | local | pose_action | 500 | 13 | 0.026000 |
| C | local | reflective_surface | 100 | 1 | 0.010000 |
| C | local | scene_environment | 600 | 1 | 0.001667 |
| C | local | target_background_contrast | 600 | 0 | 0.000000 |
| C | local | target_recognizability | 600 | 0 | 0.000000 |
| C | local | truncation_level | 600 | 0 | 0.000000 |
| C | local | truncation_parts | 500 | 0 | 0.000000 |
| C | local | visible_vehicle_parts | 100 | 0 | 0.000000 |
| C | local | visual_representation | 600 | 0 | 0.000000 |
| C | local | wheel_visibility | 100 | 0 | 0.000000 |
| C | online | background_complexity | 600 | 1 | 0.001667 |
| C | online | blur_level | 600 | 1 | 0.001667 |
| C | online | body_orientation | 600 | 18 | 0.030000 |
| C | online | body_visibility | 600 | 1 | 0.001667 |
| C | online | car_color | 100 | 9 | 0.090000 |
| C | online | group_context | 600 | 8 | 0.013333 |
| C | online | head_orientation | 500 | 11 | 0.022000 |
| C | online | head_visibility | 500 | 0 | 0.000000 |
| C | online | illumination | 600 | 2 | 0.003333 |
| C | online | interaction_context | 600 | 6 | 0.010000 |
| C | online | leg_visibility | 500 | 9 | 0.018000 |
| C | online | occluder_type | 600 | 2 | 0.003333 |
| C | online | occlusion_level | 600 | 2 | 0.003333 |
| C | online | pose_action | 500 | 8 | 0.016000 |
| C | online | reflective_surface | 100 | 12 | 0.120000 |
| C | online | scene_environment | 600 | 4 | 0.006667 |
| C | online | target_background_contrast | 600 | 1 | 0.001667 |
| C | online | target_recognizability | 600 | 1 | 0.001667 |
| C | online | truncation_level | 600 | 2 | 0.003333 |
| C | online | truncation_parts | 500 | 0 | 0.000000 |
| C | online | visible_vehicle_parts | 100 | 5 | 0.050000 |
| C | online | visual_representation | 600 | 0 | 0.000000 |
| C | online | wheel_visibility | 100 | 11 | 0.110000 |
| D | local | background_complexity | 2400 | 2 | 0.000833 |
| D | local | blur_level | 2400 | 0 | 0.000000 |
| D | local | body_orientation | 2400 | 241 | 0.100417 |
| D | local | body_visibility | 2400 | 55 | 0.022917 |
| D | local | car_color | 400 | 23 | 0.057500 |
| D | local | group_context | 2400 | 7 | 0.002917 |
| D | local | head_orientation | 2000 | 473 | 0.236500 |
| D | local | head_visibility | 2000 | 30 | 0.015000 |
| D | local | illumination | 2400 | 5 | 0.002083 |
| D | local | interaction_context | 2400 | 52 | 0.021667 |
| D | local | leg_visibility | 2000 | 37 | 0.018500 |
| D | local | occluder_type | 2400 | 0 | 0.000000 |
| D | local | occlusion_level | 2400 | 4 | 0.001667 |
| D | local | pose_action | 2000 | 80 | 0.040000 |
| D | local | reflective_surface | 400 | 47 | 0.117500 |
| D | local | scene_environment | 2400 | 3 | 0.001250 |
| D | local | target_background_contrast | 2400 | 7 | 0.002917 |
| D | local | target_recognizability | 2400 | 0 | 0.000000 |
| D | local | truncation_level | 2400 | 5 | 0.002083 |
| D | local | truncation_parts | 2000 | 2 | 0.001000 |
| D | local | visible_vehicle_parts | 400 | 8 | 0.020000 |
| D | local | visual_representation | 2400 | 6 | 0.002500 |
| D | local | wheel_visibility | 400 | 0 | 0.000000 |
| D | online | background_complexity | 2400 | 9 | 0.003750 |
| D | online | blur_level | 2400 | 4 | 0.001667 |
| D | online | body_orientation | 2400 | 243 | 0.101250 |
| D | online | body_visibility | 2400 | 65 | 0.027083 |
| D | online | car_color | 400 | 83 | 0.207500 |
| D | online | group_context | 2400 | 104 | 0.043333 |
| D | online | head_orientation | 2000 | 211 | 0.105500 |
| D | online | head_visibility | 2000 | 49 | 0.024500 |
| D | online | illumination | 2400 | 17 | 0.007083 |
| D | online | interaction_context | 2400 | 94 | 0.039167 |
| D | online | leg_visibility | 2000 | 154 | 0.077000 |
| D | online | occluder_type | 2400 | 57 | 0.023750 |
| D | online | occlusion_level | 2400 | 53 | 0.022083 |
| D | online | pose_action | 2000 | 110 | 0.055000 |
| D | online | reflective_surface | 400 | 133 | 0.332500 |
| D | online | scene_environment | 2400 | 9 | 0.003750 |
| D | online | target_background_contrast | 2400 | 35 | 0.014583 |
| D | online | target_recognizability | 2400 | 18 | 0.007500 |
| D | online | truncation_level | 2400 | 43 | 0.017917 |
| D | online | truncation_parts | 2000 | 30 | 0.015000 |
| D | online | visible_vehicle_parts | 400 | 65 | 0.162500 |
| D | online | visual_representation | 2400 | 16 | 0.006667 |
| D | online | wheel_visibility | 400 | 80 | 0.200000 |

## S13 Reanalysis entry point, prompts and implementation scope

Extract the portable ZIP and run `python scripts/run_scale_only_confirmation.py --input portable_inputs --output reproduced_scale`. Requires Python with NumPy, Pandas and scikit-learn; pinned observed versions are in environment.json. No network, GPU or API credentials are required. The bundle provides the exact native old/car prompts, three annotation schemas, two-source automatic states, all 24-classifier predictions/logits on B/C/D, programmatic score parts, frozen historical full-risk memberships, target hashes and fixed weak-state hypotheses. It reproduces the new scale contrast, all historical full-risk layer metrics, area-conditioned diagnostics and unknown/support tables. Older random grouping and mixture outputs, selection analysis tables and provenance are included with the available caches. The entry point does not re-create original image sampling, classifier training or historical VLM calls.

For the old selection breadth comparisons, fixed/equal-error/random metric tables are retained and budget AUC can be reaggregated without relabeling. Original embedding feature extraction and clustering are not regenerated by the portable scale script. Model/version strings and recorded request settings are provided without credentials. API-version aliases do not establish provider-side weight hashes.

The second standalone command is `python scripts/reaggregate_selection_evidence.py --input historical_selection --output reproduced_selection`. It recomputes all 60 historical mean-AUC contrasts, exact architecture-level sign-flip p values and four correction families, and checks them against retained comparison tables. It does not reconstruct the original bootstrap intervals. Archived implementation files support inspection of historical definitions but are not standalone end-to-end runners.

**Table S8. Historical common-quota baseline implementation details.**

| Component | Recorded implementation |
|---|---|
| Common feasibility | Within class and each full-risk quintile, allocate by the joint cell of score and unknown-dimension count using proportional largest remainder. B/D add 20 per class per layer; C adds 10. Quotas and orders are retained. |
| Coverage | Greedy sum of log(n+2)−log(n+1) over active field–state indicators whose previous count is below 20; counts persist across layers; ascending sample-ID ties. |
| Random | 1,000 without-replacement within-cell sequences; the retained random_orders.npz fixes the actual selections. |
| DeepGini | 1−sum(p_k squared), with probabilities from stabilized softmax of frozen logits; choose highest score within the common feasible set. |
| DSA | Raw classifier feature distances. Nearest training example has label equal to the candidate predicted class; divide candidate-to-anchor distance by anchor-to-nearest different-label training distance, with denominator floor 1e-12. Choose highest score. |
| k-center | L2-normalized ViT-B/16 vectors; cosine distance 1−dot product. First choose feasible target closest to the current layer centroid; later maximize nearest-selected distance across layers, with ascending ID ties. |
| Coverage without H | Remove target_scale, body_visibility, occlusion_level, truncation_level, body_orientation and target_recognizability; retain the other nine shared fields under unchanged quotas. |
| Error clusters | Per true category and frozen model/seed, L2-normalized wrong-target features; k=max(2,min(8,round(sqrt(number of errors)))) for at least two errors. Zero/one error has zero/one cluster. KMeans n_init=20, random_state=20260928+training_seed+class_offset. |

## S14 Real-image cases and practical effect sizes

The four displayed targets are illustrative, chosen by the explicit first-ID rule in case_manifest.json within the four pre-existing manuscript conditions. Predictions are drawn from frozen caches; labels are automatic and are not certified by manual ground truth. The displayed classifier input is the stored margin crop after Resize(256) and CenterCrop(224), before tensor normalization. Original-image context available to the VLM is not the classifier input. These selected cases cannot estimate field-label accuracy or population failure rates.

practical_breadth.csv reports equal-error expected normalized confusion/cluster breadth at the largest budget, in addition to the budget AUC. Fractional discoveries arise from exact error-count rarefaction; they are not integer independently verified bugs. Universe sizes vary by architecture/representation, so a normalized panel mean cannot be multiplied by a single arbitrary denominator.

**Table S9. Largest-budget equal-error breadth: Coverage (left) versus matched Random (right); normalized fractions, not percentage points or new significance tests.**

| pool | source | budget_per_class | confusion_left_mean | confusion_right_mean | confusion_delta | cluster_left_mean | cluster_right_mean | cluster_delta |
|---|---|---|---|---|---|---|---|---|
| B | local | 100 | 0.798366 | 0.761792 | 0.036575 | 0.782311 | 0.767939 | 0.014372 |
| B | online | 100 | 0.815555 | 0.786187 | 0.029368 | 0.818311 | 0.799595 | 0.018716 |
| C | local | 50 | 0.685316 | 0.683457 | 0.001858 | 0.757456 | 0.748687 | 0.008770 |
| C | online | 50 | 0.657105 | 0.624356 | 0.032749 | 0.709905 | 0.682773 | 0.027132 |
| D | local | 100 | 0.894858 | 0.856472 | 0.038386 | 0.897447 | 0.882038 | 0.015409 |
| D | online | 100 | 0.862217 | 0.852786 | 0.009432 | 0.876406 | 0.875725 | 0.000681 |
