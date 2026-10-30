# Supplementary Information: Visual-State Risk Stratification

## S1 Protocol, schema and reproducibility scope

The supplement accompanies the revised empirical manuscript. Numbers below are retained results, not new model runs. The evidence bundle includes CSV tables, protocol files and an index; image assets and full prediction caches remain in the local experiment directories and are not falsely described as publicly deposited.

The generic native object schema and the car prompt are included. The shared mapping is implemented in `scripts/dual_vlm_common.py`; source-selection and confirmation logic is preserved by the accompanying protocol documents. Unknown states remain unknown. Original-image scale is computed, not inferred by a VLM.

**Table S1. Generic native fields and allowed states. Shared-field normalization is given in the included source code.**

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

The bundle provides source-relative paths and SHA-256 checksums in `evidence_index.json`. Main Table 3 maps to random_summary.csv; Table 4 to ablation_summary.csv; Tables 5–6 to distribution/panel tables and full-stratum NPZs; Table 7 to each pool comparison table; Table 8 to frozen transfer tables; Tables 9–11 to the audited historical manuscript and attribute/disagreement outputs. These indices identify retained local evidence rather than a currently public repository.

The full Chinese evidence manuscript is separately preserved as a working archive; its 28 numbered tables, appendix and 17 figures have not been deleted. It is not advertised as an English journal supplement. Figure sources are editable SVG with English labels, plus PDF/EPS/PNG exports. The LaTeX source uses the official Springer class and contains actual author/declaration placeholders; authors must complete those facts and actual data-access conditions before submitting.
