#!/usr/bin/env python3
"""Freeze the next validation protocol and report audited data readiness."""
from collections import Counter
import json
from pathlib import Path
import pickletools
import shutil
import zipfile

import numpy as np
import ijson

from audit_fresh_confirmation_20261112 import ROOT, OUT, CLASSES, read, write, dump, sha, tokens, now

DOC = ROOT / "docs/实验方案与结果/新数据资格审计与冻结验证方案_20261112.md"


def review_opaque_caches():
    results = []
    for r in read(OUT / "scan_issues.csv"):
        if r["issues"] != "object_array_not_unpickled:parameter_names.npy":
            raise ValueError("Unexpected unresolved scan issue: " + r["path"])
        p = Path(r["path"])
        if sha(p) != r["sha256"]:
            raise ValueError("Changed opaque cache")
        with zipfile.ZipFile(p) as z, z.open("parameter_names.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            strings = [arg for op, arg, _ in pickletools.genops(f.read()) if isinstance(arg, str)]
        ids, obj, _ = tokens("\n".join(strings).encode())
        if ids or obj:
            raise ValueError("Extra source IDs require a versioned audit rerun")
        results.append(dict(path=str(p), sha256=r["sha256"], array="parameter_names", shape=list(shape),
            dtype=str(dtype), parsed_string_literals=len(strings), additional_image_ID_tokens=0,
            execution="pickletools opcode inspection only; no unpickling or execution"))
    dump("opaque_cache_review.json", dict(status="pass", files=results))


def copy_input(path):
    dest = OUT / "protocol_inputs" / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return dict(original=str(path), snapshot=str(dest), sha256=sha(dest))


def main():
    if (OUT / "completion.json").exists():
        raise FileExistsError("This readiness package is already complete")
    a = json.loads((OUT / "audit_summary.json").read_text())
    b = json.loads((OUT / "alternative_supply.json").read_text())
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification["status"] == "pass"
    sizes = json.loads((OUT / "package_size_audit.json").read_text())
    assert sizes["all_missing_package_sizes_resolved"]
    # Independently verify the image half of the 36 annotation/image bindings.
    plan = read(OUT / "alternative_provisional_plan.csv")
    smoke = []
    for c in CLASSES:
        for role in ("selection", "reference", "technical_reserve"):
            smoke += [r for r in plan if r["class_name"] == c and r["role"] == role][:2]
    wanted = {int(r["source_image_id"]):r for r in smoke}
    found = set()
    metadata = ROOT / "artifacts/objects365_independent_confirmation_20261005/metadata/zhiyuan_objv2_train.json"
    with metadata.open("rb") as f:
        for image in ijson.items(f, "images.item", use_float=True):
            r = wanted.get(image["id"])
            if r is not None:
                assert (image["width"], image["height"], image["file_name"]) == (int(r["width"]), int(r["height"]), r["source_file_name"])
                found.add(image["id"])
                if found == set(wanted):
                    break
    assert found == set(wanted)
    dump("official_image_binding_smoke.json", dict(status="pass", official_image_records_checked=len(found)))
    review_opaque_caches()
    fixed = [ROOT / "configs/category_five_dimension_pilot_v1.json",
             ROOT / "configs/recent_five_models_20261009.json",
             ROOT / "artifacts/paper_claim_validation_20261106/thresholds.json",
             ROOT / "scripts/prepare_multiclass_object_candidate_pool.py",
             ROOT / "scripts/annotate_other_classes_specific_v2_20261018.py",
             ROOT / "scripts/annotate_dual_vlm.py",
             ROOT / "scripts/run_dog_local_pipeline.py",
             ROOT / "scripts/run_design_robustness_20261110.py",
             ROOT / "scripts/verify_cross_vlm_analysis_20261112.py",
             ROOT / "scripts/audit_multiclass_near_duplicates.py",
             ROOT / "scripts/run_recent_five_models.py",
             ROOT / "output/experiments/论文关键结论验证_20261106/common.py",
             ROOT / "output/experiments/实证论文补全_20261108/execute.py"]
    for domain in ("old", "car"):
        fixed += [ROOT / f"artifacts/sixclass_independent_confirmation_20261004/configs/{domain}_annotation_schema.json",
                  ROOT / f"artifacts/sixclass_independent_confirmation_20261004/prompts/{domain}.txt"]
    fixed += [ROOT / f"artifacts/paper_claim_validation_20261106/{c}_dictionary.json" for c in CLASSES]
    fixed += [ROOT / f"artifacts/other_classes_specific_experiment_20261018/frozen_dictionaries/{c}.json" for c in CLASSES[2:]]
    snapshots = [copy_input(p) for p in fixed]
    dump("protocol_input_binding.json", dict(files=snapshots,
        inherited_code_role="Frozen algorithm definitions and annotation templates; a new-manifest execution adapter must pass smoke checks before the full run."))
    protocol = dict(version="fresh-confirmation-v1-20261112", frozen_at=now(),
        current_stage="design_frozen_data_not_qualified", human_verification=False,
        scientific_question="Does risk-constrained selection increase label-defined evaluability on project-unseen images, and does P20 add useful benefit over exact Greedy20?",
        thesis_change=False,
        sources=dict(primary_same_source_COCO="infeasible_under_current_conservative_exclusions",
            alternative="Objects365-v2 train; external-source confirmation track, not COCO same-source replication",
            alternative_status=b["status"], plan="alternative_provisional_plan.csv",
            no_claim_of_global_pretraining_independence=True),
        sampling=dict(seed=2026111201, classes=CLASSES, selection_per_class=400, reference_per_class=400,
            total_primary_images=4800, technical_reserves_per_class=100,
            role_rule="hash rank 1..400 selection; 401..800 reference; 801..850 selection technical reserve; 851..900 reference technical reserve",
            one_original_image_one_role_and_one_class=True,
            allowed_reserve_use="in frozen rank order for download/decode/official-box/exact-or-near-duplicate failures only, before annotations/predictions",
            forbidden_reserve_use="no replacement for model disagreement, unknown labels, poor support or classifier results",
            stop_if_qualified_supply_insufficient=True),
        data_gate=dict(required=["all planned originals decoded and official dimensions/bounds checked",
            "all bound paths SHA256 verified against historical registry and actual available historical original/crop content",
            "all selection/reference/train/development exact and near duplicates excluded at original-image level",
            "cross-class and cross-role duplicates excluded before any annotation",
            "manifest and source archives/range-download checks frozen; no unresolved lineage/input failures"],
            near_duplicate_rule="Conservative full-original dHash64<=4 including horizontal mirror; target-crop pHash<=4 and frozen ViT cosine>=0.980 OR pHash<=8 and cosine>=0.995; exact SHA always excluded. Apply against accessible historical originals/crops and within the new pools; missing historical originals prevent certification.",
            threshold_source="existing project acquisition and near-duplicate rules; no tuning using new outputs",
            maximum_simultaneous_staged_archive_bytes=80*1024**3,
            maximum_network_payload_bytes=120*1024**3, minimum_free_disk_after_staging_bytes=20*1024**3,
            acquisition="Sequential package streaming/extraction of frozen target basenames; do not retain full new archives or duplicate range chunks. Preserve verified target files and acquisition provenance. This changes storage strategy, not target membership.",
            package_sizes_must_be_verified_before_download=True),
        annotations=dict(models=dict(A="qwen3-vl:30b-a3b-instruct", B="sensenova-6.8-flash-lite"),
            frozen_native_model_A_digest="c871fc73fabc5516500b70a298ea25fd44a6a23d5cffc46c63b50302543e3915",
            protocol="original native old/car schemas and templates; original independent-named-v2 fine schemas/prompts for horse/sheep/person/car",
            revised_pilot_prompt_used=False,
            backend_symmetry="same task text, schema and evidence bytes for A/B; transport-only differences logged; no first-model answer sharing",
            native_evidence="original + marked original + feature crop, frozen ensure_views geometry",
            fine_evidence="original+official-target-crop contact sheet, original renderer",
            sampling=dict(temperature=0, native_seed=20260911, fine_seed=20261018,
                native_num_ctx=8192, native_num_predict=2600, fine_num_ctx=16384, fine_num_predict=2000),
            max_attempts_per_job=3,
            failure="retain raw responses and schema failures; formatting reminder only, no label remapping/adjudication or prompt redesign; unresolved jobs block affected class/source",
            unknown="excluded from positive states and from known-negative controls",
            disagreement="retain both independent versions; report agreement/unknown; no third-model majority substituted for truth",
            semantic_scope="native old/car truncation semantics are historical and differ; no claim of unified or corrected visual truth",
            base_semantic_jobs=16000, maximum_attempts=48000),
        selection=dict(methods=["R", "Greedy20", "P20"], budget_per_class=200,
            risk="equal_parts midpoint/uncertainty from frozen five-dimension config; quintiles from C_select only with sample-ID tie rule",
            cell="(class,Q,round(midpoint*1e8),round(uncertainty_width*1e8))",
            quotas="same class/Q/fine-cell quota vector for all three methods, largest remainder within Q; 40 selected per Q",
            starts=list(range(2026110600,2026110650)),
            R="same-cell random initial200, shared by Greedy20/P20 at each start",
            edits="at most20 original members removed; exact frozen Greedy and P20 implementations",
            P20="historical support-first/occurrence-second formulation and finite-solver fallback; any execution adapter must reproduce historical fixtures before use",
            solver=dict(time_limit_per_stage_seconds=20, mip_rel_gap=0, threads=1, random_seed=2026110600),
            original_numeric_thresholds="copied historical thresholds.json, no new-source recalibration",
            selection_sources="A primary; B own-source predefined robustness, reported as a change of the whole risk/coverage/initialization bundle",
            freeze_before_classifier_outputs=True,
            reference_not_used_for_selection=True),
        endpoints=dict(primary="per-class mean number of evaluable state predicates: state-positive and known same-field complement each >=20 distinct originals; macro-average six classes, then paired50 starts",
            coverage_secondary="capped-support completion mean(min(n_state/20,1)), fixed state dictionary and denominator; plus occurrence/dual-pass",
            label_evaluation="report A and B separately for fixed A selections; agreement-core separately. B own-source sets separately; no source pooling or best-source selection",
            P20_vs_R_support_rule="positive primary macro mean difference under both A and B evaluators; report every class and label source even when mixed",
            P20_vs_Greedy_rule="report primary/coverage difference together with runtime, solver statuses and constraint failures; no post-hoc minimal-useful-effect or universal superiority claim",
            inference="one new image realization;50 starts share data and are not independent datasets. Report paired distributions/min/median/max and class-specific results; no significance claim from seeds as independent replicates"),
        diagnosis=dict(role="secondary", classifiers=["yolo26s_cls", "pe_core_b16", "siglip2_base", "mobileclip2_s0", "aimv2_large"],
            prediction_seeds=[20260916,20260917,20260918], weights="15 best.pt SHA bindings in checkpoint_lineage.csv; no retraining",
            preprocessing="each model's frozen run_recent_five_models.py/model.json inference transform; preserve historical input sizes/normalization",
            correctness="float mean of three seed-wise correct indicators, not majority-vote label",
            weak="positive-vs-known-negative recall gap <=-0.10, both supports>=20",
            confirmed="same predicate weak on independent complete C_ref under same annotation source/classifier",
            main_comparison="intersection of evaluable state predicates for R/Greedy/P20 and C_ref, same universe for all method comparisons",
            additional="all-eligible counts with exact new/lost eligibility vs common-gap decomposition; raw selected error rates, Q gradients, runtime",
            no_outcome_matching_as_primary=True,
            maximum_target_model_seed_predictions=72000),
        sequence=["qualify and bind data", "automatic six-class smoke with frozen protocols; these targets remain in their frozen pools and cannot be tuned against",
            "complete two-model annotation and freeze all labels/masks", "compute risk and freeze R/Greedy/P20 selections without classifier outcomes",
            "infer frozen15 checkpoints on both new pools", "independent numerical audit and report all predefined endpoints"],
        stop_rules=["unresolved lineage/hash/near-duplicate failures block data qualification",
            "annotation smoke may repair transport/schema formatting only; semantic changes require a new version and separate pilot, and invalidate this confirmatory run",
            "no adaptive samples, thresholds, dictionaries, classifiers, seed subsets or label source after seeing outcomes",
            "insufficient support is reported as unevaluable, never filled after predictions",
            "failed or mixed outcomes are reported and bounded; no automatic relabeling/retry-until-positive"],
        result_claim_status="no new empirical efficacy results; readiness/metadata only")
    dump("frozen_validation_protocol.json", protocol)
    old_counts = a["old_reserve_status"]
    table = "\n".join(f"| {c} | {a['remaining_COCO_train_ID_candidates'][c]} | 800 |" for c in CLASSES)
    doc = f"""# 新数据资格审计与冻结验证方案

本轮完成历史资格审计、替代数据源的编号供给核查和验证设计冻结；尚未完成新图片资格认定或新的效果验证。历史原文件及用户已有编辑保持原样。

## 资格审计结果

先前仅查看11月8日的预留状态文件是不完整的：后续真实扩容已使用其中{old_counts['already_development']}个目标，另外{old_counts['previously_rejected']}个已在采集阶段被排除。原2400预留名单不能再作为未使用确认集。逐条证据见`old_reserve_lineage.csv`；保留旧状态文件作为历史快照，不覆写成新状态。

本轮登记{a['files_inventoried']}个文件，扫描{a['text_files_scanned']}个文本和{a['numpy_caches_scanned']}个NumPy缓存，并纳入图片文件名的编号。登记{a['conservative_COCO_ID_exclusions']}个保守COCO编号排除项；这不是已证实使用图片的数量，12位数字筛查可能过度排除。原始官方标注目录只表示数据供给，不当作实验使用记录；断点下载分块偏移也不当作图像编号。

| 类别 | 按当前保守规则剩余的COCO train2017编号候选 | 所需选择池＋参考池 |
|---|---:|---:|
{table}

羊类仅余11个编号候选，因此当前COCO同源路线不满足六类每类400＋400的设计。此结论针对上述筛查规则，不声称整个COCO绝对不存在其他未使用图片。剩余候选均尚未完成实际原图哈希与近重复核验。

五个分类器的项目内微调训练清单共1800目标、关联2100条不同图像/裁剪路径，文件均存在；15份best.pt已逐份计算SHA256。四个object类型缓存只含parameter_names数组，已用pickletools解析字符串操作码补查，未执行pickle，未发现新增图片编号。基础模型预训练数据不可得，因此不声称全局训练数据独立，也不宣称排除了所有未登记的项目外使用。

## 替代数据源供给

使用本地官方Objects365-v2 train原始标签，重新扫描{b['annotations_scanned']}条框标注，排除已登记的{b['historical_train_ID_exclusions']}个历史训练源编号。未读取新的模型结果，未下载新的数据集载荷，未发出VLM或分类器请求。

供给状态：`{b['status']}`。生成{b['provisional_targets']}条互不重叠的暂定编号记录，分配为选择池、参考池及技术储备，详细名单见`alternative_provisional_plan.csv`。涉及{b['required_packages']}个源数据包，其中{len(b['missing_local_archives'])}个本地尚缺；缺包大小不能按0计算。编号供给可行不等于图片独立性通过，后续仍需核验数据包、原图、哈希、近重复和技术储备。

已进行{sizes['header_requests']}次仅响应头的HTTP尺寸探测，缺包合计{sizes['verified_missing_bytes']/1024**3:.2f}GiB，当前可用磁盘{sizes['free_disk_bytes']/1024**3:.2f}GiB；若整批归档同时落盘，只剩{(sizes['free_disk_bytes']-sizes['verified_missing_bytes'])/1024**3:.2f}GiB，不够后续图片、证据与推理缓存。故在新图片下载前冻结逐包流式提取的存储策略：主样本和角色名单不变，不同时保留全部新增归档/分块；暂存上限80GiB、网络载荷上限120GiB、始终预留20GiB磁盘。初始尺寸审计中的80GiB整批下载假设保留为原始工程检查，最终执行约束以冻结协议中的分离网络/暂存限制为准。这不是模型结果驱动的采样修改。

该替代路线属于外部来源确认，不能写成新的COCO同源复现。按供给和成本做源选择，不读取效果后再挑来源。冻结每类400选择＋400参考及100技术储备，主样本共4800张、储备共600张，原图在类别与角色之间全局互斥。

## 冻结后的验证设计

1. 先完成数据资格。按冻结编号顺序仅为下载、解码、官方框或重复失败补充技术储备；禁止按unknown、VLM分歧、覆盖或错误表现换样本。采用逐包流式提取，网络载荷上限120GiB、同时暂存归档上限80GiB、始终至少留20GiB磁盘空间。
2. 全程自动标注。A为Qwen3-VL、B为SenseNova，固定原始native与named-v2细字段协议，两模型接收相同任务文字、schema及证据图片，不互看答案。修订小试验提示词不投入本次确认。原始回复、格式失败、unknown和分歧均保留；一致性不是准确率。最多16000个基础标注任务，每任务最多3次尝试，不能无限重试到满意。
3. 固定比较R、精确Greedy20、P20。每类最终200、每Q40、风险细单元配额相同、最多替换20个原成员、固定50起点、历史数值阈值和求解预算。A为主选择源；B评估同一批A集合，并另做事先规定的B自身选择敏感性，不能把整套标签切换解释成单一覆盖效应。
4. 主指标为“正状态与已知同字段补集均至少20张原图”的可评价条件数，补充封顶支持完成率；类别先宏平均，再汇总配对起点。A/B分别报告，不择优。50起点共用图片，不能当50次独立数据实验。
5. 共同可评价条件上的诊断为次指标。冻结选择后才推理15份历史权重；用三预测种子正确指示的浮点均值。正负均至少20、召回差≤−10个百分点定义弱条件，并在全新参考池验证。同一比较使用R/Greedy/P20及参考池的共同状态集合，同时报告支持量分解、真实错误率和运行成本。
6. 如实记录失败。若P20优于R但与Greedy接近，只支持覆盖/可评价性；共同条件诊断不提高就不宣称提高诊断能力。不得看结果后换阈值、标签源、种子或数据。

精确字段、角色顺序、假设、判定与停止条件见`frozen_validation_protocol.json`；提示词、字典、实现参考已复制到`protocol_inputs/`并绑定哈希。当前冻结的是设计和参考算法，新的数据接入适配器仍需在全面运行前完成历史夹具复现与六类小规模检查。

## 当前交付与下一执行点

本轮未下载新的图片、未调用新的VLM、未进行新的分类器推理。当前可复核的交付为资格报告、逐图历史谱系、保守排除登记、替代源暂定名单、训练权重绑定及冻结验证协议。4项防漏核验测试通过，独立脚本重算COCO供给并核对全部5400个暂定编号/角色；另直接核对36条官方框和36条官方图像元数据。下一步是实现并小规模验证逐包提取接入，绑定图片、完成去重；资格通过后才继续自动标注。当前不能报告“独立验证已完成”。

复现命令依次为`.venv/bin/python scripts/audit_fresh_confirmation_20261112.py`、`.venv/bin/python scripts/audit_alternative_fresh_supply_20261112.py`、`.venv/bin/python scripts/audit_fresh_package_sizes_20261112.py`、`.venv/bin/python scripts/verify_fresh_readiness_20261112.py`、`.venv/bin/python scripts/finalize_fresh_readiness_20261112.py`。脚本拒绝覆盖已完成快照，重做应使用新版本目录。原始扫描预检曾因NumPy私有接口不可用失败，已保存到`artifacts/fresh_confirmation_readiness_preflight_20261112/`，修复为公开header接口后重新完整运行。
"""
    DOC.write_text(doc)
    (OUT / "claim_evidence.md").write_text("# 本轮主张与证据\n\n| 主张 | 状态 | 证据 | 边界 |\n|---|---|---|---|\n| 原预留已失去未使用确认资格 | supported | old_reserve_lineage.csv；扩容new_manifest.csv/acquisition_rejections.csv | 原图编号谱系，非标签准确率 |\n| 当前保守COCO规则不足以完成六类400+400 | supported | audit_summary.json；remaining_coco_ID_candidates.csv；冻结标签zip | 保守数字筛查可能过度排除 |\n| 替代源编号供给 | partial | alternative_supply.json；alternative_provisional_plan.csv | 非图片哈希/近重复资格通过 |\n| 新数据上可评价性或诊断提高 | unsupported/unmeasured | 尚无新预测 | 不得报告为新效果结果 |\n")
    (OUT / "research_log.md").write_text("# 执行记录\n\n用户授权按步骤推进。首先核对原始记录，发现预留名单后续已被扩容使用；完整扫描文本、缓存与文件名，不覆盖历史快照。再进行本地替代源供给审计，固定六类400+400设计、自动标注协议、主要指标及停止规则。所有新供给仅为metadata级别，不发起模型调用。扫描预检失败与修复保留。详见audit_protocol.json、alternative_supply_protocol.json、逐文件注册及代码冻结。\n")
    (OUT / "review.md").write_text("# 复核\n\nCitation integrity：本轮无外部引文，使用本地原始记录；远端只探测数据包响应头体积。\n\nResult traceability：通过。逐图核对预留与扩容/排除清单；供给以冻结官方标签重算；训练路径与15权重哈希绑定；对象缓存用无执行的操作码补查。独立复核命令另记录verification.json，36个框及36条官方图像绑定补查通过。\n\nClaim-scope integrity：通过。仅报告历史资格和编号供给；新图像资格未通过、无新效果结果；基础预训练独立性不声称。COCO替代为Objects365属于另一个确认来源，明确披露。\n\n尚未满足的执行条件：逐包提取接入、实际原图绑定与近重复去重、标注/选择接入适配器检查。它们阻止全面新推理，不阻止本轮审计与协议交付。\n")
    code = [ROOT / "scripts" / name for name in ("audit_fresh_confirmation_20261112.py", "audit_alternative_fresh_supply_20261112.py", "audit_fresh_package_sizes_20261112.py", "verify_fresh_readiness_20261112.py", "finalize_fresh_readiness_20261112.py")]
    code += [ROOT / "tests/test_fresh_confirmation_audit.py"]
    dump("code_freeze.json", {str(p): sha(p) for p in code})
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name not in {"final_freeze.json", "completion.json"}]
    dump("final_freeze.json", {str(p): sha(p) for p in files + [DOC]})
    dump("completion.json", dict(completed_at=now(), status="readiness_audit_and_design_complete",
        empirical_validation_complete=False, data_qualified=False, new_VLM_calls=0, new_classifier_calls=0,
        report=str(DOC), frozen_files=len(files)+1, next_stage="alternative-package-and-image-qualification"))
    print(json.dumps(dict(report=str(DOC), status="readiness_audit_and_design_complete"), ensure_ascii=False))


if __name__ == "__main__":
    main()
