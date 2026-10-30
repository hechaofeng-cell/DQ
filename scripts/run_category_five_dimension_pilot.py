#!/usr/bin/env python3
"""Explore fixed five-dimension rules on existing labels and frozen predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/category_five_dimension_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/category_five_dimension_pilot_20261012"
DIMENSIONS = ("scale", "visibility", "occlusion", "truncation", "orientation")
SCHEMES = ("legacy_global", "equal_global", "equal_parts")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mapped_risk(value, mapping, config):
    if value in config["unknown_values"]:
        return None
    if value not in mapping:
        raise ValueError(f"Unmapped scored value: {value!r}")
    return float(mapping[value])


def aggregate_components(components):
    """Keep the fixed denominator, with explicit bounds for unknown values."""
    if not components or any(w <= 0 for _, w in components):
        raise ValueError("Need positive component weights")
    weight = sum(w for _, w in components)
    if not math.isclose(weight, 1.0):
        raise ValueError(f"Component weights must sum to one: {weight}")
    if any(v is not None and not 0 <= v <= 1 for v, _ in components):
        raise ValueError("Component risk outside [0, 1]")
    lower = sum(v * w for v, w in components if v is not None)
    missing = sum(w for v, w in components if v is None)
    return dict(lower=lower, upper=lower + missing, midpoint=lower + missing / 2,
                known_weight=1 - missing,
                unknown_components=sum(v is None for v, _ in components))


def scale_risk(area, config):
    if not math.isfinite(area) or not 0 < area <= 1:
        raise ValueError(f"Invalid original-image box area ratio: {area}")
    a, b, c = config["scale_edges"]
    return (3 if area < a else 2 if area < b else 1 if area < c else 0) / 3


def score_sample(area, features, category, scheme, config):
    maps = config["maps"]
    components = {"scale": [(scale_risk(area, config), 1.0)]}

    def field(name, map_name=None):
        if name not in features:
            raise ValueError(f"Required scoring field missing: {name}")
        return mapped_risk(features[name], maps[map_name or name], config)

    if scheme == "early_parts":
        map_names = {"head_visibility": "part_visibility", "muzzle_visibility": "part_visibility",
                     "mouth_visibility": "part_visibility", "leg_visibility_count": "legacy_leg_visibility",
                     "tail_visibility": "legacy_tail_visibility"}
        components["visibility"] = [(field(k, map_names[k]), w)
                                    for k, w in config["early_dog_visibility_components"].items()]
        components["orientation"] = [(field("body_orientation", "legacy_body_orientation"), .5),
                                     (field("head_orientation", "legacy_head_orientation"), .5)]
        components["occlusion"] = [(field("occlusion_level", "legacy_occlusion_level"), 1)]
        components["truncation"] = [(field("truncation_level", "legacy_truncation_level"), 1)]
    else:
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown scheme: {scheme}")
        components["occlusion"] = [(field("occlusion_level"), 1)]
        components["truncation"] = [(field("truncation_level"), 1)]
        if scheme == "equal_parts" and category != "car":
            components["visibility"] = [(field(k, "part_visibility" if k == "head_visibility" else k), w)
                                        for k, w in config["noncar_visibility_components"].items()]
            components["orientation"] = [(field(k), w) for k, w in
                                         config["noncar_orientation_components"].items()]
        else:
            components["visibility"] = [(field("body_visibility"), 1)]
            components["orientation"] = [(field("body_orientation"), 1)]
    weights = config["legacy_dimension_weights"] if scheme == "legacy_global" else config["dimension_weights"]
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Dimension weights must sum to one")
    dimensions = {d: aggregate_components(components[d]) for d in DIMENSIONS}
    fully_known = all(math.isclose(v["known_weight"], 1) for v in dimensions.values())
    evaluable = all(v["known_weight"] >= config["minimum_known_weight_per_dimension"] - 1e-12
                    for v in dimensions.values())
    result = dict(score_lower=100 * sum(weights[d] * dimensions[d]["lower"] for d in DIMENSIONS),
                  score_upper=100 * sum(weights[d] * dimensions[d]["upper"] for d in DIMENSIONS),
                  score_midpoint=100 * sum(weights[d] * dimensions[d]["midpoint"] for d in DIMENSIONS),
                  fully_known=fully_known, evaluable=evaluable,
                  unknown_components=sum(v["unknown_components"] for v in dimensions.values()),
                  scoring_components=sum(len(v) for v in components.values()))
    for d in DIMENSIONS:
        for key in ("lower", "upper", "midpoint", "known_weight"):
            result[f"{d}_{key}"] = dimensions[d][key]
    if not -1e-8 <= result["score_lower"] <= result["score_midpoint"] <= result["score_upper"] <= 100 + 1e-8:
        raise ValueError("Invalid risk interval")
    return result


def validate_label(value, row, schema):
    if value["image_id"] != row["sample_id"] or value["target_instance_id"] != row["target_instance_id"]:
        raise ValueError("Annotation target identity mismatch")
    expected = schema.get("universal_features", []) + schema.get("category_specific_features", [])
    features = value["features"]
    if set(features) != {f["feature_id"] for f in expected}:
        raise ValueError("Annotation schema field mismatch")
    for f in expected:
        v = features[f["feature_id"]]
        if f["value_type"] == "set":
            if not isinstance(v, list) or not v or len(set(v)) != len(v):
                raise ValueError("Invalid set field")
            if any(x in v and len(v) > 1 for x in f.get("exclusive_values", [])):
                raise ValueError("Exclusive set value mixed with other values")
            values = v
        else:
            if not isinstance(v, str):
                raise ValueError("Expected enum string")
            values = [v]
        if any(x not in f["possible_values"] for x in values):
            raise ValueError("Annotation enum outside frozen schema")
    return features


def validate_input_binding(candidate, annotation_input, image_size):
    """The native target_1 alias is bound to the official box via input views."""
    if annotation_input["image_id"] != candidate["sample_id"] or annotation_input["target_class"] != candidate["class_name"]:
        raise ValueError("Annotation input class/ID mismatch")
    for native, official in (("image_path", "image_path"), ("marked_image_path", "marked_image_path"),
                             ("crop_image_path", "feature_crop_path")):
        if annotation_input[native] != candidate[official]:
            raise ValueError("Annotation input views differ from candidate target views")
    width, height = image_size
    official = np.asarray(json.loads(candidate["bbox_norm_xyxy"])) * [width, height, width, height]
    native = np.asarray(json.loads(annotation_input["bbox"]))
    if native.shape != (4,) or not np.allclose(native, official, rtol=0, atol=1.1):
        raise ValueError("Annotation input box differs from official target box")
    return dict(sample_id=candidate["sample_id"], target_instance_id=annotation_input["target_instance_id"])


def five_tiers(indices, scores, sample_ids, random_seed=None):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 10:
        raise ValueError("Too few samples for five tiers")
    if random_seed is None:
        ordered = sorted(indices, key=lambda i: (scores[i], sample_ids[i]))
    else:
        rng = np.random.default_rng(random_seed)
        tie_key = {int(i): float(x) for i, x in zip(indices, rng.random(len(indices)))}
        ordered = sorted(indices, key=lambda i: (scores[i], tie_key[int(i)]))
    return [np.asarray(v, dtype=int) for v in np.array_split(ordered, 5)]


def error_auc(errors, scores):
    if len(set(errors.tolist())) != 2:
        return None
    return float(roc_auc_score(errors, scores))


def mean_or_none(values):
    valid = [x for x in values if x is not None]
    return float(np.mean(valid)) if valid else None


def source_scores(rows, labels, config, source):
    records, scores = [], {s: [] for s in SCHEMES}
    for row in rows:
        f = labels[row["sample_id"]]
        for scheme in SCHEMES:
            v = score_sample(float(row["bbox_area_ratio"]), f, row["class_name"], scheme, config)
            scores[scheme].append(v)
            records.append(dict(source=source, sample_id=row["sample_id"], class_name=row["class_name"],
                                scheme=scheme, **v,
                                **{f"feature_{k}": json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
                                   for k, x in f.items()}))
    return records, scores


def bootstrap_architectures(values, reps, seed):
    arrays = [np.asarray(v, dtype=float) for _, v in sorted(values.items()) if v]
    if not arrays:
        return None, None
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(reps):
        model_draw = rng.integers(0, len(arrays), len(arrays))
        sampled.append(np.mean([np.mean(rng.choice(arrays[j], len(arrays[j]), replace=True)) for j in model_draw]))
    lo, hi = np.quantile(sampled, [.025, .975])
    return float(lo), float(hi)


def analyze(rows, all_scores, predictions, config, output):
    ids = [r["sample_id"] for r in rows]
    truth = np.asarray([config["classes"].index(r["class_name"]) for r in rows])
    cohorts, metrics, tier_metrics, rankings, comparisons = [], [], [], [], []
    arrays = {source: {scheme: {key: np.asarray([v[key] for v in values])
                              for key in ("score_midpoint", "score_lower", "score_upper", "fully_known", "evaluable")}
                       for scheme, values in score_set.items()} for source, score_set in all_scores.items()}
    for source in config["sources"]:
        print(f"Evaluate {source}: matched cohorts, 24 prediction runs", flush=True)
        for category in config["classes"]:
            category_indices = np.asarray([i for i, r in enumerate(rows) if r["class_name"] == category])
            flags = {"complete": np.logical_and.reduce([arrays[source][s]["fully_known"] for s in SCHEMES]),
                     "midpoint": np.logical_and.reduce([arrays[source][s]["evaluable"] for s in SCHEMES])}
            for cohort in ("complete", "midpoint"):
                indices = category_indices[flags[cohort][category_indices]]
                cohorts.append(dict(source=source, class_name=category, cohort=cohort,
                                    eligible=len(indices), total=len(category_indices),
                                    retained_fraction=len(indices) / len(category_indices)))
                if len(indices) < 10:
                    continue
                for scheme in SCHEMES:
                    score = arrays[source][scheme]["score_midpoint"]
                    modes = ["midpoint", "random_ties"] if cohort == "complete" else ["midpoint", "lower", "upper"]
                    for mode in modes:
                        selected_score = score if mode in ("midpoint", "random_ties") else arrays[source][scheme][f"score_{mode}"]
                        tiers = five_tiers(indices, selected_score, ids,
                                           config["random_seed"] if mode == "random_ties" else None)
                        boundary_ties = sum(selected_score[tiers[i][-1]] == selected_score[tiers[i + 1][0]] for i in range(4))
                        if mode == "midpoint":
                            for tier, members in enumerate(tiers, 1):
                                for i in members:
                                    rankings.append(dict(source=source, class_name=category, cohort=cohort,
                                                         scheme=scheme, tier=tier, sample_id=ids[i]))
                        for model, seed, predicted in predictions:
                            errors = predicted != truth
                            auc = error_auc(errors[indices].astype(int), selected_score[indices])
                            low, high = float(errors[tiers[0]].mean()), float(errors[tiers[4]].mean())
                            metrics.append(dict(source=source, class_name=category, cohort=cohort,
                                                unknown_mode=mode, scheme=scheme, model=model, seed=seed,
                                                n=len(indices), error_auc=auc, error_gap=high - low,
                                                q1_error=low, q5_error=high, tied_tier_boundaries=boundary_ties))
                            if mode == "midpoint":
                                for tier, members in enumerate(tiers, 1):
                                    tier_metrics.append(dict(source=source, class_name=category, cohort=cohort,
                                                             scheme=scheme, model=model, seed=seed, tier=tier,
                                                             n=len(members), error_rate=float(errors[members].mean()),
                                                             risk_mean=float(selected_score[members].mean())))
    write_csv(output / "tables/cohorts.csv", cohorts)
    write_csv(output / "tables/metrics_by_class_run.csv", metrics)
    write_csv(output / "tables/tier_metrics_by_class_run.csv", tier_metrics)
    write_csv(output / "tables/tier_assignments.csv", rankings)
    class_summary = []
    grouped = defaultdict(list)
    for v in metrics:
        grouped[tuple(v[k] for k in ("source", "class_name", "cohort", "unknown_mode", "scheme"))].append(v)
    for key, group in sorted(grouped.items()):
        class_summary.append(dict(zip(("source", "class_name", "cohort", "unknown_mode", "scheme"), key),
                                  n=group[0]["n"], error_auc_mean=mean_or_none([x["error_auc"] for x in group]),
                                  error_gap_mean=float(np.mean([x["error_gap"] for x in group])),
                                  q1_error_mean=float(np.mean([x["q1_error"] for x in group])),
                                  q5_error_mean=float(np.mean([x["q5_error"] for x in group]))))
    write_csv(output / "tables/summary_by_class.csv", class_summary)
    run_macro = []
    grouped = defaultdict(list)
    for v in metrics:
        grouped[tuple(v[k] for k in ("source", "cohort", "unknown_mode", "scheme", "model", "seed"))].append(v)
    for key, group in sorted(grouped.items()):
        run_macro.append(dict(zip(("source", "cohort", "unknown_mode", "scheme", "model", "seed"), key),
                              classes=len(group), error_auc=mean_or_none([x["error_auc"] for x in group]),
                              error_gap=float(np.mean([x["error_gap"] for x in group]))))
    write_csv(output / "tables/metrics_macro_by_run.csv", run_macro)
    index = {tuple(v[k] for k in ("source", "cohort", "unknown_mode", "scheme", "model", "seed")): v for v in run_macro}
    for source in config["sources"]:
        for cohort in ("complete", "midpoint"):
            for before, after in (("legacy_global", "equal_global"), ("equal_global", "equal_parts")):
                for metric in ("error_auc", "error_gap"):
                    diffs = defaultdict(list)
                    for model, seed, _ in predictions:
                        a = index.get((source, cohort, "midpoint", before, model, seed))
                        b = index.get((source, cohort, "midpoint", after, model, seed))
                        if a and b and a[metric] is not None and b[metric] is not None:
                            diffs[model].append(b[metric] - a[metric])
                    lo, hi = bootstrap_architectures(diffs, config["model_bootstrap_replicates"], config["random_seed"])
                    model_means = [np.mean(x) for x in diffs.values()]
                    comparisons.append(dict(source=source, cohort=cohort, before=before, after=after, metric=metric,
                                            mean_delta=mean_or_none(model_means), ci_low=lo, ci_high=hi,
                                            architectures=len(diffs), positive_architectures=sum(x > 0 for x in model_means)))
    write_csv(output / "tables/paired_comparisons.csv", comparisons)
    agreement = []
    for category in config["classes"]:
        inds = np.asarray([i for i, r in enumerate(rows) if r["class_name"] == category])
        for scheme in SCHEMES:
            x, y = arrays["local"][scheme]["score_midpoint"][inds], arrays["online"][scheme]["score_midpoint"][inds]
            corr = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
            agreement.append(dict(class_name=category, scheme=scheme, n=len(inds),
                                  source_spearman=corr, source_mean_absolute_score_difference=float(np.abs(x - y).mean())))
    write_csv(output / "tables/source_score_agreement.csv", agreement)
    # Random tiers are a simple control; the target-level error panel is fixed.
    mean_errors = np.mean([pred != truth for _, _, pred in predictions], axis=0)
    random_rows = []
    rng = np.random.default_rng(config["random_seed"])
    for source in config["sources"]:
        for category in config["classes"]:
            mask = np.logical_and.reduce([arrays[source][s]["fully_known"] for s in SCHEMES])
            inds = np.asarray([i for i, r in enumerate(rows) if r["class_name"] == category and mask[i]])
            if len(inds) < 10:
                continue
            gaps = []
            for _ in range(config["random_strata_replicates"]):
                tiers = np.array_split(rng.permutation(inds), 5)
                gaps.append(float(mean_errors[tiers[-1]].mean() - mean_errors[tiers[0]].mean()))
            random_rows.append(dict(source=source, class_name=category, n=len(inds),
                                    repetitions=len(gaps), mean_error_gap=float(np.mean(gaps)),
                                    gap_q025=float(np.quantile(gaps, .025)), gap_q975=float(np.quantile(gaps, .975))))
    write_csv(output / "tables/random_tier_control.csv", random_rows)
    return class_summary, run_macro, comparisons, cohorts


def export_early(config, output, input_paths):
    path = ROOT / config["early_23_csv"]
    schema_path = ROOT / "configs/dog_feature_schema_v3_1.json"
    input_paths.extend([path, schema_path])
    schema = json.loads(schema_path.read_text())
    original = read_csv(path)
    keys = [f["feature_id"] for f in schema["universal_features"]]
    records = []
    for row in original:
        if row["feature_status"] != "ok":
            raise ValueError("Invalid early feature record")
        # Validate every original feature against its own protocol, not the later schema.
        features = {k: row[k] for k in keys}
        for f in schema["universal_features"]:
            value = features[f["feature_id"]]
            values = json.loads(value) if f["value_type"] == "set" else [value]
            if any(v not in f["possible_values"] for v in values):
                raise ValueError("Early protocol enum mismatch")
        score = score_sample(float(row["target_area_ratio"]), features, "dog", "early_parts", config)
        records.append(dict(sample_id=row["image_id"], target_instance_id=row["target_instance_id"],
                            dataset="early_dog_v3.1", classifier_status=row["classifier_status"],
                            scheme="early_parts", **score, **features))
    if len({(r["sample_id"], r["target_instance_id"]) for r in records}) != len(records):
        raise ValueError("Duplicate early target identity")
    write_csv(output / "tables/early_23_feature_scores.csv", records)
    return dict(targets=len(records), fully_known=sum(r["fully_known"] for r in records),
                evaluable=sum(r["evaluable"] for r in records),
                original_fields_preserved=len(keys),
                classifier_status_counts=dict(Counter(r["classifier_status"] for r in records)),
                score_midpoint_mean=float(np.mean([r["score_midpoint"] for r in records])),
                mean_uncertainty_width=float(np.mean([r["score_upper"] - r["score_lower"] for r in records])))


def export_inventory(rows, all_labels, schemas, config, output):
    records = []
    for source, labels in all_labels.items():
        for category in config["classes"]:
            subset = [labels[r["sample_id"]] for r in rows if r["class_name"] == category]
            schema = schemas["car" if category == "car" else "old"]
            keys = [f["feature_id"] for f in schema["universal_features"] + schema.get("category_specific_features", [])]
            scored = {"body_visibility", "occlusion_level", "truncation_level", "body_orientation"} if category == "car" else {
                "head_visibility", "leg_visibility", "occlusion_level", "truncation_level", "body_orientation", "head_orientation"}
            for key in keys:
                unknown = sum(v[key] == "unknown" or v[key] == ["unknown"] for v in subset)
                records.append(dict(source=source, class_name=category, field=key, records=len(subset),
                                    unknown_count=unknown, equal_parts_role="scored" if key in scored else "preserved_description"))
    write_csv(output / "tables/field_inventory.csv", records)


def export_report(output, summary, macro, comparisons, cohorts, early, audit, config):
    main = [v for v in macro if v["cohort"] == "complete" and v["unknown_mode"] == "midpoint"]
    aggregate = []
    for source in config["sources"]:
        for scheme in SCHEMES:
            rows = [v for v in main if v["source"] == source and v["scheme"] == scheme]
            aggregate.append(dict(source=source, scheme=scheme,
                                  error_auc_mean=mean_or_none([v["error_auc"] for v in rows]),
                                  error_gap_mean=mean_or_none([v["error_gap"] for v in rows])))
    write_csv(output / "tables/overall_summary.csv", aggregate)
    labels = {"legacy_global": "原整体评分", "equal_global": "五维等权整体", "equal_parts": "五维等权局部"}
    lines = ["# 类别细粒度五维风险评分：缓存试运行", "",
             "本次运行使用六分类现有目标标注与冻结预测。分值规则为探索性启发式规则，未根据本轮分类结果拟合或调整；结果不代表独立确认。", "",
             f"输入包含每类400个目标，共{audit['sixclass_targets']}个目标，两个状态来源，{audit['prediction_runs']}份模型／训练种子预测。", "",
             "## 做了什么", "",
             "将每维归一化为0—1，五维各20分。犬、猫、马、羊和人以头部／腿部可见性替代整体可见性，身体／头部朝向各占朝向维度的一半；车辆维持整体字段。没有把可见车轮数直接解释为可见比例，因为正常侧视也只看得到两轮。", "",
             "毛色、场景、动作、遮挡来源及部件等原有字段完整保留，未赋未经验证的风险分。局部方案是一种替代评分方案，并非将整体和局部风险全部累加。", "",
             "主比较在每个来源、每个类别内使用三方案共同完整证据目标；unknown敏感性另报中点和上下界。不同来源的完整证据集合可能不同，主表数值不作来源间的直接优劣判断。", "",
             "## 主比较", "", "| 来源 | 评分方案 | 类别宏平均错误AUROC | Q5−Q1错误率（百分点） |",
             "|---|---|---:|---:|"]
    for v in aggregate:
        auc = f"{v['error_auc_mean']:.4f}" if v["error_auc_mean"] is not None else "不可评价"
        gap = f"{100 * v['error_gap_mean']:.2f}" if v["error_gap_mean"] is not None else "不可评价"
        lines.append(f"| {v['source']} | {labels[v['scheme']]} | {auc} | {gap} |")
    lines += ["", "这次试运行可以执行，但未显示局部方案在两个标注来源下均稳定改善。下表将局部方案与相同五维等权的整体方案配对比较，避免将维度权重变化误当作新增字段的效果。", "",
              "错误AUROC评价风险分对分类错误的区分，越高越好；常数风险时可评价AUROC为0.5，无正确或无错误目标时不可评价。Q5−Q1按类别内分层计算，越大表示端点错误率区分越强，不要求所有相邻层单调。", "",
              "## 局部字段的增量", "", "| 来源 | 指标 | 局部−等权整体 | 描述性95%区间 | 正向架构数 |", "|---|---|---:|---|---:|"]
    for v in comparisons:
        if v["cohort"] == "complete" and v["after"] == "equal_parts":
            factor = 100 if v["metric"] == "error_gap" else 1
            suffix = "个百分点" if factor == 100 else ""
            lines.append(f"| {v['source']} | {v['metric']} | {v['mean_delta'] * factor:.4f}{suffix} | "
                         f"[{v['ci_low'] * factor:.4f}, {v['ci_high'] * factor:.4f}] | {v['positive_architectures']}/{v['architectures']} |")
    lines += ["", "区间按架构和架构内训练种子重采样，描述已有八架构面板；未同时重采样目标，不表示对新数据集的总体推断。", "",
              "## 主比较的样本支持", "", "| 类别 | local完整证据目标 | online完整证据目标 |", "|---|---:|---:|"]
    for category in config["classes"]:
        counts = [next(v["eligible"] for v in cohorts if v["cohort"] == "complete" and v["source"] == s and v["class_name"] == category)
                  for s in config["sources"]]
        lines.append(f"| {category} | {counts[0]} / 400 | {counts[1]} / 400 |")
    lines += ["", "## 早期23项特征", "",
              f"另对{early['targets']}条dog_v3.1有效记录试算，并逐行保留全部{early['original_fields_preserved']}项原始字段。"
              f"完整证据目标为{early['fully_known']}，按每维至少一半权重已知的规则可评价目标为{early['evaluable']}。", "",
              "可见性以头部区域、腿部、尾部各占三分之一；头部区域再将头部、口鼻、嘴部可见性各占三分之一。"
              "身体／头部朝向各占一半。眼睛开闭、舌头不可见、毛色与耳形等不自动设为高风险。"
              "尾部not_observable按未知处理，并保留分数区间，不能当作确定的最大风险。", "",
              "早期遮挡与截断的partial在本试算中各映射为0.5，是该版本独立的初始映射，不把它伪装为后来moderate。", "",
              "该596条记录的分类器状态全部为failed，没有可用分类预测。因此这部分只验证评分与字段保留，尚不能评价23项方案是否更能预测分类失败。", "",
              "## 结果如何使用", "",
              "若局部方案仅在部分来源或类别改善，按条件保留结论；若持平或下降，不据此删掉早期外观特征，而是说明该套初始评分未显示稳定增量。完整23项的独立确认、人物头发等补标、车辆部件视角协议尚需后续工作。", "",
              "## 复现与证据", "",
              "运行：`.venv/bin/python scripts/run_category_five_dimension_pilot.py`。规则在config_frozen.json，输入哈希在input_freeze.sha256.json；逐目标分数、分层、模型指标、配对差异、来源一致性及随机分层对照位于tables/。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # A compact Word result that does not depend on a Markdown converter.
    from docx import Document
    from docx.shared import Pt
    document = Document()
    document.styles["Normal"].font.size = Pt(10)
    in_table = None
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if in_table is None:
                in_table = document.add_table(rows=0, cols=len(cells))
                in_table.style = "Table Grid"
            for cell, text in zip(in_table.add_row().cells, cells):
                cell.text = text
        else:
            in_table = None
            if line.startswith("# "):
                document.add_heading(line[2:], 0)
            elif line.startswith("## "):
                document.add_heading(line[3:], 1)
            elif line:
                document.add_paragraph(line)
    document.save(output / "report.docx")
    plot_results(output, summary, aggregate, config)
    return aggregate


def plot_results(output, summary, aggregate, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    names = ["Original global", "Equal global", "Equal parts"]
    colors = ["#64748b", "#0f766e", "#c2410c"]
    for ax, source in zip(axes, config["sources"]):
        x = np.arange(len(config["classes"]))
        for j, scheme in enumerate(SCHEMES):
            values = [next(v["error_auc_mean"] for v in summary if v["source"] == source and
                           v["class_name"] == c and v["scheme"] == scheme and v["cohort"] == "complete" and
                           v["unknown_mode"] == "midpoint") for c in config["classes"]]
            ax.bar(x + (j - 1) * .24, values, width=.23, label=names[j], color=colors[j])
        ax.axhline(.5, color="black", ls=":", lw=.8)
        ax.set_xticks(x, config["classes"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{source}: common complete-evidence targets")
        ax.set_ylabel("Error AUROC (mean of 24 runs)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Exploratory cache pilot: fixed rules, no outcome fitting", fontsize=11)
    fig.tight_layout()
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(directory / f"error_discrimination.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path, output, smoke=False):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    input_paths = [config_path, Path(__file__)]
    pool = ROOT / config["sixclass_pool"]
    manifest_path = pool / "manifests/candidate_pool.csv"
    input_paths.append(manifest_path)
    all_rows = read_csv(manifest_path)
    if len({r["sample_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate candidate IDs")
    if Counter(r["class_name"] for r in all_rows) != Counter(dict.fromkeys(config["classes"], 400)):
        raise ValueError("Unexpected candidate balance")
    for row in all_rows:
        if int(row["label"]) != config["classes"].index(row["class_name"]):
            raise ValueError("Official class index mismatch")
    rows = ([r for c in config["classes"] for r in [x for x in all_rows if x["class_name"] == c][:5]]
            if smoke else all_rows)
    schemas = {}
    annotation_inputs = {}
    for domain in ("old", "car"):
        path = pool / "configs" / f"{domain}_annotation_schema.json"
        input_paths.append(path)
        schemas[domain] = json.loads(path.read_text())
        path = pool / "inputs" / f"{domain}.csv"
        input_paths.append(path)
        native_rows = read_csv(path)
        if len({v["image_id"] for v in native_rows}) != len(native_rows):
            raise ValueError("Duplicate annotation input identity")
        annotation_inputs[domain] = {v["image_id"]: v for v in native_rows}
    from PIL import Image
    bindings = {}
    for row in rows:
        domain = "car" if row["class_name"] == "car" else "old"
        with Image.open(row["image_path"]) as image:
            bindings[row["sample_id"]] = validate_input_binding(
                row, annotation_inputs[domain][row["sample_id"]], image.size)
    all_labels, all_scores, records = {}, {}, []
    for source in config["sources"]:
        labels = {}
        for row in rows:
            domain = "car" if row["class_name"] == "car" else "old"
            path = pool / "annotations" / source / domain / "parsed" / f"{row['sample_id']}.json"
            input_paths.append(path)
            labels[row["sample_id"]] = validate_label(json.loads(path.read_text()), bindings[row["sample_id"]], schemas[domain])
        all_labels[source] = labels
        current, scores = source_scores(rows, labels, config, source)
        records.extend(current)
        all_scores[source] = scores
        print(f"Scored {source}: {len(rows)} targets, three schemes", flush=True)
    write_json(output / "config_frozen.json", config)
    write_csv(output / "tables/sixclass_scores.csv", records)
    export_inventory(rows, all_labels, schemas, config, output)
    if smoke:
        write_json(output / "smoke_audit.json", dict(status="pass", targets=len(rows), records=len(records),
                                                    sources=2, schemes=3))
        print("SMOKE PASS", flush=True)
        return
    predictions = []
    for path in sorted((pool / "benchmark/runs").glob("*/*/candidate_outputs.npz")):
        input_paths.append(path)
        with np.load(path) as cache:
            if cache["sample_ids"].tolist() != [r["sample_id"] for r in rows]:
                raise ValueError(f"Prediction ID/order mismatch: {path}")
            logits = cache["logits"]
            if logits.shape != (len(rows), 6) or not np.isfinite(logits).all():
                raise ValueError("Invalid logits")
            predictions.append((path.parent.parent.name, int(path.parent.name), logits.argmax(1)))
    if len(predictions) != 24 or len({m for m, _, _ in predictions}) != 8:
        raise ValueError("Expected eight architectures, three seeds each")
    if any(sorted(seed for m, seed, _ in predictions if m == model) != [20260916, 20260917, 20260918]
           for model in {m for m, _, _ in predictions}):
        raise ValueError("Training seed panel differs from the frozen study")
    early = export_early(config, output, input_paths)
    write_json(output / "input_freeze.sha256.json", {str(p.resolve()): sha256(p) for p in input_paths})
    summary, macro, comparisons, cohorts = analyze(rows, all_scores, predictions, config, output)
    audit = dict(status="pass", sixclass_targets=len(rows), sources=len(all_labels),
                 score_records=len(records), prediction_runs=len(predictions), early=early,
                 scoring_did_not_read_predictions=True, source_schemas_validated=True,
                 prediction_identity_validated=True, output_directory=str(output.resolve()))
    aggregate = export_report(output, summary, macro, comparisons, cohorts, early, audit, config)
    write_json(output / "audit.json", audit)
    claims = ["# 本次试运行的主张与证据", "",
              "- 支持：固定五维等权评分、原评分与局部替代评分可在缓存上执行。证据：audit.json、tables/sixclass_scores.csv。",
              "- 支持：早期23项字段完整保留并可试算。证据：tables/early_23_feature_scores.csv，逐字段值域核验。",
              "- 尚未支持：596条早期23项评分能区分分类错误。原因：该批classifier_status全为failed，未移接其他目标预测。",
              "- 局部方案是否改善：按来源、类别与指标报告tables/paired_comparisons.csv，不从某一正结果推断普遍有效。",
              "- 未检验：新目标独立确认、人物头发特征的评分、车辆部件完整协议、非任意权重的最优性。", ""]
    (output / "claim_evidence.md").write_text("\n".join(claims), encoding="utf-8")
    log_path = output / "research_log.md"
    if log_path.exists():
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n执行完成：`.venv/bin/python scripts/run_category_five_dimension_pilot.py`。"
                      "先以--smoke和独立smoke目录验证，再运行完整缓存；规则未据结果修改。"
                      "完成数与输入校验见audit.json，逐项指标见tables/。\n")
    print(json.dumps(dict(overall=aggregate, early=early), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output == DEFAULT_OUTPUT:
        parser.error("Use a separate --output directory for smoke")
    run(args.config, args.output, args.smoke)
