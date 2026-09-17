#!/usr/bin/env python3
"""Add a frozen random-100 dog control to the existing COCO five-class experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_resnet18_dog_coverage_comparison import ROOT, read_csv, sha256, write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CLASS_SPECS, NAME_TO_LABEL, SEEDS, train_one


OUTPUT = ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916"
RANDOM_SEED = 20260916
KNOWN_NONLIVING_CANDIDATES = {"000000244215", "000000249619", "000000275268", "000000516265"}
CONDITIONS = ("dog496", "dog_random596", "dog596")
DISPLAY_NAMES = {"dog496": "dog496", "dog_random596": "random596", "dog596": "gap596"}


def feature_values(row, key):
    value = row.get(key, "")
    if not value:
        return []
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return [value]


def coverage_details(rows, definitions):
    counts = Counter(
        (definition["feature_id"], value)
        for row in rows
        for definition in definitions
        for value in feature_values(row, definition["feature_id"])
    )
    distribution = []
    for definition in definitions:
        feature_id = definition["feature_id"]
        for state in definition["possible_values"]:
            if state in ("unknown", "other"):
                continue
            count = counts[feature_id, state]
            distribution.append({
                "feature_id": feature_id,
                "state": state,
                "count": count,
                "normalized_coverage": min(count / 30, 1),
                "band": "missing" if count == 0 else ("sparse" if count < 10 else ("weak" if count < 30 else "covered")),
            })
    return float(np.mean([row["normalized_coverage"] for row in distribution])), distribution


def prepare_random_group():
    baseline_manifest = {row["image_id"]: row for row in read_csv(
        ROOT / "artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv"
    )}
    candidate_manifest = {row["image_id"]: row for row in read_csv(
        ROOT / "artifacts/dog_candidate400_v2_1_inputs_20260912/target_manifest.csv"
    )}
    clean_results = read_csv(ROOT / "artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv")
    clean_ids = {row["image_id"] for row in clean_results}
    baseline_clean_ids = sorted(set(baseline_manifest) & clean_ids)
    if len(baseline_clean_ids) != 496:
        raise ValueError("expected 496 clean baseline dog targets")

    eligible_ids = sorted(set(candidate_manifest) - KNOWN_NONLIVING_CANDIDATES)
    rng = random.Random(RANDOM_SEED)
    random_ids = sorted(rng.sample(eligible_ids, 100))
    gap_ids = {
        row["image_id"] for row in read_csv(
            ROOT / "artifacts/dog_v3_1_supplement_selection_20260913/selected_candidates.csv"
        )
    }
    if len(gap_ids) != 100:
        raise ValueError("expected 100 gap-selected targets")

    candidate_features = {row["image_id"]: row for row in read_csv(
        ROOT / "artifacts/dog_candidate400_features_v3_1_20260913/results.csv"
    )}
    baseline_features = {row["image_id"]: row for row in read_csv(
        ROOT / "artifacts/dog500_features_v3_1_20260913/results.csv"
    )}
    rows = read_csv(OUTPUT / "data/train_dog496.csv")
    selection_rows = []
    for rank, image_id in enumerate(random_ids, 1):
        manifest = candidate_manifest[image_id]
        feature = candidate_features[image_id]
        rows.append({
            "sample_id": f"dog_{image_id}", "image_id": image_id,
            "crop_path": manifest["crop_image_path"], "label": NAME_TO_LABEL["dog"],
            "class_name": "dog", "yolo_class_id": 16, "source": "random100",
            "bbox_area_ratio": feature["target_area_ratio"],
        })
        selection_rows.append({
            "selection_rank": rank, "image_id": image_id, "random_seed": RANDOM_SEED,
            "candidate_pool_size": len(eligible_ids), "overlaps_gap_selection": int(image_id in gap_ids),
            "dog_recognizability": feature["dog_recognizability"],
            "crop_path": manifest["crop_image_path"],
        })
    if len(rows) != 596 or len({row["image_id"] for row in rows}) != 596:
        raise ValueError("random dog group is not 596 unique targets")
    fields = ["sample_id", "image_id", "crop_path", "label", "class_name", "yolo_class_id",
              "source", "bbox_area_ratio"]
    write_csv(OUTPUT / "data/train_dog_random596.csv", rows, fields)
    write_csv(OUTPUT / "data/random100_selection.csv", selection_rows)

    schema = json.loads((ROOT / "configs/dog_feature_schema_v3_1.json").read_text())["universal_features"]
    baseline_feature_rows = [baseline_features[image_id] for image_id in baseline_clean_ids]
    random_feature_rows = baseline_feature_rows + [candidate_features[image_id] for image_id in random_ids]
    gap_feature_rows = clean_results
    coverage_rows = []
    for condition, feature_rows in (("dog496", baseline_feature_rows),
                                    ("dog_random596", random_feature_rows),
                                    ("dog596", gap_feature_rows)):
        coverage, distribution = coverage_details(feature_rows, schema)
        coverage_rows.append({
            "condition": condition, "display_name": DISPLAY_NAMES[condition], "images": len(feature_rows),
            "coverage_index": coverage,
            "missing_states": sum(row["band"] == "missing" for row in distribution),
            "sparse_states": sum(row["band"] == "sparse" for row in distribution),
            "weak_states": sum(row["band"] == "weak" for row in distribution),
        })
        write_csv(OUTPUT / f"comparison_with_random/coverage_distribution_{condition}.csv", distribution)
    write_csv(OUTPUT / "comparison_with_random/coverage_by_condition.csv", coverage_rows)
    return rows, coverage_rows, len(set(random_ids) & gap_ids)


def load_existing_metrics(condition):
    return [json.loads((OUTPUT / condition / str(seed) / "metrics.json").read_text()) for seed in SEEDS]


def summarize(all_metrics, coverage_rows, overlap):
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    for name, _ in CLASS_SPECS:
        metric_names += [f"{name}_precision", f"{name}_recall", f"{name}_f1"]
    metric_names += ["small_dog_recall", "tiny_dog_recall"]
    summary = []
    for condition in CONDITIONS:
        row = {"condition": condition, "display_name": DISPLAY_NAMES[condition],
               "runs": len(all_metrics[condition])}
        for metric in metric_names:
            values = [run[metric] for run in all_metrics[condition] if run[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)

    comparisons = (
        ("random_vs_base", "dog496", "dog_random596"),
        ("gap_vs_base", "dog496", "dog596"),
        ("gap_vs_random", "dog_random596", "dog596"),
    )
    deltas = []
    for comparison, reference, treatment in comparisons:
        for metric in metric_names:
            paired = [all_metrics[treatment][i][metric] - all_metrics[reference][i][metric]
                      for i in range(len(SEEDS))]
            deltas.append({
                "comparison": comparison, "reference": reference, "treatment": treatment,
                "metric": metric, "delta_mean": float(np.mean(paired)),
                "delta_std": float(np.std(paired, ddof=1)), "seed_deltas": json.dumps(paired),
            })
    write_csv(OUTPUT / "comparison_with_random/summary_by_condition.csv", summary)
    write_csv(OUTPUT / "comparison_with_random/deltas.csv", deltas)
    report = {
        "experiment": "ResNet-18 five-class random-100 versus gap-100 dog supplement control",
        "random_seed": RANDOM_SEED, "random_gap_overlap": overlap,
        "coverage": coverage_rows, "conditions": summary, "deltas": deltas,
        "limitations": [
            "One frozen random-100 subset; random subset selection variance is not measured.",
            "Three model-training seeds and 50 test targets per class.",
            "Target-region classification with oracle COCO boxes, not object detection.",
        ],
    }
    write_json(OUTPUT / "comparison_with_random/report.json", report)

    by_condition = {row["condition"]: row for row in summary}
    coverage = {row["condition"]: row["coverage_index"] for row in coverage_rows}
    display_metrics = ("accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1",
                       "small_dog_recall", "tiny_dog_recall")
    lines = [
        "# ResNet-18五分类：随机补图与缺口补图对照", "", "## 实验设置", "",
        "固定cat、horse、sheep、person训练数据以及验证/测试集，仅改变dog训练子集。", "",
        f"随机100从396张合格候选中按种子{RANDOM_SEED}无放回抽取；与缺口100重叠{overlap}张。", "",
        "## 覆盖率", "", "| 条件 | dog数量 | 覆盖率 |", "|---|---:|---:|",
    ]
    for condition in CONDITIONS:
        count = 496 if condition == "dog496" else 596
        lines.append(f"| {DISPLAY_NAMES[condition]} | {count} | {coverage[condition]:.2%} |")
    lines += ["", "## 模型结果", "",
              "| 指标 | dog496 | random596 | gap596 | gap-random |", "|---|---:|---:|---:|---:|"]
    for metric in display_metrics:
        values = [by_condition[c][metric + "_mean"] for c in CONDITIONS]
        stds = [by_condition[c][metric + "_std"] for c in CONDITIONS]
        lines.append(f"| {metric} | {values[0]:.4f} ± {stds[0]:.4f} | "
                     f"{values[1]:.4f} ± {stds[1]:.4f} | {values[2]:.4f} ± {stds[2]:.4f} | "
                     f"{values[2] - values[1]:+.4f} |")
    lines += ["", "## 结论边界", "",
              "- 只有一个固定随机100子集，尚未覆盖随机选样本身的方差。",
              "- 每组只有3个训练种子、每类测试目标50个；应同时报告均值、标准差和逐种子方向。",
              "- 结果评价COCO目标框裁剪五分类，不是目标检测。", ""]
    (OUTPUT / "comparison_with_random/report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def write_checksums():
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"checksums.csv", "checksums_with_random.csv"}:
            rows.append({"path": str(path.relative_to(OUTPUT)), "sha256": sha256(path),
                         "bytes": path.stat().st_size})
    write_csv(OUTPUT / "checksums_with_random.csv", rows)


def main():
    random_rows, coverage_rows, overlap = prepare_random_group()
    other_rows = read_csv(OUTPUT / "data/train_other_fixed.csv")
    eval_rows = read_csv(OUTPUT / "data/eval_fixed.csv")
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    all_metrics = defaultdict(list)
    all_metrics["dog496"] = load_existing_metrics("dog496")
    all_metrics["dog596"] = load_existing_metrics("dog596")
    for seed in SEEDS:
        metrics, _ = train_one("dog_random596", seed, random_rows + other_rows, val_rows, test_rows,
                               OUTPUT, epochs=20, workers=24)
        all_metrics["dog_random596"].append(metrics)
    report = summarize(all_metrics, coverage_rows, overlap)
    write_checksums()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
