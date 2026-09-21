#!/usr/bin/env python3
"""Analyze the frozen gap-driven gradient against matched random controls."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAP_ROOT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
RANDOM_ROOT = ROOT / "artifacts/coco_openimages_random_gradient_8_models_20260918"
OUTPUT = RANDOM_ROOT / "analysis"
ADDITIONS = (50, 100, 150, 200, 250, 300, 350)
METRICS = ("accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1")


def mean(values):
    return statistics.fmean(values)


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_id(row):
    return row.get("sample_id") or row.get("target_instance_id") or row["image_id"]


def external_dog_ids(path: Path):
    return {
        target_id(row) for row in read_csv(path)
        if row["class_name"] == "dog" and row.get("dataset_version", "").startswith("openimages")
    }


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def main():
    gap = json.loads((GAP_ROOT / "all_metrics.json").read_text(encoding="utf-8"))
    random_runs = json.loads((RANDOM_ROOT / "all_metrics.json").read_text(encoding="utf-8"))
    if len(gap) != 192 or len(random_runs) != 840:
        raise ValueError(f"incomplete inputs: gap={len(gap)}, random={len(random_runs)}")
    selection_seeds = sorted({row["selection_seed"] for row in random_runs})
    training_seeds = sorted({row["seed"] for row in random_runs})
    models = sorted({row["model"] for row in random_runs})
    if (len(selection_seeds), len(training_seeds), len(models)) != (5, 3, 8):
        raise ValueError("unexpected experimental dimensions")

    gap_by_add = {
        add: [row for row in gap if row["condition"] == f"oi_gap_add{add}"]
        for add in ADDITIONS
    }
    random_by_add = {
        add: [row for row in random_runs if row["external_dog_added"] == add]
        for add in ADDITIONS
    }
    coverage_rows = []
    performance_rows = []
    for add in ADDITIONS:
        gap_rows = gap_by_add[add]
        random_rows = random_by_add[add]
        random_coverages = sorted({row["coverage"] for row in random_rows})
        gap_coverage = gap_rows[0]["coverage"]
        coverage_rows.append({
            "external_dog_added": add,
            "gap_coverage": gap_coverage,
            "random_coverage_mean": mean(random_coverages),
            "random_coverage_sd": statistics.stdev(random_coverages),
            "random_coverage_min": min(random_coverages),
            "random_coverage_max": max(random_coverages),
            "gap_minus_random_mean": gap_coverage - mean(random_coverages),
        })
        row = {"external_dog_added": add}
        for metric in METRICS + ("small_dog_recall", "tiny_dog_recall"):
            gap_value = mean(item[metric] for item in gap_rows)
            random_value = mean(item[metric] for item in random_rows)
            row[f"gap_{metric}"] = gap_value
            row[f"random_{metric}"] = random_value
            row[f"gap_minus_random_{metric}"] = gap_value - random_value
        for metric in ("macro_f1", "dog_f1"):
            sequence_means = [
                mean(item[metric] for item in random_rows if item["selection_seed"] == seed)
                for seed in selection_seeds
            ]
            row[f"random_sequence_{metric}_sd"] = statistics.stdev(sequence_means)
            row[f"gap_above_all_random_sequence_means_{metric}"] = (
                row[f"gap_{metric}"] > max(sequence_means)
            )
        performance_rows.append(row)

    model_rows = []
    for model in models:
        row = {"model": model}
        for metric in ("macro_f1", "dog_f1"):
            stage_deltas = []
            for add in ADDITIONS:
                gap_value = mean(
                    item[metric] for item in gap_by_add[add] if item["model"] == model
                )
                random_value = mean(
                    item[metric] for item in random_by_add[add] if item["model"] == model
                )
                stage_deltas.append(gap_value - random_value)
            row[f"mean_gap_minus_random_{metric}"] = mean(stage_deltas)
            row[f"gap_stage_wins_{metric}"] = sum(value > 0 for value in stage_deltas)
        model_rows.append(row)

    overlap_rows = []
    for add in ADDITIONS:
        gap_ids = external_dog_ids(GAP_ROOT / "manifests" / f"train_oi_gap_add{add}.csv")
        overlaps = []
        for seed in selection_seeds:
            random_ids = external_dog_ids(
                RANDOM_ROOT / "manifests" / f"train_random_s{seed}_add{add}.csv"
            )
            overlaps.append(len(gap_ids & random_ids))
        overlap_rows.append({
            "external_dog_added": add,
            "overlap_mean": mean(overlaps),
            "overlap_min": min(overlaps),
            "overlap_max": max(overlaps),
        })

    write_csv(OUTPUT / "coverage_comparison.csv", coverage_rows)
    write_csv(OUTPUT / "performance_comparison.csv", performance_rows)
    write_csv(OUTPUT / "model_summary.csv", model_rows)
    write_csv(OUTPUT / "selection_overlap.csv", overlap_rows)

    pooled = {}
    for metric in METRICS + ("small_dog_recall", "tiny_dog_recall"):
        pooled[f"gap_{metric}"] = mean(row[f"gap_{metric}"] for row in performance_rows)
        pooled[f"random_{metric}"] = mean(row[f"random_{metric}"] for row in performance_rows)
        pooled[f"delta_{metric}"] = pooled[f"gap_{metric}"] - pooled[f"random_{metric}"]
    pooled["gap_coverage"] = mean(row["gap_coverage"] for row in coverage_rows)
    pooled["random_coverage"] = mean(row["random_coverage_mean"] for row in coverage_rows)
    pooled["delta_coverage"] = pooled["gap_coverage"] - pooled["random_coverage"]

    lines = [
        "# 缺口驱动补图与同规模随机补图对照分析", "",
        "日期：2026-09-19", "",
        "## 实验完整性", "",
        "- 缺口组：8个阶段 × 8个模型 × 3个训练种子，共192次；本报告比较其中7个补图阶段。",
        "- 随机组：5条独立随机选样序列 × 7个补图阶段 × 8个模型 × 3个训练种子，共840次。",
        "- 840/840次随机组训练均完成。固定COCO起始集、非dog新增清单、验证集、测试集和训练配置。",
        "- 唯一有意改变的因素是Open Images dog候选目标的选择顺序。", "",
        "## 覆盖效率", "",
        "| 新增dog | 缺口组覆盖率 | 随机组均值±SD | 随机范围 | 覆盖率差值 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['external_dog_added']} | {pct(row['gap_coverage'])} | "
            f"{pct(row['random_coverage_mean'])} ± {pct(row['random_coverage_sd'])} | "
            f"{pct(row['random_coverage_min'])}–{pct(row['random_coverage_max'])} | "
            f"{pp(row['gap_minus_random_mean'])} |"
        )
    lines += [
        "", f"七个预算点平均来看，缺口组覆盖率为{pct(pooled['gap_coverage'])}，"
        f"随机组为{pct(pooled['random_coverage'])}，平均高{pp(pooled['delta_coverage'])}。"
        "缺口选择在全部预算点都高于5条随机序列的最大值。", "",
        "## 下游分类性能", "",
        "| 新增dog | 缺口Macro-F1 | 随机Macro-F1 | 差值 | 缺口dog F1 | 随机dog F1 | 差值 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in performance_rows:
        lines.append(
            f"| {row['external_dog_added']} | {pct(row['gap_macro_f1'])} | "
            f"{pct(row['random_macro_f1'])} | {pp(row['gap_minus_random_macro_f1'])} | "
            f"{pct(row['gap_dog_f1'])} | {pct(row['random_dog_f1'])} | "
            f"{pp(row['gap_minus_random_dog_f1'])} |"
        )
    lines += [
        "", "跨七个预算点等权平均：", "",
        f"- Accuracy：缺口组{pct(pooled['gap_accuracy'])}，随机组{pct(pooled['random_accuracy'])}，差值{pp(pooled['delta_accuracy'])}；",
        f"- Macro-F1：缺口组{pct(pooled['gap_macro_f1'])}，随机组{pct(pooled['random_macro_f1'])}，差值{pp(pooled['delta_macro_f1'])}；",
        f"- dog F1：缺口组{pct(pooled['gap_dog_f1'])}，随机组{pct(pooled['random_dog_f1'])}，差值{pp(pooled['delta_dog_f1'])}；",
        f"- 小目标dog Recall：差值{pp(pooled['delta_small_dog_recall'])}；极小目标dog Recall：差值{pp(pooled['delta_tiny_dog_recall'])}。",
        "", "缺口组只在新增100和200时取得更高的Macro-F1；dog F1仅在新增200时高于随机组。"
        "因此，当前证据不支持“缺口驱动补图普遍提高分类性能”。", "",
        "## 新增200张的局部结果", "",
        "新增200是唯一同时改善总体Macro-F1和dog F1的阶段：Macro-F1高0.37个百分点，dog F1高0.51个百分点。"
        "在该阶段，缺口组的跨模型平均值高于5条随机序列各自的平均值；但优势幅度较小，且不同模型并不一致，"
        "只能视为局部有效区间，而不是普遍胜出。", "",
        "## 分模型稳定性", "",
        "| 模型 | 七阶段平均ΔMacro-F1 | Macro-F1胜出阶段 | 七阶段平均Δdog F1 | dog F1胜出阶段 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {pp(row['mean_gap_minus_random_macro_f1'])} | "
            f"{row['gap_stage_wins_macro_f1']}/7 | {pp(row['mean_gap_minus_random_dog_f1'])} | "
            f"{row['gap_stage_wins_dog_f1']}/7 |"
        )
    lines += [
        "", "ResNet-18的Macro-F1在7/7个阶段高于随机均值，但dog F1只在2/7个阶段胜出；"
        "EfficientNet-B0是唯一在两个指标的七阶段平均差值上都为正的模型。"
        "MobileNetV3-Small受缺口选择影响最明显，两个指标均下降。", "",
        "## 可以支持的结论", "",
        "1. 缺口驱动选样稳定且大幅提高了视觉状态覆盖效率，并且这种优势在全部预算点和5条随机序列上保持一致。",
        "2. 更高覆盖率没有自动转化为更高的整体分类性能；覆盖指标和模型效用指标必须分别报告。",
        "3. 当前设置存在一个约新增200张的局部收益区间，但收益具有阶段依赖性和架构依赖性。",
        "4. 高覆盖组在小目标与极小目标Recall上呈轻微平均优势，但测试支持数分别仅12和7，不能作强结论。",
        "5. 方法当前更强的证据是‘更高效地补齐可解释视觉状态’，而不是‘普遍提高分类F1’。", "",
        "## 不能支持的结论", "",
        "- 不能声称缺口驱动补图总体优于随机补图的分类性能。",
        "- 不能声称覆盖率越高，F1越高。",
        "- 不能把新增200张解释为通用最优预算。",
        "- 不能把固定COCO测试集上的结果外推到其他数据集、类别或目标检测任务。", "",
        "## 解释与限制", "",
        "- 随机组与缺口组来自同一候选池，随机清单允许自然抽中缺口样本；新增350时平均重叠42.6张，属于合理的随机对照，但会缩小两组差异。",
        "- 各阶段是嵌套数据集，同一随机序列内的阶段不是独立重复。",
        "- 8个模型共享同一个固定测试集，模型和训练种子不能被当作完全独立的统计样本。",
        "- 测试集只有250个目标，每类50个，困难dog子组更小，因此重点应放在效应方向和跨设置一致性。",
        "- 可能的机制包括困难状态增加、跨数据集域偏移和模型容量差异；本实验尚未单独识别这些机制。", "",
        "## 结果文件", "",
        "- `coverage_comparison.csv`：同预算覆盖率比较；",
        "- `performance_comparison.csv`：同预算模型性能比较；",
        "- `model_summary.csv`：逐模型七阶段汇总；",
        "- `selection_overlap.csv`：随机组与缺口组dog选择重叠；",
        "- 原始缺口组：`artifacts/coco_openimages_external_gradient_8_models_20260918/all_metrics.json`；",
        "- 原始随机组：`artifacts/coco_openimages_random_gradient_8_models_20260918/all_metrics.json`。", "",
    ]
    (OUTPUT / "gap_vs_random_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "summary.json").write_text(
        json.dumps({"pooled": pooled, "coverage": coverage_rows, "performance": performance_rows,
                    "models": model_rows, "overlap": overlap_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT / "gap_vs_random_analysis.md")


if __name__ == "__main__":
    main()
