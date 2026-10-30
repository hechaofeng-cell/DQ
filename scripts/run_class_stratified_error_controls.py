#!/usr/bin/env python3
"""Class-stratified equal-error controls for Coverage diagnostics.

The original rarefaction matched the pooled number of errors. This analysis
matches errors independently inside each true class before macro averaging, so
differences cannot be produced only by a different class composition of errors.
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_coverage_incremental_value import (
    EQUAL_ERROR_BASELINES,
    HYBRID_BASELINES,
    expected_unique_fraction,
    hybrid_orders,
    load_orders,
    normalized_auc,
    paired_stats,
)
from run_cross_state_decoupled_diagnostics import error_structures, load_cached_run
from run_multiclass_object_selection_benchmark import (
    BUDGETS,
    CLASSES,
    LABELS,
    MODELS,
    RANDOM_REPLICATES,
    SEEDS,
    build_context,
    random_orders,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/class_stratified_error_controls_20260929"


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.zeros(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def class_error_labels(context, class_name, order, budget, predicted, structures):
    globals_ = context["by_class"][class_name]["global_indices"]
    chosen = globals_[np.asarray(order[class_name][:budget])]
    confusion, clusters = [], []
    for index in chosen:
        index = int(index)
        if int(predicted[index]) == LABELS[class_name]:
            continue
        confusion.append(int(predicted[index]))
        clusters.append(int(structures[class_name]["cluster_by_global"][index]))
    return confusion, clusters


def class_stratified_pair(context, left_order, right_order, budget, predicted, structures):
    class_rows = []
    for class_name in CLASSES:
        left_conf, left_clusters = class_error_labels(
            context, class_name, left_order, budget, predicted, structures
        )
        right_conf, right_clusters = class_error_labels(
            context, class_name, right_order, budget, predicted, structures
        )
        matched = min(len(left_conf), len(right_conf))
        confusion_universe = len(structures[class_name]["possible_predictions"])
        cluster_universe = int(structures[class_name]["cluster_count"])
        class_rows.append({
            "class_name": class_name,
            "left_errors": len(left_conf),
            "right_errors": len(right_conf),
            "matched_errors": matched,
            "has_evidence": int(matched > 0),
            "left_confusion": expected_unique_fraction(
                [(class_name, value) for value in left_conf], matched, confusion_universe
            ),
            "right_confusion": expected_unique_fraction(
                [(class_name, value) for value in right_conf], matched, confusion_universe
            ),
            "left_clusters": expected_unique_fraction(
                [(class_name, value) for value in left_clusters], matched, cluster_universe
            ),
            "right_clusters": expected_unique_fraction(
                [(class_name, value) for value in right_clusters], matched, cluster_universe
            ),
        })
    output = {
        "matched_errors": float(sum(row["matched_errors"] for row in class_rows)),
        "evaluable_class_fraction": float(np.mean([row["has_evidence"] for row in class_rows])),
    }
    for key in ("left_confusion", "right_confusion", "left_clusters", "right_clusters"):
        output[key] = float(np.mean([row[key] for row in class_rows]))
    return output, class_rows


def aggregate_random_pairs(pairs):
    keys = pairs[0][0]
    aggregate = {key: float(np.mean([pair[0][key] for pair in pairs])) for key in keys}
    by_class = []
    for class_name in CLASSES:
        rows = [next(row for row in pair[1] if row["class_name"] == class_name) for pair in pairs]
        item = {"class_name": class_name}
        for key in rows[0]:
            if key != "class_name":
                item[key] = float(np.mean([row[key] for row in rows]))
        by_class.append(item)
    return aggregate, by_class


def add_delta(row):
    row["delta_confusion"] = row["left_confusion"] - row["right_confusion"]
    row["delta_clusters"] = row["left_clusters"] - row["right_clusters"]
    return row


def run(output: Path, random_replicates: int) -> None:
    context = build_context()
    random_sets = random_orders(context["by_class"], random_replicates)
    rows, class_rows, auc_rows = [], [], []
    values = defaultdict(lambda: defaultdict(list))
    hybrid_rows, hybrid_class_rows, hybrid_auc_rows = [], [], []
    hybrid_values = defaultdict(lambda: defaultdict(list))

    for model, seed in itertools.product(MODELS, SEEDS):
        features, predicted = load_cached_run(context, model, seed)
        structures = error_structures(context, features, predicted, model, seed)
        orders = load_orders(context, model, seed)
        curves = defaultdict(list)
        hybrid_curves = defaultdict(list)

        for budget in BUDGETS:
            for baseline in EQUAL_ERROR_BASELINES:
                if baseline == "random":
                    pairs = [
                        class_stratified_pair(
                            context, orders["coverage"], random_order, budget, predicted, structures
                        )
                        for random_order in random_sets
                    ]
                    metric, per_class = aggregate_random_pairs(pairs)
                else:
                    metric, per_class = class_stratified_pair(
                        context, orders["coverage"], orders[baseline], budget, predicted, structures
                    )
                record = add_delta({
                    "comparison": f"coverage_vs_{baseline}", "model": model,
                    "seed": seed, "budget": budget, **metric,
                })
                rows.append(record)
                curves[baseline].append(record)
                for item in per_class:
                    class_rows.append({
                        "comparison": f"coverage_vs_{baseline}", "model": model,
                        "seed": seed, "budget": budget, **item,
                    })

            for base in HYBRID_BASELINES:
                coverage25 = hybrid_orders(orders[base], orders["coverage"], budget)
                random_pairs = [
                    class_stratified_pair(
                        context,
                        coverage25,
                        hybrid_orders(orders[base], random_order, budget),
                        budget,
                        predicted,
                        structures,
                    )
                    for random_order in random_sets
                ]
                metric, per_class = aggregate_random_pairs(random_pairs)
                comparison = f"{base}_coverage25_vs_random25"
                record = add_delta({
                    "comparison": comparison, "model": model, "seed": seed,
                    "budget": budget, **metric,
                })
                hybrid_rows.append(record)
                hybrid_curves[base, "random25"].append(record)
                for item in per_class:
                    hybrid_class_rows.append({
                        "comparison": comparison, "model": model, "seed": seed,
                        "budget": budget, **item,
                    })

                pure_metric, pure_class = class_stratified_pair(
                    context, coverage25, orders[base], budget, predicted, structures
                )
                comparison = f"{base}_coverage25_vs_pure"
                record = add_delta({
                    "comparison": comparison, "model": model, "seed": seed,
                    "budget": budget, **pure_metric,
                })
                hybrid_rows.append(record)
                hybrid_curves[base, "pure"].append(record)
                for item in pure_class:
                    hybrid_class_rows.append({
                        "comparison": comparison, "model": model, "seed": seed,
                        "budget": budget, **item,
                    })

        for baseline in EQUAL_ERROR_BASELINES:
            item = {"comparison": f"coverage_vs_{baseline}", "model": model, "seed": seed}
            for metric in ("confusion", "clusters"):
                auc_value = normalized_auc(curves[baseline], f"delta_{metric}")
                item[f"auc_delta_{metric}"] = auc_value
                values[baseline, metric][model].append(auc_value)
            auc_rows.append(item)
        for base in HYBRID_BASELINES:
            for comparator in ("random25", "pure"):
                comparison = f"{base}_coverage25_vs_{comparator}"
                item = {"comparison": comparison, "model": model, "seed": seed}
                for metric in ("confusion", "clusters"):
                    auc_value = normalized_auc(hybrid_curves[base, comparator], f"delta_{metric}")
                    item[f"auc_delta_{metric}"] = auc_value
                    hybrid_values[base, comparator, metric][model].append(auc_value)
                hybrid_auc_rows.append(item)
        print(f"[stratified] {model} seed={seed} complete", flush=True)

    comparisons = [
        paired_stats(values[baseline, metric], f"coverage_vs_{baseline}", f"class_equal_error_{metric}")
        for baseline in EQUAL_ERROR_BASELINES for metric in ("confusion", "clusters")
    ]
    primary = [row for row in comparisons if row["comparison"] == "coverage_vs_random"]
    adjusted = holm([row["permutation_p"] for row in primary])
    for row, value in zip(primary, adjusted):
        row["holm_primary_p"] = value
    global_adjusted = holm([row["permutation_p"] for row in comparisons])
    for row, value in zip(comparisons, global_adjusted):
        row["holm_global_p"] = value

    hybrid_comparisons = [
        paired_stats(
            hybrid_values[base, comparator, metric],
            f"{base}_coverage25_vs_{comparator}",
            f"class_equal_error_{metric}",
        )
        for base in HYBRID_BASELINES for comparator in ("random25", "pure")
        for metric in ("confusion", "clusters")
    ]
    hybrid_primary = [row for row in hybrid_comparisons if row["comparison"].endswith("random25")]
    adjusted = holm([row["permutation_p"] for row in hybrid_primary])
    for row, value in zip(hybrid_primary, adjusted):
        row["holm_primary_p"] = value
    global_adjusted = holm([row["permutation_p"] for row in hybrid_comparisons])
    for row, value in zip(hybrid_comparisons, global_adjusted):
        row["holm_global_p"] = value

    write_csv(output / "tables/class_equal_error_metrics.csv", rows)
    write_csv(output / "tables/class_equal_error_by_class.csv", class_rows)
    write_csv(output / "tables/class_equal_error_auc.csv", auc_rows)
    write_csv(output / "tables/class_equal_error_comparisons.csv", comparisons)
    write_csv(output / "tables/hybrid_class_equal_error_metrics.csv", hybrid_rows)
    write_csv(output / "tables/hybrid_class_equal_error_by_class.csv", hybrid_class_rows)
    write_csv(output / "tables/hybrid_class_equal_error_auc.csv", hybrid_auc_rows)
    write_csv(output / "tables/hybrid_class_equal_error_comparisons.csv", hybrid_comparisons)

    report = [
        "# 逐类别等错误数控制", "",
        "错误数在每个真实类别内分别匹配，零匹配错误类别按双方0计入宏平均。", "",
        "## Coverage vs Random（预注册主指标族D1）", "",
        "| 指标 | AUC差值 | 95% CI | 正向运行 | Holm p | 全局Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        report.append(
            f"| {row['metric']} | {row['mean_delta_auc']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['positive_runs']}/24 | {row['holm_primary_p']:.4f} | {row['holm_global_p']:.4f} |"
        )
    report += ["", "## Coverage25 vs Random25（预注册主指标族E1）", "",
        "| 强基线 | 指标 | AUC差值 | 95% CI | 正向运行 | Holm p | 全局Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in hybrid_primary:
        base = row["comparison"].split("_coverage25")[0]
        report.append(
            f"| {base} | {row['metric']} | {row['mean_delta_auc']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['positive_runs']}/24 | {row['holm_primary_p']:.4f} | {row['holm_global_p']:.4f} |"
        )
    report += ["", "## 解释边界", "",
        "- 该控制排除了总体错误数及其类别构成差异，但仍共享同一候选池和模型运行。",
        "- 零错误类别保留在宏平均中，避免只报告有利类别。",
        "- 相对纯强基线的结果属于次要比较，不用于改写预注册主张。", "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "report_zh.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    args = parser.parse_args()
    run(args.output.resolve(), args.random_replicates)


if __name__ == "__main__":
    main()
