#!/usr/bin/env python3
"""Compare local and online VLM state labels and test conclusion sensitivity."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from run_class_stratified_error_controls import (
    add_delta,
    aggregate_random_pairs,
    class_stratified_pair,
    holm,
)
from run_coverage_incremental_value import independent_metrics, normalized_auc, paired_stats
from run_cross_state_decoupled_diagnostics import error_structures, load_cached_run
from run_multiclass_object_selection_benchmark import (
    BUDGETS,
    CLASSES,
    MODELS,
    RANDOM_REPLICATES,
    SEEDS,
    build_context,
    concave_coverage_order,
    random_orders,
    state_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts/multiclass_object_pool_v1_20260928"
BASE = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928"
DECOUPLED = ROOT / "artifacts/cross_state_decoupled_diagnostics_20260928"
CLASS_CONTROL = ROOT / "artifacts/class_stratified_error_controls_20260929"
SCHEMA = ROOT / "configs/multiclass_object_state_schema_v1_0.json"
DEFAULT_ONLINE = ROOT / "artifacts/online_vlm_reannotation_20260929/results.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/online_vlm_reannotation_20260929/analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def values(value: str) -> list[str]:
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        return [str(item) for item in parsed]
    return [text]


def auc(rows: list[dict], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["budget"]))
    x = np.asarray([0, *[int(row["budget"]) for row in ordered]], dtype=float)
    y = np.asarray([0, *[float(row[metric]) for row in ordered]], dtype=float)
    return float(np.trapezoid(y, x) / BUDGETS[-1])


def agreement(local: dict, online: dict, schema: dict) -> tuple[list[dict], dict]:
    rows = []
    all_exact = []
    for feature in schema["universal_features"]:
        name = feature["feature_id"]
        left = [local[key][name] for key in online]
        right = [online[key][name] for key in online]
        if feature["value_type"] == "set":
            exact = [set(values(a)) == set(values(b)) for a, b in zip(left, right)]
            jaccard = []
            for a, b in zip(left, right):
                first, second = set(values(a)), set(values(b))
                jaccard.append(len(first & second) / len(first | second))
            row = {
                "feature": name, "type": "set", "n": len(left),
                "exact_agreement": float(np.mean(exact)),
                "cohen_kappa": "", "mean_jaccard": float(np.mean(jaccard)),
                "local_unknown_rate": float(np.mean(["unknown" in values(item) for item in left])),
                "online_unknown_rate": float(np.mean(["unknown" in values(item) for item in right])),
            }
        else:
            exact = [a == b for a, b in zip(left, right)]
            row = {
                "feature": name, "type": "enum", "n": len(left),
                "exact_agreement": float(np.mean(exact)),
                "cohen_kappa": float(cohen_kappa_score(left, right)),
                "mean_jaccard": "",
                "local_unknown_rate": float(np.mean([item == "unknown" for item in left])),
                "online_unknown_rate": float(np.mean([item == "unknown" for item in right])),
            }
        rows.append(row)
        all_exact.extend(exact)
    summary = {
        "samples": len(online),
        "field_instances": len(all_exact),
        "micro_exact_agreement": float(np.mean(all_exact)),
        "enum_macro_exact_agreement": float(np.mean([r["exact_agreement"] for r in rows if r["type"] == "enum"])),
        "enum_macro_kappa": float(np.nanmean([r["cohen_kappa"] for r in rows if r["type"] == "enum"])),
        "set_macro_exact_agreement": float(np.mean([r["exact_agreement"] for r in rows if r["type"] == "set"])),
        "set_macro_jaccard": float(np.mean([r["mean_jaccard"] for r in rows if r["type"] == "set"])),
    }
    return rows, summary


def merged_context(base: dict, merged_features: dict, schema: dict) -> dict:
    rows = base["rows"]
    global_index = {row["sample_id"]: index for index, row in enumerate(rows)}
    by_class = {}
    for class_name in CLASSES:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        matrix, states = state_matrix(class_rows, merged_features, schema)
        by_class[class_name] = {
            "rows": class_rows, "matrix": matrix, "states": states,
            "global_indices": np.asarray([global_index[row["sample_id"]] for row in class_rows]),
        }
    return {**base, "features": merged_features, "by_class": by_class}


def coverage_orders(context: dict) -> dict[str, list[int]]:
    return {
        name: concave_coverage_order(context["by_class"][name]["rows"], context["by_class"][name]["matrix"])
        for name in CLASSES
    }


def order_stability(base: dict, changed: dict, local_orders: list[dict]) -> list[dict]:
    local_lookup = defaultdict(list)
    for row in local_orders:
        if row["method"] == "coverage":
            local_lookup[row["class_name"]].append(row)
    for class_name in CLASSES:
        local_lookup[class_name].sort(key=lambda row: int(row["rank"]))
    changed_orders = coverage_orders(changed)
    rows = []
    for class_name in CLASSES:
        changed_ids = [changed["by_class"][class_name]["rows"][index]["sample_id"] for index in changed_orders[class_name]]
        local_ids = [row["sample_id"] for row in local_lookup[class_name]]
        changed_rank = {sample_id: rank for rank, sample_id in enumerate(changed_ids)}
        tau = float(kendalltau(range(len(local_ids)), [changed_rank[item] for item in local_ids]).statistic)
        for budget in BUDGETS:
            rows.append({
                "class_name": class_name, "budget": budget,
                "jaccard": len(set(local_ids[:budget]) & set(changed_ids[:budget])) / len(set(local_ids[:budget]) | set(changed_ids[:budget])),
                "kendall_tau_full_order": tau,
            })
    return rows


def conclusion_sensitivity(context: dict, orders: dict) -> tuple[list[dict], list[dict]]:
    random_auc = {
        (row["model"], int(row["seed"])): row
        for row in read_csv(DECOUPLED / "tables/auc_by_run.csv") if row["method"] == "random"
    }
    local_auc = {
        (row["model"], int(row["seed"])): row
        for row in read_csv(DECOUPLED / "tables/auc_by_run.csv") if row["method"] == "coverage"
    }
    run_rows = []
    comparison_values = {
        "confusion": defaultdict(list), "clusters": defaultdict(list),
        "local_delta_confusion": defaultdict(list), "local_delta_clusters": defaultdict(list),
    }
    for model in MODELS:
        for seed in SEEDS:
            features, predicted = load_cached_run(context, model, seed)
            structures = error_structures(context, features, predicted, model, seed)
            curves = []
            for budget in BUDGETS:
                metric = independent_metrics(context, orders, budget, predicted, structures)
                curves.append({"budget": budget, **metric})
            changed_conf = auc(curves, "confusion_pair_coverage")
            changed_clusters = auc(curves, "error_cluster_coverage")
            random = random_auc[model, seed]
            local = local_auc[model, seed]
            row = {
                "model": model, "seed": seed,
                "online_merged_confusion_auc": changed_conf,
                "local_confusion_auc": float(local["auc_confusion_pair_coverage"]),
                "random_confusion_auc": float(random["auc_confusion_pair_coverage"]),
                "online_merged_cluster_auc": changed_clusters,
                "local_cluster_auc": float(local["auc_error_cluster_coverage"]),
                "random_cluster_auc": float(random["auc_error_cluster_coverage"]),
            }
            row["online_vs_random_confusion"] = row["online_merged_confusion_auc"] - row["random_confusion_auc"]
            row["online_vs_random_clusters"] = row["online_merged_cluster_auc"] - row["random_cluster_auc"]
            row["online_vs_local_confusion"] = row["online_merged_confusion_auc"] - row["local_confusion_auc"]
            row["online_vs_local_clusters"] = row["online_merged_cluster_auc"] - row["local_cluster_auc"]
            run_rows.append(row)
            comparison_values["confusion"][model].append(row["online_vs_random_confusion"])
            comparison_values["clusters"][model].append(row["online_vs_random_clusters"])
            comparison_values["local_delta_confusion"][model].append(row["online_vs_local_confusion"])
            comparison_values["local_delta_clusters"][model].append(row["online_vs_local_clusters"])
    comparisons = [
        paired_stats(comparison_values[name], label, metric)
        for name, label, metric in (
            ("confusion", "online_merged_coverage_vs_random", "confusion_pair_coverage"),
            ("clusters", "online_merged_coverage_vs_random", "error_cluster_coverage"),
            ("local_delta_confusion", "online_merged_vs_local_coverage", "confusion_pair_coverage"),
            ("local_delta_clusters", "online_merged_vs_local_coverage", "error_cluster_coverage"),
        )
    ]
    return run_rows, comparisons


def class_equal_error_sensitivity(
    context: dict, orders: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Re-evaluate Coverage after matching errors inside each true class."""
    random_sets = random_orders(context["by_class"], RANDOM_REPLICATES)
    local_auc = {
        (row["model"], int(row["seed"])): row
        for row in read_csv(CLASS_CONTROL / "tables/class_equal_error_auc.csv")
        if row["comparison"] == "coverage_vs_random"
    }
    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    auc_rows: list[dict] = []
    online_values = defaultdict(lambda: defaultdict(list))
    local_delta_values = defaultdict(lambda: defaultdict(list))

    for model in MODELS:
        for seed in SEEDS:
            features, predicted = load_cached_run(context, model, seed)
            structures = error_structures(context, features, predicted, model, seed)
            curve = []
            for budget in BUDGETS:
                pairs = [
                    class_stratified_pair(
                        context, orders, random_order, budget, predicted, structures
                    )
                    for random_order in random_sets
                ]
                metric, per_class = aggregate_random_pairs(pairs)
                record = add_delta({
                    "comparison": "online_merged_coverage_vs_random",
                    "model": model,
                    "seed": seed,
                    "budget": budget,
                    **metric,
                })
                metric_rows.append(record)
                curve.append(record)
                for item in per_class:
                    class_rows.append({
                        "comparison": "online_merged_coverage_vs_random",
                        "model": model,
                        "seed": seed,
                        "budget": budget,
                        **item,
                    })

            item = {
                "comparison": "online_merged_coverage_vs_random",
                "model": model,
                "seed": seed,
            }
            local = local_auc[model, seed]
            for metric in ("confusion", "clusters"):
                online_auc = normalized_auc(curve, f"delta_{metric}")
                local_value = float(local[f"auc_delta_{metric}"])
                item[f"auc_delta_{metric}"] = online_auc
                item[f"local_auc_delta_{metric}"] = local_value
                item[f"online_minus_local_{metric}"] = online_auc - local_value
                online_values[metric][model].append(online_auc)
                local_delta_values[metric][model].append(online_auc - local_value)
            auc_rows.append(item)

    comparisons = [
        paired_stats(
            online_values[metric],
            "online_merged_coverage_vs_random",
            f"class_equal_error_{metric}",
        )
        for metric in ("confusion", "clusters")
    ]
    adjusted = holm([row["permutation_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_primary_p"] = value
    comparisons.extend(
        paired_stats(
            local_delta_values[metric],
            "online_merged_vs_local_coverage",
            f"class_equal_error_{metric}",
        )
        for metric in ("confusion", "clusters")
    )
    return metric_rows, class_rows, auc_rows, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", type=Path, default=DEFAULT_ONLINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    local_rows = read_csv(POOL / "annotations/results.csv")
    local = {row["image_id"]: row for row in local_rows}
    online_rows = [row for row in read_csv(args.online) if row["status"] == "ok"]
    online = {row["image_id"]: row for row in online_rows}
    if len(online) != 200:
        raise ValueError(f"expected 200 successful online rows, got {len(online)}")
    field_rows, summary = agreement(local, online, schema)
    base = build_context()
    merged = dict(local)
    merged.update(online)
    changed = merged_context(base, merged, schema)
    orders = coverage_orders(changed)
    stability = order_stability(base, changed, read_csv(BASE / "tables/model_independent_orders.csv"))
    run_rows, comparisons = conclusion_sensitivity(changed, orders)
    strict_rows, strict_class_rows, strict_auc_rows, strict_comparisons = class_equal_error_sensitivity(
        changed, orders
    )
    write_csv(args.output / "field_agreement.csv", field_rows)
    write_csv(args.output / "selection_stability.csv", stability)
    write_csv(args.output / "sensitivity_by_run.csv", run_rows)
    write_csv(args.output / "sensitivity_comparisons.csv", comparisons)
    write_csv(args.output / "class_equal_error_metrics.csv", strict_rows)
    write_csv(args.output / "class_equal_error_by_class.csv", strict_class_rows)
    write_csv(args.output / "class_equal_error_auc.csv", strict_auc_rows)
    write_csv(args.output / "class_equal_error_comparisons.csv", strict_comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "agreement_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ordered = sorted(field_rows, key=lambda row: row["exact_agreement"])
    axes[0].barh([row["feature"] for row in ordered], [row["exact_agreement"] for row in ordered], color="#287271")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Exact agreement")
    axes[0].set_title("Local vs online VLM state labels")
    by_budget = defaultdict(list)
    for row in stability:
        by_budget[int(row["budget"])].append(float(row["jaccard"]))
    x = sorted(by_budget)
    axes[1].plot(x, [np.mean(by_budget[value]) for value in x], marker="o", color="#C44E52")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Candidates per class")
    axes[1].set_ylabel("Mean top-k Jaccard")
    axes[1].set_title("Coverage selection stability")
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output / "online_vlm_agreement.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "online_vlm_agreement.svg", bbox_inches="tight")
    plt.close(fig)

    local_comparisons = {
        row["metric"]: row
        for row in read_csv(CLASS_CONTROL / "tables/class_equal_error_comparisons.csv")
        if row["comparison"] == "coverage_vs_random"
    }
    online_comparisons = {
        row["metric"]: row
        for row in strict_comparisons
        if row["comparison"] == "online_merged_coverage_vs_random"
    }
    metrics = ("class_equal_error_confusion", "class_equal_error_clusters")
    labels = ("Confusion types", "Error clusters")
    positions = np.arange(len(metrics))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for offset, source, lookup, color in (
        (-width / 2, "Local VLM", local_comparisons, "#4C78A8"),
        (width / 2, "10% online replacement", online_comparisons, "#E45756"),
    ):
        means = np.asarray([float(lookup[metric]["mean_delta_auc"]) for metric in metrics])
        lows = np.asarray([float(lookup[metric]["ci95_low"]) for metric in metrics])
        highs = np.asarray([float(lookup[metric]["ci95_high"]) for metric in metrics])
        axis.bar(positions + offset, means, width, label=source, color=color)
        axis.errorbar(
            positions + offset,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            ecolor="#333333",
            capsize=4,
            linewidth=1.2,
        )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Coverage - Random AUC")
    axis.set_title("Class-stratified equal-error sensitivity")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output / "online_class_equal_error_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "online_class_equal_error_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    report = ["# 线上与本地VLM一致性及结论敏感性", "", "## 标注一致性", "",
        f"- 复标样本：{summary['samples']}张（每类40张，来自本地Coverage前序困难状态样本）。",
        f"- 19字段微平均精确一致率：{summary['micro_exact_agreement']:.2%}。",
        f"- enum宏平均精确一致率：{summary['enum_macro_exact_agreement']:.2%}；宏平均kappa：{summary['enum_macro_kappa']:.3f}。",
        f"- set宏平均精确一致率：{summary['set_macro_exact_agreement']:.2%}；宏平均Jaccard：{summary['set_macro_jaccard']:.3f}。", "",
        "## 结论敏感性", "",
        "将这200张线上标签替换进完整2000候选后重新生成Coverage顺序，冻结模型预测保持不变。", "",
        "| 比较 | 指标 | AUC差值 | 95% CI | 正向运行 |", "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        report.append(f"| {row['comparison']} | {row['metric']} | {row['mean_delta_auc']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['positive_runs']}/24 |")
    report += ["", "## 逐类别等错误数敏感性", "",
        "错误数在每个真实类别内分别匹配后，再比较线上合并Coverage与同一1000条Random排序。", "",
        "| 比较 | 指标 | AUC差值 | 95% CI | 正向运行 | Holm p |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in strict_comparisons:
        holm_value = row.get("holm_primary_p", "")
        holm_text = f"{holm_value:.4f}" if holm_value != "" else "—"
        report.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_delta_auc']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['positive_runs']}/24 | {holm_text} |"
        )
    report += ["", "## 边界", "",
        "- 抽样集中于Coverage前40张，刻意压力测试稀有/困难状态，不代表全池平均一致性。",
        "- 一致率衡量标注器稳定性，不等于任何一方是真值。",
        "- 敏感性分析只替换10%的候选标签，不能替代全池双标。",
        "- 线上合并与本地排序差异的置信区间跨0只表示未观察到稳定变化，不表示两种标注等效。", ""]
    (args.output / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
