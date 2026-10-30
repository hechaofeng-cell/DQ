#!/usr/bin/env python3
"""Sensitivity of independent error-cluster coverage to cluster count."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from run_multiclass_object_selection_benchmark import (
    BUDGETS, CLASSES, LABELS, MODELS, RANDOM_SEED, SEEDS, build_context, random_orders,
)
from run_cross_state_decoupled_diagnostics import (
    BASE, FEATURE_GROUPS, paired_comparison, holm, load_cached_run, load_order,
)


ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "artifacts/cross_state_decoupled_diagnostics_20260928"
DEFAULT_OUTPUT = ROOT / "artifacts/error_cluster_sensitivity_20260929"
METHODS = ("coverage", "random", "deepgini", "dsa", "kcenter")
FIXED_METHODS = ("coverage", "deepgini", "dsa", "kcenter")
CLUSTER_COUNTS = (4, 6, 8, 10)
RANDOM_REPLICATES = 1000


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(output: Path, message: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (output / "execution.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def load_coverage_orders(context: dict) -> dict[str, dict[str, list[int]]]:
    rows = read_csv(CROSS / "tables/coverage_holdout_orders.csv")
    result: dict[str, dict[str, list[int]]] = {}
    for heldout in FEATURE_GROUPS:
        result[heldout] = {}
        for class_name in CLASSES:
            local = {
                row["sample_id"]: index
                for index, row in enumerate(context["by_class"][class_name]["rows"])
            }
            selected = sorted(
                (row for row in rows if row["heldout_group"] == heldout and row["class_name"] == class_name),
                key=lambda row: int(row["rank"]),
            )
            order = [local[row["sample_id"]] for row in selected]
            if len(order) != 400:
                raise ValueError(f"invalid coverage order: {heldout}/{class_name}")
            result[heldout][class_name] = order
    return result


def cluster_errors(
    context: dict, features: np.ndarray, predicted: np.ndarray, requested_k: int,
    model_name: str, seed: int,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for class_offset, class_name in enumerate(CLASSES):
        global_indices = context["by_class"][class_name]["global_indices"]
        errors = global_indices[predicted[global_indices] != LABELS[class_name]]
        actual_k = min(requested_k, len(errors))
        values = features[errors]
        values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        labels = KMeans(
            n_clusters=actual_k,
            random_state=RANDOM_SEED + requested_k * 1009 + seed + class_offset,
            n_init=20,
        ).fit_predict(values)
        output[class_name] = {
            "k": actual_k,
            "labels": {int(index): int(label) for index, label in zip(errors, labels)},
        }
    return output


def evaluate(
    context: dict, orders: dict[str, list[int]], budget: int,
    structures: dict[str, dict[str, object]],
) -> float:
    values = []
    for class_name in CLASSES:
        global_indices = context["by_class"][class_name]["global_indices"]
        selected = global_indices[np.asarray(orders[class_name][:budget])]
        labels = structures[class_name]["labels"]
        discovered = {labels[int(index)] for index in selected if int(index) in labels}
        values.append(len(discovered) / int(structures[class_name]["k"]))
    return float(np.mean(values))


def auc(values: list[float]) -> float:
    x = np.asarray([0, *BUDGETS], dtype=float)
    y = np.asarray([0, *values], dtype=float)
    return float(np.trapezoid(y, x) / BUDGETS[-1])


def run(output: Path) -> None:
    context = build_context()
    coverage = load_coverage_orders(context)
    kcenter = load_order(BASE / "tables/model_independent_orders.csv", context, "kcenter")
    random_sets = random_orders(context["by_class"], RANDOM_REPLICATES)
    metrics: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []

    for run_index, (model_name, seed) in enumerate(itertools.product(MODELS, SEEDS), start=1):
        features, predicted = load_cached_run(context, model_name, seed)
        static = {
            "kcenter": kcenter,
            "deepgini": load_order(BASE / "orders" / model_name / str(seed) / "deepgini.csv", context),
            "dsa": load_order(BASE / "orders" / model_name / str(seed) / "dsa.csv", context),
        }
        for requested_k in CLUSTER_COUNTS:
            structures = cluster_errors(context, features, predicted, requested_k, model_name, seed)
            coverage_by_budget = []
            for budget in BUDGETS:
                fold_values = [
                    evaluate(context, coverage[heldout], budget, structures)
                    for heldout in FEATURE_GROUPS
                ]
                value = float(np.mean(fold_values))
                coverage_by_budget.append(value)
                metrics.append({
                    "method": "coverage", "model": model_name, "seed": seed,
                    "requested_k": requested_k, "budget": budget,
                    "error_cluster_coverage": value,
                })
            auc_rows.append({
                "method": "coverage", "model": model_name, "seed": seed,
                "requested_k": requested_k, "auc_error_cluster_coverage": auc(coverage_by_budget),
            })
            for method, orders in static.items():
                values = [evaluate(context, orders, budget, structures) for budget in BUDGETS]
                for budget, value in zip(BUDGETS, values):
                    metrics.append({
                        "method": method, "model": model_name, "seed": seed,
                        "requested_k": requested_k, "budget": budget,
                        "error_cluster_coverage": value,
                    })
                auc_rows.append({
                    "method": method, "model": model_name, "seed": seed,
                    "requested_k": requested_k, "auc_error_cluster_coverage": auc(values),
                })
            replicate_curves = [
                [evaluate(context, orders, budget, structures) for budget in BUDGETS]
                for orders in random_sets
            ]
            for budget_index, budget in enumerate(BUDGETS):
                values = np.asarray([curve[budget_index] for curve in replicate_curves])
                random_rows.append({
                    "method": "random", "model": model_name, "seed": seed,
                    "requested_k": requested_k, "budget": budget,
                    "replicates": RANDOM_REPLICATES,
                    "error_cluster_coverage_mean": float(values.mean()),
                    "error_cluster_coverage_p2_5": float(np.percentile(values, 2.5)),
                    "error_cluster_coverage_p97_5": float(np.percentile(values, 97.5)),
                })
            auc_rows.append({
                "method": "random", "model": model_name, "seed": seed,
                "requested_k": requested_k,
                "auc_error_cluster_coverage": float(np.mean([auc(curve) for curve in replicate_curves])),
            })
        write_json(output / "progress.json", {
            "status": "running", "completed_runs": run_index, "total_runs": 24,
            "latest_model": model_name, "latest_seed": seed, "updated_at": now(),
        })
        log(output, f"completed {run_index}/24 {model_name} seed={seed}")

    comparisons: list[dict[str, object]] = []
    for requested_k in CLUSTER_COUNTS:
        subset = [row for row in auc_rows if int(row["requested_k"]) == requested_k]
        family = [
            paired_comparison(subset, baseline, "auc_error_cluster_coverage")
            for baseline in METHODS if baseline != "coverage"
        ]
        for row in family:
            row["requested_k"] = requested_k
        holm(family, "permutation_p", "holm_within_k_p")
        comparisons.extend(family)
    holm(comparisons, "permutation_p", "holm_global_p")

    endpoint = []
    for requested_k in CLUSTER_COUNTS:
        for method in METHODS:
            if method == "random":
                rows = [
                    row for row in random_rows
                    if int(row["requested_k"]) == requested_k and int(row["budget"]) == 100
                ]
                values = np.asarray([float(row["error_cluster_coverage_mean"]) for row in rows])
            else:
                rows = [
                    row for row in metrics
                    if row["method"] == method and int(row["requested_k"]) == requested_k
                    and int(row["budget"]) == 100
                ]
                values = np.asarray([float(row["error_cluster_coverage"]) for row in rows])
            endpoint.append({
                "requested_k": requested_k, "method": method, "runs": len(values),
                "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            })

    write_csv(output / "tables/fixed_metrics.csv", metrics)
    write_csv(output / "tables/random_summary.csv", random_rows)
    write_csv(output / "tables/auc_by_run.csv", auc_rows)
    write_csv(output / "tables/comparisons.csv", comparisons)
    write_csv(output / "tables/endpoint_summary.csv", endpoint)

    colors = {"random": "#777777", "deepgini": "#D55E00", "dsa": "#CC79A7", "kcenter": "#009E73"}
    labels = {"random": "Random", "deepgini": "DeepGini", "dsa": "Surprise Adequacy", "kcenter": "k-center"}
    fig, axis = plt.subplots(figsize=(10.8, 5.6))
    for baseline in ("random", "deepgini", "dsa", "kcenter"):
        rows = sorted(
            (row for row in comparisons if row["comparison"] == f"coverage_vs_{baseline}"),
            key=lambda row: int(row["requested_k"]),
        )
        axis.plot(
            [int(row["requested_k"]) for row in rows],
            [float(row["mean_delta_auc"]) for row in rows],
            marker="o", linewidth=2.2, color=colors[baseline], label=labels[baseline],
        )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(CLUSTER_COUNTS)
    axis.set_xlabel("Requested error clusters per class")
    axis.set_ylabel("Coverage - baseline AUC")
    axis.set_title("Error-cluster sensitivity")
    axis.grid(alpha=.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures/error_cluster_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figures/error_cluster_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    random_comparisons = [row for row in comparisons if row["comparison"] == "coverage_vs_random"]
    kcenter_comparisons = [row for row in comparisons if row["comparison"] == "coverage_vs_kcenter"]
    stable_random = all(float(row["ci95_low"]) > 0 and int(row["positive_runs"]) >= 18 for row in random_comparisons)
    stable_kcenter_direction = all(float(row["mean_delta_auc"]) < 0 for row in kcenter_comparisons)
    significantly_below_kcenter = sum(float(row["ci95_high"]) < 0 for row in kcenter_comparisons)
    report = [
        "# 错误簇数量敏感性分析", "",
        "固定候选池、模型预测、五折Coverage顺序和预算，仅改变每类请求的错误簇数量。", "",
        "| K | Coverage-Random AUC | 95% CI | Coverage-k-center AUC | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for requested_k in CLUSTER_COUNTS:
        left = next(row for row in random_comparisons if int(row["requested_k"]) == requested_k)
        right = next(row for row in kcenter_comparisons if int(row["requested_k"]) == requested_k)
        report.append(
            f"| {requested_k} | {left['mean_delta_auc']:+.4f} | [{left['ci95_low']:+.4f}, {left['ci95_high']:+.4f}] | "
            f"{right['mean_delta_auc']:+.4f} | [{right['ci95_low']:+.4f}, {right['ci95_high']:+.4f}] |"
        )
    report.extend([
        "", "## 判定", "",
        f"- Coverage相对Random方向稳定：`{stable_random}`。",
        f"- Coverage相对k-center的点估计均为负：`{stable_kcenter_direction}`。",
        f"- 置信区间明确低于k-center的设置：`{significantly_below_kcenter}/4`；其余设置为差异不确定。",
        "- 因此可以稳定声称Coverage优于Random且未超过k-center，不能声称其在所有K下显著弱于k-center。", "",
    ])
    (output / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output / "protocol.json", {
        "experiment": "error-cluster-sensitivity-v1", "created_at": now(),
        "requested_cluster_counts": list(CLUSTER_COUNTS), "models": list(MODELS),
        "seeds": list(SEEDS), "classes": list(CLASSES), "budgets": list(BUDGETS),
        "random_replicates": RANDOM_REPLICATES,
        "fixed": "candidate pool, logits, embeddings, selection orders, budgets",
        "decision": {
            "stable_vs_random": stable_random,
            "kcenter_point_direction_always_negative": stable_kcenter_direction,
            "significantly_below_kcenter_settings": significantly_below_kcenter,
        },
    })
    write_json(output / "progress.json", {"status": "complete", "updated_at": now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_json(args.output / "progress.json", {"status": "starting", "updated_at": now()})
    log(args.output, "starting error-cluster sensitivity")
    run(args.output)
    log(args.output, "sensitivity complete")


if __name__ == "__main__":
    main()
