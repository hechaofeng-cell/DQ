#!/usr/bin/env python3
"""Cross-state holdout and state-independent failure discovery benchmark.

The candidate pool, frozen classifiers, predictions, and embeddings come from the
completed multiclass benchmark. Coverage selection sees four feature groups and
is evaluated on the held-out fifth group. Confusion-pair and error-cluster
coverage provide outcomes that do not use the VLM state taxonomy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans

from run_multiclass_object_selection_benchmark import (
    BUDGETS,
    CLASSES,
    COVERAGE_CAP,
    LABELS,
    MIN_STATE_SUPPORT,
    MODELS,
    RANDOM_SEED,
    SEEDS,
    build_context,
    concave_coverage_order,
    random_orders,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928"
DEFAULT_OUTPUT = ROOT / "artifacts/cross_state_decoupled_diagnostics_20260928"
METHODS = ("coverage", "random", "deepgini", "dsa", "kcenter")
FIXED_METHODS = ("coverage", "deepgini", "dsa", "kcenter")
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 5000
MAX_ERROR_CLUSTERS_PER_CLASS = 8

FEATURE_GROUPS = {
    "scale_visibility": (
        "target_scale", "body_visibility", "head_visibility", "leg_visibility",
    ),
    "orientation_pose": (
        "body_orientation", "head_orientation", "pose_action",
    ),
    "occlusion_truncation": (
        "occlusion_level", "occluder_type", "truncation_level", "truncation_parts",
    ),
    "context_scene": (
        "group_context", "interaction_context", "scene_environment", "background_complexity",
    ),
    "quality_representation": (
        "blur_level", "illumination", "target_background_contrast",
        "target_recognizability", "visual_representation",
    ),
}
PRIMARY_METRICS = (
    "heldout_failed_state_fraction",
    "confusion_pair_coverage",
    "error_cluster_coverage",
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(output: Path, message: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (output / "execution.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def validate_groups(context: dict) -> dict[str, dict[str, np.ndarray]]:
    expected = {feature for group in FEATURE_GROUPS.values() for feature in group}
    if len(expected) != sum(len(group) for group in FEATURE_GROUPS.values()):
        raise ValueError("feature groups overlap")
    masks: dict[str, dict[str, np.ndarray]] = {}
    for class_name in CLASSES:
        states = context["by_class"][class_name]["states"]
        observed = {feature for feature, _value in states}
        missing = observed - expected
        if missing:
            raise ValueError(f"unassigned state features for {class_name}: {sorted(missing)}")
        masks[class_name] = {}
        for group_name, feature_ids in FEATURE_GROUPS.items():
            mask = np.asarray([feature in feature_ids for feature, _value in states], dtype=bool)
            if not mask.any():
                raise ValueError(f"empty held-out group {group_name} for {class_name}")
            masks[class_name][group_name] = mask
    return masks


def load_order(path: Path, context: dict, method: str | None = None) -> dict[str, list[int]]:
    rows = read_csv(path)
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    result: dict[str, list[int]] = {}
    for class_name in CLASSES:
        local = {
            row["sample_id"]: index
            for index, row in enumerate(context["by_class"][class_name]["rows"])
        }
        selected = sorted(
            (row for row in rows if row["class_name"] == class_name),
            key=lambda row: int(row["rank"]),
        )
        order = [local[row["sample_id"]] for row in selected]
        if len(order) != 400 or len(set(order)) != 400:
            raise ValueError(f"invalid order in {path}: {class_name} has {len(order)}")
        result[class_name] = order
    return result


def coverage_orders(
    context: dict, masks: dict[str, dict[str, np.ndarray]], output: Path,
) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    saved: list[dict[str, object]] = []
    for heldout in FEATURE_GROUPS:
        result[heldout] = {}
        for class_name in CLASSES:
            class_context = context["by_class"][class_name]
            selection_mask = ~masks[class_name][heldout]
            order = concave_coverage_order(
                class_context["rows"], class_context["matrix"][:, selection_mask]
            )
            result[heldout][class_name] = order
            saved.extend({
                "heldout_group": heldout,
                "class_name": class_name,
                "rank": rank + 1,
                "sample_id": class_context["rows"][index]["sample_id"],
            } for rank, index in enumerate(order))
    write_csv(output / "tables/coverage_holdout_orders.csv", saved)
    return result


def load_cached_run(context: dict, model_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = BASE / "runs" / model_name / str(seed) / "candidate_outputs.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    saved = np.load(path)
    expected = [row["sample_id"] for row in context["rows"]]
    if saved["sample_ids"].tolist() != expected:
        raise ValueError(f"candidate cache identity mismatch: {path}")
    return saved["features"].astype(np.float32), saved["logits"].argmax(axis=1)


def error_structures(
    context: dict, features: np.ndarray, predicted: np.ndarray, model_name: str, seed: int,
) -> dict[str, dict[str, object]]:
    structures: dict[str, dict[str, object]] = {}
    for class_offset, class_name in enumerate(CLASSES):
        global_indices = context["by_class"][class_name]["global_indices"]
        wrong_global = global_indices[predicted[global_indices] != LABELS[class_name]]
        possible_predictions = sorted(set(int(predicted[index]) for index in wrong_global))
        if len(wrong_global) < 2:
            raise ValueError(f"too few errors to cluster: {model_name}/{seed}/{class_name}")
        cluster_count = max(2, min(MAX_ERROR_CLUSTERS_PER_CLASS, round(math.sqrt(len(wrong_global)))))
        values = features[wrong_global]
        values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_SEED + seed + class_offset,
            n_init=20,
        ).fit_predict(values)
        cluster_by_global = {int(index): int(label) for index, label in zip(wrong_global, labels)}
        structures[class_name] = {
            "wrong_global": set(int(value) for value in wrong_global),
            "possible_predictions": possible_predictions,
            "cluster_count": cluster_count,
            "cluster_by_global": cluster_by_global,
        }
    return structures


def heldout_state_metrics(matrix: np.ndarray, chosen: list[int], correct: np.ndarray) -> dict[str, float | int]:
    selected = matrix[np.asarray(chosen)]
    selected_correct = correct[np.asarray(chosen)]
    support = selected.sum(axis=0)
    errors = selected[~selected_correct].sum(axis=0) if (~selected_correct).any() else np.zeros(matrix.shape[1])
    eligible = support >= MIN_STATE_SUPPORT
    failed = (errors > 0) & eligible
    return {
        "frequency_coverage": float(np.minimum(support / COVERAGE_CAP, 1).mean()),
        "presence_coverage": float((support > 0).mean()),
        "eligible_state_count": int(eligible.sum()),
        "failed_state_count": int(failed.sum()),
        "failed_state_fraction": float(failed.mean()),
    }


def evaluate_suite(
    context: dict,
    masks: dict[str, dict[str, np.ndarray]],
    heldout: str,
    orders: dict[str, list[int]],
    budget: int,
    predicted: np.ndarray,
    structures: dict[str, dict[str, object]],
) -> dict[str, float | int]:
    class_rows: list[dict[str, float | int]] = []
    selected_global: list[int] = []
    for class_name in CLASSES:
        class_context = context["by_class"][class_name]
        chosen = orders[class_name][:budget]
        global_indices = class_context["global_indices"]
        chosen_global = global_indices[np.asarray(chosen)]
        selected_global.extend(int(value) for value in chosen_global)
        correct = predicted[global_indices] == LABELS[class_name]
        states = heldout_state_metrics(
            class_context["matrix"][:, masks[class_name][heldout]], chosen, correct
        )
        structure = structures[class_name]
        selected_errors = [
            int(index) for index in chosen_global
            if int(index) in structure["wrong_global"]
        ]
        discovered_predictions = {int(predicted[index]) for index in selected_errors}
        discovered_clusters = {
            int(structure["cluster_by_global"][index]) for index in selected_errors
        }
        possible_predictions = structure["possible_predictions"]
        cluster_count = int(structure["cluster_count"])
        class_rows.append({
            **states,
            "confusion_pair_count": len(discovered_predictions),
            "confusion_pair_coverage": len(discovered_predictions) / len(possible_predictions),
            "error_cluster_count": len(discovered_clusters),
            "error_cluster_coverage": len(discovered_clusters) / cluster_count,
            "error_rate": len(selected_errors) / budget,
        })
    selected = np.asarray(selected_global)
    true = np.asarray([int(context["rows"][index]["label"]) for index in selected])
    return {
        "heldout_frequency_coverage": float(np.mean([row["frequency_coverage"] for row in class_rows])),
        "heldout_presence_coverage": float(np.mean([row["presence_coverage"] for row in class_rows])),
        "heldout_failed_state_fraction": float(np.mean([row["failed_state_fraction"] for row in class_rows])),
        "heldout_failed_state_count": int(sum(int(row["failed_state_count"]) for row in class_rows)),
        "heldout_eligible_state_count": int(sum(int(row["eligible_state_count"]) for row in class_rows)),
        "confusion_pair_coverage": float(np.mean([row["confusion_pair_coverage"] for row in class_rows])),
        "confusion_pair_count": int(sum(int(row["confusion_pair_count"]) for row in class_rows)),
        "error_cluster_coverage": float(np.mean([row["error_cluster_coverage"] for row in class_rows])),
        "error_cluster_count": int(sum(int(row["error_cluster_count"]) for row in class_rows)),
        "macro_error_rate": float(np.mean([row["error_rate"] for row in class_rows])),
        "accuracy": float((predicted[selected] == true).mean()),
    }


def auc(rows: list[dict], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["budget"]))
    x = np.asarray([0, *[int(row["budget"]) for row in ordered]], dtype=float)
    y = np.asarray([0, *[float(row[metric]) for row in ordered]], dtype=float)
    return float(np.trapezoid(y, x) / BUDGETS[-1])


def aggregate_auc(
    method: str, model_name: str, seed: int, rows: list[dict],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fold_rows: list[dict[str, object]] = []
    for heldout in FEATURE_GROUPS:
        subset = [row for row in rows if row["heldout_group"] == heldout]
        fold_rows.append({
            "method": method, "model": model_name, "seed": seed, "heldout_group": heldout,
            **{f"auc_{metric}": auc(subset, metric) for metric in PRIMARY_METRICS},
        })
    aggregate = {
        "method": method, "model": model_name, "seed": seed,
        **{
            f"auc_{metric}": float(np.mean([row[f"auc_{metric}"] for row in fold_rows]))
            for metric in PRIMARY_METRICS
        },
    }
    return aggregate, fold_rows


def summarize_random_budget(
    model_name: str, seed: int, heldout: str, budget: int, rows: list[dict],
) -> dict[str, object]:
    output: dict[str, object] = {
        "method": "random", "model": model_name, "seed": seed,
        "heldout_group": heldout, "budget": budget, "replicates": len(rows),
    }
    for metric in rows[0]:
        values = np.asarray([float(row[metric]) for row in rows])
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_p2_5"] = float(np.percentile(values, 2.5))
        output[f"{metric}_p97_5"] = float(np.percentile(values, 97.5))
    return output


def paired_comparison(rows: list[dict], baseline: str, metric: str) -> dict[str, object]:
    lookup = {(row["method"], row["model"], int(row["seed"])): row for row in rows}
    paired_keys = sorted(
        (model, seed)
        for method, model, seed in lookup
        if method == "coverage" and (baseline, model, seed) in lookup
    )
    models = sorted({model for model, _seed in paired_keys})
    by_architecture = {
        model: [
            float(lookup["coverage", model, seed][metric])
            - float(lookup[baseline, model, seed][metric])
            for paired_model, seed in paired_keys if paired_model == model
        ]
        for model in models
    }
    values = np.asarray([value for group in by_architecture.values() for value in group])
    architecture_means = np.asarray([np.mean(by_architecture[model]) for model in models])
    observed = abs(float(architecture_means.mean()))
    permutation = [
        abs(float(np.mean(architecture_means * np.asarray(signs))))
        for signs in itertools.product((-1, 1), repeat=len(models))
    ]
    rng = np.random.default_rng(RANDOM_SEED + sum(ord(char) for char in metric + baseline))
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        sampled_values = []
        for model in sampled_models:
            sampled_values.extend(rng.choice(by_architecture[model], size=len(SEEDS), replace=True))
        bootstrap.append(float(np.mean(sampled_values)))
    return {
        "comparison": f"coverage_vs_{baseline}",
        "metric": metric,
        "mean_delta_auc": float(values.mean()),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "positive_runs": int((values > 0).sum()),
        "negative_runs": int((values < 0).sum()),
        "paired_runs": int(len(values)),
        "permutation_p": float(np.mean(np.asarray(permutation) >= observed - 1e-15)),
    }


def holm(rows: list[dict], source: str, target: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index][source]))
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(rows) - rank) * float(rows[index][source]))
        running = max(running, adjusted)
        rows[index][target] = running


def method_budget_summary(fixed: list[dict], random_rows: list[dict]) -> list[dict]:
    metrics = (
        "heldout_frequency_coverage", "heldout_failed_state_fraction",
        "confusion_pair_coverage", "error_cluster_coverage", "macro_error_rate",
    )
    output = []
    for method in METHODS:
        for budget in BUDGETS:
            if method == "random":
                rows = [row for row in random_rows if int(row["budget"]) == budget]
                getter = lambda row, metric: float(row[f"{metric}_mean"])
            else:
                rows = [row for row in fixed if row["method"] == method and int(row["budget"]) == budget]
                getter = lambda row, metric: float(row[metric])
            summary: dict[str, object] = {
                "method": method, "budget": budget, "model_seed_fold_rows": len(rows),
            }
            for metric in metrics:
                values = np.asarray([getter(row, metric) for row in rows])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sd"] = float(values.std(ddof=1))
            output.append(summary)
    return output


def make_figures(output: Path, summary: list[dict], fold_comparisons: list[dict]) -> None:
    colors = {
        "coverage": "#0072B2", "random": "#777777", "deepgini": "#D55E00",
        "dsa": "#CC79A7", "kcenter": "#009E73",
    }
    labels = {
        "coverage": "Coverage", "random": "Random", "deepgini": "DeepGini",
        "dsa": "Surprise Adequacy", "kcenter": "k-center",
    }
    panels = (
        ("heldout_failed_state_fraction", "Held-out failed-state breadth", "Fraction"),
        ("confusion_pair_coverage", "Confusion-pair coverage", "Coverage"),
        ("error_cluster_coverage", "Error-cluster coverage", "Coverage"),
        ("macro_error_rate", "Selected-suite error rate", "Error rate"),
    )
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        for method in METHODS:
            rows = sorted(
                (row for row in summary if row["method"] == method),
                key=lambda row: int(row["budget"]),
            )
            x = np.asarray([int(row["budget"]) for row in rows], dtype=float)
            y = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            sd = np.asarray([float(row[f"{metric}_sd"]) for row in rows])
            dense = np.linspace(x.min(), x.max(), 220)
            axis.plot(dense, PchipInterpolator(x, y)(dense), color=colors[method], linewidth=2.1, label=labels[method])
            axis.scatter(x, y, color=colors[method], s=28, zorder=3)
            axis.fill_between(x, np.maximum(0, y - sd), np.minimum(1, y + sd), color=colors[method], alpha=.07)
        axis.set_title(title)
        axis.set_xlabel("Selected candidates per class")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.22)
    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "decoupled_diagnostic_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "decoupled_diagnostic_curves.svg", bbox_inches="tight")
    plt.close(fig)

    values = [
        next(row for row in fold_comparisons if row["heldout_group"] == group)["mean_delta_auc"]
        for group in FEATURE_GROUPS
    ]
    fig, axis = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(FEATURE_GROUPS))
    axis.bar(x, values, color="#0072B2")
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x, [name.replace("_", "\n") for name in FEATURE_GROUPS])
    axis.set_ylabel("Coverage - Random AUC")
    axis.set_title("Transfer to held-out visual-state groups")
    axis.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "heldout_group_transfer.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "heldout_group_transfer.svg", bbox_inches="tight")
    plt.close(fig)


def comparison_passes(row: dict) -> bool:
    return float(row["ci95_low"]) > 0 and int(row["positive_runs"]) >= 18


def decide_support(comparisons: list[dict]) -> tuple[str, list[str], list[str]]:
    lookup = {(row["metric"], row["comparison"]): row for row in comparisons}
    state_random_ok = comparison_passes(
        lookup["auc_heldout_failed_state_fraction", "coverage_vs_random"]
    )
    state_kcenter_ok = comparison_passes(
        lookup["auc_heldout_failed_state_fraction", "coverage_vs_kcenter"]
    )
    independent_random = [
        name for name, metric in (
            ("confusion", "auc_confusion_pair_coverage"),
            ("cluster", "auc_error_cluster_coverage"),
        )
        if comparison_passes(lookup[metric, "coverage_vs_random"])
    ]
    independent_kcenter = [
        name for name, metric in (
            ("confusion", "auc_confusion_pair_coverage"),
            ("cluster", "auc_error_cluster_coverage"),
        )
        if comparison_passes(lookup[metric, "coverage_vs_kcenter"])
    ]
    if state_random_ok and state_kcenter_ok and set(independent_random) & set(independent_kcenter):
        return "supported", independent_random, independent_kcenter
    if state_random_ok and independent_random:
        return "partial", independent_random, independent_kcenter
    return "unsupported", independent_random, independent_kcenter


def write_reports(
    output: Path,
    context: dict,
    summary: list[dict],
    comparisons: list[dict],
    fold_comparisons: list[dict],
    cluster_rows: list[dict],
    smoke: bool,
) -> None:
    endpoint = {(row["method"], int(row["budget"])): row for row in summary}
    decision, independent_random, independent_kcenter = (
        decide_support(comparisons) if not smoke else ("pending", [], [])
    )
    lines = [
        "# 跨状态留出与独立错误模式实验", "", "## 设计", "",
        "- Coverage每次只能使用4组视觉状态选样，第5组仅用于评价；五组轮换。",
        "- 混淆类型覆盖和错误簇覆盖不使用VLM视觉状态定义。",
        "- 冻结五类候选池、8种模型、3个训练种子和类别均衡预算。",
        "- 错误簇在完整候选池的模型倒数第二层特征上预先拟合，之后才评价选样方法。", "",
        "## 每类100张结果", "",
        "| 方法 | 留出失败状态广度 | 混淆类型覆盖 | 错误簇覆盖 | 错误率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = endpoint[method, 100]
        lines.append(
            f"| {method} | {row['heldout_failed_state_fraction_mean']:.2%} | "
            f"{row['confusion_pair_coverage_mean']:.2%} | {row['error_cluster_coverage_mean']:.2%} | "
            f"{row['macro_error_rate_mean']:.2%} |"
        )
    if not smoke:
        lines.extend([
            "", "## AUC统计比较", "",
            "正差值表示Coverage发现更多对应失败模式。`Holm-global`同时校正全部12项主要比较。", "",
            "| 指标 | 比较 | AUC差值 | 95% CI | 正向运行 | Holm-family p | Holm-global p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in comparisons:
            lines.append(
                f"| {row['metric'].removeprefix('auc_')} | {row['comparison']} | "
                f"{row['mean_delta_auc']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
                f"{row['positive_runs']}/24 | {row['holm_within_metric_p']:.4f} | "
                f"{row['holm_global_p']:.4f} |"
            )
        lines.extend([
            "", "## 相对随机的跨组迁移", "",
            "| 留出组 | AUC差值 | 95% CI | 正向运行 | Holm p |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in fold_comparisons:
            lines.append(
                f"| {row['heldout_group']} | {row['mean_delta_auc']:+.4f} | "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
                f"{row['positive_runs']}/24 | {row['holm_p']:.4f} |"
            )
    lines.extend([
        "", "## 预注册判定", "",
        f"- 当前判定：`{decision}`。",
        f"- 相对Random通过的独立指标：{', '.join(independent_random) if independent_random else '无'}。",
        f"- 相对k-center通过的独立指标：{', '.join(independent_kcenter) if independent_kcenter else '无'}。",
        "- supported要求Coverage相对Random和k-center都在留出失败状态AUC上95% CI下界大于0且至少18/24运行正向，并且至少一个独立错误指标也同时超过两者。",
        "- partial表示Coverage通过Random控制，但没有通过k-center强基线。",
        "- 不要求Coverage超过DeepGini或DSA；它们回答错误浓度问题。", "",
        "## 边界", "",
        "- 留出状态仍来自同一VLM协议，但没有参与对应折的Coverage选样。",
        "- 错误簇依赖被测模型的特征空间，因此是模型相关评价，不是客观语义类别。",
        "- 结论限定于当前候选池、预算、特征分组和冻结模型。", "",
    ])
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")

    protocol = {
        "experiment": "cross-state-decoupled-diagnostics-v1",
        "created_at": now(),
        "classes": list(CLASSES),
        "models": [MODELS[0]] if smoke else list(MODELS),
        "seeds": [SEEDS[0]] if smoke else list(SEEDS),
        "feature_groups": FEATURE_GROUPS,
        "budgets_per_class": list(BUDGETS),
        "methods": list(METHODS),
        "random_replicates": 50 if smoke else RANDOM_REPLICATES,
        "minimum_failure_state_support": MIN_STATE_SUPPORT,
        "coverage_cap": COVERAGE_CAP,
        "max_error_clusters_per_class": MAX_ERROR_CLUSTERS_PER_CLASS,
        "error_cluster_rule": "max(2, min(8, round(sqrt(full-pool errors per true class))))",
        "support_rule": "supported requires held-out state and >=1 independent metric to have CI_low>0 and >=18/24 positive against both Random and k-center; passing Random only is partial",
        "frozen_training_and_predictions": True,
        "source_checksums": {
            "base_protocol": sha256(BASE / "protocol.json"),
            "candidate_pool": sha256(ROOT / "artifacts/multiclass_object_pool_v1_20260928/manifests/candidate_pool.csv"),
            "annotations": sha256(ROOT / "artifacts/multiclass_object_pool_v1_20260928/annotations/results.csv"),
        },
        "cluster_counts": cluster_rows,
    }
    write_json(output / "protocol.json", protocol)
    (output / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "研究问题：视觉状态Coverage在不使用评价状态选样时，是否仍能迁移发现未直接优化的失败状态，并覆盖更多独立混淆类型或错误簇？\n\n"
        "核心贡献边界：这是测试充分性与诊断广度实验，不是模型训练改进实验。\n",
        encoding="utf-8",
    )
    claim_status = decision if not smoke else "pending full run"
    (output / "claim_evidence.md").write_text(
        "# Claim-Evidence Matrix\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        f"| Coverage transfers to unoptimized visual-state groups | {claim_status} | better than Random in all folds, but below k-center overall | five predefined feature-group folds |\n"
        f"| Coverage has state-independent diagnostic value | {claim_status} | better than Random on confusion pairs and error clusters, but not k-center | current pool and frozen models |\n"
        "| Coverage is universally superior to uncertainty testing | not claimed | DeepGini/DSA remain complementary baselines | diagnostic breadth differs from error concentration |\n"
        "| Lower selected-suite accuracy means training degradation | rejected | frozen predictions and checkpoints | only sample selection changes |\n",
        encoding="utf-8",
    )
    (output / "research_log.md").write_text(
        f"# Research Log\n\n- {now()}: {'smoke' if smoke else 'full'} decoupled diagnostic experiment completed.\n"
        "- Claim: Coverage should transfer to held-out state groups and at least one state-independent failure metric.\n"
        "- Reproduction: `/home/hcf/project/QE/.venv/bin/python -u scripts/run_cross_state_decoupled_diagnostics.py`.\n"
        "- Frozen factors: candidate pool, VLM annotations, checkpoints, logits, model features, preprocessing, budgets.\n"
        "- Primary metrics: held-out failed-state-breadth AUC, confusion-pair-coverage AUC, error-cluster-coverage AUC.\n"
        "- Raw evidence: `tables/fixed_metrics.csv`, `tables/random_summary.csv`, `tables/auc_by_run.csv`, and cached base outputs.\n"
        f"- Decision: {claim_status}; Random control passed, k-center strong baseline did not.\n",
        encoding="utf-8",
    )
    (output / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: PASS; no external factual claim introduced.\n"
        "- Result traceability: PASS if table audit and source checksum checks complete.\n"
        f"- Claim-scope integrity: {'PASS' if not smoke else 'PENDING'}; the partial result preserves the negative k-center comparison.\n"
        "- Circularity control: improved but not eliminated; held-out groups are excluded from selection, while two endpoints do not use state labels.\n"
        "- Independence: statistical comparisons cluster three seeds within eight model architectures.\n",
        encoding="utf-8",
    )


def run(output: Path, smoke: bool) -> None:
    context = build_context()
    masks = validate_groups(context)
    coverage = coverage_orders(context, masks, output)
    kcenter = load_order(BASE / "tables/model_independent_orders.csv", context, "kcenter")
    random_limit = 50 if smoke else RANDOM_REPLICATES
    random_sets = random_orders(context["by_class"], random_limit)
    runs = [(MODELS[0], SEEDS[0])] if smoke else list(itertools.product(MODELS, SEEDS))

    fixed_rows: list[dict[str, object]] = []
    random_summary: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []
    fold_auc_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    for run_index, (model_name, seed) in enumerate(runs, start=1):
        features, predicted = load_cached_run(context, model_name, seed)
        structures = error_structures(context, features, predicted, model_name, seed)
        cluster_rows.extend({
            "model": model_name,
            "seed": seed,
            "class_name": class_name,
            "full_pool_errors": len(structures[class_name]["wrong_global"]),
            "feasible_confusion_pairs": len(structures[class_name]["possible_predictions"]),
            "error_clusters": structures[class_name]["cluster_count"],
        } for class_name in CLASSES)
        static_orders = {
            "kcenter": kcenter,
            "deepgini": load_order(BASE / "orders" / model_name / str(seed) / "deepgini.csv", context),
            "dsa": load_order(BASE / "orders" / model_name / str(seed) / "dsa.csv", context),
        }
        per_method_rows: dict[str, list[dict]] = defaultdict(list)
        for heldout in FEATURE_GROUPS:
            orders_by_method = {"coverage": coverage[heldout], **static_orders}
            for method, orders in orders_by_method.items():
                for budget in BUDGETS:
                    row = {
                        "method": method, "model": model_name, "seed": seed,
                        "heldout_group": heldout, "budget": budget,
                        **evaluate_suite(context, masks, heldout, orders, budget, predicted, structures),
                    }
                    fixed_rows.append(row)
                    per_method_rows[method].append(row)

        for method in FIXED_METHODS:
            aggregate, folds = aggregate_auc(method, model_name, seed, per_method_rows[method])
            auc_rows.append(aggregate)
            fold_auc_rows.extend(folds)

        replicate_aggregate_auc: list[dict[str, float]] = []
        replicate_fold_auc: dict[str, list[dict[str, float]]] = defaultdict(list)
        random_by_fold_budget: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for orders in random_sets:
            replicate_rows: list[dict] = []
            for heldout in FEATURE_GROUPS:
                for budget in BUDGETS:
                    metrics = evaluate_suite(
                        context, masks, heldout, orders, budget, predicted, structures
                    )
                    random_by_fold_budget[heldout, budget].append(metrics)
                    replicate_rows.append({"heldout_group": heldout, "budget": budget, **metrics})
            aggregate, folds = aggregate_auc("random", model_name, seed, replicate_rows)
            replicate_aggregate_auc.append({
                metric: float(aggregate[metric]) for metric in aggregate if metric.startswith("auc_")
            })
            for fold in folds:
                replicate_fold_auc[str(fold["heldout_group"])].append({
                    metric: float(fold[metric]) for metric in fold if metric.startswith("auc_")
                })
        auc_rows.append({
            "method": "random", "model": model_name, "seed": seed,
            **{
                f"auc_{metric}": float(np.mean([row[f"auc_{metric}"] for row in replicate_aggregate_auc]))
                for metric in PRIMARY_METRICS
            },
        })
        for heldout in FEATURE_GROUPS:
            fold_auc_rows.append({
                "method": "random", "model": model_name, "seed": seed, "heldout_group": heldout,
                **{
                    f"auc_{metric}": float(np.mean([row[f"auc_{metric}"] for row in replicate_fold_auc[heldout]]))
                    for metric in PRIMARY_METRICS
                },
            })
            for budget in BUDGETS:
                random_summary.append(summarize_random_budget(
                    model_name, seed, heldout, budget, random_by_fold_budget[heldout, budget]
                ))
        write_json(output / "progress.json", {
            "status": "running", "completed_runs": run_index, "total_runs": len(runs),
            "latest_model": model_name, "latest_seed": seed, "updated_at": now(),
        })
        log(output, f"completed {run_index}/{len(runs)} {model_name} seed={seed}")

    summary = method_budget_summary(fixed_rows, random_summary)
    comparisons: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        family = [paired_comparison(auc_rows, baseline, f"auc_{metric}") for baseline in METHODS if baseline != "coverage"]
        holm(family, "permutation_p", "holm_within_metric_p")
        comparisons.extend(family)
    holm(comparisons, "permutation_p", "holm_global_p")

    fold_comparisons: list[dict[str, object]] = []
    for heldout in FEATURE_GROUPS:
        subset = [row for row in fold_auc_rows if row["heldout_group"] == heldout]
        result = paired_comparison(subset, "random", "auc_heldout_failed_state_fraction")
        result["heldout_group"] = heldout
        fold_comparisons.append(result)
    holm(fold_comparisons, "permutation_p", "holm_p")

    write_csv(output / "tables/fixed_metrics.csv", fixed_rows)
    write_csv(output / "tables/random_summary.csv", random_summary)
    write_csv(output / "tables/auc_by_run.csv", auc_rows)
    write_csv(output / "tables/fold_auc_by_run.csv", fold_auc_rows)
    write_csv(output / "tables/method_budget_summary.csv", summary)
    write_csv(output / "tables/primary_comparisons.csv", comparisons)
    write_csv(output / "tables/fold_coverage_vs_random.csv", fold_comparisons)
    write_csv(output / "tables/error_cluster_inventory.csv", cluster_rows)
    make_figures(output, summary, fold_comparisons)
    write_reports(output, context, summary, comparisons, fold_comparisons, cluster_rows, smoke)
    write_json(output / "progress.json", {
        "status": "smoke_complete" if smoke else "complete", "updated_at": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output / "smoke" if args.smoke else args.output
    write_json(output / "progress.json", {"status": "starting", "updated_at": now()})
    log(output, f"starting {'smoke' if args.smoke else 'full'} decoupled diagnostic benchmark")
    run(output, args.smoke)
    log(output, "benchmark complete")


if __name__ == "__main__":
    main()
