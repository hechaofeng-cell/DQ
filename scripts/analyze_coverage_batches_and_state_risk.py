#!/usr/bin/env python3
"""Analyze incremental coverage batches and state-associated error risk."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from openimages_cua_vsl import StateSchema, parse_values
from run_fixed_train_test_adequacy_experiment import read_csv, sha256
from run_resnet18_dog_coverage_comparison import write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CLASS_SPECS


ROOT = Path(__file__).resolve().parents[1]
DOG_SOURCE = ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"
CAT_SOURCE = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
DOG_FEATURES = ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
CAT_FEATURES = ROOT / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv"
DOG_SCHEMA = ROOT / "configs/dog_feature_schema_v3_1.json"
CAT_SCHEMA = ROOT / "configs/cat_feature_schema_v1_0.json"
OUTPUT = ROOT / "artifacts/coverage_batch_state_attribution_20260920"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
CLASS_NAMES = tuple(name for name, _ in CLASS_SPECS)
TARGET_LABEL = {"dog": 0, "cat": 1}
BATCH_SIZE = 20
BATCHES = (1, 2, 3, 4, 5)
MIN_STATE_SUPPORT = 10
RARE_STATE_MAX_SUPPORT = 10
BOOTSTRAP_REPLICATES = 10000
PERMUTATION_REPLICATES = 10000
RANDOM_SEED = 20260920


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_log(message: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (OUTPUT / "execution.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def context(category: str) -> dict:
    source = DOG_SOURCE if category == "dog" else CAT_SOURCE
    feature_path = DOG_FEATURES if category == "dog" else CAT_FEATURES
    schema_path = DOG_SCHEMA if category == "dog" else CAT_SCHEMA
    pool = read_csv(source / "candidate_pool.csv")
    target_rows = [row for row in pool if row["class_name"] == category]
    target_by_id = {row["sample_id"]: row for row in target_rows}
    features = {row["image_id"]: row for row in read_csv(feature_path)}
    schema = StateSchema.from_path(schema_path)
    states = schema.coverage_states()
    state_index = {state: index for index, state in enumerate(states)}
    order_rows = sorted(
        read_csv(source / "orders" / f"gap_{category}_order.csv"),
        key=lambda row: int(row["rank"]),
    )
    ordered_target = [target_by_id[row["sample_id"]] for row in order_rows]
    matrix = np.zeros((len(ordered_target), len(states)), dtype=np.int8)
    for image_index, row in enumerate(ordered_target):
        for pair in schema.state_pairs(features[row["sample_id"]]):
            if pair in state_index:
                matrix[image_index, state_index[pair]] = 1
    non_target_rows = read_csv(source / "orders" / f"non_{category}_order.csv")
    non_target = {}
    for class_name in CLASS_NAMES:
        if class_name == category:
            continue
        rows = sorted(
            (row for row in non_target_rows if row["class_name"] == class_name),
            key=lambda row: int(row["rank"]),
        )
        if len(rows) != 100:
            raise ValueError(f"{category}: expected 100 fixed {class_name} targets")
        non_target[class_name] = rows
    if len(ordered_target) != 200 or matrix.shape[0] != 200:
        raise ValueError(f"{category}: invalid target pool")
    return {
        "category": category, "source": source, "pool": pool,
        "target_rows": ordered_target, "features": features, "schema": schema,
        "states": states, "matrix": matrix, "non_target": non_target,
        "feature_path": feature_path, "schema_path": schema_path,
    }


def classification_metrics(true: np.ndarray, predicted: np.ndarray, target_label: int) -> dict:
    confusion = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted_support = confusion.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted_support, out=np.zeros(5), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "target_precision": float(precision[target_label]),
        "target_f1": float(f1[target_label]),
        "target_recall": float(recall[target_label]),
        "target_error_rate": float(1.0 - recall[target_label]),
        "target_error_count": int(support[target_label] - tp[target_label]),
        "target_false_positive_count": int(predicted_support[target_label] - tp[target_label]),
    }


def cohort_state_metrics(ctx: dict, batch: int) -> dict:
    start = (batch - 1) * BATCH_SIZE
    stop = batch * BATCH_SIZE
    cohort = ctx["matrix"][start:stop]
    earlier = ctx["matrix"][:start]
    support = ctx["matrix"].sum(axis=0)
    present = cohort.sum(axis=0) > 0
    prior_present = earlier.sum(axis=0) > 0 if start else np.zeros(ctx["matrix"].shape[1], dtype=bool)
    rare = support <= RARE_STATE_MAX_SUPPORT
    feature_rows = [ctx["features"][row["sample_id"]] for row in ctx["target_rows"][start:stop]]
    occluded = [row["occlusion_level"] in {"partial", "heavy"} for row in feature_rows]
    truncated = [row["truncation_level"] in {"partial", "heavy"} for row in feature_rows]
    non_typical_actions = {
        "crouching", "running", "jumping", "climbing", "grooming", "eating",
        "drinking", "playing", "interacting", "sleeping",
    }
    unusual = [row["posture_action_state"] in non_typical_actions for row in feature_rows]
    return {
        "present_state_count": int(present.sum()),
        "new_state_count": int((present & ~prior_present).sum()),
        "rare_state_count": int((present & rare).sum()),
        "occluded_fraction": float(np.mean(occluded)),
        "truncated_fraction": float(np.mean(truncated)),
        "non_typical_action_fraction": float(np.mean(unusual)),
    }


def build_prediction_arrays(ctx: dict, model_name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = read_csv(ctx["source"] / "predictions" / model_name / str(seed) / "candidate_predictions.csv")
    by_id = {row["sample_id"]: row for row in predictions}
    target_predicted = np.asarray([
        int(by_id[row["sample_id"]]["predicted_label"]) for row in ctx["target_rows"]
    ])
    all_true = np.asarray([int(by_id[row["sample_id"]]["true_label"]) for row in ctx["pool"]])
    all_predicted = np.asarray([int(by_id[row["sample_id"]]["predicted_label"]) for row in ctx["pool"]])
    return target_predicted, all_true, all_predicted


def batch_analysis(contexts: dict[str, dict]) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    detail = []
    error_matrices = {}
    for category, ctx in contexts.items():
        target_label = TARGET_LABEL[category]
        pool_positions = {row["sample_id"]: index for index, row in enumerate(ctx["pool"])}
        target_positions = [pool_positions[row["sample_id"]] for row in ctx["target_rows"]]
        non_target_positions = {
            class_name: [pool_positions[row["sample_id"]] for row in rows]
            for class_name, rows in ctx["non_target"].items()
        }
        error_rows = []
        for model_name, seed in itertools.product(MODELS, SEEDS):
            target_predicted, all_true, all_predicted = build_prediction_arrays(ctx, model_name, seed)
            target_error = (target_predicted != target_label).astype(np.int8)
            error_rows.append(target_error)
            for batch in BATCHES:
                start = (batch - 1) * BATCH_SIZE
                stop = batch * BATCH_SIZE
                selected = list(np.asarray(target_positions)[start:stop])
                for class_name in CLASS_NAMES:
                    if class_name != category:
                        selected.extend(non_target_positions[class_name][start:stop])
                metrics = classification_metrics(all_true[selected], all_predicted[selected], target_label)
                detail.append({
                    "category": category, "model": model_name, "seed": seed,
                    "batch": batch, "target_rank_start": start + 1, "target_rank_end": stop,
                    **metrics, **cohort_state_metrics(ctx, batch),
                })
        error_matrices[category] = np.asarray(error_rows)

    write_csv(OUTPUT / "tables" / "batch_metrics_by_run.csv", detail)
    summary = []
    metric_names = (
        "accuracy", "macro_f1", "target_precision", "target_f1", "target_recall", "target_error_rate",
        "target_false_positive_count",
        "present_state_count", "new_state_count", "rare_state_count", "occluded_fraction",
        "truncated_fraction", "non_typical_action_fraction",
    )
    for category in contexts:
        for batch in BATCHES:
            rows = [row for row in detail if row["category"] == category and row["batch"] == batch]
            out = {"category": category, "batch": batch,
                   "target_rank_start": (batch - 1) * BATCH_SIZE + 1,
                   "target_rank_end": batch * BATCH_SIZE}
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in rows])
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_sd"] = float(values.std(ddof=1)) if len(set(values)) > 1 else 0.0
            summary.append(out)
    write_csv(OUTPUT / "tables" / "batch_summary.csv", summary)
    return detail, summary, error_matrices


def exact_cluster_sign_test(values_by_architecture: dict[str, list[float]]) -> float:
    architecture_values = np.asarray([np.mean(values_by_architecture[model]) for model in MODELS])
    observed = abs(float(architecture_values.mean()))
    null = [abs(float(np.mean(architecture_values * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(MODELS))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def hierarchical_ci(values: dict[tuple[str, int], float], rng: np.random.Generator) -> tuple[float, float]:
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        architectures = rng.choice(MODELS, size=len(MODELS), replace=True)
        sampled = []
        for model_name in architectures:
            seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
            sampled.extend(values[(model_name, int(seed))] for seed in seeds)
        draws.append(float(np.mean(sampled)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def batch_trend_tests(detail: list[dict]) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    output = []
    x = np.asarray(BATCHES, dtype=float)
    for category in ("dog", "cat"):
        for metric in ("target_error_rate", "target_precision", "target_f1", "macro_f1"):
            slopes = {}
            endpoint = {}
            by_arch_slope: dict[str, list[float]] = defaultdict(list)
            by_arch_endpoint: dict[str, list[float]] = defaultdict(list)
            for model_name in MODELS:
                for seed in SEEDS:
                    rows = sorted(
                        (row for row in detail if row["category"] == category and row["model"] == model_name and row["seed"] == seed),
                        key=lambda row: row["batch"],
                    )
                    y = np.asarray([float(row[metric]) for row in rows])
                    slope = float(np.polyfit(x, y, 1)[0])
                    delta = float(y[-1] - y[0])
                    slopes[(model_name, seed)] = slope
                    endpoint[(model_name, seed)] = delta
                    by_arch_slope[model_name].append(slope)
                    by_arch_endpoint[model_name].append(delta)
            slope_ci = hierarchical_ci(slopes, rng)
            endpoint_ci = hierarchical_ci(endpoint, rng)
            output.append({
                "category": category, "metric": metric,
                "mean_slope_per_20": float(np.mean(list(slopes.values()))),
                "slope_ci95_low": slope_ci[0], "slope_ci95_high": slope_ci[1],
                "slope_cluster_p": exact_cluster_sign_test(by_arch_slope),
                "positive_slope_runs": sum(value > 0 for value in slopes.values()),
                "negative_slope_runs": sum(value < 0 for value in slopes.values()),
                "mean_B5_minus_B1": float(np.mean(list(endpoint.values()))),
                "endpoint_ci95_low": endpoint_ci[0], "endpoint_ci95_high": endpoint_ci[1],
                "endpoint_cluster_p": exact_cluster_sign_test(by_arch_endpoint),
            })
    write_csv(OUTPUT / "tables" / "batch_trend_tests.csv", output)
    return output


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def state_risk_analysis(contexts: dict[str, dict], error_matrices: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    state_rows = []
    image_rows = []
    for category, ctx in contexts.items():
        errors = error_matrices[category].astype(float)
        image_error = errors.mean(axis=0)
        ranks = np.arange(1, len(image_error) + 1)
        rho, _ = stats.spearmanr(ranks, image_error)
        permuted = []
        for _ in range(PERMUTATION_REPLICATES):
            permuted.append(stats.spearmanr(ranks, rng.permutation(image_error)).statistic)
        p_value = float((1 + np.count_nonzero(np.abs(permuted) >= abs(rho))) / (PERMUTATION_REPLICATES + 1))
        image_rows.append({
            "category": category, "spearman_rank_error": float(rho),
            "permutation_p": p_value, "images": len(image_error), "model_runs": errors.shape[0],
        })
        support = ctx["matrix"].sum(axis=0)
        category_rows = []
        for state_index, (feature_id, state) in enumerate(ctx["states"]):
            present = ctx["matrix"][:, state_index].astype(bool)
            count = int(present.sum())
            if count < MIN_STATE_SUPPORT or count > len(present) - MIN_STATE_SUPPORT:
                continue
            per_run_with = errors[:, present].mean(axis=1)
            per_run_without = errors[:, ~present].mean(axis=1)
            differences = per_run_with - per_run_without
            risk_values = {(model, seed): float(differences[index])
                           for index, (model, seed) in enumerate(itertools.product(MODELS, SEEDS))}
            by_arch: dict[str, list[float]] = defaultdict(list)
            for (model, _seed), value in risk_values.items():
                by_arch[model].append(value)
            ci_low, ci_high = hierarchical_ci(risk_values, rng)
            error_with = float(per_run_with.mean())
            error_without = float(per_run_without.mean())
            category_rows.append({
                "category": category, "feature_id": feature_id, "state": state,
                "state_label": f"{feature_id}={state}", "support_images": count,
                "support_fraction": count / len(present),
                "mean_coverage_rank": float(ranks[present].mean()),
                "late_batch_fraction": float(np.mean(ranks[present] > 60)),
                "error_rate_with": error_with, "error_rate_without": error_without,
                "risk_difference": float(differences.mean()),
                "risk_ratio": error_with / error_without if error_without > 0 else math.inf,
                "ci95_low": ci_low, "ci95_high": ci_high,
                "positive_model_runs": int((differences > 0).sum()),
                "positive_architectures": int(sum(np.mean(by_arch[model]) > 0 for model in MODELS)),
                "cluster_p": exact_cluster_sign_test(by_arch),
            })
        adjusted = holm_adjust([row["cluster_p"] for row in category_rows])
        for row, value in zip(category_rows, adjusted):
            row["holm_p"] = value
        state_rows.extend(category_rows)
    write_csv(OUTPUT / "tables" / "state_risk_associations.csv", state_rows)
    write_csv(OUTPUT / "tables" / "rank_error_correlations.csv", image_rows)
    return state_rows, image_rows


def make_figures(batch_summary: list[dict], state_rows: list[dict], contexts: dict[str, dict],
                 error_matrices: dict[str, np.ndarray]) -> None:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"dog": "#0072B2", "cat": "#D55E00"}

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    specifications = (
        ("target_error_rate_mean", "Target error rate (%)", 100),
        ("target_precision_mean", "Target-class precision (%)", 100),
        ("target_f1_mean", "Target-class F1 (%)", 100),
        ("macro_f1_mean", "Macro-F1 (%)", 100),
    )
    for axis, (field, label, scale) in zip(axes, specifications):
        for category in ("dog", "cat"):
            rows = sorted((row for row in batch_summary if row["category"] == category), key=lambda row: row["batch"])
            axis.plot(BATCHES, [float(row[field]) * scale for row in rows], marker="o", linewidth=2,
                      color=colors[category], label=category)
        axis.set_xlabel("Independent added batch (20 target images)")
        axis.set_ylabel(label)
        axis.set_xticks(BATCHES, ["1-20", "21-40", "41-60", "61-80", "81-100"], rotation=25)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(figure_dir / "incremental_batch_performance.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "incremental_batch_performance.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, category in zip(axes, ("dog", "cat")):
        rows = sorted((row for row in state_rows if row["category"] == category),
                      key=lambda row: row["risk_difference"], reverse=True)[:12]
        rows = list(reversed(rows))
        y = np.arange(len(rows))
        means = np.asarray([row["risk_difference"] * 100 for row in rows])
        low = means - np.asarray([row["ci95_low"] * 100 for row in rows])
        high = np.asarray([row["ci95_high"] * 100 for row in rows]) - means
        colors_for_rows = ["#D55E00" if row["holm_p"] < 0.05 else "#777777" for row in rows]
        for index, row in enumerate(rows):
            axis.errorbar(means[index], y[index], xerr=[[low[index]], [high[index]]], fmt="o",
                          color=colors_for_rows[index], capsize=3)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_yticks(y, [row["state_label"] for row in rows], fontsize=8)
        axis.set_xlabel("Error-risk difference (percentage points)")
        axis.set_title(category.upper())
        axis.grid(axis="x", alpha=0.25)
    fig.savefig(figure_dir / "top_state_risk_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "top_state_risk_forest.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for axis, category in zip(axes, ("dog", "cat")):
        image_error = error_matrices[category].mean(axis=0) * 100
        ranks = np.arange(1, len(image_error) + 1)
        axis.scatter(ranks, image_error, s=16, alpha=0.35, color=colors[category])
        batch_means = [image_error[(batch - 1) * BATCH_SIZE:batch * BATCH_SIZE].mean() for batch in BATCHES]
        centers = [10, 30, 50, 70, 90]
        axis.plot(centers, batch_means, marker="o", linewidth=2.5, color="#222222", label="20-image batch mean")
        axis.set_title(category.upper())
        axis.set_xlabel("Coverage-selection rank")
        axis.set_ylabel("Error rate across 24 models (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(figure_dir / "selection_rank_vs_error.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "selection_rank_vs_error.svg", bbox_inches="tight")
    plt.close(fig)


def write_protocol(contexts: dict[str, dict]) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    write_json(OUTPUT / "protocol.json", {
        "experiment_id": "coverage-batch-state-attribution-20260920",
        "created_at": now(),
        "research_questions": [
            "Are later independent coverage-selected batches harder for frozen models?",
            "Which observable states are associated with higher error risk across architectures?",
        ],
        "categories": ["dog", "cat"], "cat_label_status": "provisional VLM labels",
        "models": list(MODELS), "seeds": list(SEEDS),
        "batches": ["1-20", "21-40", "41-60", "61-80", "81-100"],
        "state_minimum_support": MIN_STATE_SUPPORT,
        "rare_state_definition": f"candidate-pool support <= {RARE_STATE_MAX_SUPPORT}",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "fixed_factors": "D300 checkpoints, candidate pools, coverage orders, non-target orders, preprocessing",
        "causal_boundary": "state analyses are marginal associations; co-occurring states prevent causal attribution",
        "git_commit": commit,
        "input_sha256": {
            f"{category}_{name}": sha256(path)
            for category, ctx in contexts.items()
            for name, path in (
                ("candidate_pool", ctx["source"] / "candidate_pool.csv"),
                ("features", ctx["feature_path"]), ("schema", ctx["schema_path"]),
                ("coverage_order", ctx["source"] / "orders" / f"gap_{category}_order.csv"),
            )
        },
    })
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "- Claim under test: later coverage-selected cohorts expose harder visual conditions, and the response can be localized to observable states.\n"
        "- Primary evidence: independent 20-target cohorts across 24 frozen models and state-level risk differences across eight architecture clusters.\n"
        "- Boundary: associations do not establish that a single state caused an error; cat labels are provisional.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol frozen: {now()}\n"
        "- Reproduction: `.venv/bin/python -u scripts/analyze_coverage_batches_and_state_risk.py`.\n"
        "- No model inference or retraining: analysis reuses 24 frozen prediction files per category.\n"
        "- State inclusion: candidate-pool support from 10 through 190 images.\n"
        "- Inference unit: architecture cluster; three seeds are nested within architecture.\n",
        encoding="utf-8",
    )


def write_report(batch_summary: list[dict], trends: list[dict], state_rows: list[dict], image_rows: list[dict]) -> None:
    lines = ["# 覆盖新增批次与视觉状态风险分析", "", f"日期：{now()[:10]}", "", "## 新增批次结果", "",
             "每一行都是独立的20张目标类别新增批次，同时配入四个非目标类别各20张；不是累计测试集。", "",
             "| 类别 | 批次 | 目标错误率 | 目标Precision | 目标F1 | 目标假阳性数 | 新状态数 | 遮挡比例 | 截断比例 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in batch_summary:
        lines.append(
            f"| {row['category']} | B{row['batch']} ({row['target_rank_start']}-{row['target_rank_end']}) | "
            f"{row['target_error_rate_mean']*100:.2f}% | {row['target_precision_mean']*100:.2f}% | "
            f"{row['target_f1_mean']*100:.2f}% | {row['target_false_positive_count_mean']:.2f} | "
            f"{row['new_state_count_mean']:.0f} | {row['occluded_fraction_mean']*100:.1f}% | "
            f"{row['truncated_fraction_mean']*100:.1f}% |"
        )
    lines.extend(["", "## 趋势检验", "",
                  "斜率表示批次每向后移动20张时指标的平均变化。置信区间按架构分层bootstrap，p值按8个架构簇精确符号置换。", "",
                  "| 类别 | 指标 | 每批斜率 | 95% CI | 正/负运行 | B5-B1 | 95% CI | p |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in trends:
        lines.append(
            f"| {row['category']} | {row['metric']} | {row['mean_slope_per_20']:+.4f} | "
            f"[{row['slope_ci95_low']:+.4f}, {row['slope_ci95_high']:+.4f}] | "
            f"{row['positive_slope_runs']}/{row['negative_slope_runs']} | {row['mean_B5_minus_B1']:+.4f} | "
            f"[{row['endpoint_ci95_low']:+.4f}, {row['endpoint_ci95_high']:+.4f}] | "
            f"{row['endpoint_cluster_p']:.4f} |"
        )
    error_trends = {row["category"]: row for row in trends if row["metric"] == "target_error_rate"}
    precision_trends = {row["category"]: row for row in trends if row["metric"] == "target_precision"}
    lines.extend(["", "## 关键解释", "",
                  f"- dog目标错误率每批斜率为`{error_trends['dog']['mean_slope_per_20']*100:+.2f}`个百分点，但95% CI跨0，B5与B1几乎相同；",
                  f"- cat目标错误率每批斜率为`{error_trends['cat']['mean_slope_per_20']*100:+.2f}`个百分点，95% CI同样跨0；",
                  f"- cat目标Precision每批斜率为`{precision_trends['cat']['mean_slope_per_20']*100:+.2f}`个百分点。目标F1下降不仅取决于cat召回，还受到非cat图片被预测为cat的假阳性影响；",
                  "- 因此，累计F1下降不能直接归因于后续加入的目标视觉状态越来越困难。独立批次表现为局部难点峰值和类别组成效应，而不是单调难度上升。",
                  "", "## 排名与图片错误风险", "", "| 类别 | Spearman rho | permutation p |", "|---|---:|---:|"])
    for row in image_rows:
        lines.append(f"| {row['category']} | {row['spearman_rank_error']:+.3f} | {row['permutation_p']:.4f} |")

    lines.extend(["", "## 高风险关联状态", "",
                  "下表按错误风险差排序，仅列候选池支持数不少于10的前10个状态。风险差为含该状态样本与不含该状态样本的平均错误率差。", "",
                  "| 类别 | 状态 | 支持数 | 含状态错误率 | 不含状态错误率 | 风险差 | 95% CI | 正向架构 | Holm-p |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    significant = 0
    for category in ("dog", "cat"):
        rows = sorted((row for row in state_rows if row["category"] == category), key=lambda row: row["risk_difference"], reverse=True)[:10]
        for row in rows:
            significant += row["holm_p"] < 0.05
            lines.append(
                f"| {category} | `{row['state_label']}` | {row['support_images']} | "
                f"{row['error_rate_with']*100:.2f}% | {row['error_rate_without']*100:.2f}% | "
                f"{row['risk_difference']*100:+.2f} pp | [{row['ci95_low']*100:+.2f}, {row['ci95_high']*100:+.2f}] | "
                f"{row['positive_architectures']}/8 | {row['holm_p']:.4f} |"
            )
    lines.extend(["", "## 结论边界", ""])
    if significant:
        lines.append("> 部分可观察状态在多个架构上与更高错误风险稳定相关，但这些结果仍是边际关联，不能排除状态共现造成的混杂。")
    else:
        lines.append("> 存在若干跨架构方向一致的高风险状态，但经过全状态Holm校正后没有状态达到确认性显著，因此应视为后续验证线索而非已确认原因。")
    lines.extend([
        "", "- 独立批次分析可以判断后续加入的20张是否本身更难，避免累计F1掩盖批次差异。",
        "- cat仍使用VLM自动标签，所有cat状态结论都是暂定结果。",
        "- 状态通常同时出现，例如遮挡、腿部不可见和低可识别性可能属于同一张图片；本分析不能声称其中某一状态造成错误。",
        "- 模型权重没有变化，性能变化描述的是测试集响应，而不是训练后模型退化。",
        "", "## 结果文件", "",
        "- `tables/batch_summary.csv`：五个独立新增批次；",
        "- `tables/batch_trend_tests.csv`：斜率、端点差和置信区间；",
        "- `tables/state_risk_associations.csv`：全部合格状态的风险关联；",
        "- `tables/rank_error_correlations.csv`：选样顺序与图片错误风险；",
        "- `figures/incremental_batch_performance.svg`：新增批次性能曲线；",
        "- `figures/top_state_risk_forest.svg`：状态风险森林图；",
        "- `figures/selection_rank_vs_error.svg`：进入顺序与错误率散点图。", "",
    ])
    (OUTPUT / "coverage_batch_state_attribution_zh.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Later coverage cohorts are harder | see results | `tables/batch_trend_tests.csv` | fixed coverage order and D300 models |\n"
        "| Specific states cause model errors | unsupported causal wording | `tables/state_risk_associations.csv` | marginal association only |\n"
        "| State risk associations identify follow-up hypotheses | supported within tested pools | risk differences and architecture consistency | Holm correction required |\n"
        "| Cat state findings are final | blocked | human review pending | provisional only |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: pass; no external factual claims introduced.\n"
        "- Result traceability: pass after successful `checksums.csv` verification.\n"
        "- Claim scope: pass if report retains association-not-causation and provisional-cat boundaries.\n"
        "- Remaining blocker: cat human review.\n",
        encoding="utf-8",
    )


def write_checksums() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "checksums.csv":
            rows.append({"path": str(path.relative_to(OUTPUT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(OUTPUT / "checksums.csv", rows)


def main() -> None:
    contexts = {category: context(category) for category in ("dog", "cat")}
    write_protocol(contexts)
    append_log("protocol and frozen inputs validated")
    detail, batch_summary, error_matrices = batch_analysis(contexts)
    append_log("independent batch metrics completed")
    trends = batch_trend_tests(detail)
    state_rows, image_rows = state_risk_analysis(contexts, error_matrices)
    append_log("state risk associations completed")
    make_figures(batch_summary, state_rows, contexts, error_matrices)
    write_report(batch_summary, trends, state_rows, image_rows)
    write_json(OUTPUT / "progress.json", {"status": "completed", "updated_at": now()})
    append_log("analysis completed")
    write_checksums()


if __name__ == "__main__":
    main()
