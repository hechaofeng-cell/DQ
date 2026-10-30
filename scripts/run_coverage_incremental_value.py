#!/usr/bin/env python3
"""Evaluate the incremental diagnostic value of semantic state coverage.

This script reuses frozen candidate predictions and features. It performs:
1. equal-error-count rarefaction for confusion pairs and error clusters;
2. fixed-budget 75% strong-baseline + 25% Coverage/random hybrids;
3. three predeclared visual-state interaction analyses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import font_manager
from scipy.special import gammaln

from run_cross_state_decoupled_diagnostics import error_structures, load_cached_run, load_order
from run_multiclass_object_selection_benchmark import (
    BUDGETS,
    CLASSES,
    LABELS,
    MODELS,
    RANDOM_SEED,
    SEEDS,
    build_context,
    evaluate_suite as evaluate_state_suite,
    random_orders,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928"
POOL = ROOT / "artifacts/multiclass_object_pool_v1_20260928"
DEFAULT_OUTPUT = ROOT / "artifacts/coverage_incremental_value_20260929"
FONT_PATH = ROOT / "scripts/assets/NotoSansSC-Regular.ttf"
FIXED_METHODS = ("coverage", "deepgini", "dsa", "kcenter")
EQUAL_ERROR_BASELINES = ("random", "deepgini", "dsa", "kcenter")
HYBRID_BASELINES = ("deepgini", "kcenter")
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 5000
SUPPLEMENT_FRACTION = 0.25

if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False


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


def load_orders(context: dict, model: str, seed: int) -> dict[str, dict[str, list[int]]]:
    independent = BASE / "tables/model_independent_orders.csv"
    return {
        "coverage": load_order(independent, context, "coverage"),
        "kcenter": load_order(independent, context, "kcenter"),
        "deepgini": load_order(BASE / "orders" / model / str(seed) / "deepgini.csv", context),
        "dsa": load_order(BASE / "orders" / model / str(seed) / "dsa.csv", context),
    }


def selected_global(context: dict, orders: dict[str, list[int]], budget: int) -> list[int]:
    output: list[int] = []
    for class_name in CLASSES:
        global_indices = context["by_class"][class_name]["global_indices"]
        output.extend(int(value) for value in global_indices[np.asarray(orders[class_name][:budget])])
    return output


def error_labels(
    context: dict,
    chosen_global: list[int],
    predicted: np.ndarray,
    structures: dict[str, dict[str, object]],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    confusion: list[tuple[str, int]] = []
    clusters: list[tuple[str, int]] = []
    for index in chosen_global:
        class_name = context["rows"][index]["class_name"]
        if int(predicted[index]) == LABELS[class_name]:
            continue
        confusion.append((class_name, int(predicted[index])))
        clusters.append((class_name, int(structures[class_name]["cluster_by_global"][index])))
    return confusion, clusters


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def expected_unique_fraction(labels: list[tuple[str, int]], k: int, universe: int) -> float:
    if k <= 0 or not labels or universe <= 0:
        return 0.0
    k = min(k, len(labels))
    counts = Counter(labels)
    total = len(labels)
    expected = 0.0
    denominator = log_comb(total, k)
    for count in counts.values():
        if total - count < k:
            absent = 0.0
        else:
            absent = math.exp(log_comb(total - count, k) - denominator)
        expected += 1.0 - absent
    return expected / universe


def equal_error_pair(
    coverage_labels: tuple[list[tuple[str, int]], list[tuple[str, int]]],
    baseline_labels: tuple[list[tuple[str, int]], list[tuple[str, int]]],
    confusion_universe: int,
    cluster_universe: int,
) -> dict[str, float | int]:
    coverage_confusion, coverage_clusters = coverage_labels
    baseline_confusion, baseline_clusters = baseline_labels
    k = min(len(coverage_confusion), len(baseline_confusion))
    return {
        "matched_errors": k,
        "feasible": int(k > 0),
        "coverage_confusion": expected_unique_fraction(coverage_confusion, k, confusion_universe),
        "baseline_confusion": expected_unique_fraction(baseline_confusion, k, confusion_universe),
        "coverage_clusters": expected_unique_fraction(coverage_clusters, k, cluster_universe),
        "baseline_clusters": expected_unique_fraction(baseline_clusters, k, cluster_universe),
    }


def independent_metrics(
    context: dict,
    orders: dict[str, list[int]],
    budget: int,
    predicted: np.ndarray,
    structures: dict[str, dict[str, object]],
) -> dict[str, float | int | set[tuple[str, int]]]:
    confusion_found: set[tuple[str, int]] = set()
    cluster_found: set[tuple[str, int]] = set()
    class_confusion: list[float] = []
    class_clusters: list[float] = []
    error_rates: list[float] = []
    for class_name in CLASSES:
        global_indices = context["by_class"][class_name]["global_indices"]
        chosen = global_indices[np.asarray(orders[class_name][:budget])]
        wrong = [int(index) for index in chosen if int(predicted[index]) != LABELS[class_name]]
        predictions = {(class_name, int(predicted[index])) for index in wrong}
        clusters = {
            (class_name, int(structures[class_name]["cluster_by_global"][index]))
            for index in wrong
        }
        confusion_found.update(predictions)
        cluster_found.update(clusters)
        class_confusion.append(len(predictions) / len(structures[class_name]["possible_predictions"]))
        class_clusters.append(len(clusters) / int(structures[class_name]["cluster_count"]))
        error_rates.append(len(wrong) / budget)
    state = evaluate_state_suite(context, orders, budget, predicted)
    return {
        "confusion_pair_coverage": float(np.mean(class_confusion)),
        "error_cluster_coverage": float(np.mean(class_clusters)),
        "macro_error_rate": float(np.mean(error_rates)),
        "macro_f1": float(state["macro_f1"]),
        "macro_frequency_coverage": float(state["macro_frequency_coverage"]),
        "macro_failed_state_fraction": float(state["macro_failed_state_fraction"]),
        "confusion_set": confusion_found,
        "cluster_set": cluster_found,
    }


def hybrid_orders(
    base_orders: dict[str, list[int]],
    supplement_orders: dict[str, list[int]],
    budget: int,
) -> dict[str, list[int]]:
    supplement_count = int(round(budget * SUPPLEMENT_FRACTION))
    base_count = budget - supplement_count
    output: dict[str, list[int]] = {}
    for class_name in CLASSES:
        chosen = list(base_orders[class_name][:base_count])
        seen = set(chosen)
        for index in supplement_orders[class_name]:
            if index not in seen:
                chosen.append(index)
                seen.add(index)
            if len(chosen) == budget:
                break
        if len(chosen) != budget:
            raise ValueError(f"hybrid underfilled for {class_name}/{budget}")
        output[class_name] = chosen
    return output


def normalized_auc(rows: list[dict], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["budget"]))
    x = np.asarray([0, *[int(row["budget"]) for row in ordered]], dtype=float)
    y = np.asarray([0, *[float(row[metric]) for row in ordered]], dtype=float)
    return float(np.trapezoid(y, x) / BUDGETS[-1])


def paired_stats(
    values: dict[str, list[float]], label: str, metric: str,
) -> dict[str, object]:
    arrays = {model: np.asarray(group, dtype=float) for model, group in values.items()}
    flat = np.concatenate(list(arrays.values()))
    architecture_means = np.asarray([array.mean() for array in arrays.values()])
    observed = abs(float(architecture_means.mean()))
    null = [
        abs(float(np.mean(architecture_means * np.asarray(signs))))
        for signs in itertools.product((-1, 1), repeat=len(arrays))
    ]
    rng = np.random.default_rng(RANDOM_SEED + sum(ord(char) for char in label + metric))
    bootstrap = []
    model_names = list(arrays)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_models = rng.choice(model_names, size=len(model_names), replace=True)
        sampled: list[float] = []
        for model in sampled_models:
            sampled.extend(rng.choice(arrays[model], size=len(arrays[model]), replace=True))
        bootstrap.append(float(np.mean(sampled)))
    return {
        "comparison": label,
        "metric": metric,
        "mean_delta_auc": float(flat.mean()),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "positive_runs": int((flat > 0).sum()),
        "negative_runs": int((flat < 0).sum()),
        "zero_runs": int((flat == 0).sum()),
        "paired_runs": int(len(flat)),
        "permutation_p": float(np.mean(np.asarray(null) >= observed - 1e-15)),
    }


def run_equal_error(
    output: Path,
    context: dict,
    random_sets: list[dict[str, list[int]]],
) -> tuple[list[dict], list[dict]]:
    metric_rows: list[dict] = []
    auc_rows: list[dict] = []
    comparison_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model, seed in itertools.product(MODELS, SEEDS):
        features, predicted = load_cached_run(context, model, seed)
        structures = error_structures(context, features, predicted, model, seed)
        orders = load_orders(context, model, seed)
        confusion_universe = sum(len(structures[name]["possible_predictions"]) for name in CLASSES)
        cluster_universe = sum(int(structures[name]["cluster_count"]) for name in CLASSES)
        run_curves: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for budget in BUDGETS:
            coverage_chosen = selected_global(context, orders["coverage"], budget)
            coverage_labels = error_labels(context, coverage_chosen, predicted, structures)
            for baseline in EQUAL_ERROR_BASELINES:
                if baseline == "random":
                    pairs = []
                    for random_order in random_sets:
                        labels = error_labels(
                            context, selected_global(context, random_order, budget), predicted, structures
                        )
                        pairs.append(equal_error_pair(
                            coverage_labels, labels, confusion_universe, cluster_universe
                        ))
                    row = {
                        "baseline": baseline, "model": model, "seed": seed, "budget": budget,
                        "matched_errors_mean": float(np.mean([p["matched_errors"] for p in pairs])),
                        "matched_errors_min": int(min(p["matched_errors"] for p in pairs)),
                        "matched_errors_max": int(max(p["matched_errors"] for p in pairs)),
                        "feasible_fraction": float(np.mean([p["feasible"] for p in pairs])),
                    }
                    for key in (
                        "coverage_confusion", "baseline_confusion",
                        "coverage_clusters", "baseline_clusters",
                    ):
                        row[key] = float(np.mean([p[key] for p in pairs]))
                else:
                    labels = error_labels(
                        context, selected_global(context, orders[baseline], budget), predicted, structures
                    )
                    pair = equal_error_pair(
                        coverage_labels, labels, confusion_universe, cluster_universe
                    )
                    row = {
                        "baseline": baseline, "model": model, "seed": seed, "budget": budget,
                        "matched_errors_mean": pair["matched_errors"],
                        "matched_errors_min": pair["matched_errors"],
                        "matched_errors_max": pair["matched_errors"],
                        "feasible_fraction": pair["feasible"],
                        **{key: pair[key] for key in (
                            "coverage_confusion", "baseline_confusion",
                            "coverage_clusters", "baseline_clusters",
                        )},
                    }
                row["delta_confusion"] = float(row["coverage_confusion"]) - float(row["baseline_confusion"])
                row["delta_clusters"] = float(row["coverage_clusters"]) - float(row["baseline_clusters"])
                metric_rows.append(row)
                run_curves[baseline, "confusion"].append(row)
                run_curves[baseline, "clusters"].append(row)
        for baseline in EQUAL_ERROR_BASELINES:
            auc_row: dict[str, object] = {"baseline": baseline, "model": model, "seed": seed}
            for metric in ("confusion", "clusters"):
                value = normalized_auc(run_curves[baseline, metric], f"delta_{metric}")
                auc_row[f"auc_delta_{metric}"] = value
                comparison_values[baseline, metric][model].append(value)
            auc_rows.append(auc_row)
        log(output, f"equal-error completed {model} seed={seed}")
    comparisons = [
        paired_stats(comparison_values[baseline, metric], f"coverage_vs_{baseline}", f"equal_error_{metric}")
        for baseline in EQUAL_ERROR_BASELINES for metric in ("confusion", "clusters")
    ]
    write_csv(output / "tables/equal_error_metrics.csv", metric_rows)
    write_csv(output / "tables/equal_error_auc_by_run.csv", auc_rows)
    write_csv(output / "tables/equal_error_comparisons.csv", comparisons)
    return metric_rows, comparisons


def clean_metrics(metrics: dict) -> dict[str, float | int]:
    return {key: value for key, value in metrics.items() if not isinstance(value, set)}


def run_hybrids(
    output: Path,
    context: dict,
    random_sets: list[dict[str, list[int]]],
) -> tuple[list[dict], list[dict], list[dict]]:
    fixed_rows: list[dict] = []
    random_rows: list[dict] = []
    auc_rows: list[dict] = []
    endpoint_rows: list[dict] = []
    comparison_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = (
        "confusion_pair_coverage", "error_cluster_coverage", "macro_error_rate",
        "macro_f1", "macro_frequency_coverage", "macro_failed_state_fraction",
    )
    for model, seed in itertools.product(MODELS, SEEDS):
        features, predicted = load_cached_run(context, model, seed)
        structures = error_structures(context, features, predicted, model, seed)
        orders = load_orders(context, model, seed)
        for base in HYBRID_BASELINES:
            fixed_curves: dict[str, list[dict]] = defaultdict(list)
            random_curves: list[dict[str, list[dict]]] = [defaultdict(list) for _ in random_sets]
            for budget in BUDGETS:
                pure_metrics = independent_metrics(context, orders[base], budget, predicted, structures)
                coverage_order = hybrid_orders(orders[base], orders["coverage"], budget)
                coverage_metrics = independent_metrics(context, coverage_order, budget, predicted, structures)
                for variant, values in (("pure", pure_metrics), ("coverage25", coverage_metrics)):
                    row = {
                        "base": base, "variant": variant, "model": model, "seed": seed,
                        "budget": budget, **clean_metrics(values),
                    }
                    fixed_rows.append(row)
                    fixed_curves[variant].append(row)
                per_replicate = []
                for replicate, random_order in enumerate(random_sets):
                    order = hybrid_orders(orders[base], random_order, budget)
                    values = independent_metrics(context, order, budget, predicted, structures)
                    clean = clean_metrics(values)
                    per_replicate.append(clean)
                    random_curves[replicate]["random25"].append({"budget": budget, **clean})
                summary: dict[str, object] = {
                    "base": base, "variant": "random25", "model": model, "seed": seed,
                    "budget": budget, "replicates": len(per_replicate),
                }
                for metric in metrics:
                    values = np.asarray([float(row[metric]) for row in per_replicate])
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"{metric}_p2_5"] = float(np.percentile(values, 2.5))
                    summary[f"{metric}_p97_5"] = float(np.percentile(values, 97.5))
                random_rows.append(summary)

                if budget == BUDGETS[-1]:
                    pure_conf = pure_metrics["confusion_set"]
                    pure_clust = pure_metrics["cluster_set"]
                    cov_conf = coverage_metrics["confusion_set"]
                    cov_clust = coverage_metrics["cluster_set"]
                    random_new_conf, random_lost_conf = [], []
                    random_new_clust, random_lost_clust = [], []
                    for random_order in random_sets:
                        values = independent_metrics(
                            context, hybrid_orders(orders[base], random_order, budget),
                            budget, predicted, structures,
                        )
                        random_new_conf.append(len(values["confusion_set"] - pure_conf))
                        random_lost_conf.append(len(pure_conf - values["confusion_set"]))
                        random_new_clust.append(len(values["cluster_set"] - pure_clust))
                        random_lost_clust.append(len(pure_clust - values["cluster_set"]))
                    endpoint_rows.append({
                        "base": base, "model": model, "seed": seed, "budget": budget,
                        "coverage_new_confusion": len(cov_conf - pure_conf),
                        "coverage_lost_confusion": len(pure_conf - cov_conf),
                        "coverage_net_confusion": len(cov_conf) - len(pure_conf),
                        "random_new_confusion_mean": float(np.mean(random_new_conf)),
                        "random_lost_confusion_mean": float(np.mean(random_lost_conf)),
                        "random_net_confusion_mean": float(np.mean(random_new_conf) - np.mean(random_lost_conf)),
                        "coverage_new_clusters": len(cov_clust - pure_clust),
                        "coverage_lost_clusters": len(pure_clust - cov_clust),
                        "coverage_net_clusters": len(cov_clust) - len(pure_clust),
                        "random_new_clusters_mean": float(np.mean(random_new_clust)),
                        "random_lost_clusters_mean": float(np.mean(random_lost_clust)),
                        "random_net_clusters_mean": float(np.mean(random_new_clust) - np.mean(random_lost_clust)),
                    })
            random_auc = {
                metric: float(np.mean([
                    normalized_auc(curves["random25"], metric) for curves in random_curves
                ]))
                for metric in metrics
            }
            for variant in ("pure", "coverage25"):
                row: dict[str, object] = {"base": base, "variant": variant, "model": model, "seed": seed}
                for metric in metrics:
                    row[f"auc_{metric}"] = normalized_auc(fixed_curves[variant], metric)
                auc_rows.append(row)
            auc_rows.append({
                "base": base, "variant": "random25", "model": model, "seed": seed,
                **{f"auc_{metric}": value for metric, value in random_auc.items()},
            })
            for metric in metrics:
                cov_auc = normalized_auc(fixed_curves["coverage25"], metric)
                pure_auc = normalized_auc(fixed_curves["pure"], metric)
                comparison_values[base, "coverage25_vs_random25", metric][model].append(
                    cov_auc - random_auc[metric]
                )
                comparison_values[base, "coverage25_vs_pure", metric][model].append(
                    cov_auc - pure_auc
                )
        log(output, f"hybrid completed {model} seed={seed}")
    comparisons = [
        paired_stats(values, f"{base}_{comparison}", metric)
        for (base, comparison, metric), values in comparison_values.items()
    ]
    write_csv(output / "tables/hybrid_fixed_metrics.csv", fixed_rows)
    write_csv(output / "tables/hybrid_random_summary.csv", random_rows)
    write_csv(output / "tables/hybrid_auc_by_run.csv", auc_rows)
    write_csv(output / "tables/hybrid_comparisons.csv", comparisons)
    write_csv(output / "tables/hybrid_endpoint_new_lost.csv", endpoint_rows)
    return fixed_rows, random_rows, comparisons


def parse_values(value: str) -> list[str]:
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return [value]


def interaction_flags(row: dict[str, str], feature: dict[str, str], pair: str) -> tuple[bool, int, int]:
    if pair == "small_x_occlusion":
        valid = feature["occlusion_level"] != "unknown"
        return valid, int(float(row["bbox_area_ratio"]) < 0.03), int(feature["occlusion_level"] in {"moderate", "heavy"})
    if pair == "pose_x_truncation":
        valid = feature["pose_action"] != "unknown" and feature["body_orientation"] != "unknown" and feature["truncation_level"] != "unknown"
        unusual = feature["pose_action"] in {"running", "jumping", "riding_or_being_ridden", "interacting", "other"} or feature["body_orientation"] == "overhead_or_unusual"
        return valid, int(unusual), int(feature["truncation_level"] in {"moderate", "heavy"})
    if pair == "blur_x_low_contrast":
        valid = feature["blur_level"] != "unknown" and feature["target_background_contrast"] != "unknown"
        return valid, int(feature["blur_level"] in {"mild", "severe"}), int(feature["target_background_contrast"] == "low")
    raise KeyError(pair)


def run_interactions(output: Path, context: dict) -> tuple[list[dict], list[dict]]:
    annotations = {row["image_id"]: row for row in read_csv(POOL / "annotations/results.csv")}
    records: list[dict[str, object]] = []
    for model, seed in itertools.product(MODELS, SEEDS):
        _features, predicted = load_cached_run(context, model, seed)
        run_id = f"{model}_{seed}"
        for index, row in enumerate(context["rows"]):
            truth = LABELS[row["class_name"]]
            feature = annotations[row["sample_id"]]
            for pair in ("small_x_occlusion", "pose_x_truncation", "blur_x_low_contrast"):
                valid, a_value, b_value = interaction_flags(row, feature, pair)
                if valid:
                    records.append({
                        "pair": pair, "sample_id": row["sample_id"], "class_name": row["class_name"],
                        "run_id": run_id, "model": model, "seed": seed,
                        "a": a_value, "b": b_value, "error": int(int(predicted[index]) != truth),
                    })
    data = pd.DataFrame(records)
    cells: list[dict[str, object]] = []
    models: list[dict[str, object]] = []
    for pair, frame in data.groupby("pair", sort=False):
        for (a_value, b_value), cell in frame.groupby(["a", "b"]):
            cells.append({
                "pair": pair, "a": int(a_value), "b": int(b_value),
                "samples": int(cell["sample_id"].nunique()), "decisions": len(cell),
                "errors": int(cell["error"].sum()), "error_rate": float(cell["error"].mean()),
            })
        observed_cells = set(zip(frame["a"].astype(int), frame["b"].astype(int)))
        if len(observed_cells) < 4:
            missing = sorted(set(itertools.product((0, 1), repeat=2)) - observed_cells)
            models.append({
                "pair": pair,
                "valid_samples": int(frame["sample_id"].nunique()),
                "decisions": len(frame),
                "estimable": 0,
                "reason": f"missing cells: {missing}",
                "interaction_log_odds": "",
                "interaction_odds_ratio": "",
                "ci95_odds_ratio_low": "",
                "ci95_odds_ratio_high": "",
                "p_value": "",
            })
            continue
        fitted = smf.gee(
            "error ~ a * b + C(class_name) + C(model) + C(seed)",
            groups="sample_id",
            data=frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit()
        coefficient = float(fitted.params["a:b"])
        low, high = (float(value) for value in fitted.conf_int().loc["a:b"])
        models.append({
            "pair": pair,
            "valid_samples": int(frame["sample_id"].nunique()),
            "decisions": len(frame),
            "estimable": 1,
            "reason": "",
            "interaction_log_odds": coefficient,
            "interaction_odds_ratio": math.exp(coefficient),
            "ci95_odds_ratio_low": math.exp(low),
            "ci95_odds_ratio_high": math.exp(high),
            "p_value": float(fitted.pvalues["a:b"]),
        })
    write_csv(output / "tables/state_interaction_cells.csv", cells)
    write_csv(output / "tables/state_interaction_models.csv", models)
    make_interaction_figure(output, cells)
    log(output, "state-interaction models completed")
    return cells, models


def make_interaction_figure(output: Path, rows: list[dict]) -> None:
    titles = {
        "small_x_occlusion": "小目标 × 中重度遮挡",
        "pose_x_truncation": "非常规姿态 × 中重度截断",
        "blur_x_low_contrast": "模糊 × 低对比度",
    }
    labels = ("均无", "仅A", "仅B", "A与B")
    cells = ((0, 0), (1, 0), (0, 1), (1, 1))
    lookup = {(row["pair"], int(row["a"]), int(row["b"])): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    for axis, pair in zip(axes, titles):
        values = [float(lookup[pair, a, b]["error_rate"]) * 100 if (pair, a, b) in lookup else 0.0 for a, b in cells]
        supports = [int(lookup[pair, a, b]["samples"]) if (pair, a, b) in lookup else 0 for a, b in cells]
        bars = axis.bar(range(4), values, color=("#777777", "#3B6EA8", "#009E73", "#C84A3A"))
        axis.set_xticks(range(4), labels, rotation=18)
        axis.set_title(titles[pair])
        axis.grid(axis="y", alpha=0.2)
        for bar, value, support in zip(bars, values, supports):
            label = f"{value:.1f}%\nn={support}" if support else "无样本"
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.35, label, ha="center", fontsize=8)
    axes[0].set_ylabel("聚合分类错误率 (%)")
    fig.suptitle("预定义视觉状态组合与模型错误率")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "state_interaction_error_rates.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "state_interaction_error_rates.svg", bbox_inches="tight")
    plt.close(fig)


def make_summary_figure(output: Path, equal_rows: list[dict], hybrid_comparisons: list[dict]) -> None:
    endpoint = [row for row in equal_rows if int(row["budget"]) == 100]
    equal_values = []
    for baseline in EQUAL_ERROR_BASELINES:
        subset = [row for row in endpoint if row["baseline"] == baseline]
        equal_values.append((
            baseline,
            float(np.mean([float(row["delta_confusion"]) for row in subset])) * 100,
            float(np.mean([float(row["delta_clusters"]) for row in subset])) * 100,
        ))
    hybrid_lookup = {
        (row["comparison"], row["metric"]): row for row in hybrid_comparisons
        if "coverage25_vs_random25" in row["comparison"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x = np.arange(len(equal_values))
    width = 0.36
    axes[0].bar(x - width / 2, [row[1] for row in equal_values], width, label="混淆类型", color="#3B6EA8")
    axes[0].bar(x + width / 2, [row[2] for row in equal_values], width, label="错误簇", color="#009E73")
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_xticks(x, [row[0] for row in equal_values])
    axes[0].set_ylabel("Coverage - 基线（百分点）")
    axes[0].set_title("每类100张：等错误数后的诊断广度")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    labels, confusion, clusters = [], [], []
    for base in HYBRID_BASELINES:
        label = f"{base}_coverage25_vs_random25"
        labels.append(base)
        confusion.append(float(hybrid_lookup[label, "confusion_pair_coverage"]["mean_delta_auc"]) * 100)
        clusters.append(float(hybrid_lookup[label, "error_cluster_coverage"]["mean_delta_auc"]) * 100)
    x = np.arange(len(labels))
    axes[1].bar(x - width / 2, confusion, width, label="混淆类型", color="#3B6EA8")
    axes[1].bar(x + width / 2, clusters, width, label="错误簇", color="#009E73")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Coverage补充 - 随机补充 AUC（百分点）")
    axes[1].set_title("75%强基线 + 25%补充信号")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "incremental_value_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "incremental_value_summary.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    args = parser.parse_args()
    output = args.output.resolve()
    context = build_context()
    random_sets = random_orders(context["by_class"], args.random_replicates)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "progress.json", {"status": "running", "started_at": now()})
    log(output, f"started with {args.random_replicates} random replicates")
    equal_rows, equal_comparisons = run_equal_error(output, context, random_sets)
    write_json(output / "progress.json", {"status": "running", "stage": "hybrid", "updated_at": now()})
    fixed_rows, random_rows, hybrid_comparisons = run_hybrids(output, context, random_sets)
    write_json(output / "progress.json", {"status": "running", "stage": "interactions", "updated_at": now()})
    interaction_cells, interaction_models = run_interactions(output, context)
    make_summary_figure(output, equal_rows, hybrid_comparisons)
    protocol = {
        "experiment": "coverage-incremental-diagnostic-value-v1",
        "created_at": now(),
        "models": list(MODELS), "seeds": list(SEEDS), "classes": list(CLASSES),
        "budgets_per_class": list(BUDGETS), "random_replicates": args.random_replicates,
        "hybrid_base_fraction": 0.75, "hybrid_supplement_fraction": SUPPLEMENT_FRACTION,
        "equal_error_method": "exact expected distinct labels under sampling without replacement",
        "interaction_model": "binomial GEE with class, architecture, and seed fixed effects; sample-grouped robust covariance",
        "interaction_pairs": ["small_x_occlusion", "pose_x_truncation", "blur_x_low_contrast"],
        "frozen_predictions": True,
        "source_checksums": {
            "base_protocol": sha256(BASE / "protocol.json"),
            "candidate_pool": sha256(POOL / "manifests/candidate_pool.csv"),
            "annotations": sha256(POOL / "annotations/results.csv"),
        },
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "progress.json", {
        "status": "completed", "completed_at": now(),
        "equal_error_rows": len(equal_rows), "hybrid_fixed_rows": len(fixed_rows),
        "hybrid_random_rows": len(random_rows), "interaction_cells": len(interaction_cells),
        "interaction_models": len(interaction_models),
    })
    log(output, "completed all supplemental analyses")


if __name__ == "__main__":
    main()
