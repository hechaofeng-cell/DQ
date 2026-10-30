#!/usr/bin/env python3
"""Cross-source confirmation benchmark on the online-VLM-annotated VOC pool."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
sys.path.insert(0, str(SOURCE / "scripts"))

from run_four_model_gap_vs_random import build_model  # noqa: E402
from run_semantic_test_selection_baselines import extract_classifier_inputs  # noqa: E402
from run_class_stratified_error_controls import class_stratified_pair, holm  # noqa: E402
from run_coverage_incremental_value import paired_stats  # noqa: E402
from run_multiclass_object_selection_benchmark import (  # noqa: E402
    CLASSES,
    LABELS,
    MODELS,
    SEEDS,
    candidate_embeddings,
    concave_coverage_order,
    dsa_scores,
    evaluate_suite as evaluate_state_suite,
    kcenter_order,
    random_orders,
    ranked,
    softmax,
    state_matrix,
)

POOL = ROOT / "artifacts/voc2012_confirmation_pool_20260929"
ANNOTATIONS = ROOT / "artifacts/voc2012_confirmation_pool_20260929/online_annotations/results.csv"
SCHEMA = ROOT / "configs/multiclass_object_state_schema_v1_0.json"
TRAIN_MANIFEST = SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv"
CHECKPOINT_ROOT = SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/runs"
BASE_CACHE = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928/runs"
DEFAULT_OUTPUT = ROOT / "artifacts/voc2012_confirmation_benchmark_20260929"
BUDGETS = (10, 20, 40, 60, 80)
RANDOM_REPLICATES = 1000


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


def build_context() -> dict:
    rows = read_csv(POOL / "manifests/candidate_pool.csv")
    annotation_rows = [row for row in read_csv(ANNOTATIONS) if row["status"] == "ok"]
    features = {row["image_id"]: row for row in annotation_rows}
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if len(rows) != 500 or len(features) != 500:
        raise ValueError(f"expected 500 candidates/features, got {len(rows)}/{len(features)}")
    if set(row["sample_id"] for row in rows) != set(features):
        raise ValueError("candidate/annotation identity mismatch")
    for row in rows:
        row["label"] = str(LABELS[row["class_name"]])
    global_index = {row["sample_id"]: index for index, row in enumerate(rows)}
    by_class = {}
    for class_name in CLASSES:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        if len(class_rows) != 100:
            raise ValueError(f"{class_name}: expected 100")
        matrix, states = state_matrix(class_rows, features, schema)
        by_class[class_name] = {
            "rows": class_rows, "matrix": matrix, "states": states,
            "global_indices": np.asarray([global_index[row["sample_id"]] for row in class_rows]),
        }
    train_rows = read_csv(TRAIN_MANIFEST)
    return {"rows": rows, "features": features, "by_class": by_class, "train_rows": train_rows}


def infer(output: Path, context: dict, model_name: str, seed: int, workers: int) -> dict[str, np.ndarray]:
    run_dir = output / "runs" / model_name / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "candidate_outputs.npz"
    candidate_ids = [row["sample_id"] for row in context["rows"]]
    if path.is_file():
        saved = np.load(path)
        if saved["sample_ids"].tolist() != candidate_ids:
            raise ValueError(f"stale cache: {path}")
        candidate_features, candidate_logits = saved["features"], saved["logits"]
    else:
        checkpoint = torch.load(
            CHECKPOINT_ROOT / model_name / "oi_gap_add0" / str(seed) / "best.pt",
            map_location="cpu", weights_only=False,
        )
        model = build_model(model_name)
        model.load_state_dict(checkpoint["model"])
        model.cuda().eval()
        candidate_features, candidate_logits = extract_classifier_inputs(
            model, context["rows"], torch.device("cuda"), workers
        )
        np.savez_compressed(path, features=candidate_features, logits=candidate_logits, sample_ids=np.asarray(candidate_ids))
        del model
        torch.cuda.empty_cache()
    train = np.load(BASE_CACHE / model_name / str(seed) / "train_features.npz")
    expected_train = [row["sample_id"] for row in context["train_rows"]]
    if train["sample_ids"].tolist() != expected_train:
        raise ValueError("base train cache identity mismatch")
    probabilities = softmax(candidate_logits)
    write_csv(run_dir / "candidate_predictions.csv", [{
        "sample_id": row["sample_id"], "class_name": row["class_name"],
        "true_label": row["label"], "predicted_label": int(probabilities[index].argmax()),
        "probabilities": json.dumps(probabilities[index].tolist()),
    } for index, row in enumerate(context["rows"])])
    return {
        "candidate_features": candidate_features,
        "candidate_logits": candidate_logits,
        "train_features": train["features"],
        "train_labels": train["labels"],
    }


def make_independent_orders(context: dict, output: Path, workers: int) -> dict:
    embeddings = candidate_embeddings(context, output / "embeddings/vit_b16_candidates.npz", workers)
    orders = {"coverage": {}, "kcenter": {}}
    saved = []
    for class_name in CLASSES:
        section = context["by_class"][class_name]
        orders["coverage"][class_name] = concave_coverage_order(section["rows"], section["matrix"])
        orders["kcenter"][class_name] = kcenter_order(section["rows"], embeddings[section["global_indices"]])
        for method in orders:
            saved.extend({
                "method": method, "class_name": class_name, "rank": rank + 1,
                "sample_id": section["rows"][index]["sample_id"],
            } for rank, index in enumerate(orders[method][class_name]))
    write_csv(output / "tables/model_independent_orders.csv", saved)
    return orders


def normalized_auc(rows: list[dict], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["budget"]))
    x = np.asarray([0, *[int(row["budget"]) for row in ordered]], dtype=float)
    y = np.asarray([0, *[float(row[metric]) for row in ordered]], dtype=float)
    return float(np.trapezoid(y, x) / BUDGETS[-1])


def confirmation_error_structures(context: dict, features: np.ndarray, predicted: np.ndarray, seed: int) -> dict:
    structures = {}
    for class_offset, class_name in enumerate(CLASSES):
        global_indices = context["by_class"][class_name]["global_indices"]
        wrong = global_indices[predicted[global_indices] != LABELS[class_name]]
        possible = sorted(set(int(predicted[index]) for index in wrong))
        if len(wrong) == 0:
            labels = np.asarray([], dtype=int)
            cluster_count = 0
        elif len(wrong) == 1:
            labels = np.asarray([0], dtype=int)
            cluster_count = 1
        else:
            cluster_count = max(2, min(8, round(math.sqrt(len(wrong)))))
            values = features[wrong]
            values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            labels = KMeans(
                n_clusters=cluster_count, random_state=20260928 + seed + class_offset, n_init=20
            ).fit_predict(values)
        structures[class_name] = {
            "wrong_global": set(int(value) for value in wrong),
            "possible_predictions": possible,
            "cluster_count": cluster_count,
            "cluster_by_global": {int(index): int(label) for index, label in zip(wrong, labels)},
        }
    return structures


def confirmation_metrics(context: dict, orders: dict, budget: int, predicted: np.ndarray, structures: dict) -> dict:
    confusion_found, cluster_found = set(), set()
    class_confusion, class_clusters, error_rates = [], [], []
    evaluable_confusion, evaluable_clusters = [], []
    for class_name in CLASSES:
        global_indices = context["by_class"][class_name]["global_indices"]
        chosen = global_indices[np.asarray(orders[class_name][:budget])]
        wrong = [int(index) for index in chosen if int(predicted[index]) != LABELS[class_name]]
        predictions = {(class_name, int(predicted[index])) for index in wrong}
        clusters = {(class_name, int(structures[class_name]["cluster_by_global"][index])) for index in wrong}
        confusion_found.update(predictions)
        cluster_found.update(clusters)
        confusion_universe = len(structures[class_name]["possible_predictions"])
        cluster_universe = int(structures[class_name]["cluster_count"])
        class_confusion.append(len(predictions) / confusion_universe if confusion_universe else 0.0)
        class_clusters.append(len(clusters) / cluster_universe if cluster_universe else 0.0)
        evaluable_confusion.append(int(confusion_universe > 0))
        evaluable_clusters.append(int(cluster_universe > 0))
        error_rates.append(len(wrong) / budget)
    state = evaluate_state_suite(context, orders, budget, predicted)
    return {
        "confusion_pair_coverage": float(np.mean(class_confusion)),
        "error_cluster_coverage": float(np.mean(class_clusters)),
        "evaluable_confusion_class_fraction": float(np.mean(evaluable_confusion)),
        "evaluable_cluster_class_fraction": float(np.mean(evaluable_clusters)),
        "macro_error_rate": float(np.mean(error_rates)),
        "macro_f1": float(state["macro_f1"]),
        "macro_frequency_coverage": float(state["macro_frequency_coverage"]),
        "macro_failed_state_fraction": float(state["macro_failed_state_fraction"]),
        "confusion_set": confusion_found,
        "cluster_set": cluster_found,
    }


def run(output: Path, workers: int, random_replicates: int, smoke: bool = False) -> None:
    context = build_context()
    independent = make_independent_orders(context, output, workers)
    random_sets = random_orders(context["by_class"], random_replicates)
    metric_rows, auc_rows, equal_rows = [], [], []
    comparison_values = defaultdict(lambda: defaultdict(list))
    equal_values = defaultdict(lambda: defaultdict(list))

    experiment_runs = [(MODELS[0], SEEDS[0])] if smoke else list(itertools.product(MODELS, SEEDS))
    for model, seed in experiment_runs:
        arrays = infer(output, context, model, seed, workers)
        probabilities = softmax(arrays["candidate_logits"])
        predicted = probabilities.argmax(axis=1)
        structures = confirmation_error_structures(context, arrays["candidate_features"], predicted, seed)
        orders = dict(independent)
        orders["deepgini"] = {}
        orders["dsa"] = {}
        gini = 1 - np.square(probabilities).sum(axis=1)
        surprise = dsa_scores(
            arrays["train_features"], arrays["train_labels"], arrays["candidate_features"], predicted
        )
        for class_name in CLASSES:
            section = context["by_class"][class_name]
            indices = section["global_indices"]
            orders["deepgini"][class_name] = ranked(gini[indices], section["rows"])
            orders["dsa"][class_name] = ranked(surprise[indices], section["rows"])
        curves = defaultdict(list)
        equal_curves = defaultdict(list)
        for budget in BUDGETS:
            coverage_metric = confirmation_metrics(context, orders["coverage"], budget, predicted, structures)
            for method in ("coverage", "deepgini", "dsa", "kcenter"):
                metric = coverage_metric if method == "coverage" else confirmation_metrics(
                    context, orders[method], budget, predicted, structures
                )
                row = {"method": method, "model": model, "seed": seed, "budget": budget}
                row.update({key: value for key, value in metric.items() if not isinstance(value, set)})
                metric_rows.append(row)
                curves[method].append(row)
            random_metrics = [confirmation_metrics(context, order, budget, predicted, structures) for order in random_sets]
            row = {"method": "random", "model": model, "seed": seed, "budget": budget}
            for key in ("confusion_pair_coverage", "error_cluster_coverage", "macro_error_rate", "macro_f1", "macro_frequency_coverage", "macro_failed_state_fraction"):
                row[key] = float(np.mean([item[key] for item in random_metrics]))
            metric_rows.append(row)
            curves["random"].append(row)

            for baseline in ("random", "deepgini", "dsa", "kcenter"):
                if baseline == "random":
                    pairs = [
                        class_stratified_pair(context, orders["coverage"], order, budget, predicted, structures)[0]
                        for order in random_sets
                    ]
                    pair = {key: float(np.mean([item[key] for item in pairs])) for key in pairs[0]}
                else:
                    pair = class_stratified_pair(
                        context, orders["coverage"], orders[baseline], budget, predicted, structures
                    )[0]
                pair["delta_confusion"] = pair["left_confusion"] - pair["right_confusion"]
                pair["delta_clusters"] = pair["left_clusters"] - pair["right_clusters"]
                item = {"baseline": baseline, "model": model, "seed": seed, "budget": budget, **pair}
                equal_rows.append(item)
                equal_curves[baseline].append(item)

        for method in ("coverage", "random", "deepgini", "dsa", "kcenter"):
            row = {"method": method, "model": model, "seed": seed}
            for metric in ("confusion_pair_coverage", "error_cluster_coverage", "macro_error_rate", "macro_f1", "macro_frequency_coverage", "macro_failed_state_fraction"):
                row[f"auc_{metric}"] = normalized_auc(curves[method], metric)
            auc_rows.append(row)
        for baseline in ("random", "deepgini", "dsa", "kcenter"):
            for metric in ("confusion", "clusters"):
                value = normalized_auc(equal_curves[baseline], f"delta_{metric}")
                equal_values[baseline, metric][model].append(value)
                comparison_values[baseline, metric][model].append(
                    normalized_auc(curves["coverage"], "confusion_pair_coverage" if metric == "confusion" else "error_cluster_coverage")
                    - normalized_auc(curves[baseline], "confusion_pair_coverage" if metric == "confusion" else "error_cluster_coverage")
                )
        print(f"[confirm] {model} seed={seed} complete", flush=True)

    direct_comparisons = [
        paired_stats(comparison_values[base, metric], f"coverage_vs_{base}", metric)
        for base in ("random", "deepgini", "dsa", "kcenter") for metric in ("confusion", "clusters")
    ]
    equal_comparisons = [
        paired_stats(equal_values[base, metric], f"coverage_vs_{base}", f"class_equal_error_{metric}")
        for base in ("random", "deepgini", "dsa", "kcenter") for metric in ("confusion", "clusters")
    ]
    primary = [row for row in equal_comparisons if row["comparison"] == "coverage_vs_random"]
    for row, value in zip(primary, holm([row["permutation_p"] for row in primary])):
        row["holm_primary_p"] = value
    write_csv(output / "tables/method_budget_metrics.csv", metric_rows)
    write_csv(output / "tables/auc_by_run.csv", auc_rows)
    write_csv(output / "tables/class_equal_error_metrics.csv", equal_rows)
    write_csv(output / "tables/direct_comparisons.csv", direct_comparisons)
    write_csv(output / "tables/class_equal_error_comparisons.csv", equal_comparisons)

    summary = []
    for method in ("coverage", "random", "deepgini", "dsa", "kcenter"):
        for budget in BUDGETS:
            subset = [row for row in metric_rows if row["method"] == method and row["budget"] == budget]
            item = {"method": method, "budget": budget}
            for metric in ("confusion_pair_coverage", "error_cluster_coverage", "macro_error_rate", "macro_f1", "macro_frequency_coverage", "macro_failed_state_fraction"):
                item[f"{metric}_mean"] = float(np.mean([row[metric] for row in subset]))
            summary.append(item)
    write_csv(output / "tables/method_budget_summary.csv", summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"coverage": "#C44E52", "random": "#777777", "deepgini": "#287271", "dsa": "#4C72B0", "kcenter": "#DD8452"}
    for axis, (metric, title) in zip(axes, (
        ("confusion_pair_coverage_mean", "Confusion-type coverage"),
        ("error_cluster_coverage_mean", "Error-cluster coverage"),
        ("macro_error_rate_mean", "Error rate"),
    )):
        for method in colors:
            subset = [row for row in summary if row["method"] == method]
            axis.plot([row["budget"] for row in subset], [row[metric] for row in subset], marker="o", label=method, color=colors[method])
        axis.set_title(title)
        axis.set_xlabel("Candidates per class")
        axis.grid(alpha=.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures/confirmation_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figures/confirmation_curves.svg", bbox_inches="tight")
    plt.close(fig)

    report = ["# PASCAL VOC 2012跨数据源确认实验", "", "## 设置", "",
        "- dog、cat、horse、sheep、person各100张；源图跨类别互斥。",
        "- 使用线上sensenova-6.8-flash-lite按object_state_v1.0标注。",
        "- 冻结8架构×3种子的已有分类器，预算为每类10/20/40/60/80。",
        "- 主指标为逐类别等错误数后的混淆类型与错误簇覆盖AUC。", "",
        "## Coverage相对Random的确认结果", "",
        "| 指标 | AUC差值 | 95% CI | 正向运行 | Holm p |", "|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        report.append(f"| {row['metric']} | {row['mean_delta_auc']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['positive_runs']}/24 | {row['holm_primary_p']:.4f} |")
    report += ["", "## 边界", "",
        "- VOC 2012与当前训练数据存在年代、标注规范和图像分布差异，确认的是跨源方向稳定性，不是总体性能排行榜。",
        "- 100张/类限制了高预算端的分辨率，因此最大预算预先停在80。",
        "- 若某模型在某类全池仅有1个错误，则定义为单例错误簇；若为0个错误，该类簇覆盖按0计并报告不可评价比例。",
        "- 线上VLM状态不是人工真值；一致性实验单独报告其标注敏感性。", ""]
    (output / "report_zh.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.workers, args.random_replicates, args.smoke)


if __name__ == "__main__":
    main()
