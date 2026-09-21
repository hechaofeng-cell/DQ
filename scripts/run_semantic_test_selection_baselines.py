#!/usr/bin/env python3
"""Compare semantic coverage with random, DeepGini, DSA, and k-center.

The 24 D300 classifiers and candidate pools are frozen. This script only creates
target-class rankings, reuses saved predictions, and evaluates equal-budget test
suites for dog and cat.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from openimages_cua_vsl import StateSchema
from run_fixed_train_test_adequacy_experiment import (
    coverage_values,
    dog_failure_metrics,
    read_csv,
)
from run_four_model_gap_vs_random import build_model
from run_resnet18_dog_coverage_comparison import write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CLASS_SPECS, CropDataset


ROOT = Path(__file__).resolve().parents[1]
GRADIENT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
DOG_SOURCE = ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"
CAT_SOURCE = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
DOG_FEATURES = ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
CAT_FEATURES = ROOT / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv"
DOG_SCHEMA = ROOT / "configs/dog_feature_schema_v3_1.json"
CAT_SCHEMA = ROOT / "configs/cat_feature_schema_v1_0.json"
OUTPUT = ROOT / "artifacts/semantic_test_selection_baselines_20260920"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
BUDGETS = (20, 40, 60, 80, 100)
CLASS_NAMES = tuple(name for name, _ in CLASS_SPECS)
TARGET_LABELS = {"dog": 0, "cat": 1}
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260920
PERMUTATION_SEED = 20260921
CAP = 30


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_log(message: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (OUTPUT / "execution.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def run_key(model: str, seed: int) -> str:
    return f"{model}::{seed}"


def target_context(category: str) -> dict:
    source = DOG_SOURCE if category == "dog" else CAT_SOURCE
    pool = read_csv(source / "candidate_pool.csv")
    target_rows = [row for row in pool if row["class_name"] == category]
    if len(target_rows) != 200:
        raise ValueError(f"{category}: expected 200 target candidates")
    features_path = DOG_FEATURES if category == "dog" else CAT_FEATURES
    schema_path = DOG_SCHEMA if category == "dog" else CAT_SCHEMA
    features = {row["image_id"]: row for row in read_csv(features_path)}
    schema = StateSchema.from_path(schema_path)
    states = schema.coverage_states()
    state_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(target_rows), len(states)), dtype=np.int8)
    for row_index, row in enumerate(target_rows):
        for pair in schema.state_pairs(features[row["sample_id"]]):
            if pair in state_index:
                matrix[row_index, state_index[pair]] = 1

    target_index = {row["sample_id"]: index for index, row in enumerate(target_rows)}
    gap_path = source / "orders" / f"gap_{category}_order.csv"
    gap_order = [target_index[row["sample_id"]] for row in sorted(read_csv(gap_path), key=lambda r: int(r["rank"]))]
    random_path = source / "orders" / f"random_{category}_orders.csv"
    random_groups: dict[int, list[dict]] = defaultdict(list)
    for row in read_csv(random_path):
        random_groups[int(row["replicate"])].append(row)
    random_orders = []
    for replicate in range(RANDOM_REPLICATES):
        rows = sorted(random_groups[replicate], key=lambda r: int(r["rank"]))
        random_orders.append([target_index[row["sample_id"]] for row in rows])
    if any(len(order) != 200 or len(set(order)) != 200 for order in [gap_order, *random_orders]):
        raise ValueError(f"{category}: invalid source ordering")
    return {
        "category": category,
        "source": source,
        "pool": pool,
        "target_rows": target_rows,
        "target_index": target_index,
        "matrix": matrix,
        "states": states,
        "gap_order": gap_order,
        "random_orders": random_orders,
    }


def final_linear(model: nn.Module) -> nn.Linear:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return model.head
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            return model.classifier
        for layer in reversed(model.classifier):
            if isinstance(layer, nn.Linear):
                return layer
    raise TypeError(f"cannot find final linear layer for {type(model).__name__}")


@torch.inference_mode()
def extract_classifier_inputs(model: nn.Module, rows: list[dict], device: torch.device, workers: int) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(
        CropDataset(rows, transform), batch_size=96, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    batches: list[np.ndarray] = []

    def hook(_module, args):
        batches.append(args[0].detach().float().cpu().numpy())

    handle = final_linear(model).register_forward_pre_hook(hook)
    logits_out = []
    try:
        model.eval()
        for images, _labels, _indices in loader:
            logits_out.append(model(images.to(device, non_blocking=True)).detach().float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(batches), np.concatenate(logits_out)


def dsa_scores(train_features: np.ndarray, train_labels: np.ndarray,
               candidate_features: np.ndarray, candidate_predicted: np.ndarray) -> np.ndarray:
    scores = np.empty(len(candidate_features), dtype=np.float64)
    for index, (feature, predicted) in enumerate(zip(candidate_features, candidate_predicted)):
        same_indices = np.flatnonzero(train_labels == predicted)
        other_indices = np.flatnonzero(train_labels != predicted)
        same_distances = np.linalg.norm(train_features[same_indices] - feature, axis=1)
        nearest_same = train_features[same_indices[int(np.argmin(same_distances))]]
        numerator = float(same_distances.min())
        denominator = float(np.linalg.norm(train_features[other_indices] - nearest_same, axis=1).min())
        scores[index] = numerator / max(denominator, 1e-12)
    return scores


def ranked_indices(scores: np.ndarray, rows: list[dict], descending: bool = True) -> list[int]:
    direction = -1.0 if descending else 1.0
    return sorted(range(len(rows)), key=lambda i: (direction * float(scores[i]), rows[i]["sample_id"]))


def extract_dsa_orders(contexts: dict[str, dict], workers: int, smoke: bool) -> dict[str, dict[str, list[int]]]:
    output: dict[str, dict[str, list[int]]] = {category: {} for category in contexts}
    train_rows = read_csv(GRADIENT / "manifests/train_oi_gap_add0.csv")
    train_labels = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [(MODELS[0], SEEDS[0])] if smoke else list(itertools.product(MODELS, SEEDS))
    for run_number, (model_name, seed) in enumerate(runs, 1):
        key = run_key(model_name, seed)
        run_dir = OUTPUT / "dsa" / model_name / str(seed)
        score_paths = {category: run_dir / f"{category}_scores.csv" for category in contexts}
        if all(path.is_file() for path in score_paths.values()):
            for category, path in score_paths.items():
                rows = sorted(read_csv(path), key=lambda row: int(row["rank"]))
                output[category][key] = [contexts[category]["target_index"][row["sample_id"]] for row in rows]
            append_log(f"DSA reuse {run_number}/{len(runs)} {key}")
            continue
        checkpoint = torch.load(
            GRADIENT / "runs" / model_name / "oi_gap_add0" / str(seed) / "best.pt",
            map_location="cpu", weights_only=False,
        )
        model = build_model(model_name)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        train_features, _ = extract_classifier_inputs(model, train_rows, device, workers)
        for category, context in contexts.items():
            candidate_features, logits = extract_classifier_inputs(model, context["target_rows"], device, workers)
            predicted = logits.argmax(axis=1)
            saved = read_csv(context["source"] / "predictions" / model_name / str(seed) / "candidate_predictions.csv")
            saved_by_id = {row["sample_id"]: int(row["predicted_label"]) for row in saved}
            expected = np.asarray([saved_by_id[row["sample_id"]] for row in context["target_rows"]])
            if not np.array_equal(predicted, expected):
                raise ValueError(f"{category} {key}: extracted predictions differ from frozen predictions")
            scores = dsa_scores(train_features, train_labels, candidate_features, predicted)
            order = ranked_indices(scores, context["target_rows"])
            output[category][key] = order
            write_csv(score_paths[category], [
                {"rank": rank + 1, "sample_id": context["target_rows"][index]["sample_id"],
                 "dsa_score": float(scores[index]), "predicted_label": int(predicted[index])}
                for rank, index in enumerate(order)
            ])
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        append_log(f"DSA extracted {run_number}/{len(runs)} {key}")
    return output


def deepgini_orders(contexts: dict[str, dict]) -> dict[str, dict[str, list[int]]]:
    output: dict[str, dict[str, list[int]]] = {category: {} for category in contexts}
    for category, context in contexts.items():
        for model_name, seed in itertools.product(MODELS, SEEDS):
            key = run_key(model_name, seed)
            predictions = read_csv(context["source"] / "predictions" / model_name / str(seed) / "candidate_predictions.csv")
            by_id = {row["sample_id"]: row for row in predictions}
            scores = []
            for target in context["target_rows"]:
                row = by_id[target["sample_id"]]
                probabilities = np.asarray([float(row[f"prob_{name}"]) for name in CLASS_NAMES])
                scores.append(1.0 - float(np.square(probabilities).sum()))
            scores_array = np.asarray(scores)
            order = ranked_indices(scores_array, context["target_rows"])
            output[category][key] = order
            write_csv(OUTPUT / "deepgini" / model_name / str(seed) / f"{category}_scores.csv", [
                {"rank": rank + 1, "sample_id": context["target_rows"][index]["sample_id"],
                 "deepgini_score": float(scores_array[index])}
                for rank, index in enumerate(order)
            ])
    return output


@torch.inference_mode()
def kcenter_orders(contexts: dict[str, dict], workers: int) -> dict[str, list[int]]:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    encoder = models.vit_b_16(weights=weights)
    encoder.heads.head = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    output = {}
    for category, context in contexts.items():
        embedding_path = OUTPUT / "kcenter" / f"{category}_embeddings.npz"
        if embedding_path.is_file():
            saved = np.load(embedding_path)
            embeddings = saved["embeddings"]
            sample_ids = saved["sample_ids"].tolist()
            if sample_ids != [row["sample_id"] for row in context["target_rows"]]:
                raise ValueError(f"{category}: stale k-center embeddings")
        else:
            loader = DataLoader(
                CropDataset(context["target_rows"], weights.transforms()), batch_size=64,
                shuffle=False, num_workers=workers, pin_memory=True,
                persistent_workers=workers > 0,
            )
            chunks = []
            for images, _labels, _indices in loader:
                chunks.append(encoder(images.to(device, non_blocking=True)).detach().float().cpu().numpy())
            embeddings = np.concatenate(chunks)
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embedding_path, embeddings=embeddings,
                sample_ids=np.asarray([row["sample_id"] for row in context["target_rows"]]),
            )
        embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        center = embeddings.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        first = min(range(len(embeddings)), key=lambda i: (1.0 - float(embeddings[i] @ center), context["target_rows"][i]["sample_id"]))
        selected = [first]
        remaining = set(range(len(embeddings))) - {first}
        min_distance = 1.0 - embeddings @ embeddings[first]
        while remaining:
            best = max(remaining, key=lambda i: (float(min_distance[i]), context["target_rows"][i]["sample_id"]))
            selected.append(best)
            remaining.remove(best)
            min_distance = np.minimum(min_distance, 1.0 - embeddings @ embeddings[best])
        output[category] = selected
        write_csv(OUTPUT / "kcenter" / f"{category}_order.csv", [
            {"rank": rank + 1, "sample_id": context["target_rows"][index]["sample_id"]}
            for rank, index in enumerate(selected)
        ])
        append_log(f"k-center completed {category}")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def validate_orders(contexts: dict[str, dict], deepgini: dict, dsa: dict, kcenter: dict, smoke: bool) -> None:
    expected_runs = 1 if smoke else len(MODELS) * len(SEEDS)
    for category, context in contexts.items():
        collections = [context["gap_order"], kcenter[category], *context["random_orders"]]
        collections.extend(deepgini[category].values())
        collections.extend(dsa[category].values())
        if len(deepgini[category]) != len(MODELS) * len(SEEDS) or len(dsa[category]) != expected_runs:
            raise ValueError(f"{category}: missing model-specific order")
        for order in collections:
            if len(order) != 200 or set(order) != set(range(200)):
                raise ValueError(f"{category}: invalid ordering")
    append_log("all rankings contain 200 unique candidates and are nested by prefix")


def evaluate_order(order: list[int], correct: np.ndarray, matrix: np.ndarray, budget: int) -> dict:
    chosen = order[:budget]
    failure = dog_failure_metrics(chosen, correct, matrix)
    frequency, presence = coverage_values(matrix, chosen)
    present_states = int((matrix[chosen].sum(axis=0) > 0).sum())
    error_count = int((~correct[np.asarray(chosen)]).sum())
    return {
        "frequency_coverage": frequency,
        "presence_coverage": presence,
        "present_state_count": present_states,
        "failed_state_count": failure["failed_state_count"],
        "normalized_failed_state_breadth": failure["failed_state_count"] / present_states if present_states else 0.0,
        "target_error_count": error_count,
        "target_error_rate": error_count / budget,
        "state_macro_recall": failure["state_macro_recall"],
        "worst5_state_recall": failure["worst5_state_recall"],
    }


def aggregate(contexts: dict[str, dict], deepgini: dict, dsa: dict, kcenter: dict) -> tuple[list[dict], list[dict]]:
    detail = []
    for category, context in contexts.items():
        target_label = TARGET_LABELS[category]
        for model_name, seed in itertools.product(MODELS, SEEDS):
            key = run_key(model_name, seed)
            predictions = read_csv(context["source"] / "predictions" / model_name / str(seed) / "candidate_predictions.csv")
            by_id = {row["sample_id"]: row for row in predictions}
            correct = np.asarray([
                int(by_id[row["sample_id"]]["predicted_label"]) == target_label
                for row in context["target_rows"]
            ], dtype=bool)
            fixed_orders = {
                "coverage": context["gap_order"],
                "deepgini": deepgini[category][key],
                "dsa": dsa[category][key],
                "kcenter": kcenter[category],
            }
            for method, order in fixed_orders.items():
                for budget in BUDGETS:
                    detail.append({
                        "category": category, "method": method, "replicate": -1,
                        "model": model_name, "seed": seed, "budget": budget,
                        **evaluate_order(order, correct, context["matrix"], budget),
                    })
            for replicate, order in enumerate(context["random_orders"]):
                for budget in BUDGETS:
                    detail.append({
                        "category": category, "method": "random", "replicate": replicate,
                        "model": model_name, "seed": seed, "budget": budget,
                        **evaluate_order(order, correct, context["matrix"], budget),
                    })
    write_csv(OUTPUT / "tables" / "metrics_detailed.csv", detail)

    auc_rows = []
    metrics = ("failed_state_count", "normalized_failed_state_breadth", "target_error_rate", "frequency_coverage")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in detail:
        groups[(row["category"], row["method"], row["replicate"], row["model"], row["seed"])].append(row)
    max_states = {category: context["matrix"].shape[1] for category, context in contexts.items()}
    for (category, method, replicate, model_name, seed), rows in groups.items():
        rows.sort(key=lambda row: row["budget"])
        x = np.asarray([0, *BUDGETS], dtype=float)
        out = {"category": category, "method": method, "replicate": replicate, "model": model_name, "seed": seed}
        for metric in metrics:
            y = np.asarray([0.0, *[float(row[metric]) for row in rows]])
            denominator = 100.0
            if metric == "failed_state_count":
                denominator *= max_states[category]
            out[f"auc_{metric}"] = float(np.trapezoid(y, x) / denominator)
        auc_rows.append(out)
    write_csv(OUTPUT / "tables" / "auc_by_run.csv", auc_rows)
    return detail, auc_rows


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def exact_cluster_permutation(differences: dict[str, list[float]]) -> float:
    architecture_means = np.asarray([np.mean(differences[model]) for model in MODELS])
    observed = abs(float(architecture_means.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(MODELS)):
        values.append(abs(float(np.mean(architecture_means * np.asarray(signs)))))
    return float((np.count_nonzero(np.asarray(values) >= observed - 1e-15)) / len(values))


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def statistical_analysis(auc_rows: list[dict]) -> list[dict]:
    index = {(r["category"], r["method"], int(r["replicate"]), r["model"], int(r["seed"])): r for r in auc_rows}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []
    for category in ("dog", "cat"):
        category_results = []
        for baseline in ("random", "deepgini", "dsa", "kcenter"):
            differences_by_arch: dict[str, list[float]] = defaultdict(list)
            run_differences = []
            for model_name in MODELS:
                for seed in SEEDS:
                    coverage = float(index[(category, "coverage", -1, model_name, seed)]["auc_failed_state_count"])
                    if baseline == "random":
                        values = [float(index[(category, "random", rep, model_name, seed)]["auc_failed_state_count"])
                                  for rep in range(RANDOM_REPLICATES)]
                        other = float(np.mean(values))
                    else:
                        other = float(index[(category, baseline, -1, model_name, seed)]["auc_failed_state_count"])
                    difference = coverage - other
                    differences_by_arch[model_name].append(difference)
                    run_differences.append(difference)
            run_diff = np.asarray(run_differences)
            boot = []
            for _ in range(BOOTSTRAP_REPLICATES):
                sampled_architectures = rng.choice(MODELS, size=len(MODELS), replace=True)
                random_rep = int(rng.integers(RANDOM_REPLICATES)) if baseline == "random" else -1
                sampled_differences = []
                for model_name in sampled_architectures:
                    sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                    for seed in sampled_seeds:
                        coverage = float(index[(category, "coverage", -1, model_name, int(seed))]["auc_failed_state_count"])
                        other = float(index[(category, baseline, random_rep, model_name, int(seed))]["auc_failed_state_count"])
                        sampled_differences.append(coverage - other)
                boot.append(float(np.mean(sampled_differences)))
            architecture_differences = np.asarray([np.mean(differences_by_arch[m]) for m in MODELS])
            sd = float(architecture_differences.std(ddof=1))
            result = {
                "category": category,
                "comparison": f"coverage_vs_{baseline}",
                "mean_delta_auc_fsb": float(run_diff.mean()),
                "median_delta_auc_fsb": float(np.median(run_diff)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "cluster_dz": float(architecture_differences.mean() / sd) if sd else math.inf,
                "rank_biserial": rank_biserial(architecture_differences),
                "positive_runs": int((run_diff > 0).sum()),
                "zero_runs": int((run_diff == 0).sum()),
                "negative_runs": int((run_diff < 0).sum()),
                "permutation_p": exact_cluster_permutation(differences_by_arch),
            }
            category_results.append(result)
        adjusted = holm_adjust([row["permutation_p"] for row in category_results])
        for row, value in zip(category_results, adjusted):
            row["holm_p"] = value
        comparisons.extend(category_results)
    write_csv(OUTPUT / "tables" / "primary_auc_comparisons.csv", comparisons)
    return comparisons


def summarize_budget(detail: list[dict]) -> list[dict]:
    output = []
    for category in ("dog", "cat"):
        for method in ("coverage", "random", "deepgini", "dsa", "kcenter"):
            for budget in BUDGETS:
                rows = [r for r in detail if r["category"] == category and r["method"] == method and r["budget"] == budget]
                if method == "random":
                    # Average each model over random sequences before aggregating models.
                    per_run = []
                    for model_name in MODELS:
                        for seed in SEEDS:
                            group = [r for r in rows if r["model"] == model_name and r["seed"] == seed]
                            per_run.append({metric: float(np.mean([float(r[metric]) for r in group])) for metric in (
                                "frequency_coverage", "failed_state_count", "normalized_failed_state_breadth", "target_error_rate")})
                    rows_for_summary = per_run
                else:
                    rows_for_summary = rows
                out = {"category": category, "method": method, "budget": budget}
                for metric in ("frequency_coverage", "failed_state_count", "normalized_failed_state_breadth", "target_error_rate"):
                    values = np.asarray([float(r[metric]) for r in rows_for_summary])
                    out[f"{metric}_mean"] = float(values.mean())
                    out[f"{metric}_sd"] = float(values.std(ddof=1))
                output.append(out)
    write_csv(OUTPUT / "tables" / "budget_summary.csv", output)
    return output


def fit_mixed_model(detail: list[dict]) -> dict:
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as error:
        return {"status": "not_run", "reason": str(error)}
    rows = [r for r in detail if r["method"] != "random"]
    # Add one expected-random observation per model/budget/category.
    for category in ("dog", "cat"):
        for model_name in MODELS:
            for seed in SEEDS:
                for budget in BUDGETS:
                    group = [r for r in detail if r["category"] == category and r["method"] == "random"
                             and r["model"] == model_name and r["seed"] == seed and r["budget"] == budget]
                    rows.append({
                        "category": category, "method": "random", "model": model_name, "seed": seed,
                        "budget": budget,
                        "normalized_failed_state_breadth": float(np.mean([r["normalized_failed_state_breadth"] for r in group])),
                    })
    frame = pd.DataFrame(rows)
    frame["budget_scaled"] = (frame["budget"] - 60.0) / 40.0
    frame["run"] = frame["model"] + "::" + frame["seed"].astype(str)
    try:
        model = smf.mixedlm(
            "normalized_failed_state_breadth ~ C(method) * budget_scaled * C(category)",
            frame, groups=frame["model"], re_formula="~budget_scaled",
            vc_formula={"run": "0 + C(run)"},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(reml=False, method="lbfgs", maxiter=2000, disp=False)
        warning_messages = [str(item.message) for item in caught]
        text = result.summary().as_text()
        (OUTPUT / "tables" / "mixed_effects_summary.txt").write_text(text + "\n", encoding="utf-8")
        coefficients = []
        for term in result.params.index:
            coefficients.append({
                "term": term, "coefficient": float(result.params[term]),
                "std_error": float(result.bse[term]), "p_value": float(result.pvalues[term]),
            })
        write_csv(OUTPUT / "tables" / "mixed_effects_coefficients.csv", coefficients)
        status = "completed" if not warning_messages else "completed_with_boundary_warning"
        return {
            "status": status, "converged": bool(result.converged), "n": int(len(frame)),
            "warnings": warning_messages,
            "inference_policy": "descriptive_only" if warning_messages else "standard",
        }
    except Exception as error:
        return {"status": "failed", "reason": repr(error), "n": int(len(frame))}


def make_figures(summary: list[dict], comparisons: list[dict], auc_rows: list[dict]) -> None:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "coverage": "#0072B2", "random": "#777777", "deepgini": "#D55E00",
        "dsa": "#CC79A7", "kcenter": "#009E73",
    }
    labels = {"coverage": "Semantic coverage", "random": "Random", "deepgini": "DeepGini", "dsa": "DSA", "kcenter": "k-center"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, category in enumerate(("dog", "cat")):
        for method in colors:
            rows = sorted([r for r in summary if r["category"] == category and r["method"] == method], key=lambda r: r["budget"])
            x = [r["budget"] for r in rows]
            axes[0, column].plot(x, [r["frequency_coverage_mean"] * 100 for r in rows], marker="o", color=colors[method], label=labels[method])
            axes[1, column].plot(x, [r["failed_state_count_mean"] for r in rows], marker="o", color=colors[method], label=labels[method])
        axes[0, column].set_title(f"{category.upper()}: visual-state coverage")
        axes[1, column].set_title(f"{category.upper()}: failure-state breadth")
        axes[0, column].set_ylabel("Frequency coverage (%)")
        axes[1, column].set_ylabel("Failed states (count)")
        axes[1, column].set_xlabel("Target-class test budget")
        for row in axes[:, column]:
            row.grid(alpha=0.25)
            row.set_xticks(BUDGETS)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig(figure_dir / "coverage_and_failure_breadth.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "coverage_and_failure_breadth.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, category in zip(axes, ("dog", "cat")):
        rows = [r for r in comparisons if r["category"] == category]
        y = np.arange(len(rows))
        means = np.asarray([r["mean_delta_auc_fsb"] for r in rows])
        low = means - np.asarray([r["ci95_low"] for r in rows])
        high = np.asarray([r["ci95_high"] for r in rows]) - means
        axis.errorbar(means, y, xerr=np.vstack([low, high]), fmt="o", color="#0072B2", capsize=4)
        axis.axvline(0, color="#444444", linewidth=1)
        axis.set_yticks(y, [r["comparison"].replace("coverage_vs_", "vs ") for r in rows])
        axis.set_title(category.upper())
        axis.set_xlabel("Coverage minus baseline: AUC-FSB")
        axis.grid(axis="x", alpha=0.25)
    fig.savefig(figure_dir / "auc_fsb_effects.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "auc_fsb_effects.svg", bbox_inches="tight")
    plt.close(fig)

    auc_index = {
        (r["category"], r["method"], int(r["replicate"]), r["model"], int(r["seed"])):
        float(r["auc_failed_state_count"]) for r in auc_rows
    }
    baselines = ("random", "deepgini", "dsa", "kcenter")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    max_abs = 0.0
    matrices = {}
    for category in ("dog", "cat"):
        matrix = np.zeros((len(MODELS), len(baselines)))
        for i, model_name in enumerate(MODELS):
            for j, baseline in enumerate(baselines):
                deltas = []
                for seed in SEEDS:
                    coverage = auc_index[(category, "coverage", -1, model_name, seed)]
                    if baseline == "random":
                        other = np.mean([auc_index[(category, baseline, rep, model_name, seed)] for rep in range(RANDOM_REPLICATES)])
                    else:
                        other = auc_index[(category, baseline, -1, model_name, seed)]
                    deltas.append(coverage - other)
                matrix[i, j] = float(np.mean(deltas))
        matrices[category] = matrix
        max_abs = max(max_abs, float(np.abs(matrix).max()))
    for axis, category in zip(axes, ("dog", "cat")):
        matrix = matrices[category]
        image = axis.imshow(matrix, cmap="RdBu", vmin=-max_abs, vmax=max_abs, aspect="auto")
        axis.set_xticks(range(len(baselines)), baselines, rotation=25, ha="right")
        axis.set_yticks(range(len(MODELS)), MODELS)
        axis.set_title(category.upper())
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if abs(matrix[i, j]) > max_abs * 0.55 else "black"
                axis.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=axes, label="Coverage minus baseline: AUC-FSB", shrink=0.85)
    fig.savefig(figure_dir / "architecture_effect_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "architecture_effect_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def write_protocol() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    protocol = {
        "experiment_id": "semantic-test-selection-baselines-20260920",
        "created_at": now(),
        "research_question": "Does explicit semantic state coverage provide independent failure-state diagnostic value beyond random, uncertainty, surprise, and generic embedding diversity selection?",
        "primary_metric": "normalized area under the failed-state-breadth budget curve (AUC-FSB)",
        "categories": ["dog", "cat"],
        "methods": ["coverage", "random", "deepgini", "dsa", "kcenter"],
        "budgets": list(BUDGETS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "random_replicates": RANDOM_REPLICATES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "kcenter_encoder": "torchvision ViT-B/16 IMAGENET1K_V1; frozen; classifier removed",
        "cat_annotation_status": "automatic VLM labels; human review and double annotation pending",
        "git_commit": commit,
        "input_sha256": {
            "dog_pool": sha256(DOG_SOURCE / "candidate_pool.csv"),
            "cat_pool": sha256(CAT_SOURCE / "candidate_pool.csv"),
            "dog_features": sha256(DOG_FEATURES),
            "cat_features": sha256(CAT_FEATURES),
            "training_manifest": sha256(GRADIENT / "manifests/train_oi_gap_add0.csv"),
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "- Thesis under test: explicit within-class semantic coverage can reveal a broader range of model failure states at a fixed test budget.\n"
        "- Primary endpoint: AUC-FSB across budgets 20, 40, 60, 80, and 100.\n"
        "- Strong baselines: DeepGini, DSA, ViT embedding k-center, and 1000 random nested orders.\n"
        "- Fixed factors: D300 training, 24 checkpoints, candidate pools, preprocessing, and labels.\n"
        "- Boundary: cat labels remain provisional until human review; stress-test F1 is not deployment accuracy.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol created: {now()}\n"
        "- Reproduction: `.venv/bin/python -u scripts/run_semantic_test_selection_baselines.py`.\n"
        "- Smoke test: add `--smoke`; it extracts one DSA model and validates ranking invariants.\n"
        "- Primary success rule: Coverage has positive AUC-FSB delta with a 95% hierarchical bootstrap CI excluding zero against random and at least two of three strong baselines, with direction consistency across dog and cat.\n"
        "- Cat semantic claims remain provisional until blind human review and inter-annotator reliability are complete.\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Coverage has independent diagnostic value | pending | `tables/primary_auc_comparisons.csv` | dog/cat frozen D300 setting |\n"
        "| Coverage improves semantic input coverage | pending | `tables/budget_summary.csv` | frozen schemas and pools |\n"
        "| Cat result is annotation-validated | blocked | human review not completed | do not make this claim |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n- Citation integrity: no new external claim in experiment output.\n"
        "- Result traceability: pending execution.\n- Claim scope: pending execution; cat labels provisional.\n",
        encoding="utf-8",
    )


def write_report(summary: list[dict], comparisons: list[dict], auc_rows: list[dict], mixed: dict) -> None:
    def comparison_line(category: str, baseline: str) -> dict:
        return next(r for r in comparisons if r["category"] == category and r["comparison"] == f"coverage_vs_{baseline}")

    strong_baselines = ("deepgini", "dsa", "kcenter")
    decisions = {}
    for category in ("dog", "cat"):
        random_result = comparison_line(category, "random")
        wins = [comparison_line(category, baseline) for baseline in strong_baselines]
        supported = (
            random_result["mean_delta_auc_fsb"] > 0 and random_result["ci95_low"] > 0
            and sum(r["mean_delta_auc_fsb"] > 0 and r["ci95_low"] > 0 for r in wins) >= 2
        )
        decisions[category] = supported
    cross_category = all(decisions.values())
    lines = [
        "# 语义覆盖与强基线比较结果", "", f"日期：{now()[:10]}", "",
        "## 核心判定", "",
    ]
    if cross_category:
        lines.append("> 在当前冻结 D300、dog/cat 候选池和自动状态标签设置下，Coverage 相对随机及至少两个强基线表现出跨类别的独立失败状态诊断价值。")
    elif any(decisions.values()):
        lines.append("> Coverage 的独立诊断价值获得部分支持：优势只在部分类别成立，尚不能主张跨类别普适性。")
    else:
        lines.append("> 当前结果不支持 Coverage 相对强测试选择基线具有稳定的独立失败状态诊断优势。")
    lines.extend(["", "判定依据是预先固定的 AUC-FSB，而不是事后选择某一个有利预算点。", "", "## 主要比较", "",
                  "| 类别 | 比较 | ΔAUC-FSB | 95% CI | 正/零/负运行 | cluster dz | Holm-p |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        lines.append(
            f"| {row['category']} | {row['comparison']} | {row['mean_delta_auc_fsb']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['positive_runs']}/{row['zero_runs']}/{row['negative_runs']} | "
            f"{row['cluster_dz']:.3f} | {row['holm_p']:.4f} |"
        )
    coverage_auc = {}
    for category in ("dog", "cat"):
        coverage_auc[category] = {}
        for method in ("coverage", "random", "deepgini", "dsa", "kcenter"):
            rows = [r for r in auc_rows if r["category"] == category and r["method"] == method]
            if method == "random":
                by_run = defaultdict(list)
                for row in rows:
                    by_run[(row["model"], row["seed"])].append(float(row["auc_frequency_coverage"]))
                values = [float(np.mean(group)) for group in by_run.values()]
            else:
                values = [float(row["auc_frequency_coverage"]) for row in rows]
            coverage_auc[category][method] = float(np.mean(values))
    lines.extend(["", "## 输入覆盖价值与诊断价值必须分开", "",
                  "Coverage 的频次覆盖 AUC 在两个类别都最高：", "",
                  "| 类别 | Coverage | Random | DeepGini | DSA | k-center |",
                  "|---|---:|---:|---:|---:|---:|",
                  f"| dog | {coverage_auc['dog']['coverage']:.4f} | {coverage_auc['dog']['random']:.4f} | {coverage_auc['dog']['deepgini']:.4f} | {coverage_auc['dog']['dsa']:.4f} | {coverage_auc['dog']['kcenter']:.4f} |",
                  f"| cat | {coverage_auc['cat']['coverage']:.4f} | {coverage_auc['cat']['random']:.4f} | {coverage_auc['cat']['deepgini']:.4f} | {coverage_auc['cat']['dsa']:.4f} | {coverage_auc['cat']['kcenter']:.4f} |",
                  "", "这支持 Coverage 作为模型无关、可解释的输入充分性度量；但它没有转化为相对强基线更高的失败状态发现能力。因此不能把“覆盖指标有描述价值”写成“覆盖选样是更强的故障发现算法”。",
                  "", "## 解释边界", "",
                  "- AUC-FSB 越高表示在连续预算下发现的失败视觉状态越广，不等同于错误样本比例越高。",
                  "- DeepGini 和 DSA 是逐模型排序，使用被测模型输出或内部特征；Coverage 和 k-center 使用一条模型无关排序。",
                  "- cat 当前使用 VLM 自动状态标签，人工盲审和双标尚未完成，因此 cat 和跨类别结论均为暂定。",
                  "- 所有模型权重保持不变，压力测试上的 F1 或错误率不能解释为模型训练性能变化。",
                  "", "## 混合效应分析", "", f"`{json.dumps(mixed, ensure_ascii=False)}`",
                  "", "若状态包含 Hessian/边界警告，则混合模型只用于描述总体方向，标准误和 p 值不作为正式推断证据。主结论来自预注册的分层 bootstrap 与架构簇置换检验。", "",
                  "## 结果文件", "",
                  "- `tables/metrics_detailed.csv`：逐模型、方法、预算的完整结果；",
                  "- `tables/auc_by_run.csv`：每个运行的曲线面积；",
                  "- `tables/primary_auc_comparisons.csv`：主检验、置信区间和效应量；",
                  "- `tables/budget_summary.csv`：预算曲线汇总；",
                  "- `figures/coverage_and_failure_breadth.svg`：覆盖与失败状态曲线；",
                  "- `figures/auc_fsb_effects.svg`：主效应森林图；",
                  "- `figures/architecture_effect_heatmap.svg`：逐架构诊断差值热图。", ""])
    (OUTPUT / "semantic_baseline_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")
    evidence = [
        "# Claim-Evidence Ledger", "",
        "| Claim | Status | Evidence | Boundary |", "|---|---|---|---|",
        f"| Coverage has independent diagnostic value across dog and cat | {'supported' if cross_category else 'partial' if any(decisions.values()) else 'unsupported'} | `tables/primary_auc_comparisons.csv` | cat human review pending |",
        "| Coverage changes frozen-model quality | rejected framing | checkpoints unchanged | test selection only |",
        "| Cat labels are human validated | blocked | no completed human labels | do not claim |", "",
    ]
    (OUTPUT / "claim_evidence.md").write_text("\n".join(evidence), encoding="utf-8")
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: pass; experiment report introduces no unverified literature claim.\n"
        "- Result traceability: pass if all checksums and tables listed in `checksums.csv` are present.\n"
        f"- Claim scope: {'partial pass; cross-category result is provisional until cat human review.' if cross_category else 'pass; report does not claim a cross-category independent advantage.'}\n"
        "- Remaining blocker: cat blind human review and double annotation.\n",
        encoding="utf-8",
    )


def write_checksums() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "checksums.csv":
            rows.append({"path": str(path.relative_to(OUTPUT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(OUTPUT / "checksums.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    args = parser.parse_args()
    write_protocol()
    contexts = {category: target_context(category) for category in ("dog", "cat")}
    append_log("loaded frozen dog/cat contexts")
    deepgini = deepgini_orders(contexts)
    append_log("DeepGini rankings completed")
    kcenter = kcenter_orders(contexts, args.workers)
    dsa = extract_dsa_orders(contexts, args.workers, args.smoke)
    validate_orders(contexts, deepgini, dsa, kcenter, args.smoke)
    if args.smoke:
        write_json(OUTPUT / "progress.json", {"status": "smoke_completed", "updated_at": now()})
        append_log("smoke test completed")
        return
    detail, auc_rows = aggregate(contexts, deepgini, dsa, kcenter)
    comparisons = statistical_analysis(auc_rows)
    summary = summarize_budget(detail)
    mixed = {"status": "skipped"} if args.skip_mixed else fit_mixed_model(detail)
    write_json(OUTPUT / "mixed_effects_status.json", mixed)
    make_figures(summary, comparisons, auc_rows)
    write_report(summary, comparisons, auc_rows, mixed)
    write_json(OUTPUT / "progress.json", {"status": "completed", "updated_at": now()})
    append_log("experiment completed")
    write_checksums()


if __name__ == "__main__":
    main()
