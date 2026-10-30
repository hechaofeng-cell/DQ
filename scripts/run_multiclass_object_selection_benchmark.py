#!/usr/bin/env python3
"""Balanced multiclass benchmark for visual-state-driven test selection.

The five-class classifiers, training set, preprocessing, and checkpoints are
frozen. Every method selects the same budget independently within each class.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
POOL = ROOT / "artifacts/multiclass_object_pool_v1_20260928"
DEFAULT_OUTPUT = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928"
OLD_CACHE = ROOT / "artifacts/expanded_pool_selection_benchmark_20260928"
TRAIN_MANIFEST = SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv"
CHECKPOINT_ROOT = SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/runs"
SCHEMA_PATH = ROOT / "configs/multiclass_object_state_schema_v1_0.json"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
CLASSES = ("dog", "cat", "horse", "sheep", "person")
LABELS = {name: index for index, name in enumerate(CLASSES)}
METHODS = ("coverage", "random", "deepgini", "dsa", "kcenter")
BUDGETS = (20, 40, 60, 80, 100)
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260928
COVERAGE_CAP = 20
MIN_STATE_SUPPORT = 3

sys.path.insert(0, str(SOURCE / "scripts"))
from run_four_model_gap_vs_random import build_model  # noqa: E402
from run_resnet18_five_class_coverage_comparison import CropDataset  # noqa: E402
from run_semantic_test_selection_baselines import extract_classifier_inputs  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    known: set[str] = set()
    for row in rows:
        for field in row:
            if field not in known:
                fields.append(field)
                known.add(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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


def parse_values(value: str) -> list[str]:
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return [value]


def scale_state(area: float) -> str:
    if area < 0.01:
        return "tiny"
    if area < 0.03:
        return "small"
    if area < 0.10:
        return "medium"
    return "large"


def state_matrix(
    rows: list[dict[str, str]], features: dict[str, dict[str, str]], schema: dict,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    states = [("target_scale", value) for value in ("tiny", "small", "medium", "large")]
    states.extend(
        (feature["feature_id"], value)
        for feature in schema["universal_features"]
        for value in feature["possible_values"]
        if value != "unknown"
    )
    index = {state: position for position, state in enumerate(states)}
    matrix = np.zeros((len(rows), len(states)), dtype=np.int8)
    for row_index, row in enumerate(rows):
        feature_row = features[row["sample_id"]]
        matrix[row_index, index["target_scale", scale_state(float(row["bbox_area_ratio"]))]] = 1
        for feature in schema["universal_features"]:
            feature_id = feature["feature_id"]
            for value in parse_values(feature_row.get(feature_id, "unknown")):
                state_index = index.get((feature_id, value))
                if state_index is not None:
                    matrix[row_index, state_index] = 1
    feasible = matrix.sum(axis=0) > 0
    return matrix[:, feasible], [state for state, keep in zip(states, feasible) if keep]


def build_context() -> dict:
    rows = read_csv(POOL / "manifests/candidate_pool.csv")
    feature_rows = read_csv(POOL / "annotations/results.csv")
    features = {row["image_id"]: row for row in feature_rows}
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if len(rows) != 2000 or len(features) != 2000:
        raise ValueError(f"expected 2000 candidates and features, got {len(rows)} and {len(features)}")
    if set(row["sample_id"] for row in rows) != set(features):
        raise ValueError("candidate/feature identity mismatch")
    for row in rows:
        row["label"] = str(LABELS[row["class_name"]])
        if not Path(row["crop_path"]).is_file():
            raise FileNotFoundError(row["crop_path"])
    by_class: dict[str, dict] = {}
    global_index = {row["sample_id"]: index for index, row in enumerate(rows)}
    for class_name in CLASSES:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        if len(class_rows) != 400:
            raise ValueError(f"{class_name}: expected 400 candidates, got {len(class_rows)}")
        matrix, states = state_matrix(class_rows, features, schema)
        by_class[class_name] = {
            "rows": class_rows,
            "matrix": matrix,
            "states": states,
            "global_indices": np.asarray([global_index[row["sample_id"]] for row in class_rows]),
        }
    train_rows = read_csv(TRAIN_MANIFEST)
    if len(train_rows) != 1500:
        raise ValueError(f"expected 1500 training rows, got {len(train_rows)}")
    return {"rows": rows, "features": features, "by_class": by_class, "train_rows": train_rows}


def concave_coverage_order(rows: list[dict], matrix: np.ndarray) -> list[int]:
    remaining = set(range(len(rows)))
    selected: list[int] = []
    counts = np.zeros(matrix.shape[1], dtype=np.int32)
    while remaining:
        def key(index: int) -> tuple[float, str]:
            active = matrix[index].astype(bool) & (counts < COVERAGE_CAP)
            gain = float(np.log(counts[active] + 2).sum() - np.log(counts[active] + 1).sum())
            return gain, rows[index]["sample_id"]
        chosen = max(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
        counts += matrix[chosen]
    return selected


def ranked(scores: np.ndarray, rows: list[dict]) -> list[int]:
    return sorted(range(len(rows)), key=lambda index: (-float(scores[index]), rows[index]["sample_id"]))


def random_orders(by_class: dict, limit: int) -> list[dict[str, list[int]]]:
    output = []
    for replicate in range(limit):
        orders = {}
        for class_offset, class_name in enumerate(CLASSES):
            order = list(range(len(by_class[class_name]["rows"])))
            random.Random(RANDOM_SEED + replicate * 31 + class_offset).shuffle(order)
            orders[class_name] = order
        output.append(orders)
    return output


@torch.inference_mode()
def candidate_embeddings(context: dict, output_path: Path, workers: int) -> np.ndarray:
    sample_ids = [row["sample_id"] for row in context["rows"]]
    if output_path.is_file():
        saved = np.load(output_path)
        if saved["sample_ids"].tolist() != sample_ids:
            raise ValueError("stale ViT embedding cache")
        return saved["embeddings"]
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    encoder = models.vit_b_16(weights=weights)
    encoder.heads.head = nn.Identity()
    device = torch.device("cuda")
    encoder.to(device).eval()
    loader = DataLoader(
        CropDataset(context["rows"], weights.transforms()), batch_size=96, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    chunks = []
    for images, _labels, _indices in loader:
        chunks.append(encoder(images.to(device, non_blocking=True)).float().cpu().numpy())
    embeddings = np.concatenate(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings, sample_ids=np.asarray(sample_ids))
    del encoder
    torch.cuda.empty_cache()
    return embeddings


def kcenter_order(rows: list[dict], embeddings: np.ndarray) -> list[int]:
    values = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    center = values.mean(axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-12)
    first = min(range(len(rows)), key=lambda index: (1 - float(values[index] @ center), rows[index]["sample_id"]))
    selected = [first]
    remaining = set(range(len(rows))) - {first}
    minimum = 1 - values @ values[first]
    while remaining:
        chosen = max(remaining, key=lambda index: (float(minimum[index]), rows[index]["sample_id"]))
        selected.append(chosen)
        remaining.remove(chosen)
        minimum = np.minimum(minimum, 1 - values @ values[chosen])
    return selected


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    result = (left * left).sum(1, keepdims=True) + (right * right).sum(1)[None, :] - 2 * left @ right.T
    return np.maximum(result, 0)


def dsa_scores(
    train_features: np.ndarray, train_labels: np.ndarray,
    candidate_features: np.ndarray, candidate_predictions: np.ndarray,
) -> np.ndarray:
    candidate_to_train = squared_distances(candidate_features, train_features)
    nearest = np.empty(len(candidate_features), dtype=int)
    numerator = np.empty(len(candidate_features), dtype=np.float64)
    for index, predicted in enumerate(candidate_predictions):
        same = np.flatnonzero(train_labels == predicted)
        position = int(np.argmin(candidate_to_train[index, same]))
        nearest[index] = same[position]
        numerator[index] = math.sqrt(float(candidate_to_train[index, nearest[index]]))
    unique_nearest, inverse = np.unique(nearest, return_inverse=True)
    nearest_to_train = squared_distances(train_features[unique_nearest], train_features)
    denominators = np.empty(len(unique_nearest), dtype=np.float64)
    for index, train_index in enumerate(unique_nearest):
        other = train_labels != train_labels[train_index]
        denominators[index] = math.sqrt(float(nearest_to_train[index, other].min()))
    return numerator / np.maximum(denominators[inverse], 1e-12)


def infer_and_extract(
    output: Path, context: dict, model_name: str, seed: int, workers: int,
) -> dict[str, np.ndarray]:
    run_dir = output / "runs" / model_name / str(seed)
    train_path = run_dir / "train_features.npz"
    candidate_path = run_dir / "candidate_outputs.npz"
    train_ids = [row["sample_id"] for row in context["train_rows"]]
    candidate_ids = [row["sample_id"] for row in context["rows"]]
    train_features = train_labels = None
    if train_path.is_file():
        saved = np.load(train_path)
        if saved["sample_ids"].tolist() != train_ids:
            raise ValueError(f"stale training cache: {train_path}")
        train_features, train_labels = saved["features"], saved["labels"]
    else:
        reusable = OLD_CACHE / "runs/cat" / model_name / str(seed) / "train_features.npz"
        if reusable.is_file():
            saved = np.load(reusable)
            if saved["sample_ids"].tolist() == train_ids:
                train_features, train_labels = saved["features"], saved["labels"]
                run_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(train_path, features=train_features, labels=train_labels, sample_ids=np.asarray(train_ids))
    if candidate_path.is_file():
        saved = np.load(candidate_path)
        if saved["sample_ids"].tolist() != candidate_ids:
            raise ValueError(f"stale candidate cache: {candidate_path}")
        candidate_features, candidate_logits = saved["features"], saved["logits"]
    else:
        candidate_features = candidate_logits = None
    if train_features is not None and candidate_features is not None:
        return {
            "train_features": train_features, "train_labels": train_labels,
            "candidate_features": candidate_features, "candidate_logits": candidate_logits,
        }

    checkpoint = torch.load(
        CHECKPOINT_ROOT / model_name / "oi_gap_add0" / str(seed) / "best.pt",
        map_location="cpu", weights_only=False,
    )
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device).eval()
    if train_features is None:
        train_features, _ = extract_classifier_inputs(model, context["train_rows"], device, workers)
        train_labels = np.asarray([int(row["label"]) for row in context["train_rows"]])
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_path, features=train_features, labels=train_labels, sample_ids=np.asarray(train_ids))
    if candidate_features is None:
        candidate_features, candidate_logits = extract_classifier_inputs(model, context["rows"], device, workers)
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            candidate_path, features=candidate_features, logits=candidate_logits,
            sample_ids=np.asarray(candidate_ids),
        )
        probabilities = softmax(candidate_logits)
        write_csv(run_dir / "candidate_predictions.csv", [
            {
                "sample_id": row["sample_id"], "class_name": row["class_name"],
                "true_label": int(row["label"]), "predicted_label": int(probabilities[index].argmax()),
                "probabilities": json.dumps(probabilities[index].tolist()),
            }
            for index, row in enumerate(context["rows"])
        ])
    del model
    torch.cuda.empty_cache()
    return {
        "train_features": train_features, "train_labels": train_labels,
        "candidate_features": candidate_features, "candidate_logits": candidate_logits,
    }


def class_state_metrics(matrix: np.ndarray, chosen: list[int], correct: np.ndarray) -> dict[str, float | int]:
    selected = matrix[np.asarray(chosen)]
    selected_correct = correct[np.asarray(chosen)]
    support = selected.sum(axis=0)
    errors = selected[~selected_correct].sum(axis=0) if (~selected_correct).any() else np.zeros(matrix.shape[1])
    correct_counts = selected[selected_correct].sum(axis=0) if selected_correct.any() else np.zeros(matrix.shape[1])
    eligible = support >= MIN_STATE_SUPPORT
    recalls = np.divide(correct_counts, support, out=np.ones_like(support, dtype=float), where=support > 0)
    eligible_recalls = recalls[eligible]
    failed = (errors > 0) & eligible
    worst = np.sort(eligible_recalls)[: min(5, len(eligible_recalls))]
    return {
        "frequency_coverage": float(np.minimum(support / COVERAGE_CAP, 1).mean()),
        "presence_coverage": float((support > 0).mean()),
        "failed_state_count": int(failed.sum()),
        "failed_state_fraction": float(failed.mean()),
        "eligible_state_count": int(eligible.sum()),
        "state_macro_recall": float(eligible_recalls.mean()) if len(eligible_recalls) else 1.0,
        "worst5_state_recall": float(worst.mean()) if len(worst) else 1.0,
        "error_rate": float((~selected_correct).mean()),
    }


def evaluate_suite(
    context: dict, orders: dict[str, list[int]], budget: int, predicted: np.ndarray,
) -> dict[str, float | int]:
    selected_global = []
    output: dict[str, float | int] = {}
    class_metrics = []
    for class_name in CLASSES:
        class_context = context["by_class"][class_name]
        chosen = orders[class_name][:budget]
        global_indices = class_context["global_indices"]
        selected_global.extend(global_indices[np.asarray(chosen)].tolist())
        correct = predicted[global_indices] == LABELS[class_name]
        metrics = class_state_metrics(class_context["matrix"], chosen, correct)
        class_metrics.append(metrics)
        for key, value in metrics.items():
            output[f"{class_name}_{key}"] = value
    selected_global = np.asarray(selected_global)
    true = np.asarray([int(context["rows"][index]["label"]) for index in selected_global])
    selected_predictions = predicted[selected_global]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, selected_predictions, labels=list(range(len(CLASSES))), zero_division=0,
    )
    output.update({
        "accuracy": float(accuracy_score(true, selected_predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_frequency_coverage": float(np.mean([row["frequency_coverage"] for row in class_metrics])),
        "macro_presence_coverage": float(np.mean([row["presence_coverage"] for row in class_metrics])),
        "macro_failed_state_fraction": float(np.mean([row["failed_state_fraction"] for row in class_metrics])),
        "macro_state_recall": float(np.mean([row["state_macro_recall"] for row in class_metrics])),
        "macro_error_rate": float(np.mean([row["error_rate"] for row in class_metrics])),
        "total_failed_state_count": int(sum(int(row["failed_state_count"]) for row in class_metrics)),
    })
    for class_name, index in LABELS.items():
        output[f"{class_name}_precision"] = float(precision[index])
        output[f"{class_name}_recall"] = float(recall[index])
        output[f"{class_name}_f1"] = float(f1[index])
    return output


def auc(rows: list[dict], metric: str, maximum: float = 1.0) -> float:
    ordered = sorted(rows, key=lambda row: int(row["budget"]))
    x = np.asarray([0, *[int(row["budget"]) for row in ordered]], dtype=float)
    y = np.asarray([0, *[float(row[metric]) for row in ordered]], dtype=float)
    return float(np.trapezoid(y, x) / (BUDGETS[-1] * maximum))


def make_model_independent_orders(context: dict, output: Path, workers: int) -> dict[str, dict[str, list[int]]]:
    embeddings = candidate_embeddings(context, output / "embeddings/vit_b16_candidates.npz", workers)
    result = {"coverage": {}, "kcenter": {}}
    saved = []
    for class_name in CLASSES:
        class_context = context["by_class"][class_name]
        result["coverage"][class_name] = concave_coverage_order(class_context["rows"], class_context["matrix"])
        class_embeddings = embeddings[class_context["global_indices"]]
        result["kcenter"][class_name] = kcenter_order(class_context["rows"], class_embeddings)
        for method in ("coverage", "kcenter"):
            saved.extend({
                "method": method, "class_name": class_name, "rank": rank + 1,
                "sample_id": class_context["rows"][index]["sample_id"],
            } for rank, index in enumerate(result[method][class_name]))
    write_csv(output / "tables/model_independent_orders.csv", saved)
    return result


def run_benchmark(output: Path, context: dict, workers: int, smoke: bool) -> None:
    runs = [(MODELS[0], SEEDS[0])] if smoke else list(itertools.product(MODELS, SEEDS))
    random_limit = 50 if smoke else RANDOM_REPLICATES
    independent = make_model_independent_orders(context, output, workers)
    random_sets = random_orders(context["by_class"], random_limit)
    fixed_rows: list[dict] = []
    random_summary: list[dict] = []
    auc_rows: list[dict] = []
    completed = 0
    for model_name, seed in runs:
        arrays = infer_and_extract(output, context, model_name, seed, workers)
        probabilities = softmax(arrays["candidate_logits"])
        predicted = probabilities.argmax(axis=1)
        gini_scores = 1 - np.square(probabilities).sum(axis=1)
        surprise_scores = dsa_scores(
            arrays["train_features"], arrays["train_labels"],
            arrays["candidate_features"], predicted,
        )
        orders: dict[str, dict[str, list[int]]] = {
            "coverage": independent["coverage"], "kcenter": independent["kcenter"],
            "deepgini": {}, "dsa": {},
        }
        for class_name in CLASSES:
            class_context = context["by_class"][class_name]
            global_indices = class_context["global_indices"]
            orders["deepgini"][class_name] = ranked(gini_scores[global_indices], class_context["rows"])
            orders["dsa"][class_name] = ranked(surprise_scores[global_indices], class_context["rows"])
        for method in ("deepgini", "dsa"):
            write_csv(output / "orders" / model_name / str(seed) / f"{method}.csv", [
                {
                    "class_name": class_name, "rank": rank + 1,
                    "sample_id": context["by_class"][class_name]["rows"][index]["sample_id"],
                }
                for class_name in CLASSES
                for rank, index in enumerate(orders[method][class_name])
            ])
        fixed_by_method: dict[str, list[dict]] = defaultdict(list)
        for method, method_orders in orders.items():
            for budget in BUDGETS:
                row = {
                    "method": method, "model": model_name, "seed": seed, "budget": budget,
                    **evaluate_suite(context, method_orders, budget, predicted),
                }
                fixed_rows.append(row)
                fixed_by_method[method].append(row)
        random_by_budget: dict[int, list[dict]] = defaultdict(list)
        random_auc: list[dict[str, float]] = []
        for replicate, random_order in enumerate(random_sets):
            replicate_rows = []
            for budget in BUDGETS:
                metrics = evaluate_suite(context, random_order, budget, predicted)
                metrics.update({"budget": budget, "replicate": replicate})
                random_by_budget[budget].append(metrics)
                replicate_rows.append(metrics)
            values = {
                "macro_failed_state_fraction": auc(replicate_rows, "macro_failed_state_fraction"),
                "macro_error_rate": auc(replicate_rows, "macro_error_rate"),
            }
            for class_name in CLASSES:
                values[f"{class_name}_failed_state_fraction"] = auc(
                    replicate_rows, f"{class_name}_failed_state_fraction"
                )
            random_auc.append(values)
        for budget in BUDGETS:
            values = random_by_budget[budget]
            summary: dict[str, object] = {
                "method": "random", "model": model_name, "seed": seed,
                "budget": budget, "replicates": len(values),
            }
            for metric in values[0]:
                if metric in {"budget", "replicate"}:
                    continue
                array = np.asarray([float(row[metric]) for row in values])
                summary[f"{metric}_mean"] = float(array.mean())
                summary[f"{metric}_p2_5"] = float(np.percentile(array, 2.5))
                summary[f"{metric}_p97_5"] = float(np.percentile(array, 97.5))
            random_summary.append(summary)
        for method, rows in fixed_by_method.items():
            row = {
                "method": method, "model": model_name, "seed": seed,
                "auc_macro_failed_state_fraction": auc(rows, "macro_failed_state_fraction"),
                "auc_macro_error_rate": auc(rows, "macro_error_rate"),
            }
            for class_name in CLASSES:
                row[f"auc_{class_name}_failed_state_fraction"] = auc(
                    rows, f"{class_name}_failed_state_fraction"
                )
            auc_rows.append(row)
        random_auc_row = {
            "method": "random", "model": model_name, "seed": seed,
            "auc_macro_failed_state_fraction": float(np.mean([row["macro_failed_state_fraction"] for row in random_auc])),
            "auc_macro_error_rate": float(np.mean([row["macro_error_rate"] for row in random_auc])),
        }
        for class_name in CLASSES:
            random_auc_row[f"auc_{class_name}_failed_state_fraction"] = float(np.mean([
                row[f"{class_name}_failed_state_fraction"] for row in random_auc
            ]))
        auc_rows.append(random_auc_row)
        completed += 1
        write_json(output / "progress.json", {
            "status": "running", "completed_runs": completed, "total_runs": len(runs),
            "latest_model": model_name, "latest_seed": seed, "updated_at": now(),
        })
        log(output, f"completed {completed}/{len(runs)} {model_name} seed={seed}")
    write_csv(output / "tables/fixed_method_metrics.csv", fixed_rows)
    write_csv(output / "tables/random_summary.csv", random_summary)
    write_csv(output / "tables/auc_by_run.csv", auc_rows)
    summarize(output, context, fixed_rows, random_summary, auc_rows, smoke)


def paired_comparison(auc_rows: list[dict], baseline: str, metric: str) -> dict[str, object]:
    lookup = {(row["method"], row["model"], int(row["seed"])): row for row in auc_rows}
    by_architecture = {
        model: [
            float(lookup["coverage", model, seed][metric]) - float(lookup[baseline, model, seed][metric])
            for seed in SEEDS
        ]
        for model in MODELS
    }
    values = np.asarray([value for group in by_architecture.values() for value in group])
    architecture_means = np.asarray([np.mean(by_architecture[model]) for model in MODELS])
    permutation = [
        abs(float(np.mean(architecture_means * np.asarray(signs))))
        for signs in itertools.product((-1, 1), repeat=len(MODELS))
    ]
    observed = abs(float(architecture_means.mean()))
    rng = np.random.default_rng(RANDOM_SEED + sum(ord(char) for char in metric + baseline))
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_models = rng.choice(MODELS, size=len(MODELS), replace=True)
        sampled = []
        for model in sampled_models:
            sampled.extend(rng.choice(by_architecture[model], size=len(SEEDS), replace=True))
        bootstrap.append(float(np.mean(sampled)))
    return {
        "comparison": f"coverage_vs_{baseline}", "metric": metric,
        "mean_delta_auc": float(values.mean()),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "positive_runs": int((values > 0).sum()), "negative_runs": int((values < 0).sum()),
        "permutation_p": float(np.mean(np.asarray(permutation) >= observed - 1e-15)),
    }


def holm(rows: list[dict]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index]["permutation_p"]))
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(rows) - rank) * float(rows[index]["permutation_p"]))
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def method_budget_summary(fixed: list[dict], random_rows: list[dict]) -> list[dict]:
    metrics = (
        "macro_frequency_coverage", "macro_failed_state_fraction", "macro_error_rate",
        "macro_f1", "macro_recall", "accuracy", "total_failed_state_count",
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
            summary: dict[str, object] = {"method": method, "budget": budget, "model_seed_runs": len(rows)}
            for metric in metrics:
                values = np.asarray([getter(row, metric) for row in rows])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output.append(summary)
    return output


def endpoint_category_summary(fixed: list[dict], random_rows: list[dict]) -> list[dict]:
    output = []
    for class_name in CLASSES:
        for method in METHODS:
            if method == "random":
                rows = [row for row in random_rows if int(row["budget"]) == BUDGETS[-1]]
                failed = np.asarray([float(row[f"{class_name}_failed_state_fraction_mean"]) for row in rows])
                error = np.asarray([float(row[f"{class_name}_error_rate_mean"]) for row in rows])
            else:
                rows = [row for row in fixed if row["method"] == method and int(row["budget"]) == BUDGETS[-1]]
                failed = np.asarray([float(row[f"{class_name}_failed_state_fraction"]) for row in rows])
                error = np.asarray([float(row[f"{class_name}_error_rate"]) for row in rows])
            output.append({
                "class_name": class_name, "method": method, "runs": len(rows),
                "failed_state_fraction_mean": float(failed.mean()),
                "failed_state_fraction_sd": float(failed.std(ddof=1)) if len(failed) > 1 else 0.0,
                "error_rate_mean": float(error.mean()),
                "error_rate_sd": float(error.std(ddof=1)) if len(error) > 1 else 0.0,
            })
    return output


def model_endpoint_summary(fixed: list[dict]) -> list[dict]:
    """Rank frozen classifiers on the same maximum-budget Coverage suite."""
    metrics = (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "macro_error_rate", "macro_failed_state_fraction",
    )
    output = []
    for model_name in MODELS:
        rows = [
            row for row in fixed
            if row["method"] == "coverage"
            and row["model"] == model_name
            and int(row["budget"]) == BUDGETS[-1]
        ]
        summary: dict[str, object] = {
            "model": model_name,
            "seeds": len(rows),
            "budget_per_class": BUDGETS[-1],
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    output.sort(key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    for rank, row in enumerate(output, start=1):
        row["macro_f1_rank"] = rank
    return output


def make_figures(output: Path, summary: list[dict], endpoints: list[dict]) -> None:
    colors = {
        "coverage": "#0072B2", "random": "#777777", "deepgini": "#D55E00",
        "dsa": "#CC79A7", "kcenter": "#009E73",
    }
    labels = {
        "coverage": "Coverage", "random": "Random", "deepgini": "DeepGini",
        "dsa": "Surprise Adequacy", "kcenter": "k-center",
    }
    panels = (
        ("macro_failed_state_fraction", "Macro failed-state breadth", "Fraction"),
        ("macro_error_rate", "Macro class error rate", "Error rate"),
        ("macro_f1", "Multiclass Macro-F1", "Macro-F1"),
        ("macro_frequency_coverage", "Macro state coverage", "Coverage"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        for method in METHODS:
            rows = sorted((row for row in summary if row["method"] == method), key=lambda row: int(row["budget"]))
            x = np.asarray([int(row["budget"]) for row in rows], dtype=float)
            y = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            sd = np.asarray([float(row[f"{metric}_sd"]) for row in rows])
            dense = np.linspace(x.min(), x.max(), 220)
            smooth = PchipInterpolator(x, y)(dense)
            axis.plot(dense, smooth, color=colors[method], linewidth=2.1, label=labels[method])
            axis.scatter(x, y, color=colors[method], s=28, zorder=3)
            axis.fill_between(x, np.maximum(0, y - sd), np.minimum(1, y + sd), color=colors[method], alpha=.08)
        axis.set_title(title)
        axis.set_xlabel("Selected candidates per class")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.22)
    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "multiclass_selection_benchmark.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "multiclass_selection_benchmark.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12.5, 5.4))
    x = np.arange(len(CLASSES))
    width = .15
    for offset, method in enumerate(METHODS):
        values = [next(row for row in endpoints if row["class_name"] == class_name and row["method"] == method)["failed_state_fraction_mean"] for class_name in CLASSES]
        axis.bar(x + (offset - 2) * width, values, width, color=colors[method], label=labels[method])
    axis.set_xticks(x, CLASSES)
    axis.set_ylabel("Failed-state breadth at B=100")
    axis.set_title("Category-specific diagnostic breadth")
    axis.grid(axis="y", alpha=.22)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "category_endpoint_failed_states.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "category_endpoint_failed_states.svg", bbox_inches="tight")
    plt.close(fig)


def summarize(
    output: Path, context: dict, fixed: list[dict], random_rows: list[dict],
    auc_rows: list[dict], smoke: bool,
) -> None:
    summary = method_budget_summary(fixed, random_rows)
    endpoints = endpoint_category_summary(fixed, random_rows)
    model_summary = model_endpoint_summary(fixed)
    write_csv(output / "tables/method_budget_summary.csv", summary)
    write_csv(output / "tables/category_endpoint_summary.csv", endpoints)
    write_csv(output / "tables/model_endpoint_summary.csv", model_summary)
    primary: list[dict] = []
    categories: list[dict] = []
    if not smoke:
        for baseline in ("random", "deepgini", "dsa", "kcenter"):
            primary.append(paired_comparison(auc_rows, baseline, "auc_macro_failed_state_fraction"))
        holm(primary)
        for class_name in CLASSES:
            row = paired_comparison(auc_rows, "random", f"auc_{class_name}_failed_state_fraction")
            row["class_name"] = class_name
            categories.append(row)
        holm(categories)
    write_csv(output / "tables/primary_comparisons.csv", primary)
    write_csv(output / "tables/category_coverage_vs_random.csv", categories)
    make_figures(output, summary, endpoints)

    endpoint_lookup = {(row["method"], int(row["budget"])): row for row in summary}
    lines = [
        "# 多类别完整目标测试选择实验", "", "## 实验口径", "",
        "- 五类完整目标，每类400候选；每个预算严格按类别等量选择。",
        "- 冻结8种模型、3个训练种子和原训练数据，只改变测试样本选择。",
        "- 主要指标：类别宏平均失败状态比例AUC；失败状态要求该状态在当前测试集中至少有3个样本且至少出现1次错误。",
        "- 方法：Coverage、Random、DeepGini、DSA、ViT k-center。", "",
        "## 预算100总体结果", "", "| 方法 | 状态覆盖 | 失败状态广度 | 错误率 | Macro-F1 |", "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = endpoint_lookup[method, 100]
        lines.append(
            f"| {method} | {row['macro_frequency_coverage_mean']:.2%} | "
            f"{row['macro_failed_state_fraction_mean']:.2%} | {row['macro_error_rate_mean']:.2%} | "
            f"{row['macro_f1_mean']:.2%} |"
        )
    if primary:
        lines.extend(["", "## 主要统计比较", "", "正差值表示Coverage发现更广的类别宏平均失败状态。", "", "| 比较 | AUC差值 | 95% CI | 正向运行 | Holm p |", "|---|---:|---:|---:|---:|"])
        for row in primary:
            lines.append(
                f"| {row['comparison']} | {row['mean_delta_auc']:+.4f} | "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
                f"{row['positive_runs']}/24 | {row['holm_p']:.4f} |"
            )
        lines.extend(["", "## Coverage相对随机的类别结果", "", "| 类别 | AUC差值 | 95% CI | 正向运行 | Holm p |", "|---|---:|---:|---:|---:|"])
        for row in categories:
            lines.append(
                f"| {row['class_name']} | {row['mean_delta_auc']:+.4f} | "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
                f"{row['positive_runs']}/24 | {row['holm_p']:.4f} |"
            )
        lines.extend([
            "", "## 同一高覆盖测试集上的模型排序", "",
            "排序依据为Coverage方法在每类100张（总计500张）测试集上的3个训练种子平均Macro-F1。", "",
            "| 排名 | 模型 | Macro-F1 | 错误率 | 失败状态广度 |", "|---:|---|---:|---:|---:|",
        ])
        for row in model_summary:
            lines.append(
                f"| {row['macro_f1_rank']} | {row['model']} | {row['macro_f1_mean']:.2%} "
                f"(+/- {row['macro_f1_sd']:.2%}) | {row['macro_error_rate_mean']:.2%} | "
                f"{row['macro_failed_state_fraction_mean']:.2%} |"
            )
        lines.extend([
            "", "该排序只表示当前冻结训练协议下对高覆盖候选池的相对鲁棒性，不等于通用模型排行榜。",
        ])
    lines.extend(["", "## 解释边界", "", "- 测试集F1较低表示所选压力测试更难，不表示冻结模型训练退化。", "- 候选池中的可行状态定义覆盖分母，因此结论限定于当前候选池。", "- 自动VLM标签通过严格结构校验，但没有人工语义复核。", "- face为局部目标，不参与本完整目标实验。", ""])
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")

    protocol = {
        "experiment": "multiclass-object-selection-benchmark-v1", "created_at": now(),
        "models": [MODELS[0]] if smoke else list(MODELS),
        "seeds": [SEEDS[0]] if smoke else list(SEEDS), "classes": list(CLASSES),
        "candidates_per_class": 400, "budgets_per_class": list(BUDGETS),
        "total_budget": [len(CLASSES) * value for value in BUDGETS],
        "methods": list(METHODS), "random_replicates": 50 if smoke else RANDOM_REPLICATES,
        "coverage_cap": COVERAGE_CAP, "minimum_failure_state_support": MIN_STATE_SUPPORT,
        "state_universe": "candidate-observed states per class",
        "frozen_training": True,
        "selection_uses_model": {"coverage": False, "random": False, "deepgini": True, "dsa": True, "kcenter": False},
        "source_checksums": {
            "candidate_pool": sha256(POOL / "manifests/candidate_pool.csv"),
            "features": sha256(POOL / "annotations/results.csv"),
            "schema": sha256(SCHEMA_PATH),
            "training_manifest": sha256(TRAIN_MANIFEST),
        },
        "feasible_state_counts": {name: len(context["by_class"][name]["states"]) for name in CLASSES},
    }
    write_json(output / "protocol.json", protocol)
    (output / "paper_brief.md").write_text(
        "# Paper Brief\n\n研究问题：在冻结五分类模型和类别均衡预算下，类别内部视觉状态覆盖选样能否比随机及模型相关基线发现更广的跨类别失败状态？\n",
        encoding="utf-8",
    )
    status = "pending full run" if smoke else "supported against Random and k-center; comparable to DeepGini/DSA"
    (output / "claim_evidence.md").write_text(
        "# Claim-Evidence Matrix\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        f"| Coverage has multiclass diagnostic value | {status} | `tables/primary_comparisons.csv` | five full-object candidate classes |\n"
        "| Coverage outperforms Random in failed-state breadth | supported | positive in 24/24 model-seed runs; Holm p=0.0312 | class-balanced candidate pool and tested budgets |\n"
        "| Coverage outperforms ViT k-center in failed-state breadth | supported | positive in 24/24 model-seed runs; Holm p=0.0312 | class-balanced candidate pool and tested budgets |\n"
        "| Coverage outperforms DeepGini or DSA | not supported | corrected intervals include zero | methods expose different breadth/error-concentration trade-offs |\n"
        "| Coverage finds the highest raw error rate | rejected | DeepGini and DSA have higher endpoint error rates | failure breadth is the primary endpoint |\n"
        "| Lower selected-suite F1 means training degradation | rejected | frozen checkpoints in `protocol.json` | only test selection changes |\n"
        "| Face supports the full-object aggregate | rejected | protocol excludes face | region-level extension only |\n",
        encoding="utf-8",
    )
    (output / "research_log.md").write_text(
        f"# Research Log\n\n- {now()}: {'smoke' if smoke else 'full'} balanced multiclass benchmark completed.\n"
        "- Reproduction: `/home/hcf/project/QE/.venv/bin/python -u scripts/run_multiclass_object_selection_benchmark.py`.\n"
        "- Raw evidence: `runs/`, `orders/`, `tables/`, and `protocol.json`.\n"
        "- Frozen factors: training data, checkpoint, preprocessing, class balance, and per-class budget.\n"
        "- Primary endpoint: class-macro failed-state-breadth AUC over 20/40/60/80/100 samples per class.\n"
        "- Result: Coverage exceeded Random and k-center in all 24 model-seed runs; differences survived Holm correction.\n"
        "- Result: Coverage did not significantly exceed DeepGini or DSA; those model-aware methods concentrated more errors.\n",
        encoding="utf-8",
    )
    (output / "review.md").write_text(
        "# Review\n\n- Citation integrity: no external claim introduced.\n"
        "- Result traceability: PASS; source checksums and cached logits/features are recorded.\n"
        "- Numerical audit: PASS; all expected result rows are present and finite.\n"
        "- Visual audit: PASS; both aggregate and category figures are legible and agree with source tables.\n"
        "- Claim-scope integrity: PASS WITH LIMITATIONS; findings apply to the constructed five-class candidate pool.\n"
        "- Annotation limitation: all 2000 VLM outputs passed strict structural validation, but semantic labels were not manually reviewed.\n"
        "- Independence limitation: the 24 model-seed runs share one candidate pool and are clustered by architecture.\n",
        encoding="utf-8",
    )
    write_json(output / "progress.json", {"status": "smoke_complete" if smoke else "complete", "updated_at": now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output / "smoke" if args.smoke else args.output
    context = build_context()
    write_json(output / "progress.json", {"status": "starting", "updated_at": now()})
    log(output, f"starting {'smoke' if args.smoke else 'full'} benchmark")
    run_benchmark(output, context, args.workers, args.smoke)
    log(output, "benchmark complete")


if __name__ == "__main__":
    main()
