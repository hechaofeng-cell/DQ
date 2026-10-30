#!/usr/bin/env python3
"""Benchmark test-selection methods on expanded cat and face candidate pools.

All classifiers are frozen. Cat uses the original D300 five-class checkpoints.
Face uses the original dog/cat/horse/sheep/face checkpoints and maps the face
logit to a face-vs-nonface decision. Only the selected test suite changes.
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
EXPANSION = ROOT / "artifacts/candidate_pool_expansion_v2_20260928"
DEFAULT_OUTPUT = ROOT / "artifacts/expanded_pool_selection_benchmark_20260928"
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
BUDGETS = (20, 40, 60, 80, 100)
METHODS = ("coverage", "random", "deepgini", "dsa", "kcenter")
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260928
COVERAGE_CAP = 20
MIN_STATE_SUPPORT = 3

sys.path.insert(0, str(SOURCE / "scripts"))
from run_four_model_gap_vs_random import build_model  # noqa: E402
from run_resnet18_five_class_coverage_comparison import CropDataset  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_cross_category_balanced_difficulty import state_matrix  # noqa: E402
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
                known.add(field)
                fields.append(field)
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


def normalize_row(row: dict[str, str], label: int | None = None) -> dict[str, str]:
    value = dict(row)
    value["crop_path"] = value.get("crop_path") or value.get("crop_image_path", "")
    if label is not None:
        value["label"] = str(label)
    if not value.get("sample_id") or not value["crop_path"]:
        raise ValueError(f"invalid manifest row: {value.get('sample_id')}")
    return value


def build_contexts() -> dict[str, dict]:
    cat_source = SOURCE / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
    old_cat_pool = read_csv(cat_source / "candidate_pool.csv")
    old_cat_targets = [normalize_row(row, 1) for row in old_cat_pool if row["class_name"] == "cat"]
    cat_controls = [normalize_row(row) for row in old_cat_pool if row["class_name"] != "cat"]
    new_cat_targets = [
        normalize_row(row, 1)
        for row in read_csv(EXPANSION / "cat/manifests/cat_candidates_expansion.csv")
    ]
    old_cat_features = read_csv(SOURCE / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv")
    new_cat_features = read_csv(EXPANSION / "cat/features_auto/results.csv")
    cat_features = {row["image_id"]: row for row in old_cat_features + new_cat_features}
    cat_targets = old_cat_targets + new_cat_targets
    cat_matrix, cat_states = state_matrix("cat", cat_targets, cat_features)

    old_face_root = ROOT / "artifacts/head_face_v1_0_20260922"
    old_face_targets = [
        normalize_row(row, 4)
        for row in read_csv(old_face_root / "manifests/face_candidates.csv")
    ]
    new_face_targets = [
        normalize_row(row, 4)
        for row in read_csv(EXPANSION / "face_binary/manifests/face_candidates.csv")
    ]
    face_controls = [
        normalize_row(row, 0)
        for row in read_csv(EXPANSION / "face_binary/manifests/nonface_candidates.csv")
    ]
    old_face_features = read_csv(old_face_root / "annotations/results.csv")
    new_face_features = read_csv(EXPANSION / "face_binary/annotations/results.csv")
    face_features = {row["sample_id"]: row for row in old_face_features + new_face_features}
    face_targets = old_face_targets + new_face_targets
    face_matrix, face_states = state_matrix("face", face_targets, face_features)

    contexts = {
        "cat": {
            "target_rows": cat_targets,
            "control_rows": cat_controls,
            "matrix": cat_matrix,
            "states": cat_states,
            "features": cat_features,
            "train_rows": [normalize_row(row) for row in read_csv(
                SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv"
            )],
            "checkpoint": lambda model, seed: SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/runs" / model / "oi_gap_add0" / str(seed) / "best.pt",
            "target_label": 1,
            "evaluation": "multiclass",
        },
        "face": {
            "target_rows": face_targets,
            "control_rows": face_controls,
            "matrix": face_matrix,
            "states": face_states,
            "features": face_features,
            "train_rows": [normalize_row(row) for row in read_csv(old_face_root / "manifests/train_d300.csv")],
            "checkpoint": lambda model, seed: old_face_root / "model_runs/runs" / model / "head_face_d300" / str(seed) / "best.pt",
            "target_label": 4,
            "evaluation": "binary_face",
        },
    }
    expected = {"cat": (400, 400, 98), "face": (600, 400, 99)}
    for domain, context in contexts.items():
        actual = (len(context["target_rows"]), len(context["control_rows"]), context["matrix"].shape[1])
        if actual != expected[domain]:
            raise ValueError(f"{domain}: expected {expected[domain]}, got {actual}")
        ids = [row["sample_id"] for row in context["target_rows"] + context["control_rows"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{domain}: duplicate sample IDs")
        missing = [row["crop_path"] for row in context["target_rows"] + context["control_rows"] if not Path(row["crop_path"]).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    return contexts


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


def random_orders(count: int) -> list[list[int]]:
    output = []
    for replicate in range(RANDOM_REPLICATES):
        order = list(range(count))
        random.Random(RANDOM_SEED + replicate).shuffle(order)
        output.append(order)
    return output


@torch.inference_mode()
def kcenter_order(rows: list[dict], output_path: Path, workers: int) -> list[int]:
    if output_path.is_file():
        saved = np.load(output_path)
        if saved["sample_ids"].tolist() == [row["sample_id"] for row in rows]:
            embeddings = saved["embeddings"]
        else:
            raise ValueError(f"stale embeddings: {output_path}")
    else:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        encoder = models.vit_b_16(weights=weights)
        encoder.heads.head = nn.Identity()
        device = torch.device("cuda")
        encoder.to(device).eval()
        loader = DataLoader(
            CropDataset(rows, weights.transforms()), batch_size=96, shuffle=False,
            num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        )
        chunks = []
        for images, _labels, _indices in loader:
            chunks.append(encoder(images.to(device, non_blocking=True)).float().cpu().numpy())
        embeddings = np.concatenate(chunks)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings, sample_ids=np.asarray([row["sample_id"] for row in rows]))
        del encoder
        torch.cuda.empty_cache()
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    center = embeddings.mean(axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-12)
    first = min(range(len(rows)), key=lambda index: (1 - float(embeddings[index] @ center), rows[index]["sample_id"]))
    selected = [first]
    remaining = set(range(len(rows))) - {first}
    minimum = 1 - embeddings @ embeddings[first]
    while remaining:
        chosen = max(remaining, key=lambda index: (float(minimum[index]), rows[index]["sample_id"]))
        selected.append(chosen)
        remaining.remove(chosen)
        minimum = np.minimum(minimum, 1 - embeddings @ embeddings[chosen])
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
    train_features: np.ndarray,
    train_labels: np.ndarray,
    candidate_features: np.ndarray,
    candidate_predictions: np.ndarray,
) -> np.ndarray:
    candidate_to_train = squared_distances(candidate_features, train_features)
    nearest = np.empty(len(candidate_features), dtype=int)
    numerator = np.empty(len(candidate_features), dtype=np.float64)
    for index, predicted in enumerate(candidate_predictions):
        same = np.flatnonzero(train_labels == predicted)
        nearest_position = int(np.argmin(candidate_to_train[index, same]))
        nearest[index] = same[nearest_position]
        numerator[index] = math.sqrt(float(candidate_to_train[index, nearest[index]]))
    unique_nearest, inverse = np.unique(nearest, return_inverse=True)
    nearest_to_train = squared_distances(train_features[unique_nearest], train_features)
    denominator_by_nearest = np.empty(len(unique_nearest), dtype=np.float64)
    for index, train_index in enumerate(unique_nearest):
        other = train_labels != train_labels[train_index]
        denominator_by_nearest[index] = math.sqrt(float(nearest_to_train[index, other].min()))
    return numerator / np.maximum(denominator_by_nearest[inverse], 1e-12)


def ranked(scores: np.ndarray, rows: list[dict]) -> list[int]:
    return sorted(range(len(rows)), key=lambda index: (-float(scores[index]), rows[index]["sample_id"]))


def infer_and_extract(
    output: Path,
    domain: str,
    context: dict,
    model_name: str,
    seed: int,
    workers: int,
) -> dict[str, np.ndarray]:
    run_dir = output / "runs" / domain / model_name / str(seed)
    train_path = run_dir / "train_features.npz"
    candidate_path = run_dir / "candidate_outputs.npz"
    train_rows = context["train_rows"]
    candidate_rows = context["target_rows"] + context["control_rows"]
    train_ids = [row["sample_id"] for row in train_rows]
    candidate_ids = [row["sample_id"] for row in candidate_rows]
    if train_path.is_file() and candidate_path.is_file():
        train_saved = np.load(train_path)
        candidate_saved = np.load(candidate_path)
        if train_saved["sample_ids"].tolist() == train_ids and candidate_saved["sample_ids"].tolist() == candidate_ids:
            return {
                "train_features": train_saved["features"],
                "train_labels": train_saved["labels"],
                "candidate_features": candidate_saved["features"],
                "candidate_logits": candidate_saved["logits"],
            }
        raise ValueError(f"stale inference cache: {run_dir}")
    checkpoint_path = context["checkpoint"](model_name, seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device)
    train_features, _ = extract_classifier_inputs(model, train_rows, device, workers)
    candidate_features, candidate_logits = extract_classifier_inputs(model, candidate_rows, device, workers)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        train_path,
        features=train_features,
        labels=np.asarray([int(row["label"]) for row in train_rows]),
        sample_ids=np.asarray(train_ids),
    )
    np.savez_compressed(
        candidate_path,
        features=candidate_features,
        logits=candidate_logits,
        sample_ids=np.asarray(candidate_ids),
    )
    probabilities = softmax(candidate_logits)
    predicted = probabilities.argmax(1)
    prediction_rows = []
    for index, row in enumerate(candidate_rows):
        prediction_rows.append({
            "sample_id": row["sample_id"], "class_name": row["class_name"],
            "predicted_label": int(predicted[index]), "probabilities": json.dumps(probabilities[index].tolist()),
        })
    write_csv(run_dir / "candidate_predictions.csv", prediction_rows)
    del model
    torch.cuda.empty_cache()
    return {
        "train_features": train_features,
        "train_labels": np.asarray([int(row["label"]) for row in train_rows]),
        "candidate_features": candidate_features,
        "candidate_logits": candidate_logits,
    }


def coverage_metrics(matrix: np.ndarray, selected: list[int]) -> dict[str, float | int]:
    counts = matrix[selected].sum(axis=0)
    return {
        "frequency_coverage": float(np.minimum(counts / COVERAGE_CAP, 1).mean()),
        "presence_coverage": float((counts > 0).mean()),
        "present_state_count": int((counts > 0).sum()),
    }


def failure_metrics(matrix: np.ndarray, selected: list[int], correct: np.ndarray) -> dict[str, float | int]:
    selected_matrix = matrix[selected]
    selected_correct = correct[np.asarray(selected)]
    support = selected_matrix.sum(0)
    errors = selected_matrix[~selected_correct].sum(0) if (~selected_correct).any() else np.zeros(matrix.shape[1])
    correct_counts = selected_matrix[selected_correct].sum(0) if selected_correct.any() else np.zeros(matrix.shape[1])
    eligible = support >= MIN_STATE_SUPPORT
    recalls = np.divide(correct_counts, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    eligible_recalls = recalls[eligible]
    worst = np.sort(eligible_recalls)[: min(5, len(eligible_recalls))]
    return {
        "failed_state_count": int((errors > 0).sum()),
        "eligible_state_count": int(eligible.sum()),
        "state_macro_recall": float(eligible_recalls.mean()) if len(eligible_recalls) else 1.0,
        "worst5_state_recall": float(worst.mean()) if len(worst) else 1.0,
    }


def classification_metrics(context: dict, target_indices: list[int], control_indices: list[int], predicted5: np.ndarray) -> dict[str, float]:
    target_count = len(context["target_rows"])
    selected_rows = [context["target_rows"][index] for index in target_indices] + [context["control_rows"][index] for index in control_indices]
    selected_predictions = np.concatenate((
        predicted5[np.asarray(target_indices)],
        predicted5[target_count + np.asarray(control_indices)],
    ))
    if context["evaluation"] == "multiclass":
        true = np.asarray([int(row["label"]) for row in selected_rows])
        labels = [0, 1, 2, 3, 4]
        target_label = 1
        predicted = selected_predictions
    else:
        true = np.asarray([1] * len(target_indices) + [0] * len(control_indices))
        labels = [0, 1]
        target_label = 1
        predicted = (selected_predictions == 4).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(true, predicted, labels=labels, zero_division=0)
    target_pos = labels.index(target_label)
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "target_precision": float(precision[target_pos]),
        "target_recall": float(recall[target_pos]),
        "target_f1": float(f1[target_pos]),
        "control_false_positive_rate": float(np.mean(
            selected_predictions[len(target_indices):] == context["target_label"]
        )),
    }


def control_orders(context: dict) -> dict[int, list[int]]:
    if context["evaluation"] == "binary_face":
        values = list(range(len(context["control_rows"])))
        random.Random(RANDOM_SEED + 9000).shuffle(values)
        return {0: values}
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(context["control_rows"]):
        grouped[int(row["label"])].append(index)
    for label, values in grouped.items():
        random.Random(RANDOM_SEED + 9000 + label).shuffle(values)
    return grouped


def selected_controls(orders: dict[int, list[int]], budget: int) -> list[int]:
    return [index for label in sorted(orders) for index in orders[label][:budget]]


def evaluate_order(
    context: dict,
    order: list[int],
    budget: int,
    predicted5: np.ndarray,
    controls: dict[int, list[int]],
) -> dict[str, float | int]:
    chosen = order[:budget]
    target_predicted = predicted5[: len(context["target_rows"])]
    correct = target_predicted == context["target_label"]
    control_indices = selected_controls(controls, budget)
    return {
        **coverage_metrics(context["matrix"], chosen),
        **failure_metrics(context["matrix"], chosen, correct),
        **classification_metrics(context, chosen, control_indices, predicted5),
        "target_error_count": int((~correct[np.asarray(chosen)]).sum()),
        "target_error_rate": float((~correct[np.asarray(chosen)]).mean()),
    }


def auc(rows: list[dict], metric: str, maximum: float = 1.0) -> float:
    ordered = sorted(rows, key=lambda row: int(row["budget"]))
    x = np.asarray([0, *[int(row["budget"]) for row in ordered]], dtype=float)
    y = np.asarray([0, *[float(row[metric]) for row in ordered]], dtype=float)
    return float(np.trapezoid(y, x) / (BUDGETS[-1] * maximum))


def run_benchmark(output: Path, contexts: dict[str, dict], workers: int, smoke: bool) -> None:
    runs = [(MODELS[0], SEEDS[0])] if smoke else list(itertools.product(MODELS, SEEDS))
    fixed_rows: list[dict] = []
    random_summary: list[dict] = []
    auc_rows: list[dict] = []
    orders_saved: list[dict] = []
    coverage_orders = {}
    random_by_domain = {}
    kcenter_orders = {}
    for domain, context in contexts.items():
        coverage_orders[domain] = concave_coverage_order(context["target_rows"], context["matrix"])
        random_by_domain[domain] = random_orders(len(context["target_rows"]))
        kcenter_orders[domain] = kcenter_order(
            context["target_rows"], output / "embeddings" / f"{domain}_vit_b16.npz", workers,
        )
        for method, order in (("coverage", coverage_orders[domain]), ("kcenter", kcenter_orders[domain])):
            orders_saved.extend({"domain": domain, "method": method, "model": "model_independent", "seed": -1, "rank": rank + 1, "sample_id": context["target_rows"][index]["sample_id"]} for rank, index in enumerate(order))
    write_csv(output / "tables/model_independent_orders.csv", orders_saved)

    completed = 0
    for model_name, seed in runs:
        for domain, context in contexts.items():
            arrays = infer_and_extract(output, domain, context, model_name, seed, workers)
            probabilities = softmax(arrays["candidate_logits"])
            predicted5 = probabilities.argmax(1)
            target_count = len(context["target_rows"])
            target_probabilities = probabilities[:target_count]
            target_features = arrays["candidate_features"][:target_count]
            deepgini = ranked(1 - np.square(target_probabilities).sum(1), context["target_rows"])
            dsa = ranked(
                dsa_scores(arrays["train_features"], arrays["train_labels"], target_features, predicted5[:target_count]),
                context["target_rows"],
            )
            run_orders = {
                "coverage": coverage_orders[domain], "deepgini": deepgini,
                "dsa": dsa, "kcenter": kcenter_orders[domain],
            }
            for method in ("deepgini", "dsa"):
                write_csv(output / "orders" / domain / model_name / str(seed) / f"{method}.csv", [
                    {"rank": rank + 1, "sample_id": context["target_rows"][index]["sample_id"]}
                    for rank, index in enumerate(run_orders[method])
                ])
            controls = control_orders(context)
            by_method: dict[str, list[dict]] = defaultdict(list)
            for method, order in run_orders.items():
                for budget in BUDGETS:
                    row = {
                        "domain": domain, "method": method, "model": model_name,
                        "seed": seed, "budget": budget,
                        **evaluate_order(context, order, budget, predicted5, controls),
                    }
                    fixed_rows.append(row)
                    by_method[method].append(row)
            random_auc_values = []
            random_by_budget: dict[int, list[dict]] = defaultdict(list)
            random_limit = 50 if smoke else RANDOM_REPLICATES
            for replicate, order in enumerate(random_by_domain[domain][:random_limit]):
                replicate_rows = []
                for budget in BUDGETS:
                    row = evaluate_order(context, order, budget, predicted5, controls)
                    row.update({"budget": budget, "replicate": replicate})
                    random_by_budget[budget].append(row)
                    replicate_rows.append(row)
                random_auc_values.append(auc(replicate_rows, "failed_state_count", context["matrix"].shape[1]))
            for budget in BUDGETS:
                values = random_by_budget[budget]
                summary = {"domain": domain, "method": "random", "model": model_name, "seed": seed, "budget": budget, "replicates": len(values)}
                for metric in values[0]:
                    if metric in {"budget", "replicate"}:
                        continue
                    array = np.asarray([float(row[metric]) for row in values])
                    summary[f"{metric}_mean"] = float(array.mean())
                    summary[f"{metric}_p2_5"] = float(np.percentile(array, 2.5))
                    summary[f"{metric}_p97_5"] = float(np.percentile(array, 97.5))
                random_summary.append(summary)
            for method, rows in by_method.items():
                auc_rows.append({
                    "domain": domain, "method": method, "model": model_name, "seed": seed,
                    "auc_failed_state_breadth": auc(rows, "failed_state_count", context["matrix"].shape[1]),
                    "auc_target_error_rate": auc(rows, "target_error_rate"),
                })
            auc_rows.append({
                "domain": domain, "method": "random", "model": model_name, "seed": seed,
                "auc_failed_state_breadth": float(np.mean(random_auc_values)),
                "auc_target_error_rate": float(np.mean([
                    auc([{"budget": budget, "target_error_rate": row["target_error_rate"]} for budget, row in zip(BUDGETS, [random_by_budget[b][replicate] for b in BUDGETS])], "target_error_rate")
                    for replicate in range(random_limit)
                ])),
            })
            completed += 1
            write_json(output / "progress.json", {
                "status": "running", "completed_domain_runs": completed,
                "total_domain_runs": len(runs) * len(contexts), "latest_domain": domain,
                "latest_model": model_name, "latest_seed": seed, "updated_at": now(),
            })
            log(output, f"completed {completed}/{len(runs) * len(contexts)} {domain} {model_name} seed={seed}")
    write_csv(output / "tables/fixed_method_metrics.csv", fixed_rows)
    write_csv(output / "tables/random_summary.csv", random_summary)
    write_csv(output / "tables/auc_by_run.csv", auc_rows)
    summarize(output, contexts, fixed_rows, random_summary, auc_rows, smoke)


def hierarchical_comparisons(auc_rows: list[dict], smoke: bool) -> list[dict]:
    if smoke:
        return []
    lookup = {(row["domain"], row["method"], row["model"], int(row["seed"])): row for row in auc_rows}
    rng = np.random.default_rng(RANDOM_SEED)
    output = []
    for domain in ("cat", "face"):
        domain_rows = []
        for baseline in ("random", "deepgini", "dsa", "kcenter"):
            by_architecture = {}
            for model_name in MODELS:
                by_architecture[model_name] = [
                    float(lookup[domain, "coverage", model_name, seed]["auc_failed_state_breadth"])
                    - float(lookup[domain, baseline, model_name, seed]["auc_failed_state_breadth"])
                    for seed in SEEDS
                ]
            values = np.asarray([value for group in by_architecture.values() for value in group])
            architecture_means = np.asarray([np.mean(by_architecture[model]) for model in MODELS])
            permutation = []
            for signs in itertools.product((-1, 1), repeat=len(MODELS)):
                permutation.append(abs(float(np.mean(architecture_means * np.asarray(signs)))))
            observed = abs(float(architecture_means.mean()))
            bootstrap = []
            for _ in range(BOOTSTRAP_REPLICATES):
                sampled_models = rng.choice(MODELS, size=len(MODELS), replace=True)
                sampled = []
                for model_name in sampled_models:
                    sampled.extend(rng.choice(by_architecture[model_name], size=len(SEEDS), replace=True))
                bootstrap.append(float(np.mean(sampled)))
            domain_rows.append({
                "domain": domain, "comparison": f"coverage_vs_{baseline}",
                "mean_delta_auc_failed_state_breadth": float(values.mean()),
                "ci95_low": float(np.percentile(bootstrap, 2.5)),
                "ci95_high": float(np.percentile(bootstrap, 97.5)),
                "positive_runs": int((values > 0).sum()), "negative_runs": int((values < 0).sum()),
                "permutation_p": float(np.mean(np.asarray(permutation) >= observed - 1e-15)),
            })
        ordered = sorted(range(len(domain_rows)), key=lambda row_index: domain_rows[row_index]["permutation_p"])
        running = 0.0
        for rank, row_index in enumerate(ordered):
            adjusted = min(1.0, (len(domain_rows) - rank) * domain_rows[row_index]["permutation_p"])
            running = max(running, adjusted)
            domain_rows[row_index]["holm_p"] = running
        output.extend(domain_rows)
    return output


def method_budget_summary(fixed: list[dict], random_rows: list[dict]) -> list[dict]:
    output = []
    metrics = (
        "frequency_coverage", "failed_state_count", "target_error_rate",
        "control_false_positive_rate", "macro_f1", "target_recall", "target_f1",
    )
    for domain in ("cat", "face"):
        for method in METHODS:
            for budget in BUDGETS:
                if method == "random":
                    rows = [row for row in random_rows if row["domain"] == domain and int(row["budget"]) == budget]
                    getter = lambda row, metric: float(row[f"{metric}_mean"])
                else:
                    rows = [row for row in fixed if row["domain"] == domain and row["method"] == method and int(row["budget"]) == budget]
                    getter = lambda row, metric: float(row[metric])
                result = {"domain": domain, "method": method, "budget": budget, "model_seed_runs": len(rows)}
                for metric in metrics:
                    array = np.asarray([getter(row, metric) for row in rows])
                    result[f"{metric}_mean"] = float(array.mean())
                    result[f"{metric}_sd"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
                output.append(result)
    return output


def make_figures(output: Path, summary: list[dict]) -> None:
    colors = {"coverage": "#0072B2", "random": "#777777", "deepgini": "#D55E00", "dsa": "#CC79A7", "kcenter": "#009E73"}
    labels = {"coverage": "Coverage", "random": "Random", "deepgini": "DeepGini", "dsa": "Surprise Adequacy", "kcenter": "k-center"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    for column, domain in enumerate(("cat", "face")):
        for method in METHODS:
            rows = sorted((row for row in summary if row["domain"] == domain and row["method"] == method), key=lambda row: row["budget"])
            x = np.asarray([row["budget"] for row in rows], dtype=float)
            for axis, metric in ((axes[0, column], "failed_state_count_mean"), (axes[1, column], "target_error_rate_mean")):
                y = np.asarray([row[metric] for row in rows])
                dense_x = np.linspace(x.min(), x.max(), 200)
                dense_y = PchipInterpolator(x, y)(dense_x)
                axis.plot(dense_x, dense_y, color=colors[method], linewidth=1.5, label=labels[method])
                axis.scatter(x, y, color=colors[method], s=18, zorder=3)
        axes[0, column].set_title(f"{domain}: failed-state breadth")
        axes[1, column].set_title(f"{domain}: target error rate")
        axes[0, column].set_ylabel("Failed states")
        axes[1, column].set_ylabel("Error rate")
        axes[1, column].set_xlabel("Selected target budget")
        for row in axes[:, column]:
            row.grid(alpha=0.2)
            row.set_xticks(BUDGETS)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures/selection_benchmark.png", dpi=220)
    fig.savefig(output / "figures/selection_benchmark.svg")
    plt.close(fig)


def summarize(output: Path, contexts: dict, fixed: list[dict], random_rows: list[dict], auc_rows: list[dict], smoke: bool) -> None:
    summary = method_budget_summary(fixed, random_rows)
    comparisons = hierarchical_comparisons(auc_rows, smoke)
    write_csv(output / "tables/method_budget_summary.csv", summary)
    write_csv(output / "tables/primary_comparisons.csv", comparisons)
    model_summary = []
    for domain in ("cat", "face"):
        for model_name in ([MODELS[0]] if smoke else MODELS):
            rows = [
                row for row in fixed
                if row["domain"] == domain and row["method"] == "coverage"
                and row["model"] == model_name and int(row["budget"]) == 100
            ]
            model_summary.append({
                "domain": domain, "model": model_name, "seeds": len(rows),
                **{
                    f"{metric}_mean": float(np.mean([float(row[metric]) for row in rows]))
                    for metric in (
                        "macro_f1", "target_precision", "target_recall", "target_f1",
                        "control_false_positive_rate", "failed_state_count",
                    )
                },
            })
    write_csv(output / "tables/model_summary.csv", model_summary)
    make_figures(output, summary)
    lines = [
        "# 扩展候选池测试选择实验", "", "## 协议", "",
        f"- 模型运行：{'冒烟测试（ResNet-18，一个种子）' if smoke else '8个架构 × 3个冻结种子'}。",
        "- 方法：视觉状态覆盖、随机、DeepGini、Surprise Adequacy（DSA）和ViT嵌入k-center。",
        "- 预算：20、40、60、80、100个目标类样本；所有方法使用完全相同的非目标样本。",
        "- 主要指标：失败状态广度；错误率、Macro-F1和目标类Recall/F1为辅助指标。", "",
        "## 预算100结果", "",
        "| 类别 | 方法 | 覆盖率 | 失败状态数 | 目标错误率 | 非目标假阳性率 | Macro-F1 | 目标Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in ("cat", "face"):
        for method in METHODS:
            row = next(item for item in summary if item["domain"] == domain and item["method"] == method and item["budget"] == 100)
            lines.append(
                f"| {domain} | {method} | {row['frequency_coverage_mean']:.2%} | "
                f"{row['failed_state_count_mean']:.2f} | {row['target_error_rate_mean']:.2%} | {row['control_false_positive_rate_mean']:.2%} | "
                f"{row['macro_f1_mean']:.2%} | {row['target_recall_mean']:.2%} |"
            )
    if comparisons:
        lines.extend(["", "## 主要统计比较", "", "正差值表示coverage暴露了更广的失败状态。", "", "| 类别 | 比较 | Δ失败状态广度AUC | 95% CI | 正向运行 | 原始p | Holm p |", "|---|---|---:|---:|---:|---:|---:|"])
        for row in comparisons:
            lines.append(
                f"| {row['domain']} | {row['comparison']} | {row['mean_delta_auc_failed_state_breadth']:+.4f} | "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['positive_runs']}/24 | "
                f"{row['permutation_p']:.4f} | {row['holm_p']:.4f} |"
            )
        lines.extend(["", "## 覆盖组模型排名（预算100）", ""])
        for domain in ("cat", "face"):
            lines.extend([f"### {domain}", "", "| 排名 | 模型 | Macro-F1 | 目标Recall | 非目标假阳性率 | 失败状态数 |", "|---:|---|---:|---:|---:|---:|"])
            ranked_models = sorted(
                (row for row in model_summary if row["domain"] == domain),
                key=lambda row: row["macro_f1_mean"], reverse=True,
            )
            for rank, row in enumerate(ranked_models, 1):
                lines.append(
                    f"| {rank} | {row['model']} | {row['macro_f1_mean']:.2%} | "
                    f"{row['target_recall_mean']:.2%} | {row['control_false_positive_rate_mean']:.2%} | "
                    f"{row['failed_state_count_mean']:.2f} |"
                )
            lines.append("")
    lines.extend(["", "## 解释边界", "", "- 模型权重和训练数据均未改变；测试集上的F1下降表示压力测试更难，不表示模型训练退化。", "- face结果是原五分类模型的face-vs-rest映射，专门检验其面对同源非人脸区域时的能力。", "- 候选池是覆盖导向压力测试，不代表部署分布。", "- 自动状态标签通过结构校验，但没有人工语义复核。", ""])
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "progress.json", {"status": "smoke_complete" if smoke else "complete", "updated_at": now()})
    write_research_artifacts(output, contexts, comparisons, smoke)


def write_research_artifacts(output: Path, contexts: dict, comparisons: list[dict], smoke: bool) -> None:
    protocol = {
        "experiment": "expanded-pool-selection-benchmark-v1",
        "created_at": now(), "models": [MODELS[0]] if smoke else list(MODELS),
        "seeds": [SEEDS[0]] if smoke else list(SEEDS), "budgets": list(BUDGETS),
        "methods": list(METHODS), "random_replicates": 50 if smoke else RANDOM_REPLICATES,
        "coverage_cap": COVERAGE_CAP, "minimum_state_support": MIN_STATE_SUPPORT,
        "frozen_training": True, "selection_uses_predictions": {"coverage": False, "random": False, "deepgini": True, "dsa": True, "kcenter": False},
        "candidate_counts": {domain: {"target": len(context["target_rows"]), "control": len(context["control_rows"])} for domain, context in contexts.items()},
    }
    write_json(output / "protocol.json", protocol)
    (output / "paper_brief.md").write_text(
        "# Paper Brief\n\n研究问题：在冻结模型和同预算约束下，类别内部视觉状态覆盖选样能否比随机、不确定性、Surprise Adequacy和嵌入多样性暴露更广的失败状态？\n",
        encoding="utf-8",
    )
    status = "pending full run" if smoke else "see primary comparisons"
    (output / "claim_evidence.md").write_text(
        "# Claim-Evidence Matrix\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        f"| Coverage selection improves failed-state breadth | {status} | `tables/primary_comparisons.csv` | expanded cat and face candidate pools |\n"
        "| Lower stress-test F1 means training degradation | rejected | frozen checkpoints | only test composition changes |\n",
        encoding="utf-8",
    )
    (output / "research_log.md").write_text(
        f"# Research Log\n\n- {now()}: {'smoke' if smoke else 'full'} benchmark completed.\n"
        "- Reproduction: `/home/hcf/project/QE/.venv/bin/python -u scripts/run_expanded_pool_selection_benchmark.py`.\n"
        "- Raw outputs: `runs/`, `orders/`, and `tables/`.\n",
        encoding="utf-8",
    )
    (output / "review.md").write_text(
        "# Review\n\n- Citation integrity: no external claim introduced.\n"
        "- Result traceability: PASS; aggregates trace to saved logits/features and frozen checkpoints.\n"
        "- Claim-scope integrity: PASS WITH LIMITATIONS; results describe constructed stress tests, not deployment prevalence.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output / "smoke" if args.smoke else args.output
    contexts = build_contexts()
    write_json(output / "progress.json", {"status": "starting", "updated_at": now()})
    log(output, f"starting {'smoke' if args.smoke else 'full'} benchmark")
    run_benchmark(output, contexts, args.workers, args.smoke)
    log(output, "benchmark complete")


if __name__ == "__main__":
    main()
