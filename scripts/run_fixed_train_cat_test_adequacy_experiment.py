#!/usr/bin/env python3
"""Run the cat_v1.0 fixed-training visual test adequacy experiment."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from openimages_cua_vsl import StateSchema
from run_fixed_train_test_adequacy_experiment import (
    coverage_order, coverage_values, dog_failure_metrics as target_failure_metrics,
    now, read_csv, sha256,
)
from run_four_model_gap_vs_random import build_model
from run_resnet18_dog_coverage_comparison import write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CLASS_SPECS, CropDataset, predict


ROOT = Path(__file__).resolve().parents[1]
GRADIENT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
DATA = ROOT / "data/openimages_v7_dog_gap_v1"
FEATURES = ROOT / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv"
SCHEMA_PATH = ROOT / "configs/cat_feature_schema_v1_0.json"
OUTPUT = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
BUDGETS = (20, 40, 60, 80, 100)
RANDOM_REPLICATES = 1000
RANDOM_SEED_BASE = 20260920
TARGET_CLASS = "cat"
TARGET_LABEL = 1
CLASS_NAMES = tuple(name for name, _ in CLASS_SPECS)
MIN_STATE_SUPPORT = 3


def prepare_candidates() -> tuple[list[dict], list[dict], dict[str, dict], StateSchema]:
    rows = read_csv(DATA / "manifests/test.csv") + read_csv(DATA / "manifests/cat_difficulty_test.csv")
    for row in rows:
        row["candidate_source"] = "difficulty_test" if row["split"] == "cat_difficulty_test" else "main_test"
    expected = {name: (200 if name == TARGET_CLASS else 100) for name in CLASS_NAMES}
    counts = {name: sum(row["class_name"] == name for row in rows) for name in CLASS_NAMES}
    if counts != expected or len(rows) != 600 or len({row["sample_id"] for row in rows}) != 600:
        raise ValueError(f"invalid candidate pool: {counts}")
    if any(not Path(row["crop_path"]).is_file() for row in rows):
        raise FileNotFoundError("candidate crop is missing")
    target_rows = [row for row in rows if row["class_name"] == TARGET_CLASS]
    feature_rows = read_csv(FEATURES)
    features = {row["image_id"]: row for row in feature_rows}
    if len(features) != 200 or set(features) != {row["sample_id"] for row in target_rows}:
        raise ValueError("cat features do not match candidate targets")
    if any(row["feature_status"] != "ok" for row in feature_rows):
        raise ValueError("cat feature extraction contains failures")

    train = read_csv(GRADIENT / "manifests/train_oi_gap_add0.csv")
    train_ids = {row["image_id"] for row in train}
    test_ids = {row["image_id"] for row in rows}
    if train_ids & test_ids:
        raise ValueError("image ID leakage between D300 training and cat test pool")
    train_hashes = {row.get("image_sha256") for row in train if row.get("image_sha256")}
    test_hashes = {row.get("image_sha256") for row in rows if row.get("image_sha256")}
    if train_hashes & test_hashes:
        raise ValueError("image SHA-256 leakage between D300 training and cat test pool")
    return rows, target_rows, features, StateSchema.from_path(SCHEMA_PATH)


def state_matrix(target_rows: list[dict], features: dict[str, dict], schema: StateSchema) -> np.ndarray:
    states = schema.coverage_states()
    state_to_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(target_rows), len(states)), dtype=np.int8)
    for row_index, row in enumerate(target_rows):
        for pair in schema.state_pairs(features[row["sample_id"]]):
            if pair in state_to_index:
                matrix[row_index, state_to_index[pair]] = 1
    return matrix


def build_protocol() -> dict:
    rows, target_rows, features, schema = prepare_candidates()
    matrix = state_matrix(target_rows, features, schema)
    gap_order = coverage_order(target_rows, matrix)
    random_orders = []
    for replicate in range(RANDOM_REPLICATES):
        order = list(range(len(target_rows)))
        random.Random(RANDOM_SEED_BASE + replicate).shuffle(order)
        random_orders.append(order)
    non_target_orders = {}
    for class_index, class_name in enumerate(name for name in CLASS_NAMES if name != TARGET_CLASS):
        class_rows = sorted((row for row in rows if row["class_name"] == class_name), key=lambda row: row["sample_id"])
        random.Random(RANDOM_SEED_BASE + 10000 + class_index).shuffle(class_rows)
        non_target_orders[class_name] = class_rows

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "candidate_pool.csv", rows)
    write_csv(OUTPUT / "orders/gap_cat_order.csv", [
        {"rank": rank + 1, **target_rows[index]} for rank, index in enumerate(gap_order)
    ])
    write_csv(OUTPUT / "orders/random_cat_orders.csv", [
        {"replicate": replicate, "seed": RANDOM_SEED_BASE + replicate,
         "rank": rank + 1, "sample_id": target_rows[index]["sample_id"]}
        for replicate, order in enumerate(random_orders) for rank, index in enumerate(order)
    ])
    write_csv(OUTPUT / "orders/non_cat_order.csv", [
        {"rank": rank + 1, **row}
        for class_name in CLASS_NAMES if class_name != TARGET_CLASS
        for rank, row in enumerate(non_target_orders[class_name])
    ])

    selection_rows = []
    for budget in BUDGETS:
        gap_frequency, gap_presence = coverage_values(matrix, gap_order[:budget])
        random_frequency, random_presence = [], []
        for order in random_orders:
            frequency, presence = coverage_values(matrix, order[:budget])
            random_frequency.append(frequency)
            random_presence.append(presence)
        selection_rows.append({
            "cat_per_class": budget, "total_test_targets": 5 * budget,
            "gap_frequency_coverage": gap_frequency,
            "random_frequency_coverage_mean": float(np.mean(random_frequency)),
            "random_frequency_coverage_sd": float(np.std(random_frequency, ddof=1)),
            "random_frequency_coverage_p2_5": float(np.percentile(random_frequency, 2.5)),
            "random_frequency_coverage_p97_5": float(np.percentile(random_frequency, 97.5)),
            "gap_presence_coverage": gap_presence,
            "random_presence_coverage_mean": float(np.mean(random_presence)),
            "random_presence_coverage_sd": float(np.std(random_presence, ddof=1)),
        })
        target_ids = {target_rows[index]["sample_id"] for index in gap_order[:budget]}
        manifest = [row for row in target_rows if row["sample_id"] in target_ids]
        for class_name in CLASS_NAMES:
            if class_name != TARGET_CLASS:
                manifest.extend(non_target_orders[class_name][:budget])
        write_csv(OUTPUT / f"manifests/gap_balanced_{budget}_per_class.csv", manifest)
    write_csv(OUTPUT / "tables/selection_coverage.csv", selection_rows)

    checkpoints = [
        GRADIENT / "runs" / model / "oi_gap_add0" / str(seed) / "best.pt"
        for model in MODELS for seed in SEEDS
    ]
    if any(not path.is_file() for path in checkpoints):
        raise FileNotFoundError("a frozen D300 checkpoint is missing")
    write_json(OUTPUT / "protocol.json", {
        "experiment_id": "fixed-train-cat-visual-test-adequacy-20260920",
        "created_at": now(),
        "target_class": TARGET_CLASS,
        "state_protocol": "cat_v1.0",
        "frozen_training_condition": "oi_gap_add0 / D300",
        "fixed_checkpoints": len(checkpoints),
        "models": list(MODELS), "seeds": list(SEEDS),
        "candidate_pool": {name: (200 if name == TARGET_CLASS else 100) for name in CLASS_NAMES},
        "budgets_per_class": list(BUDGETS),
        "random_replicates": RANDOM_REPLICATES,
        "selection_blinding": "cat_v1.0 states only; no model prediction is used for selection",
        "success_criterion": "higher coverage and broader failure-state discovery than the random 95% interval at equal budget",
        "input_sha256": {
            "D300_train_manifest": sha256(GRADIENT / "manifests/train_oi_gap_add0.csv"),
            "main_test_manifest": sha256(DATA / "manifests/test.csv"),
            "cat_difficulty_manifest": sha256(DATA / "manifests/cat_difficulty_test.csv"),
            "cat_features": sha256(FEATURES),
            "schema": sha256(SCHEMA_PATH),
        },
    })
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n- Thesis: visual-state coverage can guide failure-revealing tests across object categories.\n"
        "- Cat replication: fixed D300 models, 200 cat candidates, four non-cat classes with 100 targets each, and 1000 random controls.\n"
        "- Boundary: stress-test metrics do not estimate deployment prevalence.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        f"# Research Log\n\n- Protocol frozen: {now()}\n- Target: cat with cat_v1.0.\n"
        "- Fixed factors: D300 training set, 24 checkpoints, candidate pool, non-cat orders, preprocessing, and label space.\n"
        "- Reproduction: `.venv/bin/python -u scripts/run_fixed_train_cat_test_adequacy_experiment.py`.\n",
        encoding="utf-8",
    )
    return {"rows": rows, "target_rows": target_rows, "matrix": matrix,
            "gap_order": gap_order, "random_orders": random_orders,
            "non_target_orders": non_target_orders}


def classification_metrics(true: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted_support, out=np.zeros(5), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
        "error_count": int(matrix.sum() - tp.sum()), "error_rate": float(1 - tp.sum() / matrix.sum()),
        "cat_f1": float(f1[TARGET_LABEL]), "cat_recall": float(recall[TARGET_LABEL]),
        "cat_error_count": int(support[TARGET_LABEL] - tp[TARGET_LABEL]),
        "cat_error_rate": float(1 - recall[TARGET_LABEL]),
    }


def infer(context: dict, workers: int) -> None:
    rows = context["rows"]
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(CropDataset(rows, transform), batch_size=128, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completed = 0
    for model_name in MODELS:
        for seed in SEEDS:
            path = OUTPUT / "predictions" / model_name / str(seed) / "candidate_predictions.csv"
            if not path.is_file():
                checkpoint = torch.load(
                    GRADIENT / "runs" / model_name / "oi_gap_add0" / str(seed) / "best.pt",
                    map_location="cpu", weights_only=False,
                )
                model = build_model(model_name)
                model.load_state_dict(checkpoint["model"])
                model.to(device)
                predictions = predict(model, loader, rows, device)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(path, predictions)
                del model, checkpoint
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            completed += 1
            write_json(OUTPUT / "progress.json", {
                "status": "running", "completed_runs": completed, "total_runs": 24,
                "percent": 100 * completed / 24, "updated_at": now(),
            })
            print(f"[{completed}/24] {model_name} seed={seed}", flush=True)


def aggregate(context: dict) -> None:
    rows, target_rows, matrix = context["rows"], context["target_rows"], context["matrix"]
    positions = {row["sample_id"]: index for index, row in enumerate(rows)}
    target_positions = np.asarray([positions[row["sample_id"]] for row in target_rows])
    non_target_positions = {
        name: [positions[row["sample_id"]] for row in context["non_target_orders"][name]]
        for name in CLASS_NAMES if name != TARGET_CLASS
    }
    detailed = []
    orders = [("gap", -1, context["gap_order"])] + [
        ("random", replicate, order) for replicate, order in enumerate(context["random_orders"])
    ]
    for model_name in MODELS:
        for seed in SEEDS:
            predictions = read_csv(OUTPUT / "predictions" / model_name / str(seed) / "candidate_predictions.csv")
            by_id = {row["sample_id"]: row for row in predictions}
            ordered = [by_id[row["sample_id"]] for row in rows]
            true = np.asarray([int(row["true_label"]) for row in ordered])
            predicted = np.asarray([int(row["predicted_label"]) for row in ordered])
            target_correct = predicted[target_positions] == TARGET_LABEL
            for method, replicate, order in orders:
                for budget in BUDGETS:
                    chosen = list(target_positions[order[:budget]])
                    for name in CLASS_NAMES:
                        if name != TARGET_CLASS:
                            chosen.extend(non_target_positions[name][:budget])
                    frequency, presence = coverage_values(matrix, order[:budget])
                    detailed.append({
                        "method": method, "replicate": replicate, "model": model_name, "seed": seed,
                        "cat_per_class": budget, "total_test_targets": 5 * budget,
                        "frequency_coverage": frequency, "presence_coverage": presence,
                        **classification_metrics(true[chosen], predicted[chosen]),
                        **target_failure_metrics(order[:budget], target_correct, matrix),
                    })
    write_csv(OUTPUT / "tables/metrics_detailed.csv", detailed)
    metrics = (
        "frequency_coverage", "presence_coverage", "error_count", "error_rate",
        "cat_error_count", "cat_error_rate", "macro_f1", "cat_f1", "cat_recall",
        "failed_state_count", "failed_state_fraction", "state_macro_recall", "worst5_state_recall",
    )
    summary = []
    for budget in BUDGETS:
        gap = [row for row in detailed if row["method"] == "gap" and row["cat_per_class"] == budget]
        random_rows = [row for row in detailed if row["method"] == "random" and row["cat_per_class"] == budget]
        random_by_replicate = [{
            metric: float(np.mean([row[metric] for row in random_rows if row["replicate"] == replicate and row[metric] is not None]))
            for metric in metrics
        } for replicate in range(RANDOM_REPLICATES)]
        out = {"cat_per_class": budget, "total_test_targets": 5 * budget,
               "fixed_model_runs": 24, "random_replicates": RANDOM_REPLICATES}
        for metric in metrics:
            gap_value = float(np.mean([row[metric] for row in gap if row[metric] is not None]))
            random_values = np.asarray([row[metric] for row in random_by_replicate])
            out[f"gap_{metric}"] = gap_value
            out[f"random_{metric}_mean"] = float(random_values.mean())
            out[f"random_{metric}_sd"] = float(random_values.std(ddof=1))
            out[f"random_{metric}_p2_5"] = float(np.percentile(random_values, 2.5))
            out[f"random_{metric}_p97_5"] = float(np.percentile(random_values, 97.5))
            out[f"delta_{metric}"] = gap_value - float(random_values.mean())
            out[f"gap_percentile_{metric}"] = float(np.mean(random_values <= gap_value))
        summary.append(out)
    write_csv(OUTPUT / "tables/budget_summary.csv", summary)
    write_json(OUTPUT / "summary.json", {"status": "completed", "budget_summary": summary})
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "completed_runs": 24, "total_runs": 24,
        "percent": 100.0, "updated_at": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    context = build_protocol()
    if args.prepare_only:
        write_json(OUTPUT / "progress.json", {"status": "prepared", "completed_runs": 0, "total_runs": 24, "updated_at": now()})
        print(json.dumps({"status": "prepared", "targets": len(context["rows"])}))
        return
    infer(context, args.workers)
    aggregate(context)
    print(json.dumps({"status": "completed", "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
