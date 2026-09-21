#!/usr/bin/env python3
"""Evaluate coverage-guided visual test adequacy with frozen D300 models.

The training data and 24 D300 checkpoints remain fixed. Only nested, class-balanced
test subsets change. Coverage-guided dog selection is compared with random dog
selection at the same test budget, while non-dog targets are identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from openimages_cua_vsl import StateSchema
from run_four_model_gap_vs_random import build_model
from run_resnet18_dog_coverage_comparison import write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CLASS_SPECS, CropDataset, predict


ROOT = Path(__file__).resolve().parents[1]
GRADIENT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
STATE_SOURCE = ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
DATA = ROOT / "data/openimages_v7_dog_gap_v1"
SCHEMA_PATH = ROOT / "configs/dog_feature_schema_v3_1.json"
OUTPUT = ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
BUDGETS = (20, 40, 60, 80, 100)
RANDOM_REPLICATES = 1000
RANDOM_SEED_BASE = 20260920
COVERAGE_CAP = 30
MIN_STATE_SUPPORT = 3
CLASS_NAMES = tuple(name for name, _ in CLASS_SPECS)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_candidates() -> tuple[list[dict], list[dict], dict[str, dict], StateSchema]:
    ordinary = read_csv(DATA / "manifests/test.csv")
    difficulty = read_csv(DATA / "manifests/dog_difficulty_test.csv")
    for row in ordinary:
        row["candidate_source"] = "main_test"
    for row in difficulty:
        row["candidate_source"] = "difficulty_test"
    rows = ordinary + difficulty
    by_class = Counter(row["class_name"] for row in rows)
    expected = {"dog": 200, "cat": 100, "horse": 100, "sheep": 100, "person": 100}
    if dict(by_class) != expected:
        raise ValueError(f"unexpected candidate class counts: {dict(by_class)}")
    if len(rows) != 600 or len({row["sample_id"] for row in rows}) != 600:
        raise ValueError("candidate pool must contain 600 unique targets")
    if any(not Path(row["crop_path"]).is_file() for row in rows):
        raise FileNotFoundError("candidate crop is missing")

    feature_rows = read_csv(STATE_SOURCE)
    features = {row["image_id"]: row for row in feature_rows}
    dog_rows = [row for row in rows if row["class_name"] == "dog"]
    if len(features) != 200 or set(features) != {row["sample_id"] for row in dog_rows}:
        raise ValueError("dog feature labels do not match the 200 dog candidates")
    if any(row["feature_status"] != "ok" for row in feature_rows):
        raise ValueError("dog feature labels include failed records")

    train = read_csv(GRADIENT / "manifests/train_oi_gap_add0.csv")
    train_ids = {row["image_id"] for row in train}
    candidate_ids = {row["image_id"] for row in rows}
    if train_ids & candidate_ids:
        raise ValueError("image ID leakage between D300 training and candidate test pool")
    train_hashes = {row.get("image_sha256") for row in train if row.get("image_sha256")}
    candidate_hashes = {row.get("image_sha256") for row in rows if row.get("image_sha256")}
    if train_hashes & candidate_hashes:
        raise ValueError("image SHA-256 leakage between D300 training and candidate test pool")
    return rows, dog_rows, features, StateSchema.from_path(SCHEMA_PATH)


def dog_state_matrix(
    dog_rows: list[dict], features: dict[str, dict], schema: StateSchema,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    states = schema.coverage_states()
    state_to_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(dog_rows), len(states)), dtype=np.int8)
    for row_index, row in enumerate(dog_rows):
        for pair in schema.state_pairs(features[row["sample_id"]]):
            if pair in state_to_index:
                matrix[row_index, state_to_index[pair]] = 1
    return matrix, states


def coverage_values(matrix: np.ndarray, selected: list[int]) -> tuple[float, float]:
    counts = matrix[selected].sum(axis=0) if selected else np.zeros(matrix.shape[1])
    frequency_coverage = float(np.minimum(counts / COVERAGE_CAP, 1.0).mean())
    presence_coverage = float((counts > 0).mean())
    return frequency_coverage, presence_coverage


def coverage_order(dog_rows: list[dict], matrix: np.ndarray) -> list[int]:
    remaining = set(range(len(dog_rows)))
    selected: list[int] = []
    counts = np.zeros(matrix.shape[1], dtype=np.int32)
    while remaining:
        eligible = counts < COVERAGE_CAP
        best = max(
            remaining,
            key=lambda index: (int(matrix[index, eligible].sum()), dog_rows[index]["sample_id"]),
        )
        selected.append(best)
        remaining.remove(best)
        counts += matrix[best]
    return selected


def build_protocol() -> dict:
    rows, dog_rows, features, schema = prepare_candidates()
    matrix, states = dog_state_matrix(dog_rows, features, schema)
    gap_order = coverage_order(dog_rows, matrix)

    random_orders = []
    for replicate in range(RANDOM_REPLICATES):
        order = list(range(len(dog_rows)))
        random.Random(RANDOM_SEED_BASE + replicate).shuffle(order)
        random_orders.append(order)

    non_dog_orders: dict[str, list[dict]] = {}
    for class_index, class_name in enumerate(CLASS_NAMES[1:], start=1):
        class_rows = sorted(
            (row for row in rows if row["class_name"] == class_name),
            key=lambda row: row["sample_id"],
        )
        random.Random(RANDOM_SEED_BASE + 10000 + class_index).shuffle(class_rows)
        non_dog_orders[class_name] = class_rows

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "candidate_pool.csv", rows)
    write_csv(OUTPUT / "orders/gap_dog_order.csv", [
        {"rank": rank + 1, **dog_rows[index]} for rank, index in enumerate(gap_order)
    ])
    write_csv(OUTPUT / "orders/random_dog_orders.csv", [
        {
            "replicate": replicate,
            "seed": RANDOM_SEED_BASE + replicate,
            "rank": rank + 1,
            "sample_id": dog_rows[index]["sample_id"],
        }
        for replicate, order in enumerate(random_orders)
        for rank, index in enumerate(order)
    ])
    write_csv(OUTPUT / "orders/non_dog_order.csv", [
        {"rank": rank + 1, **row}
        for class_name in CLASS_NAMES[1:]
        for rank, row in enumerate(non_dog_orders[class_name])
    ])

    selection_rows = []
    for budget in BUDGETS:
        gap_frequency, gap_presence = coverage_values(matrix, gap_order[:budget])
        random_frequency = []
        random_presence = []
        for order in random_orders:
            frequency, presence = coverage_values(matrix, order[:budget])
            random_frequency.append(frequency)
            random_presence.append(presence)
        selection_rows.append({
            "dog_per_class": budget,
            "total_test_targets": budget * len(CLASS_NAMES),
            "gap_frequency_coverage": gap_frequency,
            "random_frequency_coverage_mean": float(np.mean(random_frequency)),
            "random_frequency_coverage_sd": float(np.std(random_frequency, ddof=1)),
            "random_frequency_coverage_p2_5": float(np.percentile(random_frequency, 2.5)),
            "random_frequency_coverage_p97_5": float(np.percentile(random_frequency, 97.5)),
            "gap_presence_coverage": gap_presence,
            "random_presence_coverage_mean": float(np.mean(random_presence)),
            "random_presence_coverage_sd": float(np.std(random_presence, ddof=1)),
        })
        gap_ids = {dog_rows[index]["sample_id"] for index in gap_order[:budget]}
        manifest = [row for row in dog_rows if row["sample_id"] in gap_ids]
        for class_name in CLASS_NAMES[1:]:
            manifest.extend(non_dog_orders[class_name][:budget])
        write_csv(OUTPUT / f"manifests/gap_balanced_{budget}_per_class.csv", manifest)
    write_csv(OUTPUT / "tables/selection_coverage.csv", selection_rows)

    checkpoint_paths = [
        GRADIENT / "runs" / model / "oi_gap_add0" / str(seed) / "best.pt"
        for model in MODELS for seed in SEEDS
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing D300 checkpoints: {missing[:3]}")
    protocol = {
        "experiment_id": "fixed-train-visual-test-adequacy-20260920",
        "created_at": now(),
        "research_question": (
            "At an equal class-balanced test budget, does coverage-guided dog selection "
            "achieve higher visual-state coverage and expose more classifier failures than random selection?"
        ),
        "frozen_training_condition": "oi_gap_add0 / D300",
        "fixed_checkpoints": len(checkpoint_paths),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "candidate_pool": {"dog": 200, "cat": 100, "horse": 100, "sheep": 100, "person": 100},
        "budgets_per_class": list(BUDGETS),
        "random_replicates": RANDOM_REPLICATES,
        "coverage_cap": COVERAGE_CAP,
        "selection_blinding": "test selection uses dog_v3.1 states only; model predictions are not inputs",
        "human_review_status": "accepted_by_user_assumption_all_200_correct",
        "primary_metrics": [
            "frequency_coverage", "error_rate", "dog_error_rate", "failed_state_count",
        ],
        "secondary_metrics": ["macro_f1", "dog_f1", "dog_recall", "worst5_state_recall"],
        "success_criterion": (
            "At equal budget, gap selection exceeds the random 95% interval in coverage and "
            "has positive error-discovery gain without using predictions for selection."
        ),
        "input_sha256": {
            "D300_train_manifest": sha256(GRADIENT / "manifests/train_oi_gap_add0.csv"),
            "main_test_manifest": sha256(DATA / "manifests/test.csv"),
            "difficulty_test_manifest": sha256(DATA / "manifests/dog_difficulty_test.csv"),
            "state_features": sha256(STATE_SOURCE),
            "schema": sha256(SCHEMA_PATH),
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "- Thesis: within-class visual-state coverage can measure test adequacy and guide failure-revealing test selection under a fixed budget.\n"
        "- Primary comparison: coverage-guided versus 1000 random dog selections at identical balanced test budgets.\n"
        "- Fixed factors: D300 training data, 24 checkpoints, candidate pool, non-dog targets, preprocessing, and label space.\n"
        "- Boundary: the selected suites are stress tests, not estimates of deployment prevalence or accuracy.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol frozen: {now()}\n"
        "- Claim tested: state-coverage-guided selection exposes more valid failures than random selection at equal budget.\n"
        "- Checkpoints: 24 frozen D300 models; no retraining.\n"
        "- Candidate pool: 600 Open Images V7 test targets; 200 dog targets have accepted dog_v3.1 states.\n"
        "- Test budgets: 20, 40, 60, 80, and 100 targets per class.\n"
        "- Control: 1000 independently seeded nested random dog orders; identical nested non-dog targets.\n"
        "- Reproduction: `.venv/bin/python -u scripts/run_fixed_train_test_adequacy_experiment.py`.\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Coverage-guided tests cover states more efficiently than random tests | pending | `tables/selection_coverage.csv` | Same 200-dog candidate pool |\n"
        "| Coverage-guided tests expose more model errors | pending | `tables/budget_summary.csv` | Fixed D300 models and Open Images targets |\n"
        "| Lower test F1 means the model is worse | rejected framing | Fixed checkpoint design | Lower F1 indicates stronger stress exposure, not changed model quality |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: not applicable; no external claims introduced.\n"
        "- Result traceability: pending inference and aggregation.\n"
        "- Claim scope: test suites are diagnostic stress tests, not deployment-distribution samples.\n",
        encoding="utf-8",
    )
    return {
        "rows": rows, "dog_rows": dog_rows, "features": features, "schema": schema,
        "matrix": matrix, "states": states, "gap_order": gap_order,
        "random_orders": random_orders, "non_dog_orders": non_dog_orders,
    }


def confusion_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    matrix = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted_support, out=np.zeros(5), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
    result: dict[str, float | int] = {
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "error_count": int(matrix.sum() - tp.sum()),
        "error_rate": float(1 - tp.sum() / matrix.sum()),
        "dog_f1": float(f1[0]),
        "dog_recall": float(recall[0]),
        "dog_error_count": int(support[0] - tp[0]),
        "dog_error_rate": float(1 - recall[0]),
    }
    return result


def dog_failure_metrics(
    selected_dog: list[int], dog_correct: np.ndarray, state_matrix: np.ndarray,
) -> dict[str, float | int | None]:
    selected = np.asarray(selected_dog, dtype=int)
    selected_states = state_matrix[selected]
    correct = dog_correct[selected].astype(bool)
    failed_states = selected_states[~correct].sum(axis=0) if (~correct).any() else np.zeros(state_matrix.shape[1])
    support = selected_states.sum(axis=0)
    correct_support = selected_states[correct].sum(axis=0) if correct.any() else np.zeros(state_matrix.shape[1])
    eligible = support >= MIN_STATE_SUPPORT
    recalls = np.divide(correct_support, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    eligible_recalls = recalls[eligible]
    worst = np.sort(eligible_recalls)[: min(5, len(eligible_recalls))]
    return {
        "failed_state_count": int((failed_states > 0).sum()),
        "failed_state_fraction": float((failed_states > 0).mean()),
        "eligible_state_count": int(eligible.sum()),
        "state_macro_recall": float(eligible_recalls.mean()) if len(eligible_recalls) else None,
        "worst5_state_recall": float(worst.mean()) if len(worst) else None,
    }


def infer_candidates(context: dict, workers: int) -> None:
    rows = context["rows"]
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(
        CropDataset(rows, transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(MODELS) * len(SEEDS)
    completed = 0
    for model_name in MODELS:
        for seed in SEEDS:
            output_dir = OUTPUT / "predictions" / model_name / str(seed)
            prediction_path = output_dir / "candidate_predictions.csv"
            if prediction_path.is_file():
                completed += 1
                print(f"[{completed}/{total}] reuse {model_name} seed={seed}", flush=True)
                continue
            checkpoint_path = GRADIENT / "runs" / model_name / "oi_gap_add0" / str(seed) / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(model_name)
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            predictions = predict(model, loader, rows, device)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_csv(prediction_path, predictions)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            completed += 1
            write_json(OUTPUT / "progress.json", {
                "status": "running", "completed_runs": completed, "total_runs": total,
                "percent": 100 * completed / total, "latest_model": model_name,
                "latest_seed": seed, "updated_at": now(),
            })
            print(f"[{completed}/{total}] inferred {model_name} seed={seed}", flush=True)


def aggregate(context: dict) -> None:
    rows = context["rows"]
    dog_rows = context["dog_rows"]
    matrix = context["matrix"]
    gap_order = context["gap_order"]
    random_orders = context["random_orders"]
    non_dog_orders = context["non_dog_orders"]
    row_position = {row["sample_id"]: index for index, row in enumerate(rows)}
    dog_positions = np.asarray([row_position[row["sample_id"]] for row in dog_rows])
    fixed_non_dog_positions = {
        class_name: [row_position[row["sample_id"]] for row in non_dog_orders[class_name]]
        for class_name in CLASS_NAMES[1:]
    }

    detailed = []
    for model_name in MODELS:
        for seed in SEEDS:
            predictions = read_csv(OUTPUT / "predictions" / model_name / str(seed) / "candidate_predictions.csv")
            prediction_by_id = {row["sample_id"]: row for row in predictions}
            ordered_predictions = [prediction_by_id[row["sample_id"]] for row in rows]
            true = np.asarray([int(row["true_label"]) for row in ordered_predictions])
            predicted = np.asarray([int(row["predicted_label"]) for row in ordered_predictions])
            dog_correct = predicted[dog_positions] == 0

            for method, replicate, order in [("gap", -1, gap_order)]:
                for budget in BUDGETS:
                    positions = list(dog_positions[order[:budget]])
                    for class_name in CLASS_NAMES[1:]:
                        positions.extend(fixed_non_dog_positions[class_name][:budget])
                    metrics = confusion_metrics(true[positions], predicted[positions])
                    failure = dog_failure_metrics(order[:budget], dog_correct, matrix)
                    frequency, presence = coverage_values(matrix, order[:budget])
                    detailed.append({
                        "method": method, "replicate": replicate, "model": model_name,
                        "seed": seed, "dog_per_class": budget,
                        "total_test_targets": 5 * budget, "frequency_coverage": frequency,
                        "presence_coverage": presence, **metrics, **failure,
                    })

            for replicate, order in enumerate(random_orders):
                for budget in BUDGETS:
                    positions = list(dog_positions[order[:budget]])
                    for class_name in CLASS_NAMES[1:]:
                        positions.extend(fixed_non_dog_positions[class_name][:budget])
                    metrics = confusion_metrics(true[positions], predicted[positions])
                    failure = dog_failure_metrics(order[:budget], dog_correct, matrix)
                    frequency, presence = coverage_values(matrix, order[:budget])
                    detailed.append({
                        "method": "random", "replicate": replicate, "model": model_name,
                        "seed": seed, "dog_per_class": budget,
                        "total_test_targets": 5 * budget, "frequency_coverage": frequency,
                        "presence_coverage": presence, **metrics, **failure,
                    })
    write_csv(OUTPUT / "tables/metrics_detailed.csv", detailed)

    metrics = (
        "frequency_coverage", "presence_coverage", "error_count", "error_rate",
        "dog_error_count", "dog_error_rate", "macro_f1", "dog_f1", "dog_recall",
        "failed_state_count", "failed_state_fraction", "state_macro_recall",
        "worst5_state_recall",
    )
    summary = []
    for budget in BUDGETS:
        gap = [row for row in detailed if row["method"] == "gap" and row["dog_per_class"] == budget]
        random_rows = [row for row in detailed if row["method"] == "random" and row["dog_per_class"] == budget]
        random_by_replicate = []
        for replicate in range(RANDOM_REPLICATES):
            group = [row for row in random_rows if row["replicate"] == replicate]
            random_by_replicate.append({
                metric: float(np.mean([row[metric] for row in group if row[metric] is not None]))
                for metric in metrics
            })
        out: dict[str, float | int] = {
            "dog_per_class": budget,
            "total_test_targets": 5 * budget,
            "fixed_model_runs": len(gap),
            "random_replicates": RANDOM_REPLICATES,
        }
        for metric in metrics:
            gap_value = float(np.mean([row[metric] for row in gap if row[metric] is not None]))
            random_values = np.asarray([row[metric] for row in random_by_replicate], dtype=float)
            out[f"gap_{metric}"] = gap_value
            out[f"random_{metric}_mean"] = float(random_values.mean())
            out[f"random_{metric}_sd"] = float(random_values.std(ddof=1))
            out[f"random_{metric}_p2_5"] = float(np.percentile(random_values, 2.5))
            out[f"random_{metric}_p97_5"] = float(np.percentile(random_values, 97.5))
            out[f"delta_{metric}"] = gap_value - float(random_values.mean())
            out[f"gap_percentile_{metric}"] = float(np.mean(random_values <= gap_value))
        summary.append(out)
    write_csv(OUTPUT / "tables/budget_summary.csv", summary)

    write_json(OUTPUT / "summary.json", {
        "status": "completed",
        "fixed_checkpoints": len(MODELS) * len(SEEDS),
        "random_replicates": RANDOM_REPLICATES,
        "budget_summary": summary,
    })
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "completed_runs": len(MODELS) * len(SEEDS),
        "total_runs": len(MODELS) * len(SEEDS), "percent": 100.0,
        "updated_at": now(),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = build_protocol()
    if args.prepare_only:
        write_json(OUTPUT / "progress.json", {
            "status": "prepared", "completed_runs": 0,
            "total_runs": len(MODELS) * len(SEEDS), "updated_at": now(),
        })
        print(json.dumps({
            "status": "prepared", "output": str(OUTPUT),
            "checkpoints": len(MODELS) * len(SEEDS), "candidate_targets": len(context["rows"]),
        }))
        return
    infer_candidates(context, args.workers)
    aggregate(context)
    print(json.dumps({"status": "completed", "output": str(OUTPUT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
