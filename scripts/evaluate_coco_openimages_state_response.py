#!/usr/bin/env python3
"""Evaluate frozen coverage-gradient checkpoints on an independent dog state test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from openimages_cua_vsl import StateSchema, parse_values
from run_four_model_gap_vs_random import build_model
from run_resnet18_dog_coverage_comparison import write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CropDataset, predict


ROOT = Path(__file__).resolve().parents[1]
GRADIENT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
STATE_SOURCE = ROOT / "artifacts/openimages_v7_cua_vsl_v1"
DATA = ROOT / "data/openimages_v7_dog_gap_v1"
OUTPUT = ROOT / "artifacts/coco_openimages_state_response_20260920"
SCHEMA = ROOT / "configs/dog_feature_schema_v3_1.json"
COCO_FEATURES = ROOT / "artifacts/dog500_features_v3_1_20260913/results.csv"
ADDITIONS = (0, 50, 100, 150, 200, 250, 300, 350)
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
CAP = 30
MIN_SUPPORT = 5
ATYPICAL_ACTIONS = {
    "prone", "running", "jumping", "playing", "eating", "drinking",
    "interacting", "sniffing", "barking", "sleeping",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def state_pairs(schema: StateSchema, row: dict[str, str]) -> set[tuple[str, str]]:
    return schema.state_pairs(row)


def prepare_inputs() -> tuple[list[dict], dict[str, dict], StateSchema, Counter]:
    schema = StateSchema.from_path(SCHEMA)
    main_rows = [row for row in read_csv(DATA / "manifests/test.csv") if row["class_name"] == "dog"]
    difficulty_rows = read_csv(DATA / "manifests/dog_difficulty_test.csv")
    for row in main_rows:
        row["state_test_split"] = "main"
    for row in difficulty_rows:
        row["state_test_split"] = "difficulty"
    rows = main_rows + difficulty_rows
    if len(rows) != 200 or len({row["sample_id"] for row in rows}) != 200:
        raise ValueError("state test must contain 200 unique dog targets")
    if any(row["class_name"] != "dog" or int(row["label"]) != 0 for row in rows):
        raise ValueError("state test must contain label-0 dog targets only")
    if any(not Path(row["crop_path"]).is_file() for row in rows):
        raise FileNotFoundError("at least one state-test crop is missing")

    feature_rows = read_csv(STATE_SOURCE / "state_test_features_auto/results.csv")
    features = {row["image_id"]: row for row in feature_rows}
    if len(features) != 200 or set(features) != {row["sample_id"] for row in rows}:
        raise ValueError("state-test manifests and feature labels do not match")
    if any(row["feature_status"] != "ok" for row in feature_rows):
        raise ValueError("state-test feature file contains failed rows")

    base = read_csv(GRADIENT / "manifests/train_oi_gap_add0.csv")
    base_dog_ids = [row["image_id"] for row in base if row["class_name"] == "dog"]
    coco_features = {row["image_id"]: row for row in read_csv(COCO_FEATURES)}
    if len(base_dog_ids) != 300 or any(image_id not in coco_features for image_id in base_dog_ids):
        raise ValueError("cannot reconstruct D300 dog state counts")
    counts = Counter(
        pair for image_id in base_dog_ids for pair in state_pairs(schema, coco_features[image_id])
    )

    train_rows = []
    for path in (GRADIENT / "manifests").glob("*.csv"):
        train_rows.extend(read_csv(path))
    train_ids = {row["image_id"] for row in train_rows}
    test_ids = {row["image_id"] for row in rows}
    if train_ids & test_ids:
        raise ValueError("image_id leakage between gradient training data and state test")
    train_hashes = {row.get("image_sha256") for row in train_rows if row.get("image_sha256")}
    test_hashes = {row.get("image_sha256") for row in rows if row.get("image_sha256")}
    if train_hashes & test_hashes:
        raise ValueError("image hash leakage between gradient training data and state test")
    return rows, features, schema, counts


def recall(predictions: list[dict], member_ids: set[str]) -> float | None:
    members = [row for row in predictions if row["sample_id"] in member_ids]
    if not members:
        return None
    return sum(row["predicted_class"] == "dog" for row in members) / len(members)


def state_metrics(
    predictions: list[dict], features: dict[str, dict], schema: StateSchema,
    train_counts: Counter,
) -> tuple[dict, list[dict]]:
    prediction_ids = {row["sample_id"] for row in predictions}
    if prediction_ids != set(features):
        raise ValueError("prediction and state-feature IDs differ")
    state_rows = []
    rare_values, common_values = [], []
    for feature_id, state in schema.coverage_states():
        members = {
            sample_id for sample_id, feature_row in features.items()
            if (feature_id, state) in state_pairs(schema, feature_row)
        }
        value = recall(predictions, members)
        count = train_counts[(feature_id, state)]
        eligible = len(members) >= MIN_SUPPORT
        rare = count < CAP
        if eligible and value is not None:
            (rare_values if rare else common_values).append(value)
        state_rows.append({
            "feature_id": feature_id,
            "state": state,
            "base_train_count": count,
            "state_test_support": len(members),
            "dog_recall": value,
            "rare_by_base_cap": int(rare),
            "eligible_min_support": int(eligible),
        })
    eligible = [row["dog_recall"] for row in state_rows if row["eligible_min_support"]]
    worst = sorted(eligible)[:min(5, len(eligible))]

    def ids_where(predicate):
        return {sample_id for sample_id, feature in features.items() if predicate(feature)}

    occluded = ids_where(lambda row: row["occlusion_level"] in {"partial", "heavy"})
    truncated = ids_where(lambda row: row["truncation_level"] in {"partial", "heavy"})
    atypical = ids_where(lambda row: row["posture_action_state"] in ATYPICAL_ACTIONS)
    small = {
        row["sample_id"] for row in predictions
        if float(row.get("bbox_area_ratio") or 1.0) < 0.03
    }
    main = {row["sample_id"] for row in predictions if row["state_test_split"] == "main"}
    difficulty = {row["sample_id"] for row in predictions if row["state_test_split"] == "difficulty"}
    metrics = {
        "overall_dog_recall": recall(predictions, prediction_ids),
        "main_dog_recall": recall(predictions, main),
        "difficulty_dog_recall": recall(predictions, difficulty),
        "rare_state_macro_recall": float(np.mean(rare_values)) if rare_values else None,
        "common_state_macro_recall": float(np.mean(common_values)) if common_values else None,
        "worst5_state_recall": float(np.mean(worst)) if worst else None,
        "occluded_dog_recall": recall(predictions, occluded),
        "truncated_dog_recall": recall(predictions, truncated),
        "small_dog_recall": recall(predictions, small),
        "atypical_action_dog_recall": recall(predictions, atypical),
        "rare_state_count": len(rare_values),
        "common_state_count": len(common_values),
        "occluded_support": len(occluded),
        "truncated_support": len(truncated),
        "small_support": len(small),
        "atypical_action_support": len(atypical),
    }
    return metrics, state_rows


def aggregate(run_rows: list[dict], state_rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["external_dog_added"])].append(row)
    summary = []
    metric_names = (
        "overall_dog_recall", "main_dog_recall", "difficulty_dog_recall",
        "rare_state_macro_recall", "common_state_macro_recall", "worst5_state_recall",
        "occluded_dog_recall", "truncated_dog_recall", "small_dog_recall",
        "atypical_action_dog_recall",
    )
    for (model, add), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        out = {
            "model": model, "external_dog_added": add, "coverage": rows[0]["coverage"],
            "runs": len(rows),
        }
        for metric in metric_names:
            values = [row[metric] for row in rows if row[metric] is not None]
            out[f"{metric}_mean"] = float(np.mean(values)) if values else None
            out[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    write_csv(OUTPUT / "tables/metrics_by_run.csv", run_rows)
    write_csv(OUTPUT / "tables/metrics_by_model_stage.csv", summary)
    write_csv(OUTPUT / "tables/state_recall_by_run.csv", state_rows)

    stage_rows = []
    gradient_metrics = json.loads((GRADIENT / "all_metrics.json").read_text(encoding="utf-8"))
    for add in ADDITIONS:
        rows = [row for row in run_rows if row["external_dog_added"] == add]
        ordinary = [row for row in gradient_metrics if row["condition"] == f"oi_gap_add{add}"]
        out = {
            "external_dog_added": add,
            "dog_per_class": 300 + add,
            "coverage": rows[0]["coverage"],
            "fixed_coco_macro_f1": float(np.mean([row["macro_f1"] for row in ordinary])),
            "fixed_coco_dog_f1": float(np.mean([row["dog_f1"] for row in ordinary])),
        }
        for metric in metric_names:
            out[metric] = float(np.mean([row[metric] for row in rows if row[metric] is not None]))
        stage_rows.append(out)
    for index, row in enumerate(stage_rows):
        if index == 0:
            row.update({
                "delta_coverage": None,
                "macro_f1_sensitivity": None,
                "rare_state_sensitivity": None,
                "worst5_state_sensitivity": None,
            })
            continue
        previous = stage_rows[index - 1]
        delta_coverage = row["coverage"] - previous["coverage"]
        row["delta_coverage"] = delta_coverage
        row["macro_f1_sensitivity"] = (
            row["fixed_coco_macro_f1"] - previous["fixed_coco_macro_f1"]
        ) / delta_coverage
        row["rare_state_sensitivity"] = (
            row["rare_state_macro_recall"] - previous["rare_state_macro_recall"]
        ) / delta_coverage
        row["worst5_state_sensitivity"] = (
            row["worst5_state_recall"] - previous["worst5_state_recall"]
        ) / delta_coverage
    write_csv(OUTPUT / "tables/stage_response.csv", stage_rows)
    write_json(OUTPUT / "summary.json", {
        "scientific_status": "preliminary_automatic_state_labels",
        "completed_runs": len(run_rows),
        "state_test_targets": 200,
        "stage_response": stage_rows,
    })


def write_governance(rows: list[dict], features: dict[str, dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment_id": "coco-openimages-state-response-20260920",
        "created_at": now(),
        "objective": "measure state-stratified response as dog training-state coverage increases",
        "training_experiment": str(GRADIENT.resolve()),
        "checkpoints": len(ADDITIONS) * len(MODELS) * len(SEEDS),
        "state_test": {
            "source": "Open Images V7 test split",
            "main_dog": 100,
            "difficulty_dog": 100,
            "total": len(rows),
            "automatic_feature_labels": len(features),
            "human_review_complete": False,
        },
        "coverage_cap": CAP,
        "minimum_state_test_support": MIN_SUPPORT,
        "definitions": {
            "rare_state": "D300 frequency < 30",
            "common_state": "D300 frequency >= 30",
            "worst5_state_recall": "mean of five lowest state recalls among states with test support >= 5",
            "local_sensitivity": "adjacent metric difference divided by adjacent coverage-index difference",
        },
        "input_sha256": {
            "gradient_protocol": sha256(GRADIENT / "protocol.json"),
            "gradient_metrics": sha256(GRADIENT / "all_metrics.json"),
            "main_test_manifest": sha256(DATA / "manifests/test.csv"),
            "difficulty_test_manifest": sha256(DATA / "manifests/dog_difficulty_test.csv"),
            "state_features": sha256(STATE_SOURCE / "state_test_features_auto/results.csv"),
            "schema": sha256(SCHEMA),
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "- Thesis: interpretable within-class visual-state coverage is a dataset adequacy dimension whose relationship with classifier performance is non-monotonic and architecture-dependent.\n"
        "- Scope: dog target-region five-class classification; COCO-based training gradient; independent Open Images dog state test.\n"
        "- Primary evidence: coverage response, rare-state macro recall, worst-five-state recall, and fixed COCO Macro-F1.\n"
        "- Boundary: automatic dog_v3.1 state labels are preliminary until human review is complete.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Started: {now()}\n"
        "- Decision: reuse the frozen 100 main + 100 difficulty Open Images dog tests instead of creating a post-hoc 300-image test.\n"
        "- Claim tested: increasing dog training-state coverage changes common, rare, and worst-state recall differently.\n"
        "- Falsifiable alternatives: all state strata improve together; all decline together; or no systematic response exists.\n"
        "- Fixed factors: 192 frozen checkpoints, preprocessing, label space, and state-test targets.\n"
        "- Command: `.venv/bin/python -u scripts/evaluate_coco_openimages_state_response.py`.\n"
        "- Human review: not complete; automatic-label results are preliminary.\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| State-test evaluation is leakage-free by image ID and SHA-256 | pending run audit | `protocol.json`; runtime checks | Exact and hash duplicates only |\n"
        "| Higher training coverage improves rare-state recall | untested | `tables/stage_response.csv` after execution | Do not claim before analysis |\n"
        "| Higher coverage causes F1 decline | unsupported as a causal claim | fixed-test response plus random controls | Domain shift and selection effects remain alternatives |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: not applicable; no external claims introduced.\n"
        "- Result traceability: pending completion of 192 checkpoint evaluations.\n"
        "- Claim scope: automatic state labels must not be presented as human-verified ground truth.\n"
        "- Open blocker: human review of 200 state annotations.\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, features, schema, train_counts = prepare_inputs()
    write_governance(rows, features)
    total = len(ADDITIONS) * len(MODELS) * len(SEEDS)
    write_json(OUTPUT / "progress.json", {
        "status": "prepared" if args.prepare_only else "running",
        "completed_runs": 0, "total_runs": total, "updated_at": now(),
    })
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "runs": total, "targets": len(rows)}))
        return

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(
        CropDataset(rows, transform), batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    coverage_rows = {
        int(row["external_dog_added"]): float(row["coverage"])
        for row in read_csv(GRADIENT / "coverage_gradient.csv")
    }
    device = torch.device("cuda")
    run_rows, all_state_rows = [], []
    for add in ADDITIONS:
        condition = f"oi_gap_add{add}"
        for model_name in MODELS:
            for seed in SEEDS:
                output_dir = OUTPUT / "runs" / model_name / condition / str(seed)
                prediction_path = output_dir / "state_test_predictions.csv"
                metric_path = output_dir / "state_metrics.json"
                state_path = output_dir / "state_recall.csv"
                if prediction_path.is_file() and metric_path.is_file() and state_path.is_file():
                    predictions = read_csv(prediction_path)
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    per_state = read_csv(state_path)
                    for row in per_state:
                        row["dog_recall"] = float(row["dog_recall"]) if row["dog_recall"] else None
                else:
                    checkpoint_path = GRADIENT / "runs" / model_name / condition / str(seed) / "best.pt"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    model = build_model(model_name)
                    model.load_state_dict(checkpoint["model"])
                    model.to(device)
                    predictions = predict(model, loader, rows, device)
                    metrics, per_state = state_metrics(predictions, features, schema, train_counts)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    write_csv(prediction_path, predictions)
                    write_json(metric_path, metrics)
                    write_csv(state_path, per_state)
                    del model, checkpoint
                    torch.cuda.empty_cache()
                record = {
                    "model": model_name, "condition": condition, "seed": seed,
                    "external_dog_added": add, "coverage": coverage_rows[add], **metrics,
                }
                run_rows.append(record)
                all_state_rows.extend({
                    "model": model_name, "condition": condition, "seed": seed,
                    "external_dog_added": add, "coverage": coverage_rows[add], **row,
                } for row in per_state)
                write_json(OUTPUT / "progress.json", {
                    "status": "running", "completed_runs": len(run_rows), "total_runs": total,
                    "percent": 100 * len(run_rows) / total, "latest": record, "updated_at": now(),
                })
                print(
                    f"[{len(run_rows)}/{total}] {model_name} {condition} seed={seed} "
                    f"rare={metrics['rare_state_macro_recall']:.4f} "
                    f"worst5={metrics['worst5_state_recall']:.4f}", flush=True,
                )
    aggregate(run_rows, all_state_rows)
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "completed_runs": total, "total_runs": total,
        "percent": 100.0, "updated_at": now(),
    })
    print(json.dumps({"status": "completed", "output": str(OUTPUT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
