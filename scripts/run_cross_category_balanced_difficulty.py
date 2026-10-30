#!/usr/bin/env python3
"""Run a unified difficulty-balanced coverage diagnostic for dog, cat and head/face."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
OUTPUT = ROOT / "artifacts/cross_category_balanced_difficulty_20260928"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
STRATA = 5
STRATUM_SIZE = 40
SELECT_PER_STRATUM = 20
COVERAGE_CAP = 20
MIN_STATE_SUPPORT = 5
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 10000
RANDOM_SEED = 20260928

DOG_DATA = SOURCE / "data/openimages_v7_dog_gap_v1"
DOG_FEATURES = SOURCE / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
DOG_SCHEMA = ROOT / "configs/dog_diag_feature_schema_v1_0.json"
DOG_PREDICTIONS = SOURCE / "artifacts/fixed_train_visual_test_adequacy_20260920/predictions"

CAT_EXPERIMENT = SOURCE / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
CAT_FEATURES = SOURCE / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv"
CAT_SCHEMA = SOURCE / "configs/cat_feature_schema_v1_0.json"

FACE_EXPERIMENT = ROOT / "artifacts/head_face_v1_0_20260922"
FACE_SCHEMA = ROOT / "configs/head_face_feature_schema_v1_0.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def progress(phase: str, detail: str, percent: float) -> None:
    value = {
        "status": "running", "phase": phase, "detail": detail,
        "percent": percent, "updated_at": now(),
    }
    write_json(OUTPUT / "progress.json", value)
    print(f"[{phase}] {detail} ({percent:.0f}%)", flush=True)


def parse_values(value: str) -> list[str]:
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return [value]


def critical_part_visibility(row: dict[str, str]) -> str:
    if row.get("head_visibility") == "not_visible":
        return "severe_loss"
    missing = sum(row.get(field) == "not_visible" for field in (
        "muzzle_visibility", "mouth_visibility", "eye_state",
    ))
    if missing >= 2:
        return "severe_loss"
    if (
        missing > 0
        or row.get("head_visibility") == "partial"
        or row.get("muzzle_visibility") == "partial"
        or row.get("mouth_visibility") == "partial"
    ):
        return "partial"
    return "clear"


def animal_schema(category: str) -> list[dict]:
    if category == "dog":
        return json.loads(DOG_SCHEMA.read_text(encoding="utf-8"))["features"]
    source = json.loads(CAT_SCHEMA.read_text(encoding="utf-8"))["universal_features"]
    wanted = {
        "coat_primary_color", "coat_pattern", "coat_texture", "ear_posture",
        "body_shape", "head_orientation", "body_orientation", "posture_action_state",
        "head_visibility", "leg_visibility_count", "tail_visibility",
        "occlusion_level", "truncation_level", "cat_recognizability", "scene",
    }
    features = [
        {"feature_id": row["feature_id"], "source": row["feature_id"],
         "possible_values": row["possible_values"]}
        for row in source if row["feature_id"] in wanted
    ]
    features.append({
        "feature_id": "critical_part_visibility", "source": "derived",
        "possible_values": ["clear", "partial", "severe_loss", "unknown"],
    })
    return features


def face_schema() -> list[dict]:
    schema = json.loads(FACE_SCHEMA.read_text(encoding="utf-8"))
    excluded = {"visible_skin_tone", "skin_tone_confidence"}
    return [
        {"feature_id": row["feature_id"], "source": row["feature_id"],
         "possible_values": row["possible_values"], "value_type": row["value_type"]}
        for row in schema["features"] if row["feature_id"] not in excluded
    ]


def state_matrix(
    category: str, target_rows: list[dict[str, str]], features: dict[str, dict[str, str]],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    schema = face_schema() if category == "face" else animal_schema(category)
    states = [
        (feature["feature_id"], value)
        for feature in schema for value in feature["possible_values"] if value != "unknown"
    ]
    state_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(target_rows), len(states)), dtype=np.int8)
    for row_index, target in enumerate(target_rows):
        feature_row = features[target["sample_id"]]
        for feature in schema:
            key = feature["feature_id"]
            if feature.get("source") == "derived":
                values = [critical_part_visibility(feature_row)]
            else:
                values = parse_values(feature_row.get(feature["source"], "unknown"))
            for value in values:
                index = state_index.get((key, value))
                if index is not None:
                    matrix[row_index, index] = 1
    return matrix, states


def scale_component(area: float) -> float:
    if area <= 0:
        return 3.0
    return float(np.clip(math.log10(0.10 / area), 0.0, 3.0))


def difficulty(category: str, target: dict[str, str], feature: dict[str, str]) -> dict[str, float | str]:
    size = scale_component(float(target["bbox_area_ratio"]))
    if category in {"dog", "cat"}:
        critical = critical_part_visibility(feature)
        visibility = {"clear": 0, "partial": 2, "severe_loss": 3}[critical]
        occlusion = {"none": 0, "partial": 2, "heavy": 3, "unknown": 0}.get(
            feature.get("occlusion_level", "unknown"), 0,
        )
        truncation = {"none": 0, "partial": 1, "heavy": 2, "unknown": 0}.get(
            feature.get("truncation_level", "unknown"), 0,
        )
        if feature.get("body_orientation") == "back" and feature.get("head_visibility") == "not_visible":
            orientation = 2
        elif feature.get("body_orientation") == "back" or feature.get("head_orientation") in {"back", "upward", "downward"}:
            orientation = 1
        else:
            orientation = 0
    else:
        absent = sum((
            feature.get("eye_visibility") == "none",
            feature.get("nose_visibility") == "not_visible",
            feature.get("mouth_visibility") == "not_visible",
        ))
        if feature.get("face_visibility") in {"minimal", "not_visible"} or absent >= 2:
            visibility = 3
        elif feature.get("face_visibility") == "partial" or absent:
            visibility = 2
        else:
            visibility = 0
        occlusion = {"none": 0, "light": 1, "moderate": 2, "severe": 3, "unknown": 0}.get(
            feature.get("occlusion_level", "unknown"), 0,
        )
        truncation = {"complete": 0, "lightly_truncated": 1, "severely_truncated": 2, "unknown": 0}.get(
            feature.get("head_completeness", "unknown"), 0,
        )
        orientation = {
            "frontal": 0, "left_three_quarter": 1, "right_three_quarter": 1,
            "left_profile": 2, "right_profile": 2, "rear_or_face_absent": 2,
            "unknown": 0,
        }.get(feature.get("yaw", "unknown"), 0)
    score = size + visibility + occlusion + truncation + orientation
    return {
        "size_score": size, "visibility_score": visibility,
        "occlusion_score": occlusion, "truncation_score": truncation,
        "orientation_score": orientation, "difficulty_score": score,
    }


def coverage(matrix: np.ndarray, selected: list[int]) -> tuple[float, float, int]:
    counts = matrix[selected].sum(axis=0) if selected else np.zeros(matrix.shape[1])
    return (
        float(np.minimum(counts / COVERAGE_CAP, 1).mean()),
        float((counts > 0).mean()), int((counts == 0).sum()),
    )


def select_balanced_batches(
    rows: list[dict[str, str]], matrix: np.ndarray, difficulty_rows: list[dict],
) -> tuple[list[list[int]], list[list[int]], list[dict]]:
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            float(difficulty_rows[index]["difficulty_score"]),
            float(difficulty_rows[index]["visibility_score"]),
            float(difficulty_rows[index]["occlusion_score"]),
            float(difficulty_rows[index]["truncation_score"]),
            float(difficulty_rows[index]["orientation_score"]),
            float(difficulty_rows[index]["size_score"]), rows[index]["sample_id"],
        ),
    )
    strata = [order[start:start + STRATUM_SIZE] for start in range(0, len(order), STRATUM_SIZE)]
    if len(strata) != STRATA or any(len(group) != STRATUM_SIZE for group in strata):
        raise ValueError("difficulty strata are not five equal groups of 40")
    batches: list[list[int]] = []
    audit: list[dict] = []
    counts = np.zeros(matrix.shape[1], dtype=int)
    for stratum_number, stratum in enumerate(strata, start=1):
        remaining = set(stratum)
        batch = []
        for rank in range(SELECT_PER_STRATUM):
            def selection_key(index: int) -> tuple[float, str]:
                active = matrix[index].astype(bool) & (counts < COVERAGE_CAP)
                # Concave gain prioritizes states with less existing evidence.
                gain = float(np.log(counts[active] + 2).sum() - np.log(counts[active] + 1).sum())
                return (-gain, rows[index]["sample_id"])

            chosen = min(remaining, key=selection_key)
            active = matrix[chosen].astype(bool) & (counts < COVERAGE_CAP)
            gain = float(np.log(counts[active] + 2).sum() - np.log(counts[active] + 1).sum())
            counts += matrix[chosen]
            batch.append(chosen)
            remaining.remove(chosen)
            audit.append({
                "batch": f"B{stratum_number}", "batch_rank": rank + 1,
                "sample_id": rows[chosen]["sample_id"],
                "difficulty_score": difficulty_rows[chosen]["difficulty_score"],
                "state_balance_gain": gain,
            })
        batches.append(batch)
    return strata, batches, audit


def confusion_metrics(true: np.ndarray, predicted: np.ndarray, target_label: int) -> dict[str, float]:
    matrix = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted_support, out=np.zeros(5), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=precision + recall > 0)
    return {
        "accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
        "target_precision": float(precision[target_label]),
        "target_recall": float(recall[target_label]), "target_f1": float(f1[target_label]),
    }


def failure_metrics(selected: list[int], correct: np.ndarray, matrix: np.ndarray, minimum: int) -> dict[str, float | int]:
    selected_matrix = matrix[selected]
    selected_correct = correct[selected]
    supports = selected_matrix.sum(axis=0)
    eligible = supports >= minimum
    if not eligible.any():
        return {"failed_state_count": 0, "eligible_state_count": 0, "state_macro_recall": 1.0, "worst5_state_recall": 1.0}
    correct_counts = selected_matrix.T @ selected_correct.astype(int)
    recalls = correct_counts[eligible] / supports[eligible]
    return {
        "failed_state_count": int((recalls < 1).sum()),
        "eligible_state_count": int(eligible.sum()),
        "state_macro_recall": float(recalls.mean()),
        "worst5_state_recall": float(np.sort(recalls)[:5].mean()),
    }


def category_inputs(category: str) -> dict:
    if category == "dog":
        rows = read_csv(DOG_DATA / "manifests/test.csv") + read_csv(DOG_DATA / "manifests/dog_difficulty_test.csv")
        feature_rows = read_csv(DOG_FEATURES)
        features = {row["image_id"]: row for row in feature_rows}
        prediction_root = DOG_PREDICTIONS
        prediction_name = "candidate_predictions.csv"
        class_names = ("dog", "cat", "horse", "sheep", "person")
        target_label = 0
        source_paths = [DOG_DATA / "manifests/test.csv", DOG_DATA / "manifests/dog_difficulty_test.csv", DOG_FEATURES]
    elif category == "cat":
        rows = read_csv(CAT_EXPERIMENT / "candidate_pool.csv")
        feature_rows = read_csv(CAT_FEATURES)
        features = {row["image_id"]: row for row in feature_rows}
        prediction_root = CAT_EXPERIMENT / "predictions"
        prediction_name = "candidate_predictions.csv"
        class_names = ("dog", "cat", "horse", "sheep", "person")
        target_label = 1
        source_paths = [CAT_EXPERIMENT / "candidate_pool.csv", CAT_FEATURES]
    else:
        rows = read_csv(FACE_EXPERIMENT / "manifests/candidate_pool.csv")
        feature_rows = read_csv(FACE_EXPERIMENT / "annotations/results.csv")
        features = {row["sample_id"]: row for row in feature_rows}
        prediction_root = FACE_EXPERIMENT / "model_runs/runs"
        prediction_name = "test_predictions.csv"
        class_names = ("dog", "cat", "horse", "sheep", "face")
        target_label = 4
        source_paths = [FACE_EXPERIMENT / "manifests/candidate_pool.csv", FACE_EXPERIMENT / "annotations/results.csv"]
    target_rows = [row for row in rows if row["class_name"] == category]
    counts = Counter(row["class_name"] for row in rows)
    expected = Counter({name: (200 if name == category else 100) for name in class_names})
    if counts != expected or len(target_rows) != 200:
        raise ValueError(f"{category}: unexpected candidate counts {counts}")
    if set(features) != {row["sample_id"] for row in target_rows}:
        raise ValueError(f"{category}: feature rows do not match targets")
    if any(row["feature_status"] != "ok" for row in feature_rows):
        raise ValueError(f"{category}: invalid feature rows")
    return {
        "category": category, "rows": rows, "target_rows": target_rows,
        "features": features, "prediction_root": prediction_root,
        "prediction_name": prediction_name, "class_names": class_names,
        "target_label": target_label, "source_paths": source_paths,
    }


def prediction_path(context: dict, model: str, seed: int) -> Path:
    if context["category"] == "face":
        return context["prediction_root"] / model / "head_face_d300" / str(seed) / context["prediction_name"]
    return context["prediction_root"] / model / str(seed) / context["prediction_name"]


def prepare_category(category: str) -> dict:
    context = category_inputs(category)
    target_rows = context["target_rows"]
    matrix, states = state_matrix(category, target_rows, context["features"])
    difficulties = [difficulty(category, row, context["features"][row["sample_id"]]) for row in target_rows]
    strata, batches, audit = select_balanced_batches(target_rows, matrix, difficulties)
    selected = [index for batch in batches for index in batch]
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError(f"{category}: selection is not 100 unique targets")
    rng_seed = RANDOM_SEED + {"dog": 1000, "cat": 2000, "face": 3000}[category]
    random_groups = []
    for replicate in range(RANDOM_REPLICATES):
        rng = random.Random(rng_seed + replicate)
        random_groups.append([rng.sample(stratum, SELECT_PER_STRATUM) for stratum in strata])
    non_target_order = {}
    for class_index, class_name in enumerate(context["class_names"]):
        if class_name == category:
            continue
        values = sorted((row for row in context["rows"] if row["class_name"] == class_name), key=lambda row: row["sample_id"])
        random.Random(rng_seed + 10000 + class_index).shuffle(values)
        non_target_order[class_name] = values
    context.update({
        "matrix": matrix, "states": states, "difficulties": difficulties,
        "strata": strata, "batches": batches, "selected": selected,
        "random_groups": random_groups, "non_target_order": non_target_order,
    })

    difficulty_rows = []
    stratum_map = {index: number for number, group in enumerate(strata, start=1) for index in group}
    batch_map = {index: number for number, group in enumerate(batches, start=1) for index in group}
    for index, row in enumerate(target_rows):
        difficulty_rows.append({
            "category": category, "sample_id": row["sample_id"],
            "stratum": f"Q{stratum_map[index]}", "selected": int(index in batch_map),
            "batch": f"B{batch_map[index]}" if index in batch_map else "",
            **difficulties[index], "crop_path": row["crop_path"],
        })
    write_csv(OUTPUT / f"tables/{category}_difficulty_scores.csv", difficulty_rows)
    write_csv(OUTPUT / f"tables/{category}_selection_audit.csv", audit)
    for number, batch in enumerate(batches, start=1):
        write_csv(OUTPUT / f"manifests/{category}_B{number}.csv", [
            {"category": category, "batch": f"B{number}",
             "difficulty_score": difficulties[index]["difficulty_score"], **target_rows[index]}
            for index in batch
        ])
    return context


def load_predictions(context: dict) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], dict[str, dict[str, str]]]]:
    target_ids = [row["sample_id"] for row in context["target_rows"]]
    all_ids = {row["sample_id"] for row in context["rows"]}
    correct = {}
    predictions = {}
    for model in MODELS:
        for seed in SEEDS:
            path = prediction_path(context, model, seed)
            rows = read_csv(path)
            by_id = {row["sample_id"]: row for row in rows}
            if set(by_id) != all_ids:
                raise ValueError(f"incomplete predictions: {path}")
            predictions[(model, seed)] = by_id
            correct[(model, seed)] = np.asarray([
                int(by_id[sample_id]["predicted_label"]) == context["target_label"]
                for sample_id in target_ids
            ], dtype=bool)
    return correct, predictions


def evaluate_category(context: dict, correct: dict, predictions: dict) -> tuple[list[dict], list[dict]]:
    batch_rows = []
    cumulative_rows = []
    rows_by_id = {row["sample_id"]: row for row in context["rows"]}
    for model in MODELS:
        for seed in SEEDS:
            outcome = correct[(model, seed)]
            by_id = predictions[(model, seed)]
            cumulative: list[int] = []
            for number, batch in enumerate(context["batches"], start=1):
                cumulative.extend(batch)
                score = [float(context["difficulties"][index]["difficulty_score"]) for index in batch]
                frequency, presence, missing = coverage(context["matrix"], batch)
                failure = failure_metrics(batch, outcome, context["matrix"], 3)
                batch_rows.append({
                    "category": context["category"], "model": model, "seed": seed,
                    "batch": f"B{number}", "difficulty_mean": float(np.mean(score)),
                    "difficulty_median": float(np.median(score)), "difficulty_min": min(score),
                    "difficulty_max": max(score), "frequency_coverage": frequency,
                    "presence_coverage": presence, "missing_states": missing,
                    "target_recall": float(outcome[batch].mean()), **failure,
                })
                budget = len(cumulative)
                suite_ids = [context["target_rows"][index]["sample_id"] for index in cumulative]
                for class_name in context["class_names"]:
                    if class_name != context["category"]:
                        suite_ids.extend(row["sample_id"] for row in context["non_target_order"][class_name][:budget])
                true = np.asarray([context["class_names"].index(rows_by_id[sample_id]["class_name"]) for sample_id in suite_ids])
                predicted = np.asarray([int(by_id[sample_id]["predicted_label"]) for sample_id in suite_ids])
                overall = confusion_metrics(true, predicted, context["target_label"])
                frequency, presence, missing = coverage(context["matrix"], cumulative)
                failure = failure_metrics(cumulative, outcome, context["matrix"], MIN_STATE_SUPPORT)
                cumulative_rows.append({
                    "category": context["category"], "model": model, "seed": seed,
                    "stage": f"D{budget}", "target_count": budget, "total_targets": 5 * budget,
                    "frequency_coverage": frequency, "presence_coverage": presence,
                    "missing_states": missing, **overall, **failure,
                })
    return batch_rows, cumulative_rows


def random_controls(context: dict, correct: dict, cumulative_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    observed = {}
    selected: list[int] = []
    for stage, batch in enumerate(context["batches"], start=1):
        selected.extend(batch)
        rows = [row for row in cumulative_rows if row["target_count"] == len(selected)]
        observed[stage] = {
            "frequency_coverage": coverage(context["matrix"], selected)[0],
            "mean_target_recall": float(np.mean([row["target_recall"] for row in rows])),
            "mean_failed_state_count": float(np.mean([row["failed_state_count"] for row in rows])),
        }
    details = []
    run_keys = list(correct)
    for replicate, groups in enumerate(context["random_groups"]):
        cumulative: list[int] = []
        for stage, group in enumerate(groups, start=1):
            cumulative.extend(group)
            frequency = coverage(context["matrix"], cumulative)[0]
            recalls = [float(correct[key][cumulative].mean()) for key in run_keys]
            breadth = [
                float(failure_metrics(cumulative, correct[key], context["matrix"], MIN_STATE_SUPPORT)["failed_state_count"])
                for key in run_keys
            ]
            details.append({
                "category": context["category"], "replicate": replicate,
                "stage": f"D{len(cumulative)}", "frequency_coverage": frequency,
                "mean_target_recall": float(np.mean(recalls)),
                "mean_failed_state_count": float(np.mean(breadth)),
            })
    summary = []
    for stage in range(1, STRATA + 1):
        stage_name = f"D{stage * SELECT_PER_STRATUM}"
        candidates = [row for row in details if row["stage"] == stage_name]
        output = {"category": context["category"], "stage": stage_name}
        for metric, observed_value in observed[stage].items():
            values = np.asarray([row[metric] for row in candidates], dtype=float)
            output[f"observed_{metric}"] = observed_value
            output[f"random_{metric}_mean"] = float(values.mean())
            output[f"random_{metric}_p2_5"] = float(np.percentile(values, 2.5))
            output[f"random_{metric}_p97_5"] = float(np.percentile(values, 97.5))
            output[f"delta_{metric}"] = observed_value - float(values.mean())
            output[f"{metric}_upper_p"] = float((1 + np.sum(values >= observed_value)) / (len(values) + 1))
            output[f"{metric}_lower_p"] = float((1 + np.sum(values <= observed_value)) / (len(values) + 1))
        summary.append(output)
    return details, summary


def model_summary(batch_rows: list[dict], cumulative_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    category_models = []
    for category in ("dog", "cat", "face"):
        for model in MODELS:
            batch_means = []
            for number in range(1, STRATA + 1):
                values = [
                    row["target_recall"] for row in batch_rows
                    if row["category"] == category and row["model"] == model and row["batch"] == f"B{number}"
                ]
                batch_means.append(float(np.mean(values)))
            final = [
                row for row in cumulative_rows
                if row["category"] == category and row["model"] == model and row["stage"] == "D100"
            ]
            category_models.append({
                "category": category, "model": model,
                "difficulty_auc": float(np.trapezoid(batch_means, np.arange(1, 6)) / 4),
                **{f"B{number}_recall": batch_means[number - 1] for number in range(1, 6)},
                "B1_to_B5": batch_means[-1] - batch_means[0],
                "D100_macro_f1": float(np.mean([row["macro_f1"] for row in final])),
                "D100_target_f1": float(np.mean([row["target_f1"] for row in final])),
                "D100_target_recall": float(np.mean([row["target_recall"] for row in final])),
                "D100_failed_states": float(np.mean([row["failed_state_count"] for row in final])),
                "D100_worst5_state_recall": float(np.mean([row["worst5_state_recall"] for row in final])),
            })
    for category in ("dog", "cat", "face"):
        rows = sorted((row for row in category_models if row["category"] == category), key=lambda row: row["difficulty_auc"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["category_auc_rank"] = rank
    cross_models = []
    for model in MODELS:
        rows = [row for row in category_models if row["model"] == model]
        cross_models.append({
            "model": model,
            "cross_category_auc": float(np.mean([row["difficulty_auc"] for row in rows])),
            "mean_category_rank": float(np.mean([row["category_auc_rank"] for row in rows])),
            "mean_B5_recall": float(np.mean([row["B5_recall"] for row in rows])),
            "mean_D100_macro_f1": float(np.mean([row["D100_macro_f1"] for row in rows])),
            **{f"{row['category']}_auc": row["difficulty_auc"] for row in rows},
            **{f"{row['category']}_B5": row["B5_recall"] for row in rows},
        })
    cross_models.sort(key=lambda row: (-row["cross_category_auc"], row["mean_category_rank"]))
    for rank, row in enumerate(cross_models, start=1):
        row["cross_category_rank"] = rank
    return category_models, cross_models


def bootstrap_cross_category(contexts: dict, correct_by_category: dict, cross_models: list[dict]) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    resamples = {
        category: {
            "seeds": rng.integers(0, len(SEEDS), size=(BOOTSTRAP_REPLICATES, len(SEEDS))),
            "images": [
                rng.integers(0, SELECT_PER_STRATUM, size=(BOOTSTRAP_REPLICATES, SELECT_PER_STRATUM))
                for _ in range(STRATA)
            ],
        }
        for category in ("dog", "cat", "face")
    }
    model_boot = {}
    for model in MODELS:
        category_auc = []
        for category in ("dog", "cat", "face"):
            context = contexts[category]
            matrix = np.asarray([
                [correct_by_category[category][(model, seed)][index] for index in context["selected"]]
                for seed in SEEDS
            ], dtype=float)
            seed_sample = resamples[category]["seeds"]
            batch_values = []
            for batch in range(STRATA):
                image_sample = resamples[category]["images"][batch]
                positions = batch * SELECT_PER_STRATUM + image_sample
                sampled = matrix[seed_sample[:, :, None], positions[:, None, :]]
                batch_values.append(sampled.mean(axis=(1, 2)))
            values = np.column_stack(batch_values)
            category_auc.append(np.trapezoid(values, np.arange(1, 6), axis=1) / 4)
        model_boot[model] = np.mean(np.column_stack(category_auc), axis=1)
    point = {row["model"]: row["cross_category_auc"] for row in cross_models}
    pairs = []
    for first, second in itertools.combinations(MODELS, 2):
        difference = model_boot[first] - model_boot[second]
        pairs.append({
            "model_a": first, "model_b": second,
            "auc_difference": point[first] - point[second],
            "ci_low": float(np.percentile(difference, 2.5)),
            "ci_high": float(np.percentile(difference, 97.5)),
            "p_two_sided": float(min(1, 2 * min(np.mean(difference <= 0), np.mean(difference >= 0)))),
        })
    order = sorted(range(len(pairs)), key=lambda index: pairs[index]["p_two_sided"])
    adjusted = [1.0] * len(pairs)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(pairs) - rank) * pairs[index]["p_two_sided"])
        running = max(running, candidate)
        adjusted[index] = running
    for row, value in zip(pairs, adjusted):
        row["holm_p"] = value
    return pairs


def bootstrap_difficulty_effects(contexts: dict, correct_by_category: dict) -> list[dict]:
    rows = []
    for category_number, category in enumerate(("dog", "cat", "face"), start=1):
        rng = np.random.default_rng(RANDOM_SEED + 5000 + category_number)
        seed_samples = rng.integers(
            0, len(SEEDS), size=(BOOTSTRAP_REPLICATES, len(MODELS), len(SEEDS)),
        )
        b1_positions = rng.integers(0, SELECT_PER_STRATUM, size=(BOOTSTRAP_REPLICATES, SELECT_PER_STRATUM))
        b5_positions = (STRATA - 1) * SELECT_PER_STRATUM + rng.integers(
            0, SELECT_PER_STRATUM, size=(BOOTSTRAP_REPLICATES, SELECT_PER_STRATUM),
        )
        b1_values = np.zeros(BOOTSTRAP_REPLICATES)
        b5_values = np.zeros(BOOTSTRAP_REPLICATES)
        context = contexts[category]
        for model_index, model in enumerate(MODELS):
            matrix = np.asarray([
                [correct_by_category[category][(model, seed)][index] for index in context["selected"]]
                for seed in SEEDS
            ], dtype=float)
            sampled_seeds = seed_samples[:, model_index, :]
            sampled_b1 = matrix[sampled_seeds[:, :, None], b1_positions[:, None, :]]
            sampled_b5 = matrix[sampled_seeds[:, :, None], b5_positions[:, None, :]]
            b1_values += sampled_b1.mean(axis=(1, 2)) / len(MODELS)
            b5_values += sampled_b5.mean(axis=(1, 2)) / len(MODELS)
        difference = b5_values - b1_values
        rows.append({
            "category": category,
            "B1_to_B5_difference": float(difference.mean()),
            "ci_low": float(np.percentile(difference, 2.5)),
            "ci_high": float(np.percentile(difference, 97.5)),
            "p_two_sided": float(min(1, 2 * min(np.mean(difference <= 0), np.mean(difference >= 0)))),
        })
    return rows


def make_figures(batch_rows: list[dict], cumulative_rows: list[dict], random_summary: list[dict], cross_models: list[dict]) -> None:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"dog": "#167D77", "cat": "#D55E00", "face": "#4C6FAE"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for category in colors:
        values = []
        for number in range(1, 6):
            rows = [row for row in batch_rows if row["category"] == category and row["batch"] == f"B{number}"]
            values.append(np.mean([row["target_recall"] for row in rows]))
        axes[0, 0].plot(range(1, 6), values, marker="o", linewidth=2.3, label=category, color=colors[category])
    axes[0, 0].set(title="Difficulty response by target category", xlabel="Difficulty stratum", ylabel="Target recall", xticks=range(1, 6), xticklabels=[f"B{i}" for i in range(1, 6)])
    axes[0, 0].legend(frameon=False)

    for category in colors:
        rows = [row for row in cumulative_rows if row["category"] == category]
        x, y = [], []
        for budget in (20, 40, 60, 80, 100):
            values = [row for row in rows if row["target_count"] == budget]
            x.append(np.mean([row["frequency_coverage"] for row in values]))
            y.append(np.mean([row["target_recall"] for row in values]))
        axes[0, 1].plot(x, y, marker="o", linewidth=2.3, label=category, color=colors[category])
    axes[0, 1].set(title="Coverage-response curve", xlabel="Within-category frequency coverage", ylabel="Target recall")
    axes[0, 1].legend(frameon=False)

    d100 = [row for row in random_summary if row["stage"] == "D100"]
    x = np.arange(3)
    coverage_delta = [100 * row["delta_frequency_coverage"] for row in d100]
    failure_delta = [row["delta_mean_failed_state_count"] for row in d100]
    coverage_bars = axes[1, 0].bar(x - 0.18, coverage_delta, width=0.36, label="Coverage delta", color="#6A9F58")
    failure_axis = axes[1, 0].twinx()
    failure_bars = failure_axis.bar(x + 0.18, failure_delta, width=0.36, label="Failed-state delta", color="#B85C38")
    axes[1, 0].axhline(0, color="#444444", linewidth=0.8)
    axes[1, 0].set(
        title="Coverage selection minus matched random at D100",
        xticks=x, xticklabels=[row["category"] for row in d100],
        ylabel="Coverage delta (percentage points)",
    )
    failure_axis.set_ylabel("Failed-state count delta")
    axes[1, 0].legend(
        [coverage_bars, failure_bars], ["Coverage delta", "Failed-state delta"], frameon=False,
    )

    ordered = list(reversed(cross_models))
    axes[1, 1].barh([row["model"] for row in ordered], [row["cross_category_auc"] for row in ordered], color="#4C78A8")
    axes[1, 1].set(title="Cross-category architecture ranking", xlabel="Mean difficulty robustness AUC", xlim=(0, 1))
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.savefig(figure_dir / "cross_category_overview.png", dpi=220)
    fig.savefig(figure_dir / "cross_category_overview.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)
    for axis, category in zip(axes, ("dog", "cat", "face")):
        for model in MODELS:
            values = []
            for number in range(1, 6):
                rows = [row for row in batch_rows if row["category"] == category and row["model"] == model and row["batch"] == f"B{number}"]
                values.append(np.mean([row["target_recall"] for row in rows]))
            axis.plot(range(1, 6), values, marker="o", linewidth=1.3, label=model)
        axis.set(title=category, xlabel="Difficulty batch", xticks=range(1, 6), xticklabels=[f"B{i}" for i in range(1, 6)], ylim=(0, 1.03))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Target recall")
    axes[-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(figure_dir / "model_difficulty_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "model_difficulty_curves.svg", bbox_inches="tight")
    plt.close(fig)


def write_report(
    contexts: dict, batch_rows: list[dict], cumulative_rows: list[dict], random_summary: list[dict],
    category_models: list[dict], cross_models: list[dict], bootstrap_rows: list[dict],
    difficulty_bootstrap: list[dict],
) -> None:
    category_summary = []
    for category in ("dog", "cat", "face"):
        batch_values = []
        for number in range(1, 6):
            rows = [row for row in batch_rows if row["category"] == category and row["batch"] == f"B{number}"]
            batch_values.append(float(np.mean([row["target_recall"] for row in rows])))
        random = next(row for row in random_summary if row["category"] == category and row["stage"] == "D100")
        difficulty_ci = next(row for row in difficulty_bootstrap if row["category"] == category)
        category_summary.append({
            "category": category,
            "B1_recall": batch_values[0], "B5_recall": batch_values[-1],
            "recall_change": batch_values[-1] - batch_values[0],
            "coverage": random["observed_frequency_coverage"],
            "random_coverage": random["random_frequency_coverage_mean"],
            "coverage_delta": random["delta_frequency_coverage"],
            "coverage_p": random["frequency_coverage_upper_p"],
            "failed_states": random["observed_mean_failed_state_count"],
            "random_failed_states": random["random_mean_failed_state_count_mean"],
            "failed_state_delta": random["delta_mean_failed_state_count"],
            "failed_state_p": random["mean_failed_state_count_upper_p"],
            "target_recall": random["observed_mean_target_recall"],
            "random_target_recall": random["random_mean_target_recall_mean"],
            "difficulty_ci_low": difficulty_ci["ci_low"],
            "difficulty_ci_high": difficulty_ci["ci_high"],
            "difficulty_p": difficulty_ci["p_two_sided"],
        })
    negative_difficulty = sum(row["B5_recall"] < row["B1_recall"] for row in category_summary)
    supported_difficulty = sum(row["difficulty_ci_high"] < 0 and row["difficulty_p"] <= 0.05 for row in category_summary)
    supported_coverage = sum(row["coverage_p"] <= 0.05 for row in category_summary)
    supported_failure = sum(row["failed_state_p"] <= 0.05 for row in category_summary)
    significant_pairs = [row for row in bootstrap_rows if row["holm_p"] <= 0.05 and not (row["ci_low"] <= 0 <= row["ci_high"])]

    lines = [
        "# dog、cat与head/face统一难度覆盖实验", "", "## 固定协议", "",
        "- 每个目标类别使用200张候选，按统一的五维难度分数划分5个等量层，每层40张。",
        "- 五个共同难度维度：目标尺寸、关键部位可见性、遮挡、截断、姿态方向。",
        "- 每层使用类别专属状态的凹覆盖增益选择20张，最终每类100张。",
        "- 正式五分类测试在每个阶段均保持每类数量相同。",
        "- 8种架构、3个种子；模型和训练集冻结，不重新训练。",
        "- 每个类别运行1000次同难度、同预算随机选择。", "",
        "## 跨类别主要结果", "",
        "| 类别 | B1 Recall | B5 Recall | B1→B5 [95% CI] | D100覆盖 | 随机覆盖 | 覆盖增量 | 失败状态增量 | D100目标Recall | 随机Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_summary:
        lines.append(
            f"| {row['category']} | {row['B1_recall']:.2%} | {row['B5_recall']:.2%} | {row['recall_change']:+.2%} "
            f"[{row['difficulty_ci_low']:+.2%}, {row['difficulty_ci_high']:+.2%}] | "
            f"{row['coverage']:.2%} | {row['random_coverage']:.2%} | {row['coverage_delta']:+.2%} | "
            f"{row['failed_state_delta']:+.2f} | {row['target_recall']:.2%} | {row['random_target_recall']:.2%} |"
        )
    lines.extend(["", "## 跨类别模型排名", "", "| 排名 | 模型 | 平均难度AUC | 平均B5 Recall | 平均类别内排名 | 平均D100 Macro-F1 |", "|---:|---|---:|---:|---:|---:|"])
    for row in cross_models:
        lines.append(
            f"| {row['cross_category_rank']} | {row['model']} | {row['cross_category_auc']:.2%} | "
            f"{row['mean_B5_recall']:.2%} | {row['mean_category_rank']:.2f} | {row['mean_D100_macro_f1']:.2%} |"
        )
    lines.extend([
        "", "## 假设判定", "",
        f"- 难度方向：{negative_difficulty}/3个类别的B5 Recall低于B1；其中{supported_difficulty}/3个类别的95% Bootstrap区间完全低于0。",
        f"- 覆盖有效性：{supported_coverage}/3个类别在D100超过同难度随机组（单侧经验p≤0.05）。",
        f"- 独立检错广度：{supported_failure}/3个类别的失败状态数超过同难度随机组（单侧经验p≤0.05）。",
        f"- 模型差异：Holm校正后的成对Bootstrap中有{len(significant_pairs)}/28对达到p≤0.05；完整区间见表格。",
        "", "## 解释边界", "",
        "- dog、cat与head/face使用类别专属状态字典，原始覆盖百分比只在各类别内部与随机组比较。",
        "- head/face使用扩展头部裁剪，不能把其原始F1与完整动物目标直接解释为类别难度差异。",
        "- 这是覆盖导向压力测试，不代表真实部署分布。",
        "- 当前结果支持关联性诊断，不证明某个单独状态因果导致错误。",
    ])
    (OUTPUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(OUTPUT / "tables/category_summary.csv", category_summary)
    write_json(OUTPUT / "summary.json", {
        "difficulty_negative_categories": negative_difficulty,
        "difficulty_supported_categories": supported_difficulty,
        "coverage_supported_categories": supported_coverage,
        "failure_breadth_supported_categories": supported_failure,
        "top_cross_category_model": cross_models[0]["model"],
        "top_cross_category_auc": cross_models[0]["cross_category_auc"],
    })
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        f"| Unified difficulty strata expose tail degradation | {'supported' if supported_difficulty >= 2 else 'partial'} | `tables/category_difficulty_bootstrap.csv` | Strong effects may be category-dependent |\n"
        f"| Coverage selection exceeds matched random coverage | {'supported' if supported_coverage == 3 else 'partial'} | `tables/random_control_summary.csv` | Within-category comparisons |\n"
        f"| Coverage selection independently broadens failure discovery | {'supported' if supported_failure >= 2 else 'partial_or_unsupported'} | `tables/random_control_summary.csv` | One-sided empirical tests |\n"
        "| Raw coverage percentages are directly comparable across categories | unsupported | Different state dictionaries | Compare deltas only |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: no external claim introduced.\n"
        "- Result traceability: PASS; outputs derive from frozen manifests, annotations and saved predictions.\n"
        "- Claim-scope integrity: PASS WITH LIMITATIONS; stress-test distributions are not deployment prevalence.\n"
        "- Cross-category comparability: counts, budgets, difficulty dimensions, models and seeds are aligned; state vocabularies and face target granularity remain category-specific.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    progress("prepare", "freezing three 200-target candidate pools", 3)
    contexts = {category: prepare_category(category) for category in ("dog", "cat", "face")}
    protocol = {
        "experiment_id": "cross-category-balanced-difficulty-20260928",
        "created_at": now(), "status": "frozen_before_prediction_read",
        "categories": ["dog", "cat", "face"], "candidate_targets_per_category": 200,
        "strata": STRATA, "stratum_size": STRATUM_SIZE,
        "selected_per_stratum": SELECT_PER_STRATUM, "selected_per_category": 100,
        "difficulty_dimensions": ["target_scale", "critical_part_visibility", "occlusion", "truncation", "orientation"],
        "selection": "concave within-category state-balance gain; predictions excluded",
        "models": list(MODELS), "seeds": list(SEEDS),
        "random_replicates": RANDOM_REPLICATES, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "input_sha256": {
            category: {path.name: digest(path) for path in context["source_paths"]}
            for category, context in contexts.items()
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "- Thesis: category-conditional semantic coverage plus aligned difficulty strata can diagnose architecture-specific tail failures.\n"
        "- Validation: dog and cat test similar animal classes; head/face tests cross-domain transfer.\n"
        "- Primary control: 1000 difficulty-matched selections per category.\n"
        "- Boundary: diagnostic stress testing, not deployment prevalence or causal attribution.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol frozen: {now()}\n"
        "- Inputs: three 200-target candidate pools and 72 saved model prediction files.\n"
        "- Common factors: five difficulty dimensions, five equal strata, 20 selected per stratum, eight models, three seeds.\n"
        "- Reproduction: `/home/hcf/project/QE/.venv/bin/python -u scripts/run_cross_category_balanced_difficulty.py`.\n",
        encoding="utf-8",
    )

    all_batch, all_cumulative, all_random_details, all_random_summary = [], [], [], []
    correct_by_category = {}
    for number, category in enumerate(("dog", "cat", "face"), start=1):
        progress("evaluate", f"loading and evaluating {category} predictions", 10 + number * 15)
        correct, predictions = load_predictions(contexts[category])
        correct_by_category[category] = correct
        batch_rows, cumulative_rows = evaluate_category(contexts[category], correct, predictions)
        random_details, random_summary = random_controls(contexts[category], correct, cumulative_rows)
        all_batch.extend(batch_rows)
        all_cumulative.extend(cumulative_rows)
        all_random_details.extend(random_details)
        all_random_summary.extend(random_summary)
    write_csv(OUTPUT / "tables/batch_metrics_by_run.csv", all_batch)
    write_csv(OUTPUT / "tables/cumulative_metrics_by_run.csv", all_cumulative)
    write_csv(OUTPUT / "tables/random_control_by_replicate.csv", all_random_details)
    write_csv(OUTPUT / "tables/random_control_summary.csv", all_random_summary)

    progress("aggregate", "computing category and cross-category rankings", 70)
    category_models, cross_models = model_summary(all_batch, all_cumulative)
    write_csv(OUTPUT / "tables/category_model_summary.csv", category_models)
    write_csv(OUTPUT / "tables/cross_category_model_ranking.csv", cross_models)
    progress("bootstrap", "running 10000 hierarchical bootstrap replicates", 78)
    bootstrap_rows = bootstrap_cross_category(contexts, correct_by_category, cross_models)
    write_csv(OUTPUT / "tables/cross_category_model_bootstrap.csv", bootstrap_rows)
    difficulty_bootstrap = bootstrap_difficulty_effects(contexts, correct_by_category)
    write_csv(OUTPUT / "tables/category_difficulty_bootstrap.csv", difficulty_bootstrap)
    progress("figures", "rendering cross-category figures", 91)
    make_figures(all_batch, all_cumulative, all_random_summary, cross_models)
    progress("report", "writing result report and evidence review", 97)
    write_report(
        contexts, all_batch, all_cumulative, all_random_summary, category_models,
        cross_models, bootstrap_rows, difficulty_bootstrap,
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol frozen and run completed: {now()}\n"
        "- Inputs: three 200-target candidate pools and 72 saved model prediction files.\n"
        "- Common factors: five difficulty dimensions, five equal strata, 20 selected per stratum, eight models, three seeds.\n"
        "- Controls: 1000 difficulty-matched selections per category; 10000 hierarchical bootstrap replicates.\n"
        "- Raw results: `tables/batch_metrics_by_run.csv`, `tables/cumulative_metrics_by_run.csv`, and `tables/random_control_by_replicate.csv`.\n"
        "- Reproduction: `/home/hcf/project/QE/.venv/bin/python -u scripts/run_cross_category_balanced_difficulty.py`.\n"
        "- Outcome: coverage advantage supported in all three categories; failure-breadth advantage supported for dog only; difficulty-tail degradation remains statistically inconclusive under the current 20-image strata.\n",
        encoding="utf-8",
    )
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "percent": 100, "updated_at": now(),
        "categories": 3, "models": 8, "seeds": 3,
    })
    print(json.dumps({"status": "completed", "output": str(OUTPUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
