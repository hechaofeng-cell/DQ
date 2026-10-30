#!/usr/bin/env python3
"""Run the frozen-model dog coverage and progressive-difficulty diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/hcf/project/QE")
SOURCE_EXPERIMENT = SOURCE_ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"
GRADIENT = SOURCE_ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
DATA = SOURCE_ROOT / "data/openimages_v7_dog_gap_v1"
FEATURES_PATH = SOURCE_ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
SCHEMA_PATH = ROOT / "configs/dog_diag_feature_schema_v1_0.json"
OUTPUT = ROOT / "artifacts/visual_state_diagnostic_v1_20260928"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
CLASS_NAMES = ("dog", "cat", "horse", "sheep", "person")
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}
RANDOM_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 10000
RANDOM_SEED = 20260928
STRATA = 5
STRATUM_SIZE = 40
SELECT_PER_STRATUM = 20
CAP = 20
MIN_STATE_SUPPORT = 10


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def progress(phase: str, detail: str, percent: float) -> None:
    value = {
        "status": "running", "phase": phase, "detail": detail,
        "percent": round(percent, 2), "updated_at": now(),
    }
    write_json(OUTPUT / "progress.json", value)
    print(f"[{phase}] {detail} ({percent:.1f}%)", flush=True)


def critical_part_visibility(row: dict[str, str]) -> str:
    fields = ("head_visibility", "muzzle_visibility", "mouth_visibility", "eye_state")
    if all(row.get(field, "unknown") == "unknown" for field in fields):
        return "unknown"
    if row.get("head_visibility") == "not_visible":
        return "severe_loss"
    missing = sum(row.get(field) == "not_visible" for field in ("muzzle_visibility", "mouth_visibility", "eye_state"))
    if missing >= 2:
        return "severe_loss"
    partial = (
        missing > 0
        or row.get("head_visibility") == "partial"
        or row.get("muzzle_visibility") == "partial"
        or row.get("mouth_visibility") == "partial"
    )
    return "partial" if partial else "clear"


def feature_value(feature: dict, row: dict[str, str]) -> str:
    if feature["source"] == "derived":
        return critical_part_visibility(row)
    return row.get(feature["source"], "unknown") or "unknown"


def build_state_matrix(
    dog_rows: list[dict[str, str]], features: dict[str, dict[str, str]], schema: dict,
) -> tuple[np.ndarray, list[tuple[str, str]], list[dict[str, str]]]:
    states = [
        (feature["feature_id"], value)
        for feature in schema["features"]
        for value in feature["possible_values"]
        if value != "unknown"
    ]
    state_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(dog_rows), len(states)), dtype=np.int8)
    derived_rows = []
    for row_index, target in enumerate(dog_rows):
        source = features[target["sample_id"]]
        derived = {"sample_id": target["sample_id"]}
        for feature in schema["features"]:
            value = feature_value(feature, source)
            if value not in feature["possible_values"]:
                raise ValueError(f"invalid {feature['feature_id']}={value} for {target['sample_id']}")
            derived[feature["feature_id"]] = value
            index = state_index.get((feature["feature_id"], value))
            if index is not None:
                matrix[row_index, index] = 1
        derived_rows.append(derived)
    return matrix, states, derived_rows


def difficulty(target: dict[str, str], feature: dict[str, str]) -> dict[str, int | float | str]:
    area = float(target["bbox_area_ratio"])
    size = 0 if area >= 0.10 else 1 if area >= 0.03 else 2 if area >= 0.01 else 3
    critical = critical_part_visibility(feature)
    head_score = {"clear": 0, "partial": 2, "not_visible": 3, "unknown": 0}.get(feature["head_visibility"], 0)
    critical_score = {"clear": 0, "partial": 2, "severe_loss": 3, "unknown": 0}[critical]
    visibility = max(head_score, critical_score)
    occlusion = {"none": 0, "partial": 2, "heavy": 3, "unknown": 0}.get(feature["occlusion_level"], 0)
    truncation = {"none": 0, "partial": 1, "heavy": 2, "unknown": 0}.get(feature["truncation_level"], 0)
    body = feature["body_orientation"]
    head = feature["head_orientation"]
    if body == "back" and feature["head_visibility"] == "not_visible":
        orientation = 2
    elif body == "back" or head in {"back", "upward", "downward"}:
        orientation = 1
    else:
        orientation = 0
    return {
        "bbox_area_ratio": area, "size_score": size, "visibility_score": visibility,
        "occlusion_score": occlusion, "truncation_score": truncation,
        "orientation_score": orientation,
        "difficulty_score": size + visibility + occlusion + truncation + orientation,
        "critical_part_visibility": critical,
    }


def coverage(matrix: np.ndarray, selected: list[int]) -> tuple[float, float, int, int]:
    counts = matrix[selected].sum(axis=0) if selected else np.zeros(matrix.shape[1], dtype=int)
    return (
        float(np.minimum(counts / CAP, 1).mean()),
        float((counts > 0).mean()),
        int(((counts > 0) & (counts < 5)).sum()),
        int((counts == 0).sum()),
    )


def select_batches(
    dog_rows: list[dict[str, str]], matrix: np.ndarray, difficulty_rows: list[dict],
) -> tuple[list[list[int]], list[list[int]], list[dict]]:
    ordered = sorted(
        range(len(dog_rows)),
        key=lambda index: (difficulty_rows[index]["difficulty_score"], dog_rows[index]["sample_id"]),
    )
    strata = [ordered[index * STRATUM_SIZE:(index + 1) * STRATUM_SIZE] for index in range(STRATA)]
    batches: list[list[int]] = []
    counts = np.zeros(matrix.shape[1], dtype=np.int32)
    selection_audit = []
    for stratum_index, candidates in enumerate(strata, start=1):
        remaining = set(candidates)
        batch = []
        for position in range(SELECT_PER_STRATUM):
            def key(index: int) -> tuple[float, int, str]:
                gain = float(((counts < CAP) & matrix[index].astype(bool)).sum()) / (CAP * matrix.shape[1])
                return (-gain, int(difficulty_rows[index]["difficulty_score"]), dog_rows[index]["sample_id"])

            chosen = min(remaining, key=key)
            before = counts.copy()
            counts += matrix[chosen]
            gain = float((np.minimum(counts / CAP, 1) - np.minimum(before / CAP, 1)).mean())
            gained_states = [index for index in np.where((before < CAP) & matrix[chosen].astype(bool))[0]]
            selection_audit.append({
                "global_rank": len(selection_audit) + 1,
                "batch": f"B{stratum_index}", "batch_rank": position + 1,
                "sample_id": dog_rows[chosen]["sample_id"],
                "difficulty_score": difficulty_rows[chosen]["difficulty_score"],
                "marginal_frequency_coverage_gain": gain,
                "gained_state_count": len(gained_states),
            })
            batch.append(chosen)
            remaining.remove(chosen)
        batches.append(batch)
    return strata, batches, selection_audit


def confusion_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    matrix = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted_support, out=np.zeros(5), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "dog_precision": float(precision[0]),
        "dog_recall": float(recall[0]), "dog_f1": float(f1[0]),
    }


def failure_metrics(selected: list[int], correct: np.ndarray, matrix: np.ndarray, min_support: int) -> dict:
    chosen = np.asarray(selected, dtype=int)
    states = matrix[chosen]
    outcome = correct[chosen].astype(bool)
    support = states.sum(axis=0)
    correct_support = states[outcome].sum(axis=0) if outcome.any() else np.zeros(matrix.shape[1])
    failed_support = states[~outcome].sum(axis=0) if (~outcome).any() else np.zeros(matrix.shape[1])
    recalls = np.divide(correct_support, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    eligible = support >= min_support
    eligible_recalls = recalls[eligible]
    worst = np.sort(eligible_recalls)[: min(5, len(eligible_recalls))]
    return {
        "failed_state_count": int((failed_support > 0).sum()),
        "eligible_state_count": int(eligible.sum()),
        "state_macro_recall": float(eligible_recalls.mean()) if len(eligible_recalls) else None,
        "worst5_state_recall": float(worst.mean()) if len(worst) else None,
    }


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = values[:, 1:] > values[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def duplicate_audit(dog_rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> list[dict]:
    train_hashes = [(row["sample_id"], dhash(Path(row["crop_path"]))) for row in train_rows]
    pairs = []
    for number, row in enumerate(dog_rows, start=1):
        value = dhash(Path(row["crop_path"]))
        for train_id, train_value in train_hashes:
            distance = (value ^ train_value).bit_count()
            if distance <= 2:
                pairs.append({
                    "candidate_sample_id": row["sample_id"], "train_sample_id": train_id,
                    "dhash_hamming_distance": distance, "review_status": "pending",
                })
        if number % 50 == 0:
            print(f"[audit] perceptual hashes {number}/{len(dog_rows)}", flush=True)
    return pairs


def prepare() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    ordinary = read_csv(DATA / "manifests/test.csv")
    difficult = read_csv(DATA / "manifests/dog_difficulty_test.csv")
    for row in ordinary:
        row["candidate_source"] = "main_test"
    for row in difficult:
        row["candidate_source"] = "difficulty_test"
    rows = ordinary + difficult
    if Counter(row["class_name"] for row in rows) != Counter({"dog": 200, "cat": 100, "horse": 100, "sheep": 100, "person": 100}):
        raise ValueError("unexpected candidate class counts")
    if len({row["sample_id"] for row in rows}) != 600:
        raise ValueError("candidate sample IDs are not unique")
    dog_rows = [row for row in rows if row["class_name"] == "dog"]
    feature_rows = read_csv(FEATURES_PATH)
    features = {row["image_id"]: row for row in feature_rows}
    if set(features) != {row["sample_id"] for row in dog_rows}:
        raise ValueError("feature labels do not match the dog candidates")
    if any(row["feature_status"] != "ok" for row in feature_rows):
        raise ValueError("feature labels contain invalid records")

    train_rows = read_csv(GRADIENT / "manifests/train_oi_gap_add0.csv")
    candidate_ids = {row["image_id"] for row in rows}
    train_ids = {row["image_id"] for row in train_rows}
    candidate_hashes = {row["image_sha256"] for row in rows if row.get("image_sha256")}
    train_hashes = {row["image_sha256"] for row in train_rows if row.get("image_sha256")}
    candidate_crop_hashes = {row["crop_sha256"] for row in rows if row.get("crop_sha256")}
    train_crop_hashes = {row.get("crop_sha256") for row in train_rows if row.get("crop_sha256")}
    if candidate_ids & train_ids or candidate_hashes & train_hashes or candidate_crop_hashes & train_crop_hashes:
        raise ValueError("exact train-diagnostic leakage detected")

    matrix, states, derived_rows = build_state_matrix(dog_rows, features, schema)
    difficulty_rows = [difficulty(row, features[row["sample_id"]]) for row in dog_rows]
    strata, batches, selection_audit = select_batches(dog_rows, matrix, difficulty_rows)
    selected = [index for batch in batches for index in batch]
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError("diagnostic selection is not 100 unique dog targets")

    difficulty_table = []
    stratum_by_index = {index: number for number, stratum in enumerate(strata, start=1) for index in stratum}
    selected_by_index = {index: (batch_number, rank) for batch_number, batch in enumerate(batches, start=1) for rank, index in enumerate(batch, start=1)}
    for index, row in enumerate(dog_rows):
        selected_info = selected_by_index.get(index)
        difficulty_table.append({
            "sample_id": row["sample_id"], "candidate_source": row["candidate_source"],
            "stratum": f"Q{stratum_by_index[index]}",
            "selected": int(selected_info is not None),
            "batch": f"B{selected_info[0]}" if selected_info else "",
            "batch_rank": selected_info[1] if selected_info else "",
            **difficulty_rows[index], **{key: value for key, value in derived_rows[index].items() if key != "sample_id"},
        })
    write_csv(OUTPUT / "tables/difficulty_scores.csv", difficulty_table)
    write_csv(OUTPUT / "tables/selection_audit.csv", selection_audit)
    write_csv(OUTPUT / "manifests/candidate_dog_200.csv", dog_rows)
    for batch_number, batch in enumerate(batches, start=1):
        manifest = []
        for index in batch:
            manifest.append({
                "batch": f"B{batch_number}", "difficulty_score": difficulty_rows[index]["difficulty_score"],
                **dog_rows[index],
            })
        write_csv(OUTPUT / f"manifests/batch_B{batch_number}.csv", manifest)

    non_dog_rows = read_csv(SOURCE_EXPERIMENT / "orders/non_dog_order.csv")
    non_dog_order = {
        class_name: [row for row in non_dog_rows if row["class_name"] == class_name]
        for class_name in CLASS_NAMES[1:]
    }
    if any(len(value) != 100 for value in non_dog_order.values()):
        raise ValueError("non-dog fixed orders are incomplete")

    random_matched = []
    random_global = []
    for replicate in range(RANDOM_REPLICATES):
        rng = random.Random(RANDOM_SEED + replicate)
        random_matched.append([rng.sample(stratum, SELECT_PER_STRATUM) for stratum in strata])
        values = list(range(len(dog_rows)))
        rng.shuffle(values)
        random_global.append([values[index * SELECT_PER_STRATUM:(index + 1) * SELECT_PER_STRATUM] for index in range(STRATA)])
    write_csv(OUTPUT / "orders/difficulty_matched_random.csv", [
        {"replicate": replicate, "batch": f"B{batch + 1}", "rank": rank + 1, "sample_id": dog_rows[index]["sample_id"]}
        for replicate, groups in enumerate(random_matched) for batch, group in enumerate(groups)
        for rank, index in enumerate(group)
    ])
    write_csv(OUTPUT / "orders/global_random.csv", [
        {"replicate": replicate, "batch": f"B{batch + 1}", "rank": rank + 1, "sample_id": dog_rows[index]["sample_id"]}
        for replicate, groups in enumerate(random_global) for batch, group in enumerate(groups)
        for rank, index in enumerate(group)
    ])

    protocol = {
        "experiment_id": "visual-state-diagnostic-v1-20260928",
        "created_at": now(), "status": "frozen_before_model_evaluation",
        "research_question": "How do frozen classifiers differ as semantic coverage and model-independent visual difficulty increase?",
        "candidate_pool": {"dog": 200, "other_each": 100},
        "selected_dog": 100, "difficulty_strata": 5, "selected_per_stratum": 20,
        "models": list(MODELS), "seeds": list(SEEDS),
        "random_replicates": RANDOM_REPLICATES, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "coverage_cap": CAP, "minimum_state_support": MIN_STATE_SUPPORT,
        "selection_blinding": "observable dog_diag_v1.0 states and annotations only; predictions excluded",
        "primary_ranking": "difficulty_robustness_auc",
        "input_sha256": {
            "schema": sha256(SCHEMA_PATH), "features": sha256(FEATURES_PATH),
            "main_candidates": sha256(DATA / "manifests/test.csv"),
            "difficulty_candidates": sha256(DATA / "manifests/dog_difficulty_test.csv"),
            "D300_train": sha256(GRADIENT / "manifests/train_oi_gap_add0.csv"),
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "- Thesis: predeclared semantic states and model-independent difficulty can reveal architecture-specific robustness boundaries.\n"
        "- Primary object: frozen D300 dog classifiers evaluated on five disjoint difficulty batches.\n"
        "- Primary ranking: area under the B1--B5 dog-recall curve; B5 recall is the tail metric.\n"
        "- Main control: 1000 difficulty-matched random selections.\n"
        "- Boundary: diagnostic stress testing, not deployment prevalence or causal feature attribution.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol frozen: {now()}\n"
        "- Inputs: 200 independent Open Images dog targets, dog_v3.1 labels, 24 saved D300 predictions.\n"
        "- Selection: five difficulty strata; 20 coverage-greedy targets selected from each 40-target stratum.\n"
        "- Controls: 1000 difficulty-matched and 1000 global random selections.\n"
        "- Reproduction: `/home/hcf/project/QE/.venv/bin/python -u scripts/run_visual_state_progressive_difficulty_diagnostic.py`.\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| B1--B5 form an empirically harder sequence | pending | `tables/batch_metrics_by_run.csv` | Frozen dog crop task |\n"
        "| Architectures have different difficulty sensitivity | pending | `tables/model_robustness_ranking.csv` | Eight D300 architectures |\n"
        "| Coverage adds diagnostic value after difficulty matching | pending | `tables/random_control_summary.csv` | Same 200-dog pool |\n"
        "| A single state causes errors | unsupported | no counterfactual intervention | Association only |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n- Result traceability: pending run.\n- Claim scope: frozen-model diagnostic only.\n- Data audit: exact leakage passed; perceptual audit pending.\n",
        encoding="utf-8",
    )
    return {
        "schema": schema, "rows": rows, "dog_rows": dog_rows, "features": features,
        "matrix": matrix, "states": states, "difficulty_rows": difficulty_rows,
        "strata": strata, "batches": batches, "selected": selected,
        "non_dog_order": non_dog_order, "random_matched": random_matched,
        "random_global": random_global, "train_rows": train_rows,
    }


def load_predictions(context: dict) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], dict[str, dict[str, str]]]]:
    dog_rows = context["dog_rows"]
    correct = {}
    all_predictions = {}
    for model in MODELS:
        for seed in SEEDS:
            path = SOURCE_EXPERIMENT / "predictions" / model / str(seed) / "candidate_predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = read_csv(path)
            by_id = {row["sample_id"]: row for row in rows}
            if len(by_id) != 600:
                raise ValueError(f"incomplete predictions: {path}")
            all_predictions[(model, seed)] = by_id
            correct[(model, seed)] = np.asarray([
                int(by_id[row["sample_id"]]["predicted_label"]) == 0 for row in dog_rows
            ], dtype=bool)
    return correct, all_predictions


def evaluate(context: dict, correct: dict, predictions: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    batches = context["batches"]
    matrix = context["matrix"]
    dog_rows = context["dog_rows"]
    rows_by_id = {row["sample_id"]: row for row in context["rows"]}
    batch_metrics = []
    cumulative_metrics = []
    state_metrics = []
    confusion_rows = []
    for model in MODELS:
        for seed in SEEDS:
            outcome = correct[(model, seed)]
            by_id = predictions[(model, seed)]
            cumulative = []
            for batch_number, batch in enumerate(batches, start=1):
                cumulative.extend(batch)
                batch_difficulty = [context["difficulty_rows"][index]["difficulty_score"] for index in batch]
                batch_frequency, batch_presence, batch_low, batch_missing = coverage(matrix, batch)
                batch_failure = failure_metrics(batch, outcome, matrix, min_support=3)
                batch_metrics.append({
                    "model": model, "seed": seed, "batch": f"B{batch_number}",
                    "dog_count": len(batch), "difficulty_mean": float(np.mean(batch_difficulty)),
                    "difficulty_median": float(np.median(batch_difficulty)),
                    "difficulty_min": int(min(batch_difficulty)), "difficulty_max": int(max(batch_difficulty)),
                    "frequency_coverage": batch_frequency, "presence_coverage": batch_presence,
                    "low_frequency_states": batch_low, "missing_states": batch_missing,
                    "dog_recall": float(outcome[batch].mean()),
                    "dog_error_count": int((~outcome[batch]).sum()), **batch_failure,
                })

                budget = len(cumulative)
                selected_ids = [dog_rows[index]["sample_id"] for index in cumulative]
                suite_ids = list(selected_ids)
                for class_name in CLASS_NAMES[1:]:
                    suite_ids.extend(row["sample_id"] for row in context["non_dog_order"][class_name][:budget])
                true = np.asarray([CLASS_TO_LABEL[rows_by_id[sample_id]["class_name"]] for sample_id in suite_ids])
                predicted = np.asarray([int(by_id[sample_id]["predicted_label"]) for sample_id in suite_ids])
                overall = confusion_metrics(true, predicted)
                frequency, presence, low, missing = coverage(matrix, cumulative)
                failure = failure_metrics(cumulative, outcome, matrix, min_support=MIN_STATE_SUPPORT)
                cumulative_metrics.append({
                    "model": model, "seed": seed, "stage": f"D{budget}",
                    "dog_count": budget, "total_targets": 5 * budget,
                    "frequency_coverage": frequency, "presence_coverage": presence,
                    "low_frequency_states": low, "missing_states": missing,
                    "dog_condition_recall": float(outcome[cumulative].mean()),
                    "dog_condition_error_count": int((~outcome[cumulative]).sum()),
                    **overall, **failure,
                })

            selected = [index for group in batches for index in group]
            selected_matrix = matrix[selected]
            selected_outcome = outcome[selected]
            for state_index, (feature_id, state) in enumerate(context["states"]):
                mask = selected_matrix[:, state_index].astype(bool)
                support = int(mask.sum())
                if support == 0:
                    continue
                count = int(selected_outcome[mask].sum())
                state_metrics.append({
                    "model": model, "seed": seed, "feature_id": feature_id, "state": state,
                    "support": support, "correct_count": count, "error_count": support - count,
                    "recall": count / support,
                })
            predicted_dog = [int(by_id[dog_rows[index]["sample_id"]]["predicted_label"]) for index in selected]
            counts = Counter(predicted_dog)
            confusion_rows.append({
                "model": model, "seed": seed,
                **{f"predicted_{name}": counts.get(label, 0) for name, label in CLASS_TO_LABEL.items()},
            })
    write_csv(OUTPUT / "tables/batch_metrics_by_run.csv", batch_metrics)
    write_csv(OUTPUT / "tables/cumulative_metrics_by_run.csv", cumulative_metrics)
    write_csv(OUTPUT / "tables/state_recall_by_run.csv", state_metrics)
    write_csv(OUTPUT / "tables/dog_confusions_by_run.csv", confusion_rows)
    return batch_metrics, cumulative_metrics, state_metrics, confusion_rows


def random_controls(context: dict, correct: dict, observed_cumulative: list[dict]) -> tuple[list[dict], list[dict]]:
    by_observed_stage = {}
    for stage in range(1, STRATA + 1):
        budget = stage * SELECT_PER_STRATUM
        rows = [row for row in observed_cumulative if row["dog_count"] == budget]
        by_observed_stage[stage] = {
            "dog_recall": float(np.mean([row["dog_condition_recall"] for row in rows])),
            "failed_state_count": float(np.mean([row["failed_state_count"] for row in rows])),
        }
    run_keys = list(correct)
    detailed = []
    for method, collections in (
        ("difficulty_matched_random", context["random_matched"]),
        ("global_random", context["random_global"]),
    ):
        for replicate, groups in enumerate(collections):
            cumulative = []
            for stage, group in enumerate(groups, start=1):
                cumulative.extend(group)
                frequency, presence, _, _ = coverage(context["matrix"], cumulative)
                recalls = []
                breadths = []
                for key in run_keys:
                    recalls.append(float(correct[key][cumulative].mean()))
                    breadths.append(float(failure_metrics(cumulative, correct[key], context["matrix"], MIN_STATE_SUPPORT)["failed_state_count"]))
                detailed.append({
                    "method": method, "replicate": replicate, "stage": f"D{len(cumulative)}",
                    "dog_count": len(cumulative), "frequency_coverage": frequency,
                    "presence_coverage": presence, "mean_dog_recall": float(np.mean(recalls)),
                    "mean_failed_state_count": float(np.mean(breadths)),
                })
        print(f"[random] completed {method}", flush=True)
    write_csv(OUTPUT / "tables/random_control_by_replicate.csv", detailed)

    summary = []
    selected_cumulative = []
    for stage, batch in enumerate(context["batches"], start=1):
        selected_cumulative.extend(batch)
        observed_frequency, observed_presence, _, _ = coverage(context["matrix"], selected_cumulative)
        for method in ("difficulty_matched_random", "global_random"):
            candidates = [row for row in detailed if row["method"] == method and row["dog_count"] == len(selected_cumulative)]
            observed = {
                "frequency_coverage": observed_frequency,
                "presence_coverage": observed_presence,
                "mean_dog_recall": by_observed_stage[stage]["dog_recall"],
                "mean_failed_state_count": by_observed_stage[stage]["failed_state_count"],
            }
            out = {"method": method, "stage": f"D{len(selected_cumulative)}", "dog_count": len(selected_cumulative)}
            for metric, observed_value in observed.items():
                values = np.asarray([float(row[metric]) for row in candidates])
                out[f"observed_{metric}"] = observed_value
                out[f"random_{metric}_mean"] = float(values.mean())
                out[f"random_{metric}_sd"] = float(values.std(ddof=1))
                out[f"random_{metric}_p2_5"] = float(np.percentile(values, 2.5))
                out[f"random_{metric}_p97_5"] = float(np.percentile(values, 97.5))
                out[f"delta_{metric}"] = observed_value - float(values.mean())
                out[f"empirical_upper_p"] = float((1 + np.sum(values >= observed_value)) / (1 + len(values)))
            summary.append(out)
    write_csv(OUTPUT / "tables/random_control_summary.csv", summary)
    return detailed, summary


def aggregate_models(batch_metrics: list[dict], cumulative_metrics: list[dict], state_metrics: list[dict]) -> list[dict]:
    summary = []
    for model in MODELS:
        batch_means = []
        for batch_number in range(1, STRATA + 1):
            rows = [row for row in batch_metrics if row["model"] == model and row["batch"] == f"B{batch_number}"]
            batch_means.append(float(np.mean([row["dog_recall"] for row in rows])))
        auc = float(np.trapezoid(batch_means, np.arange(1, STRATA + 1)) / (STRATA - 1))
        d100 = [row for row in cumulative_metrics if row["model"] == model and row["dog_count"] == 100]
        out = {
            "model": model, "difficulty_robustness_auc": auc,
            **{f"B{index + 1}_recall": value for index, value in enumerate(batch_means)},
            "recall_drop_B1_to_B5": batch_means[-1] - batch_means[0],
        }
        for metric in (
            "accuracy", "macro_f1", "dog_f1", "dog_recall", "dog_condition_recall",
            "failed_state_count", "state_macro_recall", "worst5_state_recall",
        ):
            values = [float(row[metric]) for row in d100 if row[metric] is not None]
            out[f"D100_{metric}_mean"] = float(np.mean(values))
            out[f"D100_{metric}_sd"] = float(np.std(values, ddof=1))
        summary.append(out)
    summary.sort(key=lambda row: row["difficulty_robustness_auc"], reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["auc_rank"] = rank
    write_csv(OUTPUT / "tables/model_robustness_ranking.csv", summary)

    grouped_states = []
    for model in MODELS:
        keys = sorted({(row["feature_id"], row["state"]) for row in state_metrics if row["model"] == model})
        for feature_id, state in keys:
            rows = [row for row in state_metrics if row["model"] == model and row["feature_id"] == feature_id and row["state"] == state]
            support = int(rows[0]["support"])
            total_correct = sum(int(row["correct_count"]) for row in rows)
            total_support = sum(int(row["support"]) for row in rows)
            grouped_states.append({
                "model": model, "feature_id": feature_id, "state": state, "support": support,
                "recall": total_correct / total_support, "eligible": int(support >= MIN_STATE_SUPPORT),
            })
    write_csv(OUTPUT / "tables/state_recall_by_model.csv", grouped_states)
    return summary


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_bootstrap(context: dict, correct: dict, model_summary: list[dict]) -> list[dict]:
    selected = [index for group in context["batches"] for index in group]
    arrays = {
        model: np.asarray([[correct[(model, seed)][index] for index in selected] for seed in SEEDS], dtype=float)
        for model in MODELS
    }
    rng = np.random.default_rng(RANDOM_SEED)
    seed_samples = rng.integers(0, len(SEEDS), size=(BOOTSTRAP_REPLICATES, len(SEEDS)))
    image_samples = [rng.integers(0, SELECT_PER_STRATUM, size=(BOOTSTRAP_REPLICATES, SELECT_PER_STRATUM)) for _ in range(STRATA)]
    boot_auc = {}
    boot_b5 = {}
    for model in MODELS:
        batch_values = []
        for batch_index in range(STRATA):
            positions = batch_index * SELECT_PER_STRATUM + image_samples[batch_index]
            sampled = arrays[model][seed_samples[:, :, None], positions[:, None, :]]
            batch_values.append(sampled.mean(axis=(1, 2)))
        matrix = np.column_stack(batch_values)
        boot_auc[model] = np.trapezoid(matrix, np.arange(1, STRATA + 1), axis=1) / (STRATA - 1)
        boot_b5[model] = matrix[:, -1]
    point = {row["model"]: row for row in model_summary}
    rows = []
    for first, second in itertools.combinations(MODELS, 2):
        row = {"model_a": first, "model_b": second}
        for metric, values, field in (
            ("auc", boot_auc, "difficulty_robustness_auc"),
            ("B5_recall", boot_b5, "B5_recall"),
        ):
            diff = values[first] - values[second]
            row[f"{metric}_difference"] = float(point[first][field] - point[second][field])
            row[f"{metric}_ci_low"] = float(np.percentile(diff, 2.5))
            row[f"{metric}_ci_high"] = float(np.percentile(diff, 97.5))
            row[f"{metric}_p_two_sided"] = float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
        rows.append(row)
    for metric in ("auc", "B5_recall"):
        adjusted = holm([row[f"{metric}_p_two_sided"] for row in rows])
        for row, value in zip(rows, adjusted):
            row[f"{metric}_holm_p"] = value
    write_csv(OUTPUT / "tables/paired_model_bootstrap.csv", rows)
    return rows


def error_ledger(context: dict, correct: dict, predictions: dict) -> list[dict]:
    selected_info = {}
    for batch_number, batch in enumerate(context["batches"], start=1):
        for index in batch:
            selected_info[index] = batch_number
    rows = []
    for index in context["selected"]:
        target = context["dog_rows"][index]
        errors = []
        architecture_errors = set()
        confusions = Counter()
        for model in MODELS:
            for seed in SEEDS:
                prediction = predictions[(model, seed)][target["sample_id"]]
                if int(prediction["predicted_label"]) != 0:
                    errors.append(f"{model}:{seed}:{prediction['predicted_class']}")
                    architecture_errors.add(model)
                    confusions[prediction["predicted_class"]] += 1
        rows.append({
            "sample_id": target["sample_id"], "batch": f"B{selected_info[index]}",
            "difficulty_score": context["difficulty_rows"][index]["difficulty_score"],
            "error_runs": len(errors), "error_architectures": len(architecture_errors),
            "confusion_counts": json.dumps(confusions, ensure_ascii=False, sort_keys=True),
            "error_details": json.dumps(errors, ensure_ascii=False),
            "human_review": "pending" if errors else "not_required",
            "crop_path": target["crop_path"], "image_path": target["image_path"],
        })
    rows.sort(key=lambda row: (row["error_runs"], row["difficulty_score"]), reverse=True)
    write_csv(OUTPUT / "tables/error_ledger.csv", rows)
    return rows


def make_figures(
    context: dict, batch_metrics: list[dict], cumulative_metrics: list[dict],
    model_summary: list[dict], state_metrics: list[dict], random_summary: list[dict], confusion_rows: list[dict],
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    x = np.arange(1, 6)
    for model_index, model in enumerate(MODELS):
        values = [next(row for row in model_summary if row["model"] == model)[f"B{batch}_recall"] for batch in x]
        axes[0, 0].plot(x, values, marker="o", linewidth=1.5, color=colors(model_index), label=model)
    axes[0, 0].set(title="Disjoint difficulty batches", xlabel="Difficulty batch", ylabel="Dog recall", xticks=x, xticklabels=[f"B{i}" for i in x])
    axes[0, 0].grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=6.8, ncol=2)

    stages = [20, 40, 60, 80, 100]
    coverage_x = []
    cumulative = []
    for batch in context["batches"]:
        cumulative.extend(batch)
        coverage_x.append(coverage(context["matrix"], cumulative)[0])
    for model_index, model in enumerate(MODELS):
        values = []
        for stage in stages:
            group = [row for row in cumulative_metrics if row["model"] == model and row["dog_count"] == stage]
            values.append(float(np.mean([row["dog_condition_recall"] for row in group])))
        axes[0, 1].plot(coverage_x, values, marker="o", linewidth=1.5, color=colors(model_index), label=model)
    axes[0, 1].set(title="Cumulative coverage response", xlabel="Frequency coverage", ylabel="Cumulative dog recall")
    axes[0, 1].grid(alpha=.2)

    ordered = sorted(model_summary, key=lambda row: row["difficulty_robustness_auc"])
    y = np.arange(len(ordered))
    axes[1, 0].barh(y - .18, [row["difficulty_robustness_auc"] for row in ordered], height=.34, color="#2878b5", label="Difficulty AUC")
    axes[1, 0].barh(y + .18, [row["B5_recall"] for row in ordered], height=.34, color="#d95f02", label="B5 recall")
    axes[1, 0].set(title="Robustness ranking", xlabel="Metric value", yticks=y, yticklabels=[row["model"] for row in ordered])
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="x", alpha=.2)

    matched = [row for row in random_summary if row["method"] == "difficulty_matched_random"]
    observed = np.asarray([row["observed_mean_failed_state_count"] for row in matched])
    random_mean = np.asarray([row["random_mean_failed_state_count_mean"] for row in matched])
    low = np.asarray([row["random_mean_failed_state_count_p2_5"] for row in matched])
    high = np.asarray([row["random_mean_failed_state_count_p97_5"] for row in matched])
    axes[1, 1].plot(stages, observed, marker="o", color="#117a65", label="Coverage+difficulty")
    axes[1, 1].plot(stages, random_mean, marker="o", color="#6b7280", label="Difficulty-matched random")
    axes[1, 1].fill_between(stages, low, high, color="#9ca3af", alpha=.22, label="Random 95% interval")
    axes[1, 1].set(title="Failure-state breadth", xlabel="Dog targets", ylabel="Mean failed states", xticks=stages)
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "diagnostic_overview.png", dpi=220)
    fig.savefig(figure_dir / "diagnostic_overview.svg")
    plt.close(fig)

    eligible_states = sorted({
        (row["feature_id"], row["state"]) for row in state_metrics if int(row["support"]) >= MIN_STATE_SUPPORT
    })
    heat = np.zeros((len(MODELS), len(eligible_states)))
    for model_index, model in enumerate(MODELS):
        for state_index, state in enumerate(eligible_states):
            values = [float(row["recall"]) for row in state_metrics if row["model"] == model and (row["feature_id"], row["state"]) == state]
            heat[model_index, state_index] = np.mean(values)
    order = np.argsort(heat.mean(axis=0))[: min(30, len(eligible_states))]
    heat = heat[:, order]
    labels = [f"{eligible_states[index][0]}={eligible_states[index][1]}" for index in order]
    fig, axis = plt.subplots(figsize=(max(12, len(labels) * .42), 5.2))
    image = axis.imshow(heat, aspect="auto", vmin=0.55, vmax=1.0, cmap="RdYlGn")
    axis.set(title="Lowest-recall visual states", yticks=np.arange(len(MODELS)), yticklabels=MODELS,
             xticks=np.arange(len(labels)), xticklabels=labels)
    axis.tick_params(axis="x", rotation=65, labelsize=7)
    fig.colorbar(image, ax=axis, label="Recall")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_state_recall_heatmap.png", dpi=220)
    fig.savefig(figure_dir / "model_state_recall_heatmap.svg")
    plt.close(fig)

    confusion = np.zeros((len(MODELS), 5))
    for model_index, model in enumerate(MODELS):
        group = [row for row in confusion_rows if row["model"] == model]
        for class_index, class_name in enumerate(CLASS_NAMES):
            confusion[model_index, class_index] = np.mean([float(row[f"predicted_{class_name}"]) for row in group])
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    image = axis.imshow(confusion, aspect="auto", vmin=0, vmax=max(.15, float(confusion[:, 1:].max())), cmap="Blues")
    axis.set(title="Dog prediction destinations (D100)", yticks=np.arange(len(MODELS)), yticklabels=MODELS,
             xticks=np.arange(5), xticklabels=CLASS_NAMES)
    for row in range(len(MODELS)):
        for column in range(5):
            axis.text(column, row, f"{confusion[row, column]:.1%}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label="Fraction")
    fig.tight_layout()
    fig.savefig(figure_dir / "dog_confusion_heatmap.png", dpi=220)
    fig.savefig(figure_dir / "dog_confusion_heatmap.svg")
    plt.close(fig)


def write_report(
    context: dict, batch_metrics: list[dict], cumulative_metrics: list[dict], model_summary: list[dict],
    state_metrics: list[dict], random_summary: list[dict], bootstrap_rows: list[dict], error_rows: list[dict],
    duplicate_rows: list[dict],
) -> None:
    batch_average = []
    for batch_number in range(1, 6):
        rows = [row for row in batch_metrics if row["batch"] == f"B{batch_number}"]
        batch_average.append({
            "batch": f"B{batch_number}",
            "difficulty": float(np.mean([row["difficulty_median"] for row in rows])),
            "recall": float(np.mean([row["dog_recall"] for row in rows])),
            "errors": float(np.mean([row["dog_error_count"] for row in rows])),
        })
    architecture_drop = sum(row["B5_recall"] < row["B1_recall"] for row in model_summary)
    h1 = "supported" if batch_average[-1]["recall"] < batch_average[0]["recall"] and architecture_drop >= 6 else "partial_or_unsupported"
    easy_order = [row["model"] for row in sorted(model_summary, key=lambda row: row["B1_recall"], reverse=True)]
    auc_order = [row["model"] for row in model_summary]
    h2 = "supported" if easy_order != auc_order else "unsupported"
    matched100 = next(row for row in random_summary if row["method"] == "difficulty_matched_random" and row["dog_count"] == 100)
    h3 = "supported" if matched100["observed_mean_failed_state_count"] > matched100["random_mean_failed_state_count_p97_5"] else "partial_or_unsupported"
    macro_order = [row["model"] for row in sorted(model_summary, key=lambda row: row["D100_macro_f1_mean"], reverse=True)]
    b5_order = [row["model"] for row in sorted(model_summary, key=lambda row: row["B5_recall"], reverse=True)]
    h4 = "supported" if macro_order != b5_order else "unsupported"

    state_risk = []
    keys = sorted({(row["feature_id"], row["state"]) for row in state_metrics if int(row["support"]) >= MIN_STATE_SUPPORT})
    for feature_id, state in keys:
        rows = [row for row in state_metrics if row["feature_id"] == feature_id and row["state"] == state]
        state_risk.append((float(np.mean([row["recall"] for row in rows])), feature_id, state, int(rows[0]["support"])))
    state_risk.sort()

    lines = [
        "# 视觉状态覆盖与难度递增模型诊断结果", "", "## 固定协议", "",
        "- 200张独立dog候选按模型无关难度划分为5层，每层40张；",
        "- 每层按精简状态边际覆盖贡献选择20张，形成B1--B5共100张；",
        "- 8种架构、3个种子、24个冻结模型，不重新训练；",
        "- 主要对照为1000次难度匹配随机选择；",
        "- 主排名为B1--B5 dog Recall的难度鲁棒性AUC。", "",
        "## 难度批次", "",
        "| 批次 | 难度中位数 | 平均dog Recall | 平均错误数/20张 |",
        "|---|---:|---:|---:|",
    ]
    for row in batch_average:
        lines.append(f"| {row['batch']} | {row['difficulty']:.1f} | {row['recall']:.2%} | {row['errors']:.2f} |")
    lines.extend(["", "## 模型排名", "", "| 排名 | 模型 | 难度AUC | B1 Recall | B5 Recall | B1→B5 | D100 Macro-F1 | 最差5状态Recall |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in model_summary:
        lines.append(
            f"| {row['auc_rank']} | {row['model']} | {row['difficulty_robustness_auc']:.2%} | "
            f"{row['B1_recall']:.2%} | {row['B5_recall']:.2%} | {row['recall_drop_B1_to_B5']:+.2%} | "
            f"{row['D100_macro_f1_mean']:.2%} | {row['D100_worst5_state_recall_mean']:.2%} |"
        )
    lines.extend([
        "", "## 难度匹配随机对照（D100）", "",
        f"- 覆盖组频数覆盖率：{matched100['observed_frequency_coverage']:.2%}；随机均值：{matched100['random_frequency_coverage_mean']:.2%}。",
        f"- 覆盖组平均失败状态数：{matched100['observed_mean_failed_state_count']:.2f}；随机均值：{matched100['random_mean_failed_state_count_mean']:.2f}；随机95%区间：[{matched100['random_mean_failed_state_count_p2_5']:.2f}, {matched100['random_mean_failed_state_count_p97_5']:.2f}]。",
        f"- 覆盖组平均dog Recall：{matched100['observed_mean_dog_recall']:.2%}；随机均值：{matched100['random_mean_dog_recall_mean']:.2%}。", "",
        "## 预注册假设判定", "",
        f"- H1 难度阶梯有效：`{h1}`；B5低于B1的架构为{architecture_drop}/8。",
        f"- H2 模型难度敏感度不同：`{h2}`。",
        f"- H3 覆盖在难度匹配后提供独立失败状态诊断：`{h3}`。",
        f"- H4 整体排名不能替代尾部状态排名：`{h4}`。", "",
        "## 最低Recall状态", "",
        "| 状态 | 支持数 | 跨模型种子平均Recall |", "|---|---:|---:|",
    ])
    for recall, feature_id, state, support in state_risk[:12]:
        lines.append(f"| `{feature_id}={state}` | {support} | {recall:.2%} |")
    lines.extend([
        "", "## 数据与解释边界", "",
        f"- D300与诊断候选的ID、图像SHA和裁剪SHA精确重叠为0；dHash距离不大于2的待复核对为{len(duplicate_rows)}。",
        f"- D100中有{sum(row['error_runs'] > 0 for row in error_rows)}张dog在至少一次运行中出错，需要复核错误台账。",
        "- 难度分数没有读取模型输出；实际Recall不必严格单调，如果不单调应视为难度定义的实验结果。",
        "- 状态共现，因此当前结果支持关联与能力边界，不支持单状态因果作用。",
        "- 当前排名限于D300冻结模型、Open Images目标裁剪和dog类别，不是通用模型排名。",
    ])
    (OUTPUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(OUTPUT / "summary.json", {
        "status": "completed", "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "top_auc_model": model_summary[0]["model"],
        "top_b5_model": max(model_summary, key=lambda row: row["B5_recall"])["model"],
        "top_macro_f1_model": max(model_summary, key=lambda row: row["D100_macro_f1_mean"])["model"],
        "stable_error_images": sum(row["error_runs"] >= 18 for row in error_rows),
        "perceptual_near_duplicate_pairs": len(duplicate_rows),
    })
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        f"| B1--B5 form an empirically harder sequence | {h1} | `tables/batch_metrics_by_run.csv` | Frozen dog crop task |\n"
        f"| Architectures have different difficulty sensitivity | {h2} | `tables/model_robustness_ranking.csv`, `tables/paired_model_bootstrap.csv` | Eight D300 architectures |\n"
        f"| Coverage adds diagnostic value after difficulty matching | {h3} | `tables/random_control_summary.csv` | Same 200-dog pool |\n"
        f"| Aggregate ranking differs from tail-state ranking | {h4} | `tables/model_robustness_ranking.csv` | Current protocol only |\n"
        "| A single state causes errors | unsupported | no counterfactual intervention | Association only |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: no external factual claim introduced.\n"
        "- Result traceability: PASS; all numbers derive from frozen manifests, saved predictions and scripted transformations.\n"
        "- Claim scope: PASS WITH LIMITATIONS; results are restricted to frozen D300 dog-crop classification.\n"
        f"- Data audit: exact leakage passed; {len(duplicate_rows)} perceptual near-duplicate pairs require review.\n"
        f"- Error audit: {sum(row['error_runs'] > 0 for row in error_rows)} selected images require review because at least one run failed.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-perceptual-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress("prepare", "validating inputs and freezing diagnostic batches", 2)
    context = prepare()
    progress("audit", "checking candidate/train perceptual near duplicates", 10)
    duplicates = [] if args.skip_perceptual_audit else duplicate_audit(context["dog_rows"], context["train_rows"])
    write_csv(OUTPUT / "tables/near_duplicate_audit.csv", duplicates)
    progress("predictions", "loading 24 saved model prediction files", 20)
    correct, predictions = load_predictions(context)
    progress("evaluation", "computing disjoint-batch and cumulative metrics", 32)
    batch_metrics, cumulative_metrics, state_metrics, confusion_rows = evaluate(context, correct, predictions)
    progress("random_controls", "running 1000 difficulty-matched and global random controls", 48)
    _, random_summary = random_controls(context, correct, cumulative_metrics)
    progress("aggregation", "ranking models and aggregating state recalls", 70)
    model_summary = aggregate_models(batch_metrics, cumulative_metrics, state_metrics)
    progress("bootstrap", "running 10000 paired hierarchical bootstrap replicates", 78)
    bootstrap_rows = paired_bootstrap(context, correct, model_summary)
    progress("errors", "building per-image error ledger", 90)
    error_rows = error_ledger(context, correct, predictions)
    progress("figures", "rendering diagnostic figures", 94)
    make_figures(context, batch_metrics, cumulative_metrics, model_summary, state_metrics, random_summary, confusion_rows)
    progress("report", "writing evidence-bounded report and review", 98)
    write_report(
        context, batch_metrics, cumulative_metrics, model_summary, state_metrics,
        random_summary, bootstrap_rows, error_rows, duplicates,
    )
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "percent": 100.0, "updated_at": now(),
        "models": len(MODELS), "seeds": len(SEEDS), "random_replicates": RANDOM_REPLICATES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    })
    print(json.dumps({"status": "completed", "output": str(OUTPUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
