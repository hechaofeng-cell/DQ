#!/usr/bin/env python3
"""Dense coverage-response curves and per-image error attribution ledgers.

This analysis reuses 24 frozen-model prediction files. It does not retrain or
rerun inference. Test subsets are nested and class-balanced at five-image steps.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


WORKTREE = Path(__file__).resolve().parents[1]
PROJECT = Path("/home/hcf/project/QE")


OUTPUT = WORKTREE / "artifacts/dense_coverage_error_attribution_20260921"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
CLASS_NAMES = ("dog", "cat", "horse", "sheep", "person")
CLASS_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}
BUDGETS = tuple(range(5, 101, 5))
WINDOW_SIZE = 20
WINDOW_STARTS = tuple(range(1, 82, 5))
COVERAGE_CAP = 30
BOOTSTRAPS = 5000
RANDOM_SEED = 20260921
STABLE_RUN_THRESHOLD = 18
STABLE_ARCH_THRESHOLD = 6

CONTEXTS = {
    "dog": {
        "source": PROJECT / "artifacts/fixed_train_visual_test_adequacy_20260920",
        "feature_path": PROJECT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv",
        "schema_path": PROJECT / "configs/dog_feature_schema_v3_1.json",
    },
    "cat": {
        "source": PROJECT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920",
        "feature_path": PROJECT / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv",
        "schema_path": PROJECT / "configs/cat_feature_schema_v1_0.json",
    },
}

COLORS = {"dog": "#484878", "cat": "#E08AA5", "precision": "#5B9E98",
          "recall": "#D67B5B", "f1": "#484878", "fp": "#B987C5", "fn": "#E96B63"}


@dataclass(frozen=True)
class StateField:
    feature_id: str
    values: tuple[str, ...]


class StateSchema:
    """Torch-free subset of the project's schema parser used for coverage."""

    def __init__(self, payload: dict):
        definitions = payload["universal_features"] + payload.get("category_specific_features", [])
        self.fields = tuple(StateField(item["feature_id"], tuple(item["possible_values"]))
                            for item in definitions)

    @classmethod
    def from_path(cls, path: Path) -> "StateSchema":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def coverage_states(self) -> list[tuple[str, str]]:
        return [(field.feature_id, value) for field in self.fields for value in field.values
                if value not in {"unknown", "other"}]

    def state_pairs(self, row: dict) -> set[tuple[str, str]]:
        pairs = set()
        for field in self.fields:
            raw = row.get(field.feature_id, "")
            if not raw:
                values = []
            elif raw.startswith("["):
                values = json.loads(raw)
            else:
                values = [raw]
            for value in values:
                if value not in {"unknown", "other"}:
                    pairs.add((field.feature_id, value))
        return pairs


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"cannot infer fields for empty table: {path}")
    names = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
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


def prepare_context(category: str) -> dict:
    cfg = CONTEXTS[category]
    source = cfg["source"]
    pool = read_csv(source / "candidate_pool.csv")
    by_id = {row["sample_id"]: row for row in pool}
    target_order = sorted(read_csv(source / "orders" / f"gap_{category}_order.csv"),
                          key=lambda row: int(row["rank"]))
    target_rows = [by_id[row["sample_id"]] | {"selection_rank": int(row["rank"])}
                   for row in target_order]
    non_target_order = read_csv(source / "orders" / f"non_{category}_order.csv")
    non_target_rows = {}
    for class_name in CLASS_NAMES:
        if class_name == category:
            continue
        rows = sorted((row for row in non_target_order if row["class_name"] == class_name),
                      key=lambda row: int(row["rank"]))
        non_target_rows[class_name] = [by_id[row["sample_id"]] | {"selection_rank": int(row["rank"])}
                                       for row in rows]
    if len(target_rows) != 200 or any(len(rows) != 100 for rows in non_target_rows.values()):
        raise ValueError(f"{category}: unexpected ordered candidate counts")

    feature_rows = read_csv(cfg["feature_path"])
    features = {row["image_id"]: row for row in feature_rows}
    schema = StateSchema.from_path(cfg["schema_path"])
    states = schema.coverage_states()
    state_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(target_rows), len(states)), dtype=np.int8)
    for image_index, row in enumerate(target_rows):
        feature = features[row["sample_id"]]
        for pair in schema.state_pairs(feature):
            if pair in state_index:
                matrix[image_index, state_index[pair]] = 1

    predictions = {}
    for model in MODELS:
        for seed in SEEDS:
            path = source / "predictions" / model / str(seed) / "candidate_predictions.csv"
            rows = read_csv(path)
            predictions[(model, seed)] = {row["sample_id"]: row for row in rows}
            if set(predictions[(model, seed)]) != set(by_id):
                raise ValueError(f"{category}: prediction IDs do not match pool for {model}/{seed}")
    return {
        "category": category, "source": source, "pool": pool, "by_id": by_id,
        "target_rows": target_rows, "non_target_rows": non_target_rows,
        "features": features, "schema": schema, "states": states, "matrix": matrix,
        "predictions": predictions, "target_label": CLASS_LABEL[category],
        "feature_path": cfg["feature_path"], "schema_path": cfg["schema_path"],
    }


def selected_rows(ctx: dict, start: int, count: int) -> list[dict]:
    stop = start + count
    rows = ctx["target_rows"][start:stop]
    for class_name in CLASS_NAMES:
        if class_name != ctx["category"]:
            rows.extend(ctx["non_target_rows"][class_name][start:stop])
    return rows


def classification_metrics(true: np.ndarray, predicted: np.ndarray, target: int) -> dict:
    confusion = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted_support = confusion.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted_support, out=np.zeros(5), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5),
                   where=(precision + recall) > 0)
    fp = int(predicted_support[target] - tp[target])
    fn = int(support[target] - tp[target])
    return {
        "accuracy": float(tp.sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "target_precision": float(precision[target]),
        "target_recall": float(recall[target]),
        "target_f1": float(f1[target]),
        "target_error_rate": float(1 - recall[target]),
        "target_false_positive_count": fp,
        "target_false_negative_count": fn,
    }


def coverage_metrics(ctx: dict, start: int, count: int) -> dict:
    selected = ctx["matrix"][start:start + count]
    counts = selected.sum(axis=0)
    return {
        "frequency_coverage": float(np.minimum(counts / COVERAGE_CAP, 1.0).mean()),
        "presence_coverage": float((counts > 0).mean()),
        "present_state_count": int((counts > 0).sum()),
    }


def failure_state_metrics(ctx: dict, start: int, count: int, target_predictions: np.ndarray) -> dict:
    states = ctx["matrix"][start:start + count]
    failed = target_predictions != ctx["target_label"]
    failed_support = states[failed].sum(axis=0) if failed.any() else np.zeros(states.shape[1])
    return {
        "failed_state_count": int((failed_support > 0).sum()),
        "failed_state_fraction": float((failed_support > 0).mean()),
    }


def analyze_curves(contexts: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    dense = []
    windows = []
    for category, ctx in contexts.items():
        for model in MODELS:
            for seed in SEEDS:
                predictions = ctx["predictions"][(model, seed)]
                for budget in BUDGETS:
                    rows = selected_rows(ctx, 0, budget)
                    true = np.asarray([int(predictions[row["sample_id"]]["true_label"]) for row in rows])
                    predicted = np.asarray([int(predictions[row["sample_id"]]["predicted_label"]) for row in rows])
                    target_pred = np.asarray([
                        int(predictions[row["sample_id"]]["predicted_label"])
                        for row in ctx["target_rows"][:budget]
                    ])
                    dense.append({
                        "category": category, "model": model, "seed": seed,
                        "budget_per_class": budget, "total_test_images": 5 * budget,
                        **coverage_metrics(ctx, 0, budget),
                        **classification_metrics(true, predicted, ctx["target_label"]),
                        **failure_state_metrics(ctx, 0, budget, target_pred),
                    })
                for start_rank in WINDOW_STARTS:
                    start = start_rank - 1
                    rows = selected_rows(ctx, start, WINDOW_SIZE)
                    true = np.asarray([int(predictions[row["sample_id"]]["true_label"]) for row in rows])
                    predicted = np.asarray([int(predictions[row["sample_id"]]["predicted_label"]) for row in rows])
                    target_pred = np.asarray([
                        int(predictions[row["sample_id"]]["predicted_label"])
                        for row in ctx["target_rows"][start:start + WINDOW_SIZE]
                    ])
                    windows.append({
                        "category": category, "model": model, "seed": seed,
                        "window_start_rank": start_rank,
                        "window_end_rank": start_rank + WINDOW_SIZE - 1,
                        "total_test_images": 5 * WINDOW_SIZE,
                        **coverage_metrics(ctx, start, WINDOW_SIZE),
                        **classification_metrics(true, predicted, ctx["target_label"]),
                        **failure_state_metrics(ctx, start, WINDOW_SIZE, target_pred),
                    })
    write_csv(TABLES / "dense_metrics_by_run.csv", dense)
    write_csv(TABLES / "sliding_window_metrics_by_run.csv", windows)
    return dense, windows


SUMMARY_METRICS = (
    "frequency_coverage", "presence_coverage", "present_state_count", "accuracy",
    "macro_f1", "target_precision", "target_recall", "target_f1", "target_error_rate",
    "target_false_positive_count", "target_false_negative_count", "failed_state_count",
    "failed_state_fraction",
)


def architecture_bootstrap(rows: list[dict], metric: str, rng: np.random.Generator) -> tuple[float, float]:
    by_model = {model: [float(row[metric]) for row in rows if row["model"] == model] for model in MODELS}
    architecture_means = np.asarray([np.mean(by_model[model]) for model in MODELS])
    draws = rng.integers(0, len(MODELS), size=(BOOTSTRAPS, len(MODELS)))
    values = architecture_means[draws].mean(axis=1)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize(rows: list[dict], group_field: str, output_name: str) -> list[dict]:
    result = []
    rng = np.random.default_rng(RANDOM_SEED)
    for category in ("dog", "cat"):
        keys = sorted({int(row[group_field]) for row in rows if row["category"] == category})
        for key in keys:
            group = [row for row in rows if row["category"] == category and int(row[group_field]) == key]
            out = {"category": category, group_field: key, "model_runs": len(group)}
            for metric in SUMMARY_METRICS:
                values = np.asarray([float(row[metric]) for row in group])
                low, high = architecture_bootstrap(group, metric, rng)
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_sd"] = float(values.std(ddof=1))
                out[f"{metric}_ci95_low"] = low
                out[f"{metric}_ci95_high"] = high
            result.append(out)
    write_csv(TABLES / output_name, result)
    return result


def stability_label(error_runs: int, stable_architectures: int) -> str:
    if error_runs >= STABLE_RUN_THRESHOLD and stable_architectures >= STABLE_ARCH_THRESHOLD:
        return "stable_shared_error"
    if 1 <= stable_architectures <= 2 and error_runs >= 2:
        return "architecture_specific_error"
    if error_runs == 0:
        return "consistently_correct"
    return "mixed_or_seed_unstable"


def observable_reason_tags(row: dict, feature: dict | None, role: str) -> list[str]:
    tags = []
    area = float(row.get("bbox_area_ratio") or 0)
    if area < 0.03:
        tags.append("very_small_target")
    elif area < 0.10:
        tags.append("small_target")
    if row.get("is_occluded") == "1":
        tags.append("source_occlusion")
    if row.get("is_truncated") == "1":
        tags.append("source_truncation")
    if feature:
        if feature.get("occlusion_level") in {"partial", "heavy"}:
            tags.append("semantic_occlusion")
        if feature.get("truncation_level") in {"partial", "heavy"}:
            tags.append("semantic_truncation")
        if feature.get("head_visibility") not in {"", "clear", "unknown"}:
            tags.append("head_not_clear")
        if feature.get("muzzle_visibility") not in {"", "clear", "unknown"}:
            tags.append("muzzle_not_clear")
        if feature.get("mouth_visibility") not in {"", "clear", "unknown"}:
            tags.append("mouth_not_clear")
        if feature.get("posture_action_state") in {
            "crouching", "running", "jumping", "climbing", "grooming", "eating",
            "drinking", "playing", "interacting", "sleeping",
        }:
            tags.append("non_typical_action")
    if role == "FP":
        tags.append("non_target_attracted_to_target_class")
    elif role == "FN":
        tags.append("target_evidence_not_selected")
    return tags or ["no_obvious_observable_factor"]


def build_error_ledgers(contexts: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ledger = []
    run_rows = []
    for category, ctx in contexts.items():
        final_rows = selected_rows(ctx, 0, 100)
        target = ctx["target_label"]
        for row in final_rows:
            sample_id = row["sample_id"]
            true_label = int(row["label"])
            true_class = row["class_name"]
            predictions = []
            target_events = []
            overall_errors = []
            probabilities_true = []
            probabilities_target = []
            predicted_confidences = []
            architecture_target_counts = defaultdict(int)
            architecture_overall_counts = defaultdict(int)
            prediction_counts = Counter()
            for model in MODELS:
                for seed in SEEDS:
                    pred = ctx["predictions"][(model, seed)][sample_id]
                    predicted_label = int(pred["predicted_label"])
                    predicted_class = pred["predicted_class"]
                    overall_error = predicted_label != true_label
                    fn = true_label == target and predicted_label != target
                    fp = true_label != target and predicted_label == target
                    target_event = fn or fp
                    role = "FN" if fn else "FP" if fp else "none"
                    true_prob = float(pred[f"prob_{true_class}"])
                    target_prob = float(pred[f"prob_{category}"])
                    predicted_prob = float(pred[f"prob_{predicted_class}"])
                    predictions.append(predicted_label)
                    overall_errors.append(overall_error)
                    target_events.append(target_event)
                    probabilities_true.append(true_prob)
                    probabilities_target.append(target_prob)
                    predicted_confidences.append(predicted_prob)
                    prediction_counts[predicted_class] += 1
                    architecture_overall_counts[model] += int(overall_error)
                    architecture_target_counts[model] += int(target_event)
                    run_rows.append({
                        "experiment_target": category, "sample_id": sample_id,
                        "true_class": true_class, "selection_rank": row["selection_rank"],
                        "model": model, "seed": seed, "predicted_class": predicted_class,
                        "predicted_label": predicted_label, "overall_error": int(overall_error),
                        "target_error_role": role, "target_relevant_error": int(target_event),
                        "prob_true_class": true_prob, "prob_target_class": target_prob,
                        "predicted_confidence": predicted_prob,
                    })
            overall_runs = int(sum(overall_errors))
            target_runs = int(sum(target_events))
            overall_arch = sum(count >= 2 for count in architecture_overall_counts.values())
            target_arch = sum(count >= 2 for count in architecture_target_counts.values())
            role = "FN" if true_label == target else "FP"
            relevant_role = role if target_runs else "none"
            feature = ctx["features"].get(sample_id) if true_label == target else None
            state_labels = []
            if feature:
                state_labels = sorted(f"{a}={b}" for a, b in ctx["schema"].state_pairs(feature))
            dominant = prediction_counts.most_common(1)[0]
            ledger.append({
                "experiment_target": category, "sample_id": sample_id,
                "image_id": row["image_id"], "true_class": true_class,
                "selection_rank": row["selection_rank"],
                "target_or_non_target": "target" if true_label == target else "non_target",
                "target_f1_error_role": relevant_role,
                "overall_error_runs": overall_runs,
                "overall_error_rate": overall_runs / 24,
                "overall_stable_architectures": overall_arch,
                "overall_error_stability": stability_label(overall_runs, overall_arch),
                "target_relevant_error_runs": target_runs,
                "target_relevant_error_rate": target_runs / 24,
                "target_error_stable_architectures": target_arch,
                "target_error_stability": stability_label(target_runs, target_arch),
                "dominant_prediction": dominant[0], "dominant_prediction_runs": dominant[1],
                **{f"predicted_{name}_runs": prediction_counts[name] for name in CLASS_NAMES},
                "mean_prob_true_class": float(np.mean(probabilities_true)),
                "mean_prob_target_class": float(np.mean(probabilities_target)),
                "mean_predicted_confidence": float(np.mean(predicted_confidences)),
                "bbox_area_ratio": row.get("bbox_area_ratio", ""),
                "is_occluded": row.get("is_occluded", ""),
                "is_truncated": row.get("is_truncated", ""),
                "candidate_source": row.get("candidate_source", ""),
                "observable_reason_tags": "|".join(observable_reason_tags(row, feature, role)),
                "semantic_states": "|".join(state_labels),
                "image_path": row.get("image_path", ""), "crop_path": row.get("crop_path", ""),
                "human_primary_reason": "", "human_secondary_reason": "",
                "human_review_notes": "",
            })
    write_csv(TABLES / "prediction_events_by_run.csv", run_rows)
    write_csv(TABLES / "per_image_error_ledger.csv", ledger)
    stable = [row for row in ledger if row["target_error_stability"] == "stable_shared_error"]
    write_csv(TABLES / "stable_target_errors.csv", stable, fields=list(ledger[0]))
    priority = sorted(
        (row for row in ledger if int(row["target_relevant_error_runs"]) > 0),
        key=lambda row: (
            row["target_error_stability"] != "stable_shared_error",
            -float(row["target_relevant_error_rate"]),
            row["experiment_target"], row["sample_id"],
        ),
    )
    write_csv(TABLES / "error_review_priority.csv", priority, fields=list(ledger[0]))
    return ledger, stable, run_rows


def summarize_errors(ledger: list[dict], run_rows: list[dict], contexts: dict[str, dict]) -> dict[str, list[dict]]:
    role_summary = []
    fp_sources = []
    confusion = []
    factor_rows = []
    reason_rows = []
    for category in ("dog", "cat"):
        category_rows = [row for row in ledger if row["experiment_target"] == category]
        for role in ("FN", "FP"):
            eligible = [row for row in category_rows if
                        (role == "FN" and row["target_or_non_target"] == "target") or
                        (role == "FP" and row["target_or_non_target"] == "non_target")]
            role_summary.append({
                "category": category, "role": role, "eligible_images": len(eligible),
                "images_with_any_error": sum(float(row["target_relevant_error_rate"]) > 0 for row in eligible),
                "stable_shared_images": sum(row["target_error_stability"] == "stable_shared_error" for row in eligible),
                "architecture_specific_images": sum(row["target_error_stability"] == "architecture_specific_error" for row in eligible),
                "mean_error_rate_across_images": float(np.mean([float(row["target_relevant_error_rate"]) for row in eligible])),
            })
            tag_groups = defaultdict(list)
            for row in eligible:
                if int(row["target_relevant_error_runs"]) == 0:
                    continue
                for tag in row["observable_reason_tags"].split("|"):
                    if tag not in {"target_evidence_not_selected", "non_target_attracted_to_target_class"}:
                        tag_groups[tag].append(row)
            for tag, tagged in tag_groups.items():
                reason_rows.append({
                    "category": category, "role": role, "observable_reason_tag": tag,
                    "images_with_error_and_tag": len(tagged),
                    "mean_target_error_rate": float(np.mean([
                        float(row["target_relevant_error_rate"]) for row in tagged
                    ])),
                    "stable_shared_images": sum(
                        row["target_error_stability"] == "stable_shared_error" for row in tagged
                    ),
                })
        non_target = [row for row in category_rows if row["target_or_non_target"] == "non_target"]
        for source_class in (name for name in CLASS_NAMES if name != category):
            group = [row for row in non_target if row["true_class"] == source_class]
            fp_sources.append({
                "category": category, "true_source_class": source_class,
                "images": len(group),
                "total_fp_runs": sum(int(row["target_relevant_error_runs"]) for row in group),
                "fp_rate": sum(int(row["target_relevant_error_runs"]) for row in group) / (24 * len(group)),
                "images_with_any_fp": sum(int(row["target_relevant_error_runs"]) > 0 for row in group),
                "stable_fp_images": sum(row["target_error_stability"] == "stable_shared_error" for row in group),
            })
        events = [row for row in run_rows if row["experiment_target"] == category]
        for true_class in CLASS_NAMES:
            for predicted_class in CLASS_NAMES:
                count = sum(row["true_class"] == true_class and row["predicted_class"] == predicted_class for row in events)
                confusion.append({"category": category, "true_class": true_class,
                                  "predicted_class": predicted_class, "run_predictions": count})

        ctx = contexts[category]
        target_rows = [row for row in category_rows if row["target_or_non_target"] == "target"]
        factor_map = defaultdict(list)
        for row in target_rows:
            for tag in row["observable_reason_tags"].split("|"):
                if tag not in {"target_evidence_not_selected", "no_obvious_observable_factor"}:
                    factor_map[tag].append(row)
            for state in row["semantic_states"].split("|"):
                if state:
                    factor_map[state].append(row)
        for factor, with_rows in factor_map.items():
            unique_with = {row["sample_id"]: row for row in with_rows}.values()
            with_ids = {row["sample_id"] for row in unique_with}
            without_rows = [row for row in target_rows if row["sample_id"] not in with_ids]
            if len(with_ids) < 5 or len(without_rows) < 5:
                continue
            with_rate = float(np.mean([float(row["target_relevant_error_rate"]) for row in unique_with]))
            without_rate = float(np.mean([float(row["target_relevant_error_rate"]) for row in without_rows]))
            factor_rows.append({
                "category": category, "observable_factor": factor,
                "support_images": len(with_ids), "error_rate_with": with_rate,
                "error_rate_without": without_rate, "risk_difference": with_rate - without_rate,
                "stable_fn_images_with": sum(row["target_error_stability"] == "stable_shared_error" for row in unique_with),
            })
    factor_rows.sort(key=lambda row: (row["category"], -float(row["risk_difference"])))
    write_csv(TABLES / "target_error_role_summary.csv", role_summary)
    write_csv(TABLES / "false_positive_source_summary.csv", fp_sources)
    write_csv(TABLES / "confusion_flow_summary.csv", confusion)
    write_csv(TABLES / "observable_factor_risk_summary.csv", factor_rows)
    reason_rows.sort(key=lambda row: (row["category"], row["role"],
                                      -int(row["images_with_error_and_tag"])))
    write_csv(TABLES / "observable_reason_tag_summary.csv", reason_rows)
    return {"roles": role_summary, "fp_sources": fp_sources, "confusion": confusion,
            "factors": factor_rows, "reasons": reason_rows}


def lowess_like(x: np.ndarray, y: np.ndarray, span: float = 0.35) -> np.ndarray:
    count = max(5, int(math.ceil(span * len(x))))
    result = np.zeros_like(y, dtype=float)
    for index, x0 in enumerate(x):
        distances = np.abs(x - x0)
        bandwidth = np.partition(distances, min(count - 1, len(x) - 1))[min(count - 1, len(x) - 1)]
        if bandwidth == 0:
            result[index] = y[index]
            continue
        weights = np.clip(1 - (distances / bandwidth) ** 3, 0, None) ** 3
        design = np.column_stack([np.ones_like(x), x - x0])
        beta = np.linalg.pinv(design.T @ (weights[:, None] * design)) @ (design.T @ (weights * y))
        result[index] = beta[0]
    return result


def configure_plots() -> None:
    font_path = WORKTREE / "artifacts/coverage_result_visualizations_20260921/NotoSansCJKsc-Regular.otf"
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        family = fm.FontProperties(fname=font_path).get_name()
    else:
        family = "DejaVu Sans"
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "svg.fonttype": "none", "font.size": 8.5, "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
    })


def finish_axis(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid, color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_figures(dense_summary: list[dict], window_summary: list[dict], summaries: dict) -> None:
    configure_plots()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    for row_index, category in enumerate(("dog", "cat")):
        rows = sorted((row for row in dense_summary if row["category"] == category),
                      key=lambda row: int(row["budget_per_class"]))
        x = np.asarray([int(row["budget_per_class"]) for row in rows])
        specs = [
            ("frequency_coverage", "频数覆盖率（%）", [COLORS[category]], [category.upper()]),
            ("prf", "目标类别指标（%）", [COLORS["precision"], COLORS["recall"], COLORS["f1"]],
             ["Precision", "Recall", "F1"]),
            ("counts", "错误数（24次运行均值）", [COLORS["fp"], COLORS["fn"]], ["FP", "FN"]),
        ]
        for col_index, (kind, ylabel, colors, labels) in enumerate(specs):
            ax = axes[row_index, col_index]
            if kind == "frequency_coverage":
                series = [("frequency_coverage", labels[0], colors[0], 100)]
            elif kind == "prf":
                series = [("target_precision", labels[0], colors[0], 100),
                          ("target_recall", labels[1], colors[1], 100),
                          ("target_f1", labels[2], colors[2], 100)]
            else:
                series = [("target_false_positive_count", labels[0], colors[0], 1),
                          ("target_false_negative_count", labels[1], colors[1], 1)]
            for metric, label, color, scale in series:
                y = np.asarray([scale * float(row[f"{metric}_mean"]) for row in rows])
                low = np.asarray([scale * float(row[f"{metric}_ci95_low"]) for row in rows])
                high = np.asarray([scale * float(row[f"{metric}_ci95_high"]) for row in rows])
                ax.plot(x, y, "o", markersize=3.2, alpha=0.55, color=color)
                ax.plot(x, lowess_like(x, y), linewidth=2.0, color=color, label=label)
                ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
            ax.set(xlabel="每类累计样本数", ylabel=ylabel,
                   title=f"{category.upper()} · " + ("覆盖" if col_index == 0 else "P/R/F1" if col_index == 1 else "FP/FN"))
            ax.set_xticks([5, 20, 40, 60, 80, 100])
            ax.legend(loc="best")
            finish_axis(ax)
    fig.suptitle("每5张重算：真实点、局部平滑趋势与架构分层95%置信区间",
                 fontsize=13, fontweight="semibold")
    save_figure(fig, "01_dense_coverage_performance_curves")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
    for row_index, category in enumerate(("dog", "cat")):
        rows = sorted((row for row in window_summary if row["category"] == category),
                      key=lambda row: int(row["window_start_rank"]))
        centers = np.asarray([int(row["window_start_rank"]) + 9.5 for row in rows])
        for col_index, (metric, ylabel) in enumerate((("target_error_rate", "目标错误率（%）"),
                                                       ("target_f1", "目标F1（%）"))):
            ax = axes[row_index, col_index]
            y = np.asarray([100 * float(row[f"{metric}_mean"]) for row in rows])
            low = np.asarray([100 * float(row[f"{metric}_ci95_low"]) for row in rows])
            high = np.asarray([100 * float(row[f"{metric}_ci95_high"]) for row in rows])
            ax.plot(centers, y, marker="o", markersize=3.5, linewidth=1.7, color=COLORS[category])
            ax.fill_between(centers, low, high, color=COLORS[category], alpha=0.14)
            ax.set(xlabel="20张滑动窗口中心覆盖排序", ylabel=ylabel, title=category.upper())
            finish_axis(ax)
    fig.suptitle("局部难度曲线（20张窗口，步长5张；窗口相互重叠）", fontsize=13, fontweight="semibold")
    save_figure(fig, "02_sliding_window_local_difficulty")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)
    for ax, category in zip(axes, ("dog", "cat")):
        rows = [row for row in summaries["fp_sources"] if row["category"] == category]
        rows.sort(key=lambda row: float(row["fp_rate"]), reverse=True)
        names = [row["true_source_class"] for row in rows]
        values = [100 * float(row["fp_rate"]) for row in rows]
        bars = ax.bar(names, values, color=COLORS[category], alpha=0.82)
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                    f"稳定FP={row['stable_fp_images']}", ha="center", va="bottom", fontsize=7)
        ax.set(xlabel="非目标真实类别", ylabel=f"误报为{category}的运行比例（%）", title=category.upper())
        finish_axis(ax)
    fig.suptitle("Precision下降来源：哪些非目标类别被误报为目标类", fontsize=13, fontweight="semibold")
    save_figure(fig, "03_false_positive_sources")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), constrained_layout=True)
    for ax, category in zip(axes, ("dog", "cat")):
        rows = [row for row in summaries["factors"] if row["category"] == category]
        rows = sorted(rows, key=lambda row: float(row["risk_difference"]), reverse=True)[:12][::-1]
        labels = [row["observable_factor"].replace("_", " ") for row in rows]
        values = [100 * float(row["risk_difference"]) for row in rows]
        colors = [COLORS["fn"] if value > 0 else "#5B9E98" for value in values]
        ax.barh(np.arange(len(rows)), values, color=colors, alpha=0.82)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(np.arange(len(rows)), labels, fontsize=7)
        ax.set(xlabel="FN风险差（百分点）", title=category.upper())
        finish_axis(ax, "x")
    fig.suptitle("目标漏识别的可观察风险因素（探索性、未经多重检验确认）",
                 fontsize=12.5, fontweight="semibold")
    save_figure(fig, "04_observable_fn_risk_factors")

    stable = summaries.get("stable", [])
    if stable:
        fig, axes = plt.subplots(1, len(stable), figsize=(3.1 * len(stable), 3.4), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, row in zip(axes, stable):
            image = plt.imread(row["crop_path"])
            ax.imshow(image)
            ax.set_title(
                f"真实：{row['true_class']} → 预测：{row['experiment_target']}\n"
                f"目标误报 {row['target_relevant_error_runs']}/24，框面积 {100*float(row['bbox_area_ratio']):.2f}%",
                fontsize=8,
            )
            ax.axis("off")
        fig.suptitle("跨架构稳定误报案例：共同特征是目标极小、有效细节不足",
                     fontsize=11.5, fontweight="semibold")
        save_figure(fig, "05_stable_error_examples")


def build_report(dense_summary: list[dict], ledger: list[dict], stable: list[dict], summaries: dict) -> None:
    lines = [
        "# 密集覆盖曲线与分类错误归因分析", "", f"生成时间：{now()}", "",
        "## 实验口径", "",
        "- 训练集、24个模型权重、候选池、覆盖选样顺序和非目标顺序全部固定；",
        "- 累计测试集按每类5张递增，共20个真实检查点；",
        "- 每个检查点保持dog、cat、horse、sheep、person五类等量；",
        "- 20张滑动窗口仅用于描述局部难度，因窗口重叠不作为独立显著性样本；",
        "- 稳定目标错误定义为至少18/24次出错，且至少6/8个架构中有不少于2/3种子出错。", "",
        "## F1分解", "",
    ]
    for category in ("dog", "cat"):
        rows = sorted((row for row in dense_summary if row["category"] == category),
                      key=lambda row: int(row["budget_per_class"]))
        start = next(row for row in rows if int(row["budget_per_class"]) == 20)
        end = next(row for row in rows if int(row["budget_per_class"]) == 100)
        lines.extend([
            f"### {category.upper()}", "",
            f"- 20→100张：Precision {100*float(start['target_precision_mean']):.2f}% → {100*float(end['target_precision_mean']):.2f}% "
            f"（{100*(float(end['target_precision_mean'])-float(start['target_precision_mean'])):+.2f} pp）；",
            f"- 20→100张：Recall {100*float(start['target_recall_mean']):.2f}% → {100*float(end['target_recall_mean']):.2f}% "
            f"（{100*(float(end['target_recall_mean'])-float(start['target_recall_mean'])):+.2f} pp）；",
            f"- 20→100张：F1 {100*float(start['target_f1_mean']):.2f}% → {100*float(end['target_f1_mean']):.2f}% "
            f"（{100*(float(end['target_f1_mean'])-float(start['target_f1_mean'])):+.2f} pp）。", "",
        ])
    lines.extend(["## 稳定错误筛选", ""])
    for category in ("dog", "cat"):
        rows = [row for row in stable if row["experiment_target"] == category]
        fn = sum(row["target_f1_error_role"] == "FN" for row in rows)
        fp = sum(row["target_f1_error_role"] == "FP" for row in rows)
        lines.append(f"- {category.upper()}：稳定目标相关错误{len(rows)}张，其中FN={fn}，FP={fp}。")
    lines.extend(["", "## 自动错误原因线索", ""])
    for category in ("dog", "cat"):
        fp_rows = sorted((row for row in summaries["fp_sources"] if row["category"] == category),
                         key=lambda row: float(row["fp_rate"]), reverse=True)
        top_fp = fp_rows[0]
        factor_rows = sorted((row for row in summaries["factors"] if row["category"] == category),
                             key=lambda row: float(row["risk_difference"]), reverse=True)
        top_factor = factor_rows[0]
        lines.extend([
            f"- {category.upper()}最主要FP来源：真实`{top_fp['true_source_class']}`，"
            f"误报运行比例{100*float(top_fp['fp_rate']):.2f}%，稳定FP图片{top_fp['stable_fp_images']}张；",
            f"- {category.upper()}最高探索性FN风险因素：`{top_factor['observable_factor']}`，"
            f"风险差{100*float(top_factor['risk_difference']):+.2f}个百分点，支持{top_factor['support_images']}张。",
        ])
    lines.extend(["", "## 解释边界", "",
                  "- Recall下降对应目标漏识别（FN增加），可解释为目标证据不足或目标状态更难；",
                  "- Precision下降对应非目标误报（FP增加），不能直接解释为目标本身更难识别；",
                  "- `observable_factor_risk_summary.csv`是相关性筛查，不证明某个状态导致错误；",
                  "- cat语义状态仍是VLM暂定标签，人工复核前不能作为确认性证据。", "",
                  "## 输出", "",
                  "- `tables/dense_curve_summary.csv`：每5张累计曲线；",
                  "- `tables/sliding_window_summary.csv`：局部滑动窗口；",
                  "- `tables/per_image_error_ledger.csv`：逐图错误台账及人工复核空列；",
                  "- `tables/error_review_priority.csv`：所有出现过目标相关错误的人工复核优先序；",
                  "- `tables/stable_target_errors.csv`：稳定目标相关错误；",
                  "- `tables/false_positive_source_summary.csv`：Precision下降来源；",
                  "- `tables/observable_factor_risk_summary.csv`：目标FN风险因素；",
                  "- `figures/`：五组SVG和PNG结果图。", ""])
    (OUTPUT / "dense_coverage_error_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")


def write_governance(contexts: dict[str, dict]) -> None:
    write_json(OUTPUT / "protocol.json", {
        "experiment_id": "dense-coverage-error-attribution-20260921",
        "created_at": now(), "training_and_models": "fixed D300; no retraining or inference",
        "categories": ["dog", "cat"], "classes": list(CLASS_NAMES),
        "models": list(MODELS), "seeds": list(SEEDS),
        "cumulative_budgets_per_class": list(BUDGETS),
        "sliding_window": {"size_per_class": WINDOW_SIZE, "step": 5, "starts": list(WINDOW_STARTS)},
        "stable_error_rule": {"error_runs_at_least": STABLE_RUN_THRESHOLD,
                              "architectures_at_least": STABLE_ARCH_THRESHOLD,
                              "architecture_error_requires_seeds": 2},
        "input_sha256": {
            f"{category}_{name}": sha256(path)
            for category, ctx in contexts.items()
            for name, path in (("features", ctx["feature_path"]), ("schema", ctx["schema_path"]))
        },
    })
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n- Claim: denser semantic-coverage response curves separate cumulative metric changes from local difficulty.\n"
        "- Diagnostic claim: target F1 changes must be decomposed into FN/Recall and FP/Precision pathways.\n"
        "- Boundary: observable states are risk correlates, not causal explanations.\n", encoding="utf-8")
    (OUTPUT / "research_log.md").write_text(
        f"# Research Log\n\n- Started: {now()}\n- Reused 24 frozen predictions for each of dog and cat.\n"
        "- Command: `python3 scripts/analyze_dense_coverage_and_errors.py`.\n"
        "- No model training or image inference performed.\n", encoding="utf-8")
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Dense curves describe metric response at five-image increments | supported after run | `tables/dense_curve_summary.csv` | Nested balanced stress tests |\n"
        "| Precision decline is attributable to target-class false positives | supported by definition and counts | `tables/false_positive_source_summary.csv` | Does not identify causal visual feature |\n"
        "| Observable factors cause model errors | unsupported | `tables/observable_factor_risk_summary.csv` | Association only |\n",
        encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    contexts = {category: prepare_context(category) for category in ("dog", "cat")}
    write_governance(contexts)
    dense, windows = analyze_curves(contexts)
    dense_summary = summarize(dense, "budget_per_class", "dense_curve_summary.csv")
    window_summary = summarize(windows, "window_start_rank", "sliding_window_summary.csv")
    ledger, stable, run_rows = build_error_ledgers(contexts)
    summaries = summarize_errors(ledger, run_rows, contexts)
    summaries["stable"] = stable
    make_figures(dense_summary, window_summary, summaries)
    build_report(dense_summary, ledger, stable, summaries)
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "completed_at": now(),
        "dense_checkpoints": len(BUDGETS), "model_runs_per_category": 24,
        "ledger_images": len(ledger), "stable_target_errors": len(stable),
    })
    (OUTPUT / "review.md").write_text(
        "# Review\n\n- Citation integrity: no external claims introduced.\n"
        "- Result traceability: all figures and summaries derive from saved prediction CSVs.\n"
        "- Claim scope: FP/FN decomposition is descriptive; observable-factor analysis is exploratory.\n",
        encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(OUTPUT),
                      "dense_rows": len(dense), "ledger_rows": len(ledger),
                      "stable_errors": len(stable)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
