#!/usr/bin/env python3
"""Aggregate CUA-VSL runs and compute dog_v3.1 state-stratified recall."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from openimages_cua_vsl import StateSchema, feature_rows_by_id, read_csv, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/openimages_v7_cua_vsl_v1.json"
METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "dog_precision", "dog_recall", "dog_f1", "difficulty_dog_recall",
    "difficulty_under3pct_dog_recall", "difficulty_small_dog_recall",
    "difficulty_tiny_dog_recall", "difficulty_occluded_dog_recall",
    "difficulty_truncated_dog_recall", "main_rare_state_macro_recall",
    "difficulty_rare_state_macro_recall",
)


def state_recall_rows(
    predictions: list[dict], features: dict[str, dict], schema: StateSchema,
    train_counts: dict[tuple[str, str], int], cap: int, min_support: int,
) -> tuple[list[dict], float | None]:
    dog_predictions = {row["sample_id"]: row for row in predictions if row["class_name"] == "dog"}
    if set(dog_predictions) != set(features):
        missing = set(dog_predictions) ^ set(features)
        raise ValueError(f"prediction/state IDs differ: {sorted(missing)[:3]}")
    rows = []
    eligible_recalls = []
    for feature_id, state in schema.coverage_states():
        members = [
            sample_id for sample_id, feature_row in features.items()
            if (feature_id, state) in schema.state_pairs(feature_row)
        ]
        support = len(members)
        recall = (
            sum(dog_predictions[sample_id]["predicted_class"] == "dog" for sample_id in members) / support
            if support else None
        )
        rare = train_counts.get((feature_id, state), 0) < cap
        eligible = rare and support >= min_support
        if eligible and recall is not None:
            eligible_recalls.append(recall)
        rows.append({
            "feature_id": feature_id,
            "state": state,
            "train_count": train_counts.get((feature_id, state), 0),
            "test_support": support,
            "dog_recall": recall,
            "rare_by_train_cap": int(rare),
            "included_in_rare_macro": int(eligible),
        })
    macro = float(np.mean(eligible_recalls)) if eligible_recalls else None
    return rows, macro


def load_test_features(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    all_features = feature_rows_by_id(path)
    main = {sample_id: row for sample_id, row in all_features.items() if row["split"] == "main_test"}
    difficulty = {
        sample_id: row for sample_id, row in all_features.items()
        if row["split"] == "difficulty_test"
    }
    if len(main) != 100 or len(difficulty) != 100:
        raise ValueError(f"expected 100+100 test dog states, got {len(main)}+{len(difficulty)}")
    return main, difficulty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-group", choices=("smoke_runs", "runs"), default="smoke_runs")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output"])
    run_root = output / args.run_group
    metric_paths = sorted(run_root.glob("*/*/*/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"no completed metrics under {run_root}")
    schema = StateSchema.from_path(Path(config["schema"]))
    priority_rows = read_csv(output / "state_priority/state_priority.csv")
    train_counts = {
        (row["feature_id"], row["state"]): int(row["train_count"])
        for row in priority_rows
    }
    state_feature_path = output / "state_test_features_auto/results.csv"
    main_features = difficulty_features = None
    if state_feature_path.is_file():
        main_features, difficulty_features = load_test_features(state_feature_path)

    run_rows = []
    state_rows = []
    for metric_path in metric_paths:
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        run_dir = metric_path.parent
        if main_features is not None:
            main_predictions = read_csv(run_dir / "test_predictions.csv")
            difficulty_predictions = read_csv(run_dir / "difficulty_test_predictions.csv")
            main_state_rows, main_macro = state_recall_rows(
                main_predictions, main_features, schema, train_counts,
                int(config["coverage_cap"]), int(config["state_test_min_support"]),
            )
            difficulty_state_rows, difficulty_macro = state_recall_rows(
                difficulty_predictions, difficulty_features, schema, train_counts,
                int(config["coverage_cap"]), int(config["state_test_min_support"]),
            )
            metrics["main_rare_state_macro_recall"] = main_macro
            metrics["difficulty_rare_state_macro_recall"] = difficulty_macro
            for split_name, rows in (("main", main_state_rows), ("difficulty", difficulty_state_rows)):
                state_rows.extend({
                    "model": metrics["model"], "condition": metrics["condition"],
                    "seed": metrics["seed"], "test_split": split_name, **row,
                } for row in rows)
        run_rows.append(metrics)

    summary = []
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["condition"])].append(row)
    for (model, condition), rows in sorted(grouped.items()):
        summary_row = {"model": model, "condition": condition, "runs": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            summary_row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary_row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        summary.append(summary_row)

    table_dir = output / "tables"
    suffix = "smoke" if args.run_group == "smoke_runs" else "formal"
    write_csv(table_dir / f"{suffix}_results_by_seed.csv", run_rows)
    write_csv(table_dir / f"{suffix}_results_by_condition.csv", summary)
    if state_rows:
        write_csv(table_dir / f"{suffix}_state_recall_by_run.csv", state_rows)
    report = {
        "run_group": args.run_group,
        "scientific_status": "pipeline_smoke_only" if args.run_group == "smoke_runs" else "formal_pending_audit",
        "completed_runs": len(run_rows),
        "conditions": sorted({row["condition"] for row in run_rows}),
        "models": sorted({row["model"] for row in run_rows}),
        "automatic_state_labels": state_feature_path.is_file(),
        "human_state_review": (output / "state_test_features_auto/human_review.csv").is_file(),
        "summary": summary,
    }
    write_json(output / f"{suffix}_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
