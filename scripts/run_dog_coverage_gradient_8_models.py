#!/usr/bin/env python3
"""Run the gap-driven dog coverage gradient on the fixed COCO five-class task.

This phase intentionally has no random supplement control. It estimates the
descriptive response of eight classifiers to a nested coverage-driven sequence.
The random, size-matched control is reserved for a later phase.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_four_model_gap_vs_random as experiment


DATA_ROOT = ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916"
DOG_MANIFEST = DATA_ROOT / "data/train_dog596.csv"
OTHER_MANIFEST = DATA_ROOT / "data/train_other_fixed.csv"
FEATURES = ROOT / "artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv"
SCHEMA = ROOT / "configs/dog_feature_schema_v3_1.json"
OUTPUT = ROOT / "artifacts/dog_coverage_gradient_8_models_20260918"
STAGES = (300, 350, 400, 450, 500, 550, 596)
SEEDS = (20260916, 20260917, 20260918)
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def state_values(row: dict[str, str], key: str) -> list[str]:
    value = row.get(key, "")
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return [value]


def load_state_keys() -> list[str]:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return [item["feature_id"] for item in schema["universal_features"]
            if item.get("feature_id") not in {"unknown", "other"}]


def coverage(rows: list[dict[str, str]], keys: list[str]) -> float:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    definitions = {item["feature_id"]: item for item in schema["universal_features"]}
    scores = []
    for key in keys:
        states = [state for state in definitions[key].get("possible_values", [])
                  if state not in {"unknown", "other"}]
        counts = Counter(value for row in rows for value in state_values(row, key))
        if states:
            scores.extend(min(counts[state] / 30.0, 1.0) for state in states)
    return sum(scores) / len(scores) if scores else 0.0


def gap_order(dog_rows: list[dict[str, str]], feature_rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, str]]:
    features = {row["image_id"]: row for row in feature_rows}
    selected = list(dog_rows[:300])
    remaining = dog_rows[300:]
    # The first 300 are a frozen starting set. Thereafter each image is chosen
    # greedily by its marginal capped-state coverage gain, with image_id as a
    # deterministic tie-breaker.
    state_counts = Counter()
    for row in selected:
        f = features[row["image_id"]]
        for key in keys:
            for state in state_values(f, key):
                state_counts[(key, state)] += 1
    ordered = list(selected)
    while remaining:
        best = None
        best_key = None
        for row in remaining:
            f = features[row["image_id"]]
            gain = 0.0
            for key in keys:
                for state in state_values(f, key):
                    if state in {"unknown", "other"}:
                        continue
                    before = min(state_counts[(key, state)] / 30.0, 1.0)
                    after = min((state_counts[(key, state)] + 1) / 30.0, 1.0)
                    gain += after - before
            tie = row["image_id"]
            rank = (gain, tie)
            if best is None or rank > best_key:
                best, best_key = row, rank
        ordered.append(best)
        remaining.remove(best)
        f = features[best["image_id"]]
        for key in keys:
            for state in state_values(f, key):
                state_counts[(key, state)] += 1
    return ordered


def build_stage_manifests() -> dict[str, float]:
    dog_rows = read_csv(DOG_MANIFEST)
    feature_rows = read_csv(FEATURES)
    keys = load_state_keys()
    ordered = gap_order(dog_rows, feature_rows, keys)
    feature_by_id = {row["image_id"]: row for row in feature_rows}
    other_rows = read_csv(OTHER_MANIFEST)
    by_class = {}
    for class_name in ("cat", "horse", "sheep", "person"):
        rows = [row for row in other_rows if row["class_name"] == class_name]
        rows.sort(key=lambda row: row["image_id"])
        if len(rows) < max(STAGES):
            raise ValueError(f"not enough fixed {class_name} rows: {len(rows)}")
        by_class[class_name] = rows

    manifest_dir = OUTPUT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for count in STAGES:
        dogs = ordered[:count]
        stage_rows = dogs[:]
        for class_name in ("cat", "horse", "sheep", "person"):
            stage_rows.extend(by_class[class_name][:count])
        stage = f"gap{count}"
        write_csv(manifest_dir / f"train_{stage}.csv", stage_rows)
        cov_rows = [feature_by_id[row["image_id"]] for row in dogs]
        records.append({"stage": stage, "dog_count": count, "non_dog_count": count * 4,
                        "total_count": count * 5, "coverage": coverage(cov_rows, keys),
                        "dog_form_note": "toy representations remain dog label; audit metadata retained when available"})
    write_csv(OUTPUT / "coverage_gradient.csv", records)
    (OUTPUT / "selection_protocol.md").write_text(
        "# Dog Coverage Gradient Protocol\n\n"
        "This is the first, gap-only phase. No random supplement control is included. "
        "The nested dog sequence starts at 300 and is extended by marginal capped-state coverage gain. "
        "Every stage contains equal counts of dog, cat, horse, sheep, and person; each stage uses the "
        "same fixed validation/test manifest and fixed 3000 samples per training epoch. Random controls "
        "will be added in a later, size-matched phase.\n\n"
        "The reported coverage values are attained values, not forced target values.\n",
        encoding="utf-8",
    )
    return {record["stage"]: record["coverage"] for record in records}


def run_training() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coverages = build_stage_manifests()
    eval_rows = experiment.read_csv(DATA_ROOT / "data/eval_fixed.csv")
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    experiment.OUTPUT = OUTPUT
    experiment.MODEL_NAMES = MODELS
    experiment.SEEDS = SEEDS
    profiles = [experiment.model_profile(model_name) for model_name in MODELS]
    experiment.write_json(OUTPUT / "protocol.json", {
        "experiment_type": "gap-driven dog coverage gradient",
        "stages": list(STAGES), "models": list(MODELS), "seeds": list(SEEDS),
        "fixed_ratio": "dog:non-dog = 1:4; each non-dog class equals dog count",
        "epochs": 20, "samples_per_epoch": 3000, "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "selection_metric": "validation macro-F1", "input_size": "224x224",
        "coverage": coverages, "random_control": "deferred to later phase",
        "data_root": str(DATA_ROOT.resolve()),
        "eval_manifest": str((DATA_ROOT / "data/eval_fixed.csv").resolve()),
    })
    all_metrics = []
    for count in STAGES:
        stage = f"gap{count}"
        train_rows = experiment.read_csv(OUTPUT / "manifests" / f"train_{stage}.csv")
        for model_name in MODELS:
            for seed in SEEDS:
                print(f"[stage={stage} model={model_name} seed={seed}] starting", flush=True)
                metrics = experiment.train_one(model_name, stage, seed, train_rows, val_rows, test_rows)
                metrics["coverage_stage"] = stage
                metrics["coverage"] = coverages[stage]
                all_metrics.append(metrics)
                experiment.write_json(OUTPUT / "metrics_latest.json", {"completed_runs": len(all_metrics),
                    "total_runs": len(STAGES) * len(MODELS) * len(SEEDS), "latest": metrics})
    experiment.write_json(OUTPUT / "all_metrics.json", all_metrics)
    experiment.write_csv(OUTPUT / "results_by_run.csv", all_metrics)
    print(json.dumps({"output": str(OUTPUT.resolve()), "completed_runs": len(all_metrics)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run_training()
