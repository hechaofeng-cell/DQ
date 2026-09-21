#!/usr/bin/env python3
"""COCO 300 seed plus gap-driven Open Images V7 supplements.

The first phase deliberately omits a random control. Every supplement batch
adds one Open Images target per class so the five-class ratio stays 1:1:1:1:1.
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

COCO_ROOT = ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916"
OI_ROOT = ROOT / "data/openimages_v7_dog_gap_v1"
OI_FEATURES = ROOT / "artifacts/openimages_v7_dog_candidate_features_v1/results.csv"
COCO_FEATURES = ROOT / "artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv"
COCO_RAW_DOGS = ROOT / "artifacts/resnet18_dog500_vs_dog596_20260916/data/train_dog500.csv"
COCO_RAW_FEATURES = ROOT / "artifacts/dog500_features_v3_1_20260913/results.csv"
SCHEMA = ROOT / "configs/dog_feature_schema_v3_1.json"
OUTPUT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
ADDITIONS = (0, 50, 100, 150, 200, 250, 300, 350)
SEEDS = (20260916, 20260917, 20260918)
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
CLASSES = ("cat", "horse", "sheep", "person")
TOY_DOG_IDS = {"000000140444", "000000246880", "000000446990", "000000471513"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def vals(row: dict[str, str], key: str) -> list[str]:
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


def state_setup():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    keys = [item["feature_id"] for item in schema["universal_features"]]
    states = {(item["feature_id"], state)
              for item in schema["universal_features"]
              for state in item.get("possible_values", [])
              if state not in {"unknown", "other"}}
    return keys, states


def coverage(rows: list[dict[str, str]], keys: list[str], states: set[tuple[str, str]]) -> float:
    counts = Counter((key, state) for row in rows for key in keys for state in vals(row, key))
    return sum(min(counts[state] / 30.0, 1.0) for state in states) / len(states)


def select_external_dogs(coco_rows, oi_manifest, oi_features, additions, keys, states):
    feature_by_id = {row["image_id"]: row for row in oi_features}
    selected = []
    remaining = list(oi_manifest)
    counts = Counter((key, state) for row in coco_rows for key in keys for state in vals(row, key))

    def gain(manifest_row):
        feature = feature_by_id.get(manifest_row["sample_id"])
        if feature is None:
            return -1.0
        return sum(
            min((counts[(key, state)] + 1) / 30.0, 1.0) - min(counts[(key, state)] / 30.0, 1.0)
            for key in keys for state in vals(feature, key) if (key, state) in states
        )

    output = {0: []}
    for step in range(1, max(additions) + 1):
        best = max(remaining, key=lambda row: (gain(row), row["sample_id"]))
        remaining.remove(best)
        selected.append(best)
        feature = feature_by_id[best["sample_id"]]
        for key in keys:
            for state in vals(feature, key):
                counts[(key, state)] += 1
        if step in additions:
            output[step] = list(selected)
    return output, feature_by_id


def build_manifests():
    keys, states = state_setup()
    raw_dogs = read(COCO_RAW_DOGS)
    raw_by_id = {row["image_id"]: row for row in raw_dogs}
    ordinary = [row for row in raw_dogs if row["image_id"] not in TOY_DOG_IDS]
    coco_dogs = ordinary[:296] + [raw_by_id[image_id] for image_id in sorted(TOY_DOG_IDS)]
    for row in coco_dogs:
        row.update({"label": "0", "class_name": "dog", "yolo_class_id": "16"})
    coco_others = read(COCO_ROOT / "data/train_other_fixed.csv")
    coco_by_class = {name: [row for row in coco_others if row["class_name"] == name][:300]
                     for name in CLASSES}
    oi_dog_manifest = read(OI_ROOT / "manifests/dog_candidates.csv")
    oi_dog_features = read(OI_FEATURES)
    oi_base = read(OI_ROOT / "manifests/train_base.csv")
    oi_by_class = {name: [row for row in oi_base if row["class_name"] == name]
                   for name in CLASSES}
    if any(len(rows) < max(ADDITIONS) for rows in oi_by_class.values()):
        raise ValueError("Open Images non-dog candidate pool is too small")
    coco_feature_by_id = {row["image_id"]: row for row in read(COCO_RAW_FEATURES)}
    coco_feature_rows = [coco_feature_by_id[row["image_id"]] for row in coco_dogs]
    selected, oi_feature_by_id = select_external_dogs(
        coco_feature_rows, oi_dog_manifest, oi_dog_features, ADDITIONS, keys, states
    )
    manifest_dir = OUTPUT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for add in ADDITIONS:
        dogs = coco_dogs + [oi_feature_by_id[row["sample_id"]] for row in selected[add]]
        # Replace feature rows with training rows for the external dog targets.
        oi_train_by_id = {row["sample_id"]: row for row in oi_dog_manifest}
        dogs = coco_dogs + [oi_train_by_id[row["sample_id"]] for row in selected[add]]
        stage_rows = list(dogs)
        for name in CLASSES:
            stage_rows.extend(coco_by_class[name])
            stage_rows.extend(oi_by_class[name][:add])
        stage = f"oi_gap_add{add}"
        write(manifest_dir / f"train_{stage}.csv", stage_rows)
        feature_rows = coco_feature_rows + [oi_feature_by_id[row["sample_id"]] for row in selected[add]]
        records.append({"stage": stage, "external_dog_added": add,
                        "dog_count": 300 + add, "each_non_dog_count": 300 + add,
                        "total_count": 5 * (300 + add),
                        "coverage": coverage(feature_rows, keys, states)})
    write(OUTPUT / "coverage_gradient.csv", records)
    write(OUTPUT / "selected_external_dog_order.csv", selected[max(ADDITIONS)])
    return {row["stage"]: row["coverage"] for row in records}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coverages = build_manifests()
    eval_rows = experiment.read_csv(COCO_ROOT / "data/eval_fixed.csv")
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    experiment.OUTPUT = OUTPUT
    experiment.MODEL_NAMES = MODELS
    experiment.SEEDS = SEEDS
    experiment.write_json(OUTPUT / "protocol.json", {
        "experiment_type": "COCO 300 plus gap-driven Open Images V7 external supplements",
        "external_source": "Open Images V7 local frozen train candidate pool",
        "stages": list(ADDITIONS), "models": list(MODELS), "seeds": list(SEEDS),
        "class_ratio": "dog:cat:horse:sheep:person = 1:1:1:1:1",
        "coverage": coverages, "random_control": "deferred to later phase",
        "fixed_validation_test": str((COCO_ROOT / "data/eval_fixed.csv").resolve()),
        "samples_per_epoch": 3000, "epochs": 20, "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
    })
    completed = []
    for add in ADDITIONS:
        stage = f"oi_gap_add{add}"
        train_rows = experiment.read_csv(OUTPUT / "manifests" / f"train_{stage}.csv")
        for model in MODELS:
            for seed in SEEDS:
                print(f"[stage={stage} model={model} seed={seed}] starting", flush=True)
                metrics = experiment.train_one(model, stage, seed, train_rows, val_rows, test_rows)
                metrics.update({"coverage_stage": stage, "coverage": coverages[stage]})
                completed.append(metrics)
                experiment.write_json(OUTPUT / "metrics_latest.json", {
                    "completed_runs": len(completed),
                    "total_runs": len(ADDITIONS) * len(MODELS) * len(SEEDS),
                    "latest": metrics,
                })
    experiment.write_csv(OUTPUT / "results_by_run.csv", completed)
    experiment.write_json(OUTPUT / "all_metrics.json", completed)
    print(json.dumps({"output": str(OUTPUT.resolve()), "completed_runs": len(completed)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
