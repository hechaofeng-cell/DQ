#!/usr/bin/env python3
"""Run matched random-selection controls for the COCO/Open Images gradient.

The gap-driven experiment is the frozen reference. This script changes only the
selection order of external dog targets. All COCO targets, external non-dog
targets, evaluation targets, training hyperparameters, and training seeds are
kept identical to the reference experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_four_model_gap_vs_random as experiment
import run_coco_openimages_external_gradient_8_models as gap_experiment


REFERENCE = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
OUTPUT = ROOT / "artifacts/coco_openimages_random_gradient_8_models_20260918"
OI_ROOT = ROOT / "data/openimages_v7_dog_gap_v1"
OI_FEATURES = ROOT / "artifacts/openimages_v7_dog_candidate_features_v1/results.csv"
COCO_ROOT = ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916"
ADDITIONS = (50, 100, 150, 200, 250, 300, 350)
SELECTION_SEEDS = (2026091801, 2026091802, 2026091803, 2026091804, 2026091805)
TRAINING_SEEDS = (20260916, 20260917, 20260918)
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)


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


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sample_key(row: dict[str, str]) -> str:
    return row.get("sample_id") or row.get("target_instance_id") or row["image_id"]


def build_random_orders(selection_seeds: tuple[int, ...]) -> dict[int, list[dict[str, str]]]:
    candidates = read(OI_ROOT / "manifests/dog_candidates.csv")
    if len(candidates) < max(ADDITIONS):
        raise ValueError(f"dog candidate pool has only {len(candidates)} rows")
    keys = [sample_key(row) for row in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("dog candidate pool contains duplicate target identifiers")
    orders = {}
    for seed in selection_seeds:
        rows = sorted(candidates, key=sample_key)
        random.Random(seed).shuffle(rows)
        orders[seed] = rows
        write(OUTPUT / "selections" / f"random_dog_order_seed{seed}.csv", rows[:max(ADDITIONS)])
    return orders


def compute_coverage(
    base_dog_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
) -> float:
    keys, states = gap_experiment.state_setup()
    coco_features = {row["image_id"]: row for row in read(gap_experiment.COCO_RAW_FEATURES)}
    oi_features = {row["image_id"]: row for row in read(OI_FEATURES)}
    feature_rows = [coco_features[row["image_id"]] for row in base_dog_rows]
    feature_rows.extend(oi_features[sample_key(row)] for row in selected_rows)
    return gap_experiment.coverage(feature_rows, keys, states)


def audit_manifest(
    random_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    expected_per_class: int,
) -> None:
    counts = Counter(row["class_name"] for row in random_rows)
    expected = {name: expected_per_class for name in ("dog", "cat", "horse", "sheep", "person")}
    if counts != expected:
        raise ValueError(f"class-count mismatch: got {dict(counts)}, expected {expected}")
    random_non_dogs = [sample_key(row) for row in random_rows if row["class_name"] != "dog"]
    reference_non_dogs = [sample_key(row) for row in reference_rows if row["class_name"] != "dog"]
    if random_non_dogs != reference_non_dogs:
        raise ValueError("random and gap manifests do not have identical ordered non-dog targets")
    missing = [row.get("crop_path", "") for row in random_rows if not Path(row.get("crop_path", "")).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} crop files are missing; first: {missing[0]}")


def build_manifests(selection_seeds: tuple[int, ...]) -> tuple[dict[str, float], list[dict]]:
    orders = build_random_orders(selection_seeds)
    base_rows = read(REFERENCE / "manifests/train_oi_gap_add0.csv")
    base_dogs = [row for row in base_rows if row["class_name"] == "dog"]
    if len(base_dogs) != 300:
        raise ValueError(f"expected 300 base dogs, found {len(base_dogs)}")
    coverage_by_condition: dict[str, float] = {}
    audit_rows = []
    for selection_seed, order in orders.items():
        previous_ids: set[str] = set()
        for add in ADDITIONS:
            condition = f"random_s{selection_seed}_add{add}"
            reference_rows = read(REFERENCE / "manifests" / f"train_oi_gap_add{add}.csv")
            non_dogs = [row for row in reference_rows if row["class_name"] != "dog"]
            selected = order[:add]
            selected_ids = {sample_key(row) for row in selected}
            if not previous_ids.issubset(selected_ids):
                raise ValueError(f"random sequence is not nested at {condition}")
            previous_ids = selected_ids
            train_rows = base_dogs + selected + non_dogs
            audit_manifest(train_rows, reference_rows, 300 + add)
            manifest_path = OUTPUT / "manifests" / f"train_{condition}.csv"
            write(manifest_path, train_rows)
            value = compute_coverage(base_dogs, selected)
            coverage_by_condition[condition] = value
            audit_rows.append({
                "condition": condition,
                "selection_seed": selection_seed,
                "external_dog_added": add,
                "dog_count": 300 + add,
                "each_non_dog_count": 300 + add,
                "total_count": 5 * (300 + add),
                "coverage": value,
                "manifest": str(manifest_path.resolve()),
                "non_dog_manifest_matches_gap": True,
                "nested_random_dog_selection": True,
            })
    write(OUTPUT / "random_coverage_gradient.csv", audit_rows)
    return coverage_by_condition, audit_rows


def write_protocol(selection_seeds: tuple[int, ...], coverage_by_condition: dict[str, float]) -> None:
    experiment.write_json(OUTPUT / "protocol.json", {
        "experiment_type": "matched random dog selection control for the COCO/Open Images gradient",
        "created_at": now(),
        "reference_experiment": str(REFERENCE.resolve()),
        "candidate_pool": str((OI_ROOT / "manifests/dog_candidates.csv").resolve()),
        "random_sampling": "uniform permutation without replacement, independently generated per selection seed",
        "selection_seeds": list(selection_seeds),
        "additions": list(ADDITIONS),
        "models": list(MODELS),
        "training_seeds": list(TRAINING_SEEDS),
        "planned_new_runs": len(selection_seeds) * len(ADDITIONS) * len(MODELS) * len(TRAINING_SEEDS),
        "baseline": "reuse frozen oi_gap_add0 runs because D300 is identical",
        "class_ratio": "dog:cat:horse:sheep:person = 1:1:1:1:1",
        "controlled_factors": [
            "COCO base targets", "Open Images non-dog targets", "validation set", "test set",
            "model architectures", "training seeds", "optimizer", "epochs", "samples per epoch",
            "batch size", "preprocessing and augmentation",
        ],
        "only_changed_factor": "external Open Images dog target selection order: random instead of gap-driven",
        "coverage": coverage_by_condition,
        "fixed_validation_test": str((COCO_ROOT / "data/eval_fixed.csv").resolve()),
        "samples_per_epoch": 3000,
        "epochs": 20,
        "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
    })
    (OUTPUT / "research_log.md").write_text(
        "# Matched Random-Control Research Log\n\n"
        "- Claim tested: gap-driven dog selection differs from uniform random dog selection under an equal data budget.\n"
        "- Falsifiable hypothesis: at matched additions, gap selection improves coverage more efficiently and may improve Macro-F1 or dog F1.\n"
        "- Primary metrics: dog_v3.1 coverage, Macro-F1, dog F1.\n"
        "- Secondary metrics: Accuracy, Macro-Precision/Recall, dog Precision/Recall, small/tiny dog Recall.\n"
        "- Fixed factors: base data, non-dog additions, val/test, eight models, three training seeds, and all training hyperparameters.\n"
        "- Random-set variance: five independently seeded nested random dog permutations.\n"
        "- Success criterion: compare full distributions and effect sizes; no requirement that every model improve.\n"
        "- Raw output: `artifacts/coco_openimages_random_gradient_8_models_20260918`.\n"
        "- Reproduction: `.venv/bin/python -u scripts/run_coco_openimages_random_gradient_8_models.py`.\n"
        "- Resume behavior: completed runs with all four required artifacts are reused.\n",
        encoding="utf-8",
    )


def persist_progress(completed: list[dict], total: int, latest: dict | None, status: str) -> None:
    if completed:
        experiment.write_csv(OUTPUT / "results_by_run.csv", completed)
    experiment.write_json(OUTPUT / "all_metrics.json", completed)
    experiment.write_json(OUTPUT / "progress.json", {
        "status": status,
        "updated_at": now(),
        "completed_runs": len(completed),
        "total_runs": total,
        "percent": 100.0 * len(completed) / total if total else 100.0,
        "latest": latest,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="build and audit manifests without training")
    parser.add_argument(
        "--selection-seeds", type=int, nargs="+", default=list(SELECTION_SEEDS),
        help="independent random dog-selection seeds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_seeds = tuple(args.selection_seeds)
    if len(selection_seeds) != len(set(selection_seeds)):
        raise ValueError("selection seeds must be unique")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coverage_by_condition, audit_rows = build_manifests(selection_seeds)
    write_protocol(selection_seeds, coverage_by_condition)
    total = len(selection_seeds) * len(ADDITIONS) * len(MODELS) * len(TRAINING_SEEDS)
    if args.prepare_only:
        persist_progress([], total, None, "prepared")
        print(json.dumps({
            "status": "prepared", "output": str(OUTPUT.resolve()),
            "manifests": len(audit_rows), "planned_runs": total,
        }, ensure_ascii=False), flush=True)
        return

    eval_rows = experiment.read_csv(COCO_ROOT / "data/eval_fixed.csv")
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    experiment.OUTPUT = OUTPUT
    experiment.MODEL_NAMES = MODELS
    experiment.SEEDS = TRAINING_SEEDS
    completed: list[dict] = []
    persist_progress(completed, total, None, "running")
    try:
        for selection_seed in selection_seeds:
            for add in ADDITIONS:
                condition = f"random_s{selection_seed}_add{add}"
                train_rows = experiment.read_csv(OUTPUT / "manifests" / f"train_{condition}.csv")
                for model in MODELS:
                    for training_seed in TRAINING_SEEDS:
                        print(
                            f"[selection_seed={selection_seed} add={add} model={model} "
                            f"training_seed={training_seed}] starting",
                            flush=True,
                        )
                        metrics = experiment.train_one(
                            model, condition, training_seed, train_rows, val_rows, test_rows
                        )
                        metrics.update({
                            "selection_method": "uniform_random_without_replacement",
                            "selection_seed": selection_seed,
                            "external_dog_added": add,
                            "coverage": coverage_by_condition[condition],
                        })
                        completed.append(metrics)
                        persist_progress(completed, total, metrics, "running")
    except BaseException:
        persist_progress(completed, total, completed[-1] if completed else None, "interrupted")
        raise
    persist_progress(completed, total, completed[-1] if completed else None, "completed")
    print(json.dumps({
        "status": "completed", "output": str(OUTPUT.resolve()), "completed_runs": len(completed),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
