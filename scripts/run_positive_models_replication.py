#!/usr/bin/env python3
"""Replicate the gap100-vs-random100 comparison on the three selected models."""

from __future__ import annotations

import json
from collections import defaultdict

import run_four_model_gap_vs_random as experiment


SELECTED_MODELS = ("resnet18", "convnext_tiny", "swin_t")
REPLICATION_SEEDS = (20260919, 20260920, 20260921)
OUTPUT = experiment.ROOT / "artifacts/positive_models_seed_replication_20260917"


def main() -> None:
    experiment.MODEL_NAMES = SELECTED_MODELS
    experiment.SEEDS = REPLICATION_SEEDS
    experiment.OUTPUT = OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)

    profiles = [experiment.model_profile(model_name) for model_name in SELECTED_MODELS]
    experiment.write_json(OUTPUT / "protocol.json", {
        "experiment_type": "training-seed replication after exploratory model selection",
        "selection_artifact": str((experiment.ROOT / "artifacts/positive_models_all_core_metrics_20260917").resolve()),
        "models": list(SELECTED_MODELS),
        "conditions_in_run_order": list(experiment.CONDITIONS),
        "classes": [name for name, _ in experiment.CLASS_SPECS],
        "seeds": list(REPLICATION_SEEDS),
        "epochs": 20,
        "samples_per_epoch": 3000,
        "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "selection_metric": "validation macro-F1",
        "input_size": 224,
        "data_root": str(experiment.DATA_ROOT.resolve()),
        "gap_dog_manifest": str((experiment.DATA_ROOT / "data/train_dog596.csv").resolve()),
        "random_dog_manifest": str((experiment.DATA_ROOT / "data/train_dog_random596.csv").resolve()),
        "other_train_manifest": str((experiment.DATA_ROOT / "data/train_other_fixed.csv").resolve()),
        "eval_manifest": str((experiment.DATA_ROOT / "data/eval_fixed.csv").resolve()),
        "claim_boundary": (
            "New training seeds test optimization stability on the same fixed test set. "
            "They are not an independent confirmation because model selection used that test set."
        ),
    })
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        "- Decision: test whether the three mean-positive models retain the gap100 advantage under new training seeds.\n"
        "- Hypothesis: gap596 exceeds random596 on Macro-F1 and dog F1 means.\n"
        "- Models: ResNet-18, ConvNeXt-Tiny, Swin-T.\n"
        "- New seeds: 20260919, 20260920, 20260921.\n"
        "- Fixed factors: data manifests, test set, transforms, optimizer, epochs, sampling budget.\n"
        "- Primary metrics: Macro-F1 and dog F1.\n"
        "- Success: both primary mean deltas are positive for at least two of three models.\n"
        "- Partial support: only one primary metric or one model is positive.\n"
        "- Stop criterion: one complete three-seed run per condition and model; no post-result tuning.\n"
        "- Command: `.venv/bin/python -u scripts/run_positive_models_replication.py`\n"
        "- Boundary: stability replication on the reused test set, not independent confirmation.\n",
        encoding="utf-8",
    )

    gap_dogs = experiment.read_csv(experiment.DATA_ROOT / "data/train_dog596.csv")
    random_dogs = experiment.read_csv(experiment.DATA_ROOT / "data/train_dog_random596.csv")
    other_rows = experiment.read_csv(experiment.DATA_ROOT / "data/train_other_fixed.csv")
    eval_rows = experiment.read_csv(experiment.DATA_ROOT / "data/eval_fixed.csv")
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    condition_rows = {
        "gap596": gap_dogs + other_rows,
        "random596": random_dogs + other_rows,
    }

    all_metrics = defaultdict(lambda: defaultdict(list))
    for model_name in SELECTED_MODELS:
        for condition in experiment.CONDITIONS:
            for seed in REPLICATION_SEEDS:
                metrics = experiment.train_one(
                    model_name,
                    condition,
                    seed,
                    condition_rows[condition],
                    val_rows,
                    test_rows,
                )
                all_metrics[model_name][condition].append(metrics)

    summary, deltas = experiment.aggregate(all_metrics, profiles)
    experiment.make_report(summary, deltas, profiles)
    experiment.write_json(OUTPUT / "report.json", {
        "experiment_type": "training-seed replication",
        "coverage": {
            "random596": 0.8132791327913279,
            "gap596": 0.8417344173441734,
            "gap_minus_random": 0.0284552845528455,
        },
        "profiles": profiles,
        "summary": summary,
        "deltas": deltas,
        "conclusion": experiment.summarize_conclusion(deltas),
        "limitations": [
            "models selected using prior results on the same test set",
            "one random100 subset",
            "three new training seeds",
            "oracle target crops",
        ],
    })
    experiment.write_checksums()
    print(json.dumps({
        "output": str(OUTPUT.resolve()),
        "models": SELECTED_MODELS,
        "seeds": REPLICATION_SEEDS,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
