#!/usr/bin/env python3
"""New-seed replication for the two primary-metric-positive modern models."""

from __future__ import annotations

import json
from collections import defaultdict

import run_four_model_gap_vs_random as experiment


MODELS = ("efficientnet_v2_s", "maxvit_t")
SEEDS = (20260919, 20260920, 20260921)
OUTPUT = experiment.ROOT / "artifacts/modern_positive_models_replication_20260917"


def main() -> None:
    experiment.MODEL_NAMES = MODELS
    experiment.SEEDS = SEEDS
    experiment.OUTPUT = OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = [experiment.model_profile(model_name) for model_name in MODELS]
    experiment.write_json(OUTPUT / "protocol.json", {
        "experiment_type": "new-seed replication of primary-metric-positive modern models",
        "selection_source": str((experiment.ROOT / "artifacts/modern_models_screen_20260917").resolve()),
        "models": list(MODELS),
        "conditions_in_run_order": list(experiment.CONDITIONS),
        "classes": [name for name, _ in experiment.CLASS_SPECS],
        "seeds": list(SEEDS),
        "epochs": 20,
        "samples_per_epoch": 3000,
        "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "selection_metric": "validation macro-F1",
        "primary_confirmation_metrics": ["accuracy", "macro_f1", "dog_f1"],
        "strict_all_core_metrics": [
            "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "dog_precision", "dog_recall", "dog_f1",
        ],
        "input_size": 224,
        "data_root": str(experiment.DATA_ROOT.resolve()),
    })
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        "- Decision: replicate EfficientNetV2-S and MaxVit-T with three unseen training seeds.\n"
        "- Selection source: first pre-registered modern-model screen.\n"
        "- Primary confirmation rule: mean deltas for Accuracy, Macro-F1, and dog F1 are all > 0.\n"
        "- Strict secondary rule: all seven core metric mean deltas are > 0.\n"
        "- Success: both models satisfy the primary confirmation rule.\n"
        "- Partial support: exactly one model satisfies it.\n"
        "- Stop criterion: 12 complete runs; no model replacement or post-result tuning.\n"
        "- Fixed factors: manifests, test set, transforms, optimizer, batch size, epochs, sampling budget.\n"
        "- Command: `.venv/bin/python -u scripts/run_modern_positive_replication.py`\n"
        "- Boundary: same reused test set and one random100 list; this tests seed stability only.\n",
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
    for model_name in MODELS:
        for condition in experiment.CONDITIONS:
            for seed in SEEDS:
                all_metrics[model_name][condition].append(experiment.train_one(
                    model_name, condition, seed, condition_rows[condition], val_rows, test_rows
                ))
    summary, deltas = experiment.aggregate(all_metrics, profiles)
    experiment.make_report(summary, deltas, profiles)
    experiment.write_json(OUTPUT / "report.json", {
        "experiment_type": "new-seed replication",
        "profiles": profiles,
        "summary": summary,
        "deltas": deltas,
        "conclusion": experiment.summarize_conclusion(deltas),
        "limitations": ["selected on reused test set", "one random100 subset", "three new seeds"],
    })
    experiment.write_checksums()
    print(json.dumps({"output": str(OUTPUT.resolve()), "models": MODELS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
