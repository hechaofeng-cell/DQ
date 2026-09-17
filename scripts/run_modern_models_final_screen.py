#!/usr/bin/env python3
"""Final bounded pre-registered architecture screen; no further model search follows."""

from __future__ import annotations

import json
from collections import defaultdict

import run_four_model_gap_vs_random as experiment


MODELS = ("resnext50_32x4d", "convnext_small", "swin_v2_t", "regnet_x_3_2gf")
SEEDS = (20260916, 20260917, 20260918)
OUTPUT = experiment.ROOT / "artifacts/modern_models_final_screen_20260917"


def main() -> None:
    experiment.MODEL_NAMES = MODELS
    experiment.SEEDS = SEEDS
    experiment.OUTPUT = OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = [experiment.model_profile(model_name) for model_name in MODELS]
    experiment.write_json(OUTPUT / "protocol.json", {
        "experiment_type": "final bounded pre-registered architecture screen",
        "models": list(MODELS),
        "conditions_in_run_order": list(experiment.CONDITIONS),
        "classes": [name for name, _ in experiment.CLASS_SPECS],
        "seeds": list(SEEDS),
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
        "search_stop": "No additional architectures will be screened after this batch.",
        "input_note": "SwinV2-T official ImageNet weights use 256 crops; 224 is retained for cross-model protocol consistency.",
    })
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        "- Decision: perform one final bounded screen after the first modern batch yielded one strict-positive model.\n"
        "- Models fixed before results: ResNeXt-50 32x4d, ConvNeXt-Small, SwinV2-T, RegNetX-3.2GF.\n"
        "- Hypothesis: gap596 exceeds random596 on all seven core metric means.\n"
        "- Strict rule: Accuracy, Macro-Precision/Recall/F1, and dog Precision/Recall/F1 mean deltas are all > 0.\n"
        "- Success: at least one model meets the strict rule, producing at least four discovery-stage candidates overall.\n"
        "- Stop criterion: all 24 runs complete; no more architectures or post-result tuning.\n"
        "- Fixed factors: manifests, test set, transforms, optimizer, batch size, epochs, sampling budget, seeds.\n"
        "- Command: `.venv/bin/python -u scripts/run_modern_models_final_screen.py`\n"
        "- Boundary: one random100 list; test set reused; SwinV2-T kept at common 224 input.\n",
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
        "experiment_type": "final bounded pre-registered architecture screen",
        "coverage": {"random596": 0.8132791327913279, "gap596": 0.8417344173441734},
        "profiles": profiles,
        "summary": summary,
        "deltas": deltas,
        "conclusion": experiment.summarize_conclusion(deltas),
        "limitations": ["one random100 subset", "three seeds", "reused test set", "oracle crops"],
    })
    experiment.write_checksums()
    print(json.dumps({"output": str(OUTPUT.resolve()), "models": MODELS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
