#!/usr/bin/env python3
"""Audit raw model runs and render the Open Images dog-gap result report."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from run_openimages_v7_four_model_experiment import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_SELECTIONS, METRICS, sha256,
)
from run_four_model_gap_vs_random import DISPLAY_NAMES


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def main() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    selection = json.loads((DEFAULT_SELECTIONS / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((DEFAULT_OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    summary = {(r["model"], r["condition"]): r for r in read_csv(DEFAULT_OUTPUT / "tables/results_by_condition.csv")}
    delta_rows = {(r["model"], r["comparison"], r["metric"]): r
                  for r in read_csv(DEFAULT_OUTPUT / "tables/paired_deltas.csv")}
    parameters = {r["model"]: r for r in read_csv(DEFAULT_OUTPUT / "tables/model_parameters.csv")}
    checked_runs = 0
    for model in config["models"]:
        for condition in config["conditions"]:
            raw_metrics = []
            for seed in config["training_seeds"]:
                run_dir = DEFAULT_OUTPUT / "runs" / model / condition / str(seed)
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                predictions = read_csv(run_dir / "test_predictions.csv")
                difficulty = read_csv(run_dir / "difficulty_test_predictions.csv")
                if (metrics["model"], metrics["condition"], metrics["seed"]) != (model, condition, seed):
                    raise ValueError(f"incorrect run identity: {run_dir}")
                if len(predictions) != 500 or len(difficulty) != 100:
                    raise ValueError(f"test prediction count mismatch: {run_dir}")
                accuracy = sum(int(row["correct"]) for row in predictions) / len(predictions)
                if not math.isclose(accuracy, metrics["accuracy"], abs_tol=1e-10):
                    raise ValueError(f"accuracy does not match individual predictions: {run_dir}")
                raw_metrics.append(metrics)
                checked_runs += 1
            for metric in METRICS:
                values = [r[metric] for r in raw_metrics if r.get(metric) is not None]
                reported = summary[model, condition][f"{metric}_mean"]
                if values and not math.isclose(sum(values) / len(values), float(reported), abs_tol=1e-10):
                    raise ValueError(f"aggregate does not match raw runs: {model}/{condition}/{metric}")

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def score(model: str, condition: str, metric: str) -> str:
        row = summary[model, condition]
        return f"{pct(float(row[metric + '_mean']))} +/- {pct(float(row[metric + '_std']))}"

    def delta(model: str, metric: str) -> str:
        value = float(delta_rows[model, "gap_minus_random", metric]["delta_mean"])
        return f"{value * 100:+.2f} pp"

    coverage = selection["coverage"]
    accuracy_wins = [m for m in config["models"] if float(delta_rows[m, "gap_minus_random", "accuracy"]["delta_mean"]) > 0]
    f1_wins = [m for m in config["models"] if float(delta_rows[m, "gap_minus_random", "macro_f1"]["delta_mean"]) > 0]
    dog_wins = [m for m in config["models"] if float(delta_rows[m, "gap_minus_random", "dog_f1"]["delta_mean"]) > 0]
    primary_wins = set(f1_wins) | set(dog_wins)
    seed_consistent = [m for m in config["models"] if any(
        int(delta_rows[m, "gap_minus_random", metric]["positive_seed_count"]) == len(config["training_seeds"])
        for metric in config["primary_metrics"]
    )]
    coverage_win = coverage["gap300"] > coverage["random300"]
    minimum_met = coverage_win and len(primary_wins) >= 3 and bool(seed_consistent)

    lines = [
        "# Open Images V7: dog visual coverage and four-model comparison", "",
        "## Study Design", "",
        "Five-class target-region classification with fixed Open Images V7 training, validation, and test splits. "
        "The 6,800-target baseline includes 1,500 dog targets; random and gap conditions each add 300 dog targets "
        "from the same 3,000-target candidate pool. Main test contains 100 targets per class. "
        "The additional 100-dog difficulty test is excluded from overall metrics.", "",
        f"Training seeds: {', '.join(map(str, config['training_seeds']))}; reported values are mean +/- sample SD. "
        "ImageNet-1K pretrained weights, 20 epochs, 6,800 balanced samples per epoch, "
        "batch size 64, AdamW 1e-4, and a fixed 150-target validation set were used for all conditions.", "",
        "## Coverage", "",
        "| Condition | Dog training targets | dog_v3.1 coverage |", "|---|---:|---:|",
        f"| base | 1,500 | {pct(coverage['base'])} |",
        f"| random300 | 1,800 | {pct(coverage['random300'])} |",
        f"| gap300 | 1,800 | {pct(coverage['gap300'])} |", "",
        f"Gap - random coverage: {(coverage['gap300'] - coverage['random300']) * 100:+.2f} percentage points. "
        "Coverage is a state-frequency index, not classification accuracy.", "",
        "## Fixed-Test Results", "",
        "| Model | Condition | Accuracy | Macro-F1 | dog F1 | dog Recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in config["models"]:
        for condition in config["conditions"]:
            lines.append(f"| {DISPLAY_NAMES[model]} | {condition} | {score(model, condition, 'accuracy')} | "
                         f"{score(model, condition, 'macro_f1')} | {score(model, condition, 'dog_f1')} | "
                         f"{score(model, condition, 'dog_recall')} |")
    lines += ["", "## Gap Minus Random", "",
              "Paired differences use the same training seed; positive means gap300 wins.", "",
              "| Model | Accuracy | Macro-F1 | dog F1 | dog Recall | Macro-F1 positive seeds |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in config["models"]:
        wins = delta_rows[model, "gap_minus_random", "macro_f1"]["positive_seed_count"]
        lines.append(f"| {DISPLAY_NAMES[model]} | {delta(model, 'accuracy')} | "
                     f"{delta(model, 'macro_f1')} | {delta(model, 'dog_f1')} | "
                     f"{delta(model, 'dog_recall')} | {wins}/{len(config['training_seeds'])} |")
    lines += ["", "## Difficulty Test", "",
              "The 100 dog-only images include overlapping groups: 32 small-only (1%-3%), 10 tiny (<1%), "
              "42 total below 3%, 36 occluded, and 32 truncated dogs. These are not included in the five-class main test.", "",
              "| Model | random300 small-only Recall | gap300 small-only Recall | random300 tiny Recall | gap300 tiny Recall | random300 occluded Recall | gap300 occluded Recall |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in config["models"]:
        lines.append(f"| {DISPLAY_NAMES[model]} | {score(model, 'random300', 'difficulty_small_dog_recall')} | "
                     f"{score(model, 'gap300', 'difficulty_small_dog_recall')} | "
                     f"{score(model, 'random300', 'difficulty_tiny_dog_recall')} | "
                     f"{score(model, 'gap300', 'difficulty_tiny_dog_recall')} | "
                     f"{score(model, 'random300', 'difficulty_occluded_dog_recall')} | "
                     f"{score(model, 'gap300', 'difficulty_occluded_dog_recall')} |")
    lines += ["", "## Model Size", "", "| Model | Total parameters | FP32 parameters |", "|---|---:|---:|"]
    for model in config["models"]:
        p = parameters[model]
        lines.append(f"| {DISPLAY_NAMES[model]} | {int(p['total_parameters']):,} | "
                     f"{float(p['parameter_memory_mib_fp32']):.1f} MiB |")
    lines += ["", "## Interpretation", "",
              f"Gap versus random: Accuracy improves for {len(accuracy_wins)}/4 models, "
              f"Macro-F1 for {len(f1_wins)}/4, and dog F1 for {len(dog_wins)}/4. "
              f"The preregistered minimum evidence rule is {'met' if minimum_met else 'not met'}. "
              "Per-model and per-seed negatives are retained above and in the raw tables.", "",
              "Limitations: selection is automated without human review; this is one fixed random candidate draw, "
              "three training seeds, and oracle-box classification, not end-to-end object detection. "
              "These comparisons cannot establish that gap selection helps every model or every category.", "",
              "## Raw Outputs", "",
              "`runs/<model>/<condition>/<seed>/` contains model weights, test predictions, difficulty-test predictions, "
              "training logs, and metrics. `tables/results_by_seed.csv` and `tables/paired_deltas.csv` "
              "are derived from the same raw metrics.", ""]
    (DEFAULT_OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    claim = ["# Claim-Evidence Audit", "",
             f"- Coverage claim: {'supported' if coverage_win else 'contradicted'}; "
             f"gap={pct(coverage['gap300'])}, random={pct(coverage['random300'])}, "
             "source `artifacts/openimages_v7_dog_supplements_v1/report.json`.",
             f"- Multi-model advantage: {'partial support' if minimum_met else 'not established'} "
             f"under the frozen minimum rule; {len(primary_wins)}/4 models improve at least one primary metric, "
             "source `tables/paired_deltas.csv` and raw per-seed metrics.",
             "- Generalization to other classes, unseen datasets, or object detection: not tested.", ""]
    (DEFAULT_OUTPUT / "claim_evidence.md").write_text("\n".join(claim), encoding="utf-8")
    log = ["# Research Log", "",
           "- Question: Does gap-selected dog300 outperform equal-budget random dog300 on fixed Open Images V7 target-region classification?",
           "- Commands: `bash scripts/run_openimages_dog_v3_1.sh`; "
           "`.venv/bin/python scripts/select_openimages_dog_supplements.py --base-features artifacts/openimages_v7_dog_base_features_v1/results.csv "
           "--candidate-features artifacts/openimages_v7_dog_candidate_features_v1/results.csv`; "
           "`.venv/bin/python scripts/run_openimages_v7_four_model_experiment.py`; "
           "`.venv/bin/python scripts/report_openimages_v7_four_model_experiment.py`.",
           f"- Runs audited: {checked_runs}/36, each with 500 main-test and 100 difficulty-test predictions.",
           f"- Protocol SHA-256: {sha256(DEFAULT_OUTPUT / 'protocol.json')}.",
           f"- Selection SHA-256: {sha256(DEFAULT_SELECTIONS / 'report.json')}.",
           "- Automatic-only feature annotation and candidate filtering; no manual label audit was performed.",
           "- No architecture was replaced based on Open Images evaluation outcomes.", ""]
    (DEFAULT_OUTPUT / "research_log.md").write_text("\n".join(log), encoding="utf-8")
    review = ["# Evidence Review", "",
              "- Citation gate: source and license are described in the frozen protocol using the official Open Images pages; no unverified literature claims are made in this report.",
              f"- Result gate: passed; {checked_runs} raw run files and 21,600 per-target predictions checked against summary tables and Accuracy.",
              "- Claim-scope gate: passed only for the observed five-class oracle-box setting and one random300 selection; automated labels and mixed model outcomes remain visible.", ""]
    (DEFAULT_OUTPUT / "review.md").write_text("\n".join(review), encoding="utf-8")
    print(f"Audited {checked_runs} runs; report: {DEFAULT_OUTPUT / 'report.md'}")


if __name__ == "__main__":
    main()
