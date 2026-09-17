#!/usr/bin/env python3
"""Select models whose mean gap-minus-random delta is positive on every core metric."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/four_models_coco5_gap_vs_random_20260917"
DEFAULT_OUTPUT = ROOT / "artifacts/positive_models_all_core_metrics_20260917"
CORE_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "dog_precision",
    "dog_recall",
    "dog_f1",
)
SEEDS = (20260916, 20260917, 20260918)
POSITIVE_TOLERANCE = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(source: Path, model: str, condition: str, seed: int) -> Path:
    local_run = source / "runs" / model / condition / str(seed)
    if local_run.is_dir():
        return local_run
    if model == "resnet18":
        source_condition = {"gap596": "dog596", "random596": "dog_random596"}[condition]
        return ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916" / source_condition / str(seed)
    return local_run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in protocol.get("seeds", SEEDS))
    deltas = read_csv(source / "tables/gap_minus_random.csv")
    profiles = {row["model"]: row for row in read_csv(source / "tables/model_parameters.csv")}
    by_model: dict[str, dict[str, dict[str, str]]] = {}
    for row in deltas:
        by_model.setdefault(row["model"], {})[row["metric"]] = row

    selected = [
        model
        for model, metrics in by_model.items()
        if all(float(metrics[metric]["gap_minus_random_mean"]) > POSITIVE_TOLERANCE
               for metric in CORE_METRICS)
    ]
    all_seed_all_metric_winners = [
        model
        for model in selected
        if all(int(by_model[model][metric]["positive_seed_count"]) == len(seeds)
               for metric in CORE_METRICS)
    ]
    selection_rows = []
    for model in selected:
        profile = profiles[model]
        row = {
            "model": model,
            "display_name": profile["display_name"],
            "total_parameters": profile["total_parameters"],
        }
        for metric in CORE_METRICS:
            row[f"delta_{metric}"] = by_model[model][metric]["gap_minus_random_mean"]
        selection_rows.append(row)

    checkpoint_rows = []
    for model in selected:
        for condition in ("gap596", "random596"):
            for seed in seeds:
                run_dir = checkpoint_path(source, model, condition, seed)
                for required in ("best.pt", "metrics.json", "test_predictions.csv", "train_log.csv"):
                    if not (run_dir / required).is_file():
                        raise FileNotFoundError(run_dir / required)
                weight = run_dir / "best.pt"
                checkpoint_rows.append({
                    "model": model,
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": str(weight.resolve()),
                    "checkpoint_sha256": sha256(weight),
                    "metrics": str((run_dir / "metrics.json").resolve()),
                    "predictions": str((run_dir / "test_predictions.csv").resolve()),
                    "train_log": str((run_dir / "train_log.csv").resolve()),
                })

    output.mkdir(parents=True, exist_ok=True)
    selection_fields = ["model", "display_name", "total_parameters"] + [
        f"delta_{metric}" for metric in CORE_METRICS
    ]
    checkpoint_fields = [
        "model", "condition", "seed", "checkpoint", "checkpoint_sha256",
        "metrics", "predictions", "train_log",
    ]
    write_csv(output / "selected_models.csv", selection_rows, selection_fields)
    write_csv(output / "checkpoint_manifest.csv", checkpoint_rows, checkpoint_fields)
    selection = {
        "source": str(source),
        "selection_rule": (
            f"gap_minus_random_mean > {POSITIVE_TOLERANCE:g} for every core metric; "
            "tolerance excludes floating-point representations of zero"
        ),
        "core_metrics": list(CORE_METRICS),
        "selected_models": selected,
        "selected_count": len(selected),
        "all_seed_all_metric_winner_count": len(all_seed_all_metric_winners),
        "interpretation": (
            "Selection is based on three-seed means. No model improved every core metric "
            "in every seed, so the selected models are candidates for replication, not "
            "confirmed universal winners."
        ),
        "recommended_replication_seeds": [max(seeds) + offset for offset in (1, 2, 3)],
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metric_labels = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro-Precision",
        "macro_recall": "Macro-Recall",
        "macro_f1": "Macro-F1",
        "dog_precision": "dog Precision",
        "dog_recall": "dog Recall",
        "dog_f1": "dog F1",
    }
    lines = [
        "# 全部核心指标均提升的模型筛选", "",
        "筛选规则：七项核心指标的三训练种子均值均满足 `gap596 - random596 > 0`。", "",
        "| 模型 | 参数量 | " + " | ".join(metric_labels[m] for m in CORE_METRICS) + " |",
        "|---|---:|" + "---:|" * len(CORE_METRICS),
    ]
    for row in selection_rows:
        values = [f"{float(row[f'delta_{metric}']):+.2%}" for metric in CORE_METRICS]
        lines.append(
            f"| {row['display_name']} | {int(row['total_parameters']):,} | "
            + " | ".join(values) + " |"
        )
    lines += [
        "", "## 结论边界", "",
        "- 入选模型：" + "、".join(profiles[model]["display_name"] for model in selected) + "。",
        "- 这里的“全部提升”指七项核心指标的三种子均值全部为正。",
        "- 若要求每个种子上的每项指标都为正，则当前没有模型满足。",
        "- 这些模型由已有测试结果筛选，因此后续实验属于复现/稳定性验证；若要做独立确认，应使用未参与筛选的新测试集。",
        f"- `checkpoint_manifest.csv` 索引{len(checkpoint_rows)}组已有权重、指标、预测和训练日志。", "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
