#!/usr/bin/env python3
"""Run the preregistered four-model Open Images V7 dog-gap experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from run_four_model_gap_vs_random import DISPLAY_NAMES, build_model, model_profile
from run_resnet18_dog_coverage_comparison import ROOT, write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CropDataset, evaluate, predict, seed_everything


DEFAULT_CONFIG = ROOT / "configs/openimages_v7_dog_gap_experiment_v1.json"
DEFAULT_DATA = ROOT / "data/openimages_v7_dog_gap_v1"
DEFAULT_SELECTIONS = ROOT / "artifacts/openimages_v7_dog_supplements_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/openimages_v7_four_model_experiment_v1"
METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "dog_precision", "dog_recall", "dog_f1", "small_dog_recall",
    "occluded_dog_recall", "truncated_dog_recall",
    "difficulty_dog_recall", "difficulty_small_dog_recall",
    "difficulty_tiny_dog_recall", "difficulty_occluded_dog_recall",
    "difficulty_truncated_dog_recall",
)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifests(train_rows: dict, val_rows: list[dict], test_rows: list[dict],
                    difficulty_rows: list[dict], config: dict) -> None:
    counts = config["counts"]
    base = train_rows["base"]
    base_ids = {row["sample_id"] for row in base}
    if len(base_ids) != len(base) or len(base) != sum(counts["train_base_by_class"].values()):
        raise ValueError("invalid baseline training manifest")
    for class_name, expected in counts["train_base_by_class"].items():
        if sum(row["class_name"] == class_name for row in base) != expected:
            raise ValueError(f"invalid baseline count for {class_name}")
    for condition in ("random300", "gap300"):
        rows = train_rows[condition]
        ids = {row["sample_id"] for row in rows}
        if len(ids) != len(rows) or len(rows) != len(base) + counts["supplement_budget"]:
            raise ValueError(f"invalid {condition} count or duplicate sample")
        if not base_ids <= ids or any(row["class_name"] != "dog" for row in rows if row["sample_id"] not in base_ids):
            raise ValueError(f"{condition} must contain the unchanged baseline and only added dog targets")
    if any(len(rows) != expected for rows, expected in (
        (val_rows, counts["validation_per_class"] * len(config["classes"])),
        (test_rows, counts["test_per_class"] * len(config["classes"])),
        (difficulty_rows, counts["difficulty_test_dog"]),
    )):
        raise ValueError("validation or test manifest has an unexpected size")
    train_image_ids = {row["image_id"] for rows in train_rows.values() for row in rows}
    val_ids = {row["image_id"] for row in val_rows}
    test_ids = {row["image_id"] for row in test_rows}
    difficulty_ids = {row["image_id"] for row in difficulty_rows}
    if train_image_ids & (val_ids | test_ids | difficulty_ids) or val_ids & (test_ids | difficulty_ids) or test_ids & difficulty_ids:
        raise ValueError("image overlap across training, validation, or test splits")
    if any(not Path(row["crop_path"]).is_file() for rows in (*train_rows.values(), val_rows, test_rows, difficulty_rows) for row in rows):
        raise ValueError("at least one target crop is missing")


def slice_recall(predictions: list[dict], field: str, expected: str = "1") -> float | None:
    rows = [
        row for row in predictions
        if row["class_name"] == "dog" and row.get(field) == expected
    ]
    if not rows:
        return None
    return sum(int(row["correct"]) for row in rows) / len(rows)


def train_one(
    model_name: str, condition: str, seed: int, train_rows: list[dict], val_rows: list[dict],
    test_rows: list[dict], difficulty_rows: list[dict], output: Path, epochs: int,
    workers: int, samples_per_epoch: int,
) -> dict:
    run_dir = output / "runs" / model_name / condition / str(seed)
    required = ("best.pt", "metrics.json", "test_predictions.csv", "difficulty_test_predictions.csv", "train_log.csv")
    if all((run_dir / name).is_file() for name in required):
        print(f"[{model_name} {condition} seed={seed}] reusing completed run", flush=True)
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    seed_everything(seed)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
    ])
    class_counts = Counter(int(row["label"]) for row in train_rows)
    weights = [1 / class_counts[int(row["label"])] for row in train_rows]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True, generator=generator)
    train_loader = DataLoader(
        CropDataset(train_rows, train_transform), batch_size=64, sampler=sampler,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0, generator=generator,
    )
    val_loader = DataLoader(
        CropDataset(val_rows, eval_transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        CropDataset(test_rows, eval_transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    difficulty_loader = DataLoader(
        CropDataset(difficulty_rows, eval_transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )

    device = torch.device("cuda")
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        scheduler.step()
        val_predictions = predict(model, val_loader, val_rows, device)
        val_metrics = evaluate(val_predictions)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[{model_name} {condition} seed={seed}] {epoch}/{epochs} "
            f"loss={loss_sum / seen:.4f} val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    predictions = predict(model, test_loader, test_rows, device)
    difficulty_predictions = predict(model, difficulty_loader, difficulty_rows, device)
    metrics = evaluate(predictions)
    metrics["occluded_dog_recall"] = slice_recall(predictions, "is_occluded")
    metrics["truncated_dog_recall"] = slice_recall(predictions, "is_truncated")
    metrics["difficulty_dog_recall"] = sum(int(row["correct"]) for row in difficulty_predictions) / len(difficulty_predictions)
    metrics["difficulty_small_dog_recall"] = slice_recall(difficulty_predictions, "bbox_area_band", "small")
    metrics["difficulty_tiny_dog_recall"] = slice_recall(difficulty_predictions, "bbox_area_band", "tiny")
    metrics["difficulty_occluded_dog_recall"] = slice_recall(difficulty_predictions, "is_occluded")
    metrics["difficulty_truncated_dog_recall"] = slice_recall(difficulty_predictions, "is_truncated")
    metrics.update({
        "model": model_name,
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_counts": dict(class_counts),
        "samples_per_epoch": samples_per_epoch,
        "elapsed_seconds": time.perf_counter() - started,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "model_name": model_name, "condition": condition, "seed": seed, "metrics": metrics},
        run_dir / "best.pt",
    )
    write_csv(run_dir / "train_log.csv", history)
    write_csv(run_dir / "test_predictions.csv", predictions)
    write_csv(run_dir / "difficulty_test_predictions.csv", difficulty_predictions)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def aggregate(all_metrics: dict, models: list[str], conditions: list[str], seeds: list[int], output: Path) -> None:
    summary = []
    seed_rows = []
    for model_name in models:
        for condition in conditions:
            runs = all_metrics[model_name][condition]
            row = {"model": model_name, "display_name": DISPLAY_NAMES[model_name], "condition": condition, "runs": len(runs)}
            for metric in METRICS:
                values = [run[metric] for run in runs if run.get(metric) is not None]
                row[f"{metric}_mean"] = float(np.mean(values)) if values else None
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            summary.append(row)
            for run in runs:
                seed_rows.append({
                    "model": model_name, "condition": condition, "seed": run["seed"],
                    **{metric: run.get(metric) for metric in METRICS},
                    "best_epoch": run["best_epoch"], "elapsed_seconds": run["elapsed_seconds"],
                })

    deltas = []
    for model_name in models:
        for comparison, left, right in (
            ("gap_minus_random", "gap300", "random300"),
            ("gap_minus_base", "gap300", "base"),
            ("random_minus_base", "random300", "base"),
        ):
            for metric in METRICS:
                paired = []
                for index, _ in enumerate(seeds):
                    left_value = all_metrics[model_name][left][index].get(metric)
                    right_value = all_metrics[model_name][right][index].get(metric)
                    if left_value is not None and right_value is not None:
                        paired.append(left_value - right_value)
                deltas.append({
                    "model": model_name, "display_name": DISPLAY_NAMES[model_name],
                    "comparison": comparison, "metric": metric,
                    "delta_mean": float(np.mean(paired)) if paired else None,
                    "delta_std": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0 if paired else None,
                    "positive_seed_count": sum(value > 0 for value in paired),
                    "negative_seed_count": sum(value < 0 for value in paired),
                    "seed_deltas": json.dumps(paired),
                })
    write_csv(output / "tables/results_by_condition.csv", summary)
    write_csv(output / "tables/results_by_seed.csv", seed_rows)
    write_csv(output / "tables/paired_deltas.csv", deltas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["models"]
    conditions = config["conditions"]
    seeds = config["training_seeds"]
    samples_per_epoch = args.samples_per_epoch or config["samples_per_epoch"]
    train_rows = {
        condition: read_csv(args.selections / f"manifests/train_{condition}.csv")
        for condition in conditions
    }
    val_rows = read_csv(args.data / "manifests/validation.csv")
    test_rows = read_csv(args.data / "manifests/test.csv")
    difficulty_rows = read_csv(args.data / "manifests/dog_difficulty_test.csv")
    check_manifests(train_rows, val_rows, test_rows, difficulty_rows, config)
    for row in difficulty_rows:
        area = float(row["bbox_area_ratio"])
        row["bbox_area_band"] = "tiny" if area < .01 else "small" if area < .03 else "regular"
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment_id": config["experiment_id"],
        "models": models,
        "conditions": conditions,
        "seeds": seeds,
        "epochs": args.epochs,
        "samples_per_epoch": samples_per_epoch,
        "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "selection_metric": "validation macro-F1",
        "input_size": 224,
        "primary_metrics": config["primary_metrics"],
        "data": str(args.data.resolve()),
        "selections": str(args.selections.resolve()),
        "selection_review_status": json.loads((args.selections / "report.json").read_text(encoding="utf-8"))["selection_review_status"],
        "input_sha256": {
            "config": sha256(args.config),
            **{condition: sha256(args.selections / f"manifests/train_{condition}.csv") for condition in conditions},
            "validation": sha256(args.data / "manifests/validation.csv"),
            "test": sha256(args.data / "manifests/test.csv"),
            "dog_difficulty_test": sha256(args.data / "manifests/dog_difficulty_test.csv"),
        },
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("experiment protocol or input manifests changed; use a new output directory")
    write_json(protocol_path, protocol)
    write_csv(args.output / "tables/model_parameters.csv", [model_profile(name) for name in models])

    all_metrics = defaultdict(lambda: defaultdict(list))
    for model_name in models:
        for condition in conditions:
            for seed in seeds:
                metrics = train_one(
                    model_name, condition, seed, train_rows[condition], val_rows, test_rows,
                    difficulty_rows, args.output, args.epochs, args.workers, samples_per_epoch,
                )
                all_metrics[model_name][condition].append(metrics)
    aggregate(all_metrics, models, conditions, seeds, args.output)
    print(f"completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
