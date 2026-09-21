#!/usr/bin/env python3
"""Estimate dog state utility from stratified out-of-fold classifier predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from openimages_cua_vsl import (
    StateSchema,
    feature_rows_by_id,
    finite,
    normalize_priority_scores,
    read_csv,
    seed_everything,
    state_counts,
    write_csv,
    write_json,
)
from run_four_model_gap_vs_random import build_model
from run_resnet18_five_class_coverage_comparison import CropDataset, evaluate, predict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/openimages_v7_cua_vsl_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transforms_for_imagenet():
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
    ])
    return train_transform, eval_transform


def run_fold(
    model_name: str, fold: int, seed: int, train_rows: list[dict], holdout_rows: list[dict],
    val_rows: list[dict], output: Path, epochs: int, workers: int,
) -> list[dict]:
    run_dir = output / "oof" / model_name / f"fold_{fold}"
    prediction_path = run_dir / "holdout_predictions.csv"
    if prediction_path.is_file() and (run_dir / "metrics.json").is_file():
        print(f"[{model_name} fold={fold}] reusing completed fold", flush=True)
        return read_csv(prediction_path)

    seed_everything(seed)
    train_transform, eval_transform = transforms_for_imagenet()
    class_counts = Counter(int(row["label"]) for row in train_rows)
    weights = [1 / class_counts[int(row["label"])] for row in train_rows]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(train_rows), replacement=True, generator=generator)
    train_loader = DataLoader(
        CropDataset(train_rows, train_transform), batch_size=64, sampler=sampler,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0, generator=generator,
    )
    val_loader = DataLoader(
        CropDataset(val_rows, eval_transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    holdout_loader = DataLoader(
        CropDataset(holdout_rows, eval_transform), batch_size=128, shuffle=False,
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
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[{model_name} fold={fold}] {epoch}/{epochs} "
            f"loss={loss_sum / seen:.4f} val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no best state was selected")
    model.load_state_dict(best_state)
    holdout_predictions = predict(model, holdout_loader, holdout_rows, device)
    holdout_metrics = evaluate(holdout_predictions)
    elapsed = time.perf_counter() - started
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state, "model_name": model_name, "fold": fold, "seed": seed,
            "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        },
        run_dir / "best.pt",
    )
    write_csv(run_dir / "train_log.csv", history)
    write_csv(prediction_path, holdout_predictions)
    write_json(run_dir / "metrics.json", {
        "model": model_name,
        "fold": fold,
        "seed": seed,
        "train_size": len(train_rows),
        "holdout_size": len(holdout_rows),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "holdout_macro_f1": holdout_metrics["macro_f1"],
        "holdout_dog_recall": holdout_metrics["dog_recall"],
        "elapsed_seconds": elapsed,
    })
    return holdout_predictions


def aggregate_priorities(
    model_predictions: dict[str, list[dict]], base_features: dict[str, dict],
    schema: StateSchema, cap: int, beta: float, output: Path,
) -> dict:
    dog_feature_rows = list(base_features.values())
    counts = state_counts(dog_feature_rows, schema)
    all_states = schema.coverage_states()
    model_stats = {}
    per_model_errors = {}
    for model_name, predictions in model_predictions.items():
        dog_predictions = [row for row in predictions if row["class_name"] == "dog"]
        if len(dog_predictions) != len(base_features):
            raise ValueError(
                f"{model_name}: expected {len(base_features)} dog OOF predictions, got {len(dog_predictions)}"
            )
        dog_by_id = {row["sample_id"]: row for row in dog_predictions}
        if set(dog_by_id) != set(base_features):
            raise ValueError(f"{model_name}: dog OOF IDs do not match base feature IDs")
        global_errors = [1 - int(dog_by_id[sample_id]["correct"]) for sample_id in base_features]
        global_error = float(np.mean(global_errors))
        state_support = Counter()
        state_false_negative = Counter()
        for sample_id, feature_row in base_features.items():
            missed = 1 - int(dog_by_id[sample_id]["correct"])
            for state in schema.state_pairs(feature_row):
                state_support[state] += 1
                state_false_negative[state] += missed
        errors = {
            state: finite(
                (state_false_negative[state] + beta * global_error) / (state_support[state] + beta),
                f"{model_name}:{state}",
            )
            for state in all_states
        }
        per_model_errors[model_name] = errors
        model_stats[model_name] = {
            "dog_oof_samples": len(dog_predictions),
            "dog_oof_error": global_error,
            "dog_oof_recall": 1.0 - global_error,
        }

    rows = []
    raw_coverage = []
    raw_utility = []
    for feature_id, state in all_states:
        key = (feature_id, state)
        deficit = max(0.0, cap - counts[key]) / cap
        model_error_values = [per_model_errors[name][key] for name in model_predictions]
        mean_error = float(np.mean(model_error_values))
        joint = deficit * mean_error
        row = {
            "feature_id": feature_id,
            "state": state,
            "train_count": counts[key],
            "coverage_deficit": deficit,
            "mean_smoothed_error": mean_error,
            "joint_priority": joint,
        }
        for model_name in model_predictions:
            row[f"{model_name}_smoothed_error"] = per_model_errors[model_name][key]
        rows.append(row)
        raw_coverage.append(deficit)
        raw_utility.append(joint)

    coverage_norm = normalize_priority_scores(raw_coverage)
    utility_norm = normalize_priority_scores(raw_utility)
    for row, coverage_value, utility_value in zip(rows, coverage_norm, utility_norm):
        row["coverage_priority_norm"] = coverage_value
        row["utility_priority_norm"] = utility_value
    rows.sort(key=lambda row: (-row["joint_priority"], row["feature_id"], row["state"]))
    write_csv(output / "state_priority" / "state_priority.csv", rows)
    report = {
        "states": len(rows),
        "coverage_cap": cap,
        "utility_shrinkage_beta": beta,
        "reference_models": model_stats,
        "missing_states": sum(row["train_count"] == 0 for row in rows),
        "states_below_cap": sum(row["train_count"] < cap for row in rows),
        "top_joint_priorities": rows[:20],
    }
    write_json(output / "state_priority" / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = Path(config["data"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    schema = StateSchema.from_path(Path(config["schema"]))
    base_rows = read_csv(data / "manifests/train_base.csv")
    val_rows = read_csv(data / "manifests/validation.csv")
    base_features = feature_rows_by_id(Path(config["base_features"]))
    dog_ids = {row["sample_id"] for row in base_rows if row["class_name"] == "dog"}
    if dog_ids != set(base_features):
        raise ValueError("base dog features do not match the frozen base training manifest")

    folds = int(config["utility_folds"])
    labels = np.asarray([int(row["label"]) for row in base_rows])
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(config["utility_fold_seed"])
    )
    split_indices = list(splitter.split(np.zeros(len(base_rows)), labels))
    write_json(output / "oof" / "fold_assignment.json", {
        "folds": folds,
        "fold_seed": config["utility_fold_seed"],
        "assignments": [
            {
                "fold": fold,
                "holdout_sample_ids": [base_rows[index]["sample_id"] for index in holdout_indices],
            }
            for fold, (_, holdout_indices) in enumerate(split_indices)
        ],
    })

    epochs = args.epochs or int(config["epochs"])
    model_predictions = {}
    for model_name in config["utility_reference_models"]:
        predictions = []
        for fold, (train_indices, holdout_indices) in enumerate(split_indices):
            train_rows = [base_rows[index] for index in train_indices]
            holdout_rows = [base_rows[index] for index in holdout_indices]
            seed = int(config["utility_training_seeds"][fold])
            predictions.extend(
                run_fold(
                    model_name, fold, seed, train_rows, holdout_rows, val_rows,
                    output, epochs, args.workers,
                )
            )
        if len(predictions) != len(base_rows) or len({row["sample_id"] for row in predictions}) != len(base_rows):
            raise ValueError(f"{model_name}: incomplete or duplicate OOF predictions")
        predictions.sort(key=lambda row: row["sample_id"])
        write_csv(output / "oof" / model_name / "oof_predictions.csv", predictions)
        model_predictions[model_name] = predictions

    report = aggregate_priorities(
        model_predictions, base_features, schema,
        int(config["coverage_cap"]), float(config["utility_shrinkage_beta"]), output,
    )
    protocol = {
        "experiment_id": config["experiment_id"],
        "stage": "oof_utility_complete",
        "epochs": epochs,
        "input_sha256": {
            "config": sha256(args.config),
            "train_base": sha256(data / "manifests/train_base.csv"),
            "validation": sha256(data / "manifests/validation.csv"),
            "schema": sha256(Path(config["schema"])),
            "base_features": sha256(Path(config["base_features"])),
        },
        "report": report,
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
