#!/usr/bin/env python3
"""Train the CUA-VSL ablation matrix with frozen manifests and budgets."""
from __future__ import annotations

import argparse
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

from openimages_cua_vsl import (
    StateAwareCropDataset,
    StateSchema,
    VisualStateMultiTaskModel,
    class_balanced_state_weights,
    feature_rows_by_id,
    load_state_priority,
    read_csv,
    seed_everything,
    validate_no_split_overlap,
    visual_state_loss,
    write_csv,
    write_json,
)
from run_resnet18_five_class_coverage_comparison import evaluate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/openimages_v7_cua_vsl_v1.json"
DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "convnext_tiny": "ConvNeXt-Tiny",
    "maxvit_t": "MaxViT-T",
    "efficientnet_v2_s": "EfficientNetV2-S",
}


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


@torch.inference_mode()
def predict(model, loader, rows: list[dict], device: torch.device) -> list[dict]:
    model.eval()
    output_rows = [None] * len(rows)
    class_names = ("dog", "cat", "horse", "sheep", "person")
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True), include_states=False)["class_logits"]
        probabilities = logits.softmax(1).cpu().numpy()
        predicted = logits.argmax(1).cpu().numpy()
        labels = batch["label"].numpy()
        for index, label, prediction, probs in zip(
            batch["index"].tolist(), labels.tolist(), predicted.tolist(), probabilities.tolist()
        ):
            output_rows[index] = {
                **rows[index],
                "true_label": label,
                "predicted_label": prediction,
                "predicted_class": class_names[prediction],
                "correct": int(label == prediction),
                **{f"prob_{name}": probs[class_index] for class_index, name in enumerate(class_names)},
            }
    return output_rows


def slice_recall(predictions: list[dict], field: str, expected: str = "1") -> float | None:
    rows = [
        row for row in predictions
        if row["class_name"] == "dog" and row.get(field) == expected
    ]
    if not rows:
        return None
    return sum(int(row["predicted_class"] == "dog") for row in rows) / len(rows)


def area_recall(
    predictions: list[dict], upper: float, lower: float | None = None,
) -> float | None:
    rows = [
        row for row in predictions
        if row["class_name"] == "dog"
        and float(row["bbox_area_ratio"]) < upper
        and (lower is None or float(row["bbox_area_ratio"]) >= lower)
    ]
    if not rows:
        return None
    return sum(int(row["predicted_class"] == "dog") for row in rows) / len(rows)


def make_dataset(
    rows: list[dict], transform, schema: StateSchema, features: dict[str, dict],
    state_priority: dict[tuple[str, str], float] | None = None,
    state_weight_alpha: float = 0.0, state_weight_max: float = 3.0,
) -> StateAwareCropDataset:
    return StateAwareCropDataset(
        rows, transform, schema, features, state_priority,
        state_weight_alpha=state_weight_alpha, state_weight_max=state_weight_max,
    )


def run_one(
    model_name: str, condition_name: str, condition: dict, seed: int,
    train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], difficulty_rows: list[dict],
    schema: StateSchema, features: dict[str, dict], coverage_priority: dict, utility_priority: dict,
    output: Path, epochs: int, samples_per_epoch: int, batch_size: int, workers: int,
    lambda_state: float, sample_alpha: float, state_weight_alpha: float, max_multiplier: float,
    run_group: str,
) -> dict:
    run_dir = output / run_group / model_name / condition_name / str(seed)
    required = ("best.pt", "metrics.json", "train_log.csv", "test_predictions.csv")
    if all((run_dir / name).is_file() for name in required):
        print(f"[{model_name} {condition_name} seed={seed}] reusing completed run", flush=True)
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    sampling_mode = condition["sampling"]
    auxiliary_mode = condition["state_auxiliary"]
    sampling_priority = None
    if sampling_mode == "coverage":
        sampling_priority = coverage_priority
    elif sampling_mode == "utility":
        sampling_priority = utility_priority
    elif sampling_mode != "ordinary":
        raise ValueError(f"unknown sampling mode: {sampling_mode}")
    auxiliary_priority = None
    auxiliary_alpha = 0.0
    if auxiliary_mode == "coverage":
        auxiliary_priority = coverage_priority
        auxiliary_alpha = state_weight_alpha
    elif auxiliary_mode == "utility":
        auxiliary_priority = utility_priority
        auxiliary_alpha = state_weight_alpha
    elif auxiliary_mode == "uniform":
        auxiliary_priority = None
    elif auxiliary_mode != "none":
        raise ValueError(f"unknown auxiliary mode: {auxiliary_mode}")

    seed_everything(seed)
    train_transform, eval_transform = transforms_for_imagenet()
    train_dataset = make_dataset(
        train_rows, train_transform, schema, features,
        auxiliary_priority, auxiliary_alpha, max_multiplier,
    )
    eval_features = {}
    val_dataset = make_dataset(val_rows, eval_transform, schema, eval_features)
    test_dataset = make_dataset(test_rows, eval_transform, schema, eval_features)
    difficulty_dataset = make_dataset(difficulty_rows, eval_transform, schema, eval_features)
    weights = class_balanced_state_weights(
        train_rows, schema, features, sampling_priority,
        alpha=sample_alpha if sampling_priority is not None else 0.0,
        max_multiplier=max_multiplier,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True, generator=generator)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    difficulty_loader = DataLoader(
        difficulty_dataset, batch_size=128, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )

    device = torch.device("cuda")
    model = VisualStateMultiTaskModel(model_name, schema, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    use_auxiliary = auxiliary_mode != "none"
    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_sum = class_sum = state_sum = enum_sum = set_sum = 0.0
        seen = 0
        sampled_indices = Counter()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(images, include_states=use_auxiliary)
                class_loss = criterion(outputs["class_logits"], labels)
                if use_auxiliary:
                    state_loss, state_details = visual_state_loss(outputs, batch, schema)
                else:
                    state_loss = class_loss * 0.0
                    state_details = {"enum_loss": 0.0, "set_loss": 0.0}
                loss = class_loss + (lambda_state * state_loss if use_auxiliary else 0.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = len(labels)
            total_sum += loss.item() * count
            class_sum += class_loss.item() * count
            state_sum += state_loss.item() * count
            enum_sum += state_details["enum_loss"] * count
            set_sum += state_details["set_loss"] * count
            seen += count
            sampled_indices.update(batch["index"].tolist())
        scheduler.step()
        val_predictions = predict(model, val_loader, val_rows, device)
        val_metrics = evaluate(val_predictions)
        dog_exposures = [
            sampled_indices[index]
            for index, row in enumerate(train_rows) if row["class_name"] == "dog"
        ]
        history.append({
            "epoch": epoch,
            "train_total_loss": total_sum / seen,
            "train_class_loss": class_sum / seen,
            "train_state_loss": state_sum / seen,
            "train_enum_loss": enum_sum / seen,
            "train_set_loss": set_sum / seen,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "dog_exposure_min": min(dog_exposures) if dog_exposures else 0,
            "dog_exposure_max": max(dog_exposures) if dog_exposures else 0,
            "dog_exposure_mean": float(np.mean(dog_exposures)) if dog_exposures else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[{model_name} {condition_name} seed={seed}] {epoch}/{epochs} "
            f"class={class_sum/seen:.4f} state={state_sum/seen:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no best state was selected")
    model.load_state_dict(best_state)
    test_predictions = predict(model, test_loader, test_rows, device)
    difficulty_predictions = predict(model, difficulty_loader, difficulty_rows, device)
    metrics = evaluate(test_predictions)
    metrics.update({
        "difficulty_dog_recall": sum(
            int(row["predicted_class"] == "dog") for row in difficulty_predictions
        ) / len(difficulty_predictions),
        "difficulty_under3pct_dog_recall": area_recall(difficulty_predictions, 0.03),
        "difficulty_small_dog_recall": area_recall(difficulty_predictions, 0.03, 0.01),
        "difficulty_tiny_dog_recall": area_recall(difficulty_predictions, 0.01),
        "difficulty_occluded_dog_recall": slice_recall(difficulty_predictions, "is_occluded"),
        "difficulty_truncated_dog_recall": slice_recall(difficulty_predictions, "is_truncated"),
        "model": model_name,
        "condition": condition_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "lambda_state": lambda_state,
        "sample_alpha": sample_alpha,
        "state_weight_alpha": state_weight_alpha,
        "max_multiplier": max_multiplier,
        "sampling_mode": sampling_mode,
        "auxiliary_mode": auxiliary_mode,
        "elapsed_seconds": time.perf_counter() - started,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state, "model_name": model_name, "condition": condition_name,
            "seed": seed, "metrics": metrics, "schema_mapping": schema.mapping(),
        },
        run_dir / "best.pt",
    )
    write_csv(run_dir / "train_log.csv", history)
    write_csv(run_dir / "test_predictions.csv", test_predictions)
    write_csv(run_dir / "difficulty_test_predictions.csv", difficulty_predictions)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def parse_list(value: str | None, default: list[str | int]) -> list:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models")
    parser.add_argument("--conditions")
    parser.add_argument("--seeds")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.smoke == args.formal:
        raise SystemExit("choose exactly one of --smoke or --formal")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    hyper = config["smoke_only_hyperparameters"]
    if args.formal and hyper["status"] != "frozen_for_formal_training":
        raise SystemExit(
            "formal hyperparameters are not frozen; discuss smoke results and update config status first"
        )
    output = Path(config["output"])
    data = Path(config["data"])
    schema = StateSchema.from_path(Path(config["schema"]))
    state_priority_path = output / "state_priority" / "state_priority.csv"
    if not state_priority_path.is_file():
        raise FileNotFoundError("OOF state priority is missing")
    coverage_priority = load_state_priority(state_priority_path, "coverage_priority_norm")
    utility_priority = load_state_priority(state_priority_path, "utility_priority_norm")
    features = {
        **feature_rows_by_id(Path(config["base_features"])),
        **feature_rows_by_id(Path(config["candidate_features"])),
    }
    val_rows = read_csv(data / "manifests/validation.csv")
    test_rows = read_csv(data / "manifests/test.csv")
    difficulty_rows = read_csv(data / "manifests/dog_difficulty_test.csv")
    models = parse_list(args.models, ["resnet18"] if args.smoke else config["models"])
    conditions = parse_list(args.conditions, ["E0", "E5", "E8"] if args.smoke else list(config["conditions"]))
    seeds = [int(value) for value in parse_list(
        args.seeds, [config["training_seeds_minimum"][0]] if args.smoke else config["training_seeds_minimum"]
    )]
    epochs = 2 if args.smoke else int(config["epochs"])
    samples_per_epoch = 512 if args.smoke else int(config["samples_per_epoch"])
    run_group = "smoke_runs" if args.smoke else "runs"

    all_metrics = defaultdict(lambda: defaultdict(list))
    manifest_hashes = {}
    for condition_name in conditions:
        condition = config["conditions"][condition_name]
        manifest_path = output / "manifests" / f"train_{condition['data']}.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"selection manifest is missing: {manifest_path}")
        train_rows = read_csv(manifest_path)
        validate_no_split_overlap({
            "train": train_rows, "validation": val_rows,
            "test": test_rows, "difficulty": difficulty_rows,
        })
        train_dog_ids = {row["sample_id"] for row in train_rows if row["class_name"] == "dog"}
        missing_features = train_dog_ids - set(features)
        if missing_features:
            raise ValueError(f"missing dog features: {sorted(missing_features)[:3]}")
        manifest_hashes[condition_name] = sha256(manifest_path)
        for model_name in models:
            if model_name not in DISPLAY_NAMES:
                raise ValueError(f"unsupported model: {model_name}")
            for seed in seeds:
                metrics = run_one(
                    model_name, condition_name, condition, seed,
                    train_rows, val_rows, test_rows, difficulty_rows,
                    schema, features, coverage_priority, utility_priority,
                    output, epochs, samples_per_epoch, int(config["batch_size"]), args.workers,
                    float(hyper["lambda_state"]), float(hyper["sample_alpha"]),
                    float(hyper["state_weight_alpha"]), float(hyper["max_multiplier"]),
                    run_group,
                )
                all_metrics[model_name][condition_name].append(metrics)

    rows = []
    for model_name in models:
        for condition_name in conditions:
            for metrics in all_metrics[model_name][condition_name]:
                rows.append(metrics)
    table_dir = output / "tables"
    table_name = "smoke_results.csv" if args.smoke else "results_by_seed.csv"
    write_csv(table_dir / table_name, rows)
    protocol = {
        "experiment_id": config["experiment_id"],
        "run_group": run_group,
        "models": models,
        "conditions": conditions,
        "seeds": seeds,
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "hyperparameters": hyper,
        "manifest_sha256": manifest_hashes,
        "state_priority_sha256": sha256(state_priority_path),
    }
    write_json(output / f"{run_group}_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
