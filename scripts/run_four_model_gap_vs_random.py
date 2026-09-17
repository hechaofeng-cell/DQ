#!/usr/bin/env python3
"""Run four ImageNet-pretrained classifiers on gap-100 and random-100 COCO5 datasets."""
from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from run_resnet18_dog_coverage_comparison import ROOT, read_csv, sha256, write_csv, write_json
from run_resnet18_five_class_coverage_comparison import (
    CLASS_SPECS,
    CropDataset,
    SEEDS,
    evaluate,
    predict,
    seed_everything,
)


DATA_ROOT = ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916"
OUTPUT = ROOT / "artifacts/four_models_coco5_gap_vs_random_20260917"
CONDITIONS = ("gap596", "random596")
MODEL_NAMES = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-T",
    "regnet_y_3_2gf": "RegNetY-3.2GF",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "maxvit_t": "MaxVit-T",
    "vit_b_16": "ViT-B/16",
    "resnext50_32x4d": "ResNeXt-50 32x4d",
    "convnext_small": "ConvNeXt-Small",
    "swin_v2_t": "SwinV2-T",
    "regnet_x_3_2gf": "RegNetX-3.2GF",
}


def build_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_SPECS))
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_SPECS))
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_SPECS))
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, len(CLASS_SPECS))
    elif model_name == "regnet_y_3_2gf":
        model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_SPECS))
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "maxvit_t":
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_SPECS))
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_SPECS))
    elif model_name == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_SPECS))
    elif model_name == "swin_v2_t":
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, len(CLASS_SPECS))
    elif model_name == "regnet_x_3_2gf":
        model = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_SPECS))
    else:
        raise ValueError(f"unknown model: {model_name}")
    return model


def model_profile(model_name):
    model = build_model(model_name)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
    return {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_memory_mib_fp32": state_bytes / 1024 / 1024,
        "input_size": "224x224",
        "pretraining": "ImageNet-1K",
    }


def train_one(model_name, condition, seed, train_rows, val_rows, test_rows, epochs=20, workers=12):
    run_dir = OUTPUT / "runs" / model_name / condition / str(seed)
    required = ("best.pt", "metrics.json", "test_predictions.csv", "train_log.csv")
    if all((run_dir / filename).is_file() for filename in required):
        print(f"[{model_name} {condition} seed={seed}] reusing completed run", flush=True)
        return json.loads((run_dir / "metrics.json").read_text())

    seed_everything(seed)
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(),
        normalization,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization,
    ])
    class_counts = Counter(int(row["label"]) for row in train_rows)
    sample_weights = [1 / class_counts[int(row["label"])] for row in train_rows]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=3000, replacement=True, generator=generator)
    train_loader = DataLoader(
        CropDataset(train_rows, train_transform), batch_size=64, sampler=sampler,
        num_workers=workers, pin_memory=True, persistent_workers=True, generator=generator,
    )
    val_loader = DataLoader(
        CropDataset(val_rows, eval_transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        CropDataset(test_rows, eval_transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=True,
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
        running_loss, correct, seen = 0.0, 0, 0
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
            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        scheduler.step()
        val_predictions = predict(model, val_loader, val_rows, device)
        val_metrics = evaluate(val_predictions)
        history.append({
            "epoch": epoch, "train_loss": running_loss / seen, "train_accuracy": correct / seen,
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(f"[{model_name} {condition} seed={seed}] epoch {epoch}/{epochs} "
              f"loss={running_loss/seen:.4f} val_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_predictions = predict(model, test_loader, test_rows, device)
    metrics = evaluate(test_predictions)
    metrics.update({
        "model": model_name, "condition": condition, "seed": seed,
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "train_counts": dict(class_counts), "samples_per_epoch": 3000,
        "elapsed_seconds": time.perf_counter() - started,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "model_name": model_name, "condition": condition,
                "seed": seed, "metrics": metrics}, run_dir / "best.pt")
    write_csv(run_dir / "train_log.csv", history)
    write_csv(run_dir / "test_predictions.csv", test_predictions)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def import_resnet18_metrics():
    source_condition = {"gap596": "dog596", "random596": "dog_random596"}
    imported = defaultdict(list)
    for condition in CONDITIONS:
        for seed in SEEDS:
            source = DATA_ROOT / source_condition[condition] / str(seed) / "metrics.json"
            metrics = json.loads(source.read_text())
            metrics["model"] = "resnet18"
            metrics["condition"] = condition
            metrics["source_metrics_path"] = str(source.resolve())
            imported[condition].append(metrics)
    return imported


def aggregate(all_metrics, profiles):
    metric_names = [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "dog_precision", "dog_recall", "dog_f1", "small_dog_recall", "tiny_dog_recall",
    ]
    summary = []
    deltas = []
    seed_rows = []
    for model_name in MODEL_NAMES:
        for condition in CONDITIONS:
            runs = all_metrics[model_name][condition]
            row = {"model": model_name, "display_name": DISPLAY_NAMES[model_name],
                   "condition": condition, "runs": len(runs)}
            for metric in metric_names:
                values = [run[metric] for run in runs if run[metric] is not None]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row["best_epoch_mean"] = float(np.mean([run["best_epoch"] for run in runs]))
            elapsed = [run.get("elapsed_seconds") for run in runs if run.get("elapsed_seconds") is not None]
            row["elapsed_seconds_mean"] = float(np.mean(elapsed)) if elapsed else None
            summary.append(row)
            for run in runs:
                seed_rows.append({"model": model_name, "condition": condition, "seed": run["seed"],
                                  **{metric: run[metric] for metric in metric_names},
                                  "best_epoch": run["best_epoch"],
                                  "elapsed_seconds": run.get("elapsed_seconds", "")})
        for metric in metric_names:
            gap = all_metrics[model_name]["gap596"]
            random_runs = all_metrics[model_name]["random596"]
            paired = [gap[index][metric] - random_runs[index][metric] for index in range(len(SEEDS))]
            deltas.append({
                "model": model_name, "display_name": DISPLAY_NAMES[model_name], "metric": metric,
                "gap_minus_random_mean": float(np.mean(paired)),
                "gap_minus_random_std": float(np.std(paired, ddof=1)),
                "positive_seed_count": sum(value > 0 for value in paired),
                "zero_seed_count": sum(value == 0 for value in paired),
                "negative_seed_count": sum(value < 0 for value in paired),
                "seed_deltas": json.dumps(paired),
            })
    write_csv(OUTPUT / "tables/model_parameters.csv", profiles)
    write_csv(OUTPUT / "tables/results_by_condition.csv", summary)
    write_csv(OUTPUT / "tables/gap_minus_random.csv", deltas)
    write_csv(OUTPUT / "tables/results_by_seed.csv", seed_rows)
    return summary, deltas


def summarize_conclusion(deltas):
    model_count = len(MODEL_NAMES)
    delta_rows = {(row["model"], row["metric"]): row for row in deltas}
    positive_accuracy = [model for model in MODEL_NAMES
                         if delta_rows[model, "accuracy"]["gap_minus_random_mean"] > 0]
    positive_macro_f1 = [model for model in MODEL_NAMES
                         if delta_rows[model, "macro_f1"]["gap_minus_random_mean"] > 0]
    positive_dog_f1 = [model for model in MODEL_NAMES
                       if delta_rows[model, "dog_f1"]["gap_minus_random_mean"] > 0]
    consistent_accuracy = [DISPLAY_NAMES[model] for model in MODEL_NAMES
                           if delta_rows[model, "accuracy"]["positive_seed_count"] == len(SEEDS)]
    consistent_dog_f1 = [DISPLAY_NAMES[model] for model in MODEL_NAMES
                         if delta_rows[model, "dog_f1"]["positive_seed_count"] == len(SEEDS)]
    consistent_accuracy_text = "、".join(consistent_accuracy) if consistent_accuracy else "没有模型"
    consistent_dog_f1_text = "、".join(consistent_dog_f1) if consistent_dog_f1 else "没有模型"
    return (
        f"缺口组相对随机组在{len(positive_accuracy)}/{model_count}个模型上提高Accuracy、"
        f"在{len(positive_macro_f1)}/{model_count}个模型上提高Macro-F1，并在"
        f"{len(positive_dog_f1)}/{model_count}个模型上提高dog F1。"
        f"其中，{consistent_accuracy_text}的Accuracy在{len(SEEDS)}个训练种子中均胜出；"
        f"{consistent_dog_f1_text}的dog F1在{len(SEEDS)}个种子中均胜出。"
        "结果整体偏向缺口补图，却仍不是所有架构、所有指标一致提升；"
        "在只有一个random100清单的条件下，不能据此宣称普遍优于随机补图。"
    )


def make_report(summary, deltas, profiles):
    rows = {(row["model"], row["condition"]): row for row in summary}
    delta_rows = {(row["model"], row["metric"]): row for row in deltas}
    model_count = len(MODEL_NAMES)
    lines = [
        f"# {model_count}模型：缺口100与随机100对照实验", "", "## 实验口径", "",
        f"{model_count}个ImageNet预训练模型在同一COCO五分类目标区域数据集上训练。固定其他四类、验证集、测试集和训练超参数，仅比较dog496+缺口100与dog496+随机100。每个条件运行{len(SEEDS)}个相同训练种子。", "",
        "## 核心结论", "",
        summarize_conclusion(deltas), "",
        "## 数据覆盖率", "",
        "| 条件 | dog数量 | dog_v3.1覆盖率 |", "|---|---:|---:|",
        "| random596 | 596 | 81.33% |", "| gap596 | 596 | 84.17% |", "",
        "## 模型参数", "",
        "| 模型 | 总参数量 | 可训练参数量 | FP32参数内存 | 输入 |", "|---|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        lines.append(f"| {profile['display_name']} | {profile['total_parameters']:,} | "
                     f"{profile['trainable_parameters']:,} | {profile['parameter_memory_mib_fp32']:.2f} MiB | "
                     f"{profile['input_size']} |")
    lines += ["", "## 主要结果", "",
              "数值为3个训练种子的均值±样本标准差；差值为gap596-random596。", "",
              "| 模型 | 条件 | Accuracy | Macro-F1 | dog Precision | dog Recall | dog F1 |", 
              "|---|---|---:|---:|---:|---:|---:|"]
    for model_name in MODEL_NAMES:
        for condition in CONDITIONS:
            row = rows[model_name, condition]
            values = []
            for metric in ("accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1"):
                values.append(f"{row[metric + '_mean']:.2%} ± {row[metric + '_std']:.2%}")
            lines.append(f"| {DISPLAY_NAMES[model_name]} | {condition} | " + " | ".join(values) + " |")
    lines += ["", "## 缺口组相对随机组", "",
              "| 模型 | ΔAccuracy | ΔMacro-F1 | Δdog Precision | Δdog Recall | Δdog F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for model_name in MODEL_NAMES:
        values = [delta_rows[model_name, metric]["gap_minus_random_mean"]
                  for metric in ("accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1")]
        lines.append(f"| {DISPLAY_NAMES[model_name]} | " + " | ".join(f"{value:+.2%}" for value in values) + " |")
    lines += ["", "## 缺口组胜出种子数", "",
              "分母为3；仅表示训练种子方向，不是统计显著性。", "",
              "| 模型 | Accuracy | Macro-F1 | dog Precision | dog Recall | dog F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for model_name in MODEL_NAMES:
        wins = [delta_rows[model_name, metric]["positive_seed_count"]
                for metric in ("accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1")]
        lines.append(f"| {DISPLAY_NAMES[model_name]} | " + " | ".join(f"{value}/3" for value in wins) + " |")
    lines += ["", "## 困难目标", "",
              "| 模型 | random小目标Recall | gap小目标Recall | 差值 | random极小目标Recall | gap极小目标Recall | 差值 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model_name in MODEL_NAMES:
        random_row, gap_row = rows[model_name, "random596"], rows[model_name, "gap596"]
        lines.append(f"| {DISPLAY_NAMES[model_name]} | {random_row['small_dog_recall_mean']:.2%} | "
                     f"{gap_row['small_dog_recall_mean']:.2%} | "
                     f"{gap_row['small_dog_recall_mean'] - random_row['small_dog_recall_mean']:+.2%} | "
                     f"{random_row['tiny_dog_recall_mean']:.2%} | {gap_row['tiny_dog_recall_mean']:.2%} | "
                     f"{gap_row['tiny_dog_recall_mean'] - random_row['tiny_dog_recall_mean']:+.2%} |")
    lines += ["", "## 固定训练参数", "",
              "| 参数 | 设置 |", "|---|---|",
              "| 预训练 | ImageNet-1K |", "| 输入尺寸 | 224x224 |", "| 优化器 | AdamW |",
              "| 初始学习率 | 1e-4 |", "| Weight decay | 1e-4 |", "| Batch size | 64 |",
              "| Epoch | 20 |", "| 每轮采样量 | 3000，类别均衡有放回采样 |",
              "| 模型选择 | 固定验证集Macro-F1最高轮 |",
              f"| 训练种子 | {'、'.join(str(seed) for seed in SEEDS)} |",
              "| 固定测试集 | 250个目标，每类50个 |", "", "## 结论边界", "",
              "- 比较评价目标区域五分类，不是目标检测。",
              "- 只有一个固定random100数据清单；3个训练种子不能替代多个随机选样清单。",
              "- 小目标dog测试样本12个、极小目标7个，困难目标结果只能作为趋势。", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_checksums():
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "checksums.csv":
            rows.append({"path": str(path.relative_to(OUTPUT)), "sha256": sha256(path),
                         "bytes": path.stat().st_size})
    write_csv(OUTPUT / "checksums.csv", rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = [model_profile(model_name) for model_name in MODEL_NAMES]
    write_json(OUTPUT / "protocol.json", {
        "models": list(MODEL_NAMES), "conditions_in_run_order": list(CONDITIONS),
        "classes": [name for name, _ in CLASS_SPECS], "seeds": list(SEEDS),
        "epochs": 20, "samples_per_epoch": 3000, "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "selection_metric": "validation macro-F1", "input_size": 224,
        "data_root": str(DATA_ROOT.resolve()),
        "gap_dog_manifest": str((DATA_ROOT / "data/train_dog596.csv").resolve()),
        "random_dog_manifest": str((DATA_ROOT / "data/train_dog_random596.csv").resolve()),
        "other_train_manifest": str((DATA_ROOT / "data/train_other_fixed.csv").resolve()),
        "eval_manifest": str((DATA_ROOT / "data/eval_fixed.csv").resolve()),
    })
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        "- Question: Does gap-selected dog100 outperform random dog100 across eight model architectures?\n"
        "- Primary metrics: Macro-F1 and dog F1; secondary: Accuracy, dog Precision/Recall.\n"
        "- Fixed factors: COCO5 manifests, transforms, optimizer, epochs, sampling budget, seeds, val/test.\n"
        "- Command: `.venv/bin/python -u scripts/run_four_model_gap_vs_random.py`\n"
        f"- Raw output: `{OUTPUT}`\n"
        "- Claim boundary: one random100 subset; cross-model consistency does not measure selection-set variance.\n",
        encoding="utf-8",
    )

    gap_dogs = read_csv(DATA_ROOT / "data/train_dog596.csv")
    random_dogs = read_csv(DATA_ROOT / "data/train_dog_random596.csv")
    other_rows = read_csv(DATA_ROOT / "data/train_other_fixed.csv")
    eval_rows = read_csv(DATA_ROOT / "data/eval_fixed.csv")
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    condition_rows = {"gap596": gap_dogs + other_rows, "random596": random_dogs + other_rows}

    all_metrics = defaultdict(lambda: defaultdict(list))
    imported = import_resnet18_metrics()
    for condition in CONDITIONS:
        all_metrics["resnet18"][condition] = imported[condition]

    for model_name in MODEL_NAMES[1:]:
        for condition in CONDITIONS:
            for seed in SEEDS:
                metrics = train_one(model_name, condition, seed, condition_rows[condition],
                                    val_rows, test_rows)
                all_metrics[model_name][condition].append(metrics)

    summary, deltas = aggregate(all_metrics, profiles)
    make_report(summary, deltas, profiles)
    write_json(OUTPUT / "report.json", {
        "coverage": {"random596": 0.8132791327913279, "gap596": 0.8417344173441734,
                     "gap_minus_random": 0.0284552845528455},
        "profiles": profiles, "summary": summary, "deltas": deltas,
        "conclusion": summarize_conclusion(deltas),
        "limitations": ["one random100 subset", "three training seeds", "oracle target crops"],
    })
    write_checksums()
    print((OUTPUT / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
