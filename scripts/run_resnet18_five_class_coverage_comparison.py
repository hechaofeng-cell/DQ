#!/usr/bin/env python3
"""Compare clean dog496 and coverage-improved dog596 in a fixed COCO five-class host dataset."""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from run_resnet18_dog_coverage_comparison import (
    ROOT,
    crop_target,
    download_one,
    parse_labels,
    read_csv,
    sha256,
    write_csv,
    write_json,
)


CLASS_SPECS = (
    ("dog", 16),
    ("cat", 15),
    ("horse", 17),
    ("sheep", 18),
    ("person", 0),
)
NAME_TO_LABEL = {name: index for index, (name, _) in enumerate(CLASS_SPECS)}
YOLO_TO_NAME = {class_id: name for name, class_id in CLASS_SPECS}
SEEDS = (20260916, 20260917, 20260918)
OTHER_TRAIN_PER_CLASS = 596
VAL_PER_CLASS = 15
TEST_PER_CLASS = 50


def largest_target(lines, class_id):
    candidates = [line for line in lines if len(line) == 5 and int(line[0]) == class_id]
    return max(candidates, key=lambda x: float(x[3]) * float(x[4])) if candidates else None


def label_candidates(split):
    selected_ids = {class_id for _, class_id in CLASS_SPECS}
    candidates = defaultdict(list)
    archive_path = ROOT / "data/coco2017/coco2017labels.zip"
    prefix = f"coco/labels/{split}/"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith(".txt"):
                continue
            lines = parse_labels(archive.read(name))
            present = {int(line[0]) for line in lines if len(line) == 5}
            image_id = Path(name).stem
            for class_id in selected_ids & present:
                candidates[class_id].append({
                    "image_id": image_id,
                    "target": largest_target(lines, class_id),
                    "contains_dog": 16 in present,
                })
    return candidates


def choose_unique(candidates, counts, forbidden, seed, reject_dog_for_non_dog):
    rng = random.Random(seed)
    used = set(forbidden)
    chosen = {}
    # Allocate the rare classes first so co-occurring images do not starve them.
    for class_id in sorted(counts, key=lambda cid: len(candidates[cid])):
        pool = [row for row in candidates[class_id]
                if row["image_id"] not in used
                and not (reject_dog_for_non_dog and class_id != 16 and row["contains_dog"])]
        rng.shuffle(pool)
        count = counts[class_id]
        if len(pool) < count:
            raise ValueError(f"class {class_id}: need {count}, only {len(pool)} available")
        chosen[class_id] = pool[:count]
        used.update(row["image_id"] for row in chosen[class_id])
    return chosen


def download_images(rows, split, image_dir, workers):
    image_dir.mkdir(parents=True, exist_ok=True)
    image_ids = sorted({row["image_id"] for row in rows})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, iid, split, image_dir): iid for iid in image_ids}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 100 == 0 or index == len(image_ids):
                print(f"[prepare {split}] downloaded {index}/{len(image_ids)}", flush=True)


def prepare_dog_rows():
    baseline = read_csv(ROOT / "artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv")
    candidate = read_csv(ROOT / "artifacts/dog_candidate400_v2_1_inputs_20260912/target_manifest.csv")
    clean = read_csv(ROOT / "artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv")
    clean_by_id = {row["image_id"]: row for row in clean}
    targets = {row["image_id"]: row for row in baseline + candidate}
    baseline_ids = {row["image_id"] for row in baseline}
    clean_baseline_ids = sorted(baseline_ids & set(clean_by_id))
    clean_all_ids = sorted(clean_by_id)
    if len(clean_baseline_ids) != 496 or len(clean_all_ids) != 596:
        raise ValueError("expected clean dog496 and dog596")

    def build(ids, source_name):
        rows = []
        for iid in ids:
            source = "baseline496" if iid in baseline_ids else "supplement100"
            rows.append({
                "sample_id": f"dog_{iid}", "image_id": iid,
                "crop_path": targets[iid]["crop_image_path"],
                "label": NAME_TO_LABEL["dog"], "class_name": "dog", "yolo_class_id": 16,
                "source": source if source_name == "dog596" else "baseline496",
                "bbox_area_ratio": clean_by_id[iid]["target_area_ratio"],
            })
        return rows

    universe = {row["image_id"] for row in baseline + candidate}
    return build(clean_baseline_ids, "dog496"), build(clean_all_ids, "dog596"), universe


def crop_selected(chosen, split, image_dir, crop_dir):
    output = []
    for class_id, rows in chosen.items():
        class_name = YOLO_TO_NAME[class_id]
        for row in rows:
            iid = row["image_id"]
            crop_path = crop_dir / class_name / f"{iid}.jpg"
            if crop_path.exists():
                details = {"bbox_area_ratio": float(row["target"][3]) * float(row["target"][4])}
            else:
                details = crop_target(image_dir / f"{iid}.jpg", row["target"], crop_path)
            output.append({
                "sample_id": f"{class_name}_{iid}", "image_id": iid,
                "crop_path": str(crop_path.resolve()), "label": NAME_TO_LABEL[class_name],
                "class_name": class_name, "yolo_class_id": class_id,
                "source": split, "bbox_area_ratio": details["bbox_area_ratio"],
            })
    return output


def prepare_data(output, workers):
    dog496, dog596, dog_universe = prepare_dog_rows()
    train_candidates = label_candidates("train2017")
    other_counts = {class_id: OTHER_TRAIN_PER_CLASS for name, class_id in CLASS_SPECS if name != "dog"}
    chosen_train = choose_unique(train_candidates, other_counts, dog_universe, 20260916, True)
    flat_train = [row for rows in chosen_train.values() for row in rows]
    train_images = output / "data/train_images"
    download_images(flat_train, "train2017", train_images, workers)
    other_train = crop_selected(chosen_train, "train2017", train_images, output / "data/train_crops")

    eval_candidates = label_candidates("val2017")
    eval_counts = {class_id: VAL_PER_CLASS + TEST_PER_CLASS for _, class_id in CLASS_SPECS}
    chosen_eval = choose_unique(eval_candidates, eval_counts, set(), 20260917, False)
    flat_eval = [row for rows in chosen_eval.values() for row in rows]
    eval_images = output / "data/eval_images"
    download_images(flat_eval, "val2017", eval_images, workers)
    eval_rows = crop_selected(chosen_eval, "val2017", eval_images, output / "data/eval_crops")
    by_class_seen = Counter()
    for row in eval_rows:
        index = by_class_seen[row["class_name"]]
        row["split"] = "val" if index < VAL_PER_CLASS else "test"
        by_class_seen[row["class_name"]] += 1

    fields = ["sample_id", "image_id", "crop_path", "label", "class_name", "yolo_class_id",
              "source", "bbox_area_ratio"]
    write_csv(output / "data/train_dog496.csv", dog496, fields)
    write_csv(output / "data/train_dog596.csv", dog596, fields)
    write_csv(output / "data/train_other_fixed.csv", other_train, fields)
    write_csv(output / "data/eval_fixed.csv", eval_rows, fields + ["split"])
    return dog496, dog596, other_train, eval_rows


class CropDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.rows[index]["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.rows[index]["label"]), index


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def predict(model, loader, rows, device):
    model.eval()
    output = [None] * len(rows)
    for images, labels, indices in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities = logits.softmax(1).cpu().numpy()
        predicted = logits.argmax(1).cpu().numpy()
        for idx, label, pred, probs in zip(indices.tolist(), labels.tolist(), predicted.tolist(), probabilities.tolist()):
            output[idx] = {
                **rows[idx], "true_label": label, "predicted_label": pred,
                "predicted_class": CLASS_SPECS[pred][0], "correct": int(label == pred),
                **{f"prob_{name}": probs[class_index] for class_index, (name, _) in enumerate(CLASS_SPECS)},
            }
    return output


def evaluate(predictions):
    labels = np.asarray([int(row["true_label"]) for row in predictions])
    predicted = np.asarray([int(row["predicted_label"]) for row in predictions])
    class_labels = list(range(len(CLASS_SPECS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=class_labels, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=class_labels).tolist(),
    }
    for index, (name, _) in enumerate(CLASS_SPECS):
        metrics[f"{name}_precision"] = float(precision[index])
        metrics[f"{name}_recall"] = float(recall[index])
        metrics[f"{name}_f1"] = float(f1[index])
        metrics[f"{name}_support"] = int(support[index])
    areas = np.asarray([float(row["bbox_area_ratio"]) for row in predictions])
    dog_mask = labels == NAME_TO_LABEL["dog"]
    for name, threshold in (("small_dog", 0.03), ("tiny_dog", 0.01)):
        mask = dog_mask & (areas < threshold)
        metrics[f"{name}_support"] = int(mask.sum())
        metrics[f"{name}_recall"] = float(np.mean(predicted[mask] == NAME_TO_LABEL["dog"])) if mask.any() else None
    return metrics


def train_one(condition, seed, train_rows, val_rows, test_rows, output, epochs, workers):
    run_dir = output / condition / str(seed)
    if all((run_dir / name).exists() for name in ("best.pt", "metrics.json", "test_predictions.csv")):
        print(f"[{condition} seed={seed}] reusing completed run", flush=True)
        return json.loads((run_dir / "metrics.json").read_text()), read_csv(run_dir / "test_predictions.csv")
    seed_everything(seed)
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1), transforms.ToTensor(), normalization,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization,
    ])
    class_counts = Counter(int(row["label"]) for row in train_rows)
    weights = [1 / class_counts[int(row["label"])] for row in train_rows]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=3000, replacement=True, generator=generator)
    loader_workers = min(workers, 12)
    train_loader = DataLoader(CropDataset(train_rows, train_transform), batch_size=64, sampler=sampler,
                              num_workers=loader_workers, pin_memory=True, persistent_workers=True,
                              generator=generator)
    val_loader = DataLoader(CropDataset(val_rows, eval_transform), batch_size=128, shuffle=False,
                            num_workers=loader_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(CropDataset(test_rows, eval_transform), batch_size=128, shuffle=False,
                             num_workers=loader_workers, pin_memory=True, persistent_workers=True)
    device = torch.device("cuda")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_SPECS))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
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
        history.append({"epoch": epoch, "train_loss": running_loss / seen,
                        "train_accuracy": correct / seen, "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"], "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"[{condition} seed={seed}] epoch {epoch}/{epochs} loss={running_loss/seen:.4f} "
              f"val_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_predictions = predict(model, test_loader, test_rows, device)
    metrics = evaluate(test_predictions)
    metrics.update({"condition": condition, "seed": seed, "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1, "train_counts": dict(class_counts),
                    "samples_per_epoch": 3000})
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "condition": condition, "seed": seed, "metrics": metrics}, run_dir / "best.pt")
    write_csv(run_dir / "train_log.csv", history)
    write_csv(run_dir / "test_predictions.csv", test_predictions)
    write_json(run_dir / "metrics.json", metrics)
    return metrics, test_predictions


def summarize(all_metrics, output):
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    for name, _ in CLASS_SPECS:
        metric_names += [f"{name}_precision", f"{name}_recall", f"{name}_f1"]
    metric_names += ["small_dog_recall", "tiny_dog_recall"]
    summary = []
    for condition in ("dog496", "dog596"):
        row = {"condition": condition, "runs": len(all_metrics[condition])}
        for metric in metric_names:
            values = [run[metric] for run in all_metrics[condition] if run[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    deltas = []
    for metric in metric_names:
        paired = [all_metrics["dog596"][i][metric] - all_metrics["dog496"][i][metric]
                  for i in range(len(SEEDS))]
        deltas.append({"metric": metric, "delta_mean": float(np.mean(paired)),
                       "delta_std": float(np.std(paired, ddof=1)), "seed_deltas": json.dumps(paired)})
    write_csv(output / "comparison/summary_by_condition.csv", summary)
    write_csv(output / "comparison/deltas.csv", deltas)
    report = {
        "experiment": "ImageNet-pretrained ResNet-18 five-class target-region classification",
        "classes": [name for name, _ in CLASS_SPECS],
        "coverage": {"dog496": 0.7864, "dog596": 0.8417, "delta": 0.0553},
        "conditions": summary,
        "deltas": deltas,
        "limitations": [
            "Target-region classification with oracle COCO boxes, not whole-image detection.",
            "The comparison tests adding 100 gap-selected dog targets; no equal-budget random-dog control.",
        ],
    }
    write_json(output / "comparison/report.json", report)
    lines = [
        "# ResNet-18五分类：dog496与dog596比较", "", "## 实验设置", "",
        "类别：dog、cat、horse、sheep、person；COCO目标框裁剪；固定其他四类、验证集和测试集；3个随机种子。", "",
        "dog覆盖率：78.64% -> 84.17%（+5.53个百分点）。", "", "## 结果", "",
        "| 指标 | dog496（均值±标准差） | dog596（均值±标准差） | 差值 |", "|---|---:|---:|---:|",
    ]
    display_metrics = ["accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1",
                       "cat_recall", "horse_recall", "sheep_recall", "person_recall",
                       "small_dog_recall", "tiny_dog_recall"]
    for metric in display_metrics:
        a, b = summary[0], summary[1]
        delta = next(row["delta_mean"] for row in deltas if row["metric"] == metric)
        lines.append(f"| {metric} | {a[metric + '_mean']:.4f} ± {a[metric + '_std']:.4f} | "
                     f"{b[metric + '_mean']:.4f} ± {b[metric + '_std']:.4f} | {delta:+.4f} |")
    lines += ["", "## 结论边界", "", "- 结果评价目标区域五分类，不是目标检测。",
              "- 没有等预算随机dog补图对照，因此不能把差值单独归因于覆盖驱动选择。", ""]
    (output / "comparison/report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/resnet18_coco5_dog496_vs_dog596_20260916")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": "torchvision resnet18 ImageNet pretrained", "task": "five_class_target_crop",
        "classes": [name for name, _ in CLASS_SPECS], "conditions": ["dog496", "dog596"],
        "seeds": list(SEEDS), "epochs": args.epochs, "samples_per_epoch": 3000, "batch_size": 64,
        "optimizer": "AdamW(lr=1e-4,weight_decay=1e-4)", "selection_metric": "validation macro-F1",
        "other_train_per_class": OTHER_TRAIN_PER_CLASS, "validation_per_class": VAL_PER_CLASS,
        "test_per_class": TEST_PER_CLASS, "source": "COCO 2017",
    }
    write_json(args.output / "protocol.json", protocol)
    dog496, dog596, other, eval_rows = prepare_data(args.output, args.workers)
    if args.prepare_only:
        print(json.dumps({"dog496": len(dog496), "dog596": len(dog596), "other": len(other),
                          "val": sum(r["split"] == "val" for r in eval_rows),
                          "test": sum(r["split"] == "test" for r in eval_rows)}, indent=2))
        return
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    all_metrics = defaultdict(list)
    for condition, dog_rows in (("dog496", dog496), ("dog596", dog596)):
        train_rows = dog_rows + other
        for seed in SEEDS:
            metrics, _ = train_one(condition, seed, train_rows, val_rows, test_rows,
                                   args.output, args.epochs, args.workers)
            all_metrics[condition].append(metrics)
    report = summarize(all_metrics, args.output)
    checksums = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "checksums.csv":
            checksums.append({"path": str(path.relative_to(args.output)), "sha256": sha256(path),
                              "bytes": path.stat().st_size})
    write_csv(args.output / "checksums.csv", checksums)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
