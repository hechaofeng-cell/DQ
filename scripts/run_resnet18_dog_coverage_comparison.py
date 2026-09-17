#!/usr/bin/env python3
"""Compare ImageNet-pretrained ResNet-18 on dog500 versus clean dog596."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
DOG_ID = 16
HARD_NEGATIVE_IDS = (0, 14, 15, 17, 18, 19, 21, 22, 23, 77)
SEEDS = (20260916, 20260917, 20260918)
EXCLUDED_NONLIVING = {"000000140444", "000000246880", "000000446990", "000000471513"}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_labels(content: bytes):
    return [line.split() for line in content.decode("utf-8").splitlines() if line.strip()]


def largest_box(lines, class_id=None, allowed=None):
    candidates = []
    for line in lines:
        if len(line) != 5:
            continue
        cid = int(line[0])
        if class_id is not None and cid != class_id:
            continue
        if allowed is not None and cid not in allowed:
            continue
        candidates.append(line)
    if not candidates:
        return None
    return max(candidates, key=lambda x: float(x[3]) * float(x[4]))


def crop_target(image_path: Path, line, output: Path):
    _, cx, cy, width, height = line
    cx, cy, width, height = map(float, (cx, cy, width, height))
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        x1 = max(0, round((cx - width / 2) * image.width))
        y1 = max(0, round((cy - height / 2) * image.height))
        x2 = min(image.width, round((cx + width / 2) * image.width))
        y2 = min(image.height, round((cy + height / 2) * image.height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"empty box for {image_path}")
        crop = image.crop((x1, y1, x2, y2))
        scale = min(960 / crop.width, 960 / crop.height)
        resized = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        crop = crop.resize(resized, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1024, 1024), "white")
        canvas.paste(crop, ((1024 - crop.width) // 2, (1024 - crop.height) // 2))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, quality=95)
    return {
        "bbox_area_ratio": width * height,
        "target_class_id": int(line[0]),
        "crop_sha256": sha256(output),
    }


def download_one(image_id: str, split: str, output: Path):
    path = output / f"{image_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        with Image.open(path) as image:
            image.verify()
        return path
    temporary = path.with_suffix(".jpg.part")
    url = f"http://images.cocodataset.org/{split}/{image_id}.jpg"
    last_error = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "QE-ResNet18-comparison/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                stream.write(response.read())
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
            return path
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed {image_id}: {last_error}")


def prepare_training_manifests(output: Path):
    baseline_manifest = read_csv(ROOT / "artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv")
    candidate_manifest = read_csv(ROOT / "artifacts/dog_candidate400_v2_1_inputs_20260912/target_manifest.csv")
    clean_ids = {row["image_id"] for row in read_csv(
        ROOT / "artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv"
    )}
    all_targets = {row["image_id"]: row for row in baseline_manifest + candidate_manifest}
    before = [{"sample_id": "dog_" + row["image_id"], "image_id": row["image_id"],
               "crop_path": row["crop_image_path"], "label": 1, "source": "baseline500",
               "bbox_area_ratio": ""} for row in baseline_manifest]
    after = [{"sample_id": "dog_" + iid, "image_id": iid,
              "crop_path": all_targets[iid]["crop_image_path"], "label": 1,
              "source": "baseline496" if iid in {r["image_id"] for r in baseline_manifest} else "supplement100",
              "bbox_area_ratio": ""} for iid in sorted(clean_ids)]
    negative_dir = output / "data/train_negative_crops"
    negatives = []
    source_rows = {row["image_id"]: row for row in read_csv(ROOT / "data/coco2017/dog500/manifest.csv")}
    for iid, row in sorted(source_rows.items()):
        lines = [line.split() for line in Path(row["label_path"]).read_text().splitlines() if line.strip()]
        target = largest_box(lines, allowed=set(range(80)) - {DOG_ID})
        if target is None:
            continue
        crop_path = negative_dir / f"{iid}.jpg"
        details = crop_target(Path(row["image_path"]), target, crop_path) if not crop_path.exists() else {
            "bbox_area_ratio": float(target[3]) * float(target[4]), "target_class_id": int(target[0]),
            "crop_sha256": sha256(crop_path)}
        negatives.append({"sample_id": "notdog_" + iid, "image_id": iid, "crop_path": str(crop_path.resolve()),
                          "label": 0, "source": "fixed_train_negative", "bbox_area_ratio": details["bbox_area_ratio"],
                          "target_class_id": details["target_class_id"]})
    fields = ["sample_id", "image_id", "crop_path", "label", "source", "bbox_area_ratio", "target_class_id"]
    for rows in (before, after):
        for row in rows:
            row.setdefault("target_class_id", DOG_ID)
    write_csv(output / "data/train_dog500.csv", before, fields)
    write_csv(output / "data/train_dog596.csv", after, fields)
    write_csv(output / "data/train_negatives_fixed.csv", negatives, fields)
    return before, after, negatives


def choose_stratified_negatives(by_class, count, seed):
    rng = random.Random(seed)
    pools = {}
    for class_id, rows in by_class.items():
        values = rows[:]
        rng.shuffle(values)
        pools[class_id] = values
    chosen, used = [], set()
    classes = sorted(pools)
    while len(chosen) < count:
        progressed = False
        for class_id in classes:
            while pools[class_id] and pools[class_id][-1]["image_id"] in used:
                pools[class_id].pop()
            if pools[class_id] and len(chosen) < count:
                row = pools[class_id].pop()
                chosen.append(row)
                used.add(row["image_id"])
                progressed = True
        if not progressed:
            raise ValueError("insufficient stratified negatives")
    return chosen


def prepare_eval_manifests(output: Path, workers: int):
    archive_path = ROOT / "data/coco2017/coco2017labels.zip"
    dog_rows, negative_by_class = [], defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.startswith("coco/labels/val2017/") or not name.endswith(".txt"):
                continue
            lines = parse_labels(archive.read(name))
            if not lines:
                continue
            image_id = Path(name).stem
            dog = largest_box(lines, class_id=DOG_ID)
            if dog is not None:
                dog_rows.append({"image_id": image_id, "target": dog})
                continue
            for class_id in HARD_NEGATIVE_IDS:
                target = largest_box(lines, class_id=class_id)
                if target is not None:
                    negative_by_class[class_id].append({"image_id": image_id, "target": target})
    rng = random.Random(20260916)
    rng.shuffle(dog_rows)
    if len(dog_rows) != 177:
        raise ValueError(f"expected 177 val dog images, got {len(dog_rows)}")
    negatives = choose_stratified_negatives(negative_by_class, len(dog_rows), 20260916)
    eval_rows = []
    for index, row in enumerate(dog_rows):
        eval_rows.append({**row, "label": 1, "split": "val" if index < 50 else "test"})
    for index, row in enumerate(negatives):
        eval_rows.append({**row, "label": 0, "split": "val" if index < 50 else "test"})
    image_dir = output / "data/eval_images"
    crop_dir = output / "data/eval_crops"
    image_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({row["image_id"] for row in eval_rows})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, iid, "val2017", image_dir): iid for iid in wanted}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0:
                print(f"[prepare] downloaded {index}/{len(wanted)}", flush=True)
    output_rows = []
    for row in eval_rows:
        crop_path = crop_dir / f"{row['label']}_{row['image_id']}.jpg"
        details = crop_target(image_dir / f"{row['image_id']}.jpg", row["target"], crop_path)
        output_rows.append({"sample_id": ("dog_" if row["label"] else "notdog_") + row["image_id"],
                            "image_id": row["image_id"], "crop_path": str(crop_path.resolve()),
                            "label": row["label"], "split": row["split"],
                            "bbox_area_ratio": details["bbox_area_ratio"],
                            "target_class_id": details["target_class_id"]})
    fields = ["sample_id", "image_id", "crop_path", "label", "split", "bbox_area_ratio", "target_class_id"]
    write_csv(output / "data/eval_manifest.csv", output_rows, fields)
    return output_rows


class CropDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), int(row["label"]), index


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_dict(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "non_dog_precision": float(precision[0]), "non_dog_recall": float(recall[0]), "non_dog_f1": float(f1[0]),
        "dog_precision": float(precision[1]), "dog_recall": float(recall[1]), "dog_f1": float(f1[1]),
        "support_non_dog": int(support[0]), "support_dog": int(support[1]),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


@torch.inference_mode()
def predict(model, loader, rows, device):
    model.eval()
    output = [None] * len(rows)
    for images, labels, indices in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities = logits.softmax(dim=1)[:, 1].cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()
        for idx, label, pred, prob in zip(indices.tolist(), labels.tolist(), predictions.tolist(), probabilities.tolist()):
            output[idx] = {**rows[idx], "true_label": label, "predicted_label": pred, "dog_probability": prob,
                           "correct": int(label == pred)}
    return output


def evaluate_rows(predictions):
    labels = [int(row["true_label"]) for row in predictions]
    predicted = [int(row["predicted_label"]) for row in predictions]
    metrics = metric_dict(labels, predicted)
    dog_small = [row for row in predictions if int(row["true_label"]) == 1 and float(row["bbox_area_ratio"]) < 0.03]
    dog_tiny = [row for row in predictions if int(row["true_label"]) == 1 and float(row["bbox_area_ratio"]) < 0.01]
    metrics["small_dog_support"] = len(dog_small)
    metrics["small_dog_recall"] = sum(int(r["predicted_label"]) == 1 for r in dog_small) / len(dog_small) if dog_small else None
    metrics["tiny_dog_support"] = len(dog_tiny)
    metrics["tiny_dog_recall"] = sum(int(r["predicted_label"]) == 1 for r in dog_tiny) / len(dog_tiny) if dog_tiny else None
    return metrics


def array_metric(labels, predicted, areas, metric):
    if metric == "accuracy":
        return float(np.mean(labels == predicted))
    tp = np.sum((labels == 1) & (predicted == 1))
    fn = np.sum((labels == 1) & (predicted == 0))
    fp = np.sum((labels == 0) & (predicted == 1))
    tn = np.sum((labels == 0) & (predicted == 0))
    dog_precision = tp / (tp + fp) if tp + fp else 0.0
    dog_recall = tp / (tp + fn) if tp + fn else 0.0
    dog_f1 = 2 * dog_precision * dog_recall / (dog_precision + dog_recall) if dog_precision + dog_recall else 0.0
    non_dog_precision = tn / (tn + fn) if tn + fn else 0.0
    non_dog_recall = tn / (tn + fp) if tn + fp else 0.0
    non_dog_f1 = (2 * non_dog_precision * non_dog_recall / (non_dog_precision + non_dog_recall)
                  if non_dog_precision + non_dog_recall else 0.0)
    values = {
        "macro_precision": (dog_precision + non_dog_precision) / 2,
        "macro_recall": (dog_recall + non_dog_recall) / 2,
        "macro_f1": (dog_f1 + non_dog_f1) / 2,
        "dog_precision": dog_precision,
        "dog_recall": dog_recall,
        "dog_f1": dog_f1,
        "non_dog_recall": non_dog_recall,
    }
    if metric in values:
        return float(values[metric])
    threshold = 0.03 if metric == "small_dog_recall" else 0.01
    mask = (labels == 1) & (areas < threshold)
    return float(np.mean(predicted[mask] == 1))


def bootstrap_delta(before, after, metric, iterations=10000, seed=20260916):
    if [r["sample_id"] for r in before] != [r["sample_id"] for r in after]:
        raise ValueError("prediction manifests are not aligned")
    rng = np.random.default_rng(seed)
    labels = np.asarray([int(row["true_label"]) for row in before])
    before_predicted = np.asarray([int(row["predicted_label"]) for row in before])
    after_predicted = np.asarray([int(row["predicted_label"]) for row in after])
    areas = np.asarray([float(row["bbox_area_ratio"]) for row in before])
    size = len(labels)
    values = np.empty(iterations, dtype=np.float64)
    for _ in range(iterations):
        indices = rng.integers(0, size, size=size)
        sampled_labels = labels[indices]
        sampled_areas = areas[indices]
        values[_] = (array_metric(sampled_labels, after_predicted[indices], sampled_areas, metric)
                     - array_metric(sampled_labels, before_predicted[indices], sampled_areas, metric))
    return {"lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975))}


def train_one(condition, seed, train_rows, val_rows, test_rows, output: Path, epochs: int):
    seed_everything(seed)
    device = torch.device("cuda")
    normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1), transforms.ToTensor(), normalization,
    ])
    eval_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization])
    train_dataset = CropDataset(train_rows, train_transform)
    class_counts = Counter(int(row["label"]) for row in train_rows)
    weights = [1 / class_counts[int(row["label"])] for row in train_rows]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=1000, replacement=True, generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=8, pin_memory=True,
                              persistent_workers=True, generator=generator)
    val_loader = DataLoader(CropDataset(val_rows, eval_transform), batch_size=64, shuffle=False, num_workers=8,
                            pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(CropDataset(test_rows, eval_transform), batch_size=64, shuffle=False, num_workers=8,
                             pin_memory=True, persistent_workers=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
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
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        val_metrics = evaluate_rows(val_predictions)
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "train_accuracy": correct / seen,
                        "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"[{condition} seed={seed}] epoch {epoch}/{epochs} loss={running_loss/seen:.4f} "
              f"val_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_predictions = predict(model, test_loader, test_rows, device)
    metrics = evaluate_rows(test_predictions)
    metrics.update({"condition": condition, "seed": seed, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                    "train_positive": class_counts[1], "train_negative": class_counts[0], "samples_per_epoch": 1000})
    run_dir = output / condition / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "condition": condition, "seed": seed, "metrics": metrics}, run_dir / "best.pt")
    write_csv(run_dir / "train_log.csv", history)
    write_csv(run_dir / "test_predictions.csv", test_predictions)
    write_json(run_dir / "metrics.json", metrics)
    return metrics, test_predictions


def summarize(all_metrics, predictions, output: Path):
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "dog_precision", "dog_recall",
                    "dog_f1", "non_dog_recall", "small_dog_recall", "tiny_dog_recall"]
    rows = []
    for condition in ("dog500", "dog596"):
        values = all_metrics[condition]
        row = {"condition": condition, "runs": len(values)}
        for metric in metric_names:
            observed = [x[metric] for x in values if x[metric] is not None]
            row[metric + "_mean"] = float(np.mean(observed))
            row[metric + "_std"] = float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0
        rows.append(row)
    deltas = []
    for metric in metric_names:
        paired = [all_metrics["dog596"][i][metric] - all_metrics["dog500"][i][metric] for i in range(len(SEEDS))]
        seed_cis = [bootstrap_delta(predictions["dog500"][i], predictions["dog596"][i], metric)
                    for i in range(len(SEEDS))]
        deltas.append({"metric": metric, "delta_mean": float(np.mean(paired)),
                       "delta_std": float(np.std(paired, ddof=1)),
                       "seed_deltas": json.dumps(paired),
                       "paired_bootstrap_lower_mean": float(np.mean([x["lower_95"] for x in seed_cis])),
                       "paired_bootstrap_upper_mean": float(np.mean([x["upper_95"] for x in seed_cis]))})
    write_csv(output / "comparison/summary_by_condition.csv", rows)
    write_csv(output / "comparison/deltas.csv", deltas)
    report = {"experiment": "ImageNet-pretrained ResNet-18 dog/not-dog target-region classification",
              "comparison": "original dog500 versus clean coverage-improved dog596",
              "coverage": {"dog500": 0.7864, "dog596": 0.8417, "delta": 0.0553},
              "conditions": rows, "deltas": deltas,
              "limitations": ["Binary target-region classification with oracle COCO boxes, not object detection.",
                              "The 500-to-596 change adds 100 selected images and removes four nonliving representations.",
                              "No equal-budget random supplement control; deltas cannot be attributed solely to coverage."]}
    write_json(output / "comparison/report.json", report)
    lines = ["# ImageNet预训练ResNet-18：dog500与dog596比较结果", "", "## 实验口径", "",
             "ImageNet预训练ResNet-18；COCO真实框裁剪；dog/非dog二分类；3个训练种子；验证集与测试集固定。", "",
             "视觉状态覆盖率：78.64% -> 84.17%（+5.53个百分点）。", "",
             "## 结果", "", "| 指标 | dog500（均值±标准差） | dog596（均值±标准差） | 差值 |", "|---|---:|---:|---:|"]
    for metric in metric_names:
        a = rows[0][metric + "_mean"]
        b = rows[1][metric + "_mean"]
        delta = next(x for x in deltas if x["metric"] == metric)["delta_mean"]
        a_std = rows[0][metric + "_std"]
        b_std = rows[1][metric + "_std"]
        lines.append(f"| {metric} | {a:.4f} ± {a_std:.4f} | {b:.4f} ± {b_std:.4f} | {delta:+.4f} |")
    lines += ["", "## 结论边界", "", "- 该结果评价目标区域二分类，不是目标检测。",
              "- dog596相对dog500同时增加100张补图并删除4张非真实犬。",
              "- 尚无等预算随机补图对照，不能把性能差值单独归因于覆盖驱动选择。", ""]
    (output / "comparison/report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/resnet18_dog500_vs_dog596_20260916")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": "torchvision resnet18 ImageNet pretrained", "task": "dog_vs_non_dog_target_crop",
                "conditions": ["dog500", "dog596"], "seeds": list(SEEDS), "epochs": args.epochs,
                "samples_per_epoch": 1000, "batch_size": 32, "optimizer": "AdamW(lr=1e-4,weight_decay=1e-4)",
                "selection_metric": "validation macro-F1", "test_source": "COCO val2017",
                "dog500_definition": "original 500 dog targets",
                "dog596_definition": "clean baseline496 plus selected supplement100",
                "negative_training_data": "fixed largest non-dog target from each usable baseline image"}
    write_json(args.output / "protocol.json", protocol)
    before, after, negatives = prepare_training_manifests(args.output)
    eval_rows = prepare_eval_manifests(args.output, args.workers)
    val_rows = [row for row in eval_rows if row["split"] == "val"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    all_metrics = defaultdict(list)
    predictions = defaultdict(list)
    for condition, positives in (("dog500", before), ("dog596", after)):
        train_rows = positives + negatives
        for seed in SEEDS:
            metrics, values = train_one(condition, seed, train_rows, val_rows, test_rows, args.output, args.epochs)
            all_metrics[condition].append(metrics)
            predictions[condition].append(values)
    report = summarize(all_metrics, predictions, args.output)
    checksums = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "checksums.csv":
            checksums.append({"path": str(path.relative_to(args.output)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(args.output / "checksums.csv", checksums)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
