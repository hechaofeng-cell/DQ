#!/usr/bin/env python3
"""Materialize a deterministic person subset from the existing COCO YOLO data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


SOURCE_PRIORITY = {
    "coco500": 0,
    "dog500": 1,
    "dog_candidate400_20260912": 2,
    "dog500_incomplete_20260911": 3,
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_person_boxes(label_path, width, height):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            continue
        try:
            _, cx, cy, box_width, box_height = map(float, parts)
        except ValueError:
            continue
        x1 = max(0.0, (cx - box_width / 2) * width)
        y1 = max(0.0, (cy - box_height / 2) * height)
        x2 = min(float(width), (cx + box_width / 2) * width)
        y2 = min(float(height), (cy + box_height / 2) * height)
        if x2 > x1 and y2 > y1:
            boxes.append([round(x1), round(y1), round(x2), round(y2)])
    return boxes


def image_for_stem(image_dir, stem):
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def collect_candidates(root):
    candidates = {}
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image_dir = source_dir / "images"
        label_dir = source_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        priority = SOURCE_PRIORITY.get(source_dir.name, 10)
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_for_stem(image_dir, label_path.stem)
            if image_path is None:
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes = parse_person_boxes(label_path, width, height)
            except (OSError, ValueError):
                continue
            if not boxes:
                continue
            record = {
                "image_id": label_path.stem,
                "source_dataset": source_dir.name,
                "image_path": image_path,
                "label_path": label_path,
                "image_sha256": sha256(image_path),
                "width": width,
                "height": height,
                "person_boxes": boxes,
                "target_bbox": max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])),
                "priority": priority,
            }
            dedupe_key = record["image_sha256"]
            previous = candidates.get(dedupe_key)
            if previous is None or (record["priority"], record["image_id"]) < (previous["priority"], previous["image_id"]):
                candidates[dedupe_key] = record
    return sorted(candidates.values(), key=lambda item: (item["priority"], item["image_id"]))


def materialize(record, output):
    image_id = record["image_id"]
    source = record["image_path"]
    box = record["target_bbox"]
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    image.save(output / "images" / f"{image_id}.jpg", quality=95)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(box, outline=(255, 0, 0), width=max(3, round(min(image.size) / 200)))
    draw.text((box[0] + 4, box[1] + 4), "[A]", fill=(255, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    marked.save(output / "marked" / f"{image_id}.jpg", quality=95)
    x1, y1, x2, y2 = box
    pad_x, pad_y = round((x2 - x1) * 0.05), round((y2 - y1) * 0.05)
    crop_box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(image.width, x2 + pad_x), min(image.height, y2 + pad_y))
    crop = image.crop(crop_box)
    scale = min(960 / crop.width, 960 / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1024, 1024), "white")
    canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
    canvas.save(output / "crops" / f"{image_id}.jpg", quality=95)
    shutil.copy2(record["label_path"], output / "labels" / f"{image_id}.txt")
    return {
        "image_id": image_id,
        "target_instance_id": "target_1",
        "image_path": f"images/{image_id}.jpg",
        "bbox": json.dumps(box, separators=(",", ":")),
        "marked_image_path": f"marked/{image_id}.jpg",
        "crop_image_path": f"crops/{image_id}.jpg",
        "target_class": "person",
        "source_dataset": record["source_dataset"],
        "source_image_sha256": record["image_sha256"],
        "person_count_in_image": len(record["person_boxes"]),
        "target_area_ratio": round((box[2] - box[0]) * (box[3] - box[1]) / (record["width"] * record["height"]), 6),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/coco2017"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/person300_dataset_v1"))
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args(argv)
    candidates = collect_candidates(args.source_root.resolve())
    if len(candidates) < args.count:
        raise SystemExit(f"only_{len(candidates)}_unique_person_images_available")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output_exists:{output}")
    for name in ("images", "marked", "crops", "labels"):
        (output / name).mkdir(parents=True, exist_ok=True)
    selected = candidates[:args.count]
    rows = [materialize(record, output) for record in selected]
    fields = ["image_id", "target_instance_id", "image_path", "bbox", "marked_image_path", "crop_image_path", "target_class", "source_dataset", "source_image_sha256", "person_count_in_image", "target_area_ratio"]
    with (output / "target_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "prepared",
        "count": len(rows),
        "source_root": str(args.source_root.resolve()),
        "unique_person_images_available": len(candidates),
        "selected_source_counts": {source: sum(row["source_dataset"] == source for row in rows) for source in sorted({row["source_dataset"] for row in rows})},
        "selected_person_count_distribution": {str(count): sum(int(row["person_count_in_image"]) == count for row in rows) for count in sorted({int(row["person_count_in_image"]) for row in rows})},
        "target_policy": "one_largest_person_bbox_per_unique_image",
        "deduplication": "sha256",
    }
    (output / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
