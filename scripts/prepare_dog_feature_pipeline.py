#!/usr/bin/env python3
"""Materialize the frozen dog-v1 target manifest and the three model inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

DOG_ID = 16
FIELDS = ["image_id", "target_instance_id", "target_class_id", "target_class", "bbox",
          "image_path", "marked_image_path", "crop_image_path", "split", "image_sha256",
          "label_sha256", "protocol_version"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dog_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        bits = line.split()
        if len(bits) != 5 or int(bits[0]) != DOG_ID:
            continue
        _, cx, cy, width, height = map(float, bits)
        boxes.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    return boxes


def absolute_box(box, width, height):
    x1, y1, x2, y2 = box
    return [max(0, round(x1 * width)), max(0, round(y1 * height)),
            min(width, round(x2 * width)), min(height, round(y2 * height))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/coco2017/dog500"))
    ap.add_argument("--manifest", type=Path, help="source manifest; defaults to DATASET/manifest.csv")
    ap.add_argument("--output", type=Path, default=Path("artifacts/dog_feature_pipeline_v1"))
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--protocol-version", default="dog-v1")
    ap.add_argument("--expected-count", type=int, default=500)
    ap.add_argument("--candidate-split", action="store_true", help="mark every row as candidate")
    args = ap.parse_args()
    manifest = args.manifest or (args.dataset / "manifest.csv")
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    if len(rows) != args.expected_count or len({r["image_id"] for r in rows}) != args.expected_count:
        raise SystemExit(f"expected exactly {args.expected_count} unique images")
    rng = random.Random(args.seed)
    ordered = sorted(rows, key=lambda r: r["image_id"])
    rng.shuffle(ordered)
    splits = ({r["image_id"]: "candidate" for r in ordered} if args.candidate_split else
              {r["image_id"]: ("calibration" if i < 100 else "development" if i < 300 else "validation") for i, r in enumerate(ordered)})
    (args.output / "marked").mkdir(parents=True, exist_ok=True)
    (args.output / "crops").mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        label_path = Path(row["label_path"])
        if not label_path.is_absolute():
            label_path = Path.cwd() / label_path
        boxes = dog_boxes(label_path)
        if not boxes:
            raise SystemExit(f"no dog box: {row['image_id']}")
        target = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            box = absolute_box(target, image.width, image.height)
            marked = image.copy()
            draw = ImageDraw.Draw(marked)
            draw.rectangle(box, outline=(255, 0, 0), width=max(3, min(image.size) // 150))
            draw.text((box[0] + 4, box[1] + 4), "[A]", fill=(255, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
            marked_path = args.output / "marked" / f"{row['image_id']}.jpg"
            crop_path = args.output / "crops" / f"{row['image_id']}.jpg"
            marked.save(marked_path, quality=95)
            crop = image.crop(box)
            scale = min(960 / crop.width, 960 / crop.height)
            resized = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
            crop = crop.resize(resized, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1024, 1024), "white")
            canvas.paste(crop, ((1024 - crop.width) // 2, (1024 - crop.height) // 2))
            canvas.save(crop_path, quality=95)
        output_rows.append({
            "image_id": row["image_id"], "target_instance_id": "target_1", "target_class_id": DOG_ID,
            "target_class": "dog", "bbox": json.dumps(box, separators=(",", ":")),
            "image_path": str(image_path.resolve()), "marked_image_path": str(marked_path.resolve()),
            "crop_image_path": str(crop_path.resolve()), "split": splits[row["image_id"]],
            "image_sha256": sha256(image_path), "label_sha256": sha256(label_path),
            "protocol_version": args.protocol_version,
        })
    with (args.output / "target_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(output_rows)
    with (args.output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "split"]); writer.writeheader()
        writer.writerows({"image_id": r["image_id"], "split": r["split"]} for r in output_rows)
    split_names = sorted({r["split"] for r in output_rows})
    summary = {"status": "prepared", "count": len(output_rows), "splits": {s: sum(r["split"] == s for r in output_rows) for s in split_names}, "dog_class_id": DOG_ID, "protocol_version": args.protocol_version, "crop_policy": "upscale_longest_side_to_960_on_1024_canvas"}
    (args.output / "prepare_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
