#!/usr/bin/env python3
"""Create dog_v3.1 three-view inputs for the frozen main and difficulty tests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from openimages_cua_vsl import read_csv, write_csv, write_json
from prepare_openimages_v7_experiment import absolute_box, sha256


ROOT = Path(__file__).resolve().parents[1]


def materialize_row(row: dict, output: Path, group: str) -> dict:
    image_path = Path(row["image_path"])
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    box = absolute_box(json.loads(row["bbox_norm_xyxy"]), image.width, image.height, 0)
    marked_path = output / "tri_view" / group / "marked" / f"{row['sample_id']}.jpg"
    crop_path = output / "tri_view" / group / "crops" / f"{row['sample_id']}.jpg"
    marked_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not marked_path.is_file():
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        line_width = max(3, min(image.size) // 150)
        draw.rectangle(box, outline=(255, 0, 0), width=line_width)
        draw.text((box[0] + 4, box[1] + 4), "[A]", fill=(255, 0, 0), stroke_width=2, stroke_fill="white")
        marked.save(marked_path, quality=95)
    if not crop_path.is_file():
        target_crop = image.crop(box)
        scale = min(960 / target_crop.width, 960 / target_crop.height)
        resized = target_crop.resize(
            (max(1, round(target_crop.width * scale)), max(1, round(target_crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (1024, 1024), "white")
        canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
        canvas.save(crop_path, quality=95)
    return {
        "image_id": row["sample_id"],
        "target_instance_id": "target_1",
        "target_class_id": "/m/0bt9lr",
        "target_class": "dog",
        "bbox": json.dumps(box, separators=(",", ":")),
        "image_path": str(image_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "crop_image_path": str(crop_path.resolve()),
        "split": group,
        "image_sha256": row["image_sha256"] or sha256(image_path),
        "label_sha256": "",
        "protocol_version": "dog_v3.1-openimages-v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/openimages_v7_dog_gap_v1"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_inputs",
    )
    args = parser.parse_args()
    groups = {
        "main_test": [
            row for row in read_csv(args.data / "manifests/test.csv")
            if row["class_name"] == "dog"
        ],
        "difficulty_test": read_csv(args.data / "manifests/dog_difficulty_test.csv"),
    }
    all_image_ids = [row["image_id"] for rows in groups.values() for row in rows]
    if len(all_image_ids) != len(set(all_image_ids)):
        raise ValueError("main and difficulty dog tests overlap")
    feature_rows = []
    for group, rows in groups.items():
        if any(row["class_name"] != "dog" for row in rows):
            raise ValueError(f"{group} contains non-dog rows")
        feature_rows.extend(materialize_row(row, args.output, group) for row in rows)
    fields = [
        "image_id", "target_instance_id", "target_class_id", "target_class", "bbox",
        "image_path", "marked_image_path", "crop_image_path", "split", "image_sha256",
        "label_sha256", "protocol_version",
    ]
    manifest_path = args.output / "dog_feature_test_manifest.csv"
    write_csv(manifest_path, feature_rows, fields)
    write_json(args.output / "report.json", {
        "main_test_dog": len(groups["main_test"]),
        "difficulty_test_dog": len(groups["difficulty_test"]),
        "total": len(feature_rows),
        "manifest": str(manifest_path.resolve()),
    })
    print(json.dumps({"total": len(feature_rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
