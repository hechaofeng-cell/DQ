#!/usr/bin/env python3
"""Add a frozen 100-target difficult-cat test to the existing Open Images pool."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from prepare_openimages_v7_experiment import (
    MANIFEST_FIELDS, absolute_box, eligible, image_targets, load_json,
    manifest_row, sha256, write_csv, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/openimages_v7_dog_gap_v1"
CONFIG = ROOT / "configs/openimages_v7_dog_gap_experiment_v1.json"
OUTPUT_MANIFEST = DATA / "manifests/cat_difficulty_test.csv"
FEATURE_ROOT = ROOT / "artifacts/openimages_v7_cat_test_v1/inputs"
SEED = 20260920
COUNT = 100


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def choose_difficult(rows: list[dict], used: set[str]) -> list[dict]:
    pool = [row for row in rows if row["ImageID"] not in used]
    random.Random(SEED).shuffle(pool)
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def area(row: dict) -> float:
        return (float(row["XMax"]) - float(row["XMin"])) * (float(row["YMax"]) - float(row["YMin"]))

    def take(predicate, limit: int) -> None:
        for row in pool:
            if len(selected) >= COUNT or limit <= 0:
                break
            if row["ImageID"] not in selected_ids and predicate(row):
                selected.append(row)
                selected_ids.add(row["ImageID"])
                limit -= 1

    take(lambda row: area(row) < .01, 10)
    take(lambda row: .01 <= area(row) < .03, 35)
    take(lambda row: row["IsOccluded"] == "1", 25)
    take(lambda row: row["IsTruncated"] == "1", 25)
    take(lambda row: True, COUNT - len(selected))
    if len(selected) != COUNT:
        raise ValueError(f"need {COUNT} difficult cats, selected {len(selected)}")
    return selected


def download(row: dict) -> tuple[str, str]:
    target = DATA / "images/test" / f"{row['image_id']}.jpg"
    if target.is_file() and target.stat().st_size > 0:
        try:
            with Image.open(target) as image:
                image.load()
            return row["image_id"], "reused"
        except OSError:
            target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jpg.part")
    url = f"https://open-images-dataset.s3.amazonaws.com/test/{row['image_id']}.jpg"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "QE-cat-test-preparer/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(target)
        with Image.open(target) as image:
            image.load()
        return row["image_id"], "downloaded"
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return row["image_id"], f"error:{exc}"


def materialize(row: dict) -> tuple[dict, dict]:
    image_path = DATA / "images/test" / f"{row['image_id']}.jpg"
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    norm = json.loads(row["bbox_norm_xyxy"])
    target_box = absolute_box(norm, image.width, image.height, 0)
    crop_box = absolute_box(norm, image.width, image.height, .05)

    group = "main_test" if row["split"] == "test" else "difficulty_test"
    crop_path = (
        Path(row["crop_path"]) if row.get("crop_path")
        else DATA / f"crops/{row['split']}/cat" / f"{row['sample_id']}.jpg"
    )
    marked_path = FEATURE_ROOT / f"tri_view/{group}/marked" / f"{row['sample_id']}.jpg"
    feature_crop_path = FEATURE_ROOT / f"tri_view/{group}/crops" / f"{row['sample_id']}.jpg"
    for path in (crop_path, marked_path, feature_crop_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(crop_path, quality=95)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    width = max(3, min(image.size) // 150)
    draw.rectangle(target_box, outline=(255, 0, 0), width=width)
    draw.text((target_box[0] + 4, target_box[1] + 4), "[A]", fill=(255, 0, 0), stroke_width=2, stroke_fill="white")
    marked.save(marked_path, quality=95)
    target_crop = image.crop(target_box)
    scale = min(960 / target_crop.width, 960 / target_crop.height)
    resized = target_crop.resize(
        (max(1, round(target_crop.width * scale)), max(1, round(target_crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (1024, 1024), "white")
    canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
    canvas.save(feature_crop_path, quality=95)

    row.update({
        "image_path": str(image_path.resolve()),
        "crop_path": str(crop_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "feature_crop_path": str(feature_crop_path.resolve()),
        "image_sha256": sha256(image_path),
        "crop_sha256": sha256(crop_path),
    })
    feature = {
        "image_id": row["sample_id"],
        "target_instance_id": "target_1",
        "target_class_id": "/m/01yrx",
        "target_class": "cat",
        "bbox": json.dumps(target_box, separators=(",", ":")),
        "image_path": str(image_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "crop_image_path": str(feature_crop_path.resolve()),
        "split": group,
        "image_sha256": row["image_sha256"],
        "label_sha256": "",
        "protocol_version": "cat_v1.0-openimages-v1",
    }
    return row, feature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = load_json(CONFIG)
    annotations = read(DATA / "annotations/test_five_classes.csv")
    targets = image_targets(annotations, {mid: name for name, mid in config["classes"].items()}, config["box_filters"])
    # Exclude only the pre-existing frozen tests. Excluding this script's own
    # output would change the deterministic selection on a resume.
    used = {
        row["image_id"]
        for path in (DATA / "manifests/test.csv", DATA / "manifests/dog_difficulty_test.csv")
        for row in read(path)
        if row.get("source_split") == "test"
    }
    selected = choose_difficult(targets["cat"], used)
    rows = [manifest_row(row, config, "test", "cat_difficulty_test", index) for index, row in enumerate(selected, 1)]

    statuses = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            statuses[status.split(":", 1)[0]] += 1
            if status.startswith("error:"):
                raise RuntimeError(status)
            if index % 20 == 0 or index == len(rows):
                print(f"[download] {index}/{len(rows)} {dict(statuses)}", flush=True)

    materialized, difficulty_feature_rows = [], []
    for index, row in enumerate(rows, 1):
        updated, feature = materialize(row)
        materialized.append(updated)
        difficulty_feature_rows.append(feature)
        if index % 20 == 0 or index == len(rows):
            print(f"[materialize] {index}/{len(rows)}", flush=True)
    write_csv(OUTPUT_MANIFEST, materialized, MANIFEST_FIELDS)
    main_cat_rows = [
        row for row in read(DATA / "manifests/test.csv") if row["class_name"] == "cat"
    ]
    main_feature_rows = []
    for index, row in enumerate(main_cat_rows, 1):
        _, feature = materialize(row)
        main_feature_rows.append(feature)
        if index % 20 == 0 or index == len(main_cat_rows):
            print(f"[main-cat-triview] {index}/{len(main_cat_rows)}", flush=True)
    fields = [
        "image_id", "target_instance_id", "target_class_id", "target_class", "bbox",
        "image_path", "marked_image_path", "crop_image_path", "split", "image_sha256",
        "label_sha256", "protocol_version",
    ]
    write_csv(FEATURE_ROOT / "cat_difficulty_feature_manifest.csv", difficulty_feature_rows, fields)
    write_csv(
        FEATURE_ROOT / "cat_feature_test_manifest.csv",
        main_feature_rows + difficulty_feature_rows,
        fields,
    )
    report = {
        "status": "complete",
        "source": "Open Images test split",
        "selection_seed": SEED,
        "selected": len(materialized),
        "main_test_cat": len(main_feature_rows),
        "total_cat_feature_targets": len(main_feature_rows) + len(difficulty_feature_rows),
        "download_status": dict(statuses),
        "difficulty_counts": {
            "tiny_lt_1pct": sum(float(row["bbox_area_ratio"]) < .01 for row in materialized),
            "small_1_to_3pct": sum(.01 <= float(row["bbox_area_ratio"]) < .03 for row in materialized),
            "occluded": sum(row["is_occluded"] == "1" for row in materialized),
            "truncated": sum(row["is_truncated"] == "1" for row in materialized),
        },
        "existing_test_image_ids_excluded": len(used),
        "manifest_sha256": sha256(OUTPUT_MANIFEST),
    }
    write_json(FEATURE_ROOT.parent / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
