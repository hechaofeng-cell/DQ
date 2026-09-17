#!/usr/bin/env python3
"""Prepare a frozen Open Images V7 five-class dog-gap experiment.

The script streams the official annotation CSVs, retains only the five target
classes, creates image-disjoint manifests, downloads selected source images,
and materializes target crops plus dog tri-view inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/openimages_v7_dog_gap_experiment_v1.json"
DEFAULT_OUTPUT = ROOT / "data/openimages_v7_dog_gap_v1"
ANNOTATION_FIELDS = [
    "ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax",
    "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
    "IsDepiction", "IsInside",
]
MANIFEST_FIELDS = [
    "sample_id", "image_id", "target_instance_id", "class_name", "label",
    "source_split", "bbox_norm_xyxy", "bbox_area_ratio", "is_occluded",
    "is_truncated", "is_group_of", "is_depiction", "is_inside", "split",
    "dataset_version", "image_path", "crop_path", "marked_image_path",
    "feature_crop_path", "image_sha256", "crop_sha256",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ANNOTATION_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_rows(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "QE-openimages-preparer/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        lines = (line.decode("utf-8") for line in response)
        yield from csv.DictReader(lines)


def filter_annotations(config: dict, output: Path, force: bool) -> None:
    wanted = set(config["classes"].values())
    annotation_dir = output / "annotations"
    summary = {}
    for split, key in (("train", "train_boxes"), ("validation", "validation_boxes"), ("test", "test_boxes")):
        target = annotation_dir / f"{split}_five_classes.csv"
        if target.exists() and not force:
            rows = list(csv.DictReader(target.open(encoding="utf-8", newline="")))
            summary[split] = {"rows": len(rows), "by_mid": dict(Counter(row["LabelName"] for row in rows))}
            print(f"[{split}] reusing {target} ({len(rows):,} rows)", flush=True)
            continue
        rows = []
        print(f"[{split}] streaming {config['official_urls'][key]}", flush=True)
        for index, row in enumerate(stream_rows(config["official_urls"][key]), 1):
            if row["LabelName"] in wanted:
                rows.append({field: row.get(field, "") for field in ANNOTATION_FIELDS})
            if index % 1_000_000 == 0:
                print(f"[{split}] scanned {index:,}; retained {len(rows):,}", flush=True)
        write_csv(target, rows, ANNOTATION_FIELDS)
        summary[split] = {"rows": len(rows), "by_mid": dict(Counter(row["LabelName"] for row in rows))}
        print(f"[{split}] retained {len(rows):,} rows", flush=True)
    write_json(output / "annotation_summary.json", summary)


def eligible(row: dict, filters: dict) -> bool:
    area = (float(row["XMax"]) - float(row["XMin"])) * (float(row["YMax"]) - float(row["YMin"]))
    return (
        float(row["Confidence"]) >= float(filters["confidence"])
        and int(row["IsGroupOf"]) == int(filters["is_group_of"])
        and int(row["IsDepiction"]) == int(filters["is_depiction"])
        and int(row["IsInside"]) == int(filters["is_inside"])
        and area >= float(filters["minimum_area_ratio"])
    )


def image_targets(rows: list[dict], mid_to_name: dict[str, str], filters: dict) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        if eligible(row, filters):
            grouped[(mid_to_name[row["LabelName"]], row["ImageID"])].append(row)
    by_class = defaultdict(list)
    for (class_name, image_id), boxes in grouped.items():
        target = max(
            boxes,
            key=lambda row: (float(row["XMax"]) - float(row["XMin"]))
            * (float(row["YMax"]) - float(row["YMin"])),
        )
        target = dict(target)
        target["class_name"] = class_name
        target["same_class_box_count"] = len(boxes)
        by_class[class_name].append(target)
    return by_class


def choose(rows: list[dict], count: int, used_images: set[str], rng: random.Random, label: str) -> list[dict]:
    pool = [row for row in rows if row["ImageID"] not in used_images]
    rng.shuffle(pool)
    if len(pool) < count:
        raise ValueError(f"{label}: need {count:,} images, only {len(pool):,} eligible")
    selected = pool[:count]
    used_images.update(row["ImageID"] for row in selected)
    return selected


def manifest_row(row: dict, config: dict, source_split: str, split: str, rank: int) -> dict:
    class_name = row["class_name"]
    bbox = [float(row[key]) for key in ("XMin", "YMin", "XMax", "YMax")]
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return {
        "sample_id": f"oi7-{source_split}-{row['ImageID']}-{class_name}",
        "image_id": row["ImageID"],
        "target_instance_id": f"target_{rank:06d}",
        "class_name": class_name,
        "label": config["class_labels"][class_name],
        "source_split": source_split,
        "bbox_norm_xyxy": json.dumps(bbox, separators=(",", ":")),
        "bbox_area_ratio": f"{area:.8f}",
        "is_occluded": row["IsOccluded"],
        "is_truncated": row["IsTruncated"],
        "is_group_of": row["IsGroupOf"],
        "is_depiction": row["IsDepiction"],
        "is_inside": row["IsInside"],
        "split": split,
        "dataset_version": config["experiment_id"],
        "image_path": "",
        "crop_path": "",
        "marked_image_path": "",
        "feature_crop_path": "",
        "image_sha256": "",
        "crop_sha256": "",
    }


def choose_dog_difficulty(rows: list[dict], count: int, used_images: set[str], rng: random.Random) -> list[dict]:
    pool = [row for row in rows if row["ImageID"] not in used_images]
    rng.shuffle(pool)
    selected = []
    selected_ids = set()

    def area(row: dict) -> float:
        return (float(row["XMax"]) - float(row["XMin"])) * (float(row["YMax"]) - float(row["YMin"]))

    def take(predicate, limit: int) -> None:
        for row in pool:
            if len(selected) >= count or limit <= 0:
                break
            if row["ImageID"] not in selected_ids and predicate(row):
                selected.append(row)
                selected_ids.add(row["ImageID"])
                limit -= 1

    # Preserve all available small targets first, then add explicit difficult attributes.
    take(lambda row: area(row) < .01, 10)
    take(lambda row: .01 <= area(row) < .03, 35)
    take(lambda row: row["IsOccluded"] == "1", 25)
    take(lambda row: row["IsTruncated"] == "1", 25)
    take(lambda row: True, count - len(selected))
    if len(selected) != count:
        raise ValueError(f"difficulty dog test: need {count}, only selected {len(selected)}")
    used_images.update(selected_ids)
    return selected


def build_plan(config: dict, output: Path) -> None:
    rng = random.Random(config["selection_seed"])
    mid_to_name = {mid: name for name, mid in config["classes"].items()}
    split_targets = {}
    for split in ("train", "validation", "test"):
        path = output / "annotations" / f"{split}_five_classes.csv"
        if not path.exists():
            raise FileNotFoundError(f"run metadata first: {path}")
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        split_targets[split] = image_targets(rows, mid_to_name, config["box_filters"])

    counts = config["counts"]
    train_used = set()
    base_selected = {}
    train_counts = counts["train_base_by_class"]
    # Dog is selected first so its baseline and candidate pool cannot overlap other targets.
    base_selected["dog"] = choose(
        split_targets["train"]["dog"], train_counts["dog"], train_used, rng, "train dog base"
    )
    dog_candidates = choose(
        split_targets["train"]["dog"], counts["train_dog_candidate"], train_used, rng, "train dog candidates"
    )
    other_classes = ("cat", "horse", "sheep", "person")
    for class_name in sorted(other_classes, key=lambda name: len(split_targets["train"][name])):
        base_selected[class_name] = choose(
            split_targets["train"][class_name], train_counts[class_name], train_used, rng,
            f"train {class_name} base",
        )

    base_rows = []
    for class_name in config["classes"]:
        base_rows.extend(
            manifest_row(row, config, "train", "train", index)
            for index, row in enumerate(base_selected[class_name], 1)
        )
    candidate_rows = [
        manifest_row(row, config, "train", "candidate", index)
        for index, row in enumerate(dog_candidates, 1)
    ]

    eval_manifests = {}
    test_used = set()
    for source_split, split_name, count_key in (
        ("validation", "val", "validation_per_class"),
        ("test", "test", "test_per_class"),
    ):
        used = set()
        rows = []
        for class_name in sorted(config["classes"], key=lambda name: len(split_targets[source_split][name])):
            selected = choose(
                split_targets[source_split][class_name], counts[count_key], used, rng,
                f"{source_split} {class_name}",
            )
            rows.extend(
                manifest_row(row, config, source_split, split_name, index)
                for index, row in enumerate(selected, 1)
            )
        eval_manifests[split_name] = rows
        if split_name == "test":
            test_used = used

    difficulty_selected = choose_dog_difficulty(
        split_targets["test"]["dog"], counts["difficulty_test_dog"], test_used, rng
    )
    difficulty_rows = [
        manifest_row(row, config, "test", "difficulty_test", index)
        for index, row in enumerate(difficulty_selected, 1)
    ]

    manifest_dir = output / "manifests"
    write_csv(manifest_dir / "train_base.csv", base_rows, MANIFEST_FIELDS)
    write_csv(manifest_dir / "dog_candidates.csv", candidate_rows, MANIFEST_FIELDS)
    write_csv(manifest_dir / "validation.csv", eval_manifests["val"], MANIFEST_FIELDS)
    write_csv(manifest_dir / "test.csv", eval_manifests["test"], MANIFEST_FIELDS)
    write_csv(manifest_dir / "dog_difficulty_test.csv", difficulty_rows, MANIFEST_FIELDS)

    all_rows = base_rows + candidate_rows + eval_manifests["val"] + eval_manifests["test"] + difficulty_rows
    image_list = output / "image_ids.txt"
    image_list.write_text(
        "\n".join(sorted({f"{row['source_split']}/{row['image_id']}" for row in all_rows})) + "\n",
        encoding="ascii",
    )
    report = {
        "experiment_id": config["experiment_id"],
        "counts": {
            "train_base": len(base_rows),
            "train_base_by_class": dict(Counter(row["class_name"] for row in base_rows)),
            "dog_candidates": len(candidate_rows),
            "validation": len(eval_manifests["val"]),
            "test": len(eval_manifests["test"]),
            "dog_difficulty_test": len(difficulty_rows),
            "unique_source_images": len({row["image_id"] for row in all_rows}),
        },
        "image_disjoint_checks": {
            "base_candidate_overlap": len({r["image_id"] for r in base_rows} & {r["image_id"] for r in candidate_rows}),
            "train_validation_overlap": len({r["image_id"] for r in base_rows + candidate_rows} & {r["image_id"] for r in eval_manifests["val"]}),
            "train_test_overlap": len({r["image_id"] for r in base_rows + candidate_rows} & {r["image_id"] for r in eval_manifests["test"]}),
            "main_difficulty_test_overlap": len({r["image_id"] for r in eval_manifests["test"]} & {r["image_id"] for r in difficulty_rows}),
        },
        "status": "planned_not_downloaded",
    }
    write_json(output / "plan_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def download_one(item: tuple[str, str], image_root: Path, force: bool) -> tuple[str, str]:
    split, image_id = item
    target = image_root / split / f"{image_id}.jpg"
    if target.exists() and target.stat().st_size > 0 and not force:
        try:
            with Image.open(target) as image:
                image.load()
            return image_id, "reused"
        except (OSError, ValueError):
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jpg.part")
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "QE-openimages-preparer/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(target)
        with Image.open(target) as image:
            image.load()
        return image_id, "downloaded"
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return image_id, f"error:{exc}"


def download_images(output: Path, workers: int, force: bool) -> None:
    list_path = output / "image_ids.txt"
    items = [tuple(line.strip().split("/", 1)) for line in list_path.read_text().splitlines() if line.strip()]
    statuses = Counter()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, item, output / "images", force): item for item in items}
        for index, future in enumerate(as_completed(futures), 1):
            image_id, status = future.result()
            statuses[status.split(":", 1)[0]] += 1
            if status.startswith("error:"):
                failures.append({"image_id": image_id, "error": status[6:]})
            if index % 100 == 0 or index == len(items):
                print(f"[download] {index:,}/{len(items):,} {dict(statuses)}", flush=True)
    write_json(output / "download_report.json", {"planned": len(items), "status_counts": statuses, "failures": failures})
    if failures:
        raise SystemExit(f"{len(failures)} image downloads failed; rerun with the same command to resume")


def absolute_box(norm_box: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = norm_box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, round((x1 - pad_x) * width)),
        max(0, round((y1 - pad_y) * height)),
        min(width, round((x2 + pad_x) * width)),
        min(height, round((y2 + pad_y) * height)),
    )


def materialize(output: Path, padding: float) -> None:
    manifest_dir = output / "manifests"
    for manifest_path in sorted(manifest_dir.glob("*.csv")):
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
        updated = []
        for index, row in enumerate(rows, 1):
            image_path = output / "images" / row["source_split"] / f"{row['image_id']}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                norm_box = json.loads(row["bbox_norm_xyxy"])
                target_box = absolute_box(norm_box, image.width, image.height, 0)
                crop_box = absolute_box(norm_box, image.width, image.height, padding)
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    raise ValueError(f"invalid crop box for {row['sample_id']}: {crop_box}")
                crop = image.crop(crop_box)
                crop_path = output / "crops" / row["split"] / row["class_name"] / f"{row['sample_id']}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=95)
                row["image_path"] = str(image_path.resolve())
                row["crop_path"] = str(crop_path.resolve())
                row["image_sha256"] = sha256(image_path)
                row["crop_sha256"] = sha256(crop_path)
                if row["class_name"] == "dog" and row["split"] in {"train", "candidate"}:
                    tri_root = output / "dog_tri_view" / row["split"]
                    marked_path = tri_root / "marked" / f"{row['sample_id']}.jpg"
                    feature_crop_path = tri_root / "crops" / f"{row['sample_id']}.jpg"
                    marked_path.parent.mkdir(parents=True, exist_ok=True)
                    feature_crop_path.parent.mkdir(parents=True, exist_ok=True)
                    marked = image.copy()
                    draw = ImageDraw.Draw(marked)
                    line_width = max(3, min(image.size) // 150)
                    draw.rectangle(target_box, outline=(255, 0, 0), width=line_width)
                    draw.text((target_box[0] + 4, target_box[1] + 4), "[A]", fill=(255, 0, 0), stroke_width=2, stroke_fill="white")
                    marked.save(marked_path, quality=95)
                    target_crop = image.crop(target_box)
                    scale = min(960 / target_crop.width, 960 / target_crop.height)
                    feature_crop = target_crop.resize(
                        (max(1, round(target_crop.width * scale)), max(1, round(target_crop.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    canvas = Image.new("RGB", (1024, 1024), "white")
                    canvas.paste(feature_crop, ((1024 - feature_crop.width) // 2, (1024 - feature_crop.height) // 2))
                    canvas.save(feature_crop_path, quality=95)
                    row["marked_image_path"] = str(marked_path.resolve())
                    row["feature_crop_path"] = str(feature_crop_path.resolve())
            updated.append(row)
            if index % 250 == 0 or index == len(rows):
                print(f"[materialize] {manifest_path.name} {index:,}/{len(rows):,}", flush=True)
        write_csv(manifest_path, updated, MANIFEST_FIELDS)

    dog_rows = []
    dog_rows_by_split = defaultdict(list)
    for name in ("train_base.csv", "dog_candidates.csv"):
        for row in csv.DictReader((manifest_dir / name).open(encoding="utf-8", newline="")):
            if row["class_name"] == "dog":
                with Image.open(row["image_path"]) as image:
                    target_box = absolute_box(
                        json.loads(row["bbox_norm_xyxy"]), image.width, image.height, 0
                    )
                feature_row = {
                    "image_id": row["sample_id"],
                    "target_instance_id": "target_1",
                    "target_class_id": "/m/0bt9lr",
                    "target_class": "dog",
                    "bbox": json.dumps(target_box, separators=(",", ":")),
                    "image_path": row["image_path"],
                    "marked_image_path": row["marked_image_path"],
                    "crop_image_path": row["feature_crop_path"],
                    "split": row["split"],
                    "image_sha256": row["image_sha256"],
                    "label_sha256": "",
                    "protocol_version": "dog_v3.1-openimages-v1",
                }
                dog_rows.append(feature_row)
                dog_rows_by_split[row["split"]].append(feature_row)
    feature_fields = [
        "image_id", "target_instance_id", "target_class_id", "target_class", "bbox",
        "image_path", "marked_image_path", "crop_image_path", "split", "image_sha256",
        "label_sha256", "protocol_version",
    ]
    write_csv(output / "dog_feature_target_manifest.csv", dog_rows, feature_fields)
    write_csv(output / "dog_feature_base_manifest.csv", dog_rows_by_split["train"], feature_fields)
    write_csv(output / "dog_feature_candidate_manifest.csv", dog_rows_by_split["candidate"], feature_fields)
    write_json(output / "materialize_report.json", {
        "status": "complete",
        "dog_feature_targets": len(dog_rows),
        "dog_feature_base_targets": len(dog_rows_by_split["train"]),
        "dog_feature_candidate_targets": len(dog_rows_by_split["candidate"]),
        "crop_padding": padding,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("metadata", "plan", "download", "materialize", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "frozen_config.json", config)
    if args.command in {"metadata", "all"}:
        filter_annotations(config, args.output, args.force)
    if args.command in {"plan", "all"}:
        build_plan(config, args.output)
    if args.command in {"download", "all"}:
        download_images(args.output, args.workers, args.force)
    if args.command in {"materialize", "all"}:
        materialize(args.output, args.padding)


if __name__ == "__main__":
    main()
