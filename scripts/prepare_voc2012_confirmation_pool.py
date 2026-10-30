#!/usr/bin/env python3
"""Build a balanced cross-source PASCAL VOC 2012 confirmation pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
ARCHIVE = SOURCE / "data/pascal_voc2012_dog_validation_v1/raw/VOCtrainval_11-May-2012.tar"
CURRENT_POOL = ROOT / "artifacts/multiclass_object_pool_v1_20260928/manifests/candidate_pool.csv"
TRAIN_MANIFEST = SOURCE / "artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/voc2012_confirmation_pool_20260929"
CLASSES = ("dog", "cat", "horse", "sheep", "person")
LABELS = {name: index for index, name in enumerate(CLASSES)}
TARGET_PER_CLASS = 100
SEED = 20260929


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def parse_candidates(archive: tarfile.TarFile) -> dict[str, list[dict]]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    members = [member for member in archive.getmembers() if "/Annotations/" in member.name and member.name.endswith(".xml")]
    for member in members:
        stream = archive.extractfile(member)
        if stream is None:
            continue
        root = ET.fromstring(stream.read())
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        if width <= 0 or height <= 0:
            continue
        grouped: dict[str, list[ET.Element]] = defaultdict(list)
        for item in root.findall("object"):
            name = item.findtext("name", "")
            if name in CLASSES and item.findtext("difficult", "0") == "0":
                grouped[name].append(item)
        for class_name, objects in grouped.items():
            def object_area(item: ET.Element) -> int:
                box = item.find("bndbox")
                if box is None:
                    return 0
                return max(0, round(float(box.findtext("xmax", "0"))) - round(float(box.findtext("xmin", "0")))) * max(
                    0, round(float(box.findtext("ymax", "0"))) - round(float(box.findtext("ymin", "0")))
                )

            target = max(objects, key=object_area)
            box = target.find("bndbox")
            if box is None:
                continue
            x1 = max(0, round(float(box.findtext("xmin", "1"))) - 1)
            y1 = max(0, round(float(box.findtext("ymin", "1"))) - 1)
            x2 = min(width, round(float(box.findtext("xmax", str(width)))))
            y2 = min(height, round(float(box.findtext("ymax", str(height)))))
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1) / (width * height)
            by_class[class_name].append({
                "stem": Path(member.name).stem,
                "class_name": class_name,
                "bbox": [x1, y1, x2, y2],
                "bbox_area_ratio": area,
                "truncated": int(target.findtext("truncated", "0")),
                "same_class_count": len(objects),
                "width": width,
                "height": height,
            })
    return by_class


def candidate_priority(row: dict) -> tuple:
    area = row["bbox_area_ratio"]
    area_bin = 0 if area < .03 else 1 if area < .10 else 2 if area < .35 else 3
    hard = int(row["truncated"] or row["same_class_count"] > 1)
    return (area_bin, -hard, stable_key(SEED, row["class_name"], row["stem"]))


def select_balanced(candidates: dict[str, list[dict]], count: int) -> dict[str, list[dict]]:
    selected: dict[str, list[dict]] = {}
    used_stems: set[str] = set()
    for class_name in CLASSES:
        groups: dict[int, list[dict]] = defaultdict(list)
        for row in candidates[class_name]:
            area = row["bbox_area_ratio"]
            groups[0 if area < .03 else 1 if area < .10 else 2 if area < .35 else 3].append(row)
        for rows in groups.values():
            rows.sort(key=candidate_priority)
        chosen: list[dict] = []
        # Round-robin across scale bins prevents the pool from being dominated by large targets.
        while len(chosen) < count and any(groups.values()):
            for bin_id in (0, 1, 2, 3):
                while groups[bin_id] and groups[bin_id][0]["stem"] in used_stems:
                    groups[bin_id].pop(0)
                if groups[bin_id] and len(chosen) < count:
                    item = groups[bin_id].pop(0)
                    chosen.append(item)
                    used_stems.add(item["stem"])
        if len(chosen) != count:
            raise ValueError(f"{class_name}: selected {len(chosen)}, expected {count}")
        selected[class_name] = chosen
    return selected


def render(
    output: Path, archive: tarfile.TarFile, selected: dict[str, list[dict]], target_per_class: int
) -> tuple[list[dict], int]:
    current_hashes = {row["image_sha256"] for row in read_csv(CURRENT_POOL) if row.get("image_sha256")}
    train_crop_hashes = {
        digest_path(Path(row["crop_path"])) for row in read_csv(TRAIN_MANIFEST)
        if row.get("crop_path") and Path(row["crop_path"]).is_file()
    }
    rows: list[dict] = []
    rejected_exact = 0
    for class_name in CLASSES:
        rank = 0
        candidates = selected[class_name]
        # Add deterministic reserves in case an exact duplicate is detected.
        for item in candidates:
            member_name = f"VOCdevkit/VOC2012/JPEGImages/{item['stem']}.jpg"
            source = archive.extractfile(member_name)
            if source is None:
                continue
            data = source.read()
            image_hash = digest_bytes(data)
            if image_hash in current_hashes:
                rejected_exact += 1
                continue
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            x1, y1, x2, y2 = item["bbox"]
            dx, dy = (x2 - x1) * .05, (y2 - y1) * .05
            crop_box = (max(0, x1 - dx), max(0, y1 - dy), min(image.width, x2 + dx), min(image.height, y2 + dy))
            target = image.crop(crop_box)
            target_buffer = io.BytesIO()
            target.save(target_buffer, format="JPEG", quality=95)
            crop_hash = digest_bytes(target_buffer.getvalue())
            if crop_hash in train_crop_hashes:
                rejected_exact += 1
                continue
            rank += 1
            sample_id = f"voc2012-{item['stem']}-{class_name}"
            image_path = output / "images" / f"{item['stem']}.jpg"
            crop_path = output / "rendered/crops" / class_name / f"{sample_id}.jpg"
            marked_path = output / "rendered/marked" / class_name / f"{sample_id}.jpg"
            feature_path = output / "rendered/feature_crops" / class_name / f"{sample_id}.jpg"
            for path in (image_path, crop_path, marked_path, feature_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                image_path.write_bytes(data)
            crop_path.write_bytes(target_buffer.getvalue())
            marked = image.copy()
            draw = ImageDraw.Draw(marked)
            line_width = max(3, min(image.size) // 150)
            draw.rectangle((x1, y1, x2, y2), outline=(220, 48, 47), width=line_width)
            draw.text((x1 + 4, y1 + 4), "[A]", fill=(220, 48, 47), stroke_width=2, stroke_fill="white")
            marked.save(marked_path, quality=92)
            scale = min(960 / target.width, 960 / target.height)
            resized = target.resize((max(1, round(target.width * scale)), max(1, round(target.height * scale))), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1024, 1024), "white")
            canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
            canvas.save(feature_path, quality=95)
            rows.append({
                "sample_id": sample_id,
                "image_id": f"voc2012-{item['stem']}",
                "target_instance_id": f"target_voc_{class_name}",
                "class_name": class_name,
                "label": LABELS[class_name],
                "source_split": "voc2012_trainval",
                "bbox_norm_xyxy": json.dumps([x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height], separators=(",", ":")),
                "bbox_area_ratio": f"{item['bbox_area_ratio']:.8f}",
                "is_occluded": "",
                "is_truncated": item["truncated"],
                "is_group_of": 0,
                "is_depiction": 0,
                "is_inside": 0,
                "split": "confirmation_candidate",
                "dataset_version": "pascal-voc2012-trainval",
                "image_path": str(image_path.resolve()),
                "crop_path": str(crop_path.resolve()),
                "marked_image_path": str(marked_path.resolve()),
                "feature_crop_path": str(feature_path.resolve()),
                "image_sha256": image_hash,
                "crop_sha256": crop_hash,
                "candidate_source": "pascal_voc2012_confirmation",
                "class_candidate_rank": rank,
                "pool_protocol": "voc2012_confirmation_pool_v1",
            })
            if rank == target_per_class:
                break
        if rank != target_per_class:
            raise ValueError(f"{class_name}: only {rank} after duplicate filtering")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("source image reused across classes")
    return rows, rejected_exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=TARGET_PER_CLASS)
    args = parser.parse_args()
    target_per_class = args.per_class
    output = args.output.resolve()
    with tarfile.open(ARCHIVE, "r") as archive:
        candidates = parse_candidates(archive)
        # Select reserves so exact-hash exclusions cannot underfill the pool.
        selected = select_balanced(candidates, target_per_class + 100)
        rows, rejected = render(output, archive, selected, target_per_class)
    rows.sort(key=lambda row: (CLASSES.index(row["class_name"]), int(row["class_candidate_rank"])))
    write_csv(output / "manifests/candidate_pool.csv", rows)
    inputs = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            norm = json.loads(row["bbox_norm_xyxy"])
            box = [round(norm[0] * image.width), round(norm[1] * image.height), round(norm[2] * image.width), round(norm[3] * image.height)]
        inputs.append({
            "image_id": row["sample_id"], "target_instance_id": "target_1",
            "target_class_id": row["class_name"], "target_class": row["class_name"],
            "bbox": json.dumps(box, separators=(",", ":")), "image_path": row["image_path"],
            "marked_image_path": row["marked_image_path"], "crop_image_path": row["feature_crop_path"],
            "split": "confirmation_candidate", "image_sha256": row["image_sha256"],
            "label_sha256": "", "protocol_version": "object_state_v1.0-voc2012-confirmation-v1",
        })
    write_csv(output / "inputs/object_state_feature_manifest.csv", inputs)
    audit = {
        "status": "complete", "source": "PASCAL VOC 2012 trainval", "classes": list(CLASSES),
        "per_class": target_per_class, "total": len(rows),
        "counts": dict(Counter(row["class_name"] for row in rows)),
        "unique_source_images": len({row["image_id"] for row in rows}),
        "unique_image_hashes": len({row["image_sha256"] for row in rows}),
        "unique_crop_hashes": len({row["crop_sha256"] for row in rows}),
        "exact_hash_rejections": rejected,
        "current_pool_exact_image_overlap": 0,
        "training_exact_crop_overlap": 0,
        "archive_sha256": digest_path(ARCHIVE),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
