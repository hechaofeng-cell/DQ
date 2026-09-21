#!/usr/bin/env python3
"""Build an independent PASCAL VOC 2012 dog validation candidate pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/pascal_voc2012_dog_validation_v1"
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize(
            (9, 8), Image.Resampling.LANCZOS
        )
    pixels = list(image.getdata())
    value = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(output: Path) -> Path:
    archive = output / "raw" / Path(VOC_URL).name
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl", "--location", "--fail", "--retry", "5", "--retry-delay", "5",
            "--continue-at", "-", "--output", str(archive), VOC_URL,
        ],
        check=True,
    )
    write_json(output / "archive_provenance.json", {
        "source_url": VOC_URL,
        "archive_path": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_hash(archive),
    })
    return archive


def extract(archive: Path, output: Path) -> Path:
    extracted = output / "raw" / "extracted"
    voc_root = extracted / "VOCdevkit" / "VOC2012"
    required = voc_root / "Annotations"
    if required.is_dir():
        return voc_root
    extracted.mkdir(parents=True, exist_ok=True)
    destination = extracted.resolve()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            member_path = (extracted / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f"unsafe archive member: {member.name}")
        bundle.extractall(extracted)
    if not required.is_dir():
        raise FileNotFoundError(f"VOC annotations missing after extraction: {required}")
    return voc_root


def reference_images() -> list[tuple[str, Path]]:
    roots = [
        ("coco2017", ROOT / "data/coco2017/coco500/images"),
        ("coco2017", ROOT / "data/coco2017/dog500/images"),
        ("coco2017", ROOT / "data/coco2017/dog500_incomplete_20260911/images"),
        ("coco2017", ROOT / "data/coco2017/dog_candidate400_20260912/images"),
        ("coco2017", ROOT / "data/coco2017/selected_val_images"),
        ("openimages_v7", ROOT / "data/openimages_v7_dog_gap_v1/images"),
    ]
    seen: set[Path] = set()
    result = []
    for source, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            resolved = path.resolve()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and resolved not in seen:
                result.append((source, resolved))
                seen.add(resolved)
    return result


def build_reference_index(output: Path) -> list[dict]:
    cache = output / "audit" / "reference_image_hashes.csv"
    sources = reference_images()
    expected_paths = {str(path) for _, path in sources}
    cached: dict[str, dict] = {}
    if cache.is_file():
        with cache.open(encoding="utf-8-sig", newline="") as stream:
            cached = {row["path"]: row for row in csv.DictReader(stream)}
    rows = []
    for index, (source, path) in enumerate(sources, start=1):
        key = str(path)
        row = cached.get(key)
        if row and int(row.get("size_bytes", -1)) == path.stat().st_size:
            rows.append(row)
            continue
        try:
            rows.append({
                "source": source,
                "path": key,
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "dhash64": f"{difference_hash(path):016x}",
                "status": "ok",
            })
        except Exception as error:
            rows.append({
                "source": source,
                "path": key,
                "size_bytes": path.stat().st_size,
                "sha256": "",
                "dhash64": "",
                "status": f"error:{type(error).__name__}",
            })
        if index % 1000 == 0:
            print(f"hashed reference images: {index}/{len(sources)}", flush=True)
    rows = [row for row in rows if row["path"] in expected_paths]
    write_csv(
        cache, rows,
        ["source", "path", "size_bytes", "sha256", "dhash64", "status"],
    )
    return [row for row in rows if row["status"] == "ok"]


def int_value(node: ET.Element, path: str, default: int = 0) -> int:
    text = node.findtext(path)
    return int(text) if text is not None else default


def parse_dog_targets(voc_root: Path) -> tuple[list[dict], dict]:
    rows = []
    counters = Counter()
    image_set_path = voc_root / "ImageSets" / "Main" / "dog_trainval.txt"
    image_set_rows = [line.split() for line in image_set_path.read_text().splitlines() if line.strip()]
    positive_ids = [image_id for image_id, label in image_set_rows if label == "1"]
    counters["trainval_image_set_rows"] = len(image_set_rows)
    counters["official_positive_dog_images"] = len(positive_ids)
    for image_id in sorted(positive_ids):
        annotation_path = voc_root / "Annotations" / f"{image_id}.xml"
        root = ET.parse(annotation_path).getroot()
        dogs = [obj for obj in root.findall("object") if obj.findtext("name") == "dog"]
        if not dogs:
            raise ValueError(f"official positive dog image has no dog annotation: {image_id}")
        counters["images_with_dog"] += 1
        counters["dog_instances"] += len(dogs)
        usable = [obj for obj in dogs if int_value(obj, "difficult") == 0]
        counters["non_difficult_dog_instances"] += len(usable)
        if not usable:
            counters["difficult_only_images"] += 1
            continue

        def area(obj: ET.Element) -> int:
            box = obj.find("bndbox")
            if box is None:
                return 0
            return max(0, int_value(box, "xmax") - int_value(box, "xmin")) * max(
                0, int_value(box, "ymax") - int_value(box, "ymin")
            )

        target = max(usable, key=area)
        box = target.find("bndbox")
        if box is None:
            counters["missing_box"] += 1
            continue
        filename = root.findtext("filename") or f"{image_id}.jpg"
        image_path = voc_root / "JPEGImages" / filename
        if not image_path.is_file():
            counters["missing_image"] += 1
            continue
        width = int_value(root, "size/width")
        height = int_value(root, "size/height")
        x1 = max(0, int_value(box, "xmin") - 1)
        y1 = max(0, int_value(box, "ymin") - 1)
        x2 = min(width, int_value(box, "xmax"))
        y2 = min(height, int_value(box, "ymax"))
        if x2 <= x1 or y2 <= y1:
            counters["invalid_box"] += 1
            continue
        rows.append({
            "sample_id": f"voc2012_dog_{image_id}",
            "source_dataset": "PASCAL VOC 2012 trainval",
            "source_split": "trainval",
            "source_image_id": image_id,
            "image_path": str(image_path.resolve()),
            "bbox_xyxy": json.dumps([x1, y1, x2, y2], separators=(",", ":")),
            "bbox_area_ratio": (x2 - x1) * (y2 - y1) / (width * height),
            "voc_pose": target.findtext("pose") or "Unspecified",
            "voc_truncated": int_value(target, "truncated"),
            "voc_difficult": int_value(target, "difficult"),
            "dog_instances_in_image": len(dogs),
        })
    counters["selected_largest_non_difficult_targets"] = len(rows)
    return rows, dict(counters)


def materialize(output: Path, voc_root: Path, near_threshold: int) -> None:
    references = build_reference_index(output)
    exact_index = {row["sha256"]: row for row in references}
    perceptual = [(int(row["dhash64"], 16), row) for row in references]
    targets, source_counts = parse_dog_targets(voc_root)
    all_rows = []
    frozen_rows = []
    status_counts = Counter()
    for index, row in enumerate(targets, start=1):
        image_path = Path(row["image_path"])
        sha256 = file_hash(image_path)
        dhash = difference_hash(image_path)
        exact = exact_index.get(sha256)
        nearest_distance = 65
        nearest = None
        if exact is None:
            for reference_hash, reference_row in perceptual:
                distance = (dhash ^ reference_hash).bit_count()
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = reference_row
                    if distance == 0:
                        break
        if exact is not None:
            status = "excluded_exact_duplicate"
            match = exact
            distance_value = 0
        elif nearest is not None and nearest_distance <= near_threshold:
            status = "held_near_duplicate"
            match = nearest
            distance_value = nearest_distance
        else:
            status = "eligible"
            match = None
            distance_value = nearest_distance if nearest is not None else ""
        status_counts[status] += 1
        enriched = {
            **row,
            "image_sha256": sha256,
            "image_dhash64": f"{dhash:016x}",
            "duplicate_status": status,
            "nearest_reference_source": match["source"] if match else "",
            "nearest_reference_path": match["path"] if match else "",
            "nearest_reference_dhash_distance": distance_value,
            "candidate_status": "frozen_candidate" if status == "eligible" else "excluded_pending_audit",
        }
        all_rows.append(enriched)
        if status != "eligible":
            continue
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        x1, y1, x2, y2 = json.loads(row["bbox_xyxy"])
        marked_path = output / "tri_view" / "marked" / f"{row['sample_id']}.jpg"
        crop_path = output / "tri_view" / "crops" / f"{row['sample_id']}.jpg"
        marked_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        line_width = max(3, min(image.size) // 150)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=line_width)
        draw.text((x1 + 4, y1 + 4), "[A]", fill=(255, 0, 0), stroke_width=2, stroke_fill="white")
        marked.save(marked_path, quality=95)
        target_crop = image.crop((x1, y1, x2, y2))
        scale = min(960 / target_crop.width, 960 / target_crop.height)
        resized = target_crop.resize(
            (max(1, round(target_crop.width * scale)), max(1, round(target_crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (1024, 1024), "white")
        canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
        canvas.save(crop_path, quality=95)
        frozen_rows.append({
            "image_id": row["sample_id"],
            "target_instance_id": "target_1",
            "target_class_id": "voc:dog",
            "target_class": "dog",
            "bbox": row["bbox_xyxy"],
            "image_path": str(image_path.resolve()),
            "marked_image_path": str(marked_path.resolve()),
            "crop_image_path": str(crop_path.resolve()),
            "split": "independent_candidate",
            "image_sha256": sha256,
            "label_sha256": file_hash(voc_root / "Annotations" / f"{row['source_image_id']}.xml"),
            "protocol_version": "dog_v3.1-pascal-voc2012-independent-v1",
        })
        if index % 100 == 0:
            print(f"audited VOC dog targets: {index}/{len(targets)}", flush=True)

    all_fields = list(all_rows[0])
    write_csv(output / "manifests" / "all_dog_targets_audit.csv", all_rows, all_fields)
    feature_fields = list(frozen_rows[0])
    feature_manifest = output / "manifests" / "dog_feature_candidate_manifest.csv"
    write_csv(feature_manifest, frozen_rows, feature_fields)
    write_json(output / "report.json", {
        "source": "PASCAL VOC 2012 trainval",
        "source_url": VOC_URL,
        "selection_rule": "largest non-difficult dog instance per image",
        "model_outputs_used_for_selection": False,
        "near_duplicate_metric": "64-bit difference hash on full source image",
        "near_duplicate_hold_threshold": near_threshold,
        "source_counts": source_counts,
        "reference_image_count": len(references),
        "duplicate_status_counts": dict(status_counts),
        "frozen_candidate_count": len(frozen_rows),
        "feature_manifest": str(feature_manifest.resolve()),
    })
    print(json.dumps({
        "frozen_candidate_count": len(frozen_rows),
        "duplicate_status_counts": dict(status_counts),
        "feature_manifest": str(feature_manifest),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "materialize", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-threshold", type=int, default=3)
    args = parser.parse_args()
    archive = args.output / "raw" / Path(VOC_URL).name
    if args.command in {"download", "all"}:
        archive = download(args.output)
    if args.command in {"materialize", "all"}:
        if not archive.is_file():
            raise FileNotFoundError(f"download archive first: {archive}")
        materialize(args.output, extract(archive, args.output), args.near_threshold)


if __name__ == "__main__":
    main()
