#!/usr/bin/env python3
"""Build an image-disjoint, balanced five-class full-object candidate pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
OPENIMAGES = SOURCE / "data/openimages_v7_dog_gap_v1"
VOC2012 = SOURCE / "data/pascal_voc2012_dog_validation_v1/raw/extracted/VOCdevkit/VOC2012"
DEFAULT_OUTPUT = ROOT / "artifacts/multiclass_object_pool_v1_20260928"
SEED = 20260928
CLASSES = ("dog", "cat", "horse", "sheep", "person")
TARGET_PER_CLASS = 400
NEW_PER_CONTROL_CLASS = 300

sys.path.insert(0, str(SOURCE / "scripts"))
from prepare_openimages_v7_experiment import (  # noqa: E402
    absolute_box,
    image_targets,
    load_json,
    manifest_row,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    known: set[str] = set()
    for row in rows:
        for field in row:
            if field not in known:
                fields.append(field)
                known.add(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def area(row: dict[str, str]) -> float:
    return (float(row["XMax"]) - float(row["XMin"])) * (
        float(row["YMax"]) - float(row["YMin"])
    )


def all_used_openimages_ids() -> set[str]:
    used: set[str] = set()
    for path in sorted((OPENIMAGES / "manifests").glob("*.csv")):
        used.update(row["image_id"] for row in read_csv(path) if row.get("image_id"))
    expansion = ROOT / "artifacts/candidate_pool_expansion_v2_20260928/cat/manifests/cat_candidates_expansion.csv"
    if expansion.is_file():
        used.update(row["image_id"] for row in read_csv(expansion))
    return used


def stratified_select(
    rows: list[dict[str, str]], class_name: str, count: int, used: set[str]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    pool = [row for row in rows if row["ImageID"] not in used]
    pool.sort(key=lambda row: (
        int(row.get("IsDepiction", "0") != "0"),
        stable_key(SEED, class_name, row["_source_split"], row["ImageID"]),
    ))
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    stages: Counter[str] = Counter()

    def take(name: str, predicate, limit: int) -> None:
        for row in pool:
            if len(selected) >= count or stages[name] >= limit:
                break
            if row["ImageID"] in selected_ids or not predicate(row):
                continue
            selected.append(row)
            selected_ids.add(row["ImageID"])
            stages[name] += 1

    take("tiny_lt_1pct", lambda row: area(row) < 0.01, 15)
    take("small_1_to_3pct", lambda row: 0.01 <= area(row) < 0.03, 60)
    take("occluded_and_truncated", lambda row: row["IsOccluded"] == "1" and row["IsTruncated"] == "1", 30)
    take("occluded", lambda row: row["IsOccluded"] == "1", 60)
    take("truncated", lambda row: row["IsTruncated"] == "1", 60)
    take("diversity_fill", lambda row: True, count - len(selected))
    if len(selected) != count:
        raise ValueError(f"{class_name}: need {count}, selected {len(selected)}")
    used.update(selected_ids)
    return selected, dict(stages)


def sheep_targets_with_depictions(
    rows: list[dict[str, str]], sheep_mid: str
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["LabelName"] != sheep_mid:
            continue
        if float(row["Confidence"]) < 1 or row["IsGroupOf"] != "0" or row["IsInside"] != "0":
            continue
        if area(row) < 0.0001:
            continue
        grouped[row["ImageID"]].append(row)
    result = []
    for boxes in grouped.values():
        target = max(
            boxes,
            key=lambda row: (row["IsDepiction"] == "0", area(row)),
        )
        item = dict(target)
        item["class_name"] = "sheep"
        item["same_class_box_count"] = len(boxes)
        result.append(item)
    return result


def voc_sheep_rows(count: int, config: dict) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for annotation_path in sorted((VOC2012 / "Annotations").glob("*.xml")):
        root = ET.parse(annotation_path).getroot()
        objects = [
            item for item in root.findall("object")
            if item.findtext("name") == "sheep" and item.findtext("difficult", "0") == "0"
        ]
        if not objects:
            continue
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        if width <= 0 or height <= 0:
            continue

        def object_area(item: ET.Element) -> int:
            box = item.find("bndbox")
            if box is None:
                return 0
            return (int(box.findtext("xmax", "0")) - int(box.findtext("xmin", "0"))) * (
                int(box.findtext("ymax", "0")) - int(box.findtext("ymin", "0"))
            )

        target = max(objects, key=object_area)
        box = target.find("bndbox")
        if box is None:
            continue
        x1 = max(0, int(box.findtext("xmin", "1")) - 1)
        y1 = max(0, int(box.findtext("ymin", "1")) - 1)
        x2 = min(width, int(box.findtext("xmax", str(width))))
        y2 = min(height, int(box.findtext("ymax", str(height))))
        if x2 <= x1 or y2 <= y1:
            continue
        stem = annotation_path.stem
        image_path = VOC2012 / "JPEGImages" / f"{stem}.jpg"
        if not image_path.is_file():
            continue
        image_id = f"voc2012-{stem}"
        candidates.append({
            "sample_id": f"{image_id}-sheep",
            "image_id": image_id,
            "target_instance_id": "target_voc_sheep",
            "class_name": "sheep",
            "label": config["class_labels"]["sheep"],
            "source_split": "voc2012",
            "bbox_norm_xyxy": json.dumps([x1 / width, y1 / height, x2 / width, y2 / height], separators=(",", ":")),
            "bbox_area_ratio": f"{((x2 - x1) * (y2 - y1) / (width * height)):.8f}",
            "is_occluded": "",
            "is_truncated": target.findtext("truncated", "0"),
            "is_group_of": "0",
            "is_depiction": "0",
            "is_inside": "0",
            "split": "multiclass_candidate_v1",
            "dataset_version": "pascal-voc2012",
            "image_path": str(image_path.resolve()),
            "crop_path": "",
            "marked_image_path": "",
            "feature_crop_path": "",
            "image_sha256": "",
            "crop_sha256": "",
            "candidate_source": "pascal_voc2012_sheep_fallback",
        })
    candidates.sort(key=lambda row: stable_key(SEED, "voc-sheep", row["image_id"]))
    if len(candidates) < count:
        raise ValueError(f"VOC2012 sheep fallback needs {count}, found {len(candidates)}")
    return candidates[:count]


def download_one(row: dict[str, object]) -> tuple[str, str]:
    split = str(row["source_split"])
    image_id = str(row["image_id"])
    target = OPENIMAGES / "images" / split / f"{image_id}.jpg"
    if target.is_file() and target.stat().st_size:
        try:
            with Image.open(target) as image:
                image.load()
            return image_id, "reused"
        except OSError:
            target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".jpg.part")
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "QE-multiclass-pool/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            partial.replace(target)
            with Image.open(target) as image:
                image.load()
            return image_id, "downloaded"
        except Exception as error:
            last_error = error
            partial.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt * 2)
    return image_id, f"error:{last_error}"


def ensure_views(row: dict[str, object], output: Path) -> dict[str, object]:
    image_path = Path(str(row["image_path"]))
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    norm = json.loads(str(row["bbox_norm_xyxy"]))
    target_box = absolute_box(norm, image.width, image.height, 0)
    crop_box = absolute_box(norm, image.width, image.height, 0.05)
    sample_id = str(row["sample_id"])
    class_name = str(row["class_name"])
    existing_crop = Path(str(row.get("crop_path", "")))
    crop_path = existing_crop if existing_crop.is_file() else output / "rendered/crops" / class_name / f"{sample_id}.jpg"
    marked_path = Path(str(row.get("marked_image_path", "")))
    if not marked_path.is_file():
        marked_path = output / "rendered/marked" / class_name / f"{sample_id}.jpg"
    feature_crop_path = Path(str(row.get("feature_crop_path", "")))
    if not feature_crop_path.is_file():
        feature_crop_path = output / "rendered/feature_crops" / class_name / f"{sample_id}.jpg"
    for path in (crop_path, marked_path, feature_crop_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not crop_path.is_file():
        image.crop(crop_box).save(crop_path, quality=95)
    if not marked_path.is_file():
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        width = max(3, min(image.size) // 150)
        draw.rectangle(target_box, outline=(220, 48, 47), width=width)
        draw.text((target_box[0] + 4, target_box[1] + 4), "[A]", fill=(220, 48, 47), stroke_width=2, stroke_fill="white")
        marked.save(marked_path, quality=92)
    if not feature_crop_path.is_file():
        target = image.crop(target_box)
        scale = min(960 / target.width, 960 / target.height)
        resized = target.resize(
            (max(1, round(target.width * scale)), max(1, round(target.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (1024, 1024), "white")
        canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
        canvas.save(feature_crop_path, quality=95)
    row.update({
        "crop_path": str(crop_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "feature_crop_path": str(feature_crop_path.resolve()),
        "image_sha256": row.get("image_sha256") or sha256(image_path),
        "crop_sha256": row.get("crop_sha256") or sha256(crop_path),
    })
    return row


def prepare(output: Path, workers: int) -> None:
    config = load_json(SOURCE / "configs/openimages_v7_dog_gap_experiment_v1.json")
    old_pool_path = SOURCE / "artifacts/fixed_train_cat_visual_test_adequacy_20260920/candidate_pool.csv"
    cat_new_path = ROOT / "artifacts/candidate_pool_expansion_v2_20260928/cat/manifests/cat_candidates_expansion.csv"
    dog_source_path = OPENIMAGES / "manifests/dog_candidates.csv"
    old_pool = read_csv(old_pool_path)
    cat_new = read_csv(cat_new_path)
    dog_source = read_csv(dog_source_path)

    chosen: list[dict[str, object]] = []
    provenance: Counter[str] = Counter()
    for row in old_pool:
        item: dict[str, object] = dict(row)
        item["candidate_source"] = "frozen_openimages_test_pool"
        chosen.append(item)
        provenance[str(item["candidate_source"])] += 1
    for row in cat_new:
        item = dict(row)
        item["candidate_source"] = "cat_expansion_v2"
        chosen.append(item)
        provenance[str(item["candidate_source"])] += 1

    dog_pool = sorted(dog_source, key=lambda row: stable_key(SEED, "dog-existing", row["image_id"]))
    dog_selected = dog_pool[: TARGET_PER_CLASS - 100]
    for row in dog_selected:
        item = dict(row)
        item["candidate_source"] = "frozen_openimages_dog_candidates"
        chosen.append(item)
        provenance[str(item["candidate_source"])] += 1

    targets: dict[str, list[dict[str, str]]] = {name: [] for name in CLASSES}
    for source_split in ("train", "validation", "test"):
        annotations = read_csv(OPENIMAGES / "annotations" / f"{source_split}_five_classes.csv")
        split_targets = image_targets(
            annotations,
            {mid: name for name, mid in config["classes"].items()},
            config["box_filters"],
        )
        for class_name in CLASSES:
            candidates = (
                sheep_targets_with_depictions(annotations, config["classes"]["sheep"])
                if class_name == "sheep"
                else split_targets[class_name]
            )
            for target in candidates:
                target["_source_split"] = source_split
                targets[class_name].append(target)
    used = all_used_openimages_ids()
    new_rows: list[dict[str, object]] = []
    stage_audit: dict[str, dict[str, int]] = {}
    for class_name in ("sheep", "horse", "person"):
        if class_name == "sheep":
            natural = [row for row in targets[class_name] if row.get("IsDepiction") == "0"]
            depictions = [row for row in targets[class_name] if row.get("IsDepiction") == "1"]
            natural_count = min(
                NEW_PER_CONTROL_CLASS,
                len({row["ImageID"] for row in natural if row["ImageID"] not in used}),
            )
            selected, stages = stratified_select(natural, class_name, natural_count, used)
            depiction_count = min(
                30,
                NEW_PER_CONTROL_CLASS - len(selected),
                len({row["ImageID"] for row in depictions if row["ImageID"] not in used}),
            )
            if depiction_count:
                depiction_selected, _ = stratified_select(
                    depictions, f"{class_name}-depiction", depiction_count, used
                )
                selected.extend(depiction_selected)
        else:
            selected, stages = stratified_select(
                targets[class_name], class_name, NEW_PER_CONTROL_CLASS, used
            )
        stages["selected_depictions"] = sum(row.get("IsDepiction") == "1" for row in selected)
        stages["selected_openimages"] = len(selected)
        stage_audit[class_name] = stages
        for index, target in enumerate(selected, 1):
            item = manifest_row(target, config, target["_source_split"], "multiclass_candidate_v1", index)
            item["candidate_source"] = "multiclass_expansion_v1"
            new_rows.append(item)

    sheep_shortfall = NEW_PER_CONTROL_CLASS - stage_audit["sheep"]["selected_openimages"]
    voc_rows = voc_sheep_rows(sheep_shortfall, config)
    stage_audit["sheep"]["selected_voc2012"] = len(voc_rows)
    new_rows.extend(voc_rows)

    statuses: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloadable = [row for row in new_rows if row["source_split"] in {"train", "validation", "test"}]
        futures = [executor.submit(download_one, row) for row in downloadable]
        for index, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            statuses[status.split(":", 1)[0]] += 1
            if status.startswith("error:"):
                raise RuntimeError(status)
            if index % 50 == 0 or index == len(downloadable):
                print(f"[download] {index}/{len(downloadable)} {dict(statuses)}", flush=True)
    for row in new_rows:
        if row["source_split"] in {"train", "validation", "test"}:
            row["image_path"] = str((OPENIMAGES / "images" / str(row["source_split"]) / f"{row['image_id']}.jpg").resolve())
        chosen.append(row)
        provenance[str(row["candidate_source"])] += 1

    counts_before = Counter(str(row["class_name"]) for row in chosen)
    if counts_before != Counter({name: TARGET_PER_CLASS for name in CLASSES}):
        raise ValueError(f"unbalanced pool: {counts_before}")
    source_ids = [str(row["image_id"]) for row in chosen]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source-image overlap across candidate classes")

    rendered: list[dict[str, object]] = []
    for index, row in enumerate(chosen, 1):
        rendered.append(ensure_views(row, output))
        if index % 100 == 0 or index == len(chosen):
            print(f"[render] {index}/{len(chosen)}", flush=True)
    rendered.sort(key=lambda row: (CLASSES.index(str(row["class_name"])), stable_key(SEED, row["sample_id"])))
    class_ranks: Counter[str] = Counter()
    for row in rendered:
        class_name = str(row["class_name"])
        class_ranks[class_name] += 1
        row["class_candidate_rank"] = class_ranks[class_name]
        row["pool_protocol"] = "multiclass_object_pool_v1"
    write_csv(output / "manifests/candidate_pool.csv", rendered)

    feature_rows = []
    for row in rendered:
        with Image.open(str(row["image_path"])) as image:
            box = absolute_box(json.loads(str(row["bbox_norm_xyxy"])), image.width, image.height, 0)
        feature_rows.append({
            "image_id": row["sample_id"],
            "target_instance_id": "target_1",
            "target_class_id": config["classes"][str(row["class_name"])],
            "target_class": row["class_name"],
            "bbox": json.dumps(box, separators=(",", ":")),
            "image_path": row["image_path"],
            "marked_image_path": row["marked_image_path"],
            "crop_image_path": row["feature_crop_path"],
            "split": "candidate",
            "image_sha256": row["image_sha256"],
            "label_sha256": "",
            "protocol_version": "object_state_v1.0-openimages-v1",
        })
    write_csv(output / "inputs/object_state_feature_manifest.csv", feature_rows)

    train_ids = {
        row["image_id"]
        for row in read_csv(OPENIMAGES / "manifests/train_base.csv")
    }
    report = {
        "version": "multiclass_object_pool_v1_20260928",
        "created_at": now(),
        "classes": list(CLASSES),
        "target_per_class": TARGET_PER_CLASS,
        "total_candidates": len(rendered),
        "class_counts": dict(Counter(str(row["class_name"]) for row in rendered)),
        "candidate_sources": dict(provenance),
        "new_selection_stages": stage_audit,
        "download_status": dict(statuses),
        "unique_source_images": len(set(source_ids)),
        "train_source_overlap": len(train_ids & set(source_ids)),
        "annotation_protocol": "object_state_v1.0",
        "annotation_status": "pending",
        "face_in_primary_pool": False,
    }
    write_json(output / "protocol.json", report)
    if report["train_source_overlap"]:
        raise ValueError(f"candidate/train leakage: {report['train_source_overlap']}")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.output, args.workers)


if __name__ == "__main__":
    main()
