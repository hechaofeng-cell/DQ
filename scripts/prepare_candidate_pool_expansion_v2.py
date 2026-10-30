#!/usr/bin/env python3
"""Expand the cat pool and build a confusable WIDER FACE binary pool.

The existing experiments are immutable. This script writes a new versioned
artifact and only reuses the Open Images raw-image cache.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
OPENIMAGES = SOURCE / "data/openimages_v7_dog_gap_v1"
WIDER = SOURCE / "data/widerface_head_v1/raw/widerface"
DEFAULT_OUTPUT = ROOT / "artifacts/candidate_pool_expansion_v2_20260928"
SEED = 20260928
CAT_MID = "/m/01yrx"

sys.path.insert(0, str(SOURCE / "scripts"))
from prepare_openimages_v7_experiment import (  # noqa: E402
    absolute_box,
    image_targets,
    load_json,
    manifest_row,
)

sys.path.insert(0, str(ROOT / "scripts"))
from prepare_widerface_head_experiment import (  # noqa: E402
    FaceTarget,
    expanded_head_box,
    one_target_per_image,
    parse_annotations,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    known = set(fields)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fields.append(field)
                known.add(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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


def cat_area(row: dict[str, str]) -> float:
    return (float(row["XMax"]) - float(row["XMin"])) * (
        float(row["YMax"]) - float(row["YMin"])
    )


def used_openimages_ids() -> set[str]:
    used: set[str] = set()
    for path in sorted((OPENIMAGES / "manifests").glob("*.csv")):
        for row in read_csv(path):
            image_id = row.get("image_id", "")
            if image_id:
                used.add(image_id)
    return used


def choose_cat_targets(rows: list[dict[str, str]], count: int) -> tuple[list[dict[str, str]], dict]:
    used = used_openimages_ids()
    pool = [row for row in rows if row["ImageID"] not in used]
    pool.sort(key=lambda row: stable_key(SEED, "cat", row["ImageID"]))
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    stage_counts: Counter[str] = Counter()

    def take(name: str, predicate, limit: int) -> None:
        for row in pool:
            if stage_counts[name] >= limit:
                break
            if row["ImageID"] in selected_ids or not predicate(row):
                continue
            selected.append(row)
            selected_ids.add(row["ImageID"])
            stage_counts[name] += 1

    # The Open Images test split has exhausted nearly all small cats. The train
    # split supplies image-disjoint small targets and richer truncation states.
    take("tiny_lt_1pct", lambda row: cat_area(row) < 0.01, 10)
    take("small_1_to_3pct", lambda row: 0.01 <= cat_area(row) < 0.03, 50)
    take(
        "occluded_and_truncated",
        lambda row: row["IsOccluded"] == "1" and row["IsTruncated"] == "1",
        40,
    )
    take("occluded", lambda row: row["IsOccluded"] == "1", 50)
    take("truncated", lambda row: row["IsTruncated"] == "1", 40)
    take("diversity_fill", lambda row: True, count - len(selected))
    if len(selected) != count:
        raise ValueError(f"cat expansion requires {count} targets, found {len(selected)}")
    audit = {
        "available_after_exclusion": len(pool),
        "excluded_image_ids": len(used),
        "selection_stage_counts": dict(stage_counts),
    }
    return selected, audit


def download_openimages(row: dict[str, object]) -> tuple[str, str]:
    split = str(row["source_split"])
    image_id = str(row["image_id"])
    target = OPENIMAGES / "images" / split / f"{image_id}.jpg"
    if target.is_file() and target.stat().st_size:
        try:
            with Image.open(target) as image:
                image.verify()
            return image_id, "reused"
        except OSError:
            target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".jpg.part")
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "QE-pool-expansion/2.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(target)
        with Image.open(target) as image:
            image.verify()
        return image_id, "downloaded"
    except Exception as error:
        partial.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return image_id, f"error:{error}"


def render_cat(row: dict[str, object], output: Path) -> tuple[dict[str, object], dict[str, object]]:
    image_path = OPENIMAGES / "images" / str(row["source_split"]) / f"{row['image_id']}.jpg"
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    norm = json.loads(str(row["bbox_norm_xyxy"]))
    target_box = absolute_box(norm, image.width, image.height, 0)
    crop_box = absolute_box(norm, image.width, image.height, 0.05)
    sample_id = str(row["sample_id"])
    crop_path = output / "cat/rendered/crops" / f"{sample_id}.jpg"
    marked_path = output / "cat/rendered/marked" / f"{sample_id}.jpg"
    feature_crop_path = output / "cat/rendered/feature_crops" / f"{sample_id}.jpg"
    for path in (crop_path, marked_path, feature_crop_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(crop_path, quality=95)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    width = max(3, min(image.size) // 150)
    draw.rectangle(target_box, outline=(220, 48, 47), width=width)
    draw.text((target_box[0] + 4, target_box[1] + 4), "[A]", fill=(220, 48, 47), stroke_width=2, stroke_fill="white")
    marked.save(marked_path, quality=94)
    target = image.crop(target_box)
    scale = min(960 / target.width, 960 / target.height)
    resized = target.resize(
        (max(1, round(target.width * scale)), max(1, round(target.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (1024, 1024), "white")
    canvas.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
    canvas.save(feature_crop_path, quality=95)
    row.update(
        {
            "image_path": str(image_path.resolve()),
            "crop_path": str(crop_path.resolve()),
            "marked_image_path": str(marked_path.resolve()),
            "feature_crop_path": str(feature_crop_path.resolve()),
            "image_sha256": sha256(image_path),
            "crop_sha256": sha256(crop_path),
        }
    )
    feature = {
        "image_id": sample_id,
        "target_instance_id": "target_1",
        "target_class_id": CAT_MID,
        "target_class": "cat",
        "bbox": json.dumps(target_box, separators=(",", ":")),
        "image_path": str(image_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "crop_image_path": str(feature_crop_path.resolve()),
        "split": "candidate_expansion_v2",
        "image_sha256": row["image_sha256"],
        "label_sha256": "",
        "protocol_version": "cat_v1.0-openimages-expansion-v2",
    }
    return row, feature


def prepare_cat(output: Path, count: int, workers: int) -> dict:
    config = load_json(SOURCE / "configs/openimages_v7_dog_gap_experiment_v1.json")
    annotations = read_csv(OPENIMAGES / "annotations/train_five_classes.csv")
    targets = image_targets(
        annotations,
        {mid: name for name, mid in config["classes"].items()},
        config["box_filters"],
    )["cat"]
    selected, selection_audit = choose_cat_targets(targets, count)
    rows = [
        manifest_row(row, config, "train", "cat_candidate_expansion_v2", index)
        for index, row in enumerate(selected, 1)
    ]
    statuses: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_openimages, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            statuses[status.split(":", 1)[0]] += 1
            if status.startswith("error:"):
                raise RuntimeError(status)
            if index % 25 == 0 or index == len(rows):
                print(f"[cat-download] {index}/{len(rows)} {dict(statuses)}", flush=True)
    rendered, features = [], []
    for index, row in enumerate(rows, 1):
        item, feature = render_cat(row, output)
        rendered.append(item)
        features.append(feature)
        if index % 25 == 0 or index == len(rows):
            print(f"[cat-render] {index}/{len(rows)}", flush=True)
    write_csv(output / "cat/manifests/cat_candidates_expansion.csv", rendered)
    write_csv(output / "cat/inputs/cat_feature_expansion_manifest.csv", features)
    existing = read_csv(SOURCE / "artifacts/openimages_v7_cat_test_v1/inputs/cat_feature_test_manifest.csv")
    write_csv(output / "cat/inputs/cat_feature_pool_400.csv", existing + features)
    all_ids = [str(row["image_id"]) for row in rendered]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) & used_openimages_ids():
        raise ValueError("cat image-ID leakage or duplication")
    metrics = {
        **selection_audit,
        "new_targets": len(rendered),
        "combined_feature_pool": len(existing) + len(features),
        "download_status": dict(statuses),
        "tiny_lt_1pct": sum(float(row["bbox_area_ratio"]) < 0.01 for row in rendered),
        "small_1_to_3pct": sum(0.01 <= float(row["bbox_area_ratio"]) < 0.03 for row in rendered),
        "occluded": sum(row["is_occluded"] == "1" for row in rendered),
        "truncated": sum(row["is_truncated"] == "1" for row in rendered),
        "unique_image_hashes": len({row["image_sha256"] for row in rendered}),
    }
    write_json(output / "cat/audit.json", metrics)
    return metrics


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def choose_face_positives(targets: list[FaceTarget], used_paths: set[str], count: int) -> list[FaceTarget]:
    pool = [target for target in one_target_per_image(targets, 48) if target.relative_path not in used_paths]
    pool.sort(key=lambda item: stable_key(SEED, "face-positive", item.relative_path))
    selected: list[FaceTarget] = []
    selected_paths: set[str] = set()

    def take(predicate, limit: int) -> None:
        taken = 0
        for item in pool:
            if taken >= limit:
                break
            if item.relative_path in selected_paths or not predicate(item):
                continue
            selected.append(item)
            selected_paths.add(item.relative_path)
            taken += 1

    take(lambda item: item.pose > 0, 80)
    take(lambda item: item.occlusion >= 2, 100)
    take(lambda item: item.blur >= 2, 100)
    take(lambda item: item.difficulty >= 5, 80)
    take(lambda item: True, count - len(selected))
    if len(selected) != count:
        raise ValueError(f"face expansion requires {count} positives, found {len(selected)}")
    return selected


def clipped_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, int(round(left))),
        max(0, int(round(top))),
        min(width, int(round(right))),
        min(height, int(round(bottom))),
    )


def render_face_positive(target: FaceTarget, output: Path, rank: int) -> dict[str, object]:
    raw = WIDER / f"WIDER_{target.split}/images" / target.relative_path
    key = stable_key(target.split, target.relative_path)[:16]
    sample_id = f"wider-{target.split}-{key}-face-v2"
    face_box = (target.x, target.y, target.x + target.width, target.y + target.height)
    head_float = expanded_head_box((target.x, target.y, target.width, target.height))
    rendered = output / "face_binary/rendered/positive"
    face_path = rendered / "face" / f"{sample_id}.jpg"
    head_path = rendered / "head" / f"{sample_id}.jpg"
    marked_path = rendered / "marked" / f"{sample_id}.jpg"
    for path in (face_path, head_path, marked_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    head_box = clipped_box(head_float, image.width, image.height)
    image.crop(face_box).save(face_path, quality=95)
    image.crop(head_box).save(head_path, quality=95)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(head_box, outline=(28, 170, 89), width=max(2, min(image.size) // 220))
    draw.rectangle(face_box, outline=(220, 48, 47), width=max(2, min(image.size) // 220))
    draw.text((head_box[0] + 3, head_box[1] + 3), "[A] FACE", fill="white", stroke_width=2, stroke_fill="black", font=ImageFont.load_default())
    marked.save(marked_path, quality=94)
    return {
        "sample_id": sample_id,
        "image_id": f"wider-{target.split}-{key}",
        "target_instance_id": "face_000001",
        "class_name": "face",
        "target_class": "face",
        "label": 1,
        "source_split": target.split,
        "split": "candidate",
        "dataset_version": "widerface-binary-v2-20260928",
        "source_relative_path": target.relative_path,
        "image_path": str(raw.resolve()),
        "crop_path": str(head_path.resolve()),
        "crop_image_path": str(head_path.resolve()),
        "face_crop_path": str(face_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "bbox": json.dumps(list(head_box)),
        "face_bbox_xyxy": json.dumps(list(face_box)),
        "bbox_area_ratio": round((head_box[2] - head_box[0]) * (head_box[3] - head_box[1]) / (image.width * image.height), 8),
        "wider_blur": target.blur,
        "wider_expression": target.expression,
        "wider_illumination": target.illumination,
        "wider_occlusion": target.occlusion,
        "wider_pose": target.pose,
        "wider_difficulty_score": target.difficulty,
        "image_sha256": sha256(raw),
        "crop_sha256": sha256(head_path),
        "selection_rank": rank,
    }


def find_negative_box(
    image_size: tuple[int, int],
    faces: list[FaceTarget],
    reference: FaceTarget,
    seed_key: str,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    # Preserve the reference face aspect and approximate scale, but cap the
    # crop at one third of the current image. Without this cap, a large face
    # from another WIDER image makes a zero-overlap negative geometrically
    # impossible in many otherwise useful scenes.
    if width < 150 or height < 150:
        return None
    crop_w = min(width // 3, max(48, round(reference.width * 1.35)))
    crop_h = min(height // 3, max(48, round(reference.height * 1.45)))
    forbidden = []
    for face in faces:
        margin_x = round(face.width * 0.25)
        margin_y = round(face.height * 0.25)
        forbidden.append((
            max(0, face.x - margin_x), max(0, face.y - margin_y),
            min(width, face.x + face.width + margin_x),
            min(height, face.y + face.height + margin_y),
        ))
    rng = random.Random(int(stable_key(SEED, "negative", seed_key)[:16], 16))
    candidates = []
    for _ in range(300):
        x1 = rng.randint(0, width - crop_w)
        y1 = rng.randint(0, height - crop_h)
        box = (x1, y1, x1 + crop_w, y1 + crop_h)
        if all(intersection_area(box, face_box) == 0 for face_box in forbidden):
            # Prefer boxes close to a face, which reduces scene/background shortcuts.
            distance = min(
                abs((box[0] + box[2]) / 2 - (face.x + face.width / 2))
                + abs((box[1] + box[3]) / 2 - (face.y + face.height / 2))
                for face in faces
            )
            candidates.append((distance, box))
    return min(candidates, default=(None, None), key=lambda item: item[0])[1]


def render_face_negative(
    relative_path: str,
    faces: list[FaceTarget],
    reference: FaceTarget,
    box: tuple[int, int, int, int],
    output: Path,
    rank: int,
) -> dict[str, object]:
    raw = WIDER / "WIDER_val/images" / relative_path
    key = stable_key("val", relative_path)[:16]
    sample_id = f"wider-val-{key}-nonface-v2"
    rendered = output / "face_binary/rendered/negative"
    crop_path = rendered / "crops" / f"{sample_id}.jpg"
    marked_path = rendered / "marked" / f"{sample_id}.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    marked_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.crop(box).save(crop_path, quality=95)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(box, outline=(39, 110, 210), width=max(2, min(image.size) // 220))
    draw.text((box[0] + 3, box[1] + 3), "[A] NON-FACE", fill="white", stroke_width=2, stroke_fill="black", font=ImageFont.load_default())
    marked.save(marked_path, quality=94)
    return {
        "sample_id": sample_id,
        "image_id": f"wider-val-{key}",
        "target_instance_id": "nonface_000001",
        "class_name": "nonface",
        "target_class": "nonface",
        "label": 0,
        "source_split": "val",
        "split": "candidate",
        "dataset_version": "widerface-binary-v2-20260928",
        "source_relative_path": relative_path,
        "image_path": str(raw.resolve()),
        "crop_path": str(crop_path.resolve()),
        "crop_image_path": str(crop_path.resolve()),
        "face_crop_path": "",
        "marked_image_path": str(marked_path.resolve()),
        "bbox": json.dumps(list(box)),
        "face_bbox_xyxy": "",
        "bbox_area_ratio": round((box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height), 8),
        "wider_blur": "",
        "wider_expression": "",
        "wider_illumination": "",
        "wider_occlusion": "",
        "wider_pose": "",
        "wider_difficulty_score": "",
        "image_sha256": sha256(raw),
        "crop_sha256": sha256(crop_path),
        "selection_rank": rank,
        "negative_reference_face": json.dumps(asdict(reference), ensure_ascii=False),
        "maximum_face_intersection_px": max(intersection_area(box, (f.x, f.y, f.x + f.width, f.y + f.height)) for f in faces),
    }


def prepare_face_binary(output: Path, count: int) -> dict:
    annotations = parse_annotations(WIDER / "wider_face_split/wider_face_val_bbx_gt.txt", "val")
    valid = [item for item in annotations if item.invalid == 0]
    grouped_all: dict[str, list[FaceTarget]] = defaultdict(list)
    for item in annotations:
        grouped_all[item.relative_path].append(item)
    previous_manifest = ROOT / "artifacts/head_face_v1_0_20260922/manifests/face_candidates.csv"
    used_paths = {row["source_relative_path"] for row in read_csv(previous_manifest)}
    previous_val = ROOT / "artifacts/head_face_v1_0_20260922/manifests/face_validation.csv"
    used_paths.update(row["source_relative_path"] for row in read_csv(previous_val))
    positives = choose_face_positives(valid, used_paths, count)
    positive_rows = []
    for index, target in enumerate(positives, 1):
        positive_rows.append(render_face_positive(target, output, index))
        if index % 50 == 0 or index == len(positives):
            print(f"[face-positive] {index}/{len(positives)}", flush=True)
    blocked = used_paths | {item.relative_path for item in positives}
    negative_rows = []
    candidate_images = sorted(
        (path for path in grouped_all if path not in blocked),
        key=lambda path: stable_key(SEED, "negative-image", path),
    )
    references = sorted(positives, key=lambda item: stable_key(SEED, "reference", item.relative_path))
    for relative_path in candidate_images:
        if len(negative_rows) == count:
            break
        raw = WIDER / "WIDER_val/images" / relative_path
        with Image.open(raw) as image:
            size = ImageOps.exif_transpose(image).size
        reference = references[len(negative_rows) % len(references)]
        faces = grouped_all[relative_path]
        box = find_negative_box(size, faces, reference, relative_path)
        if box is None:
            continue
        negative_rows.append(
            render_face_negative(relative_path, faces, reference, box, output, len(negative_rows) + 1)
        )
        if len(negative_rows) % 50 == 0:
            print(f"[face-negative] {len(negative_rows)}/{count}", flush=True)
    if len(negative_rows) != count:
        raise ValueError(f"face expansion requires {count} negatives, found {len(negative_rows)}")
    write_csv(output / "face_binary/manifests/face_candidates.csv", positive_rows)
    write_csv(output / "face_binary/manifests/nonface_candidates.csv", negative_rows)
    write_csv(output / "face_binary/manifests/candidate_pool.csv", positive_rows + negative_rows)
    positive_sources = {row["source_relative_path"] for row in positive_rows}
    negative_sources = {row["source_relative_path"] for row in negative_rows}
    if positive_sources & negative_sources or positive_sources & used_paths or negative_sources & used_paths:
        raise ValueError("WIDER source-image leakage")
    if any(int(row["maximum_face_intersection_px"]) != 0 for row in negative_rows):
        raise ValueError("a non-face crop intersects a WIDER face annotation")
    metrics = {
        "positive_targets": len(positive_rows),
        "negative_targets": len(negative_rows),
        "previous_wider_images_excluded": len(used_paths),
        "positive_pose_atypical": sum(item.pose > 0 for item in positives),
        "positive_occluded": sum(item.occlusion > 0 for item in positives),
        "positive_blurred": sum(item.blur > 0 for item in positives),
        "positive_difficulty_ge_5": sum(item.difficulty >= 5 for item in positives),
        "source_image_overlap": 0,
        "negative_face_intersections": 0,
        "unique_positive_hashes": len({row["image_sha256"] for row in positive_rows}),
        "unique_negative_hashes": len({row["image_sha256"] for row in negative_rows}),
    }
    write_json(output / "face_binary/audit.json", metrics)
    return metrics


def write_research_files(output: Path, cat: dict | None, face: dict | None) -> None:
    protocol = {
        "version": "candidate-pool-expansion-v2-20260928",
        "created_at": now(),
        "seed": SEED,
        "old_experiments_modified": False,
        "cat": cat,
        "face_binary": face,
        "automatic_quality_gates": [
            "unique sample and source image IDs",
            "cryptographic image and crop hashes",
            "all images decode successfully",
            "no overlap with frozen manifests",
            "no intersection with any WIDER face annotation, including invalid annotations, in non-face crops",
        ],
    }
    write_json(output / "protocol.json", protocol)
    (output / "paper_brief.md").write_text(
        "# Paper Brief\n\n"
        "研究问题：扩大候选池并提高困难状态密度后，覆盖导向测试能否在 cat 和 face 域获得比旧候选池更强的独立诊断增益？\n\n"
        "本版本只构建数据，不使用分类模型预测选样。cat 扩展保持 cat_v1.0；face 建立同源 face/nonface 二分类候选池，以消除 face-vs-animal 的任务捷径。\n",
        encoding="utf-8",
    )
    (output / "claim_evidence.md").write_text(
        "# Claim-Evidence Matrix\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Candidate database was expanded without modifying frozen experiments | supported | `protocol.json`, manifests | data construction only |\n"
        "| New cat images are image-disjoint from existing Open Images manifests | supported | `cat/audit.json` | ID/hash audit |\n"
        "| Non-face crops do not intersect WIDER face boxes | supported | `face_binary/audit.json` | annotation-box criterion, not pixel segmentation |\n"
        "| Expansion improves diagnostic effectiveness | pending | future frozen-model experiment | cannot be claimed from construction alone |\n",
        encoding="utf-8",
    )
    (output / "research_log.md").write_text(
        f"# Research Log\n\n- {now()}: candidate-pool expansion v2 materialized.\n"
        "- Existing dog, cat, and head/face experiments were left immutable.\n"
        "- Feature annotation and frozen-model evaluation are separate downstream stages.\n",
        encoding="utf-8",
    )
    (output / "review.md").write_text(
        "# Review\n\n"
        "- Data integrity gates: automatic and auditable.\n"
        "- Selection leakage: no classifier prediction used.\n"
        "- Remaining risk: Open Images and WIDER annotations can contain label omissions; negative crops are guaranteed only against recorded WIDER boxes.\n"
        "- Claim scope: no performance or coverage-improvement claim is made before annotation and evaluation.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--component", choices=("all", "cat", "face"), default="all")
    parser.add_argument("--cat-count", type=int, default=200)
    parser.add_argument("--face-count", type=int, default=400)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cat = prepare_cat(args.output, args.cat_count, args.workers) if args.component in {"all", "cat"} else None
    face = prepare_face_binary(args.output, args.face_count) if args.component in {"all", "face"} else None
    write_research_files(args.output, cat, face)
    result = {"status": "complete", "output": str(args.output.resolve()), "cat": cat, "face_binary": face}
    write_json(args.output / "expansion_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
