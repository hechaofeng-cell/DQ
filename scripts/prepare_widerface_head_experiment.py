#!/usr/bin/env python3
"""Prepare a D300-equivalent head/face state-coverage experiment.

The existing five-class target-crop protocol is retained, with ``person``
replaced by ``face``. WIDER FACE supplies one annotated face target per source
image. A face crop is preserved for auditing, while an expanded head crop is
used by both the classifier and the state annotator so hair and headwear remain
visible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE_ROOT = Path("/home/hcf/project/QE")
DEFAULT_DATA = SOURCE_ROOT / "data/widerface_head_v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/head_face_v1_0_20260922"
DEMOGRAPHICS_VAL = SOURCE_ROOT / "data/widerface_demographics_annotations/annotations/demographics_val.csv"
BASE_TRAIN = SOURCE_ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv"
BASE_VAL = SOURCE_ROOT / "data/openimages_v7_dog_gap_v1/manifests/validation.csv"
BASE_CANDIDATE = SOURCE_ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920/candidate_pool.csv"
CLASS_NAMES = ("dog", "cat", "horse", "sheep", "face")
SEED = 20260922


@dataclass(frozen=True)
class FaceTarget:
    split: str
    relative_path: str
    face_index: int
    x: int
    y: int
    width: int
    height: int
    blur: int
    expression: int
    illumination: int
    occlusion: int
    pose: int
    invalid: int

    @property
    def difficulty(self) -> int:
        return 2 * self.blur + self.expression + self.illumination + 2 * self.occlusion + self.pose


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_key(*values: object) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()


def parse_annotations(path: Path, split: str) -> list[FaceTarget]:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[FaceTarget] = []
    cursor = 0
    while cursor < len(lines):
        relative_path = lines[cursor].strip()
        count = int(lines[cursor + 1])
        cursor += 2
        if count == 0:
            # Official WIDER FACE files store one all-zero placeholder line.
            if cursor < len(lines) and len(lines[cursor].split()) >= 10:
                cursor += 1
            continue
        for face_index in range(count):
            values = [int(value) for value in lines[cursor].split()]
            cursor += 1
            if len(values) < 10:
                raise ValueError(f"invalid annotation: {relative_path} face={face_index}")
            x, y, width, height, blur, expression, illumination, invalid, occlusion, pose = values[:10]
            output.append(FaceTarget(
                split, relative_path, face_index, x, y, width, height,
                blur, expression, illumination, occlusion, pose, invalid,
            ))
    return output


def one_target_per_image(targets: list[FaceTarget], min_side: int) -> list[FaceTarget]:
    grouped: dict[str, list[FaceTarget]] = defaultdict(list)
    for target in targets:
        if target.invalid == 0 and min(target.width, target.height) >= min_side:
            grouped[target.relative_path].append(target)
    selected = []
    for relative_path, values in grouped.items():
        selected.append(max(values, key=lambda item: (item.difficulty, item.width * item.height, -item.face_index)))
    return selected


def choose_train(targets: list[FaceTarget], count: int) -> list[FaceTarget]:
    ordered = sorted(targets, key=lambda item: stable_key(SEED, "train", item.relative_path))
    return ordered[:count]


def read_skin_metadata(path: Path) -> dict[tuple[str, int], str]:
    output = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relative_path = row["file"].removeprefix("images/")
            if row["perceived_skin"] in {"I", "II", "III", "IV", "V", "VI"}:
                output[(relative_path, int(row["face"]))] = row["perceived_skin"]
    return output


def choose_val_and_candidate(
    targets: list[FaceTarget], skin_metadata: dict[tuple[str, int], str], val_count: int, candidate_count: int,
) -> tuple[list[FaceTarget], list[FaceTarget]]:
    eligible = [item for item in targets if (item.relative_path, item.face_index) in skin_metadata]
    strata: dict[tuple[str, int, int, int, int], list[FaceTarget]] = defaultdict(list)
    for item in eligible:
        scale = 0 if min(item.width, item.height) < 80 else 1 if min(item.width, item.height) < 160 else 2
        skin = skin_metadata[(item.relative_path, item.face_index)]
        strata[(skin, scale, item.blur, item.occlusion, item.pose)].append(item)
    for key in strata:
        strata[key].sort(key=lambda item: stable_key(SEED, "candidate", item.relative_path))
    candidate: list[FaceTarget] = []
    used_images = set()
    skin_levels = ("I", "II", "III", "IV", "V", "VI")
    quotas = {skin: candidate_count // len(skin_levels) for skin in skin_levels}
    for skin in skin_levels[: candidate_count % len(skin_levels)]:
        quotas[skin] += 1
    for skin in skin_levels:
        keys = sorted((key for key in strata if key[0] == skin), key=lambda key: stable_key(SEED, "stratum", key))
        while sum(skin_metadata[(item.relative_path, item.face_index)] == skin for item in candidate) < quotas[skin] and keys:
            next_keys = []
            for key in keys:
                while strata[key] and strata[key][-1].relative_path in used_images:
                    strata[key].pop()
                current = sum(skin_metadata[(item.relative_path, item.face_index)] == skin for item in candidate)
                if strata[key] and current < quotas[skin]:
                    selected = strata[key].pop()
                    candidate.append(selected)
                    used_images.add(selected.relative_path)
                if strata[key]:
                    next_keys.append(key)
            keys = next_keys
    if len(candidate) != candidate_count:
        raise ValueError(f"only {len(candidate)} eligible validation targets")
    remaining_images = {item.relative_path for item in targets} - used_images
    ordered = sorted((item for item in targets if item.relative_path in remaining_images), key=lambda item: stable_key(SEED, "validation", item.relative_path, item.face_index))
    validation, validation_images = [], set()
    for item in ordered:
        if item.relative_path in validation_images:
            continue
        validation.append(item)
        validation_images.add(item.relative_path)
        if len(validation) == val_count:
            break
    if len(validation) != val_count:
        raise ValueError(f"only {len(validation)} validation targets")
    return validation, candidate


def expanded_head_box(face_box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x, y, width, height = face_box
    return x - 0.30 * width, y - 0.50 * height, x + 1.30 * width, y + 1.20 * height


def crop_with_padding(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    left, top, right, bottom = map(lambda value: int(round(value)), box)
    source_box = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        raise ValueError(f"empty clipped head box: {box}")
    return image.crop(source_box)


def render_target(target: FaceTarget, data_root: Path, output_root: Path, role: str, rank: int) -> dict[str, object]:
    raw = data_root / "raw/widerface" / f"WIDER_{target.split}" / "images" / target.relative_path
    if not raw.is_file():
        raise FileNotFoundError(raw)
    key = stable_key(target.split, target.relative_path)[:16]
    sample_id = f"wider-{target.split}-{key}-face"
    image_id = f"wider-{target.split}-{key}"
    face_box = (target.x, target.y, target.width, target.height)
    head_box = expanded_head_box(face_box)
    rendered = output_root / "rendered" / role
    face_path = rendered / "face" / f"{sample_id}.jpg"
    head_path = rendered / "head" / f"{sample_id}.jpg"
    marked_path = rendered / "marked" / f"{sample_id}.jpg"
    for parent in (face_path.parent, head_path.parent, marked_path.parent):
        parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw) as source:
        image = source.convert("RGB")
        face = image.crop((target.x, target.y, target.x + target.width, target.y + target.height))
        head = crop_with_padding(image, head_box)
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        draw.rectangle(head_box, outline=(28, 180, 90), width=max(2, round(min(image.size) / 250)))
        draw.rectangle((target.x, target.y, target.x + target.width, target.y + target.height), outline=(220, 50, 47), width=max(2, round(min(image.size) / 250)))
        draw.text((max(0, int(head_box[0])), max(0, int(head_box[1]) - 14)), "HEAD / FACE", fill=(255, 255, 255), font=ImageFont.load_default())
        face.save(face_path, quality=95)
        head.save(head_path, quality=95)
        marked.save(marked_path, quality=92)
        image_width, image_height = image.size
    x1, y1, x2, y2 = head_box
    clipped = (max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2))
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "target_instance_id": "face_000001",
        "class_name": "face",
        "target_class": "face",
        "label": 4,
        "source_split": target.split,
        "split": role,
        "dataset_version": "widerface-head-v1-20260922",
        "source_relative_path": target.relative_path,
        "image_path": str(raw.resolve()),
        "crop_path": str(head_path.resolve()),
        "crop_image_path": str(head_path.resolve()),
        "face_crop_path": str(face_path.resolve()),
        "marked_image_path": str(marked_path.resolve()),
        "bbox": json.dumps([round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]),
        "face_bbox_xywh": json.dumps(list(face_box)),
        "head_bbox_clipped_xyxy": json.dumps([round(value, 3) for value in clipped]),
        "bbox_area_ratio": round((clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) / (image_width * image_height), 8),
        "head_box_truncated": int(x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height),
        "wider_blur": target.blur,
        "wider_expression": target.expression,
        "wider_illumination": target.illumination,
        "wider_occlusion": target.occlusion,
        "wider_pose": target.pose,
        "wider_difficulty_score": target.difficulty,
        "image_sha256": digest(raw),
        "crop_sha256": digest(head_path),
        "selection_rank": rank,
    }


def normalize_existing(row: dict[str, str], role: str) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    output.update({"target_class": row["class_name"], "split": role, "crop_image_path": row["crop_path"]})
    output.setdefault("bbox", row.get("bbox_norm_xyxy", ""))
    output.setdefault("face_crop_path", "")
    output.setdefault("head_bbox_clipped_xyxy", "")
    output.setdefault("head_box_truncated", "")
    output.setdefault("wider_blur", "")
    output.setdefault("wider_expression", "")
    output.setdefault("wider_illumination", "")
    output.setdefault("wider_occlusion", "")
    output.setdefault("wider_pose", "")
    output.setdefault("wider_difficulty_score", "")
    output.setdefault("selection_rank", "")
    return output


def harmonize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    preferred = [
        "sample_id", "image_id", "target_instance_id", "class_name", "target_class", "label",
        "source_split", "split", "dataset_version", "source_relative_path", "image_path", "crop_path",
        "crop_image_path", "face_crop_path", "marked_image_path", "bbox", "face_bbox_xywh",
        "head_bbox_clipped_xyxy", "bbox_area_ratio", "head_box_truncated", "wider_blur",
        "wider_expression", "wider_illumination", "wider_occlusion", "wider_pose",
        "wider_difficulty_score", "image_sha256", "crop_sha256", "selection_rank",
    ]
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    fields = preferred + extras
    return [{field: row.get(field, "") for field in fields} for row in rows]


def assert_counts(rows: list[dict[str, object]], expected: dict[str, int], name: str) -> None:
    counts = Counter(str(row["class_name"]) for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"{name} counts {dict(counts)} != {expected}")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError(f"duplicate sample ID in {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = args.data_root / "raw/widerface"
    train_targets = parse_annotations(raw / "wider_face_split/wider_face_train_bbx_gt.txt", "train")
    val_targets = parse_annotations(raw / "wider_face_split/wider_face_val_bbx_gt.txt", "val")
    skin_metadata = read_skin_metadata(DEMOGRAPHICS_VAL)
    train_faces = choose_train(one_target_per_image(train_targets, 64), 300)
    validation_faces, candidate_faces = choose_val_and_candidate(
        [item for item in val_targets if item.invalid == 0 and min(item.width, item.height) >= 48],
        skin_metadata, 30, 200,
    )

    rendered_train = [render_target(item, args.data_root, args.output, "train", index + 1) for index, item in enumerate(train_faces)]
    rendered_val = [render_target(item, args.data_root, args.output, "validation", index + 1) for index, item in enumerate(validation_faces)]
    rendered_candidate = [render_target(item, args.data_root, args.output, "candidate", index + 1) for index, item in enumerate(candidate_faces)]
    for row, target in zip(rendered_candidate, candidate_faces):
        row["perceived_skin_metadata"] = skin_metadata[(target.relative_path, target.face_index)]
        row["perceived_skin_metadata_source"] = "Amazon WiderFace Demographics validation human annotation"

    base_train = [normalize_existing(row, "train") for row in read_csv(BASE_TRAIN) if row["class_name"] != "person"]
    base_val = [normalize_existing(row, "validation") for row in read_csv(BASE_VAL) if row["class_name"] != "person"]
    base_candidate_rows = read_csv(BASE_CANDIDATE)
    nonface_candidate = []
    for name in CLASS_NAMES[:-1]:
        values = sorted((row for row in base_candidate_rows if row["class_name"] == name), key=lambda row: row["sample_id"])
        nonface_candidate.extend(normalize_existing(row, "candidate") for row in values[:100])

    train = harmonize(base_train + rendered_train)
    validation = harmonize(base_val + rendered_val)
    candidate = harmonize(nonface_candidate + rendered_candidate)
    expected_train = {name: 300 for name in CLASS_NAMES}
    expected_val = {name: 30 for name in CLASS_NAMES}
    expected_candidate = {name: (200 if name == "face" else 100) for name in CLASS_NAMES}
    assert_counts(train, expected_train, "train")
    assert_counts(validation, expected_val, "validation")
    assert_counts(candidate, expected_candidate, "candidate")
    if {row["image_sha256"] for row in rendered_train} & {row["image_sha256"] for row in rendered_val + rendered_candidate}:
        raise ValueError("WIDER source image leakage")

    write_csv(args.output / "manifests/train_d300.csv", train)
    write_csv(args.output / "manifests/validation.csv", validation)
    write_csv(args.output / "manifests/candidate_pool.csv", candidate)
    write_csv(args.output / "manifests/face_train.csv", harmonize(rendered_train))
    write_csv(args.output / "manifests/face_validation.csv", harmonize(rendered_val))
    write_csv(args.output / "manifests/face_candidates.csv", harmonize(rendered_candidate))
    audit = {
        "dataset": "WIDER FACE",
        "task": "five-class target-region classification with person replaced by face",
        "classes": list(CLASS_NAMES),
        "train_counts": expected_train,
        "validation_counts": expected_val,
        "candidate_counts": expected_candidate,
        "budgets_per_class": [20, 40, 60, 80, 100],
        "face_box_expansion": {"left": 0.30, "right": 0.30, "top": 0.50, "bottom": 0.20},
        "minimum_face_side": {"train": 64, "validation_and_candidate": 48},
        "selection_seed": SEED,
        "identity_labels_used": False,
        "ethnicity_inferred": False,
        "candidate_skin_stratification": Counter(row["perceived_skin_metadata"] for row in rendered_candidate),
        "candidate_skin_metadata_license": "CC BY-NC 4.0",
    }
    (args.output / "dataset_protocol.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", **audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
