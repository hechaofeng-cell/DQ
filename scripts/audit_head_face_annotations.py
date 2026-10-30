#!/usr/bin/env python3
"""Create a deterministic human-review queue for head_face_v1.0 labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "artifacts/head_face_v1_0_20260922"


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


def as_set(value: str) -> set[str]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return set()
    return set(parsed) if isinstance(parsed, list) else set()


def main() -> None:
    features = read_csv(EXPERIMENT / "annotations/results.csv")
    manifest = {row["sample_id"]: row for row in read_csv(EXPERIMENT / "manifests/face_candidates.csv")}
    queue = []
    for row in features:
        reasons = []
        truncation = as_set(row["head_truncation_parts"])
        occluders = as_set(row["occluder_types"])
        if row["head_completeness"] == "complete" and truncation != {"none"}:
            reasons.append("complete_head_with_truncation")
        if row["head_completeness"] in {"lightly_truncated", "severely_truncated"} and "none" in truncation:
            reasons.append("truncated_head_with_none_part")
        if row["occlusion_level"] == "none" and occluders != {"none"}:
            reasons.append("no_occlusion_with_occluder")
        if row["occlusion_level"] != "none" and "none" in occluders:
            reasons.append("occlusion_with_none_occluder")
        if row["hair_visibility"] == "not_visible" and row["hair_color"] not in {"not_visible", "unknown"}:
            reasons.append("hidden_hair_with_color")
        if row["hair_visibility"] == "not_visible" and row["hair_texture"] not in {"not_visible", "unknown"}:
            reasons.append("hidden_hair_with_texture")
        if row["visible_skin_tone"] == "unknown" and row["skin_tone_confidence"] not in {"unknown", "low"}:
            reasons.append("unknown_skin_with_confident_label")
        if row["visible_skin_tone"] != "unknown" and row["skin_tone_confidence"] == "unknown":
            reasons.append("skin_label_without_confidence")
        if row["face_visibility"] in {"not_visible", "minimal"} and any(row[key] == "clear" for key in ("nose_visibility", "mouth_visibility")):
            reasons.append("minimal_face_with_clear_part")
        source = manifest[row["sample_id"]]
        if source["head_box_truncated"] == "1" and row["head_completeness"] == "complete":
            reasons.append("bbox_truncation_disagrees_with_label")
        if row["skin_tone_confidence"] in {"low", "unknown"}:
            reasons.append("skin_tone_review")
        if row["head_recognizability"] in {"low", "unrecognizable"}:
            reasons.append("hard_target_review")
        if reasons:
            queue.append({
                "priority": len(reasons), "sample_id": row["sample_id"],
                "reasons": "|".join(reasons), "image_path": source["image_path"],
                "marked_image_path": source["marked_image_path"], "head_crop_path": source["crop_path"],
                "face_crop_path": source["face_crop_path"],
            })
    queue.sort(key=lambda row: (-int(row["priority"]), row["sample_id"]))
    write_csv(EXPERIMENT / "annotations/human_review_queue.csv", queue)
    summary = {
        "status": "automatic_labels_pending_human_review",
        "total_labels": len(features), "queued_for_review": len(queue),
        "queue_fraction": len(queue) / len(features) if features else None,
        "recommended_double_annotation": min(60, len(features)),
    }
    (EXPERIMENT / "annotations/audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
