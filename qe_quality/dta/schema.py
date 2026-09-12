"""Frozen phase-one semantic and geometry contracts."""

import math

CLASS_NAMES = {
    1: "person", 2: "vehicle", 3: "building", 4: "furniture",
    5: "electronic", 6: "indoor_item", 7: "outdoor_item",
    8: "natural_scene", 9: "animal", 10: "food", 11: "plant",
    12: "tool", 13: "clothing", 14: "text_document", 15: "other",
}
OUTLINES = {"clear", "partial", "unclear"}
OCCLUSIONS = {"none", "partial", "heavy", "unknown"}
SUBJECT_ROLES = {
    "foreground_displayed_object", "foreground_operated_object",
    "foreground_dominant_object", "action_agent", "scene_or_group", "ambiguous",
}
SEMANTIC_FIELDS = {
    "image_id", "class_id", "class_name", "class_confidence",
    "fine_name", "subject_role", "selection_confidence", "subject_selection_evidence",
    "outline_visibility", "occlusion_level", "target_bbox_hint", "evidence",
}


def validate_semantic(value, image_id):
    if set(value) != SEMANTIC_FIELDS:
        raise ValueError("semantic_fields_mismatch")
    if value["image_id"] != image_id:
        raise ValueError("image_id_mismatch")
    class_id = value["class_id"]
    confidence = value["class_confidence"]
    if type(class_id) is not int or class_id not in CLASS_NAMES:
        raise ValueError("invalid_class_id")
    if value["class_name"] != CLASS_NAMES[class_id]:
        raise ValueError("class_id_name_mismatch")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("invalid_class_confidence")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid_class_confidence")
    if value["outline_visibility"] not in OUTLINES:
        raise ValueError("invalid_outline_visibility")
    if value["occlusion_level"] not in OCCLUSIONS:
        raise ValueError("invalid_occlusion_level")
    if value["subject_role"] not in SUBJECT_ROLES:
        raise ValueError("invalid_subject_role")
    if not isinstance(value["fine_name"], str) or not value["fine_name"].strip():
        raise ValueError("invalid_fine_name")
    selection_confidence = value["selection_confidence"]
    if (isinstance(selection_confidence, bool) or
            not isinstance(selection_confidence, (int, float)) or
            not math.isfinite(float(selection_confidence)) or not 0 <= float(selection_confidence) <= 1):
        raise ValueError("invalid_selection_confidence")
    selection_evidence = value["subject_selection_evidence"]
    if not isinstance(selection_evidence, str) or not selection_evidence.strip():
        raise ValueError("empty_subject_selection_evidence")
    box = value["target_bbox_hint"]
    if (not isinstance(box, list) or len(box) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box)):
        raise ValueError("invalid_target_bbox_hint")
    box = [float(v) for v in box]
    if any(not math.isfinite(v) or not 0 <= v <= 1000 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("invalid_target_bbox_hint")
    scale = 1000 if max(box) > 1 else 1
    normalized_box = [v / scale for v in box]
    evidence = value["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("empty_evidence")
    normalized_id = 15 if confidence < .6 else class_id
    return {
        **value,
        "class_confidence": confidence,
        "fine_name": value["fine_name"].strip(),
        "selection_confidence": float(selection_confidence),
        "subject_selection_evidence": selection_evidence.strip(),
        "raw_target_bbox_hint": box,
        "target_bbox_scale": scale,
        "target_bbox_hint": normalized_box,
        "evidence": evidence.strip(),
        "raw_class_id": class_id,
        "raw_class_name": value["class_name"],
        "class_id": normalized_id,
        "class_name": CLASS_NAMES[normalized_id],
    }


def prompt_schema():
    return {
        "type": "object",
        "properties": {
            "image_id": {"type": "string"},
            "class_id": {"type": "integer", "minimum": 1, "maximum": 15},
            "class_name": {"type": "string", "enum": list(CLASS_NAMES.values())},
            "class_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "fine_name": {"type": "string"},
            "subject_role": {"type": "string", "enum": sorted(SUBJECT_ROLES)},
            "selection_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "subject_selection_evidence": {"type": "string"},
            "outline_visibility": {"type": "string", "enum": sorted(OUTLINES)},
            "occlusion_level": {"type": "string", "enum": sorted(OCCLUSIONS)},
            "target_bbox_hint": {
                "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4, "maxItems": 4,
            },
            "evidence": {"type": "string"},
        },
        "required": sorted(SEMANTIC_FIELDS),
        "additionalProperties": False,
    }


def pad_box(box, width, height, fraction=.05):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * fraction, (y2 - y1) * fraction
    return [
        max(0., (x1 - dx) * width), max(0., (y1 - dy) * height),
        min(float(width), (x2 + dx) * width), min(float(height), (y2 + dy) * height),
    ]


def geometry_from_mask(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty_sam_mask")
    height, width = mask.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    cx, cy = float(xs.mean()) / width, float(ys.mean()) / height
    horizontal = "left" if cx < 1 / 3 else "right" if cx >= 2 / 3 else "center"
    vertical = "top" if cy < 1 / 3 else "bottom" if cy >= 2 / 3 else "middle"
    legacy = "center" if horizontal == "center" and vertical == "middle" else (
        vertical if vertical != "middle" else horizontal
    )
    return {
        "bbox": bbox,
        "area_ratio": float(mask.astype(bool).mean()),
        "horizontal_position": horizontal,
        "vertical_position": vertical,
        "legacy_position": legacy,
    }
