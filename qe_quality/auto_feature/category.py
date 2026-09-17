"""Contracts for the blind category-discovery stage."""

from collections import Counter
import math


CATEGORY_CANDIDATES = [
    "person", "vehicle", "building", "furniture", "electronic",
    "indoor_item", "outdoor_item", "natural_scene", "animal", "food",
    "plant", "tool", "clothing", "text_document", "other", "unknown",
]


def validate_category_result(value, row):
    """Validate one independent category response without using target_class."""
    required = {
        "image_id", "target_instance_id", "predicted_category", "fine_name",
        "confidence", "evidence",
    }
    if set(value) != required:
        raise ValueError("category_fields_mismatch")
    if value["image_id"] != row["image_id"]:
        raise ValueError("category_image_id_mismatch")
    if value["target_instance_id"] != row["target_instance_id"]:
        raise ValueError("category_target_instance_id_mismatch")
    if value["predicted_category"] not in CATEGORY_CANDIDATES:
        raise ValueError("invalid_predicted_category")
    if not isinstance(value["fine_name"], str) or not value["fine_name"].strip():
        raise ValueError("invalid_fine_name")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("invalid_category_confidence")
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise ValueError("invalid_category_confidence")
    if not isinstance(value["evidence"], str) or not value["evidence"].strip():
        raise ValueError("empty_category_evidence")
    return {
        **value,
        "fine_name": value["fine_name"].strip(),
        "confidence": float(confidence),
        "evidence": value["evidence"].strip(),
    }


def aggregate_category(results, min_consensus=0.8, min_confidence=0.6):
    """Return a deterministic batch category or raise on an unsafe mixture."""
    usable = [x for x in results if x["predicted_category"] != "unknown"]
    if not usable:
        raise ValueError("mixed_or_unresolved_category")
    counts = Counter(x["predicted_category"] for x in usable)
    category, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    consensus = count / len(usable)
    confidence = sum(x["confidence"] for x in usable if x["predicted_category"] == category) / count
    if consensus < min_consensus or confidence < min_confidence:
        raise ValueError("mixed_or_unresolved_category")
    names = [x["fine_name"] for x in usable if x["predicted_category"] == category]
    fine_name = Counter(names).most_common(1)[0][0]
    return {
        "category": category,
        "fine_name": fine_name,
        "consensus": round(consensus, 6),
        "confidence": round(confidence, 6),
        "usable_count": len(usable),
        "category_counts": dict(sorted(counts.items())),
    }
