#!/usr/bin/env python3
"""Apply declared cat_v1.0 dependency rules to rejected raw VLM responses."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qe_quality.dta.io import unique_json_object

from run_dog_local_pipeline import validate_feature


OUTPUT = ROOT / "artifacts/openimages_v7_cat_test_v1/features_auto"
INPUT = ROOT / "artifacts/openimages_v7_cat_test_v1/inputs/cat_feature_test_manifest.csv"
SCHEMA = ROOT / "configs/cat_feature_schema_v1_0.json"


def main() -> None:
    rows = {row["image_id"]: row for row in csv.DictReader(INPUT.open(encoding="utf-8-sig", newline=""))}
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors = json.loads((OUTPUT / "errors.json").read_text(encoding="utf-8"))
    audit = []
    for error in errors:
        if error["stage"] != "features" or error["error"] != "mouth_dependency_mismatch":
            raise ValueError(f"unsupported repair: {error}")
        image_id = error["image_id"]
        raw_path = OUTPUT / "raw/features" / f"{image_id}_attempt1.json"
        wrapper = json.loads(raw_path.read_text(encoding="utf-8"))
        result = unique_json_object(wrapper["message"]["content"])
        features = result["features"]
        if features["mouth_visibility"] != "not_visible":
            raise ValueError(f"unexpected dependency state for {image_id}")
        before_state = features["mouth_state"]
        before_tongue = features["tongue_visibility"]
        features["mouth_state"] = "unknown"
        features["tongue_visibility"] = "unknown"
        result["evidence"]["mouth_state"] = "目标嘴部不可见，无法判断嘴部状态。"
        result["evidence"]["tongue_visibility"] = "目标嘴部不可见，无法判断舌头是否可见。"
        validate_feature(result, rows[image_id], schema)
        parsed_path = OUTPUT / "parsed/features" / f"{image_id}.json"
        parsed_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit.append({
            "image_id": image_id,
            "rule": "mouth_visibility_not_visible_dependencies_unknown",
            "mouth_state_before": before_state,
            "mouth_state_after": "unknown",
            "tongue_visibility_before": before_tongue,
            "tongue_visibility_after": "unknown",
            "raw_response": str(raw_path.resolve()),
            "corrected_record": str(parsed_path.resolve()),
        })
    audit_path = OUTPUT / "dependency_repairs.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps({"repaired": len(audit), "audit": str(audit_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
