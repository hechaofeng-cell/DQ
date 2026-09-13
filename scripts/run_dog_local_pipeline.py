#!/usr/bin/env python3
"""Run dog-v1 feature annotation and independent classification with local Qwen-VL."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qe_quality.dta.io import atomic_json, unique_json_object

COCO_CANDIDATES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush", "unknown"]


def now():
    return datetime.now(timezone.utc).isoformat()


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    parsed = urllib.parse.urlsplit(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if parsed.hostname in {"localhost", "127.0.0.1", "::1"} else urllib.request.build_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace"), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), time.monotonic() - started


def ask(config, prompt, row, images, raw_path, schema=None):
    content = prompt + "\nimage_id: " + row["image_id"] + "\ntarget_instance_id: target_1"
    body = {"model": config["model"], "stream": False, "messages": [{"role": "user", "content": content, "images": [image_b64(p) for p in images]}], "options": {"temperature": 0, "num_ctx": config.get("num_ctx", 8192), "num_predict": config.get("num_predict", 1200), "seed": config.get("seed", 20260911)}}
    if schema:
        body["format"] = schema
    attempts = config.get("max_retries", 2) + 1
    last = None
    for attempt in range(1, attempts + 1):
        status, text, elapsed = post(config["url"], body, config.get("timeout_seconds", 900))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.with_name(f"{raw_path.stem}_attempt{attempt}.json").write_text(text, encoding="utf-8")
        last = text
        if status == 200:
            try:
                response = json.loads(text)
                result = unique_json_object(response["message"]["content"])
                result["elapsed_seconds"] = elapsed
                return result
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                error = str(exc)
        else:
            error = f"ollama_http_{status}"
        if attempt < attempts:
            time.sleep(config.get("retry_wait_seconds", 10) * attempt)
    raise ValueError(error if last is not None else "ollama_failed")


def feature_schema(schema_path: Path):
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    fields = {}
    for item in schema["universal_features"] + schema.get("category_specific_features", []):
        kind = {"set": "array", "integer": "integer", "number": "number"}.get(item["value_type"], "string")
        fields[item["feature_id"]] = {"type": kind}
    return {"type": "object", "properties": {"schema_version": {"type": "string"}, "image_id": {"type": "string"}, "target_instance_id": {"type": "string"}, "features": {"type": "object", "properties": fields, "additionalProperties": False}, "evidence": {"type": "object", "additionalProperties": {"type": "string"}}}, "required": ["schema_version", "image_id", "target_instance_id", "features", "evidence"], "additionalProperties": False}


def validate_feature(result, row, schema):
    if result.get("schema_version") != schema["schema_version"] or result.get("image_id") != row["image_id"] or result.get("target_instance_id") != "target_1":
        raise ValueError("feature_identity_mismatch")
    definitions = {item["feature_id"]: item for item in schema["universal_features"] + schema.get("category_specific_features", [])}
    features = result.get("features")
    if set(features or {}) != set(definitions):
        raise ValueError("feature_fields_mismatch")
    for key, value in features.items():
        definition = definitions[key]
        allowed = definition.get("possible_values", [])
        if definition["value_type"] == "enum" and value not in allowed:
            raise ValueError(f"invalid_feature_value:{key}")
        if definition["value_type"] == "integer" and (type(value) is not int or value < 0):
            raise ValueError(f"invalid_feature_value:{key}")
        if definition["value_type"] == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"invalid_feature_value:{key}")
        if definition["value_type"] == "set" and (not isinstance(value, list) or any(v not in allowed for v in value)):
            raise ValueError(f"invalid_feature_value:{key}")
        if definition["value_type"] == "set":
            if not value or len(value) != len(set(value)):
                raise ValueError(f"invalid_feature_value:{key}")
            if any(exclusive in value for exclusive in definition.get("exclusive_values", [])) and len(value) != 1:
                raise ValueError(f"invalid_feature_value:{key}")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(features):
        raise ValueError("evidence_fields_mismatch")
    if any(not isinstance(value, str) or not value.strip() or re.search(r"https?://|www\\.", value, re.I) for value in evidence.values()):
        raise ValueError("invalid_evidence")
    if features.get("mouth_visibility") == "not_visible" and (features.get("mouth_state") != "unknown" or features.get("tongue_visibility") != "unknown"):
        raise ValueError("mouth_dependency_mismatch")
    if features.get("tail_visibility") == "not_observable" and features.get("tail_posture") != "unknown":
        raise ValueError("tail_dependency_mismatch")
    return result


def validate_class(result, row):
    if result.get("image_id") != row["image_id"] or result.get("target_instance_id") != "target_1":
        raise ValueError("classification_identity_mismatch")
    if result.get("predicted_class") not in COCO_CANDIDATES:
        raise ValueError("invalid_predicted_class")
    if result.get("prediction_status") not in {"ok", "unknown"}:
        raise ValueError("invalid_prediction_status")
    if not isinstance(result.get("evidence"), str) or not result["evidence"].strip():
        raise ValueError("empty_classification_evidence")
    result["classification_correct"] = result["predicted_class"] == row["target_class"]
    return result


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def program_fields(row, feature):
    bbox = json.loads(row["bbox"])
    with Image.open(row["image_path"]) as image:
        width, height = image.size
    x1, y1, x2, y2 = bbox
    area = ((x2 - x1) * (y2 - y1)) / (width * height)
    center_x, center_y = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
    truncation = "none" if x1 > 0 and y1 > 0 and x2 < width and y2 < height else "partial"
    factors = []
    if area < .03: factors.append("small_target")
    if feature.get("occlusion_level") in {"partial", "heavy"}: factors.append("occlusion")
    if truncation != "none": factors.append("truncation")
    if feature.get("target_background_contrast") == "low": factors.append("low_contrast")
    if feature.get("motion_blur") in {"mild", "strong"}: factors.append("blur")
    if feature.get("background_complexity") == "complex": factors.append("complex_background")
    if feature.get("pose") in {"jumping", "running"}: factors.append("unusual_pose")
    if any(feature.get(k) in {"partial", "not_visible", "one", "none"} for k in ("head_visibility", "muzzle_visibility", "eye_visibility", "tail_visibility", "leg_visibility")): factors.append("missing_key_parts")
    label_path = Path(row["image_path"]).parent.parent / "labels" / (Path(row["image_path"]).stem + ".txt")
    visible_dogs = sum(line.split()[0] == "16" for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()) if label_path.exists() else None
    return {"target_bbox": row["bbox"], "target_area_ratio": round(area, 6), "target_center_x": round(center_x, 6), "target_center_y": round(center_y, 6), "image_width": width, "image_height": height, "visible_dog_count_in_scene": visible_dogs, "truncation_level_from_bbox": truncation, "feature_unknown_count": sum(v == "unknown" for v in feature.values()), "visibility_score": None, "candidate_difficulty_factors": json.dumps(sorted(set(factors)), ensure_ascii=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("artifacts/dog_feature_pipeline_v1/target_manifest.csv"))
    ap.add_argument("--schema", type=Path, default=Path("configs/dog_feature_schema_v2.json"))
    ap.add_argument("--feature-prompt", type=Path, default=Path("prompts/dog_feature_annotation_v2.txt"))
    ap.add_argument("--class-prompt", type=Path, default=Path("prompts/dog_independent_classifier_v1.txt"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/dog_feature_pipeline_v1/results"))
    ap.add_argument("--url", default="http://localhost:11434/api/chat")
    ap.add_argument("--model", default="qwen3-vl:30b-a3b-instruct")
    ap.add_argument("--num-predict", type=int, default=2200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N rows")
    ap.add_argument("--stages", choices=("all", "features", "classifier"), default="all")
    args = ap.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig", newline="")))
    if args.limit:
        rows = rows[:args.limit]
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    compact_schema = {"schema_version": schema["schema_version"], "category": schema["category"], "features": [{k: item.get(k, True) for k in ("feature_id", "value_type", "possible_values", "allow_unknown")} for item in schema["universal_features"] + schema.get("category_specific_features", [])]}
    feature_prompt = args.feature_prompt.read_text(encoding="utf-8") + "\n特征字典如下：\n" + json.dumps(compact_schema, ensure_ascii=False)
    class_prompt = args.class_prompt.read_text(encoding="utf-8") + "\n候选类别列表：" + json.dumps(COCO_CANDIDATES, ensure_ascii=False)
    config = {"url": args.url, "model": args.model, "num_ctx": 8192, "num_predict": args.num_predict, "timeout_seconds": 900, "max_retries": 3, "retry_wait_seconds": 20, "seed": 20260911}
    for sub in ("raw/features", "raw/classifier", "parsed/features", "parsed/classifier"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    status_path = args.output / "pipeline_status.json"
    input_protocol = rows[0].get("protocol_version", "unknown") if rows else "unknown"
    status = {"status": "running", "stage": "features", "started_at": now(), "planned": len(rows), "completed_features": 0, "completed_classifier": 0, "failed": 0, "gpu_model": args.model, "schema_version": schema["schema_version"], "input_protocol_version": input_protocol}
    atomic_json(status_path, status)
    features, classes, errors = {}, {}, []
    if args.stages in {"all", "features"}:
      for index, row in enumerate(rows, 1):
          status.update(stage="features", current_image=row["image_id"], heartbeat=now(), completed_features=len(features), completed_classifier=len(classes), failed=len(errors)); atomic_json(status_path, status)
          key = row["image_id"]
          feature_file = args.output / "parsed/features" / f"{key}.json"
          try:
              if args.resume and feature_file.exists():
                  value = json.loads(feature_file.read_text(encoding="utf-8")); result = validate_feature(value, row, schema)
              else:
                  result = validate_feature(ask(config, feature_prompt, row, [Path(row["image_path"]), Path(row["marked_image_path"]), Path(row["crop_image_path"])], args.output / "raw/features" / f"{key}.json", feature_schema(args.schema)), row, schema)
                  atomic_json(feature_file, result)
              features[key] = result
              print(f"[features] {index}/{len(rows)} {key} ok", flush=True)
          except Exception as exc:
              errors.append({"stage": "features", "image_id": key, "error": str(exc)}); print(f"[features] {index}/{len(rows)} {key} ERROR {exc}", flush=True)
    if args.stages in {"all", "classifier"}:
      for index, row in enumerate(rows, 1):
          status.update(stage="classifier", current_image=row["image_id"], heartbeat=now(), completed_features=len(features), completed_classifier=len(classes), failed=len(errors))
          key = row["image_id"]
          class_file = args.output / "parsed/classifier" / f"{key}.json"
          try:
              if args.resume and class_file.exists():
                  result = validate_class(json.loads(class_file.read_text(encoding="utf-8")), row)
              else:
                  result = validate_class(ask(config, class_prompt, row, [Path(row["image_path"]), Path(row["marked_image_path"]), Path(row["crop_image_path"])], args.output / "raw/classifier" / f"{key}.json"), row)
                  atomic_json(class_file, result)
              classes[key] = result
              print(f"[classifier] {index}/{len(rows)} {key} {result['predicted_class']}", flush=True)
          except Exception as exc:
              errors.append({"stage": "classifier", "image_id": key, "error": str(exc)}); print(f"[classifier] {index}/{len(rows)} {key} ERROR {exc}", flush=True)
    result_rows = []
    for row in rows:
        key = row["image_id"]
        feature = features.get(key, {}); cls = classes.get(key, {})
        flat = {"image_id": key, "target_instance_id": row["target_instance_id"], "target_class": row["target_class"], "split": row["split"], "predicted_class": cls.get("predicted_class", ""), "classification_correct": cls.get("classification_correct", ""), "feature_status": "ok" if feature else "failed", "classifier_status": cls.get("prediction_status", "failed"), "protocol_version": schema["schema_version"], "input_protocol_version": row.get("protocol_version", "unknown")}
        for field, value in feature.get("features", {}).items(): flat[field] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
        flat.update(program_fields(row, feature.get("features", {})))
        result_rows.append(flat)
    fields = ["image_id", "target_instance_id", "target_class", "split", "predicted_class", "classification_correct", "feature_status", "classifier_status", "protocol_version", "input_protocol_version", "target_bbox", "target_area_ratio", "target_center_x", "target_center_y", "image_width", "image_height", "visible_dog_count_in_scene", "truncation_level_from_bbox", "feature_unknown_count", "visibility_score", "candidate_difficulty_factors"] + [x["feature_id"] for x in schema["universal_features"]]
    save_csv(args.output / "results.csv", result_rows, fields)
    stats = {"protocol_version": schema["schema_version"], "input_protocol_version": input_protocol, "planned": len(rows), "feature_ok": len(features), "classifier_ok": len(classes), "classification_correct": sum(bool(x.get("classification_correct")) for x in classes.values()), "classification_accuracy": sum(bool(x.get("classification_correct")) for x in classes.values()) / len(classes) if classes else None, "errors": len(errors), "unknown_predictions": sum(x.get("predicted_class") == "unknown" for x in classes.values())}
    atomic_json(args.output / "errors.json", errors); atomic_json(args.output / "report.json", stats); status.update(status="complete", stage="report", heartbeat=now(), completed_features=len(features), completed_classifier=len(classes), failed=len(errors), current_image=None); atomic_json(status_path, status)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
