"""Resumable automatic category, schema, and feature pipeline."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

from PIL import Image

from qe_quality.dta.io import atomic_json, unique_json_object
from .category import CATEGORY_CANDIDATES, aggregate_category, validate_category_result
from .schema_generator import build_schema_prompt, representative_rows
from .schema_validator import feature_output_schema, fingerprint_schema, freeze_schema, validate_feature_result, validate_schema


def now():
    return datetime.now(timezone.utc).isoformat()


def _image_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    parsed = urllib.parse.urlsplit(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if parsed.hostname in {"localhost", "127.0.0.1", "::1"} else urllib.request.build_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def call_vlm(config, prompt, images, output_path, response_schema=None):
    body = {
        "model": config["model"], "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [_image_b64(path) for path in images]}],
        "options": {"num_ctx": config.get("num_ctx", 8192), "temperature": config.get("temperature", 0), "num_predict": config.get("num_predict", 2200), "seed": config.get("seed", 20260913)},
    }
    if response_schema is not None:
        body["format"] = response_schema
    attempts = int(config.get("max_retries", 2)) + 1
    last_error = "vlm_failed"
    for attempt in range(1, attempts + 1):
        status, text = _post(config["url"], body, config.get("timeout_seconds", 900))
        attempt_path = Path(output_path).with_name(f"{Path(output_path).stem}_attempt{attempt}.json")
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path.write_text(text, encoding="utf-8")
        if status != 200:
            last_error = f"vlm_http_{status}"
        else:
            try:
                content = json.loads(text)["message"]["content"]
                return unique_json_object(content)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
        if attempt < attempts:
            time.sleep(config.get("retry_wait_seconds", 5) * attempt)
    raise ValueError(last_error)


def load_manifest(path):
    manifest_path = Path(path).resolve()
    required = {"image_id", "target_instance_id", "image_path", "bbox", "marked_image_path", "crop_image_path", "target_class"}
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty_target_manifest")
    if not required.issubset(rows[0]):
        raise ValueError("target_manifest_fields_mismatch")
    ids = set()
    for row in rows:
        if row["image_id"] in ids:
            raise ValueError("duplicate_image_id")
        ids.add(row["image_id"])
        for key in ("image_path", "marked_image_path", "crop_image_path"):
            candidate = Path(row[key]).expanduser()
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            if not candidate.is_file():
                raise FileNotFoundError(str(candidate))
            row[key] = str(candidate.resolve())
        try:
            box = json.loads(row["bbox"])
            if len(box) != 4 or not box[0] < box[2] or not box[1] < box[3]:
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid_bbox:{row['image_id']}") from exc
    return rows


def _category_response_schema():
    return {"type": "object", "properties": {"image_id": {"type": "string"}, "target_instance_id": {"type": "string"}, "predicted_category": {"type": "string", "enum": CATEGORY_CANDIDATES}, "fine_name": {"type": "string"}, "confidence": {"type": "number"}, "evidence": {"type": "string"}}, "required": ["image_id", "target_instance_id", "predicted_category", "fine_name", "confidence", "evidence"], "additionalProperties": False}


def _program_fields(row, feature_values):
    box = json.loads(row["bbox"])
    with Image.open(row["image_path"]) as image:
        width, height = image.size
    x1, y1, x2, y2 = map(float, box)
    area = (x2 - x1) * (y2 - y1) / (width * height)
    truncation = "none" if x1 > 0 and y1 > 0 and x2 < width and y2 < height else "partial"
    return {"target_bbox": row["bbox"], "image_width": width, "image_height": height, "target_area_ratio": round(area, 6), "target_center_x": round((x1 + x2) / 2 / width, 6), "target_center_y": round((y1 + y2) / 2 / height, 6), "truncation_level_from_bbox": truncation, "feature_unknown_count": sum(value == "unknown" for value in feature_values.values())}


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(manifest, output, policy_path, category_prompt_path, schema_prompt_path, feature_prompt_path, config, resume=False, reuse_category_from=None):
    rows = load_manifest(manifest)
    output = Path(output)
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    atomic_json(status_path, {"status": "running", "stage": "category", "started_at": now(), "planned": len(rows)})
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8-sig"))
    category_prompt = Path(category_prompt_path).read_text(encoding="utf-8") + "\n候选类别：" + json.dumps(CATEGORY_CANDIDATES, ensure_ascii=False)
    categories, errors = {}, []
    (output / "raw/category").mkdir(parents=True, exist_ok=True)
    (output / "parsed/category").mkdir(parents=True, exist_ok=True)
    summary = None
    if reuse_category_from:
        source = Path(reuse_category_from)
        try:
            source_results = json.loads((source / "category_results.json").read_text(encoding="utf-8"))
            source_summary = json.loads((source / "category_summary.json").read_text(encoding="utf-8"))
            categories = {row["image_id"]: validate_category_result(source_results[row["image_id"]], row) for row in rows}
            summary = aggregate_category(list(categories.values()), policy.get("min_category_consensus", .8), policy.get("min_category_confidence", .6))
            if summary["category"] != source_summary.get("category"):
                raise ValueError("reused_category_summary_mismatch")
            atomic_json(output / "category_results.json", categories)
            atomic_json(output / "category_summary.json", summary)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            atomic_json(output / "errors.json", [{"stage": "category_reuse", "error": str(exc)}])
            raise ValueError("category_reuse_failed") from exc
    max_category_rounds = int(policy.get("max_category_rounds", 2))
    for category_round in range(1, max_category_rounds + 1) if summary is None else []:
        categories, errors = {}, []
        for row in rows:
            key = row["image_id"]
            parsed_path = output / "parsed/category" / f"{key}.json"
            try:
                if resume and category_round == 1 and parsed_path.exists():
                    result = json.loads(parsed_path.read_text(encoding="utf-8"))
                else:
                    correction = ""
                    if category_round > 1:
                        correction = "\n上一轮类别结果在整批一致性检查中不通过。请重新检查目标主体，给出最可能的候选类别，不要使用 unknown，除非目标确实不可识别。"
                    result = call_vlm(config, category_prompt + correction + f"\nimage_id: {key}\ntarget_instance_id: {row['target_instance_id']}", [row["image_path"], row["marked_image_path"], row["crop_image_path"]], output / "raw/category" / f"{key}_round{category_round}.json", _category_response_schema())
                    atomic_json(parsed_path, result)
                value = validate_category_result(result, row)
                if value["confidence"] < policy.get("min_category_confidence", .6) and category_round < max_category_rounds:
                    raise ValueError("category_confidence_below_threshold")
                categories[key] = value
            except Exception as exc:
                errors.append({"stage": "category", "image_id": key, "round": category_round, "error": str(exc)})
        if errors:
            if category_round < max_category_rounds:
                continue
            atomic_json(output / "errors.json", errors)
            raise ValueError("category_stage_failed")
        try:
            summary = aggregate_category(list(categories.values()), policy.get("min_category_consensus", .8), policy.get("min_category_confidence", .6))
            break
        except ValueError as exc:
            if category_round >= max_category_rounds:
                atomic_json(output / "errors.json", [{"stage": "category", "round": category_round, "error": str(exc)}])
                raise
    if summary is None:
        raise ValueError("mixed_or_unresolved_category")
    atomic_json(output / "category_results.json", categories)
    atomic_json(output / "category_summary.json", summary)
    atomic_json(status_path, {"status": "running", "stage": "schema", "started_at": now(), "planned": len(rows), "category": summary["category"]})
    samples = representative_rows(rows, int(policy.get("schema_sample_size", 12)))
    schema_raw = output / "raw/schema/schema_proposal.json"
    schema_prompt = build_schema_prompt(Path(schema_prompt_path).read_text(encoding="utf-8"), summary, policy)
    schema = None
    max_schema_attempts = int(policy.get("max_schema_attempts", 3))
    schema_errors = []
    if resume and (output / "schema_frozen.json").exists():
        schema = json.loads((output / "schema_frozen.json").read_text(encoding="utf-8"))
        validate_schema({key: schema[key] for key in ("schema_version", "category", "features", "rules")}, policy)
        if schema.get("schema_fingerprint") != fingerprint_schema(schema):
            raise ValueError("schema_fingerprint_mismatch")
        if schema.get("category") != summary["category"]:
            raise ValueError("schema_category_mismatch")
    else:
        for schema_attempt in range(1, max_schema_attempts + 1):
            try:
                feedback = ""
                if schema_errors:
                    feedback = "\n上一份 schema 未通过程序校验，必须修正以下错误后重新输出：" + "; ".join(schema_errors[-3:])
                proposal = call_vlm(config, schema_prompt + feedback, [row["crop_image_path"] for row in samples], output / "raw/schema" / f"schema_proposal_round{schema_attempt}.json", {"type": "object"})
                atomic_json(output / "schema_proposal.json", proposal)
                validate_schema(proposal, policy)
                if proposal["category"] != summary["category"]:
                    raise ValueError("schema_category_mismatch")
                schema = freeze_schema(proposal, policy)
                atomic_json(output / "schema_frozen.json", schema)
                (output / "schema_fingerprint.txt").write_text(schema["schema_fingerprint"] + "\n", encoding="utf-8")
                atomic_json(output / "schema_validation.json", {"status": "valid", "attempts": schema_attempt, "schema_fingerprint": schema["schema_fingerprint"]})
                break
            except Exception as exc:
                schema_errors.append(str(exc))
        if schema is None:
            atomic_json(output / "schema_validation.json", {"status": "invalid", "attempts": max_schema_attempts, "errors": schema_errors})
            atomic_json(output / "errors.json", errors + [{"stage": "schema", "error": schema_errors[-1] if schema_errors else "schema_generation_failed"}])
            raise ValueError("schema_generation_failed")
    atomic_json(status_path, {"status": "running", "stage": "features", "started_at": now(), "planned": len(rows), "category": summary["category"], "schema_fingerprint": schema["schema_fingerprint"]})
    feature_prompt = Path(feature_prompt_path).read_text(encoding="utf-8") + "\n冻结 schema：\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    feature_schema = feature_output_schema(schema)
    features = {}
    max_feature_validation_attempts = int(policy.get("max_feature_validation_attempts", 3))
    (output / "raw/features").mkdir(parents=True, exist_ok=True)
    (output / "parsed/features").mkdir(parents=True, exist_ok=True)
    for row in rows:
        key = row["image_id"]
        parsed_path = output / "parsed/features" / f"{key}.json"
        last_error = None
        for feature_attempt in range(1, max_feature_validation_attempts + 1):
            try:
                if resume and feature_attempt == 1 and parsed_path.exists():
                    result = json.loads(parsed_path.read_text(encoding="utf-8"))
                else:
                    correction = ""
                    if last_error:
                        correction = (
                            "\n上一份特征 JSON 未通过严格校验，必须修正以下错误后重新输出："
                            + str(last_error)
                            + "。set 字段不能输出空数组；无法判断时使用 [\"unknown\"]；数组值不能重复。"
                        )
                    result = call_vlm(config, feature_prompt + correction + f"\nimage_id: {key}\ntarget_instance_id: {row['target_instance_id']}", [row["image_path"], row["marked_image_path"], row["crop_image_path"]], output / "raw/features" / f"{key}_validation{feature_attempt}.json", feature_schema)
                validated = validate_feature_result(result, row, schema)
                atomic_json(parsed_path, validated)
                features[key] = validated
                break
            except Exception as exc:
                last_error = str(exc)
                if feature_attempt >= max_feature_validation_attempts:
                    errors.append({"stage": "features", "image_id": key, "error": last_error})
    result_rows = []
    for row in rows:
        key = row["image_id"]
        category = categories[key]
        feature = features.get(key, {})
        values = feature.get("features", {})
        result_row = {"image_id": key, "target_instance_id": row["target_instance_id"], "target_class": row["target_class"], "predicted_category": category["predicted_category"], "category_confidence": category["confidence"], "classification_correct": bool(row["target_class"]) and category["predicted_category"] == row["target_class"], "schema_version": schema["schema_version"], "schema_fingerprint": schema["schema_fingerprint"], "feature_status": "ok" if feature else "failed", **_program_fields(row, values)}
        result_row.update({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in values.items()})
        result_rows.append(result_row)
    base_fields = ["image_id", "target_instance_id", "target_class", "predicted_category", "category_confidence", "classification_correct", "schema_version", "schema_fingerprint", "feature_status", "target_bbox", "image_width", "image_height", "target_area_ratio", "target_center_x", "target_center_y", "truncation_level_from_bbox", "feature_unknown_count"]
    feature_fields = [item["feature_id"] for item in schema["features"]]
    fields = base_fields + feature_fields
    _write_csv(output / "results.csv", result_rows, fields)
    atomic_json(output / "errors.json", errors)
    distributions = {}
    for item in schema["features"]:
        feature_id = item["feature_id"]
        counts = Counter()
        for value in (features[key]["features"].get(feature_id) for key in features):
            counts[json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, list) else str(value)] += 1
        distributions[feature_id] = dict(sorted(counts.items()))
    atomic_json(output / "feature_distribution.json", distributions)
    correct = sum(row["classification_correct"] is True for row in result_rows)
    report = {"status": "complete" if not errors else "needs_improvement", "planned": len(rows), "category_ok": len(categories), "feature_ok": len(features), "errors": len(errors), "classification_correct": correct, "classification_accuracy": correct / len(rows) if rows else None, "category_summary": summary, "schema_version": schema["schema_version"], "schema_fingerprint": schema["schema_fingerprint"]}
    atomic_json(output / "report.json", report)
    atomic_json(status_path, {**report, "stage": "report", "finished_at": now()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/auto_feature_policy_v1.json"))
    parser.add_argument("--category-prompt", type=Path, default=Path("prompts/category_discovery_v1.txt"))
    parser.add_argument("--schema-prompt", type=Path, default=Path("prompts/schema_generator_v1.txt"))
    parser.add_argument("--feature-prompt", type=Path, default=Path("prompts/feature_annotation_auto_v1.txt"))
    parser.add_argument("--url", default="http://localhost:11434/api/chat")
    parser.add_argument("--model", default="qwen3-vl:30b-a3b-instruct")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-category-from", type=Path, default=None, help="reuse validated category_results.json and category_summary.json from a prior run")
    parser.add_argument("--num-predict", type=int, default=2200)
    args = parser.parse_args(argv)
    config = {"url": args.url, "model": args.model, "num_ctx": 8192, "temperature": 0, "num_predict": args.num_predict, "timeout_seconds": 900, "max_retries": 2, "retry_wait_seconds": 5, "seed": 20260913}
    try:
        report = run_pipeline(args.manifest, args.output, args.policy, args.category_prompt, args.schema_prompt, args.feature_prompt, config, args.resume, args.reuse_category_from)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"auto-feature: {exc}\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
