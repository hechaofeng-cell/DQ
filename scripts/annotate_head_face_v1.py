#!/usr/bin/env python3
"""Annotate the frozen 200-face candidate pool with head_face_v1.0."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qe_quality.auto_feature.pipeline import call_vlm
from qe_quality.auto_feature.schema_validator import feature_output_schema, validate_feature_result, validate_schema


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "artifacts/head_face_v1_0_20260922"
DEFAULT_SCHEMA = ROOT / "configs/head_face_feature_schema_v1_0.json"
DEFAULT_POLICY = ROOT / "configs/auto_feature_policy_head_face_v1_0.json"
DEFAULT_PROMPT = ROOT / "prompts/feature_annotation_head_face_v1_0.txt"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model", default="qwen3-vl:30b-a3b-instruct")
    parser.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--annotation-status",
        default="automatic_vlm_pending_human_review",
        help="provenance label written to progress.json; does not alter validation",
    )
    args = parser.parse_args()

    rows = read_csv(args.experiment / "manifests/face_candidates.csv")
    if args.limit is not None:
        rows = rows[: args.limit]
    schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
    validate_schema(schema, policy)
    response_schema = feature_output_schema(schema)
    prompt = DEFAULT_PROMPT.read_text(encoding="utf-8") + "\n冻结 schema：\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    config = {
        "url": args.url,
        "model": args.model,
        "num_ctx": 16384,
        "temperature": 0,
        "num_predict": 4096,
        "seed": 20260922,
        "timeout_seconds": 900,
        "max_retries": 2,
        "retry_wait_seconds": 5,
    }
    output = args.experiment / "annotations"
    parsed_dir = output / "parsed"
    raw_dir = output / "raw"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for index, row in enumerate(rows, start=1):
        parsed_path = parsed_dir / f"{row['sample_id']}.json"
        try:
            if args.resume and parsed_path.is_file():
                result = json.loads(parsed_path.read_text(encoding="utf-8"))
            else:
                request_row = {"image_id": row["sample_id"], "target_instance_id": row["target_instance_id"]}
                images = [row["image_path"], row["marked_image_path"], row["crop_image_path"], row["face_crop_path"]]
                validation_error = None
                result = None
                for semantic_attempt in range(1, 4):
                    correction = ""
                    if validation_error:
                        correction = (
                            "\n上一份JSON未通过严格校验，错误为：" + validation_error
                            + "。请重新检查全部字段。set字段至少一个值；none和unknown不能与其他值并存。"
                        )
                    candidate = call_vlm(
                        config,
                        prompt + correction + f"\nimage_id: {row['sample_id']}\ntarget_instance_id: {row['target_instance_id']}",
                        images,
                        raw_dir / f"{row['sample_id']}_semantic{semantic_attempt}.json",
                        response_schema,
                    )
                    try:
                        result = validate_feature_result(candidate, request_row, schema)
                        break
                    except ValueError as error:
                        validation_error = str(error)
                if result is None:
                    raise ValueError(validation_error or "semantic_validation_failed")
                parsed_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[annotate] {index}/{len(rows)} {row['sample_id']} ok", flush=True)
        except Exception as error:
            errors.append({"sample_id": row["sample_id"], "error": str(error)})
            print(f"[annotate] {index}/{len(rows)} {row['sample_id']} ERROR {error}", flush=True)
        (output / "progress.json").write_text(json.dumps({
            "status": "running", "completed": index - len(errors), "failed": len(errors),
            "planned": len(rows), "latest": row["sample_id"], "updated_at": now(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = []
    distributions = {item["feature_id"]: Counter() for item in schema["features"]}
    source = {row["sample_id"]: row for row in rows}
    for parsed_path in sorted(parsed_dir.glob("*.json")):
        result = json.loads(parsed_path.read_text(encoding="utf-8"))
        if result["image_id"] not in source:
            continue
        features = result["features"]
        row = source[result["image_id"]]
        flattened: dict[str, object] = {
            "sample_id": result["image_id"], "target_instance_id": result["target_instance_id"],
            "schema_version": result["schema_version"], "feature_status": "ok",
            "wider_blur": row["wider_blur"], "wider_illumination": row["wider_illumination"],
            "wider_occlusion": row["wider_occlusion"], "wider_pose": row["wider_pose"],
            "head_box_truncated": row.get("head_box_truncated", ""),
        }
        for key, value in features.items():
            flattened[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            if isinstance(value, list):
                for state in value:
                    distributions[key][state] += 1
            else:
                distributions[key][value] += 1
        flat.append(flattened)
    write_csv(output / "results.csv", flat)
    (output / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "feature_distribution.json").write_text(json.dumps({
        key: dict(sorted(values.items())) for key, values in distributions.items()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "complete" if len(flat) == len(rows) and not errors else "needs_review"
    (output / "progress.json").write_text(json.dumps({
        "status": status, "completed": len(flat), "failed": len(errors), "planned": len(rows),
        "updated_at": now(), "annotation_status": args.annotation_status,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "complete":
        raise RuntimeError(f"annotation incomplete: {len(flat)}/{len(rows)}, errors={len(errors)}")


if __name__ == "__main__":
    main()
