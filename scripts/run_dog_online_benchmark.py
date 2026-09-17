#!/usr/bin/env python3
"""Blind online-VLM classification benchmark for the clean dog596 release."""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.clients import image_uri, load_online_config, post_json
from qe_quality.dta.io import atomic_csv, atomic_json, fingerprint, read_csv, unique_json_object
from scripts.run_dog_local_pipeline import COCO_CANDIDATES, validate_class

BASELINE_MANIFEST = ROOT / "artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv"
CANDIDATE_MANIFEST = ROOT / "artifacts/dog_candidate400_v2_1_inputs_20260912/target_manifest.csv"
CLEAN_RESULTS = ROOT / "artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv"
PROMPT_PATH = ROOT / "prompts/dog_independent_classifier_v1.txt"
CONFIG_PATH = ROOT / "artifacts/feature_pilot_20260908/api_config.local.json"
DEFAULT_OUTPUT = ROOT / "artifacts/dog596_online_sensenova_20260914"


def now():
    return datetime.now(timezone.utc).isoformat()


def request_one(config, prompt, row, raw_dir):
    content = [
        {"type": "text", "text": prompt + "\nimage_id: " + row["image_id"] + "\ntarget_instance_id: target_1"},
        {"type": "image_url", "image_url": {"url": image_uri(row["image_path"])}},
        {"type": "image_url", "image_url": {"url": image_uri(row["marked_image_path"])}},
        {"type": "image_url", "image_url": {"url": image_uri(row["crop_image_path"])}},
    ]
    body = {**config.get("request_parameters", {}), "model": config["model"],
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}]}
    attempts = config.get("max_retries", 5) + 1
    last_finished = None
    for attempt in range(1, attempts + 1):
        if last_finished is not None:
            time.sleep(max(0, config.get("interval_seconds", 10) - (time.monotonic() - last_finished)))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        status, text, elapsed, headers = post_json(config["url"], body,
            {"Authorization": "Bearer " + config["api_key"]}, config.get("timeout_seconds", 600))
        last_finished = time.monotonic()
        safe = text.replace(config["api_key"], "[REDACTED]")
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{row['image_id']}_{stamp}_attempt{attempt}.json").write_text(safe, encoding="utf-8")
        if status in {429, 500, 502, 503, 504} and attempt < attempts:
            try:
                retry = float(headers.get("Retry-After", config.get("retry_wait_seconds", 120)))
            except (TypeError, ValueError):
                retry = config.get("retry_wait_seconds", 120)
            time.sleep(max(retry, config.get("retry_wait_seconds", 120)))
            continue
        if status != 200:
            raise ValueError(f"online_http_{status}")
        try:
            outer = json.loads(safe)
            result = unique_json_object(outer["choices"][0]["message"]["content"])
            result = validate_class(result, row)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            if attempt < attempts:
                body["messages"] += [
                    {"role": "assistant", "content": outer.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(locals().get("outer"), dict) else ""},
                    {"role": "user", "content": f"上一条输出未通过严格校验：{exc}。请只返回符合指定字段和候选类别的JSON对象。"},
                ]
                continue
            raise
        result.update({"elapsed_seconds": elapsed, "backend": "sensenova_online", "attempts": attempt})
        return result
    raise ValueError("online_retries_exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = load_online_config(args.config)
    config.update({"interval_seconds": 10, "retry_wait_seconds": 120, "max_retries": 5})
    clean = read_csv(CLEAN_RESULTS)
    wanted = {r["image_id"] for r in clean}
    manifests = {r["image_id"]: r for r in read_csv(BASELINE_MANIFEST) + read_csv(CANDIDATE_MANIFEST)}
    rows = [manifests[r["image_id"]] for r in clean]
    assert len(rows) == len(wanted) == 596
    if args.limit:
        rows = rows[:args.limit]
    prompt = PROMPT_PATH.read_text(encoding="utf-8") + "\n候选类别列表：" + json.dumps(COCO_CANDIDATES, ensure_ascii=False)
    protocol = {"protocol_version": "dog596-online-blind-classification-v1", "model": config["model"],
                "images": len(rows), "inputs": ["original", "A_marked", "enlarged_crop"],
                "prompt_sha256": fingerprint(prompt), "candidate_classes": COCO_CANDIDATES,
                "target_label_sent_to_model": False, "dog_features_sent_to_model": False,
                "other_model_results_sent_to_model": False}
    protocol["protocol_hash"] = fingerprint(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "protocol.json", protocol)
    parsed_dir = args.output / "parsed"
    raw_dir = args.output / "raw"
    status_path = args.output / "pipeline_status.json"
    results, errors = {}, []
    for row in rows:
        path = parsed_dir / f"{row['image_id']}.json"
        if args.resume and path.exists():
            try:
                results[row["image_id"]] = validate_class(json.loads(path.read_text(encoding="utf-8")), row)
            except Exception:
                pass
    started = now()
    last_image_finished = None
    for index, row in enumerate(rows, 1):
        key = row["image_id"]
        atomic_json(status_path, {"status": "running", "stage": "online_classifier", "planned": len(rows),
            "completed": len(results), "failed": len(errors), "remaining": len(rows)-index+1,
            "current_image": key, "started_at": started, "heartbeat": now(), "model": config["model"]})
        if key in results:
            print(f"[online] {index}/{len(rows)} {key} resume", flush=True)
            continue
        try:
            if last_image_finished is not None:
                time.sleep(max(0, config.get("interval_seconds", 10) - (time.monotonic() - last_image_finished)))
            value = request_one(config, prompt, row, raw_dir)
            atomic_json(parsed_dir / f"{key}.json", value)
            results[key] = value
            print(f"[online] {index}/{len(rows)} {key} {value['predicted_class']} ok", flush=True)
        except Exception as exc:
            errors.append({"image_id": key, "error": str(exc), "at": now()})
            print(f"[online] {index}/{len(rows)} {key} ERROR {exc}", flush=True)
        finally:
            last_image_finished = time.monotonic()
        atomic_json(args.output / "errors.json", errors)
    output_rows = []
    for source in rows:
        value = results.get(source["image_id"], {})
        output_rows.append({"image_id": source["image_id"], "target_instance_id": "target_1",
            "dataset_source": "baseline496" if source["image_id"] in {r["image_id"] for r in read_csv(BASELINE_MANIFEST)} else "supplement100",
            "predicted_class": value.get("predicted_class", ""), "prediction_status": value.get("prediction_status", "failed"),
            "classification_correct": value.get("classification_correct", ""), "evidence": value.get("evidence", ""),
            "elapsed_seconds": value.get("elapsed_seconds", ""), "attempts": value.get("attempts", "")})
    atomic_csv(args.output / "results.csv", output_rows, list(output_rows[0]))
    counts = Counter(r["predicted_class"] for r in output_rows if r["prediction_status"] != "failed")
    report = {"protocol_version": protocol["protocol_version"], "protocol_hash": protocol["protocol_hash"],
        "model": config["model"], "planned": len(rows), "successful": len(results), "failed": len(errors),
        "dog_predictions": counts["dog"], "unknown_predictions": counts["unknown"],
        "dog_positive_recall": counts["dog"] / len(results) if results else None,
        "prediction_distribution": dict(counts), "features_visible_to_model": False,
        "ground_truth_visible_to_model": False}
    atomic_json(args.output / "report.json", report)
    final = "complete" if not errors and len(results) == len(rows) else "needs_retry"
    atomic_json(status_path, {"status": final, "stage": "report", "planned": len(rows), "completed": len(results),
        "failed": len(errors), "remaining": len(rows)-len(results), "current_image": None,
        "started_at": started, "heartbeat": now(), "model": config["model"]})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
