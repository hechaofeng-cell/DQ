#!/usr/bin/env python3
"""Annotate a frozen multiclass target manifest with the online SenseNova VLM."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hcf/project/QE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

from qe_quality.dta.clients import image_uri, load_online_config, post_json
from qe_quality.dta.io import atomic_json, unique_json_object
from run_dog_local_pipeline import validate_feature

DEFAULT_CONFIG = Path("/home/hcf/project/QE/artifacts/feature_pilot_20260908/api_config.local.json")
DEFAULT_SCHEMA = ROOT / "configs/multiclass_object_state_schema_v1_0.json"
DEFAULT_PROMPT = ROOT / "prompts/feature_annotation_multiclass_object_v1_0.txt"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class StartLimiter:
    def __init__(self, interval_seconds: float):
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_start = time.monotonic() + self.interval


def bounded_image_uri(path: str, max_side: int = 1600) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_one(config, prompt, row, schema, raw_dir: Path, limiter: StartLimiter) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt + "\nimage_id: " + row["image_id"] + "\ntarget_instance_id: target_1"},
        {"type": "image_url", "image_url": {"url": image_uri(row["image_path"])}},
        {"type": "image_url", "image_url": {"url": image_uri(row["marked_image_path"])}},
        {"type": "image_url", "image_url": {"url": image_uri(row["crop_image_path"])}},
    ]}]
    max_attempts = int(config.get("max_retries", 5)) + 1
    last_error = "online_retries_exhausted"
    compressed_transport = False
    for attempt in range(1, max_attempts + 1):
        body = {
            **config.get("request_parameters", {}),
            "model": config["model"],
            "max_tokens": max(3000, int(config.get("request_parameters", {}).get("max_tokens", 0))),
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        limiter.wait()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        status, text, elapsed, headers = post_json(
            config["url"], body, {"Authorization": "Bearer " + config["api_key"]},
            int(config.get("timeout_seconds", 600)),
        )
        safe = text.replace(config["api_key"], "[REDACTED]")
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{row['image_id']}_{stamp}_attempt{attempt}.json").write_text(safe, encoding="utf-8")
        if status == 413 and not compressed_transport and attempt < max_attempts:
            for item, field in zip(messages[0]["content"][1:], ("image_path", "marked_image_path", "crop_image_path")):
                item["image_url"]["url"] = bounded_image_uri(row[field])
            compressed_transport = True
            last_error = "online_http_413_original_transport"
            continue
        if status in {429, 500, 502, 503, 504}:
            last_error = f"online_http_{status}"
            if attempt < max_attempts:
                try:
                    wait = float(headers.get("Retry-After", config.get("retry_wait_seconds", 30)))
                except (TypeError, ValueError):
                    wait = float(config.get("retry_wait_seconds", 30))
                time.sleep(max(5.0, wait))
                continue
        if status != 200:
            raise ValueError(f"online_http_{status}")
        try:
            outer = json.loads(safe)
            content = outer["choices"][0]["message"]["content"]
            candidate = unique_json_object(content)
            definitions = {
                item["feature_id"]
                for item in schema["universal_features"] + schema.get("category_specific_features", [])
            }
            feature_values = candidate.get("features")
            if isinstance(feature_values, dict) and set(feature_values) - definitions == {"category"}:
                if feature_values["category"] != row["target_class"]:
                    raise ValueError("extra_category_target_mismatch")
                del feature_values["category"]
                evidence = candidate.get("evidence")
                if isinstance(evidence, dict):
                    evidence.pop("category", None)
            evidence = candidate.get("evidence")
            if isinstance(evidence, dict):
                for key, value in list(evidence.items()):
                    if isinstance(value, list):
                        evidence[key] = ", ".join(map(str, value))
            result = validate_feature(candidate, row, schema)
            usage = outer.get("usage", {}) if isinstance(outer, dict) else {}
            result["online_elapsed_seconds"] = elapsed
            result["online_attempts"] = attempt
            result["online_prompt_tokens"] = usage.get("prompt_tokens", "")
            result["online_completion_tokens"] = usage.get("completion_tokens", "")
            result["online_compressed_transport"] = compressed_transport
            return result
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
            last_error = f"semantic:{error}"
            if attempt < max_attempts:
                messages.extend([
                    {"role": "assistant", "content": content if "content" in locals() else ""},
                    {"role": "user", "content": (
                        "上一条输出未通过严格校验，错误为：" + str(error)
                        + "。请依据冻结schema重新输出完整JSON，不要添加任何说明。"
                    )},
                ])
                continue
    raise ValueError(last_error)


def flatten(row: dict[str, str], result: dict | None, error: str = "") -> dict:
    output = {
        "image_id": row["image_id"], "target_instance_id": "target_1",
        "target_class": row["target_class"], "status": "ok" if result else "failed",
        "error": error, "protocol_version": result.get("schema_version", "") if result else "",
        "online_elapsed_seconds": result.get("online_elapsed_seconds", "") if result else "",
        "online_attempts": result.get("online_attempts", "") if result else "",
        "online_prompt_tokens": result.get("online_prompt_tokens", "") if result else "",
        "online_completion_tokens": result.get("online_completion_tokens", "") if result else "",
        "online_compressed_transport": result.get("online_compressed_transport", "") if result else "",
    }
    if result:
        for key, value in result["features"].items():
            output[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-interval", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_csv(args.input)
    if args.limit:
        rows = rows[:args.limit]
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    compact = {
        "schema_version": schema["schema_version"], "category": schema["category"],
        "features": [
            {key: item.get(key, True) for key in ("feature_id", "value_type", "possible_values", "allow_unknown")}
            for item in schema["universal_features"] + schema.get("category_specific_features", [])
        ],
    }
    prompt = args.prompt.read_text(encoding="utf-8") + "\n特征字典如下：\n" + json.dumps(compact, ensure_ascii=False)
    config = load_online_config(args.config)
    config.update({"max_retries": 5, "retry_wait_seconds": 30})
    args.output.mkdir(parents=True, exist_ok=True)
    parsed_dir, raw_dir = args.output / "parsed", args.output / "raw"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "object-state-online-sensenova-v1", "created_at": now(),
        "model": config["model"], "schema_version": schema["schema_version"],
        "planned": len(rows), "workers": args.workers, "start_interval_seconds": args.start_interval,
        "inputs": ["original", "A_marked", "enlarged_crop"], "target_label_visible": True,
        "deterministic_normalization": [
            "drop redundant features.category only when it equals target_class",
            "join list-valued evidence into a string; feature values are unchanged",
            "on HTTP 413 only, resend all three views as max-side-1600 JPEG",
        ],
    }
    atomic_json(args.output / "protocol.json", protocol)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    pending = []
    for row in rows:
        path = parsed_dir / f"{row['image_id']}.json"
        if args.resume and path.is_file():
            try:
                results[row["image_id"]] = validate_feature(json.loads(path.read_text(encoding="utf-8")), row, schema)
                continue
            except Exception:
                pass
        pending.append(row)
    limiter = StartLimiter(args.start_interval)
    started = now()
    atomic_json(args.output / "pipeline_status.json", {
        "status": "running", "started_at": started, "heartbeat": now(), "planned": len(rows),
        "completed": len(results), "failed": 0, "model": config["model"],
    })
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_rows = {
            executor.submit(request_one, config, prompt, row, schema, raw_dir, limiter): row
            for row in pending
        }
        for completed, future in enumerate(as_completed(future_rows), 1):
            row = future_rows[future]
            key = row["image_id"]
            try:
                result = future.result()
                atomic_json(parsed_dir / f"{key}.json", result)
                results[key] = result
                print(f"[online] {len(results) + len(errors)}/{len(rows)} {key} ok", flush=True)
            except Exception as error:
                errors[key] = str(error)
                print(f"[online] {len(results) + len(errors)}/{len(rows)} {key} ERROR {error}", flush=True)
            atomic_json(args.output / "pipeline_status.json", {
                "status": "running", "started_at": started, "heartbeat": now(), "planned": len(rows),
                "completed": len(results), "failed": len(errors), "model": config["model"],
            })
    output_rows = [flatten(row, results.get(row["image_id"]), errors.get(row["image_id"], "")) for row in rows]
    write_csv(args.output / "results.csv", output_rows)
    atomic_json(args.output / "errors.json", [{"image_id": key, "error": value} for key, value in errors.items()])
    report = {
        **protocol, "successful": len(results), "failed": len(errors),
        "mean_elapsed_seconds": (
            sum(float(value.get("online_elapsed_seconds", 0)) for value in results.values()) / len(results)
            if results else None
        ),
        "prompt_tokens": sum(int(value.get("online_prompt_tokens") or 0) for value in results.values()),
        "completion_tokens": sum(int(value.get("online_completion_tokens") or 0) for value in results.values()),
    }
    atomic_json(args.output / "report.json", report)
    atomic_json(args.output / "pipeline_status.json", {
        "status": "complete" if not errors else "needs_retry", "started_at": started,
        "heartbeat": now(), "planned": len(rows), "completed": len(results), "failed": len(errors),
        "model": config["model"],
    })
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
