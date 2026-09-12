#!/usr/bin/env python3
"""Local human-review server for the dog v2.1 feature experiment."""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import random
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.io import atomic_json

SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Store:
    def __init__(self, results, manifest, schema, output, queue_size=120, seed=20260912):
        self.results_dir = Path(results).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = {r["image_id"]: r for r in read_csv(self.results_dir / "results.csv")}
        self.manifest = {r["image_id"]: r for r in read_csv(manifest)}
        self.schema = json.loads(Path(schema).read_text(encoding="utf-8"))
        self.definitions = self.schema["universal_features"] + self.schema.get("category_specific_features", [])
        self.queue_path = self.output / "review_queue.json"
        self.queue = self._queue(queue_size, seed)

    def _queue(self, size, seed):
        if self.queue_path.exists():
            value = json.loads(self.queue_path.read_text(encoding="utf-8"))
            if value.get("protocol") != "dog-v2.1-upscaled-crop":
                raise ValueError("review_queue_protocol_mismatch")
            return value["items"]
        rng = random.Random(seed)
        selected, reasons = [], {}

        def add(ids, reason, limit=None):
            ids = list(ids)
            if limit is not None:
                rng.shuffle(ids); ids = ids[:limit]
            for image_id in ids:
                is_new = image_id not in selected
                if is_new and len(selected) < size:
                    selected.append(image_id)
                elif is_new:
                    continue
                if is_new or not reason.startswith("random"):
                    reasons.setdefault(image_id, []).append(reason)

        for split, count in (("calibration", 10), ("development", 20), ("validation", 20)):
            add((i for i, r in self.rows.items() if r["split"] == split), f"random_{split}", count)
        add((i for i, r in self.rows.items() if r["classification_correct"] == "False"), "classification_error")
        add((i for i, r in self.rows.items() if r["feature_status"] != "ok"), "feature_failure")
        add((i for i, r in sorted(self.rows.items(), key=lambda x: float(x[1]["target_area_ratio"]))[:25]), "small_target")
        add((i for i, r in self.rows.items() if int(r.get("feature_unknown_count") or 0) >= 4), "high_unknown")
        add((i for i, r in self.rows.items() if r.get("occlusion_level") == "heavy" or r.get("outline_visibility") == "unclear"), "visibility_risk", 20)
        add(self.rows, "random_fill")
        items = [{"image_id": i, "reasons": sorted(set(reasons.get(i, ["random_fill"])))} for i in selected]
        atomic_json(self.queue_path, {"protocol": "dog-v2.1-upscaled-crop", "seed": seed, "count": len(items), "items": items})
        return items

    def feature(self, image_id):
        path = self.results_dir / "parsed/features" / f"{image_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def review_path(self, reviewer, version, image_id):
        if not all(SAFE.fullmatch(v or "") for v in (reviewer, version, image_id)):
            raise ValueError("invalid_reviewer_version_or_image_id")
        return self.output / "reviews" / version / reviewer / f"{image_id}.json"

    def payload(self, reviewer="", version="v1"):
        cases = []
        completed = 0
        for item in self.queue:
            image_id = item["image_id"]
            row, manifest = self.rows[image_id], self.manifest[image_id]
            review = None
            if reviewer and SAFE.fullmatch(reviewer) and SAFE.fullmatch(version):
                path = self.review_path(reviewer, version, image_id)
                if path.exists():
                    review = json.loads(path.read_text(encoding="utf-8")); completed += 1
            cases.append({
                "image_id": image_id, "reasons": item["reasons"], "split": row["split"],
                "predicted_class": row["predicted_class"], "classification_correct": row["classification_correct"],
                "target_area_ratio": row["target_area_ratio"], "feature_status": row["feature_status"],
                "feature": self.feature(image_id), "review": review,
                "images": {"source": f"/media/source/{image_id}", "marked": f"/media/marked/{image_id}", "crop": f"/media/crop/{image_id}"},
            })
        return {"schema_version": self.schema["schema_version"], "definitions": self.definitions,
                "total": len(cases), "completed": completed, "cases": cases}

    def media(self, kind, image_id):
        row = self.manifest.get(image_id)
        if not row:
            return None
        key = {"source": "image_path", "marked": "marked_image_path", "crop": "crop_image_path"}.get(kind)
        path = Path(row[key]).resolve() if key else None
        return path if path and path.is_file() else None

    def save(self, value):
        reviewer = str(value.get("reviewer", "")).strip()
        version = str(value.get("review_version", "v1")).strip()
        image_id = str(value.get("image_id", "")).strip()
        if image_id not in {x["image_id"] for x in self.queue}:
            raise ValueError("image_not_in_review_queue")
        features = value.get("human_features")
        expected = {x["feature_id"]: set(x["possible_values"]) for x in self.definitions}
        if not isinstance(features, dict) or set(features) != set(expected):
            raise ValueError("incomplete_human_features")
        if any(features[k] not in expected[k] for k in expected):
            raise ValueError("invalid_human_feature_value")
        if value.get("target_match") not in {"yes", "no", "uncertain"}:
            raise ValueError("invalid_target_match")
        if value.get("human_class") not in {"dog", "not_dog", "uncertain"}:
            raise ValueError("invalid_human_class")
        if str(value.get("srl")) not in {"0", "1", "2", "3", "4", "5"}:
            raise ValueError("invalid_srl")
        if value.get("is_recognizable") not in {"yes", "no", "uncertain"}:
            raise ValueError("invalid_recognizability")
        saved = {**value, "reviewer": reviewer, "review_version": version,
                 "schema_version": self.schema["schema_version"], "reviewed_at": datetime.now(timezone.utc).isoformat()}
        path = self.review_path(reviewer, version, image_id)
        if path.exists():
            history = path.parent / "history" / image_id
            history.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            atomic_json(history / f"{stamp}.json", json.loads(path.read_text(encoding="utf-8")))
        atomic_json(path, saved)
        return saved


def handler(store):
    html_path = ROOT / "scripts/dog_human_review_app.html"
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)): body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str): body = body.encode()
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            url = urlsplit(self.path); path = url.path
            if path == "/": return self.send(200, html_path.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/data":
                q = parse_qs(url.query); return self.send(200, store.payload(q.get("reviewer", [""])[0], q.get("version", ["v1"])[0]))
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "media":
                media = store.media(parts[1], unquote(parts[2]))
                if media: return self.send(200, media.read_bytes(), mimetypes.guess_type(media.name)[0] or "application/octet-stream")
            self.send(404, {"error": "not_found"})
        def do_POST(self):
            if urlsplit(self.path).path != "/api/reviews": return self.send(404, {"error": "not_found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000: raise ValueError("invalid_request_size")
                self.send(200, {"ok": True, "review": store.save(json.loads(self.rfile.read(length)))})
            except (ValueError, TypeError, json.JSONDecodeError) as exc: self.send(400, {"error": str(exc)})
        def log_message(self, fmt, *args): print("[dog-review] " + fmt % args, flush=True)
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "artifacts/dog_feature_pipeline_v2_1_upscaled_20260912/results")
    ap.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv")
    ap.add_argument("--schema", type=Path, default=ROOT / "configs/dog_feature_schema_v2.json")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/dog_feature_pipeline_v2_1_upscaled_20260912/human_review")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(); store = Store(args.results, args.manifest, args.schema, args.output)
    server = ThreadingHTTPServer((args.host, args.port), handler(store))
    print(f"Dog human review: http://{args.host}:{args.port} ({len(store.queue)} images)", flush=True)
    server.serve_forever()


if __name__ == "__main__": main()
