#!/usr/bin/env python3
"""Review the fixed targeted-supplement queue without changing source results."""
import argparse
import csv
import io
import json
import mimetypes
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.io import atomic_json, read_csv, read_json

EXPERIMENT = ROOT / "artifacts/dog_targeted_supplement_v1_20260912"


class Store:
    def __init__(self, output=EXPERIMENT / "human_review"):
        self.output = Path(output)
        self.schema = read_json(ROOT / "configs/dog_feature_schema_v2.json")
        self.definitions = {d["feature_id"]: d for d in self.schema["universal_features"]}
        self.manifest = {r["image_id"]: r for r in read_csv(EXPERIMENT / "combined_manifest_600.csv")}
        self.cases = []
        for row in read_csv(EXPERIMENT / "review_queue.csv"):
            self.cases.append({"image_id": row["image_id"], "group": "critical" if row["review_reason"] == "contributes_missing_or_sparse_state" else "random", "gap_states": json.loads(row["gap_states"]), "quality_flag": ""})
        for row in read_csv(EXPERIMENT / "baseline_audit_flags.csv"):
            self.cases.append({"image_id": row["image_id"], "group": "baseline", "gap_states": [], "quality_flag": row["quality_flag"]})
        self.cases.sort(key=lambda c: ({"baseline": 0, "critical": 1, "random": 2}[c["group"]], c["image_id"]))
        self.case_by_id = {c["image_id"]: c for c in self.cases}
        self.features = {}
        for case in self.cases:
            directory = "dog_feature_pipeline_v2_1_upscaled_20260912" if case["group"] == "baseline" else "dog_candidate400_features_v2_1_20260912"
            self.features[case["image_id"]] = read_json(ROOT / f"artifacts/{directory}/results/parsed/features/{case['image_id']}.json")

    def records(self):
        return [read_json(self.output / "current" / f"{c['image_id']}.json") for c in self.cases if (self.output / "current" / f"{c['image_id']}.json").is_file()]

    def payload(self):
        reviews = {r["image_id"]: r for r in self.records()}
        return {"experiment_id": "dog-targeted-supplement-v1-20260912", "definitions": list(self.definitions.values()), "cases": [{**c, "feature": self.features[c["image_id"]], "review": reviews.get(c["image_id"]), "images": {k: f"/media/{k}/{c['image_id']}" for k in ("source", "marked", "crop")}} for c in self.cases]}

    def media(self, kind, image_id):
        if image_id not in self.case_by_id:
            return None
        key = {"source": "image_path", "marked": "marked_image_path", "crop": "crop_image_path"}.get(kind)
        return Path(self.manifest[image_id][key]) if key else None

    def save(self, value):
        image_id = value.get("image_id")
        if image_id not in self.case_by_id:
            raise ValueError("图片不在固定复核队列中")
        reviewer = value.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 80:
            raise ValueError("请填写审核员，最长80字符")
        for key, allowed in {"target_match": {"yes", "no", "uncertain"}, "target_kind": {"real_dog", "representation", "other", "uncertain"}, "decision": {"accept", "reject", "uncertain"}}.items():
            if value.get(key) not in allowed:
                raise ValueError("请完成目标确认和处置")
        if value["decision"] == "accept" and (value["target_match"] != "yes" or value["target_kind"] != "real_dog"):
            raise ValueError("仅A框匹配且为真实犬的图片可以保留")
        expected = {s.split("=", 1)[0] for s in self.case_by_id[image_id]["gap_states"]}
        answers = value.get("human_features")
        if not isinstance(answers, dict) or set(answers) != expected:
            raise ValueError("关键特征字段不完整")
        if any(not isinstance(v, str) or v not in self.definitions[k]["possible_values"] for k, v in answers.items()):
            raise ValueError("请确认每个关键特征，或填写unknown")
        if not isinstance(value.get("notes", ""), str) or len(value.get("notes", "")) > 5000:
            raise ValueError("备注格式错误")
        stamp = datetime.now(timezone.utc)
        saved = {"image_id": image_id, "reviewer": reviewer.strip(), "review_version": "supplement_review_v1", "reviewed_at": stamp.isoformat(), "target_match": value["target_match"], "target_kind": value["target_kind"], "decision": value["decision"], "human_features": answers, "notes": value.get("notes", ""), "group": self.case_by_id[image_id]["group"], "model_features": self.features[image_id]["features"]}
        atomic_json(self.output / "history" / image_id / f"{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json", saved)
        atomic_json(self.output / "current" / f"{image_id}.json", saved)
        return saved


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, content_type="application/json; charset=utf-8"):
            data = json.dumps(value, ensure_ascii=False).encode() if isinstance(value, (list, dict)) else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self.send(200, (ROOT / "scripts/dog_supplement_review_app.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/fonts/NotoSansSC-Regular.ttf":
                return self.send(200, (ROOT / "scripts/assets/NotoSansSC-Regular.ttf").read_bytes(), "font/ttf")
            if path == "/api/data":
                return self.send(200, store.payload())
            if path == "/api/export.json":
                return self.send(200, store.records())
            if path == "/api/export.csv":
                stream = io.StringIO()
                fields = ["image_id", "group", "reviewer", "reviewed_at", "target_match", "target_kind", "decision", "human_features", "notes"]
                writer = csv.DictWriter(stream, fields, extrasaction="ignore")
                writer.writeheader()
                for row in store.records():
                    writer.writerow({**row, "human_features": json.dumps(row["human_features"], ensure_ascii=False)})
                return self.send(200, ("\ufeff" + stream.getvalue()).encode(), "text/csv; charset=utf-8")
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "media":
                media = store.media(parts[1], unquote(parts[2]))
                if media and media.is_file():
                    return self.send(200, media.read_bytes(), mimetypes.guess_type(media.name)[0] or "application/octet-stream")
            self.send(404, {"error": "not_found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/reviews":
                return self.send(404, {"error": "not_found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 100000:
                    raise ValueError("请求大小无效")
                self.send(200, {"review": store.save(json.loads(self.rfile.read(length)))})
            except (ValueError, TypeError, KeyError) as error:
                self.send(400, {"error": str(error)})

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    store = Store()
    print(f"Review: http://localhost:{args.port}/ ({len(store.cases)} images)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(store)).serve_forever()
