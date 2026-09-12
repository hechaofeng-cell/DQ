"""Local browser UI for adjudicating DTA class disagreements."""

import json
import mimetypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .io import atomic_csv, atomic_json, read_csv, read_json
from .reports import REVIEW_FIELDS
from .schema import CLASS_NAMES, OCCLUSIONS, OUTLINES, SUBJECT_ROLES


QUEUE_NAME = "class_review_queue.json"


class ReviewStore:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.manifest = {row["image_id"]: row for row in read_csv(self.output / "manifest.csv")}
        self.queue_path = self.output / QUEUE_NAME
        self.review_path = self.output / "human_review.csv"
        self.queue = self._load_or_create_queue()

    def _result(self, backend, image_id):
        path = self.output / "parsed" / backend / f"{image_id}.json"
        return read_json(path)["result"] if path.exists() else None

    def _load_or_create_queue(self):
        if self.queue_path.exists():
            queue = read_json(self.queue_path)
            if not isinstance(queue, list) or any(image_id not in self.manifest for image_id in queue):
                raise ValueError("invalid_class_review_queue")
            return queue
        queue = []
        for image_id in sorted(self.manifest):
            local, online = self._result("local", image_id), self._result("online", image_id)
            if local and online and local["class_id"] != online["class_id"]:
                queue.append(image_id)
        atomic_json(self.queue_path, queue)
        return queue

    def reviews(self):
        return {row["image_id"]: row for row in read_csv(self.review_path)}

    def payload(self):
        reviews = self.reviews()
        cases = []
        for image_id in self.queue:
            row = self.manifest[image_id]
            review = reviews.get(image_id, {})
            cases.append({
                "image_id": image_id,
                "source_class_name": row["source_class_name"],
                "image_url": f"/media/source/{image_id}",
                "overlay_url": f"/media/overlay/{image_id}",
                "local": self._result("local", image_id),
                "online": self._result("online", image_id),
                "review": {key: review.get(key, "") for key in REVIEW_FIELDS},
            })
        completed = sum(bool(case["review"].get("human_class_id")) for case in cases)
        return {
            "classes": CLASS_NAMES,
            "subject_roles": sorted(SUBJECT_ROLES),
            "outlines": sorted(OUTLINES),
            "occlusions": sorted(OCCLUSIONS),
            "completed": completed,
            "total": len(cases),
            "cases": cases,
        }

    def media_path(self, kind, image_id):
        if image_id not in self.manifest:
            return None
        if kind == "source":
            path = Path(self.manifest[image_id]["path"])
        elif kind == "overlay":
            path = self.output / "mask_overlays" / f"{image_id}.jpg"
        else:
            return None
        return path if path.is_file() else None

    def save(self, image_id, values):
        if image_id not in self.queue:
            raise ValueError("image_not_in_class_review_queue")
        required = {
            "human_class_id", "human_subject_role", "human_outline_visibility",
            "human_occlusion_level", "sam_target_match", "sam_mask_acceptable", "reviewer",
        }
        if any(not str(values.get(key, "")).strip() for key in required):
            raise ValueError("incomplete_review")
        class_id = str(values["human_class_id"])
        if class_id not in {str(value) for value in CLASS_NAMES}:
            raise ValueError("invalid_human_class_id")
        if values["human_subject_role"] not in SUBJECT_ROLES:
            raise ValueError("invalid_human_subject_role")
        if values["human_outline_visibility"] not in OUTLINES:
            raise ValueError("invalid_human_outline_visibility")
        if values["human_occlusion_level"] not in OCCLUSIONS:
            raise ValueError("invalid_human_occlusion_level")
        for key in ("sam_target_match", "sam_mask_acceptable"):
            if values[key] not in {"yes", "no", "unknown"}:
                raise ValueError(f"invalid_{key}")

        rows = read_csv(self.review_path)
        by_id = {row["image_id"]: row for row in rows}
        if image_id not in by_id:
            manifest = self.manifest[image_id]
            by_id[image_id] = {
                "image_id": image_id, "sha256": manifest["sha256"],
                "selection_reason": "class_disagreement",
                **{field: "" for field in REVIEW_FIELDS[3:]},
            }
            rows.append(by_id[image_id])
        row = by_id[image_id]
        for key in required | {"notes"}:
            row[key] = str(values.get(key, "")).strip()
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_csv(self.review_path, rows, REVIEW_FIELDS)
        return {key: row.get(key, "") for key in REVIEW_FIELDS}


def handler_for(store):
    html_path = Path(__file__).with_name("review_app.html")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/data":
                self._send(200, store.payload())
                return
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "media":
                media = store.media_path(parts[1], unquote(parts[2]))
                if media:
                    self._send(200, media.read_bytes(), mimetypes.guess_type(media.name)[0] or "application/octet-stream")
                    return
            self._send(404, {"error": "not_found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/reviews":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid_request_size")
                value = json.loads(self.rfile.read(length))
                review = store.save(str(value.get("image_id", "")), value.get("review", {}))
                self._send(200, {"ok": True, "review": review})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, pattern, *args):
            print(f"[review] {self.address_string()} {pattern % args}", flush=True)

    return Handler


def serve_review(output, host="127.0.0.1", port=8765):
    store = ReviewStore(output)
    server = ThreadingHTTPServer((host, port), handler_for(store))
    print(f"DTA review: http://{host}:{server.server_port} ({len(store.queue)} images)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

