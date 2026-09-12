"""Ollama and OpenAI-compatible multimodal clients with preserved raw responses."""

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .io import atomic_json, unique_json_object
from .schema import prompt_schema, validate_semantic


def image_uri(path):
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def post_json(url, body, headers, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    started = time.monotonic()
    parsed = urllib.parse.urlsplit(url)
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if parsed.hostname in {"localhost", "127.0.0.1", "::1"}
              else urllib.request.build_opener())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return response.status, payload, time.monotonic() - started, dict(response.headers)
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8", errors="replace")
        return error.code, payload, time.monotonic() - started, dict(error.headers)


def ollama_annotation(config, prompt, row, raw_path):
    image = base64.b64encode(Path(row["path"]).read_bytes()).decode("ascii")
    body = {
        "model": config["model"], "stream": False,
        "messages": [{"role": "user", "content": prompt + "\nimage_id: " + row["image_id"], "images": [image]}],
        "format": prompt_schema(),
        "options": {
            "num_ctx": config["num_ctx"], "temperature": config["temperature"],
            "num_predict": config["num_predict"], "seed": config.get("seed", 20260910),
        },
    }
    max_attempts = config.get("max_retries", 3) + 1
    for attempt in range(1, max_attempts + 1):
        status, text, elapsed, _ = post_json(config["url"], body, {}, config["timeout_seconds"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path = raw_path.with_name(f"{raw_path.stem}_attempt{attempt}.json")
        attempt_path.write_text(text, encoding="utf-8")
        if status in {502, 503, 504} and attempt < max_attempts:
            time.sleep(config.get("retry_wait_seconds", 15) * attempt)
            continue
        if status != 200:
            raise ValueError(f"ollama_http_{status}")
        try:
            response = json.loads(text)
            content = response["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("invalid_ollama_response") from exc
        try:
            result = validate_semantic(unique_json_object(content), row["image_id"])
        except ValueError as exc:
            if attempt >= max_attempts:
                raise
            body["messages"] = [
                body["messages"][0],
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    f"上一条 JSON 未通过严格校验：{exc}。请修正该错误，尤其确保 class_id 与 "
                    "class_name 严格对应，并且只返回完整 JSON 对象。"
                )},
            ]
            time.sleep(config.get("retry_wait_seconds", 15))
            continue
        result.update({"elapsed_seconds": elapsed, "backend": "ollama", "attempts": attempt})
        return result
    raise ValueError("ollama_retries_exhausted")


def load_online_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    key = os.environ.get("SENSENOVA_API_KEY") or value.get("api_key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("missing_online_api_key")
    return {**value, "api_key": key.strip()}


def online_annotation(config, prompt, row, raw_dir, pacing):
    body = {
        **config.get("request_parameters", {}),
        "model": config["model"],
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri(row["path"])}},
            {"type": "text", "text": prompt + "\nimage_id: " + row["image_id"]},
        ]}],
    }
    attempts = config.get("max_retries", 5) + 1
    for attempt in range(1, attempts + 1):
        pacing.wait()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        raw_path = raw_dir / f"{row['image_id']}_{stamp}_attempt{attempt}.json"
        status, text, elapsed, headers = post_json(
            config["url"], body, {"Authorization": "Bearer " + config["api_key"]},
            config.get("timeout_seconds", 600),
        )
        pacing.finished()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        safe = text.replace(config["api_key"], "[REDACTED]")
        raw_path.write_text(safe, encoding="utf-8")
        if status == 429 and attempt < attempts:
            retry = headers.get("Retry-After")
            try:
                retry = float(retry)
            except (TypeError, ValueError):
                retry = config.get("retry_wait_seconds", 120)
            time.sleep(max(retry, config.get("retry_wait_seconds", 120)))
            continue
        if status != 200:
            raise ValueError(f"online_http_{status}")
        try:
            response = json.loads(safe)
            content = response["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("invalid_online_response") from exc
        result = validate_semantic(unique_json_object(content), row["image_id"])
        result.update({"elapsed_seconds": elapsed, "backend": "online", "attempts": attempt})
        return result
    raise ValueError("online_retries_exhausted")


class Pacing:
    def __init__(self, interval):
        self.interval = interval
        self.last_finished = None

    def wait(self):
        if self.last_finished is not None:
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_finished)))

    def finished(self):
        self.last_finished = time.monotonic()


def save_parsed(path, value, protocol_hash):
    atomic_json(path, {"protocol_hash": protocol_hash, "result": value})
