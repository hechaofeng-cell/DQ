"""Resumable orchestration for the 100-image DTA reliability gate."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .clients import Pacing, load_online_config, ollama_annotation, online_annotation, save_parsed
from .geometry import run_sam
from .io import atomic_csv, atomic_json, digest, fingerprint, read_json
from .reports import report_all


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path):
    path = Path(path).resolve()
    value = read_json(path)
    root = path.parent.parent
    for key in ("input_dir", "classes_file", "prompt", "online_config"):
        candidate = Path(value[key]).expanduser()
        value[key] = str((root / candidate).resolve()) if not candidate.is_absolute() else str(candidate.resolve())
    checkpoint = Path(value["sam"]["checkpoint"]).expanduser()
    value["sam"]["checkpoint"] = str((root / checkpoint).resolve()) if not checkpoint.is_absolute() else str(checkpoint.resolve())
    value["config_path"] = str(path)
    return value


def prepare_manifest(config):
    input_dir = Path(config["input_dir"])
    classes = {row["class_id"]: row["class_name"] for row in read_json(config["classes_file"])}
    images = sorted(path for path in input_dir.glob("*/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    rows = [{"image_id": path.stem, "path": str(path.resolve()), "sha256": digest(path),
             "source_class_id": path.parent.name, "source_class_name": classes.get(path.parent.name, "")}
            for path in images]
    expected, per_class = config["expected_count"], config["expected_per_class"]
    counts = {}
    for row in rows:
        counts[row["source_class_id"]] = counts.get(row["source_class_id"], 0) + 1
    if len(rows) != expected or len({r["image_id"] for r in rows}) != expected:
        raise ValueError(f"expected_{expected}_unique_images")
    if not counts or any(value != per_class for value in counts.values()):
        raise ValueError("unexpected_per_class_counts")
    if any(not row["source_class_name"] for row in rows):
        raise ValueError("unknown_source_class")
    return rows


def public_online_config(path):
    value = read_json(path)
    return {key: value.get(key) for key in ("url", "model", "timeout_seconds", "request_parameters")}


def build_protocol(config, rows, prompt):
    implementation = {
        path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    value = {
        "protocol_version": config["protocol_version"],
        "manifest": [{k: row[k] for k in ("image_id", "sha256", "source_class_id")} for row in rows],
        "prompt_sha256": fingerprint(prompt),
        "ollama": config["ollama"],
        "sam": {**config["sam"], "checkpoint_sha256": digest(config["sam"]["checkpoint"])},
        "online": public_online_config(config["online_config"]),
        "online_runtime": config.get("online_runtime", {}),
        "review_seed": config["review_seed"],
        "implementation_sha256": implementation,
    }
    return value, fingerprint(value)


class Status:
    def __init__(self, output, protocol_hash, total):
        self.path = output / "pipeline_status.json"
        self.value = read_json(self.path) if self.path.exists() else {
            "status": "running", "stage": "prepare", "started_at": now(), "pid": os.getpid(),
            "protocol_hash": protocol_hash, "planned": total,
        }
        self.value.update({"status": "running", "pid": os.getpid(), "heartbeat": now()})
        self.save()

    def update(self, **values):
        self.value.update(values, heartbeat=now())
        self.save()

    def save(self):
        atomic_json(self.path, self.value)


def load_results(directory, protocol_hash):
    results = {}
    if not directory.exists():
        return results
    for path in sorted(directory.glob("*.json")):
        value = read_json(path)
        if value.get("protocol_hash") != protocol_hash:
            raise ValueError(f"resume_protocol_mismatch:{path}")
        results[path.stem] = value["result"]
    return results


def run_annotations(backend, config, prompt, rows, output, protocol_hash, status, errors):
    directory = output / "parsed" / backend
    raw_dir = output / "raw" / backend
    directory.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(directory, protocol_hash)
    pacing = Pacing(config.get("interval_seconds", 10)) if backend == "online" else None
    total = len(rows)
    stage_started = time.monotonic()
    for index, row in enumerate(rows, 1):
        image_id = row["image_id"]
        elapsed = time.monotonic() - stage_started
        attempted = max(1, index - 1)
        remaining_seconds = elapsed / attempted * (total - index + 1)
        status.update(stage=f"{backend}_vlm", current_image=image_id, completed=len(results),
                      failed=sum(e["stage"] == backend for e in errors), remaining=total - len(results),
                      estimated_remaining_seconds=round(remaining_seconds, 1),
                      **{f"{backend}_json_success_rate": len(results) / total})
        if image_id in results:
            continue
        try:
            if backend == "local":
                result = ollama_annotation(config, prompt, row, raw_dir / f"{image_id}.json")
            else:
                result = online_annotation(config, prompt, row, raw_dir, pacing)
            save_parsed(directory / f"{image_id}.json", result, protocol_hash)
            results[image_id] = result
            status.update(completed=len(results), remaining=total - len(results),
                          **{f"{backend}_json_success_rate": len(results) / total})
            print(f"[{backend}] {index}/{total} {image_id} ok {result['elapsed_seconds']:.2f}s", flush=True)
        except Exception as exc:
            errors.append({"stage": backend, "image_id": image_id, "error": str(exc)})
            print(f"[{backend}] {index}/{total} {image_id} ERROR {exc}", flush=True)
    return results


def unload_ollama(config):
    try:
        from .clients import post_json
        post_json(config["url"], {"model": config["model"], "keep_alive": 0}, {}, 30)
    except Exception as exc:
        print(f"[local] model unload warning: {exc}", flush=True)


def artifact_hashes(output):
    excluded = {"artifact_hashes.json", "pipeline_status.json", "background.log"}
    values = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded and ".tmp" not in path.name:
            values[str(path.relative_to(output))] = digest(path)
    atomic_json(output / "artifact_hashes.json", values)


def run_phase1(config_path, output_path, resume=False, smoke_only=False):
    config = load_config(config_path)
    output = Path(output_path).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("raw/local", "raw/online", "parsed/local", "parsed/online", "parsed/sam", "masks", "mask_overlays"):
        (output / name).mkdir(parents=True, exist_ok=True)
    rows = prepare_manifest(config)
    prompt = Path(config["prompt"]).read_text(encoding="utf-8-sig")
    protocol, protocol_hash = build_protocol(config, rows, prompt)
    protocol["protocol_hash"] = protocol_hash
    if (output / "protocol.json").exists():
        if read_json(output / "protocol.json").get("protocol_hash") != protocol_hash:
            raise ValueError("output_protocol_mismatch")
    else:
        atomic_json(output / "protocol.json", protocol)
        atomic_csv(output / "manifest.csv", rows,
                   ["image_id", "path", "sha256", "source_class_id", "source_class_name"])
    status = Status(output, protocol_hash, len(rows))
    errors = []
    selected_rows = rows
    if smoke_only:
        wanted = config["smoke_ids"]
        selected_rows = [row for row in rows if row["image_id"] in wanted]
        if [row["image_id"] for row in selected_rows] != sorted(wanted):
            raise ValueError("smoke_ids_not_found")
    try:
        local = run_annotations("local", config["ollama"], prompt, selected_rows, output,
                                protocol_hash, status, errors)
        unload_ollama(config["ollama"])

        def sam_progress(index, total, image_id, error):
            if error:
                errors.append({"stage": "sam", "image_id": image_id, "error": error})
            status.update(stage="sam", current_image=image_id, completed=index - (1 if error else 0),
                          failed=sum(e["stage"] == "sam" for e in errors), remaining=total - index,
                          sam_success_rate=(index - sum(e["stage"] == "sam" for e in errors)) / total)
            print(f"[sam] {index}/{total} {image_id} {'ERROR ' + error if error else 'ok'}", flush=True)

        sam = run_sam(config["sam"], selected_rows, local, output, protocol_hash, sam_progress)
        if smoke_only:
            passed = len(local) == len(selected_rows) and len(sam) == len(selected_rows) and not errors
            smoke_report = {"status": "smoke_complete" if passed else "smoke_failed", "planned": len(selected_rows),
                            "local": len(local), "sam": len(sam), "errors": errors}
            atomic_json(output / "smoke_report.json", smoke_report)
            status.update(status=smoke_report["status"], stage="smoke_review", completed=len(sam),
                          failed=len(errors), remaining=0, current_image=None)
            artifact_hashes(output)
            if not passed:
                raise ValueError("smoke_gate_failed")
            return smoke_report

        online_config = load_online_config(config["online_config"])
        online_config.update(config.get("online_runtime", {}))
        online = run_annotations("online", online_config, prompt, rows, output,
                                 protocol_hash, status, errors)
        status.update(stage="report", current_image=None, completed=0, remaining=0)
        report = report_all(output, rows, local, online, sam, errors, config, protocol_hash)
        final_status = "complete" if report["decision"] == "GO" else (
            "needs_review" if report["decision"] == "NEEDS_REVIEW" else "failed")
        status.update(status=final_status, stage="report", decision=report["decision"],
                      completed=report["counts"]["features"], failed=report["counts"]["errors"], remaining=0)
        artifact_hashes(output)
        return report
    except BaseException as exc:
        status.update(status="failed", last_error=str(exc), current_image=None)
        artifact_hashes(output)
        raise
