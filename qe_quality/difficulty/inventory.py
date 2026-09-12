"""Read historical GPU files and original bytes without rerunning inference."""

import gzip
import hashlib
import io
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from . import VERSION
from .io import (digest, fingerprint, new_output, read_csv, read_json, seal_run,
                 snapshot, verify_snapshot, write_csv, write_json)


def issue(image_id, code, detail, model=""):
    return {"original_id": image_id, "code": code, "model": model, "detail": detail}


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("prediction_dir", "source_audit_dir", "archive"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    scoring, validation = config["scoring_models"], config["validation_models"]
    if (not scoring or not validation or len(set(scoring + validation)) != len(scoring + validation)
            or any(not m.replace("_", "").isalnum() for m in scoring + validation)):
        raise ValueError("Scoring and validation panels must be nonempty, unique and disjoint")
    edges = config["diagnostic_band_edges"]
    if edges[0] != 0 or edges[-1] != 1 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("Diagnostic bands must strictly partition [0, 1]")
    policy = config["calibration"]
    if policy["method"] != "scalar_temperature_nll":
        raise ValueError("Unsupported calibration method")
    if any(policy[k] < 1 for k in ("min_per_part", "min_per_class", "min_errors_per_model")):
        raise ValueError("Calibration evidence guards must be positive")
    lo, hi = policy["temperature_bounds"]
    if not 0 < lo < 1 < hi or not math.isfinite(hi):
        raise ValueError("Temperature bounds must be finite and contain 1")
    if config["random_review_per_class"] < 1:
        raise ValueError("Random review must cover each class")
    return config


class Groups:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def read_predictions(directory, models, bases, class_order, manifest, issues):
    by_member = {r["member"]: r for r in bases}
    manifest_index = {r["member"]: i for i, r in enumerate(manifest)}
    predictions = {r["image_id"]: {} for r in bases}
    for model in models:
        path = directory / f"{model}.jsonl.gz"
        if not path.is_file():
            for base in bases:
                issues.append(issue(base["image_id"], "missing_prediction_file", str(path), model))
            continue
        seen = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Cannot align malformed JSON: {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Prediction record is not an object: {path}:{line_number}")
                member = record.get("member")
                if member not in by_member:
                    continue
                base = by_member[member]
                image_id = base["image_id"]
                seen[member] += 1
                errors = []
                logits = record.get("logits_10", record.get("logits"))
                if record.get("error"):
                    errors.append("inference_error")
                if (not isinstance(logits, list) or len(logits) != len(class_order)
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in logits)):
                    errors.append("invalid_logits")
                else:
                    predicted = class_order[max(range(len(logits)), key=logits.__getitem__)]
                    if record.get("prediction") != predicted:
                        errors.append("argmax_mismatch")
                    if record.get("correct") is not (predicted == base["class_id"]):
                        errors.append("historical_correctness_mismatch")
                if record.get("index") != manifest_index[member]:
                    errors.append("manifest_index_mismatch")
                if seen[member] > 1:
                    errors.append("duplicate_prediction")
                for code in errors:
                    issues.append(issue(image_id, code, f"{path.name}:{line_number}", model))
                previous = predictions[image_id].get(model)
                predictions[image_id][model] = {
                    "prediction": record.get("prediction"), "logits": None if "invalid_logits" in errors else logits,
                    "valid": not errors and not (previous and not previous["valid"]),
                    "member": member, "raw_line": line_number,
                }
        for member, base in by_member.items():
            if not seen[member]:
                issues.append(issue(base["image_id"], "missing_prediction", member, model))
    return predictions


def prepare(config_path, output):
    config = load_config(config_path)
    source = Path(config["prediction_dir"])
    audit = Path(config["source_audit_dir"])
    historical = read_json(source / "protocol.json")
    archive = Path(config.get("archive") or historical["archive"]).resolve()
    config["archive"] = str(archive)
    classes = read_json(source / "classes.json")
    class_order = [c["class_id"] for c in classes]
    if len(set(class_order)) != len(class_order) or class_order != historical["class_order"]:
        raise ValueError("Class order is ambiguous or differs from historical logits")
    config["class_order"] = class_order
    config["classes"] = classes
    config["implementation_version"] = VERSION
    models = config["scoring_models"] + config["validation_models"]
    model_metadata = {}
    for model in models:
        meta = read_json(source / f"{model}_run.json")
        if meta.get("model") != model or not meta.get("weights") or not meta.get("preprocess"):
            raise ValueError(f"Missing model identity or preprocessing: {model}")
        model_metadata[model] = meta
    config["model_metadata"] = model_metadata
    manifest = read_csv(source / "manifest.csv")
    if len({r["member"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate manifest member")
    bases = [r for r in manifest if r["group"] == config["baseline_group"]]
    if len(bases) != config["expected_count"] or len({r["image_id"] for r in bases}) != len(bases):
        raise ValueError("Baseline count or ID uniqueness check failed")
    if any(r["role"] not in {"original", "original_copy"} for r in bases):
        raise ValueError("Transform/auxiliary image cannot be a baseline original")
    # Preserve every historical file, including older pilot results outside the GPU run.
    history_paths = [p for p in source.parent.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    inputs = snapshot(history_paths + [Path(config_path), archive])
    issues = []
    groups = Groups()
    hashes, first_hash, id_hashes = {}, {}, defaultdict(set)
    originals_by_id = defaultdict(list)
    from PIL import Image
    with zipfile.ZipFile(archive) as images:
        names = images.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members make byte provenance ambiguous")
        if archive.stat().st_size != historical["archive_bytes"]:
            raise ValueError("Archive size differs from historical prediction input")
        for row in manifest:
            if row["role"] not in {"original", "original_copy"}:
                continue
            member, image_id = row["member"], row["image_id"]
            try:
                content = images.read(member)
            except (KeyError, zipfile.BadZipFile) as exc:
                issues.append(issue(image_id, "missing_or_corrupt_original", str(exc)))
                continue
            sha = hashlib.sha256(content).hexdigest()
            hashes[member] = sha
            id_hashes[image_id].add(sha)
            groups.join(image_id, first_hash.setdefault(sha, image_id))
            if row["role"] == "original":
                originals_by_id[image_id].append(row)
        for base in bases:
            try:
                with Image.open(io.BytesIO(images.read(base["member"]))) as image:
                    image.verify()
            except Exception as exc:
                issues.append(issue(base["image_id"], "image_decode_error", str(exc)))
    for image_id, variants in id_hashes.items():
        if len(variants) > 1:
            issues.append(issue(image_id, "same_id_different_original_bytes", str(len(variants))))
    old_audit = {r["image_id"]: r for r in read_csv(audit / "old_pilot_audit.csv")}
    old_manifest_path = source.parent / "manifest.csv"
    old_hashes = {r["image_id"]: r["sha256"] for r in read_csv(old_manifest_path)} if old_manifest_path.exists() else {}
    official = {r["image_id"]: r for r in read_csv(audit / "validation_candidates.csv")}
    predictions = read_predictions(source, models, bases, class_order, manifest, issues)
    rows = []
    for base in bases:
        image_id = base["image_id"]
        sha = hashes.get(base["member"], "")
        previous = old_audit.get(image_id, {})
        verified = official.get(image_id, {})
        risk = "unknown"
        evidence = "No matching byte-verified membership evidence in existing audit"
        if (sha and sha == old_hashes.get(image_id)
                and previous.get("official_training_bbox_id_match", "").lower() == "true"):
            risk = "known_training_overlap"
            evidence = "source_audit/old_pilot_audit.csv + old manifest SHA-256 + current bytes"
        elif (sha and sha == verified.get("sha256") and base["class_id"] == verified.get("true_class_id")):
            risk = "official_validation_verified_not_proof_of_unseen"
            evidence = "source_audit/validation_candidates.csv matched ID, bytes and label"
        originals = originals_by_id[image_id]
        if not originals:
            issues.append(issue(image_id, "canonical_original_not_found", base["member"]))
        if any(r["class_id"] != base["class_id"] for r in originals):
            issues.append(issue(image_id, "original_label_conflict", base["member"]))
        rows.append({"original_id": image_id, "source_group_id": groups.root(image_id),
                     "archive": str(archive), "path": base["member"],
                     "canonical_paths": json.dumps([r["member"] for r in originals]),
                     "sha256": sha, "source": "uploaded_archive; inherited dataset label",
                     "inherited_class_id": base["class_id"], "reference_class_id": base["class_id"],
                     "review_status": "pending", "reviewer": "", "reviewer_kind": "",
                     "source_status": risk, "source_evidence": evidence,
                     "validation_model_source_status": "unknown",
                     "split": "unassigned", "score_version": config["protocol_version"]})
    root_labels = defaultdict(set)
    for row in rows:
        root_labels[row["source_group_id"]].add(row["inherited_class_id"])
    for row in rows:
        if len(root_labels[row["source_group_id"]]) > 1:
            issues.append(issue(row["original_id"], "duplicate_label_conflict", row["source_group_id"]))
    group_predictions = defaultdict(list)
    for row in rows:
        group_predictions[row["source_group_id"]].append(row)
    for group_rows in group_predictions.values():
        for model in models:
            values = {predictions[r["original_id"]][model]["prediction"] for r in group_rows
                      if predictions[r["original_id"]].get(model, {}).get("valid")}
            if len(values) > 1:
                for row in group_rows:
                    issues.append(issue(row["original_id"], "same_source_prediction_conflict",
                                        row["source_group_id"], model))
                    if model in predictions[row["original_id"]]:
                        predictions[row["original_id"]][model]["valid"] = False
    relevant_roots = {r["source_group_id"] for r in rows}
    baseline_ids = {r["original_id"] for r in rows}
    baseline_members = {b["member"] for b in bases}
    lineage = []
    for row in manifest:
        root = groups.root(row["image_id"])
        if root in relevant_roots:
            lineage.append({"member": row["member"], "image_id": row["image_id"],
                            "original_id": row["image_id"] if row["image_id"] in baseline_ids else root,
                            "source_group_id": root, "role": row["role"],
                            "sha256": hashes.get(row["member"], ""),
                            "pairing_evidence": "shared_image_id_or_exact_original_hash",
                            "included_in_scoring": row["member"] in baseline_members})
    rng = random.Random(config["review_seed"])
    random_ids = set()
    for cls in class_order:
        candidates = sorted(r["original_id"] for r in rows if r["inherited_class_id"] == cls)
        random_ids.update(rng.sample(candidates, min(len(candidates), config["random_review_per_class"])))
    queue = []
    for row in rows:
        model_rows = predictions[row["original_id"]]
        valid = [r["prediction"] for r in model_rows.values() if r["valid"]]
        reasons = []
        if row["original_id"] in random_ids:
            reasons.append("stratified_random")
        if any(p != row["reference_class_id"] for p in valid):
            reasons.append("any_model_error")
        if len(set(valid)) > 1:
            reasons.append("model_disagreement")
        if len(valid) != len(models):
            reasons.append("missing_or_invalid_prediction")
        if any(i["original_id"] == row["original_id"] for i in issues):
            reasons.append("data_issue")
        if reasons:
            queue.append({"original_id": row["original_id"], "sha256": row["sha256"],
                          "path": row["path"], "inherited_class_id": row["inherited_class_id"],
                          "review_reasons": ";".join(reasons)})
    issues.append(issue("", "historical_input_binding_limit", "Historical GPU records lack per-image hashes; current archive size matches, current bytes are hashed, original inference byte identity cannot be retrospectively proven."))
    issues.append(issue("", "near_duplicate_audit_pending", "Identity and exact bytes are grouped; visual near duplicates require further audit before independent validation."))
    config["protocol_fingerprint"] = fingerprint(config)
    output = new_output(output, [source.parent, archive, Path(config_path)])
    write_json(output / "config.json", config)
    write_json(output / "predictions.json", predictions)
    write_csv(output / "samples.csv", rows)
    write_csv(output / "lineage.csv", lineage)
    write_csv(output / "review_queue.csv", queue,
              ["original_id", "sha256", "path", "inherited_class_id", "review_reasons"])
    write_csv(output / "reviews_template.csv", [
        {"original_id": r["original_id"], "sha256": r["sha256"], "review_status": "pending",
         "reference_class_id": r["inherited_class_id"], "reviewer": "", "reviewer_kind": "",
         "reviewed_at": "", "evidence": ""} for r in rows])
    write_csv(output / "splits_template.csv", [
        {"original_id": r["original_id"], "source_group_id": r["source_group_id"],
         "split": "unassigned"} for r in rows])
    write_csv(output / "issues.csv", issues, ["original_id", "code", "model", "detail"])
    write_json(output / "inventory.json", {
        "baseline_count": len(rows), "class_counts": dict(Counter(r["reference_class_id"] for r in rows)),
        "unique_source_groups": len(relevant_roots), "lineage_roles": dict(Counter(r["role"] for r in lineage)),
        "source_counts": dict(Counter(r["source_status"] for r in rows)),
        "review_queue_count": len(queue), "random_review_count": len(random_ids),
        "valid_predictions": {m: sum(v.get(m, {}).get("valid", False) for v in predictions.values()) for m in models},
        "inherited_label_errors": {m: sum(v.get(m, {}).get("valid", False) and v[m]["prediction"] != r["reference_class_id"]
                                          for r in rows for v in [predictions[r["original_id"]]]) for m in models},
        "split_decision": "Unassigned until reviewed coverage and errors justify a use allocation; no default ratios.",
    })
    verify_snapshot(inputs)
    write_json(output / "provenance.json", {"input_sha256": inputs, "historical_files_unchanged": True,
                                             "archive_sha256": digest(archive),
                                             "preparation_code_sha256": snapshot(Path(__file__).parent.glob("*.py"))})
    seal_run(output)
    return output
