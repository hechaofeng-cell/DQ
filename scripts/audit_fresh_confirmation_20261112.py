#!/usr/bin/env python3
"""Outcome-blind lineage and supply audit. Never calls a VLM or a classifier.

Historical inputs remain read-only. Raw source annotation catalogues describe
availability, not prior experimental use; their IDs are not an exclusion list.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/fresh_confirmation_readiness_20261112"
CLASSES = ["dog", "cat", "horse", "sheep", "person", "car"]
COCO = dict(zip(CLASSES, (16, 15, 17, 18, 0, 2)))
TEXT = {".csv", ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".py", ".log"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "vendor", "node_modules"}
CATALOGUES = {
    ROOT / "artifacts/objects365_independent_confirmation_20261005/metadata/zhiyuan_objv2_train.json",
    ROOT / "artifacts/objects365_independent_confirmation_20261005/metadata/zhiyuan_objv2_val.json",
}
ID_RX = re.compile(rb"(?<!\d)(\d{12})(?!\d)")
OBJ_RX = re.compile(rb"(?:objects365_v[12]_0*|obj365-train-0*|objects365-v2-train-0*)(\d+)")
SHA_RX = re.compile(rb"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tokens(content):
    return (set(x.decode() for x in ID_RX.findall(content)),
            set(str(int(x)) for x in OBJ_RX.findall(content)),
            set(x.decode().lower() for x in SHA_RX.findall(content)))


def file_evidence(path):
    """Read text in overlapping blocks, or only string arrays from NumPy caches."""
    ids, obj, hashes = tokens(str(path.relative_to(ROOT)).encode())
    h = hashlib.sha256()
    stat = path.stat()
    issues = []
    kind = "path_only"
    if path in CATALOGUES:
        return path, ids, obj, hashes, dict(path=str(path), bytes=stat.st_size,
            sha256=sha(path), kind="raw_source_catalogue_not_usage", issues="")
    if path.suffix.lower() in TEXT:
        kind = "text"
        tail = b""
        with path.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
                a, c, d = tokens(tail + b)
                ids.update(a); obj.update(c); hashes.update(d)
                tail = b[-256:]
    elif path.suffix.lower() in {".npz", ".npy"}:
        kind = "numpy_cache"
        # Numeric arrays cannot supply textual sample identities. Read headers
        # before decompression; never unpickle historical object arrays.
        try:
            if path.suffix.lower() == ".npz":
                with zipfile.ZipFile(path) as z:
                    members = []
                    for member in z.namelist():
                        if not member.endswith(".npy"):
                            continue
                        with z.open(member) as f:
                            version = np.lib.format.read_magic(f)
                            if version == (1, 0):
                                _, _, dtype = np.lib.format.read_array_header_1_0(f)
                            elif version == (2, 0):
                                _, _, dtype = np.lib.format.read_array_header_2_0(f)
                            else:
                                raise ValueError("Unsupported NumPy header version: " + str(version))
                        if dtype.hasobject:
                            issues.append("object_array_not_unpickled:" + member)
                        elif dtype.kind in "US" or dtype.fields:
                            members.append(member[:-4])
                if members:
                    with np.load(path, allow_pickle=False) as z:
                        for member in members:
                            a, c, d = tokens("\n".join(z[member].astype(str).reshape(-1)).encode())
                            ids.update(a); obj.update(c); hashes.update(d)
            else:
                a = np.load(path, allow_pickle=False, mmap_mode="r")
                if a.dtype.kind in "US":
                    x, y, z = tokens("\n".join(a.astype(str).reshape(-1)).encode())
                    ids.update(x); obj.update(y); hashes.update(z)
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            issues.append(type(exc).__name__ + ":" + str(exc)[:160])
        h_value = sha(path)
    else:
        h_value = ""
    if kind == "text":
        h_value = h.hexdigest()
    changed = path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != stat.st_size
    if changed:
        issues.append("changed_during_scan")
    return path, ids, obj, hashes, dict(path=str(path), bytes=stat.st_size,
        sha256=h_value, kind=kind, issues=";".join(issues))


def enumerate_coco(archive, forbidden):
    counts = Counter()
    remaining = []
    inverse = {v: k for k, v in COCO.items()}
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not member.startswith("coco/labels/train2017/") or not member.endswith(".txt"):
                continue
            iid = Path(member).stem
            if iid in forbidden:
                continue
            best = {}
            for line in z.read(member).decode().splitlines():
                p = line.split()
                if len(p) != 5 or int(p[0]) not in inverse:
                    continue
                label = int(p[0]); cx, cy, w, h = map(float, p[1:])
                if not 0 < w <= 1 or not 0 < h <= 1:
                    continue
                box = [max(0., cx-w/2), max(0., cy-h/2), min(1., cx+w/2), min(1., cy+h/2)]
                if label not in best or w*h > best[label][0]:
                    best[label] = w*h, box
            for label, (area, box) in best.items():
                c = inverse[label]; counts[c] += 1
                remaining.append(dict(class_name=c, image_id=iid, bbox_area_ratio=area,
                    bbox_norm_xyxy=json.dumps(box), label_archive_member=member,
                    status="ID_screen_only_not_image_certified"))
    return {c: counts[c] for c in CLASSES}, remaining


def audit():
    if OUT.exists():
        raise FileExistsError("Use a new version for a new historical snapshot: " + str(OUT))
    OUT.mkdir(parents=True)
    dump("audit_protocol.json", dict(created_at=now(), scope="accessible project history at snapshot time",
        roots=[str(ROOT / x) for x in ("artifacts", "data", "output", "configs", "scripts", "docs", "qe_quality")],
        excluded_directories=sorted(SKIP_DIRS), raw_source_catalogues=list(map(str, sorted(CATALOGUES))),
        exclusion="All filename IDs plus 12-digit tokens in historical text/string caches; broad conservative exclusions, not a count of proven used images. Download range .part numbers are excluded.",
        object_exclusion="Objects365 train filename / sample-ID tokens; raw annotation catalogues do not constitute usage.",
        desired_pools=dict(select_per_class=400, reference_per_class=400, classes=CLASSES),
        human_review=False, model_calls=False, pretraining_independence_certified=False))
    paths = []
    for top in ("artifacts", "data", "output", "configs", "scripts", "docs", "qe_quality"):
        for d, ds, fs in os.walk(ROOT / top):
            ds[:] = sorted(x for x in ds if x not in SKIP_DIRS and (Path(d) / x) != OUT
                and not x.endswith(".parts"))
            for f in sorted(fs):
                p = Path(d) / f
                # Do not read secret-bearing local API configuration.
                if "api_config.local" in f or f.startswith(".env"):
                    continue
                if p.suffix.lower() in TEXT | {".npz", ".npy", ".jpg", ".jpeg", ".png", ".webp"}:
                    paths.append(p)
    paths.sort()
    ids, obj, hashes = set(), set(), set()
    provenance = {}; obj_provenance = {}; records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for n, (path, a, b, c, record) in enumerate(executor.map(file_evidence, paths), 1):
            for iid in a - ids:
                provenance[iid] = str(path)
            for iid in b - obj:
                obj_provenance[iid] = str(path)
            ids.update(a); obj.update(b); hashes.update(c); records.append(record)
            if n % 10000 == 0:
                print(json.dumps(dict(stage="scan", files=n, total=len(paths), coco_IDs=len(ids), objects_IDs=len(obj))), flush=True)
    write("historical_file_registry.csv", records, ["path", "bytes", "sha256", "kind", "issues"])
    write("coco_ID_exclusions.csv", [dict(image_id=x, first_evidence=provenance[x]) for x in sorted(ids)], ["image_id", "first_evidence"])
    write("objects365_train_ID_exclusions.csv", [dict(image_id=x, first_evidence=obj_provenance[x]) for x in sorted(obj, key=int)], ["image_id", "first_evidence"])
    (OUT / "historical_hash_tokens.txt").write_text("\n".join(sorted(hashes)) + "\n")
    reserve = read(ROOT / "artifacts/empirical_completion_20261108/fresh_candidate_reserve_provisional.csv")
    expanded = {r["image_id"]: r for r in read(ROOT / "artifacts/candidate_pool_real_expansion_20261108/new_manifest.csv")}
    rejected = {r["image_id"]: r for r in read(ROOT / "artifacts/candidate_pool_real_expansion_20261108/acquisition_rejections.csv")}
    overlaps = []
    for r in reserve:
        e = expanded.get(r["image_id"])
        reason = "already_development" if e else "previously_rejected" if r["image_id"] in rejected else "unresolved_history"
        overlaps.append(dict(class_name=r["class_name"], image_id=r["image_id"], reserve_sample_id=r["sample_id"],
            current_status=reason, development_sample_id=e["sample_id"] if e else "",
            rejection_reason=rejected.get(r["image_id"], {}).get("reason", "")))
    write("old_reserve_lineage.csv", overlaps, list(overlaps[0]))
    archive = ROOT / "data/coco2017/coco2017labels.zip"
    supply, remaining = enumerate_coco(archive, ids)
    write("remaining_coco_ID_candidates.csv", remaining, ["class_name", "image_id", "bbox_area_ratio", "bbox_norm_xyxy", "label_archive_member", "status"])
    config_path = ROOT / "configs/recent_five_models_20261009.json"
    config = json.loads(config_path.read_text())
    training = ROOT / config["train_manifest"]
    train_rows = read(training)
    bound_paths = [Path(r[k]) for r in train_rows for k in ("image_path", "crop_path") if r.get(k)]
    outside = sorted({str(p) for p in bound_paths if not p.resolve().is_relative_to(ROOT)})
    missing = sorted({str(p) for p in bound_paths if not p.is_file()})
    checkpoints = []
    for model in config["models"]:
        for seed in config["seeds"]:
            p = ROOT / "artifacts/recent_five_models_20261009/runs" / model / str(seed) / "best.pt"
            checkpoints.append(dict(model=model, seed=seed, path=str(p), exists=p.is_file(), sha256=sha(p) if p.is_file() else ""))
    write("checkpoint_lineage.csv", checkpoints, ["model", "seed", "path", "exists", "sha256"])
    dump("training_lineage.json", dict(config=str(config_path), config_sha256=sha(config_path),
        train_manifest=str(training), train_manifest_sha256=sha(training), targets=len(train_rows),
        bound_image_paths=len(set(map(str,bound_paths))), missing_paths=missing, outside_project_paths=outside,
        checkpoints=len(checkpoints), all_checkpoints_exist=all(r["exists"] for r in checkpoints),
        boundary="Project fine-tuning inputs only; unavailable foundation-model pretraining manifests and unrelated locations are not certified."))
    issue_rows = [r for r in records if r["issues"]]
    write("scan_issues.csv", issue_rows, ["path", "bytes", "sha256", "kind", "issues"])
    result = dict(completed_at=now(), status="not_ready_for_new_inference", files_inventoried=len(paths),
        text_files_scanned=sum(r["kind"] == "text" for r in records), numpy_caches_scanned=sum(r["kind"] == "numpy_cache" for r in records),
        conservative_COCO_ID_exclusions=len(ids), objects365_train_ID_exclusions=len(obj), scan_issue_files=len(issue_rows),
        old_reserve_targets=len(reserve), old_reserve_status=dict(Counter(r["current_status"] for r in overlaps)),
        development_overlap_by_class=dict(Counter(r["class_name"] for r in overlaps if r["current_status"] == "already_development")),
        remaining_COCO_train_ID_candidates=supply, required_per_class=800,
        classes_below_required=[c for c in CLASSES if supply[c] < 800], source_archive_sha256=sha(archive),
        remaining_candidates_image_bound=0, remaining_candidates_near_duplicate_checked=0,
        new_VLM_calls=0, new_classifier_calls=0, independent_confirmation_completed=False,
        lineage_scope="Accessible project records; conservative ID screening is not exact-hash/near-duplicate certification or foundation pretraining exclusion.",
        next_action="Audit alternative source supply using existing local Objects365 official metadata; do not silently shrink the six-class 400+400 design.")
    dump("audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
