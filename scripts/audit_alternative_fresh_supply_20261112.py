#!/usr/bin/env python3
"""Count unused Objects365 source IDs from local metadata, without image/model calls.

This is a feasibility screen, not an image-level independence certificate.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import ijson

from audit_fresh_confirmation_20261112 import OUT, ROOT, CLASSES, read, dump, write, sha, now

SOURCE = ROOT / "artifacts/objects365_independent_confirmation_20261005"
SEED = 2026111201


def key(*parts):
    return hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).hexdigest()


def package(filename):
    m = re.search(r"(?:images/)?(v[12]/patch\d+)/", filename)
    if not m:
        raise ValueError("Unrecognized official Objects365 filename")
    return m[1]


def main():
    if (OUT / "alternative_supply.json").exists():
        raise FileExistsError("Alternative source snapshot already exists")
    dump("alternative_supply_protocol.json", dict(created_at=now(), seed=SEED,
        purpose="Outcome-blind feasibility of another source after excluding project history",
        source="local official Objects365-v2 train metadata", selection="largest valid noncrowd nonignored box, area>=0.001; historical train-ID exclusion before counting",
        package_order="SHA256(seed|package|name), prefix until each class has >=1000 ID-screened candidates",
        class_allocation="scarce classes first, globally unique original-image IDs, hash order within class",
        target_pools="400 selection + 400 reference + 100 technical reserves per class",
        counts_are_pre_hash_and_near_duplicate_upper_bounds=True,
        remote_requests=False, image_read=False, classifier_outputs_read=False))
    frozen = json.loads((SOURCE / "metadata/source_audit.json").read_text())
    path = SOURCE / "metadata/zhiyuan_objv2_train.json"
    digest = sha(path)
    if digest != frozen["annotation_json_sha256"]:
        raise ValueError("Official metadata differs from historical source audit")
    forbidden = {int(r["image_id"]) for r in read(OUT / "objects365_train_ID_exclusions.csv")}
    inverse = {int(v): k for k, v in frozen["mapping"].items()}
    images = {}
    with path.open("rb") as f:
        for r in ijson.items(f, "images.item", use_float=True):
            if r["id"] not in forbidden:
                images[r["id"]] = (r["file_name"], r["width"], r["height"])
    best = {}; scanned = 0; invalid = Counter()
    with path.open("rb") as f:
        for ann in ijson.items(f, "annotations.item", use_float=True):
            scanned += 1
            if scanned % 2000000 == 0:
                print(json.dumps(dict(stage="alternative_source", annotations=scanned, candidates=len(best))), flush=True)
            c = inverse.get(ann["category_id"])
            if c is None or ann["image_id"] not in images:
                continue
            if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                invalid["crowd_or_ignore"] += 1; continue
            name, iw, ih = images[ann["image_id"]]
            x, y, w, h = map(float, ann["bbox"])
            if min(w, h, iw, ih) <= 0 or min(x, y) < 0 or x+w > iw+1e-3 or y+h > ih+1e-3:
                invalid["invalid_box"] += 1; continue
            if w*h/(iw*ih) < .001:
                invalid["area_below_0.001"] += 1; continue
            k = c, ann["image_id"]
            previous = best.get(k)
            if previous is None or (w*h, -ann["id"]) > (previous[0], -previous[1]):
                best[k] = (w*h, ann["id"], (x, y, w, h), int(ann["category_id"]))
    full = Counter(); by_package = defaultdict(Counter)
    for (c, iid) in best:
        full[c] += 1
        by_package[package(images[iid][0])][c] += 1
    packages = sorted(by_package, key=lambda x: key("package", x))
    cumulative = Counter(); prefix = []
    for p in packages:
        cumulative.update(by_package[p]); prefix.append(p)
        if all(cumulative[c] >= 1000 for c in CLASSES):
            break
    grouped = defaultdict(list)
    for (c, iid), b in best.items():
        filename, iw, ih = images[iid]
        if package(filename) not in prefix:
            continue
        grouped[c].append((key(c, iid), iid, filename, iw, ih, b))
    order = sorted(CLASSES, key=lambda c: (len(grouped[c]), c))
    used = set(); plan = []; availability = {}
    for c in order:
        options = sorted(r for r in grouped[c] if r[1] not in used)
        availability[c] = len(options)
        for n, (rank, iid, filename, iw, ih, b) in enumerate(options[:900]):
            area, aid, box, cid = b
            # Hash order determines roles before image content or annotation results.
            role = "selection" if n < 400 else "reference" if n < 800 else "technical_reserve"
            used.add(iid)
            x,y,w,h = box
            plan.append(dict(class_name=c, source_image_id=iid, source_file_name=filename,
                package=package(filename), role=role, rank=n+1, order_sha256=rank,
                width=iw, height=ih, category_id=cid, annotation_id=aid,
                bbox_norm_xyxy=json.dumps([x/iw,y/ih,(x+w)/iw,(y+h)/ih]), bbox_area_ratio=area/(iw*ih),
                status="provisional_metadata_only_not_certified"))
    write("alternative_provisional_plan.csv", plan, list(plan[0]))
    package_rows = []
    planned = Counter(r["package"] for r in plan)
    cached = {r["name"]: r for r in frozen["packages"]}
    for p in prefix:
        archive = SOURCE / "data/packages" / (p.replace("/", "_") + ".tar.gz")
        record = cached.get(p, {})
        package_rows.append(dict(package=p, targets_in_plan=planned[p],
            local_archive=str(archive), archive_exists=archive.is_file(),
            recorded_bytes=record.get("bytes", ""),
            archive_download_audit_exists=archive.with_suffix(".gz.download.json").is_file(),
            qualified_ID_counts=json.dumps(dict(by_package[p]))))
    write("alternative_packages.csv", package_rows, list(package_rows[0]))
    result = dict(completed_at=now(), status="metadata_feasible_image_gate_pending" if all(availability[c]>=900 for c in CLASSES) else "insufficient_metadata_supply",
        source_metadata_sha256=digest, annotations_scanned=scanned, historical_train_ID_exclusions=len(forbidden),
        unused_qualified_ID_upper_bounds={c:full[c] for c in CLASSES},
        selected_package_prefix=prefix, globally_disjoint_availability=availability,
        provisional_targets=len(plan), provisional_role_counts=dict(Counter(r["role"] for r in plan)),
        required_packages=len(prefix), missing_local_archives=[r["package"] for r in package_rows if not r["archive_exists"]],
        unavailable_package_sizes_are_not_zero=True, image_SHA_and_near_duplicate_audit_completed=False,
        no_new_image_download_or_model_calls=True,
        interpretation="A feasible source-ID plan only. Source switch is an external-source confirmation track, not a new COCO same-source replication. No fresh efficacy result exists.")
    dump("alternative_supply.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
