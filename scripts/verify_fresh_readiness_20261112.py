#!/usr/bin/env python3
"""Independent raw-record checks for the readiness package (not efficacy)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import ijson

from audit_fresh_confirmation_20261112 import ROOT, OUT, CLASSES, COCO, read, dump, sha, now


def main():
    a = json.loads((OUT / "audit_summary.json").read_text())
    reserve = read(ROOT / "artifacts/empirical_completion_20261108/fresh_candidate_reserve_provisional.csv")
    expanded = read(ROOT / "artifacts/candidate_pool_real_expansion_20261108/new_manifest.csv")
    rejected = read(ROOT / "artifacts/candidate_pool_real_expansion_20261108/acquisition_rejections.csv")
    before = {r["image_id"] for r in reserve}
    used = {r["image_id"] for r in expanded}
    bad = {r["image_id"] for r in rejected}
    assert len(before) == 2400 and len(before & used) == 2396
    assert before - used <= bad and len(before - used) == 4
    assert a["old_reserve_status"] == {"already_development": 2396, "previously_rejected": 4}
    forbidden = {r["image_id"] for r in read(OUT / "coco_ID_exclusions.csv")}
    counters = Counter()
    # Independent presence counting, rather than the auditor's largest-box rows.
    with zipfile.ZipFile(ROOT / "data/coco2017/coco2017labels.zip") as z:
        for member in z.namelist():
            if not member.startswith("coco/labels/train2017/") or not member.endswith(".txt"):
                continue
            if Path(member).stem in forbidden:
                continue
            present = set()
            for line in z.read(member).decode().splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                if 0 < float(values[3]) <= 1 and 0 < float(values[4]) <= 1:
                    present.add(int(values[0]))
            counters.update(c for c, category in COCO.items() if category in present)
    assert {c: counters[c] for c in CLASSES} == a["remaining_COCO_train_ID_candidates"]
    plan = read(OUT / "alternative_provisional_plan.csv")
    history = {r["image_id"] for r in read(OUT / "objects365_train_ID_exclusions.csv")}
    assert len(plan) == len({r["source_image_id"] for r in plan}) == 5400
    assert not {r["source_image_id"] for r in plan} & history
    for c in CLASSES:
        rows = sorted((r for r in plan if r["class_name"] == c), key=lambda r: int(r["rank"]))
        assert len(rows) == 900 and [int(r["rank"]) for r in rows] == list(range(1, 901))
        assert Counter(r["role"] for r in rows) == {"selection":400,"reference":400,"technical_reserve":100}
        keys = [hashlib.sha256(f"2026111201|{c}|{r['source_image_id']}".encode()).hexdigest() for r in rows]
        assert keys == sorted(keys) == [r["order_sha256"] for r in rows]
        for r in rows:
            n = int(r["rank"])
            assert r["role"] == ("selection" if n <= 400 else "reference" if n <= 800 else "technical_reserve")
    # Check two targets in every class/role directly against official raw boxes.
    smoke = []
    for c in CLASSES:
        for role in ("selection", "reference", "technical_reserve"):
            smoke += [r for r in plan if r["class_name"] == c and r["role"] == role][:2]
    wanted = {int(r["annotation_id"]):r for r in smoke}
    found = set()
    source = ROOT / "artifacts/objects365_independent_confirmation_20261005/metadata/zhiyuan_objv2_train.json"
    with source.open("rb") as f:
        for ann in ijson.items(f, "annotations.item", use_float=True):
            r = wanted.get(ann["id"])
            if r is None:
                continue
            assert ann["image_id"] == int(r["source_image_id"])
            assert ann["category_id"] == int(r["category_id"])
            assert not ann.get("iscrowd", 0) and not ann.get("ignore", 0)
            iw, ih = int(r["width"]), int(r["height"])
            x,y,w,h = ann["bbox"]
            expected = [x/iw,y/ih,(x+w)/iw,(y+h)/ih]
            assert max(abs(a-b) for a,b in zip(expected,json.loads(r["bbox_norm_xyxy"]))) < 1e-12
            assert abs(w*h/(iw*ih)-float(r["bbox_area_ratio"])) < 1e-12
            found.add(ann["id"])
    assert found == set(wanted)
    checkpoints = read(OUT / "checkpoint_lineage.csv")
    assert len(checkpoints) == len({(r["model"],r["seed"]) for r in checkpoints}) == 15
    assert all(Path(r["path"]).is_file() and len(r["sha256"]) == 64 for r in checkpoints)
    checks = dict(completed_at=now(), status="pass", old_reserve_records_checked=2400,
        COCO_supply_independently_recounted=True, all_provisional_ID_and_role_bindings_checked=5400,
        raw_official_annotation_smoke_checked=len(found), bound_classifier_checkpoints=15,
        new_classifier_or_VLM_calls=0,
        boundary="Only readiness and source binding; original image pixels/hash/near-duplicate gates still pending.")
    dump("verification.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
