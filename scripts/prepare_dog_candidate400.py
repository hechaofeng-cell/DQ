#!/usr/bin/env python3
"""Build a reproducible 400-image COCO train2017 dog gap candidate pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def download(item, images):
    image_id = item["image_id"]
    path = images / f"{image_id}.jpg"
    url = f"http://images.cocodataset.org/train2017/{image_id}.jpg"
    if path.exists():
        try:
            with Image.open(path) as image: image.verify()
            return image_id, path, url
        except Exception: path.unlink()
    temporary = path.with_suffix(".jpg.part")
    error = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "QE-dog-candidate-builder/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                stream.write(response.read())
            with Image.open(temporary) as image: image.verify()
            temporary.replace(path)
            return image_id, path, url
        except Exception as exc:
            error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(attempt * 2)
    raise RuntimeError(f"download_failed:{image_id}:{error}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-zip", type=Path, default=Path("data/coco2017/coco2017labels.zip"))
    ap.add_argument("--exclude", type=Path, default=Path("data/coco2017/dog500/manifest.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/coco2017/dog_candidate400_20260912"))
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, labels = args.output / "images", args.output / "labels"
    images.mkdir(exist_ok=True); labels.mkdir(exist_ok=True)
    exclude = {r["image_id"] for r in csv.DictReader(args.exclude.open(encoding="utf-8-sig", newline=""))}
    candidates = []
    with zipfile.ZipFile(args.labels_zip) as archive:
        for name in archive.namelist():
            if not name.startswith("coco/labels/train2017/") or not name.endswith(".txt"): continue
            image_id = Path(name).stem
            if image_id in exclude: continue
            content = archive.read(name)
            annotations = [line.split() for line in content.decode("utf-8").splitlines() if line.strip()]
            dogs = [x for x in annotations if x[0] == "16"]
            if not dogs: continue
            target = max(dogs, key=lambda x: float(x[3]) * float(x[4]))
            cx, cy, width, height = map(float, target[1:])
            area = width * height
            flags = {
                "tiny_target": area < .01,
                "small_target": .01 <= area < .03,
                "multiple_dogs": len(dogs) >= 2,
                "boundary_target": cx - width / 2 <= .02 or cy - height / 2 <= .02 or cx + width / 2 >= .98 or cy + height / 2 >= .98,
                "person_cooccurrence": any(x[0] == "0" for x in annotations),
            }
            candidates.append({"image_id": image_id, "label_bytes": content, "dog_count": len(dogs), "target_bbox_yolo": " ".join(target[1:]), "target_area_ratio": area, **flags})
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected, used = [], set()
    strata = [("tiny_target", "tiny_target"), ("small_target", "small_target"), ("multiple_dogs", "multiple_dogs"), ("boundary_target", "boundary_target"), ("person_cooccurrence", "person_cooccurrence")]
    for stratum, flag in strata:
        pool = [x for x in candidates if x[flag] and x["image_id"] not in used]
        if len(pool) < 80: raise SystemExit(f"insufficient_{stratum}:{len(pool)}")
        for item in pool[:80]: item["selection_stratum"] = stratum; selected.append(item); used.add(item["image_id"])
    selected.sort(key=lambda x: x["image_id"])
    for item in selected: (labels / f"{item['image_id']}.txt").write_bytes(item["label_bytes"])
    completed = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, item, images): item for item in selected}
        for index, future in enumerate(as_completed(futures), 1):
            image_id, path, url = future.result(); completed[image_id] = (path, url)
            if index % 25 == 0: print(f"downloaded {index}/400", flush=True)
    rows = []
    for item in selected:
        path, url = completed[item["image_id"]]; label = labels / f"{item['image_id']}.txt"
        rows.append({"candidate_id": f"dogcand_{len(rows)+1:04d}", "image_id": item["image_id"], "image_path": str(path), "label_path": str(label), "image_url": url, "selection_stratum": item["selection_stratum"], "tiny_target": item["tiny_target"], "small_target": item["small_target"], "multiple_dogs": item["multiple_dogs"], "boundary_target": item["boundary_target"], "person_cooccurrence": item["person_cooccurrence"], "dog_count": item["dog_count"], "target_bbox_yolo": item["target_bbox_yolo"], "target_area_ratio": item["target_area_ratio"], "image_sha256": sha256(path), "label_sha256": sha256(label), "source_split": "COCO train2017", "candidate_status": "unreviewed"})
    manifest = args.output / "candidate_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"status": "complete", "count": len(rows), "seed": args.seed, "excluded_existing_ids": len(exclude), "unique_image_ids": len({r["image_id"] for r in rows}), "unique_image_hashes": len({r["image_sha256"] for r in rows}), "selection_strata": {name: sum(r["selection_stratum"] == name for r in rows) for name, _ in strata}, "source": "COCO train2017", "purpose": "candidate_pool_not_final_validation"}
    (args.output / "candidate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
