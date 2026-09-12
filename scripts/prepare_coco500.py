#!/usr/bin/env python3
"""Select and download 500 COCO val2017 images with non-empty bbox labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("data/coco2017/coco2017labels.zip"))
    ap.add_argument("--output", type=Path, default=Path("data/coco2017/coco500"))
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--class-id", type=int, default=16, help="COCO YOLO class id; dog=16")
    ap.add_argument("--class-name", default="dog")
    ap.add_argument("--split", choices=("train2017", "val2017"), default="train2017")
    args = ap.parse_args()

    if args.output.exists() and (args.output / "manifest.csv").exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / "images"
    labels = args.output / "labels"
    images.mkdir()
    labels.mkdir()

    with zipfile.ZipFile(args.zip) as z:
        names = [n for n in z.namelist() if n.startswith(f"coco/labels/{args.split}/") and n.endswith(".txt")]
        candidates = []
        for name in names:
            content = z.read(name)
            lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
            if any(int(line.split()[0]) == args.class_id for line in lines):
                image_id = Path(name).stem
                candidates.append((image_id, content))
    if len(candidates) < args.count:
        raise SystemExit(f"Only {len(candidates)} non-empty val labels found")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = sorted(candidates[: args.count])

    rows = []
    for i, (image_id, label_bytes) in enumerate(selected, 1):
        image_path = images / f"{image_id}.jpg"
        label_path = labels / f"{image_id}.txt"
        url = f"http://images.cocodataset.org/{args.split}/{image_id}.jpg"
        last_error = None
        for attempt in range(1, 6):
            try:
                urllib.request.urlretrieve(url, image_path)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 5:
                    raise RuntimeError(f"failed to download {url} after 5 attempts") from exc
                time.sleep(2 * attempt)
        label_path.write_bytes(label_bytes)
        annotation_lines = [line for line in label_bytes.decode("utf-8").splitlines() if line.strip()]
        annotation_count = sum(1 for line in annotation_lines if int(line.split()[0]) == args.class_id)
        rows.append({
            "sample_id": f"coco500_{i:04d}",
            "image_id": image_id,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "source": "COCO val2017 / Ultralytics YOLO labels",
            "image_url": url,
            "annotation_count": annotation_count,
            "image_sha256": sha256(image_path),
            "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        })
        if i % 25 == 0:
            print(f"downloaded {i}/{args.count}", flush=True)

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "selection.json").write_text(json.dumps({
        "dataset": f"COCO {args.split}",
        "count": args.count,
        "seed": args.seed,
        "class": args.class_name,
        "class_id": args.class_id,
        "selection_rule": "val2017 YOLO annotation containing at least one requested class bbox; one image per image_id",
        "bbox_required": True,
        "source_label_archive": str(args.zip),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"created {args.output} with {len(rows)} images", flush=True)


if __name__ == "__main__":
    main()
