#!/usr/bin/env python3
"""Freeze the original-pool subset used for online-VLM agreement analysis."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "artifacts/multiclass_object_pool_v1_20260928/inputs/object_state_feature_manifest.csv"
ORDERS = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928/tables/model_independent_orders.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/online_vlm_reannotation_20260929/input_manifest.csv"
CLASSES = ("dog", "cat", "horse", "sheep", "person")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = {row["image_id"]: row for row in read_csv(INPUTS)}
    orders = [row for row in read_csv(ORDERS) if row["method"] == "coverage"]
    selected = [
        row for row in orders
        if row["class_name"] in CLASSES and int(row["rank"]) <= args.per_class
    ]
    selected.sort(key=lambda row: (CLASSES.index(row["class_name"]), int(row["rank"])))
    output_rows = [inputs[row["sample_id"]] for row in selected]
    counts = Counter(row["target_class"] for row in output_rows)
    if counts != Counter({name: args.per_class for name in CLASSES}):
        raise ValueError(f"unbalanced subset: {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    protocol = {
        "selection": "first K candidates per class from frozen local-VLM Coverage order",
        "per_class": args.per_class,
        "total": len(output_rows),
        "classes": list(CLASSES),
        "purpose": "challenging-state online/local VLM agreement and annotation sensitivity",
    }
    (args.output.parent / "subset_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == "__main__":
    main()
