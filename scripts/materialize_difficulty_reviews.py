"""Bind explicitly recorded visual decisions to prepared byte hashes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notes = json.loads(args.notes.read_text())
    notes_hash = hashlib.sha256(args.notes.read_bytes()).hexdigest()
    with (args.prepared / "samples.csv").open(encoding="utf-8-sig") as stream:
        samples = {r["original_id"]: r for r in csv.DictReader(stream)}
    with (args.prepared / "review_queue.csv").open(encoding="utf-8-sig") as stream:
        queued = {r["original_id"] for r in csv.DictReader(stream)}
    decisions = notes["decisions"]
    if len({r[0] for r in decisions}) != len(decisions) or {r[0] for r in decisions} != queued:
        raise ValueError("Visual decisions must cover the selected queue exactly once")
    fields = ["original_id", "sha256", "review_status", "reference_class_id", "reviewer", "reviewer_kind", "reviewed_at", "evidence"]
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_id, status, evidence in decisions:
            row = samples[image_id]
            writer.writerow({"original_id": image_id, "sha256": row["sha256"], "review_status": status,
                "reference_class_id": row["inherited_class_id"], "reviewer": notes["reviewer"],
                "reviewer_kind": notes["reviewer_kind"], "reviewed_at": notes["reviewed_at"],
                "evidence": evidence + f" Notes SHA-256: {notes_hash}"})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
