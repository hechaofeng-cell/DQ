"""Immutable run artifacts and content-addressed input provenance."""

import csv
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_output(path, protected=()):
    path = Path(path).resolve()
    for source in protected:
        source = Path(source).resolve()
        if path == source or path in source.parents or source in path.parents:
            raise ValueError(f"Output must be separate from input: {source}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot(paths):
    return {str(Path(p).resolve()): digest(p) for p in sorted(set(map(Path, paths))) if Path(p).is_file()}


def verify_snapshot(hashes):
    changed = [p for p, sha in hashes.items() if not Path(p).is_file() or digest(p) != sha]
    if changed:
        raise ValueError(f"Input changed or disappeared: {changed}")


def seal_run(output):
    write_json(output / "artifact_hashes.json", {
        p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()
    })


def verify_run(output):
    output = Path(output)
    for name, sha in read_json(output / "artifact_hashes.json").items():
        if Path(name).name != name or digest(output / name) != sha:
            raise ValueError(f"Artifact integrity failure: {name}")
