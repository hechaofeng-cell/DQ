#!/usr/bin/env python3
"""Replay frozen coverage and diagnostic analyses from a self-contained bundle."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(bundle):
    records = json.loads((bundle / "manifest.sha256.json").read_text())
    for record in records:
        path = (bundle / record["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError(f"Invalid bundle member: {record['path']}")
        if sha(path) != record["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {record['path']}")
    return len(records)


def compare_csv(actual, expected):
    with actual.open(encoding="utf-8-sig", newline="") as handle:
        a = list(csv.reader(handle))
    with expected.open(encoding="utf-8-sig", newline="") as handle:
        b = list(csv.reader(handle))
    if not a or not b or a[0] != b[0] or len(a) != len(b):
        raise ValueError(f"CSV shape or header mismatch: {expected.name}")
    for row_index, (left, right) in enumerate(zip(a[1:], b[1:]), 2):
        if len(left) != len(right):
            raise ValueError(f"CSV row width mismatch: {expected.name}:{row_index}")
        for column, (x, y) in enumerate(zip(left, right)):
            if x == y:
                continue
            try:
                xx, yy = float(x), float(y)
                equal = (math.isnan(xx) and math.isnan(yy)) or math.isclose(
                    xx, yy, rel_tol=1e-10, abs_tol=1e-10
                )
            except ValueError:
                equal = False
            if not equal:
                raise ValueError(
                    f"CSV value mismatch: {expected.name}:{row_index} "
                    f"column {a[0][column]}: {x!r} != {y!r}"
                )
    return {"file": str(expected.name), "rows": len(a) - 1, "columns": len(a[0])}


def compare_npz(actual, expected):
    import numpy as np

    with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError(f"NPZ key mismatch: {expected.name}")
        for key in b.files:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"NPZ array mismatch: {expected.name}:{key}")
        return {"file": expected.name, "arrays": len(b.files)}


def replay(bundle, output):
    started = time.monotonic()
    bundle, output = bundle.resolve(), output.resolve()
    count = verify_manifest(bundle)
    # A new output directory keeps both frozen inputs and previous replays intact.
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    shutil.copytree(bundle / "inputs", workspace)
    shutil.copytree(bundle / "snapshot/scripts", workspace / "scripts")
    shutil.copytree(bundle / "snapshot/configs", workspace / "configs")
    logs = output / "logs"
    logs.mkdir()
    main_script = "scripts/run_coverage_validation_20261018.py"
    steps = [[main_script, phase] for phase in (
        "check", "select", "discover", "reliability", "sensitivity", "confirm"
    )]
    steps.append(["scripts/run_automatic_diagnostic_robustness_20261018.py"])
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = "2"
    for index, step in enumerate(steps, 1):
        name = step[-1] if len(step) > 1 else "automatic_robustness"
        print(f"[{index}/{len(steps)}] {name}", flush=True)
        log = logs / f"{index:02d}_{name}.log"
        with log.open("w") as handle:
            result = subprocess.run(
                [sys.executable, *step], cwd=workspace, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Replay failed at {name}; see {log}")
    generated = workspace / "artifacts/coverage_diagnostic_validation_20261018"
    csv_results, array_results = [], []
    for expected in sorted((bundle / "expected").rglob("*.csv")):
        relative = expected.relative_to(bundle / "expected")
        record = compare_csv(generated / relative, expected)
        record["file"] = str(relative)
        csv_results.append(record)
    for expected in sorted((bundle / "expected").glob("*.npz")):
        array_results.append(compare_npz(generated / expected.name, expected))
    for name in ("implementation_checks.json", "selection_audit.json",
                 "external_confirmation_audit.json", "automatic_robustness/audit.json"):
        if json.loads((generated / name).read_text())["status"] != "pass":
            raise ValueError(f"Analysis audit failed: {name}")
    if verify_manifest(bundle) != count:
        raise ValueError("Bundle manifest changed during replay")
    import numpy as np

    report = {
        "status": "pass", "manifest_files_verified": count,
        "csv_tables": csv_results, "npz_files": array_results,
        "python": sys.version, "numpy": np.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "scope": "Frozen-label, embedding and prediction reanalysis only",
        "training_inference_vlm_calls": 0,
        "historical_inputs_modified": False,
        "human_semantic_accuracy_validated": False,
    }
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"PASS: {len(csv_results)} tables and {len(array_results)} array files match.", flush=True)
    print(f"Verification: {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for regenerated results and verification")
    args = parser.parse_args()
    replay(args.bundle, args.output)
