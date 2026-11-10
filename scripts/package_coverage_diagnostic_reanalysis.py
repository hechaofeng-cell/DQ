#!/usr/bin/env python3
"""Package the frozen 2026-10-18 analyses without images or trained weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("artifacts/coverage_diagnostic_validation_20261018")
SOURCE = Path("artifacts/other_classes_specific_experiment_20261018")
BASE = Path("artifacts/coco_five_dimension_fine_selection_20261015")
EXTERNAL = Path("artifacts/objects365_independent_confirmation_20261005")
CLASSES = ("horse", "sheep", "person", "car")
MODELS = ("yolo26s_cls", "pe_core_b16", "siglip2_base", "mobileclip2_s0", "aimv2_large")
SEEDS = (20260916, 20260917, 20260918)
SCRIPTS = (
    "coverage_validation_common_20261018.py",
    "run_coverage_validation_20261018.py",
    "run_automatic_diagnostic_robustness_20261018.py",
    "run_category_five_dimension_pilot.py",
)
TABLES = (
    "coverage_metrics.csv", "coverage_runs.csv", "quotas.csv",
    "discovered_conditions.csv", "discovery_groups.csv",
    "annotation_repeat_pairs.csv", "annotation_repeat_summary.csv",
    "synthetic_missingness.csv", "external_condition_results.csv",
    "external_diagnostic_groups.csv", "external_diagnostic_summary.csv",
    "automatic_robustness/external_risk_scores.csv",
    "automatic_robustness/unique_state_model_checks.csv",
    "automatic_robustness/condition_checks.csv",
    "automatic_robustness/group_checks.csv",
    "automatic_robustness/summary.csv",
    "automatic_robustness/deduplicated_conditions.csv",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(destination):
    copied = []

    def copy(relative, target):
        source = ROOT / relative
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        copied.append({"source": str(relative), "member": str(target),
                       "sha256": sha(source), "bytes": source.stat().st_size})

    def input_file(relative):
        copy(relative, Path("inputs") / relative)

    for name in SCRIPTS:
        copy(Path("scripts") / name, Path("snapshot/scripts") / name)
    config = Path("configs/category_five_dimension_pilot_v1.json")
    copy(config, Path("snapshot") / config)
    copy(Path("scripts/replay_coverage_diagnostic_bundle.py"), Path("reproduce.py"))
    copy(Path("docs/实验方案与结果/latest_coverage_reanalysis_20261018.md"), Path("README.md"))
    for name in ("protocol.json", "external_manifest.csv", "repeat_manifest.csv",
                 "automatic_robustness/protocol.json"):
        input_file(STUDY / name)
    for relative in sorted((ROOT / STUDY / "dictionaries").glob("*.json")):
        input_file(relative.relative_to(ROOT))
    for relative in sorted((ROOT / STUDY / "annotations").rglob("*_parsed.json")):
        input_file(relative.relative_to(ROOT))
    for name in ("manifest.csv", "native_features.jsonl"):
        input_file(BASE / name)
    input_file(SOURCE / "risk_scores_and_tiers.csv")
    for category in CLASSES:
        for name in ("state_matrix.npz", "state_dictionary.json"):
            input_file(SOURCE / f"{category}_{name}")
    for row in rows(STUDY / "repeat_manifest.csv"):
        input_file(SOURCE / f"annotations_v2/{row['class_name']}/{row['sample_id']}_parsed.json")
    input_file(Path("artifacts/sixclass_independent_confirmation_20261004/embeddings/vit_b16_candidates.npz"))
    for model in MODELS:
        for seed in SEEDS:
            for pool in ("B", "D"):
                input_file(Path(f"artifacts/recent_five_models_20261009/runs/{model}/{seed}/{pool}_outputs.npz"))
    for row in rows(STUDY / "external_manifest.csv"):
        domain = "car" if row["class_name"] == "car" else "old"
        input_file(EXTERNAL / f"annotations/local/{domain}/parsed/{row['sample_id']}.json")
    for name in TABLES:
        copy(STUDY / name, Path("expected") / name)
    for name in ("selections.npz", *(f"external_{c}_matrix.npz" for c in CLASSES)):
        copy(STUDY / name, Path("expected") / name)
    for name in ("conditions_before_external_predictions.json", "analysis_input_freeze.json",
                 "selection_before_prediction_freeze.json", "external_confirmation_audit.json",
                 "execution_environment.json", "automatic_robustness/audit.json"):
        copy(STUDY / name, Path("provenance") / name)
    for name in ("dataset_license_inventory_20261007.md",
                 "覆盖与诊断补充实验结果_20261018.md", "全自动诊断稳健性实验结果_20261018.md"):
        copy(Path("docs") / name, Path("reports") / name)
    import numpy as np

    metadata = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "study": "coverage_diagnostic_validation_20261018",
        "copied_files": copied, "source_files_sha256": "recorded per file",
        "python": sys.version, "numpy": np.__version__,
        "historical_timestamps_recreated": False,
        "raw_images_weights_secrets_included": False,
    }
    (destination / "provenance/package.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    (destination / "requirements.txt").write_text(f"numpy=={np.__version__}\n")
    records = [{"path": str(p.relative_to(destination)), "sha256": sha(p),
                "bytes": p.stat().st_size}
               for p in sorted(destination.rglob("*")) if p.is_file()]
    (destination / "manifest.sha256.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # 2026-11-10 docs 重组：既有打包件归档于 docs/archive/。
    parser.add_argument("--output", type=Path, default=ROOT / "docs/archive/coverage_diagnostic_reanalysis_20261018.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve the previous bundle")
    with tempfile.TemporaryDirectory(prefix="qe-coverage-package-") as directory:
        temporary = Path(directory)
        records = build(temporary)
        with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(temporary.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(temporary)))
    print(json.dumps({"package": str(args.output), "sha256": sha(args.output),
                      "payload_files": len(records),
                      "uncompressed_bytes": sum(r["bytes"] for r in records),
                      "archive_bytes": args.output.stat().st_size}, indent=2))
