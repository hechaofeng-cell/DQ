#!/usr/bin/env python3
"""Build an auditable, non-secret submission freeze package."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/submission_freeze_20260929"

CORE_DISTRIBUTIONS = (
    "torch", "torchvision", "numpy", "scipy", "scikit-learn",
    "matplotlib", "pandas", "Pillow", "python-docx", "requests",
)

EXPERIMENTS = (
    "multiclass_object_pool_v1_20260928",
    "multiclass_object_selection_benchmark_20260928",
    "cross_state_decoupled_diagnostics_20260928",
    "error_cluster_sensitivity_20260929",
    "near_duplicate_audit_20260929",
    "multiclass_failure_atlas_20260929",
    "coverage_incremental_value_20260929",
    "class_stratified_error_controls_20260929",
    "online_vlm_reannotation_20260929",
    "voc2012_confirmation_pool_20260929",
    "voc2012_confirmation_benchmark_20260929",
    "empirical_paper_20260929",
)

CRITICAL_FILES = (
    "configs/multiclass_object_state_schema_v1_0.json",
    "prompts/feature_annotation_multiclass_object_v1_0.txt",
    "docs/visual_state_coverage_empirical_study_draft_20260929.md",
    "docs/visual_state_coverage_empirical_study_revised_20260929.docx",
    "scripts/run_multiclass_object_selection_benchmark.py",
    "scripts/run_cross_state_decoupled_diagnostics.py",
    "scripts/run_error_cluster_sensitivity.py",
    "scripts/run_coverage_incremental_value.py",
    "scripts/run_class_stratified_error_controls.py",
    "scripts/analyze_online_vlm_agreement.py",
    "scripts/run_voc2012_confirmation_benchmark.py",
    "scripts/audit_multiclass_near_duplicates.py",
    "scripts/audit_strengthening_results.py",
    "scripts/build_submission_freeze.py",
    "artifacts/multiclass_object_pool_v1_20260928/protocol.json",
    "artifacts/multiclass_object_pool_v1_20260928/manifests/candidate_pool.csv",
    "artifacts/multiclass_object_pool_v1_20260928/annotations/results.csv",
    "artifacts/multiclass_object_selection_benchmark_20260928/protocol.json",
    "artifacts/multiclass_object_selection_benchmark_20260928/tables/primary_comparisons.csv",
    "artifacts/multiclass_object_selection_benchmark_20260928/tables/model_independent_orders.csv",
    "artifacts/cross_state_decoupled_diagnostics_20260928/protocol.json",
    "artifacts/cross_state_decoupled_diagnostics_20260928/tables/primary_comparisons.csv",
    "artifacts/error_cluster_sensitivity_20260929/protocol.json",
    "artifacts/error_cluster_sensitivity_20260929/tables/comparisons.csv",
    "artifacts/near_duplicate_audit_20260929/audit_summary.json",
    "artifacts/multiclass_failure_atlas_20260929/summary.json",
    "artifacts/coverage_incremental_value_20260929/protocol.json",
    "artifacts/coverage_incremental_value_20260929/tables/statistical_comparisons_corrected.csv",
    "artifacts/class_stratified_error_controls_20260929/protocol.json",
    "artifacts/class_stratified_error_controls_20260929/tables/class_equal_error_comparisons.csv",
    "artifacts/class_stratified_error_controls_20260929/tables/hybrid_class_equal_error_comparisons.csv",
    "artifacts/online_vlm_reannotation_20260929/protocol.json",
    "artifacts/online_vlm_reannotation_20260929/results.csv",
    "artifacts/online_vlm_reannotation_20260929/analysis/agreement_summary.json",
    "artifacts/online_vlm_reannotation_20260929/analysis/class_equal_error_comparisons.csv",
    "artifacts/voc2012_confirmation_pool_20260929/audit.json",
    "artifacts/voc2012_confirmation_pool_20260929/online_annotations/report.json",
    "artifacts/voc2012_confirmation_benchmark_20260929/protocol.json",
    "artifacts/voc2012_confirmation_benchmark_20260929/tables/direct_comparisons.csv",
    "artifacts/voc2012_confirmation_benchmark_20260929/tables/class_equal_error_comparisons.csv",
    "artifacts/empirical_paper_20260929/claim_evidence.md",
    "artifacts/empirical_paper_20260929/paper_brief.md",
    "artifacts/empirical_paper_20260929/research_log.md",
    "artifacts/empirical_paper_20260929/review.md",
    "artifacts/empirical_paper_20260929/strengthening_audit.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    versions = {}
    for distribution in CORE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [relative for relative in CRITICAL_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing critical freeze inputs: {missing}")

    versions = package_versions()
    environment = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }
    try:
        import torch
        environment["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as error:  # pragma: no cover - records environment failure without hiding it
        environment["torch_runtime_error"] = repr(error)
    write_json(OUT / "environment.json", environment)
    (OUT / "requirements-core.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in versions.items()),
        encoding="utf-8",
    )

    inventory = {}
    for experiment in EXPERIMENTS:
        directory = ROOT / "artifacts" / experiment
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = [path for path in directory.rglob("*") if path.is_file()]
        inventory[experiment] = {
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "protocol": str((directory / "protocol.json").relative_to(ROOT))
            if (directory / "protocol.json").is_file() else None,
        }
    write_json(OUT / "experiment_inventory.json", inventory)

    manifest = {}
    for relative in CRITICAL_FILES:
        path = ROOT / relative
        manifest[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "csv_rows": csv_rows(path),
        }
    write_json(OUT / "artifact_manifest.sha256.json", manifest)

    protocol = {
        "freeze_date": "2026-09-29",
        "status": "experiments_complete",
        "research_scope": "visual-state coverage for diagnostic image-classification test selection",
        "main_pool": {"classes": 5, "candidates": 2000, "per_class": 400},
        "models": 8,
        "training_seeds": 3,
        "budgets_per_class": [20, 40, 60, 80, 100],
        "selection_methods": ["Coverage", "Random", "DeepGini", "DSA", "k-center"],
        "confirmatory_configuration": {"source": "PASCAL VOC 2012", "candidates": 500},
        "online_reannotation": {"main_hard_subset": 200, "voc_pool": 500},
        "final_primary_results": {
            "local_class_equal_error_confusion_auc_delta": 0.0222,
            "local_class_equal_error_cluster_auc_delta": 0.0115,
            "online_merged_confusion_auc_delta": 0.0247,
            "online_merged_cluster_auc_delta": 0.0167,
            "voc_confusion_auc_delta": 0.0008,
            "voc_cluster_auc_delta": -0.0008,
        },
        "stopping_rule": "No result-contingent experiment additions after the registered online-label sensitivity analysis.",
    }
    write_json(OUT / "final_protocol.json", protocol)

    (OUT / "reproduction_commands.md").write_text("""# Reproduction Commands

Run from the repository root with the frozen project environment:

```bash
/home/hcf/project/QE/.venv/bin/python scripts/analyze_online_vlm_agreement.py
/home/hcf/project/QE/.venv/bin/python scripts/export_empirical_paper_docx.py docs/visual_state_coverage_empirical_study_draft_20260929.md docs/visual_state_coverage_empirical_study_revised_20260929.docx
/home/hcf/project/QE/.venv/bin/python scripts/audit_strengthening_results.py
/home/hcf/project/QE/.venv/bin/python scripts/build_submission_freeze.py
```

The online VLM responses are cached under the two online-annotation artifact directories, so the final statistical analysis does not require a new API call. Full model inference and source-pool reconstruction commands remain in each experiment protocol and execution log. Absolute local dataset paths must be replaced with the public download roots when preparing the anonymous artifact.
""", encoding="utf-8")

    (OUT / "data_release_and_licenses.md").write_text("""# Data Release Boundary

- Release image identifiers, source split, target boxes, hashes, state annotations, selected orders and processing scripts.
- Do not redistribute source images unless the corresponding dataset terms explicitly permit it.
- Provide download and crop reconstruction scripts for Open Images and PASCAL VOC records.
- Remove local absolute paths, credentials and private API configuration from the anonymous package.
- Before upload, the authors must verify the current terms of each source dataset and the target venue's artifact policy. This is an administrative publication check, not a missing experiment.
""", encoding="utf-8")

    (OUT / "submission_readiness.md").write_text("""# Submission Readiness

## Scientific work complete

- [x] Five-class balanced main pool and fixed model predictions
- [x] Eight architectures, three seeds and five budget levels
- [x] Random, DeepGini, DSA and k-center same-budget baselines
- [x] State-held-out and state-independent failure metrics
- [x] Equal-error and class-stratified equal-error controls
- [x] Fixed-budget complementary-selection control
- [x] Error-cluster-count sensitivity and duplicate/leakage audit
- [x] Failure atlas and class/state error analysis
- [x] Online VLM reannotation sensitivity on 200 difficult samples
- [x] Independent-source PASCAL VOC confirmation configuration
- [x] Negative and non-replication results retained in the manuscript
- [x] Unified numerical audit, environment capture and SHA-256 manifest

## Author/venue actions only

- [ ] Add author names, affiliations, funding, acknowledgements and conflict statement
- [ ] Select a venue and apply its template, length and anonymization rules
- [ ] Confirm dataset redistribution terms and package data as IDs/download scripts
- [ ] Perform language polishing and final PDF visual inspection

No additional result-contingent experiment is required for the present empirical claim. A third source or human adjudication would strengthen external/annotation validity but is correctly scoped as future work rather than a hidden prerequisite.
""", encoding="utf-8")

    audit = {
        "status": "pass",
        "critical_files": len(manifest),
        "missing_files": missing,
        "manifest_entries_with_hash": sum(bool(item["sha256"]) for item in manifest.values()),
        "strengthening_audit": json.loads(
            (ROOT / "artifacts/empirical_paper_20260929/strengthening_audit.json").read_text(encoding="utf-8")
        ),
    }
    write_json(OUT / "freeze_audit.json", audit)
    print(json.dumps({"status": "pass", "output": str(OUT), "critical_files": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
