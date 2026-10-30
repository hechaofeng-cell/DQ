#!/usr/bin/env python3
"""Audit strengthening artifacts and manuscript number traceability."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: str, expected: float, tolerance: float = 5e-5) -> None:
    if abs(float(actual) - expected) > tolerance:
        raise ValueError(f"number mismatch: {actual} vs {expected}")


def main() -> None:
    class_root = ROOT / "artifacts/class_stratified_error_controls_20260929/tables"
    voc_root = ROOT / "artifacts/voc2012_confirmation_benchmark_20260929/tables"
    online_root = ROOT / "artifacts/online_vlm_reannotation_20260929"
    expected_counts = {
        class_root / "class_equal_error_metrics.csv": 480,
        class_root / "class_equal_error_by_class.csv": 2400,
        class_root / "class_equal_error_auc.csv": 96,
        class_root / "class_equal_error_comparisons.csv": 8,
        class_root / "hybrid_class_equal_error_metrics.csv": 480,
        class_root / "hybrid_class_equal_error_by_class.csv": 2400,
        class_root / "hybrid_class_equal_error_auc.csv": 96,
        class_root / "hybrid_class_equal_error_comparisons.csv": 8,
        voc_root / "method_budget_metrics.csv": 600,
        voc_root / "auc_by_run.csv": 120,
        voc_root / "class_equal_error_metrics.csv": 480,
        voc_root / "direct_comparisons.csv": 8,
        voc_root / "class_equal_error_comparisons.csv": 8,
        voc_root / "method_budget_summary.csv": 25,
        voc_root / "model_independent_orders.csv": 1000,
        online_root / "results.csv": 200,
        online_root / "analysis/field_agreement.csv": 19,
        online_root / "analysis/selection_stability.csv": 25,
        online_root / "analysis/sensitivity_by_run.csv": 24,
        online_root / "analysis/sensitivity_comparisons.csv": 4,
        online_root / "analysis/class_equal_error_metrics.csv": 120,
        online_root / "analysis/class_equal_error_by_class.csv": 600,
        online_root / "analysis/class_equal_error_auc.csv": 24,
        online_root / "analysis/class_equal_error_comparisons.csv": 4,
    }
    observed = {}
    for path, expected in expected_counts.items():
        count = len(rows(path))
        observed[str(path.relative_to(ROOT))] = count
        if count != expected:
            raise ValueError(f"{path}: expected {expected}, got {count}")

    class_comparisons = rows(class_root / "class_equal_error_comparisons.csv")
    lookup = {(row["comparison"], row["metric"]): row for row in class_comparisons}
    close(lookup["coverage_vs_random", "class_equal_error_confusion"]["mean_delta_auc"], .0222, 8e-5)
    close(lookup["coverage_vs_random", "class_equal_error_clusters"]["mean_delta_auc"], .0115, 8e-5)
    close(lookup["coverage_vs_random", "class_equal_error_confusion"]["holm_primary_p"], .015625)
    close(lookup["coverage_vs_random", "class_equal_error_clusters"]["holm_primary_p"], .0546875)

    voc_comparisons = rows(voc_root / "class_equal_error_comparisons.csv")
    lookup = {(row["comparison"], row["metric"]): row for row in voc_comparisons}
    close(lookup["coverage_vs_random", "class_equal_error_confusion"]["mean_delta_auc"], .00075959)
    close(lookup["coverage_vs_random", "class_equal_error_clusters"]["mean_delta_auc"], -.00077174)
    close(lookup["coverage_vs_random", "class_equal_error_confusion"]["holm_primary_p"], 1.0)

    online_report = json.loads((online_root / "report.json").read_text(encoding="utf-8"))
    voc_online_report = json.loads((ROOT / "artifacts/voc2012_confirmation_pool_20260929/online_annotations/report.json").read_text(encoding="utf-8"))
    pool_audit = json.loads((ROOT / "artifacts/voc2012_confirmation_pool_20260929/audit.json").read_text(encoding="utf-8"))
    agreement = json.loads((online_root / "analysis/agreement_summary.json").read_text(encoding="utf-8"))
    if (online_report["successful"], online_report["failed"]) != (200, 0):
        raise ValueError("original-pool online annotation incomplete")
    if (voc_online_report["successful"], voc_online_report["failed"]) != (500, 0):
        raise ValueError("VOC online annotation incomplete")
    if pool_audit["total"] != 500 or pool_audit["unique_source_images"] != 500:
        raise ValueError("VOC pool audit failed")
    close(str(agreement["micro_exact_agreement"]), .59605263)
    close(str(agreement["enum_macro_kappa"]), .44366048)

    online_strict = rows(online_root / "analysis/class_equal_error_comparisons.csv")
    lookup = {(row["comparison"], row["metric"]): row for row in online_strict}
    close(lookup["online_merged_coverage_vs_random", "class_equal_error_confusion"]["mean_delta_auc"], .02470249)
    close(lookup["online_merged_coverage_vs_random", "class_equal_error_clusters"]["mean_delta_auc"], .01674655)
    close(lookup["online_merged_coverage_vs_random", "class_equal_error_confusion"]["holm_primary_p"], .015625)
    close(lookup["online_merged_coverage_vs_random", "class_equal_error_clusters"]["holm_primary_p"], .015625)

    manuscript = ROOT / "docs/visual_state_coverage_empirical_study_draft_20260929.md"
    text = manuscript.read_text(encoding="utf-8")
    required = (
        "+0.0222", "+0.0115", "+0.0247", "+0.0167",
        "+0.0008", "-0.0008", "59.61%", "0.444", "未复现",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ValueError(f"manuscript missing audited values: {missing}")

    word_document = ROOT / "docs/visual_state_coverage_empirical_study_revised_20260929.docx"
    manifest_files = [*expected_counts, manuscript,
        ROOT / "artifacts/voc2012_confirmation_pool_20260929/audit.json",
        online_root / "report.json",
        ROOT / "artifacts/voc2012_confirmation_pool_20260929/online_annotations/report.json",
        online_root / "analysis/report_zh.md",
        online_root / "analysis/online_class_equal_error_sensitivity.png",
        online_root / "analysis/online_class_equal_error_sensitivity.svg",
    ]
    if word_document.is_file():
        manifest_files.append(word_document)
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in manifest_files}
    output = ROOT / "artifacts/empirical_paper_20260929"
    (output / "strengthening_audit.json").write_text(json.dumps({
        "status": "pass", "audited_rows": sum(observed.values()),
        "tables": observed, "online_annotations": {"main_subset": 200, "voc": 500},
        "online_label_sensitivity": {
            "class_equal_error_confusion_auc": .02470249,
            "class_equal_error_cluster_auc": .01674655,
            "holm_primary_p": .015625,
        },
        "manuscript_values_checked": list(required),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "strengthening_manifest.sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "audited_rows": sum(observed.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
