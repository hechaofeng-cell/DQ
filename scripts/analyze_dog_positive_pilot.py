#!/usr/bin/env python3
"""Apply acceptance rules for a dog-only positive-sample pilot."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.io import atomic_json, atomic_text

CONDITIONAL = {"tail_position"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    coverage = json.loads((args.results / "automatic_coverage_report.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((args.results / "results.csv").open(encoding="utf-8-sig", newline="")))
    total = len(rows)
    classifier_responses = sum(r["classifier_status"] in {"ok", "unknown"} for r in rows)
    dog_hits = sum(r["predicted_class"] == "dog" for r in rows)
    refusals = sum(r["predicted_class"] == "unknown" for r in rows)
    concrete_errors = total - dog_hits - refusals
    feature_rows = []
    for item in coverage["features"]:
        structural = "PASS" if item["presence_rate"] >= .98 and item["validity_rate"] >= .95 and item["evidence_rate"] >= .90 else ("WARNING" if item["presence_rate"] >= .95 and item["validity_rate"] >= .90 and item["evidence_rate"] >= .80 else "FAIL")
        observability = "CONDITIONAL" if item["feature_id"] in CONDITIONAL else ("LOW" if item["unknown_rate"] > .50 else "MODERATE" if item["unknown_rate"] > .20 else "NORMAL")
        feature_rows.append({**item, "structural_status": structural, "observability_status": observability})
    gates = {
        "parseable_responses_at_least_99pct": coverage["parseable_response_rate"] >= .99,
        "strict_responses_at_least_90pct": coverage["strict_response_success_rate"] >= .90,
        "all_fields_present_at_least_98pct": all(x["presence_rate"] >= .98 for x in feature_rows),
        "all_enum_validity_at_least_90pct": all(x["validity_rate"] >= .90 for x in feature_rows),
        "classifier_response_rate_100pct": classifier_responses == total,
        "dog_hit_rate_at_least_90pct": dog_hits / total >= .90,
    }
    report = {"evaluation_scope": "dog_positive_samples_only", "denominator": total, "feature_parseable_rate": coverage["parseable_response_rate"], "feature_strict_protocol_rate": coverage["strict_response_success_rate"], "classifier_response_rate": classifier_responses / total, "dog_hit_rate": dog_hits / total, "classifier_refusal_rate": refusals / total, "concrete_misclassification_rate": concrete_errors / total, "conditional_features": sorted(CONDITIONAL), "features": feature_rows, "gates": gates, "decision": "PASS_DOG_POSITIVE_PILOT" if all(gates.values()) else "NEEDS_IMPROVEMENT"}
    atomic_json(args.results / "dog_positive_pilot_report.json", report)
    lines = ["# Dog 正样本试验验收报告", "", f"决策：**{report['decision']}**", "", f"- Dog 识别命中率：{report['dog_hit_rate']:.1%}", f"- 分类拒答率：{report['classifier_refusal_rate']:.1%}", f"- 明确误分类率：{report['concrete_misclassification_rate']:.1%}", f"- 特征可解析率：{report['feature_parseable_rate']:.1%}", f"- 严格协议合规率：{report['feature_strict_protocol_rate']:.1%}", "", "`tail_position` 是条件可观察特征；不可见时的 `unknown` 是有效结果，不作为失败。", "", "| 门槛 | 结果 |", "|---|---|"]
    lines += [f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in gates.items()]
    atomic_text(args.results / "dog_positive_pilot_report.md", "\n".join(lines) + "\n")
    print(json.dumps({"decision": report["decision"], "dog_hit_rate": report["dog_hit_rate"], "feature_strict_protocol_rate": report["feature_strict_protocol_rate"], "gates": gates}, ensure_ascii=False))


if __name__ == "__main__": main()
