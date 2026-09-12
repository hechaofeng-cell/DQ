#!/usr/bin/env python3
"""Compute automatic dog feature coverage without requiring human review."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.io import atomic_json, atomic_text, unique_json_object


def raw_result(raw_dir, image_id):
    attempts = sorted(raw_dir.glob(f"{image_id}_attempt*.json"))
    for path in reversed(attempts):
        try:
            outer = json.loads(path.read_text(encoding="utf-8"))
            return unique_json_object(outer["message"]["content"]), "parseable_raw"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return {}, "unparseable"


def status(presence, validity, unknown_rate, evidence_rate, effective):
    if presence < .95 or validity < .98 or evidence_rate < .90 or effective < .60:
        return "GAP"
    if unknown_rate > .20 or effective < .80:
        return "WARNING"
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    args = ap.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    definitions = {x["feature_id"]: x for x in schema["universal_features"] + schema.get("category_specific_features", [])}
    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    total = len(manifest)
    metrics = {key: {"present": 0, "missing": 0, "valid": 0, "invalid": 0, "unknown": 0, "valid_non_unknown": 0, "evidence_valid": 0, "evidence_missing_or_weak": 0} for key in definitions}
    accepted = parseable = 0
    for row in manifest:
        image_id = row["image_id"]
        parsed = args.results / "parsed/features" / f"{image_id}.json"
        if parsed.exists():
            value = json.loads(parsed.read_text(encoding="utf-8")); accepted += 1; parseable += 1
        else:
            value, state = raw_result(args.results / "raw/features", image_id)
            parseable += state == "parseable_raw"
        features, evidence = value.get("features", {}), value.get("evidence", {})
        for key, definition in definitions.items():
            metric = metrics[key]
            if key not in features:
                metric["missing"] += 1
                metric["evidence_missing_or_weak"] += 1
                continue
            metric["present"] += 1
            item = features[key]
            allowed = definition["possible_values"]
            valid = item in allowed
            metric["valid" if valid else "invalid"] += 1
            if valid and item == "unknown": metric["unknown"] += 1
            if valid and item != "unknown": metric["valid_non_unknown"] += 1
            text = evidence.get(key)
            evidence_ok = isinstance(text, str) and len(text.strip()) >= 8 and text.strip().lower() != str(item).lower()
            metric["evidence_valid" if evidence_ok else "evidence_missing_or_weak"] += 1
    rows = []
    for key, metric in metrics.items():
        presence = metric["present"] / total
        validity = metric["valid"] / metric["present"] if metric["present"] else 0
        unknown = metric["unknown"] / metric["valid"] if metric["valid"] else 0
        evidence = metric["evidence_valid"] / total
        effective = metric["valid_non_unknown"] / total
        rows.append({"feature_id": key, **metric, "presence_rate": presence, "validity_rate": validity, "unknown_rate": unknown, "evidence_rate": evidence, "effective_coverage": effective, "status": status(presence, validity, unknown, evidence, effective)})
    response_rate = accepted / total
    report = {"protocol": manifest[0].get("protocol_version") if manifest else None, "denominator": total, "accepted_responses": accepted, "strict_response_success_rate": response_rate, "parseable_responses": parseable, "parseable_response_rate": parseable / total if total else 0, "rules": {"field_presence_pass": ">=95%", "enum_validity_pass": ">=98% of present values", "unknown_warning": ">20%", "unknown_gap": ">50%", "evidence_pass": ">=90% of all 500", "effective_coverage_pass": ">=80%", "effective_coverage_warning": "60%-80%", "effective_coverage_gap": "<60%"}, "features": rows, "summary": {s: sum(r["status"] == s for r in rows) for s in ("PASS", "WARNING", "GAP")}, "overall_status": "PASS" if response_rate >= .95 and all(r["status"] == "PASS" for r in rows) else "NEEDS_IMPROVEMENT"}
    atomic_json(args.results / "automatic_coverage_report.json", report)
    with (args.results / "automatic_feature_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# 犬类特征自动覆盖报告", "", f"- 固定分母：{total}", f"- 严格有效响应：{accepted}/{total} ({response_rate:.1%})", f"- 可解析响应：{parseable}/{total} ({parseable/total:.1%})", f"- 判定：**{report['overall_status']}**", "", "| 特征 | 出现率 | 合法率 | unknown率 | 有效覆盖率 | evidence率 | 状态 |", "|---|---:|---:|---:|---:|---:|---|"]
    lines += [f"| {r['feature_id']} | {r['presence_rate']:.1%} | {r['validity_rate']:.1%} | {r['unknown_rate']:.1%} | {r['effective_coverage']:.1%} | {r['evidence_rate']:.1%} | {r['status']} |" for r in rows]
    atomic_text(args.results / "automatic_coverage_report.md", "\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in ("denominator", "accepted_responses", "strict_response_success_rate", "parseable_responses", "summary", "overall_status")}, ensure_ascii=False))


if __name__ == "__main__": main()
