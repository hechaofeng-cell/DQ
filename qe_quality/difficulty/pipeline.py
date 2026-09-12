"""Review-gated scoring, explicit use partitions and frozen validation."""

import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .io import (digest, fingerprint, new_output, read_csv, read_json, seal_run,
                 snapshot, verify_run, verify_snapshot, write_csv, write_json)
from .metrics import discrimination, independent_check, panel_stability, ranks, spearman
from .scoring import ACCEPTED, calibrate, probabilities, score_a, score_b


PARTS = {"unassigned", "calibration_fit", "calibration_check", "development", "validation"}
REVIEW_STATES = {"pending", "confirmed", "corrected", "uncertain", "out_of_scope", "exclude"}
DATA_BLOCKERS = {"missing_or_corrupt_original", "image_decode_error", "same_id_different_original_bytes",
                 "canonical_original_not_found", "original_label_conflict", "duplicate_label_conflict"}


def apply_reviews(rows, reviews, classes):
    for row in rows:
        row.setdefault("reviewed_at", "")
        row.setdefault("evidence", "")
    lookup = {r["original_id"]: r for r in rows}
    seen = set()
    for review in reviews:
        image_id = review["original_id"]
        if image_id in seen or image_id not in lookup:
            raise ValueError(f"Duplicate or unknown review ID: {image_id}")
        seen.add(image_id)
        row = lookup[image_id]
        if not row["sha256"] or review["sha256"] != row["sha256"]:
            raise ValueError(f"Review bytes do not match: {image_id}")
        status = review["review_status"]
        if status not in REVIEW_STATES:
            raise ValueError(f"Unknown review status: {status}")
        reference = review["reference_class_id"]
        if status in ACCEPTED:
            if (reference not in classes or not review.get("reviewer") or not review.get("reviewed_at")
                    or not review.get("evidence") or review.get("reviewer_kind") not in {"human", "assistant_visual"}):
                raise ValueError(f"Accepted review needs supported class, reviewer, kind, time and evidence: {image_id}")
            if (status == "confirmed") != (reference == row["inherited_class_id"]):
                raise ValueError(f"Use corrected status for a changed label: {image_id}")
        elif reference != row["inherited_class_id"]:
            raise ValueError(f"An unaccepted review cannot change the label: {image_id}")
        row.update({k: review.get(k, "") for k in (
            "reference_class_id", "review_status", "reviewer", "reviewer_kind", "reviewed_at", "evidence")})
    return rows


def apply_splits(rows, assignments):
    lookup = {r["original_id"]: r for r in rows}
    seen = set()
    for assignment in assignments:
        image_id = assignment["original_id"]
        if image_id in seen or image_id not in lookup:
            raise ValueError(f"Duplicate or unknown split ID: {image_id}")
        seen.add(image_id)
        row = lookup[image_id]
        if assignment["source_group_id"] != row["source_group_id"] or assignment["split"] not in PARTS:
            raise ValueError(f"Invalid split/group: {image_id}")
        row["split"] = assignment["split"]
    if assignments and seen != set(lookup):
        raise ValueError("Explicit splits must assign every baseline ID, including unassigned rows")
    group_splits = defaultdict(set)
    group_labels = defaultdict(set)
    for row in rows:
        group_splits[row["source_group_id"]].add(row["split"])
        if row["review_status"] in ACCEPTED:
            group_labels[row["source_group_id"]].add(row["reference_class_id"])
    if any(len(s) > 1 for s in group_splits.values()):
        raise ValueError("Same-source images cross use partitions (including unassigned)")
    if any(len(s) > 1 for s in group_labels.values()):
        raise ValueError("Conflicting accepted labels within a source group")
    return rows


def exclusions(rows, predictions, issues, config):
    blockers = defaultdict(set)
    for problem in issues:
        if problem["code"] in DATA_BLOCKERS or (problem["code"] == "same_source_prediction_conflict"
                                               and problem["model"] in config["scoring_models"]):
            blockers[problem["original_id"]].add(problem["code"])
    for row in rows:
        reasons = set(blockers[row["original_id"]])
        if row["reference_class_id"] not in config["class_order"]:
            reasons.add("unsupported_class")
        for model in config["scoring_models"]:
            if not predictions[row["original_id"]].get(model, {}).get("valid"):
                reasons.add("missing_or_invalid_scoring_prediction:" + model)
        row["data_exclusion_reasons"] = ";".join(sorted(reasons))
        if row["review_status"] not in ACCEPTED:
            reasons.add("label_" + row["review_status"])
        row["exclusion_reasons"] = ";".join(sorted(reasons))


def calculate_rows(rows, predictions, config, calibration, stage):
    models = config["scoring_models"]
    all_models = models + config["validation_models"]
    temperatures = {m: r["temperature"] for m, r in calibration["models"].items() if "temperature" in r}
    calibrated = calibration["status"] == "calibration_checked_exploratory"
    score_version = config["protocol_version"] + ":" + fingerprint({
        "protocol": config["protocol_fingerprint"], "calibration": calibration,
        "code": snapshot(Path(__file__).parent.glob("*.py")),
    })[:16]
    result = []
    for original in rows:
        row = dict(original)
        observed = predictions[row["original_id"]]
        for model in all_models:
            p = observed.get(model, {})
            row[model + "_prediction"] = p.get("prediction", "")
            row[model + "_correct"] = (p["prediction"] == row["reference_class_id"]
                                       if p.get("valid") and row["reference_class_id"] in config["class_order"] else None)
            row[model + "_raw_logits"] = json.dumps(p.get("logits")) if p.get("valid") else ""
            row[model + "_prediction_status"] = "valid" if p.get("valid") else "missing_or_invalid"
            row[model + "_true_calibrated_probability"] = None
        row.update(score_a=None, score_b=None, diagnostic_a_inherited=None,
                   score_version=score_version, scoring_models=",".join(models),
                   scoring_model_count=len(models), score_a_status="excluded",
                   score_b_status="pending_validation", risk_flags=";".join([
                       row["source_status"], "validation_model_overlap_unknown", "near_duplicates_not_cleared",
                       "historical_prediction_bytes_not_bound", "not_validated_difficulty"]))
        active = row["split"] == "validation" if stage == "validation" else row["split"] != "validation"
        if not active:
            row["score_a_status"] = "withheld_for_other_stage"
            row["score_b_status"] = "withheld_for_other_stage"
            for model in all_models:
                row[model + "_prediction"] = ""
                row[model + "_correct"] = None
                row[model + "_raw_logits"] = ""
                row[model + "_prediction_status"] = "withheld_for_other_stage"
            result.append(row)
            continue
        pred = {m: observed[m]["prediction"] for m in models if observed.get(m, {}).get("valid")}
        if not row["data_exclusion_reasons"] and row["review_status"] == "pending":
            row["diagnostic_a_inherited"] = score_a(pred, row["inherited_class_id"], models)
        if not row["exclusion_reasons"]:
            row["score_a"] = score_a(pred, row["reference_class_id"], models)
            row["score_a_status"] = "reviewed_exploratory"
            if calibrated and row["split"] not in {"calibration_fit", "calibration_check"}:
                logits = {m: observed[m]["logits"] for m in models}
                j = config["class_order"].index(row["reference_class_id"])
                row["score_b"] = score_b(logits, j, models, temperatures)
                row["score_b_status"] = "calibration_checked_exploratory"
                for m in models:
                    row[m + "_true_calibrated_probability"] = probabilities(logits[m], temperatures[m])[j]
            elif row["split"] in {"calibration_fit", "calibration_check"}:
                row["score_b_status"] = "calibration_data_not_evaluation"
        else:
            row["score_b_status"] = "excluded:" + row["exclusion_reasons"]
        result.append(row)
    return result


def b_stability(rows, models):
    selected = [r for r in rows if r["score_b"] is not None]
    output = {}
    for removed in models:
        remaining = [m for m in models if m != removed]
        if not remaining:
            continue
        base = [r["score_b"] for r in selected]
        alt = [1 - sum(r[m + "_true_calibrated_probability"] for m in remaining) / len(remaining) for r in selected]
        shifts = [abs(a - b) for a, b in zip(ranks(base), ranks(alt))]
        output["remove_" + removed] = {"n": len(base), "spearman": spearman(base, alt),
            "mean_absolute_rank_shift": sum(shifts) / len(shifts) if shifts else None,
            "max_absolute_rank_shift": max(shifts, default=None)}
    return output


def summarize_subset(rows, config):
    output = {}
    for field in ("score_a", "score_b", "diagnostic_a_inherited"):
        values = [r[field] for r in rows if r[field] is not None]
        output[field] = {"discrimination": discrimination(values), "independent_models": {
            model: independent_check(rows, field, model, config["diagnostic_band_edges"])
            for model in config["validation_models"]}}
    output["stability_a"] = panel_stability(rows, config["scoring_models"])
    output["stability_a_inherited"] = panel_stability(rows, config["scoring_models"], "diagnostic_a_inherited")
    output["stability_b"] = b_stability(rows, config["scoring_models"])
    paired = [r for r in rows if r["score_a"] is not None and r["score_b"] is not None]
    ties = defaultdict(list)
    for row in paired:
        ties[row["score_a"]].append(row["score_b"])
    output["method_comparison"] = {"paired_n": len(paired),
        "spearman_a_b": spearman([r["score_a"] for r in paired], [r["score_b"] for r in paired]),
        "b_discrimination_within_a_ties": {str(a): discrimination(v) for a, v in ties.items()},
        "interpretation": "More unique decimals alone do not establish additional predictive value."}
    return output


def build_report(rows, config, calibration, stage):
    # Never pool calibration, development and validation to claim validation.
    parts = ["validation"] if stage == "validation" else ["unassigned", "development"]
    analyses = {}
    for part in parts:
        selected = [r for r in rows if r["split"] == part]
        by_class = {c: summarize_subset([r for r in selected if r["reference_class_id"] == c], config)
                    for c in config["class_order"]}
        analysis = summarize_subset(selected, config)
        analysis["by_class"] = by_class
        analysis["macro_class_auc"] = {}
        for field in ("score_a", "score_b", "diagnostic_a_inherited"):
            analysis["macro_class_auc"][field] = {}
            for model in config["validation_models"]:
                values = [v[field]["independent_models"][model]["error_auc"] for v in by_class.values()]
                defined = [v for v in values if v is not None]
                analysis["macro_class_auc"][field][model] = {
                    "classes_with_defined_auc": len(defined),
                    "mean_auc": sum(defined) / len(defined) if defined else None,
                    "caveat": "Classes without both errors and correct predictions are undefined, not zero."}
        analyses[part] = analysis
    reasons = ["Reference/validation model training overlap is unresolved.",
               "Visual near duplicates have not been cleared.",
               "Historical GPU output does not bind each prediction to a byte hash.",
               "Historical predictions have already been explored; a retrospective partition is not a blind test."]
    pending = sum(r["review_status"] not in ACCEPTED for r in rows)
    if pending:
        reasons.append(f"{pending} labels remain unaccepted or excluded.")
    if stage != "validation":
        reasons.append("No frozen validation evaluation was performed in this run.")
    if calibration["status"] != "calibration_checked_exploratory":
        reasons.append("B calibration evidence is insufficient or did not pass its independent check.")
    return {
        "engineering_status": "passed_automatic_checks", "research_status": "exploratory_only_not_ready_for_formal_grouping",
        "stage": stage, "baseline_count": len(rows), "review_counts": dict(Counter(r["review_status"] for r in rows)),
        "reviewer_kind_counts": dict(Counter(r.get("reviewer_kind", "") or "none" for r in rows)),
        "source_counts": dict(Counter(r["source_status"] for r in rows)),
        "score_a_count": sum(r["score_a"] is not None for r in rows),
        "score_b_count": sum(r["score_b"] is not None for r in rows),
        "inherited_diagnostic_count": sum(r["diagnostic_a_inherited"] is not None for r in rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "split_coverage": {p: {"classes": dict(Counter(r["reference_class_id"] for r in rows if r["split"] == p)),
                               "accepted_labels": sum(r["split"] == p and r["review_status"] in ACCEPTED for r in rows)} for p in sorted(PARTS)},
        "reviewed_availability_by_class": {c: {
            "eligible_a": sum(r["reference_class_id"] == c and not r["exclusion_reasons"] for r in rows),
            "model_errors": {m: sum(r["reference_class_id"] == c and not r["exclusion_reasons"]
                                     and r[m + "_correct"] is False for r in rows)
                             for m in config["scoring_models"] + config["validation_models"]},
        } for c in config["class_order"]},
        "calibration": calibration, "analyses": analyses, "unresolved": reasons,
        "limitations": ["Review-selected samples enrich model errors; their error rates are not population estimates.",
                        "Assistant visual review is not human label certification.",
                        "Diagnostic bands are fixed numerical intervals, not easy/medium/hard grades.",
                        "A removed-panel comparison changes the definition; it is only a sensitivity analysis.",
                        "No automatic scientific acceptance, thresholds, resampling or metamorphic tests."],
    }


def render_report(report):
    lines = ["# 图片难度评分与验证报告", "",
             "工程验收：自动检查通过。研究验收：仅探索性结果，当前评分尚不适合正式分组。", "",
             f"基准原图 {report['baseline_count']} 张；审核后 A 候选分数 {report['score_a_count']} 张；"
             f"未审核标签诊断值 {report['inherited_diagnostic_count']} 张；B 分数 {report['score_b_count']} 张。", "",
             "完整逐图结果见 scores.csv，数值与类别分析见 report.json，输入与运行信息见 run.json。", "",
             "## 审核与来源", "", f"审核状态：`{json.dumps(report['review_counts'], ensure_ascii=False)}`。", "",
             f"来源状态：`{json.dumps(report['source_counts'], ensure_ascii=False)}`。", "",
             "assistant_visual 表示助手视觉检查，不是人工认证。错误/分歧定向检查的样本不能代表总体；"
             "pending 仅有继承标签诊断值，uncertain/out_of_scope/exclude 无分数。", "",
             "## A / B 与独立模型检查", ""]
    for part, analysis in report["analyses"].items():
        lines.extend([f"### {part}", "", "各列分开分析，不能将审核选样的错误率当作总体错误率。", "",
                      "| 字段 | 数量 | 不同分数 | 最大同分组 |", "| --- | ---: | ---: | ---: |"])
        for field in ("score_a", "score_b", "diagnostic_a_inherited"):
            d = analysis[field]["discrimination"]
            lines.append(f"| {field} | {d['n']} | {d['unique_scores']} | {d['largest_tie_count']} |")
        lines.extend(["", "固定数值区间的辅助验证模型错误率（95% Wilson 区间；按同源组计数）：", "",
                      "| 字段 | 模型 | 区间 | 样本组 | 错误数 | 错误率 | 95% 区间 |",
                      "| --- | --- | --- | ---: | ---: | --- | --- |"])
        for field in ("score_a", "score_b", "diagnostic_a_inherited"):
            for model, check in analysis[field]["independent_models"].items():
                for band in check["bands"]:
                    if not band["n_groups"]:
                        continue
                    interval = band["wilson_95"]
                    bound = "]" if band["upper_inclusive"] else ")"
                    lines.append(f"| {field} | {model} | [{band['lower']}, {band['upper']}{bound} | "
                                 f"{band['n_groups']} | {band['errors']} | {band['error_rate']:.3f} | "
                                 f"[{interval[0]:.3f}, {interval[1]:.3f}] |")
        lines.extend(["", "移除模型的 Spearman、平均/最大名次变化、各类别诊断和类别宏平均 AUC 均保存在 report.json。"
                      "常量排序或某类无错误时返回 null，不制造相关系数。", ""])
    lines.extend(["## 校准", "", f"状态：`{report['calibration']['status']}`。", ""])
    lines.extend("- " + reason for reason in report["calibration"]["reasons"])
    lines.extend(["", "B 只使用独立校准后正确类别的概率；校准拟合与校准检查分开，未满足条件不生成 B。", "",
                  "## 未解决问题", ""])
    lines.extend("- " + reason for reason in report["unresolved"] + report["limitations"])
    lines.extend(["", "证据边界：可计算、可追溯、可复算不代表评分方法有效。此报告不划分等级、不调整分布、不运行蜕变测试。", ""])
    return "\n".join(lines)


def automatic_checks(rows, predictions, config, calibration, stage):
    if len(rows) != config["expected_count"] or len({r["original_id"] for r in rows}) != len(rows):
        raise ValueError("Score row cardinality mismatch")
    apply_splits([dict(r) for r in rows], [])
    temps = {m: r["temperature"] for m, r in calibration["models"].items() if "temperature" in r}
    for row in rows:
        for field in ("score_a", "score_b", "diagnostic_a_inherited"):
            value = row[field]
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Nonfinite or out-of-range score")
        if row["score_a"] is not None:
            if row["exclusion_reasons"] or row["review_status"] not in ACCEPTED:
                raise ValueError("Excluded row was scored")
            expected = sum(row[m + "_prediction"] != row["reference_class_id"] for m in config["scoring_models"]) / len(config["scoring_models"])
            if row["score_a"] != expected:
                raise ValueError("A formula recomputation failed")
        if row["diagnostic_a_inherited"] is not None:
            if row["review_status"] != "pending" or row["data_exclusion_reasons"]:
                raise ValueError("Invalid unreviewed diagnostic")
            expected = sum(row[m + "_prediction"] != row["inherited_class_id"] for m in config["scoring_models"]) / len(config["scoring_models"])
            if row["diagnostic_a_inherited"] != expected:
                raise ValueError("Inherited diagnostic recomputation failed")
        if row["score_b"] is not None:
            if calibration["status"] != "calibration_checked_exploratory" or row["score_a"] is None:
                raise ValueError("B bypassed calibration or label gate")
            expected_p = []
            for model in config["scoring_models"]:
                z = predictions[row["original_id"]][model]["logits"]
                j = config["class_order"].index(row["reference_class_id"])
                # Independent formula in log space, not the score_b implementation.
                log_den = math.log(sum(math.exp((v - max(z)) / temps[model]) for v in z))
                expected_p.append(math.exp((z[j] - max(z)) / temps[model] - log_den))
            if not math.isclose(row["score_b"], 1 - sum(expected_p) / len(expected_p), abs_tol=1e-12):
                raise ValueError("B formula recomputation failed")
        if (stage == "development" and row["split"] == "validation"
                or stage == "validation" and row["split"] != "validation"):
            if any(row[f] is not None for f in ("score_a", "score_b", "diagnostic_a_inherited")):
                raise ValueError("Evaluation stage isolation failed")
    return {"row_alignment": True, "fixed_denominator_and_formula_recomputation": True,
            "exclusions_and_finite_values": True, "source_group_isolation": True,
            "validation_stage_isolation": True, "calibration_gate": True,
            "disjoint_model_roles": not bool(set(config["scoring_models"]) & set(config["validation_models"]))}


def evaluate(prepared, output, reviews_path=None, splits_path=None, frozen=None):
    started = datetime.now(timezone.utc).isoformat()
    prepared = Path(prepared).resolve()
    verify_run(prepared)
    config = read_json(prepared / "config.json")
    historical = read_json(prepared / "provenance.json")["input_sha256"]
    verify_snapshot(historical)
    predictions = read_json(prepared / "predictions.json")
    issues = read_csv(prepared / "issues.csv")
    external = snapshot([p for p in (reviews_path, splits_path, frozen) if p])
    if frozen:
        if reviews_path or splits_path:
            raise ValueError("Frozen validation cannot change reviews or splits")
        verify_run(Path(frozen).parent)
        lock = read_json(frozen)
        verify_snapshot(lock["development_artifacts"])
        verify_snapshot(lock["code_sha256"])
        if lock["prepared_artifact_hash"] != digest(prepared / "artifact_hashes.json"):
            raise ValueError("Frozen protocol refers to a different prepared dataset")
        if lock["protocol_fingerprint"] != config["protocol_fingerprint"]:
            raise ValueError("Frozen protocol mismatch")
        rows = lock["effective_samples"]
        calibration = lock["calibration"]
        stage = "validation"
    else:
        rows = read_csv(prepared / "samples.csv")
        apply_reviews(rows, read_csv(reviews_path) if reviews_path else [], config["class_order"])
        apply_splits(rows, read_csv(splits_path) if splits_path else [])
        exclusions(rows, predictions, issues, config)
        calibration = calibrate(rows, predictions, config)
        stage = "development"
    scores = calculate_rows(rows, predictions, config, calibration, stage)
    checks = automatic_checks(scores, predictions, config, calibration, stage)
    report = build_report(scores, config, calibration, stage)
    protected = [prepared, Path(config["prediction_dir"]).parent] + [Path(p) for p in (reviews_path, splits_path, frozen) if p]
    output = new_output(output, protected)
    write_csv(output / "scores.csv", scores)
    write_json(output / "effective_samples.json", rows)
    write_json(output / "config.json", config)
    write_json(output / "calibration.json", calibration)
    write_json(output / "report.json", report)
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(render_report(report))
    queue = {r["original_id"]: r["review_reasons"] for r in read_csv(prepared / "review_queue.csv")}
    write_csv(output / "sample_checks.csv", [
        {"original_id": r["original_id"], "selection": queue.get(r["original_id"], "not_selected"),
         "review_status": r["review_status"], "reviewer_kind": r.get("reviewer_kind", ""),
         "evidence": r.get("evidence", ""), "exclusion_reasons": r["exclusion_reasons"]} for r in scores])
    failures = [r for r in scores if r["exclusion_reasons"] or any(r[m + "_correct"] is False
                for m in config["scoring_models"] + config["validation_models"])]
    write_csv(output / "failure_cases.csv", failures, list(scores[0]))
    verify_snapshot(historical)
    verify_snapshot(external)
    checks["historical_files_unchanged"] = True
    write_json(output / "checks.json", checks)
    code_paths = list(Path(__file__).parent.glob("*.py"))
    write_json(output / "run.json", {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": platform.python_version(), "platform": platform.platform(),
        "stage": stage, "prepared": str(prepared), "prepared_artifact_hash": digest(prepared / "artifact_hashes.json"),
        "external_input_sha256": external, "code_sha256": snapshot(code_paths),
        "protocol_fingerprint": config["protocol_fingerprint"],
        "inference_run": False, "metamorphic_test_run": False, "dataset_mutation": False,
    })
    seal_run(output)
    return output


def freeze(development, output):
    development = Path(development).resolve()
    verify_run(development)
    run = read_json(development / "run.json")
    rows = read_json(development / "effective_samples.json")
    if run["stage"] != "development" or not {"development", "validation"} <= {r["split"] for r in rows}:
        raise ValueError("Freeze requires explicit development and validation partitions")
    if not all(any(r["split"] == p and r["review_status"] in ACCEPTED and not r["exclusion_reasons"]
                   for r in rows) for p in ("development", "validation")):
        raise ValueError("Freeze requires eligible reviewed rows in both partitions")
    output = new_output(output, [development, Path(run["prepared"]), Path(read_json(development / "config.json")["prediction_dir"]).parent])
    write_json(output / "frozen.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prepared_artifact_hash": run["prepared_artifact_hash"],
        "protocol_fingerprint": run["protocol_fingerprint"],
        "effective_samples": rows, "calibration": read_json(development / "calibration.json"),
        "development_artifacts": snapshot(development.iterdir()),
        "code_sha256": run["code_sha256"],
        "interpretation": "Operational rule freeze only; retrospective historical data is not certified blind or uncontaminated.",
    })
    seal_run(output)
    return output
