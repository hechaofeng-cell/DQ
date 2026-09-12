"""Phase-one descriptive, agreement, review, and gate reports."""

import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path

from .io import atomic_csv, atomic_json, atomic_text, read_csv
from .schema import CLASS_NAMES, OCCLUSIONS, OUTLINES, SUBJECT_ROLES

FEATURE_FIELDS = [
    "image_id", "sha256", "source_class_id", "source_class_name", "class_id", "class_name",
    "raw_class_id", "raw_class_name", "fine_name", "class_confidence", "subject_role",
    "selection_confidence", "subject_selection_evidence", "outline_visibility", "occlusion_level",
    "target_bbox_hint", "raw_target_bbox_hint", "target_bbox_scale", "bbox", "area_ratio", "horizontal_position", "vertical_position",
    "legacy_position", "sam_score", "background_complement_applied", "evidence", "local_elapsed_seconds", "sam_elapsed_seconds",
    "end_to_end_seconds", "protocol_hash",
]
REVIEW_FIELDS = [
    "image_id", "sha256", "selection_reason", "human_class_id", "human_subject_role", "human_outline_visibility",
    "human_occlusion_level", "sam_target_match", "sam_mask_acceptable", "reviewer",
    "reviewed_at", "notes",
]


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def fmt(value):
    return "N/A" if value is None else f"{value:.1%}"


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def kappa(left, right):
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    lcount, rcount, total = Counter(left), Counter(right), len(left)
    expected = sum(lcount[key] * rcount[key] for key in set(lcount) | set(rcount)) / total ** 2
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def build_features(rows, local, sam, protocol_hash):
    output = []
    for row in rows:
        image_id = row["image_id"]
        if image_id not in local or image_id not in sam:
            continue
        semantic, geometry = local[image_id], sam[image_id]
        local_time = semantic["elapsed_seconds"]
        sam_time = geometry["elapsed_seconds"]
        output.append({
            **{key: row[key] for key in ("image_id", "sha256", "source_class_id", "source_class_name")},
            **{key: semantic[key] for key in (
                "class_id", "class_name", "raw_class_id", "raw_class_name", "fine_name", "class_confidence",
                "subject_role", "selection_confidence", "subject_selection_evidence",
                "outline_visibility", "occlusion_level", "target_bbox_hint", "raw_target_bbox_hint",
                "target_bbox_scale", "evidence",
            )},
            **{key: geometry[key] for key in (
                "bbox", "area_ratio", "horizontal_position", "vertical_position", "legacy_position", "sam_score",
                "background_complement_applied",
            )},
            "local_elapsed_seconds": local_time, "sam_elapsed_seconds": sam_time,
            "end_to_end_seconds": local_time + sam_time, "protocol_hash": protocol_hash,
        })
    return output


def write_feature_files(output, features):
    serialized = []
    for row in features:
        value = dict(row)
        for field in ("target_bbox_hint", "raw_target_bbox_hint", "bbox"):
            value[field] = __import__("json").dumps(value[field], separators=(",", ":"))
        serialized.append(value)
    atomic_csv(output / "features.csv", serialized, FEATURE_FIELDS)
    try:
        import pandas as pd
        temporary = output / "features.parquet.tmp"
        pd.DataFrame(features, columns=FEATURE_FIELDS).to_parquet(temporary, index=False)
        temporary.replace(output / "features.parquet")
    except ImportError as exc:
        raise ValueError("pandas_and_pyarrow_required_for_parquet") from exc


def deterministic_review_ids(rows, seed, per_class=2):
    selected = set()
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_class_id"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['image_id']}".encode()).hexdigest())
        selected.update(row["image_id"] for row in values[:per_class])
    return selected


def write_review_template(output, rows, local, online, seed):
    path = output / "human_review.csv"
    if path.exists():
        return
    fixed = deterministic_review_ids(rows, seed)
    disagreement = set()
    for image_id in set(local) & set(online):
        if any(local[image_id][field] != online[image_id][field]
               for field in ("class_id", "subject_role", "outline_visibility", "occlusion_level")):
            disagreement.add(image_id)
    by_id = {row["image_id"]: row for row in rows}
    review_rows = []
    for image_id in sorted(fixed | disagreement):
        reasons = []
        if image_id in fixed:
            reasons.append("fixed_stratified_20")
        if image_id in disagreement:
            reasons.append("semantic_disagreement")
        review_rows.append({
            "image_id": image_id, "sha256": by_id[image_id]["sha256"],
            "selection_reason": "+".join(reasons),
            **{field: "" for field in REVIEW_FIELDS[3:]},
        })
    atomic_csv(path, review_rows, REVIEW_FIELDS)


def read_reviews(path, rows):
    if not Path(path).exists():
        return []
    by_id = {row["image_id"]: row for row in rows}
    reviews = []
    for row in read_csv(path):
        if not any(row.get(field, "").strip() for field in REVIEW_FIELDS[3:]):
            continue
        image_id = row.get("image_id")
        if image_id not in by_id or row.get("sha256") != by_id[image_id]["sha256"]:
            raise ValueError("human_review_input_mismatch")
        if row["human_class_id"] not in {str(value) for value in CLASS_NAMES}:
            raise ValueError("invalid_human_class_id")
        if row["human_subject_role"] not in SUBJECT_ROLES:
            raise ValueError("invalid_human_subject_role")
        if row["human_outline_visibility"] not in OUTLINES or row["human_occlusion_level"] not in OCCLUSIONS:
            raise ValueError("invalid_human_semantic_review")
        if row["sam_target_match"] not in {"yes", "no", "unknown"} or row["sam_mask_acceptable"] not in {"yes", "no", "unknown"}:
            raise ValueError("invalid_human_sam_review")
        if not row["reviewer"].strip() or not row["reviewed_at"].strip():
            raise ValueError("missing_human_review_provenance")
        reviews.append(row)
    return reviews


def report_all(output, rows, local, online, sam, errors, config, protocol_hash):
    features = build_features(rows, local, sam, protocol_hash)
    write_feature_files(output, features)
    write_review_template(output, rows, local, online, config["review_seed"])
    reviews = read_reviews(output / "human_review.csv", rows)

    common = sorted(set(local) & set(online))
    agreement = {}
    for field in ("class_id", "subject_role", "outline_visibility", "occlusion_level"):
        left, right = [local[key][field] for key in common], [online[key][field] for key in common]
        agreement[field] = {"n": len(common), "exact": fraction(sum(a == b for a, b in zip(left, right)), len(common)),
                            "kappa": kappa(left, right)}
    disagreement_rows = []
    for image_id in common:
        for field in ("class_id", "subject_role", "outline_visibility", "occlusion_level"):
            if local[image_id][field] != online[image_id][field]:
                disagreement_rows.append({"image_id": image_id, "field": field,
                                          "local": local[image_id][field], "online": online[image_id][field]})
    atomic_csv(output / "disagreements.csv", disagreement_rows,
               ["image_id", "field", "local", "online"])
    hint_ious = []
    for key in common:
        a, b = local[key]["target_bbox_hint"], online[key]["target_bbox_hint"]
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        hint_ious.append(intersection / union if union else 0)

    human = {}
    for backend, values in (("local", local), ("online", online)):
        human[backend] = {}
        for model_field, review_field in (("class_id", "human_class_id"),
                                          ("subject_role", "human_subject_role"),
                                          ("outline_visibility", "human_outline_visibility"),
                                          ("occlusion_level", "human_occlusion_level")):
            eligible = [r for r in reviews if r["image_id"] in values]
            hits = sum(str(values[r["image_id"]][model_field]) == r[review_field] for r in eligible)
            human[backend][model_field] = {"correct": hits, "n": len(eligible), "rate": fraction(hits, len(eligible))}

    fixed_reviews = [r for r in reviews if "fixed_stratified_20" in r["selection_reason"]]
    sam_known = [r for r in fixed_reviews if r["sam_target_match"] != "unknown" and r["sam_mask_acceptable"] != "unknown"]
    sam_accepted = sum(r["sam_target_match"] == "yes" and r["sam_mask_acceptable"] == "yes" for r in sam_known)
    timings = [row["end_to_end_seconds"] for row in features]
    evidence_lengths = [len("".join(local[key]["evidence"].split())) for key in local]
    total = len(rows)
    gates = {
        "class_agreement_at_least_85pct": agreement["class_id"]["exact"] is not None and agreement["class_id"]["exact"] >= .85,
        "sam_complete_and_review_at_least_90pct": len(sam) == total and len(sam_known) >= 20 and fraction(sam_accepted, len(sam_known)) >= .9,
        "both_json_success_at_least_95pct": len(local) / total >= .95 and len(online) / total >= .95,
        "local_end_to_end_p95_at_most_5s": percentile(timings, .95) is not None and percentile(timings, .95) <= 5,
        "mean_evidence_at_least_100_chars_no_empty": len(evidence_lengths) == total and min(evidence_lengths, default=0) > 0 and statistics.mean(evidence_lengths) >= 100,
    }
    reviewed_enough = len(sam_known) >= 20
    decision = "GO" if all(gates.values()) else ("NEEDS_REVIEW" if not reviewed_enough else
               "REVISE" if sum(not value for value in gates.values()) <= 2 else "STOP")
    report = {
        "protocol_hash": protocol_hash, "planned": total,
        "counts": {"local": len(local), "sam": len(sam), "online": len(online), "features": len(features),
                   "human_reviews": len(reviews), "sam_reviewed": len(sam_known), "errors": len(errors)},
        "agreement": agreement,
        "target_hint_iou": {"n": len(hint_ious), "mean": statistics.mean(hint_ious) if hint_ious else None},
        "human_agreement": human,
        "sam_review": {"accepted": sam_accepted, "n": len(sam_known), "rate": fraction(sam_accepted, len(sam_known))},
        "latency_seconds": {"mean": statistics.mean(timings) if timings else None,
                            "p50": percentile(timings, .5), "p95": percentile(timings, .95)},
        "evidence_chars": {"mean": statistics.mean(evidence_lengths) if evidence_lengths else None,
                           "minimum": min(evidence_lengths) if evidence_lengths else None},
        "gates": gates, "decision": decision,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "checks.json", gates)
    atomic_csv(output / "errors.csv", errors, ["stage", "image_id", "error"])
    _write_markdown(output, features, report)
    return report


def _write_markdown(output, features, report):
    class_counts = Counter(row["class_name"] for row in features)
    outline_counts = Counter(row["outline_visibility"] for row in features)
    occlusion_counts = Counter(row["occlusion_level"] for row in features)
    distribution = ["# 阶段 1 分布", "", f"有效特征：{len(features)} 张。", "", "## L1 类别", ""]
    distribution += [f"- {key}: {value}" for key, value in sorted(class_counts.items())]
    distribution += ["", "## 轮廓", ""] + [f"- {k}: {v}" for k, v in sorted(outline_counts.items())]
    distribution += ["", "## 遮挡", ""] + [f"- {k}: {v}" for k, v in sorted(occlusion_counts.items())]
    atomic_text(output / "distribution.md", "\n".join(distribution) + "\n")

    covered = len(class_counts)
    missing = [name for name in CLASS_NAMES.values() if name not in class_counts]
    coverage = ["# 阶段 1 探索性覆盖", "", f"15 个 L1 类别中观察到 {covered} 类。",
                "", "未观察类别：" + (", ".join(missing) if missing else "无"), "",
                "本报告仅描述 Imagenette 100 张，不代表真实业务总体的数据充分性。", ""]
    atomic_text(output / "coverage_report.md", "\n".join(coverage))

    lines = ["# 阶段 1 一致率与验收", "", f"决策：**{report['decision']}**", "",
             "## 本地与在线", "", "| 字段 | 样本数 | 一致率 | Kappa |", "|---|---:|---:|---:|"]
    for field, value in report["agreement"].items():
        kval = "N/A" if value["kappa"] is None else f"{value['kappa']:.3f}"
        lines.append(f"| {field} | {value['n']} | {fmt(value['exact'])} | {kval} |")
    lines += ["", "## 五项门槛", ""] + [f"- [{'x' if ok else ' '}] {name}" for name, ok in report["gates"].items()]
    lines += ["", f"SAM 人工抽检：{report['sam_review']['accepted']}/{report['sam_review']['n']}。",
              f"端到端 P95：{report['latency_seconds']['p95']} 秒。", "",
              "逐图分歧见 `disagreements.csv`；人工复核结论见 `human_review.csv`。", ""]
    atomic_text(output / "agreement_report.md", "\n".join(lines))
