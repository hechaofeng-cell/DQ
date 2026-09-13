#!/usr/bin/env python3
"""Build a non-overwriting, self-contained release from the fixed human audit."""
import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.io import atomic_csv, atomic_json, atomic_text, digest, read_csv, read_json
from scripts.dog_supplement_review_server import Store, EXPERIMENT
from scripts.run_dog_targeted_supplement import latest_feature_payload, scenario_counts
from scripts.run_dog_local_pipeline import program_fields

DEFAULT = ROOT / "artifacts/dog594_release_v1_20260913"
VERSION = "dog594-reviewed-release-v1"
RETAIN_UNKNOWN = "000000030731"


def verify(output):
    marker = read_json(output / "FROZEN.json")
    checksums = read_csv(output / "checksums.csv")
    assert digest(output / "checksums.csv") == marker["checksums_sha256"]
    for row in checksums:
        assert digest(output / row["path"]) == row["sha256"], row["path"]
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    assert actual == {r["path"] for r in checksums} | {"checksums.csv", "FROZEN.json"}
    manifest = read_csv(output / "manifest.csv")
    assert len(manifest) == len({r["image_id"] for r in manifest}) == 594
    assert len({r["image_sha256"] for r in manifest}) == 594
    features = read_csv(output / "features.csv")
    assert {r["image_id"] for r in features} == {r["image_id"] for r in manifest}
    for row in manifest:
        for key in ("image_path", "marked_image_path", "crop_image_path", "label_path"):
            assert (output / row[key]).is_file()
        assert digest(output / row["image_path"]) == row["image_sha256"]
        assert digest(output / row["label_path"]) == row["label_sha256"]
    print(json.dumps({"verified": True, "images": 594, "hashed_files": len(checksums), "output": str(output)}))


def build(output):
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    store = Store()
    reviews = {r["image_id"]: r for r in store.records()}
    assert set(reviews) == set(store.case_by_id), "review_queue_incomplete"
    rejected = {iid for iid, r in reviews.items() if r["decision"] == "reject"}
    assert len(rejected) == 6 and all(reviews[i]["target_kind"] == "representation" for i in rejected)
    uncertain = {iid for iid, r in reviews.items() if r["decision"] == "uncertain"}
    assert uncertain == {RETAIN_UNKNOWN}
    assert reviews[RETAIN_UNKNOWN]["target_kind"] == "real_dog"
    assert reviews[RETAIN_UNKNOWN]["human_features"]["coat_length"] == "unknown"
    assert len(store.manifest) == 600
    output.mkdir(parents=True)
    definitions = store.definitions
    shutil.copy2(ROOT / "configs/dog_feature_schema_v2.json", output / "dog_feature_schema_v2.json")
    shutil.copy2(Path(__file__), output / "freeze_script.py")
    for name in ("report.json", "pre_post_state_coverage.csv", "combined_manifest_600.csv", "review_queue.csv"):
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXPERIMENT / name, target)
    for iid, review in reviews.items():
        atomic_json(output / "review_snapshot" / "current" / f"{iid}.json", review)
        history = store.output / "history" / iid
        if history.is_dir():
            shutil.copytree(history, output / "review_snapshot" / "history" / iid)
    manifest, feature_rows, changes, excluded, invalid_fields = [], [], [], [], []
    baseline_counts, final_counts = Counter(), Counter()
    evidence_counts = Counter()
    strict = Counter()
    classification = Counter()
    for iid, original in sorted(store.manifest.items()):
        review = reviews.get(iid)
        if iid in rejected:
            excluded.append({"image_id": iid, "dataset_source": original["dataset_source"], "reason": "human_confirmed_nonliving_representation", "reviewer": review["reviewer"]})
            continue
        baseline = original["dataset_source"] == "baseline500"
        directory = "dog_feature_pipeline_v2_1_upscaled_20260912" if baseline else "dog_candidate400_features_v2_1_20260912"
        result_dir = ROOT / "artifacts" / directory / "results"
        payload = latest_feature_payload(result_dir, iid)
        identity_valid = payload.get("image_id") == iid and payload.get("target_instance_id") == "target_1"
        model_features = payload["features"]
        effective = dict(model_features)
        provenance = {key: "model" for key in definitions}
        for key, value in (review["human_features"] if review else {}).items():
            effective[key] = value
            provenance[key] = "human_review"
            if model_features.get(key) != value:
                changes.append({"image_id": iid, "feature_id": key, "model_value": model_features.get(key), "effective_value": value, "reviewer": review["reviewer"], "reviewed_at": review["reviewed_at"], "notes": review["notes"]})
        assert set(effective) == set(definitions)
        field_status = {}
        for key, definition in definitions.items():
            if (not identity_valid and provenance[key] != "human_review") or effective[key] not in definition["possible_values"]:
                reason = "response_identity_mismatch" if not identity_valid else "invalid_enum_not_unknown"
                invalid_fields.append({"image_id": iid, "feature_id": key, "raw_value": effective[key], "status": reason})
                effective[key] = None
                provenance[key] = "model_invalid_response"
                field_status[key] = reason
            else:
                field_status[key] = "unknown" if effective[key] == "unknown" else "valid"
        atomic_json(output / "model_features" / f"{iid}.json", payload)
        atomic_json(output / "effective_features" / f"{iid}.json", {
            "schema_version": "dog_v2", "release_version": VERSION, "image_id": iid,
            "target_instance_id": "target_1", "features": effective, "feature_provenance": provenance, "field_status": field_status,
            "model_evidence": payload.get("evidence", {}),
            "human_review": review, "evidence_note": "Model evidence is historical and may contradict corrected values; human notes are not fabricated evidence."})
        for task in ("features", "classifier"):
            for response in (result_dir / "raw" / task).glob(f"{iid}_attempt*.json"):
                destination = output / "raw" / task / response.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(response, destination)
        assert list((output / "raw/features").glob(f"{iid}_attempt*.json")), f"missing_raw:{iid}"
        row = dict(original)
        row.update({"protocol_version": VERSION, "input_protocol_version": original["protocol_version"],
                    "selection_status": "frozen_release", "quality_flag": "", "human_review_status": "unreviewed" if not review else ("reviewed_retained_unknown" if iid == RETAIN_UNKNOWN else "reviewed_retained")})
        source_label = Path(original["image_path"]).parent.parent / "labels" / f"{iid}.txt"
        for key, folder, source in [("image_path", "images", Path(original["image_path"])),
                                   ("marked_image_path", "marked", Path(original["marked_image_path"])),
                                   ("crop_image_path", "crops", Path(original["crop_image_path"])),
                                   ("label_path", "labels", source_label)]:
            destination = output / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row[key] = str(destination.relative_to(output))
        assert digest(output / row["image_path"]) == original["image_sha256"]
        assert digest(output / row["label_path"]) == original["label_sha256"]
        manifest.append(row)
        programs = program_fields(original, effective)
        classifier = result_dir / "parsed/classifier" / f"{iid}.json"
        prediction = ""
        if baseline and classifier.exists():
            classified = read_json(classifier)
            assert classified["image_id"] == iid and classified["target_instance_id"] == "target_1"
            prediction = classified["predicted_class"]
            atomic_json(output / "classification" / f"{iid}.json", classified)
            classification["available"] += 1
            classification["dog"] += prediction == "dog"
            classification["unknown"] += prediction == "unknown"
        strict["baseline" if baseline else "supplement"] += (result_dir / "parsed/features" / f"{iid}.json").exists()
        feature_rows.append({"image_id": iid, "target_instance_id": "target_1", "dataset_source": row["dataset_source"],
            "split": row["split"], "release_version": VERSION, "human_review_status": row["human_review_status"],
            "model_response_status": "strict_accepted" if (result_dir / "parsed/features" / f"{iid}.json").exists() else "parseable_raw_not_strict_accepted",
            "predicted_class": prediction, "classification_correct": prediction == "dog" if prediction else "",
            "classifier_status": "historical_baseline" if prediction else "not_evaluated_in_release",
            **programs, **effective})
        final_counts.update(effective.items())
        if baseline:
            baseline_counts.update(effective.items())
        for key in definitions:
            evidence = payload.get("evidence", {}).get(key)
            evidence_counts[key] += isinstance(evidence, str) and len(evidence.strip()) >= 8 and evidence.strip().lower() != str(model_features[key]).lower()
    assert len(manifest) == 594
    atomic_csv(output / "manifest.csv", manifest, list(manifest[0]))
    atomic_csv(output / "features.csv", feature_rows, list(feature_rows[0]))
    atomic_csv(output / "corrections.csv", changes, list(changes[0]))
    atomic_csv(output / "excluded.csv", excluded, list(excluded[0]))
    atomic_csv(output / "invalid_fields.csv", invalid_fields, ["image_id", "feature_id", "raw_value", "status"])
    distribution = []
    for key, definition in definitions.items():
        for state in definition["possible_values"]:
            n = final_counts[(key, state)]
            subgroup = [r for r in feature_rows if r[key] == state and r["predicted_class"]]
            distribution.append({"feature_id": key, "state": state, "count": n, "rate": n / 594,
                "baseline496_count": baseline_counts[(key, state)], "supplement98_count": n - baseline_counts[(key, state)],
                "baseline_classified_count": len(subgroup), "baseline_classification_errors": sum(r["predicted_class"] != "dog" for r in subgroup)})
    atomic_csv(output / "feature_state_distribution.csv", distribution, list(distribution[0]))
    old_states = read_csv(EXPERIMENT / "pre_post_state_coverage.csv")
    coverage = [{"feature_id": r["feature_id"], "state": r["state"], "historical500_count": int(r["baseline_count"]),
        "clean_baseline496_count": baseline_counts[(r["feature_id"], r["state"])],
        "release594_count": final_counts[(r["feature_id"], r["state"])],
        "frozen_target": int(r["target_count"]), "historical_gap": r["is_gap"] == "True",
        "target_met": final_counts[(r["feature_id"], r["state"])] >= int(r["target_count"])} for r in old_states]
    atomic_csv(output / "coverage.csv", coverage, list(coverage[0]))
    remaining = [r for r in coverage if r["historical_gap"] and not r["target_met"]]
    atomic_csv(output / "remaining_gaps.csv", remaining, list(coverage[0]))
    index = sum(min(r["release594_count"] / 30, 1) for r in coverage) / len(coverage)
    base_index = sum(min(r["clean_baseline496_count"] / 30, 1) for r in coverage) / len(coverage)
    scenarios = scenario_counts(feature_rows)
    scenario_index = sum(min(n / 30, 1) for n in scenarios.values()) / len(scenarios)
    strict_rate = sum(strict.values()) / 594
    evidence_rate = sum(evidence_counts.values()) / (594 * len(definitions))
    enum_validity = 1 - len(invalid_fields) / (594 * len(definitions))
    annotation_score = 20 * (1 + strict_rate + enum_validity + evidence_rate) / 4
    score = 20 + annotation_score + 35 * index + 15 * scenario_index + 10
    agreement = {}
    for group in ("critical", "random"):
        pairs = [(r["model_features"][k], v) for r in reviews.values() if r["group"] == group and r["decision"] == "accept" for k, v in r["human_features"].items()]
        agreement[group] = {"matched": sum(a == b for a, b in pairs), "reviewed_fields": len(pairs), "rate": sum(a == b for a, b in pairs) / len(pairs)}
    report = {"release_version": VERSION, "status": "frozen", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "images": 594, "baseline_images": 496, "supplement_images": 98, "excluded_images": 6,
        "reviewed_retained_images": 42, "unreviewed_images": 552, "reviewed_total_including_excluded": 48,
        "retained_uncertain_case": {"image_id": RETAIN_UNKNOWN, "resolution": "retain_real_dog_with_coat_length_unknown", "authority": "user_requested_594_release"},
        "corrected_fields": len(changes), "invalid_fields_by_reason": dict(Counter(r["status"] for r in invalid_fields)), "coverage": {"evaluable_states": len(coverage), "clean_baseline_index": base_index,
            "release_index": index, "gain_percentage_points": 100 * (index - base_index), "historical_gaps": 26,
            "resolved_historical_gaps": 26 - len(remaining), "remaining_historical_gaps": len(remaining),
            "formula": "mean(min(state_count/30,1)); exclude unknown/other; retain historical gap targets"},
        "features": [{"feature_id": key, "unknown_count": final_counts[(key, "unknown")], "unknown_rate": final_counts[(key, "unknown")] / 594,
            "model_evidence_format_valid_count": evidence_counts[key]} for key in definitions],
        "annotation": {"model_strict_accepted_count": sum(strict.values()), "model_strict_response_rate": strict_rate,
            "effective_usable_field_rate": enum_validity, "invalid_fields": len(invalid_fields), "mean_model_evidence_format_rate": evidence_rate,
            "human_model_agreement_accepted_samples": agreement, "all_fields_human_verified": False},
        "challenge_scenarios": scenarios, "readiness_score": round(score, 1),
        "score_components": {"input_integrity": 20, "annotation_format": annotation_score, "state_coverage": 35 * index,
            "challenge_scenarios": 15 * scenario_index, "lineage_reproducibility": 10}, "score_is_not_accuracy": True,
        "classification": {**dict(classification), "baseline_images": 496,
            "baseline_dog_positive_recall": classification["dog"] / 496, "supplement_evaluated": False,
            "release594_recall": None, "note": "Only historical baseline blind classifications reused; no new classifier inference."},
        "limitations": ["552 images are not human reviewed; reviewed images have only selected fields checked.",
            "Visibility scores remain null: enum occlusion/truncation cannot establish numeric ratios.",
            "Model evidence is unchanged historical evidence, not evidence for human-corrected labels.",
            "Original splits are provenance only, not an untouched holdout after dataset analysis.",
            "No claim of improved recognition performance, universal recognition, or complete coverage."],
        "paths_relative_to": "release_root"}
    atomic_json(output / "report.json", report)
    atomic_text(output / "report.md", f"# 犬类594张正式版 v1\n\n状态：冻结（数据构建版本，不代表全部标签已人工验证）。\n\n"
        f"- 数量：594张＝496张基线＋98张补图；排除6张犬形表现物。\n"
        f"- 应用人工修正：{len(changes)}个字段；保留42张已复核样本，552张未复核。\n"
        f"- 非法或响应ID不匹配字段：{len(invalid_fields)}个，正式值留空，invalid_fields.csv记录原值和原因；不自动转成unknown。\n"
        f"- 覆盖指数：清理基线 {100*base_index:.2f}% → 正式版 {100*index:.2f}%；提升 {100*(index-base_index):.2f} 个百分点。\n"
        f"- 原26个缺口关闭 {26-len(remaining)} 个，剩余 {len(remaining)} 个。\n"
        f"- 自定义就绪度：{score:.1f}，不是识别准确率。\n"
        f"- 原模型严格响应率：{100*strict_rate:.2f}%；人工修正不改变原响应是否严格成功。\n"
        f"- 基线496张历史dog召回：{100*classification['dog']/496:.2f}%；新增98张未运行盲分类，594张整体召回未知。\n\n"
        "## 冻结说明\n\n毛长无法确认的000000030731按用户决定保留，coat_length=unknown；不计作可确定毛长的覆盖。\n"
        "全部20字段保存来源。模型证据保留原文，不能当成人工修正后的证据。\n"
        "覆盖分布不等于识别准确率；未全面人工验证，不声称模型性能提升。\n")
    atomic_text(output / "README.md", "# dog594 release v1\n\n"
        "所有manifest路径相对于本目录。images/marked/crops/labels包含三种输入和COCO YOLO标签。\n"
        "features.csv是有效特征表，effective_features保留逐字段来源及人工审核；model_features和raw保留原模型结果。\n"
        "corrections.csv记录修正，excluded.csv记录排除，review_snapshot保留审核历史。\n"
        "invalid_fields.csv记录字典外值；有效JSON中留null、CSV中留空，与unknown明确区分。\n"
        "dog_feature_schema_v2.json定义不变；本版改变的是样本清单与人工修正层。\n"
        "classification只复用原基线，新增样本不标为分类通过。visibility_score为空，不从等级伪造比例。\n"
        "冻结后不得原地修改；后续变化创建新版本。该目录保留原图来源及标签，不授予额外图片再分发许可。\n\n"
        "校验命令：\n```bash\n/home/hcf/project/QE/.venv/bin/python /home/hcf/project/QE/scripts/freeze_dog594_release.py --verify --output " + str(output) + "\n```\n")
    paths = sorted(p for p in output.rglob("*") if p.is_file())
    atomic_csv(output / "checksums.csv", [{"path": str(p.relative_to(output)), "sha256": digest(p), "bytes": p.stat().st_size} for p in paths], ["path", "sha256", "bytes"])
    atomic_json(output / "FROZEN.json", {"release_version": VERSION, "status": "frozen", "images": 594,
        "frozen_at": report["frozen_at"], "checksums_sha256": digest(output / "checksums.csv"), "manifest_sha256": digest(output / "manifest.csv")})
    verify(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (verify if args.verify else build)(args.output.resolve())
