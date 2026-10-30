#!/usr/bin/env python3
"""Build an aggregate failure taxonomy and representative-image atlas."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from matplotlib import font_manager

from run_multiclass_object_selection_benchmark import CLASSES, LABELS, MODELS, SEEDS


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts/multiclass_object_pool_v1_20260928"
RUNS = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928/runs"
DEFAULT_OUTPUT = ROOT / "artifacts/multiclass_failure_atlas_20260929"
RUN_COUNT = len(MODELS) * len(SEEDS)
FONT_PATH = ROOT / "scripts/assets/NotoSansSC-Regular.ttf"

if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False

GROUPS = {
    "small_target": "小目标",
    "limited_visibility": "关键部位不可见",
    "occlusion": "中重度遮挡",
    "truncation": "中重度截断",
    "unusual_pose": "动态或非常规姿态",
    "adverse_context_quality": "复杂场景或成像质量",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_values(value: str) -> list[str]:
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return [value]


def member(group: str, row: dict[str, str], feature: dict[str, str]) -> bool:
    area = float(row["bbox_area_ratio"])
    if group == "small_target":
        return area < 0.03
    if group == "limited_visibility":
        return feature["body_visibility"] in {"partial", "minimal"} or feature["head_visibility"] in {"partial", "not_visible"}
    if group == "occlusion":
        return feature["occlusion_level"] in {"moderate", "heavy"}
    if group == "truncation":
        return feature["truncation_level"] in {"moderate", "heavy"}
    if group == "unusual_pose":
        return (
            feature["pose_action"] in {"running", "jumping", "riding_or_being_ridden", "interacting", "other"}
            or feature["body_orientation"] == "overhead_or_unusual"
        )
    if group == "adverse_context_quality":
        return (
            feature["background_complexity"] == "complex"
            or feature["blur_level"] in {"mild", "severe"}
            or feature["illumination"] != "normal"
            or feature["target_background_contrast"] == "low"
            or feature["visual_representation"] != "natural_photo"
        )
    raise KeyError(group)


def state_pairs(feature: dict[str, str], area: float) -> list[tuple[str, str]]:
    if area < 0.01:
        scale = "tiny"
    elif area < 0.03:
        scale = "small"
    elif area < 0.10:
        scale = "medium"
    else:
        scale = "large"
    output = [("target_scale", scale)]
    excluded = {
        "image_id", "target_instance_id", "target_class", "split", "predicted_class",
        "classification_correct", "feature_status", "classifier_status", "protocol_version",
        "input_protocol_version", "target_bbox", "target_area_ratio", "target_center_x",
        "target_center_y", "image_width", "image_height", "visible_dog_count_in_scene",
        "truncation_level_from_bbox", "feature_unknown_count", "visibility_score",
        "candidate_difficulty_factors",
    }
    for key, value in feature.items():
        if key in excluded:
            continue
        for item in parse_values(value):
            if item and item != "unknown":
                output.append((key, item))
    return output


def make_group_plot(rows: list[dict[str, object]], output: Path) -> None:
    labels = [str(row["group_zh"]) for row in rows]
    values = [float(row["error_rate"]) * 100 for row in rows]
    colors = ["#C84A3A" if row["group_id"] != "all_candidates" else "#3B6EA8" for row in rows]
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1])
    axis.set_xlabel("聚合分类错误率 (%)")
    axis.set_xlim(0, max(values) * 1.18)
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values[::-1]):
        axis.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)
    axis.set_title("不同视觉挑战组的聚合错误率（8模型 × 3种子）")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_confusion_plot(matrix: np.ndarray, output: Path) -> None:
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_title("候选池聚合混淆矩阵")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            value = normalized[row, column] * 100
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color="white" if value > 50 else "black", fontsize=9)
    fig.colorbar(image, ax=axis, label="行归一化比例")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_atlas(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.2))
    for axis, row in zip(axes.flat, rows):
        with Image.open(str(row["crop_path"])) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.axis("off")
        axis.set_title(
            f"{row['group_zh']} | {row['class_name']}\n"
            f"错误 {row['failure_count']}/{RUN_COUNT}，常见误判 {row['top_wrong_class']}",
            fontsize=9,
        )
    for axis in axes.flat[len(rows):]:
        axis.axis("off")
    fig.suptitle("典型失败案例图谱：每类挑战选择两个跨模型高频错误样本", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(POOL / "manifests/candidate_pool.csv")
    feature_rows = read_csv(POOL / "annotations/results.csv")
    features = {row["image_id"]: row for row in feature_rows}
    sample_ids = [row["sample_id"] for row in rows]
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    truths = np.asarray([LABELS[row["class_name"]] for row in rows], dtype=np.int64)
    predictions = []
    for model in MODELS:
        for seed in SEEDS:
            stored = np.load(RUNS / model / str(seed) / "candidate_outputs.npz", allow_pickle=True)
            order = {str(sample_id): index for index, sample_id in enumerate(stored["sample_ids"])}
            logits = stored["logits"][[order[sample_id] for sample_id in sample_ids]]
            predictions.append(np.argmax(logits, axis=1))
    prediction_matrix = np.asarray(predictions, dtype=np.int64)
    failures = prediction_matrix != truths[None, :]
    failure_count = failures.sum(axis=0)

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for run_predictions in prediction_matrix:
        np.add.at(confusion, (truths, run_predictions), 1)
    make_confusion_plot(confusion, output / "figures/aggregate_confusion_matrix.png")

    group_rows: list[dict[str, object]] = []
    group_indices: dict[str, list[int]] = {}
    all_groups = {"all_candidates": "全部候选"} | GROUPS
    for group_id, group_zh in all_groups.items():
        if group_id == "all_candidates":
            indices = list(range(len(rows)))
        else:
            indices = [
                index for index, row in enumerate(rows)
                if member(group_id, row, features[row["sample_id"]])
            ]
        group_indices[group_id] = indices
        per_class = Counter(rows[index]["class_name"] for index in indices)
        group_rows.append({
            "group_id": group_id,
            "group_zh": group_zh,
            "samples": len(indices),
            "decisions": len(indices) * RUN_COUNT,
            "errors": int(failures[:, indices].sum()),
            "error_rate": float(failures[:, indices].mean()),
            **{f"{class_name}_samples": per_class[class_name] for class_name in CLASSES},
        })
    write_csv(output / "tables/challenge_group_error_rates.csv", group_rows)
    make_group_plot(group_rows, output / "figures/challenge_group_error_rates.png")

    state_counts: Counter[tuple[str, str]] = Counter()
    state_errors: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows):
        for state in state_pairs(features[row["sample_id"]], float(row["bbox_area_ratio"])):
            state_counts[state] += RUN_COUNT
            state_errors[state] += int(failure_count[index])
    state_rows = [{
        "feature": state[0], "state": state[1],
        "sample_support": state_counts[state] // RUN_COUNT,
        "decisions": state_counts[state], "errors": state_errors[state],
        "error_rate": state_errors[state] / state_counts[state],
    } for state in state_counts if state_counts[state] // RUN_COUNT >= 20]
    state_rows.sort(key=lambda row: (-float(row["error_rate"]), -int(row["sample_support"]), str(row["feature"]), str(row["state"])))
    write_csv(output / "tables/state_error_rates_support20.csv", state_rows)

    atlas_rows: list[dict[str, object]] = []
    used: set[int] = set()
    for group_id, group_zh in GROUPS.items():
        candidates = sorted(
            group_indices[group_id],
            key=lambda index: (-int(failure_count[index]), rows[index]["sample_id"]),
        )
        chosen = []
        for index in candidates:
            if index not in used:
                chosen.append(index)
                used.add(index)
            if len(chosen) == 2:
                break
        for index in chosen:
            wrong = Counter(
                int(prediction_matrix[run, index])
                for run in range(RUN_COUNT) if failures[run, index]
            )
            top_wrong = CLASSES[wrong.most_common(1)[0][0]] if wrong else "none"
            feature = features[rows[index]["sample_id"]]
            atlas_rows.append({
                "group_id": group_id, "group_zh": group_zh,
                "sample_id": rows[index]["sample_id"], "class_name": rows[index]["class_name"],
                "crop_path": rows[index]["crop_path"],
                "failure_count": int(failure_count[index]),
                "failure_rate": float(failure_count[index] / RUN_COUNT),
                "top_wrong_class": top_wrong,
                "top_wrong_count": wrong.most_common(1)[0][1] if wrong else 0,
                "target_scale": "tiny" if float(rows[index]["bbox_area_ratio"]) < .01 else "small" if float(rows[index]["bbox_area_ratio"]) < .03 else "medium" if float(rows[index]["bbox_area_ratio"]) < .1 else "large",
                "body_visibility": feature["body_visibility"],
                "occlusion_level": feature["occlusion_level"],
                "truncation_level": feature["truncation_level"],
                "pose_action": feature["pose_action"],
                "background_complexity": feature["background_complexity"],
                "blur_level": feature["blur_level"],
                "illumination": feature["illumination"],
                "visual_representation": feature["visual_representation"],
            })
    write_csv(output / "tables/representative_failures.csv", atlas_rows)
    make_atlas(atlas_rows, output / "figures/failure_case_atlas.png")

    confusion_rows = []
    for true_index, true_class in enumerate(CLASSES):
        for predicted_index, predicted_class in enumerate(CLASSES):
            confusion_rows.append({
                "true_class": true_class, "predicted_class": predicted_class,
                "count": int(confusion[true_index, predicted_index]),
                "row_rate": float(confusion[true_index, predicted_index] / confusion[true_index].sum()),
            })
    write_csv(output / "tables/aggregate_confusion_matrix.csv", confusion_rows)

    top_states = state_rows[:12]
    summary = {
        "analysis": "multiclass-failure-atlas-v1", "created_at": now(),
        "models": list(MODELS), "seeds": list(SEEDS), "runs": RUN_COUNT,
        "classes": list(CLASSES), "candidate_count": len(rows),
        "overall_error_rate": float(failures.mean()),
        "challenge_groups": group_rows,
        "top_supported_failure_states": top_states,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    challenge_lines = "\n".join(
        f"| {row['group_zh']} | {row['samples']} | {float(row['error_rate']) * 100:.2f}% |"
        for row in group_rows
    )
    state_lines = "\n".join(
        f"| `{row['feature']}={row['state']}` | {row['sample_support']} | {float(row['error_rate']) * 100:.2f}% |"
        for row in top_states
    )
    report = f"""# 多类别失败模式图谱

基于固定候选池的 {len(rows)} 张图像，聚合 {len(MODELS)} 个模型、{len(SEEDS)} 个训练种子的 {RUN_COUNT} 组预测。整体分类错误率为 {failures.mean() * 100:.2f}%。同一图像可同时属于多个挑战组，因此各组不是互斥分层。

## 六类视觉挑战

| 挑战组 | 样本数 | 聚合错误率 |
|---|---:|---:|
{challenge_lines}

## 高错误率状态（支持度至少20张）

| 状态 | 支持度 | 聚合错误率 |
|---|---:|---:|
{state_lines}

代表样本见 `figures/failure_case_atlas.png`，完整状态结果、挑战组结果与混淆矩阵见 `tables/`。这些结果描述当前五类候选池和冻结分类器上的关联，不应解释为单一视觉因素的因果效应。
"""
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
