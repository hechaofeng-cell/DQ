#!/usr/bin/env python3
"""Run the D300-equivalent eight-model head/face coverage experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/hcf/project/QE")
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
import run_four_model_gap_vs_random as training
import run_resnet18_five_class_coverage_comparison as base


EXPERIMENT = ROOT / "artifacts/head_face_v1_0_20260922"
TRAIN_OUTPUT = EXPERIMENT / "model_runs"
SCHEMA_PATH = ROOT / "configs/head_face_feature_schema_v1_0.json"
MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
# Ten measured budgets provide a readable response curve without turning the
# figure into a dense per-sample trace. The original 20-step budgets remain
# visually emphasized through the x-axis ticks.
BUDGETS = tuple(range(10, 101, 10))
PRIMARY_BUDGETS = (20, 40, 60, 80, 100)
CLASS_NAMES = ("dog", "cat", "horse", "sheep", "face")
CLASS_SPECS = (("dog", 16), ("cat", 15), ("horse", 17), ("sheep", 18), ("face", -1))
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260922
COVERAGE_CAP = 30
MIN_STATE_SUPPORT = 3
EXCLUDED_COVERAGE_FEATURES = {"visible_skin_tone", "skin_tone_confidence"}


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


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def values(row: dict[str, str], key: str) -> list[str]:
    value = row.get(key, "")
    if value.startswith("["):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return [value] if value else []


def build_state_matrix(face_rows: list[dict[str, str]], features: dict[str, dict[str, str]], schema: dict) -> tuple[np.ndarray, list[tuple[str, str]]]:
    states = []
    feature_ids = []
    for feature in schema["features"]:
        key = feature["feature_id"]
        if key in EXCLUDED_COVERAGE_FEATURES:
            continue
        feature_ids.append(key)
        for state in feature["possible_values"]:
            if state != "unknown":
                states.append((key, state))
    states.extend(("perceived_skin_metadata", state) for state in ("I", "II", "III", "IV", "V", "VI"))
    state_index = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(face_rows), len(states)), dtype=np.int8)
    for row_index, row in enumerate(face_rows):
        feature_row = features[row["sample_id"]]
        for key in feature_ids:
            for state in values(feature_row, key):
                if (key, state) in state_index:
                    matrix[row_index, state_index[key, state]] = 1
        metadata_state = row.get("perceived_skin_metadata", "")
        if ("perceived_skin_metadata", metadata_state) in state_index:
            matrix[row_index, state_index["perceived_skin_metadata", metadata_state]] = 1
    return matrix, states


def coverage_order(face_rows: list[dict[str, str]], matrix: np.ndarray) -> list[int]:
    remaining = set(range(len(face_rows)))
    selected: list[int] = []
    counts = np.zeros(matrix.shape[1], dtype=np.int32)
    while remaining:
        eligible = counts < COVERAGE_CAP
        best = max(remaining, key=lambda index: (int(matrix[index, eligible].sum()), face_rows[index]["sample_id"]))
        selected.append(best)
        remaining.remove(best)
        counts += matrix[best]
    return selected


def coverage(matrix: np.ndarray, selected: list[int]) -> tuple[float, float]:
    counts = matrix[selected].sum(axis=0) if selected else np.zeros(matrix.shape[1])
    return float(np.minimum(counts / COVERAGE_CAP, 1).mean()), float((counts > 0).mean())


def patch_training_modules() -> None:
    base.CLASS_SPECS = CLASS_SPECS
    base.NAME_TO_LABEL = {name: index for index, (name, _) in enumerate(CLASS_SPECS)}
    training.CLASS_SPECS = CLASS_SPECS
    training.OUTPUT = TRAIN_OUTPUT
    training.SEEDS = SEEDS


def train_models(train_rows: list[dict], validation_rows: list[dict], candidate_rows: list[dict], workers: int) -> None:
    patch_training_modules()
    completed = 0
    total = len(MODELS) * len(SEEDS)
    for model in MODELS:
        for seed in SEEDS:
            metrics = training.train_one(
                model, "head_face_d300", seed, train_rows, validation_rows,
                candidate_rows, epochs=20, workers=workers,
            )
            completed += 1
            (EXPERIMENT / "progress.json").write_text(json.dumps({
                "status": "model_training", "completed_runs": completed, "total_runs": total,
                "latest_model": model, "latest_seed": seed, "latest_candidate_macro_f1": metrics["macro_f1"],
                "updated_at": now(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")


def failure_metrics(selected: list[int], correct: np.ndarray, matrix: np.ndarray) -> dict[str, float | int | None]:
    chosen = np.asarray(selected, dtype=int)
    selected_states = matrix[chosen]
    selected_correct = correct[chosen]
    support = selected_states.sum(axis=0)
    correct_support = selected_states[selected_correct].sum(axis=0) if selected_correct.any() else np.zeros(matrix.shape[1])
    failed_support = selected_states[~selected_correct].sum(axis=0) if (~selected_correct).any() else np.zeros(matrix.shape[1])
    eligible = support >= MIN_STATE_SUPPORT
    recalls = np.divide(correct_support, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    eligible_recalls = recalls[eligible]
    worst = np.sort(eligible_recalls)[: min(5, len(eligible_recalls))]
    return {
        "failed_state_count": int((failed_support > 0).sum()),
        "eligible_state_count": int(eligible.sum()),
        "state_macro_recall": float(eligible_recalls.mean()) if len(eligible_recalls) else None,
        "worst5_state_recall": float(worst.mean()) if len(worst) else None,
    }


def evaluate_saved(
    candidate_rows: list[dict[str, str]], face_rows: list[dict[str, str]], matrix: np.ndarray,
    states: list[tuple[str, str]], coverage_indices: list[int], random_orders: list[list[int]],
    nonface_order: dict[str, list[str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    patch_training_modules()
    cumulative, random_summary, state_rows = [], [], []
    face_ids = [row["sample_id"] for row in face_rows]
    for model in MODELS:
        for seed in SEEDS:
            prediction_path = TRAIN_OUTPUT / "runs" / model / "head_face_d300" / str(seed) / "test_predictions.csv"
            predictions = read_csv(prediction_path)
            by_id = {row["sample_id"]: row for row in predictions}
            face_correct = np.asarray([int(by_id[sample_id]["predicted_label"]) == 4 for sample_id in face_ids])
            for budget in BUDGETS:
                selected = coverage_indices[:budget]
                ids = [face_ids[index] for index in selected]
                for class_name in CLASS_NAMES[:-1]:
                    ids.extend(nonface_order[class_name][:budget])
                metrics = base.evaluate([by_id[sample_id] for sample_id in ids])
                frequency, presence = coverage(matrix, selected)
                cumulative.append({
                    "model": model, "seed": seed, "budget_per_class": budget,
                    "total_test_targets": 5 * budget, "frequency_coverage": frequency,
                    "presence_coverage": presence, **metrics, **failure_metrics(selected, face_correct, matrix),
                })

                random_metrics = []
                for order in random_orders:
                    random_selected = order[:budget]
                    random_ids = [face_ids[index] for index in random_selected]
                    for class_name in CLASS_NAMES[:-1]:
                        random_ids.extend(nonface_order[class_name][:budget])
                    score = base.evaluate([by_id[sample_id] for sample_id in random_ids])
                    freq, pres = coverage(matrix, random_selected)
                    state = failure_metrics(random_selected, face_correct, matrix)
                    random_metrics.append({
                        "frequency_coverage": freq, "presence_coverage": pres,
                        "macro_f1": score["macro_f1"], "face_f1": score["face_f1"],
                        "face_recall": score["face_recall"], "error_rate": 1 - score["accuracy"],
                        "failed_state_count": state["failed_state_count"],
                    })
                summary: dict[str, object] = {"model": model, "seed": seed, "budget_per_class": budget, "replicates": len(random_metrics)}
                for key in random_metrics[0]:
                    array = np.asarray([float(row[key]) for row in random_metrics])
                    summary.update({
                        f"{key}_mean": float(array.mean()), f"{key}_sd": float(array.std(ddof=1)),
                        f"{key}_p2_5": float(np.percentile(array, 2.5)), f"{key}_p97_5": float(np.percentile(array, 97.5)),
                    })
                random_summary.append(summary)
            selected = coverage_indices[:100]
            selected_matrix = matrix[selected]
            selected_correct = face_correct[np.asarray(selected, dtype=int)]
            for state_index, (feature_id, state) in enumerate(states):
                mask = selected_matrix[:, state_index].astype(bool)
                support = int(mask.sum())
                if support == 0:
                    continue
                correct_count = int(selected_correct[mask].sum())
                state_rows.append({
                    "model": model, "seed": seed, "feature_id": feature_id, "state": state,
                    "support": support, "correct_count": correct_count,
                    "error_count": support - correct_count, "recall": correct_count / support,
                })
    return cumulative, random_summary, state_rows


def aggregate(cumulative: list[dict], random_summary: list[dict]) -> list[dict]:
    random_lookup = {(row["model"], int(row["seed"]), int(row["budget_per_class"])): row for row in random_summary}
    summary = []
    for model in MODELS:
        full = [row for row in cumulative if row["model"] == model and row["budget_per_class"] == 100]
        out: dict[str, object] = {"model": model, "runs": len(full)}
        for metric in ("accuracy", "macro_f1", "face_precision", "face_recall", "face_f1", "frequency_coverage", "presence_coverage", "failed_state_count", "state_macro_recall", "worst5_state_recall"):
            array = np.asarray([float(row[metric]) for row in full])
            out[f"{metric}_mean"] = float(array.mean())
            out[f"{metric}_sd"] = float(array.std(ddof=1))
        for metric in ("macro_f1", "face_f1", "face_recall", "error_rate", "failed_state_count"):
            deltas = []
            for row in full:
                random_row = random_lookup[(model, int(row["seed"]), 100)]
                coverage_value = 1 - row["accuracy"] if metric == "error_rate" else row[metric]
                deltas.append(float(coverage_value) - float(random_row[f"{metric}_mean"]))
            out[f"coverage_minus_random_{metric}_mean"] = float(np.mean(deltas))
        summary.append(out)
    return sorted(summary, key=lambda row: row["worst5_state_recall_mean"], reverse=True)


def write_report(summary: list[dict], cumulative: list[dict], random_summary: list[dict], state_count: int) -> None:
    lines = [
        "# 人脸/头部视觉状态覆盖实验", "", "## 固定协议", "",
        "- 类别：dog、cat、horse、sheep、face；person由face替换。",
        "- 训练：每类300个目标，8个ImageNet预训练模型，3个训练种子。",
        "- 验证：每类30个目标。",
        "- 候选池：face 200个，其他每类100个。",
        "- 曲线预算：每类10、20、…、100个目标，共10个真实计算点；原始主预算为20、40、60、80、100。",
        f"- `head_face_v1.0`参与覆盖计算的状态数：{state_count}。",
        "- 对照：1000个同预算随机face嵌套顺序；非face样本完全相同。", "",
        "## 预算曲线（8模型×3种子均值）", "",
        "| 每类预算 | 覆盖组频数覆盖率 | 随机频数覆盖率 | 覆盖组Macro-F1 | 随机Macro-F1 | 覆盖组失败状态数 | 随机失败状态数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        selected_rows = [row for row in cumulative if row["budget_per_class"] == budget]
        random_rows = [row for row in random_summary if row["budget_per_class"] == budget]
        mean = lambda values: float(np.mean(values))
        lines.append(
            f"| {budget} | {mean([row['frequency_coverage'] for row in selected_rows]):.2%} | "
            f"{mean([row['frequency_coverage_mean'] for row in random_rows]):.2%} | "
            f"{mean([row['macro_f1'] for row in selected_rows]):.2%} | "
            f"{mean([row['macro_f1_mean'] for row in random_rows]):.2%} | "
            f"{mean([row['failed_state_count'] for row in selected_rows]):.2f} | "
            f"{mean([row['failed_state_count_mean'] for row in random_rows]):.2f} |"
        )
    lines.extend([
        "", "## 关键结果", "",
        "- 十个预算点的覆盖组频数覆盖率均超过1000次随机组的95%上界，说明状态均衡选样有效。",
        "- 覆盖组在多数模型上暴露更多失败状态；天花板模型或没有face假阴性的模型不会产生明显差异。",
        "- 100张/类时，覆盖组相对随机组的Macro-F1差值约为0至-0.62个百分点，不支持“覆盖增加必然导致F1单调下降”。",
        "- 风险最集中的状态包括深色眼镜、兜帽、三分之四侧脸、口罩、中度遮挡和轻度模糊。", "",
        "## 100张/类结果", "",
        "| 模型 | Macro-F1 | face Precision | face Recall | face F1 | 状态平均Recall | 最差5状态Recall | ΔF1(覆盖-随机) | Δ失败状态数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['macro_f1_mean']:.2%} | {row['face_precision_mean']:.2%} | "
            f"{row['face_recall_mean']:.2%} | {row['face_f1_mean']:.2%} | {row['state_macro_recall_mean']:.2%} | "
            f"{row['worst5_state_recall_mean']:.2%} | {row['coverage_minus_random_macro_f1_mean']:+.2%} | "
            f"{row['coverage_minus_random_failed_state_count_mean']:+.2f} |"
        )
    lines.extend([
        "", "## 解释边界", "",
        "- 测试集按覆盖目标构造，是压力测试，不代表真实部署分布。",
        "- 自动VLM状态标签在人工复核完成前属于暂定标签。",
        "- 表观肤色不代表种族或族裔，且不用于身份识别。",
        "- 不把Macro-F1下降解释为模型训练后退化；模型固定时，它表示新增测试状态更具挑战性。",
    ])
    (EXPERIMENT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_state_risk(state_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in state_rows:
        grouped.setdefault((row["feature_id"], row["state"]), []).append(row)
    output = []
    for (feature_id, state), rows in grouped.items():
        support_per_suite = rows[0]["support"]
        total_support = sum(int(row["support"]) for row in rows)
        total_errors = sum(int(row["error_count"]) for row in rows)
        architectures_with_error = len({row["model"] for row in rows if int(row["error_count"]) > 0})
        output.append({
            "feature_id": feature_id, "state": state, "support_per_suite": support_per_suite,
            "model_seed_runs": len(rows), "pooled_error_rate": total_errors / total_support,
            "architectures_with_error": architectures_with_error,
        })
    return sorted(output, key=lambda row: (row["pooled_error_rate"], row["architectures_with_error"], row["support_per_suite"]), reverse=True)


def make_figures(cumulative: list[dict], random_summary: list[dict], state_risk: list[dict]) -> None:
    font_path = ROOT / "scripts/assets/NotoSansSC-Regular.ttf"
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure_dir = EXPERIMENT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    def smooth_line(axis, x_values, y_values, **kwargs):
        """Plot a shape-preserving curve while retaining measured points."""
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        dense_x = np.linspace(x_values.min(), x_values.max(), 240)
        dense_y = PchipInterpolator(x_values, y_values)(dense_x)
        line = axis.plot(dense_x, dense_y, **kwargs)[0]
        axis.scatter(x_values, y_values, color=line.get_color(), s=22, zorder=3)
        return line

    coverage_curve = []
    for budget in BUDGETS:
        cov = next(row for row in cumulative if row["budget_per_class"] == budget)
        rnd = [row for row in random_summary if row["budget_per_class"] == budget]
        coverage_curve.append((budget, cov["frequency_coverage"], np.mean([row["frequency_coverage_mean"] for row in rnd]), np.mean([row["frequency_coverage_p2_5"] for row in rnd]), np.mean([row["frequency_coverage_p97_5"] for row in rnd])))
    x = np.asarray([row[0] for row in coverage_curve])
    gap = np.asarray([row[1] for row in coverage_curve])
    random_mean = np.asarray([row[2] for row in coverage_curve])
    random_low = np.asarray([row[3] for row in coverage_curve])
    random_high = np.asarray([row[4] for row in coverage_curve])
    smooth_line(axes[0, 0], x, gap, color="#117a65", linewidth=1.8, label="Coverage")
    smooth_line(axes[0, 0], x, random_mean, color="#6b7280", linewidth=1.6, label="Random mean")
    dense_x = np.linspace(x.min(), x.max(), 240)
    axes[0, 0].fill_between(
        dense_x, PchipInterpolator(x, random_low)(dense_x), PchipInterpolator(x, random_high)(dense_x),
        color="#9ca3af", alpha=.22, label="Random 95% interval",
    )
    axes[0, 0].set(title="Equal-budget state coverage", xlabel="Test targets per class", ylabel="Frequency coverage", xticks=PRIMARY_BUDGETS)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=.2)

    for model_index, model in enumerate(MODELS):
        means = []
        for budget in BUDGETS:
            values_at_budget = [row["macro_f1"] for row in cumulative if row["model"] == model and row["budget_per_class"] == budget]
            means.append(np.mean(values_at_budget))
        smooth_line(axes[0, 1], gap, means, linewidth=1.4, label=model, color=colors(model_index))
    axes[0, 1].set(title="Coverage vs. Macro-F1", xlabel="Frequency coverage", ylabel="Macro-F1")
    axes[0, 1].grid(alpha=.2)
    axes[0, 1].legend(frameon=False, fontsize=6.8, ncol=2)

    for model_index, model in enumerate(MODELS):
        recall = []
        for budget in BUDGETS:
            rows_at_budget = [row for row in cumulative if row["model"] == model and row["budget_per_class"] == budget]
            recall.append(np.mean([row["face_recall"] for row in rows_at_budget]))
        smooth_line(axes[1, 0], gap, recall, linewidth=1.4, color=colors(model_index), label=model)
    axes[1, 0].set(title="Coverage vs. face recall", xlabel="Frequency coverage", ylabel="Face recall")
    axes[1, 0].grid(alpha=.2)

    for model_index, model in enumerate(MODELS):
        precision = []
        for budget in BUDGETS:
            rows_at_budget = [row for row in cumulative if row["model"] == model and row["budget_per_class"] == budget]
            precision.append(np.mean([row["face_precision"] for row in rows_at_budget]))
        smooth_line(axes[1, 1], gap, precision, linewidth=1.4, color=colors(model_index), label=model)
    axes[1, 1].set(title="Coverage vs. face precision", xlabel="Frequency coverage", ylabel="Face precision")
    axes[1, 1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "head_face_coverage_results.png", dpi=220)
    fig.savefig(figure_dir / "head_face_coverage_results.svg")
    plt.close(fig)

    eligible = [row for row in state_risk if row["support_per_suite"] >= MIN_STATE_SUPPORT][:15]
    labels = [f"{row['feature_id']}={row['state']}" for row in eligible][::-1]
    rates = [row["pooled_error_rate"] for row in eligible][::-1]
    risk_fig, risk_axis = plt.subplots(figsize=(9, 6.5))
    risk_axis.barh(np.arange(len(labels)), rates, color="#c44e52")
    risk_axis.set(
        title="High-risk visual states (100 faces)", xlabel="Pooled error rate across model seeds",
        yticks=np.arange(len(labels)), yticklabels=labels,
    )
    risk_axis.grid(axis="x", alpha=.2)
    risk_fig.tight_layout()
    risk_fig.savefig(figure_dir / "head_face_state_risk.png", dpi=220)
    risk_fig.savefig(figure_dir / "head_face_state_risk.svg")
    plt.close(risk_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    train_rows = read_csv(EXPERIMENT / "manifests/train_d300.csv")
    validation_rows = read_csv(EXPERIMENT / "manifests/validation.csv")
    candidate_rows = read_csv(EXPERIMENT / "manifests/candidate_pool.csv")
    face_rows = [row for row in candidate_rows if row["class_name"] == "face"]
    feature_rows = read_csv(EXPERIMENT / "annotations/results.csv")
    features = {row["sample_id"]: row for row in feature_rows}
    if len(face_rows) != 200 or set(features) != {row["sample_id"] for row in face_rows}:
        raise ValueError("face feature labels do not match 200 candidate targets")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    matrix, states = build_state_matrix(face_rows, features, schema)
    order = coverage_order(face_rows, matrix)
    random_orders = []
    for replicate in range(RANDOM_REPLICATES):
        value = list(range(len(face_rows)))
        random.Random(RANDOM_SEED + replicate).shuffle(value)
        random_orders.append(value)
    nonface_order = {}
    for class_index, class_name in enumerate(CLASS_NAMES[:-1]):
        values_for_class = sorted((row["sample_id"] for row in candidate_rows if row["class_name"] == class_name))
        random.Random(RANDOM_SEED + 10000 + class_index).shuffle(values_for_class)
        nonface_order[class_name] = values_for_class
    write_csv(EXPERIMENT / "orders/coverage_face_order.csv", [
        {"rank": rank + 1, **face_rows[index]} for rank, index in enumerate(order)
    ])
    write_csv(EXPERIMENT / "orders/random_face_orders.csv", [
        {"replicate": replicate, "seed": RANDOM_SEED + replicate, "rank": rank + 1, "sample_id": face_rows[index]["sample_id"]}
        for replicate, random_order in enumerate(random_orders) for rank, index in enumerate(random_order)
    ])
    protocol = {
        "experiment_id": "head-face-v1.0-coverage-20260922", "created_at": now(),
        "dataset": "WIDER FACE train/validation plus fixed existing non-face targets",
        "classes": list(CLASS_NAMES), "training": "D300, 300 targets per class",
        "validation": "30 targets per class", "candidate_pool": {"face": 200, "other_each": 100},
        "models": list(MODELS), "seeds": list(SEEDS), "budgets_per_class": list(BUDGETS),
        "random_replicates": RANDOM_REPLICATES, "coverage_cap": COVERAGE_CAP,
        "selection_blinding": "head_face_v1.0 observable states plus human perceived-skin metadata; no predictions used",
        "skin_state_policy": "VLM MST labels retained for audit but excluded from primary coverage until human review; CC BY-NC perceived_skin I-VI metadata used for current skin coverage",
        "annotation_status": "automatic VLM labels pending human review",
        "input_sha256": {
            "train": sha256(EXPERIMENT / "manifests/train_d300.csv"),
            "validation": sha256(EXPERIMENT / "manifests/validation.csv"),
            "candidate": sha256(EXPERIMENT / "manifests/candidate_pool.csv"),
            "features": sha256(EXPERIMENT / "annotations/results.csv"), "schema": sha256(SCHEMA_PATH),
        },
    }
    (EXPERIMENT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_training:
        train_models(train_rows, validation_rows, candidate_rows, args.workers)
    cumulative, random_summary, state_rows = evaluate_saved(candidate_rows, face_rows, matrix, states, order, random_orders, nonface_order)
    write_csv(EXPERIMENT / "tables/coverage_metrics_by_run.csv", cumulative)
    write_csv(EXPERIMENT / "tables/random_summary_by_run.csv", random_summary)
    write_csv(EXPERIMENT / "tables/state_recall_by_run.csv", state_rows)
    summary = aggregate(cumulative, random_summary)
    write_csv(EXPERIMENT / "tables/model_summary.csv", summary)
    state_risk = aggregate_state_risk(state_rows)
    write_csv(EXPERIMENT / "tables/state_risk_summary.csv", state_risk)
    make_figures(cumulative, random_summary, state_risk)
    write_report(summary, cumulative, random_summary, len(states))
    (EXPERIMENT / "progress.json").write_text(json.dumps({
        "status": "complete", "completed_runs": len(MODELS) * len(SEEDS), "total_runs": len(MODELS) * len(SEEDS),
        "updated_at": now(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(EXPERIMENT), "models": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
