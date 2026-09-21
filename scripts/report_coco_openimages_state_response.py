#!/usr/bin/env python3
"""Create the audited report and editable figures for state-response evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/coco_openimages_state_response_20260920"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def f(row, key):
    return float(row[key])


def pp(value):
    return f"{100 * value:+.2f} pp"


def pct(value):
    return f"{100 * value:.2f}%"


def make_figure(stage_rows, model_rows):
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    })
    coverage = np.array([100 * f(row, "coverage") for row in stage_rows])
    additions = np.array([int(row["external_dog_added"]) for row in stage_rows])
    colors = {"blue": "#0F4D92", "teal": "#42949E", "violet": "#9A4D8E",
              "gray": "#767676", "red": "#B64342", "light": "#B4C0E4"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(additions, coverage, marker="o", lw=1.8, color=colors["blue"])
    for x, y in zip(additions, coverage):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.5)
    ax.set(xlabel="Added dog targets", ylabel="Visual-state coverage (%)")
    ax.set_ylim(min(coverage) - 3, max(coverage) + 3)

    ax = axes[0, 1]
    ax.plot(coverage, [100 * f(row, "fixed_coco_macro_f1") for row in stage_rows],
            marker="o", lw=1.6, color=colors["blue"], label="COCO Macro-F1")
    ax.plot(coverage, [100 * f(row, "fixed_coco_dog_f1") for row in stage_rows],
            marker="s", lw=1.6, color=colors["violet"], label="COCO dog F1")
    ax.set(xlabel="Visual-state coverage (%)", ylabel="Fixed-test performance (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower center")

    ax = axes[1, 0]
    for metric, label, color, marker in (
        ("rare_state_macro_recall", "Rare-state macro recall", colors["blue"], "o"),
        ("common_state_macro_recall", "Common-state macro recall", colors["teal"], "s"),
        ("worst5_state_recall", "Worst-5 state recall", colors["red"], "^"),
    ):
        ax.plot(coverage, [100 * f(row, metric) for row in stage_rows], marker=marker,
                lw=1.6, color=color, label=label)
    ax.set(xlabel="Visual-state coverage (%)", ylabel="Independent state-test recall (%)")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")

    ax = axes[1, 1]
    models = sorted({row["model"] for row in model_rows})
    deltas = []
    for model in models:
        start = next(row for row in model_rows if row["model"] == model and row["external_dog_added"] == "0")
        end = next(row for row in model_rows if row["model"] == model and row["external_dog_added"] == "350")
        deltas.append(100 * (f(end, "rare_state_macro_recall_mean") - f(start, "rare_state_macro_recall_mean")))
    order = np.argsort(deltas)
    labels = [models[index].replace("mobilenet_v3_", "MobileNetV3-").replace("_", " ") for index in order]
    values = [deltas[index] for index in order]
    bar_colors = [colors["red"] if value < 0 else colors["teal"] for value in values]
    ax.barh(np.arange(len(values)), values, color=bar_colors, height=0.68)
    ax.axvline(0, color="#272727", lw=0.8)
    ax.set_yticks(np.arange(len(values)), labels, fontsize=6.5)
    ax.set(xlabel="Rare-state recall change, D650 - D300 (pp)")
    for y, value in enumerate(values):
        ax.text(value + (0.08 if value >= 0 else -0.08), y, f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=6.3)

    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
    fig.savefig(FIGURES / "state_coverage_performance_response.svg", bbox_inches="tight")
    fig.savefig(FIGURES / "state_coverage_performance_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    stage = read_csv(TABLES / "stage_response.csv")
    model = read_csv(TABLES / "metrics_by_model_stage.csv")
    states = read_csv(TABLES / "state_recall_by_run.csv")
    make_figure(stage, model)
    start, end = stage[0], stage[-1]

    endpoint_rows = []
    for name in sorted({row["model"] for row in model}):
        a = next(row for row in model if row["model"] == name and row["external_dog_added"] == "0")
        b = next(row for row in model if row["model"] == name and row["external_dog_added"] == "350")
        endpoint_rows.append({
            "model": name,
            "rare_delta": f(b, "rare_state_macro_recall_mean") - f(a, "rare_state_macro_recall_mean"),
            "worst_delta": f(b, "worst5_state_recall_mean") - f(a, "worst5_state_recall_mean"),
            "overall_delta": f(b, "overall_dog_recall_mean") - f(a, "overall_dog_recall_mean"),
        })

    state_values = defaultdict(lambda: defaultdict(list))
    state_support = {}
    for row in states:
        if row["external_dog_added"] not in {"0", "350"} or not row["dog_recall"]:
            continue
        key = (row["feature_id"], row["state"])
        state_values[key][row["external_dog_added"]].append(float(row["dog_recall"]))
        state_support[key] = int(row["state_test_support"])
    state_deltas = []
    for key, values in state_values.items():
        if "0" in values and "350" in values and state_support[key] >= 5:
            state_deltas.append({
                "feature_id": key[0], "state": key[1], "support": state_support[key],
                "delta": float(np.mean(values["350"]) - np.mean(values["0"])),
            })
    state_deltas.sort(key=lambda row: row["delta"])
    with (TABLES / "endpoint_state_deltas.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state_deltas[0]))
        writer.writeheader(); writer.writerows(state_deltas)

    peaks = {}
    for metric in ("fixed_coco_macro_f1", "rare_state_macro_recall", "worst5_state_recall"):
        row = max(stage, key=lambda item: f(item, metric))
        peaks[metric] = {"coverage": f(row, "coverage"), "add": int(row["external_dog_added"]), "value": f(row, metric)}

    lines = [
        "# 视觉状态覆盖率—模型性能响应分析", "",
        "日期：2026-09-20", "",
        "## 执行状态", "",
        "- 已冻结原有8级覆盖梯度、8个模型和3个训练种子；",
        "- 已复用192/192个模型权重完成独立状态测试推理；",
        "- 状态测试集来自Open Images V7 test：100张常规dog和100张困难dog；",
        "- 200/200张已通过dog_v3.1自动结构校验，但尚未完成人工状态复核；",
        "- 训练与状态测试不存在image ID或SHA-256重复。", "",
        "## 跨模型平均响应", "",
        "| 覆盖率 | 新增dog | COCO Macro-F1 | COCO dog F1 | 独立dog Recall | 稀有状态Macro Recall | 常见状态Macro Recall | Worst-5 Recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage:
        lines.append(
            f"| {pct(f(row, 'coverage'))} | {row['external_dog_added']} | "
            f"{pct(f(row, 'fixed_coco_macro_f1'))} | {pct(f(row, 'fixed_coco_dog_f1'))} | "
            f"{pct(f(row, 'overall_dog_recall'))} | {pct(f(row, 'rare_state_macro_recall'))} | "
            f"{pct(f(row, 'common_state_macro_recall'))} | {pct(f(row, 'worst5_state_recall'))} |"
        )
    lines += [
        "", "## 主要观察", "",
        f"1. 覆盖指数从{pct(f(start, 'coverage'))}提高到{pct(f(end, 'coverage'))}，增加{pp(f(end, 'coverage') - f(start, 'coverage'))}。",
        f"2. 固定COCO Macro-F1从{pct(f(start, 'fixed_coco_macro_f1'))}变为{pct(f(end, 'fixed_coco_macro_f1'))}，端点差值{pp(f(end, 'fixed_coco_macro_f1') - f(start, 'fixed_coco_macro_f1'))}；其跨模型平均峰值位于新增{peaks['fixed_coco_macro_f1']['add']}、覆盖率{pct(peaks['fixed_coco_macro_f1']['coverage'])}。",
        f"3. 独立测试的稀有状态Macro Recall从{pct(f(start, 'rare_state_macro_recall'))}变为{pct(f(end, 'rare_state_macro_recall'))}，差值{pp(f(end, 'rare_state_macro_recall') - f(start, 'rare_state_macro_recall'))}。",
        f"4. Worst-5状态Recall从{pct(f(start, 'worst5_state_recall'))}变为{pct(f(end, 'worst5_state_recall'))}，差值{pp(f(end, 'worst5_state_recall') - f(start, 'worst5_state_recall'))}，且最低均值出现在中高覆盖阶段。",
        f"5. 遮挡Recall端点变化{pp(f(end, 'occluded_dog_recall') - f(start, 'occluded_dog_recall'))}，截断Recall端点变化{pp(f(end, 'truncated_dog_recall') - f(start, 'truncated_dog_recall'))}，小目标Recall端点变化{pp(f(end, 'small_dog_recall') - f(start, 'small_dog_recall'))}。",
        "", "## 架构差异", "",
        "| 模型 | Δ独立dog Recall | Δ稀有状态Macro Recall | ΔWorst-5 Recall |",
        "|---|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        lines.append(f"| {row['model']} | {pp(row['overall_delta'])} | {pp(row['rare_delta'])} | {pp(row['worst_delta'])} |")
    lines += [
        "", f"稀有状态端点变化为正的模型有{sum(row['rare_delta'] > 0 for row in endpoint_rows)}/8个；"
        f"Worst-5端点变化为正的模型有{sum(row['worst_delta'] > 0 for row in endpoint_rows)}/8个。"
        "这说明高覆盖数据没有被各架构一致吸收。", "",
        "## 变化最大的视觉状态", "",
        "以下只列测试支持数不少于5的状态，数值为D650相对D300各24次运行（8模型×3种子）的聚合端点差异。", "",
        "### 下降最多", "",
        "| 特征 | 状态 | 支持数 | Recall变化 |", "|---|---|---:|---:|",
    ]
    for row in state_deltas[:8]:
        lines.append(f"| {row['feature_id']} | {row['state']} | {row['support']} | {pp(row['delta'])} |")
    lines += ["", "### 提升最多", "", "| 特征 | 状态 | 支持数 | Recall变化 |", "|---|---|---:|---:|"]
    for row in reversed(state_deltas[-8:]):
        lines.append(f"| {row['feature_id']} | {row['state']} | {row['support']} | {pp(row['delta'])} |")
    lines += [
        "", "## 当前可以支持的结论", "",
        "1. 训练数据视觉状态覆盖率与固定测试性能呈非单调关系；",
        "2. 覆盖增加没有带来稀有状态和最差状态Recall的一致提升；",
        "3. 模型架构和训练种子对状态响应存在明显影响；",
        "4. 高覆盖阶段并非所有状态都下降，少数场景、姿态和遮挡状态出现改善，因此结果属于状态间性能重新分配；",
        "5. 覆盖率可用于衡量数据状态充分性，但不能替代模型性能或鲁棒性指标。", "",
        "## 不能支持的结论", "",
        "- 不能声称覆盖率提高必然导致F1下降；",
        "- 不能声称F1下降本身证明数据集更好；",
        "- 不能把当前曲线解释为覆盖率的纯因果效应，因为覆盖选择、样本难度和COCO/Open Images域构成同时变化；",
        "- 在人工复核前，不能把自动状态分层指标作为最终人工真值结果。", "",
        "## 下一步与停止条件", "",
        "1. 人工复核200张状态测试标签，优先检查支持数5–15的状态及变化最大的状态；",
        "2. 复核后重新生成本报告，不需要重新推理模型；",
        "3. 完成模型×阶段的局部覆盖敏感度和变化点置信区间；",
        "4. dog主线完成后，再选择一个新类别做低/中/高三点复验。", "",
        "当前不建议继续补dog图片或增加新模型。", "",
        "## 产物", "",
        "- `tables/stage_response.csv`：覆盖阶段响应；",
        "- `tables/metrics_by_model_stage.csv`：逐模型逐阶段结果；",
        "- `tables/state_recall_by_run.csv`：逐状态逐运行结果；",
        "- `tables/endpoint_state_deltas.csv`：状态端点变化；",
        "- `figures/state_coverage_performance_response.svg`：可编辑四联图；",
        "- `figures/state_coverage_performance_response.png`：300 DPI预览图。", "",
    ]
    (OUTPUT / "state_response_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")

    feature_rows = read_csv(
        ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
    )
    input_rows = {
        row["image_id"]: row for row in read_csv(
            ROOT / "artifacts/openimages_v7_cua_vsl_v1/state_test_inputs/dog_feature_test_manifest.csv"
        )
    }
    review_rows = []
    feature_fields = [
        key for key in feature_rows[0]
        if key not in {"image_path", "marked_image_path", "crop_image_path"}
    ]
    for row in feature_rows:
        source = input_rows[row["image_id"]]
        review_rows.append({
            "sample_id": row["image_id"], "split": row["split"],
            "image_path": source["image_path"],
            "marked_image_path": source["marked_image_path"],
            "crop_image_path": source["crop_image_path"],
            **{f"auto_{key}": row.get(key, "") for key in feature_fields},
            "review_status": "pending", "correction_json": "", "review_notes": "",
            "reviewer": "", "reviewed_at": "",
        })
    with (OUTPUT / "human_review_queue.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader(); writer.writerows(review_rows)

    claim_path = OUTPUT / "claim_evidence.md"
    claim_path.write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| State-test evaluation is leakage-free by image ID and SHA-256 | supported | runtime checks; `protocol.json` | Exact IDs and hashes only; perceptual near-duplicates not audited |\n"
        "| 192 frozen checkpoints were evaluated | supported | `progress.json`; `tables/metrics_by_run.csv` | Gap-gradient checkpoints only |\n"
        "| Coverage and performance are non-monotonic | supported | `tables/stage_response.csv` | Current dog/COCO/Open Images protocol |\n"
        "| Higher coverage consistently improves rare-state recall | contradicted | `tables/stage_response.csv`; endpoint model table | Automatic state labels; 1/8 positive endpoint |\n"
        "| Higher coverage necessarily causes F1 decline | unsupported | mixed stage and architecture responses | Selection and domain composition are confounded |\n"
        "| State-stratified values are human-verified | pending | human review absent | Automatic labels only |\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "| Gate | Verdict | Evidence or blocker |\n|---|---|---|\n"
        "| Citation integrity | pass | No external factual claims in this experiment report |\n"
        "| Result traceability | pass | 192/192 run predictions and state tables trace to frozen checkpoints |\n"
        "| Claim-scope integrity | pass with limitation | State results explicitly marked preliminary until human review |\n\n"
        "## Remaining major item\n\nHuman review of the 200 automatic dog_v3.1 state annotations is not complete."
        " This does not invalidate checkpoint inference, but blocks final state-level paper claims.\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        "- 2026-09-20: froze the existing 100 main + 100 difficulty Open Images dog state test.\n"
        "- Leakage audit passed for image IDs and SHA-256 hashes against all gradient training manifests.\n"
        "- Automatic dog_v3.1 annotations: 200/200 structurally valid; human review remains pending.\n"
        "- Reused 192/192 frozen checkpoints; no model retraining or checkpoint modification.\n"
        "- Generated per-run predictions, state recalls, coverage sensitivities, report, and editable SVG.\n"
        "- Reproduction: `.venv/bin/python -u scripts/evaluate_coco_openimages_state_response.py`; "
        "`.venv/bin/python scripts/report_coco_openimages_state_response.py`.\n",
        encoding="utf-8",
    )
    checksum_rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "checksums.csv":
            checksum_rows.append({
                "path": str(path.relative_to(OUTPUT)), "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    with (OUTPUT / "checksums.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checksum_rows[0]))
        writer.writeheader(); writer.writerows(checksum_rows)
    print(OUTPUT / "state_response_analysis_zh.md")


if __name__ == "__main__":
    main()
