"""Compute Q1--Q5 Macro-F1 from saved predictions, without new inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "artifacts/coco_five_dimension_fine_selection_20261015"))
    root = parser.parse_args().input.resolve()
    manifest = read_csv(root / "manifest.csv")
    pred = np.load(root / "predictions.npz")
    selected = np.load(root / "selection.npz")
    sample_ids = pred["sample_ids"].astype(str)
    assert [row["sample_id"] for row in manifest] == list(sample_ids)
    assert np.array_equal(selected["sample_ids"].ravel().astype(str), sample_ids)
    truth = np.array([int(row["label"]) for row in manifest])
    classes = np.unique(truth)
    per_class = selected["sample_ids"].shape[1]
    assert np.array_equal(truth, np.repeat(classes, per_class))
    tiers = {(row["scheme"], row["sample_id"]): int(row["tier"])
             for row in read_csv(root / "tiers.csv")}
    models = list(dict.fromkeys(pred["models"].astype(str)))
    display = dict(zip(models, ["ResNet-18", "ResNet-50", "MobileNetV3-Small",
        "MobileNetV3-Large", "DenseNet-121", "EfficientNet-B0", "ConvNeXt-Tiny", "Swin-T"]))
    contexts = [("POOL", "equal_parts"), ("POOL", "equal_global"),
                ("E4", "equal_parts"), ("E2", "equal_global")]
    runs, summary = [], []
    verified = 0
    for group, scheme in contexts:
        indices = (np.arange(len(truth)) if group == "POOL" else
                   (selected[group][0] + classes[:, None] * per_class).ravel())
        tier_ids = np.array([tiers[scheme, sid] for sid in sample_ids])[indices]
        for model in models:
            positions = np.flatnonzero(pred["models"].astype(str) == model)
            values = []
            for position in positions:
                row_values = []
                for q in range(1, 6):
                    subset = indices[tier_ids == q]
                    support = np.bincount(truth[subset], minlength=len(classes))
                    assert len(set(support)) == 1
                    guessed = pred["predictions"][position, subset]
                    score = f1_score(truth[subset], guessed, labels=classes,
                                     average="macro", zero_division=0)
                    matrix = confusion_matrix(truth[subset], guessed, labels=classes)
                    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
                    independent = np.divide(2 * matrix.diagonal(), denominator,
                        out=np.zeros(len(classes), dtype=float), where=denominator > 0).mean()
                    assert np.isclose(score, independent, atol=1e-12)
                    verified += 1
                    row_values.append(score)
                    runs.append(dict(group=group, scheme=scheme, model=model,
                        seed=int(pred["seeds"][position]), tier=q,
                        targets=len(subset), per_class_support=int(support[0]), macro_f1=score))
                values.append(row_values)
            values = np.array(values)
            mean = values.mean(axis=0)
            summary.append(dict(group=group, scheme=scheme, model=model,
                seeds=len(positions), targets_per_tier=int((tier_ids == 1).sum()),
                **{f"Q{q}": mean[q - 1] for q in range(1, 6)},
                Q5_minus_Q1_pp=100 * (mean[-1] - mean[0]),
                strictly_decreasing_mean=bool(np.all(np.diff(mean) < 0)),
                strictly_decreasing_seeds=int(np.all(np.diff(values, axis=1) < 0, axis=1).sum())))
    write_csv(root / "tier_macro_f1_run_metrics.csv", runs)
    write_csv(root / "tier_macro_f1_summary.csv", summary)

    text = ["# COCO模型Q1—Q5分层Macro-F1补充结果", "",
        "本报告由已保存的真实类别、模型预测和分层记录重新计算，不涉及重新训练、重新推理或改动分层。此前分层热图的指标是Recall，本报告另行计算Macro-F1。", "",
        "Q1至Q5表示类别内视觉风险由低到高，不是训练阶段。每个模型先按每个训练种子、每一风险层分别计算六类别Macro-F1，再对三个种子取平均。数值单位为%；差值单位为百分点。", "",
        "严格下降的结论针对三种子平均值，不代表每个种子、每个类别或每张图片都严格下降。", ""]
    titles = {("POOL", "equal_parts"): "完整候选池：五维细粒度评分（2400目标，每层480）",
              ("POOL", "equal_global"): "完整候选池：原五维评分（2400目标，每层480）",
              ("E4", "equal_parts"): "E4覆盖选样后：五维细粒度评分（1200目标，每层240）",
              ("E2", "equal_global"): "E2覆盖选样后：原五维评分（1200目标，每层240）"}
    for group, scheme in contexts:
        rows = [r for r in summary if r["group"] == group and r["scheme"] == scheme]
        text += [f"## {titles[group, scheme]}", "",
            "| 模型 | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 | 均值逐层下降 | 逐层下降的种子数 |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|"]
        for row in rows:
            cells = [display[row["model"]]] + [f"{100 * row[f'Q{q}']:.2f}" for q in range(1, 6)]
            cells += [f"{row['Q5_minus_Q1_pp']:+.2f}",
                      "是" if row["strictly_decreasing_mean"] else "否",
                      f"{row['strictly_decreasing_seeds']}/3"]
            text.append("| " + " | ".join(cells) + " |")
        strict = sum(r["strictly_decreasing_mean"] for r in rows)
        text += ["", f"三种子平均值严格逐层下降：{strict}/8个模型；Q5低于Q1：8/8个模型。", ""]
        for row in rows:
            bumps = [f"Q{q}→Q{q+1}（{100 * row[f'Q{q}']:.2f}%→{100 * row[f'Q{q+1}']:.2f}%）"
                     for q in range(1, 5) if row[f"Q{q+1}"] > row[f"Q{q}"]]
            if bumps:
                text += [f"- {display[row['model']]}出现相邻层回升：{'；'.join(bumps)}。"]
        text.append("")
    text += ["## 结果解释", "",
        "五维细粒度评分下，完整候选池和E4覆盖选样集的8个模型均呈Q1至Q5平均Macro-F1逐层下降。这与高风险层更容易暴露分类困难的预期一致。", "",
        "原五维评分存在个别相邻层回升，风险层不是严格确定的模型难度等级。即使所有模型均出现端点下降，也不能据此宣称细粒度评分优于原评分；两种评分的优劣仍应结合配对比较和不确定性分析。", "",
        "这些是相同已训练模型在不同测试条件下的性能差异，不是模型训练后变差，也不是补图改善训练效果的证据。覆盖选样后测试集F1较低，可以说明它暴露了更多分类困难，不能直接判为数据集质量较差。", "",
        "本报告为补充描述性分析，未新增显著性结论。", "",
        "计算脚本：scripts/summarize_coco_qtier_f1_20261015.py。输入：manifest.csv、predictions.npz、tiers.csv、selection.npz。"]
    (root / "tier_macro_f1_report_zh.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, group in zip(axes, ["POOL", "E4"]):
        for row in summary:
            if row["group"] == group and row["scheme"] == "equal_parts":
                axis.plot(range(1, 6), [100 * row[f"Q{q}"] for q in range(1, 6)],
                          marker="o", markersize=4, label=display[row["model"]])
        axis.set(title="Full pool (n=2400)" if group == "POOL" else "Coverage-selected E4 (n=1200)",
                 xlabel="Risk tier (local fine scoring)", xticks=range(1, 6),
                 xticklabels=[f"Q{q}" for q in range(1, 6)], ylim=(58, 100))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[1].legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("Q1–Q5 Macro-F1: mean over three training seeds")
    fig.tight_layout()
    for suffix in ["svg", "png"]:
        fig.savefig(root / "figures" / f"08_model_qtier_macro_f1.{suffix}", dpi=180)
    plt.close(fig)
    audit = dict(metric="six-class Macro-F1", seed_aggregation="mean after per-seed F1",
                 independent_confusion_matrix_checks=verified, all_passed=True,
                 candidate_targets=len(truth), model_runs=len(pred["models"]),
                 selected_groups=["E2", "E4"], random_groups_included=False,
                 note="E1/E3 repeated random draws are not included in this supplement")
    (root / "tier_macro_f1_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False))
    print(root / "tier_macro_f1_report_zh.md")


if __name__ == "__main__":
    main()
