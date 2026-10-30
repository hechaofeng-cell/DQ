#!/usr/bin/env python3
"""Evaluate the old eight frozen models on the newly re-ranked cat Q1--Q5 tiers."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/cat_dog_style_risk_20261017"
OUT = BASE / "legacy_eight"
POOL = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920/candidate_pool.csv"
PRED_ROOT = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920/predictions"
MODELS = ("resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
          "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t")
DISPLAY = {"resnet18":"ResNet-18", "resnet50":"ResNet-50", "mobilenet_v3_small":"MobileNetV3-S",
           "mobilenet_v3_large":"MobileNetV3-L", "densenet121":"DenseNet-121", "efficientnet_b0":"EfficientNet-B0",
           "convnext_tiny":"ConvNeXt-T", "swin_t":"Swin-T"}
SEEDS = (20260916, 20260917, 20260918)
CLASSES = ("dog", "cat", "horse", "sheep", "person")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    return {"accuracy": float((y == pred).mean()), "macro_precision": float(p.mean()),
            "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
            "cat_f1": float(f[1]), "cat_recall": float(r[1])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL, dtype={"sample_id": str})
    risk = pd.read_csv(BASE / "cat_dog_style_risk_scores.csv", dtype={"sample_id": str})
    risk = risk.sort_values("risk_rank")
    assert risk.Q.value_counts().sort_index().to_dict() == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert set(risk.sample_id) == set(pool.loc[pool.class_name == "cat", "sample_id"])
    # Keep a fixed balanced negative set in every Q layer.
    negatives = pd.concat([
        pool[pool.class_name == c].sort_values("sample_id").iloc[:40] for c in CLASSES if c != "cat"
    ], ignore_index=True)
    q_frames = {}
    for q in range(1, 6):
        cats = pool[pool.sample_id.isin(risk.loc[risk.Q == q, "sample_id"])]
        frame = pd.concat([cats, negatives], ignore_index=True)
        assert frame.class_name.value_counts().to_dict() == {"cat": 40, "dog": 40, "horse": 40, "sheep": 40, "person": 40}
        q_frames[q] = frame
    pool_index = {sid: i for i, sid in enumerate(pool.sample_id)}
    rows = []
    prediction_dump = []
    for model in MODELS:
        for seed in SEEDS:
            path = PRED_ROOT / model / str(seed) / "candidate_predictions.csv"
            pred_df = pd.read_csv(path, dtype={"sample_id": str})
            by_id = pred_df.set_index("sample_id")
            assert set(by_id.index) == set(pool.sample_id)
            pred_by_id = by_id.predicted_label.astype(int).to_dict()
            for q, frame in q_frames.items():
                y = frame.label.to_numpy(dtype=int)
                pred = np.array([pred_by_id[sid] for sid in frame.sample_id], dtype=int)
                m = metrics(y, pred)
                rows.append({"model": model, "display_model": DISPLAY[model], "seed": seed,
                             "Q": q, "n_total": len(frame), "n_per_class": 40, **m})
            prediction_dump.append({"model": model, "seed": seed, "prediction_file": str(path)})
    run_df = pd.DataFrame(rows)
    run_df.to_csv(OUT / "q1_q5_macro_f1_runs.csv", index=False)
    summary = run_df.groupby(["model", "display_model", "Q"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        cat_f1_mean=("cat_f1", "mean"), cat_recall_mean=("cat_recall", "mean"))
    summary.to_csv(OUT / "q1_q5_macro_f1_summary.csv", index=False)
    effects = []
    for model in MODELS:
        f = summary[summary.model == model].sort_values("Q")
        y = f.macro_f1_mean.to_numpy()
        effects.append({"model": model, "display_model": DISPLAY[model], "Q1_macro_f1": y[0],
                        "Q5_macro_f1": y[-1], "Q5_minus_Q1": y[-1] - y[0],
                        "monotonic_nonincreasing": bool(np.all(np.diff(y) <= 1e-12))})
    pd.DataFrame(effects).to_csv(OUT / "q1_q5_macro_f1_effect.csv", index=False)

    font = "/home/hcf/.local/share/fonts/NotoSansCJKsc-Regular.otf"
    if Path(font).exists(): font_manager.fontManager.addfont(font)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, .875, len(MODELS)))
    for color, model in zip(colors, MODELS):
        f = summary[summary.model == model].sort_values("Q")
        ax.errorbar(f.Q, 100*f.macro_f1_mean, yerr=100*f.macro_f1_std.fillna(0), marker="o",
                    linewidth=1.9, capsize=3, color=color, label=DISPLAY[model])
    ax.set_xticks(range(1, 6), ["Q1\n易", "Q2", "Q3", "Q4", "Q5\n难"])
    ax.set_xlabel("猫类有序风险层（按新风险分数重新排序）", fontsize=12)
    ax.set_ylabel("五分类 Macro-F1 (%)", fontsize=12)
    ax.set_title("旧八模型在猫类新 Q1–Q5 风险分层上的 Macro-F1", fontsize=15, weight="bold")
    ax.set_ylim(0, 100); ax.grid(alpha=.22); ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "猫类新风险分层_旧八模型_Q1-Q5_Macro-F1.png", dpi=220)
    fig.savefig(OUT / "猫类新风险分层_旧八模型_Q1-Q5_Macro-F1.svg"); plt.close(fig)
    (OUT / "report.md").write_text(
        "# 旧八模型猫类新风险分层复评\n\n"
        "旧八模型使用新狗式风险分数排序后的猫类 Q1–Q5；没有沿用旧的 Q 层。每层 40 个猫目标，并加入每个负类 40 个样本组成五分类平衡测试集。\n\n"
        "主结果见 `q1_q5_macro_f1_summary.csv`、`q1_q5_macro_f1_effect.csv` 和折线图。\n",
        encoding="utf-8")
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == "__main__":
    main()
