#!/usr/bin/env python3
"""Re-run the cat risk experiment with the dog-style ordered difficulty protocol.

The five major cat dimensions remain a coverage dictionary in the accompanying
report. Q1--Q5 are formed only from ordered visual difficulty factors shared
with the dog experiment: target scale, critical visibility, occlusion,
truncation, and orientation. Frozen recent-five checkpoints are evaluated on a
600-target balanced candidate pool (200 cats and 100 examples for each
negative class), with 40 targets per cat risk tier.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/cat_dog_style_risk_20261017"
POOL = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920/candidate_pool.csv"
FEATURES = ROOT / "artifacts/openimages_v7_cat_test_v1/features_auto/results.csv"
MODEL_OUT = ROOT / "artifacts/recent_five_models_20261009"

sys.path.insert(0, str(ROOT / "scripts"))
import run_cross_category_balanced_difficulty as dog_style  # noqa: E402
import run_recent_five_models as recent  # noqa: E402

MODELS = ("yolo26s_cls", "pe_core_b16", "siglip2_base", "mobileclip2_s0", "aimv2_large")
SEEDS = (20260916, 20260917, 20260918)
DISPLAY = recent.DISPLAY
CLASSES = tuple(recent.CLASSES)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def difficulty_rows(pool: pd.DataFrame, features: dict[str, dict[str, str]]) -> list[dict]:
    target = pool[pool.class_name == "cat"].copy().reset_index(drop=True)
    rows = []
    for r in target.to_dict("records"):
        d = dog_style.difficulty("cat", r, features[r["sample_id"]])
        rows.append({"sample_id": r["sample_id"], "class_name": "cat", **d,
                     "bbox_area_ratio": float(r["bbox_area_ratio"]), "crop_path": r["crop_path"]})
    rows.sort(key=lambda x: (x["difficulty_score"], x["visibility_score"],
                             x["occlusion_score"], x["truncation_score"],
                             x["orientation_score"], x["size_score"], x["sample_id"]))
    for i, row in enumerate(rows):
        row["Q"] = i // 40 + 1
        row["risk_rank"] = i + 1
    return rows


def infer(pool: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    outputs = {}
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        print(f"[model] {model_name}", flush=True)
        model, meta = recent.load_model(model_name)
        model.cuda()
        for seed in SEEDS:
            checkpoint_path = MODEL_OUT / "runs" / model_name / str(seed) / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
            print(f"  [infer] seed={seed} targets={len(pool)}", flush=True)
            logits = recent.predict(model, pool, meta)
            outputs[(model_name, seed)] = logits
            np.savez_compressed(OUT / "predictions" / f"{model_name}_{seed}.npz",
                                sample_ids=pool.sample_id.to_numpy(dtype=str),
                                labels=pool.label.to_numpy(dtype=int), logits=logits)
        del model
        torch.cuda.empty_cache()
    return outputs


def metrics_for(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(6), zero_division=0)
    return {"accuracy": float((y == pred).mean()), "macro_precision": float(p.mean()),
            "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
            "cat_f1": float(f[1]), "cat_recall": float(r[1])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL, dtype={"sample_id": str, "crop_path": str})
    assert pool.class_name.value_counts().to_dict() == {"cat": 200, "sheep": 100, "horse": 100, "dog": 100, "person": 100}
    assert set(pool.class_name) <= set(CLASSES)
    feature_df = pd.read_csv(FEATURES, dtype={"image_id": str})
    features = {r["image_id"]: r for r in feature_df.to_dict("records")}
    cat_ids = set(pool.loc[pool.class_name == "cat", "sample_id"])
    assert cat_ids <= set(features)
    risk = difficulty_rows(pool, features)
    write_csv(OUT / "cat_dog_style_risk_scores.csv", risk)
    risk_df = pd.DataFrame(risk)
    print("[risk] Q means", risk_df.groupby("Q").difficulty_score.mean().round(3).to_dict(), flush=True)

    # The same 40 negatives per class are used at every Q, so only cat risk changes.
    negative = []
    for name in CLASSES:
        if name == "cat":
            continue
        negative.append(pool[pool.class_name == name].sort_values("sample_id").iloc[:40])
    negative_df = pd.concat(negative, ignore_index=True)
    q_rows = []
    for q in range(1, 6):
        cat_q = pool[pool.sample_id.isin(risk_df.loc[risk_df.Q == q, "sample_id"])]
        q_rows.append(pd.concat([cat_q, negative_df], ignore_index=True))

    logits_by_run = infer(pool)
    summary = []
    runs = []
    for model_name in MODELS:
        for seed in SEEDS:
            logits = logits_by_run[(model_name, seed)]
            pred = logits.argmax(axis=1)
            pool_index = {sid: i for i, sid in enumerate(pool.sample_id)}
            for q, frame in enumerate(q_rows, start=1):
                idx = np.array([pool_index[sid] for sid in frame.sample_id])
                m = metrics_for(frame.label.to_numpy(dtype=int), pred[idx])
                runs.append({"model": model_name, "display_model": DISPLAY[model_name], "seed": seed,
                             "Q": q, "n_total": len(idx), "n_per_class": 40, **m})
    run_df = pd.DataFrame(runs)
    run_df.to_csv(OUT / "q1_q5_macro_f1_runs.csv", index=False)
    grouped = run_df.groupby(["model", "display_model", "Q"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        cat_f1_mean=("cat_f1", "mean"), cat_recall_mean=("cat_recall", "mean"))
    grouped.to_csv(OUT / "q1_q5_macro_f1_summary.csv", index=False)
    for model_name in MODELS:
        vals = grouped[grouped.model == model_name].sort_values("Q").macro_f1_mean.to_numpy()
        summary.append({"model": model_name, "display_model": DISPLAY[model_name],
                        "Q1_macro_f1": vals[0], "Q5_macro_f1": vals[-1],
                        "Q5_minus_Q1": vals[-1] - vals[0], "monotonic_nonincreasing": bool(np.all(np.diff(vals) <= 1e-12))})
    write_csv(OUT / "q1_q5_macro_f1_effect.csv", summary)

    # Plot mean Macro-F1 with seed standard deviation. This is the dog-style
    # six-class metric; Cat-F1 is retained in CSV as a category diagnostic.
    zh_font = "/home/hcf/.local/share/fonts/NotoSansCJKsc-Regular.otf"
    if Path(zh_font).exists():
        font_manager.fontManager.addfont(zh_font)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = ["#1769AA", "#D55E00", "#009E73", "#7B2CBF", "#E69F00"]
    for color, model_name in zip(colors, MODELS):
        frame = grouped[grouped.model == model_name].sort_values("Q")
        ax.errorbar(frame.Q, frame.macro_f1_mean * 100, yerr=frame.macro_f1_std.fillna(0) * 100,
                    marker="o", linewidth=2.2, capsize=3, color=color, label=DISPLAY[model_name])
    ax.set_xticks(range(1, 6), ["Q1\n易", "Q2", "Q3", "Q4", "Q5\n难"])
    ax.set_xlabel("猫类有序风险层（狗类同口径）", fontsize=12)
    ax.set_ylabel("六分类 Macro-F1 (%)", fontsize=12)
    ax.set_title("猫类狗式风险分层：最新五模型 Q1–Q5 Macro-F1", fontsize=15, weight="bold")
    ax.set_ylim(0, 100); ax.grid(alpha=.22); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "猫类狗式风险分层_最新五模型_Q1-Q5_Macro-F1.png", dpi=220)
    fig.savefig(OUT / "猫类狗式风险分层_最新五模型_Q1-Q5_Macro-F1.svg")
    plt.close(fig)

    protocol = {
        "version": "cat-dog-style-risk-v1-20261017", "candidate_pool": str(POOL),
        "target_count": 200, "tier_size": 40, "balanced_eval": "40 per class x 6 classes per Q",
        "risk_dimensions": ["target_scale", "critical_part_visibility", "occlusion", "truncation", "orientation"],
        "coverage_dictionary": "five major cat classes retained separately; not used as nominal rarity risk",
        "models": list(MODELS), "seeds": list(SEEDS), "metric": "six-class Macro-F1; Cat-F1 diagnostic",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# 猫类狗式有序风险重做（2026-10-17）", "",
        "本版本按狗类实验口径重做猫类风险分层。五大类只用于状态字典覆盖与配额；Q1–Q5仅由尺度、关键部位可见性、遮挡、截断、朝向五个有序难度因素决定。", "",
        "- 候选池：200 个猫目标；每个 Q 层 40 个目标。", "- 每个 Q 使用六分类平衡评估，每类 40 个样本。",
        "- 模型：最新五个冻结模型；3 个 seed。", "- 主指标：六分类 Macro-F1；Cat-F1 和 Cat-Recall 作为诊断。", "",
        "## Q1–Q5 结果", "", "见 `q1_q5_macro_f1_summary.csv`、`q1_q5_macro_f1_effect.csv` 和折线图。", "",
        "## 解释边界", "", "若 Q5−Q1 未稳定为负，说明该数据池仍不足以验证风险分层，不能通过图形或分数人为制造下降趋势。", "",
        "## 文件", "", "- `cat_dog_style_risk_scores.csv`：狗式风险分数与 Q 分层。", "- `q1_q5_macro_f1_runs.csv`：逐模型、逐 seed、逐 Q 指标。", "- `猫类狗式风险分层_最新五模型_Q1-Q5_Macro-F1.svg`：最终折线图。", "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(OUT), "effect": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
