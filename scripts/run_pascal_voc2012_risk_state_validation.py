#!/usr/bin/env python3
"""Validate preregistered dog visibility-risk states on PASCAL VOC 2012."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision import transforms

from run_four_model_gap_vs_random import build_model
from run_resnet18_dog_coverage_comparison import write_csv, write_json
from run_resnet18_five_class_coverage_comparison import CLASS_SPECS, CropDataset, predict


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "data/pascal_voc2012_dog_validation_v1"
FEATURE_ROOT = ROOT / "artifacts/pascal_voc2012_dog_validation_v1/features_auto"
CHECKPOINT_ROOT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
OUTPUT = ROOT / "artifacts/pascal_voc2012_risk_state_validation_20260920"

MODELS = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
)
SEEDS = (20260916, 20260917, 20260918)
GROUP_SIZE = 50
CONTROL_SIZE = 150
SELECTION_SEED = 20260920
BOOTSTRAP_REPLICATES = 10000
GROUPS = (
    "head_partial",
    "mouth_not_visible_only",
    "muzzle_partial_only",
    "all_clear_control",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_family(value: str) -> str:
    if value.startswith("indoor_"):
        return "indoor"
    if value.startswith("outdoor_"):
        return "outdoor"
    return value


def group_candidates(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "head_partial": [row for row in rows if row["head_visibility"] == "partial"],
        "mouth_not_visible_only": [
            row for row in rows
            if row["mouth_visibility"] == "not_visible"
            and row["head_visibility"] != "partial"
            and row["muzzle_visibility"] != "partial"
        ],
        "muzzle_partial_only": [
            row for row in rows
            if row["muzzle_visibility"] == "partial"
            and row["head_visibility"] != "partial"
            and row["mouth_visibility"] != "not_visible"
        ],
        "all_clear_control": [
            row for row in rows
            if row["head_visibility"] == "clear"
            and row["muzzle_visibility"] == "clear"
            and row["mouth_visibility"] == "clear"
        ],
    }


def sample_risk_groups(candidates: dict[str, list[dict]]) -> list[dict]:
    selected = []
    for offset, group in enumerate(GROUPS[:3]):
        pool = sorted(candidates[group], key=lambda row: row["image_id"])
        if len(pool) < GROUP_SIZE:
            raise ValueError(f"insufficient {group} candidates: {len(pool)}")
        random.Random(SELECTION_SEED + offset).shuffle(pool)
        for row in pool[:GROUP_SIZE]:
            selected.append({**row, "validation_group": group})
    if len({row["image_id"] for row in selected}) != len(selected):
        raise ValueError("risk-state groups overlap")
    return selected


def matched_controls(risk_rows: list[dict], controls: list[dict]) -> list[dict]:
    if len(risk_rows) != CONTROL_SIZE:
        raise ValueError("risk sample count must equal control sample count")
    all_rows = risk_rows + controls
    log_areas = np.asarray([math.log(max(float(row["target_area_ratio"]), 1e-8)) for row in all_rows])
    mean, std = float(log_areas.mean()), float(log_areas.std(ddof=1))
    std = std or 1.0
    occ = {"none": 0, "partial": 1, "heavy": 2}
    trunc = {"none": 0, "partial": 1, "heavy": 2}

    def cost(left: dict, right: dict) -> float:
        area_cost = abs(
            (math.log(max(float(left["target_area_ratio"]), 1e-8)) - mean) / std
            - (math.log(max(float(right["target_area_ratio"]), 1e-8)) - mean) / std
        )
        return (
            area_cost
            + 0.75 * abs(occ[left["occlusion_level"]] - occ[right["occlusion_level"]])
            + 0.75 * abs(trunc[left["truncation_level"]] - trunc[right["truncation_level"]])
            + 0.25 * int(scene_family(left["scene"]) != scene_family(right["scene"]))
        )

    matrix = np.asarray([[cost(risk, control) for control in controls] for risk in risk_rows])
    risk_indices, control_indices = linear_sum_assignment(matrix)
    if len(risk_indices) != CONTROL_SIZE:
        raise ValueError("control matching did not produce 150 pairs")
    selected = []
    for risk_index, control_index in zip(risk_indices.tolist(), control_indices.tolist()):
        selected.append({
            **controls[control_index],
            "validation_group": "all_clear_control",
            "matched_risk_image_id": risk_rows[risk_index]["image_id"],
            "matched_risk_group": risk_rows[risk_index]["validation_group"],
            "matching_cost": float(matrix[risk_index, control_index]),
        })
    return selected


def make_classifier_crop(row: dict, output: Path) -> Path:
    crop_path = output / "crops" / f"{row['image_id']}.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if crop_path.is_file():
        return crop_path
    source = Path(row["image_path"])
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    x1, y1, x2, y2 = json.loads(row["target_bbox"])
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError(f"invalid crop for {row['image_id']}: {(x1, y1, x2, y2)} / {image.size}")
    image.crop((x1, y1, x2, y2)).save(crop_path, quality=95)
    return crop_path


def balance_rows(rows: list[dict]) -> list[dict]:
    output = []
    for group in GROUPS:
        subset = [row for row in rows if row["validation_group"] == group]
        output.append({
            "validation_group": group,
            "count": len(subset),
            "target_area_ratio_mean": float(np.mean([float(row["target_area_ratio"]) for row in subset])),
            "target_area_ratio_median": float(np.median([float(row["target_area_ratio"]) for row in subset])),
            "occlusion_partial_or_heavy_fraction": float(np.mean([
                row["occlusion_level"] in {"partial", "heavy"} for row in subset
            ])),
            "truncation_partial_or_heavy_fraction": float(np.mean([
                row["truncation_level"] in {"partial", "heavy"} for row in subset
            ])),
        })
    return output


def prepare() -> list[dict]:
    features = read_csv(FEATURE_ROOT / "results.csv")
    if len(features) != 1286 or any(row["feature_status"] != "ok" for row in features):
        raise ValueError("PASCAL dog_v3.1 features are incomplete")
    manifest = {
        row["image_id"]: row
        for row in read_csv(INPUT_ROOT / "manifests/dog_feature_candidate_manifest.csv")
    }
    if set(manifest) != {row["image_id"] for row in features}:
        raise ValueError("feature and source manifests do not match")
    joined = [{**row, **manifest[row["image_id"]]} for row in features]
    candidates = group_candidates(joined)
    risk = sample_risk_groups(candidates)
    controls = matched_controls(risk, sorted(candidates["all_clear_control"], key=lambda row: row["image_id"]))
    selected = risk + controls
    if len(selected) != 300 or len({row["image_id"] for row in selected}) != 300:
        raise ValueError("validation manifest must contain 300 unique images")
    rows = []
    for row in selected:
        crop_path = make_classifier_crop(row, OUTPUT)
        rows.append({
            **row,
            "matched_risk_image_id": row.get("matched_risk_image_id", ""),
            "matched_risk_group": row.get("matched_risk_group", ""),
            "matching_cost": row.get("matching_cost", ""),
            "sample_id": row["image_id"],
            "class_name": "dog",
            "label": 0,
            "crop_path": str(crop_path.resolve()),
            "classifier_crop_sha256": sha256(crop_path),
        })
    write_csv(OUTPUT / "manifests/selected_300.csv", rows)
    write_csv(OUTPUT / "tables/selection_balance.csv", balance_rows(rows))

    checkpoints = [
        CHECKPOINT_ROOT / "runs" / model / "oi_gap_add0" / str(seed) / "best.pt"
        for model in MODELS for seed in SEEDS
    ]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing[:3]}")
    protocol = {
        "experiment_id": "pascal-voc2012-risk-state-validation-20260920",
        "created_at": now(),
        "claim": "Visibility-risk states discovered on Open Images have higher dog error rates than matched all-clear PASCAL controls.",
        "independent_source": "PASCAL VOC 2012 trainval",
        "fixed_training_condition": "Open Images D300 / oi_gap_add0",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "checkpoint_count": 24,
        "selection_seed": SELECTION_SEED,
        "group_sizes": {group: sum(row["validation_group"] == group for row in rows) for group in GROUPS},
        "selection_blinding": "Groups were selected using dog_v3.1 states and metadata only, before PASCAL model inference.",
        "human_review": "not performed, per user instruction; VLM state labels are accepted as provisional automatic labels",
        "classifier_input": "tight PASCAL dog bounding-box crop; ImageNet evaluation transform",
        "primary_metric": "dog error-rate difference from matched all-clear control",
        "inference_only": True,
        "input_sha256": {
            "source_manifest": sha256(INPUT_ROOT / "manifests/dog_feature_candidate_manifest.csv"),
            "state_features": sha256(FEATURE_ROOT / "results.csv"),
            "selected_manifest": sha256(OUTPUT / "manifests/selected_300.csv"),
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "research_log.md").write_text(
        "# Research Log\n\n"
        f"- Protocol frozen: {now()}\n"
        "- Test: cross-source validation of three preregistered visibility-risk groups.\n"
        "- Data: 300 PASCAL VOC 2012 dog targets selected without model predictions.\n"
        "- Models: 24 frozen Open Images D300 checkpoints; no retraining.\n"
        "- Human review: omitted per user instruction; labels remain VLM-provisional.\n"
        "- Reproduction: `.venv/bin/python -u scripts/run_pascal_voc2012_risk_state_validation.py`.\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| The three state groups have higher dog error than matched controls | pending | `tables/group_comparisons.csv` | VLM-provisional labels; PASCAL VOC 2012; 24 fixed models |\n"
        "| A state causes classification errors | unsupported | observational matched comparison | State co-occurrence and residual confounding remain |\n",
        encoding="utf-8",
    )
    return rows


def infer(rows: list[dict], workers: int) -> None:
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(
        CropDataset(rows, transform), batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completed = 0
    for model_name in MODELS:
        for seed in SEEDS:
            destination = OUTPUT / "predictions" / model_name / str(seed) / "predictions.csv"
            if destination.is_file():
                completed += 1
                print(f"[{completed}/24] reuse {model_name} seed={seed}", flush=True)
                continue
            checkpoint_path = CHECKPOINT_ROOT / "runs" / model_name / "oi_gap_add0" / str(seed) / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(model_name)
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            predictions = predict(model, loader, rows, device)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_csv(destination, predictions)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            completed += 1
            write_json(OUTPUT / "progress.json", {
                "status": "running", "completed_runs": completed, "total_runs": 24,
                "latest_model": model_name, "latest_seed": seed, "updated_at": now(),
            })
            print(f"[{completed}/24] inferred {model_name} seed={seed}", flush=True)


def cluster_interval(values: list[float], random_seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(random_seed)
    array = np.asarray(values, dtype=float)
    sampled = generator.choice(array, size=(BOOTSTRAP_REPLICATES, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def exact_sign_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    permuted = []
    for signs in itertools.product((-1, 1), repeat=len(array)):
        permuted.append(abs(float(np.mean(array * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def aggregate(rows: list[dict]) -> None:
    selected_by_id = {row["sample_id"]: row for row in rows}
    prediction_rows = []
    run_rows = []
    architecture_group_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    architecture_matched_control_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    for model_name in MODELS:
        for seed in SEEDS:
            predictions = read_csv(OUTPUT / "predictions" / model_name / str(seed) / "predictions.csv")
            if len(predictions) != 300 or {row["sample_id"] for row in predictions} != set(selected_by_id):
                raise ValueError(f"invalid predictions for {model_name}/{seed}")
            for prediction in predictions:
                source = selected_by_id[prediction["sample_id"]]
                prediction_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "sample_id": prediction["sample_id"],
                    "validation_group": source["validation_group"],
                    "matched_risk_group": source["matched_risk_group"],
                    "error": 1 - int(prediction["correct"]),
                    "predicted_class": prediction["predicted_class"],
                    "prob_dog": float(prediction["prob_dog"]),
                    "target_area_ratio": float(source["target_area_ratio"]),
                    "occlusion_level": source["occlusion_level"],
                    "truncation_level": source["truncation_level"],
                })
            for group in GROUPS:
                subset = [row for row in predictions if selected_by_id[row["sample_id"]]["validation_group"] == group]
                error_rate = float(np.mean([1 - int(row["correct"]) for row in subset]))
                run_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "validation_group": group,
                    "n": len(subset),
                    "error_count": sum(1 - int(row["correct"]) for row in subset),
                    "error_rate": error_rate,
                    "dog_recall": 1 - error_rate,
                    "mean_prob_dog": float(np.mean([float(row["prob_dog"]) for row in subset])),
                })
                architecture_group_rates[model_name, group].append(error_rate)
            for group in GROUPS[:3]:
                subset = [
                    row for row in predictions
                    if selected_by_id[row["sample_id"]]["validation_group"] == "all_clear_control"
                    and selected_by_id[row["sample_id"]]["matched_risk_group"] == group
                ]
                architecture_matched_control_rates[model_name, group].append(
                    float(np.mean([1 - int(row["correct"]) for row in subset]))
                )
    write_csv(OUTPUT / "tables/predictions_long.csv", prediction_rows)
    write_csv(OUTPUT / "tables/run_group_metrics.csv", run_rows)

    group_summary = []
    for group in GROUPS:
        architecture_rates = [
            float(np.mean(architecture_group_rates[model, group])) for model in MODELS
        ]
        lower, upper = cluster_interval(architecture_rates, SELECTION_SEED + GROUPS.index(group))
        group_summary.append({
            "validation_group": group,
            "n_images": sum(row["validation_group"] == group for row in rows),
            "mean_error_rate_24_runs": float(np.mean([
                row["error_rate"] for row in run_rows if row["validation_group"] == group
            ])),
            "architecture_cluster_ci_low": lower,
            "architecture_cluster_ci_high": upper,
            "mean_dog_recall_24_runs": 1 - float(np.mean([
                row["error_rate"] for row in run_rows if row["validation_group"] == group
            ])),
        })
    write_csv(OUTPUT / "tables/group_summary.csv", group_summary)

    comparisons = []
    for index, group in enumerate(GROUPS[:3]):
        risk_rates = [float(np.mean(architecture_group_rates[model, group])) for model in MODELS]
        matched_control_rates = [
            float(np.mean(architecture_matched_control_rates[model, group])) for model in MODELS
        ]
        deltas = [
            risk_rate - control_rate
            for risk_rate, control_rate in zip(risk_rates, matched_control_rates)
        ]
        lower, upper = cluster_interval(deltas, SELECTION_SEED + 100 + index)
        comparisons.append({
            "validation_group": group,
            "mean_risk_error_rate": float(np.mean(risk_rates)),
            "mean_matched_control_error_rate": float(np.mean(matched_control_rates)),
            "mean_error_rate_difference": float(np.mean(deltas)),
            "architecture_cluster_ci_low": lower,
            "architecture_cluster_ci_high": upper,
            "positive_architectures": sum(value > 0 for value in deltas),
            "zero_architectures": sum(value == 0 for value in deltas),
            "negative_architectures": sum(value < 0 for value in deltas),
            "exact_sign_permutation_p": exact_sign_p(deltas),
        })
    write_csv(OUTPUT / "tables/group_comparisons.csv", comparisons)

    adjusted = []
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        frame = pd.DataFrame(prediction_rows)
        frame["log_area"] = np.log(frame["target_area_ratio"].clip(lower=1e-8))
        fit = smf.gee(
            "error ~ C(validation_group, Treatment(reference='all_clear_control')) + log_area + C(occlusion_level) + C(truncation_level) + C(model) + C(seed)",
            groups="sample_id", data=frame, family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        for group in GROUPS[:3]:
            key = f"C(validation_group, Treatment(reference='all_clear_control'))[T.{group}]"
            coefficient = float(fit.params[key])
            lower, upper = fit.conf_int().loc[key].tolist()
            adjusted.append({
                "validation_group": group,
                "adjusted_odds_ratio": math.exp(coefficient),
                "ci_low": math.exp(float(lower)),
                "ci_high": math.exp(float(upper)),
                "p": float(fit.pvalues[key]),
                "model": "GEE binomial; image clusters; model and seed fixed effects",
            })
        write_csv(OUTPUT / "tables/adjusted_gee.csv", adjusted)
    except Exception as error:
        write_json(OUTPUT / "tables/adjusted_gee_error.json", {"error": repr(error)})

    display = {
        "head_partial": "Head partial",
        "mouth_not_visible_only": "Mouth not visible only",
        "muzzle_partial_only": "Muzzle partial only",
        "all_clear_control": "All-clear control",
    }
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    means = [row["mean_error_rate_24_runs"] for row in group_summary]
    lower = [means[i] - group_summary[i]["architecture_cluster_ci_low"] for i in range(4)]
    upper = [group_summary[i]["architecture_cluster_ci_high"] - means[i] for i in range(4)]
    colors = ["#C44E52", "#DD8452", "#E5AE38", "#4C72B0"]
    axis.bar(range(4), np.asarray(means) * 100, color=colors, width=0.68)
    axis.errorbar(range(4), np.asarray(means) * 100, yerr=np.asarray([lower, upper]) * 100,
                  fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    axis.set_xticks(range(4), [display[group] for group in GROUPS], rotation=12, ha="right")
    axis.set_ylabel("Dog error rate (%)")
    axis.set_title("Independent PASCAL VOC 2012 risk-state validation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "figures/group_error_rates.png", dpi=220)
    figure.savefig(OUTPUT / "figures/group_error_rates.svg")
    plt.close(figure)

    lines = [
        "# PASCAL VOC 2012 dog风险状态独立验证", "",
        "## 实验设置", "",
        "- 300张PASCAL VOC 2012 dog目标；状态分组仅使用VLM自动标签，没有人工复核；",
        "- 24个冻结分类模型：8个架构 × 3个训练种子；不重新训练；",
        "- 风险发现来自Open Images，验证图片来自PASCAL VOC 2012；",
        "- 分类输入为紧致dog目标框裁剪。", "",
        "## 组别结果", "",
        "| 组别 | 图片数 | 平均错误率 | 架构bootstrap 95% CI | dog Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in group_summary:
        lines.append(
            f"| {row['validation_group']} | {row['n_images']} | {100*row['mean_error_rate_24_runs']:.2f}% | "
            f"[{100*row['architecture_cluster_ci_low']:.2f}%, {100*row['architecture_cluster_ci_high']:.2f}%] | "
            f"{100*row['mean_dog_recall_24_runs']:.2f}% |"
        )
    lines.extend(["", "## 相对清晰对照", "", "| 风险组 | 错误率差 | 方向一致架构 | p |", "|---|---:|---:|---:|"])
    for row in comparisons:
        lines.append(
            f"| {row['validation_group']} | {100*row['mean_error_rate_difference']:+.2f} pp "
            f"[{100*row['architecture_cluster_ci_low']:+.2f}, {100*row['architecture_cluster_ci_high']:+.2f}] | "
            f"{row['positive_architectures']}/8 | {row['exact_sign_permutation_p']:.4f} |"
        )
    if adjusted:
        lines.extend(["", "## 协变量调整", "", "| 风险组 | 调整后OR | 95% CI | p |", "|---|---:|---:|---:|"])
        for row in adjusted:
            lines.append(
                f"| {row['validation_group']} | {row['adjusted_odds_ratio']:.3f} | "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['p']:.4g} |"
            )
    lines.extend([
        "", "## 结论边界", "",
        "- 该实验检验跨数据源风险关联复现，不检验因果关系；",
        "- VLM状态标签未经人工复核，因此结论属于自动标签条件下的结果；",
        "- 头部部分可见组允许与嘴部/口鼻状态共现，应解释为复合视觉退化组；",
        "- 模型权重完全固定，错误率差表示测试状态暴露差异，不表示模型训练后性能变化。", "",
    ])
    (OUTPUT / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUTPUT / "summary.json", {
        "status": "completed",
        "fixed_checkpoint_runs": 24,
        "selected_images": 300,
        "group_summary": group_summary,
        "comparisons": comparisons,
        "adjusted_gee": adjusted,
    })
    write_json(OUTPUT / "progress.json", {
        "status": "completed", "completed_runs": 24, "total_runs": 24, "updated_at": now(),
    })
    (OUTPUT / "review.md").write_text(
        "# Review\n\n"
        "- Citation integrity: not applicable; no external literature claims introduced.\n"
        "- Result traceability: passed; summary values derive from 24 preserved prediction files.\n"
        "- Claim scope: passed with limitation; association only, VLM-provisional state labels.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = prepare()
    if args.prepare_only:
        write_json(OUTPUT / "progress.json", {
            "status": "prepared", "completed_runs": 0, "total_runs": 24, "updated_at": now(),
        })
        print(json.dumps({
            "status": "prepared",
            "selected": len(rows),
            "groups": dict(Counter(row["validation_group"] for row in rows)),
            "output": str(OUTPUT.resolve()),
        }, ensure_ascii=False, indent=2))
        return
    infer(rows, args.workers)
    aggregate(rows)
    print(json.dumps({"status": "completed", "output": str(OUTPUT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
