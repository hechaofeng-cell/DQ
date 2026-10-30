#!/usr/bin/env python3
"""Validate representative frozen models on predefined high-risk dog states."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKTREE = Path(__file__).resolve().parents[1]
PROJECT = Path("/home/hcf/project/QE")
SOURCE = PROJECT / "artifacts/fixed_train_visual_test_adequacy_20260920"
FEATURE_PATH = PROJECT / "artifacts/openimages_v7_cua_vsl_v1/state_test_features_auto/results.csv"
RUNS = WORKTREE / "artifacts/foundation_model_state_robustness_20260922/runs"
OUTPUT = WORKTREE / "artifacts/high_risk_state_challenge_20260922"
FONT_PATH = PROJECT / "scripts/assets/NotoSansSC-Regular.ttf"

MODELS = ("siglip2_base", "aimv2_large", "swin_t")
DISPLAY = {"siglip2_base": "SigLIP2-B", "aimv2_large": "AIMv2-L", "swin_t": "Swin-T"}
COLORS = {"siglip2_base": "#B47F98", "aimv2_large": "#9D6268", "swin_t": "#7884B4"}
SEEDS = (20260916, 20260917, 20260918)
CLASS_NAMES = ("dog", "cat", "horse", "sheep", "person")
LABELS = {name: index for index, name in enumerate(CLASS_NAMES)}
MAX_DOG_PER_SCENARIO = 30
MIN_SUPPORT = 10
BOOTSTRAP_REPLICATES = 10000
RANDOM_SEED = 20260922
AUDIT_REVIEW_IDS = {
    "oi7-test-6defbc6b78e9d624-dog": "crop is visually ambiguous between a very small dog and cat",
    "oi7-test-f4dde5593cfc7b94-dog": "target visually resembles an arctic fox rather than a domestic dog",
}

SCENARIOS = (
    ("small_target", "小目标（原框面积<3%）", "bbox_area_ratio", "lt", 0.03),
    ("head_partial", "头部部分可见", "head_visibility", "eq", "partial"),
    ("muzzle_partial", "口鼻部分可见", "muzzle_visibility", "eq", "partial"),
    ("mouth_not_visible", "嘴部不可见", "mouth_visibility", "eq", "not_visible"),
    ("head_occluded", "头部遮挡", "occlusion_parts", "contains", "head"),
    ("left_side", "左侧身", "body_orientation", "eq", "left_side"),
    ("walking", "行走姿态", "posture_action_state", "eq", "walking"),
    ("outdoor_grass", "户外草地", "scene", "eq", "outdoor_grass"),
    ("outdoor_snow", "户外雪地", "scene", "eq", "outdoor_snow"),
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_style() -> None:
    family = "DejaVu Sans"
    if FONT_PATH.is_file():
        fm.fontManager.addfont(str(FONT_PATH))
        family = fm.FontProperties(fname=str(FONT_PATH)).get_name()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 8.2, "axes.titlesize": 9.3, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .8,
        "xtick.direction": "out", "ytick.direction": "out", "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def state_match(row: pd.Series, field: str, operation: str, value: object) -> bool:
    if operation == "lt":
        return float(row[field]) < float(value)
    if operation == "eq":
        return str(row[field]) == str(value)
    if operation == "contains":
        raw = str(row[field])
        try:
            parsed = json.loads(raw)
            return value in parsed
        except json.JSONDecodeError:
            return str(value) in raw
    raise ValueError(operation)


def classification_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    pred_support = confusion.sum(axis=0).astype(float)
    precision = np.divide(tp, pred_support, out=np.zeros(5), where=pred_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "macro_recall": float(recall.mean()),
        "dog_precision": float(precision[0]),
        "dog_recall": float(recall[0]),
        "dog_f1": float(f1[0]),
    }


def bootstrap_dog_recall(prediction_matrix: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    # Hierarchical resampling treats head seed and dog image as the uncertainty units.
    seed_count, image_count = prediction_matrix.shape
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        seed_indices = rng.integers(0, seed_count, seed_count)
        image_indices = rng.integers(0, image_count, image_count)
        values[index] = prediction_matrix[np.ix_(seed_indices, image_indices)].mean()
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def prepare_manifests() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    pool = pd.read_csv(SOURCE / "candidate_pool.csv")
    order = pd.read_csv(SOURCE / "orders/gap_dog_order.csv").sort_values("rank")
    non_dog = pd.read_csv(SOURCE / "orders/non_dog_order.csv").sort_values(["class_name", "rank"])
    features = pd.read_csv(FEATURE_PATH)
    dog = order.merge(features, left_on="sample_id", right_on="image_id", how="left", validate="one_to_one", suffixes=("", "_feature"))
    if dog["feature_status"].ne("ok").any():
        raise ValueError("dog feature table contains unsuccessful rows")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifests").mkdir(exist_ok=True)
    selected: dict[str, pd.DataFrame] = {}
    manifest_rows = []
    for scenario_id, scenario_label, field, operation, value in SCENARIOS:
        mask = dog.apply(lambda row: state_match(row, field, operation, value), axis=1)
        eligible = dog.loc[mask].sort_values("rank")
        if len(eligible) < MIN_SUPPORT:
            raise ValueError(f"{scenario_id}: only {len(eligible)} eligible dog images")
        chosen_dog = eligible.head(MAX_DOG_PER_SCENARIO).copy()
        n = len(chosen_dog)
        pieces = [pool[pool.sample_id.isin(chosen_dog.sample_id)].copy()]
        for class_name in CLASS_NAMES[1:]:
            ids = non_dog[non_dog.class_name == class_name].head(n).sample_id
            pieces.append(pool[pool.sample_id.isin(ids)].copy())
        challenge = pd.concat(pieces, ignore_index=True)
        challenge["scenario_id"] = scenario_id
        challenge["scenario_label"] = scenario_label
        challenge["scenario_support"] = len(eligible)
        challenge["selected_dog_count"] = n
        challenge.to_csv(OUTPUT / "manifests" / f"{scenario_id}.csv", index=False)
        selected[scenario_id] = challenge
        manifest_rows.append({
            "scenario_id": scenario_id, "scenario_label": scenario_label,
            "definition": f"{field} {operation} {value}", "eligible_dog_count": len(eligible),
            "selected_dog_count": n, "targets_per_class": n, "total_targets": len(challenge),
        })
    manifest_summary = pd.DataFrame(manifest_rows)
    manifest_summary.to_csv(OUTPUT / "tables_scenario_manifest.csv", index=False)
    return manifest_summary, selected


def run_analysis(manifest_summary: pd.DataFrame, selected: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    confusion_rows = []
    dog_prediction_rows = []
    dog_prediction_cache: dict[tuple[str, str], list[np.ndarray]] = {}
    for model in MODELS:
        for seed in SEEDS:
            predictions = pd.read_csv(RUNS / model / str(seed) / "candidate_predictions.csv")
            by_id = predictions.set_index("sample_id")
            for scenario_id, challenge in selected.items():
                rows = by_id.loc[challenge.sample_id]
                true = rows.true_label.to_numpy(dtype=int)
                predicted = rows.predicted_label.to_numpy(dtype=int)
                metrics = classification_metrics(true, predicted)
                dog_mask = true == 0
                dog_correct = (predicted[dog_mask] == 0).astype(float)
                dog_prediction_cache.setdefault((model, scenario_id), []).append(dog_correct)
                dog_rows = challenge.iloc[np.flatnonzero(dog_mask)]
                for sample_id, predicted_label, correct in zip(dog_rows.sample_id, predicted[dog_mask], dog_correct):
                    dog_prediction_rows.append({
                        "model": model, "model_display": DISPLAY[model], "seed": seed,
                        "scenario_id": scenario_id, "scenario_label": challenge.scenario_label.iloc[0],
                        "sample_id": sample_id, "predicted_label": int(predicted_label),
                        "predicted_class": CLASS_NAMES[int(predicted_label)], "correct": int(correct),
                    })
                metric_rows.append({
                    "model": model, "model_display": DISPLAY[model], "seed": seed,
                    "scenario_id": scenario_id,
                    "scenario_label": challenge.scenario_label.iloc[0],
                    "dog_count": int(dog_mask.sum()), "total_count": len(true), **metrics,
                })
                confusion = np.bincount(true * 5 + predicted, minlength=25).reshape(5, 5)
                for true_label, true_name in enumerate(CLASS_NAMES):
                    for predicted_label, predicted_name in enumerate(CLASS_NAMES):
                        if true_label != predicted_label and confusion[true_label, predicted_label] > 0:
                            confusion_rows.append({
                                "model": model, "seed": seed, "scenario_id": scenario_id,
                                "scenario_label": challenge.scenario_label.iloc[0],
                                "true_class": true_name, "predicted_class": predicted_name,
                                "count": int(confusion[true_label, predicted_label]),
                            })

    metrics = pd.DataFrame(metric_rows)
    confusions = pd.DataFrame(confusion_rows)
    dog_predictions = pd.DataFrame(dog_prediction_rows)
    metrics.to_csv(OUTPUT / "tables_metrics_by_run.csv", index=False)
    confusions.to_csv(OUTPUT / "tables_confusions_by_run.csv", index=False)
    dog_predictions.to_csv(OUTPUT / "tables_dog_predictions_by_scenario.csv", index=False)

    summary_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    metric_names = ("accuracy", "macro_f1", "macro_recall", "dog_precision", "dog_recall", "dog_f1")
    for model in MODELS:
        for scenario_id in manifest_summary.scenario_id:
            rows = metrics[(metrics.model == model) & (metrics.scenario_id == scenario_id)]
            result = {
                "model": model, "model_display": DISPLAY[model], "scenario_id": scenario_id,
                "scenario_label": rows.scenario_label.iloc[0], "dog_count": int(rows.dog_count.iloc[0]),
            }
            for metric in metric_names:
                result[f"{metric}_mean"] = rows[metric].mean()
                result[f"{metric}_sd"] = rows[metric].std(ddof=1)
            matrix = np.vstack(dog_prediction_cache[(model, scenario_id)])
            ci_low, ci_high = bootstrap_dog_recall(matrix, rng)
            result["dog_recall_ci95_low"] = ci_low
            result["dog_recall_ci95_high"] = ci_high
            summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "tables_scenario_model_summary.csv", index=False)

    model_summary = summary.groupby(["model", "model_display"], as_index=False).agg(
        mean_scenario_accuracy=("accuracy_mean", "mean"),
        mean_scenario_macro_f1=("macro_f1_mean", "mean"),
        mean_scenario_dog_recall=("dog_recall_mean", "mean"),
        worst_scenario_dog_recall=("dog_recall_mean", "min"),
        scenario_dog_recall_sd=("dog_recall_mean", "std"),
    ).sort_values(["mean_scenario_dog_recall", "mean_scenario_macro_f1"], ascending=False)
    model_summary.insert(0, "challenge_rank", np.arange(1, len(model_summary) + 1))
    model_summary.to_csv(OUTPUT / "tables_model_challenge_ranking.csv", index=False)

    confusion_summary = confusions.groupby(["model", "true_class", "predicted_class"], as_index=False)["count"].sum()
    confusion_summary = confusion_summary.sort_values(["model", "count"], ascending=[True, False])
    confusion_summary.to_csv(OUTPUT / "tables_confusion_summary.csv", index=False)

    pool = pd.read_csv(SOURCE / "candidate_pool.csv", usecols=["sample_id", "image_path", "crop_path", "bbox_area_ratio"])
    features = pd.read_csv(FEATURE_PATH)
    feature_columns = [
        "image_id", "head_visibility", "muzzle_visibility", "mouth_visibility",
        "body_orientation", "posture_action_state", "occlusion_level", "occlusion_parts", "scene",
    ]
    errors = dog_predictions[dog_predictions.correct == 0]
    error_atlas = errors.groupby(["model", "model_display", "sample_id"], as_index=False).agg(
        error_seed_scenario_count=("correct", "size"),
        error_seed_count=("seed", "nunique"),
        predicted_classes=("predicted_class", lambda values: ";".join(sorted(set(values)))),
        scenarios=("scenario_label", lambda values: ";".join(sorted(set(values)))),
    )
    error_atlas = error_atlas.merge(pool, on="sample_id", how="left").merge(
        features[feature_columns], left_on="sample_id", right_on="image_id", how="left",
    )
    error_atlas["audit_review_required"] = error_atlas.sample_id.isin(AUDIT_REVIEW_IDS).astype(int)
    error_atlas["audit_note"] = error_atlas.sample_id.map(AUDIT_REVIEW_IDS).fillna("")
    error_atlas.to_csv(OUTPUT / "tables_dog_error_atlas.csv", index=False)

    sensitivity = dog_predictions[~dog_predictions.sample_id.isin(AUDIT_REVIEW_IDS)].groupby(
        ["model", "model_display", "scenario_id", "scenario_label"], as_index=False
    ).agg(dog_recall_after_audit=("correct", "mean"), dog_predictions_after_audit=("correct", "size"))
    sensitivity_ranking = sensitivity.groupby(["model", "model_display"], as_index=False).agg(
        mean_scenario_dog_recall_after_audit=("dog_recall_after_audit", "mean"),
        worst_scenario_dog_recall_after_audit=("dog_recall_after_audit", "min"),
    ).sort_values(["mean_scenario_dog_recall_after_audit", "worst_scenario_dog_recall_after_audit"], ascending=False)
    sensitivity_ranking.insert(0, "sensitivity_rank", np.arange(1, len(sensitivity_ranking) + 1))
    sensitivity.to_csv(OUTPUT / "tables_label_audit_sensitivity_by_scenario.csv", index=False)
    sensitivity_ranking.to_csv(OUTPUT / "tables_label_audit_sensitivity_ranking.csv", index=False)
    return metrics, summary, model_summary, confusion_summary


def make_figures(summary: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    setup_style()
    scenario_order = [item[0] for item in SCENARIOS]
    labels = [item[1] for item in SCENARIOS]
    matrix = summary.pivot(index="scenario_id", columns="model", values="dog_recall_mean").loc[scenario_order, MODELS]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("risk", ["#B05D6B", "#F2E7E5", "#EDF0F6", "#586FA5"])
    norm = mpl.colors.Normalize(vmin=max(0.5, float(matrix.min().min()) - .05), vmax=1.0)
    for i in range(len(scenario_order)):
        for j in range(len(MODELS)):
            value = matrix.iloc[i, j]
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=cmap(norm(value)), edgecolor="white", linewidth=.8))
            ax.text(j, i, f"{value * 100:.1f}%", ha="center", va="center", fontsize=8.2,
                    color="white" if value < .7 or value > .94 else "#333333")
    ax.set(xlim=(-.5, len(MODELS) - .5), ylim=(len(scenario_order) - .5, -.5))
    ax.set_xticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS])
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title("高风险视觉状态下的dog Recall", fontsize=11, fontweight="bold", pad=10)
    fig.text(.5, -.01, "每格为3个线性头种子的平均值；同一图片可同时属于多个状态", ha="center", fontsize=7.2, color="#555555")
    fig.savefig(OUTPUT / "figure_state_challenge_heatmap.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_state_challenge_heatmap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    order = model_summary.sort_values("mean_scenario_dog_recall")
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    axes[0].barh(y, order.mean_scenario_dog_recall * 100, color=[COLORS[m] for m in order.model])
    axes[0].set(yticks=y, yticklabels=order.model_display, xlabel="9场景平均dog Recall（%）", xlim=(50, 101), title="平均挑战集表现")
    axes[1].barh(y, order.worst_scenario_dog_recall * 100, color=[COLORS[m] for m in order.model])
    axes[1].set(yticks=y, yticklabels=order.model_display, xlabel="最差场景dog Recall（%）", xlim=(0, 101), title="最弱场景表现")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#D9D9D9", linewidth=.55, alpha=.7); ax.set_axisbelow(True)
    fig.suptitle("代表模型的高风险状态鲁棒性排名", fontsize=11, fontweight="bold")
    fig.savefig(OUTPUT / "figure_model_challenge_ranking.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_model_challenge_ranking.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_protocol(manifest_summary: pd.DataFrame) -> None:
    inputs = {
        "candidate_pool": SOURCE / "candidate_pool.csv",
        "coverage_order": SOURCE / "orders/gap_dog_order.csv",
        "non_dog_order": SOURCE / "orders/non_dog_order.csv",
        "dog_features": FEATURE_PATH,
    }
    protocol = {
        "experiment_id": "high-risk-state-challenge-20260922",
        "created_at": now(),
        "question": "Do the previously identified high-risk dog states reproduce distinct weaknesses in representative frozen models?",
        "models": list(MODELS), "seeds": list(SEEDS), "classes": list(CLASS_NAMES),
        "selection": {
            "maximum_dog_per_scenario": MAX_DOG_PER_SCENARIO,
            "minimum_state_support": MIN_SUPPORT,
            "rule": "coverage-rank order; use all eligible if support <30, otherwise first 30",
            "balance": "for n dog images, include the first n fixed targets from each of cat, horse, sheep, and person",
        },
        "primary_metric": "dog recall for each state-specific challenge set",
        "secondary_metrics": ["accuracy", "macro_f1", "macro_recall", "dog_precision", "dog_f1"],
        "uncertainty": "3 head seeds; 10000-replicate hierarchical bootstrap over seeds and dog images for dog recall",
        "causal_boundary": "states overlap; results are scenario-specific associations, not isolated causal effects",
        "manual_audit_sensitivity": AUDIT_REVIEW_IDS,
        "scenario_counts": manifest_summary.to_dict(orient="records"),
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    manifest_summary, selected = prepare_manifests()
    metrics, summary, model_summary, confusion_summary = run_analysis(manifest_summary, selected)
    make_figures(summary, model_summary)
    write_protocol(manifest_summary)
    print(model_summary.to_string(index=False))
    print("\nTop confusions")
    print(confusion_summary.groupby("model", group_keys=False).head(5).to_string(index=False))
    print("\nLabel-audit sensitivity")
    print(pd.read_csv(OUTPUT / "tables_label_audit_sensitivity_ranking.csv").to_string(index=False))


if __name__ == "__main__":
    main()
