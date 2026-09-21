#!/usr/bin/env python3
"""Create editable Chinese result figures for the Open Images V7 experiment."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/openimages_v7_four_model_experiment_v1"
SELECTION = ROOT / "artifacts/openimages_v7_dog_supplements_v1/report.json"
OUTPUT = ARTIFACT / "figures"
FONT_DIR = OUTPUT / "fonts"
MODELS = ["resnet18", "convnext_tiny", "maxvit_t", "efficientnet_v2_s"]
MODEL_LABELS = ["ResNet-18", "ConvNeXt-Tiny", "MaxViT-T", "EfficientNetV2-S"]
CONDITIONS = ["base", "random300", "gap300"]
SEEDS = [20260916, 20260917, 20260918]

COLORS = {
    "base": "#606060",
    "random300": "#B4C0E4",
    "gap300": "#484878",
    "accuracy": "#606060",
    "macro_f1": "#7884B4",
    "dog_f1": "#E4A9BC",
}


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def setup_style() -> None:
    regular = FONT_DIR / "NotoSansCJK-Regular.ttc"
    bold = FONT_DIR / "NotoSansCJK-Bold.ttc"
    if not regular.is_file() or not bold.is_file():
        raise FileNotFoundError("Noto CJK font files are missing from the figure folder")
    font_manager.fontManager.addfont(regular)
    font_manager.fontManager.addfont(bold)
    family = font_manager.FontProperties(fname=str(regular)).get_name()
    mpl.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family, "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "axes.unicode_minus": False,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", facecolor="white", bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E4E4E4", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_coverage(ax: plt.Axes, coverage: dict, compact: bool = False) -> None:
    keys = ["base", "random300", "gap300"]
    labels = ["基线\n1,500", "随机补图\n+300", "缺口补图\n+300"]
    values = [coverage[key] * 100 for key in keys]
    bars = ax.bar(labels, values, width=0.62, color=[COLORS[key] for key in keys], zorder=3)
    ax.axhline(90, color="#9B6A7B", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.text(2.46 if not compact else 2.35, 90.5, "90%目标线", color="#815969", ha="right", fontsize=8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.annotate("+3.85 pp", xy=(2, values[2]), xytext=(1.25, 97.2),
                arrowprops={"arrowstyle": "->", "color": "#484878", "lw": 1.0},
                color="#484878", fontweight="bold", ha="center")
    ax.set_ylim(0, 102)
    ax.set_ylabel("dog_v3.1覆盖指数（%）")
    ax.set_title("视觉状态覆盖达到90%以上", loc="left", fontweight="bold")
    clean_axis(ax)


def plot_delta_axis(ax: plt.Axes, delta_rows: dict, metric: str, title: str,
                    color: str, compact: bool = False) -> None:
    y = np.arange(len(MODELS))
    for index, model in enumerate(MODELS):
        row = delta_rows[(model, "gap_minus_random", metric)]
        seeds = np.asarray(json.loads(row["seed_deltas"]), dtype=float) * 100
        mean = seeds.mean()
        sd = seeds.std(ddof=1)
        jitter = np.asarray([-0.09, 0.0, 0.09])
        ax.scatter(seeds, index + jitter, s=18 if compact else 24, facecolors="white",
                   edgecolors=color, linewidth=0.8, alpha=0.95, zorder=4)
        ax.errorbar(mean, index, xerr=sd, fmt="o", color=color, ecolor=color,
                    markersize=5.5 if compact else 6.5, capsize=3, linewidth=1.2, zorder=5)
        ax.text(mean + (0.12 if mean >= 0 else -0.12), index - 0.17, f"{mean:+.2f}",
                color=color, fontsize=7.5, ha="left" if mean >= 0 else "right", va="center")
    ax.axvline(0, color="#303030", linewidth=0.9)
    ax.set_yticks(y, MODEL_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel("缺口补图 - 随机补图（百分点）")
    ax.set_title(title, loc="left", fontweight="bold")
    clean_axis(ax, "x")


def plot_absolute_performance(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    specs = [("macro_f1", "Macro-F1", (92.5, 99.2)), ("dog_f1", "Dog F1", (92.0, 100.1))]
    x = np.arange(len(MODELS))
    offsets = [-0.20, 0.0, 0.20]
    for ax, (metric, title, limits) in zip(axes, specs):
        for offset, condition in zip(offsets, CONDITIONS):
            rows = [summary[(model, condition)] for model in MODELS]
            means = [float(row[f"{metric}_mean"]) * 100 for row in rows]
            stds = [float(row[f"{metric}_std"]) * 100 for row in rows]
            ax.errorbar(x + offset, means, yerr=stds, fmt="o", color=COLORS[condition],
                        ecolor=COLORS[condition], markersize=6, capsize=3, linewidth=1.2,
                        label={"base": "基线", "random300": "随机补图", "gap300": "缺口补图"}[condition])
        ax.set_xticks(x, MODEL_LABELS, rotation=14, ha="right")
        ax.set_ylim(*limits)
        ax.set_ylabel(f"{title}（%）")
        ax.set_title(f"{title}：三个训练条件", loc="left", fontweight="bold")
        clean_axis(ax)
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    fig.suptitle("固定测试集性能：均值 ± 样本标准差（3个种子）", fontsize=13, fontweight="bold")
    save(fig, "fig2_absolute_performance")


def difficulty_values() -> tuple[np.ndarray, list[dict]]:
    definitions = [
        ("全部困难dog", lambda row: True, 100),
        ("小目标<3%", lambda row: float(row["bbox_area_ratio"]) < .03, 42),
        ("极小目标<1%", lambda row: float(row["bbox_area_ratio"]) < .01, 10),
        ("遮挡", lambda row: row["is_occluded"] == "1", 36),
        ("截断", lambda row: row["is_truncated"] == "1", 32),
    ]
    values = np.zeros((len(MODELS), len(definitions)))
    data_rows = []
    for model_index, model in enumerate(MODELS):
        for slice_index, (slice_name, predicate, support) in enumerate(definitions):
            paired = []
            for seed in SEEDS:
                condition_recall = {}
                for condition in ("random300", "gap300"):
                    rows = read_csv(ARTIFACT / "runs" / model / condition / str(seed) / "difficulty_test_predictions.csv")
                    selected = [row for row in rows if predicate(row)]
                    condition_recall[condition] = sum(int(row["correct"]) for row in selected) / len(selected)
                paired.append((condition_recall["gap300"] - condition_recall["random300"]) * 100)
            values[model_index, slice_index] = np.mean(paired)
            data_rows.append({
                "panel": "difficulty_delta", "model": model, "condition": "gap_minus_random",
                "metric": slice_name, "mean": np.mean(paired), "std": np.std(paired, ddof=1),
                "seed_values": json.dumps(paired), "support": support,
                "source": "difficulty_test_predictions.csv",
            })
    return values, data_rows


def plot_difficulty_heatmap(ax: plt.Axes, values: np.ndarray, compact: bool = False) -> None:
    cmap = LinearSegmentedColormap.from_list("nmi_delta", ["#7884B4", "#F7F7F7", "#E4A9BC"])
    limit = 17
    rows, columns = values.shape
    image = ax.pcolormesh(
        np.arange(columns + 1) - 0.5, np.arange(rows + 1) - 0.5, values,
        cmap=cmap, vmin=-limit, vmax=limit, shading="flat",
        edgecolors="white", linewidth=0.8,
    )
    xlabels = ["全部\nn=100", "小目标<3%\nn=42", "极小目标<1%\nn=10", "遮挡\nn=36", "截断\nn=32"]
    ax.set_xticks(np.arange(5), xlabels)
    ax.set_yticks(np.arange(len(MODELS)), MODEL_LABELS)
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    color="white" if abs(value) > 9 else "#282828", fontsize=7.2 if compact else 8)
    ax.set_title("困难dog Recall差值（缺口 - 随机）", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    return image


def write_figure_data(coverage: dict, summary: dict, delta_rows: dict, difficulty_rows: list[dict]) -> None:
    rows = []
    for condition in CONDITIONS:
        rows.append({"panel": "coverage", "model": "", "condition": condition,
                     "metric": "dog_v3.1_coverage", "mean": coverage[condition] * 100,
                     "std": "", "seed_values": "", "support": {"base": 1500, "random300": 1800, "gap300": 1800}[condition],
                     "source": "openimages_v7_dog_supplements_v1/report.json"})
    for (model, condition), row in summary.items():
        for metric in ("accuracy", "macro_f1", "dog_f1", "dog_recall"):
            rows.append({"panel": "absolute_performance", "model": model, "condition": condition,
                         "metric": metric, "mean": float(row[f"{metric}_mean"]) * 100,
                         "std": float(row[f"{metric}_std"]) * 100, "seed_values": "", "support": 500,
                         "source": "tables/results_by_condition.csv"})
    for model in MODELS:
        for metric in ("accuracy", "macro_f1", "dog_f1"):
            row = delta_rows[(model, "gap_minus_random", metric)]
            rows.append({"panel": "main_delta", "model": model, "condition": "gap_minus_random",
                         "metric": metric, "mean": float(row["delta_mean"]) * 100,
                         "std": float(row["delta_std"]) * 100,
                         "seed_values": json.dumps([value * 100 for value in json.loads(row["seed_deltas"])]),
                         "support": 500, "source": "tables/paired_deltas.csv"})
    rows.extend(difficulty_rows)
    fields = ["panel", "model", "condition", "metric", "mean", "std", "seed_values", "support", "source"]
    with (OUTPUT / "figure_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    coverage = selection["coverage"]
    summary_rows = read_csv(ARTIFACT / "tables/results_by_condition.csv")
    summary = {(row["model"], row["condition"]): row for row in summary_rows}
    paired_rows = read_csv(ARTIFACT / "tables/paired_deltas.csv")
    delta_rows = {(row["model"], row["comparison"], row["metric"]): row for row in paired_rows}
    difficulty, difficulty_rows = difficulty_values()
    write_figure_data(coverage, summary, delta_rows, difficulty_rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    plot_coverage(ax, coverage)
    fig.suptitle("缺口驱动选图实现覆盖目标", fontsize=13, fontweight="bold")
    save(fig, "fig1_coverage")

    plot_absolute_performance(summary)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1", "dog_f1"),
                                 ("Accuracy变化", "Macro-F1变化", "Dog F1变化")):
        plot_delta_axis(ax, delta_rows, metric, title, COLORS[metric])
    fig.suptitle("等预算主比较：缺口补图并未让所有模型受益", fontsize=13, fontweight="bold")
    save(fig, "fig3_gap_vs_random")

    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    image = plot_difficulty_heatmap(ax, difficulty)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.03)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Recall变化（百分点）")
    fig.suptitle("困难状态收益具有模型依赖性", fontsize=13, fontweight="bold")
    save(fig, "fig4_difficulty_heatmap")

    fig = plt.figure(figsize=(13.33, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.86, 1.14), height_ratios=(1, 1))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    plot_coverage(ax_a, coverage, compact=True)
    plot_delta_axis(ax_b, delta_rows, "macro_f1", "Macro-F1：仅2/4模型均值为正", COLORS["macro_f1"], compact=True)
    plot_delta_axis(ax_c, delta_rows, "dog_f1", "Dog F1：收益同样不一致", COLORS["dog_f1"], compact=True)
    image = plot_difficulty_heatmap(ax_d, difficulty, compact=True)
    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D")
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.034, pad=0.025)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Recall变化（pp）")
    fig.suptitle("视觉覆盖显著提高，但分类收益不具普遍性", fontsize=16, fontweight="bold")
    fig.text(0.5, -0.01, "均值 ± 样本标准差，3个训练种子；正值表示gap300优于random300",
             ha="center", fontsize=9, color="#555555")
    save(fig, "fig0_overview_four_panel")
    print(f"wrote figures to {OUTPUT}")


if __name__ == "__main__":
    main()
