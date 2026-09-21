#!/usr/bin/env python3
"""Create editable result figures for the COCO + Open Images coverage gradient."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918"
OUTPUT = ARTIFACT / "figures"
RAW = ARTIFACT / "all_metrics.json"
FONT_ROOT = ROOT / "artifacts/openimages_v7_four_model_experiment_v1/figures/fonts"

STAGES = [f"oi_gap_add{value}" for value in (0, 50, 100, 150, 200, 250, 300, 350)]
MODELS = [
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "densenet121", "efficientnet_b0", "convnext_tiny", "swin_t",
]
LABELS = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50",
    "mobilenet_v3_small": "MobileNetV3-Small", "mobilenet_v3_large": "MobileNetV3-Large",
    "densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-T",
}
COLORS = {
    "resnet18": "#484878", "resnet50": "#7884B4",
    "mobilenet_v3_small": "#A77C91", "mobilenet_v3_large": "#D59AAF",
    "densenet121": "#4D8F88", "efficientnet_b0": "#76A56F",
    "convnext_tiny": "#B28759", "swin_t": "#6F6F6F",
}
MARKERS = ["o", "s", "^", "D", "P", "v", "X", "h"]


def setup_style() -> None:
    regular = FONT_ROOT / "NotoSansCJK-Regular.ttc"
    bold = FONT_ROOT / "NotoSansCJK-Bold.ttc"
    font_manager.fontManager.addfont(regular)
    font_manager.fontManager.addfont(bold)
    family = font_manager.FontProperties(fname=str(regular)).get_name()
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family, "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.4,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "figure.dpi": 160, "savefig.dpi": 300,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", facecolor="white", bbox_inches="tight", dpi=300)
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    if grid:
        ax.grid(axis=grid, color="#E8E8E8", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def aggregate():
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["condition"]].append(row)
    coverage = np.array([np.mean([row["coverage"] for row in grouped[MODELS[0], stage]]) * 100
                         for stage in STAGES])
    summary = {}
    metrics = ["accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1",
               "small_dog_recall", "tiny_dog_recall"]
    for model in MODELS:
        for stage in STAGES:
            summary[model, stage] = {
                metric: (np.mean([row[metric] for row in grouped[model, stage]]) * 100,
                         np.std([row[metric] for row in grouped[model, stage]], ddof=1) * 100)
                for metric in metrics
            }
    return rows, coverage, summary


def write_data(rows, coverage, summary) -> None:
    output_rows = []
    for index, stage in enumerate(STAGES):
        for model in MODELS:
            record = {"stage": stage, "coverage_pct": coverage[index], "model": model,
                      "model_label": LABELS[model]}
            for metric in ("accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1",
                           "small_dog_recall", "tiny_dog_recall"):
                mean, std = summary[model, stage][metric]
                record[f"{metric}_mean_pct"] = mean
                record[f"{metric}_std_pct"] = std
            output_rows.append(record)
    fields = list(output_rows[0])
    with (OUTPUT / "figure_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)


def plot_coverage(ax: plt.Axes, coverage: np.ndarray, compact: bool = False) -> None:
    counts = np.array([300, 350, 400, 450, 500, 550, 600, 650])
    ax.plot(counts, coverage, color="#484878", marker="o", linewidth=2.0,
            markersize=5, zorder=3)
    ax.fill_between(counts, 65, coverage, color="#E4E4F0", alpha=0.55, zorder=1)
    for x, value in zip(counts, coverage):
        ax.text(x, value + 0.75, f"{value:.2f}", ha="center", va="bottom",
                fontsize=7 if compact else 8, color="#484878")
    peak_index = 4
    ax.axvline(counts[peak_index], color="#A77C91", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.text(counts[peak_index] + 8, 67.0, "跨模型平均性能峰值阶段", rotation=90,
            color="#8C5D73", fontsize=7.2, va="bottom")
    ax.set_xlim(280, 670)
    ax.set_ylim(65, 94)
    ax.set_xticks(counts)
    ax.set_xlabel("每个类别的训练目标数")
    ax.set_ylabel("dog_v3.1覆盖指数（%）")
    ax.set_title("外部按需补图形成连续覆盖梯度", loc="left", fontweight="bold")
    clean_axis(ax)


def plot_model_lines(ax: plt.Axes, coverage: np.ndarray, summary, metric: str,
                     title: str, compact: bool = False) -> None:
    for index, model in enumerate(MODELS):
        means = np.array([summary[model, stage][metric][0] for stage in STAGES])
        stds = np.array([summary[model, stage][metric][1] for stage in STAGES])
        ax.plot(coverage, means, color=COLORS[model], marker=MARKERS[index],
                markersize=3.8 if compact else 4.6, linewidth=1.25, label=LABELS[model], zorder=3)
        if not compact:
            ax.fill_between(coverage, means - stds, means + stds,
                            color=COLORS[model], alpha=0.07, linewidth=0)
    average = np.array([np.mean([summary[model, stage][metric][0] for model in MODELS])
                        for stage in STAGES])
    ax.plot(coverage, average, color="#202020", linewidth=2.4, marker="o",
            markersize=5.4, label="8模型平均", zorder=5)
    peak = int(np.argmax(average))
    ax.scatter([coverage[peak]], [average[peak]], s=56, facecolor="white",
               edgecolor="#202020", linewidth=1.4, zorder=6)
    ax.annotate(f"平均峰值 {average[peak]:.2f}%\n覆盖率 {coverage[peak]:.2f}%",
                xy=(coverage[peak], average[peak]), xytext=(coverage[peak] - 4.0, average[peak] + 2.0),
                arrowprops={"arrowstyle": "->", "color": "#404040", "lw": 0.8},
                fontsize=7.2, ha="center")
    all_values = [summary[model, stage][metric][0] for model in MODELS for stage in STAGES]
    ax.set_ylim(min(all_values) - 2.0, max(all_values) + 2.8)
    ax.set_xlabel("dog_v3.1覆盖指数（%）")
    ax.set_ylabel(f"{title}（%）")
    ax.set_title(f"{title}随覆盖率呈非单调变化", loc="left", fontweight="bold")
    clean_axis(ax)


def delta_matrix(summary):
    metrics = ["accuracy", "macro_f1", "dog_precision", "dog_recall", "dog_f1"]
    values = np.zeros((len(MODELS), len(metrics)))
    for i, model in enumerate(MODELS):
        for j, metric in enumerate(metrics):
            values[i, j] = summary[model, STAGES[-1]][metric][0] - summary[model, STAGES[0]][metric][0]
    return metrics, values


def plot_delta_heatmap(ax: plt.Axes, summary, compact: bool = False):
    metrics, values = delta_matrix(summary)
    labels = ["Accuracy", "Macro-F1", "dog Precision", "dog Recall", "dog F1"]
    limit = max(8.1, float(np.ceil(np.max(np.abs(values)))))
    cmap = LinearSegmentedColormap.from_list("delta", ["#7884B4", "#F7F7F7", "#D58A9F"])
    image = ax.pcolormesh(
        np.arange(values.shape[1] + 1) - 0.5,
        np.arange(values.shape[0] + 1) - 0.5,
        values, cmap=cmap, vmin=-limit, vmax=limit, shading="flat",
        edgecolors="white", linewidth=0.8,
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(MODELS)), [LABELS[model] for model in MODELS])
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    ax.set_ylim(values.shape[0] - 0.5, -0.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    fontsize=7 if compact else 8,
                    color="white" if abs(value) > limit * 0.60 else "#252525")
    ax.set_title("高覆盖终点相对起点的变化", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows, coverage, summary = aggregate()
    write_data(rows, coverage, summary)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    plot_coverage(ax, coverage)
    fig.suptitle("COCO起始集 + Open Images V7按需补图", fontsize=13, fontweight="bold")
    save(fig, "fig1_coverage_gradient")

    for metric, stem, title in (("macro_f1", "fig2_macro_f1_curves", "Macro-F1"),
                                ("dog_f1", "fig3_dog_f1_curves", "dog F1")):
        fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
        plot_model_lines(ax, coverage, summary, metric, title)
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
        fig.suptitle("8个模型，均值 ± 样本标准差（3个训练种子）", fontsize=13, fontweight="bold")
        save(fig, stem)

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    image = plot_delta_heatmap(ax, summary)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    colorbar.set_label("91.08%覆盖 - 68.64%覆盖（百分点）")
    colorbar.solids.set_rasterized(False)
    fig.suptitle("高覆盖率对不同架构产生不同影响", fontsize=13, fontweight="bold")
    save(fig, "fig4_endpoint_delta_heatmap")

    fig = plt.figure(figsize=(13.33, 8.0))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(0.92, 1.08), height_ratios=(1, 1),
        left=0.07, right=0.96, top=0.88, bottom=0.22, hspace=0.39, wspace=0.25,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    plot_coverage(axes[0], coverage, compact=True)
    plot_model_lines(axes[1], coverage, summary, "macro_f1", "Macro-F1", compact=True)
    plot_model_lines(axes[2], coverage, summary, "dog_f1", "dog F1", compact=True)
    image = plot_delta_heatmap(axes[3], summary, compact=True)
    for ax, label in zip(axes, "ABCD"):
        panel_label(ax, label)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.035), fontsize=7.2)
    colorbar = fig.colorbar(image, ax=axes[3], fraction=0.032, pad=0.025)
    colorbar.set_label("终点 - 起点（pp）")
    colorbar.solids.set_rasterized(False)
    fig.suptitle("覆盖率提高并不保证模型性能单调提升", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.115, "COCO 300张/类起始集；Open Images V7按需等比例补图；固定COCO测试集",
             ha="center", color="#555555", fontsize=8)
    save(fig, "fig0_overview_four_panel")
    print(f"wrote figures to {OUTPUT}")


if __name__ == "__main__":
    main()
