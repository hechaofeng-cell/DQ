#!/usr/bin/env python3
"""Plot recent-five Q1-minus-Q5 F1 gaps using all three frozen runs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/recent_five_models_20261009/tables/risk_gaps_runs.csv"
LAYERS = ROOT / "artifacts/recent_five_models_20261009/tables/risk_layers_runs.csv"
OUT = ROOT / "artifacts/current_method_conclusions_20261012"
FONT = ROOT / "scripts/assets/NotoSansSC-Regular.ttf"
SEEDS = (20260916, 20260917, 20260918)
MODELS = (
    "yolo26s_cls", "pe_core_b16", "siglip2_base", "mobileclip2_s0", "aimv2_large"
)
NAMES = {
    "yolo26s_cls": "YOLO26-S-cls",
    "pe_core_b16": "PE-Core-B16",
    "siglip2_base": "SigLIP2-B",
    "mobileclip2_s0": "MobileCLIP2-S0",
    "aimv2_large": "AIMv2-L",
}
PANELS = tuple((pool, source) for pool in ("B", "C", "D") for source in ("local", "online"))


def summarize() -> pd.DataFrame:
    runs = pd.read_csv(SOURCE)
    layers = pd.read_csv(LAYERS)
    keys = ["model", "pool", "source", "seed"]
    assert not runs.duplicated(keys).any()
    assert len(runs) == len(MODELS) * len(PANELS) * len(SEEDS) == 90
    assert set(runs.model) == set(MODELS)
    assert set(runs.seed) == set(SEEDS)
    assert np.allclose(runs.q1_minus_q5, runs.q1_macro_f1 - runs.q5_macro_f1, atol=1e-12)
    records = []
    for model in MODELS:
        for pool, source in PANELS:
            selected = runs[(runs.model == model) & (runs.pool == pool) & (runs.source == source)].set_index("seed").loc[list(SEEDS)]
            assert len(selected) == 3
            supports = layers[(layers.model == model) & (layers.pool == pool) & (layers.source == source)].support.unique()
            assert len(supports) == 1
            assert supports[0] == (120 if pool == "C" else 480)
            row = {
                "model": model,
                "model_label": NAMES[model],
                "pool": pool,
                "source": source,
                "targets_per_full_layer": int(supports[0]),
                "seed_count": len(selected),
                "seeds": ";".join(map(str, SEEDS)),
                "q1_macro_f1_mean": selected.q1_macro_f1.mean(),
                "q5_macro_f1_mean": selected.q5_macro_f1.mean(),
                "q1_minus_q5_mean_pp": 100 * selected.q1_minus_q5.mean(),
                "q1_minus_q5_seed_sd_pp": 100 * selected.q1_minus_q5.std(ddof=1),
            }
            for seed in SEEDS:
                row[f"seed_{seed}_q1_macro_f1"] = selected.loc[seed, "q1_macro_f1"]
                row[f"seed_{seed}_q5_macro_f1"] = selected.loc[seed, "q5_macro_f1"]
                row[f"seed_{seed}_gap_pp"] = 100 * selected.loc[seed, "q1_minus_q5"]
            records.append(row)
    summary = pd.DataFrame(records)
    assert len(summary) == 30 and summary.seed_count.eq(3).all()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "tables/recent_models_gap.csv", index=False)
    return summary


def main() -> None:
    summary = summarize()
    # Read the saved summary table so the figure and delivered CSV share a source.
    saved = pd.read_csv(OUT / "tables/recent_models_gap.csv")
    assert np.allclose(saved.q1_minus_q5_mean_pp, summary.q1_minus_q5_mean_pp, atol=1e-12)
    indexed = saved.set_index(["model", "pool", "source"])
    values = np.array([[indexed.loc[(model, pool, source), "q1_minus_q5_mean_pp"] for pool, source in PANELS] for model in MODELS])
    assert values.shape == (5, 6) and np.isfinite(values).all()
    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })
    cmap = LinearSegmentedColormap.from_list("risk_gap_blue", ["#F4F9FD", "#74A9CF", "#124A77"])
    fig, ax = plt.subplots(figsize=(9.4, 4.45))
    fig.subplots_adjust(left=0.205, right=0.865, top=0.79, bottom=0.16)
    mesh = ax.pcolormesh(np.arange(7), np.arange(6), values, cmap=cmap, vmin=0, vmax=45, shading="flat", edgecolors="white", linewidth=1, rasterized=False)
    ax.set_ylim(5, 0)
    ax.set_xticks(np.arange(6) + 0.5, [f"{pool}\n{'本地' if source == 'local' else '线上'}" for pool, source in PANELS])
    ax.set_yticks(np.arange(5) + 0.5, [NAMES[model] for model in MODELS])
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(5):
        for col in range(6):
            ax.text(col + 0.5, row + 0.5, f"{values[row, col]:.1f}", ha="center", va="center", fontsize=11, color="white" if values[row, col] >= 30 else "#102C41")
    cax = fig.add_axes([0.89, 0.16, 0.016, 0.63])
    colorbar = fig.colorbar(mesh, cax=cax, ticks=[0, 15, 30, 45])
    colorbar.solids.set_rasterized(False)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=9, width=0.6)
    colorbar.set_label("端点落差（百分点）", fontsize=10, labelpad=10)
    fig.suptitle("扩展验证：五模型的风险端点落差", fontsize=15, y=0.965)
    fig.text(0.52, 0.865, "Q1−Q5 Macro-F1（百分点）；三个训练种子均值", ha="center", fontsize=10, color="#4B4B4B")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"figures/06_recent_models_gap.{extension}", dpi=240)
    plt.close(fig)
    print(f"Verified 90 runs, 30 means, three seeds in every cell; range {values.min():.6f}–{values.max():.6f} pp.")


if __name__ == "__main__":
    main()
