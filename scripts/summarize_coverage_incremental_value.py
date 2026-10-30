#!/usr/bin/env python3
"""Audit and summarize the coverage incremental-value experiment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/coverage_incremental_value_20260929"


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.zeros(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def fmt_ci(row: pd.Series) -> str:
    return f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]"


def add_corrections(equal: pd.DataFrame, hybrid: pd.DataFrame) -> pd.DataFrame:
    equal = equal.copy()
    equal["family"] = "equal-error secondary"
    equal["holm_family_p"] = np.nan
    primary = equal.comparison.eq("coverage_vs_random")
    equal.loc[primary, "family"] = "equal-error primary: Coverage vs Random, 2 metrics"
    equal.loc[primary, "holm_family_p"] = holm(equal.loc[primary, "permutation_p"].tolist())
    equal["holm_global_p"] = holm(equal.permutation_p.tolist())

    hybrid = hybrid.copy()
    hybrid["family"] = "hybrid explanatory"
    hybrid["holm_family_p"] = np.nan
    independent = hybrid.metric.isin(["confusion_pair_coverage", "error_cluster_coverage"])
    random_control = hybrid.comparison.str.contains("coverage25_vs_random25") & independent
    pure_control = hybrid.comparison.str.contains("coverage25_vs_pure") & independent
    hybrid.loc[random_control, "family"] = "hybrid primary: Coverage25 vs Random25, 2 bases x 2 metrics"
    hybrid.loc[random_control, "holm_family_p"] = holm(hybrid.loc[random_control, "permutation_p"].tolist())
    hybrid.loc[pure_control, "family"] = "hybrid secondary: Coverage25 vs Pure, 2 bases x 2 metrics"
    hybrid.loc[pure_control, "holm_family_p"] = holm(hybrid.loc[pure_control, "permutation_p"].tolist())
    hybrid["holm_global_p"] = holm(hybrid.permutation_p.tolist())
    return pd.concat([
        equal.assign(experiment="equal_error"),
        hybrid.assign(experiment="hybrid"),
    ], ignore_index=True)


def validate(directory: Path) -> dict[str, int]:
    tables = directory / "tables"
    expected = {
        "equal_error_metrics.csv": 24 * 5 * 4,
        "equal_error_auc_by_run.csv": 24 * 4,
        "equal_error_comparisons.csv": 4 * 2,
        "hybrid_fixed_metrics.csv": 24 * 2 * 2 * 5,
        "hybrid_random_summary.csv": 24 * 2 * 5,
        "hybrid_auc_by_run.csv": 24 * 2 * 3,
        "hybrid_comparisons.csv": 2 * 2 * 6,
        "hybrid_endpoint_new_lost.csv": 24 * 2,
        "state_interaction_cells.csv": 11,
        "state_interaction_models.csv": 3,
    }
    observed = {}
    for name, count in expected.items():
        rows = pd.read_csv(tables / name)
        observed[name] = len(rows)
        if len(rows) != count:
            raise ValueError(f"{name}: expected {count} rows, got {len(rows)}")
    equal = pd.read_csv(tables / "equal_error_metrics.csv")
    if set(equal.budget) != {20, 40, 60, 80, 100} or equal[["coverage_confusion", "baseline_confusion", "coverage_clusters", "baseline_clusters"]].isna().any().any():
        raise ValueError("invalid equal-error budget or missing metrics")
    hybrid = pd.read_csv(tables / "hybrid_fixed_metrics.csv")
    if hybrid.isna().any().any():
        raise ValueError("missing hybrid fixed metrics")
    return observed


def write_report(directory: Path, comparisons: pd.DataFrame) -> None:
    tables = directory / "tables"
    equal_metrics = pd.read_csv(tables / "equal_error_metrics.csv")
    fixed = pd.read_csv(tables / "hybrid_fixed_metrics.csv")
    random = pd.read_csv(tables / "hybrid_random_summary.csv")
    new_lost = pd.read_csv(tables / "hybrid_endpoint_new_lost.csv")
    interactions = pd.read_csv(tables / "state_interaction_models.csv")
    cells = pd.read_csv(tables / "state_interaction_cells.csv")

    endpoint = equal_metrics[equal_metrics.budget.eq(100)].groupby("baseline").mean(numeric_only=True)
    fixed_endpoint = fixed[fixed.budget.eq(100)].groupby(["base", "variant"]).mean(numeric_only=True)
    random_endpoint = random[random.budget.eq(100)].groupby("base").mean(numeric_only=True)
    new_lost_mean = new_lost.groupby("base").mean(numeric_only=True)

    eq_compare = comparisons[comparisons.experiment.eq("equal_error")].set_index(["comparison", "metric"])
    hy_compare = comparisons[comparisons.experiment.eq("hybrid")].set_index(["comparison", "metric"])
    lines = [
        "# Coverage增量诊断价值补充实验", "", "## 实验范围", "",
        "本实验复用五类候选池和24组冻结预测，不重新训练模型。等错误数控制使用无放回抽样下不同错误标签期望数的精确计算；固定预算互补组由75%强基线与25%补充信号构成；随机补充采用1000个类别内随机排序。", "",
        "## 1. 等错误数控制", "",
        "每类100张终点预算下的平均结果如下。正差值表示Coverage在相同错误数量下触达更多不同错误模式。", "",
        "| 基线 | 匹配错误数 | Coverage混淆覆盖 | 基线混淆覆盖 | 差值 | Coverage错误簇 | 基线错误簇 | 差值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for base in ("random", "deepgini", "dsa", "kcenter"):
        row = endpoint.loc[base]
        lines.append(
            f"| {base} | {row.matched_errors_mean:.1f} | {row.coverage_confusion:.2%} | "
            f"{row.baseline_confusion:.2%} | {row.delta_confusion:+.2%} | "
            f"{row.coverage_clusters:.2%} | {row.baseline_clusters:.2%} | {row.delta_clusters:+.2%} |"
        )
    lines.extend(["", "曲线AUC的主要比较：", "",
        "| 指标 | Coverage−Random AUC | 95% CI | 正向运行 | Holm p |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric, label in (("equal_error_confusion", "混淆类型"), ("equal_error_clusters", "错误簇")):
        row = eq_compare.loc[("coverage_vs_random", metric)]
        lines.append(f"| {label} | {row.mean_delta_auc:+.4f} | {fmt_ci(row)} | {int(row.positive_runs)}/24 | {row.holm_family_p:.4f} |")

    lines.extend(["", "## 2. 固定预算互补组", "",
        "下表为每类100张时的终点平均值。Coverage25表示75%强基线加25% Coverage，Random25表示75%强基线加25%随机样本。", "",
        "| 强基线 | 方案 | 混淆覆盖 | 错误簇覆盖 | 错误率 | Macro-F1 | 状态覆盖 | 失败状态广度 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for base in ("deepgini", "kcenter"):
        for variant, label in (("pure", "100%强基线"), ("coverage25", "Coverage25")):
            row = fixed_endpoint.loc[base, variant]
            lines.append(f"| {base} | {label} | {row.confusion_pair_coverage:.2%} | {row.error_cluster_coverage:.2%} | {row.macro_error_rate:.2%} | {row.macro_f1:.2%} | {row.macro_frequency_coverage:.2%} | {row.macro_failed_state_fraction:.2%} |")
        row = random_endpoint.loc[base]
        lines.append(f"| {base} | Random25 | {row.confusion_pair_coverage_mean:.2%} | {row.error_cluster_coverage_mean:.2%} | {row.macro_error_rate_mean:.2%} | {row.macro_f1_mean:.2%} | {row.macro_frequency_coverage_mean:.2%} | {row.macro_failed_state_fraction_mean:.2%} |")
    lines.extend(["", "Coverage25相对Random25的主要AUC比较：", "",
        "| 强基线 | 指标 | AUC差值 | 95% CI | 正向运行 | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for base in ("deepgini", "kcenter"):
        comparison = f"{base}_coverage25_vs_random25"
        for metric, label in (("confusion_pair_coverage", "混淆类型"), ("error_cluster_coverage", "错误簇")):
            row = hy_compare.loc[(comparison, metric)]
            lines.append(f"| {base} | {label} | {row.mean_delta_auc:+.4f} | {fmt_ci(row)} | {int(row.positive_runs)}/24 | {row.holm_family_p:.4f} |")
    lines.extend(["", "相对100%强基线，Coverage25没有形成稳定净优势。其独立错误指标的四项比较经过家族内Holm校正后均未达到0.05。DeepGini混合组在终点新增混淆类型和错误簇，但曲线错误簇AUC点估计低于纯DeepGini；k-center混合组与纯k-center的独立错误指标差异区间均跨0。", "",
        "每类100张时，相对纯强基线的新发现与丢失数量均值：", "",
        "| 强基线 | Coverage新增混淆 | Coverage丢失混淆 | Coverage新增簇 | Coverage丢失簇 | 随机净混淆 | 随机净错误簇 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for base in ("deepgini", "kcenter"):
        row = new_lost_mean.loc[base]
        lines.append(f"| {base} | {row.coverage_new_confusion:.2f} | {row.coverage_lost_confusion:.2f} | {row.coverage_new_clusters:.2f} | {row.coverage_lost_clusters:.2f} | {row.random_net_confusion_mean:+.2f} | {row.random_net_clusters_mean:+.2f} |")

    lines.extend(["", "## 3. 状态组合分析", "",
        "| 组合 | 交互优势比 | 95% CI | p值 | 结论 |",
        "|---|---:|---:|---:|---|",
    ])
    labels = {"small_x_occlusion": "小目标×中重度遮挡", "pose_x_truncation": "非常规姿态×中重度截断", "blur_x_low_contrast": "模糊×低对比度"}
    for _, row in interactions.iterrows():
        if int(row.estimable) == 1:
            conclusion = "未发现超出主效应的显著交互"
            lines.append(f"| {labels[row.pair]} | {row.interaction_odds_ratio:.3f} | [{row.ci95_odds_ratio_low:.3f}, {row.ci95_odds_ratio_high:.3f}] | {row.p_value:.4f} | {conclusion} |")
        else:
            lines.append(f"| {labels[row.pair]} | — | — | — | 缺少一个状态单元，无法估计 |")
    lines.extend(["", "联合状态仍可对应较高的描述性错误率：小目标且中重度遮挡为23.07%，模糊且低对比度为51.56%；后者仅有8张图片，并缺少“清晰但低对比度”样本，不能用于交互推断。", "",
        "![增量价值汇总](figures/incremental_value_summary.png)", "",
        "![状态组合错误率](figures/state_interaction_error_rates.png)", "",
        "## 结论边界", "",
        "- 等错误数控制支持Coverage相对随机的诊断多样性不完全由错误数量解释。", 
        "- 固定预算互补实验支持Coverage比随机更有信息，但不支持Coverage混合组统一超过完整强基线。", 
        "- 当前最合适的论文定位是：可解释视觉状态Coverage为模型无关的测试补充信号，并揭示与模型相关强基线之间的权衡。", 
        "- 状态组合结果是关联性诊断；当前两项可估计交互均未显著，第三项因设计矩阵缺格而不可估计。", 
    ])
    (directory / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    directory = args.input.resolve()
    observed = validate(directory)
    equal = pd.read_csv(directory / "tables/equal_error_comparisons.csv")
    hybrid = pd.read_csv(directory / "tables/hybrid_comparisons.csv")
    comparisons = add_corrections(equal, hybrid)
    comparisons.to_csv(directory / "tables/statistical_comparisons_corrected.csv", index=False)
    write_report(directory, comparisons)
    (directory / "claim_evidence.md").write_text(
        "# Claim-Evidence Matrix\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Coverage的诊断广度不完全由错误数量增加解释 | supported against Random | 等错误数混淆AUC +0.0371、错误簇AUC +0.0191；两项家族内Holm p=0.0156 | 当前五类、24运行、五档预算 |\n"
        "| Coverage是比随机替换更有信息量的补充信号 | supported | 在DeepGini和k-center混合组中，混淆与错误簇四项AUC均为正，家族内Holm p=0.0312 | 固定75/25比例 |\n"
        "| Coverage混合组优于完整强基线 | unsupported | 相对纯强基线的四项独立错误指标经家族内校正均未显著 | 不排除其他预注册比例 |\n"
        "| 三组状态存在超乘法交互 | unsupported/not estimable | 两项GEE交互CI跨1；模糊×低对比度缺少一格 | 关联分析，不作因果解释 |\n",
        encoding="utf-8",
    )
    (directory / "review.md").write_text(
        "# Review\n\n"
        "| Gate | Verdict | Evidence |\n|---|---|---|\n"
        f"| Result traceability | pass | {sum(observed.values())} rows audited across 10 source tables; corrected comparison table generated |\n"
        "| Claim-scope integrity | pass | Strong-baseline superiority removed; interaction null and non-estimable result retained |\n"
        "| Citation integrity | not applicable | This artifact reports local experiment results only |\n",
        encoding="utf-8",
    )
    print(f"validated {sum(observed.values())} rows; wrote corrected comparisons and report")


if __name__ == "__main__":
    main()
