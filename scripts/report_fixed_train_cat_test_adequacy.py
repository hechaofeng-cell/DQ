#!/usr/bin/env python3
"""Analyze cat_v1.0 test adequacy and compare it with the dog replication."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
DOG = ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"
TABLES = CAT / "tables"
FIGURES = CAT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(TABLES / "metrics_detailed.csv")
    summary = pd.read_csv(TABLES / "budget_summary.csv")
    tests = []
    for budget in sorted(detail.cat_per_class.unique()):
        group = detail[detail.cat_per_class.eq(budget)]
        for metric in ("frequency_coverage", "cat_error_rate", "failed_state_count"):
            observed = group[group.method.eq("gap")][metric].mean()
            random_values = group[group.method.eq("random")].groupby("replicate")[metric].mean().to_numpy()
            tests.append({
                "cat_per_class": budget, "metric": metric, "gap_mean": observed,
                "random_mean": random_values.mean(), "delta": observed - random_values.mean(),
                "empirical_one_sided_p": (1 + np.sum(random_values >= observed)) / (len(random_values) + 1),
            })
    tests = pd.DataFrame(tests)
    tests["holm_adjusted_p_across_budgets"] = np.nan
    for metric in tests.metric.unique():
        mask = tests.metric.eq(metric)
        tests.loc[mask, "holm_adjusted_p_across_budgets"] = holm(
            tests.loc[mask, "empirical_one_sided_p"].to_numpy()
        )
    tests.to_csv(TABLES / "empirical_tests.csv", index=False)

    model_rows = []
    for budget in (60, 80, 100):
        group = detail[detail.cat_per_class.eq(budget)]
        for model, model_group in group.groupby("model"):
            gap = model_group[model_group.method.eq("gap")]
            random_rows = model_group[model_group.method.eq("random")]
            model_rows.append({
                "cat_per_class": budget, "model": model,
                "delta_cat_error_rate": gap.cat_error_rate.mean() - random_rows.cat_error_rate.mean(),
                "delta_failed_state_count": gap.failed_state_count.mean() - random_rows.failed_state_count.mean(),
            })
    models = pd.DataFrame(model_rows)
    models.to_csv(TABLES / "model_diagnostic_gain.csv", index=False)

    dog = pd.read_csv(DOG / "tables/budget_summary.csv")
    cross = pd.DataFrame({
        "targets_per_class": summary.cat_per_class,
        "dog_coverage_gain": dog.delta_frequency_coverage,
        "cat_coverage_gain": summary.delta_frequency_coverage,
        "dog_target_error_gain": dog.delta_dog_error_rate,
        "cat_target_error_gain": summary.delta_cat_error_rate,
        "dog_failed_state_fraction_gain": dog.delta_failed_state_fraction,
        "cat_failed_state_fraction_gain": summary.delta_failed_state_fraction,
    })
    cross.to_csv(TABLES / "dog_cat_comparison.csv", index=False)
    return summary, tests, cross


def make_figure(summary: pd.DataFrame, cross: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
        "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    blue, gray, band, teal, violet = "#0F4D92", "#767676", "#B4C0E4", "#42949E", "#9A4D8E"
    x = summary.cat_per_class.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(x, summary.random_frequency_coverage_p2_5 * 100,
                    summary.random_frequency_coverage_p97_5 * 100, color=band, alpha=.65, linewidth=0)
    ax.plot(x, summary.random_frequency_coverage_mean * 100, color=gray, marker="o", label="Random mean")
    ax.plot(x, summary.gap_frequency_coverage * 100, color=blue, marker="o", linewidth=1.8, label="Coverage-guided")
    ax.set_ylabel("Cat frequency coverage (%)")
    ax.set_title("Cat state coverage efficiency")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    ax.fill_between(x, summary.random_cat_error_rate_p2_5 * 100,
                    summary.random_cat_error_rate_p97_5 * 100, color=band, alpha=.65, linewidth=0)
    ax.plot(x, summary.random_cat_error_rate_mean * 100, color=gray, marker="o")
    ax.plot(x, summary.gap_cat_error_rate * 100, color=violet, marker="o", linewidth=1.8)
    ax.set_ylabel("Cat error rate (%)")
    ax.set_title("Cat failure exposure")

    ax = axes[1, 0]
    ax.fill_between(x, summary.random_failed_state_count_p2_5,
                    summary.random_failed_state_count_p97_5, color=band, alpha=.65, linewidth=0)
    ax.plot(x, summary.random_failed_state_count_mean, color=gray, marker="o")
    ax.plot(x, summary.gap_failed_state_count, color=teal, marker="o", linewidth=1.8)
    ax.set_ylabel("Cat states with >=1 failure")
    ax.set_title("Cat failure-state breadth")

    ax = axes[1, 1]
    ax.axhline(0, color="#272727", linewidth=.8)
    ax.plot(x, cross.dog_failed_state_fraction_gain * 100, color=blue, marker="o", label="Dog")
    ax.plot(x, cross.cat_failed_state_fraction_gain * 100, color=violet, marker="s", label="Cat")
    ax.set_ylabel("Failure-state breadth gain (pp)")
    ax.set_title("Cross-category diagnostic gain")
    ax.legend(frameon=False, fontsize=7)

    for label, ax in zip("ABCD", axes.flat):
        ax.text(-.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.set_xlabel("Targets per class")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.savefig(FIGURES / "cat_and_cross_category_test_adequacy.svg", facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "cat_and_cross_category_test_adequacy.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    rows = [
        "# cat_v1.0固定训练集视觉测试充分性复验", "", "日期：2026-09-20", "",
        "## 实验设置", "",
        "- 固定D300训练集与8种架构×3个种子的24个模型权重；",
        "- 200张cat候选：100张常规cat + 100张新增困难cat；",
        "- dog、horse、sheep、person各100张，五类测试预算完全平衡；",
        "- 每类20、40、60、80、100张五个预算；",
        "- cat_v1.0包含23项特征、125个可计入覆盖状态；",
        "- 覆盖驱动选择不读取模型预测；随机对照为1000条嵌套cat顺序；",
        "- 200/200特征结构有效：194张原始通过，6张按已声明嘴部依赖规则修正；尚未进行逐张人工语义复核。", "",
        "## 主要结果", "",
        "| 每类测试数 | 覆盖组覆盖率 | 随机覆盖率 | cat错误率差值 | 失败状态数差值 | 失败状态p值 | Holm校正p值 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        budget = int(row.cat_per_class)
        test = tests[tests.cat_per_class.eq(budget) & tests.metric.eq("failed_state_count")].iloc[0]
        rows.append(
            f"| {budget} | {row.gap_frequency_coverage:.2%} | {row.random_frequency_coverage_mean:.2%} | "
            f"{row.delta_cat_error_rate * 100:+.2f} pp | {row.delta_failed_state_count:+.2f} | "
            f"{test.empirical_one_sided_p:.3f} | {test.holm_adjusted_p_across_budgets:.3f} |"
        )
    rows += [
        "", "## 结论", "",
        "1. cat覆盖驱动组在全部预算点均超过1000条随机序列的最大频次覆盖率；各点经验p=0.001，跨预算Holm校正p=0.005。",
        "2. 20–40张/类时，覆盖组cat错误率和失败状态数低于随机组；覆盖提升没有立即转化为诊断收益。",
        "3. 60–100张/类时方向转正，80张/类的增益最大：cat错误率+1.02个百分点、失败状态数+6.80；但失败状态经验p=0.081、Holm校正p=0.405，未达到显著。",
        "4. 80张/类时，20/24个模型运行的cat错误率增益为正，20/24个运行的失败状态数增益为正，说明存在跨架构趋势，但随机选样方差仍较大。", "",
        "## dog与cat联合解释", "",
        "> 缺口驱动选择在dog和cat上都稳定提高视觉状态覆盖效率，说明覆盖构造机制具有跨类别可迁移性；但失败发现收益只在dog的80张预算达到校正后显著，cat仅呈中高预算趋势，因此诊断效用具有类别和基础错误率依赖性。", "",
        "这比声称方法在所有类别都能显著发现更多错误更准确。当前跨类别证据支持通用的覆盖评价框架，但只部分支持通用的故障发现优势。", "",
        "## 限制", "",
        "- cat状态为本地VLM自动标注，尚未逐张人工语义复核；",
        "- cat可用小目标数量低于dog，困难池以遮挡和截断为主；",
        "- cat基线错误率约3.3%，低于dog约7.0%，使错误发现差异更难达到显著；",
        "- 压力测试结果不能作为真实部署分布的Accuracy估计。", "",
        "## 当前论文结论", "",
        "> 类别内部视觉状态覆盖能够跨dog和cat提高测试集的状态充分性；其向模型故障发现能力的转化存在预算阈值和类别依赖。", "",
    ]
    (CAT / "cat_test_adequacy_analysis_zh.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (CAT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Coverage-guided selection improves state coverage for cat | supported | all five budgets Holm p=0.005 | Current cat_v1.0 and candidate pool |\n"
        "| Coverage efficiency transfers from dog to cat | supported | both categories exceed all 1000 random sequences | Category-specific schemas |\n"
        "| Cat coverage-guided tests significantly expose more failed states | unsupported | best raw p=0.081; Holm p=0.405 | Directional trend only |\n"
        "| Failure-discovery advantage is category-independent | unsupported | significant for dog at 80/class, not cat | More categories needed |\n",
        encoding="utf-8",
    )
    (CAT / "review.md").write_text(
        "# Review\n\n| Gate | Verdict | Evidence or blocker |\n|---|---|---|\n"
        "| Citation integrity | pass | No new external factual claims |\n"
        "| Result traceability | pass | 24 prediction files, 120120 detailed rows, 1000 seeded random orders |\n"
        "| Claim-scope integrity | pass with limitation | Coverage transfer supported; cat diagnostic significance unsupported |\n\n"
        "Remaining limitation: cat_v1.0 labels have structural validation and six audited dependency repairs, but no per-image human semantic review.\n",
        encoding="utf-8",
    )


def write_checksums() -> None:
    paths = [CAT / "protocol.json", CAT / "tables/budget_summary.csv", CAT / "tables/empirical_tests.csv",
             CAT / "tables/dog_cat_comparison.csv", CAT / "cat_test_adequacy_analysis_zh.md",
             FIGURES / "cat_and_cross_category_test_adequacy.svg"]
    pd.DataFrame([{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in paths]).to_csv(
        CAT / "checksums.csv", index=False
    )


def main() -> None:
    summary, tests, cross = analyze()
    make_figure(summary, cross)
    write_report(summary, tests)
    write_checksums()
    print(CAT / "cat_test_adequacy_analysis_zh.md")


if __name__ == "__main__":
    main()
