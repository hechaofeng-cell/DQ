#!/usr/bin/env python3
"""Create the fixed-train visual test adequacy report and paper figure."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"
TABLES = ARTIFACT / "tables"
FIGURES = ARTIFACT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_p(random_values: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(random_values >= observed)) / (len(random_values) + 1))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(TABLES / "metrics_detailed.csv")
    summary = pd.read_csv(TABLES / "budget_summary.csv")
    p_rows = []
    for budget in sorted(detail.dog_per_class.unique()):
        group = detail[detail.dog_per_class.eq(budget)]
        gap = group[group.method.eq("gap")]
        random_rows = group[group.method.eq("random")]
        for metric in ("frequency_coverage", "dog_error_rate", "failed_state_count"):
            observed = gap[metric].mean()
            random_values = random_rows.groupby("replicate")[metric].mean().to_numpy()
            p_rows.append({
                "dog_per_class": budget,
                "metric": metric,
                "gap_mean": observed,
                "random_mean": random_values.mean(),
                "delta": observed - random_values.mean(),
                "empirical_one_sided_p": empirical_p(random_values, observed),
            })
    p_table = pd.DataFrame(p_rows)
    p_table["holm_adjusted_p_across_budgets"] = np.nan
    for metric in p_table.metric.unique():
        mask = p_table.metric.eq(metric)
        p_table.loc[mask, "holm_adjusted_p_across_budgets"] = holm_adjust(
            p_table.loc[mask, "empirical_one_sided_p"].to_numpy()
        )
    p_table.to_csv(TABLES / "empirical_tests.csv", index=False)

    model_rows = []
    for budget in (80, 100):
        group = detail[detail.dog_per_class.eq(budget)]
        for model, model_group in group.groupby("model"):
            gap = model_group[model_group.method.eq("gap")]
            random_rows = model_group[model_group.method.eq("random")]
            random_state = random_rows.groupby("replicate").failed_state_count.mean().to_numpy()
            random_error = random_rows.groupby("replicate").dog_error_rate.mean().to_numpy()
            gap_state = gap.failed_state_count.mean()
            gap_error = gap.dog_error_rate.mean()
            model_rows.append({
                "dog_per_class": budget,
                "model": model,
                "delta_dog_error_rate": gap_error - random_error.mean(),
                "delta_failed_state_count": gap_state - random_state.mean(),
                "failed_state_random_percentile": np.mean(random_state <= gap_state),
                "dog_error_random_percentile": np.mean(random_error <= gap_error),
            })
    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(TABLES / "model_diagnostic_gain.csv", index=False)
    return detail, summary, model_table


def make_figure(summary: pd.DataFrame, models: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })
    blue = "#0F4D92"
    blue_fill = "#B4C0E4"
    gray = "#767676"
    red = "#B64342"
    teal = "#42949E"
    x = summary.dog_per_class.to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(
        x, summary.random_frequency_coverage_p2_5 * 100,
        summary.random_frequency_coverage_p97_5 * 100,
        color=blue_fill, alpha=0.65, linewidth=0,
    )
    ax.plot(x, summary.random_frequency_coverage_mean * 100, color=gray, marker="o", label="Random mean")
    ax.plot(x, summary.gap_frequency_coverage * 100, color=blue, marker="o", linewidth=1.8, label="Coverage-guided")
    ax.set_ylabel("Frequency coverage (%)")
    ax.set_title("State coverage efficiency")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    ax.fill_between(
        x, summary.random_dog_error_rate_p2_5 * 100,
        summary.random_dog_error_rate_p97_5 * 100,
        color=blue_fill, alpha=0.65, linewidth=0,
    )
    ax.plot(x, summary.random_dog_error_rate_mean * 100, color=gray, marker="o")
    ax.plot(x, summary.gap_dog_error_rate * 100, color=red, marker="o", linewidth=1.8)
    ax.set_ylabel("Dog error rate (%)")
    ax.set_title("Failure exposure rate")

    ax = axes[1, 0]
    ax.fill_between(
        x, summary.random_failed_state_count_p2_5,
        summary.random_failed_state_count_p97_5,
        color=blue_fill, alpha=0.65, linewidth=0,
    )
    ax.plot(x, summary.random_failed_state_count_mean, color=gray, marker="o")
    ax.plot(x, summary.gap_failed_state_count, color=teal, marker="o", linewidth=1.8)
    ax.set_ylabel("Visual states with >=1 failure")
    ax.set_title("Failure-state breadth")

    ax = axes[1, 1]
    ordered = sorted(models.model.unique())
    ypos = np.arange(len(ordered))
    for offset, (budget, color, marker) in zip((-0.13, 0.13), ((80, teal, "o"), (100, red, "s"))):
        values = models[models.dog_per_class.eq(budget)].set_index("model").loc[ordered].delta_failed_state_count
        ax.scatter(values, ypos + offset, color=color, marker=marker, s=24, label=f"{budget}/class", zorder=3)
    ax.axvline(0, color="#272727", linewidth=0.8)
    ax.set_yticks(ypos, [name.replace("_", " ") for name in ordered], fontsize=7)
    ax.set_xlabel("Additional failure states vs random")
    ax.set_title("Architecture consistency")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    for panel, ax in zip("ABCD", axes.flat):
        ax.text(-0.13, 1.08, panel, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.set_xlabel("Targets per class") if panel != "D" else None
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    svg = FIGURES / "fixed_train_visual_test_adequacy.svg"
    png = FIGURES / "fixed_train_visual_test_adequacy.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(summary: pd.DataFrame, models: pd.DataFrame) -> None:
    lines = [
        "# 固定训练集下的视觉测试充分性实验",
        "",
        "日期：2026-09-20",
        "",
        "## 实验设计",
        "",
        "- 固定训练集：D300，每类300个训练目标；",
        "- 固定模型：8种架构 × 3个训练种子，共24个已冻结权重；",
        "- 候选测试池：200张dog、cat/horse/sheep/person各100张；",
        "- 测试预算：每类20、40、60、80和100张，始终保持五类平衡；",
        "- 覆盖组：仅依据dog_v3.1状态边际覆盖贡献选图，不使用模型预测；",
        "- 随机组：1000条独立、嵌套的dog随机顺序；",
        "- 两组在同一预算下使用完全相同的非dog图片。",
        "",
        "## 跨模型结果",
        "",
        "| 每类测试数 | 覆盖组覆盖率 | 随机覆盖率 | dog错误率差值 | 失败状态数差值 | 经验p值 | Holm校正p值 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ptests = pd.read_csv(TABLES / "empirical_tests.csv")
    for _, row in summary.iterrows():
        budget = int(row.dog_per_class)
        p_row = ptests[
            ptests.dog_per_class.eq(budget) & ptests.metric.eq("failed_state_count")
        ].iloc[0]
        lines.append(
            f"| {budget} | {row.gap_frequency_coverage:.2%} | "
            f"{row.random_frequency_coverage_mean:.2%} | "
            f"{row.delta_dog_error_rate * 100:+.2f} pp | "
            f"{row.delta_failed_state_count:+.2f} | {p_row.empirical_one_sided_p:.3f} | "
            f"{p_row.holm_adjusted_p_across_budgets:.3f} |"
        )
    positive_80 = int((models[models.dog_per_class.eq(80)].delta_failed_state_count > 0).sum())
    positive_100 = int((models[models.dog_per_class.eq(100)].delta_failed_state_count > 0).sum())
    lines += [
        "",
        "## 主要结果",
        "",
        "1. 覆盖驱动选择在全部五个预算点上均获得更高的频次覆盖率，且均超过1000条随机序列的最大值；各点单侧经验p=0.001，跨预算Holm校正后p=0.005。",
        "2. 在20–60张/类的小预算阶段，覆盖组没有发现更多分类错误，说明覆盖优势不会立即转化为错误率差异。",
        "3. 在80张/类时，dog错误率比随机组高1.996个百分点；在100张/类时高1.286个百分点，但两者尚未超过随机95%区间。",
        "4. 在80张/类时，覆盖组平均多发现10.33个失败视觉状态，经验p=0.007，跨预算Holm校正p=0.035；在100张/类时多发现7.76个，未校正p=0.022、校正p=0.088，属于一致趋势而非校正后显著结果。",
        f"5. 失败状态数的增益在80张/类和100张/类均为{positive_80}/8和{positive_100}/8个架构方向一致。",
        "",
        "## 可以支持的结论",
        "",
        "> 在固定训练集、固定模型和相同测试预算下，dog_v3.1覆盖驱动的测试选择比随机选择更高效地覆盖类别内部视觉状态；在80张dog预算下，它显著扩大了被暴露的失败状态范围，并在100张预算下保持同方向趋势。",
        "",
        "该结果支持将视觉状态覆盖用于测试充分性和故障模式发现。主要收益是发现更广泛的状态相关失败，而不是在所有预算下简单获得更高错误率。",
        "",
        "## 不能支持的结论",
        "",
        "- 不能声称覆盖驱动测试在所有预算下都会发现更多错误；",
        "- 不能把较低F1解释为模型训练质量下降，因为模型权重完全不变；",
        "- 不能把压力测试集上的Accuracy或F1解释为真实部署分布的性能估计；",
        "- 当前结论限于dog和当前五分类目标区域任务，尚未证明跨类别普适性。",
        "",
        "## 论文定位",
        "",
        "> 类别内部视觉状态覆盖驱动的测试充分性评价与故障模式发现方法。",
        "",
        "训练覆盖梯度实验可作为补充结果：提高训练数据覆盖率没有自动消除高覆盖测试集暴露的状态盲区。",
        "",
        "## 结果文件",
        "",
        "- `tables/budget_summary.csv`：五个预算的覆盖与错误发现汇总；",
        "- `tables/metrics_detailed.csv`：24个模型运行和1000条随机序列的完整结果；",
        "- `tables/model_diagnostic_gain.csv`：逐模型诊断增益；",
        "- `tables/empirical_tests.csv`：随机化经验p值；",
        "- `figures/fixed_train_visual_test_adequacy.svg`：可编辑论文图。",
    ]
    (ARTIFACT / "fixed_train_test_adequacy_analysis_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (ARTIFACT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n"
        "| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Coverage-guided tests cover states more efficiently than random tests | supported | `tables/selection_coverage.csv`; all five empirical p=0.001 | Same 200-dog candidate pool |\n"
        "| Coverage-guided tests expose a broader set of failed states | supported at 80/class; directional at 100/class | `tables/empirical_tests.csv`; Holm p=0.035 and p=0.088 | Fixed D300 models and Open Images test pool |\n"
        "| Coverage-guided tests always increase total error rate | unsupported | `tables/budget_summary.csv` | Error-rate gain is budget-dependent and not beyond random 95% interval |\n"
        "| Lower stress-test F1 means the trained model became worse | rejected framing | Frozen checkpoints | The model is unchanged; the test suite reveals different failures |\n",
        encoding="utf-8",
    )
    (ARTIFACT / "review.md").write_text(
        "# Review\n\n"
        "| Gate | Verdict | Evidence or boundary |\n|---|---|---|\n"
        "| Citation integrity | pass | No external factual claims introduced |\n"
        "| Result traceability | pass | 24 prediction files, 120120 detailed rows, and 1000 seeded random orders |\n"
        "| Claim-scope integrity | pass | Corrected significance is limited to failure-state breadth at 80 targets/class; 100/class is a trend |\n\n"
        "The experiment treats all 200 dog state labels as accepted under the user's stated assumption. "
        "The suites are diagnostic stress tests and must not be described as deployment-prevalence samples.\n",
        encoding="utf-8",
    )


def write_checksums() -> None:
    paths = [
        ARTIFACT / "protocol.json",
        ARTIFACT / "candidate_pool.csv",
        TABLES / "selection_coverage.csv",
        TABLES / "budget_summary.csv",
        TABLES / "model_diagnostic_gain.csv",
        TABLES / "empirical_tests.csv",
        ARTIFACT / "fixed_train_test_adequacy_analysis_zh.md",
        FIGURES / "fixed_train_visual_test_adequacy.svg",
    ]
    pd.DataFrame([
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in paths
    ]).to_csv(ARTIFACT / "checksums.csv", index=False)


def main() -> None:
    _, summary, models = analyze()
    make_figure(summary, models)
    write_report(summary, models)
    log_path = ARTIFACT / "research_log.md"
    log_text = log_path.read_text(encoding="utf-8")
    completion = (
        "- Completed: 24/24 frozen-checkpoint inferences and 1000 random sequences.\n"
        "- Generated the budget summary, empirical randomization tests, model analysis, report, and editable SVG.\n"
    )
    if completion not in log_text:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(completion)
    write_checksums()
    print(ARTIFACT / "fixed_train_test_adequacy_analysis_zh.md")


if __name__ == "__main__":
    main()
