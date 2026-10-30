#!/usr/bin/env python3
"""Reproduce conclusion figures and tables from completed, unchanged experiments."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/current_method_conclusions_20261012'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
POOLS = {
    'B': 'sixclass_independent_confirmation_20261004',
    'C': 'sixclass_voc_external_confirmation_20261005',
    'D': 'objects365_independent_confirmation_20261005',
}
POOL_LABELS = {'B': 'B · COCO', 'C': 'C · VOC2007', 'D': 'D · Objects365'}
SOURCES = {'local': '本地', 'online': '线上'}
SOURCE_COLORS = {'local': '#3775BA', 'online': '#B87543'}
GRAY = '#9CA3AA'
PURPLE = '#8A70A4'
READ_FILES = {}


def read(relative):
    path = ROOT / relative
    READ_FILES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def save_table(table, name):
    table.to_csv(TABLES / (name + '.csv'), index=False)
    return table


def setup():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    font = ROOT / 'scripts/assets/NotoSansSC-Regular.ttf'
    fontManager.addfont(str(font))
    family = FontProperties(fname=font).get_name()
    plt.rcParams.update({
        'font.family': family, 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'xtick.major.width': .8, 'ytick.major.width': .8,
        'legend.frameon': False, 'figure.facecolor': 'white',
        'axes.unicode_minus': False,
    })


def save_figure(figure, name):
    for extension in ('svg', 'png', 'pdf'):
        figure.savefig(FIGURES / (name + '.' + extension), dpi=300,
                       bbox_inches='tight', facecolor='white')
    plt.close(figure)


def aggregate():
    coverage, contrasts, states, transfer, cases, strong = [], [], [], [], [], []
    numeric_checks = 0
    for pool, directory in POOLS.items():
        prefix = 'artifacts/' + directory
        fixed = read(prefix + '/comparison/tables/fixed_metrics.csv')
        random = read(prefix + '/comparison/tables/random_metrics.csv')
        selected = fixed[fixed.method == 'coverage']
        for (source, budget), group in selected.groupby(['source', 'budget']):
            assert np.ptp(group.macro_frequency_coverage.to_numpy()) < 1e-12
            reference = random[(random.source == source) & (random.budget == budget)]
            assert len(group) == len(reference) == 24
            c, r = group.macro_frequency_coverage.mean(), reference.macro_frequency_coverage_mean.mean()
            coverage.append({'pool': pool, 'source': source, 'per_class_budget': budget,
                             'total_budget': budget * 6, 'coverage': c, 'random': r,
                             'delta_pp': 100 * (c - r)})
        comparisons = read(prefix + '/comparison/tables/comparisons.csv')
        equal_error = read(prefix + '/comparison/tables/equal_error_metrics.csv')
        for row in comparisons[comparisons.baseline == 'random'].itertuples():
            assert row.representation == 'classifier'
            subset = equal_error[(equal_error.source == row.source) &
                                 (equal_error.baseline == 'random') &
                                 (equal_error.representation == 'classifier')]
            auc = []
            for _, group in subset.groupby(['model', 'seed']):
                group = group.sort_values('budget')
                x = np.r_[0, group.budget.to_numpy()]
                y = np.r_[0, group['delta_' + row.metric].to_numpy()]
                auc.append(np.trapezoid(y, x) / x[-1])
            assert len(auc) == 24
            assert abs(np.mean(auc) - row.mean_delta_auc) < 1e-11
            numeric_checks += 1
            contrasts.append({'pool': pool, 'source': row.source, 'metric': row.metric,
                              'delta_auc': row.mean_delta_auc,
                              'ci_low': row.ci95_low, 'ci_high': row.ci95_high,
                              'holm_primary4_p': row.holm_primary4_p,
                              'holm_all20_p': row.holm_all20_p})
        for row in comparisons[comparisons.baseline.isin(['deepgini', 'dsa', 'kcenter'])].itertuples():
            strong.append({'pool': pool, 'source': row.source, 'baseline': row.baseline,
                           'metric': row.metric, 'delta_auc': row.mean_delta_auc,
                           'holm_strong12_p': row.holm_strong12_p})
        state = read(prefix + '/transfer/tables/state_validation.csv')
        grouped = state.groupby(['source', 'model', 'class_name', 'feature', 'value']).agg(
            support=('B_support', 'min'), gap=('B_gap', 'mean'), seed_count=('seed', 'nunique'))
        assert len(grouped) == 175 and (grouped.seed_count == 3).all()
        for source, group in grouped.groupby(level='source'):
            eligible = group.support >= 20
            states.append({'pool': pool, 'source': source, 'conditions': len(group),
                           'eligible': int(eligible.sum()),
                           'negative_mean': int((eligible & (group.gap < 0)).sum()),
                           'nonnegative_mean': int((eligible & (group.gap >= 0)).sum()),
                           'insufficient': int((~eligible).sum())})
        for source, group in state.groupby('source'):
            eligible = group.B_support >= 20
            states[-2 if source == 'local' else -1].update({
                'seed_rows': len(group),
                'negative_seed_rows': int((eligible & (group.B_gap < 0)).sum()),
                'nonnegative_seed_rows': int((eligible & (group.B_gap >= 0)).sum()),
                'insufficient_seed_rows': int((~eligible).sum())})
        comparison = read(prefix + '/transfer/tables/comparisons.csv')
        for row in comparison.itertuples():
            per_run = state[(state.source == row.source) & state.B_evaluable].groupby(
                ['model', 'seed']).B_gap.mean()
            architecture_mean = per_run.groupby(level='model').mean().mean()
            assert abs(architecture_mean - row.mean_delta_auc) < 1e-11
            numeric_checks += 1
            transfer.append({'pool': pool, 'source': row.source,
                             'recall_gap_pp': row.mean_delta_auc * 100,
                             'ci_low_pp': row.ci95_low * 100, 'ci_high_pp': row.ci95_high * 100,
                             'holm_two_p': row.holm_two_p,
                             'evaluable_architectures': per_run.index.get_level_values('model').nunique()})
        if pool == 'B':
            case_specs = [
                ('local', 'resnet50', 'dog', 'body_orientation', 'back', 'ResNet-50 · 背向dog'),
                ('local', 'resnet50', 'cat', 'target_scale', 'small', 'ResNet-50 · 小尺度cat'),
                ('online', 'resnet50', 'sheep', 'blur_level', 'severe', 'ResNet-50 · 严重模糊sheep'),
                ('local', 'densenet121', 'dog', 'target_scale', 'medium', 'DenseNet-121 · 中等尺度dog'),
            ]
            for source, model, cls, feature, value, label in case_specs:
                part = state[(state.source == source) & (state.model == model) &
                             (state.class_name == cls) & (state.feature == feature) & (state.value == value)]
                assert len(part) == 3 and part.B_support.nunique() == 1
                cases.append({'label': label, 'source': source, 'support': int(part.B_support.iloc[0]),
                              'class_recall': part.B_class_recall.mean(),
                              'state_recall': part.B_recall.mean(), 'gap_pp': part.B_gap.mean() * 100})
    save_table(pd.DataFrame(coverage), 'coverage_curves')
    save_table(pd.DataFrame(contrasts), 'diagnostic_auc_contrasts')
    save_table(pd.DataFrame(strong), 'strong_baseline_contrasts')
    save_table(pd.DataFrame(states), 'condition_support')
    save_table(pd.DataFrame(transfer), 'condition_recall_contrasts')
    save_table(pd.DataFrame(cases), 'condition_cases')
    risk = read('artifacts/random_stratification_ablation_20261006/tables/random_summary.csv')
    save_table(risk, 'risk_random_control')
    area = read('artifacts/scale_only_confirmation_20261007/tables/summary.csv')
    save_table(area, 'risk_area_control')
    area_conditions = read('artifacts/scale_only_confirmation_20261007/tables/area_condition_summary.csv')
    area_rows = []
    for (pool, source), group in area_conditions.groupby(['pool', 'source']):
        area_rows.append({'pool': pool, 'source': source, 'conditions': len(group),
                          'eligible': int(group.evaluable.sum()),
                          'negative': int((group.evaluable & (group.area_adjusted_recall_gap < 0)).sum())})
    save_table(pd.DataFrame(area_rows), 'area_condition_support')
    mixture = read('artifacts/controlled_risk_distribution_20261005/tables/panel_summary.csv')
    mixture = mixture[(mixture.total == 120) & (mixture.budget_role == 'common')].copy()
    assert len(mixture) == 126
    save_table(mixture, 'risk_mixture_curves')
    endpoints = []
    for (pool, source), group in mixture.groupby(['pool', 'source']):
        group = group.sort_values('high_risk_percent')
        assert np.array_equal(group.high_risk_percent.to_numpy(), np.arange(0, 101, 5))
        endpoints.append({'pool': pool, 'source': source,
                          'low_f1_percent': group.macro_f1_mean.iloc[0] * 100,
                          'high_f1_percent': group.macro_f1_mean.iloc[-1] * 100,
                          'delta_pp': (group.macro_f1_mean.iloc[-1] - group.macro_f1_mean.iloc[0]) * 100})
    save_table(pd.DataFrame(endpoints), 'risk_mixture_endpoints')
    return numeric_checks


def coverage_plot():
    table = pd.read_csv(TABLES / 'coverage_curves.csv')
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), sharey=True)
    for row, source in enumerate(SOURCES):
        for col, pool in enumerate(POOLS):
            ax = axes[row, col]
            data = table[(table.pool == pool) & (table.source == source)].sort_values('per_class_budget')
            ax.plot(data.per_class_budget, 100 * data.coverage, color=SOURCE_COLORS[source],
                    marker='o', ms=4, lw=1.6, label='Coverage')
            ax.plot(data.per_class_budget, 100 * data.random, color=GRAY, linestyle='--',
                    marker='s', ms=3.5, lw=1.3, label='Random均值')
            ax.set_title(POOL_LABELS[pool] + ' · ' + SOURCES[source])
            ax.set_ylim(0, 80)
            ax.set_xticks(data.per_class_budget)
            ax.set_xlabel('每类测试目标数')
            if col == 0:
                ax.set_ylabel('频次覆盖（%）')
            last = data.iloc[-1]
            ax.annotate(f'+{last.delta_pp:.2f} pp', (last.per_class_budget, 100 * last.coverage),
                        xytext=(-6, 8), textcoords='offset points', ha='right', fontsize=9)
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle('共同风险配额下，覆盖引导选样增加状态支持', fontsize=13, y=1.01)
    fig.tight_layout()
    save_figure(fig, '01_coverage_curves')


def diagnostic_plot():
    table = pd.read_csv(TABLES / 'diagnostic_auc_contrasts.csv')
    labels = [POOL_LABELS[p] + ' · ' + SOURCES[s] for p in POOLS for s in SOURCES]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for ax, metric, title, color in zip(axes, ['confusion', 'clusters'],
                                      ['混淆类型广度', '错误表征簇广度'], ['#3775BA', PURPLE]):
        for y, (pool, source) in enumerate((p, s) for p in POOLS for s in SOURCES):
            record = table[(table.pool == pool) & (table.source == source) & (table.metric == metric)].iloc[0]
            significant = record.holm_primary4_p < .05
            ax.errorbar(record.delta_auc, y,
                        xerr=[[record.delta_auc - record.ci_low], [record.ci_high - record.delta_auc]],
                        fmt='o', color=color, mfc=color if significant else 'white',
                        ms=6, lw=1.3, capsize=3)
            ax.text(.066, y, f'{record.delta_auc:+.4f}', va='center', ha='right', fontsize=9)
        ax.axvline(0, color='#777777', lw=.8, linestyle='--')
        ax.set_xlim(-.013, .068)
        ax.set_title(title)
        ax.set_xlabel('Coverage − Random：归一化预算AUC差')
        ax.set_yticks(range(6), labels)
        ax.invert_yaxis()
    # Shared y axes only need one inversion.
    if not axes[0].yaxis_inverted():
        axes[0].invert_yaxis()
    fig.suptitle('等错误数下的额外诊断广度：收益有条件', fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, '02_diagnostic_gains')


def risk_plot():
    random = pd.read_csv(TABLES / 'risk_random_control.csv')
    area = pd.read_csv(TABLES / 'risk_area_control.csv')
    labels = [POOL_LABELS[p] + ' · ' + SOURCES[s] for p in POOLS for s in SOURCES]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for y, (pool, source) in enumerate((p, s) for p in POOLS for s in SOURCES):
        row = random[(random.pool == pool) & (random.source == source)].iloc[0]
        ax = axes[0]
        ax.plot([-100 * row.random_sampling_high, -100 * row.random_sampling_low], [y, y],
                color='#BFC5CB', lw=7, solid_capstyle='round')
        gap = -100 * row.observed_delta
        ax.plot(gap, y, 'D', color=SOURCE_COLORS[source], ms=6)
        ax.text(gap + 1.0, y, f'{gap:.2f}', va='center', fontsize=9)
        row = area[(area.pool == pool) & (area.source == source)].iloc[0]
        ax = axes[1]
        estimate, low, high = 100 * row.increment_mean, 100 * row.ci_low, 100 * row.ci_high
        ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]],
                    fmt='o', ms=6, color=SOURCE_COLORS[source], capsize=3, lw=1.2)
        ax.text(10.7, y, f'{estimate:+.2f}', ha='right', va='center', fontsize=9)
    for ax in axes:
        ax.axvline(0, color='#777777', lw=.8, linestyle='--')
        ax.set_yticks(range(6), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-7, 49)
    axes[0].set_title('风险分层与随机划层')
    axes[0].set_xlabel('Q1 − Q5 Macro-F1（百分点）')
    axes[1].set_xlim(-5, 11)
    axes[1].set_title('完整评分与仅面积分层')
    axes[1].set_xlabel('完整落差 − 仅面积落差（百分点）')
    fig.suptitle('风险分层区分表现，但完整评分并非普遍强于面积', fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, '03_risk_controls')


def mixture_plot():
    table = pd.read_csv(TABLES / 'risk_mixture_curves.csv')
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    for ax, pool in zip(axes, POOLS):
        for source in SOURCES:
            group = table[(table.pool == pool) & (table.source == source)].sort_values('high_risk_percent')
            ax.plot(group.high_risk_percent, 100 * group.macro_f1_mean, color=SOURCE_COLORS[source],
                    linestyle='-' if source == 'local' else '--', marker='o' if source == 'local' else 's',
                    markevery=[0, 5, 10, 15, 20], ms=4, lw=1.6, label=SOURCES[source])
        ax.set_title(POOL_LABELS[pool])
        ax.set_xlabel('Q5目标占比（%）')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(40, 100)
    axes[0].set_ylabel('Macro-F1（%）')
    axes[0].legend(loc='lower left')
    fig.suptitle('固定120个测试目标：风险构成改变测得性能', fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, '04_risk_mixture')


def condition_plot():
    support = pd.read_csv(TABLES / 'condition_support.csv')
    contrasts = pd.read_csv(TABLES / 'condition_recall_contrasts.csv')
    labels = [POOL_LABELS[p] + ' · ' + SOURCES[s] for p in POOLS for s in SOURCES]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    for y, (pool, source) in enumerate((p, s) for p in POOLS for s in SOURCES):
        row = support[(support.pool == pool) & (support.source == source)].iloc[0]
        left = 0
        for field, color, label in [('negative_mean', '#3775BA', '种子平均负差'),
                                    ('nonnegative_mean', '#CAA37A', '种子平均非负差'),
                                    ('insufficient', '#DCE0E4', '支持不足')]:
            value = int(row[field])
            axes[0].barh(y, value, left=left, height=.62, color=color, label=label if y == 0 else None)
            if value >= 5:
                axes[0].text(left + value / 2, y, str(value), ha='center', va='center',
                             fontsize=9, color='white' if field == 'negative_mean' else '#292929')
            left += value
        if row.eligible < 5:
            axes[0].text(3.5, y, f'可评价{int(row.eligible)}项', ha='left', va='center', fontsize=8)
        row = contrasts[(contrasts.pool == pool) & (contrasts.source == source)].iloc[0]
        axes[1].errorbar(row.recall_gap_pp, y,
                        xerr=[[row.recall_gap_pp - row.ci_low_pp], [row.ci_high_pp - row.recall_gap_pp]],
                        fmt='o', color=SOURCE_COLORS[source],
                        mfc=SOURCE_COLORS[source] if pool != 'C' else 'white', capsize=3, ms=6, lw=1.2)
        axes[1].text(12, y, f'{row.recall_gap_pp:+.2f}', ha='right', va='center', fontsize=9)
    axes[0].set_title('候选条件的支持量与平均方向')
    axes[0].set_xlim(0, 90)
    axes[0].set_xlabel('状态—架构组合数')
    axes[0].set_yticks(range(6), labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc='upper center', bbox_to_anchor=(.52, -.16), ncol=3, fontsize=8)
    axes[1].set_title('可评价条件的架构汇总Recall差')
    axes[1].axvline(0, color='#777777', lw=.8, linestyle='--')
    axes[1].set_xlim(-35, 13)
    axes[1].set_xlabel('条件Recall − 类别Recall（百分点）')
    fig.suptitle('新目标复查：B/D支持充分，C多数条件无法评价', fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, '05_condition_confirmation')


def case_plot():
    table = pd.read_csv(TABLES / 'condition_cases.csv')
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    for y, row in enumerate(table.itertuples()):
        class_r, state_r = row.class_recall * 100, row.state_recall * 100
        ax.plot([class_r, state_r], [y, y], color='#ABB1B8', lw=2)
        ax.plot(class_r, y, 's', color=GRAY, ms=6, label='类别总体Recall' if y == 0 else None)
        ax.plot(state_r, y, 'o', color='#3775BA' if row.gap_pp < 0 else '#B87543', ms=7,
                label='条件Recall' if y == 0 else None)
        ax.text(state_r, y + .17, f'{state_r:.2f}%', ha='center', fontsize=9)
        ax.text(class_r, y - .23, f'{class_r:.2f}%', ha='center', fontsize=9, color='#555555')
        ax.text(101, y, f'{row.gap_pp:+.2f} pp', va='center', fontsize=9)
    ax.set_yticks(range(4), [f'{r.label}\n{SOURCES[r.source]} · n={r.support}' for r in table.itertuples()])
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-.6, 3.6)
    ax.invert_yaxis()
    ax.set_xlabel('Recall（%）')
    ax.legend(loc='upper center', bbox_to_anchor=(.54, -.20), ncol=2)
    ax.set_title('条件报告将整体分数转为具体后续测试对象', fontsize=13, pad=15)
    fig.tight_layout()
    save_figure(fig, '07_condition_cases')


def main():
    setup()
    numeric_checks = aggregate()
    for plot in (coverage_plot, diagnostic_plot, risk_plot, mixture_plot, condition_plot, case_plot):
        plot()
    audit = {'source_csv_sha256': READ_FILES, 'recomputed_primary_auc_and_condition_means': numeric_checks,
             'same_experiment_data': True, 'new_training_or_vlm_requests': False,
             'core_panel': '8 architectures x 3 seeds',
             'extension_is_separate': True, 'svg_editable_text': True,
             'random_intervals': 'fixed-pool random grouping percentiles, not classifier confidence intervals',
             'auc_and_condition_intervals': 'existing architecture/seed hierarchical bootstrap intervals',
             'mixture_curves': 'existing 120-target, 21-point, 1000-resample panel means'}
    (OUT / 'result_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'numeric_checks': numeric_checks, 'source_files': len(READ_FILES),
                      'output': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
