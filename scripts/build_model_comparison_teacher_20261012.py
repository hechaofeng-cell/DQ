#!/usr/bin/env python3
"""Compare models under one shared visual-state testing setting, using saved results."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageDraw

import build_teacher_results_20261012 as export

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/model_comparison_teacher_20261012'
FIG = OUT / 'figures'
TABLES = OUT / 'tables'
NAME = '视觉状态驱动测试_模型对比结果_教师汇报版_20261012'
SOURCE = ROOT / 'docs' / (NAME + '.md')
MAIN = ROOT / 'artifacts/sixclass_independent_confirmation_20261004'
RECENT = ROOT / 'artifacts/recent_five_models_20261009'
MODELS = {
    'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50',
    'mobilenet_v3_small': 'MobileNetV3-Small', 'mobilenet_v3_large': 'MobileNetV3-Large',
    'densenet121': 'DenseNet-121', 'efficientnet_b0': 'EfficientNet-B0',
    'convnext_tiny': 'ConvNeXt-Tiny', 'swin_t': 'Swin-T',
}
NEW = {'yolo26s_cls': 'YOLO26-S-cls', 'pe_core_b16': 'PE-Core-B16',
       'siglip2_base': 'SigLIP2 Base', 'mobileclip2_s0': 'MobileCLIP2-S0',
       'aimv2_large': 'AIMv2-Large'}
CLASSES = ['dog', 'cat', 'horse', 'sheep', 'person', 'car']
SEEDS = [20260916, 20260917, 20260918]
CASES = [('dog', 'body_orientation', 'back', '背向dog'),
         ('cat', 'target_scale', 'small', '小尺度cat'),
         ('sheep', 'blur_level', 'severe', '严重模糊sheep'),
         ('dog', 'target_scale', 'medium', '中等尺度dog')]
BLUE, GRAY, ORANGE, TEAL = '#3574A8', '#8B939B', '#BB7546', '#42949E'
INPUTS, FIGURE_NAMES = {}, []


def record(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return pd.read_csv(record(path))


def save_table(frame, name):
    frame.to_csv(TABLES / (name + '.csv'), index=False)
    return frame


def aggregate_runs(frame, columns, key='model'):
    grouped = frame.groupby(key)[columns].agg(['mean', 'std', 'min', 'max'])
    grouped.columns = ['_'.join(column) for column in grouped.columns]
    return grouped.reindex(MODELS if key == 'model' else None).reset_index()


def prepare():
    manifest = read(MAIN / 'manifests/candidate_pool.csv')
    truth = np.array([CLASSES.index(name) for name in manifest.class_name])
    assert len(manifest) == 2400 and (manifest.groupby('class_name').size() == 400).all()
    archive = record(ROOT / 'artifacts/random_stratification_ablation_20261006/ablation/B_local.npz')
    with np.load(archive) as cached:
        assert cached['sample_ids'].tolist() == manifest.sample_id.tolist()
        assert cached['variants'][0] == 'full'
        predictions, models, seeds = cached['predictions'], cached['models'], cached['seeds']
        orders, saved_f1 = cached['orders'][0], cached['f1'][0]
        assert sorted(orders.ravel().tolist()) == list(range(2400))
        assert set(models) == set(MODELS) and set(seeds) == set(SEEDS)
        overall, tiers = [], []
        for index, (model, seed) in enumerate(zip(models, seeds)):
            pred = predictions[index]
            values = {'model': model, 'seed': seed, 'targets': 2400,
                      'accuracy': accuracy_score(truth, pred),
                      'macro_precision': precision_score(truth, pred, average='macro', labels=range(6)),
                      'macro_recall': recall_score(truth, pred, average='macro', labels=range(6)),
                      'macro_f1': f1_score(truth, pred, average='macro', labels=range(6))}
            for category, name in enumerate(CLASSES):
                values[name + '_recall'] = np.mean(pred[truth == category] == category)
            overall.append(values)
            for tier in range(5):
                idx = orders[:, tier, :].ravel()
                value = f1_score(truth[idx], pred[idx], average='macro', labels=range(6))
                assert abs(value - saved_f1[index, tier]) < 1e-12
                tiers.append({'model': model, 'seed': seed, 'tier': tier + 1,
                              'targets': len(idx), 'macro_f1': value})
    overall = save_table(pd.DataFrame(overall), 'overall_runs')
    tiers = save_table(pd.DataFrame(tiers), 'tier_runs')
    assert (overall.groupby('model').size() == 3).all()
    published = read(ROOT / 'artifacts/model_results_report_20261008/model_full_pool_scores.csv').query('pool == "B"')
    for row in published.itertuples():
        model = next(key for key, name in MODELS.items() if name == row.model)
        assert abs(overall.query('model == @model').macro_f1.mean() * 100 - row.mean_f1_100) < 1e-9
    full = read(MAIN / 'confirmation/tables/full_stratum_metrics.csv')
    full = full.query('source == "local" and unknown_mode == "midpoint"')
    checked = tiers.merge(full[['model', 'seed', 'tier', 'macro_f1']], on=['model', 'seed', 'tier'], suffixes=('_direct', '_recorded'))
    assert len(checked) == 120 and np.max(abs(checked.macro_f1_direct - checked.macro_f1_recorded)) < 1e-12
    summary = aggregate_runs(overall, ['macro_f1', 'accuracy', 'dog_recall'])
    tier_summary = tiers.groupby(['model', 'tier']).macro_f1.agg(['mean', 'std', 'min', 'max']).reset_index()
    q = tier_summary.pivot(index='model', columns='tier', values='mean')
    summary['q1'] = summary.model.map(q[1])
    summary['q5'] = summary.model.map(q[5])
    summary['q1_minus_q5_pp'] = (summary.q1 - summary.q5) * 100
    for tier in range(1, 6):
        summary[f'q{tier}'] = summary.model.map(q[tier])
    save_table(summary, 'model_summary')
    save_table(tier_summary, 'tier_summary')

    fixed = read(MAIN / 'comparison/tables/fixed_metrics.csv').query('source == "local" and method == "coverage"')
    random = read(MAIN / 'comparison/tables/random_metrics.csv').query('source == "local"')
    coverage = []
    for budget, data in fixed.groupby('budget'):
        ref = random.query('budget == @budget')
        assert len(data) == len(ref) == 24
        assert np.ptp(data.macro_frequency_coverage) < 1e-12
        assert np.ptp(ref.macro_frequency_coverage_mean) < 1e-12
        c, r = data.macro_frequency_coverage.mean(), ref.macro_frequency_coverage_mean.mean()
        coverage.append({'per_class_budget': budget, 'total_budget': budget * 6,
                         'coverage': c, 'random': r, 'delta_pp': 100 * (c - r)})
    coverage = save_table(pd.DataFrame(coverage), 'shared_coverage')
    equal = read(MAIN / 'comparison/tables/equal_error_metrics.csv')
    equal = equal.query('source == "local" and baseline == "random" and representation == "classifier"')
    assert len(equal) == 120
    auc = []
    for (model, seed), data in equal.groupby(['model', 'seed']):
        data = data.sort_values('budget')
        x = np.r_[0, data.budget.to_numpy()]
        assert x.tolist() == [0, 20, 40, 60, 80, 100]
        for metric in ('confusion', 'clusters'):
            left = np.trapezoid(np.r_[0, data['left_' + metric].to_numpy()], x) / x[-1]
            right = np.trapezoid(np.r_[0, data['right_' + metric].to_numpy()], x) / x[-1]
            delta = np.trapezoid(np.r_[0, data['delta_' + metric].to_numpy()], x) / x[-1]
            assert abs(delta - (left - right)) < 1e-12
            auc.append({'model': model, 'seed': seed, 'metric': metric,
                        'coverage_auc': left, 'random_auc': right, 'delta_auc': delta})
    auc = save_table(pd.DataFrame(auc), 'diagnostic_auc_runs')
    diagnosis = auc.groupby(['model', 'metric']).delta_auc.agg(['mean', 'std', 'min', 'max']).reset_index()
    diagnosis['positive_seeds'] = diagnosis.apply(
        lambda row: int(((auc.model == row.model) & (auc.metric == row.metric) & (auc.delta_auc > 0)).sum()), axis=1)
    save_table(diagnosis, 'diagnostic_auc_summary')
    comparisons = read(MAIN / 'comparison/tables/comparisons.csv').query('source == "local" and baseline == "random"')
    for row in comparisons.itertuples():
        assert abs(auc.query('metric == @row.metric').delta_auc.mean() - row.mean_delta_auc) < 1e-12

    mixture = read(ROOT / 'artifacts/controlled_risk_distribution_20261005/tables/model_summary.csv')
    mixture = mixture.query('pool == "B" and source == "local" and total == 120 and budget_role == "common"')
    assert len(mixture) == 8 * 21
    save_table(mixture[['model', 'high_risk_percent', 'macro_f1_mean', 'macro_f1_sampling_low', 'macro_f1_sampling_high']], 'mixture_curves')
    endpoints = mixture.query('high_risk_percent in [0, 100]').pivot(index='model', columns='high_risk_percent', values='macro_f1_mean')
    endpoints.columns = ['low_f1', 'high_f1']
    endpoints['delta_pp'] = 100 * (endpoints.high_f1 - endpoints.low_f1)
    endpoints = save_table(endpoints.reindex(MODELS).reset_index(), 'mixture_endpoints')

    states = read(MAIN / 'confirmation/tables/state_recall.csv').query('source == "local"')
    cases = []
    for category, feature, value, label in CASES:
        subset = states.query('class_name == @category and feature == @feature and value == @value')
        assert len(subset) == 24 and subset.support.nunique() == 1 and subset.support.min() >= 20
        for model, data in subset.groupby('model'):
            ref = overall.query('model == @model')[['seed', category + '_recall']]
            pair = data.merge(ref, on='seed', validate='one_to_one')
            gap = pair.recall - pair[category + '_recall']
            cases.append({'model': model, 'condition': label, 'class_name': category,
                          'feature': feature, 'value': value, 'support': int(data.support.iloc[0]),
                          'state_recall': data.recall.mean(), 'state_recall_sd': data.recall.std(),
                          'class_recall': pair[category + '_recall'].mean(), 'gap_pp': 100 * gap.mean()})
    cases = save_table(pd.DataFrame(cases), 'common_condition_comparison')

    validation = read(MAIN / 'transfer/tables/state_validation.csv').query('source == "local"')
    grouped = validation.groupby(['model', 'class_name', 'feature', 'value']).agg(
        support=('B_support', 'min'), gap=('B_gap', 'mean'), seed_count=('seed', 'nunique')).reset_index()
    assert len(grouped) == 87 and (grouped.seed_count == 3).all()
    counts = []
    for model in MODELS:
        data = grouped.query('model == @model')
        eligible = data.support >= 20
        per_seed = validation.query('model == @model and B_support >= 20').groupby('seed').B_gap.mean()
        counts.append({'model': model, 'candidates': len(data), 'eligible': int(eligible.sum()),
                       'negative_mean': int((eligible & (data.gap < 0)).sum()),
                       'nonnegative_mean': int((eligible & (data.gap >= 0)).sum()),
                       'insufficient': int((~eligible).sum()), 'mean_gap_pp': 100 * per_seed.mean()})
    counts = save_table(pd.DataFrame(counts), 'condition_confirmation_models')
    assert counts[['eligible', 'negative_mean', 'nonnegative_mean', 'insufficient']].sum().tolist() == [78, 69, 9, 9]

    new_overall = read(RECENT / 'tables/overall_runs.csv').query('pool == "B"')
    new_tiers = read(RECENT / 'tables/risk_layers_runs.csv').query('pool == "B" and source == "local"')
    assert len(new_overall) == 15 and len(new_tiers) == 75
    assert set(new_overall.model) == set(NEW)
    new_summary = new_overall.groupby('model').macro_f1.agg(['mean', 'std', 'min', 'max']).reindex(NEW)
    new_q = new_tiers.groupby(['model', 'tier']).macro_f1.mean().unstack()
    new_summary['q1'] = new_q[1]
    new_summary['q5'] = new_q[5]
    new_summary['gap_pp'] = 100 * (new_summary.q1 - new_summary.q5)
    save_table(new_summary.reset_index(), 'extension_summary')
    save_table(new_overall, 'extension_overall_runs')
    save_table(new_tiers, 'extension_tier_runs')
    return summary, tiers, overall, diagnosis, auc, mixture, endpoints, cases, counts, new_summary, new_overall, new_q, coverage


def style():
    font_manager.fontManager.addfont(str(ROOT / 'scripts/assets/NotoSansSC-Regular.ttf'))
    plt.rcParams.update({'font.family': 'Noto Sans SC', 'font.size': 14, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.linewidth': .7, 'axes.spines.top': False,
                         'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out',
                         'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def save(fig, stem):
    for axis in fig.axes:
        colorbar = getattr(axis, '_colorbar', None)
        if colorbar is not None and colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(FIG / (stem + '.' + suffix), bbox_inches='tight', dpi=240, facecolor='white')
    plt.close(fig)
    FIGURE_NAMES.append(stem)


def heatmap(ax, data, labels, columns, title, *, difference=False):
    if difference:
        norm, cmap = TwoSlopeNorm(vmin=-55, vcenter=0, vmax=55), 'RdBu'
    else:
        norm, cmap = plt.Normalize(vmin=30, vmax=100), 'Blues'
    rows, cols = data.shape
    im = ax.pcolormesh(np.arange(cols+1)-.5, np.arange(rows+1)-.5, data,
                       cmap=cmap, norm=norm, shading='flat', rasterized=False)
    ax.set_xlim(-.5, cols-.5)
    ax.set_ylim(rows-.5, -.5)
    ax.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(labels)), yticklabels=labels, title=title)
    ax.tick_params(length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            color = 'white' if (abs(value) > 34 if difference else value > 77) else '#272727'
            ax.text(col, row, f'{value:+.1f}' if difference else f'{value:.1f}', ha='center', va='center', color=color, fontsize=14)
    return im


def plots(data):
    summary, tiers, overall, diagnosis, auc, mixture, endpoints, cases, counts, new_summary, new_overall, new_q, coverage = data
    y = np.arange(8)
    labels = list(MODELS.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), layout='constrained')
    for ax, metric, label in zip(axes, ['macro_f1', 'dog_recall'], ['a  六类别总体Macro-F1', 'b  dog类别总体Recall']):
        for index, model in enumerate(MODELS):
            values = 100 * overall.query('model == @model')[metric].to_numpy()
            ax.plot([values.min(), values.max()], [index, index], color=GRAY, lw=2)
            ax.scatter(values, index + np.array([-.11, 0, .11]), s=15, color=GRAY, zorder=3)
            ax.scatter(values.mean(), index, s=48, marker='D', color=BLUE, zorder=4)
            ax.text(101.2, index, f'{values.mean():.2f}', va='center', fontsize=13)
        ax.set(yticks=y, yticklabels=labels if ax is axes[0] else [], ylim=(7.6, -.6),
               xlim=(60, 107), xticks=[60, 70, 80, 90, 100], xlabel='得分（%）', title=label)
    save(fig, '01_overall_model_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained')
    for ax, metric, title in zip(axes, ['confusion', 'clusters'], ['a  混淆类型广度', 'b  错误表征簇广度']):
        for index, model in enumerate(MODELS):
            values = 100 * auc.query('model == @model and metric == @metric').delta_auc.to_numpy()
            mean = values.mean()
            ax.barh(index, mean, height=.48, color=BLUE if mean >= 0 else ORANGE, alpha=.7)
            ax.scatter(values, index + np.array([-.13, 0, .13]), color='#32383E', s=22, zorder=4)
            ax.text(5.0, index, f'{mean:+.2f}', va='center', fontsize=13)
        ax.axvline(0, color='#535A60', lw=.8)
        ax.set(yticks=y, yticklabels=labels if ax is axes[0] else [], ylim=(7.6, -.6),
               xlim=(-1.8, 5.9), xlabel='Coverage − Random：广度AUC × 100', title=title)
    save(fig, '02_coverage_diagnostic_model_gains')

    matrix = summary[[f'q{tier}' for tier in range(1, 6)]].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(12, 5.7), layout='constrained')
    im = heatmap(ax, matrix, labels, [f'Q{i}' for i in range(1, 6)], '相同风险分层中的八种模型：Macro-F1（%）')
    fig.colorbar(im, ax=ax, shrink=.8, label='Macro-F1（%）')
    save(fig, '03_five_risk_layers_models')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
    for index, row in enumerate(summary.itertuples()):
        q1, q5 = row.q1 * 100, row.q5 * 100
        axes[0].plot([q5, q1], [index, index], color='#CCD2D7', lw=2)
        axes[0].scatter(q1, index, marker='s', color=GRAY, s=42, label='Q1' if index == 0 else '')
        axes[0].scatter(q5, index, color=BLUE, s=42, label='Q5' if index == 0 else '')
        axes[0].text(q5 - .8, index, f'{q5:.1f}', va='center', ha='right', fontsize=12)
        axes[0].text(q1 + .8, index, f'{q1:.1f}', va='center', fontsize=12)
        axes[1].barh(index, row.q1_minus_q5_pp, color=BLUE, alpha=.7, height=.5)
        axes[1].text(row.q1_minus_q5_pp + .5, index, f'{row.q1_minus_q5_pp:.2f}', va='center', fontsize=13)
    axes[0].legend(loc='lower right', ncol=2)
    axes[0].set(xlim=(56, 104), xlabel='Macro-F1（%）', title='a  Q1与Q5的实际表现')
    axes[1].set(xlim=(0, 35), xlabel='Q1 − Q5（百分点）', title='b  高低风险层落差')
    for ax in axes:
        ax.set(yticks=y, yticklabels=labels if ax is axes[0] else [], ylim=(7.6, -.6))
    save(fig, '04_risk_gap_models')

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7), layout='constrained', sharex=True, sharey=True)
    for ax, (model, label) in zip(axes.ravel(), MODELS.items()):
        subset = mixture.query('model == @model').sort_values('high_risk_percent')
        ax.fill_between(subset.high_risk_percent, subset.macro_f1_sampling_low * 100,
                        subset.macro_f1_sampling_high * 100, color=BLUE, alpha=.10, lw=0)
        ax.plot(subset.high_risk_percent, subset.macro_f1_mean * 100, color=BLUE, lw=1.7,
                marker='o', markevery=[0, 5, 10, 15, 20], ms=3)
        last = subset.iloc[-1]
        ax.annotate(f'{last.macro_f1_mean * 100:.1f}', (100, last.macro_f1_mean * 100),
                    (-4, 7), textcoords='offset points', ha='right', color=BLUE, fontsize=12)
        ax.set(title=label, xlim=(0, 103), ylim=(40, 100), xticks=[0, 50, 100], yticks=[40, 60, 80, 100])
    for ax in axes[:, 0]:
        ax.set_ylabel('Macro-F1（%）')
    for ax in axes[-1]:
        ax.set_xlabel('Q5占比（%）')
    save(fig, '05_fixed_budget_mixture_models')

    condition_labels = [case[-1] for case in CASES]
    supports = [int(cases.query('condition == @label').support.iloc[0]) for label in condition_labels]
    columns = [f'{label}\nn={support}' for label, support in zip(condition_labels, supports)]
    absolute = cases.pivot(index='model', columns='condition', values='state_recall').reindex(index=MODELS, columns=condition_labels).to_numpy() * 100
    gap = cases.pivot(index='model', columns='condition', values='gap_pp').reindex(index=MODELS, columns=condition_labels).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), layout='constrained')
    im = heatmap(axes[0], absolute, labels, columns, 'a  相同视觉条件下的Recall（%）')
    fig.colorbar(im, ax=axes[0], shrink=.75, pad=.02)
    im = heatmap(axes[1], gap, [''] * 8, columns, 'b  条件Recall − 类别总体（百分点）', difference=True)
    fig.colorbar(im, ax=axes[1], shrink=.75, pad=.02)
    save(fig, '06_common_visual_conditions_models')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
    left = np.zeros(8)
    for key, label, color in [('negative_mean', '平均仍较弱', BLUE), ('nonnegative_mean', '平均差非负', ORANGE), ('insufficient', '支持不足', '#D0D5D9')]:
        values = counts[key].to_numpy()
        axes[0].barh(y, values, left=left, color=color, label=label, height=.6)
        for index, value in enumerate(values):
            if value:
                axes[0].text(left[index] + value / 2, index, str(value), ha='center', va='center', color='white' if color == BLUE else '#272727', fontsize=13)
        left += values
    axes[0].legend(loc='lower right', fontsize=12)
    axes[0].set(xlim=(0, 17), xlabel='既定候选条件组合数', title='a  各模型的条件复查记录')
    axes[1].barh(y, counts.mean_gap_pp, color=BLUE, height=.6, alpha=.75)
    for index, value in enumerate(counts.mean_gap_pp):
        axes[1].text(value - .5, index, f'{value:.2f}', va='center', ha='right', fontsize=13)
    axes[1].axvline(0, color=GRAY, lw=.7)
    axes[1].set(xlim=(-37, 1), xlabel='可评价条件的平均Recall差（百分点）', title='b  各模型清单内的平均差')
    for ax in axes:
        ax.set(yticks=y, yticklabels=labels if ax is axes[0] else [], ylim=(7.6, -.6))
    save(fig, '07_condition_confirmation_models')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained', gridspec_kw={'width_ratios': [1.3, 1]})
    matrix = new_q.reindex(NEW).to_numpy() * 100
    im = heatmap(axes[0], matrix, list(NEW.values()), [f'Q{i}' for i in range(1, 6)], 'a  扩展模型五层Macro-F1（%）')
    fig.colorbar(im, ax=axes[0], shrink=.8, pad=.02)
    for index, model in enumerate(NEW):
        values = new_overall.query('model == @model').macro_f1.to_numpy() * 100
        axes[1].plot([values.min(), values.max()], [index, index], color=GRAY, lw=2)
        axes[1].scatter(values, index + np.array([-.10, 0, .10]), color=GRAY, s=24)
        axes[1].scatter(values.mean(), index, color=BLUE, marker='D', s=50)
        axes[1].text(94, index, f'{values.mean():.2f}', va='center', fontsize=13)
    axes[1].set(yticks=range(5), yticklabels=[''] * 5, ylim=(4.6, -.6),
                xlim=(45, 102), xticks=[50, 60, 70, 80, 90], xlabel='全池Macro-F1（%）', title='b  三个种子的总体得分')
    save(fig, '08_extension_models_separate')


def image(stem, caption):
    return f'![{caption}](../artifacts/model_comparison_teacher_20261012/figures/{stem}.png)'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def write_report(data):
    summary, tiers, overall, diagnosis, auc, mixture, endpoints, cases, counts, new_summary, new_overall, new_q, coverage = data
    s = summary.set_index('model')
    def score(model, column, digits=2):
        return f'{s.loc[model, column] * 100:.{digits}f}'
    def diag(model, metric):
        return diagnosis.query('model == @model and metric == @metric').iloc[0]
    common = coverage.iloc[-1]
    overall_table = table(['模型', '全池F1：均值±SD', 'Q5 F1', 'Q1−Q5落差'],
                          [[MODELS[r.model], f'{r.macro_f1_mean*100:.2f}±{r.macro_f1_std*100:.2f}%',
                            f'{r.q5*100:.2f}%', f'{r.q1_minus_q5_pp:.2f}个百分点'] for r in summary.itertuples()])
    diag_table = table(['模型', '混淆广度增量×100', '表征簇广度增量×100', '正增益种子：混淆／簇'],
                       [[label, f'{diag(m,"confusion")["mean"]*100:+.3f}', f'{diag(m,"clusters")["mean"]*100:+.3f}',
                         f'{diag(m,"confusion").positive_seeds}/3 ／ {diag(m,"clusters").positive_seeds}/3'] for m, label in MODELS.items()])
    tier_table = table(['模型', 'Q2 F1', 'Q3 F1', 'Q4 F1'],
                       [[MODELS[r.model], f'{r.q2*100:.2f}%', f'{r.q3*100:.2f}%', f'{r.q4*100:.2f}%'] for r in summary.itertuples()])
    mixture_table = table(['模型', 'Q5占比0%：F1', 'Q5占比100%：F1', '变化'],
                          [[MODELS[r.model], f'{r.low_f1*100:.2f}%', f'{r.high_f1*100:.2f}%', f'{r.delta_pp:.2f}个百分点'] for r in endpoints.itertuples()])
    counts_table = table(['模型', '可评价组合', '平均仍较弱', '支持不足'],
                         [[MODELS[r.model], r.eligible, r.negative_mean, r.insufficient] for r in counts.itertuples()])
    new_table = table(['扩展模型', '全池F1：均值±SD', 'Q5 F1', 'Q1−Q5落差'],
                      [[NEW[m], f'{r["mean"]*100:.2f}±{r["std"]*100:.2f}%', f'{r.q5*100:.2f}%',
                        f'{r.gap_pp:.2f}个百分点'] for m, r in new_summary.iterrows()])
    case_lookup = cases.set_index(['model', 'condition'])
    condition_table = table(['对比条件', 'ResNet-50 Recall', 'ConvNeXt-Tiny Recall', '同批目标数'],
                            [[label, f'{case_lookup.loc[("resnet50",label),"state_recall"]*100:.2f}%',
                              f'{case_lookup.loc[("convnext_tiny",label),"state_recall"]*100:.2f}%',
                              case_lookup.loc[("resnet50",label),'support']] for label in [case[-1] for case in CASES]])
    text = f'''# 视觉状态驱动测试：模型之间有什么区别？

教师汇报版｜2026年10月12日｜8种主模型＋5种独立扩展模型

## 一、先看模型总体表现，再看被总体分数掩盖的差异

本报告将模型作为比较对象，统一使用COCO新目标确认池的2,400个目标、同一本地视觉状态标注及同一风险分层。任务为dog、cat、horse、sheep、person、car六类别目标区域分类；主实验每种模型保留3个训练种子。

研究方法的作用是：用覆盖选样增加状态支持，用风险分层揭示条件差异，再把模型结果写成可复查的条件报告。以下各图使用相同模型顺序，分数为三种子均值；百分比表示成绩，百分点表示差值。SD为三个训练种子的样本标准差，表示训练波动。

{image('01_overall_model_comparison','图1 相同目标上的模型总体表现与dog类别表现')}

图1说明：菱形为均值，灰点为三个种子，灰线为最小—最大范围；不是置信区间。总体F1在完整2,400目标上直接计算，不是五个风险层F1的简单平均。

{overall_table}

**当前数值比较中，ConvNeXt-Tiny的总体F1为{score('convnext_tiny','macro_f1_mean')}%，高风险Q5为{score('convnext_tiny','q5')}%，均为主组最高。** Swin-T次之。该排序限于本任务与实际训练设置，未进行逐对模型优劣显著性检验。

<!-- PAGEBREAK -->

## 二、覆盖提高以后，对不同模型的测试有何作用？

相同600个测试目标预算下，Coverage频次覆盖为{common.coverage*100:.2f}%，Random为{common.random*100:.2f}%，增加{common.delta_pp:.2f}个百分点。八个模型共用选样结果，因此覆盖率相同；真正需要逐模型比较的是增加覆盖后得到的诊断信息。

在相同类别和风险配额内，再匹配各类错误数量，比较Coverage与Random覆盖的混淆类型和错误表征簇。这样可检查“不是仅仅因为错误更多，是否仍能得到更广的失败信息”。

{image('02_coverage_diagnostic_model_gains','图2 覆盖选样相对随机选样的逐模型诊断增量')}

图2说明：柱为三种子平均增量，黑点为逐种子结果；零线右侧有利于Coverage。数值为标准化预算AUC的差×100，即曲线积分除以最大预算100后比较；不是分类F1或ROC-AUC。预算为每类20—100个目标，包含零预算零广度起点。

{diag_table}

**混淆类型广度的平均增量在8个模型上均为正；表征簇广度在7个模型上为正。** MobileNetV3-Small两项均为3/3种子正增量；EfficientNet-B0的表征簇平均增量为负，必须保留这一结果。

可以向老师解释：“同样提高了测试状态覆盖，对不同模型带来的诊断收益并不相同。”表征簇及其分母随模型变化，应比较模型内部Coverage−Random的差；本图不用于模型能力排名，也未给单个模型做显著性声明。现有结果未支持全面优于强基线。

<!-- PAGEBREAK -->

## 三、相同风险层下，各模型的表现如何变化？

风险规则根据尺度、可见性、遮挡、截断和朝向等条件分为Q1—Q5。Q1为低风险层，Q5为高风险层；每层480个目标、六类别各80个。所有模型使用完全相同的层成员。

{image('03_five_risk_layers_models','图3 八种模型在五个共同风险层中的表现')}

图3说明：每行是模型，每列是风险层，格内数值为三种子平均Macro-F1（%）。颜色越深，分类表现越好。分层依据不使用被测分类器的预测或错误。

{tier_table}

**分层把总体指标展开成模型的条件表现。** 例如DenseNet-121的总体F1为{score('densenet121','macro_f1_mean')}%，高于ResNet-18的{score('resnet18','macro_f1_mean')}%；但Q5中前者为{score('densenet121','q5')}%，后者为{score('resnet18','q5')}%。总体次序在高风险条件中发生了变化。

可以向老师解释：“总体分数只能说明平均表现，分层能指出模型在何种目标上失去优势。”本设置的层均值下降不构成普遍单调规律；完整风险评分也未普遍强于仅用目标面积分层。

<!-- PAGEBREAK -->

## 四、比较高风险绝对表现与高低风险落差

仅看Q1−Q5的下降幅度容易误判：两层都低的模型，也可能下降较少。因此同时看Q5实际分数和层间落差。

{image('04_risk_gap_models','图4 各模型的高风险成绩与高低风险落差')}

图4说明：左图灰方块为Q1、蓝圆点为Q5，连线表示高低层成绩跨度；右图为Q1−Q5的差值。两边均为三种子均值，不能将落差更小自动解释为模型更强。

MobileNetV3-Small的层间落差为{s.loc['mobilenet_v3_small','q1_minus_q5_pp']:.2f}个百分点，Q5为{score('mobilenet_v3_small','q5')}%；ConvNeXt-Tiny落差为{s.loc['convnext_tiny','q1_minus_q5_pp']:.2f}个百分点，Q5为{score('convnext_tiny','q5')}%。后者在高风险目标上的实际成绩更高。

DenseNet-121的落差为{s.loc['densenet121','q1_minus_q5_pp']:.2f}个百分点，是本主组最大值；它的Q1成绩较高，但Q5优势未保留。ResNet-50相对ResNet-18则在总体与Q5中都取得较高均值。

**该比较回答的是“哪些模型在相同视觉条件下更容易暴露性能不足”，而不是给模型建立脱离任务的永久排名。** 模型侧差异提供了本方法的诊断对象；具体原因仍需另外实验验证。

### 这一页可以怎样讲？

“我的方法不只是让每个模型得到一个总分，而是同时报告容易条件、困难条件和落差。例如DenseNet总体比ResNet-18高，但在最高风险层中反而低；这类差别是总体评价容易漏掉的。”

<!-- PAGEBREAK -->

## 五、测试量固定后，各模型怎样响应高风险占比？

每次固定120个目标、六类各20个，让Q5占比从0%增加至100%。0%来自Q1—Q4并集，100%来自Q5；其余比例在这两部分抽样。每个比例重复抽样1000次，不重新训练模型。

{image('05_fixed_budget_mixture_models','图5 固定120目标时各模型的风险构成响应')}

图5说明：八个小图共用坐标。实线为种子与重复抽样的平均F1；浅带为当前池内重复抽样的中心95%范围，不是新数据性能置信区间。线尾数字为100% Q5的平均成绩。

{mixture_table}

**不同模型随高风险占比上升而呈现不同的成绩水平和端点变化。** 这里的变化由测试构成产生，说明向老师汇报模型成绩时，需要交代测试了多少高风险目标。

图5的100% Q5只抽取120个目标，图3／4的完整Q5使用480个目标；二者F1不必相等。图中的下降不表示模型训练后退化，也不证明每个类别Recall都同步下降。

<!-- PAGEBREAK -->

## 六、同一种视觉条件，哪些模型更容易识别？

选取背向dog、小尺度cat、严重模糊sheep、中等尺度dog四个条件作解释案例。各条件内八模型面对相同目标；条件不是独立错误类型，也不是全面挑选模型优劣的基准。

{image('06_common_visual_conditions_models','图6 同一视觉条件下的逐模型Recall及相对类别差')}

图6说明：左图为条件Recall，右图为条件Recall减对应类别总体Recall；右图负数表示该条件较弱，正数表示较强。n是目标数，所有格为三种子均值；中等尺度dog同时保留非负差，避免只展示薄弱案例。

{condition_table}

在背向dog条件中，ResNet-50的Recall为{case_lookup.loc[('resnet50','背向dog'),'state_recall']*100:.2f}%，ConvNeXt-Tiny为{case_lookup.loc[('convnext_tiny','背向dog'),'state_recall']*100:.2f}%；在小尺度cat条件中，两者又呈现另一组差异。

**左图用于相同目标上的模型比较；右图用于解释“这一条件相对该模型自身类别水平是否偏弱”。** 两种问题需要分开。例如模型条件Recall较高，仍可能比它自身类别总体低；不能只凭右图差值排模型强弱。

可以向老师解释：“分层提示哪里表现下降，状态报告进一步指出具体是哪类条件。不同模型在相同条件下的表现不同，所以后续测试能针对模型展开。”这里的关联不证明某个状态单独导致错误。

<!-- PAGEBREAK -->

## 七、各模型已有薄弱条件在新目标中是否延续？

发现阶段为每个模型形成候选条件清单；本页沿用清单检查新目标。支持数至少20时，比较三个种子平均条件Recall与类别总体Recall，并保留平均非负差及支持不足的条件。

{image('07_condition_confirmation_models','图7 逐模型的薄弱条件复查记录')}

图7说明：左图把各模型清单分为平均仍较弱、平均差非负、支持不足；右图为各自可评价清单的平均Recall差。候选清单不同，右图及组合数量不用于模型优劣排名；计数不是逐项显著或独立复现。

{counts_table}

**本设置87个既定组合中，78个可评价，69个保留平均负差，9个平均非负，9个支持不足。** ConvNeXt-Tiny有5个可评价组合且5个平均仍较弱；这说明总体表现较高的模型也有可复查的薄弱条件。

“条件数少”不自动表示模型更好，因为发现阶段清单按模型分别形成。更公平的模型比较应使用图3、图4和图6的共同层／共同条件；本页回答的是各模型自己的诊断线索能否在新目标上继续检查。

<!-- PAGEBREAK -->

## 八、另外五种模型的分层表现与训练波动

扩展组为YOLO26-S-cls、PE-Core-B16、SigLIP2 Base、MobileCLIP2-S0、AIMv2-Large，各3个训练种子。沿用相同目标池和本地风险层，但预训练来源、规模及适配配置与主组不同，单独报告。

{image('08_extension_models_separate','图8 五种扩展模型的分层差异与种子波动')}

图8说明：左图为五层平均F1，右图为三个种子各自的全池F1及均值；灰线为最小—最大范围。保留所有种子，包含低值结果。

{new_table}

本设置五模型的平均Q1−Q5落差均为正。PE-Core-B16的总体F1种子SD为{new_summary.loc['pe_core_b16','std']*100:.2f}个百分点，说明当前训练适配对种子敏感；不能据此断言该架构固有能力较差。扩展组未单独重复覆盖选样诊断、随机划层或仅面积对照，不把主组对照结果移作这五模型的直接证据。

## 可以直接向老师说明的结论

“我把模型放到共同视觉条件下比较。覆盖选样让测试状态支持更充分，并在不同模型上产生不同的诊断增量；风险分层揭示总体成绩不能反映的模型差别；条件报告把这些差别转成背向、小目标、模糊等具体可复查的测试线索。当前贡献是模型测试与诊断方法的实证证据，尚未证明再训练后准确率提升。”

模型数值比较是已测任务中的观察结果，未宣称通用架构排名或逐模型显著提升。数据来自既有预测、分层、等错误数比较及复查记录；本次仅重新汇总与绘图，没有新增训练或VLM请求。
'''
    SOURCE.write_text(text, encoding='utf-8')


def claim_evidence(data):
    (OUT / 'claim_evidence.md').write_text('''# 模型对比结果的证据边界

| 结论 | 复算输出 | 可以使用的解释 | 边界 |
|---|---|---|---|
| 主组总体与dog表现不同 | overall_runs.csv、model_summary.csv | 同池三种子平均的观察差异 | 不宣称通用或显著架构排名 |
| Coverage的诊断增量因模型而异 | diagnostic_auc_runs.csv | 模型内同风险配额、等错误数的AUC差 | 不给单模型显著性星号，错误簇分母因模型变化 |
| 主组共用状态覆盖值 | shared_coverage.csv | 600目标时频次覆盖增加 | 不是每个模型独有覆盖或准确率增益 |
| 五层与高低层表现不同 | tier_runs.csv、tier_summary.csv | 共同层的条件表现 | 不保证其他来源或任务严格单调 |
| 测试构成引起测得成绩变化 | mixture_curves.csv、mixture_endpoints.csv | 120目标固定预算的构成响应 | 不是训练收益，抽样区间不是部署置信区间 |
| 共同状态表现存在模型差别 | common_condition_comparison.csv | 同一条件目标上的Recall与模型内相对差 | 四个说明案例，状态关联不是错误因果 |
| 候选条件可在新目标继续检查 | condition_confirmation_models.csv | 既定条件平均负差／非负／支持不足 | 模型清单不同，不比较原始数量为架构能力 |
| 扩展五模型存在端点差及种子波动 | extension_summary.csv、extension_overall_runs.csv | 同一主设置中的扩展观察 | 不继承主8模型的覆盖或风险对照结论 |

复算命令：`.venv/bin/python scripts/build_model_comparison_teacher_20261012.py`。
源文件校验值、各项重算及图表数量见 result_audit.json 与 document_audit.json。
''', encoding='utf-8')
    metadata = {'setting': 'B / local / midpoint', 'main_models': list(MODELS), 'seeds': SEEDS,
                'input_sha256': INPUTS, 'direct_overall_prediction_runs': 24,
                'direct_tier_metrics_checked': 120, 'diagnostic_auc_runs': 48,
                'shared_coverage_verified_model_independent': True,
                'mixture_points': 168, 'common_condition_model_rows': 32,
                'condition_counts': {'candidates': 87, 'eligible': 78, 'negative_mean': 69, 'nonnegative_mean': 9, 'insufficient': 9},
                'extension_models': list(NEW), 'extension_overall_runs': 15,
                'no_new_training_or_vlm_requests': True}
    (OUT / 'result_audit.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def document_export():
    export.NAME, export.SOURCE, export.OUT, export.LATEX = NAME, SOURCE, OUT, OUT / 'latex'
    blocks = export.source_blocks()
    word = export.word_export(blocks, table_widths={4: [4.4, 4.1, 4.1, 4.4]})
    pdf = export.pdf_export(blocks, table_weights={4: [.26, .24, .24, .26]})
    sys.path.insert(0, str(ROOT / 'artifacts/teacher_methodology_20261011/vendor'))
    import pymupdf
    from lxml import etree
    document = pymupdf.open(pdf)
    assert len(document) == 8, len(document)
    preview = OUT / 'preview'
    preview.mkdir(exist_ok=True)
    thumbs = []
    for index, page in enumerate(document):
        text = page.get_text()
        assert '\ufffd' not in text
        assert ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、"][index] in text
        path = preview / f'page_{index+1:02d}.png'
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(path)
        im = Image.open(path).convert('RGB')
        im.thumbnail((380, 545))
        thumbs.append(im)
    contact = Image.new('RGB', (1600, 1180), '#EEEEEE')
    draw = ImageDraw.Draw(contact)
    for index, thumb in enumerate(thumbs):
        x, y = 10 + (index % 4) * 400, 25 + (index // 4) * 590
        contact.paste(thumb, (x, y))
        draw.text((x, y-18), f'Page {index+1}', fill='black')
    contact.save(preview / 'contact_sheet.png')
    log = (OUT / 'latex_compilation.log').read_text(encoding='utf-8')
    assert not re.search(r'Missing character|Undefined control sequence|Overfull', log)
    with zipfile.ZipFile(word) as package:
        xml = etree.fromstring(package.read('word/document.xml'))
        assert len(xml.findall('.//w:drawing', export.methods.NS)) == 8
        assert len(xml.findall('.//w:tbl', export.methods.NS)) == 7
        assert len(xml.findall('.//w:br[@w:type="page"]', export.methods.NS)) == 7
    archive = SOURCE.with_name(NAME + '_图表与数据.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
        for directory in (FIG, TABLES):
            for path in directory.iterdir():
                package.write(path, path.relative_to(OUT))
        for path in (SOURCE, word, pdf, OUT / 'claim_evidence.md', OUT / 'result_audit.json'):
            package.write(path, path.name)
        portable = SOURCE.read_text(encoding='utf-8').replace('../artifacts/model_comparison_teacher_20261012/', '')
        package.writestr('可移植图文版.md', portable)
        for path in (Path(__file__), ROOT / 'scripts/build_teacher_results_20261012.py',
                     ROOT / 'scripts/export_current_method_conclusions_20261012.py',
                     ROOT / 'scripts/recompile_methods_latex_20261012.py'):
            package.write(path, 'scripts/' + path.name)
    audit = {'pages': 8, 'figures': 8, 'tables': 7, 'word_rendered': False,
             'pdf_checked': True, 'figures_svg_editable': True,
             'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, word, pdf, archive)}}
    (OUT / 'document_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    data = prepare()
    style()
    plots(data)
    write_report(data)
    claim_evidence(data)
    document_export()


if __name__ == '__main__':
    main()
