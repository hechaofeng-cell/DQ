#!/usr/bin/env python3
"""Editable figures for paired annotation diagnostics; measured points remain."""
from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.interpolate import PchipInterpolator

from dual_vlm_common import CLASSES, ROOT, STUDY, read_csv, sha256, write_json

COLORS = {'local': '#0F4D92', 'online': '#9A4D8E', 'random': '#767676'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Liberation Sans'],
                     'font.size': 9, 'axes.linewidth': .8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False})


def tidy(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.set_xlabel('Test budget per class')


def save(figure, directory, stem):
    figure.savefig(directory / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    figure.savefig(directory / f'{stem}.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    root = ET.parse(directory / f'{stem}.svg').getroot()
    namespace = {'s': 'http://www.w3.org/2000/svg'}
    if not root.get('viewBox') or not root.findall('.//s:text', namespace) or root.findall('.//s:image', namespace):
        raise ValueError(f'SVG must retain editable vector text: {stem}')


def curve(axis, x, y, label, color, linestyle='-'):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dense = np.linspace(x.min(), x.max(), 161)
    axis.plot(dense, PchipInterpolator(x, y)(dense), color=color, linestyle=linestyle, linewidth=1.5, label=label)
    axis.plot(x, y, 'o', color=color, markersize=4)


def make_figures(output):
    directory = output / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    fixed = read_csv(output / 'tables/cross_fixed_metrics.csv')
    random = read_csv(output / 'tables/cross_random_summary.csv')
    strict = read_csv(output / 'tables/equal_error_metrics.csv')
    budgets = sorted({int(row['budget']) for row in fixed})
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.9), constrained_layout=True)
    for backend in ('local', 'online'):
        opposite = 'online' if backend == 'local' else 'local'
        cross = [row for row in fixed if row['method'] == 'coverage' and row['selection_source'] == backend
                 and row['evaluation_source'] == opposite]
        same = [row for row in fixed if row['method'] == 'coverage' and row['selection_source'] == backend
                and row['evaluation_source'] == backend]
        def means(rows, metric):
            return [np.mean([float(row[metric]) for row in rows if int(row['budget']) == budget]) for budget in budgets]
        curve(axes[0, 0], budgets, means(cross, 'macro_frequency_coverage'),
              f'{backend.capitalize()} selection → {opposite} labels', COLORS[backend])
        random_opposite = [row for row in random if row['evaluation_source'] == opposite]
        curve(axes[0, 0], budgets, means(random_opposite, 'macro_frequency_coverage_mean'),
              f'Random → {opposite} labels', COLORS[backend], '--')
        curve(axes[0, 1], budgets, means(same, 'macro_f1'), f'{backend.capitalize()} selection', COLORS[backend])
        independent = [row for row in strict if row['selection_source'] == backend and row['comparison'] == 'coverage_vs_random']
        for axis, metric in ((axes[1, 0], 'delta_confusion'), (axes[1, 1], 'delta_clusters')):
            curve(axis, budgets, means(independent, metric), backend.capitalize(), COLORS[backend])
            axis.axhline(0, color='#b0b0b0', linewidth=.7, linestyle='--')
    random_one = [row for row in random if row['evaluation_source'] == 'local']
    random_f1 = [np.mean([float(row['macro_f1_mean']) for row in random_one if int(row['budget']) == budget]) for budget in budgets]
    curve(axes[0, 1], budgets, random_f1, 'Random', COLORS['random'], '--')
    titles = ['Cross-source state coverage', 'Frozen-classifier Macro-F1',
              'Equal-error confusion breadth gain', 'Equal-error cluster breadth gain']
    ylabels = ['Frequency coverage', 'Macro-F1', 'Coverage − Random', 'Coverage − Random']
    for letter, axis, title, ylabel in zip('abcd', axes.ravel(), titles, ylabels):
        tidy(axis)
        axis.set_title(title, loc='left', fontsize=10)
        axis.set_ylabel(ylabel)
        axis.set_xticks(budgets)
        axis.legend(fontsize=7)
        axis.text(-.12, 1.05, letter, transform=axis.transAxes, fontsize=10, fontweight='bold')
    for axis in axes[0]:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    save(figure, directory, 'dual_vlm_diagnostics')

    agreement = [row for row in read_csv(output / 'tables/annotation_agreement.csv') if row['shared'] == 'True']
    features = list(dict.fromkeys(row['feature'] for row in agreement))
    matrix = np.asarray([[float(next(row['exact_agreement'] for row in agreement if row['class_name'] == name and row['feature'] == feature))
                         for feature in features] for name in CLASSES])
    figure, axis = plt.subplots(figsize=(11.2, 4.6), constrained_layout=True)
    palette = LinearSegmentedColormap.from_list('agreement', ['#f5f5f5', '#c0d5eb', '#0F4D92'])
    heatmap = axis.pcolormesh(np.arange(len(features) + 1) - .5, np.arange(7) - .5, matrix,
                             vmin=0, vmax=1, cmap=palette, shading='flat', edgecolors='white', linewidth=.4)
    axis.set_ylim(5.5, -.5)
    axis.set_yticks(range(6), CLASSES)
    axis.set_xticks(range(len(features)), [name.replace('_', ' ') for name in features], rotation=45, ha='right', fontsize=8)
    for i in range(6):
        for j in range(len(features)):
            axis.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center', fontsize=7,
                      color='white' if matrix[i,j] > .72 else '#272727')
    axis.set_title('Online/local exact label agreement (includes unknown)', loc='left', fontsize=10)
    colorbar = figure.colorbar(heatmap, ax=axis, shrink=.85, format=PercentFormatter(1), label='Agreement')
    colorbar.solids.set_rasterized(False)
    save(figure, directory, 'annotation_agreement')

    overlap = read_csv(output / 'tables/selection_overlap.csv')
    figure, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    palette = ['#0F4D92', '#9A4D8E', '#42949E', '#8B6F47', '#767676', '#B64342']
    for name, color in zip(CLASSES, palette):
        rows = [row for row in overlap if row['class_name'] == name]
        rows.sort(key=lambda row: int(row['budget']))
        curve(axis, [int(row['budget']) for row in rows], [float(row['overlap_fraction']) for row in rows], name, color)
    tidy(axis)
    axis.set_xticks(budgets)
    axis.set_ylim(0, 1.02)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylabel('Shared images / budget')
    axis.set_title('Overlap of local and online Coverage selections', loc='left', fontsize=10)
    axis.legend(ncol=3, fontsize=8)
    save(figure, directory, 'selection_overlap')

    rankings = read_csv(output / 'tables/model_rankings.csv')
    endpoint = max(budgets)
    models = [row['model'] for row in rankings if row['selection_source'] == 'local' and int(row['budget']) == endpoint]
    rows = {backend: {row['model']: row for row in rankings if row['selection_source'] == backend and int(row['budget']) == endpoint}
            for backend in ('local', 'online')}
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    positions = np.arange(len(models))
    for backend, shift in (('local', -.17), ('online', .17)):
        axis.barh(positions + shift, [float(rows[backend][model]['macro_f1_mean']) for model in models], height=.32,
                  xerr=[float(rows[backend][model]['macro_f1_sd']) for model in models], color=COLORS[backend],
                  label=backend.capitalize(), error_kw={'elinewidth': .8, 'capsize': 2})
    axis.set_yticks(positions, models)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel('Macro-F1 (seed mean ± SD)')
    axis.set_title(f'Frozen models on Coverage tests: {endpoint} images/class', loc='left', fontsize=10)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(fontsize=8)
    save(figure, directory, 'model_comparison')
    shutil.copyfile(Path(__file__), directory / 'plot_dual_vlm_study.py')
    write_json(directory / 'figure_provenance.json', {
        'script': str(Path(__file__).resolve()), 'interpolation': 'PCHIP, 161 display positions; markers are measured budgets',
        'diagnostic_curves': 'means across 8 architectures x 3 seeds; inference in independent_comparisons.csv',
        'model_error_bars': 'descriptive SD across three seeds, not a confidence interval',
        'annotation_agreement': 'automatic annotator agreement, not accuracy',
        'tables_sha256': {str(path.relative_to(output)): sha256(path) for path in (output / 'tables').glob('*.csv')}})


if __name__ == '__main__':
    make_figures(STUDY / 'analysis')
