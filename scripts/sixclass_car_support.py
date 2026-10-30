"""Adapters for a separate six-class run; never modify the five-class freeze."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/hcf/project/QE')
STUDY = ROOT / 'artifacts/sixclass_car_extension_20261004'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
LABELS = {name: index for index, name in enumerate(CLASSES)}
SPECS = (('dog', 16), ('cat', 15), ('horse', 17), ('sheep', 18), ('person', 0), ('car', 2))

import run_multiclass_object_selection_benchmark as benchmark
sys.path.insert(0, str(ROOT / 'scripts'))
import run_cross_state_decoupled_diagnostics as decoupled
import run_coverage_incremental_value as incremental
import run_class_stratified_error_controls as stratified
import run_voc2012_confirmation_benchmark as confirmation
import run_four_model_gap_vs_random as training
import run_resnet18_five_class_coverage_comparison as classifier


def configure(study: Path = STUDY) -> None:
    global STUDY
    STUDY = study
    training.CLASS_SPECS = SPECS
    classifier.CLASS_SPECS = SPECS
    classifier.NAME_TO_LABEL = LABELS
    training.OUTPUT = study / 'training'
    for module in (benchmark, decoupled, incremental, stratified, confirmation):
        module.CLASSES = CLASSES
        module.LABELS = LABELS
        module.BUDGETS = (20, 40, 60, 80, 100)
        module.build_context = build_context
    benchmark.POOL = study
    benchmark.TRAIN_MANIFEST = study / 'manifests/train.csv'
    benchmark.SCHEMA_PATH = study / 'configs/shared_object_schema.json'
    benchmark.CHECKPOINT_ROOT = study / 'training/runs'
    benchmark.OLD_CACHE = study / 'disabled_five_class_cache'
    benchmark.summarize = summarize_benchmark
    decoupled.BASE = study / 'benchmark'
    incremental.BASE = study / 'benchmark'
    decoupled.FEATURE_GROUPS = {
        'scale_visibility': ('target_scale', 'body_visibility'),
        'orientation_pose': ('body_orientation',),
        'occlusion_truncation': ('occlusion_level', 'occluder_type', 'truncation_level'),
        'context_scene': ('group_context', 'interaction_context', 'scene_environment', 'background_complexity'),
        'quality_representation': ('blur_level', 'illumination', 'target_background_contrast',
                                   'target_recognizability', 'visual_representation'),
    }
    decoupled.error_structures = error_structures
    stratified.error_structures = error_structures
    incremental.error_structures = error_structures
    decoupled.evaluate_suite = heldout_metrics
    decoupled.write_reports = report_decoupled


def build_context() -> dict:
    rows = benchmark.read_csv(STUDY / 'manifests/candidate_pool.csv')
    labels = benchmark.read_csv(STUDY / 'annotations/results.csv')
    features = {row['image_id']: row for row in labels if row['status'] == 'ok'}
    if len(rows) != 2400 or len(features) != 2400:
        raise ValueError(f'six-class candidate/label counts: {len(rows)}/{len(features)}')
    if {row['sample_id'] for row in rows} != set(features):
        raise ValueError('candidate/feature identity mismatch')
    schema = json.loads((STUDY / 'configs/shared_object_schema.json').read_text())
    by_class = {}
    indices = {row['sample_id']: index for index, row in enumerate(rows)}
    for row in rows:
        row['label'] = str(LABELS[row['class_name']])
        if not Path(row['crop_path']).is_file():
            raise FileNotFoundError(row['crop_path'])
    for name in CLASSES:
        subset = [row for row in rows if row['class_name'] == name]
        if len(subset) != 400:
            raise ValueError(f'{name}: expected 400 candidates')
        matrix, states = benchmark.state_matrix(subset, features, schema)
        by_class[name] = {'rows': subset, 'matrix': matrix, 'states': states,
                          'global_indices': np.asarray([indices[row['sample_id']] for row in subset])}
    train_rows = benchmark.read_csv(STUDY / 'manifests/train.csv')
    if len(train_rows) != 1800:
        raise ValueError('expected 1800 training targets')
    return {'rows': rows, 'features': features, 'by_class': by_class, 'train_rows': train_rows}


def error_structures(context, features, predicted, model, seed):
    return confirmation.confirmation_error_structures(context, features, predicted, seed)


def heldout_metrics(context, masks, heldout, orders, budget, predicted, structures):
    class_rows, selected_global = [], []
    for name in CLASSES:
        section = context['by_class'][name]
        chosen = orders[name][:budget]
        global_indices = section['global_indices']
        chosen_global = global_indices[np.asarray(chosen)]
        selected_global.extend(chosen_global.tolist())
        correct = predicted[global_indices] == LABELS[name]
        state = decoupled.heldout_state_metrics(section['matrix'][:, masks[name][heldout]], chosen, correct)
        structure = structures[name]
        errors = [int(index) for index in chosen_global if int(index) in structure['wrong_global']]
        predictions = {int(predicted[index]) for index in errors}
        clusters = {structure['cluster_by_global'][index] for index in errors}
        pc, kc = len(structure['possible_predictions']), structure['cluster_count']
        class_rows.append({**state, 'confusion_pair_count': len(predictions),
                           'confusion_pair_coverage': len(predictions) / pc if pc else 0.,
                           'error_cluster_count': len(clusters),
                           'error_cluster_coverage': len(clusters) / kc if kc else 0.,
                           'error_rate': len(errors) / budget})
    selected = np.asarray(selected_global)
    true = np.asarray([int(context['rows'][i]['label']) for i in selected])
    return {
        'heldout_frequency_coverage': float(np.mean([row['frequency_coverage'] for row in class_rows])),
        'heldout_presence_coverage': float(np.mean([row['presence_coverage'] for row in class_rows])),
        'heldout_failed_state_fraction': float(np.mean([row['failed_state_fraction'] for row in class_rows])),
        'heldout_failed_state_count': sum(int(row['failed_state_count']) for row in class_rows),
        'heldout_eligible_state_count': sum(int(row['eligible_state_count']) for row in class_rows),
        'confusion_pair_coverage': float(np.mean([row['confusion_pair_coverage'] for row in class_rows])),
        'confusion_pair_count': sum(row['confusion_pair_count'] for row in class_rows),
        'error_cluster_coverage': float(np.mean([row['error_cluster_coverage'] for row in class_rows])),
        'error_cluster_count': sum(row['error_cluster_count'] for row in class_rows),
        'macro_error_rate': float(np.mean([row['error_rate'] for row in class_rows])),
        'accuracy': float((predicted[selected] == true).mean()),
    }


def summarize_benchmark(output, context, fixed, random_rows, auc_rows, smoke):
    summary = benchmark.method_budget_summary(fixed, random_rows)
    for item in summary:
        for metric in ('macro_precision',):
            if item['method'] == 'random':
                values = [float(row[f'{metric}_mean']) for row in random_rows if int(row['budget']) == item['budget']]
            else:
                values = [float(row[metric]) for row in fixed if row['method'] == item['method'] and int(row['budget']) == item['budget']]
            item[f'{metric}_mean'] = float(np.mean(values))
            item[f'{metric}_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
    endpoints = benchmark.endpoint_category_summary(fixed, random_rows)
    model_summary = benchmark.model_endpoint_summary(fixed)
    for name, rows in (('method_budget_summary', summary), ('category_endpoint_summary', endpoints),
                       ('model_endpoint_summary', model_summary)):
        benchmark.write_csv(output / f'tables/{name}.csv', rows)
    comparisons = []
    if not smoke:
        comparisons = [benchmark.paired_comparison(auc_rows, base, 'auc_macro_failed_state_fraction')
                       for base in ('random', 'deepgini', 'dsa', 'kcenter')]
        benchmark.holm(comparisons)
    benchmark.write_csv(output / 'tables/primary_comparisons.csv', comparisons)
    benchmark.make_figures(output, summary, endpoints)
    benchmark.write_json(output / 'protocol.json', {
        'experiment': 'sixclass-car-shared-state-benchmark', 'classes': list(CLASSES),
        'models': [benchmark.MODELS[0]] if smoke else list(benchmark.MODELS),
        'seeds': [benchmark.SEEDS[0]] if smoke else list(benchmark.SEEDS),
        'shared_vlm_fields': 14, 'target_scale': 'program-derived', 'candidates_per_class': 400,
        'train_per_class': 300, 'budgets_per_class': list(benchmark.BUDGETS),
        'methods': list(benchmark.METHODS), 'random_replicates': 50 if smoke else 1000,
        'coverage_cap': 20, 'minimum_failure_state_support': 3,
        'source_checksums': {str(path.relative_to(STUDY)): benchmark.sha256(path) for path in (
            STUDY / 'manifests/train.csv', STUDY / 'manifests/candidate_pool.csv',
            STUDY / 'annotations/results.csv', STUDY / 'configs/shared_object_schema.json')},
        'limitations': ['five old classes have local labels, car has online labels',
                        'six-class metrics are not directly subtractable from old five-class metrics'],
    })
    lines = ['# 六分类共享状态基准', '', '每类400候选、训练300；14项共享状态加目标尺度。',
             '只改变测试选样，六分类checkpoint与预测冻结。', '',
             '| 方法 | 预算/类 | 覆盖 | Accuracy | Precision | Recall | Macro-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['method']} | {row['budget']} | {row['macro_frequency_coverage_mean']:.2%} | "
                     f"{row['accuracy_mean']:.2%} | {row['macro_precision_mean']:.2%} | "
                     f"{row['macro_recall_mean']:.2%} | {row['macro_f1_mean']:.2%} |")
    (output / 'report_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    benchmark.write_json(output / 'progress.json', {'status': 'smoke_complete' if smoke else 'complete',
                                                   'updated_at': benchmark.now()})


def report_decoupled(output, context, summary, comparisons, fold_comparisons, cluster_rows, smoke):
    benchmark.write_json(output / 'protocol.json', {
        'experiment': 'sixclass-car-shared-state-holdout', 'classes': list(CLASSES),
        'feature_groups': decoupled.FEATURE_GROUPS, 'budgets_per_class': list(benchmark.BUDGETS),
        'cluster_counts': cluster_rows, 'random_replicates': 50 if smoke else 1000,
        'error_cluster_rule': 'zero errors:0; one:1; otherwise max(2,min(8,round(sqrt(n))))',
        'source_checksums': {str(path.relative_to(STUDY)): benchmark.sha256(path) for path in (
            STUDY / 'benchmark/protocol.json', STUDY / 'manifests/candidate_pool.csv', STUDY / 'annotations/results.csv')},
    })
    lines = ['# 六类共享状态轮换留出', '', 'Coverage只用四组排序，第五组只评价，五组轮换。', '',
             '| 比较 | 指标 | AUC差 | 95% CI | 正向运行 | 全局Holm p |', '|---|---|---:|---|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_delta_auc']:+.4f} | "
                     f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] | {row['positive_runs']}/24 | {row['holm_global_p']:.4f} |")
    (output / 'report_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def category_specific_diagnostics(study: Path = STUDY) -> None:
    context = build_context()
    old_schema = json.loads((ROOT / 'configs/multiclass_object_state_schema_v1_0.json').read_text())
    car_schema = json.loads((study / 'configs/car_annotation_schema.json').read_text())
    old_features = {row['image_id']: row for row in benchmark.read_csv(
        ROOT / 'artifacts/multiclass_object_pool_v1_20260928/annotations/results.csv')}
    car_features = {row['image_id']: row for row in benchmark.read_csv(study / 'online_car_annotations/results.csv')}
    native, state_rows, coverage_rows = {}, [], []
    for name in CLASSES:
        schema = car_schema if name == 'car' else old_schema
        schema = {**schema, 'universal_features': schema['universal_features'] + schema.get('category_specific_features', [])}
        feature_rows = car_features if name == 'car' else old_features
        native[name] = benchmark.state_matrix(context['by_class'][name]['rows'], feature_rows, schema)
    random_sets = benchmark.random_orders(context['by_class'], 1000)
    for model in benchmark.MODELS:
        for seed in benchmark.SEEDS:
            features, predicted = decoupled.load_cached_run(context, model, seed)
            orders = incremental.load_orders(context, model, seed)
            for name in CLASSES:
                section = context['by_class'][name]
                correct = predicted[section['global_indices']] == LABELS[name]
                matrix, states = native[name]
                support = matrix.sum(axis=0)
                errors = matrix[~correct].sum(axis=0)
                for index, (field, value) in enumerate(states):
                    state_rows.append({'model': model, 'seed': seed, 'class_name': name,
                                       'feature': field, 'value': value, 'support': int(support[index]),
                                       'errors': int(errors[index]), 'error_rate': float(errors[index] / support[index])})
                for budget in benchmark.BUDGETS:
                    for method in benchmark.METHODS:
                        selected_orders = [item[name][:budget] for item in random_sets] if method == 'random' else [orders[method][name][:budget]]
                        values = [benchmark.class_state_metrics(matrix, indices, correct) for indices in selected_orders]
                        coverage_rows.append({'model': model, 'seed': seed, 'class_name': name, 'method': method,
                                              'budget': budget, 'native_state_count': matrix.shape[1],
                                              **{key: float(np.mean([item[key] for item in values])) for key in values[0]}})
            print(f'[native states] {model}/{seed} complete', flush=True)
    benchmark.write_csv(study / 'diagnostics/native_state_error_rates.csv', state_rows)
    benchmark.write_csv(study / 'diagnostics/native_state_budget_metrics.csv', coverage_rows)
    # Predeclared car-only secondary tests; errors remain matched inside car.
    per_class = benchmark.read_csv(study / 'strict/tables/class_equal_error_by_class.csv')
    curves = defaultdict(list)
    for row in per_class:
        if row['class_name'] == 'car':
            for field in ('left_confusion', 'right_confusion', 'left_clusters', 'right_clusters'):
                row[field] = float(row[field])
            row['delta_confusion'] = row['left_confusion'] - row['right_confusion']
            row['delta_clusters'] = row['left_clusters'] - row['right_clusters']
            curves[row['comparison'], row['model'], row['seed']].append(row)
    values = defaultdict(lambda: defaultdict(list))
    car_auc = []
    for (comparison, model, seed), rows in curves.items():
        for metric in ('confusion', 'clusters'):
            value = incremental.normalized_auc(rows, f'delta_{metric}')
            values[comparison, metric][model].append(value)
            car_auc.append({'comparison': comparison, 'model': model, 'seed': seed, 'metric': metric, 'auc_delta': value})
    comparisons = [incremental.paired_stats(groups, comparison, f'car_equal_error_{metric}')
                   for (comparison, metric), groups in values.items()]
    primary = [row for row in comparisons if row['comparison'] == 'coverage_vs_random']
    for row, adjusted in zip(primary, stratified.holm([row['permutation_p'] for row in primary])):
        row['holm_car_secondary_p'] = adjusted
    for row, adjusted in zip(comparisons, stratified.holm([row['permutation_p'] for row in comparisons])):
        row['holm_global_p'] = adjusted
    benchmark.write_csv(study / 'diagnostics/car_equal_error_auc.csv', car_auc)
    benchmark.write_csv(study / 'diagnostics/car_equal_error_comparisons.csv', comparisons)


def plot_and_report(study: Path = STUDY) -> None:
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    plt.rcParams['svg.fonttype'] = 'none'
    rows = benchmark.read_csv(study / 'benchmark/tables/method_budget_summary.csv')
    colors = {'coverage': '#C44E52', 'random': '#777777', 'deepgini': '#287271', 'dsa': '#4C72B0', 'kcenter': '#DD8452'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = (('macro_frequency_coverage_mean', 'Shared state coverage'), ('macro_f1_mean', 'Macro F1'),
               ('accuracy_mean', 'Accuracy'), ('macro_precision_mean', 'Macro precision'),
               ('macro_recall_mean', 'Macro recall'), ('macro_error_rate_mean', 'Error rate'))
    for axis, (metric, title) in zip(axes.flat, metrics):
        for method, color in colors.items():
            subset = sorted([row for row in rows if row['method'] == method], key=lambda row: int(row['budget']))
            x = np.asarray([int(row['budget']) for row in subset])
            y = np.asarray([float(row[metric]) for row in subset])
            dense = np.linspace(x.min(), x.max(), 160)
            axis.plot(dense, PchipInterpolator(x, y)(dense), color=color, label=method)
            axis.scatter(x, y, s=16, color=color)
        axis.set_title(title)
        axis.set_xlabel('Test samples per class')
        axis.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figures = study / 'figures'
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / 'sixclass_budget_response.png', dpi=220)
    fig.savefig(figures / 'sixclass_budget_response.svg')
    plt.close(fig)
    comparisons = benchmark.read_csv(study / 'strict/tables/class_equal_error_comparisons.csv')
    car_comparisons = benchmark.read_csv(study / 'diagnostics/car_equal_error_comparisons.csv')
    lines = ['# 加入car的六分类扩展实验结果', '',
             '完整设置见运行前方案。原五类各300训练/400候选，car同规模新增；24个重新训练的六分类模型运行。',
             '14项共享状态+目标尺度参与主选样；原五类19字段和car18字段全部保留用于原生状态诊断。', '',
             '## 主指标：六类逐类别等错误数', '',
             '| 指标 | Coverage−Random AUC | 95% CI | 正向运行 | Holm p |', '|---|---:|---|---:|---:|']
    for row in comparisons:
        if row['comparison'] == 'coverage_vs_random':
            lines.append(f"| {row['metric']} | {float(row['mean_delta_auc']):+.4f} | "
                         f"[{float(row['ci95_low']):+.4f},{float(row['ci95_high']):+.4f}] | {row['positive_runs']}/24 | {float(row['holm_primary_p']):.4f} |")
    lines += ['', '## 次要指标：car单类逐类别等错误数', '',
              '| 指标 | AUC差 | 95% CI | 正向运行 | car家族Holm p |', '|---|---:|---|---:|---:|']
    for row in car_comparisons:
        if row['comparison'] == 'coverage_vs_random':
            lines.append(f"| {row['metric']} | {float(row['mean_delta_auc']):+.4f} | "
                         f"[{float(row['ci95_low']):+.4f},{float(row['ci95_high']):+.4f}] | {row['positive_runs']}/24 | {float(row['holm_car_secondary_p']):.4f} |")
    passed = [row['metric'] for row in comparisons if row['comparison'] == 'coverage_vs_random'
              and float(row['mean_delta_auc']) > 0 and float(row['holm_primary_p']) < .05]
    lines += ['', '## 判定与边界', '',
              f'预定两项主指标通过数：{len(passed)}/2。通过指标：{", ".join(passed) or "无"}。',
              'car单类结果用于判断刚性类别的诊断表现，不能由六类平均为正替代。',
              'F1波动是测试分布变化；没有要求Coverage选样F1单调下降。',
              '原五类本地标注、car线上标注存在类别与标注器混杂；六类结果不与旧五类F1作直接差值。',
              '曲线由5个真实预算点PCHIP插值，圆点是实测值，插值不增加证据。', '',
              '![六类预算响应](figures/sixclass_budget_response.png)', '',
              '## 文件', '', '- `benchmark/tables/`：五方法F1、Accuracy、Precision、Recall和覆盖指标。',
              '- `strict/tables/`：逐类别等错误数及75/25互补控制。',
              '- `holdout/tables/`：五组状态轮换留出。',
              '- `diagnostics/`：原五类专属状态、car专属状态与car独立比较。',
              '- `training/runs/`：六分类权重、训练日志和独立基础测试预测。', '']
    (study / 'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    (study / 'claim_evidence.md').write_text(
        '# Claim-Evidence Matrix\n\n六类Coverage相对Random诊断优势的主指标通过数：'
        f'{len(passed)}/2，源表为`strict/tables/class_equal_error_comparisons.csv`。\n\n'
        'car泛化须独立读取`diagnostics/car_equal_error_comparisons.csv`；结果不支持任意类别通用或因果归因。\n', encoding='utf-8')
