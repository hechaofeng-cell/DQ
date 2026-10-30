#!/usr/bin/env python3
"""Cross-evaluate paired VLM labels against immutable classifier predictions.

No training or classifier inference is performed here. State denominators are
aligned to the per-class union. Primary diagnostics use no VLM state values.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

from dual_vlm_common import (BUDGETS, CLASSES, MODELS, ROOT, SEEDS, SOURCE, STUDY,
                             job_path, now, read_csv, schema_path, sha256, verify_freeze,
                             write_csv, write_json)
from run_dog_local_pipeline import validate_feature
import sixclass_car_support as support

support.configure(SOURCE)
b = support.benchmark
control = support.stratified
stats = support.incremental
BACKENDS = ('local', 'online')
LABELS = dict(zip(CLASSES, range(6)))
GROUPS = support.decoupled.FEATURE_GROUPS


def note(output, stage, **extra):
    value = {'status': 'running', 'stage': stage, 'heartbeat': now(), **extra}
    write_json(output / 'progress.json', value)
    print(f'[{now()}] {stage} {json.dumps(extra, ensure_ascii=False)}', flush=True)


def native_labels(smoke=False):
    labels = {backend: {} for backend in BACKENDS}
    manifest_rows = []
    for domain in ('old', 'car'):
        schema = json.loads(schema_path(domain).read_text())
        rows = read_csv(STUDY / 'inputs' / f'{domain}.csv')
        manifest_rows.extend(rows)
        for backend in BACKENDS:
            for row in rows:
                path = job_path(backend, domain) / 'parsed' / f"{row['image_id']}.json"
                if not path.exists():
                    if smoke:
                        continue
                    raise ValueError(f'missing paired annotation: {backend}/{row["image_id"]}')
                value = validate_feature(json.loads(path.read_text()), row, schema)
                labels[backend][row['image_id']] = value
    if not smoke and any(len(values) != 2400 for values in labels.values()):
        raise ValueError('formal analysis requires 2400 strict annotations for EACH backend')
    return labels, manifest_rows


def contexts(labels, smoke=False):
    all_rows = read_csv(STUDY / 'manifests/candidate_pool.csv')
    paired = set(labels['local']) & set(labels['online'])
    if smoke:
        rows = [row for name in CLASSES for row in
                [item for item in all_rows if item['class_name'] == name and item['sample_id'] in paired][:3]]
        if len(rows) != 18:
            raise ValueError('smoke needs three real paired targets per class')
    else:
        rows = all_rows
        if len(paired) != 2400:
            raise ValueError('paired target identities are incomplete')
    for row in rows:
        row['label'] = str(LABELS[row['class_name']])
    schema = json.loads((STUDY / 'configs/shared_object_schema.json').read_text())
    states = [('target_scale', value) for value in ('tiny', 'small', 'medium', 'large')]
    states += [(item['feature_id'], value) for item in schema['universal_features']
               for value in item['possible_values'] if value != 'unknown']
    index = {state: i for i, state in enumerate(states)}
    result = {backend: {'rows': rows, 'by_class': {}, 'features': {}, 'train_rows': []} for backend in BACKENDS}
    for backend in BACKENDS:
        result[backend]['features'] = {
            row['sample_id']: {field: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                               for field, value in labels[backend][row['sample_id']]['features'].items()}
            for row in rows}
    universe = []
    for name in CLASSES:
        globals_ = np.asarray([i for i, row in enumerate(rows) if row['class_name'] == name])
        subset = [rows[i] for i in globals_]
        matrices = {}
        for backend in BACKENDS:
            matrix = np.zeros((len(subset), len(states)), dtype=np.int8)
            for row_i, row in enumerate(subset):
                matrix[row_i, index['target_scale', b.scale_state(float(row['bbox_area_ratio']))]] = 1
                values = labels[backend][row['sample_id']]['features']
                for item in schema['universal_features']:
                    field = item['feature_id']
                    value = values[field]
                    for entry in value if isinstance(value, list) else [value]:
                        if (field, entry) in index:
                            matrix[row_i, index[field, entry]] = 1
            matrices[backend] = matrix
        feasible = (matrices['local'].sum(axis=0) + matrices['online'].sum(axis=0)) > 0
        aligned_states = [state for state, keep in zip(states, feasible) if keep]
        for backend in BACKENDS:
            result[backend]['by_class'][name] = {'rows': subset, 'matrix': matrices[backend][:, feasible],
                                                 'states': aligned_states, 'global_indices': globals_}
        universe += [{'class_name': name, 'feature': field, 'value': value,
                      'local_support': int(matrices['local'][:, index[field, value]].sum()),
                      'online_support': int(matrices['online'][:, index[field, value]].sum())}
                     for field, value in aligned_states]
    return result, universe


def agreement(labels, rows, output):
    shared = {item['feature_id'] for item in json.loads((STUDY / 'configs/shared_object_schema.json').read_text())['universal_features']}
    records, frequencies = [], []
    for name in CLASSES:
        keys = [row['sample_id'] for row in rows if row['class_name'] == name]
        schema = json.loads(schema_path('car' if name == 'car' else 'old').read_text())
        for definition in schema['universal_features'] + schema.get('category_specific_features', []):
            field = definition['feature_id']
            sets = {backend: [set(value) if isinstance(value, list) else {value}
                              for key in keys for value in [labels[backend][key]['features'][field]]]
                    for backend in BACKENDS}
            exact = [left == right for left, right in zip(sets['local'], sets['online'])]
            jaccard = [len(left & right) / len(left | right) for left, right in zip(sets['local'], sets['online'])]
            encoded = {backend: ['|'.join(sorted(value)) for value in values] for backend, values in sets.items()}
            kappa = ''
            degenerate = len(set(encoded['local'] + encoded['online'])) == 1
            if not degenerate:
                value = float(cohen_kappa_score(encoded['local'], encoded['online']))
                if math.isfinite(value):
                    kappa = value
            records.append({'class_name': name, 'feature': field, 'shared': field in shared,
                            'value_type': definition['value_type'], 'n': len(keys),
                            'exact_agreement': float(np.mean(exact)), 'mean_jaccard': float(np.mean(jaccard)),
                            'cohen_kappa': kappa, 'kappa_degenerate': degenerate,
                            **{f'{backend}_unknown_rate': float(np.mean(['unknown' in value for value in sets[backend]]))
                               for backend in BACKENDS}})
            for value in definition['possible_values']:
                frequencies.append({'class_name': name, 'feature': field, 'value': value,
                                    'shared': field in shared, 'n': len(keys),
                                    **{f'{backend}_count': sum(value in entry for entry in sets[backend]) for backend in BACKENDS}})
    write_csv(output / 'tables/annotation_agreement.csv', records)
    write_csv(output / 'tables/state_frequencies.csv', frequencies)
    return records


def source_order(path, context, method=None):
    saved = read_csv(path)
    output = {}
    for name in CLASSES:
        mapping = {row['sample_id']: i for i, row in enumerate(context['by_class'][name]['rows'])}
        selected = sorted((row for row in saved if row['class_name'] == name
                           and row['sample_id'] in mapping and (method is None or row.get('method') == method)),
                          key=lambda row: int(row['rank']))
        order = [mapping[row['sample_id']] for row in selected]
        if len(order) != len(mapping) or len(set(order)) != len(mapping):
            raise ValueError(f'frozen baseline order mismatch: {path}/{name}')
        output[name] = order
    return output


def prediction_cache(context, model, seed):
    path = SOURCE / 'benchmark/runs' / model / str(seed) / 'candidate_outputs.npz'
    with np.load(path) as cache:
        ids = cache['sample_ids'].tolist()
        full = [row['sample_id'] for row in read_csv(STUDY / 'manifests/candidate_pool.csv')]
        if ids != full or cache['logits'].shape != (2400, 6):
            raise ValueError(f'frozen classifier identity/shape mismatch: {path}')
        lookup = {key: i for i, key in enumerate(ids)}
        subset = np.asarray([lookup[row['sample_id']] for row in context['rows']])
        return cache['features'][subset].astype(np.float32), cache['logits'][subset].argmax(axis=1)


def group_masks(context):
    fields = {field for values in GROUPS.values() for field in values}
    if len(fields) != sum(len(values) for values in GROUPS.values()):
        raise ValueError('overlapping holdout groups')
    output = {}
    for name in CLASSES:
        states = context['by_class'][name]['states']
        if {field for field, _value in states} - fields:
            raise ValueError('unassigned shared field')
        output[name] = {group: np.asarray([field in values for field, _value in states]) for group, values in GROUPS.items()}
    return output


def all_coverage_orders(contexts_, output):
    complete, heldout, records = {}, {}, []
    for backend, context in contexts_.items():
        masks = group_masks(context)
        complete[backend] = {}
        heldout[backend] = {group: {} for group in GROUPS}
        for name in CLASSES:
            section = context['by_class'][name]
            complete[backend][name] = b.concave_coverage_order(section['rows'], section['matrix'])
            for group in GROUPS:
                heldout[backend][group][name] = b.concave_coverage_order(section['rows'], section['matrix'][:, ~masks[name][group]])
            for group, order in [('all', complete[backend][name]),
                                 *[(group, heldout[backend][group][name]) for group in GROUPS]]:
                records += [{'selection_source': backend, 'heldout_group': group, 'class_name': name, 'rank': rank + 1,
                             'sample_id': section['rows'][i]['sample_id']} for rank, i in enumerate(order)]
    write_csv(output / 'tables/coverage_orders.csv', records)
    return complete, heldout


def overlap(orders, output, budgets):
    rows, rank_rows = [], []
    for name in CLASSES:
        for budget in budgets:
            left, right = set(orders['local'][name][:budget]), set(orders['online'][name][:budget])
            rows.append({'class_name': name, 'budget': budget, 'intersection': len(left & right),
                         'overlap_fraction': len(left & right) / budget,
                         'jaccard': len(left & right) / len(left | right)})
        local_rank = {value: rank for rank, value in enumerate(orders['local'][name])}
        online_rank = {value: rank for rank, value in enumerate(orders['online'][name])}
        ids = sorted(local_rank)
        coefficient = float(spearmanr([local_rank[i] for i in ids], [online_rank[i] for i in ids]).statistic)
        rank_rows.append({'class_name': name, 'n': len(ids), 'full_order_spearman': coefficient})
    write_csv(output / 'tables/selection_overlap.csv', rows)
    write_csv(output / 'tables/selection_rank_agreement.csv', rank_rows)
    return rows


def prepared_selection(context, orders, budget):
    chosen = {name: np.asarray([order[name][:budget] for order in orders], dtype=int) for name in CLASSES}
    if any(indices.shape != (len(orders), budget) for indices in chosen.values()):
        raise ValueError('underfilled selection')
    globals_ = np.concatenate([context['by_class'][name]['global_indices'][chosen[name]] for name in CLASSES], axis=1)
    return {'chosen': chosen, 'globals': globals_, 'budget': budget}


def state_stats(context, selection, predicted=None):
    output = {}
    for name in CLASSES:
        section = context['by_class'][name]
        selected = section['matrix'][selection['chosen'][name]]
        support_counts = selected.sum(axis=1)
        if predicted is None:
            errors = None
        else:
            wrong = predicted[section['global_indices'][selection['chosen'][name]]] != LABELS[name]
            errors = np.einsum('rbs,rb->rs', selected, wrong.astype(np.int8), dtype=np.int32)
        output[name] = (support_counts, errors)
    return output


def measures(context, selection, predicted, counts, heldout=None):
    repetitions = len(selection['globals'])
    budget = selection['budget']
    predictions = predicted[selection['globals']]
    true = np.repeat(np.arange(6), budget)
    offsets = np.arange(repetitions)[:, None] * 36
    encoded = offsets + true[None, :] * 6 + predictions
    confusion = np.bincount(encoded.ravel(), minlength=repetitions * 36).reshape(repetitions, 6, 6)
    tp = np.diagonal(confusion, axis1=1, axis2=2)
    precision = np.divide(tp, confusion.sum(axis=1), out=np.zeros_like(tp, dtype=float), where=confusion.sum(axis=1) > 0)
    recall = tp / budget
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=precision + recall > 0)
    result = {'accuracy': tp.sum(axis=1) / (budget * 6), 'macro_precision': precision.mean(axis=1),
              'macro_recall': recall.mean(axis=1), 'macro_f1': f1.mean(axis=1)}
    masks = group_masks(context) if heldout is not None else None
    aggregate = defaultdict(list)
    for name in CLASSES:
        supports, errors = counts[name]
        if errors is None:
            raise ValueError('state error counts missing')
        if masks is not None:
            supports, errors = supports[:, masks[name][heldout]], errors[:, masks[name][heldout]]
        eligible = supports >= b.MIN_STATE_SUPPORT
        state_recall = np.divide(supports - errors, supports, out=np.ones_like(supports, dtype=float), where=supports > 0)
        width = supports.shape[1]
        values = {
            'frequency_coverage': np.minimum(supports / b.COVERAGE_CAP, 1).mean(axis=1) if width else np.zeros(repetitions),
            'presence_coverage': (supports > 0).mean(axis=1) if width else np.zeros(repetitions),
            'failed_state_count': ((errors > 0) & eligible).sum(axis=1),
            'failed_state_fraction': ((errors > 0) & eligible).mean(axis=1) if width else np.zeros(repetitions),
            'eligible_state_count': eligible.sum(axis=1),
            'state_macro_recall': np.divide((state_recall * eligible).sum(axis=1), eligible.sum(axis=1),
                                          out=np.ones(repetitions), where=eligible.sum(axis=1) > 0),
            'error_rate': 1 - recall[:, LABELS[name]],
        }
        for metric, array in values.items():
            result[f'{name}_{metric}'] = array
            aggregate[metric].append(array)
        result[f'{name}_precision'] = precision[:, LABELS[name]]
        result[f'{name}_recall'] = recall[:, LABELS[name]]
        result[f'{name}_f1'] = f1[:, LABELS[name]]
    for target, metric in [('macro_frequency_coverage', 'frequency_coverage'), ('macro_presence_coverage', 'presence_coverage'),
                           ('macro_failed_state_fraction', 'failed_state_fraction'), ('macro_state_recall', 'state_macro_recall'),
                           ('macro_error_rate', 'error_rate')]:
        result[target] = np.asarray(aggregate[metric]).mean(axis=0)
    result['total_failed_state_count'] = np.asarray(aggregate['failed_state_count']).sum(axis=0)
    return result


def measured_one(context, order, budget, predicted, heldout=None):
    selection = prepared_selection(context, [order], budget)
    counts = state_stats(context, selection, predicted)
    values = measures(context, selection, predicted, counts, heldout)
    return {key: float(value[0]) for key, value in values.items()}


def random_summary(values):
    return {f'{key}_{suffix}': float(function(value)) for key, value in values.items()
            for suffix, function in [('mean', np.mean), ('p2_5', lambda x: np.percentile(x, 2.5)),
                                      ('p97_5', lambda x: np.percentile(x, 97.5))]}


def validate_batch_metrics(context, predicted, random_sets):
    # Reversible diagnostic: compare a nontrivial multi-suite vectorized path to
    # the frozen scalar evaluator on real predictions and unknown-bearing labels.
    for budget in (1, 3):
        selection = prepared_selection(context, random_sets[:3], budget)
        values = measures(context, selection, predicted, state_stats(context, selection, predicted))
        for i, order in enumerate(random_sets[:3]):
            reference = b.evaluate_suite(context, order, budget, predicted)
            for key in reference:
                if key in values and not np.isclose(reference[key], values[key][i], atol=1e-12, rtol=0):
                    raise ValueError(f'vectorized metric mismatch: {key}/{budget}/{i}: {reference[key]} vs {values[key][i]}')


def run_analysis(output, smoke=False):
    verify_code_freeze()
    note(output, 'validate_annotations')
    labels, manifest_rows = native_labels(smoke)
    if not smoke:
        for backend in BACKENDS:
            flat = []
            for row in manifest_rows:
                parsed = labels[backend][row['image_id']]
                flat.append({'image_id': row['image_id'], 'target_class': row['target_class'], 'status': 'ok',
                             'annotation_backend': backend, 'schema_version': parsed['schema_version'],
                             **{field: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                                for field, value in parsed['features'].items()}})
            write_csv(STUDY / 'annotations' / backend / 'paired_results.csv', flat)
        write_json(STUDY / 'paired_annotation_freeze.sha256.json', {
            str(path.relative_to(STUDY)): sha256(path) for backend in BACKENDS for domain in ('old', 'car')
            for path in (job_path(backend, domain) / 'parsed').glob('*.json')})
    contexts_, universe = contexts(labels, smoke)
    context = contexts_['local']
    budgets = (1, 2, 3) if smoke else BUDGETS
    runs = [(MODELS[0], SEEDS[0])] if smoke else list(itertools.product(MODELS, SEEDS))
    repetitions = 20 if smoke else 1000
    b.BUDGETS = budgets
    stats.BUDGETS = budgets
    write_csv(output / 'tables/aligned_state_universe.csv', universe)
    agreements = agreement(labels, context['rows'], output)
    complete, heldout = all_coverage_orders(contexts_, output)
    overlap_rows = overlap(complete, output, budgets)
    random_sets = b.random_orders(context['by_class'], repetitions)
    random_selections = {budget: prepared_selection(context, random_sets, budget) for budget in budgets}
    supports = {backend: {budget: state_stats(ctx, random_selections[budget])
                         for budget in budgets} for backend, ctx in contexts_.items()}
    # Selection identity is shared across label sources, including all baselines.
    kcenter = source_order(SOURCE / 'benchmark/tables/model_independent_orders.csv', context, 'kcenter')
    fixed_rows, random_rows, holdout_rows, holdout_random = [], [], [], []
    strict_rows, strict_classes, auc_rows, class_auc_rows = [], [], [], []
    values = defaultdict(lambda: defaultdict(list))
    holdout_auc = defaultdict(lambda: defaultdict(list))
    for run_i, (model, seed) in enumerate(runs, 1):
        note(output, 'diagnostics', completed_runs=run_i - 1, total_runs=len(runs), model=model, seed=seed)
        features, predicted = prediction_cache(context, model, seed)
        structures = support.error_structures(context, features, predicted, model, seed)
        frozen = {'kcenter': kcenter, **{method: source_order(SOURCE / 'benchmark/orders' / model / str(seed) / f'{method}.csv', context)
                                       for method in ('deepgini', 'dsa')}}
        if run_i == 1:
            for ctx in contexts_.values():
                validate_batch_metrics(ctx, predicted, random_sets)
        random_means = {}
        random_group_means = {}
        for evaluator, ctx in contexts_.items():
            random_means[evaluator] = {}
            random_group_means[evaluator] = defaultdict(dict)
            for budget in budgets:
                counts = state_stats(ctx, random_selections[budget], predicted)
                for name in CLASSES:
                    if not np.array_equal(counts[name][0], supports[evaluator][budget][name][0]):
                        raise ValueError('random support changed with classifier')
                batch = measures(ctx, random_selections[budget], predicted, counts)
                random_means[evaluator][budget] = {key: float(value.mean()) for key, value in batch.items()}
                random_rows.append({'evaluation_source': evaluator, 'method': 'random', 'model': model, 'seed': seed,
                                    'budget': budget, 'replicates': repetitions, **random_summary(batch)})
                for group in GROUPS:
                    group_batch = measures(ctx, random_selections[budget], predicted, counts, group)
                    random_group_means[evaluator][group][budget] = {key: float(value.mean()) for key, value in group_batch.items()}
                    holdout_random.append({'selection_source': 'online' if evaluator == 'local' else 'local',
                                           'evaluation_source': evaluator, 'heldout_group': group, 'method': 'random',
                                           'model': model, 'seed': seed, 'budget': budget, 'replicates': repetitions,
                                           **random_summary(group_batch)})
        for selector in BACKENDS:
            orders = {'coverage': complete[selector], **frozen}
            curves = defaultdict(list)
            class_curves = defaultdict(list)
            for evaluator, ctx in contexts_.items():
                for method, order in orders.items():
                    for budget in budgets:
                        metrics = measured_one(ctx, order, budget, predicted)
                        fixed_rows.append({'selection_source': selector, 'evaluation_source': evaluator,
                                           'method': method, 'model': model, 'seed': seed, 'budget': budget, **metrics})
            for budget in budgets:
                for baseline in ('random', 'deepgini', 'dsa', 'kcenter'):
                    if baseline == 'random':
                        pairs = [control.class_stratified_pair(context, orders['coverage'], order, budget, predicted, structures)
                                 for order in random_sets]
                        metrics, per_class = control.aggregate_random_pairs(pairs)
                    else:
                        metrics, per_class = control.class_stratified_pair(context, orders['coverage'], orders[baseline], budget, predicted, structures)
                    record = control.add_delta({'selection_source': selector, 'comparison': f'coverage_vs_{baseline}',
                                                'model': model, 'seed': seed, 'budget': budget, **metrics})
                    strict_rows.append(record)
                    curves[baseline].append(record)
                    for item in per_class:
                        class_record = control.add_delta({'selection_source': selector, 'comparison': f'coverage_vs_{baseline}',
                                                        'model': model, 'seed': seed, 'budget': budget, **item})
                        strict_classes.append(class_record)
                        class_curves[baseline, item['class_name']].append(class_record)
            for baseline, rows in curves.items():
                item = {'selection_source': selector, 'comparison': f'coverage_vs_{baseline}', 'model': model, 'seed': seed}
                for metric in ('confusion', 'clusters'):
                    value = stats.normalized_auc(rows, f'delta_{metric}')
                    item[f'auc_delta_{metric}'] = value
                    values[selector, baseline, metric][model].append(value)
                auc_rows.append(item)
            for (baseline, name), rows in class_curves.items():
                class_auc_rows.append({'selection_source': selector, 'comparison': f'coverage_vs_{baseline}',
                                       'model': model, 'seed': seed, 'class_name': name,
                                       **{f'auc_delta_{metric}': stats.normalized_auc(rows, f'delta_{metric}') for metric in ('confusion', 'clusters')}})
            opposite = 'online' if selector == 'local' else 'local'
            ctx = contexts_[opposite]
            for group in GROUPS:
                for method in ('coverage', 'deepgini', 'dsa', 'kcenter'):
                    order = heldout[selector][group] if method == 'coverage' else frozen[method]
                    curve = []
                    for budget in budgets:
                        metrics = measured_one(ctx, order, budget, predicted, group)
                        holdout_rows.append({'selection_source': selector, 'evaluation_source': opposite,
                                             'heldout_group': group, 'method': method, 'model': model, 'seed': seed,
                                             'budget': budget, **metrics})
                        curve.append({'budget': budget, 'delta_failure': metrics['macro_failed_state_fraction']
                                      - random_group_means[opposite][group][budget]['macro_failed_state_fraction']})
                    if method == 'coverage':
                        holdout_auc[selector, group][model].append(stats.normalized_auc(curve, 'delta_failure'))
        note(output, 'diagnostics', completed_runs=run_i, total_runs=len(runs), model=model, seed=seed)
    comparisons = []
    for (selector, baseline, metric), grouped in values.items():
        row = stats.paired_stats(grouped, f'{selector}_coverage_vs_{baseline}', f'class_equal_error_{metric}')
        row.update(selection_source=selector, baseline=baseline)
        comparisons.append(row)
    primary = [row for row in comparisons if row['baseline'] == 'random']
    for row, adjusted in zip(primary, control.holm([row['permutation_p'] for row in primary])):
        row['holm_primary_four_p'] = adjusted
    for row, adjusted in zip(comparisons, control.holm([row['permutation_p'] for row in comparisons])):
        row['holm_all_sixteen_p'] = adjusted
    source_comparisons = []
    for metric in ('confusion', 'clusters'):
        grouped = {model: (np.asarray(values['online', 'random', metric][model])
                           - np.asarray(values['local', 'random', metric][model])).tolist() for model, _seed in runs}
        source_comparisons.append(stats.paired_stats(grouped, 'online_minus_local_random_gain', f'class_equal_error_{metric}'))
    for row, adjusted in zip(source_comparisons, control.holm([row['permutation_p'] for row in source_comparisons])):
        row['holm_two_p'] = adjusted
    category_comparisons = []
    for selector in BACKENDS:
        for name in CLASSES:
            for metric in ('confusion', 'clusters'):
                grouped = defaultdict(list)
                for row in class_auc_rows:
                    if row['selection_source'] == selector and row['class_name'] == name and row['comparison'] == 'coverage_vs_random':
                        grouped[row['model']].append(row[f'auc_delta_{metric}'])
                record = stats.paired_stats(grouped, f'{selector}_{name}_coverage_vs_random', metric)
                category_comparisons.append({**record, 'selection_source': selector, 'class_name': name})
    car = [row for row in category_comparisons if row['class_name'] == 'car']
    for row, adjusted in zip(car, control.holm([row['permutation_p'] for row in car])):
        row['holm_car_four_p'] = adjusted
    for row, adjusted in zip(category_comparisons, control.holm([row['permutation_p'] for row in category_comparisons])):
        row['holm_categories_exploratory_p'] = adjusted
    holdout_comparisons = []
    for (selector, group), grouped in holdout_auc.items():
        record = stats.paired_stats(grouped, f'{selector}_cross_source_{group}_coverage_vs_random', 'heldout_failed_state_fraction')
        holdout_comparisons.append({**record, 'selection_source': selector, 'heldout_group': group})
    for row, adjusted in zip(holdout_comparisons, control.holm([row['permutation_p'] for row in holdout_comparisons])):
        row['holm_holdout_ten_p'] = adjusted
    tables = {'cross_fixed_metrics': fixed_rows, 'cross_random_summary': random_rows,
              'equal_error_metrics': strict_rows, 'equal_error_by_class': strict_classes,
              'equal_error_auc': auc_rows, 'equal_error_class_auc': class_auc_rows,
              'independent_comparisons': comparisons, 'annotation_source_gain_comparisons': source_comparisons,
              'category_independent_comparisons': category_comparisons,
              'cross_source_holdout_metrics': holdout_rows, 'cross_source_holdout_random': holdout_random,
              'cross_source_holdout_comparisons': holdout_comparisons}
    for name, rows in tables.items():
        write_csv(output / 'tables' / f'{name}.csv', rows)
    rank_rows, rank_agreement = rankings(fixed_rows, output, budgets)
    summarize(output, agreements, overlap_rows, comparisons, source_comparisons,
              fixed_rows, random_rows, rank_rows, rank_agreement, smoke)
    audit_results(output, tables, contexts_, runs, budgets, repetitions, smoke)
    if not smoke:
        record_completion(output)
    note(output, 'complete', completed_runs=len(runs), total_runs=len(runs))
    write_json(output / 'progress.json', {'status': 'smoke_complete' if smoke else 'complete', 'heartbeat': now()})


def rankings(fixed, output, budgets):
    rows, agreements = [], []
    observed_models = sorted({row['model'] for row in fixed})
    for budget in budgets:
        ranks = {}
        for source in BACKENDS:
            data = [row for row in fixed if row['selection_source'] == source and row['evaluation_source'] == source
                    and row['method'] == 'coverage' and row['budget'] == budget]
            summaries = []
            for model in observed_models:
                group = [row for row in data if row['model'] == model]
                item = {'selection_source': source, 'budget': budget, 'model': model, 'seeds': len(group)}
                for metric in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
                    values = [row[metric] for row in group]
                    item[f'{metric}_mean'] = float(np.mean(values))
                    item[f'{metric}_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
                summaries.append(item)
            summaries.sort(key=lambda row: (-row['macro_f1_mean'], row['model']))
            # Average ranks preserve ties rather than manufacture a preference.
            from scipy.stats import rankdata
            scores = [row['macro_f1_mean'] for row in summaries]
            averaged = rankdata(-np.asarray(scores), method='average')
            ranks[source] = {}
            for row, rank in zip(summaries, averaged):
                row['macro_f1_rank'] = float(rank)
                ranks[source][row['model']] = float(rank)
            rows.extend(summaries)
        tau = kendalltau([ranks['local'][model] for model in observed_models],
                         [ranks['online'][model] for model in observed_models]).statistic if len(observed_models) > 1 else None
        agreements.append({'budget': budget, 'models': len(observed_models),
                           'kendall_tau': float(tau) if tau is not None and math.isfinite(tau) else '',
                           'mean_absolute_rank_change': float(np.mean([abs(ranks['local'][model] - ranks['online'][model]) for model in observed_models]))})
    write_csv(output / 'tables/model_rankings.csv', rows)
    write_csv(output / 'tables/model_ranking_agreement.csv', agreements)
    return rows, agreements


def summarize(output, agreements, overlap_rows, comparisons, source_comparisons, fixed, random_rows, ranks, rank_agreement, smoke):
    primary = [row for row in comparisons if row['baseline'] == 'random']
    passed = [row for row in primary if row['mean_delta_auc'] > 0 and row['holm_primary_four_p'] < .05]
    shared = [row for row in agreements if row['shared']]
    weighted = sum(row['exact_agreement'] * row['n'] for row in shared) / sum(row['n'] for row in shared)
    summary = {'status': 'technical_smoke_only' if smoke else 'complete', 'primary_passed': len(passed),
               'primary_tests': len(primary), 'shared_exact_agreement': weighted,
               'supported_within_setting': not smoke and len(passed) == 4,
               'source_gain_difference_not_equivalence': source_comparisons,
               'model_rank_agreement': rank_agreement,
               'interpretation': 'Both automatic annotators support the two independent Random comparisons in this pool.'
                                 if not smoke and len(passed) == 4 else 'Partial or unresolved cross-annotator support; inspect all tests and directions.',
               'limitations': ['automatic label agreement is not human accuracy', 'same pool and classifier training configuration',
                              'historical/current model aliases and constrained decoding differ', 'negative strong-baseline results retained']}
    write_json(output / 'summary.json', summary)
    lines = ['# 六类别线上／本地VLM对齐结果', '',
             '技术smoke，仅验证管线，不作为研究证据。' if smoke else '同一2400目标、同一24个冻结分类模型预测；仅改变状态标注与Coverage选样。', '',
             f'14共享字段精确一致率（含unknown）：{weighted:.2%}。该值不是人工准确率。', '',
             '| 标注来源 | 独立指标 | Coverage−Random AUC | 95% CI | 正向运行 | 四项Holm p | 全16项Holm p |',
             '|---|---|---:|---|---:|---:|---:|']
    for row in primary:
        lines.append(f"| {row['selection_source']} | {row['metric']} | {row['mean_delta_auc']:+.4f} | "
                     f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] | {row['positive_runs']}/{row['paired_runs']} | "
                     f"{row['holm_primary_four_p']:.4f} | {row['holm_all_sixteen_p']:.4f} |")
    lines += ['', f'预定主指标通过{len(passed)}/4。两来源四项均通过才支持本设置下跨标注源复现；未显著来源差异不能解释为等价。', '',
              '强基线、car次要对比与其他类别探索结果全部见tables/，不能由相对随机的结果替代。', '',
              '## 同预算最终测试指标', '', '| 选样来源 | 评价来源 | 方法 | 预算/类 | 覆盖 | 错误率 | Macro-F1 |',
              '|---|---|---|---:|---:|---:|---:|']
    endpoint = max(row['budget'] for row in fixed)
    for selector, evaluator, method in itertools.product(BACKENDS, BACKENDS, ('coverage', 'deepgini', 'dsa', 'kcenter', 'random')):
        rows = [row for row in fixed if row['selection_source'] == selector and row['evaluation_source'] == evaluator
                and row['method'] == method and row['budget'] == endpoint]
        if method == 'random':
            rows = [row for row in random_rows if row['evaluation_source'] == evaluator and row['budget'] == endpoint]
        suffix = '_mean' if method == 'random' else ''
        means = [float(np.mean([row[field + suffix] for row in rows]))
                 for field in ('macro_frequency_coverage', 'macro_error_rate', 'macro_f1')]
        lines.append(f'| {selector} | {evaluator} | {method} | {endpoint} | {means[0]:.2%} | {means[1]:.2%} | {means[2]:.2%} |')
    lines += ['', '同一选样集合的分类指标不随状态评价来源变化；状态指标随评价标签变化。', '',
              '## 跨来源留出与模型排名', '',
              '每次先移除状态组再选样，用另一VLM评价留出组；结果为辅助指标。排名按同预算Coverage测试集三种子平均Macro-F1，限本训练/测试设置。', '',
              '完整原生19/18字段保存在annotations/；主实验使用14共享字段+程序尺度。unknown不计入覆盖，覆盖分母为两来源可行状态并集。', '',
              '![对齐诊断曲线](figures/dual_vlm_diagnostics.png)', '',
              '所有曲线圆点来自实际预算评测；插值只是视觉辅助，不新增统计证据。']
    (output / 'report_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not smoke:
        from plot_dual_vlm_study import make_figures
        make_figures(output)


def audit_results(output, tables, contexts_, runs, budgets, repetitions, smoke):
    expected = {'cross_fixed_metrics': len(runs) * 2 * 2 * 4 * len(budgets),
                'cross_random_summary': len(runs) * 2 * len(budgets),
                'equal_error_metrics': len(runs) * 2 * 4 * len(budgets),
                'equal_error_by_class': len(runs) * 2 * 4 * len(budgets) * 6,
                'equal_error_auc': len(runs) * 2 * 4,
                'cross_source_holdout_metrics': len(runs) * 2 * 5 * 4 * len(budgets),
                'cross_source_holdout_random': len(runs) * 2 * 5 * len(budgets)}
    for name, count in expected.items():
        if len(tables[name]) != count:
            raise ValueError(f'row audit: {name} {len(tables[name])} != {count}')
    for name, rows in tables.items():
        for row in rows:
            if any(isinstance(value, (float, np.floating)) and not math.isfinite(value) for value in row.values()):
                raise ValueError(f'nonfinite result in {name}')
    for name in CLASSES:
        left, right = [contexts_[backend]['by_class'][name] for backend in BACKENDS]
        if left['states'] != right['states'] or left['matrix'].shape != right['matrix'].shape:
            raise ValueError(f'state denominator mismatch: {name}')
    # Cross-evaluating tags must NEVER change Accuracy/F1 for the same selected images.
    lookup = {(row['selection_source'], row['evaluation_source'], row['method'], row['model'], row['seed'], row['budget']): row
              for row in tables['cross_fixed_metrics']}
    for key, row in lookup.items():
        other = lookup[key[0], 'online' if key[1] == 'local' else 'local', *key[2:]]
        for metric in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
            if row[metric] != other[metric]:
                raise ValueError('state labels unexpectedly changed classifier metrics')
        if key[2] != 'coverage':
            other_selector = lookup['online' if key[0] == 'local' else 'local', *key[1:]]
            for metric in ('accuracy', 'macro_frequency_coverage', 'macro_f1'):
                if row[metric] != other_selector[metric]:
                    raise ValueError('frozen strong baseline changed with annotator selection source')
    source_check = verify_freeze()
    code_check = verify_code_freeze()
    write_json(output / 'final_audit.json', {'status': 'pass', 'technical_smoke_only': smoke,
               'actual_candidate_targets': len(contexts_['local']['rows']), 'classifier_runs': len(runs),
               'random_replicates': repetitions, 'observed_rows': {key: len(rows) for key, rows in tables.items()},
               'same_state_denominators': True, 'classification_label_source_invariance': True,
               'frozen_baseline_invariance': True, 'source_freeze': source_check, 'analysis_code_freeze': code_check})
    write_json(output / 'artifact_manifest.sha256.json', {str(path.relative_to(output)): sha256(path)
               for path in output.rglob('*') if path.is_file() and path.name not in ('artifact_manifest.sha256.json', 'progress.json')})


def verify_code_freeze():
    path = STUDY / 'analysis_code_freeze.sha256.json'
    names = ('analyze_dual_vlm_study.py', 'plot_dual_vlm_study.py', 'dual_vlm_common.py',
             'annotate_dual_vlm.py', 'run_dual_vlm_study.py', 'sixclass_car_support.py',
             'run_dog_local_pipeline.py', 'annotate_multiclass_online_sensenova.py',
             'run_class_stratified_error_controls.py', 'run_coverage_incremental_value.py',
             'run_cross_state_decoupled_diagnostics.py', 'run_multiclass_object_selection_benchmark.py',
             'run_voc2012_confirmation_benchmark.py')
    if not path.exists():
        write_json(path, {str(ROOT / 'scripts' / name): sha256(ROOT / 'scripts' / name) for name in names})
    hashes = json.loads(path.read_text())
    changed = [name for name, digest in hashes.items() if sha256(name) != digest]
    if changed:
        raise ValueError(f'analysis code changed after smoke freeze: {changed}')
    return {'status': 'pass', 'files': len(hashes)}


def record_completion(output):
    summary = json.loads((output / 'summary.json').read_text())
    comparisons = read_csv(output / 'tables/independent_comparisons.csv')
    lines = ['# 结论与证据', '',
             '全部2400目标现有线上／本地双标注，原生字段保持完整；结构校验通过不代表人工语义准确。', '',
             '| 选样来源 | 独立指标 | AUC差值 | 四项Holm p | 支持状态 |', '|---|---|---:|---:|---|']
    for row in comparisons:
        if row['baseline'] == 'random':
            supported = float(row['mean_delta_auc']) > 0 and float(row['holm_primary_four_p']) < .05
            lines.append(f"| {row['selection_source']} | {row['metric']} | {float(row['mean_delta_auc']):+.4f} | "
                         f"{float(row['holm_primary_four_p']):.4f} | {'本设置内支持' if supported else '未达到预定支持条件'} |")
    lines += ['', '证据：analysis/tables/independent_comparisons.csv、equal_error_auc.csv及final_audit.json。', '',
              '跨标注源收益差值见annotation_source_gain_comparisons.csv；未显著差异不是等价证据。',
              '全部强基线、car次要比较、类别探索、交叉状态留出及模型排名保留。不能推广为任意数据集或类别普遍优势。']
    (STUDY / 'claim_evidence.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (STUDY / 'research_log.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n## {now()} 正式分析完成\n\n')
        stream.write(f"本地/线上2400目标各一套严格标签；24个冻结分类运行，全预算和全部对照完成。主检验通过{summary['primary_passed']}/4。")
        stream.write('结果位于analysis/report_zh.md及tables/；source/code/指标不变性审计通过。所有图保留实测点及可编辑SVG。\n')
    (STUDY / 'review.md').write_text(
        '# 最终审查\n\n- 文献门：无新增外部文献主张，不适用。\n'
        '- 结果门：PASS；源图、源结果、权重、预测、程序哈希不变，目标身份及计数、有限值、状态分母、分类指标和基线不变性全部通过。\n'
        f"- 范围门：PASS；主检验通过{summary['primary_passed']}/4，结论限同一六类池及固定模型配置，不以不显著差异声称等价。\n"
        '- 自动标签一致性不代表人工准确，历史/当前模型别名、类别原生字段和传输解码差异作为限制保留。\n'
        '- 强基线负结果、car独立结果及全比较Holm敏感性全部保留。\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_analysis(STUDY / ('analysis_smoke' if args.smoke else 'analysis'), args.smoke)


if __name__ == '__main__':
    main()
