#!/usr/bin/env python3
"""Independent arithmetic and provenance audit for the QE three-strategy test.

This checker deliberately does not import the experiment's selection or metrics
functions. It reconstructs state activation, risk intervals and confusion
matrices from frozen source records and raw cached classifier outputs.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/QE_VisualState_CoverRisk_v1_20261013'
OUT = ROOT / 'artifacts/QE_ThreeStrategy_EqualRatio_20261013'
OLD = ROOT / 'artifacts/QE_ThreeStrategy_Full_20261013'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
BUDGETS = (20, 40, 100, 200, 400, 600, 800, 1000, 1200)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def close(actual, expected, label):
    assert np.isclose(float(actual), float(expected), atol=1e-9, rtol=0), (label, actual, expected)


def verify_source_arrays():
    state_rows = json.loads((BASE / 'coverage_dictionary.json').read_text())['states']
    states = [tuple(s) for s in state_rows]
    assert len(states) == len(set(states)) == 71
    risk_config = json.loads((BASE / 'risk_protocol.json').read_text())
    records = {r['sample_id']: r for r in map(json.loads, (BASE / 'prepared/records.jsonl').read_text().splitlines())}
    arrays = {}
    for source in ('local', 'online'):
        root = BASE if source == 'local' else BASE / 'online_analysis'
        arrays[source] = {}
        for c in CLASSES:
            z = dict(np.load(root / 'prepared' / f'{c}.npz', allow_pickle=False))
            assert z['matrix'].shape == (1300, 71)
            assert len(set(z['sample_ids'])) == 1300
            for i, sid in enumerate(z['sample_ids']):
                record = records[str(sid)]
                assert record['class_name'] == c
                features = record['annotations'][source]['native']['features']
                area = float(record['bbox_area_ratio'])
                scale = 'tiny' if area < .01 else 'small' if area < .03 else 'medium' if area < .1 else 'large'
                expected = []
                for field, state in states:
                    value = scale if field == 'target_scale' else features[field]
                    expected.append(int(state in value if isinstance(value, list) else state == value))
                assert np.array_equal(z['matrix'][i], expected), (source, c, sid, 'activation')
                lows = [sum(area < e for e in risk_config['scale_edges']) / 3]
                highs = lows.copy()
                for field in ('body_visibility', 'occlusion_level', 'truncation_level', 'body_orientation'):
                    mapping = risk_config['maps'][field]
                    lows.append(mapping.get(features[field], 0))
                    highs.append(mapping.get(features[field], 1))
                weights = [risk_config['dimension_weights'][name] for name in ('scale', 'visibility', 'occlusion', 'truncation', 'orientation')]
                lo = float(np.dot(lows, weights) * 100)
                hi = float(np.dot(highs, weights) * 100)
                for key, value in zip(('risk_lower', 'risk_midpoint', 'risk_upper'), (lo, (lo + hi) / 2, hi)):
                    close(z[key][i], value, (source, c, sid, key))
            arrays[source][c] = z
    return arrays, records


def state_metrics(z, indices):
    indices = np.asarray(indices, dtype=int)
    assert len(set(indices.tolist())) == len(indices)
    assert indices.min() >= 0 and indices.max() < 1300
    counts = z['matrix'][indices].sum(axis=0)
    lower = z['risk_lower'][indices]
    upper = z['risk_upper'][indices]
    return dict(presence_coverage=float(np.count_nonzero(counts) / 71),
                support_completion=float(np.minimum(counts, 20).sum() / (71 * 20)),
                risk_lower=float(lower.mean()), risk_midpoint=float(z['risk_midpoint'][indices].mean()),
                risk_upper=float(upper.mean()), high_risk_q5_fraction=float(z['high_risk'][indices].mean()),
                unknown_target_fraction=float(np.mean(upper > lower + 1e-8)),
                unknown_dimension_fraction=float(np.mean((upper - lower) / 100)))


def confusion_metrics(predictions, labels):
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    tp = confusion.diagonal()
    den = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, den, out=np.zeros(6, dtype=float), where=den != 0)
    errors = int(confusion.sum() - tp.sum())
    pairs = confusion.copy()
    np.fill_diagonal(pairs, 0)
    return dict(accuracy=float(tp.sum() / confusion.sum()), macro_f1=float(f1.mean()),
                error_count=errors, error_rate=float(errors / confusion.sum()),
                confusion_pair_coverage=float(np.count_nonzero(pairs) / 30))


def rarefied_breadth(counts, q):
    """Exact finite-population expected distinct wrong labels / five.

    q is matched per true class between compared strategies; q == 0 yields 0
    rather than an invented positive diagnosis.
    """
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    q = int(q)
    assert 0 <= q <= n
    if q == 0:
        return 0.
    denom = math.comb(n, q)
    return sum(1 - (math.comb(n - int(k), q) / denom if n - int(k) >= q else 0) for k in counts if k > 0) / 5


def selfcheck():
    values = (0, 0, 1, 2)
    exact = np.mean([len({values[i] for i in indices}) / 5 for indices in itertools.combinations(range(4), 2)])
    close(rarefied_breadth([2, 1, 1, 0, 0], 2), exact, 'independent rarefaction exhaustive check')
    metrics = confusion_metrics(np.array([0, 1, 1, 3, 5, 5]), np.arange(6))
    close(metrics['error_count'], 2, 'toy confusion')


def independent_structure(z, budget):
    order = sorted(range(1300), key=lambda i: (z['risk_midpoint'][i], str(z['sample_ids'][i])))
    tiers = np.zeros(1300, dtype=int)
    for rank, index in enumerate(order):
        tiers[index] = rank // 260 + 1
    unknown = np.rint((z['risk_upper'] - z['risk_lower']) / 20).astype(int)
    groups = {}
    for i in range(1300):
        key = (int(tiers[i]), float(z['risk_midpoint'][i]), int(unknown[i]))
        groups.setdefault(key, []).append(i)
    quotas = {}
    for tier in range(1, 6):
        keys = sorted(k for k in groups if k[0] == tier)
        b = budget // 5
        for k in keys:
            quotas[k] = len(groups[k]) * b // 260
        left = b - sum(quotas[k] for k in keys)
        priorities = sorted(keys, key=lambda k: (-(len(groups[k]) * b % 260), k))
        for key in priorities[:left]:
            quotas[key] += 1
    index_keys = {i: k for k, ids in groups.items() for i in ids}
    return tiers, groups, quotas, index_keys


def audit_selections(arrays):
    verified = []
    quota_rows = read_csv(OUT / 'selection/quotas.csv')
    trace_rows = read_csv(OUT / 'selection/greedy_trace.csv')
    trace_index = {(r['source'], int(r['budget_per_class']), r['category'], r['method'], int(r['rank'])): r for r in trace_rows}
    for source in ('local', 'online'):
        for budget in BUDGETS:
            selections = dict(np.load(OUT / 'selection' / f'{source}_n{budget * 6}.npz'))
            assert set(selections) == {'coverage', 'risk', 'risk_coverage'}
            for c, category in enumerate(CLASSES):
                z = arrays[source][category]
                tiers, groups, quotas, index_keys = independent_structure(z, budget)
                recorded = [r for r in quota_rows if r['source'] == source and int(r['budget_per_class']) == budget and r['category'] == category]
                expected = {(int(r['tier']), float(r['risk_midpoint']), int(r['unknown_count'])): int(r['quota']) for r in recorded}
                assert expected == quotas
                rng = np.random.default_rng(20261013 + (0 if source == 'local' else 100000) + budget * 100 + c)
                repeats = 1 if budget == 1300 else 1000
                assert selections['risk'].shape == (repeats, 6, budget)
                for repeat in range(repeats):
                    chosen = []
                    for k in sorted(groups):
                        if quotas[k]:
                            chosen.extend(rng.choice(groups[k], quotas[k], replace=False).tolist())
                    assert np.array_equal(selections['risk'][repeat, c], chosen), (source, budget, category, repeat, 'random replay')
                for method in ('coverage', 'risk_coverage'):
                    ids = selections[method][c]
                    assert len(ids) == len(set(ids.tolist())) == budget
                    assert int(ids.min()) >= 0 and int(ids.max()) < 1300
                    if method in ('coverage', 'risk_coverage'):
                        observed = Counter(index_keys[int(i)] for i in ids)
                        assert observed == Counter({k: v for k, v in quotas.items() if v})
                    counts = np.zeros(71, dtype=int)
                    used = Counter()
                    available = set(range(1300))
                    for rank, index in enumerate(ids, 1):
                        feasible = np.asarray([i for i in sorted(available) if used[index_keys[i]] < quotas[index_keys[i]]], dtype=int)
                        weight = np.zeros(71)
                        mask = counts < 20
                        weight[mask] = np.log1p(1 / (counts[mask] + 1))
                        gains = z['matrix'][feasible].astype(float) @ weight
                        maximum = gains.max()
                        tied = feasible[np.abs(gains - maximum) <= 1e-12]
                        chosen = min(tied, key=lambda i: str(z['sample_ids'][i]))
                        assert chosen == int(index), (source, budget, category, method, 'greedy maximal feasible gain')
                        trace = trace_index[(source, budget, category, method, rank)]
                        assert trace['sample_id'] == str(z['sample_ids'][chosen])
                        close(trace['concave_gain'], maximum, (source, budget, category, method, rank, 'gain'))
                        before = np.minimum(counts, 20).sum() / 1420
                        available.remove(chosen)
                        used[index_keys[chosen]] += 1
                        counts += z['matrix'][chosen]
                        completion = np.minimum(counts, 20).sum() / 1420
                        close(trace['frequency_coverage_gain'], completion - before, (source, budget, category, method, rank, 'completion gain'))
                        close(trace['support_completion'], completion, (source, budget, category, method, rank, 'completion'))
                        close(trace['presence_coverage'], np.count_nonzero(counts) / 71, (source, budget, category, method, rank, 'presence'))
            verified.append(dict(source=source, budget_per_class=budget, fixed_methods=2, random_repeats=repeats))
            print(f'Selection audit passed: {source}, {budget}/class', flush=True)
    return verified


def independent_condition(arrays, selection):
    result = {}
    for c, category in enumerate(CLASSES):
        z = arrays[category]
        ids = selection[:, c]
        counts = z['matrix'][ids].sum(axis=1)
        other = z['online_matrix'][ids].sum(axis=1)
        parts = dict(presence_coverage=np.count_nonzero(counts, axis=1) / 71,
                     support_completion=np.minimum(counts, 20).sum(axis=1) / 1420,
                     other_label_presence_coverage=np.count_nonzero(other, axis=1) / 71,
                     other_label_support_completion=np.minimum(other, 20).sum(axis=1) / 1420,
                     risk_unknown_fraction=(z['risk_upper'][ids] > z['risk_lower'][ids] + 1e-8).mean(axis=1))
        for suffix in ('lower', 'midpoint', 'upper'):
            parts['risk_' + suffix] = z['risk_' + suffix][ids].mean(axis=1)
        tiers, _, _, _ = independent_structure(z, selection.shape[-1])
        for t in range(1, 6):
            parts[f'q{t}_fraction'] = (tiers[ids] == t).mean(axis=1)
        for key, value in parts.items():
            result[key] = result.get(key, np.zeros(len(selection))) + value / 6
    return result


def independent_model_metrics(predictions, selections, budget):
    global_ids = selections + np.arange(6)[None, :, None] * 1300
    chosen = predictions[:, global_ids.reshape(len(selections), -1)].reshape(24, len(selections), 6, budget)
    counts = np.zeros((24, len(selections), 6, 6), dtype=int)
    for label in range(6):
        counts[:, :, :, label] = np.count_nonzero(chosen == label, axis=-1)
    correct = np.diagonal(counts, axis1=-2, axis2=-1)
    denominators = counts.sum(-2) + counts.sum(-1)
    f1 = np.divide(2 * correct, denominators, out=np.zeros_like(correct, dtype=float), where=denominators > 0)
    wrong = counts.copy()
    wrong[:, :, np.arange(6), np.arange(6)] = 0
    metrics = dict(accuracy=correct.sum(-1) / (6 * budget), macro_f1=f1.mean(-1),
                   errors=6 * budget - correct.sum(-1), confusion_type_count=np.count_nonzero(wrong, axis=(-2, -1)),
                   confusion_breadth=np.count_nonzero(wrong, axis=(-2, -1)) / 30)
    for c, category in enumerate(CLASSES):
        metrics[f'{category}_recall'] = correct[:, :, c] / budget
    return metrics, wrong


def independent_rarefaction(counts, q):
    """Product of sequential non-hit probabilities, independent of log-gamma."""
    n = counts.sum(-1)
    absent = np.ones(counts.shape, dtype=float)
    for draw in range(int(q.max(initial=0))):
        active = (q > draw)[..., None]
        den = (n - draw)[..., None]
        num = np.maximum(0, den - counts)
        fraction = np.divide(num, den, out=np.ones_like(num, dtype=float), where=den > 0)
        absent *= np.where(active, fraction, 1)
    return np.sum(1 - absent, axis=-1) / 5


def audit_results(arrays):
    merged = dict(np.load(OUT / 'merged_predictions.npz', allow_pickle=False))
    assert merged['predictions'].shape == (24, 7800)
    master_ids = [str(s) for c in CLASSES for s in arrays['local'][c]['sample_ids']]
    assert merged['sample_ids'].tolist() == master_ids
    summary = read_csv(OUT / 'summary.csv')
    run_rows = read_csv(OUT / 'model_run_metrics.csv')
    paired_rows = read_csv(OUT / 'paired_model_run_metrics.csv')
    condition_rows = read_csv(OUT / 'condition_metrics.csv')
    expected_settings = len(BUDGETS) * 2
    expected_conditions = 2 * len(BUDGETS) * 1002
    assert len(summary) == expected_settings * 3 and len(run_rows) == expected_settings * 3 * 24
    assert len(paired_rows) == expected_settings * 2 * 24 and len(condition_rows) == expected_conditions
    conditions_by_key = {(r['source'], int(r['budget_per_class']), r['method'], int(r['repetition'])): r for r in condition_rows}
    runs_by_key = {(r['source'], int(r['budget_per_class']), r['method'], r['model'], int(r['seed'])): r for r in run_rows}
    paired_by_key = {(r['source'], int(r['budget_per_class']), r['contrast'], r['model'], int(r['seed'])): r for r in paired_rows}
    audit_count = 0
    for source in ('local', 'online'):
        for budget in BUDGETS:
            saved = dict(np.load(OUT / 'selection' / f'{source}_n{budget * 6}.npz'))
            raw = dict(np.load(OUT / 'raw' / f'{source}_n{budget * 6}.npz'))
            metrics = {}
            wrong = {}
            for method, selection in saved.items():
                if selection.ndim == 2:
                    selection = selection[None]
                condition = independent_condition(arrays[source], selection)
                metrics[method], wrong[method] = independent_model_metrics(merged['predictions'], selection, budget)
                assert np.array_equal(wrong[method], raw[f'wrong_counts_{method}'])
                row = next(r for r in summary if r['source'] == source and int(r['budget_per_class']) == budget and r['method'] == method)
                for key, values in condition.items():
                    assert np.allclose(values, raw[f'condition_{method}_{key}'], atol=1e-9, rtol=0), (source, budget, method, key)
                    close(row[key], values.mean(), (source, budget, method, key, 'summary'))
                    for repeat, value in enumerate(values):
                        close(conditions_by_key[(source, budget, method, repeat)][key], value, ('condition', source, budget, method, repeat, key))
                    if method == 'risk':
                        for quantile, suffix in ((.025, '_selection_p025'), (.975, '_selection_p975')):
                            close(row[key + suffix], np.quantile(values, quantile), (source, budget, method, key, suffix))
                for key, values in metrics[method].items():
                    assert np.allclose(values, raw[f'metrics_{method}_{key}'], atol=1e-9, rtol=0), (source, budget, method, key)
                    close(row[key], values.mean(), (source, budget, method, key, 'summary'))
                    for r in range(24):
                        model = str(merged['models'][r]);seed = int(merged['seeds'][r])
                        close(runs_by_key[(source, budget, method, model, seed)][key], values[r].mean(), ('model row', source, budget, method, model, seed, key))
                    if method == 'risk':
                        for quantile, suffix in ((.025, '_selection_p025'), (.975, '_selection_p975')):
                            close(row[key + suffix], np.quantile(values.mean(0), quantile), (source, budget, method, key, suffix))
                audit_count += values.size
            for left in ('coverage', 'risk_coverage'):
                fixed = wrong[left][:, 0]
                random = wrong['risk']
                q = np.minimum(fixed.sum(-1)[:, None], random.sum(-1))
                fixed_breadth = independent_rarefaction(np.broadcast_to(fixed[:, None], random.shape), q).mean(-1)
                random_breadth = independent_rarefaction(random, q).mean(-1)
                delta = fixed_breadth - random_breadth
                evidence = (q > 0).mean(-1)
                expected = dict(equal_error_confusion_delta=delta, evaluable_class_fraction=evidence,
                                left_equal_error_breadth=fixed_breadth, right_equal_error_breadth=random_breadth)
                for key, value in expected.items():
                    assert np.allclose(raw[f'paired_{left}_{key}'], value, atol=1e-9, rtol=0), (source, budget, left, key, 'rarefaction')
                for r in range(24):
                    model = str(merged['models'][r]);seed = int(merged['seeds'][r])
                    recorded = paired_by_key[(source, budget, left + '_minus_risk', model, seed)]
                    close(recorded['equal_error_confusion_delta'], delta[r].mean(), ('paired', source, budget, left, r))
                    close(recorded['evaluable_class_fraction'], evidence[r].mean(), ('paired evidence', source, budget, left, r))
                    for column, key in [('error_delta', 'errors'), ('raw_confusion_breadth_delta', 'confusion_breadth'), ('macro_f1_delta', 'macro_f1')]:
                        close(recorded[column], metrics[left][key][r, 0] - metrics['risk'][key][r].mean(), (source, budget, left, r, column))
            for key in ('risk_lower', 'risk_midpoint', 'risk_upper', 'risk_unknown_fraction', 'q1_fraction', 'q2_fraction', 'q3_fraction', 'q4_fraction', 'q5_fraction'):
                close(raw['condition_risk_' + key].mean(), raw['condition_risk_coverage_' + key].mean(), (source, budget, 'risk matching', key))
            assert np.array_equal(saved['coverage'], saved['risk_coverage'])
            for key in metrics['coverage']:
                assert np.array_equal(metrics['coverage'][key], metrics['risk_coverage'][key]), (source, budget, key, 'identical greedy names')
            old_selections = dict(np.load(OLD / 'selection' / f'{source}_n{budget * 6}.npz'))
            old_raw = dict(np.load(OLD / 'raw' / f'{source}_n{budget * 6}.npz'))
            for method in ('risk', 'risk_coverage'):
                assert np.array_equal(saved[method], old_selections[method]), (source, budget, method, 'old matched comparison index equality')
                for key in metrics[method]:
                    assert np.array_equal(metrics[method][key], old_raw[f'metrics_{method}_{key}']), (source, budget, method, key, 'old numerical equality')
            assert np.array_equal(saved['coverage'], old_selections['risk_coverage'])
            print(f'Result arithmetic audit passed: {source}, {budget}/class', flush=True)
    return dict(settings=expected_settings, summary_rows=len(summary), model_rows=len(run_rows), paired_rows=len(paired_rows),
                condition_rows=len(condition_rows), raw_observations_recomputed=audit_count,
                all_three_share_exact_risk_quotas=True, coverage_combination_ordered_targets_and_metrics_identical=True, old_matched_risk_and_combination_results_unchanged=True)


def audit_tests():
    pairs = read_csv(OUT / 'paired_model_run_metrics.csv')
    tests = read_csv(OUT / 'primary_tests.csv')
    assert len(tests) == 2 * len(BUDGETS)
    probabilities = []
    for row in tests:
        chosen = [r for r in pairs if r['source'] == row['source'] and r['budget_per_class'] == row['budget_per_class'] and r['contrast'] == 'risk_coverage_minus_risk']
        architectures = sorted({r['model'] for r in chosen})
        assert len(architectures) == 8
        values = np.array([np.mean([float(r['equal_error_confusion_delta']) for r in chosen if r['model'] == m]) for m in architectures])
        observed = abs(values.mean())
        null = [abs(np.dot(values, signs) / 8) for signs in itertools.product((-1, 1), repeat=8)]
        probability = sum(value >= observed - 1e-12 for value in null) / 256
        probabilities.append(probability)
        close(row['signflip_two_sided_p'], probability, ('signflip', row['source'], row['budget_per_class']))
        close(row['mean_equal_error_delta'], values.mean(), ('delta', row['source'], row['budget_per_class']))
        assert int(row['positive_architectures']) == np.count_nonzero(values > 0)
    previous = 0.
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda i: probabilities[i])):
        previous = max(previous, min(1, (len(tests) - rank) * probabilities[i]))
        column = 'holm_eighteen_p' if 'holm_eighteen_p' in tests[i] else 'holm_p'
        close(tests[i][column], previous, ('Holm eighteen', i))


def audit_final_report():
    return audit_equal_ratio_report()


def audit_source_correct_manifests(arrays):
    data = ROOT / 'data/QE_VisualState_CoverRisk_v1_20261013'
    master = {r['sample_id']: r for r in read_csv(data / 'manifest.csv')}
    scores = {(r['annotation_backend'], r['sample_id']): r for r in read_csv(ROOT / 'artifacts/QE_FeatureRisk_Diagnostics_20261013/target_scores.csv')}
    checked = 0
    for source in ('local', 'online'):
        for budget in BUDGETS:
            selections = dict(np.load(OUT / 'selection' / f'{source}_n{budget * 6}.npz'))
            for method in ('coverage', 'risk', 'risk_coverage'):
                rows = read_csv(OUT / 'test_sets' / f'{source}_{method}_n{budget * 6}.csv')
                selection = selections[method][0] if selections[method].ndim == 3 else selections[method]
                expected = [str(arrays[source][c]['sample_ids'][i]) for ci, c in enumerate(CLASSES) for i in selection[ci]]
                assert [r['sample_id'] for r in rows] == expected and len(set(expected)) == budget * 6
                for row in rows:
                    sid = row['sample_id']; official = master[sid]; actual = scores[source, sid]
                    assert row['annotation_backend'] == source and row['risk_tier'] == actual['risk_tier']
                    for key in ('sample_id', 'class_name', 'label', 'source_pool', 'bbox_norm_xyxy', 'bbox_area_ratio', 'image_path', 'crop_path', 'marked_image_path', 'feature_crop_path', 'annotation_path'):
                        assert row[key] == official[key], (source, budget, method, sid, key)
                    for suffix in ('lower', 'midpoint', 'upper'):
                        close(row['score_' + suffix], actual['global_risk_' + suffix], (source, budget, method, sid, 'actual risk alias'))
                    for key in ('score_lower', 'score_midpoint', 'score_upper', 'fully_known', 'evaluable', 'high_risk_q5'):
                        assert row['master_local_' + key] == official[key]
                    assert row['high_risk_q5'] == str(int(actual['risk_tier_integer']) == 5)
                    for key in ('fully_known', 'evaluable'):
                        assert row[key] == actual[key]
                checked += 1
    return checked


def audit_equal_ratio_report():
    from collections import Counter
    from docx import Document
    report_freeze = json.loads((OUT / 'equal_ratio_report_freeze.sha256.json').read_text())
    for path, expected in report_freeze.items():
        assert digest(path) == expected, ('final report freeze', path)
    summary = read_csv(OUT / 'summary.csv'); lookup = {(r['source'], int(r['budget_per_class']), r['method']): r for r in summary}
    all_rows = read_csv(OUT / 'all_budget_results.csv'); assert len(all_rows) == 54
    for row in all_rows:
        actual = lookup[row['source'], int(row['budget_per_class']), row['method']]
        for key in row:
            assert row[key] == actual[key], ('all budget report table', key)
    ratios = read_csv(OUT / 'equal_ratio_proportions.csv'); assert len(ratios) == 324
    for row in ratios:
        budget = int(row['budget_per_class'])
        assert int(row['targets_per_repetition']) == budget
        close(row['class_fraction'], 1/6, 'equal class fraction')
        for tier in range(1, 6):
            assert int(row[f'Q{tier}_targets_per_repetition']) == budget // 5
            close(row[f'Q{tier}_fraction'], .2, 'equal tier fractions')
    scores = {(r['annotation_backend'], r['sample_id']): r for r in read_csv(ROOT / 'artifacts/QE_FeatureRisk_Diagnostics_20261013/target_scores.csv')}
    manifest_index = read_csv(OUT / 'primary_manifest_index.csv'); assert len(manifest_index) == 6
    metrics = {(r['source'], r['method']): r for r in read_csv(OUT / 'primary_1200_rep0_metrics.csv')}
    dimensions = {(r['source'], r['method'], r['dimension']): r for r in read_csv(OUT / 'primary_1200_rep0_dimension_scores.csv')}
    merged = dict(np.load(OUT / 'merged_predictions.npz', allow_pickle=False)); predindex = {str(s): i for i, s in enumerate(merged['sample_ids'])}
    for item in manifest_index:
        source, method = item['source'], item['method']
        rows = read_csv(item['path']); original = read_csv(OUT / 'test_sets' / f'{source}_{method}_n1200.csv')
        assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in original]
        assert len(rows) == 1200 and Counter((r['class_name'], int(r['risk_tier_integer'])) for r in rows) == Counter({(c,t):40 for c in CLASSES for t in range(1,6)})
        for row in rows:
            actual = scores[source, row['sample_id']]
            for suffix in ('lower', 'midpoint', 'upper'):
                close(row['score_' + suffix], actual['global_risk_' + suffix], (source, method, 'primary source score'))
            assert row['annotation_backend'] == source and row['selection_repetition'] == '0'
            for key in ('image_path', 'crop_path', 'marked_image_path', 'feature_crop_path', 'annotation_path'):
                assert (Path(row['dataset_root']) / row[key]).is_file(), (source, method, key)
        gi = [predindex[r['sample_id']] for r in rows]
        labels = np.array([int(r['label']) for r in rows]); per_run = [confusion_metrics(p, labels) for p in merged['predictions'][:, gi]]
        for key, target in [('accuracy', 'accuracy'), ('macro_f1', 'macro_f1'), ('error_count', 'errors')]:
            close(metrics[source,method][target], np.mean([r[key] for r in per_run]), (source, method, 'rep0 metrics', key))
        for suffix in ('lower', 'midpoint', 'upper'):
            close(metrics[source,method]['risk_'+suffix], np.mean([float(r['score_'+suffix]) for r in rows]), (source,method,'rep0 risk',suffix))
        assert int(metrics[source,method]['Q5_targets']) == 240
        for d in ('scale','visibility','occlusion','truncation','orientation'):
            for suffix in ('lower','midpoint','upper'):
                close(dimensions[source,method,d][suffix+'_points'],np.mean([float(r[d+'_contribution_'+suffix+'_0_20']) for r in rows]),(source,method,d,suffix))
            close(dimensions[source,method,d]['unknown_fraction'],np.mean([float(r[d+'_known_weight'])<1 for r in rows]),(source,method,d,'unknown'))
    text = (OUT/'report.md').read_text(); names={'coverage':'配额约束覆盖','risk':'配额约束随机','risk_coverage':'风险＋覆盖（同覆盖组）'}
    for row in all_rows:
        cells=[row['source'],row['total_targets'],names[row['method']],f"{100*float(row['presence_coverage']):.3f}",f"{100*float(row['support_completion']):.3f}",f"{float(row['risk_midpoint']):.3f}",f"{100*float(row['macro_f1']):.3f}"]
        assert '| '+' | '.join(cells)+' |' in text
    primary = read_csv(OUT/'primary_1200_summary.csv'); assert len(primary)==6
    for row in primary:
        cells=[row['source'],names[row['method']],f"{100*float(row['presence_coverage']):.3f}",f"{100*float(row['support_completion']):.3f}",f"{float(row['risk_lower']):.3f}/{float(row['risk_midpoint']):.3f}/{float(row['risk_upper']):.3f}",f"{100*float(row['q5_fraction']):.1f}",f"{100*float(row['risk_unknown_fraction']):.3f}",f"{100*float(row['macro_f1']):.3f}",f"{float(row['errors']):.3f}"]
        assert '| '+' | '.join(cells)+' |' in text
    tests = read_csv(OUT/'primary_tests.csv')
    for row in tests:
        s,b=row['source'],int(row['budget_per_class']);left,right=lookup[s,b,'coverage'],lookup[s,b,'risk']
        cells=[s,str(b*6),f"{100*(float(left['support_completion'])-float(right['support_completion'])):+.3f}",f"{100*float(row['mean_equal_error_delta']):+.3f}",f"{float(row['holm_eighteen_p']):.6f}"]
        assert '| '+' | '.join(cells)+' |' in text
    assert '实际只有两种独立策略' in text and '没有7800全池选样终点' in text and '0.0078125' in text and '0.05/18' in text
    doc=Document(OUT/'report.docx');assert len(doc.tables)==3
    return dict(status='pass',frozen_report_paths=len(report_freeze),summary54_verified=True,ratio_rows324_verified=True,
                source_correct_primary_manifests6=True,rep0_metrics_recomputed_from24_predictions=True,
                all_three_exactly_same_class_tier_ratios=True,two_independent_algorithms_explained=True,word_tables3=True)


def main():
    selfcheck()
    freeze_counts = {}
    for name in ('selection/input_freeze.sha256.json', 'prediction_input_freeze.sha256.json', 'analysis_result_freeze.sha256.json'):
        mapping = json.loads((OUT / name).read_text())
        for path, expected in mapping.items():
            assert digest(path) == expected, (name, path, 'freeze changed')
        freeze_counts[name] = len(mapping)
    arrays, records = verify_source_arrays()
    selections = audit_selections(arrays)
    result = audit_results(arrays)
    audit_tests()
    result['source_correct_test_manifests_verified'] = audit_source_correct_manifests(arrays)
    result.update(status='pass', state_matrices_rebuilt=12, source_risks_recomputed=15600,
                  selection_settings=selections, frozen_paths_verified=freeze_counts,
                  rarefaction='independent sequential-probability product, full 24 by 1000 by 6 counts',
                  tests='8 architecture units, 256 sign flips, eighteen source-budget tests Holm independently recomputed',
                  semantic_label_accuracy_verified=False, classifier_retraining=False,
                  classifier_reinference=False, cached_prediction_binding_audit='see separate prediction audit')
    (OUT / 'independent_result_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'independent_result_audit_freeze.sha256.json').write_text(json.dumps({str(Path(__file__).resolve()): digest(__file__), str(OUT / 'analysis_result_freeze.sha256.json'): digest(OUT / 'analysis_result_freeze.sha256.json')}, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def refresh_final_report():
    """Refresh presentation audit only after proving core analysis unchanged."""
    previous = json.loads((OUT / 'independent_result_audit.json').read_text())
    recorded = json.loads((OUT / 'independent_result_audit_freeze.sha256.json').read_text())
    analysis_path = str(OUT / 'analysis_result_freeze.sha256.json')
    assert previous['status'] == 'pass' and digest(analysis_path) == recorded[analysis_path]
    for name in previous['frozen_paths_verified']:
        mapping = json.loads((OUT / name).read_text())
        for path, expected in mapping.items():
            assert digest(path) == expected, (name, path, 'freeze changed')
        previous['frozen_paths_verified'][name] = len(mapping)
    previous['final_report'] = audit_final_report()
    previous['presentation_refresh'] = 'Core frozen analysis unchanged; final midpoint-score report, efficiency tables, significance-resolution wording and Word independently reverified.'
    (OUT / 'independent_result_audit.json').write_text(json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
    frozen_paths = [Path(__file__).resolve(), OUT / 'analysis_result_freeze.sha256.json', OUT / 'equal_ratio_report_freeze.sha256.json']
    (OUT / 'independent_result_audit_freeze.sha256.json').write_text(json.dumps({str(p): digest(p) for p in frozen_paths}, indent=2) + '\n')
    print(json.dumps(dict(status='pass', core_analysis_unchanged=True, final_report=previous['final_report']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        refresh_final_report()
    else:
        main()
