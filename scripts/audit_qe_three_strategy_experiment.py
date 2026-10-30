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
OUT = ROOT / 'artifacts/QE_ThreeStrategy_Test_20261013'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')


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
        for budget in (20, 40, 100):
            selections = dict(np.load(OUT / 'selection' / f'{source}_n{budget * 6}.npz'))
            assert set(selections) == {'coverage', 'risk', 'risk_coverage'}
            for c, category in enumerate(CLASSES):
                z = arrays[source][category]
                tiers, groups, quotas, index_keys = independent_structure(z, budget)
                recorded = [r for r in quota_rows if r['source'] == source and int(r['budget_per_class']) == budget and r['category'] == category]
                expected = {(int(r['tier']), float(r['risk_midpoint']), int(r['unknown_count'])): int(r['quota']) for r in recorded}
                assert expected == quotas
                rng = np.random.default_rng(20261013 + (0 if source == 'local' else 100000) + budget * 100 + c)
                for repeat in range(1000):
                    chosen = []
                    for k in sorted(groups):
                        if quotas[k]:
                            chosen.extend(rng.choice(groups[k], quotas[k], replace=False).tolist())
                    assert np.array_equal(selections['risk'][repeat, c], chosen), (source, budget, category, repeat, 'random replay')
                for method in ('coverage', 'risk_coverage'):
                    ids = selections[method][c]
                    assert len(ids) == len(set(ids.tolist())) == budget
                    assert int(ids.min()) >= 0 and int(ids.max()) < 1300
                    if method == 'risk_coverage':
                        observed = Counter(index_keys[int(i)] for i in ids)
                        assert observed == Counter({k: v for k, v in quotas.items() if v})
                    counts = np.zeros(71, dtype=int)
                    used = Counter()
                    available = set(range(1300))
                    for rank, index in enumerate(ids, 1):
                        feasible = np.asarray([i for i in sorted(available) if method == 'coverage' or used[index_keys[i]] < quotas[index_keys[i]]], dtype=int)
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
            verified.append(dict(source=source, budget_per_class=budget, fixed_methods=2, random_repeats=1000))
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
    assert len(summary) == 18 and len(run_rows) == 432 and len(paired_rows) == 288 and len(condition_rows) == 6012
    conditions_by_key = {(r['source'], int(r['budget_per_class']), r['method'], int(r['repetition'])): r for r in condition_rows}
    runs_by_key = {(r['source'], int(r['budget_per_class']), r['method'], r['model'], int(r['seed'])): r for r in run_rows}
    paired_by_key = {(r['source'], int(r['budget_per_class']), r['contrast'], r['model'], int(r['seed'])): r for r in paired_rows}
    audit_count = 0
    for source in ('local', 'online'):
        for budget in (20, 40, 100):
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
    return dict(settings=6, summary_rows=18, model_rows=432, paired_rows=288, condition_rows=6012, raw_observations_recomputed=audit_count)


def audit_tests():
    pairs = read_csv(OUT / 'paired_model_run_metrics.csv')
    tests = read_csv(OUT / 'primary_tests.csv')
    assert len(tests) == 6
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
    for rank, i in enumerate(sorted(range(6), key=lambda i: probabilities[i])):
        previous = max(previous, min(1, (6 - rank) * probabilities[i]))
        close(tests[i]['holm_six_p'], previous, ('Holm six', i))


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
    result.update(status='pass', state_matrices_rebuilt=12, source_risks_recomputed=15600,
                  selection_settings=selections, frozen_paths_verified=freeze_counts,
                  rarefaction='independent sequential-probability product, full 24 by 1000 by 6 counts',
                  tests='8 architecture units, 256 sign flips, six-test Holm independently recomputed',
                  semantic_label_accuracy_verified=False, classifier_retraining=False,
                  classifier_reinference=False, cached_prediction_binding_audit='see separate prediction audit')
    (OUT / 'independent_result_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'independent_result_audit_freeze.sha256.json').write_text(json.dumps({str(Path(__file__).resolve()): digest(__file__), str(OUT / 'analysis_result_freeze.sha256.json'): digest(OUT / 'analysis_result_freeze.sha256.json')}, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
