#!/usr/bin/env python3
"""Independently verify materialized QE visual-state datasets and solver results."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = 'QE_VisualState_CoverRisk_v1_20261013'
OUT = ROOT / 'artifacts' / NAME
DATA = ROOT / 'data' / NAME
AUDIT = OUT / 'independent_export_audit'
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
    assert np.isclose(float(actual), float(expected), atol=1e-8, rtol=0), (label, actual, expected)


def risk(area, features, config):
    maps = config['maps']
    levels = [sum(area < e for e in config['scale_edges']) / 3]
    keys = ('body_visibility', 'occlusion_level', 'truncation_level', 'body_orientation')
    lower = levels + [maps[k].get(features[k], 0) for k in keys]
    upper = levels + [maps[k].get(features[k], 1) for k in keys]
    weights = [config['dimension_weights'][k] for k in ('scale', 'visibility', 'occlusion', 'truncation', 'orientation')]
    lo, hi = float(np.dot(lower, weights) * 100), float(np.dot(upper, weights) * 100)
    return lo, (lo + hi) / 2, hi


def metrics(z, indices):
    indices = np.array(indices, dtype=int)
    assert len(indices) == len(set(indices.tolist())) and len(indices)
    assert indices.min() >= 0 and indices.max() < len(z['sample_ids'])
    support = z['matrix'][indices].sum(axis=0)
    reachable = z['matrix'].sum(axis=0) > 0
    online = z['online_matrix'][indices].sum(axis=0)
    return dict(n=len(indices), reached_states=int((support > 0).sum()), required_states=71,
                presence_coverage=float((support > 0).mean()),
                reachable_presence_coverage=float((support[reachable] > 0).mean()),
                support_completion=float(np.minimum(support / 20, 1).mean()),
                risk_lower=float(z['risk_lower'][indices].mean()),
                risk_midpoint=float(z['risk_midpoint'][indices].mean()),
                risk_upper=float(z['risk_upper'][indices].mean()),
                high_risk_q5_fraction=float(z['high_risk'][indices].mean()),
                online_presence_coverage=float((online > 0).mean()),
                online_support_completion=float(np.minimum(online / 20, 1).mean()))


def main():
    assert (DATA / 'manifest.csv').is_file(), 'Dataset export has not completed'
    AUDIT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((OUT / 'input_freeze.sha256.json').read_text())
    for number, (path, expected) in enumerate(freeze.items(), 1):
        assert digest(path) == expected, ('frozen input changed', path)
        if number % 10000 == 0:
            print(f'Frozen inputs verified: {number}/{len(freeze)}', flush=True)
    config = json.loads((OUT / 'risk_protocol.json').read_text())
    states = [tuple(x) for x in json.loads((OUT / 'coverage_dictionary.json').read_text())['states']]
    assert len(states) == len(set(states)) == 71
    records = [json.loads(line) for line in (OUT / 'prepared/records.jsonl').read_text().splitlines()]
    source_records = {r['sample_id']: r for r in records}
    rows = read_csv(DATA / 'manifest.csv')
    assert len(rows) == len(source_records) == 7800
    lookup = {r['sample_id']: r for r in rows}
    assert len(lookup) == 7800 and set(lookup) == set(source_records)
    assert Counter(r['class_name'] for r in rows) == Counter(dict.fromkeys(CLASSES, 1300))
    reference = read_csv(OUT / 'audit/reference_targets.csv')
    assert len(reference) == 2190
    reference_hashes = {r['crop_sha256'] for r in reference}
    images, crops, fine_ids = set(), set(), set()
    audit_freeze = {str(Path(__file__).resolve()): digest(__file__)}
    for n, row in enumerate(rows, 1):
        sid = row['sample_id']; original = source_records[sid]
        assert row['class_name'] == original['class_name'] and int(row['label']) == CLASSES.index(row['class_name'])
        assert row['source_pool'] == original['source_pool']
        for key in ('image_path', 'crop_path', 'marked_image_path', 'feature_crop_path'):
            path = Path(row[key])
            assert not path.is_absolute() and '..' not in path.parts
            actual = digest(DATA / path)
            assert actual == digest(original[key]), (sid, key)
            if key == 'image_path':
                assert actual not in images; images.add(actual)
            if key == 'crop_path':
                assert actual not in crops and actual not in reference_hashes; crops.add(actual)
        ann = json.loads((DATA / row['annotation_path']).read_text())
        assert ann['sample_id'] == sid and ann['official_label'] == int(row['label'])
        assert ann['bbox_norm_xyxy'] == json.loads(original['bbox_norm_xyxy'])
        assert ann['source_pool'] == row['source_pool']
        for backend in ('local', 'online'):
            source = Path(json.loads((OUT / 'protocol.json').read_text())['source_pools'][row['source_pool']])
            native_path = source / 'annotations' / backend / ('car' if row['class_name'] == 'car' else 'old') / 'parsed' / f'{sid}.json'
            native = json.loads(native_path.read_text())
            assert ann[backend]['native'] == native == original['annotations'][backend]['native'], (sid, backend, 'native preservation')
            assert set(native['features']) == set(native['evidence'])
            computed = risk(float(row['bbox_area_ratio']), native['features'], config)
            for key, value in zip(('score_lower', 'score_midpoint', 'score_upper'), computed):
                close(ann[backend]['risk'][key], value, (sid, backend, key))
                if backend == 'local':close(row[key], value, (sid, key))
        fresh_path = ROOT / 'artifacts/fresh_vlm_feature_scoring_20261012/annotations' / f'{sid}.json'
        if fresh_path.exists():
            assert ann['fine_feature_supplement'] == json.loads(fresh_path.read_text())
            fine_ids.add(sid);audit_freeze[str(fresh_path)] = digest(fresh_path)
        else:assert 'fine_feature_supplement' not in ann
        if n % 1300 == 0:print(f'Materialized objects verified: {n}/7800', flush=True)
    assert len(fine_ids) == 60
    arrays = {c:dict(np.load(OUT / 'prepared' / f'{c}.npz', allow_pickle=False)) for c in CLASSES}
    for c, z in arrays.items():
        assert z['matrix'].shape == z['online_matrix'].shape == (1300, 71)
        assert set(z['sample_ids']) == {r['sample_id'] for r in rows if r['class_name'] == c}
        for i, sid in enumerate(z['sample_ids']):
            row = lookup[str(sid)]
            for backend, key in (('local', 'matrix'), ('online', 'online_matrix')):
                features = source_records[str(sid)]['annotations'][backend]['native']['features']
                area = float(row['bbox_area_ratio']);scale = ('tiny' if area < .01 else 'small' if area < .03 else 'medium' if area < .1 else 'large')
                expected = []
                for field, state in states:
                    value = scale if field == 'target_scale' else features[field]
                    expected.append(int(state in value if isinstance(value, list) else state == value))
                assert np.array_equal(z[key][i], expected), (sid, backend, 'matrix')
            for key, column in (('risk_lower', 'score_lower'), ('risk_midpoint', 'score_midpoint'), ('risk_upper', 'score_upper')):
                close(z[key][i], row[column], (sid, key))
            assert int(z['high_risk'][i]) == int(row['high_risk_q5'])
        order = sorted(range(1300), key=lambda i:(z['risk_midpoint'][i], str(z['sample_ids'][i])))
        assert set(np.flatnonzero(z['high_risk'])) == set(order[-260:])
    category_metrics = read_csv(OUT / 'selected_metrics_by_category.csv')
    aggregate = read_csv(OUT / 'strategy_comparison.csv')
    verified = []
    for path in sorted((DATA / 'subsets').glob('*.csv')):
        subset = read_csv(path); assert len({r['sample_id'] for r in subset}) == len(subset)
        actual = []
        for row in subset:assert row == lookup[row['sample_id']], (path, row['sample_id'])
        for c in CLASSES:
            z = arrays[c];ids = {r['sample_id'] for r in subset if r['class_name'] == c}
            indices = [i for i, sid in enumerate(z['sample_ids']) if sid in ids]
            m = metrics(z, indices);actual.append(m)
            recorded = next(r for r in category_metrics if r['strategy'] == path.stem and r['category'] == c)
            for key, value in m.items():close(recorded[key], value, (path.stem, c, key))
        recorded = next(r for r in aggregate if r['strategy'] == path.stem)
        close(recorded['total_targets'], len(subset), (path.stem, 'total_targets'))
        for key in actual[0]:
            if key not in ('n', 'reached_states', 'required_states'):close(recorded[key], np.mean([m[key] for m in actual]), (path.stem, key))
        verified.append(dict(strategy=path.stem, targets=len(subset), presence_coverage=float(recorded['presence_coverage']), risk_lower=float(recorded['risk_lower'])))
    solutions = json.loads((OUT / 'optimization/solutions.json').read_text())
    certificates = []
    items = [('minimum_cover', c, v, None) for c, v in solutions['minimum_cover'].items()]
    items += [('risk_maximum', c, v, int(k)) for k, classes in solutions['budgets'].items() for c, v in classes.items()]
    items += [('q5_maximum', c, v, int(solutions['compact_budget_per_class'])) for c, v in solutions['highest_q5_under_full_coverage'].items()]
    for mode, c, item, budget in items:
        z = arrays[c];m = metrics(z, item['selected_indices'])
        if budget is not None:assert m['n'] == budget
        assert m['reachable_presence_coverage'] == 1
        assert item['sample_ids'] == z['sample_ids'][item['selected_indices']].tolist()
        close(item['risk_lower_mean'], m['risk_lower'], (mode, c, 'risk_mean'))
        close(item['high_risk_rate'], m['high_risk_q5_fraction'], (mode, c, 'Q5'))
        proved = item['solver_proved_optimal']
        if proved:
            assert item['solver_status'] == 0 and item['solver_incumbent_verified'] and not item['used_fallback']
            close(item['mip_gap'], 0, (mode, c, 'optimal gap'))
        certificates.append(dict(mode=mode, category=c, count=m['n'], solver_proved_optimal=proved, mip_gap=item['mip_gap'], objective_bound=item['objective_bound']))
    # Reconstruct all 1000 random realizations with the registered seed.
    random_rows = read_csv(OUT / 'random_comparison.csv')
    assert len(random_rows) == 1000
    rng = np.random.default_rng(20261013);budget = int(solutions['compact_budget_per_class'])
    for repeat, recorded in enumerate(random_rows):
        values = [metrics(arrays[c], rng.choice(1300, size=budget, replace=False)) for c in CLASSES]
        assert int(recorded['repeat']) == repeat
        for key in values[0]:
            if key not in ('n', 'reached_states', 'required_states'):close(recorded[key], np.mean([r[key] for r in values]), ('random', repeat, key))
    report = (OUT / 'report.md').read_text()
    assert '没有重新训练、执行分类器' in report and '不是实测分类错误概率' in report
    assert '不可达状态' in report and '尚未全量标注' in report
    result = dict(status='pass', targets=7800, local_native_preserved=7800, online_native_preserved=7800,
                  fine_supplements_exactly_preserved=60, source_freeze_paths_verified=len(freeze),
                  reference_crops=2190, exact_train_eval_crop_overlap=0, actual_original_duplicate_groups=0,
                  actual_crop_duplicate_groups=0, label_matrices_independently_reconstructed=12,
                  risk_scores_independently_recomputed=15600, random_trials_recomputed=1000,
                  subsets=verified, solver_certificates=certificates,
                  coverage_scope='426 frozen single-field states; unreachable states remain in denominator',
                  risk_scope='five-dimensional equal_global state proxy, not classifier error probability',
                  classifier_reinference=False, semantic_labels_fully_human_verified=False,
                  newly_pooled_cross_source_near_duplicate_audit=False)
    (AUDIT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for path in [DATA / 'manifest.csv', OUT / 'strategy_comparison.csv', OUT / 'optimization/solutions.json', OUT / 'report.md']:
        audit_freeze[str(path)] = digest(path)
    (AUDIT / 'input_freeze.sha256.json').write_text(json.dumps(audit_freeze, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('subsets', 'solver_certificates')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
