#!/usr/bin/env python3
"""Independent verification of annotation-source variants and ratio certificates."""
import json
from collections import Counter
from fractions import Fraction

import numpy as np

from audit_qe_coverrisk_dataset import AUDIT, CLASSES, DATA, OUT, close, digest, metrics, read_csv, risk


def main():
    freeze_counts = {}
    for file in ('input_freeze.sha256.json', 'final_result_freeze.sha256.json', 'independent_export_audit/input_freeze.sha256.json'):
        mapping = json.loads((OUT / file).read_text())
        for path, expected in mapping.items():
            assert digest(path) == expected, (file, 'freeze changed', path)
        freeze_counts[file] = len(mapping)
    master = {r['sample_id']:r for r in read_csv(DATA / 'manifest.csv')}
    records = {r['sample_id']:r for r in map(json.loads, (OUT / 'prepared/records.jsonl').read_text().splitlines())}
    config = json.loads((OUT / 'risk_protocol.json').read_text())
    arrays = {source:{c:dict(np.load(root / 'prepared' / f'{c}.npz', allow_pickle=False)) for c in CLASSES}
              for source, root in (('local', OUT), ('online', OUT / 'online_analysis'))}
    for source, classes in arrays.items():
        for c, z in classes.items():
            local = arrays['local'][c]
            assert np.array_equal(z['sample_ids'], local['sample_ids'])
            if source == 'online':
                assert np.array_equal(z['matrix'], local['online_matrix'])
                assert np.array_equal(z['online_matrix'], local['matrix'])
            for i, sid in enumerate(z['sample_ids']):
                item = records[str(sid)];computed = risk(float(item['bbox_area_ratio']), item['annotations'][source]['native']['features'], config)
                for key, value in zip(('risk_lower', 'risk_midpoint', 'risk_upper'), computed):close(z[key][i], value, (source, sid, key))
            order = sorted(range(1300), key=lambda i:(z['risk_midpoint'][i], str(z['sample_ids'][i])))
            assert set(np.flatnonzero(z['high_risk'])) == set(order[-260:])
    comparisons = read_csv(OUT / 'variant_comparison.csv')
    category_rows = read_csv(OUT / 'variant_metrics_by_category.csv')
    paths = list((DATA / 'variants').glob('*.csv'))
    assert len(paths) == len(comparisons) == 9 and len(category_rows) == 54
    verified = []
    switched = ('score_lower', 'score_midpoint', 'score_upper', 'high_risk_q5', 'fully_known', 'evaluable')
    for path in paths:
        rows = read_csv(path); name = path.stem
        source = rows[0]['annotation_source'];assert source in ('local', 'online')
        assert len({r['sample_id'] for r in rows}) == len(rows)
        counts = Counter(r['class_name'] for r in rows);assert len(set(counts.values())) == 1 and set(counts) == set(CLASSES)
        actual = []
        for r in rows:
            sid = r['sample_id']; m = master[sid];assert r['annotation_source'] == source
            for key in m:
                if key not in switched:assert r[key] == m[key], (name, sid, key)
            for key in switched:assert r['master_local_' + key] == m[key], (name, sid, 'master_local_' + key)
            z = arrays[source][r['class_name']];i = int(np.flatnonzero(z['sample_ids'] == sid)[0])
            for key, column in (('risk_lower', 'score_lower'), ('risk_midpoint', 'score_midpoint'), ('risk_upper', 'score_upper')):close(r[column], z[key][i], (name, sid, column))
            assert int(r['high_risk_q5']) == z['high_risk'][i]
            stored = records[sid]['annotations'][source]['risk']
            assert r['fully_known'] == str(stored['fully_known']) and r['evaluable'] == str(stored['evaluable']), (name, sid, 'source knowledge flags')
        for c in CLASSES:
            z = arrays[source][c];selected = {r['sample_id'] for r in rows if r['class_name'] == c}
            indices = [i for i, sid in enumerate(z['sample_ids']) if sid in selected]
            computed = metrics(z, indices);actual.append(computed)
            recorded = next(r for r in category_rows if r['strategy'] == name and r['category'] == c)
            assert recorded['annotation_source'] == source
            for key, value in computed.items():close(recorded[key], value, (name, c, key))
        recorded = next(r for r in comparisons if r['strategy'] == name)
        assert recorded['annotation_source'] == source
        close(recorded['total_targets'], len(rows), (name, 'total'))
        for key in actual[0]:
            if key not in ('n', 'reached_states', 'required_states'):close(recorded[key], np.mean([r[key] for r in actual]), (name, key))
        verified.append(dict(strategy=name, source=source, targets=len(rows), coverage=float(recorded['presence_coverage']), risk_lower=float(recorded['risk_lower']), q5_fraction=float(recorded['high_risk_q5_fraction'])))
    certificates = []
    for source, root in (('local', OUT), ('online', OUT / 'online_analysis')):
        classes = arrays[source]
        fixed = json.loads((root / 'optimization/solutions.json').read_text())
        for phase, items, k in [('minimum_cover', fixed['minimum_cover'], None)] + [('risk_maximum', v, int(k)) for k, v in fixed['budgets'].items()] + [('q5_maximum', fixed['highest_q5_under_full_coverage'], fixed['compact_budget_per_class'])]:
            for c, item in items.items():
                m = metrics(classes[c], item['selected_indices'])
                assert m['reachable_presence_coverage'] == 1
                if k is not None:assert m['n'] == k
                close(m['risk_lower'], item['risk_lower_mean'], (source, phase, c))
                close(m['high_risk_q5_fraction'], item['high_risk_rate'], (source, phase, c, 'Q5'))
                if item['solver_proved_optimal']:
                    assert item['solver_status'] == 0 and item['solver_incumbent_verified'] and not item['used_fallback']
                    close(item['mip_gap'], 0, (source, phase, c, 'gap'))
        results = json.loads((root / 'global_ratio/solutions.json').read_text())
        protocol = results['protocol'];reach = sum(int((z['matrix'].sum(axis=0) > 0).sum()) for z in classes.values())
        assert protocol['reachable_state_constraints'] == reach and protocol['constraint_matrix_shape'] == [reach + 6, 7801]
        assert protocol['minimum_budget_per_class'] == fixed['compact_budget_per_class'] == 12 and protocol['maximum_budget_per_class'] == 1300
        for objective in ('risk_mean', 'high_risk_fraction'):
            result = results[objective];budget = result['budget_per_class'];assert 12 <= budget <= 1300
            assert result['total_targets'] == budget * 6
            total_risk = 0.;total_q5 = 0;scale = 12 if objective == 'risk_mean' else 1
            for c in CLASSES:
                z = classes[c];indices = result['selected_indices'][c];m = metrics(z, indices)
                assert m['n'] == budget and m['reachable_presence_coverage'] == 1
                assert result['classes'][c]['sample_ids'] == z['sample_ids'][indices].tolist()
                total_risk += float(z['risk_lower'][indices].sum());total_q5 += int(z['high_risk'][indices].sum())
            close(result['risk_lower_mean'], total_risk / (budget * 6), (source, objective, 'risk'))
            close(result['high_risk_fraction'], total_q5 / (budget * 6), (source, objective, 'Q5'))
            exact_numerator = round(total_risk * 12) if objective == 'risk_mean' else total_q5
            ratio = Fraction(0, 1)
            for iteration in result['iterations']:
                assert iteration['solver_status'] == 0 and iteration['solver_proved_optimal'] and iteration['verified_feasible_incumbent']
                close(iteration['mip_gap'], 0, (source, objective, 'gap'))
                assert iteration['input_ratio_numerator'] == ratio.numerator and iteration['input_ratio_denominator'] == ratio.denominator
                num = iteration['selected_objective_numerator'];den = iteration['selected_objective_denominator']
                assert den == iteration['budget_per_class'] * 6 and 12 <= iteration['budget_per_class'] <= 1300
                improvement = num * ratio.denominator - ratio.numerator * den
                assert iteration['exact_improvement_numerator'] == improvement
                assert iteration['exact_improvement_denominator'] == ratio.denominator
                close(iteration['improvement'], improvement / ratio.denominator, (source, objective, 'improvement'))
                assert np.isclose(iteration['solver_minimization_fun'], -improvement / ratio.denominator, atol=1e-5)
                assert np.isclose(iteration['solver_minimization_dual_bound'], iteration['solver_minimization_fun'], atol=1e-5)
                ratio = Fraction(num, den)
            last = result['iterations'][-1]
            assert last['exact_improvement_numerator'] == 0 and result['global_optimum_proved']
            assert last['selected_objective_numerator'] == exact_numerator and last['selected_objective_denominator'] == budget * 6
            close(result['objective_value'], float(ratio) / scale, (source, objective, 'objective value'))
            manifest = read_csv(DATA / 'variants' / f'{source}_global_{objective}.csv')
            expected = {str(classes[c]['sample_ids'][i]) for c in CLASSES for i in result['selected_indices'][c]}
            assert {r['sample_id'] for r in manifest} == expected
            certificates.append(dict(source=source, objective=objective, proved=True, count=budget * 6, final_exact_improvement_numerator=0, linear_iterations=len(result['iterations'])))
    merged = read_csv(OUT / 'merged_strategy_comparison.csv');core = read_csv(OUT / 'strategy_comparison.csv')
    assert len(merged) == len(core) + len(comparisons) == 17
    for r in core + comparisons:
        matching = next(m for m in merged if m['strategy'] == r['strategy'])
        for key in r:assert matching[key] == r[key], (r['strategy'], key, 'merged table')
    report = (OUT / 'report.md').read_text()
    assert '不把两种自动标注的状态并集' in report
    assert '不是实测分类错误概率' in report and '不对所有可能图像' in report
    result = dict(status='pass', variants=verified, global_ratio_certificates=certificates,
                  source_specific_risks_recomputed=15600, freeze_counts=freeze_counts,
                  merged_comparison_rows=17, source_union_coverage_not_used=True,
                  optimum_scope='equal class counts, K in 12..1300, frozen per-source attainable single-field states, support >= 1',
                  proof_verification='independent primal feasibility and exact ratio arithmetic plus recorded solver optimality/dual-bound certificates; MILP not rerun')
    (AUDIT / 'variant_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (AUDIT / 'variant_input_freeze.sha256.json').write_text(json.dumps({str(p):digest(p) for p in [__file__, OUT / 'final_result_freeze.sha256.json', OUT / 'variant_comparison.csv', OUT / 'merged_strategy_comparison.csv']}, indent=2) + '\n')
    primary = json.loads((AUDIT / 'audit.json').read_text());assert primary['status'] == 'pass'
    final = dict(status='pass', dataset=str(DATA), targets=7800, source_annotations_preserved=15600,
                 fine_supplements_preserved=60, primary_subsets=len(primary['subsets']), variants=9,
                 global_ratio_objectives_verified=4, local_fixed_dictionary_presence_upper_bound=407/426,
                 online_fixed_dictionary_presence_upper_bound=424/426,
                 primary_audit=str(AUDIT / 'audit.json'), variant_audit=str(AUDIT / 'variant_audit.json'),
                 semantic_labels_fully_human_verified=False, pooled_near_duplicate_audit=False,
                 classifier_reinference=False)
    (OUT / 'final_audit.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
