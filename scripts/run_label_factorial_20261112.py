"""Cached-only risk x coverage source experiment. Historical artifacts are read-only."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import verify_cross_vlm_analysis_20261112 as prior

study, old, cc = prior.study, prior.old, prior.cc
ROOT, PREV = study.ROOT, study.OUT
OUT = ROOT / 'artifacts/label_factorial_20261112'
CLASSES, METHODS, SEEDS = prior.CLASSES, prior.METHODS, prior.SEEDS
MODES = ['original', 'recalibrated']


def dump(name, value):
    cc.dump(OUT / name, value)


def write(name, rows):
    cc.write(OUT / name, rows)


def event(stage, **values):
    cc.log(OUT, stage, **values)


def key(risk, coverage, mode, c, method, seed):
    return f'{risk}{coverage}__{mode}__{c}__{method}__{seed}'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'completion.json').exists(), 'Completed experiment is immutable'
    cc.verify_freeze(PREV / 'input_freeze.json')
    cc.verify_freeze(PREV / 'label_matrix_freeze.json')
    cc.verify_freeze(PREV / 'selection_freeze.json')
    protocol = dict(
        scope='Cached historical sensitivity analysis, not semantic accuracy or fresh confirmation',
        hypothesis='Coverage-label changes alone can change P20-vs-Greedy20 diagnostic ranking with risk cells and initial samples fixed',
        design='2 risk sources x 2 coverage sources x 2 threshold regimes; every selection evaluated under 2 discovery and 2 historical reference label sources',
        sources=study.SOURCES, classes=CLASSES, methods=METHODS, seeds=SEEDS,
        budgets=dict(candidate_per_class=400, selected_per_class=200, support_cap=20, removal_limit=20),
        fixed_within_risk='Exact initial R selection, risk score cells, integer quotas, IDs, state dictionary, algorithm and solver options',
        risk_switch='Changes risk cells, quotas and initial R sets as one bundle; cannot isolate score vs quotas vs starts',
        thresholds=dict(original='Original A numeric thresholds unchanged in all arms',
                        recalibrated='0.95 x G coverage under each risk/coverage combination, prediction-blind'),
        reuse='Frozen diagonal AA/BB selections reused where thresholds match; no stochastic solver reruns to favor results',
        diagnostic='State and known same-field complement each >=20 targets; 3-prediction-seed mean recall gap <=-10pp; cross-reference intersection',
        primary='P20 minus Greedy20 confirmed conditions per start, summed across 6 classes and 5 classifiers, separately for every evaluator/reference pair',
        secondary='P20 minus R, capped-support completion, original/recalibrated gate pass counts, actual classifier error, paired-start sign counts',
        followup='Always decompose P20-vs-Greedy differences by field, including discovery-only and reference-only label switches; no further label/prompt tuning',
        success='A ranking reversal within fixed risk, threshold regime and evaluator/reference establishes coverage-driven selection sensitivity; evaluator changes of fixed sets establish measurement sensitivity',
        stop='Complete all 50 fixed starts, audit all selections and diagonal reproduction; then support/weaken claim, no search for favorable seeds or prompts',
        historical_predictions_previously_seen=True, new_vlm_calls=0, new_training=False,
    )
    cc.freeze(OUT / 'protocol.json', protocol)
    paths = [Path(__file__), Path(prior.__file__), Path(study.__file__), Path(old.__file__),
             Path(cc.__file__), Path(cc.old.__file__), Path(study.robustness.__file__),
             ROOT / 'output/experiments/实证论文补全_20261108/execute.py',
             ROOT / 'configs/category_five_dimension_pilot_v1.json',
             PREV / 'selections.npz', PREV / 'selection_runs.csv', PREV / 'summary.csv',
             PREV / 'source_thresholds.csv', PREV / 'prediction_freeze.json',
             PREV / 'input_freeze.json', PREV / 'label_matrix_freeze.json',
             old.OUT / 'thresholds.json']
    paths += list((PREV).glob('*_matrix.npz'))
    # Hash references to all prior inputs, including native labels and dictionaries.
    inputs = json.loads((PREV / 'input_freeze.json').read_text())['files'].copy()
    inputs.update({str(p): study.sha(p) for p in paths})
    cc.freeze(OUT / 'input_freeze.json', dict(files=inputs))
    (OUT / 'paper_brief.md').write_text('保持原论文的风险配额受控覆盖选样主线。本轮仅定位历史跨VLM结果变化，不能证明标签正确或在新数据泛化。\n')
    if not (OUT / 'research_log.md').exists():
        (OUT / 'research_log.md').write_text('# 执行记录\n\n先冻结2×2设计、固定50起点与停止条件，再生成缺失组合；最后读取缓存预测。已有历史结果已知，不称为盲验证。\n')
    dump('environment.json', dict(python=__import__('sys').version, numpy=np.__version__,
                                  scipy=__import__('scipy').__version__, solver=old.SOLVER))
    event('prepared')


def contexts():
    original = old.contexts()
    rows, labels = prior.load_native()
    risks, matrices = {}, {}
    for phase in rows:
        for source in study.SOURCES:
            for c in CLASSES:
                rr = [r for r in rows[phase] if r['class_name'] == c]
                ct = prior.risk_context(c, rr, labels[phase][source])
                ct['states'] = original[c]['states']
                with np.load(PREV / f'{phase}_{source}_{c}_matrix.npz', allow_pickle=False) as z:
                    assert np.array_equal(ct['ids'], z['sample_ids'])
                    assert np.array_equal(ct['cell'], z['cells'])
                    assert np.array_equal(ct['tiers'], z['tiers'])
                    matrices[phase, source, c] = z['matrix'].astype(bool)
                risks[phase, source, c] = ct
                if phase == 'B' and source == 'A':
                    assert ct['keys'] == original[c]['keys']
                    for f in ['ids', 'cell', 'tiers', 'matrix']:
                        expected = matrices[phase, source, c] if f == 'matrix' else ct[f]
                        assert np.array_equal(expected, original[c][f])
    return rows, risks, matrices


def select():
    cc.verify_freeze(OUT / 'input_freeze.json')
    rows, risks, matrices = contexts()
    spec = importlib.util.spec_from_file_location('matched_greedy_factorial', ROOT / 'output/experiments/实证论文补全_20261108/execute.py')
    matched = importlib.util.module_from_spec(spec); spec.loader.exec_module(matched)
    originals = json.loads((old.OUT / 'thresholds.json').read_text())
    stored = np.load(PREV / 'selections.npz', allow_pickle=False)
    stored_records = {(r['source'], r['threshold_mode'], r['class_name'], r['method'], int(r['selection_seed'])): r
                      for r in study.read(PREV / 'selection_runs.csv')}
    selections, records, thresholds = {}, [], []
    for risk in study.SOURCES:
        for coverage in study.SOURCES:
            for ci, c in enumerate(CLASSES):
                ct = dict(risks['B', risk, c], matrix=matrices['B', coverage, c])
                g, _ = cc.baseline(ct, 200, 'G', 20)
                gm = cc.count_metrics(ct['matrix'][g].sum(0), 20)
                own = {f: .95 * gm[f] for f in ('occurrence', 'support_completion')}
                for mode in MODES:
                    th = originals[c] if mode == 'original' else own
                    thresholds.append(dict(risk_source=risk, coverage_source=coverage, threshold_mode=mode,
                                           class_name=c, **{f: th[f] for f in own}))
                    for seed in SEEDS:
                        s0 = cc.initial_selection(ct, 200, seed, ci)
                        assert np.array_equal(s0, stored[f'{risk}__original__{c}__R__{seed}'])
                        for method in METHODS:
                            tick = time.monotonic()
                            # Both AA threshold modes are mathematically identical.
                            reused = risk == coverage
                            if reused:
                                reuse_mode = 'original' if risk == 'A' else mode
                                if risk == 'A':
                                    assert all(abs(own[f] - originals[c][f]) < 1e-12 for f in own)
                                selected = stored[f'{risk}__{reuse_mode}__{c}__{method}__{seed}']
                                info = stored_records[risk, reuse_mode, c, method, seed]
                            elif method == 'Greedy20':
                                selected, candidate, trace, reason = matched.greedy(ct, s0, th)
                                matched.check(ct, s0, selected, candidate, trace, th)
                                info = dict(stop_reason=reason)
                                if seed == SEEDS[0]:
                                    s2, c2, t2, r2 = matched.greedy(ct, s0, th)
                                    assert np.array_equal(s2, selected) and np.array_equal(c2, candidate)
                                    assert t2 == trace and r2 == reason
                            else:
                                selected, same_s0, info = old.run_method(ct, method, seed, ci, th)
                                assert np.array_equal(same_s0, s0)
                            cc.old.validate_selection(ct, selected, 200)
                            removed = len(set(s0) - set(selected)); assert removed <= 20
                            if method == 'R':
                                assert np.array_equal(selected, s0)
                            metrics = cc.count_metrics(ct['matrix'][selected].sum(0), 20)
                            selections[key(risk, coverage, mode, c, method, seed)] = selected
                            records.append(dict(risk_source=risk, coverage_source=coverage, threshold_mode=mode,
                                class_name=c, method=method, selection_seed=seed, removed=removed,
                                own_pass=cc.passes(metrics, th), reused=reused, seconds=time.monotonic() - tick,
                                stop_reason=info.get('stop_reason', ''), optimal_certified=info.get('optimal_certified', ''),
                                stage1_status=info.get('stage1_status', ''), stage2_status=info.get('stage2_status', '')))
                    event('selection_class', risk=risk, coverage=coverage, mode=mode, class_name=c, records=len(records))
                np.savez_compressed(OUT / 'selections_progress.npz', **selections)
                write('selection_runs_progress.csv', records)
    np.savez_compressed(OUT / 'selections.npz', **selections)
    write('selection_runs.csv', records); write('thresholds.csv', thresholds)
    cc.freeze(OUT / 'selection_freeze.json', dict(prediction_values_read_in_selection=False,
        files={str(OUT / n): study.sha(OUT / n) for n in ['selections.npz', 'selection_runs.csv', 'thresholds.csv']}))
    dump('smoke_audit.json', dict(status='pass', all_initial_sets_reproduced=True,
        first_start_greedy_determinism_all_hybrid_classes=True, exact_risk_quotas_and_20_removal_bound_all_runs=True))


def evaluate():
    cc.verify_freeze(OUT / 'input_freeze.json'); cc.verify_freeze(OUT / 'selection_freeze.json')
    cc.verify_freeze(PREV / 'prediction_freeze.json')
    rows, risks, matrices = contexts()
    selections = np.load(OUT / 'selections.npz', allow_pickle=False)
    runs = study.read(OUT / 'selection_runs.csv')
    th = {(r['risk_source'], r['coverage_source'], r['threshold_mode'], r['class_name']):
          {f: float(r[f]) for f in ['occurrence', 'support_completion']} for r in study.read(OUT / 'thresholds.csv')}
    pred, hashes, raw = {}, {}, {}
    for p in rows:
        pred[p], hh = cc.old.predictions(p, rows[p]); hashes.update(hh)
        offset = {r['sample_id']: i for i, r in enumerate(rows[p])}
        for source in study.SOURCES:
            for ci, c in enumerate(CLASSES):
                ct = risks[p, source, c]; x = matrices[p, source, c]
                control = prior.value_known_mask(x, ct['states'])
                pos = [offset[s] for s in ct['ids']]
                stats = {}
                for model in old.MODELS:
                    correct = (pred[p][model][:, pos] == ci).mean(0)
                    stats[model] = correct, prior.reference_stats(x, control, correct)
                raw[p, source, c] = x, control, stats
    cc.freeze(OUT / 'prediction_freeze.json', dict(files=hashes))
    cov, diag = [], []
    field_counts = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for rr in runs:
        risk, coverage, mode, c, method, seed = [rr[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'class_name', 'method', 'selection_seed']]
        selected = selections[key(risk, coverage, mode, c, method, seed)]
        base = dict(risk_source=risk, coverage_source=coverage, threshold_mode=mode,
                    class_name=c, method=method, selection_seed=int(seed))
        for evaluator in study.SOURCES:
            x, control, stats = raw['B', evaluator, c]
            a, b = x[selected], control[selected]
            metrics = cc.count_metrics(a.sum(0), 20)
            cov.append(dict(**base, evaluator=evaluator, occurrence_pct=100 * metrics['occurrence'],
                support_completion_pct=100 * metrics['support_completion'],
                pass_original=cc.passes(metrics, th['A', 'A', 'original', c]),
                pass_recalibrated=cc.passes(metrics, th[risk, evaluator, 'recalibrated', c]), removed=int(rr['removed'])))
            for model in old.MODELS:
                correct = stats[model][0]
                n1, n0, gap, eligible, proposed = prior.reference_stats(a, b, correct[selected])
                for reference in study.SOURCES:
                    d = raw['D', reference, c][2][model][1]
                    confirmed = proposed & d[4]
                    diag.append(dict(**base, evaluator=evaluator, reference=reference, model=model,
                        eligible=int(eligible.sum()), proposed=int(proposed.sum()),
                        reference_evaluable=int((proposed & d[3]).sum()), confirmed=int(confirmed.sum()),
                        error_pct=100 * float(1 - correct[selected].mean())))
                    for j, (ns, field, value) in enumerate(risks['B', evaluator, c]['states']):
                        fk = (risk, coverage, mode, evaluator, reference, c, ns, field, method)
                        field_counts[fk] += np.array([eligible[j], proposed[j], proposed[j] & d[3][j], confirmed[j]], dtype=np.int64)
        if len(cov) % 800 == 0:
            event('evaluation_progress', coverage_rows=len(cov), diagnostic_rows=len(diag))
    write('coverage.csv', cov); write('diagnostic.csv', diag)
    fields = []
    names = ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference', 'class_name', 'namespace', 'field', 'method']
    for fk, counts in field_counts.items():
        fields.append(dict(zip(names, fk), **{f + '_per_start': float(v / len(SEEDS)) for f, v in
            zip(['eligible', 'proposed', 'reference_evaluable', 'confirmed'], counts)}))
    write('by_field.csv', fields)
    summarize(cov, diag)
    audit(risks, matrices, selections, runs, cov, diag)


def summarize(cov, diag):
    dc, cg, paired_values = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))
    for r in diag:
        k = tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference', 'method'])
        dc[k].append(r)
        paired_values[k[:-1]][r['selection_seed'], r['method']] += r['confirmed']
    for r in cov:
        cg[tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'method'])].append(r)
    summary, paired = [], []
    for k, dd in dc.items():
        risk, coverage, mode, evaluator, reference, method = k
        ccrows = cg[risk, coverage, mode, evaluator, method]
        assert len(dd) == 1500 and len(ccrows) == 300
        summary.append(dict(risk_source=risk, coverage_source=coverage, threshold_mode=mode,
            evaluator=evaluator, reference=reference, method=method,
            support_completion_pct=float(np.mean([r['support_completion_pct'] for r in ccrows])),
            occurrence_pct=float(np.mean([r['occurrence_pct'] for r in ccrows])),
            pass_original_count=sum(r['pass_original'] for r in ccrows),
            pass_recalibrated_count=sum(r['pass_recalibrated'] for r in ccrows),
            confirmed_per_start=sum(r['confirmed'] for r in dd) / len(SEEDS),
            proposed_per_start=sum(r['proposed'] for r in dd) / len(SEEDS),
            error_pct=float(np.mean([r['error_pct'] for r in dd])),
            mean_removed=float(np.mean([r['removed'] for r in ccrows]))))
    for k, values in paired_values.items():
        for method, comparator in [('P20', 'Greedy20'), ('P20', 'R'), ('Greedy20', 'R')]:
            delta = np.array([values[seed, method] - values[seed, comparator] for seed in SEEDS])
            paired.append(dict(zip(['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference'], k),
                method=method, comparator=comparator, mean_confirmed_difference=float(delta.mean()),
                positive_starts=int((delta > 0).sum()), ties=int((delta == 0).sum()), negative_starts=int((delta < 0).sum()),
                SD_fixed_starts=float(delta.std(ddof=1))))
    write('summary.csv', summary); write('paired_differences.csv', paired)


def audit(risks, matrices, selections, runs, cov, diag):
    # Independently recompute every quota, membership, coverage metric and gate.
    th = {(r['risk_source'], r['coverage_source'], r['threshold_mode'], r['class_name']):
          {f: float(r[f]) for f in ['occurrence', 'support_completion']} for r in study.read(OUT / 'thresholds.csv')}
    for r in runs:
        risk, coverage, mode, c, method, seed = [r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'class_name', 'method', 'selection_seed']]
        ix = selections[key(risk, coverage, mode, c, method, seed)]
        ct = risks['B', risk, c]
        cc.old.validate_selection(ct, ix, 200)
        s0 = selections[key(risk, 'A', mode, c, 'R', seed)]
        assert len(set(s0) - set(ix)) == int(r['removed']) <= 20
    for r in cov:
        risk, coverage, mode, c, method, seed = [r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'class_name', 'method', 'selection_seed']]
        ix = selections[key(risk, coverage, mode, c, method, seed)]
        counts = matrices['B', r['evaluator'], c][ix].sum(0)
        occurrence = float((counts > 0).mean()); support = float(np.minimum(counts, 20).sum() / (20 * len(counts)))
        assert abs(100 * occurrence - r['occurrence_pct']) < 1e-10
        assert abs(100 * support - r['support_completion_pct']) < 1e-10
        for suffix, thresholds in [('original', th['A', 'A', 'original', c]),
                                   ('recalibrated', th[risk, r['evaluator'], 'recalibrated', c])]:
            assert r['pass_' + suffix] == (occurrence + 1e-12 >= thresholds['occurrence'] and support + 1e-12 >= thresholds['support_completion'])
    previous = {(r['selection_source'], r['threshold_mode'], r['evaluator'], r['reference_label_source'], r['method']): r
                for r in study.read(PREV / 'summary.csv')}
    summary = study.read(OUT / 'summary.csv'); reproduced = 0
    for r in summary:
        if r['risk_source'] != r['coverage_source']:
            continue
        source = r['risk_source']; mode = 'original' if source == 'A' else r['threshold_mode']
        expected = previous[source, mode, r['evaluator'], r['reference'], r['method']]
        for f in ['support_completion_pct', 'occurrence_pct', 'confirmed_per_start', 'proposed_per_start', 'error_pct']:
            assert abs(float(r[f]) - float(expected[f])) < 1e-9, (r, f)
        reproduced += 1
    # Field aggregation independently conserves all diagnostic counts.
    sums = defaultdict(float)
    for r in study.read(OUT / 'by_field.csv'):
        k = tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference', 'method'])
        sums[k] += float(r['confirmed_per_start'])
    for r in summary:
        k = tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference', 'method'])
        assert abs(sums[k] - float(r['confirmed_per_start'])) < 1e-9
    cc.verify_freeze(OUT / 'input_freeze.json'); cc.verify_freeze(OUT / 'selection_freeze.json')
    dump('validation.json', dict(status='pass', selected_sets=len(runs), coverage_rows=len(cov),
        diagnostic_rows=len(diag), diagonal_summary_rows_reproduced=reproduced,
        same_initial_sets_across_coverage=True, exact_quota_and_removal_checks=True,
        field_totals_conserve_confirmed_counts=True, label_accuracy_established=False))
    event('audit_pass', selections=len(runs), diagnostic_rows=len(diag))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'select', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'select': select, 'evaluate': evaluate}[args.stage]()
