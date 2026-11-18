"""Reproducible source-agreement, selection and diagnostic sensitivity analyses."""
from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

import verify_cross_vlm_20261112 as study

ROOT, OUT, BASE = study.ROOT, study.OUT, study.BASE
cc, old = study.cc, study.historical
CLASSES, METHODS, SEEDS = study.CLASSES, study.METHODS, old.SEEDS
UNKNOWN = {'unknown', 'not_observable', 'not_applicable', 'missing'}


def canonical(value):
    return tuple(sorted(value)) if isinstance(value, list) else value


def is_unknown(value):
    return bool(set(value) & UNKNOWN) if isinstance(value, list) else value in UNKNOWN


def agreement(left, right):
    assert len(left) == len(right) and len(left)
    a, b = list(map(canonical, left)), list(map(canonical, right))
    same = np.array([x == y for x, y in zip(a, b)])
    ua, ub = np.array(list(map(is_unknown, left))), np.array(list(map(is_unknown, right)))
    known = ~ua & ~ub
    ca, cb = Counter(a), Counter(b)
    chance = sum(ca[v] * cb[v] for v in ca) / len(a) ** 2
    observed = float(same.mean())
    is_set = isinstance(left[0], list)
    return dict(n=len(a), exact_agreement_pct=100 * observed,
                cohen_kappa=(observed - chance) / (1 - chance) if chance < 1 - 1e-15 else None,
                A_unknown_pct=100 * float(ua.mean()), B_unknown_pct=100 * float(ub.mean()),
                both_known_n=int(known.sum()),
                both_known_agreement_pct=100 * float(same[known].mean()) if known.any() else None,
                A_known_B_unknown=int((~ua & ub).sum()), A_unknown_B_known=int((ua & ~ub).sum()),
                both_unknown=int((ua & ub).sum()),
                mean_jaccard=(float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y))
                                             for x, y in zip(left, right)])) if is_set else None))


def load_native():
    rows = study.manifests()
    labels = {p: {s: {} for s in study.SOURCES} for p in rows}
    for p, rr in rows.items():
        for r in rr:
            for s in study.SOURCES:
                labels[p][s][r['sample_id']] = json.loads(study.native_path(p, s, r).read_text())['features']
    return rows, labels


def risk_context(c, rows, features):
    assert len({r['image_sha256'] for r in rows}) == len(rows), 'Diagnostics require distinct original images per class'
    scores = [study.robustness.score_sample(float(r['bbox_area_ratio']), features[r['sample_id']],
                                            c, 'equal_parts', study.robustness.CONFIG) for r in rows]
    midpoint = np.array([r['score_midpoint'] for r in scores])
    widths = np.array([r['score_upper'] - r['score_lower'] for r in scores])
    ids = np.array([r['sample_id'] for r in rows])
    tiers = np.zeros(len(rows), int)
    for q, ix in enumerate(study.robustness.five_tiers(np.arange(len(rows)), midpoint, ids), 1):
        tiers[ix] = q
    rawkeys = [(int(tiers[i]), round(float(midpoint[i]) * 1e8), round(float(widths[i]) * 1e8))
               for i in range(len(rows))]
    keys = sorted(set(rawkeys)); index = {v: i for i, v in enumerate(keys)}
    return dict(ids=ids, rows=rows, score=midpoint, tiers=tiers, keys=keys,
                widths=widths, cell=np.array([index[k] for k in rawkeys]),
                embeddings=np.zeros((len(rows), 1)))


def native_analysis(rows, labels, original):
    fields, transitions, risks, risks_summary = [], [], [], []
    contexts = {p: {s: {} for s in study.SOURCES} for p in rows}
    for phase, rr in rows.items():
        for c in CLASSES:
            cr = [r for r in rr if r['class_name'] == c]
            la, lb = labels[phase]['A'], labels[phase]['B']
            field_ids = list(la[cr[0]['sample_id']])
            for f in field_ids:
                a, b = [[ll[r['sample_id']][f] for r in cr] for ll in (la, lb)]
                # Geometric scale is program-computed and cannot be a VLM agreement win.
                fields.append(dict(phase=phase, class_name=c, namespace='native', field=f, **agreement(a, b)))
                for (va, vb), n in Counter((str(canonical(x)), str(canonical(y))) for x, y in zip(a, b)).items():
                    transitions.append(dict(phase=phase, class_name=c, field=f, A_value=va, B_value=vb, n=n))
            for source in study.SOURCES:
                contexts[phase][source][c] = risk_context(c, cr, labels[phase][source])
            a, b = [contexts[phase][s][c] for s in study.SOURCES]
            if phase == 'B':
                assert np.array_equal(a['ids'], original[c]['ids'])
                assert np.allclose(a['score'], original[c]['score'], rtol=0, atol=1e-10)
                assert np.array_equal(a['tiers'], original[c]['tiers'])
                assert a['keys'] == original[c]['keys'] and np.array_equal(a['cell'], original[c]['cell'])
            for i, r in enumerate(cr):
                risks.append(dict(phase=phase, class_name=c, sample_id=r['sample_id'],
                    A_score=float(a['score'][i]), B_score=float(b['score'][i]),
                    A_Q=int(a['tiers'][i]), B_Q=int(b['tiers'][i]),
                    A_width=float(a['widths'][i]), B_width=float(b['widths'][i]),
                    cell_changed=a['keys'][a['cell'][i]] != b['keys'][b['cell'][i]]))
            risks_summary.append(dict(phase=phase, class_name=c, targets=len(cr),
                score_MAE=float(np.abs(a['score'] - b['score']).mean()),
                score_spearman=float(spearmanr(a['score'], b['score']).statistic),
                Q_changed_pct=100 * float((a['tiers'] != b['tiers']).mean()),
                cell_changed_pct=100 * study.robustness.cell_change_fraction(a['keys'], a['cell'], b['keys'], b['cell']),
                A_cells=len(a['keys']), B_cells=len(b['keys'])))
        study.event('native_analysis', phase=phase)
    study.write('native_field_agreement.csv', fields)
    study.write('native_value_transitions.csv', transitions)
    study.write('risk_per_target.csv', risks)
    study.write('risk_summary.csv', risks_summary)
    study.dump('native_summary.json', dict(paired_targets=sum(map(len, rows.values())),
        field_instances=sum(r['n'] for r in fields),
        micro_agreement_pct=sum(r['exact_agreement_pct'] * r['n'] for r in fields) / sum(r['n'] for r in fields),
        macro_field_class_agreement_pct=float(np.mean([r['exact_agreement_pct'] for r in fields])),
        B_Q_changed_pct=float(np.mean([r['Q_changed_pct'] for r in risks_summary if r['phase'] == 'B'])),
        B_cell_changed_pct=float(np.mean([r['cell_changed_pct'] for r in risks_summary if r['phase'] == 'B'])),
        accuracy_established=False, results=risks_summary))
    return contexts


def value_known_mask(matrix, states):
    """A field's known alternative states only; unknown never becomes a negative."""
    controls = np.zeros_like(matrix, dtype=bool)
    groups = defaultdict(list)
    for j, state in enumerate(states):
        groups[tuple(state[:2])].append(j)
    for ix in groups.values():
        controls[:, ix] = matrix[:, ix].any(axis=1)[:, None] & ~matrix[:, ix]
    return controls


def matrix_for(rows, native, specific, states):
    x = np.zeros((len(rows), len(states)), bool)
    for i, r in enumerate(rows):
        for j, (namespace, field, state) in enumerate(states):
            if field == 'target_scale':
                area = float(r['bbox_area_ratio'])
                v = 'tiny' if area < .01 else 'small' if area < .03 else 'medium' if area < .1 else 'large'
            else:
                features = (specific[r['sample_id']] if namespace == 'specific' and
                            r['class_name'] not in ('dog', 'cat') else native[r['sample_id']])
                v = features[field]
            x[i, j] = state in v if isinstance(v, list) else state == v
    return x


def load_specific(rows):
    features = {p: {s: {} for s in study.SOURCES} for p in rows}
    agreements, provenance, incomplete = [], [], []
    raw_index = None
    label_freeze = {}
    for p, rr in rows.items():
        for c in CLASSES:
            if c in ('dog', 'cat'):
                continue
            fields = json.loads(study.dictionary(c).read_text())['features']
            cr = [r for r in rr if r['class_name'] == c]
            for r in cr:
                sid = r['sample_id']
                pa, ev = study.specific_original(p, r)
                pb = OUT / 'specific_B' / p / c / f'{sid}.json'
                if not pb.exists():
                    incomplete.append(dict(phase=p, class_name=c, sample_id=sid)); continue
                ja, jb = [json.loads(q.read_text()) for q in (pa, pb)]
                study.VALIDATE(jb['features'], fields)
                assert jb['sample_id'] == sid and jb['model'] == study.SOURCES['B']
                assert jb['evidence_sha256'] == study.sha(ev)
                assert jb['dictionary_sha256'] == study.sha(study.dictionary(c))
                if 'raw_path' in jb:
                    raw_path = Path(jb['raw_path'])
                else:
                    if raw_index is None:
                        raw_index = {study.sha(q): q for q in (OUT / 'raw_specific').glob('**/*.json')
                                     if not q.name.endswith('.meta.json')}
                    raw_path = raw_index[jb['raw_sha256']]
                assert study.sha(raw_path) == jb['raw_sha256']
                raw = json.loads(raw_path.read_text())
                assert raw['model'] == study.SOURCES['B']
                assert raw['choices'][0]['finish_reason'] == 'stop'
                raw_features = study.unique_json_object(raw['choices'][0]['message']['content'])
                assert raw_features == jb['features'], 'Parsed labels differ from actual model answer'
                label_freeze.update({str(pb): study.sha(pb), str(raw_path): study.sha(raw_path)})
                for s, j in [('A', ja), ('B', jb)]:
                    features[p][s][sid] = j['features']
                provenance.append(dict(phase=p, class_name=c, sample_id=sid, path=str(pb), sha256=study.sha(pb),
                                       raw_path=str(raw_path), raw_sha256=study.sha(raw_path)))
            matched = [r for r in cr if r['sample_id'] in features[p]['B']]
            if matched:
                for field in fields:
                    f = field['feature_id']
                    a, b = [[features[p][s][r['sample_id']][f] for r in matched] for s in study.SOURCES]
                    agreements.append(dict(phase=p, class_name=c, namespace='specific', field=f, **agreement(a, b)))
    if agreements:
        study.write('specific_field_agreement.csv', agreements)
    if provenance:
        study.write('new_specific_provenance.csv', provenance)
    study.dump('specific_completeness.json', dict(expected=2400, completed=len(provenance), missing=len(incomplete)))
    if incomplete:
        study.write('specific_missing.csv', incomplete)
        raise ValueError(f'{len(incomplete)} independent specific annotations missing; native analyses retained')
    cc.freeze(OUT / 'new_label_freeze.json', {'files': label_freeze})
    return features


def reference_stats(x, controls, correct):
    n1, n0 = x.sum(0), controls.sum(0)
    r1 = np.divide(x.T @ correct, n1, out=np.zeros(x.shape[1]), where=n1 > 0)
    r0 = np.divide(controls.T @ correct, n0, out=np.zeros(x.shape[1]), where=n0 > 0)
    gap = 100 * (r1 - r0)
    eligible = (n1 >= 20) & (n0 >= 20)
    return n1, n0, gap, eligible, eligible & (gap <= -10 + 1e-12)


def stored_methods():
    main = np.load(BASE / 'main_sets.npz', allow_pickle=False)
    gs = np.load(ROOT / 'artifacts/empirical_completion_20261108/greedy_sets.npz', allow_pickle=False)
    return {k: z[k] for z in (main, gs) for k in z.files}


def generate_sets(contexts):
    spec = importlib.util.spec_from_file_location('matched_greedy_20261108',
        ROOT / 'output/experiments/实证论文补全_20261108/execute.py')
    matched = importlib.util.module_from_spec(spec); spec.loader.exec_module(matched)
    # Freeze exact implementation, not an approximation of the previous greedy rule.
    thresholds_a = json.loads((BASE / 'thresholds.json').read_text())
    historical_sets = stored_methods()
    generated, records, threshold_rows = {}, [], []
    for source in study.SOURCES:
        for ci, c in enumerate(CLASSES):
            ct = contexts['B'][source][c]
            g, _ = cc.baseline(ct, 200, 'G', 20)
            gm = cc.count_metrics(ct['matrix'][g].sum(0), 20)
            own_threshold = {k: .95 * gm[k] for k in ('occurrence', 'support_completion')}
            for k in own_threshold:
                if source == 'A':
                    assert abs(own_threshold[k] - thresholds_a[c][k]) < 1e-10, ('threshold reproduction', c, k)
            threshold_rows.append(dict(source=source, class_name=c,
                occurrence=own_threshold['occurrence'], support_completion=own_threshold['support_completion']))
            modes = ['original'] if source == 'A' else ['original', 'recalibrated']
            for mode in modes:
                th = thresholds_a[c] if mode == 'original' else own_threshold
                for seed in SEEDS:
                    s0 = cc.initial_selection(ct, 200, seed, ci)
                    for method in METHODS:
                        tick = time.monotonic()
                        info = dict(stop_reason='frozen_historical_selection')
                        if source == 'A':
                            selected = historical_sets[old.key(c, method, seed)]
                            assert np.array_equal(np.sort(s0), np.sort(historical_sets[old.key(c, 'R', seed)]))
                        elif method == 'Greedy20':
                            selected, candidate, trace, reason = matched.greedy(ct, s0, th)
                            matched.check(ct, s0, selected, candidate, trace, th)
                            info = dict(stop_reason=reason)
                        else:
                            selected, same_s0, info = old.run_method(ct, method, seed, ci, th)
                            assert np.array_equal(same_s0, s0)
                        cc.old.validate_selection(ct, selected, 200)
                        removed = len(set(s0) - set(selected)); assert removed <= 20
                        key = f'{source}__{mode}__{c}__{method}__{seed}'
                        generated[key] = selected
                        metrics = cc.count_metrics(ct['matrix'][selected].sum(0), 20)
                        records.append(dict(source=source, threshold_mode=mode, class_name=c,
                            method=method, selection_seed=seed, removed=removed,
                            own_pass=cc.passes(metrics, th), seconds=time.monotonic() - tick,
                            stop_reason=info.get('stop_reason', ''),
                            optimal_certified=info.get('optimal_certified', ''),
                            stage1_status=info.get('stage1_status', ''), stage2_status=info.get('stage2_status', '')))
                study.event('selection_complete', source=source, mode=mode, class_name=c, selections=len(records))
                # Persist progress, so solver or process interruptions are inspectable.
                np.savez_compressed(OUT / 'new_selections_progress.npz', **generated)
                study.write('selection_runs_progress.csv', records)
    np.savez_compressed(OUT / 'selections.npz', **generated)
    study.write('selection_runs.csv', records)
    study.write('source_thresholds.csv', threshold_rows)
    study.dump('selection_freeze.json', dict(predictions_read_in_selection=False,
        historical_predictions_previously_analyzed=True,
        files={str(OUT / n): study.sha(OUT / n) for n in ['selections.npz', 'selection_runs.csv', 'source_thresholds.csv']}))
    return generated, records, threshold_rows


def evaluate(contexts, selections, selection_runs, thresholds, rows):
    # Classifier outputs are opened only after new selections are frozen.
    predictions, prediction_hashes = {}, {}
    for p in rows:
        predictions[p], hashes = cc.old.predictions(p, rows[p]); prediction_hashes.update(hashes)
    offsets = {p: {r['sample_id']: i for i, r in enumerate(rr)} for p, rr in rows.items()}
    source_thresholds = {(r['source'], r['class_name']): r for r in thresholds}
    original_thresholds = json.loads((BASE / 'thresholds.json').read_text())
    raw_cache, reference_rows, risk_gradients = {}, [], []
    for source in study.SOURCES:
        for ci, c in enumerate(CLASSES):
            for phase in rows:
                ct = contexts[phase][source][c]
                x = ct['matrix']; controls = value_known_mask(x, ct['states'])
                pos = [offsets[phase][s] for s in ct['ids']]
                stats = {}
                for model in old.MODELS:
                    correct = (predictions[phase][model][:, pos] == ci).mean(0)
                    q1, q5 = ct['tiers'] == 1, ct['tiers'] == 5
                    e1, e5 = 100 * float(1 - correct[q1].mean()), 100 * float(1 - correct[q5].mean())
                    risk_gradients.append(dict(phase=phase, source=source, class_name=c, model=model,
                        Q1_error_pct=e1, Q5_error_pct=e5, Q5_minus_Q1_pp=e5 - e1))
                    stats[model] = (correct, reference_stats(x, controls, correct))
                    if phase == 'D':
                        n1, n0, delta, eligible, weak = stats[model][1]
                        for j, (ns, f, v) in enumerate(ct['states']):
                            reference_rows.append(dict(source=source, class_name=c, model=model,
                                state_index=j, namespace=ns, field=f, value=v,
                                state_targets=int(n1[j]), control_targets=int(n0[j]),
                                delta_pp=float(delta[j]), evaluable=bool(eligible[j]), weak=bool(weak[j])))
                raw_cache[phase, source, c] = x, controls, stats
    cov_rows, diag_rows, overlap_rows = [], [], []
    for record in selection_runs:
        source, mode, c, method, seed = (record[k] for k in ['source', 'threshold_mode', 'class_name', 'method', 'selection_seed'])
        selected = selections[f'{source}__{mode}__{c}__{method}__{seed}']
        aselected = selections[f'A__original__{c}__{method}__{seed}']
        overlap_rows.append(dict(selection_source=source, threshold_mode=mode, class_name=c, method=method,
            selection_seed=seed, retained_fraction_vs_A=len(set(selected) & set(aselected)) / 200,
            jaccard_vs_A=len(set(selected) & set(aselected)) / len(set(selected) | set(aselected))))
        for evaluator in study.SOURCES:
            ct = contexts['B'][evaluator][c]
            x, controls, stats = raw_cache['B', evaluator, c]
            a, b = x[selected], controls[selected]
            metrics = cc.count_metrics(a.sum(0), 20)
            cov_rows.append(dict(selection_source=source, threshold_mode=mode, evaluator=evaluator,
                class_name=c, method=method, selection_seed=seed,
                occurrence_pct=100 * metrics['occurrence'], support_completion_pct=100 * metrics['support_completion'],
                supported_fraction_pct=100 * metrics['support_threshold_fraction'],
                pass_original_threshold=cc.passes(metrics, original_thresholds[c]),
                pass_evaluator_threshold=cc.passes(metrics, source_thresholds[evaluator, c]), removed=record['removed']))
            for model in old.MODELS:
                correct = stats[model][0]
                n1, n0, delta, eligible, proposed = reference_stats(a, b, correct[selected])
                for reference in study.SOURCES:
                    d_stats = raw_cache['D', reference, c][2][model][1]
                    confirmed = proposed & d_stats[4]
                    diag_rows.append(dict(selection_source=source, threshold_mode=mode,
                        discovery_label_source=evaluator, reference_label_source=reference,
                        class_name=c, method=method, selection_seed=seed, model=model,
                        eligible=int(eligible.sum()), proposed=int(proposed.sum()),
                        reference_evaluable=int((proposed & d_stats[3]).sum()), confirmed=int(confirmed.sum()),
                        error_pct=100 * float(1 - correct[selected].mean())))
    study.write('coverage_cross_evaluation.csv', cov_rows)
    study.write('diagnostic_cross_evaluation.csv', diag_rows)
    study.write('D_condition_reference.csv', reference_rows)
    study.write('selection_overlap.csv', overlap_rows)
    study.write('risk_error_gradient.csv', risk_gradients)
    study.dump('prediction_freeze.json', {'files': prediction_hashes})
    summary_rows = []
    grouped = defaultdict(list)
    for r in cov_rows:
        grouped[tuple(r[k] for k in ['selection_source', 'threshold_mode', 'evaluator', 'method'])].append(r)
    for (s, mode, evaluator, method), rr in grouped.items():
        for reference in study.SOURCES:
            dd = [r for r in diag_rows if r['selection_source'] == s and r['threshold_mode'] == mode and
                  r['discovery_label_source'] == evaluator and r['reference_label_source'] == reference and r['method'] == method]
            assert len(rr) == 300 and len(dd) == 1500
            summary_rows.append(dict(selection_source=s, threshold_mode=mode, evaluator=evaluator,
                reference_label_source=reference, method=method,
                occurrence_pct=float(np.mean([r['occurrence_pct'] for r in rr])),
                support_completion_pct=float(np.mean([r['support_completion_pct'] for r in rr])),
                pass_original_count=sum(r['pass_original_threshold'] for r in rr),
                pass_evaluator_count=sum(r['pass_evaluator_threshold'] for r in rr),
                mean_removed=float(np.mean([r['removed'] for r in rr])),
                error_pct=float(np.mean([r['error_pct'] for r in dd])),
                proposed_per_start=sum(r['proposed'] for r in dd) / len(SEEDS),
                confirmed_per_start=sum(r['confirmed'] for r in dd) / len(SEEDS)))
    study.write('summary.csv', summary_rows)
    # Preserve the 50 paired starts rather than implying 50 independent datasets.
    diag_index = defaultdict(lambda: defaultdict(int))
    for r in diag_rows:
        key = tuple(r[k] for k in ['selection_source', 'threshold_mode', 'discovery_label_source', 'reference_label_source'])
        diag_index[key][r['selection_seed'], r['method']] += r['confirmed']
    paired = []
    for key, values in diag_index.items():
        for method, comparator in [('Greedy20', 'R'), ('P20', 'R'), ('P20', 'Greedy20')]:
            diff = np.array([values[s, method] - values[s, comparator] for s in SEEDS])
            paired.append(dict(selection_source=key[0], threshold_mode=key[1], evaluator=key[2],
                reference_label_source=key[3], method=method, comparator=comparator,
                mean_confirmed_difference=float(diff.mean()), SD_fixed_starts=float(diff.std(ddof=1)),
                positive_starts=int((diff > 0).sum()), ties=int((diff == 0).sum()), negative_starts=int((diff < 0).sum())))
    study.write('paired_diagnostic_differences.csv', paired)
    class_summary = []
    scenarios = list(dict.fromkeys((r['selection_source'], r['threshold_mode']) for r in cov_rows))
    for s, mode in scenarios:
        for evaluator in study.SOURCES:
            for c in CLASSES:
                for method in METHODS:
                    rr = [r for r in cov_rows if (r['selection_source'], r['threshold_mode'], r['evaluator'],
                          r['class_name'], r['method']) == (s, mode, evaluator, c, method)]
                    class_summary.append(dict(selection_source=s, threshold_mode=mode, evaluator=evaluator,
                        class_name=c, method=method, occurrence_pct=float(np.mean([r['occurrence_pct'] for r in rr])),
                        support_completion_pct=float(np.mean([r['support_completion_pct'] for r in rr])),
                        pass_original_count=sum(r['pass_original_threshold'] for r in rr),
                        pass_evaluator_count=sum(r['pass_evaluator_threshold'] for r in rr)))
    study.write('by_class.csv', class_summary)
    historical_summary = json.loads((ROOT / 'artifacts/empirical_completion_20261108/summary_corrected.json').read_text())['methods']
    for r in summary_rows:
        if (r['selection_source'], r['threshold_mode'], r['evaluator'], r['reference_label_source']) == ('A', 'original', 'A', 'A'):
            expected = historical_summary[r['method']]
            for f in ['occurrence_pct', 'support_completion_pct', 'error_pct']:
                assert abs(r[f] - expected[f]) < 1e-9, (r['method'], f)
            assert abs(r['confirmed_per_start'] - expected['confirmed']) < 1e-9
    study.dump('result_traceability.json', dict(status='pass', historical_A_metrics_and_counts_reproduced=True,
        coverage_rows=len(cov_rows), diagnostic_rows=len(diag_rows), conditions_reference_rows=len(reference_rows),
        new_classifier_inference=False, new_training=False, same_image_per_target=True))
    return summary_rows


def run():
    cc.verify_freeze(OUT / 'input_freeze.json')
    analysis_inputs = [Path(__file__), ROOT / 'scripts/report_cross_vlm_20261112.py',
                       ROOT / 'scripts/repair_cross_vlm_format_20261112.py',
                       ROOT / 'scripts/repair_cross_vlm_constrained_20261112.py',
                       ROOT / 'output/experiments/实证论文补全_20261108/execute.py',
                       ROOT / 'scripts/run_category_five_dimension_pilot.py',
                       OUT / 'specific_runtime.json', OUT / 'format_repair_protocol.json',
                       OUT / 'format_repair_audit.json', OUT / 'constrained_repair_protocol.json',
                       OUT / 'constrained_repair_audit.json']
    cc.freeze(OUT / 'analysis_code_freeze.json', {'files': {str(p): study.sha(p) for p in analysis_inputs}})
    rows, labels = load_native()
    original = old.contexts()
    contexts = native_analysis(rows, labels, original)
    specific = load_specific(rows)
    for p, rr in rows.items():
        for source in study.SOURCES:
            for c in CLASSES:
                ct = contexts[p][source][c]
                states = original[c]['states']
                ct['states'] = states
                ct['matrix'] = matrix_for(ct['rows'], labels[p][source], specific[p][source], states)
                if source == 'A':
                    with np.load(BASE / f'{p}_{c}_matrix.npz', allow_pickle=False) as z:
                        assert np.array_equal(ct['matrix'], z['matrix'])
                        assert np.array_equal(ct['ids'], z['ids'] if 'ids' in z else z['sample_ids'])
                np.savez_compressed(OUT / f'{p}_{source}_{c}_matrix.npz', matrix=ct['matrix'],
                                    sample_ids=ct['ids'], tiers=ct['tiers'], cells=ct['cell'])
    study.dump('label_matrix_freeze.json', {'files': {str(p): study.sha(p) for p in OUT.glob('*_matrix.npz')}})
    selections, records, thresholds = generate_sets(contexts)
    summary = evaluate(contexts, selections, records, thresholds, rows)
    study.dump('summary.json', dict(scope='Historical cross-VLM consistency and downstream robustness; no semantic ground truth or fresh unseen-data validation',
        sources=study.SOURCES, paired_targets=3600, new_specific_annotations=2400,
        selected_per_class=200, selection_starts=50, support_cap=20, removal_limit=20,
        aggregation='coverage macro over 6 classes x 50 starts; diagnostic counts summed over 6 classes x 5 models, then averaged over 50 starts',
        results=summary))
    audit()
    study.event('analysis_complete')


def audit():
    for n in ['input_freeze.json', 'analysis_code_freeze.json', 'new_label_freeze.json', 'label_matrix_freeze.json', 'selection_freeze.json', 'prediction_freeze.json']:
        cc.verify_freeze(OUT / n)
    matrices = {}
    for p in OUT.glob('*_matrix.npz'):
        with np.load(p, allow_pickle=False) as z:
            matrices[p.stem.removesuffix('_matrix')] = z['matrix']
    selections = np.load(OUT / 'selections.npz', allow_pickle=False)
    rows = study.read(OUT / 'coverage_cross_evaluation.csv')
    for r in rows:
        s, mode, c, m, seed, evaluator = [r[k] for k in ['selection_source', 'threshold_mode', 'class_name', 'method', 'selection_seed', 'evaluator']]
        ix = selections[f'{s}__{mode}__{c}__{m}__{seed}']
        counts = matrices[f'B_{evaluator}_{c}'][ix].sum(0)
        occurrence = float((counts > 0).mean() * 100)
        support = float(np.minimum(counts, 20).sum() / (20 * len(counts)) * 100)
        assert abs(occurrence - float(r['occurrence_pct'])) < 1e-10
        assert abs(support - float(r['support_completion_pct'])) < 1e-10
    summary = study.read(OUT / 'summary.csv')
    diagnostic = study.read(OUT / 'diagnostic_cross_evaluation.csv')
    for r in summary:
        matching = [d for d in diagnostic if all(d[k] == r[rk] for k, rk in [
            ('selection_source', 'selection_source'), ('threshold_mode', 'threshold_mode'),
            ('discovery_label_source', 'evaluator'), ('reference_label_source', 'reference_label_source'), ('method', 'method')])]
        assert len(matching) == 1500
        assert abs(sum(int(d['confirmed']) for d in matching) / 50 - float(r['confirmed_per_start'])) < 1e-10
    result = dict(status='pass', coverage_rows_independently_recomputed=len(rows),
                  diagnostic_aggregates_recomputed=len(summary), input_and_selection_freezes_verified=True,
                  label_accuracy_established=False, fresh_generalization_test=False)
    study.dump('validation.json', result)


if __name__ == '__main__':
    import time
    run()
