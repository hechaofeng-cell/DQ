"""Exact support/gap accounting on frozen cross-VLM and consensus selections."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import run_label_factorial_20261112 as previous

study, cc, old = previous.study, previous.cc, previous.old
ROOT = study.ROOT
OUT = ROOT / 'artifacts/label_support_diagnosis_20261112'
CONSENSUS = ROOT / 'artifacts/label_consensus_check_20261112'
METHODS = ['R', 'Greedy20', 'P20']
VIEWS = ['A', 'B', 'C']
AXES = [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B'), ('C', 'C')]
COMPARISONS = [('P20', 'R'), ('P20', 'Greedy20'), ('Greedy20', 'R')]
GROUP = ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference']


def dump(name, value):
    cc.dump(OUT / name, value)


def write(name, rows):
    cc.write(OUT / name, rows)


def decompose(method_eligible, comparator_eligible, method_confirmed, comparator_confirmed):
    """A mutually exclusive four-way partition, with no percentage attribution."""
    em, eb, cm, cb = [np.asarray(a, dtype=bool) for a in
                      [method_eligible, comparator_eligible, method_confirmed, comparator_confirmed]]
    assert em.shape == eb.shape == cm.shape == cb.shape
    assert not (cm & ~em).any() and not (cb & ~eb).any()
    new, lost = cm & ~cb, cb & ~cm
    masks = dict(new_from_eligibility=new & ~eb, new_on_common_eligible=new & eb,
                 lost_from_eligibility=lost & ~em, lost_on_common_eligible=lost & em)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    counts['eligibility_net'] = counts['new_from_eligibility'] - counts['lost_from_eligibility']
    counts['common_gap_net'] = counts['new_on_common_eligible'] - counts['lost_on_common_eligible']
    counts['total_difference'] = int(cm.sum()) - int(cb.sum())
    assert counts['total_difference'] == counts['eligibility_net'] + counts['common_gap_net']
    assert counts['common_gap_net'] == int((cm & eb).sum()) - int((cb & em).sum())
    return counts, masks


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'completion.json').exists(), 'Completed experiment cannot be overwritten'
    for root in [previous.OUT, CONSENSUS]:
        cc.verify_freeze(root / 'final_outputs_freeze.json')
        cc.verify_freeze(root / 'input_freeze.json')
    protocol = dict(
        scope='Exploratory historical mechanism accounting after known cross-VLM and consensus results; no new model calls',
        hypothesis='P20 gains in confirmed conditions primarily reflect support eligibility rather than a stable advantage on commonly evaluable conditions',
        selections='All 8 risk/coverage/threshold scenarios, 50 starts, 6 classes, R/Greedy20/P20; 7200 frozen selected sets, no new selection',
        models=old.MODELS, evaluation_axes=AXES, sources=dict(A=study.SOURCES['A'], B=study.SOURCES['B'], C='state-membership agreement intersection'),
        fixed='Frozen labels, sample IDs, initial/selected sets, risk cells/quotas, support20, gap-10pp, 5 classifiers x 3 prediction seeds, historical D',
        diagnostic='Same-field known complement; state/control each >=20 distinct source images; three-seed mean recall gap <=-10pp in discovery and reference',
        primary='Exact confirmed-count difference = net eligibility transitions + net weak-gap transitions on pairwise common eligible states',
        secondary=dict(within_label='Restrict to conditions evaluable by all 3 methods and reference under the current label axis',
                       across_labels='Restrict to conditions evaluable by all 3 methods under all A/B/C discovery labels and all A/B/C D reference labels'),
        denominator='Common eligibility uses label support only, never selects on recall gaps or confirmation outcomes; common state IDs do not imply identical target membership or sample counts',
        interpretation='Gap component is arithmetic threshold crossing in different selected samples; not stronger ground-truth diagnostic power or a causal mechanism',
        reproduction='All relevant frozen diagonal/cross-source/consensus diagnostics, plus 100 original A per-start R comparisons',
        decision='Report all signs and common-eligibility differences, not just favorable combinations; if benefit mostly eligibility, describe improved evaluability with bounded diagnostic evidence',
        stop='One complete bounded decomposition and common-eligibility check; no support/gap threshold tuning, no prompt tuning, no search for favorable states',
        no_human_verification=True, no_new_vlm_calls=True, no_new_classifier_inference=True,
    )
    cc.freeze(OUT / 'protocol.json', protocol)
    paths = [Path(__file__), ROOT / 'scripts/report_label_support_diagnosis_20261112.py',
             ROOT / 'tests/test_label_support_diagnosis.py', Path(previous.__file__), Path(previous.prior.__file__),
             Path(cc.__file__), Path(cc.old.__file__),
             ROOT / 'artifacts/empirical_completion_20261108/mechanism_per_start.csv']
    for root, names in [
        (previous.OUT, ['selections.npz', 'selection_freeze.json', 'diagnostic.csv', 'prediction_freeze.json', 'final_outputs_freeze.json']),
        (CONSENSUS, ['diagnostic.csv', 'prediction_freeze.json', 'final_outputs_freeze.json'])]:
        paths += [root / name for name in names]
    paths += list(previous.PREV.glob('*_matrix.npz')) + list(CONSENSUS.glob('*_consensus.npz'))
    cc.freeze(OUT / 'input_freeze.json', dict(files={str(p): study.sha(p) for p in paths}))
    (OUT / 'paper_brief.md').write_text('保持风险配额受控覆盖选样主线；本轮区分历史诊断计数收益中的支持量作用与共同可评价状态上的变化，不改变论文中心主张。\n')
    (OUT / 'research_log.md').write_text(
        '# 执行记录\n\n2026-11-12：用户要求继续验证。原A已有类似支持量分解，作为复现控制；本轮补齐跨标签/共识、P20-vs-Greedy20和三方法共同可评价集合。\n\n'
        '先冻结假设、四项恒等分解、两种共同支持口径和停止条件。仍使用历史已知数据，不称为预注册或新数据确认。\n')
    cc.log(OUT, 'prepared')


def load():
    rows, risks, matrices = previous.contexts()
    raw, reference, hashes = {}, {}, {}
    for phase in rows:
        prediction, hh = cc.old.predictions(phase, rows[phase]); hashes.update(hh)
        offset = {r['sample_id']: i for i, r in enumerate(rows[phase])}
        for ci, c in enumerate(previous.CLASSES):
            a, b = [matrices[phase, source, c] for source in ['A', 'B']]
            states = risks[phase, 'A', c]['states']
            ac, bc = [previous.prior.value_known_mask(x, states) for x in [a, b]]
            xx, yy = dict(A=a, B=b, C=a & b), dict(A=ac, B=bc, C=ac & bc)
            with np.load(CONSENSUS / f'{phase}_{c}_consensus.npz', allow_pickle=False) as z:
                assert np.array_equal(z['positive'], xx['C']) and np.array_equal(z['negative'], yy['C'])
                assert np.array_equal(z['sample_ids'], risks[phase, 'A', c]['ids'])
            pos = [offset[s] for s in risks[phase, 'A', c]['ids']]
            correct = {model: (prediction[model][:, pos] == ci).mean(0) for model in old.MODELS}
            for source in VIEWS:
                raw[phase, source, c] = xx[source], yy[source], correct
                if phase == 'D':
                    for model in old.MODELS:
                        reference[source, c, model] = previous.prior.reference_stats(xx[source], yy[source], correct[model])
    cc.freeze(OUT / 'prediction_freeze.json', dict(files=hashes))
    return risks, raw, reference


def run():
    assert not (OUT / 'completion.json').exists(), 'Completed experiment cannot be overwritten'
    cc.verify_freeze(OUT / 'input_freeze.json')
    cc.verify_freeze(previous.OUT / 'prediction_freeze.json')
    risks, raw, reference = load()
    selections = np.load(previous.OUT / 'selections.npz', allow_pickle=False)
    # Existing count indexes are verification targets, not inputs to decomposition.
    expected = {}
    for root, consensus in [(previous.OUT, False), (CONSENSUS, True)]:
        for r in study.read(root / 'diagnostic.csv'):
            if r['method'] not in METHODS:
                continue
            k = tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'class_name', 'selection_seed', 'model', 'method'])
            ev, ref = ('C', 'C') if consensus else (r['evaluator'], r['reference'])
            expected[k + (ev, ref)] = tuple(int(r[f]) for f in ['eligible', 'proposed', 'reference_evaluable', 'confirmed'])
    diagnostic, decompositions = [], []
    fields = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    audited = 0
    for risk in ['A', 'B']:
        for coverage in ['A', 'B']:
            for mode in previous.MODES:
                for c in previous.CLASSES:
                    states = risks['B', 'A', c]['states']
                    groups = defaultdict(list)
                    for j, (ns, field, value) in enumerate(states):
                        groups[ns, field].append(j)
                    for seed in previous.SEEDS:
                        selected = {method: selections[previous.key(risk, coverage, mode, c, method, seed)] for method in METHODS}
                        local = {}
                        for source in VIEWS:
                            x, control, correct = raw['B', source, c]
                            for method in METHODS:
                                ix = selected[method]
                                for model in old.MODELS:
                                    local[source, method, model] = previous.prior.reference_stats(x[ix], control[ix], correct[model][ix])
                        for model in old.MODELS:
                            all_common = np.logical_and.reduce([local[s, m, model][3] for s in VIEWS for m in METHODS])
                            all_common &= np.logical_and.reduce([reference[s, c, model][3] for s in VIEWS])
                            # Consensus eligibility implies A/B eligibility because both arms are subsets.
                            c_common = np.logical_and.reduce([local['C', m, model][3] for m in METHODS]) & reference['C', c, model][3]
                            assert np.array_equal(c_common, all_common)
                            for evaluator, ref in AXES:
                                d = reference[ref, c, model]
                                same_label_common = np.logical_and.reduce([local[evaluator, m, model][3] for m in METHODS]) & d[3]
                                base = dict(risk_source=risk, coverage_source=coverage, threshold_mode=mode,
                                            class_name=c, selection_seed=seed, model=model, evaluator=evaluator, reference=ref)
                                confirmations = {}
                                for method in METHODS:
                                    n1, n0, gap, eligible, weak = local[evaluator, method, model]
                                    confirmed = weak & d[4]; confirmations[method] = confirmed
                                    counts = (int(eligible.sum()), int(weak.sum()), int((weak & d[3]).sum()), int(confirmed.sum()))
                                    ek = (risk, coverage, mode, c, str(seed), model, method, evaluator, ref)
                                    assert expected[ek] == counts, (ek, expected[ek], counts)
                                    diagnostic.append(dict(**base, method=method, eligible=counts[0], proposed=counts[1],
                                        reference_evaluable=counts[2], confirmed=counts[3],
                                        same_label_common_n=int(same_label_common.sum()),
                                        same_label_common_confirmed=int((confirmed & same_label_common).sum()),
                                        all_label_common_n=int(all_common.sum()),
                                        all_label_common_confirmed=int((confirmed & all_common).sum())))
                                for method, comparator in COMPARISONS:
                                    em, eb = [local[evaluator, m, model][3] for m in [method, comparator]]
                                    cm, cb = [confirmations[m] for m in [method, comparator]]
                                    counts, masks = decompose(em, eb, cm, cb)
                                    pair_common = em & eb & d[3]
                                    record = dict(**base, method=method, comparator=comparator, **counts,
                                        pair_common_n=int(pair_common.sum()),
                                        same_label_common_n=int(same_label_common.sum()),
                                        same_label_common_difference=int((cm & same_label_common).sum()) - int((cb & same_label_common).sum()),
                                        all_label_common_n=int(all_common.sum()),
                                        all_label_common_difference=int((cm & all_common).sum()) - int((cb & all_common).sum()))
                                    assert int((cm & pair_common).sum()) - int((cb & pair_common).sum()) == counts['common_gap_net']
                                    decompositions.append(record)
                                    for (ns, field), jj in groups.items():
                                        fk = (risk, coverage, mode, evaluator, ref, c, ns, field, method, comparator)
                                        fields[fk] += [int(masks[name][jj].sum()) for name in
                                                       ['new_from_eligibility', 'new_on_common_eligible', 'lost_from_eligibility', 'lost_on_common_eligible']]
                                    if seed == previous.SEEDS[0]:
                                        # Independent set accounting, no boolean partition helper.
                                        me, be, mc, bc = [set(np.flatnonzero(a).tolist()) for a in [em, eb, cm, cb]]
                                        new, lost = mc - bc, bc - mc
                                        wanted = [len(new - be), len(new & be), len(lost - me), len(lost & me)]
                                        assert wanted == [counts[k] for k in ['new_from_eligibility', 'new_on_common_eligible', 'lost_from_eligibility', 'lost_on_common_eligible']]
                                        audited += 1
                    cc.log(OUT, 'class_complete', risk=risk, coverage=coverage, threshold_mode=mode, class_name=c,
                           diagnostics=len(diagnostic), decompositions=len(decompositions))
    write('diagnostic.csv', diagnostic); write('mechanism.csv', decompositions)
    ff = []
    names = GROUP + ['class_name', 'namespace', 'field', 'method', 'comparator']
    for k, values in fields.items():
        ff.append(dict(zip(names, k), **{f + '_per_start': float(v / 50) for f, v in zip(
            ['new_from_eligibility', 'new_on_common_eligible', 'lost_from_eligibility', 'lost_on_common_eligible'], values)}))
    write('by_field.csv', ff)
    summarize(diagnostic, decompositions)
    reproduced = historical_audit(decompositions)
    independent_field_audit(decompositions, ff)
    cc.verify_freeze(OUT / 'input_freeze.json'); cc.verify_freeze(OUT / 'prediction_freeze.json')
    dump('validation.json', dict(status='pass', original_diagnostic_rows_reproduced=len(diagnostic),
        decomposition_identities=len(decompositions), independent_set_partition_rows=audited,
        original_A_mechanism_rows_reproduced=reproduced, field_sums_conserve_components=True,
        common_eligibility_masks_use_no_outcome_values=True, no_new_selection=True,
        semantic_accuracy_established=False))
    cc.log(OUT, 'analysis_complete', diagnostics=len(diagnostic), audited_partitions=audited)


def summarize(diagnostic, decomposition):
    totals, dg = defaultdict(lambda: defaultdict(int)), defaultdict(list)
    for r in decomposition:
        k = tuple(r[f] for f in GROUP + ['method', 'comparator', 'selection_seed'])
        for f in ['new_from_eligibility', 'new_on_common_eligible', 'lost_from_eligibility', 'lost_on_common_eligible',
                  'eligibility_net', 'common_gap_net', 'total_difference', 'pair_common_n',
                  'same_label_common_n', 'same_label_common_difference', 'all_label_common_n', 'all_label_common_difference']:
            totals[k][f] += r[f]
    per_start = [dict(zip(GROUP + ['method', 'comparator', 'selection_seed'], k), **values) for k, values in totals.items()]
    write('mechanism_per_start.csv', per_start)
    grouped = defaultdict(list)
    for r in per_start:
        grouped[tuple(r[f] for f in GROUP + ['method', 'comparator'])].append(r)
    summary = []
    for k, rr in grouped.items():
        assert len(rr) == 50
        row = dict(zip(GROUP + ['method', 'comparator'], k))
        for f in totals[next(iter(totals))]:
            row[f] = float(np.mean([r[f] for r in rr]))
        for f in ['total_difference', 'common_gap_net', 'same_label_common_difference', 'all_label_common_difference']:
            a = np.array([r[f] for r in rr])
            row[f + '_positive_starts'] = int((a > 0).sum())
            row[f + '_ties'] = int((a == 0).sum())
            row[f + '_negative_starts'] = int((a < 0).sum())
        summary.append(row)
    write('summary.csv', summary)
    for r in diagnostic:
        dg[tuple(r[f] for f in GROUP + ['method'])].append(r)
    ds = []
    for k, rr in dg.items():
        assert len(rr) == 1500
        ds.append(dict(zip(GROUP + ['method'], k), **{f + '_per_start': sum(r[f] for r in rr) / 50 for f in
            ['eligible', 'confirmed', 'same_label_common_n', 'same_label_common_confirmed', 'all_label_common_n', 'all_label_common_confirmed']}))
    write('method_summary.csv', ds)


def historical_audit(decomposition):
    grouped = defaultdict(lambda: defaultdict(int))
    for r in decomposition:
        if tuple(r[f] for f in GROUP) != ('A', 'A', 'original', 'A', 'A') or r['comparator'] != 'R':
            continue
        k = (r['selection_seed'], r['method'])
        grouped[k]['new_from_eligibility'] += r['new_from_eligibility']
        grouped[k]['new_despite_R_eligible'] += r['new_on_common_eligible']
        grouped[k]['lost_confirmed'] += r['lost_from_eligibility'] + r['lost_on_common_eligible']
    original = study.read(ROOT / 'artifacts/empirical_completion_20261108/mechanism_per_start.csv')
    for r in original:
        expected = grouped[int(r['selection_seed']), r['method']]
        for f, v in expected.items():
            assert v == int(r[f]), (r, f, v)
    assert len(grouped) == len(original) == 100
    return len(original)


def independent_field_audit(decomposition, fields):
    raw, by_field = defaultdict(lambda: np.zeros(4, dtype=int)), defaultdict(lambda: np.zeros(4))
    kk = GROUP + ['class_name', 'method', 'comparator']
    names = ['new_from_eligibility', 'new_on_common_eligible', 'lost_from_eligibility', 'lost_on_common_eligible']
    for r in decomposition:
        raw[tuple(r[f] for f in kk)] += [r[f] for f in names]
    for r in fields:
        by_field[tuple(r[f] for f in kk)] += [r[f + '_per_start'] for f in names]
    assert set(raw) == set(by_field)
    for k in raw:
        np.testing.assert_allclose(raw[k] / 50, by_field[k], atol=1e-10, rtol=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.stage]()
