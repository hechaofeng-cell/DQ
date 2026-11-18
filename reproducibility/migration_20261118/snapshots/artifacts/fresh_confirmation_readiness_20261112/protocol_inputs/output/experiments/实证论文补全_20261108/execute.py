"""Matched Greedy20 control and bounded historical diagnostic reanalysis.

Run with .venv/bin/python; original artifacts are strictly read-only.
"""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/home/hcf/project/QE')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/paper_claim_validation_20261106'
OUT = ROOT / 'artifacts/empirical_completion_20261108'
sys.path.insert(0, str(ROOT / 'output/experiments/论文关键结论验证_20261106'))
import common as old
cc = old.cc
CLASSES, MODELS, SEEDS = old.CLASSES, old.MODELS, old.SEEDS
METHODS = ['R', 'C20', 'Greedy20', 'P20', 'G']


def read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def dump(name, obj):
    cc.dump(OUT / name, obj)


def write(name, rows):
    cc.write(OUT / name, rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'date': '2026-11-08', 'status': 'exploratory execution on historical automatic labels; not corrected-label or fresh validation',
        'classes': CLASSES, 'models': MODELS, 'selection_seeds': SEEDS, 'methods': METHODS,
        'budgets': {'pool_per_class': 400, 'selected_per_class': 200, 'support_cap': 20, 'max_original_members_removed': 20},
        'Greedy20': {
            'start': 'exact stored R start; exact same risk cells and quotas as original P20',
            'skip': 'initial dual gate pass returns the identical initial set',
            'legal_moves': 'remove a still-selected original member; add a not-yet-added member outside original S0 in the same exact risk cell; added members never removed',
            'objective': 'maximize integer capped-support increment; tie maximize integer occurrence increment; tie outgoing sample ID then incoming sample ID ascending',
            'stopping': '20 moves or no lexicographically positive move; do not stop early at gate',
            'deployment': 'return candidate only if both frozen gates pass and both metrics do not decrease from S0; otherwise return S0 and preserve candidate metrics separately',
            'negative_moves': 'a positive support increment can decrease occurrence; final occurrence nondecrease remains mandatory',
        },
        'controls': 'R/C20/P20/G reuse exact frozen historical selections, not newly rerun solvers; no cross-era timing superiority claim',
        'smoke': 'first 10 of the original 50 starts for every class; audit quotas, membership, bounds, deterministic tie rule and deployment; all smoke starts retained in full results',
        'diagnosis': 'unchanged known same-field complement; both arms >=20 targets and >=20 original images; gap of three-seed mean correctness <= -10pp; historical D all-state cache',
        'mechanisms': 'new confirmed due to eligibility + new despite prior eligibility - lost confirmed; directly-scored vs other fields; actual selected error rate',
        'risk_adjustment': {
            'strata': 'frozen B Q1..Q5; D relative Q1..Q5 from equal_parts risk score and sample-ID tie rule within the fixed D200/class',
            'weights': 'full source/class candidate-pool state shares across its five strata, fixed before selected prediction reads; same weights for both arms and all methods, renormalized over retained strata',
            'eligibility': 'each retained stratum has >=5 targets per arm; retained totals >=20 per arm; retained state fraction >=0.80; all samples within a class have distinct source-image hashes',
            'no_adaptation': 'no merging strata, no alternate retention cutoff, no changing gap threshold after results',
            'interpretation': 'bounded descriptive within-risk comparison; relative Q are source specific; not causal, not error matching, not proof of independence from the risk definition',
        },
        'equal_error_auxiliary': {
            'unit': 'within each fixed class, model, prediction seed and selection start',
            'draw': 'same minimum incorrect-target count across all five methods; 50 deterministic matched random subsamples without replacement per method',
            'outcome': 'number of distinct wrong predicted classes; zero errors gives breadth zero for all methods',
            'interpretation': 'coarse confusion-category breadth conditional on equal error counts; changes formal quotas and is not root-cause diversity or causal isolation',
        },
        'annotation_review': {'random_per_class': 80, 'seed': 2026110803, 'total': 480,
            'targeted': 'the prior 60 targeted cases remain separate; overlaps tracked, never pooled into random accuracy estimates',
            'human_status': 'not started; two independent reviewers plus adjudication required; AI judgments never called human labels'},
        'fresh_validation': 'hold inference until semantic protocol and complete affected reannotation are resolved; unused means never analyzed, not just excluded from latest D',
        'stop': 'fixed inputs, all 50 paired starts; no outcome-based retuning; failed checks block affected claims; preserve unfavorable results',
    }
    cc.freeze(OUT / 'protocol.json', protocol)
    paths = [BASE / n for n in ['protocol.json', 'main_sets.npz', 'thresholds.json', 'B_manifest.csv', 'D_manifest.csv', 'reference_all_states.csv', 'diagnostic_by_selection.csv', 'selection_runs.csv']]
    paths += [BASE / f'{c}_{suffix}' for c in CLASSES for suffix in ['dictionary.json']]
    paths += [BASE / f'{phase}_{c}_matrix.npz' for c in CLASSES for phase in ['B', 'D']]
    paths += [HERE / 'execute.py', ROOT / 'configs/category_five_dimension_pilot_v1.json', Path(old.__file__), ROOT / 'scripts/conditional_coverage_common.py', ROOT / 'scripts/coverage_validation_common_20261018.py']
    cc.freeze(OUT / 'input_freeze.json', {'files': {str(p): sha(p) for p in paths}})
    dump('environment.json', {'python': sys.version, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__})
    (OUT / 'paper_brief.md').write_text('# 本轮执行边界\n\n主线保持风险配额受控的视觉状态覆盖选样与模型评价。新增同起点同20次替换上限的简单贪心，核查优化是否必要；补充诊断增益机制、风险分层后的差异和等错误数的粗粒度混淆类型。全部数值仍使用原自动标注及已分析的历史参考数据，不能当作语义修正版或全新验证。人工审查和真正新验证单独列完成状态。\n')
    (OUT / 'research_log.md').write_text('# 执行记录\n\n2026-11-08：用户授权按补全方案直接执行。protocol.json在本轮新选择产生前冻结；历史B/D结论已经看过，所以不是首次盲验证。原输入只读，新输出写入独立目录。\n')
    print('Frozen execution protocol and historical inputs.', flush=True)


def greedy(ctx, s0, threshold):
    X = ctx['matrix'].astype(np.int16)
    counts = X[s0].sum(0)
    before = cc.count_metrics(counts, 20)
    if cc.passes(before, threshold):
        return s0.copy(), s0.copy(), [], 'initial_pass'
    original = set(map(int, s0))
    order = sorted(range(len(X)), key=lambda i: str(ctx['ids'][i]))
    pairs = [(i, j) for i in order if i in original for j in order if j not in original and ctx['cell'][i] == ctx['cell'][j]]
    outgoing = np.array([a for a, b in pairs], dtype=int)
    incoming = np.array([b for a, b in pairs], dtype=int)
    delta = X[incoming] - X[outgoing]
    removed, added, trace = set(), set(), []
    reason = 'change_limit'
    for step in range(20):
        active = np.array([i not in removed and j not in added for i, j in pairs])
        ix = np.flatnonzero(active)
        if not len(ix):
            reason = 'no_legal_pair'; break
        candidate = counts[None, :] + delta[ix]
        gain_t = np.minimum(candidate, 20).sum(1) - np.minimum(counts, 20).sum()
        gain_a = (candidate > 0).sum(1) - (counts > 0).sum()
        best_t = gain_t.max()
        tied = np.flatnonzero(gain_t == best_t)
        best_a = gain_a[tied].max()
        if best_t < 0 or (best_t == 0 and best_a <= 0):
            reason = 'no_positive_move'; break
        # pairs are already in outgoing-ID, incoming-ID order.
        local = int(tied[np.flatnonzero(gain_a[tied] == best_a)[0]])
        p = int(ix[local]); a, b = pairs[p]
        removed.add(a); added.add(b); counts += delta[p]
        trace.append({'step': step + 1, 'outgoing_id': str(ctx['ids'][a]), 'incoming_id': str(ctx['ids'][b]),
                      'support_unit_gain': int(best_t), 'occurrence_state_gain': int(best_a)})
    candidate_set = np.array(sorted((original - removed) | added), dtype=int)
    final = cc.count_metrics(X[candidate_set].sum(0), 20)
    accept = cc.passes(final, threshold) and all(final[k] + 1e-12 >= before[k] for k in ['occurrence', 'support_completion'])
    return (candidate_set if accept else s0.copy()), candidate_set, trace, reason if accept else 'fallback_' + reason


def check(ctx, s0, selected, candidate, trace, threshold):
    for s in [selected, candidate]:
        cc.old.validate_selection(ctx, s, 200)
        assert len(set(map(int, s0)) - set(map(int, s))) <= 20
    assert len({r['outgoing_id'] for r in trace}) == len(trace)
    assert len({r['incoming_id'] for r in trace}) == len(trace)
    assert set(r['outgoing_id'] for r in trace) <= set(ctx['ids'][s0])
    assert not set(r['incoming_id'] for r in trace) & set(ctx['ids'][s0])
    before = cc.count_metrics(ctx['matrix'][s0].sum(0), 20)
    after = cc.count_metrics(ctx['matrix'][selected].sum(0), 20)
    assert all(after[k] + 1e-12 >= before[k] for k in ['occurrence', 'support_completion'])
    if not np.array_equal(selected, s0):
        assert cc.passes(after, threshold)


def select():
    cc.verify_freeze(OUT / 'input_freeze.json')
    assert not (OUT / 'greedy_sets.npz').exists()
    ctx = old.contexts(); thresholds = json.loads((BASE / 'thresholds.json').read_text())
    starts = np.load(BASE / 'main_sets.npz', allow_pickle=False)
    selected_sets, candidates, rows, traces = {}, {}, [], []
    for stage, seeds in [('smoke', SEEDS[:10]), ('remaining', SEEDS[10:])]:
        for c in CLASSES:
            ct = ctx[c]
            with np.load(BASE / f'B_{c}_matrix.npz') as z:
                assert np.array_equal(z['matrix'], ct['matrix']) and np.array_equal(z['cells'], ct['cell'])
            for seed in seeds:
                s0 = starts[old.key(c, 'R', seed)]
                tick = time.perf_counter()
                selected, candidate, trace, reason = greedy(ct, s0, thresholds[c])
                elapsed = time.perf_counter() - tick
                check(ct, s0, selected, candidate, trace, thresholds[c])
                if stage == 'smoke':
                    s2, c2, t2, r2 = greedy(ct, s0, thresholds[c])
                    assert np.array_equal(s2, selected) and np.array_equal(c2, candidate) and t2 == trace and r2 == reason
                k = old.key(c, 'Greedy20', seed)
                selected_sets[k] = selected; candidates[k] = candidate
                a = cc.count_metrics(ct['matrix'][selected].sum(0), 20)
                b = cc.count_metrics(ct['matrix'][candidate].sum(0), 20)
                rows.append({'class_name': c, 'method': 'Greedy20', 'selection_seed': seed, 'stop_reason': reason,
                             'seconds': elapsed, 'removed_targets': len(set(s0) - set(selected)), 'candidate_removed': len(trace),
                             'occurrence_pct': 100 * a['occurrence'], 'support_completion_pct': 100 * a['support_completion'],
                             'final_pass': cc.passes(a, thresholds[c]), 'candidate_pass': cc.passes(b, thresholds[c]),
                             'candidate_occurrence_pct': 100 * b['occurrence'], 'candidate_support_completion_pct': 100 * b['support_completion']})
                traces += [{'class_name': c, 'selection_seed': seed, **t} for t in trace]
            print(json.dumps({'phase': stage, 'class': c, 'completed': len(rows)}), flush=True)
        if stage == 'smoke':
            dump('smoke_audit.json', {'starts': 60, 'quota_and_change_bounds': 'pass', 'deterministic_full_traces': 'pass', 'deployment': 'pass'})
    np.savez_compressed(OUT / 'greedy_sets.npz', **selected_sets)
    np.savez_compressed(OUT / 'greedy_candidate_sets.npz', **candidates)
    write('greedy_selection_runs.csv', rows); write('greedy_move_trace.csv', traces)
    cc.freeze(OUT / 'selection_freeze.json', {'historical_predictions_previously_seen': True, 'prediction_values_read_in_this_execution': False,
        'files': {str(OUT / n): sha(OUT / n) for n in ['protocol.json', 'greedy_sets.npz', 'greedy_candidate_sets.npz', 'greedy_selection_runs.csv', 'greedy_move_trace.csv']}})
    print('All 300 Greedy20 selections frozen.', flush=True)


def mask_stats(X, control, correct):
    n1, n0 = X.sum(0), control.sum(0)
    r1 = np.divide(X.T @ correct, n1, out=np.zeros(X.shape[1]), where=n1 > 0)
    r0 = np.divide(control.T @ correct, n0, out=np.zeros(X.shape[1]), where=n0 > 0)
    return n1, n0, 100 * (r1 - r0)


def adjusted_structure(X, control, tiers, weights):
    a = np.array([X[tiers == q].sum(0) for q in range(1, 6)])
    b = np.array([control[tiers == q].sum(0) for q in range(1, 6)])
    keep = (a >= 5) & (b >= 5)
    retained1 = (a * keep).sum(0)
    eligible = (retained1 >= 20) & ((b * keep).sum(0) >= 20) & (retained1 >= .8 * X.sum(0))
    w = weights * keep
    w = np.divide(w, w.sum(0), out=np.zeros_like(w, dtype=float), where=w.sum(0) > 0)
    return a, b, w, eligible


def adjusted_gap(X, control, tiers, correct, structure):
    a, b, w, eligible = structure
    h1 = np.array([X[tiers == q].T @ correct[tiers == q] for q in range(1, 6)])
    h0 = np.array([control[tiers == q].T @ correct[tiers == q] for q in range(1, 6)])
    r1 = np.divide(h1, a, out=np.zeros_like(h1), where=a > 0)
    r0 = np.divide(h0, b, out=np.zeros_like(h0), where=b > 0)
    return 100 * ((r1 - r0) * w).sum(0)


def analyze():
    cc.verify_freeze(OUT / 'input_freeze.json'); cc.verify_freeze(OUT / 'selection_freeze.json')
    assert not (OUT / 'summary.json').exists()
    ctx = old.contexts(); ext, annotation_paths = old.external_contexts(ctx)
    orig = np.load(BASE / 'main_sets.npz', allow_pickle=False)
    gs = np.load(OUT / 'greedy_sets.npz', allow_pickle=False)
    sets = {**{k: orig[k] for k in orig.files}, **{k: gs[k] for k in gs.files}}
    brows, drows = read(BASE / 'B_manifest.csv'), read(BASE / 'D_manifest.csv')
    bp, bhash = cc.old.predictions('B', brows); dp, dhash = cc.old.predictions('D', drows)
    offsets = {p: {r['sample_id']: i for i, r in enumerate(rr)} for p, rr in [('B', brows), ('D', drows)]}
    from run_category_five_dimension_pilot import score_sample, five_tiers
    config = json.loads((ROOT / 'configs/category_five_dimension_pilot_v1.json').read_text())
    ref = {(r['class_name'], r['model'], int(r['state_index'])): r for r in read(BASE / 'reference_all_states.csv')}
    diagnostics, conditions, adjusted_conditions, coverage, adjust_groups, risk_rows = [], [], [], [], [], []
    confirmed, eligibles, adjustment_flags = {}, {}, {}
    correct_by, raw_by, all_state_fields = {}, {}, {}
    dlabel_paths = []
    direct = {'target_scale', 'occlusion_level', 'truncation_level', 'body_orientation'}
    for ci, c in enumerate(CLASSES):
        ct, dt = ctx[c], ext[c]
        assert len({r['image_sha256'] for r in dt['rows']}) == len(dt['rows']) == 200
        X, control = cc.condition_masks(ct); DX, DC = cc.condition_masks(dt)
        positions = {phase: np.array([offsets[phase][s] for s in data['ids']]) for phase, data in [('B', ct), ('D', dt)]}
        scores = []
        for r in dt['rows']:
            p = old.DROOT / f"annotations/local/{'car' if c == 'car' else 'old'}/parsed/{r['sample_id']}.json"
            j = json.loads(p.read_text()); assert j['image_id'] == r['sample_id']
            score = score_sample(float(r['bbox_area_ratio']), j['features'], c, 'equal_parts', config)
            scores.append(score['score_midpoint']); dlabel_paths.append(p)
        dq = np.zeros(200, dtype=int)
        for q, ix in enumerate(five_tiers(np.arange(200), scores, dt['ids']), 1): dq[ix] = q
        for i, r in enumerate(dt['rows']): risk_rows.append({'class_name': c, 'sample_id': r['sample_id'], 'Q': int(dq[i]), 'score_midpoint': scores[i]})
        bw = np.array([X[ct['tiers'] == q].sum(0) for q in range(1, 6)], dtype=float)
        dw = np.array([DX[dq == q].sum(0) for q in range(1, 6)], dtype=float)
        ds = adjusted_structure(DX, DC, dq, dw)
        dflags, dgaps, corr = {}, {}, {}
        for model in MODELS:
            correct_by[c, model] = (bp[model][:, positions['B']] == ci).mean(0)
            raw_by[c, model] = bp[model][:, positions['B']]
            corr[model] = correct_by[c, model]
            dc = (dp[model][:, positions['D']] == ci).mean(0)
            n1, n0, delta = mask_stats(DX, DC, dc)
            # Reproduce every cached D gap before introducing adjustment.
            for j in range(DX.shape[1]): assert abs(delta[j] - float(ref[c, model, j]['reference_delta_pp'])) < 1e-8
            dgaps[model] = adjusted_gap(DX, DC, dq, dc, ds)
            dflags[model] = ds[3] & (dgaps[model] <= -10 + 1e-12)
        scorer = direct | ({'body_visibility'} if c == 'car' else {'head_visibility', 'leg_visibility', 'head_orientation'})
        all_state_fields[c] = [ns == 'shared' and f in scorer for ns, f, v in ct['states']]
        for seed in SEEDS:
            for method in METHODS:
                s = sets[old.key(c, method, seed)]; A, B = X[s], control[s]
                n = A.sum(0); metrics = cc.count_metrics(n, 20)
                th = json.loads((BASE / 'thresholds.json').read_text())[c]
                coverage.append({'class_name': c, 'selection_seed': seed, 'method': method,
                                 'occurrence_pct': 100 * metrics['occurrence'], 'support_completion_pct': 100 * metrics['support_completion'],
                                 'supported_states': int((n >= 20).sum()), 'pass': cc.passes(metrics, th),
                                 'removed': len(set(sets[old.key(c, 'R', seed)]) - set(s))})
                eligible = (A.sum(0) >= 20) & (B.sum(0) >= 20)
                eligibles[c, seed, method] = set(np.flatnonzero(eligible))
                bs = adjusted_structure(A, B, ct['tiers'][s], bw)
                adjustment_flags[c, seed, method] = bs[3]
                for model in MODELS:
                    n1, n0, delta = mask_stats(A, B, corr[model][s])
                    proposed = np.flatnonzero(eligible & (delta <= -10 + 1e-12))
                    order = sorted(proposed, key=lambda j: (float(delta[j]), tuple(ct['states'][j])))
                    confirmed_set = {int(j) for j in proposed if ref[c, model, int(j)]['reference_weak'] == 'True'}
                    confirmed[c, model, seed, method] = confirmed_set
                    evaluable = sum(ref[c, model, int(j)]['reference_evaluable'] == 'True' for j in proposed)
                    top = set(order[:3])
                    diagnostics.append({'class_name': c, 'model': model, 'selection_seed': seed, 'method': method,
                                        'eligible': int(eligible.sum()), 'proposed': len(proposed), 'evaluable': evaluable,
                                        'confirmed': len(confirmed_set), 'top3_confirmed': len(top & confirmed_set),
                                        'error_pct': 100 * (1 - corr[model][s].mean())})
                    for rank, j in enumerate(order, 1):
                        ns, f, v = ct['states'][j]
                        conditions.append({'class_name': c, 'model': model, 'selection_seed': seed, 'method': method,
                                           'state_index': int(j), 'namespace': ns, 'field': f, 'value': v, 'rank': rank,
                                           'state_targets': int(n1[j]), 'control_targets': int(n0[j]), 'delta_pp': float(delta[j]),
                                           'reference_evaluable': ref[c, model, int(j)]['reference_evaluable'],
                                           'confirmed': int(j) in confirmed_set})
                    agap = adjusted_gap(A, B, ct['tiers'][s], corr[model][s], bs)
                    aweak = bs[3] & (agap <= -10 + 1e-12)
                    aconfirmed = aweak & dflags[model]
                    adjust_groups.append({'class_name': c, 'model': model, 'selection_seed': seed, 'method': method,
                                          'eligible': int(bs[3].sum()), 'proposed': int(aweak.sum()),
                                          'reference_evaluable': int((aweak & ds[3]).sum()), 'confirmed': int(aconfirmed.sum())})
                    for j in np.flatnonzero(aweak):
                        ns, f, v = ct['states'][j]
                        adjusted_conditions.append({'class_name': c, 'model': model, 'selection_seed': seed, 'method': method,
                                                    'state_index': int(j), 'namespace': ns, 'field': f, 'value': v,
                                                    'delta_pp': float(agap[j]), 'D_delta_pp': float(dgaps[model][j]),
                                                    'D_eligible': bool(ds[3][j]), 'confirmed': bool(aconfirmed[j]),
                                                    'direct_risk_field': all_state_fields[c][j]})
        print(json.dumps({'phase': 'analysis', 'class': c, 'diagnostic_groups': len(diagnostics)}), flush=True)
    write('coverage_runs.csv', coverage); write('diagnostic_runs.csv', diagnostics); write('diagnostic_conditions.csv', conditions)
    write('risk_adjusted_groups.csv', adjust_groups); write('risk_adjusted_conditions.csv', adjusted_conditions); write('D_relative_risk_strata.csv', risk_rows)
    dump('prediction_and_D_label_hashes.json', {'B_predictions': bhash, 'D_predictions': dhash, 'D_native_labels': {str(p): sha(p) for p in dlabel_paths}})
    # Summaries aggregate counts over six classes x five models, then mean over 50 starts.
    byseed = defaultdict(list)
    for r in diagnostics: byseed[r['selection_seed'], r['method']].append(r)
    adjseed = defaultdict(list)
    for r in adjust_groups: adjseed[r['selection_seed'], r['method']].append(r)
    historical = {(int(r['selection_seed']), r['method']): r for r in read(BASE / 'diagnostic_by_selection.csv') if r['scope'] == 'all'}
    totals, mechanisms = [], []
    for seed in SEEDS:
        for method in METHODS:
            rr = byseed[seed, method]; aa = adjseed[seed, method]
            cov = [r for r in coverage if r['selection_seed'] == seed and r['method'] == method]
            r = {'selection_seed': seed, 'method': method, **{k: sum(x[k] for x in rr) for k in ['proposed', 'evaluable', 'confirmed', 'top3_confirmed']},
                 'error_pct': float(np.mean([x['error_pct'] for x in rr])), 'eligible_per_class': float(np.mean([x['eligible'] for x in rr])),
                 'unique_confirmed_states': sum(len(set().union(*(confirmed[c, model, seed, method] for model in MODELS))) for c in CLASSES),
                 'occurrence_pct': float(np.mean([x['occurrence_pct'] for x in cov])),
                 'support_completion_pct': float(np.mean([x['support_completion_pct'] for x in cov])),
                 'pass_count': sum(x['pass'] for x in cov), 'mean_removed': float(np.mean([x['removed'] for x in cov])),
                 **{'adjusted_' + k: sum(x[k] for x in aa) for k in ['eligible', 'proposed', 'reference_evaluable', 'confirmed']}}
            if method != 'Greedy20':
                for k in ['proposed', 'evaluable', 'confirmed']: assert r[k] == int(historical[seed, method][k]), (seed, method, k)
            totals.append(r)
        for comparator in ['Greedy20', 'P20']:
            r = {'selection_seed': seed, 'method': comparator, **{k: 0 for k in ['new_from_eligibility', 'new_despite_R_eligible', 'lost_confirmed', 'new_other_fields', 'lost_other_fields', 'R_common_eligible_confirmed', 'method_common_eligible_confirmed']}}
            for c in CLASSES:
                for model in MODELS:
                    a = confirmed[c, model, seed, 'R']; b = confirmed[c, model, seed, comparator]
                    ea, eb = eligibles[c, seed, 'R'], eligibles[c, seed, comparator]
                    new, lost = b - a, a - b
                    r['new_from_eligibility'] += len(new - ea); r['new_despite_R_eligible'] += len(new & ea)
                    r['lost_confirmed'] += len(lost)
                    r['new_other_fields'] += sum(not all_state_fields[c][j] for j in new)
                    r['lost_other_fields'] += sum(not all_state_fields[c][j] for j in lost)
                    r['R_common_eligible_confirmed'] += len(a & ea & eb)
                    r['method_common_eligible_confirmed'] += len(b & ea & eb)
            original_totals = {t['method']: t for t in totals if t['selection_seed'] == seed}
            assert r['new_from_eligibility'] + r['new_despite_R_eligible'] - r['lost_confirmed'] == original_totals[comparator]['confirmed'] - original_totals['R']['confirmed']
            mechanisms.append(r)
    write('per_start_totals.csv', totals); write('mechanism_per_start.csv', mechanisms)
    # Error-count matching is a separate auxiliary subsampling experiment.
    eqrows = []
    for seed in SEEDS:
        acc = defaultdict(list)
        for ci, c in enumerate(CLASSES):
            for mi, model in enumerate(MODELS):
                p = raw_by[c, model]
                for pi in range(p.shape[0]):
                    errors = {m: sets[old.key(c, m, seed)][p[pi, sets[old.key(c, m, seed)]] != ci] for m in METHODS}
                    size = min(map(len, errors.values()))
                    for repeat in range(50):
                        # The same uniform ordering over candidate indices is shared across methods.
                        order = np.random.default_rng(np.random.SeedSequence([2026110805, seed, ci, mi, pi, repeat])).permutation(400)
                        rank = np.empty(400, dtype=int); rank[order] = np.arange(400)
                        for m, indices in errors.items():
                            pick = indices[np.argsort(rank[indices])[:size]]
                            acc[m].append(len(set(p[pi, pick].tolist())))
                    for m in METHODS:
                        eqrows.append({'selection_seed': seed, 'class_name': c, 'model': model, 'prediction_seed': old.PSEEDS[pi],
                                       'method': m, 'available_errors': len(errors[m]), 'matched_errors': size,
                                       'mean_confusion_types': float(np.mean(acc[m][-50:]))})
        if (seed - SEEDS[0] + 1) % 10 == 0: print(f'Equal-error starts completed: {seed - SEEDS[0] + 1}/50', flush=True)
    write('equal_error_confusion_runs.csv', eqrows)
    means = {}
    for m in METHODS:
        tt = [r for r in totals if r['method'] == m]
        mm = {k: float(np.mean([r[k] for r in tt])) for k in tt[0] if k not in ['selection_seed', 'method']}
        mm['confirmation_pct_pooled'] = 100 * mm['confirmed'] / mm['evaluable'] if mm['evaluable'] else None
        mm['evaluable_pct_pooled'] = 100 * mm['evaluable'] / mm['proposed'] if mm['proposed'] else None
        mm['dual_pass_total'] = int(sum(r['pass_count'] for r in tt))
        mm['equal_error_confusion_types'] = float(np.mean([r['mean_confusion_types'] for r in eqrows if r['method'] == m]))
        means[m] = mm
    mechanism_means = {m: {k: float(np.mean([r[k] for r in mechanisms if r['method'] == m])) for k in mechanisms[0] if k not in ['method', 'selection_seed']} for m in ['Greedy20', 'P20']}
    paired = []
    for scope, field in [('raw', 'confirmed'), ('adjusted', 'adjusted_confirmed')]:
        for a, b in [('P20', 'Greedy20'), ('Greedy20', 'R'), ('P20', 'R')]:
            lookup = {(r['selection_seed'], r['method']): r for r in totals}
            differences = [lookup[s, a][field] - lookup[s, b][field] for s in SEEDS]
            paired.append({'scope': scope, 'method_A': a, 'method_B': b, 'mean_A_minus_B': float(np.mean(differences)),
                           'SD_across_fixed_starts': float(np.std(differences, ddof=1)), 'positive_starts': sum(d > 0 for d in differences),
                           'tie_starts': sum(d == 0 for d in differences), 'negative_starts': sum(d < 0 for d in differences)})
    write('paired_comparisons.csv', paired)
    class_summaries = []
    for c in CLASSES:
        for m in METHODS:
            ccov = [r for r in coverage if r['class_name'] == c and r['method'] == m]
            dd = [r for r in diagnostics if r['class_name'] == c and r['method'] == m]
            class_summaries.append({'class_name': c, 'method': m, 'pass_count': sum(r['pass'] for r in ccov),
                                    'occurrence_pct': float(np.mean([r['occurrence_pct'] for r in ccov])),
                                    'support_completion_pct': float(np.mean([r['support_completion_pct'] for r in ccov])),
                                    'confirmed_per_start_all5models': sum(r['confirmed'] for r in dd) / 50})
    write('by_class.csv', class_summaries)
    dump('summary.json', {'scope': 'Historical, original automatic labels; new matched greedy selections and descriptive reanalysis; human correction and fresh confirmation pending',
                         'aggregation': 'counts sum over six classes and five models per start, then average 50 starts; percentages macro over six classes except explicitly pooled confirmation fractions',
                         'methods': means, 'mechanisms': mechanism_means, 'paired': paired})
    dump('execution_checks.json', {'all_300_greedy_selections_valid': True, 'smoke_60_deterministic': True,
                                  'all_200_old_method_start_diagnostic_totals_reproduced': True,
                                  'all_3585_D_state_model_gaps_reproduced': True,
                                  'all_100_mechanism_identities': True,
                                  'new_corrected_labels': False, 'new_model_inference': False, 'fresh_validation': False})
    print(json.dumps(means, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'select', 'analyze'])
    globals()[p.parse_args().phase]()
