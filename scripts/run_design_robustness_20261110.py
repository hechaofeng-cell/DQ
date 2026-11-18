#!/usr/bin/env python
"""设计稳健性补全_20261110：标签翻转扰动、L/配额敏感性轴、固定资格图像簇 bootstrap。

三组补全实验，全部沿用冻结的 20261106 实验代码与输入（严格只读）：
1. axes    —— P20 的 L∈{10,40} 与均分/池比例/风险加权配额结构。
2. perturb —— 字段级标签翻转扰动：含共享风险字段，经风险分数→Q层→精细单元→配额全链路传导；
              比较原冻结阈值与噪声池预测前重标定阈值；按干净标签参照检查达标。
3. boot    —— 固定资格的图像簇重采样：资格以点估计冻结，只对差值重采样，
              并同时计算旧式"逐复本重开门"版本，直接对照不连续性修复效果。

Run with .venv/bin/python; original artifacts are read-only.
"""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import copy
import json
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/design_robustness_20261110_v2'
BASE = ROOT / 'artifacts/paper_claim_validation_20261106'
CONFIG = json.loads((ROOT / 'configs/category_five_dimension_pilot_v1.json').read_text())
sys.path.insert(0, str(ROOT / 'output/experiments/论文关键结论验证_20261106'))
import common as frozen  # 冻结的 20261106 实现，只读
cc = frozen.cc
sys.path.insert(0, str(ROOT / 'scripts'))
from run_category_five_dimension_pilot import score_sample, five_tiers

CLASSES = frozen.CLASSES
SEEDS = frozen.SEEDS                       # 2026110600–2026110649
MODELS = frozen.MODELS
L_AXIS = [10, 40]                          # 主设置 L=20 已在主线
QUOTA_SCHEMES = ['equal', 'pool_proportional', 'risk_weighted']
RATES = [0.05, 0.10, 0.20]                 # 字段级标签错误率
THRESHOLD_MODES = ['frozen_clean', 'recalibrated_corrupt']
PERT_SEEDS = SEEDS[:8]                     # 扰动轴用前 8 个选样起点
PERT_REPS = 10
BOOT = 1000
BOOT_SEED = 2026111001
TOL = cc.TOL
INPUTS = [BASE / n for n in ['main_sets.npz', 'thresholds.json', 'selection_runs.csv',
                             'reference_all_states.csv', 'B_manifest.csv', 'D_manifest.csv',
                             'discovery_groups.csv', 'discovered_conditions.csv', 'diagnostic_runs.csv']]
INPUTS += [frozen.SPEC / 'risk_scores_and_tiers.csv', ROOT / 'configs/category_five_dimension_pilot_v1.json',
           Path(frozen.__file__), Path(cc.__file__), Path(cc.old.__file__),
           ROOT / 'scripts/run_category_five_dimension_pilot.py', Path(__file__),
           ROOT / 'scripts/report_design_robustness_20261110.py', ROOT / 'tests/test_design_robustness.py']
NATIVE_SCHEMA_ROOT = ROOT / 'artifacts/sixclass_independent_confirmation_20261004/configs'
UNKNOWN = {'unknown', 'not_observable', 'not_applicable', 'missing'}


def dump(name, obj):
    cc.dump(OUT / name, obj)


def write(name, rows, fields=None):
    cc.write(OUT / name, rows, fields)


def read(path):
    return cc.read(path)


def log(stage, **kw):
    obj = dict(time=cc.old.now(), stage=stage, **kw)
    print(json.dumps(obj, ensure_ascii=False), flush=True)
    with (OUT / 'research_log.md').open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def field_groups(states):
    groups = defaultdict(list)
    for j, (ns, field, value) in enumerate(states):
        groups[(ns, field)].append(j)
    return list(groups.items())


def scoring_fields(class_name):
    return ['occlusion_level', 'truncation_level', 'body_orientation'] + (
        ['body_visibility'] if class_name == 'car' else
        ['head_visibility', 'leg_visibility', 'head_orientation'])


def native_features():
    with (frozen.BASE / 'native_features.jsonl').open() as handle:
        return {r['sample_id']: r['features'] for r in map(json.loads, handle)}


def coverage_schema(class_name, states):
    """Keep annotation provenance: four expanded dictionaries do not replace native risk labels."""
    domain = 'car' if class_name == 'car' else 'old'
    native = json.loads((NATIVE_SCHEMA_ROOT / f'{domain}_annotation_schema.json').read_text())
    native = {f['feature_id']: f for f in native['universal_features'] + native.get('category_specific_features', [])}
    specific = {} if class_name in ['dog', 'cat'] else {
        f['feature_id']: f for f in json.loads(
            (frozen.SPEC / f'frozen_dictionaries/{class_name}.json').read_text())['features']}
    metadata = {}
    for key, js in field_groups(states):
        ns, field = key
        if field == 'target_scale':
            metadata[key] = dict(value_type='enum', exclusive_values=[])
            continue
        f = specific[field] if ns == 'specific' and class_name not in ['dog', 'cat'] else native[field]
        values = f.get('possible_values', [v['value_id'] for v in f.get('values', [])])
        assert {states[j][2] for j in js} == set(values) - UNKNOWN, (class_name, key)
        metadata[key] = f
    return metadata


def rescore(class_name, rows, features):
    """Score native risk records, never substitute a coverage-only field with the same name."""
    mids, widths = [], []
    for i in range(len(rows)):
        s = score_sample(float(rows[i]['bbox_area_ratio']),
                         features[i],
                         class_name, 'equal_parts', CONFIG)
        mids.append(s['score_midpoint'])
        widths.append(s['score_upper'] - s['score_lower'])
    mids = np.asarray(mids)
    widths = np.asarray(widths)
    tiers = np.zeros(len(rows), int)
    for q, ix in enumerate(five_tiers(np.arange(len(rows)), mids,
                                      np.array([r['sample_id'] for r in rows])), 1):
        tiers[ix] = q
    rawkeys = [(int(tiers[i]), round(float(mids[i]) * 1e8), round(float(widths[i]) * 1e8))
               for i in range(len(rows))]
    keys = sorted(set(rawkeys))
    lookup = {k: n for n, k in enumerate(keys)}
    return mids, tiers, keys, np.array([lookup[k] for k in rawkeys])


def tier_allocation(tiers, scheme, budget=200):
    counts = np.bincount(tiers, minlength=6)[1:]
    weights = np.ones(5) if scheme == 'equal' else counts.astype(float)
    if scheme == 'risk_weighted':
        weights *= np.arange(1, 6)
    elif scheme not in ['equal', 'pool_proportional']:
        raise ValueError(scheme)
    exact = budget * weights / weights.sum()
    result = np.floor(exact).astype(int)
    for j in sorted(range(5), key=lambda j: (-(exact[j] - result[j]), j))[:budget - result.sum()]:
        result[j] += 1
    assert result.sum() == budget and np.all(result <= counts)
    return result.tolist()


def cell_change_fraction(old_keys, old_cell, new_keys, new_cell):
    # Integer cell IDs are independently re-enumerated and cannot be compared.
    return float(np.mean([old_keys[a] != new_keys[b] for a, b in zip(old_cell, new_cell)]))


def quotas_for(keys, cell, tier_alloc):
    """按 Q 层预算 tier_alloc（Q1..Q5）分配精细单元配额；层内按单元候选数比例、最大余数法、同余按单元键序（复刻旧实现的分配规则）。"""
    counts = np.bincount(cell, minlength=len(keys))
    result = np.zeros(len(counts), int)
    for q in range(1, 6):
        ix = [j for j, k in enumerate(keys) if k[0] == q]
        total = int(counts[ix].sum())
        assert total, f'tier {q} has no candidates'
        numer = {j: counts[j] * tier_alloc[q - 1] for j in ix}
        for j in ix:
            result[j] = numer[j] // total
        rem = tier_alloc[q - 1] - int(result[ix].sum())
        for j in sorted(ix, key=lambda j: (-(numer[j] % total), keys[j]))[:rem]:
            result[j] += 1
    assert result.sum() == sum(tier_alloc) and np.all(result <= counts)
    return result


def initial_for(cell, quota, seed, ci):
    """复刻 cc.initial_selection 的随机流（SeedSequence([seed, ci])，单元内无放回），仅替换配额。"""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(ci)]))
    parts = [rng.choice(np.flatnonzero(cell == c), int(n), replace=False)
             for c, n in enumerate(quota) if n]
    return np.concatenate(parts).astype(int)


def validate_for(cell, quota, tiers, selected, tier_alloc):
    selected = np.asarray(selected)
    assert len(selected) == len(set(selected.tolist())) == sum(tier_alloc)
    assert np.array_equal(np.bincount(cell[selected], minlength=len(quota)), quota)
    assert np.bincount(tiers[selected], minlength=6)[1:].tolist() == list(tier_alloc)


def priority_for(ctx, s0, threshold, limit, quota):
    """复刻 frozen.priority（P20 两阶段 MILP），仅替换单元配额向量 q。"""
    X = ctx['matrix']
    n0 = X[s0].sum(0)
    before = cc.count_metrics(n0, 20)
    if cc.passes(before, threshold):
        return s0.copy(), dict(stop_reason='initial_pass', triggered=False, optimal_certified=False,
                               stage1_status=-1, stage2_status=-1)
    if limit == 0:
        return s0.copy(), dict(stop_reason='zero_change_limit', triggered=True, optimal_certified=False,
                               stage1_status=-1, stage2_status=-1)
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csr_matrix, hstack, vstack, eye, diags
    N, S = X.shape
    q = quota
    Xt = csr_matrix(X.T.astype(float))
    Z = csr_matrix((S, S))
    Aq = csr_matrix((np.ones(N), (ctx['cell'], np.arange(N))), shape=(len(q), N))
    retain = np.zeros(N + 2 * S)
    retain[s0] = 1
    sumy = np.r_[np.zeros(N), np.ones(S), np.zeros(S)]
    sumz = np.r_[np.zeros(N + S), np.ones(S)]
    minocc = max(int((n0 > 0).sum()), int(np.ceil(threshold['occurrence'] * S - 1e-9)))
    minunits = max(int(np.minimum(n0, 20).sum()), int(np.ceil(threshold['support_completion'] * S * 20 - 1e-9)))
    A = vstack([hstack([Aq, csr_matrix((len(q), 2 * S))]), hstack([Xt, -eye(S), Z]),
                hstack([Xt, Z, -eye(S)]), hstack([Xt, Z, -diags(X.sum(0).astype(float))]),
                csr_matrix(retain[None]), csr_matrix(sumy[None]), csr_matrix(sumz[None])], format='csr')
    lo = np.r_[q, np.zeros(S), np.zeros(S), np.full(S, -np.inf), 200 - limit, minunits, minocc]
    hi = np.r_[q, np.full(S, np.inf), np.full(S, np.inf), np.zeros(S), 200, np.inf, np.inf]
    bounds = Bounds(np.zeros(N + 2 * S), np.r_[np.ones(N), np.full(S, 20), np.ones(S)])
    integer = np.r_[np.ones(N), np.zeros(S), np.ones(S)]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
        a = milp(-sumy, integrality=integer, bounds=bounds,
                 constraints=LinearConstraint(A, lo, hi), options=frozen.SOLVER)
        if a.x is None:
            return s0.copy(), dict(stop_reason='infeasible' if a.status == 2 else 'no_feasible_incumbent',
                                   triggered=True, optimal_certified=False,
                                   stage1_status=int(a.status), stage2_status=-1)
        s1 = np.flatnonzero(a.x[:N] > .5)
        units = int(np.minimum(X[s1].sum(0), 20).sum())
        lo2 = lo.copy()
        lo2[-2] = units
        b = milp(-sumz, integrality=integer, bounds=bounds,
                 constraints=LinearConstraint(A, lo2, hi), options=frozen.SOLVER)
    selected = np.flatnonzero(b.x[:N] > .5) if b.x is not None else s1
    metrics = cc.count_metrics(X[selected].sum(0), 20)
    assert cc.passes(metrics, threshold) and metrics['occurrence'] + 1e-12 >= before['occurrence'] \
        and metrics['support_completion'] + 1e-12 >= before['support_completion']
    assert len(set(s0) - set(selected)) <= limit
    return selected, dict(stop_reason='lexicographic_optimal' if a.status == b.status == 0 else 'feasible_solver_limit',
                           triggered=True, optimal_certified=bool(a.status == b.status == 0),
                           stage1_status=int(a.status), stage2_status=int(b.status))


def corrupt_matrix(X, states, groups, metadata, rng, rate):
    """Flip known fields using schema-valid alternatives, preserving unknown and geometry.

    Enum replacement introduces both FP and FN. Set errors add/drop one value,
    or replace a singleton; exclusive values can never coexist with other values.
    """
    X = X.copy()
    for i in range(len(X)):
        for key, js in groups:
            ns, field = key
            if field == 'target_scale':
                continue
            if rng.random() >= rate:
                continue
            active = [j for j in js if X[i, j]]
            if not active:
                continue
            inactive = [j for j in js if not X[i, j]]
            if metadata[key]['value_type'] == 'enum':
                assert len(active) == 1
                alternatives = [{j} for j in inactive]
            else:
                exclusive = {j for j in js if states[j][2] in metadata[key].get('exclusive_values', [])}
                current = set(active)
                alternatives = ([current | {j} for j in inactive] +
                                [current - {j} for j in active] +
                                ([{j} for j in inactive] if len(active) == 1 else []))
                alternatives = [v for v in alternatives if v and (len(v) == 1 or not v & exclusive)]
            if alternatives:
                new = alternatives[int(rng.integers(len(alternatives)))]
                X[i, js] = [j in new for j in js]
    return X


def perturb_inputs(class_name, ctx, features, rng, rate):
    """One error per annotation field; shared/native-derived coverage labels stay synchronized."""
    states, clean = ctx['states'], ctx['matrix']
    groups = field_groups(states)
    X = corrupt_matrix(clean, states, groups, coverage_schema(class_name, states), rng, rate)
    risk = copy.deepcopy(features)
    linked = {field: js for (ns, field), js in groups
              if ns == 'shared' or class_name in ['dog', 'cat']}
    for field in scoring_fields(class_name):
        if field in linked:
            js = linked[field]
            for i, f in enumerate(risk):
                # Preserve exact native unknown token, not an invented known negative.
                if features[i][field] in UNKNOWN:
                    assert not X[i, js].any()
                    continue
                values = [states[j][2] for j in js if X[i, j]]
                assert len(values) == 1
                f[field] = values[0]
        else:
            map_name = 'part_visibility' if field == 'head_visibility' else field
            values = list(CONFIG['maps'][map_name])
            for i, f in enumerate(risk):
                v = f[field]
                if v in UNKNOWN or rng.random() >= rate:
                    continue
                f[field] = str(rng.choice([w for w in values if w != v]))
    return X, risk


def clean_rescore_audit(ctx):
    native = native_features()
    spec = {r['sample_id']: r for r in read(frozen.SPEC / 'risk_scores_and_tiers.csv')}
    result = []
    for c, ct in ctx.items():
        coverage_schema(c, ct['states'])
        features = [native[s] for s in ct['ids']]
        mids, tiers, keys, cell = rescore(c, ct['rows'], features)
        worst = max(abs(mids[i] - float(spec[s]['score_midpoint'])) for i, s in enumerate(ct['ids']))
        assert worst == 0 and keys == ct['keys']
        assert np.array_equal(tiers, ct['tiers']) and np.array_equal(cell, ct['cell'])
        witness, _ = cc.baseline(ct, 200, 'G', 20)
        metrics = cc.count_metrics(ct['matrix'][witness].sum(0), 20)
        frozen_thresholds = json.loads((BASE / 'thresholds.json').read_text())[c]
        assert all(abs(.95 * metrics[k] - frozen_thresholds[k]) < 1e-12 for k in frozen_thresholds)
        X0, f0 = perturb_inputs(c, ct, features, np.random.default_rng(123), 0)
        assert np.array_equal(X0, ct['matrix']) and f0 == features
        result.append(dict(class_name=c, max_midpoint_error=float(worst), cells_equal=True,
                           clean_thresholds_equal=True, zero_noise_identity=True))
    return result


def input_paths():
    paths = INPUTS + [frozen.BASE / n for n in ['manifest.csv', 'native_features.jsonl', 'state_dictionary.json']]
    paths += [NATIVE_SCHEMA_ROOT / f'{d}_annotation_schema.json' for d in ['old', 'car']]
    for c in CLASSES[2:]:
        paths += [frozen.SPEC / f'{c}_state_matrix.npz', frozen.SPEC / f'{c}_state_dictionary.json',
                  frozen.SPEC / f'frozen_dictionaries/{c}.json']
    _, ext_paths = frozen.external_contexts(frozen.contexts())
    paths += ext_paths + [frozen.EXT / 'external_manifest.csv', frozen.DROOT / 'manifests/candidate_pool.csv']
    # Freeze cached predictions before any values are opened by the bootstrap phase.
    paths += [ROOT / f'artifacts/recent_five_models_20261009/runs/{m}/{seed}/{phase}_outputs.npz'
              for m in MODELS for seed in cc.old.SEEDS for phase in ['B', 'D']]
    return sorted(set(paths))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Use a new directory; do not overwrite a frozen run.'
    clean_audit = clean_rescore_audit(frozen.contexts())
    cc.freeze(OUT / 'protocol.json', {
        'date': '2026-11-10', 'status': 'robustness extension on frozen historical inputs; exploratory, not a fresh validation',
        'bases': 'frozen 20261106 P20 implementation and artifacts; thresholds from the frozen prediction-free calibration',
        'axes_phase': {'change_limit_axis': L_AXIS, 'main_setting': 'L=20, equal 40x5 quota (mainline)',
                       'quota_schemes': QUOTA_SCHEMES,
                       'risk_weights': [1, 2, 3, 4, 5],
                       'quota_axis_note': 'same frozen thresholds and seeds; largest remainder; pool proportional equals equal because each Q has 80 candidates',
                       'reproduction_check': 'frozen.priority must reproduce mainline P20 sets and records; priority_for with equal quota must equal frozen.priority'},
        'perturb_phase': {'rates': RATES, 'repeats': PERT_REPS, 'seeds': PERT_SEEDS,
                          'error_model': 'per known field with prob rate: enum replacement; schema-valid set add/drop or singleton replacement; unknown unchanged; geometric target_scale unchanged',
                          'provenance': 'native risk records retained; shared fields and dog/cat native-derived specific fields synchronized with coverage; four expanded specific annotations remain independent, including same-name fields',
                          'propagation': 'joint corrupted coverage and native risk -> score_sample(equal_parts) -> Q tiers -> fine risk cells -> quotas -> initial selection -> P20',
                          'thresholds': THRESHOLD_MODES,
                          'threshold_modes': 'frozen_clean=deployment against trusted thresholds; recalibrated_corrupt=full pre-prediction propagation, 0.95 times same-pool G witness, frozen before P20 in each noise world',
                          'views': 'view=metrics on corrupted matrix; clean=metrics on frozen clean labels, not semantic ground truth'},
        'boot_phase': {'replicates': BOOT, 'seed': BOOT_SEED, 'unit': 'original image (one distinct image per target in B and D pools)',
                       'fix': 'eligibility fixed at point estimate; resample recall contrasts for all eligible states, with B proposals frozen for D count summaries',
                       'comparison': 'the old per-replicate re-gated version is computed alongside to display the discontinuity fix',
                       'pairing': 'full B/D pool image weights drawn once per class and shared across methods, models and seeds; selected subsets receive their original image weights',
                       'interval_scope': 'approximate pointwise percentile intervals conditional on fixed historical pools, labels and selections; no multiplicity or post-selection calibration',
                       'count_scope': 'thresholded counts remain discrete; their percentiles are descriptive, not calibrated confidence intervals',
                       'legacy_gate': 'both weighted target counts and distinct resampled original-image counts must reach 20'},
        'read_only_inputs': True, 'new_model_inference': False, 'new_training': False})
    paths = input_paths()
    cc.freeze(OUT / 'input_freeze.json', {'files': {str(p): cc.sha(p) for p in paths}})
    dump('clean_rescore_audit.json', clean_audit)
    import scipy
    dump('environment.json', dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    (OUT / 'task_list.md').write_text(
        '- [x] Trace native risk versus coverage annotation provenance; verify six clean classes\n'
        '- [ ] Run L and equal/proportional/risk-weighted quota axes\n'
        '- [ ] Run schema-valid joint label-error propagation\n'
        '- [ ] Run fixed-eligibility image-cluster bootstrap and legacy comparison\n'
        '- [ ] Audit raw outputs and write bounded findings\n')
    log('prepare_complete', inputs=len(paths), command=' '.join(sys.argv), clean_classes=len(clean_audit))


def axes():
    cc.verify_freeze(OUT / 'input_freeze.json')
    assert not (OUT / 'axes_runs.csv').exists(), 'axes outputs exist.'
    ctx = frozen.contexts()
    thresholds = json.loads((BASE / 'thresholds.json').read_text())
    starts = np.load(BASE / 'main_sets.npz', allow_pickle=False)
    mainline = {(r['class_name'], int(r['selection_seed'])): r for r in read(BASE / 'selection_runs.csv')
                if r['method'] == 'P20' and r['change_limit'] == '20'}
    rows, selected_sets = [], {}
    # 复现校验 1：frozen.priority 在冻结 R 起点上重放主线 P20 全部 300 个结果。
    for ci, c in enumerate(CLASSES):
        for seed in SEEDS:
            s0 = starts[frozen.key(c, 'R', seed)]
            s, work = frozen.priority(ctx[c], s0, thresholds[c], 20)
            assert np.array_equal(s, starts[frozen.key(c, 'P20', seed)])
            m = cc.count_metrics(ctx[c]['matrix'][s].sum(0), 20)
            r = mainline[c, seed]
            assert abs(100 * m['occurrence'] - float(r['occurrence_pct'])) < 1e-9
            assert abs(100 * m['support_completion'] - float(r['support_completion_pct'])) < 1e-9
            assert r['stop_reason'] == work['stop_reason'] and r['triggered'] == str(work['triggered'])
            assert int(r['removed_targets']) == len(set(s0) - set(s))
    log('mainline_reproduction_validated', runs=len(CLASSES) * len(SEEDS))
    # 复现校验 2：priority_for（等分配额）与 frozen.priority 逐位一致（用 dog 全部起点）。
    for ci, c in enumerate(CLASSES[:1]):
        q = cc.old.quotas(ctx[c], 200)
        for seed in SEEDS:
            s0 = starts[frozen.key(c, 'R', seed)]
            a, _ = frozen.priority(ctx[c], s0, thresholds[c], 20)
            b, _ = priority_for(ctx[c], s0, thresholds[c], 20, q)
            assert np.array_equal(a, b)
    log('priority_copy_validated', class_name=CLASSES[0], runs=len(SEEDS))

    def record(axis, setting, c, seed, elapsed, s0, s, work, metrics, alloc=None):
        selected_sets[f'{setting}__{c}__{seed}'] = s
        initial = cc.count_metrics(ctx[c]['matrix'][s0].sum(0), 20)
        rows.append(dict(axis=axis, setting=setting, class_name=c, selection_seed=seed,
                         tier_allocation=json.dumps(alloc or [40] * 5),
                         seconds=elapsed, triggered=work['triggered'], stop_reason=work['stop_reason'],
                         optimal_certified=work['optimal_certified'], removed_targets=len(set(s0) - set(s)),
                         initial_pass=cc.passes(initial, thresholds[c]),
                         final_pass=cc.passes(metrics, thresholds[c]),
                         occurrence_pct=100 * metrics['occurrence'],
                         support_completion_pct=100 * metrics['support_completion'],
                         supported_states=int((ctx[c]['matrix'][s].sum(0) >= 20).sum())))

    # 轴 1：移出上限 L∈{10,40}（同冻结 R 起点、同阈值）。
    for L in L_AXIS:
        for ci, c in enumerate(CLASSES):
            for seed in SEEDS:
                s0 = starts[frozen.key(c, 'R', seed)]
                tick = time.perf_counter()
                s, work = frozen.priority(ctx[c], s0, thresholds[c], L)
                elapsed = time.perf_counter() - tick
                m = cc.count_metrics(ctx[c]['matrix'][s].sum(0), 20)
                record('change_limit', f'L={L}', c, seed, elapsed, s0, s, work, m)
            log('axis_complete', axis='change_limit', setting=f'L={L}', class_name=c)
    # 轴 2：原池按分位数均分，池比例是等分设置的预先声明等价性对照。
    for name in QUOTA_SCHEMES:
        for ci, c in enumerate(CLASSES):
            keys, cell, tiers = ctx[c]['keys'], ctx[c]['cell'], ctx[c]['tiers']
            alloc = tier_allocation(tiers, name)
            quota = quotas_for(keys, cell, alloc)
            for seed in SEEDS:
                s0 = initial_for(cell, quota, seed, ci)
                validate_for(cell, quota, tiers, s0, alloc)
                if name in ['equal', 'pool_proportional']:
                    assert np.array_equal(s0, starts[frozen.key(c, 'R', seed)])
                tick = time.perf_counter()
                s, work = priority_for(ctx[c], s0, thresholds[c], 20, quota)
                elapsed = time.perf_counter() - tick
                validate_for(cell, quota, tiers, s, alloc)
                m = cc.count_metrics(ctx[c]['matrix'][s].sum(0), 20)
                if name in ['equal', 'pool_proportional']:
                    assert np.array_equal(s, starts[frozen.key(c, 'P20', seed)])
                record('quota_structure', name, c, seed, elapsed, s0, s, work, m, alloc)
            log('axis_complete', axis='quota_structure', setting=name, class_name=c)
    write('axes_runs.csv', rows)
    np.savez_compressed(OUT / 'axes_sets.npz', **selected_sets)
    dump('axes_audit.json', {'status': 'pass', 'runs': len(rows),
                             'mainline_reproduced': len(CLASSES) * len(SEEDS), 'priority_copy_validated': len(SEEDS),
                             'pool_proportional_equals_equal': True,
                             'prediction_blind': True, 'classifier_predictions_read': False})


def perturb():
    cc.verify_freeze(OUT / 'input_freeze.json')
    assert not (OUT / 'label_perturbation_runs.csv').exists(), 'perturb outputs exist.'
    ctx = frozen.contexts()
    thresholds = json.loads((BASE / 'thresholds.json').read_text())
    audit = clean_rescore_audit(ctx)
    native = native_features()
    baseline = np.load(BASE / 'main_sets.npz', allow_pickle=False)
    rows, worlds, selected_sets, witnesses = [], [], {}, {}
    for ci, c in enumerate(CLASSES):
        states = ctx[c]['states']
        groups = field_groups(states)
        rows_c = ctx[c]['rows']
        clean = ctx[c]['matrix']
        features = [native[s] for s in ctx[c]['ids']]
        shared = np.array([ns == 'shared' for ns, _, _ in states])
        for rate in RATES:
            for rep in range(PERT_REPS):
                rng = np.random.default_rng(np.random.SeedSequence([2026111020, ci, int(rate * 1000), rep]))
                Xp, risk_features = perturb_inputs(c, ctx[c], features, rng, rate)
                mids, tiers, keys, cell = rescore(c, rows_c, risk_features)
                quota = quotas_for(keys, cell, [40] * 5)
                tier_changed = float(np.mean(tiers != ctx[c]['tiers']))
                cell_changed = cell_change_fraction(ctx[c]['keys'], ctx[c]['cell'], keys, cell)
                assert cell_changed + 1e-12 >= tier_changed
                risk_changes = sum(a[f] != b[f] for a, b in zip(features, risk_features)
                                   for f in scoring_fields(c))
                worlds.append(dict(class_name=c, rate=rate, repeat=rep,
                                   false_positive_bits=int((Xp & ~clean).sum()),
                                   false_negative_bits=int((clean & ~Xp).sum()),
                                   shared_changed_bits=int((Xp[:, shared] != clean[:, shared]).sum()),
                                   native_risk_changed_fields=risk_changes,
                                   score_change_fraction=float(np.mean(np.abs(mids - ctx[c]['score']) > 1e-9)),
                                   tier_change_fraction=tier_changed, cell_change_fraction=cell_changed))
                pctx = dict(matrix=Xp, cell=cell, keys=keys, tiers=tiers,
                            ids=ctx[c]['ids'], rows=rows_c, states=states, embeddings=ctx[c]['embeddings'])
                witness, _ = cc.baseline(pctx, 200, 'G', 20)
                witnesses[f'{c}__{rate}__{rep}'] = witness
                witness_metrics = cc.count_metrics(Xp[witness].sum(0), 20)
                recalibrated = {k: .95 * witness_metrics[k] for k in thresholds[c]}
                worlds[-1].update(threshold_occurrence=recalibrated['occurrence'],
                                  threshold_support_completion=recalibrated['support_completion'])
                for seed in PERT_SEEDS:
                    s0 = initial_for(cell, quota, seed, ci)
                    validate_for(cell, quota, tiers, s0, [40] * 5)
                    for mode in THRESHOLD_MODES:
                        record_perturb_selection(mode, thresholds[c] if mode == 'frozen_clean' else recalibrated,
                                                 c, rate, rep, seed, ci, s0, pctx, quota, clean, Xp,
                                                 thresholds[c], baseline, tier_changed, cell_changed,
                                                 rows, selected_sets)
                log('perturb_world_complete', class_name=c, rate=rate, repeat=rep)
    write('label_perturbation_runs.csv', rows)
    write('label_perturbation_worlds.csv', worlds)
    np.savez_compressed(OUT / 'label_perturbation_sets.npz', **selected_sets)
    np.savez_compressed(OUT / 'label_perturbation_witnesses.npz', **witnesses)
    dump('perturb_audit.json', {'status': 'pass', 'runs': len(rows),
                                'clean_rescore_matches_frozen_spec': audit,
                                'coverage_reference_is_not_semantic_truth': True,
                                'cell_change_uses_semantic_keys': True,
                                'thresholds': THRESHOLD_MODES,
                                'prediction_blind': True, 'classifier_predictions_read': False})


def record_perturb_selection(mode, threshold, c, rate, rep, seed, ci, s0, pctx, quota,
                             clean, Xp, original_threshold, baseline, tier_changed, cell_changed,
                             rows, selected_sets):
    tick = time.perf_counter()
    s, work = priority_for(pctx, s0, threshold, 20, quota)
    elapsed = time.perf_counter() - tick
    validate_for(pctx['cell'], quota, pctx['tiers'], s, [40] * 5)
    view = cc.count_metrics(Xp[s].sum(0), 20)
    truth = cc.count_metrics(clean[s].sum(0), 20)
    selected_sets[f'{c}__{rate}__{rep}__{seed}__{mode}'] = s
    clean_s = baseline[frozen.key(c, 'P20', seed)]
    rows.append(dict(class_name=c, rate=rate, repeat=rep, selection_seed=seed,
                     threshold_mode=mode, threshold_occurrence=threshold['occurrence'],
                     threshold_support_completion=threshold['support_completion'],
                     seconds=elapsed, triggered=work['triggered'],
                     stop_reason=work['stop_reason'], optimal_certified=work['optimal_certified'],
                     removed_targets=len(set(s0) - set(s)),
                     view_pass=cc.passes(view, threshold),
                     view_occurrence_pct=100 * view['occurrence'],
                     view_support_pct=100 * view['support_completion'],
                     clean_pass=cc.passes(truth, original_threshold),
                     clean_occurrence_pct=100 * truth['occurrence'],
                     clean_support_pct=100 * truth['support_completion'],
                     changed_vs_clean_P20=len(set(s) - set(clean_s)),
                     tier_change_fraction=tier_changed, cell_change_fraction=cell_changed))


def bootstrap_contrasts(weights, positive, negative, correct):
    """Cluster weights are shared across contrasts; missing groups remain undefined."""
    n1, n0 = weights @ positive, weights @ negative
    r1 = np.divide(weights @ (positive * correct[:, None]), n1,
                   out=np.full(n1.shape, np.nan), where=n1 > 0)
    r0 = np.divide(weights @ (negative * correct[:, None]), n0,
                   out=np.full(n0.shape, np.nan), where=n0 > 0)
    # Legacy gate counts distinct original images, not bootstrap multiplicity.
    present = (weights > 0).astype(float)
    legacy = (n1 >= 20) & (n0 >= 20) & (present @ positive >= 20) & (present @ negative >= 20)
    return 100 * (r1 - r0), legacy


def contrast_record(delta, legacy):
    finite = np.isfinite(delta)
    assert finite.any(), 'No defined bootstrap contrasts'
    lo, hi = np.quantile(delta[finite], [.025, .975])
    old = finite & legacy
    old_lo, old_hi = np.quantile(delta[old], [.025, .975]) if old.any() else ('', '')
    weak = finite & (delta <= -10 + TOL)
    return dict(boot_p025=float(lo), boot_p975=float(hi),
                valid_replicates=int(finite.sum()),
                boot_leq_minus10=float(weak[finite].mean()),
                legacy_eligible_replicates=int(old.sum()),
                legacy_effect_p025=old_lo, legacy_effect_p975=old_hi,
                boot_leq_minus10_regated=float((weak & legacy).mean()))


def boot():
    """Pointwise effect intervals; discrete confirmation-count summaries stay descriptive."""
    import csv
    cc.verify_freeze(OUT / 'input_freeze.json')
    assert not (OUT / 'bootstrap_conditions.csv').exists(), 'boot outputs exist.'
    ctx = frozen.contexts()
    ext, _ = frozen.external_contexts(ctx)
    starts = np.load(BASE / 'main_sets.npz', allow_pickle=False)
    brows, drows = read(BASE / 'B_manifest.csv'), read(BASE / 'D_manifest.csv')
    for pool in [brows, drows]:
        assert len({r['image_id'] for r in pool}) == len(pool)
        assert len({r['image_sha256'] for r in pool}) == len(pool)
    bpred, bhash = cc.old.predictions('B', brows)
    dpred, dhash = cc.old.predictions('D', drows)
    boffsets = {r['sample_id']: i for i, r in enumerate(brows)}
    doffsets = {r['sample_id']: i for i, r in enumerate(drows)}
    ref = {(r['class_name'], r['model'], int(r['state_index'])): r for r in read(BASE / 'reference_all_states.csv')}
    discovery_groups = {(r['class_name'], r['method'], int(r['selection_seed']), r['model']): r
                        for r in read(BASE / 'discovery_groups.csv')}
    discovered = {(r['class_name'], r['method'], int(r['selection_seed']), r['model'], int(r['state_index']))
                  for r in read(BASE / 'discovered_conditions.csv')}
    rng = np.random.default_rng(BOOT_SEED)
    methods = ['R', 'C20', 'CS', 'P20', 'G', 'SR']
    group_rows, count_rows, d_effect_rows = [], [], []
    count_arrays = {}
    total_conditions = 0
    fields = ['class_name', 'method', 'selection_seed', 'model', 'state_index', 'namespace', 'field', 'value',
              'proposed', 'state_targets', 'control_targets', 'delta_pp', 'boot_p025', 'boot_p975',
              'valid_replicates', 'boot_leq_minus10', 'legacy_eligible_replicates',
              'legacy_effect_p025', 'legacy_effect_p975', 'boot_leq_minus10_regated']
    temporary = OUT / 'bootstrap_conditions.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ci, c in enumerate(CLASSES):
            ct, dt = ctx[c], ext[c]
            # A cluster is an original image; verify the historical one-target-per-image contract.
            for data in [ct, dt]:
                assert len({r['image_id'] for r in data['rows']}) == len(data['rows'])
                assert len({r['image_sha256'] for r in data['rows']}) == len(data['rows'])
            X, control = cc.condition_masks(ct)
            DX, DC = cc.condition_masks(dt)
            pos = np.array([boffsets[s] for s in ct['ids']])
            dpos = np.array([doffsets[s] for s in dt['ids']])
            WB = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=BOOT).astype(float)
            WD = rng.multinomial(len(dpos), np.full(len(dpos), 1 / len(dpos)), size=BOOT).astype(float)
            S = X.shape[1]
            d_eligible, d_point, d_boot, d_legacy = {}, {}, {}, {}
            for m in MODELS:
                dc = (dpred[m][:, dpos] == ci).mean(0)
                n1D, n0D = DX.sum(0), DC.sum(0)
                r1D = np.divide(DX.T @ dc, n1D, out=np.zeros(S), where=n1D > 0)
                r0D = np.divide(DC.T @ dc, n0D, out=np.zeros(S), where=n0D > 0)
                deltaD = 100 * (r1D - r0D)
                for j in range(S):
                    assert abs(deltaD[j] - float(ref[c, m, j]['reference_delta_pp'])) < 1e-8
                eligible = (n1D >= 20) & (n0D >= 20)
                assert np.array_equal(eligible, [ref[c, m, j]['reference_evaluable'] == 'True' for j in range(S)])
                d_eligible[m], d_point[m] = eligible, deltaD
                effects, legacy = bootstrap_contrasts(WD, DX, DC, dc)
                d_boot[m], d_legacy[m] = effects, legacy
                for j in np.flatnonzero(eligible):
                    ns, field, value = ct['states'][j]
                    d_effect_rows.append(dict(class_name=c, model=m, state_index=int(j),
                                             namespace=ns, field=field, value=value,
                                             state_targets=int(n1D[j]), control_targets=int(n0D[j]),
                                             delta_pp=float(deltaD[j]), **contrast_record(effects[:, j], legacy[:, j])))
            # Deterministic G/SR repeat across selection seeds; cache their interval calculations.
            cache = {}
            for seed in SEEDS:
                for method in methods:
                    s = np.sort(starts[frozen.key(c, method, seed)])
                    cache_key = tuple(s.tolist())
                    if cache_key not in cache:
                        A, B = X[s], control[s]
                        n1, n0 = A.sum(0), B.sum(0)
                        known = (n1 >= 20) & (n0 >= 20)
                        by_model = {}
                        for m in MODELS:
                            y = (bpred[m][:, pos] == ci).mean(0)[s]
                            r1 = np.divide(A.T @ y, n1, out=np.zeros(S), where=n1 > 0)
                            r0 = np.divide(B.T @ y, n0, out=np.zeros(S), where=n0 > 0)
                            delta = 100 * (r1 - r0)
                            proposed = np.flatnonzero(known & (delta <= -10 + TOL))
                            js = np.flatnonzero(known)
                            effects, legacy = bootstrap_contrasts(WB[:, s], A[:, js], B[:, js], y)
                            records = []
                            for k, j in enumerate(js):
                                ns, field, value = ct['states'][j]
                                records.append(dict(state_index=int(j), namespace=ns, field=field, value=value,
                                                    proposed=bool(j in proposed), state_targets=int(n1[j]),
                                                    control_targets=int(n0[j]), delta_pp=float(delta[j]),
                                                    **contrast_record(effects[:, k], legacy[:, k])))
                            by_model[m] = dict(known=int(known.sum()), proposed=proposed, records=records)
                        cache[cache_key] = by_model
                    fixed_counts, regated_counts = np.zeros(BOOT, int), np.zeros(BOOT, int)
                    point_confirmed = 0
                    for m, data in cache[cache_key].items():
                        proposed = data['proposed']
                        # Reproduce the original discovery proposal set exactly before applying reference draws.
                        actual = {j for j in range(S) if (c, method, seed, m, j) in discovered}
                        assert set(proposed.tolist()) == actual
                        old_group = discovery_groups[c, method, seed, m]
                        assert len(proposed) == int(old_group['proposed'])
                        assert data['known'] == int(old_group['discovery_evaluable_states'])
                        group_rows.append(dict(class_name=c, method=method, selection_seed=seed, model=m,
                                               eligible=data['known'], proposed=len(proposed),
                                               D_evaluable_proposed=int(d_eligible[m][proposed].sum())))
                        js = proposed[d_eligible[m][proposed]]
                        if len(js):
                            effects = d_boot[m][:, js]
                            assert np.isfinite(effects).all(), 'Undefined D draw; do not count as negative'
                            fixed = effects <= -10 + TOL
                            legacy = fixed & d_legacy[m][:, js]
                            assert np.all(legacy <= fixed)
                            fixed_counts += fixed.sum(1)
                            regated_counts += legacy.sum(1)
                            point_confirmed += int((d_point[m][js] <= -10 + TOL).sum())
                        for row in data['records']:
                            writer.writerow(dict(class_name=c, method=method, selection_seed=seed, model=m, **row))
                            total_conditions += 1
                    count_rows.append(dict(class_name=c, method=method, selection_seed=seed,
                                           point_confirmed=point_confirmed,
                                           fixed_p025=float(np.quantile(fixed_counts, .025)),
                                           fixed_median=float(np.median(fixed_counts)),
                                           fixed_p975=float(np.quantile(fixed_counts, .975)),
                                           regated_p025=float(np.quantile(regated_counts, .025)),
                                           regated_median=float(np.median(regated_counts)),
                                           regated_p975=float(np.quantile(regated_counts, .975)),
                                           fixed_mean=float(fixed_counts.mean()),
                                           regated_mean=float(regated_counts.mean())))
                    count_arrays[f'{c}__{method}__{seed}__fixed'] = fixed_counts
                    count_arrays[f'{c}__{method}__{seed}__legacy'] = regated_counts
                if (seed - SEEDS[0] + 1) % 10 == 0:
                    log('boot_progress', class_name=c, completed_seeds=seed - SEEDS[0] + 1)
            log('boot_class_complete', class_name=c, condition_rows=total_conditions)
    temporary.replace(OUT / 'bootstrap_conditions.csv')
    write('bootstrap_reference_effects.csv', d_effect_rows)
    write('bootstrap_groups.csv', group_rows)
    write('bootstrap_counts.csv', count_rows)
    np.savez_compressed(OUT / 'bootstrap_count_draws.npz', **count_arrays)
    dump('boot_audit.json', {'status': 'pass', 'replicates': BOOT, 'seed': BOOT_SEED,
                             'D_point_deltas_and_B_proposals_reproduced': True,
                             'eligibility': 'fixed at point estimate; all eligible B and D contrasts reported',
                             'legacy_gate_includes_distinct_original_images': True,
                             'count_percentiles_are_descriptive_not_confidence_intervals': True,
                             'effect_intervals_are_approximate_pointwise_not_post_selection_calibrated': True,
                             'paired_full_pool_image_weights': True, 'prediction_hashes': {'B': bhash, 'D': dhash},
                             'new_inference': False, 'historical_data_only': True,
                             'condition_rows': total_conditions})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'axes', 'perturb', 'boot'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    globals()[args.phase]()
