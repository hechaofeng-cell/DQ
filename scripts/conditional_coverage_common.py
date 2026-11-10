"""Prediction-blind, matched-cell conditional coverage selection utilities."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import coverage_validation_common_20261018 as old

ROOT = old.ROOT
METHODS = ['R', 'G', 'SR', 'KC', 'CS', 'AS']
DEFAULT_OUT = ROOT / 'artifacts/risk_first_conditional_coverage_20261105'
PLAN = ROOT / 'configs/risk_first_conditional_coverage_experiment_plan_20261105.json'
TOL = 1e-12


def dump(path, obj):
    old.dump(path, obj)


def read(path):
    return old.read(path)


def write(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError(f'Empty table needs declared columns: {path}')
    old.write(path, rows, fields)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(path, obj):
    old.freeze(path, obj)


def log(out, stage, **kw):
    obj = dict(time=old.now(), stage=stage, **kw)
    print(json.dumps(obj, ensure_ascii=False), flush=True)
    with (Path(out) / 'research_log.md').open('a') as f:
        f.write('\n' + json.dumps(obj, ensure_ascii=False) + '\n')


def load_contexts():
    contexts = old.load_contexts()
    for ctx in contexts.values():
        assert ctx['matrix'].dtype == bool
        assert len(ctx['ids']) == len(set(ctx['ids']))
        assert len(ctx['rows']) == len(ctx['ids'])
        assert not any(s[2] in ('unknown', 'not_applicable', 'missing') for s in ctx['states'])
        # Stable pair order also implements the frozen sample-ID tie rule.
        order = sorted(range(len(ctx['ids'])), key=lambda i: ctx['ids'][i])
        outs, ins = [], []
        for i in order:
            for j in order:
                if i != j and ctx['cell'][i] == ctx['cell'][j]:
                    outs.append(i)
                    ins.append(j)
        ctx['pair_out'] = np.asarray(outs, dtype=int)
        ctx['pair_in'] = np.asarray(ins, dtype=int)
        ctx['pair_delta'] = (ctx['matrix'][ins].astype(np.int8)
                             - ctx['matrix'][outs].astype(np.int8))
    return contexts


def count_metrics(counts, cap=20):
    counts = np.asarray(counts)
    return dict(occurrence=float(np.mean(counts > 0)),
                support_completion=float(np.minimum(counts / cap, 1).mean()),
                support_threshold_fraction=float(np.mean(counts >= cap)),
                capped_log=float(np.log1p(np.minimum(counts, cap).astype(np.float64)).mean() / np.log1p(cap)))


def coverage_rows(ctx, selected, cap=20):
    counts = ctx['matrix'][selected].sum(0)
    shared = np.array([s[0] == 'shared' for s in ctx['states']])
    return {ns: count_metrics(counts[mask], cap) for ns, mask in
            [('all', np.ones(len(counts), bool)), ('shared', shared), ('specific', ~shared)]
            if np.any(mask)}


def deficit(metrics, thresholds):
    return sum(max(0., (thresholds[k] - metrics[k]) / max(thresholds[k], TOL))
               for k in ['occurrence', 'support_completion'])


def passes(metrics, thresholds):
    return all(metrics[k] + TOL >= thresholds[k]
               for k in ['occurrence', 'support_completion'])


def initial_selection(ctx, budget, seed, class_index):
    # Independent reproducible class streams; no classifier input.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(class_index)]))
    q = old.quotas(ctx, budget)
    parts = [rng.choice(np.flatnonzero(ctx['cell'] == cell), int(n), replace=False)
             for cell, n in enumerate(q) if n]
    selected = np.concatenate(parts).astype(int)
    old.validate_selection(ctx, selected, budget)
    return selected


def baseline(ctx, budget, method, cap=20):
    """Instrument the existing baselines without changing their choice rule."""
    X = ctx['matrix']
    q = old.quotas(ctx, budget)
    used = np.zeros(len(q), int)
    available = np.ones(len(X), bool)
    count = np.zeros(X.shape[1], int)
    selected = []
    static = (X / np.maximum(X.sum(0), 1)).sum(1) if method == 'SR' else None
    E = ctx['embeddings']
    centroid = E.mean(0)
    centroid /= max(np.linalg.norm(centroid), TOL)
    distance = None
    scores = 0
    for _ in range(budget):
        feasible = np.flatnonzero(available & (used[ctx['cell']] < q[ctx['cell']]))
        scores += len(feasible)
        if method == 'G':
            marginal = np.where(count < cap, np.log(count + 2) - np.log(count + 1), 0)
            gain = np.sum(X[feasible] * marginal, axis=1)
        elif method == 'SR':
            gain = static[feasible]
        elif method == 'KC':
            gain = E[feasible] @ centroid if distance is None else distance[feasible]
        else:
            raise ValueError(method)
        ties = feasible[np.isclose(gain, gain.max(), atol=TOL, rtol=0)]
        i = int(min(ties, key=lambda k: ctx['ids'][k]))
        selected.append(i)
        available[i] = False
        used[ctx['cell'][i]] += 1
        count += X[i]
        if method == 'KC':
            d = np.maximum(0, 1 - E @ E[i])
            distance = d if distance is None else np.minimum(distance, d)
    result = np.array(selected, int)
    old.validate_selection(ctx, result, budget)
    return result, dict(add_candidate_scores=int(scores), pair_scores=0, full_scans=0,
                        accepted_swaps=0, stop_reason='complete')


def swap_select(ctx, initial, thresholds, *, cap=20, conditional=True,
                max_swaps=None, max_pairs=2_000_000, validate=True):
    X = ctx['matrix']
    selected = np.array(initial, dtype=int, copy=True)
    budget = len(selected)
    if max_swaps is None:
        max_swaps = budget
    old.validate_selection(ctx, selected, budget)
    picked = np.zeros(len(X), bool)
    picked[selected] = True
    counts = X[selected].sum(0).astype(np.int16)
    current = count_metrics(counts, cap)
    trace = []
    work = dict(add_candidate_scores=0, pair_scores=0, full_scans=0, accepted_swaps=0,
                discarded_partial_scan_pairs=0)
    reason = None
    while True:
        if conditional and passes(current, thresholds):
            reason = 'initial_pass' if not trace else 'post_swap_pass'
            break
        if len(trace) >= max_swaps:
            reason = 'acceptance_limit'
            break
        legal = picked[ctx['pair_out']] & ~picked[ctx['pair_in']]
        ix = np.flatnonzero(legal)
        if not len(ix):
            reason = 'no_legal_pair'
            break
        remaining = max_pairs - work['pair_scores']
        if remaining < len(ix):
            # Evaluate (and discard) the real partial scan, as the protocol requires.
            if remaining:
                partial = counts[None] + ctx['pair_delta'][ix[:remaining]]
                np.minimum(partial / cap, 1).mean(1)
                np.log1p(np.minimum(partial, cap).astype(np.float64)).mean(1)
            work['pair_scores'] += remaining
            work['discarded_partial_scan_pairs'] = remaining
            reason = 'search_limit'
            break
        n = counts[None] + ctx['pair_delta'][ix]
        assert np.all(n >= 0)
        occurrence = np.mean(n > 0, axis=1)
        support = np.minimum(n / cap, 1).mean(1)
        utility = np.log1p(np.minimum(n, cap).astype(np.float64)).mean(1) / np.log1p(cap)
        D = (np.maximum(0, (thresholds['occurrence'] - occurrence) /
                        max(thresholds['occurrence'], TOL)) +
             np.maximum(0, (thresholds['support_completion'] - support) /
                        max(thresholds['support_completion'], TOL)))
        previous_D = deficit(current, thresholds)
        eligible = ((D < previous_D - TOL) |
                    ((np.abs(D - previous_D) <= TOL) & (utility > current['capped_log'] + TOL)))
        if current['occurrence'] + TOL >= thresholds['occurrence']:
            eligible &= occurrence + TOL >= thresholds['occurrence']
        if current['support_completion'] + TOL >= thresholds['support_completion']:
            eligible &= support + TOL >= thresholds['support_completion']
        work['pair_scores'] += len(ix)
        work['full_scans'] += 1
        if not np.any(eligible):
            reason = 'local_no_improvement'
            break
        candidates = np.flatnonzero(eligible)
        candidates = candidates[D[candidates] <= D[candidates].min() + TOL]
        candidates = candidates[utility[candidates] >= utility[candidates].max() - TOL]
        k = int(candidates[0])
        pair = int(ix[k])
        outgoing, incoming = int(ctx['pair_out'][pair]), int(ctx['pair_in'][pair])
        assert picked[outgoing] and not picked[incoming]
        assert ctx['cell'][outgoing] == ctx['cell'][incoming]
        before = current
        counts += ctx['pair_delta'][pair]
        selected[np.flatnonzero(selected == outgoing)[0]] = incoming
        picked[outgoing] = False
        picked[incoming] = True
        current = count_metrics(counts, cap)
        if validate:
            old.validate_selection(ctx, selected, budget)
            assert np.array_equal(counts, X[selected].sum(0))
        trace.append(dict(step=len(trace) + 1, outgoing_id=str(ctx['ids'][outgoing]),
                          incoming_id=str(ctx['ids'][incoming]), cell=int(ctx['cell'][outgoing]),
                          occurrence_before=before['occurrence'], occurrence_after=current['occurrence'],
                          support_before=before['support_completion'], support_after=current['support_completion'],
                          deficit_before=previous_D, deficit_after=deficit(current, thresholds),
                          utility_before=before['capped_log'], utility_after=current['capped_log'],
                          legal_pairs=len(ix), cumulative_pair_scores=work['pair_scores'],
                          acceptance='lower_deficit' if D[k] < previous_D - TOL else 'equal_deficit_higher_utility'))
    work.update(accepted_swaps=len(trace), stop_reason=reason,
                final_pass=passes(current, thresholds))
    return selected, work, trace


def verify_freeze(path):
    obj = json.loads(Path(path).read_text())
    for name, digest in obj['files'].items():
        assert sha(name) == digest, f'Frozen file changed: {name}'
    return obj


def selection_key(class_name, method, seed):
    return f'{class_name}__{method}__s{seed}'


def condition_masks(ctx):
    """Known same-field complements. Unknown has no encoded state and is excluded."""
    X = ctx['matrix']
    controls = np.zeros_like(X)
    for j, state in enumerate(ctx['states']):
        field = [i for i, s in enumerate(ctx['states']) if s[:2] == state[:2]]
        controls[:, j] = X[:, field].any(1) & ~X[:, j]
    return X, controls


def image_support(rows, mask):
    return len({r['image_id'] for r, keep in zip(rows, mask) if keep})


def parsed_args(description, phases=None):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    if phases:
        parser.add_argument('phase', choices=phases)
    return parser
