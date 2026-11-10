"""Small exhaustive-oracle checks for the research algorithm, not result fitting."""
import itertools
import numpy as np
import conditional_coverage_common as cc


def context(matrix, cells=None):
    X = np.array(matrix, bool)
    if cells is None:
        # Five tiers are required by the real quota validator.
        cells = np.repeat(np.arange(5), len(X) // 5)
    c = dict(matrix=X, cell=np.array(cells), ids=np.array([f'{i:03}' for i in range(len(X))]),
             keys=[(i + 1, 0, 0) for i in range(5)], tiers=np.array(cells) + 1)
    pairs = [(i, j) for i in range(len(X)) for j in range(len(X))
             if i != j and cells[i] == cells[j]]
    c['pair_out'] = np.array([i for i, j in pairs], int)
    c['pair_in'] = np.array([j for i, j in pairs], int)
    c['pair_delta'] = X[c['pair_in']].astype(np.int8) - X[c['pair_out']].astype(np.int8)
    return c


def oracle_step(ctx, selected, thresholds, cap):
    current = cc.count_metrics(ctx['matrix'][selected].sum(0), cap)
    D = cc.deficit(current, thresholds)
    candidates = []
    for outgoing in sorted(selected):
        for incoming in range(len(ctx['matrix'])):
            if incoming in selected or ctx['cell'][outgoing] != ctx['cell'][incoming]:
                continue
            proposed = np.array([incoming if i == outgoing else i for i in selected])
            m = cc.count_metrics(ctx['matrix'][proposed].sum(0), cap)
            if any(current[k] + cc.TOL >= thresholds[k] and m[k] + cc.TOL < thresholds[k]
                   for k in ['occurrence', 'support_completion']):
                continue
            d = cc.deficit(m, thresholds)
            if d < D - cc.TOL or (abs(d-D) <= cc.TOL and m['capped_log'] > current['capped_log'] + cc.TOL):
                candidates.append((d, -m['capped_log'], outgoing, incoming))
    if not candidates:
        return None
    best_d = min(r[0] for r in candidates)
    candidates = [r for r in candidates if r[0] <= best_d + cc.TOL]
    best_negative_utility = min(r[1] for r in candidates)
    candidates = [r for r in candidates if r[1] <= best_negative_utility + cc.TOL]
    return min(candidates, key=lambda r: (r[2], r[3]))


def run_checks():
    checks = []
    # Compare vectorized scans against independent whole-selection enumeration.
    rng = np.random.default_rng(19)
    for case in range(100):
        ctx = context(rng.random((15, 8)) < .3)
        s0 = np.arange(0, 15, 3)
        thresholds = dict(occurrence=.8, support_completion=.65)
        expected = oracle_step(ctx, s0, thresholds, 3)
        actual, work, trace = cc.swap_select(ctx, s0, thresholds, cap=3,
                                            conditional=False, max_swaps=1)
        if expected is None:
            assert len(trace) == 0
        else:
            assert (trace[0]['outgoing_id'], trace[0]['incoming_id']) == (
                str(ctx['ids'][expected[2]]), str(ctx['ids'][expected[3]]))
        cc.old.validate_selection(ctx, actual, 5)
    checks.append('100_random_exhaustive_whole_set_oracle_cases')
    ctx = context(rng.random((15, 8)) < .4)
    s0 = np.arange(0, 15, 3)
    current = cc.count_metrics(ctx['matrix'][s0].sum(0), 3)
    thresholds = {k: current[k] * .9 for k in ['occurrence', 'support_completion']}
    selected, work, trace = cc.swap_select(ctx, s0, thresholds, cap=3)
    assert np.array_equal(selected, s0) and work['pair_scores'] == 0 and not trace
    checks.append('initial_pass_noop_zero_search')
    # A new state cannot be counted without removing the outgoing last support.
    ctx = context([[1, 0], [0, 1]] + [[0, 0], [0, 0]] * 4)
    s0 = np.arange(0, 10, 2)
    _, work, trace = cc.swap_select(ctx, s0, dict(occurrence=1., support_completion=.5), cap=1)
    assert not trace and work['stop_reason'] == 'local_no_improvement'
    checks.append('outgoing_last_support_loss_not_ignored')
    # Cap and first-occurrence boundaries have hand-computed values.
    m = cc.count_metrics(np.array([0, 1, 19, 20, 21]), 20)
    assert np.isclose(m['occurrence'], .8)
    assert np.isclose(m['support_completion'], (0 + .05 + .95 + 1 + 1)/5)
    assert np.isclose(m['support_threshold_fraction'], .4)
    checks.append('one_and_K_boundary_metrics')
    _, work, trace = cc.swap_select(ctx, s0, dict(occurrence=1., support_completion=1.), max_pairs=1)
    assert work['stop_reason'] == 'search_limit' and work['pair_scores'] == 1 and not trace
    checks.append('partial_scan_discarded_and_budget_counted')
    ctx2 = context([[1, 0]] * 5, cells=list(range(5)))
    _, work, _ = cc.swap_select(ctx2, np.arange(5), dict(occurrence=1., support_completion=1.))
    assert work['stop_reason'] == 'no_legal_pair'
    checks.append('no_legal_pair_distinguished')
    for case in range(30):
        ctx = context(rng.random((15, 8)) < .3)
        s0 = np.arange(0, 15, 3)
        threshold = dict(occurrence=.9, support_completion=.5)
        a, wa, ta = cc.swap_select(ctx, s0, threshold, cap=3)
        b, wb, tb = cc.swap_select(ctx, s0, threshold, cap=3, conditional=False)
        assert ta == tb[:len(ta)]
    checks.append('CS_AS_identical_prefix_and_incremental_full_recount')
    for bad in [np.array([0, 0, 6, 9, 12]), np.array([0, 1, 6, 9, 12])]:
        try:
            cc.old.validate_selection(ctx, bad, 5)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid duplicate/quota accepted')
    checks.append('duplicates_and_wrong_cells_rejected')
    return dict(status='pass', checks=checks, prediction_files_read=False)


if __name__ == '__main__':
    args = cc.parsed_args(__doc__).parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = run_checks()
    cc.dump(args.out / 'implementation_checks.json', result)
    cc.log(args.out, 'synthetic_checks_complete', **result)
