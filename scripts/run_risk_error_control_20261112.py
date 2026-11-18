"""Historical sensitivity to risk standardization and outcome-matched subsampling."""
from __future__ import annotations

from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import run_label_support_diagnosis_20261112 as support

previous, study, cc, old = support.previous, support.study, support.cc, support.old
ROOT = support.ROOT
OUT = ROOT / 'artifacts/risk_error_control_20261112'
METHODS, VIEWS, AXES = support.METHODS, support.VIEWS, support.AXES
COMPARISONS = support.COMPARISONS
PATHS = dict(pooled=('pooled', 'pooled'), retained=('retained', 'retained'),
             standardized=('standardized', 'standardized'),
             discovery_standardized=('standardized', 'pooled'),
             reference_standardized=('pooled', 'standardized'))
ANCHORS = [('A', 'A', 'original'), ('B', 'B', 'recalibrated')]
REPEATS = 20


def dump(name, value):
    cc.dump(OUT / name, value)


def write(name, rows):
    cc.write(OUT / name, rows)


class Stream:
    def __init__(self, name):
        self.file = (OUT / name).open('w', encoding='utf-8-sig', newline='')
        self.writer = None
        self.count = 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row); self.count += 1

    def close(self):
        self.file.close()


def strata_counts(x, y, tiers):
    return np.array([x[tiers == q].sum(0) for q in range(1, 6)]), np.array([y[tiers == q].sum(0) for q in range(1, 6)])


def structure(x, y, tiers, weights, keep=None):
    a, b = strata_counts(x, y, tiers)
    own_keep = (a >= 5) & (b >= 5)
    keep = own_keep if keep is None else np.asarray(keep, bool)
    assert np.all(keep <= own_keep)
    n1, n0 = (a * keep).sum(0), (b * keep).sum(0)
    eligible = (n1 >= 20) & (n0 >= 20) & (n1 >= .8 * x.sum(0))
    w = weights * keep
    w = np.divide(w, w.sum(0), out=np.zeros_like(w, dtype=float), where=w.sum(0) > 0)
    return dict(a=a, b=b, keep=keep, eligible=eligible, weights=w)


def gap_bundle(x, y, tiers, correct, st):
    # All versions use the same original selected set; retained versions mask by state-specific strata.
    h1 = np.array([x[tiers == q].T @ correct[tiers == q] for q in range(1, 6)])
    h0 = np.array([y[tiers == q].T @ correct[tiers == q] for q in range(1, 6)])
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    pooled = 100 * (ratio(h1.sum(0), st['a'].sum(0)) - ratio(h0.sum(0), st['b'].sum(0)))
    retained = 100 * (ratio((h1 * st['keep']).sum(0), (st['a'] * st['keep']).sum(0)) -
                      ratio((h0 * st['keep']).sum(0), (st['b'] * st['keep']).sum(0)))
    standardized = 100 * ((ratio(h1, st['a']) - ratio(h0, st['b'])) * st['weights']).sum(0)
    return dict(pooled=pooled, retained=retained, standardized=standardized)


def match_outcomes(selected, tiers, correct, ranks):
    """Same error and correct counts within each Q, using shared random image ranks."""
    matched, cells = {m: [] for m in METHODS}, []
    for q in range(1, 6):
        for outcome in [False, True]:
            members = {m: ix[(tiers[ix] == q) & (correct[ix] == outcome)] for m, ix in selected.items()}
            n = min(len(v) for v in members.values())
            for m, ix in members.items():
                matched[m].extend(ix[np.argsort(ranks[ix], kind='stable')[:n]].tolist())
            cells.append((q, outcome, n))
    matched = {m: np.array(v, dtype=int) for m, v in matched.items()}
    for m, ix in matched.items():
        assert len(ix) == len(set(ix)) and set(ix) <= set(selected[m])
        for q, outcome, n in cells:
            assert int(((tiers[ix] == q) & (correct[ix] == outcome)).sum()) == n
    assert len({len(v) for v in matched.values()}) == 1
    return matched, cells


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'completion.json').exists(), 'Completed output cannot be overwritten'
    for root in [previous.OUT, support.CONSENSUS, support.OUT]:
        cc.verify_freeze(root / 'final_outputs_freeze.json')
    cc.verify_freeze(previous.OUT / 'input_freeze.json')
    protocol = dict(
        hypothesis='Residual confirmed-condition gains on common evaluable states may change after risk standardization or exact within-Q error/correct-count matching',
        scope='Exploratory historical analysis following known support decomposition; not causal identification or new-data confirmation',
        primary=dict(scenarios='All8 frozen risk x coverage x threshold combinations;50 starts;6 classes;R/Greedy20/P20;5 classifiers;three-seed mean correctness',
            risk='Selection risk source provides frozen B and D class-relative Q1-Q5; same Q within each scenario across evaluators and methods',
            retained='For each state use only Q with >=5 positive and >=5 known negative under all3 methods; each method retains >=20 per arm and >=80% original state positives; reference D has same criteria independently',
            weighting='Full source/class pool state-positive Q shares, fixed per label view and phase; normalize on shared retained Q; same weights for positive/negative and all methods',
            universe='Intersection of adjusted-support eligibility for all3 methods and D; independent of correctness values; pooled/retained/standardized paths use identical state IDs',
            paths=PATHS, gap_threshold_pp=-10, no_model_output_selection=True,
            limits='Common retained Q can exclude rare states; B and D have separate state-specific weights; Q adjustment does not equal actual-error matching and cannot establish causality'),
        auxiliary=dict(anchors=ANCHORS,starts=previous.SEEDS,repeats=REPEATS,prediction_seeds=cc.old.SEEDS,
            matching='Within each class/model/prediction-seed/Q, sample min(error counts) and min(correct counts) across all3 methods; shared deterministic random rank on original image IDs',
            RNG='SeedSequence([2026111207, anchor_index, selection_seed, class_index, model_index, prediction_seed_index, repeat])',
            diagnostic='Single-prediction-seed correctness in B and unchanged same-seed D; state/control each>=20, gap<=-10pp; use only states evaluable by all3 original and matched methods and D per draw',
            interpretation='Outcome-conditioned sensitivity analysis; reduces N, changes fine-cell quotas and sometimes Q proportions; changes diagnostic estimand from primary 3-seed-mean rule; matching does not prove causal independence',
            aggregation='Sum conditions over6 classes x5 classifiers within start/prediction-seed/repeat, then mean over3 prediction seeds and20 draws within start, then50 starts; draws not independent datasets'),
        controls='Reproduce 180000 previous raw diagnostics and4500 original A legacy adjusted groups; independent scalar gap and outcome-balance checks',
        stop='Complete frozen primary and two bounded auxiliary anchors; no threshold/prompt/seed searches or dropping unfavorable results',
        no_new_vlm_calls=True,no_new_classifier_inference=True,no_human_verification=True)
    cc.freeze(OUT / 'protocol.json', protocol)
    paths = [Path(__file__), ROOT / 'scripts/report_risk_error_control_20261112.py', ROOT / 'tests/test_risk_error_control.py',
             Path(support.__file__), Path(previous.__file__), Path(previous.prior.__file__), Path(cc.__file__), Path(cc.old.__file__),
             ROOT / 'output/experiments/实证论文补全_20261108/execute.py',
             ROOT / 'artifacts/empirical_completion_20261108/risk_adjusted_groups.csv',
             previous.OUT / 'selections.npz', previous.OUT / 'diagnostic.csv', previous.OUT / 'prediction_freeze.json',
             support.CONSENSUS / 'diagnostic.csv', support.OUT / 'final_outputs_freeze.json']
    inputs = json.loads((previous.OUT / 'input_freeze.json').read_text())['files'].copy()
    paths += list(previous.PREV.glob('*_matrix.npz')) + list(support.CONSENSUS.glob('*_consensus.npz'))
    inputs.update({str(p): study.sha(p) for p in paths})
    cc.freeze(OUT / 'input_freeze.json', dict(files=inputs))
    (OUT / 'paper_brief.md').write_text('保持风险配额受控覆盖选样主线，检验历史诊断的风险构成及实际错误率敏感性，不改变论文中心主张。\n')
    (OUT / 'research_log.md').write_text('# 执行记录\n\n用户授权继续检验样本难度与实际错误率混杂。先冻结全量风险分层主分析和两锚点、50起点、20次重抽的结果条件化辅助检查。仅用冻结缓存，不称为首次盲验证或因果控制。\n')
    cc.log(OUT, 'prepared')


def load():
    rows, risks, matrices = previous.contexts()
    raw, hashes = {}, {}
    for phase in rows:
        preds, hh = cc.old.predictions(phase, rows[phase]); hashes.update(hh)
        offsets = {r['sample_id']: i for i, r in enumerate(rows[phase])}
        for ci, c in enumerate(previous.CLASSES):
            ids = risks[phase, 'A', c]['ids']; pos = [offsets[s] for s in ids]
            correct = {m: preds[m][:, pos] == ci for m in old.MODELS}
            states = risks[phase, 'A', c]['states']
            a,b = [matrices[phase, s, c] for s in ['A','B']]
            ac,bc = [previous.prior.value_known_mask(x,states) for x in [a,b]]
            for source,x,y in [('A',a,ac),('B',b,bc),('C',a&b,ac&bc)]:
                raw[phase,source,c] = x,y,correct
    cc.freeze(OUT / 'prediction_freeze.json', dict(files=hashes))
    return risks, raw


def legacy_audit(risks, raw, z):
    wanted = {(r['class_name'],r['model'],int(r['selection_seed']),r['method']):r for r in
              study.read(ROOT/'artifacts/empirical_completion_20261108/risk_adjusted_groups.csv') if r['method'] in METHODS}
    actual=0
    for c in previous.CLASSES:
        x,y,correct=raw['B','A',c];dx,dy,dc=raw['D','A',c]
        q=risks['B','A',c]['tiers'];dq=risks['D','A',c]['tiers']
        w=strata_counts(x,y,q)[0];dw=strata_counts(dx,dy,dq)[0]
        ds=structure(dx,dy,dq,dw)
        for seed in previous.SEEDS:
            for method in METHODS:
                ix=z[previous.key('A','A','original',c,method,seed)]
                st=structure(x[ix],y[ix],q[ix],w)
                for model in old.MODELS:
                    g=gap_bundle(x[ix],y[ix],q[ix],correct[model].mean(0)[ix],st)['standardized']
                    dg=gap_bundle(dx,dy,dq,dc[model].mean(0),ds)['standardized']
                    weak=st['eligible']&(g<=-10+1e-12);dweak=ds['eligible']&(dg<=-10+1e-12)
                    expected=wanted[c,model,seed,method]
                    counts=[int(st['eligible'].sum()),int(weak.sum()),int((weak&ds['eligible']).sum()),int((weak&dweak).sum())]
                    assert counts==[int(expected[k]) for k in ['eligible','proposed','reference_evaluable','confirmed']]
                    actual+=1
    assert actual==len(wanted)==4500
    return actual


def scalar_check(x,y,q,correct,st,bundle):
    for j in np.flatnonzero(st['eligible']):
        present=x[:,j];absent=y[:,j];keep=st['keep'][:,j]
        chosen=np.array([bool(keep[int(t)-1]) for t in q])
        pooled=100*(float(correct[present].mean())-float(correct[absent].mean()))
        retained=100*(float(correct[present&chosen].mean())-float(correct[absent&chosen].mean()))
        standardized=sum(100*float(st['weights'][t-1,j])*(float(correct[present&(q==t)].mean())-
                          float(correct[absent&(q==t)].mean())) for t in range(1,6) if keep[t-1])
        for name,value in [('pooled',pooled),('retained',retained),('standardized',standardized)]:
            assert abs(value-bundle[name][j])<1e-9


def primary():
    assert not (OUT/'completion.json').exists()
    cc.verify_freeze(OUT/'input_freeze.json');cc.verify_freeze(previous.OUT/'prediction_freeze.json')
    risks,raw=load();z=np.load(previous.OUT/'selections.npz',allow_pickle=False)
    legacy=legacy_audit(risks,raw,z)
    target={}
    for root,consensus in [(previous.OUT,False),(support.CONSENSUS,True)]:
        for r in study.read(root/'diagnostic.csv'):
            if r['method'] not in METHODS:continue
            k=tuple(r[f] for f in ['risk_source','coverage_source','threshold_mode','class_name','selection_seed','model','method'])
            target[k+(('C','C') if consensus else (r['evaluator'],r['reference']))]=int(r['confirmed'])
    stream=Stream('primary_diagnostic.csv');errors=Stream('actual_error.csv');support_stream=Stream('primary_support.csv')
    totals=defaultdict(lambda:defaultdict(float));reference_cache={};reproduced=scalar_rows=0
    for risk in ['A','B']:
        for c in previous.CLASSES:
            dq=risks['D',risk,c]['tiers']
            for source in VIEWS:
                dx,dy,correct=raw['D',source,c];w=strata_counts(dx,dy,dq)[0];st=structure(dx,dy,dq,w)
                reference_cache[risk,source,c]=(st,{m:gap_bundle(dx,dy,dq,correct[m].mean(0),st) for m in old.MODELS})
    for risk in ['A','B']:
        for cover in ['A','B']:
            for mode in previous.MODES:
                for c in previous.CLASSES:
                    q=risks['B',risk,c]['tiers'];correct=raw['B','A',c][2]
                    for seed in previous.SEEDS:
                        selected={m:z[previous.key(risk,cover,mode,c,m,seed)] for m in METHODS}
                        local={};common={}
                        for source in VIEWS:
                            x,y,_=raw['B',source,c];weights=strata_counts(x,y,q)[0]
                            keep=np.logical_and.reduce([(strata_counts(x[ix],y[ix],q[ix])[0]>=5)&
                                (strata_counts(x[ix],y[ix],q[ix])[1]>=5) for ix in selected.values()])
                            sts={m:structure(x[ix],y[ix],q[ix],weights,keep) for m,ix in selected.items()}
                            common[source]=np.logical_and.reduce([st['eligible'] for st in sts.values()])
                            for method,ix in selected.items():
                                st=sts[method]
                                for model in old.MODELS:
                                    g=gap_bundle(x[ix],y[ix],q[ix],correct[model].mean(0)[ix],st)
                                    local[source,method,model]=(st,g)
                                    if seed==previous.SEEDS[0] and model==old.MODELS[0]:
                                        scalar_check(x[ix],y[ix],q[ix],correct[model].mean(0)[ix],st,g);scalar_rows+=1
                        for model in old.MODELS:
                            for method,ix in selected.items():
                                acc=correct[model].mean(0)
                                for tier in [0,1,2,3,4,5]:
                                    jj=ix if tier==0 else ix[q[ix]==tier]
                                    errors.add(dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,class_name=c,
                                        selection_seed=seed,model=model,method=method,Q=tier,n=len(jj),error_pct=100*float(1-acc[jj].mean())))
                            for ev,ref in AXES:
                                ds,dg=reference_cache[risk,ref,c]
                                universe=common[ev]&ds['eligible']
                                support_stream.add(dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,class_name=c,
                                    selection_seed=seed,model=model,evaluator=ev,reference=ref,common_n=int(universe.sum()),
                                    mean_B_retained_Q=float(np.mean(local[ev,'R',model][0]['keep'].sum(0)[universe])) if universe.any() else 0))
                                for method,ix in selected.items():
                                    st,g=local[ev,method,model]
                                    x,y,_=raw['B',ev,c];dx,dy,dc=raw['D',ref,c]
                                    baseline=previous.prior.reference_stats(x[ix],y[ix],correct[model].mean(0)[ix])[4]
                                    dbase=previous.prior.reference_stats(dx,dy,dc[model].mean(0))[4]
                                    ek=(risk,cover,mode,c,str(seed),model,method,ev,ref)
                                    assert int((baseline&dbase).sum())==target[ek];reproduced+=1
                                    for path,(bp,dp) in PATHS.items():
                                        confirmed=universe&(g[bp]<=-10+1e-12)&(dg[model][dp]<=-10+1e-12)
                                        row=dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,class_name=c,
                                            selection_seed=seed,model=model,evaluator=ev,reference=ref,method=method,path=path,
                                            common_n=int(universe.sum()),confirmed=int(confirmed.sum()))
                                        stream.add(row)
                                        k=(risk,cover,mode,ev,ref,path,method,seed)
                                        totals[k]['confirmed']+=row['confirmed'];totals[k]['common_n']+=row['common_n']
                    cc.log(OUT,'primary_class',risk=risk,coverage=cover,threshold_mode=mode,class_name=c,rows=stream.count)
    stream.close();errors.close();support_stream.close()
    primary_summarize(totals)
    dump('primary_validation.json',dict(status='pass',primary_rows=stream.count,original_raw_diagnostic_rows_reproduced=reproduced,
        legacy_A_adjusted_rows_reproduced=legacy,scalar_gap_bundles_checked=scalar_rows,
        same_universe_across_paths_and_methods=True,support_masks_use_no_correctness_values=True))
    cc.freeze(OUT/'primary_output_freeze.json',dict(files={str(OUT/n):study.sha(OUT/n) for n in
        ['primary_diagnostic.csv','actual_error.csv','primary_support.csv','primary_per_start.csv','primary_summary.csv','primary_validation.json']}))


def primary_summarize(totals):
    per=[dict(zip(['risk_source','coverage_source','threshold_mode','evaluator','reference','path','method','selection_seed'],k),**v) for k,v in totals.items()]
    write('primary_per_start.csv',per)
    indexed={k:v for k,v in totals.items()};summary=[]
    for risk in ['A','B']:
        for cover in ['A','B']:
            for mode in previous.MODES:
                for ev,ref in AXES:
                    for path in PATHS:
                        for method,comparator in COMPARISONS:
                            left=[indexed[risk,cover,mode,ev,ref,path,method,s] for s in previous.SEEDS]
                            right=[indexed[risk,cover,mode,ev,ref,path,comparator,s] for s in previous.SEEDS]
                            assert all(a['common_n']==b['common_n'] for a,b in zip(left,right))
                            delta=np.array([a['confirmed']-b['confirmed'] for a,b in zip(left,right)])
                            summary.append(dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,evaluator=ev,reference=ref,
                                path=path,method=method,comparator=comparator,mean_difference=float(delta.mean()),
                                method_confirmed=float(np.mean([r['confirmed'] for r in left])),
                                comparator_confirmed=float(np.mean([r['confirmed'] for r in right])),
                                common_n=float(np.mean([r['common_n'] for r in left])),positive_starts=int((delta>0).sum()),
                                ties=int((delta==0).sum()),negative_starts=int((delta<0).sum())))
    write('primary_summary.csv',summary)


def matched():
    assert not (OUT/'completion.json').exists()
    cc.verify_freeze(OUT/'input_freeze.json');cc.verify_freeze(OUT/'primary_output_freeze.json')
    risks,raw=load();z=np.load(previous.OUT/'selections.npz',allow_pickle=False)
    totals=defaultdict(lambda:defaultdict(float));balance=Stream('matched_balance.csv')
    smoke={};independent=empty=0
    for ai,(risk,cover,mode) in enumerate(ANCHORS):
        for ci,c in enumerate(previous.CLASSES):
            q=risks['B',risk,c]['tiers'];correctness=raw['B','A',c][2]
            for seed in previous.SEEDS:
                original={m:z[previous.key(risk,cover,mode,c,m,seed)] for m in METHODS}
                for mi,model in enumerate(old.MODELS):
                    for pi,pseed in enumerate(cc.old.SEEDS):
                        correct=correctness[model][pi]
                        original_stats={};d_stats={}
                        for source in VIEWS:
                            x,y,_=raw['B',source,c];dx,dy,dc=raw['D',source,c]
                            d_stats[source]=previous.prior.reference_stats(dx,dy,dc[model][pi].astype(float))
                            for m,ix in original.items():
                                original_stats[source,m]=previous.prior.reference_stats(x[ix],y[ix],correct[ix].astype(float))
                        for repeat in range(REPEATS):
                            rng=np.random.default_rng(np.random.SeedSequence([2026111207,ai,seed,ci,mi,pi,repeat]))
                            ranks=rng.random(len(q))
                            chosen,cells=match_outcomes(original,q,correct,ranks)
                            n=len(chosen['R']);nerr=int((~correct[chosen['R']]).sum());empty+=int(n==0)
                            if seed==previous.SEEDS[0] and mi==pi==repeat==0:
                                for m,ix in chosen.items():
                                    smoke[f'{ai}__{c}__{m}']=ix
                                    for tier,outcome,wanted in cells:
                                        got=sum(int(q[int(j)])==tier and bool(correct[int(j)])==outcome for j in ix.tolist())
                                        assert got==wanted
                                    independent+=1
                            row=dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,class_name=c,selection_seed=seed,
                                model=model,prediction_seed=pseed,repeat=repeat,matched_n=n,matched_errors=nerr,empty_match=n==0,
                                **{m+'_original_errors':int((~correct[ix]).sum()) for m,ix in original.items()},
                                **{f'Q{tier}_matched_n':sum(v for q0,truth,v in cells if q0==tier) for tier in range(1,6)})
                            balance.add(row)
                            new_stats={}
                            for source in VIEWS:
                                x,y,_=raw['B',source,c]
                                for m,ix in chosen.items():
                                    new_stats[source,m]=previous.prior.reference_stats(x[ix],y[ix],correct[ix].astype(float))
                            for ev,ref in AXES:
                                universe=np.logical_and.reduce([original_stats[ev,m][3] for m in METHODS]+[new_stats[ev,m][3] for m in METHODS])&d_stats[ref][3]
                                original_common=np.logical_and.reduce([original_stats[ev,m][3] for m in METHODS])&d_stats[ref][3]
                                for phase,stats,sets in [('original',original_stats,original),('matched',new_stats,chosen)]:
                                    for m,ix in sets.items():
                                        confirmed=universe&stats[ev,m][4]&d_stats[ref][4]
                                        k=(risk,cover,mode,seed,pseed,repeat,ev,ref,phase,m)
                                        t=totals[k];t['confirmed']+=int(confirmed.sum());t['common_n']+=int(universe.sum())
                                        t['original_common_n']+=int(original_common.sum());t['selected_n']+=len(ix)
                                        t['error_targets']+=int((~correct[ix]).sum())
                                        t['error_pct_sum']+=100*float((~correct[ix]).mean()) if len(ix) else 0
            cc.log(OUT,'matched_class',risk=risk,coverage=cover,threshold_mode=mode,class_name=c,balance_units=balance.count)
    balance.close();np.savez_compressed(OUT/'matched_smoke_indices.npz',**smoke)
    names=['risk_source','coverage_source','threshold_mode','selection_seed','prediction_seed','repeat','evaluator','reference','phase','method']
    perdraw=[]
    for k,t in totals.items():
        perdraw.append(dict(zip(names,k),confirmed=t['confirmed'],common_n=t['common_n'],original_common_n=t['original_common_n'],
            mean_selected_n=t['selected_n']/30,error_pct=t['error_pct_sum']/30,total_error_targets=t['error_targets']))
    write('matched_per_draw.csv',perdraw)
    perstart=defaultdict(list)
    for row in perdraw:
        k=tuple(row[f] for f in ['risk_source','coverage_source','threshold_mode','selection_seed','evaluator','reference','phase','method'])
        perstart[k].append(row)
    averages={}
    start_rows=[]
    for k,rr in perstart.items():
        assert len(rr)==3*REPEATS
        averages[k]={f:float(np.mean([r[f] for r in rr])) for f in
            ['confirmed','common_n','original_common_n','mean_selected_n','error_pct','total_error_targets']}
        start_rows.append(dict(zip(['risk_source','coverage_source','threshold_mode','selection_seed','evaluator','reference','phase','method'],k),**averages[k]))
    write('matched_per_start.csv',start_rows)
    summary=[]
    for risk,cover,mode in ANCHORS:
        for ev,ref in AXES:
            for phase in ['original','matched']:
                for method,comparator in COMPARISONS:
                    left=[averages[risk,cover,mode,s,ev,ref,phase,method] for s in previous.SEEDS]
                    right=[averages[risk,cover,mode,s,ev,ref,phase,comparator] for s in previous.SEEDS]
                    for a,b in zip(left,right):
                        assert abs(a['common_n']-b['common_n'])<1e-9
                        if phase=='matched':
                            assert abs(a['error_pct']-b['error_pct'])<1e-9
                            assert abs(a['mean_selected_n']-b['mean_selected_n'])<1e-9
                    delta=np.array([a['confirmed']-b['confirmed'] for a,b in zip(left,right)])
                    summary.append(dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,evaluator=ev,reference=ref,
                        phase=phase,method=method,comparator=comparator,mean_difference=float(delta.mean()),
                        method_confirmed=float(np.mean([r['confirmed'] for r in left])),
                        comparator_confirmed=float(np.mean([r['confirmed'] for r in right])),
                        common_n=float(np.mean([r['common_n'] for r in left])),
                        original_common_n=float(np.mean([r['original_common_n'] for r in left])),
                        mean_selected_n=float(np.mean([r['mean_selected_n'] for r in left])),
                        method_error_pct=float(np.mean([r['error_pct'] for r in left])),
                        comparator_error_pct=float(np.mean([r['error_pct'] for r in right])),
                        positive_starts=int((delta>1e-10).sum()),ties=int((np.abs(delta)<=1e-10).sum()),negative_starts=int((delta<-1e-10).sum())))
    write('matched_summary.csv',summary)
    dump('matched_validation.json',dict(status='pass',balance_units=balance.count,empty_matches=empty,
        independent_smoke_index_sets=independent,all_Q_by_error_correct_counts_matched=True,
        all_matched_members_subset_of_original=True,matched_error_rates_and_sample_sizes_equal=True,
        common_state_universe_identical_original_and_matched=True,uses_outcomes_to_subsample=True,
        changes_selected_N_and_fine_quotas=True,diagnostic_rule='single prediction seed, not primary three-seed mean'))
    cc.verify_freeze(OUT/'input_freeze.json')
    cc.freeze(OUT/'matched_output_freeze.json',dict(files={str(OUT/n):study.sha(OUT/n) for n in
        ['matched_balance.csv','matched_smoke_indices.npz','matched_per_draw.csv','matched_per_start.csv','matched_summary.csv','matched_validation.json']}))
    cc.log(OUT,'matched_complete',balance_units=balance.count,empty_matches=empty)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','primary','matched'])
    args=parser.parse_args();{'prepare':prepare,'primary':primary,'matched':lambda: matched()}[args.stage]()
