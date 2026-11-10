"""Execute the frozen six-method experiment, then prediction-blind sensitivities."""
import json
import resource
import time

import numpy as np

import conditional_coverage_common as cc


TRACE_FIELDS = ['class_name','method','seed','step','outgoing_id','incoming_id','cell',
 'occurrence_before','occurrence_after','support_before','support_after','deficit_before',
 'deficit_after','utility_before','utility_after','legal_pairs','cumulative_pair_scores','acceptance']


def run_method(ctx, method, budget, cap, thresholds, seed, class_index):
    start = time.perf_counter()
    trace = []
    if method in ['G','SR','KC']:
        initial = None
        selected, work = cc.baseline(ctx, budget, method, cap)
    else:
        initial = cc.initial_selection(ctx, budget, seed, class_index)
        if method == 'R':
            selected = initial.copy()
            work = dict(add_candidate_scores=0,pair_scores=0,full_scans=0,accepted_swaps=0,stop_reason='complete')
        else:
            selected, work, trace = cc.swap_select(ctx, initial, thresholds, cap=cap,
                                                  conditional=method=='CS')
    final = cc.coverage_rows(ctx, selected, cap)
    elapsed = time.perf_counter() - start
    initial_cv = cc.coverage_rows(ctx, initial, cap)['all'] if initial is not None else None
    work = dict(work)
    work.pop('final_pass',None)
    record = dict(method=method, seed=seed, budget=budget, support_cap=cap, seconds=elapsed,
        **work, final_pass=cc.passes(final['all'],thresholds),
        initial_pass=cc.passes(initial_cv,thresholds) if initial_cv else '',
        initial_occurrence=initial_cv['occurrence'] if initial_cv else '',
        initial_support_completion=initial_cv['support_completion'] if initial_cv else '',
        threshold_occurrence=thresholds['occurrence'],
        threshold_support_completion=thresholds['support_completion'],
        process_high_water_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    for ns,metrics in final.items():
        record.update({f'{ns}_{metric}':value for metric,value in metrics.items()})
    return selected, initial, record, trace


def smoke(out):
    cc.verify_freeze(out/'threshold_freeze.json')
    ctxs = cc.load_contexts()
    thresholds = json.loads((out/'thresholds.json').read_text())
    records = []
    for ci,(c,ctx) in enumerate(ctxs.items()):
        for method, old_name in [('G','expanded_greedy'),('SR','static_rarity'),('KC','vit_kcenter')]:
            a,_ = cc.baseline(ctx,200,method)
            assert np.array_equal(a,cc.old.choose(ctx,200,old_name))
        for seed in range(2026110000,2026110005):
            a,initial,r,ta = run_method(ctx,'CS',200,20,thresholds[c],seed,ci)
            b,initial2,s,tb = run_method(ctx,'AS',200,20,thresholds[c],seed,ci)
            assert np.array_equal(initial,initial2)
            assert ta == tb[:len(ta)]
            records.extend([dict(class_name=c,**r),dict(class_name=c,**s)])
        cc.log(out,'historical_smoke_class_complete',class_name=c)
    cc.write(out/'smoke_runs.csv',records)
    cc.dump(out/'smoke_audit.json',dict(status='pass',records=len(records),
        baselines_reproduced=12, CS_AS_prefixes_equal=True,classifier_predictions_read=False))


def primary(out):
    cc.verify_freeze(out/'threshold_freeze.json')
    cc.verify_freeze(out/'input_hashes.json')
    assert json.loads((out/'implementation_checks.json').read_text())['status']=='pass'
    assert json.loads((out/'smoke_audit.json').read_text())['status']=='pass'
    assert not (out/'selected_sets.npz').exists(), 'Primary outputs already exist.'
    plan=json.loads((out/'protocol.json').read_text())
    seeds=plan['phases']['retrospective']['selection_seeds']
    thresholds=json.loads((out/'thresholds.json').read_text())
    contexts=cc.load_contexts()
    rng=np.random.default_rng(plan['execution']['timing_order_seed'])
    records=[];timings=[];traces=[];choices={};initials={}
    for ci,(c,ctx) in enumerate(contexts.items()):
        # NumPy path warmup; not counted as a main repetition.
        run_method(ctx,'R',200,20,thresholds[c],seeds[0],ci)
        tasks=[(method,-1,repeat) for repeat in range(3) for method in ['G','SR','KC']]
        tasks += [(method,seed,repeat) for seed in seeds for method in ['R','CS','AS']
                  for repeat in range(3 if seed in seeds[:10] else 1)]
        order=rng.permutation(len(tasks))
        seen={}
        for number,pos in enumerate(order):
            method,seed,repeat=tasks[pos]
            selected,initial,record,trace=run_method(ctx,method,200,20,thresholds[c],seed,ci)
            key=cc.selection_key(c,method,seed)
            timings.append(dict(class_name=c,method=method,seed=seed,replay=repeat,
                                seconds=record['seconds'],task_order=number))
            if key in seen:
                assert np.array_equal(seen[key],selected), 'Nondeterministic timing replay'
            else:
                seen[key]=selected.copy()
            # Replay zero is the preregistered primary timing and quality record.
            if repeat==0:
                choices[key]=selected
                if initial is not None:
                    initials[f'{c}__s{seed}']=initial
                records.append(dict(class_name=c,**record))
                traces.extend(dict(class_name=c,method=method,seed=seed,**t) for t in trace)
            if (number+1)%50==0:
                cc.log(out,'primary_progress',class_name=c,tasks_done=number+1,tasks_total=len(tasks))
        for seed in seeds:
            s0=initials[f'{c}__s{seed}']
            assert np.array_equal(choices[cc.selection_key(c,'R',seed)],s0)
        class_records=[r for r in records if r['class_name']==c]
        cc.write(out/f'selection_runs_{c}.csv',class_records,
                 list(dict.fromkeys(k for r in class_records for k in r)))
        np.savez_compressed(out/f'selected_sets_{c}.npz',**{k:v for k,v in choices.items() if k.startswith(c+'__')})
        cc.log(out,'primary_class_complete',class_name=c,quality_runs=len(seen))
    # Stable column schema across deterministic and stochastic methods.
    fields=list(dict.fromkeys(k for r in records for k in r))
    records=[{k:r.get(k,'') for k in fields} for r in records]
    cc.write(out/'selection_runs.csv',records,fields)
    cc.write(out/'timing_replays.csv',timings)
    cc.write(out/'swap_trace.csv',traces,TRACE_FIELDS)
    np.savez_compressed(out/'selected_sets.npz',**choices)
    np.savez_compressed(out/'initial_sets.npz',**initials)
    cc.freeze(out/'selection_before_prediction_freeze.json',dict(time=cc.old.now(),
        files={str(out/n):cc.sha(out/n) for n in ['protocol.json','input_hashes.json','threshold_freeze.json',
            'selected_sets.npz','initial_sets.npz','selection_runs.csv','swap_trace.csv','thresholds.json']},
        classifier_predictions_read=False,quality_runs=len(records),final_six_class_targets=1200))
    cc.log(out,'primary_selection_frozen',quality_runs=len(records),classifier_predictions_read=False)


def sensitivities(out):
    cc.verify_freeze(out/'selection_before_prediction_freeze.json')
    assert not (out/'sensitivity_runs.csv').exists(), 'Sensitivity outputs already exist.'
    plan=json.loads((out/'protocol.json').read_text())
    seeds=plan['phases']['retrospective']['selection_seeds']
    frozen=json.loads((out/'thresholds.json').read_text())
    contexts=cc.load_contexts()
    records=[];choices={};riskrows=[]
    with np.load(out/'threshold_witnesses.npz',allow_pickle=False) as z:
        witnesses={k:z[k].copy() for k in z.files}
    for ci,(c,fullctx) in enumerate(contexts.items()):
        settings=[('budget',b,20,.95,'full') for b in [50,100,300]]
        settings += [('threshold',200,20,f,'full') for f in [.9,1.0]]
        settings += [('support_cap',200,k,.95,'full') for k in [10,30]]
        settings += [('dictionary',200,20,.95,'shared')]
        for axis,budget,cap,fraction,dictionary in settings:
            ctx=fullctx
            if dictionary=='shared':
                ctx=fullctx.copy()
                mask=np.array([s[0]=='shared' for s in ctx['states']])
                ctx['matrix']=fullctx['matrix'][:,mask]
                ctx['states']=[s for s in fullctx['states'] if s[0]=='shared']
                ctx['pair_delta']=fullctx['pair_delta'][:,mask]
            witness_cv=cc.coverage_rows(ctx,witnesses[c],cap)['all']
            thresholds=frozen[c] if axis=='budget' else {
                k:witness_cv[k]*fraction for k in ['occurrence','support_completion']}
            tag=f'{axis}__b{budget}__k{cap}__f{fraction}__{dictionary}'
            tasks=[(m,-1) for m in ['G','SR','KC']]
            tasks += [(m,seed) for seed in seeds for m in ['R','CS','AS']]
            # Fixed independent order randomization for exploratory timing only.
            for method,seed in tasks:
                sel,initial,record,trace=run_method(ctx,method,budget,cap,thresholds,seed,ci)
                common=cc.coverage_rows(fullctx,sel,20)['all']
                record.update(class_name=c,axis=axis,dictionary=dictionary,fraction=fraction,
                    common_K20_full_occurrence=common['occurrence'],
                    common_K20_full_support=common['support_completion'])
                records.append(record)
                choices[f'{c}__{tag}__{method}__s{seed}']=sel
            cc.log(out,'sensitivity_setting_complete',class_name=c,setting=tag,quality_runs=len(tasks))
            cc.write(out/f'sensitivity_checkpoint_{c}.csv',
                [r for r in records if r['class_name']==c],
                list(dict.fromkeys(k for r in records for k in r)))
        # Reproduce prespecified missingness stress without claiming semantic accuracy.
        rng=np.random.default_rng(np.random.SeedSequence([2026110598,ci]))
        for rate in [.05,.1,.2]:
            for rep in range(20):
                perturbed=fullctx.copy();X=fullctx['matrix'].copy()
                for ix in fullctx['field_indices']:
                    if fullctx['states'][int(ix[0])][0]!='specific':
                        continue
                    hidden=rng.random(len(X))<rate
                    X[np.ix_(hidden,ix)]=False
                perturbed['matrix']=X
                perturbed['pair_delta']=(X[perturbed['pair_in']].astype(np.int8)-
                                           X[perturbed['pair_out']].astype(np.int8))
                seed=seeds[rep]
                sel,initial,record,trace=run_method(perturbed,'CS',200,20,frozen[c],seed,ci)
                actual=cc.coverage_rows(fullctx,sel)['all']
                riskrows.append(dict(class_name=c,missing_rate=rate,repeat=rep,
                    **record,clean_occurrence=actual['occurrence'],clean_support=actual['support_completion']))
    fields=list(dict.fromkeys(k for r in records for k in r))
    cc.write(out/'sensitivity_runs.csv',records,fields)
    cc.write(out/'missingness_runs.csv',riskrows,list(dict.fromkeys(k for r in riskrows for k in r)))
    np.savez_compressed(out/'sensitivity_sets.npz',**choices)
    cc.freeze(out/'sensitivity_freeze.json',dict(time=cc.old.now(),
        files={str(out/n):cc.sha(out/n) for n in ['sensitivity_runs.csv','missingness_runs.csv','sensitivity_sets.npz']},
        classifier_predictions_used_for_selection=False,one_factor_at_a_time=True))
    cc.log(out,'sensitivity_complete',quality_runs=len(records),missingness_runs=len(riskrows))


if __name__=='__main__':
    args=cc.parsed_args(__doc__,['smoke','primary','sensitivity']).parse_args()
    {'smoke':smoke,'primary':primary,'sensitivity':sensitivities}[args.phase](args.out)
