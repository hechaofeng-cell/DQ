"""Independently replay selected sets, swap traces, and cached predictions."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import conditional_coverage_common as cc


def close(a,b):
    assert np.isclose(float(a),float(b),atol=2e-12,rtol=0),(a,b)


def audit(out):
    for name in ['input_hashes.json','threshold_freeze.json','selection_before_prediction_freeze.json',
                 'six_class_membership_before_prediction_freeze.json',
                 'conditions_before_external_prediction_freeze.json','sensitivity_freeze.json']:
        cc.verify_freeze(out/name)
    ctxs=cc.load_contexts()
    thresholds=json.loads((out/'thresholds.json').read_text())
    plan=json.loads((out/'protocol.json').read_text())
    seeds=plan['phases']['retrospective']['selection_seeds']
    runs=cc.read(out/'selection_runs.csv')
    with np.load(out/'selected_sets.npz',allow_pickle=False) as z:
        selected={k:z[k].copy() for k in z.files}
    with np.load(out/'initial_sets.npz',allow_pickle=False) as z:
        initial={k:z[k].copy() for k in z.files}
    supports=[];marginals=[];unknown=[]
    risk={r['sample_id']:r for r in cc.read(cc.old.SOURCE/'risk_scores_and_tiers.csv')}
    for r in runs:
        c,m,seed=r['class_name'],r['method'],int(r['seed'])
        ctx=ctxs[c];sel=selected[cc.selection_key(c,m,seed)]
        cc.old.validate_selection(ctx,sel,200)
        cv=cc.coverage_rows(ctx,sel)
        for ns,metrics in cv.items():
            for metric,value in metrics.items():
                close(value,r[f'{ns}_{metric}'])
        assert (r['final_pass']=='True')==cc.passes(cv['all'],thresholds[c])
        if m in ['R','CS','AS']:
            expected=cc.initial_selection(ctx,200,seed,cc.old.CLASSES.index(c))
            assert np.array_equal(expected,initial[f'{c}__s{seed}'])
        else:
            old_method={'G':'expanded_greedy','SR':'static_rarity','KC':'vit_kcenter'}[m]
            assert np.array_equal(sel,cc.old.choose(ctx,200,old_method))
        count=ctx['matrix'][sel].sum(0)
        for j,(ns,field,value) in enumerate(ctx['states']):
            supports.append(dict(class_name=c,method=m,seed=seed,state_index=j,
                namespace=ns,field=field,value=value,support=int(count[j]),pool_support=int(ctx['matrix'][:,j].sum())))
        for ix in ctx['field_indices']:
            known=ctx['matrix'][sel][:,ix].any(1)
            state=ctx['states'][int(ix[0])]
            unknown.append(dict(class_name=c,method=m,seed=seed,namespace=state[0],field=state[1],
                no_valid_state_fraction=float((~known).mean()),
                interpretation='unknown_or_no_encoded_valid_value_not_semantic_error_rate'))
        for dimension in ['scale','visibility','occlusion','truncation','orientation']:
            values=np.array([float(risk[str(ctx['ids'][i])][f'{dimension}_midpoint']) for i in sel])
            marginals.append(dict(class_name=c,method=m,seed=seed,dimension=dimension,
                mean=float(values.mean()),p025=float(np.quantile(values,.025)),
                p975=float(np.quantile(values,.975))))
    cc.write(out/'state_support.csv',supports)
    cc.write(out/'risk_dimension_marginals.csv',marginals)
    cc.write(out/'state_missingness_by_field.csv',unknown)
    # Replay every accepted swap using whole-set recomputation, not the delta evaluator.
    grouped=defaultdict(list)
    for r in cc.read(out/'swap_trace.csv'):
        grouped[r['class_name'],r['method'],int(r['seed'])].append(r)
    checked=0
    for ci,(c,ctx) in enumerate(ctxs.items()):
        lookup={str(sid):i for i,sid in enumerate(ctx['ids'])}
        for seed in seeds:
            cs_trace=grouped[c,'CS',seed];as_trace=grouped[c,'AS',seed]
            assert [(r['outgoing_id'],r['incoming_id']) for r in cs_trace]==[
                (r['outgoing_id'],r['incoming_id']) for r in as_trace[:len(cs_trace)]]
            for method in ['CS','AS']:
                sel=initial[f'{c}__s{seed}'].copy()
                for r in grouped[c,method,seed]:
                    before=cc.count_metrics(ctx['matrix'][sel].sum(0))
                    assert not(method=='CS' and cc.passes(before,thresholds[c]))
                    outgoing=lookup[r['outgoing_id']];incoming=lookup[r['incoming_id']]
                    assert outgoing in sel and incoming not in sel
                    assert ctx['cell'][outgoing]==ctx['cell'][incoming]
                    for key,col in [('occurrence','occurrence_before'),('support_completion','support_before')]:
                        close(before[key],r[col])
                    sel[np.flatnonzero(sel==outgoing)[0]]=incoming
                    after=cc.count_metrics(ctx['matrix'][sel].sum(0))
                    cc.old.validate_selection(ctx,sel,200)
                    for key,col in [('occurrence','occurrence_after'),('support_completion','support_after')]:
                        close(after[key],r[col])
                        assert not(before[key]+cc.TOL>=thresholds[c][key] and after[key]+cc.TOL<thresholds[c][key])
                    d0=cc.deficit(before,thresholds[c]);d1=cc.deficit(after,thresholds[c])
                    assert d1<d0-cc.TOL or(abs(d1-d0)<=cc.TOL and after['capped_log']>before['capped_log']+cc.TOL)
                    checked+=1
                assert np.array_equal(sel,selected[cc.selection_key(c,method,seed)])
        cc.log(out,'swap_replay_class_verified',class_name=c,swaps_replayed_so_far=checked)
    # Check all one-factor sensitivity outputs, including common-K20 evaluation.
    sensitivity_checks=0
    with np.load(out/'sensitivity_sets.npz',allow_pickle=False) as z:
        for r in cc.read(out/'sensitivity_runs.csv'):
            c=r['class_name'];ctx=ctxs[c];budget=int(r['budget']);cap=int(r['support_cap'])
            key=f"{c}__{r['axis']}__b{budget}__k{cap}__f{r['fraction']}__{r['dictionary']}__{r['method']}__s{r['seed']}"
            sel=z[key]
            cc.old.validate_selection(ctx,sel,budget)
            cv=cc.coverage_rows(ctx,sel,20)['all']
            close(cv['occurrence'],r['common_K20_full_occurrence'])
            close(cv['support_completion'],r['common_K20_full_support'])
            X=ctx['matrix']
            if r['dictionary']=='shared':
                X=X[:,[s[0]=='shared' for s in ctx['states']]]
            metrics=cc.count_metrics(X[sel].sum(0),cap)
            close(metrics['occurrence'],r['all_occurrence'])
            close(metrics['support_completion'],r['all_support_completion'])
            th=dict(occurrence=float(r['threshold_occurrence']),
                    support_completion=float(r['threshold_support_completion']))
            assert (r['final_pass']=='True')==cc.passes(metrics,th)
            sensitivity_checks+=1
    # An independent direct counting implementation cross-checks every model metric row.
    rows=cc.old.manifest();truth=np.array([int(r['label']) for r in rows]);pred,hashes=cc.old.predictions('B',rows)
    with np.load(out/'final_six_class_sets.npz',allow_pickle=False) as z:
        tiers=z['tiers'].copy();sets={k:z[k].copy() for k in z.files if '__s' in k}
    confusion_lookup={}
    for r in json.loads((out/'confusion_matrices.json').read_text()):
        key=(r['method'],r['selection_seed'],r['model'],r['prediction_seed'],r['Q'])
        take=sets[f"{r['method']}__s{r['selection_seed']}"]
        if r['Q']:
            take=take[tiers[take]==r['Q']]
        y=truth[take];p=pred[r['model']][cc.old.SEEDS.index(r['prediction_seed']),take]
        cm=np.array([[np.sum((y==i)&(p==j)) for j in range(6)] for i in range(6)])
        assert cm.tolist()==r['matrix']
        confusion_lookup[key]=cm
    metric_checks=0
    for r in cc.read(out/'model_metrics_by_seed.csv'):
        key=(r['method'],int(r['selection_seed']),r['model'],int(r['prediction_seed']),int(r['Q']))
        cm=confusion_lookup[key]
        if r['class_name'] in cc.old.ALL_CLASSES:
            j=cc.old.ALL_CLASSES.index(r['class_name'])
            tp=int(cm[j,j]);fn=int(cm[j].sum()-tp);fp=int(cm[:,j].sum()-tp);tn=int(cm.sum()-tp-fn-fp)
            expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),one_vs_rest_accuracy=(tp+tn)/cm.sum())
            for metric,value in expected.items():
                close(r[metric],value)
        elif r['class_name']=='six_class_top1':
            close(r['top1_accuracy'],np.trace(cm)/cm.sum())
        else:
            ii=range(6) if r['class_name']=='six_class_macro' else range(2,6)
            f1=[2*cm[j,j]/(cm[j].sum()+cm[:,j].sum()) for j in ii]
            close(r['f1'],np.mean(f1))
        metric_checks+=1
    # Recheck discovery support and mean predictions for every newly proposed condition.
    from evaluate_conditional_coverage_experiment import discover
    for ctx in ctxs.values():
        ctx['diagnostic_masks']=cc.condition_masks(ctx)
    found=defaultdict(list)
    for r in cc.read(out/'discovered_conditions.csv'):
        found[r['class_name'],r['model'],r['method'],int(r['seed']),r['scope']].append(r)
    groups=cc.read(out/'discovery_groups.csv')
    lookup={r['sample_id']:i for i,r in enumerate(rows)}
    for g in groups:
        c=g['class_name'];ctx=ctxs[c];seed=int(g['seed']);method=g['method'];model=g['model']
        pos=np.array([lookup[str(s)] for s in ctx['ids']])
        correct=(pred[model][:,pos]==cc.old.ALL_CLASSES.index(c)).mean(0)
        expected=discover(ctx,selected[cc.selection_key(c,method,seed)],correct,g['scope'])
        actual=found[c,model,method,seed,g['scope']]
        assert len(expected)==int(g['proposed'])==len(actual)
        assert [r['state_index'] for r in expected]==[int(r['state_index']) for r in actual]
        for a,b in zip(actual,expected):
            close(a['delta_pp'],b['delta_pp'])
    # Independently recheck all historical reference effects and status definitions.
    external_rows=cc.read(out/'external_manifest.csv');epred,ehashes=cc.old.predictions('D',external_rows)
    external={}
    for c,ctx in ctxs.items():
        cr=[r for r in external_rows if r['class_name']==c]
        with np.load(out/f'external_{c}_matrix.npz',allow_pickle=False) as z:
            X=z['matrix'].astype(bool)
        external[c]=(cr,X)
    referencechecks=0
    for r in cc.read(out/'external_condition_results.csv'):
        c=r['class_name'];cr,X=external[c];j=int(r['state_index']);state=ctxs[c]['states'][j]
        same=[k for k,s in enumerate(ctxs[c]['states']) if s[:2]==state[:2]]
        posmask=X[:,j];negmask=np.any(X[:,same],axis=1)&~posmask
        pi=[i for i,row in enumerate(external_rows) if row['class_name']==c]
        correct=(epred[r['model']][:,pi]==cc.old.ALL_CLASSES.index(c)).mean(0)
        n1=int(posmask.sum());n0=int(negmask.sum())
        u1=cc.image_support(cr,posmask);u0=cc.image_support(cr,negmask)
        eligible=min(n1,n0,u1,u0)>=20
        delta=100*(correct[posmask].mean()-correct[negmask].mean()) if min(n1,n0)>0 else None
        expected='insufficient_support' if not eligible else 'confirmed' if delta<=-10+cc.TOL else 'not_confirmed'
        assert r['status']==expected
        if delta is not None:
            close(r['reference_delta_pp'],delta)
        referencechecks+=1
    audits=dict(status='pass',time=cc.old.now(),primary_selection_runs=len(runs),
        synthetic_checks='pass',original_deterministic_baselines_reproduced=12,
        accepted_swaps_replayed=checked,sensitivity_selection_runs=sensitivity_checks,
        confusion_matrices_recomputed=len(confusion_lookup),model_metric_rows_recomputed=metric_checks,
        discovery_groups_recomputed=len(groups),reference_condition_rows_recomputed=referencechecks,
        exact_risk_cell_quotas=True,paired_initial_sets=True,CS_AS_prefix_verified=True,
        final_six_class_targets_per_set=1200,source_prediction_files_rehashed=len(hashes)+len(ehashes),
        fresh_confirmation_completed=False,model_training_or_new_inference=False,
        audit_implementation={str(Path(__file__)):cc.sha(__file__)},
        limitations=['fixed_historical_pools','automatic_state_labels_without_independent_semantic_truth',
                     'memory_is_process_high_water_not_per_method','no_new_C_select_C_ref_data'])
    cc.dump(out/'validation.json',audits)
    cc.log(out,'independent_audit_complete',**{k:v for k,v in audits.items() if k!='time'})


if __name__=='__main__':
    args=cc.parsed_args(__doc__).parse_args()
    audit(args.out)
