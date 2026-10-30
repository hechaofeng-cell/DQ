"""Actual matched-baseline, sensitivity and externally confirmed diagnostic experiments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from coverage_validation_common_20261018 import *

METHODS=['expanded_greedy','shared_greedy','static_rarity','vit_kcenter','same_cell_random']


def self_check():
    ctx=load_contexts()
    for c,x in ctx.items():
        a=choose(x,200,'expanded_greedy')
        assert np.array_equal(a,x['original']),f'Cannot reproduce primary greedy for {c}'
        for budget in [50,100,200,300]:
            for method in METHODS[:-1]:validate_selection(x,choose(x,budget,method),budget)
        # Direct objective increment equals the marginal formula, including cap saturation.
        counts=np.array([0,1,19,20,21]);present=np.array([1,1,1,1,1])
        actual=(np.log1p(np.minimum(counts+present,20))-np.log1p(np.minimum(counts,20))).sum()
        formula=(np.where(counts<20,np.log(counts+2)-np.log(counts+1),0)*present).sum()
        assert np.isclose(actual,formula)
    dump(OUT/'implementation_checks.json',dict(status='pass',original_primary_selection_reproduced=4,
        all_budget_method_quotas=True,capped_log_increment=True))


def select_stage():
    contexts=load_contexts(); results=[]; choices={}; quota_rows=[]; run_rows=[]
    paths=[OUT/'protocol.json',Path(__file__),ROOT/'scripts/coverage_validation_common_20261018.py',
           BASE/'manifest.csv',SOURCE/'risk_scores_and_tiers.csv',ROOT/'artifacts/sixclass_independent_confirmation_20261004/embeddings/vit_b16_candidates.npz']
    paths += [SOURCE/f'{c}_{name}' for c in CLASSES for name in ['state_matrix.npz','state_dictionary.json']]
    freeze(OUT/'analysis_input_freeze.json',{str(p):sha(p) for p in paths})
    for c,ctx in contexts.items():
        for budget in [50,100,200,300]:
            q=quotas(ctx,budget)
            quota_rows.extend(dict(class_name=c,budget=budget,Q=k[0],midpoint=k[1]/1e8,width=k[2]/1e8,quota=int(q[j])) for j,k in enumerate(ctx['keys']))
            settings=[(m,20,'state') for m in METHODS]
            if budget==200:settings += [('expanded_greedy',10,'state'),('expanded_greedy',30,'state'),('expanded_greedy',20,'field')]
            for method,cap,weight in settings:
                tag=f'{c}__b{budget}__{method}__k{cap}__{weight}'
                sel=random_choices(ctx,budget) if method=='same_cell_random' else choose(ctx,budget,method,cap,weight)[None]
                for ss in sel:validate_selection(ctx,ss,budget)
                choices[tag]=sel
                # All methods evaluated at common caps, including cap sensitivity selected sets.
                for evalcap in ([10,20,30] if budget==200 else [20]):
                    cv=coverage(ctx,sel,evalcap)
                    for ns,met in cv.items():
                        for metric,v in met.items():
                            rec=dict(class_name=c,budget=budget,method=method,selection_cap=cap,weighting=weight,
                                evaluation_cap=evalcap,namespace=ns,metric=metric,repeats=len(v),mean_pct=float(v.mean()),
                                p025_pct=float(np.quantile(v,.025)),p975_pct=float(np.quantile(v,.975)))
                            results.append(rec)
                            if evalcap==20 and ns=='all':
                                run_rows.extend(dict(class_name=c,budget=budget,method=method,selection_cap=cap,weighting=weight,metric=metric,
                                                     repeat=i,value_pct=float(value)) for i,value in enumerate(v))
        log('selection_class_complete',class_name=c)
    np.savez_compressed(OUT/'selections.npz',**choices)
    write(OUT/'coverage_metrics.csv',results);write(OUT/'coverage_runs.csv',run_rows);write(OUT/'quotas.csv',quota_rows)
    freeze(OUT/'selection_before_prediction_freeze.json',{str(OUT/n):sha(OUT/n) for n in ['selections.npz','coverage_metrics.csv','quotas.csv','analysis_input_freeze.json']})
    dump(OUT/'selection_audit.json',dict(status='pass',classifier_predictions_read=False,settings=len(choices),risk_cell_quotas_verified=True,
        primary_current_selection_reproduced=True,random_repeats=1000,independent_budgets=[50,100,200,300]))


def primary_choices():
    with np.load(OUT/'selections.npz',allow_pickle=False) as z:
        return {(c,m):z[f'{c}__b200__{m}__k20__state'].copy() for c in CLASSES for m in METHODS}


def condition_candidates(ctx,sel,correct,scope):
    X=ctx['matrix'][sel];count=X.sum(0);rec=np.divide(X.T@correct[sel],count,out=np.full(len(count),np.nan),where=count>0)
    ref=float(correct[sel].mean());delta=rec-ref
    valid=(count>=20)&(delta<=-.1+1e-12)
    if scope=='specific':valid[:71]=False
    ii=sorted(np.flatnonzero(valid),key=lambda j:(delta[j],tuple(ctx['states'][j])))[:3]
    return [dict(state_index=int(j),namespace=ctx['states'][j][0],field=ctx['states'][j][1],value=ctx['states'][j][2],
        discovery_support=int(count[j]),discovery_recall=float(rec[j]),discovery_class_recall=ref,discovery_delta_pp=float(delta[j]*100)) for j in ii]


def discover_stage():
    # No external prediction file is opened in this stage.
    contexts=load_contexts(); choices=primary_choices();rows=manifest();pred,hashes=predictions('B',rows)
    records=[];groups=[]
    for c,ctx in contexts.items():
        positions=np.array([i for i,r in enumerate(rows) if r['class_name']==c]);label=ALL_CLASSES.index(c)
        for model in MODELS:
            correct=(pred[model][:,positions]==label).mean(0)
            for method in METHODS:
                sets=choices[c,method][:100] if method=='same_cell_random' else choices[c,method]
                for rep,sel in enumerate(sets):
                    for scope in ['all','specific']:
                        key=dict(class_name=c,model=model,method=method,repeat=rep,scope=scope)
                        found=condition_candidates(ctx,sel,correct,scope)
                        groups.append(dict(**key,proposed=len(found)))
                        for rank,f in enumerate(found,1):records.append(dict(**key,rank=rank,**f))
    write(OUT/'discovered_conditions.csv',records)
    write(OUT/'discovery_groups.csv',groups)
    freeze_path=OUT/'conditions_before_external_predictions.json'
    freeze_time=json.loads(freeze_path.read_text())['time'] if freeze_path.exists() else now()
    freeze(freeze_path,dict(time=freeze_time,protocol_sha256=sha(OUT/'protocol.json'),
        selection_sha256=sha(OUT/'selections.npz'),conditions_sha256=sha(OUT/'discovered_conditions.csv'),
        group_sha256=sha(OUT/'discovery_groups.csv'),external_predictions_read=False,source_prediction_sha256=hashes))
    log('conditions_frozen',conditions=len(records),groups=len(groups),external_predictions_read=False)


def reliability_stage():
    contexts=load_contexts(); selections=primary_choices();rows=[]
    # Actual repeatability is computed only from genuine new model outputs.
    missing=[]
    for row in read(OUT/'repeat_manifest.csv'):
        c=row['class_name'];sid=row['sample_id'];new=OUT/'annotations/repeat'/c/f'{sid}_parsed.json'
        if not new.exists():missing.append(sid);continue
        old=json.loads((SOURCE/'annotations_v2'/c/f'{sid}_parsed.json').read_text())['features']
        fresh=json.loads(new.read_text())['features']; schema=json.loads((OUT/'dictionaries'/f'{c}.json').read_text())['features']
        ctx=contexts[c];i=ctx['ids'].tolist().index(sid)
        for f in schema:
            field=f['feature_id'];a=old[field];b=fresh[field];aa=set(a) if isinstance(a,list) else {a};bb=set(b) if isinstance(b,list) else {b}
            rows.append(dict(class_name=c,sample_id=sid,Q=row['Q'],field=field,value_type=f['value_type'],
                originally_selected=int(i in selections[c,'expanded_greedy'][0]),exact_agreement=int(aa==bb),
                jaccard=len(aa&bb)/len(aa|bb),old_unknown=int(aa=={'unknown'}),new_unknown=int(bb=={'unknown'}),
                original_value=json.dumps(a,ensure_ascii=False),repeat_value=json.dumps(b,ensure_ascii=False)))
    if rows:write(OUT/'annotation_repeat_pairs.csv',rows)
    summary=[]
    for c in CLASSES:
        rr=[r for r in rows if r['class_name']==c]
        if rr:summary.append(dict(class_name=c,targets=len({r['sample_id'] for r in rr}),fields=len(rr),
            exact_agreement_pct=100*np.mean([r['exact_agreement'] for r in rr]),mean_jaccard=np.mean([r['jaccard'] for r in rr]),
            unknown_agreement_pct=100*np.mean([r['old_unknown']==r['new_unknown'] for r in rr]),human_accuracy='not_measured'))
    if summary:write(OUT/'annotation_repeat_summary.csv',summary)
    dump(OUT/'reliability_audit.json',dict(repeat_status='pass' if not missing else 'incomplete',missing=missing,
        human_accuracy='pending_real_human_labels',repeat_is_accuracy=False,semantic_truth_available=False))


def sensitivity_stage():
    contexts=load_contexts();rows=[]; choices=primary_choices();rng=np.random.default_rng(20261018)
    for c,ctx in contexts.items():
        reference=set(choices[c,'expanded_greedy'][0].tolist())
        for rate in [.05,.1,.2]:
            for rep in range(20):
                modified=ctx['matrix'].copy()
                # Only new specific fields are perturbed; risk inputs are always frozen.
                for ix in ctx['field_indices']:
                    if ix[0]<71:continue
                    hide=rng.random(400)<rate
                    modified[np.ix_(hide,ix)]=False
                sel=choose(ctx,200,'expanded_greedy',matrix=modified)
                cv=coverage(ctx,sel)['all']
                rows.append(dict(class_name=c,missingness_rate=rate,repeat=rep,
                    presence_pct=float(cv['presence_pct'][0]),support_completion_pct=float(cv['support_completion_pct'][0]),
                    retained_selected_pct=100*len(reference & set(sel.tolist()))/200))
        log('synthetic_missingness_complete',class_name=c)
    write(OUT/'synthetic_missingness.csv',rows)


def external_contexts():
    rows=read(OUT/'external_manifest.csv');states={c:json.loads((SOURCE/f'{c}_state_dictionary.json').read_text())['states'] for c in CLASSES};out={}
    for c in CLASSES:
        cr=[r for r in rows if r['class_name']==c];X=np.zeros((len(cr),len(states[c])),bool)
        for i,row in enumerate(cr):
            sid=row['sample_id'];domain='car' if c=='car' else 'old'
            native=json.loads((EXTERNAL/f'annotations/local/{domain}/parsed/{sid}.json').read_text())
            assert native['image_id']==sid
            shared=native['features'];specific_path=OUT/f'annotations/external/{c}/{sid}_parsed.json'
            specific=json.loads(specific_path.read_text())
            assert specific['original_sha256']==row['image_sha256'] and specific['dictionary_sha256']==sha(OUT/'dictionaries'/f'{c}.json')
            for j,(ns,field,value) in enumerate(states[c]):
                if ns=='shared' and field=='target_scale':
                    area=float(row['bbox_area_ratio']);v='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large'
                else:v=(shared if ns=='shared' else specific['features'])[field]
                X[i,j]=(value in v) if isinstance(v,list) else value==v
        out[c]=dict(rows=cr,matrix=X,states=states[c])
        np.savez_compressed(OUT/f'external_{c}_matrix.npz',sample_ids=np.array([r['sample_id'] for r in cr]),matrix=X)
    return rows,out


def confirm_stage():
    frozen=json.loads((OUT/'conditions_before_external_predictions.json').read_text())
    assert sha(OUT/'discovered_conditions.csv')==frozen['conditions_sha256']
    assert sha(OUT/'protocol.json')==frozen['protocol_sha256']
    assert sha(OUT/'selections.npz')==frozen['selection_sha256']
    external_rows,contexts=external_contexts()
    # This is the first access to external prediction logits by this experiment.
    pred,hashes=predictions('D',external_rows);conditions=read(OUT/'discovered_conditions.csv');groups=read(OUT/'discovery_groups.csv')
    summary=[];detail=[];cache={};rng=np.random.default_rng(20261019)
    for c,ctx in contexts.items():
        positions=[i for i,r in enumerate(external_rows) if r['class_name']==c];label=ALL_CLASSES.index(c)
        X=ctx['matrix'];n=X.sum(0)
        resamples=rng.multinomial(len(X),np.full(len(X),1/len(X)),size=1000)
        for model in MODELS:
            correct=(pred[model][:,positions]==label).mean(0)
            ref=float(correct.mean());rec=np.divide(X.T@correct,n,out=np.full(len(n),np.nan),where=n>0)
            boot_n=resamples@X;boot_corr=resamples@(X*correct[:,None]);boot_ref=(resamples@correct)/len(X)
            boot_rec=np.divide(boot_corr,boot_n,out=np.full(boot_corr.shape,np.nan),where=boot_n>0)
            for j in set(int(r['state_index']) for r in conditions if r['class_name']==c and r['model']==model):
                delta=float((rec[j]-ref)*100) if n[j] else None
                values=100*(boot_rec[:,j]-boot_ref);values=values[np.isfinite(values)]
                low,high=np.quantile(values,[.025,.975]) if len(values) else [None,None]
                cache[c,model,j]=dict(external_support=int(n[j]),external_recall=float(rec[j]) if n[j] else None,
                    external_class_recall=ref,external_delta_pp=delta,bootstrap_p025_pp=float(low) if low is not None else None,
                    bootstrap_p975_pp=float(high) if high is not None else None,
                    status='insufficient_support' if n[j]<20 else ('direction_confirmed' if delta<0 else 'not_confirmed'))
    for r in conditions:detail.append(dict(**r,**cache[r['class_name'],r['model'],int(r['state_index'])]))
    index={}
    keyfields=['class_name','model','method','repeat','scope']
    for r in detail:index.setdefault(tuple(r[k] for k in keyfields),[]).append(r)
    for g in groups:
        rr=index.get(tuple(g[k] for k in keyfields),[]);count=Counter(r['status'] for r in rr)
        eligible=count['direction_confirmed']+count['not_confirmed'];proposed=int(g['proposed'])
        summary.append(dict(**g,evaluable=eligible,confirmed=count['direction_confirmed'],not_confirmed=count['not_confirmed'],
            insufficient=count['insufficient_support'],confirmed_among_evaluable_pct=100*count['direction_confirmed']/eligible if eligible else '',
            unsupported_pct=100*count['insufficient_support']/proposed if proposed else '',
            confirmed_ci_below_zero=sum(r['status']=='direction_confirmed' and r['bootstrap_p975_pp']<0 for r in rr)))
    write(OUT/'external_condition_results.csv',detail);write(OUT/'external_diagnostic_groups.csv',summary)
    aggregates=[]
    for scope in ['all','specific']:
        for method in METHODS:
            rr=[r for r in summary if r['method']==method and r['scope']==scope];replicates=100 if method=='same_cell_random' else 1
            counts={k:sum(float(r[k]) for r in rr)/replicates for k in ['proposed','evaluable','confirmed','not_confirmed','insufficient','confirmed_ci_below_zero']}
            aggregates.append(dict(scope=scope,method=method,**counts,
                confirmation_pct=100*counts['confirmed']/counts['evaluable'] if counts['evaluable'] else '',
                unsupported_pct=100*counts['insufficient']/counts['proposed'] if counts['proposed'] else '',
                independent_units='4 classes x 5 model architectures; conditions correlated',random_repetitions=replicates))
    write(OUT/'external_diagnostic_summary.csv',aggregates)
    dump(OUT/'external_confirmation_audit.json',dict(status='pass',targets=len(external_rows),models=5,seeds=3,
        condition_freeze_verified=True,same_specific_dictionary=True,class_support=200,
        retrospective=True,human_label_truth=False,prediction_sha256=hashes,
        bootstrap_unit='image target (all images unique within confirmation sample)',bootstrap_repeats=1000))
    log('external_confirmation_complete',conditions=len(detail))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','select','discover','sensitivity','reliability','confirm']);args=parser.parse_args()
    {'check':self_check,'select':select_stage,'discover':discover_stage,'sensitivity':sensitivity_stage,'reliability':reliability_stage,'confirm':confirm_stage}[args.phase]()
