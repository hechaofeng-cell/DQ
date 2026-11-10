"""Evaluate frozen six-class sets and freeze newly defined diagnostic conditions."""
import json
from pathlib import Path

import numpy as np

import conditional_coverage_common as cc


CONDITION_FIELDS=['class_name','model','method','seed','scope','rank','state_index','namespace',
 'field','value','state_targets','control_targets','state_images','control_images',
 'state_recall','control_recall','delta_pp']


def confusion_metrics(cm):
    tp=np.diag(cm).astype(float)
    fp=cm.sum(0)-tp
    fn=cm.sum(1)-tp
    total=int(cm.sum())
    tn=total-tp-fp-fn
    def divide(a,b):
        return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    return dict(precision=divide(tp,tp+fp),recall=divide(tp,tp+fn),
                f1=divide(2*tp,2*tp+fp+fn),one_vs_rest_accuracy=(tp+tn)/total,
                tp=tp,fp=fp,fn=fn,tn=tn)


def discover(ctx,selected,correct,scope):
    X,controls=ctx['diagnostic_masks'] if 'diagnostic_masks' in ctx else cc.condition_masks(ctx)
    A=X[selected];B=controls[selected]
    positive=A.sum(0);negative=B.sum(0)
    pa=np.divide(A.T@correct[selected],positive,out=np.full(X.shape[1],np.nan),where=positive>0)
    pb=np.divide(B.T@correct[selected],negative,out=np.full(X.shape[1],np.nan),where=negative>0)
    valid=(positive>=20)&(negative>=20)&(pa-pb<=-.1+cc.TOL)
    if scope=='specific':
        valid &= np.array([s[0]=='specific' for s in ctx['states']])
    eligible=[]
    rows=[ctx['rows'][int(i)] for i in selected]
    for j in np.flatnonzero(valid):
        a_images=cc.image_support(rows,A[:,j])
        b_images=cc.image_support(rows,B[:,j])
        if min(a_images,b_images)<20:
            continue
        eligible.append((int(j),a_images,b_images))
    eligible.sort(key=lambda t:(float(pa[t[0]]-pb[t[0]]),tuple(ctx['states'][t[0]])))
    result=[]
    for rank,(j,a_images,b_images) in enumerate(eligible[:3],1):
        ns,field,value=ctx['states'][j]
        result.append(dict(rank=rank,state_index=j,namespace=ns,field=field,value=value,
            state_targets=int(positive[j]),control_targets=int(negative[j]),
            state_images=a_images,control_images=b_images,state_recall=float(pa[j]),
            control_recall=float(pb[j]),delta_pp=float(100*(pa[j]-pb[j]))))
    return result


def evaluate(out):
    cc.verify_freeze(out/'selection_before_prediction_freeze.json')
    cc.verify_freeze(out/'input_hashes.json')
    assert not (out/'model_metrics_by_seed.csv').exists(),'Evaluation outputs exist.'
    contexts=cc.load_contexts()
    for ctx in contexts.values():
        ctx['diagnostic_masks']=cc.condition_masks(ctx)
    plan=json.loads((out/'protocol.json').read_text())
    source_rows=cc.old.manifest()
    ids=[r['sample_id'] for r in source_rows]
    lookup={sid:i for i,sid in enumerate(ids)}
    truth=np.array([int(r['label']) for r in source_rows])
    risk={r['sample_id']:r for r in cc.read(cc.old.SOURCE/'risk_scores_and_tiers.csv')}
    tiers=np.array([int(risk[sid]['Q']) for sid in ids])
    fixed=[lookup[r['sample_id']] for r in cc.read(out/'fixed_background.csv')]
    with np.load(out/'selected_sets.npz',allow_pickle=False) as z:
        choices={k:z[k].copy() for k in z.files}
    setups=[(m,-1) for m in ['G','SR','KC']]
    setups += [(m,seed) for seed in plan['phases']['retrospective']['selection_seeds'] for m in ['R','CS','AS']]
    all_sets={};membership=[]
    for method,seed in setups:
        indices=list(fixed)
        for c,ctx in contexts.items():
            selected=choices[cc.selection_key(c,method,seed)]
            indices.extend(lookup[str(ctx['ids'][i])] for i in selected)
        a=np.array(indices,int)
        assert len(a)==len(set(a))==1200
        assert np.bincount(truth[a],minlength=6).tolist()==[200]*6
        for q in range(1,6):
            assert np.bincount(truth[a[tiers[a]==q]],minlength=6).tolist()==[40]*6
        all_sets[f'{method}__s{seed}']=a
        membership.extend(dict(method=method,seed=seed,sample_id=ids[i],
            class_name=source_rows[i]['class_name'],image_id=source_rows[i]['image_id'],Q=int(tiers[i])) for i in a)
    np.savez_compressed(out/'final_six_class_sets.npz',sample_ids=np.array(ids),tiers=tiers,**all_sets)
    cc.write(out/'final_six_class_membership.csv',membership)
    cc.freeze(out/'six_class_membership_before_prediction_freeze.json',dict(time=cc.old.now(),
        files={str(out/n):cc.sha(out/n) for n in ['final_six_class_sets.npz','final_six_class_membership.csv']},
        classifier_predictions_read=False))
    # All selected target identities and all Q memberships are now frozen.
    pred,hashes=cc.old.predictions('B',source_rows)
    source_hashes={str(Path(__file__)):cc.sha(__file__),
                   str(cc.ROOT/'scripts/conditional_coverage_common.py'):cc.sha(cc.ROOT/'scripts/conditional_coverage_common.py')}
    outputs=[];confusions=[];conditions=[];groups=[]
    for method,seed in setups:
        indices=all_sets[f'{method}__s{seed}']
        for model in cc.old.MODELS:
            for pi,pseed in enumerate(cc.old.SEEDS):
                prediction=pred[model][pi]
                for q in range(0,6):
                    take=indices if q==0 else indices[tiers[indices]==q]
                    cm=np.bincount(truth[take]*6+prediction[take],minlength=36).reshape(6,6)
                    metrics=confusion_metrics(cm)
                    key=dict(method=method,selection_seed=seed,model=model,prediction_seed=pseed,Q=q)
                    confusions.append(dict(**key,matrix=cm.tolist()))
                    for ci,c in enumerate(cc.old.ALL_CLASSES):
                        outputs.append(dict(**key,class_name=c,targets=int(cm[ci].sum()),
                            **{k:float(v[ci]) for k,v in metrics.items()}))
                    common={k:'' for k in ['precision','recall','one_vs_rest_accuracy','tp','fp','fn','tn']}
                    outputs.append(dict(**key,class_name='six_class_macro',targets=len(take),
                        f1=float(metrics['f1'].mean()),**common))
                    outputs.append(dict(**key,class_name='four_target_macro',targets=len(take),
                        f1=float(metrics['f1'][2:].mean()),**common))
                    outputs.append(dict(**key,class_name='six_class_top1',targets=len(take),
                        f1='',**{**common,'one_vs_rest_accuracy':''},
                        top1_accuracy=float(np.trace(cm)/cm.sum())))
            for c,ctx in contexts.items():
                pos=np.array([lookup[str(sid)] for sid in ctx['ids']])
                correct=(pred[model][:,pos]==cc.old.ALL_CLASSES.index(c)).mean(0)
                selected=choices[cc.selection_key(c,method,seed)]
                for scope in ['all','specific']:
                    found=discover(ctx,selected,correct,scope)
                    key=dict(class_name=c,model=model,method=method,seed=seed,scope=scope)
                    groups.append(dict(**key,proposed=len(found)))
                    conditions.extend(dict(**key,**r) for r in found)
        if method=='AS' and seed%10==9:
            cc.log(out,'model_evaluation_progress',selection_seed=seed)
    fields=list(dict.fromkeys(k for r in outputs for k in r))
    cc.write(out/'model_metrics_by_seed.csv',outputs,fields)
    cc.dump(out/'confusion_matrices.json',confusions)
    cc.write(out/'discovered_conditions.csv',conditions,CONDITION_FIELDS)
    cc.write(out/'discovery_groups.csv',groups)
    cc.freeze(out/'conditions_before_external_prediction_freeze.json',dict(time=cc.old.now(),
        files={str(out/n):cc.sha(out/n) for n in ['protocol.json','selected_sets.npz',
            'discovered_conditions.csv','discovery_groups.csv']},
        discovery_predictions=hashes,implementation=source_hashes,
        reference_prediction_files_read=False,
        historical_reference_has_been_analyzed_previously=True,
        same_field_controls=True,unknown_is_not_absence=True))
    cc.dump(out/'prediction_audit.json',dict(status='pass',discovery_prediction_files=len(hashes),
        target_count=2400,selected_per_run=1200,models=5,prediction_seeds=3,
        new_model_inference=False,new_model_training=False,prediction_hashes=hashes,
        metrics_rows=len(outputs),confusions=len(confusions),condition_rows=len(conditions)))
    cc.log(out,'evaluation_and_condition_freeze_complete',condition_rows=len(conditions),
           model_metric_rows=len(outputs),new_inference=False)


if __name__=='__main__':
    args=cc.parsed_args(__doc__).parse_args()
    evaluate(args.out)
