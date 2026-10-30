#!/usr/bin/env python3
"""Class-balanced random stratification and frozen-score dimension ablations."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from automatic_strengthening_common import labels_for
from analyze_sixclass_difficulty import risk,MAPS
from dual_vlm_common import CLASSES,MODELS,SEEDS,read_csv,write_csv,write_json,sha256,now
from run_class_stratified_error_controls import holm
from run_coverage_incremental_value import paired_stats

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/random_stratification_ablation_20261006'
PROTOCOL=ROOT/'docs/random_stratification_ablation_protocol_20261006.md'
POOLS={'B':ROOT/'artifacts/sixclass_independent_confirmation_20261004',
       'C':ROOT/'artifacts/sixclass_voc_external_confirmation_20261005',
       'D':ROOT/'artifacts/objects365_independent_confirmation_20261005'}
RUNS=list(itertools.product(MODELS,SEEDS))
VARIANTS=['full','drop_size',*['drop_'+field for field in MAPS],'equal_normalized']


def note(out,stage,**extra):
    value=dict(status='running',stage=stage,time=now(),**extra)
    write_json(out/'progress.json',value)
    line=json.dumps(value,ensure_ascii=False)
    with (out/'run.log').open('a') as stream:stream.write(line+'\n')
    print(line,flush=True)


def classification(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1).astype(float)
    incoming=cm.sum(-2);support=cm.sum(-1)
    precision=np.divide(tp,incoming,out=np.zeros_like(tp),where=incoming>0)
    recall=tp/support;f1=2*tp/(incoming+support)
    return dict(accuracy=tp.sum(-1)/support.sum(-1),macro_precision=precision.mean(-1),
        macro_recall=recall.mean(-1),macro_f1=f1.mean(-1)),recall


def stratified_scores(prediction,orders):
    # orders is repeat x class x five layers x targets-per-class-layer.
    repeats=orders.shape[0];budget=orders.shape[-1]
    cm=np.zeros((repeats,5,6,6),dtype=np.int16)
    for c in range(6):
        pred=prediction[orders[:,c]]
        for label in range(6):cm[:,:,c,label]=(pred==label).sum(-1)
    assert np.all(cm.sum(-1)==budget)
    return classification(cm)


def check_direct(rows,pred,orders,values,recall):
    y=np.array([int(row['label']) for row in rows])
    for repeat in sorted(set((0,len(orders)//2,len(orders)-1))):
        for tier in (0,2,4):
            selected=orders[repeat,:,tier].reshape(-1)
            assert len(selected)==len(set(selected.tolist()))
            pp,rr,ff,_=precision_recall_fscore_support(y[selected],pred[selected],labels=range(6),zero_division=0)
            assert np.allclose(values['macro_f1'][repeat,tier],ff.mean(),atol=1e-12)
            assert np.allclose(values['macro_precision'][repeat,tier],pp.mean(),atol=1e-12)
            assert np.allclose(recall[repeat,tier],rr,atol=1e-12)


def tier_orders(rows,parts,variant):
    orders=[];scores=[]
    for c,name in enumerate(CLASSES):
        indices=np.array([i for i,row in enumerate(rows) if row['class_name']==name])
        values=parts[indices]
        if variant=='full':score=values.sum(1)
        elif variant=='equal_normalized':
            # Preserve historical Python summation order: fields first, size last.
            score=np.array([v[0]/3+sum(v[j]/max(mapping.values()) for j,mapping in enumerate(MAPS.values(),1)) for v in values])
        else:
            j=['size',*MAPS].index(variant.removeprefix('drop_'))
            score=values.sum(1)-values[:,j]
        order=sorted(range(len(indices)),key=lambda j:(float(score[j]),rows[indices[j]]['sample_id']))
        orders.append(indices[order].reshape(5,-1));scores.append((indices,score))
    return np.stack(orders),scores


def load_pool(pool,path,runs):
    rows=read_csv(path/'manifests/candidate_pool.csv');ids=[r['sample_id'] for r in rows]
    assert len(ids)==len(set(ids))==len({r['image_id'] for r in rows})
    assert Counter(row['class_name'] for row in rows)==Counter(dict.fromkeys(CLASSES,len(rows)//6))
    labels,failures=labels_for(rows,path);assert not failures
    preds=[]
    for model,seed in runs:
        with np.load(path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
            assert saved['sample_ids'].tolist()==ids and np.isfinite(saved['logits']).all()
            preds.append(saved['logits'].argmax(1))
    part={}
    for source in labels:
        records=[risk(row,labels[source][row['sample_id']]) for row in rows]
        part[source]=np.array([[record[field] for field in ['size',*MAPS]] for record in records])
        assert np.allclose(part[source].sum(1),[record['h_midpoint'] for record in records])
    return rows,np.array(preds),part


def main(smoke=False):
    out=BASE/'smoke_corrected' if smoke else BASE
    for directory in ('tables','random_strata','ablation','figures'):(out/directory).mkdir(parents=True,exist_ok=True)
    pools={'B':POOLS['B']} if smoke else POOLS
    runs=[RUNS[0],RUNS[-1]] if smoke else RUNS
    repeats=25 if smoke else 10000
    inputs=[Path(__file__),PROTOCOL,ROOT/'artifacts/ranking_attribute_validation_20261005/tables/risk_assignments.csv']
    inputs.extend(ROOT/'scripts'/file for file in ('analyze_sixclass_difficulty.py','automatic_strengthening_common.py',
        'dual_vlm_common.py','run_dog_local_pipeline.py','run_class_stratified_error_controls.py','run_coverage_incremental_value.py'))
    for pool,path in pools.items():
        inputs.extend([path/'manifests/candidate_pool.csv',path/'confirmation/tables/full_stratum_metrics.csv'])
        inputs.extend(sorted((path/'annotations').glob('*/*/parsed/*.json')))
        inputs.extend(sorted((path/'configs').glob('*.json')))
        inputs.extend(path/'benchmark/runs'/m/str(s)/'candidate_outputs.npz' for m,s in runs)
        if pool in ('B','C'):
            inputs.extend(path/'robustness/tables'/file for file in ('risk_variant_metrics.csv','risk_variant_comparisons.csv'))
    frozen={str(p):sha256(p) for p in inputs};freeze_file=out/'input_freeze.sha256.json'
    if freeze_file.exists():assert json.loads(freeze_file.read_text())==frozen,'Frozen input changed'
    else:write_json(freeze_file,frozen)
    write_json(out/'registration.json',dict(recorded_at=now(),smoke=smoke,repeats=repeats,
        randomization_seeds={p:2026100611+list(POOLS).index(p) for p in pools},
        panel_statistic='mean of eight architecture means of three-seed Q5-minus-Q1 Macro-F1',
        p='two-sided (1+extreme)/(R+1), tol1e-12; Holm across six pool-source settings',
        ablation='five leave-one-dimension-out plus historical equal-normalized; redivide classes into quintiles',
        ablation_holm_family='twelve comparisons per pool, as historical',posthoc=True,
        no_new_training=True,no_new_inference=True,no_new_vlm=True))
    random_rows=[];random_models=[];run_rows=[];ablation_rows=[];membership=[];assignments=[]
    old_risk=pd.read_csv(ROOT/'artifacts/ranking_attribute_validation_20261005/tables/risk_assignments.csv')
    q5_checks=ablation_checks=0
    for pool,path in pools.items():
        note(out,'load_pool',pool=pool,smoke=smoke)
        rows,predictions,parts=load_pool(pool,path,runs);ids=[r['sample_id'] for r in rows]
        per_class=len(rows)//6;per_tier=per_class//5
        rng=np.random.default_rng(2026100611+list(POOLS).index(pool))
        random_order=[]
        for name in CLASSES:
            indices=np.array([i for i,row in enumerate(rows) if row['class_name']==name],dtype=np.int16)
            random_order.append(np.array([rng.permutation(indices) for _ in range(repeats)],dtype=np.int16))
        random_order=np.stack(random_order,axis=1).reshape(repeats,6,5,per_tier)
        assert np.all(np.sort(random_order.reshape(repeats,-1),axis=1)==np.arange(len(rows)))
        np.savez_compressed(out/'random_strata'/f'{pool}_orders.npz',orders=random_order,
            sample_ids=np.array(ids),classes=np.array(CLASSES),seed=2026100611+list(POOLS).index(pool))
        random_f1=np.empty((repeats,len(runs),5));random_recall=np.empty((repeats,len(runs),5,6),dtype=np.float32)
        for run,(model,seed) in enumerate(runs):
            note(out,'random_stratification',pool=pool,run=run+1,runs=len(runs),model=model,seed=seed,repeats=repeats)
            metrics,recall=stratified_scores(predictions[run],random_order)
            random_f1[:,run]=metrics['macro_f1'];random_recall[:,run]=recall
            assert np.allclose(metrics['accuracy'],metrics['macro_recall'])
            if run in (0,len(runs)-1):check_direct(rows,predictions[run],random_order,metrics,recall)
        panel_curve=random_f1.mean(1);panel_delta=panel_curve[:,-1]-panel_curve[:,0]
        np.savez_compressed(out/'random_strata'/f'{pool}_metrics.npz',f1=random_f1,
            class_recall=random_recall,panel_curve=panel_curve,panel_delta=panel_delta,
            models=np.array([m for m,s in runs]),seeds=np.array([s for m,s in runs]))
        old=pd.read_csv(path/'confirmation/tables/full_stratum_metrics.csv')
        old_ablation=pd.read_csv(path/'robustness/tables/risk_variant_metrics.csv') if pool in ('B','C') else None
        for source,values in parts.items():
            variant_f1={};variant_orders={}
            for variant in VARIANTS:
                orders,scores=tier_orders(rows,values,variant);variant_orders[variant]=orders
                for indices,score in scores:
                    tiers=np.zeros(len(rows),dtype=int)
                    for t in range(5):tiers[orders[:,t].reshape(-1)]=t+1
                    assignments.extend(dict(pool=pool,source=source,variant=variant,sample_id=rows[i]['sample_id'],
                        class_name=rows[i]['class_name'],score=float(s),tier=int(tiers[i])) for i,s in zip(indices,score))
                if variant=='full':
                    prior=old_risk[(old_risk.pool==pool)&(old_risk.source==source)&(old_risk.unknown_mode=='midpoint')].set_index('sample_id')
                    new_tiers=np.empty(len(rows),int)
                    for t in range(5):new_tiers[orders[:,t].reshape(-1)]=t+1
                    assert np.array_equal(prior.loc[ids].tier.to_numpy(),new_tiers)
                f1=[]
                for run,(model,seed) in enumerate(runs):
                    metrics,recall=stratified_scores(predictions[run],orders[None])
                    f1.append(metrics['macro_f1'][0])
                    check_direct(rows,predictions[run],orders[None],metrics,recall)
                    for t in range(5):
                        run_rows.append(dict(pool=pool,source=source,variant=variant,model=model,seed=seed,tier=t+1,
                            **{k:float(v[0,t]) for k,v in metrics.items()},
                            **{name+'_recall':float(recall[0,t,c]) for c,name in enumerate(CLASSES)}))
                    if variant=='full':
                        p=old[(old.source==source)&(old.unknown_mode=='midpoint')&(old.model==model)&(old.seed==seed)].sort_values('tier')
                        assert len(p)==5 and np.allclose(p.macro_f1,metrics['macro_f1'][0],atol=1e-12);q5_checks+=5
                    elif old_ablation is not None:
                        p=old_ablation[(old_ablation.source==source)&(old_ablation.variant==variant)&(old_ablation.model==model)&(old_ablation.seed==seed)].sort_values('tier')
                        assert len(p)==5 and np.allclose(p.macro_f1,metrics['macro_f1'][0],atol=1e-12);ablation_checks+=5
                variant_f1[variant]=np.array(f1)
            full=variant_f1['full'];observed=(full[:,-1]-full[:,0]).mean()
            extreme=int((np.abs(panel_delta)>=abs(observed)-1e-12).sum())
            random_rows.append(dict(pool=pool,source=source,targets=len(rows),per_class_layer=per_tier,layer_total=6*per_tier,
                repeats=repeats,observed_delta=float(observed),random_mean=float(panel_delta.mean()),random_sd=float(panel_delta.std(ddof=1)),
                random_sampling_low=float(np.percentile(panel_delta,2.5)),random_sampling_high=float(np.percentile(panel_delta,97.5)),
                random_abs_max=float(np.abs(panel_delta).max()),extreme_count=extreme,mc_two_sided_p=(1+extreme)/(repeats+1)))
            if not smoke:
                for mi,model in enumerate(MODELS):
                    null=random_f1[:,mi*3:(mi+1)*3].mean(1)
                    delta=null[:,-1]-null[:,0];obs=(full[mi*3:(mi+1)*3,-1]-full[mi*3:(mi+1)*3,0]).mean()
                    count=int((np.abs(delta)>=abs(obs)-1e-12).sum())
                    random_models.append(dict(pool=pool,source=source,model=model,observed_delta=float(obs),
                        random_mean=float(delta.mean()),random_sampling_low=float(np.percentile(delta,2.5)),
                        random_sampling_high=float(np.percentile(delta,97.5)),extreme_count=count,
                        mc_two_sided_p=(count+1)/(repeats+1),scope='descriptive model-specific comparison'))
            for variant,f1 in variant_f1.items():
                delta=f1[:,-1]-f1[:,0];full_delta=full[:,-1]-full[:,0]
                if smoke:continue
                groups={model:delta[mi*3:(mi+1)*3].tolist() for mi,model in enumerate(MODELS)}
                record=dict(pool=pool,source=source,variant=variant,**paired_stats(groups,f'{source}_{variant}','Q5_minus_Q1_macro_f1'))
                change={model:(delta-full_delta)[mi*3:(mi+1)*3].tolist() for mi,model in enumerate(MODELS)}
                stats=paired_stats(change,f'{pool}_{source}_{variant}_minus_full','Q5_minus_Q1_macro_f1_change')
                record.update(change_vs_full=stats['mean_delta_auc'],change_ci95_low=stats['ci95_low'],change_ci95_high=stats['ci95_high'])
                ablation_rows.append(record)
                for c,name in enumerate(CLASSES):
                    for t in (0,4):
                        original=set(variant_orders['full'][c,t]);new=set(variant_orders[variant][c,t])
                        membership.append(dict(pool=pool,source=source,variant=variant,class_name=name,tier=t+1,
                            support=len(new),overlap=len(original&new),jaccard=len(original&new)/len(original|new),
                            retention=len(original&new)/len(original)))
            np.savez_compressed(out/'ablation'/f'{pool}_{source}.npz',f1=np.stack(list(variant_f1.values())),
                variants=np.array(VARIANTS),orders=np.stack([variant_orders[v] for v in VARIANTS]),
                risk_parts=values,predictions=predictions,sample_ids=np.array(ids),
                models=np.array([m for m,s in runs]),seeds=np.array([s for m,s in runs]))
            note(out,'ablation_complete',pool=pool,source=source,variants=len(VARIANTS))
    if not smoke:
        for row,p in zip(random_rows,holm([r['mc_two_sided_p'] for r in random_rows])):row['holm_six_p']=p
        for pool in pools:
            selected=[r for r in ablation_rows if r['pool']==pool and r['variant']!='full']
            assert len(selected)==12
            for row,p in zip(selected,holm([r['permutation_p'] for r in selected])):row['holm_twelve_p']=p
            if pool in ('B','C'):
                previous=pd.read_csv(POOLS[pool]/'robustness/tables/risk_variant_comparisons.csv')
                for row in selected:
                    old=previous[(previous.source==row['source'])&(previous.variant==row['variant'])].iloc[0]
                    for key in ('mean_delta_auc','ci95_low','ci95_high','permutation_p','holm_twelve_p'):
                        assert np.isclose(row[key],old[key],atol=1e-12),(pool,row['variant'],key)
        for filename,records in [('random_summary',random_rows),('random_model_summary',random_models),
            ('ablation_summary',ablation_rows),('ablation_run_metrics',run_rows),('membership',membership),('risk_assignments',assignments)]:
            write_csv(out/'tables'/f'{filename}.csv',records)
        aggregate=pd.DataFrame(membership).groupby(['pool','source','variant','tier'])[['jaccard','retention']].mean().reset_index()
        aggregate.to_csv(out/'tables/membership_summary.csv',index=False)
    assert all(sha256(p)==digest for p,digest in frozen.items())
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    audit=dict(status='pass',completed_at=now(),smoke=smoke,repeats=repeats,
        old_full_layer_values_verified=q5_checks,old_ablation_values_verified=ablation_checks,
        freeze_files_unchanged=len(frozen),random_summary_rows=len(random_rows),ablation_summary_rows=len(ablation_rows),
        no_new_training=True,no_new_inference=True,no_new_vlm=True)
    write_json(out/'analysis_audit.json',audit);write_json(out/'progress.json',dict(**audit,stage='analysis_complete'))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:main(args.smoke)
    except Exception:
        out=BASE/'smoke_corrected' if args.smoke else BASE
        write_json(out/'progress.json',dict(status='failed',time=now(),traceback=traceback.format_exc()))
        raise
