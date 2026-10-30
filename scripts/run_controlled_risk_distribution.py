#!/usr/bin/env python3
"""Frozen-prediction, fixed-budget risk-mixture experiments with paired sampling."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import kendalltau, rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from automatic_strengthening_common import labels_for
from analyze_sixclass_difficulty import risk
from dual_vlm_common import CLASSES, MODELS, SEEDS, read_csv, write_csv, write_json, sha256, now
from run_multiclass_object_selection_benchmark import scale_state

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/controlled_risk_distribution_20261005'
PROTOCOL = ROOT / 'docs/controlled_risk_distribution_protocol_20261005.md'
POOLS = {'B':ROOT/'artifacts/sixclass_independent_confirmation_20261004',
         'C':ROOT/'artifacts/sixclass_voc_external_confirmation_20261005',
         'D':ROOT/'artifacts/objects365_independent_confirmation_20261005'}
RUNS = list(itertools.product(MODELS, SEEDS))
LABELS = dict(zip(CLASSES, range(6)))
METRICS = ('accuracy','macro_precision','macro_recall','macro_f1','failed_state_breadth',
           'confusion_breadth','cluster_breadth','equal_error_confusion_delta','equal_error_cluster_delta',
           'equal_error_evaluable_class_fraction','frequency_coverage','presence_coverage',
           'eligible_state_fraction','mean_H')


def note(out, stage, **extra):
    value = dict(status='running', stage=stage, heartbeat=now(), **extra)
    write_json(out/'progress.json',value)
    line=json.dumps(value,ensure_ascii=False)
    print(line,flush=True)
    with (out/'run.log').open('a') as stream:stream.write(line+'\n')


def prefix(values):
    shape=(values.shape[0],1,*values.shape[2:])
    return np.concatenate([np.zeros(shape,dtype=values.dtype),np.cumsum(values,axis=1,dtype=values.dtype)],axis=1)


def mixture_counts(high, low, high_n, low_n):
    return (prefix(high)[:,high_n]+prefix(low)[:,low_n]).swapaxes(0,1)


def rarefied(counts, q, universe):
    if universe==0:return np.zeros(q.shape)
    total=counts.sum(-1)
    remain=total[...,None]-counts
    qq=q[...,None]
    available=remain>=qq
    log_den=gammaln(total+1)-gammaln(q+1)-gammaln(total-q+1)
    log_num=gammaln(remain+1)-gammaln(qq+1)-gammaln(remain-qq+1)
    absent=np.ones(counts.shape,dtype=float)
    valid=(qq>0)&(counts>0)&available
    absent[valid]=np.exp(np.minimum(0,(log_num-log_den[...,None])[valid]))
    absent[(qq>0)&(counts>0)&~available]=0
    return (1-absent).sum(-1)/universe


def top_two(values):
    cutoff=np.sort(values,axis=-1)[...,-2:-1]
    above=values>cutoff;same=values==cutoff
    return above.astype(float)+same*(2-above.sum(-1,keepdims=True))/same.sum(-1,keepdims=True)


def basic_math_check():
    # Compare exact rarefaction to exhaustive without-replacement subsets.
    labels=np.array([0,0,1,2]);counts=np.array([[2,1,1]])
    brute=np.mean([len(set(labels[list(indices)]))/3 for indices in itertools.combinations(range(4),2)])
    assert np.isclose(rarefied(counts,np.array([2]),3)[0],brute)
    assert np.allclose(top_two(np.ones(8)),.25)


def prepare_pool(pool, path, runs, out):
    rows=read_csv(path/'manifests/candidate_pool.csv');ids=[r['sample_id'] for r in rows]
    assert len(set(ids))==len(rows)==len({r['image_id'] for r in rows})
    y=np.array([LABELS[r['class_name']] for r in rows])
    assert Counter(y.tolist())==Counter(dict.fromkeys(range(6),len(rows)//6))
    labels,failures=labels_for(rows,path);assert not failures
    predictions=[]
    for model,seed in runs:
        with np.load(path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
            assert saved['sample_ids'].tolist()==ids and np.isfinite(saved['logits']).all()
            predictions.append(saved['logits'].argmax(1))
    with np.load(path/'embeddings/vit_b16_candidates.npz') as saved:
        assert saved['sample_ids'].tolist()==ids
        embeddings=saved['embeddings'].astype(np.float32)
        assert np.isfinite(embeddings).all()
    embeddings/=np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    predictions=np.asarray(predictions)
    clusters=np.full(predictions.shape,-1,dtype=np.int16)
    cluster_universe=np.zeros((len(runs),6),dtype=int)
    confusion_universe=np.zeros((len(runs),6),dtype=int)
    inventory=[]
    for run,(model,seed) in enumerate(runs):
        note(out,'freeze_error_clusters',pool=pool,run=run+1,total=len(runs),model=model,seed=seed)
        for c,name in enumerate(CLASSES):
            wrong=np.flatnonzero((y==c)&(predictions[run]!=y));m=len(wrong)
            k=0 if m==0 else 1 if m==1 else max(2,min(8,round(np.sqrt(m))))
            assignment=np.array([],dtype=int) if not m else np.array([0]) if m==1 else KMeans(
                n_clusters=k,random_state=20260928+seed+c,n_init=20).fit_predict(embeddings[wrong])
            assert len(set(assignment.tolist()))==k
            clusters[run,wrong]=assignment
            cluster_universe[run,c]=k
            confusion_universe[run,c]=len(set(predictions[run,wrong].tolist()))
            inventory.append(dict(pool=pool,model=model,seed=seed,class_name=name,full_errors=m,
                clusters=k,confusion_types=int(confusion_universe[run,c])))
    np.savez_compressed(out/'clusters'/f'{pool}.npz',sample_ids=np.array(ids),clusters=clusters,
        predictions=predictions,cluster_universe=cluster_universe,confusion_universe=confusion_universe)
    return rows,ids,y,labels,predictions,clusters,cluster_universe,confusion_universe,inventory


def contexts(rows, ids, y, labels):
    # State activation is identical to the existing shared-state protocol; unknown supplies no evidence.
    result={s:[] for s in ('local','online')};universe=[];tiers={s:np.zeros(len(rows),dtype=int) for s in result}
    for c,name in enumerate(CLASSES):
        indices=np.flatnonzero(y==c)
        active={s:[] for s in result}
        state_set=set()
        for source in result:
            for i in indices:
                values={('target_scale',scale_state(float(rows[i]['bbox_area_ratio'])))}
                features=labels[source][ids[i]]
                for field in SHARED_FIELDS:
                    entry=features[field]
                    values.update((field,value) for value in (entry if isinstance(entry,list) else [entry]) if value!='unknown')
                active[source].append(values);state_set.update(values)
        states=sorted(state_set);lookup={s:j for j,s in enumerate(states)}
        universe.extend(dict(class_name=name,feature=f,value=v) for f,v in states)
        for source in result:
            matrix=np.zeros((len(indices),len(states)),dtype=np.int16)
            for i,values in enumerate(active[source]):
                matrix[i,[lookup[s] for s in values]]=1
            scores=np.array([risk(rows[i],labels[source][ids[i]])['h_midpoint'] for i in indices])
            ordered=sorted(range(len(indices)),key=lambda j:(scores[j],ids[indices[j]]))
            for t,portion in enumerate(np.split(np.array(ordered),5),1):tiers[source][indices[portion]]=t
            result[source].append(dict(indices=indices,matrix=matrix,scores=scores))
    return result,universe,tiers


SHARED_FIELDS=()


def sampling_orders(pool, source, sections, tiers, repeats, max_budget, seed, out):
    rng=np.random.default_rng(seed);high=[];low=[]
    for section in sections:
        indices=section['indices']
        upper=indices[tiers[indices]==5];other=indices[tiers[indices]!=5]
        assert len(upper)>=max_budget and len(other)>=max_budget
        high.append(np.array([rng.permutation(upper)[:max_budget] for _ in range(repeats)]))
        low.append(np.array([rng.permutation(other)[:max_budget] for _ in range(repeats)]))
    high=np.stack(high,axis=1).astype(np.int16);low=np.stack(low,axis=1).astype(np.int16)
    assert all(len(set(np.concatenate([high[r,c],low[r,c]])))==2*max_budget for r in range(repeats) for c in range(6))
    np.savez_compressed(out/'sampling'/f'{pool}_{source}.npz',high=high,low=low,seed=seed)
    return high,low


def evaluate(pool,path,source,sections,rows,y,predictions,clusters,k_universe,p_universe,
             high,low,budget,proportions,runs,out):
    repeats=high.shape[0];points=len(proportions);reference=proportions.index(20)
    nh=np.array([budget*p//100 for p in proportions]);nl=budget-nh
    assert np.array_equal(nh*100,budget*np.array(proportions))
    metrics=np.zeros((points,repeats,len(runs),len(METRICS)),dtype=np.float64)
    recalls=np.zeros((points,repeats,len(runs),6),dtype=np.float64)
    prepared=[]
    for c,section in enumerate(sections):
        global_to_local={int(v):j for j,v in enumerate(section['indices'])}
        hi=np.array([[global_to_local[int(v)] for v in order] for order in high[:,c,:budget]])
        lo=np.array([[global_to_local[int(v)] for v in order] for order in low[:,c,:budget]])
        hs=section['matrix'][hi];ls=section['matrix'][lo]
        support=mixture_counts(hs,ls,nh,nl)
        eligible=support>=3
        coverage=np.minimum(support/20,1).mean(-1)
        presence=(support>0).mean(-1);eligible_fraction=eligible.mean(-1)
        H=mixture_counts(section['scores'][hi],section['scores'][lo],nh,nl)/budget
        prepared.append((hs,ls,eligible))
        for key,values in [('frequency_coverage',coverage),('presence_coverage',presence),
                           ('eligible_state_fraction',eligible_fraction),('mean_H',H)]:
            metrics[...,METRICS.index(key)]+=values[...,None]/6
    old=pd.read_csv(path/'confirmation/tables/full_stratum_metrics.csv')
    checks=0
    for run,(model,seed) in enumerate(runs):
        note(out,'evaluate_mixtures',pool=pool,source=source,total=6*budget,run=run+1,
             runs=len(runs),model=model,seed=seed)
        prediction=predictions[run]
        confusion=np.zeros((points,repeats,6,6),dtype=np.int16)
        for c in range(6):
            high_pred=prediction[high[:,c,:budget]];low_pred=prediction[low[:,c,:budget]]
            for label in range(6):
                confusion[:,:,c,label]=mixture_counts((high_pred==label).astype(np.int16),
                    (low_pred==label).astype(np.int16),nh,nl)
            hs,ls,eligible=prepared[c]
            error_support=mixture_counts(hs*(high_pred!=c)[...,None],ls*(low_pred!=c)[...,None],nh,nl)
            breadth=((error_support>0)&eligible).mean(-1)
            metrics[:,:,run,METRICS.index('failed_state_breadth')]+=breadth/6
            type_counts=confusion[:,:,c,:].copy();type_counts[:,:,c]=0
            n_errors=type_counts.sum(-1);base_count=type_counts[reference]
            q=np.minimum(n_errors,base_count.sum(-1)[None,:])
            denom=int(p_universe[run,c])
            raw=(type_counts>0).sum(-1)/denom if denom else np.zeros(n_errors.shape)
            rare=rarefied(type_counts,q,denom)
            base_rare=rarefied(np.broadcast_to(base_count,type_counts.shape),q,denom)
            metrics[:,:,run,METRICS.index('confusion_breadth')]+=raw/6
            metrics[:,:,run,METRICS.index('equal_error_confusion_delta')]+=(rare-base_rare)/6
            metrics[:,:,run,METRICS.index('equal_error_evaluable_class_fraction')]+=(q>0)/6
            cluster_count=np.zeros((points,repeats,int(k_universe[run,c])),dtype=np.int16)
            high_cluster=clusters[run,high[:,c,:budget]];low_cluster=clusters[run,low[:,c,:budget]]
            for label in range(int(k_universe[run,c])):
                cluster_count[:,:,label]=mixture_counts((high_cluster==label).astype(np.int16),
                    (low_cluster==label).astype(np.int16),nh,nl)
            assert np.array_equal(cluster_count.sum(-1),n_errors)
            k=int(k_universe[run,c]);raw=(cluster_count>0).sum(-1)/k if k else np.zeros(n_errors.shape)
            base_cluster=cluster_count[reference]
            rare=rarefied(cluster_count,q,k)
            base_rare=rarefied(np.broadcast_to(base_cluster,cluster_count.shape),q,k)
            metrics[:,:,run,METRICS.index('cluster_breadth')]+=raw/6
            metrics[:,:,run,METRICS.index('equal_error_cluster_delta')]+=(rare-base_rare)/6
        diagonal=np.diagonal(confusion,axis1=-2,axis2=-1)
        precision=np.divide(diagonal,confusion.sum(-2),out=np.zeros(diagonal.shape,dtype=float),where=confusion.sum(-2)>0)
        recall=diagonal/budget
        f1=np.divide(2*diagonal,confusion.sum(-2)+confusion.sum(-1),out=np.zeros(diagonal.shape,dtype=float),
                     where=(confusion.sum(-2)+confusion.sum(-1))>0)
        for key,value in [('accuracy',diagonal.sum(-1)/(6*budget)),('macro_precision',precision.mean(-1)),
                          ('macro_recall',recall.mean(-1)),('macro_f1',f1.mean(-1))]:
            metrics[:,:,run,METRICS.index(key)]=value
        recalls[:,:,run]=recall
        assert np.all(confusion.sum(-1)==budget)
        # The maximal feasible high-risk budget must reproduce the original Q5 exactly.
        if budget==len(rows)//30:
            prior=old[(old.source==source)&(old.unknown_mode=='midpoint')&(old.tier==5)&(old.model==model)&(old.seed==seed)]
            assert len(prior)==1 and np.allclose(f1[-1].mean(-1),prior.macro_f1.iloc[0],atol=1e-12)
            checks+=1
        if run in (0,len(runs)-1):
            for p in (0,reference,points-1):
                selected=np.concatenate([np.concatenate([high[0,c,:nh[p]],low[0,c,:nl[p]]]) for c in range(6)])
                assert len(selected)==len(set(selected.tolist()))==6*budget
                pp,rr,ff,_=precision_recall_fscore_support(y[selected],prediction[selected],labels=np.arange(6),zero_division=0)
                assert np.allclose(ff.mean(),metrics[p,0,run,METRICS.index('macro_f1')])
                assert np.allclose(rr,recalls[p,0,run])
                assert np.isclose(accuracy_score(y[selected],prediction[selected]),metrics[p,0,run,0])
    assert np.isfinite(metrics).all() and np.isfinite(recalls).all()
    assert np.allclose(metrics[:,:,:,0],metrics[:,:,:,2])
    assert np.allclose(metrics[reference,:,:,METRICS.index('equal_error_confusion_delta')],0)
    assert np.allclose(metrics[reference,:,:,METRICS.index('equal_error_cluster_delta')],0)
    np.savez_compressed(out/'raw'/f'{pool}_{source}_n{6*budget}.npz',metrics=metrics.astype(np.float32),
        class_recall=recalls.astype(np.float32),metric_names=np.array(METRICS),classes=np.array(CLASSES),
        proportions=np.array(proportions),models=np.array([m for m,s in runs]),seeds=np.array([s for m,s in runs]))
    return metrics,recalls,checks


def summarize(pool,source,budget,proportions,metrics,recalls,runs):
    info=dict(pool=pool,source=source,total=6*budget,per_class=budget,
              budget_role='common' if budget==20 else 'scale_sensitivity')
    # Main outputs average seeds inside each architecture; shared draws preserve pairing.
    values=metrics.reshape(len(proportions),metrics.shape[1],8,3,len(METRICS)).mean(3)
    class_values=recalls.reshape(len(proportions),metrics.shape[1],8,3,6).mean(3)
    model_rows=[];class_rows=[];panel_rows=[];run_rows=[];contrasts=[]
    score=values[...,METRICS.index('macro_f1')]
    point=score.mean(1);ranking=rankdata(-point,axis=-1,method='average')
    draw_ranks=rankdata(-score,axis=-1,method='average');top=top_two(score)
    ref=proportions.index(20)
    for p,ratio in enumerate(proportions):
        for model_i,model in enumerate(MODELS):
            row=dict(**info,high_risk_percent=ratio,model=model,rank=float(ranking[p,model_i]),
                rank_sampling_low=float(np.percentile(draw_ranks[p,:,model_i],2.5)),
                rank_sampling_high=float(np.percentile(draw_ranks[p,:,model_i],97.5)),
                top2_frequency=float(top[p,:,model_i].mean()),
                point_kendall_vs20=float(kendalltau(point[p],point[ref]).statistic))
            for j,key in enumerate(METRICS):
                value=values[p,:,model_i,j]
                row.update({key+'_mean':float(value.mean()),key+'_sampling_low':float(np.percentile(value,2.5)),
                            key+'_sampling_high':float(np.percentile(value,97.5))})
            model_rows.append(row)
            for c,name in enumerate(CLASSES):
                v=class_values[p,:,model_i,c]
                class_rows.append(dict(**info,high_risk_percent=ratio,model=model,class_name=name,
                    recall_mean=float(v.mean()),recall_sampling_low=float(np.percentile(v,2.5)),recall_sampling_high=float(np.percentile(v,97.5))))
        panel=dict(**info,high_risk_percent=ratio)
        for j,key in enumerate(METRICS):
            v=values[p,:,:,j].mean(-1)
            panel.update({key+'_mean':float(v.mean()),key+'_sampling_low':float(np.percentile(v,2.5)),key+'_sampling_high':float(np.percentile(v,97.5))})
        panel_rows.append(panel)
        for run,(model,seed) in enumerate(runs):
            run_rows.append(dict(**info,high_risk_percent=ratio,model=model,seed=seed,
                **{key:float(metrics[p,:,run,j].mean()) for j,key in enumerate(METRICS)}))
    for model_i,model in [(i,m) for i,m in enumerate(MODELS)]+[(None,'panel_mean')]:
        row=dict(**info,model=model,contrast='100_minus_0_percent_high_risk')
        for j,key in enumerate(METRICS):
            delta=values[-1,:,:,j]-values[0,:,:,j]
            delta=delta.mean(-1) if model_i is None else delta[:,model_i]
            row.update({key+'_delta':float(delta.mean()),key+'_sampling_low':float(np.percentile(delta,2.5)),
                        key+'_sampling_high':float(np.percentile(delta,97.5))})
        contrasts.append(row)
    return model_rows,class_rows,panel_rows,run_rows,contrasts


def main(smoke=False):
    out=BASE/'smoke' if smoke else BASE
    for folder in ('tables','raw','sampling','clusters','figures'):(out/folder).mkdir(parents=True,exist_ok=True)
    runs=[RUNS[0],RUNS[-1]] if smoke else RUNS
    proportions=[0,20,100] if smoke else list(range(0,101,5))
    repeats=12 if smoke else 1000
    pools={'B':POOLS['B']} if smoke else POOLS
    sources=('local',) if smoke else ('local','online')
    inputs=[Path(__file__),PROTOCOL,
        ROOT/'artifacts/ranking_attribute_validation_20261005/tables/risk_assignments.csv']
    inputs += [ROOT/'scripts'/name for name in ('analyze_sixclass_difficulty.py',
        'automatic_strengthening_common.py','run_multiclass_object_selection_benchmark.py',
        'run_dog_local_pipeline.py','dual_vlm_common.py')]
    for pool,path in pools.items():
        inputs+=[path/'manifests/candidate_pool.csv',path/'configs/shared_object_schema.json',path/'confirmation/tables/full_stratum_metrics.csv',
                 path/'embeddings/vit_b16_candidates.npz']
        inputs+=sorted((path/'annotations').glob('*/*/parsed/*.json'))
        inputs+=[path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz' for model,seed in runs]
    mapping={str(p):sha256(p) for p in inputs}
    frozen=out/'input_freeze.sha256.json'
    if frozen.exists():assert json.loads(frozen.read_text())==mapping,'frozen experiment input changed'
    else:write_json(frozen,mapping)
    write_json(out/'registration.json',dict(created_at=now(),posthoc_design=True,smoke=smoke,
        repetitions=repeats,high_risk_percent=proportions,high_risk_definition='original midpoint Q5',
        common_total=120,sensitivity_total=480,interval='95% repeated-sampling range; not mean confidence interval',
        no_new_training=True,no_new_vlm_requests=True,no_new_classifier_inference=True))
    basic_math_check()
    all_rows=[[],[],[],[],[]];inventory=[];q5_checks=0
    global SHARED_FIELDS
    for pool,path in pools.items():
        note(out,'prepare_pool',pool=pool,smoke=smoke)
        rows,ids,y,labels,predictions,clusters,ku,pu,items=prepare_pool(pool,path,runs,out);inventory.extend(items)
        SHARED_FIELDS=tuple(x['feature_id'] for x in json.loads((path/'configs/shared_object_schema.json').read_text())['universal_features'])
        assert len(SHARED_FIELDS)==14
        sections,universe,tiers=contexts(rows,ids,y,labels)
        write_csv(out/'tables'/f'{pool}_state_universe.csv',universe)
        old_risk=pd.read_csv(ROOT/'artifacts/ranking_attribute_validation_20261005/tables/risk_assignments.csv')
        budgets=(20,) if smoke or pool=='C' else (20,80)
        for source in sources:
            prior=old_risk[(old_risk.pool==pool)&(old_risk.source==source)&(old_risk.unknown_mode=='midpoint')].set_index('sample_id')
            assert np.array_equal(tiers[source],prior.loc[ids].tier.to_numpy())
            h,l=sampling_orders(pool,source,sections[source],tiers[source],repeats,max(budgets),
                202610060+list(POOLS).index(pool)*2+list(('local','online')).index(source),out)
            for budget in budgets:
                metrics,recalls,checks=evaluate(pool,path,source,sections[source],rows,y,predictions,clusters,ku,pu,
                    h,l,budget,proportions,runs,out);q5_checks+=checks
                if not smoke:
                    summaries=summarize(pool,source,budget,proportions,metrics,recalls,runs)
                    for target,items in zip(all_rows,summaries):target.extend(items)
    if not smoke:
        for name,items in zip(('model_summary','class_recall','panel_summary','run_summary','distribution_contrasts'),all_rows):
            write_csv(out/'tables'/f'{name}.csv',items)
    write_csv(out/'tables/cluster_inventory.csv',inventory)
    assert all(sha256(p)==digest for p,digest in mapping.items())
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    audit=dict(status='pass',completed_at=now(),smoke=smoke,repetitions=repeats,proportion_points=len(proportions),
        freeze_files_unchanged=len(mapping),old_q5_metrics_verified=q5_checks,
        model_summary_rows=len(all_rows[0]),class_recall_rows=len(all_rows[1]),panel_rows=len(all_rows[2]),
        no_new_training=True,no_new_classifier_inference=True,no_new_vlm_requests=True,
        interval='95% repeated-sampling range conditional on finite pools and frozen predictions')
    write_json(out/'analysis_audit.json',audit)
    write_json(out/'progress.json',dict(**audit,stage='analysis_complete'))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:main(args.smoke)
    except Exception:
        out=BASE/'smoke' if args.smoke else BASE;out.mkdir(parents=True,exist_ok=True)
        write_json(out/'progress.json',dict(status='failed',heartbeat=now(),traceback=traceback.format_exc()))
        raise
