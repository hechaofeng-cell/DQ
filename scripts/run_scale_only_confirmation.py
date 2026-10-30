#!/usr/bin/env python3
"""Output-independent area control; portable frozen-prediction reanalysis.

Prepare once in the original workspace, then run --input DIRECTORY anywhere.
No training, classifier inference, network request, or human labeling occurs.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scale_only_confirmation_20261007'
POOL_PATHS = {'B':'sixclass_independent_confirmation_20261004',
              'C':'sixclass_voc_external_confirmation_20261005',
              'D':'objects365_independent_confirmation_20261005'}
CLASSES = ['dog','cat','horse','sheep','person','car']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dest):
    dest.mkdir(parents=True, exist_ok=True)
    frozen = {}
    def freeze(p):
        frozen[str(p.relative_to(ROOT))] = sha(p)
    for p in [Path(__file__),ROOT/'docs/scale_only_confirmation_protocol_20261007.md']:
        freeze(p)
    for pool, directory in POOL_PATHS.items():
        print('prepare',pool,flush=True)
        base=ROOT/'artifacts'/directory
        manifest=base/'manifests/candidate_pool.csv'; freeze(manifest)
        rows=pd.read_csv(manifest, dtype={'sample_id':str,'image_id':str})
        cache=ROOT/f'artifacts/random_stratification_ablation_20261006/ablation/{pool}_local.npz'
        freeze(cache)
        with np.load(cache) as z:
            ids=z['sample_ids']; predictions=z['predictions'];models=z['models'];seeds=z['seeds']
        assert rows.sample_id.tolist()==ids.tolist()
        shared=['sample_id','image_id','class_name','label','bbox_area_ratio','bbox_norm_xyxy',
                'dataset_version','source_split','image_sha256','crop_sha256']
        rows[[c for c in shared if c in rows]].to_csv(dest/f'{pool}_targets.csv',index=False)
        native={source:{} for source in ('local','online')}
        values={};orders={}; logits=[]
        for source in native:
            p=ROOT/f'artifacts/random_stratification_ablation_20261006/ablation/{pool}_{source}.npz';freeze(p)
            with np.load(p) as z:
                assert np.array_equal(z['predictions'],predictions)
                assert z['sample_ids'].tolist()==ids.tolist()
                assert z['variants'][0]=='full'
                values[source]=z['risk_parts'];orders[source]=z['orders'][0]
            for r in rows.to_dict('records'):
                domain='car' if r['class_name']=='car' else 'old'
                p=base/f"annotations/{source}/{domain}/parsed/{r['sample_id']}.json";freeze(p)
                native[source][r['sample_id']]=json.loads(p.read_text())['features']
        for model,seed in zip(models,seeds):
            p=base/f'benchmark/runs/{model}/{seed}/candidate_outputs.npz';freeze(p)
            with np.load(p) as z:
                assert z['sample_ids'].tolist()==ids.tolist()
                logits.append(z['logits'])
        assert np.array_equal(np.array(logits).argmax(-1),predictions)
        np.savez_compressed(dest/f'{pool}_predictions.npz',sample_ids=ids,
            predictions=predictions.astype(np.int8),logits=np.array(logits),models=models,seeds=seeds,
            local_parts=values['local'],online_parts=values['online'],
            local_full_orders=orders['local'],online_full_orders=orders['online'])
        dump(dest/f'{pool}_states.json',native)
        state_path=base/'transfer/tables/state_validation.csv';freeze(state_path)
        hypotheses=pd.read_csv(state_path)[['source','model','class_name','feature','value']].drop_duplicates()
        assert len(hypotheses)==175
        hypotheses.to_csv(dest/f'{pool}_frozen_hypotheses.csv',index=False)
    dump(dest/'original_input_hashes.json',frozen)
    assert all(sha(ROOT/p)==digest for p,digest in frozen.items())


def orders_for(rows, scores, tie=None):
    result=[]
    for category in CLASSES:
        idx=np.flatnonzero(rows.class_name.to_numpy()==category)
        if tie is None:secondary=rows.sample_id.to_numpy()[idx]
        else:secondary=tie[idx]
        order=np.lexsort((secondary,scores[idx]))
        result.append(idx[order].reshape(5,-1))
    result=np.array(result)
    assert sorted(result.reshape(-1).tolist())==list(range(len(rows)))
    return result


def metrics(predictions, orders):
    # repeat x class x layer x support; output repeat x run x layer.
    if orders.ndim==3:orders=orders[None]
    cm=np.zeros((len(orders),len(predictions),5,6,6),dtype=np.int32)
    for c in range(6):
        selected=predictions[:,orders[:,c]].transpose(1,0,2,3)
        for label in range(6):cm[:,:,:,c,label]=(selected==label).sum(-1)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    true=cm.sum(-1);incoming=cm.sum(-2)
    precision=np.divide(tp,incoming,out=np.zeros_like(tp,dtype=float),where=incoming>0)
    recall=tp/true;f1=2*tp/(true+incoming)
    return {'macro_f1':f1.mean(-1),'macro_precision':precision.mean(-1),
            'macro_recall':recall.mean(-1),'accuracy':tp.sum(-1)/true.sum(-1)},recall


def stats(deltas, models, seed):
    names=list(dict.fromkeys(models));arrays=np.array([deltas[models==m] for m in names])
    means=arrays.mean(1);n=len(names)
    signs=np.array(list(itertools.product([-1,1],repeat=n)))
    null=np.abs((signs*means).mean(1));observed=abs(means.mean())
    rng=np.random.default_rng(seed)
    arch=rng.integers(0,n,(5000,n));ss=rng.integers(0,arrays.shape[1],(5000,n,arrays.shape[1]))
    draws=arrays[arch[:,:,None],ss].mean((1,2))
    return {'increment_mean':float(means.mean()),'ci_low':float(np.quantile(draws,.025)),
            'ci_high':float(np.quantile(draws,.975)),
            'p':float((null>=observed-1e-15).mean()),
            'positive_runs':int((deltas>0).sum()),'negative_runs':int((deltas<0).sum())}


def holm(p):
    order=np.argsort(p);result=np.zeros(len(p));running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(p)-rank)*p[i]));result[i]=running
    return result


def area_condition(rows, states, predictions, models, seeds, area_orders, hypotheses, pool):
    area_tier=np.zeros(len(rows),int)
    for t in range(5):area_tier[area_orders[:,t].reshape(-1)]=t+1
    y=rows.label.to_numpy();records=[]
    for h in hypotheses.to_dict('records'):
        if h['feature']=='target_scale':continue
        category=rows.class_name.to_numpy()==h['class_name']
        native=[states[h['source']][sid].get(h['feature'],'unknown') for sid in rows.sample_id]
        def active(v):return h['value'] in v if isinstance(v,list) else v==h['value']
        state=category & np.array([active(v) for v in native])
        known=category & np.array([v!='unknown' and v!=['unknown'] for v in native])
        other=known & ~state;matched=[]
        for tier in range(1,6):
            s=np.flatnonzero(state & (area_tier==tier));r=np.flatnonzero(other & (area_tier==tier))
            if len(s)>=2 and len(r)>=2:matched.append((s,r))
        n=sum(len(s) for s,r in matched);ref=sum(len(r) for s,r in matched)
        total=int(state.sum());fraction=n/total if total else 0.
        eligible=total>=20 and n>=20 and ref>=20 and fraction>=.5
        for run in np.flatnonzero(models==h['model']):
            correct=predictions[run]==y
            gap=None
            if eligible:
                gap=sum(len(s)*(correct[s].mean()-correct[r].mean()) for s,r in matched)/n
            records.append(dict(pool=pool,**h,seed=int(seeds[run]),state_support=total,
                matched_state_support=n,matched_reference_support=ref,retained_fraction=fraction,
                matched_bins=len(matched),evaluable=eligible,area_adjusted_recall_gap=gap))
    return records


def analyze(inputs, out, smoke=False):
    out.mkdir(parents=True,exist_ok=True);(out/'tables').mkdir(exist_ok=True)
    summary=[];runs=[];members=[];profiles=[];unknown=[];conditions=[];tie_rows=[]
    audit_checks=0;repeats=10 if smoke else 1000
    pools=['B'] if smoke else list(POOL_PATHS)
    files=list(inputs.glob('*'));before={p.name:sha(p) for p in files}
    for pi,pool in enumerate(pools):
        print('analyze',pool,flush=True)
        rows=pd.read_csv(inputs/f'{pool}_targets.csv',dtype={'sample_id':str,'image_id':str})
        states=json.loads((inputs/f'{pool}_states.json').read_text())
        hypotheses=pd.read_csv(inputs/f'{pool}_frozen_hypotheses.csv')
        with np.load(inputs/f'{pool}_predictions.npz') as z:
            assert rows.sample_id.tolist()==z['sample_ids'].tolist()
            preds=z['predictions'];models=z['models'];seeds=z['seeds']
            parts={s:z[s+'_parts'] for s in states};frozen={s:z[s+'_full_orders'] for s in states}
        if smoke:
            select=np.array([0,len(preds)-1]);preds=preds[select];models=models[select];seeds=seeds[select]
        assert rows.sample_id.is_unique and rows.groupby('class_name').size().nunique()==1
        area=-rows.bbox_area_ratio.to_numpy()
        assert np.isfinite(area).all() and (area<0).all() and (area>=-1).all()
        area_order=orders_for(rows,area)
        area_metrics,_=metrics(preds,area_order);area_f1=area_metrics['macro_f1'][0]
        rng=np.random.default_rng(2026100711+pi)
        # Shared tie priorities per pool; no prediction or outcome is used.
        priorities=np.array([rng.permutation(len(rows)) for _ in range(repeats)])
        area_random=np.array([orders_for(rows,area,p) for p in priorities])
        area_tie_f1=metrics(preds,area_random)[0]['macro_f1']
        for source in states:
            full_scores=parts[source].sum(1);full_order=orders_for(rows,full_scores)
            assert np.array_equal(full_order,frozen[source])
            full_metrics,recall=metrics(preds,full_order);full_f1=full_metrics['macro_f1'][0]
            # Verify original endpoints and direct sklearn aggregation.
            for scheme,order,mm in [('full',full_order,full_metrics),('area_only',area_order,area_metrics)]:
                scheme_recall=metrics(preds,order)[1]
                for ri,(model,seed) in enumerate(zip(models,seeds)):
                    for t in range(5):
                        picked=order[:,t].reshape(-1)
                        p,r,f,_=precision_recall_fscore_support(rows.label.to_numpy()[picked],preds[ri,picked],labels=range(6),zero_division=0)
                        assert np.isclose(f.mean(),mm['macro_f1'][0,ri,t],atol=1e-12)
                        audit_checks+=1
                        runs.append(dict(pool=pool,source=source,scheme=scheme,model=model,seed=int(seed),tier=t+1,
                            **{k:float(v[0,ri,t]) for k,v in mm.items()},
                            **{name+'_recall':float(scheme_recall[0,ri,t,c]) for c,name in enumerate(CLASSES)}))
                score=full_scores if scheme=='full' else area
                for ci,category in enumerate(CLASSES):
                    for t in range(5):
                        ids=order[ci,t]
                        profiles.append(dict(pool=pool,source=source,scheme=scheme,class_name=category,tier=t+1,
                            support=len(ids),score_min=float(score[ids].min()),score_max=float(score[ids].max()),
                            area_min=float(-area[ids].max()),area_max=float(-area[ids].min())))
                        members.extend(dict(pool=pool,source=source,scheme=scheme,sample_id=rows.sample_id.iloc[i],
                            class_name=category,tier=t+1,score=float(score[i])) for i in ids)
            full_gap=full_f1[:,0]-full_f1[:,-1];area_gap=area_f1[:,0]-area_f1[:,-1]
            increment=full_gap-area_gap
            rec=dict(pool=pool,source=source,targets=len(rows),layer_total=len(rows)//5,
                full_gap=float(full_gap.mean()),area_gap=float(area_gap.mean()))
            if not smoke:rec.update(stats(increment,models,2026100740+2*pi+(source=='online')))
            else:rec['increment_mean']=float(increment.mean())
            fset=set(full_order[:,4].reshape(-1));aset=set(area_order[:,4].reshape(-1))
            rec.update(q5_retention=len(fset&aset)/len(fset),q5_jaccard=len(fset&aset)/len(fset|aset))
            full_random=np.array([orders_for(rows,full_scores,p) for p in priorities])
            full_tie=metrics(preds,full_random)[0]['macro_f1']
            tie_inc=((full_tie[:,:,0]-full_tie[:,:,-1])-(area_tie_f1[:,:,0]-area_tie_f1[:,:,-1])).mean(1)
            rec.update(tie_repeats=repeats,tie_increment_mean=float(tie_inc.mean()),
                tie_increment_low=float(np.quantile(tie_inc,.025)),tie_increment_high=float(np.quantile(tie_inc,.975)))
            tie_rows.extend(dict(pool=pool,source=source,repeat=i,increment=float(v)) for i,v in enumerate(tie_inc))
            summary.append(rec)
            for ri,(model,seed) in enumerate(zip(models,seeds)):
                correct=preds[ri]==rows.label.to_numpy()
                # Report overlap and distinct mistakes as descriptive sets, not matched-error breadth.
                full_errors={i for i in fset if not correct[i]};area_errors={i for i in aset if not correct[i]}
                runs.append(dict(pool=pool,source=source,scheme='endpoint_sets',model=model,seed=int(seed),
                    full_gap=float(full_gap[ri]),area_gap=float(area_gap[ri]),increment=float(increment[ri]),
                    full_unique_errors=len(full_errors-area_errors),area_unique_errors=len(area_errors-full_errors)))
            conditions.extend(area_condition(rows,states,preds,models,seeds,area_order,hypotheses[hypotheses.source==source],pool))
            for category in CLASSES:
                idx=np.flatnonzero(rows.class_name.to_numpy()==category)
                fields=states[source][rows.sample_id.iloc[idx[0]]].keys()
                for field in fields:
                    count=sum(states[source][rows.sample_id.iloc[i]][field] in ('unknown', ['unknown']) for i in idx)
                    unknown.append(dict(pool=pool,source=source,class_name=category,field=field,
                        targets=len(idx),unknown=count,unknown_fraction=count/len(idx)))
    if not smoke:
        for r,p in zip(summary,holm([r['p'] for r in summary])):r['holm_six_p']=float(p)
    for name,data in [('summary',summary),('run_metrics',runs),('memberships',members),('stratum_profiles',profiles),
                      ('unknown_rates',unknown),('area_condition_runs',conditions),('tie_sensitivity',tie_rows)]:
        pd.DataFrame(data).to_csv(out/'tables'/f'{name}.csv',index=False)
    cond=pd.DataFrame(conditions)
    keys=['pool','source','model','class_name','feature','value']
    grouped=cond.groupby(keys,dropna=False).agg(state_support=('state_support','first'),
        matched_state_support=('matched_state_support','first'),matched_reference_support=('matched_reference_support','first'),
        retained_fraction=('retained_fraction','first'),matched_bins=('matched_bins','first'),
        evaluable=('evaluable','first'),area_adjusted_recall_gap=('area_adjusted_recall_gap','mean')).reset_index()
    grouped.to_csv(out/'tables/area_condition_summary.csv',index=False)
    assert all(sha(inputs/name)==digest for name,digest in before.items())
    audit=dict(status='pass',smoke=smoke,sklearn_cells_checked=audit_checks,
        portable_inputs=len(files),unchanged_input_files=len(before),settings=len(summary),
        tie_repeats=repeats,no_training=True,no_inference=True,no_vlm=True,
        full_order_matches_historical=True,semantic_ground_truth_validated=False)
    dump(out/'audit.json',audit);dump(out/'progress.json',dict(audit,stage='complete'))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--input',type=Path,default=OUT/'portable_inputs')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.input)
    analyze(args.input,args.output,args.smoke)
