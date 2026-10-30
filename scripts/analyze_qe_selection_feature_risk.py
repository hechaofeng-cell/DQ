#!/usr/bin/env python3
"""Trace selected-set states, five risk dimensions and observed classifier errors."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/QE_VisualState_CoverRisk_v1_20261013'
EXPERIMENT = ROOT/'artifacts/QE_ThreeStrategy_Full_20261013'
OUTPUT = ROOT/'artifacts/QE_FeatureRisk_Diagnostics_20261013'
CLASSES = ('dog','cat','horse','sheep','person','car')
DIMENSIONS = ('scale','visibility','occlusion','truncation','orientation')
BUDGETS = (20,40,100,200,400,600,800,1000,1200,1300)
METHODS = ('coverage','risk','risk_coverage')

def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))

def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stat(values, prefix):
    values=np.asarray(values,dtype=float)
    return {prefix+'_mean':float(values.mean()),prefix+'_selection_p025':float(np.quantile(values,.025)),prefix+'_selection_p975':float(np.quantile(values,.975))}

def tiers(z):
    ids=sorted(range(len(z['sample_ids'])),key=lambda i:(z['risk_midpoint'][i],str(z['sample_ids'][i])))
    result=np.zeros(len(ids),dtype=int)
    for k,portion in enumerate(np.array_split(ids,5),1):result[portion]=k
    assert np.array_equal(result==5,z['high_risk'].astype(bool))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    records_path=SOURCE/'prepared/records.jsonl'
    records={r['sample_id']:r for r in [json.loads(line) for line in records_path.read_text().splitlines()]}
    states=json.loads((SOURCE/'coverage_dictionary.json').read_text())['states'];assert len(states)==71
    summary=read_csv(EXPERIMENT/'summary.csv');summary_lookup={(r['source'],int(r['budget_per_class']),r['method']):r for r in summary}
    pred_path=EXPERIMENT/'merged_predictions.npz'
    with np.load(pred_path) as z:pred={k:z[k] for k in z.files}
    assert pred['predictions'].shape==(24,7800) and len(records)==7800
    global_idx={sid:i for i,sid in enumerate(pred['sample_ids'].tolist())};assert set(global_idx)==set(records)
    freeze={str(p):sha(p) for p in (Path(__file__).resolve(),records_path,SOURCE/'coverage_dictionary.json',EXPERIMENT/'summary.csv',pred_path)}
    layer_rows=[];layer_totals=[];dim_rows=[];dim_totals=[];state_rows=[];composition_rows=[];composition_totals=[]
    tier_profiles=[];tier_runs=[];tier_sources=[];sample_tier_rows=[];score_verifications=0
    for backend in ('local','online'):
        prepared=SOURCE/'prepared' if backend=='local' else SOURCE/'online_analysis/prepared'
        arrays={};parts={};class_tiers={};class_source={};class_global={}
        for ci,category in enumerate(CLASSES):
            path=prepared/f'{category}.npz';freeze[str(path)]=sha(path)
            with np.load(path) as z:arrays[category]={k:z[k] for k in z.files}
            z=arrays[category];assert len(z['sample_ids'])==1300
            rs=[records[str(s)] for s in z['sample_ids']];assert all(r['class_name']==category and int(r['label'])==ci for r in rs)
            risk=[r['annotations'][backend]['risk'] for r in rs]
            for key in ('lower','midpoint','upper'):
                assert np.allclose([r['score_'+key] for r in risk],z['risk_'+key],atol=1e-10,rtol=0)
                assert np.allclose([sum(20*r[d+'_'+key] for d in DIMENSIONS) for r in risk],z['risk_'+key],atol=1e-10,rtol=0)
            score_verifications+=len(rs)
            parts[category]={d:{key:np.array([r[d+'_'+key] for r in risk],dtype=float) for key in ('lower','midpoint','upper','known_weight')} for d in DIMENSIONS}
            class_tiers[category]=tiers(z);t=class_tiers[category]
            class_source[category]=np.array([r['source_pool'] for r in rs]);class_global[category]=np.array([global_idx[str(s)] for s in z['sample_ids']])
            for i,r in enumerate(rs):sample_tier_rows.append(dict(backend=backend,sample_id=r['sample_id'],category=category,source_pool=r['source_pool'],risk_layer=int(t[i]),risk_lower=float(z['risk_lower'][i]),risk_midpoint=float(z['risk_midpoint'][i]),risk_upper=float(z['risk_upper'][i])))
            previous_max=None
            for tier in range(1,6):
                member=t==tier;mid=z['risk_midpoint'][member];lo=z['risk_lower'][member];hi=z['risk_upper'][member]
                next_min=float(z['risk_midpoint'][t==tier+1].min()) if tier<5 else None
                tier_profiles.append(dict(backend=backend,category=category,risk_layer=tier,targets=int(member.sum()),risk_midpoint_min=float(mid.min()),risk_midpoint_max=float(mid.max()),risk_midpoint_mean=float(mid.mean()),risk_lower_mean=float(lo.mean()),risk_upper_mean=float(hi.mean()),unknown_score_fraction=float((hi>lo+1e-9).mean()),tied_with_previous=previous_max is not None and np.isclose(float(mid.min()),previous_max),tied_with_next=next_min is not None and np.isclose(float(mid.max()),next_min),tier_definition='per-category pooled midpoint quantile; sample_id breaks ties; relative difficulty, not absolute error risk'))
                previous_max=float(mid.max())
                errors=(pred['predictions'][:,class_global[category][member]]!=ci)
                for run,(model,seed) in enumerate(zip(pred['models'],pred['seeds'])):
                    tier_runs.append(dict(backend=backend,category=category,risk_layer=tier,model=str(model),seed=int(seed),targets=int(member.sum()),errors=int(errors[run].sum()),accuracy=float(1-errors[run].mean()),error_rate=float(errors[run].mean()),risk_midpoint_mean=float(mid.mean()),risk_lower_mean=float(lo.mean()),risk_upper_mean=float(hi.mean())))
                for pool in sorted(set(class_source[category])):
                    count=int((class_source[category][member]==pool).sum());tier_sources.append(dict(backend=backend,category=category,risk_layer=tier,source_pool=pool,targets=count,fraction=count/int(member.sum())))
        pools=sorted({r['source_pool'] for r in records.values()})
        for budget in BUDGETS:
            selection_path=EXPERIMENT/'selection'/f'{backend}_n{6*budget}.npz';freeze[str(selection_path)]=sha(selection_path)
            with np.load(selection_path) as z:selections={k:z[k] for k in z.files}
            assert set(selections)==set(METHODS)
            for method in METHODS:
                selected=selections[method];selected=selected[None] if selected.ndim==2 else selected
                reps=len(selected);assert selected.shape[1:]==(6,budget)
                assert reps==(1000 if method=='risk' and budget<1300 else 1)
                total_layers=np.zeros((reps,5));total_dims={d:{k:np.zeros(reps) for k in ('lower','midpoint','upper','unknown')} for d in DIMENSIONS}
                total_pools={pool:np.zeros(reps) for pool in pools}
                totalpresence=np.zeros(reps);totalsupport=np.zeros(reps)
                for ci,category in enumerate(CLASSES):
                    ids=selected[:,ci];z=arrays[category];assert all(len(set(i.tolist()))==budget for i in ids)
                    base=dict(backend=backend,budget_per_class=budget,total_targets=6*budget,method=method,category=category,repetitions=reps)
                    for tier in range(1,6):
                        counts=(class_tiers[category][ids]==tier).sum(1);total_layers[:,tier-1]+=counts
                        layer_rows.append(dict(**base,risk_layer=tier,**stat(counts,'targets'),**stat(counts/budget,'fraction')))
                    for d in DIMENSIONS:
                        p=parts[category][d];values={key:20*p[key][ids].mean(1) for key in ('lower','midpoint','upper')};values['unknown']=(p['known_weight'][ids]<1-1e-9).mean(1)
                        for key,v in values.items():total_dims[d][key]+=v/6
                        dim_rows.append(dict(**base,dimension=d,maximum_dimension_points=20,**{k:v for key in ('midpoint','lower','upper') for k,v in stat(values[key],key+'_points').items()},**stat(values['unknown'],'unknown_fraction')))
                    counts=z['matrix'][ids].sum(1);presence=counts>0;completion=np.minimum(counts/20,1)
                    totalpresence+=presence.mean(1)/6;totalsupport+=completion.mean(1)/6
                    for j,(field,state) in enumerate(states):
                        state_rows.append(dict(**base,field=field,state=state,support_requirement=20,**stat(counts[:,j],'targets'),**stat(counts[:,j]/budget,'fraction'),presence_coverage_probability=float(presence[:,j].mean()),zero_support_probability=float((counts[:,j]==0).mean()),**stat(completion[:,j],'support_completion'),fully_supported_probability=float((counts[:,j]>=20).mean())))
                    for pool in pools:
                        pc=(class_source[category][ids]==pool).sum(1);total_pools[pool]+=pc
                        composition_rows.append(dict(**base,source_pool=pool,**stat(pc,'targets'),**stat(pc/budget,'fraction')))
                totalbase=dict(backend=backend,budget_per_class=budget,total_targets=6*budget,method=method,category='ALL_CLASSES',repetitions=reps)
                for tier in range(1,6):layer_totals.append(dict(**totalbase,risk_layer=tier,**stat(total_layers[:,tier-1],'targets'),**stat(total_layers[:,tier-1]/(6*budget),'fraction')))
                for d in DIMENSIONS:dim_totals.append(dict(**totalbase,dimension=d,maximum_dimension_points=20,**{k:v for key in ('midpoint','lower','upper') for k,v in stat(total_dims[d][key],key+'_points').items()},**stat(total_dims[d]['unknown'],'unknown_fraction')))
                for pool in pools:composition_totals.append(dict(**totalbase,source_pool=pool,**stat(total_pools[pool],'targets'),**stat(total_pools[pool]/(6*budget),'fraction')))
                expected=summary_lookup[backend,budget,method]
                assert np.isclose(totalpresence.mean(),float(expected['presence_coverage']),atol=1e-10)
                assert np.isclose(totalsupport.mean(),float(expected['support_completion']),atol=1e-10)
                for key in ('lower','midpoint','upper'):
                    assert np.isclose(sum(v[key].mean() for v in total_dims.values()),float(expected['risk_'+key]),atol=1e-9,rtol=0)
                assert np.isclose(total_layers[:,4].mean()/(6*budget),float(expected['q5_fraction']),atol=1e-10)
                if method in ('risk','risk_coverage'):
                    assert np.all(total_layers==budget*6/5)
                print(json.dumps(dict(backend=backend,budget=budget,method=method,repetitions=reps,verified=True)),flush=True)
    tables={'selection_risk_layers.csv':layer_rows,'selection_risk_layer_totals.csv':layer_totals,'selection_dimension_scores.csv':dim_rows,'selection_dimension_score_totals.csv':dim_totals,'selection_feature_states.csv':state_rows,'selection_source_composition.csv':composition_rows,'selection_source_composition_totals.csv':composition_totals,'risk_tier_profiles.csv':tier_profiles,'risk_tier_model_run_metrics.csv':tier_runs,'risk_tier_source_composition.csv':tier_sources,'target_risk_tier_assignments.csv':sample_tier_rows}
    pooled=[]
    for backend in ('local','online'):
        for tier in range(1,6):
            for run,(model,seed) in enumerate(zip(pred['models'],pred['seeds'])):
                rows=[r for r in tier_runs if r['backend']==backend and r['risk_layer']==tier and r['model']==str(model) and r['seed']==int(seed)]
                assert len(rows)==6;targets=sum(r['targets'] for r in rows);errors=sum(r['errors'] for r in rows)
                pooled.append(dict(backend=backend,category='ALL_CLASSES',risk_layer=tier,model=str(model),seed=int(seed),targets=targets,errors=errors,accuracy=1-errors/targets,error_rate=errors/targets,risk_midpoint_mean=float(np.mean([r['risk_midpoint_mean'] for r in rows])),risk_lower_mean=float(np.mean([r['risk_lower_mean'] for r in rows])),risk_upper_mean=float(np.mean([r['risk_upper_mean'] for r in rows]))))
    tables['risk_tier_pooled_model_metrics.csv']=pooled
    tier_summary=[]
    for backend in ('local','online'):
        for category in (*CLASSES,'ALL_CLASSES'):
            for tier in range(1,6):
                rows=[r for r in (pooled if category=='ALL_CLASSES' else tier_runs) if r['backend']==backend and r['category']==category and r['risk_layer']==tier]
                assert len(rows)==24
                tier_summary.append(dict(backend=backend,category=category,risk_layer=tier,model_runs=24,targets_per_run=rows[0]['targets'],mean_risk_midpoint=rows[0]['risk_midpoint_mean'],mean_risk_lower=rows[0]['risk_lower_mean'],mean_risk_upper=rows[0]['risk_upper_mean'],mean_accuracy=float(np.mean([r['accuracy'] for r in rows])),mean_error_rate=float(np.mean([r['error_rate'] for r in rows])),min_run_error_rate=float(min(r['error_rate'] for r in rows)),max_run_error_rate=float(max(r['error_rate'] for r in rows))))
    tables['risk_tier_error_relationship.csv']=tier_summary
    monotonic=[]
    for backend in ('local','online'):
        for category in (*CLASSES,'ALL_CLASSES'):
            rows=sorted([r for r in tier_summary if r['backend']==backend and r['category']==category],key=lambda r:r['risk_layer'])
            errors=np.array([r['mean_error_rate'] for r in rows]);adjacent=np.diff(errors)
            relevant=pooled if category=='ALL_CLASSES' else tier_runs
            run_monotonic=0
            for model,seed in zip(pred['models'],pred['seeds']):
                rr=sorted([r for r in relevant if r['backend']==backend and r['category']==category and r['model']==str(model) and r['seed']==int(seed)],key=lambda r:r['risk_layer'])
                assert len(rr)==5
                run_monotonic+=int(np.all(np.diff([r['error_rate'] for r in rr])>=-1e-12))
            monotonic.append(dict(backend=backend,category=category,mean_error_nondecreasing_Q1_to_Q5=bool(np.all(adjacent>=-1e-12)),highest_mean_error_tier=int(np.argmax(errors)+1),Q5_minus_Q1_error_rate_pp=100*float(errors[4]-errors[0]),Q5_minus_Q4_error_rate_pp=100*float(errors[4]-errors[3]),descending_adjacent_steps=int((adjacent<-1e-12).sum()),model_runs_with_nondecreasing_tier_error=run_monotonic,model_runs=24,interpretation='observed descriptive association only; no monotonicity assumed or calibrated-risk claim'))
    tables['risk_tier_monotonicity.csv']=monotonic
    full_equality_checks=0
    for source in ('local','online'):
        for rows,identity,measures in [(layer_rows,('category','risk_layer'),('targets_mean','fraction_mean')),(dim_rows,('category','dimension'),('midpoint_points_mean','lower_points_mean','upper_points_mean','unknown_fraction_mean')),(state_rows,('category','field','state'),('targets_mean','presence_coverage_probability','support_completion_mean','zero_support_probability'))]:
            full=[r for r in rows if r['backend']==source and r['budget_per_class']==1300]
            reference={tuple(r[k] for k in identity):r for r in full if r['method']=='coverage'}
            for r in full:
                ref=reference[tuple(r[k] for k in identity)]
                for key in measures:assert np.isclose(r[key],ref[key],atol=1e-12,rtol=0)
                full_equality_checks+=1
    for name,rows in tables.items():write_csv(out/name,rows)
    for name,rows in [('example_1200_risk_layers.csv',layer_totals),('example_1200_dimension_scores.csv',dim_totals),('example_1200_source_composition.csv',composition_totals)]:write_csv(out/name,[r for r in rows if r['total_targets']==1200])
    protocol=dict(state_definition='71 fixed states per category, 426 class-state units; unknown excluded without deleting missing states',risk_score='equal_global five dimensions each 20 points; visibility uses body_visibility and orientation uses body_orientation; head/legs/nominal properties do not contribute',main_score='risk_midpoint with lower/upper retained; midpoint does not resolve uncertainty',tiers='per class merged1300 midpoint order with sample_id tie-break, five equal groups260; Q5 is relative top20%, not calibrated failure probability',selection_interval='2.5% and97.5% quantiles over risk random repetitions; fixed methods one repetition so identical endpoints; no unneeded per-repetition target detail expansion',error_definition='official class labels vs24 frozen model predictions; errors observed after risk definition; no new inference/training',source_effect='pooled risk and coverage comparisons do not identify causal effects; source composition and unmeasured conditions can differ',new60_policy='fresh60 finer annotations are separate supplementary layer, not complete common coverage across7800',full_endpoint='all three methods select same7800 targets; per-class/tier/dimension distributions identical')
    (out/'selection_analysis_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    audit=dict(status='pass',backend_targets_recomputed=score_verifications,settings=60,selected_risk_layer_rows=len(layer_rows),selected_dimension_rows=len(dim_rows),selected_state_rows=len(state_rows),risk_tier_model_run_rows=len(tier_runs),all_selected_coverages_and_risk_scores_match_full_experiment=True,whole_pool_q5_equal260_per_class=True,random_risk_and_combo_each_layer_share_20_percent=True,full_endpoint_tier_dimension_state_rows_equality_verified=full_equality_checks)
    (out/'selection_analysis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    generated=list(tables)+['example_1200_risk_layers.csv','example_1200_dimension_scores.csv','example_1200_source_composition.csv','selection_analysis_protocol.json','selection_analysis_audit.json']
    freeze.update({str(out/name):sha(out/name) for name in generated})
    (out/'selection_analysis_freeze.sha256.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
