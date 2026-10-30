#!/usr/bin/env python3
"""Post-hoc, model-output-independent risk strata on the amended paired pool."""
from __future__ import annotations
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dual_vlm_common import ROOT, CLASSES, MODELS, SEEDS, read_csv, write_csv, write_json, sha256, now
import analyze_dual_vlm_study as a

SOURCE=ROOT/'artifacts/sixclass_dual_vlm_replaced_20261004'
OUTPUT=ROOT/'artifacts/sixclass_difficulty_supplement_20261004'
MAPS={'body_visibility':{'full':0,'most':0,'partial':2,'minimal':3},
      'occlusion_level':{'none':0,'light':1,'moderate':2,'heavy':3},
      'truncation_level':{'none':0,'light':.5,'moderate':1,'heavy':2},
      'body_orientation':{'front':0,'back':1,'left_side':0,'right_side':0,'three_quarter':0,'overhead_or_unusual':2}}


def note(stage,**extras):
    value=dict(status='running',stage=stage,heartbeat=now(),**extras)
    write_json(OUTPUT/'progress.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def risk(row,features):
    area=float(row['bbox_area_ratio'])
    size=0 if area>=.10 else 1 if area>=.03 else 2 if area>=.01 else 3
    low,high,unknown=float(size),float(size),0
    parts={'size':size}
    for field,values in MAPS.items():
        value=features[field]
        if value=='unknown':
            lo,hi=0,max(values.values())
            unknown+=1
        else:
            lo=hi=values[value]
        low+=lo; high+=hi
        parts[field]=(lo+hi)/2
    return dict(h_lower=low,h_upper=high,h_midpoint=(low+high)/2,unknown_dimensions=unknown,**parts)


def metrics(rows,predicted,indices):
    true=np.asarray([int(rows[i]['label']) for i in indices])
    pred=predicted[np.asarray(indices)]
    precision,recall,f1,_=precision_recall_fscore_support(true,pred,labels=list(range(6)),zero_division=0)
    return dict(accuracy=float(accuracy_score(true,pred)),macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()),macro_f1=float(f1.mean()),
                **{f'{name}_recall':float(recall[i]) for i,name in enumerate(CLASSES)})


def configure():
    a.STUDY=SOURCE
    a.job_path=lambda backend,domain:SOURCE/'annotations'/backend/domain
    a.schema_path=lambda domain:SOURCE/'configs'/('car_annotation_schema.json' if domain=='car' else 'old_annotation_schema.json')
    labels,_=a.native_labels(False)
    contexts,_=a.contexts(labels,False)
    return labels,contexts


def build_orders(contexts,labels):
    score_rows,assignments,orders,random_orders,profiles=[],{},{},{},[]
    rng=np.random.default_rng(20261004)
    for source,context in contexts.items():
        risks={r['sample_id']:risk(r,labels[source][r['sample_id']]['features']) for r in context['rows']}
        orders[source]={}; random_orders[source]=[{c:[] for c in CLASSES} for _ in range(1000)]
        for mode in ('midpoint','lower','upper'):
            assignments[source,mode]={}
            for name in CLASSES:
                section=context['by_class'][name]
                ranked=sorted(range(400),key=lambda i:(risks[section['rows'][i]['sample_id']][f'h_{mode}'],section['rows'][i]['sample_id']))
                assignments[source,mode][name]=[ranked[i*80:(i+1)*80] for i in range(5)]
        for name in CLASSES:
            section=context['by_class'][name]
            matrix=section['matrix']
            counts=np.zeros(matrix.shape[1],dtype=np.int32)
            sequence=[]
            tiers=assignments[source,'midpoint'][name]
            for tier,subset in enumerate(tiers,1):
                remaining=set(subset); chosen=[]
                for _ in range(20):
                    def key(i):
                        active=matrix[i].astype(bool)&(counts<20)
                        gain=float((np.log(counts[active]+2)-np.log(counts[active]+1)).sum())
                        return (-gain,section['rows'][i]['sample_id'])
                    winner=min(remaining,key=key)
                    chosen.append(winner); remaining.remove(winner); counts+=matrix[winner]
                sequence.extend(chosen)
                cells=defaultdict(list)
                def cell(i):
                    rr=risks[section['rows'][i]['sample_id']]
                    return (rr['h_midpoint'],rr['unknown_dimensions'])
                for i in subset: cells[cell(i)].append(i)
                wanted=Counter(cell(i) for i in chosen)
                for repetition in range(1000):
                    picked=[]
                    for identity,number in sorted(wanted.items()):
                        picked.extend(rng.choice(cells[identity],size=number,replace=False).tolist())
                    assert Counter(cell(i) for i in picked)==wanted and len(set(picked))==20
                    random_orders[source][repetition][name].extend(picked)
                scores=[risks[section['rows'][i]['sample_id']]['h_midpoint'] for i in subset]
                profiles.append(dict(source=source,class_name=name,tier=tier,candidates=80,selected=20,
                                     h_min=min(scores),h_max=max(scores),h_mean=float(np.mean(scores)),
                                     tied_with_next=bool(tier<5 and max(scores)==risks[section['rows'][tiers[tier][0]]['sample_id']]['h_midpoint'])))
                for i in subset:
                    row=section['rows'][i]; rr=risks[row['sample_id']]
                    score_rows.append(dict(source=source,class_name=name,sample_id=row['sample_id'],tier=tier,
                                           selected=i in chosen,**rr))
            orders[source][name]=sequence
    write_csv(OUTPUT/'tables/difficulty_scores.csv',score_rows)
    write_csv(OUTPUT/'tables/stratum_profiles.csv',profiles)
    order_rows=[]
    for source,context in contexts.items():
        for name,sequence in orders[source].items():
            order_rows.extend(dict(source=source,class_name=name,tier=i//20+1,rank=i+1,
                                   sample_id=context['by_class'][name]['rows'][j]['sample_id']) for i,j in enumerate(sequence))
    write_csv(OUTPUT/'tables/stratified_coverage_order.csv',order_rows)
    random_path=OUTPUT/'random_orders.npz'
    np.savez_compressed(random_path,**{source:np.asarray([[order[c] for c in CLASSES] for order in values],dtype=np.int16)
                                      for source,values in random_orders.items()})
    return assignments,orders,random_orders


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    note('freeze_labels_and_risk_rules')
    labels,contexts=configure()
    assignments,orders,random_sets=build_orders(contexts,labels)
    paths=[Path(__file__),ROOT/'docs/sixclass_difficulty_supplement_protocol_20261004.md',
           OUTPUT/'tables/difficulty_scores.csv',OUTPUT/'tables/stratified_coverage_order.csv',OUTPUT/'random_orders.npz',
           SOURCE/'amendment_input_freeze.sha256.json',SOURCE/'prediction_freeze.sha256.json']
    paths+=list((SOURCE/'annotations').glob('*/*/parsed/*.json'))
    paths+=list((SOURCE/'benchmark/runs').glob('*/*/candidate_outputs.npz'))
    write_json(OUTPUT/'input_code_order_freeze.sha256.json',{str(p):sha256(p) for p in paths})
    full,batches,cumulative,random_records,strict,states=[],[],[],[],[],[]
    contrasts=defaultdict(lambda:defaultdict(list)); gains=defaultdict(lambda:defaultdict(list))
    rows=contexts['local']['rows']
    for run_i,(model,seed) in enumerate(itertools.product(MODELS,SEEDS),1):
        note('frozen_prediction_analysis',completed_runs=run_i-1,total_runs=24,model=model,seed=seed)
        with np.load(SOURCE/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
            assert saved['sample_ids'].tolist()==[r['sample_id'] for r in rows]
            predicted=saved['logits'].argmax(1); features=saved['features']
        structures=a.support.error_structures(contexts['local'],features,predicted,model,seed)
        for source,context in contexts.items():
            for mode in ('midpoint','lower','upper'):
                curve=[]
                for tier in range(1,6):
                    indices=[int(context['by_class'][c]['global_indices'][i]) for c in CLASSES for i in assignments[source,mode][c][tier-1]]
                    value=metrics(rows,predicted,indices)
                    full.append(dict(source=source,unknown_mode=mode,model=model,seed=seed,tier=tier,n=480,**value))
                    curve.append(value['macro_f1'])
                contrasts[source,mode][model].append(curve[-1]-curve[0])
            previous=-1
            strict_curve=[]
            for tier in range(1,6):
                budget=tier*20
                indices=[int(context['by_class'][c]['global_indices'][i]) for c in CLASSES for i in orders[source][c][budget-20:budget]]
                batches.append(dict(source=source,model=model,seed=seed,tier=tier,n=120,**metrics(rows,predicted,indices)))
                measured=a.measured_one(context,orders[source],budget,predicted)
                assert measured['macro_frequency_coverage']>=previous-1e-12
                previous=measured['macro_frequency_coverage']
                opposite='online' if source=='local' else 'local'
                cross=a.measured_one(contexts[opposite],orders[source],budget,predicted)
                cumulative.append(dict(source=source,model=model,seed=seed,tier=tier,budget=budget,n=budget*6,
                                       cross_source_coverage=cross['macro_frequency_coverage'],**measured))
                selection=a.prepared_selection(context,random_sets[source],budget)
                values=a.measures(context,selection,predicted,a.state_stats(context,selection,predicted))
                random_records.append(dict(source=source,model=model,seed=seed,tier=tier,budget=budget,
                                           **a.random_summary(values)))
                pairs=[a.control.class_stratified_pair(context,orders[source],random_order,budget,predicted,structures) for random_order in random_sets[source]]
                aggregate,_=a.control.aggregate_random_pairs(pairs)
                value=a.control.add_delta(dict(source=source,model=model,seed=seed,tier=tier,budget=budget,**aggregate))
                strict.append(value); strict_curve.append(value)
            for metric in ('confusion','clusters'):
                gains[source,metric][model].append(a.stats.normalized_auc(strict_curve,f'delta_{metric}'))
            for name in CLASSES:
                section=context['by_class'][name]
                correct=predicted[section['global_indices']]==CLASSES.index(name)
                for i,(field,value) in enumerate(section['states']):
                    keep=section['matrix'][:,i].astype(bool)
                    if keep.sum()>=20:
                        states.append(dict(source=source,class_name=name,model=model,seed=seed,feature=field,value=value,
                                           support=int(keep.sum()),recall=float(correct[keep].mean())))
    comparisons=[]
    for (source,mode),grouped in contrasts.items():
        comparisons.append(dict(source=source,unknown_mode=mode,
                                **a.stats.paired_stats(grouped,f'{source}_{mode}_Q5_minus_Q1','macro_f1')))
    for row,p in zip(comparisons,a.control.holm([r['permutation_p'] for r in comparisons])):row['holm_six_p']=p
    matched=[]
    for (source,metric),grouped in gains.items():
        matched.append(dict(source=source,**a.stats.paired_stats(grouped,f'{source}_coverage_vs_exact_risk_random',metric)))
    for row,p in zip(matched,a.control.holm([r['permutation_p'] for r in matched])):row['holm_four_p']=p
    for filename,data in (('full_stratum_metrics',full),('selected_batch_metrics',batches),('cumulative_metrics',cumulative),
                          ('matched_random_summary',random_records),('matched_equal_error_metrics',strict),
                          ('difficulty_contrasts',comparisons),('matched_equal_error_comparisons',matched),('state_recall',states)):
        write_csv(OUTPUT/'tables'/f'{filename}.csv',data)
    rankings=[]
    for source in ('local','online'):
        for model in MODELS:
            curve=[float(np.mean([r['macro_f1'] for r in full if r['source']==source and r['unknown_mode']=='midpoint' and r['model']==model and r['tier']==tier])) for tier in range(1,6)]
            endpoint=[r['macro_f1'] for r in cumulative if r['source']==source and r['model']==model and r['budget']==100]
            rankings.append(dict(source=source,model=model,full_strata_auc=float(np.trapezoid(curve,x=np.arange(1,6))/4),
                                 low_risk_f1=curve[0],high_risk_f1=curve[-1],drop=curve[-1]-curve[0],
                                 selected_cumulative_f1=float(np.mean(endpoint))))
    write_csv(OUTPUT/'tables/model_robustness.csv',rankings)
    expected={'full':720,'batches':240,'cumulative':240,'random':240,'strict':240}
    for key,data in [('full',full),('batches',batches),('cumulative',cumulative),('random',random_records),('strict',strict)]:
        assert len(data)==expected[key]
        assert all(np.isfinite(value) for row in data for value in row.values() if isinstance(value,(float,np.floating)))
    frozen=json.loads((OUTPUT/'input_code_order_freeze.sha256.json').read_text())
    assert all(sha256(p)==digest for p,digest in frozen.items())
    write_json(OUTPUT/'summary.json',dict(difficulty_contrasts=comparisons,matched_equal_error=matched,
               status='complete',analysis_type='post_hoc_supplement',unknown_policy='midpoint; lower/upper sensitivity',
               random_control='1000 exact per-class per-stratum (H,unknown_dimensions) matched subsets'))
    write_json(OUTPUT/'final_audit.json',dict(status='pass',targets=2400,runs=24,random_replicates=1000,
               full_stratum_n=480,selected_batch_n=120,selected_endpoint_n=600,observed_rows=expected,
               input_freeze_unchanged=True,source_order_unchanged=True,random_risk_histograms_exact=True))
    note('complete',completed_runs=24,total_runs=24)
    write_json(OUTPUT/'progress.json',dict(status='complete',heartbeat=now()))


if __name__=='__main__':
    run()
