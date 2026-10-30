#!/usr/bin/env python3
"""Freeze risk quotas and model-independent orders before B predictions."""
import itertools
import json
from collections import Counter,defaultdict
import numpy as np
from independent_confirmation_common import *
from analyze_sixclass_difficulty import risk

EXCLUDE={'target_scale','body_visibility','occlusion_level','truncation_level','body_orientation','target_recognizability'}

def largest_remainder(cells,total=20):
    keys=sorted(cells);expected={k:total*len(cells[k])/80 for k in keys}
    counts={k:int(np.floor(expected[k])) for k in keys}
    remainder=total-sum(counts.values())
    for k in sorted(keys,key=lambda k:(-(expected[k]-counts[k]),k))[:remainder]:counts[k]+=1
    assert sum(counts.values())==total and all(n<=len(cells[k]) for k,n in counts.items())
    return counts

def constrained(section,tiers,cells,quotas,matrix=None,scores=None,embeddings=None):
    sequence=[];counts=np.zeros(matrix.shape[1],dtype=np.int32) if matrix is not None else None
    vectors=None;distance=None
    if embeddings is not None:vectors=embeddings/np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    for tier,subset in enumerate(tiers):
        remaining=set(subset);used=Counter()
        lookup={i:cell for cell,items in cells[tier].items() for i in items}
        for _ in range(20):
            feasible=[i for i in remaining if used[lookup[i]]<quotas[tier][lookup[i]]]
            assert feasible
            def key(i):
                if matrix is not None:
                    active=matrix[i].astype(bool)&(counts<20)
                    value=float((np.log(counts[active]+2)-np.log(counts[active]+1)).sum())
                elif scores is not None:value=float(scores[i])
                elif distance is None:
                    center=vectors[subset].mean(0);center/=max(np.linalg.norm(center),1e-12)
                    value=float(vectors[i]@center)
                else:value=float(distance[i])
                return (-value,section['rows'][i]['sample_id'])
            chosen=min(feasible,key=key);sequence.append(chosen);remaining.remove(chosen);used[lookup[chosen]]+=1
            if matrix is not None:counts+=matrix[chosen]
            if vectors is not None:
                current=1-vectors@vectors[chosen]
                distance=current if distance is None else np.minimum(distance,current)
        assert used==Counter({k:v for k,v in quotas[tier].items() if v})
    return sequence

def build():
    labels,contexts,universe=context();records=[];scores=[];quota_rows=[];fixed={};random={}
    with np.load(BASE/'audit/accepted_vit.npz') as saved:
        mapping={k:i for i,k in enumerate(saved['sample_ids'].tolist())}
        embeddings=saved['embeddings'][[mapping[r['sample_id']] for r in contexts['local']['rows']]]
    np.savez_compressed(BASE/'embeddings/vit_b16_candidates.npz',sample_ids=np.asarray([r['sample_id'] for r in contexts['local']['rows']]),embeddings=embeddings)
    rng=np.random.default_rng(20261004)
    for source,ctx in contexts.items():
        fixed[source]={method:{} for method in ('coverage','coverage_without_H','kcenter','cross_coverage')}
        random[source]=[{c:[] for c in CLASSES} for _ in range(1000)]
        for name in CLASSES:
            section=ctx['by_class'][name]
            values=[risk(row,labels[source][row['sample_id']]['features']) for row in section['rows']]
            ranked=sorted(range(400),key=lambda i:(values[i]['h_midpoint'],section['rows'][i]['sample_id']))
            tiers=[ranked[t*80:(t+1)*80] for t in range(5)]
            cells=[];quotas=[]
            for tier,subset in enumerate(tiers,1):
                grouping=defaultdict(list)
                for i in subset:
                    r=values[i];grouping[r['h_midpoint'],r['unknown_dimensions']].append(i)
                    scores.append(dict(source=source,class_name=name,sample_id=section['rows'][i]['sample_id'],tier=tier,**r))
                quota=largest_remainder(grouping);cells.append(grouping);quotas.append(quota)
                quota_rows.extend(dict(source=source,class_name=name,tier=tier,h_midpoint=k[0],unknown_dimensions=k[1],candidates=len(grouping[k]),quota=v) for k,v in quota.items())
                for repeat in range(1000):
                    for k in sorted(grouping):
                        if quota[k]:random[source][repeat][name].extend(rng.choice(grouping[k],quota[k],replace=False).tolist())
            keep=np.asarray([field not in EXCLUDE for field,value in section['states']])
            opposite=contexts['online' if source=='local' else 'local']['by_class'][name]
            fixed[source]['coverage'][name]=constrained(section,tiers,cells,quotas,matrix=section['matrix'])
            fixed[source]['coverage_without_H'][name]=constrained(section,tiers,cells,quotas,matrix=section['matrix'][:,keep])
            fixed[source]['cross_coverage'][name]=constrained(section,tiers,cells,quotas,matrix=opposite['matrix'])
            fixed[source]['kcenter'][name]=constrained(section,tiers,cells,quotas,embeddings=embeddings[section['global_indices']])
        for method,orders in fixed[source].items():
            records.extend(dict(source=source,method=method,class_name=c,rank=i+1,tier=i//20+1,sample_id=contexts[source]['by_class'][c]['rows'][j]['sample_id']) for c in CLASSES for i,j in enumerate(orders[c]))
    write_csv(BASE/'selection/risk_scores.csv',scores);write_csv(BASE/'selection/quotas.csv',quota_rows)
    write_csv(BASE/'selection/model_independent_orders.csv',records)
    write_csv(BASE/'selection/aligned_state_universe.csv',universe)
    np.savez_compressed(BASE/'selection/random_orders.npz',**{s:np.asarray([[order[c] for c in CLASSES] for order in groups],dtype=np.int16) for s,groups in random.items()})
    write_json(BASE/'selection/model_independent_orders.json',fixed)
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/analyze_sixclass_difficulty.py',ROOT/'scripts/analyze_independent_sixclass.py',ROOT/'scripts/run_sixclass_independent_confirmation.py',
           BASE/'manifests/candidate_pool.csv',BASE/'transfer/weak_states_A_frozen.csv',BASE/'embeddings/vit_b16_candidates.npz']
    paths+=list((BASE/'selection').glob('*'))+list((BASE/'annotations').glob('*/*/parsed/*.json'))
    freeze(BASE/'pre_prediction_freeze.sha256.json',paths)
    write_json(BASE/'selection/audit.json',dict(status='pass',quota_rule='whole stratum proportional largest remainder, no classifier outputs',random_replicates=1000,
               model_independent_methods=list(fixed['local']),prediction_read=False,classes=6,targets=2400))
    return labels,contexts,fixed,random

def load():
    labels,contexts,_=context();fixed=json.loads((BASE/'selection/model_independent_orders.json').read_text())
    with np.load(BASE/'selection/random_orders.npz') as saved:
        random={s:[{c:arr[i,j].tolist() for j,c in enumerate(CLASSES)} for i in range(len(arr))] for s,arr in saved.items()}
    return labels,contexts,fixed,random

def risk_structure(source,name,contexts):
    rows=read_csv(BASE/'selection/risk_scores.csv');section=contexts[source]['by_class'][name]
    lookup={r['sample_id']:i for i,r in enumerate(section['rows'])};tiers=[];cells=[];quotas=[]
    quota_rows=read_csv(BASE/'selection/quotas.csv')
    for tier in range(1,6):
        subset=[r for r in rows if r['source']==source and r['class_name']==name and int(r['tier'])==tier]
        groups=defaultdict(list)
        for r in subset:groups[float(r['h_midpoint']),int(r['unknown_dimensions'])].append(lookup[r['sample_id']])
        q={(float(r['h_midpoint']),int(r['unknown_dimensions'])):int(r['quota']) for r in quota_rows if r['source']==source and r['class_name']==name and int(r['tier'])==tier}
        tiers.append([lookup[r['sample_id']] for r in subset]);cells.append(groups);quotas.append(q)
    return tiers,cells,quotas

if __name__=='__main__':
    (BASE/'embeddings').mkdir(parents=True,exist_ok=True);build()
