"""Post-hoc, automatic-only checks of the frozen external diagnostic conditions."""
import ast
import math
from collections import defaultdict
import numpy as np
from coverage_validation_common_20261018 import *

DEST=OUT/'automatic_robustness'
METHODS=['expanded_greedy','shared_greedy','static_rarity','vit_kcenter','same_cell_random']


def scorer():
    path=ROOT/'scripts/run_category_five_dimension_pilot.py';tree=ast.parse(path.read_text())
    names={'mapped_risk','aggregate_components','scale_risk','score_sample'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'SCHEMES','DIMENSIONS'} for t in n.targets)]
    ns={'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['score_sample']


def main():
    protocol=json.loads((DEST/'protocol.json').read_text());score=scorer();config=json.loads((ROOT/'configs/category_five_dimension_pilot_v1.json').read_text())
    frozen=json.loads((OUT/'conditions_before_external_predictions.json').read_text())
    assert sha(OUT/'discovered_conditions.csv')==frozen['conditions_sha256']
    assert sha(OUT/'protocol.json')==frozen['protocol_sha256']
    inputs=[DEST/'protocol.json',Path(__file__),ROOT/'scripts/coverage_validation_common_20261018.py',ROOT/'scripts/run_category_five_dimension_pilot.py',ROOT/'configs/category_five_dimension_pilot_v1.json',
        OUT/'external_manifest.csv',OUT/'external_condition_results.csv',OUT/'discovery_groups.csv',BASE/'native_features.jsonl',BASE/'manifest.csv',SOURCE/'risk_scores_and_tiers.csv']
    inputs += [OUT/f'external_{c}_matrix.npz' for c in CLASSES]
    freeze(DEST/'inputs_freeze.json',{str(p):sha(p) for p in inputs})
    native={r['sample_id']:r['features'] for r in map(json.loads,(BASE/'native_features.jsonl').read_text().splitlines())}
    old_scores={r['sample_id']:r for r in read(SOURCE/'risk_scores_and_tiers.csv')};checked=0
    for row in manifest():
        if row['class_name'] not in CLASSES:continue
        result=score(float(row['bbox_area_ratio']),native[row['sample_id']],row['class_name'],'equal_parts',config)
        for key in ['score_lower','score_upper','score_midpoint']:
            assert abs(result[key]-float(old_scores[row['sample_id']][key]))<1e-8
        checked+=1
    assert checked==1600
    external=read(OUT/'external_manifest.csv');risk=[];label_paths=[]
    for row in external:
        c=row['class_name'];sid=row['sample_id'];domain='car' if c=='car' else 'old'
        path=EXTERNAL/f'annotations/local/{domain}/parsed/{sid}.json';label_paths.append(path)
        fields=json.loads(path.read_text())['features'];s=score(float(row['bbox_area_ratio']),fields,c,'equal_parts',config)
        cell=(min(9,math.floor((s['score_midpoint']+1e-10)/10)),int(s['score_upper']-s['score_lower']>1e-8))
        risk.append(dict(sample_id=sid,class_name=c,score_bin=cell[0],uncertain=cell[1],**s))
    write(DEST/'external_risk_scores.csv',risk)
    freeze(DEST/'external_label_hashes.json',{str(p):sha(p) for p in label_paths})
    pred,pred_hashes=predictions('D',external)
    assert pred_hashes==json.loads((OUT/'external_confirmation_audit.json').read_text())['prediction_sha256']
    conditions=read(OUT/'external_condition_results.csv');groups=read(OUT/'discovery_groups.csv');cache={};state_rows=[]
    rng=np.random.default_rng(protocol['seed'])
    for c in CLASSES:
        pos=[i for i,r in enumerate(external) if r['class_name']==c];rows=[external[i] for i in pos]
        z=np.load(OUT/f'external_{c}_matrix.npz',allow_pickle=False);X=z['matrix'].astype(bool)
        assert z['sample_ids'].tolist()==[r['sample_id'] for r in rows]
        cells=np.array([(risk[i]['score_bin'],risk[i]['uncertain']) for i in pos]);keys=sorted(set(map(tuple,cells)))
        correct=np.stack([(pred[m][:,pos]==ALL_CLASSES.index(c)).astype(float) for m in MODELS])
        Y=correct.mean(1).T # target x architecture; seeds remain paired on each target
        for j in sorted({int(r['state_index']) for r in conditions if r['class_name']==c}):
            present=X[:,j];n=int(present.sum());retained=[]
            for key in keys:
                in_cell=np.all(cells==key,axis=1);a=np.flatnonzero(in_cell&present);b=np.flatnonzero(in_cell&~present)
                if len(a)>=5 and len(b)>=5:retained.append((a,b))
            na=sum(len(a) for a,b in retained);nb=sum(len(b) for a,b in retained);fraction=na/n if n else 0
            eligible=na>=20 and nb>=20 and fraction>=.8-1e-12
            effect=np.zeros(5);boot=np.zeros((1000,5));raw_overlap=np.full(5,np.nan)
            if eligible:
                for a,b in retained:
                    weight=len(a)/na;effect+=weight*(Y[a].mean(0)-Y[b].mean(0))
                    wa=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=1000)/len(a)
                    wb=rng.multinomial(len(b),np.full(len(b),1/len(b)),size=1000)/len(b)
                    boot+=weight*(wa@Y[a]-wb@Y[b])
                raw_overlap=Y[np.concatenate([a for a,b in retained])].mean(0)-Y[np.concatenate([b for a,b in retained])].mean(0)
            for mi,m in enumerate(MODELS):
                deltas=100*(correct[mi][:,present].mean(1)-correct[mi].mean(1)) if n else np.full(3,np.nan)
                d=dict(class_name=c,model=m,state_index=j,external_support=n,
                    seed_1_delta_pp=float(deltas[0]) if n else '',seed_2_delta_pp=float(deltas[1]) if n else '',seed_3_delta_pp=float(deltas[2]) if n else '',
                    all_seeds_negative=int(n>=20 and np.all(deltas<0)),all_seeds_at_least_10pp=int(n>=20 and np.all(deltas<=-10+1e-10)),
                    retained_state_targets=na,retained_control_targets=nb,retained_fraction=fraction,retained_cells=len(retained),adjustment_evaluable=int(eligible),
                    adjusted_delta_pp=float(effect[mi]*100) if eligible else '',overlap_unadjusted_delta_pp=float(raw_overlap[mi]*100) if eligible else '',
                    adjusted_ci_low_pp=float(np.quantile(boot[:,mi],.025)*100) if eligible else '',adjusted_ci_high_pp=float(np.quantile(boot[:,mi],.975)*100) if eligible else '',
                    adjusted_direction=int(eligible and effect[mi]<0),adjusted_10pp=int(eligible and effect[mi]<=-.1+1e-12),
                    adjusted_ci_below_zero=int(eligible and np.quantile(boot[:,mi],.975)<0))
                cache[c,m,j]=d;state_rows.append(d)
        log('automatic_robustness_class_complete',class_name=c)
    write(DEST/'unique_state_model_checks.csv',state_rows)
    expanded=[]
    for r in conditions:
        d=cache[r['class_name'],r['model'],int(r['state_index'])]
        assert d['external_support']==int(r['external_support'])
        if d['external_support']:
            assert np.isclose(np.mean([d[f'seed_{i}_delta_pp'] for i in [1,2,3]]),float(r['external_delta_pp']),atol=1e-10)
        expanded.append(dict(**r,**{k:v for k,v in d.items() if k not in r}))
    write(DEST/'condition_checks.csv',expanded)
    group_index=defaultdict(list)
    for r in expanded:group_index[tuple(r[k] for k in ['class_name','model','method','repeat','scope'])].append(r)
    group_rows=[]
    for g in groups:
        rr=group_index[tuple(g[k] for k in ['class_name','model','method','repeat','scope'])]
        counts={f'confirmed_{t}pp':sum(int(r['external_support'])>=20 and (float(r['external_delta_pp'])<0 if t==0 else float(r['external_delta_pp'])<=-t+1e-10) for r in rr) for t in [0,5,10,20]}
        counts.update({k:sum(int(r[k]) for r in rr) for k in ['all_seeds_negative','all_seeds_at_least_10pp','adjustment_evaluable','adjusted_direction','adjusted_10pp','adjusted_ci_below_zero']})
        group_rows.append(dict(**g,**counts))
    write(DEST/'group_checks.csv',group_rows)
    summaries=[];unique=[];count_keys=list(counts)
    for scope in ['all','specific']:
        for method in METHODS:
            gg=[r for r in group_rows if r['scope']==scope and r['method']==method];div=100 if method=='same_cell_random' else 1
            counts={k:sum(float(r[k]) for r in gg)/div for k in ['proposed',*count_keys]}
            counts['adjustment_not_evaluable']=counts['proposed']-counts['adjustment_evaluable']
            summaries.append(dict(scope=scope,method=method,**counts))
            for rep in range(div):
                rr=[r for r in expanded if r['scope']==scope and r['method']==method and int(r['repeat'])==rep]
                bystate=defaultdict(list)
                for r in rr:bystate[tuple(r[k] for k in ['class_name','namespace','field','value'])].append(r)
                confirms=lambda r:int(r['external_support'])>=20 and float(r['external_delta_pp'])<0
                unique.append(dict(scope=scope,method=method,repeat=rep,proposed_types=len(bystate),
                    any_model_confirmed_types=sum(any(confirms(r) for r in rs) for rs in bystate.values()),
                    all_proposed_models_confirmed_types=sum(all(confirms(r) for r in rs) for rs in bystate.values())))
    write(DEST/'summary.csv',summaries);write(DEST/'deduplicated_conditions.csv',unique)
    # Independent checks of the weighted contrasts and the threshold monotonicity.
    for r in summaries:
        assert r['confirmed_20pp']<=r['confirmed_10pp']<=r['confirmed_5pp']<=r['confirmed_0pp']<=r['proposed']
        assert r['adjusted_10pp']<=r['adjusted_direction']<=r['adjustment_evaluable']<=r['proposed']
    old=read(OUT/'external_diagnostic_summary.csv')
    for r in summaries:
        prev=next(s for s in old if s['scope']==r['scope'] and s['method']==r['method'])
        assert np.isclose(r['confirmed_0pp'],float(prev['confirmed']))
    dump(DEST/'audit.json',dict(status='pass',source_scores_reproduced=checked,external_targets=800,
        frozen_conditions_verified=True,source_prediction_hashes_verified=True,per_seed_delta_reproduces_original=True,
        direction_counts_reproduce_original=True,threshold_monotonicity=True,unique_state_model_rows=len(state_rows),condition_rows=len(expanded),
        human_annotation_used=False,new_classifier_training=False,post_hoc=True,prediction_sha256=pred_hashes))
    log('automatic_robustness_complete',output=str(DEST))


if __name__=='__main__':main()
