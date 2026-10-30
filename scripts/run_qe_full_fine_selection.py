#!/usr/bin/env python3
"""Fixed full-fine dictionaries, then class-balanced coverage/risk selection."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

import run_fresh_vlm_feature_scoring as fresh
from build_category_dimension_review import grouping
from build_qe_visualstate_coverrisk_dataset import POOLS

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/QE_FullFine_v4_20261013'
CLASSES=fresh.CLASSES
DIMENSIONS=('scale','visibility','occlusion','truncation','orientation')
DEFAULT_BUDGETS=(20,40,100,200,400,600,800)
MIXTURES={'balanced':'0.2,0.2,0.2,0.2,0.2','challenge':'0.1,0.15,0.2,0.25,0.3'}
UNKNOWN={'unknown','not_observable'}
GROUP={'尺度':'scale','可见性':'visibility','遮挡':'occlusion','截断':'truncation','朝向':'orientation','关联条件':'auxiliary'}
MODELS=('resnet18','resnet50','mobilenet_v3_small','mobilenet_v3_large','densenet121','efficientnet_b0','convnext_tiny','swin_t')
SEEDS=(20260916,20260917,20260918)

def dump(path,obj):fresh.atomic_json(path,obj)
def read_csv(path):return fresh.pilot.read_csv(path)
def write_csv(path,rows):fresh.pilot.write_csv(path,rows)
def sha(path):return fresh.pilot.sha256(path)

def code_dependencies():
    names=('run_qe_full_fine_selection.py','run_fresh_vlm_feature_scoring.py','run_category_five_dimension_pilot.py','run_dog_local_pipeline.py','build_category_dimension_review.py','build_qe_visualstate_coverrisk_dataset.py','run_qe_three_strategy_full_experiment.py','summarize_qe_three_strategy_full_experiment.py','export_qe_full_fine_feature_scores.py')
    return [ROOT/'scripts'/name for name in names]

def support_counts(m,selection,chunk=64):
    counts=np.empty((len(selection),m.shape[1]),dtype=np.int32)
    for start in range(0,len(selection),chunk):counts[start:start+chunk]=m[selection[start:start+chunk]].sum(1)
    return counts

def excluded_state_reason(schema,field,value):
    if schema.get('category')!='car' and field in ('torso_visibility','limb_visibility') and value=='not_in_view':
        return 'v4 semantic protocol forbids noncar torso/limb not_in_view; original-frame loss is not_visible, insufficient evidence is unknown'
    return None

def dictionary(schema,apply_semantic_constraints=True):
    states=[dict(field='target_scale',value=x,coverage_group='scale',source='official_bbox') for x in ('tiny','small','medium','large')]
    for field in schema['universal_features']+schema.get('category_specific_features',[]):
        assert field['value_type'] in ('enum','set')
        dim,sub=grouping(field['feature_id'])
        for value in field['possible_values']:
            if value not in UNKNOWN:
                if apply_semantic_constraints and excluded_state_reason(schema,field['feature_id'],value):continue
                states.append(dict(field=field['feature_id'],value=value,coverage_group=GROUP[dim],subgroup=sub,source='VLM',nominal_not_ordinal_risk=GROUP[dim]=='auxiliary'))
    assert len({(x['field'],x['value']) for x in states})==len(states)
    return states

def prepare(root,out):
    root=root.resolve();out.mkdir(parents=True,exist_ok=True)
    entries=[];excluded=[];freeze={str(path.resolve()):sha(path) for path in code_dependencies()}
    for category in CLASSES:
        path=root/'schemas'/f'{category}.json'
        schema=json.loads(path.read_text()) if path.exists() else fresh.build_schema(category)
        if path.exists():freeze[str(path)]=sha(path)
        states=dictionary(schema)
        raw=dictionary(schema,False);omitted=[dict(class_name=category,field=s['field'],value=s['value'],reason=excluded_state_reason(schema,s['field'],s['value']),exclusion_basis='structurally inadmissible by frozen v4 semantic protocol, not removed because unobserved') for s in raw if excluded_state_reason(schema,s['field'],s['value'])]
        excluded.extend(omitted)
        dump(out/'coverage_dictionaries'/f'{category}.json',dict(category=category,schema_version=schema['schema_version'],raw_schema_state_count=len(raw),fixed_state_count=len(states),effective_valid_state_count=len(states),excluded_structural_states=omitted,states=states))
        entries.append(dict(class_name=category,fields=len(schema['universal_features']),raw_schema_states=len(raw),excluded_structural_states=len(omitted),effective_valid_states=len(states),declared_states=len(states),normal_not_in_view_states=sum(s['value']=='not_in_view' for s in states),auxiliary_states=sum(s['coverage_group']=='auxiliary' for s in states)))
    write_csv(out/'dictionary_counts.csv',entries)
    write_csv(out/'excluded_dictionary_states.csv',excluded)
    protocol=dict(version='qe_full_fine_selection_v4',candidate_targets_required=7800,per_class_required=1300,
        prerequisite='all7800 fine feature JSONs and score records structurally validated before any selection; incomplete annotations block selection',
        dictionary='all semantically admissible category-specific enum/set states plus4 geometric-scale states; unobserved admissible states retained; denominator varies by class',
        structural_exclusions='ten v4-forbidden noncar torso/limb not_in_view states removed before observing pool; raw schema931, effective921; explicit reasons exported',
        unknown='unknown/not_observable do not count as coverage; not_visible and normal not_in_view remain valid states',
        nominal='hair,color,texture,appearance and context contribute categorical coverage only; no arbitrary ordinal risk',
        overlap='nested head/face/mouth states may all be covered by one image; no independence assumed or joint-coverage claim',
        groups='five dimensions plus auxiliary/appearance/context separate; nominal attributes are not forced into five risk dimensions',
        support_cap='explicit CLI threshold, recommend old20 only as borrowed convention',
        aggregation='must explicitly choose class_macro or pooled_micro; report both regardless',
        risk_definition='use selected frozen fine score version, midpoint ordering with unknown interval retained; no classifier feedback',
        layers='for each class sort all1300 totalmidpoint then sample_id, equal260 in Q1lowest to Q5highest; pool layer sizes20percent do not determine selected proportions',
        selected_proportions='five user-approved ratios required explicitly; same layer quota per class; rounding largest remainder, ties lower-tier',
        methods=dict(coverage='within class budget unrestricted-risk concave coverage greedy',risk='within user-approved class-layer quotas random without replacement',risk_coverage='within same class-layer quotas concave coverage greedy'),
        no_hidden_jointquota='only Q-layer quota, no added midpoint/unknown-cell quota unless separately requested',
        identity='if later coverage is forced to same risk quota and uses same objective/tie, it equals combo and becomes two independent strategies',
        stopping='fixed user budget, no claim of sufficient coverage; cannot-fillquota blocks output, remaining state gaps reported',
        predictions_read_for_selection=False,approved_mix_ratios=MIXTURES,
        approved_reporting_aggregation='class_macro',planned_budgets_per_class=list(DEFAULT_BUDGETS),
        primary_budget_per_class=200,primary_budget_origin='1200total is a context-based default, not a user-specified count; may be revised',
        progressive_class_balance='six classes increase together; batches20 per class; risk random within-tier fixed permutation prefixes; combo budgets independently solved',
        challenge_feasibility='Q5 has260 candidates;30percent quota prevents budget above866, so planned sweep stops800',
        coverage_reference='same one unrestricted-risk coverage result per budget for both mixtures, not two independent coverage algorithms',
        primary_diagnostics='combo vs same-tier-quota random, equal-error confusion breadth,8architectures averaged3seeds; all14 mixture-budget tests Holm; unconstrained coverage comparison descriptive')
    dump(out/'design_protocol.json',protocol);dump(out/'registration_input_freeze.sha256.json',freeze)
    return entries

def parse_ratios(text):
    ratios=[Fraction(x.strip()) for x in text.split(',')]
    if len(ratios)!=5 or any(x<0 for x in ratios) or sum(ratios)!=1:raise ValueError('Five selected-layer ratios must be nonnegative and sum exactly to1 (e.g. user-approved .1,.2,.3,.2,.2).')
    return ratios

def quotas(ratios,budget):
    want=[r*budget for r in ratios];result=[int(x) for x in want]
    for i in sorted(range(5),key=lambda i:(-(want[i]-result[i]),i))[:budget-sum(result)]:result[i]+=1
    if any(x>260 for x in result):raise ValueError(f'Quota exceeds260 available targets in a class risk layer: {result}')
    assert sum(result)==budget
    return result

def matrix(rows,annotations,states):
    m=np.zeros((len(rows),len(states)),dtype=np.uint8)
    index={(s['field'],s['value']):j for j,s in enumerate(states)}
    for i,row in enumerate(rows):
        area=float(row['bbox_area_ratio']);scale='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large'
        m[i,index['target_scale',scale]]=1
        for field,value in annotations[row['sample_id']]['features'].items():
            for v in value if isinstance(value,list) else [value]:
                if v not in UNKNOWN:
                    if (field,v) not in index:raise ValueError(f"Structurally inadmissible annotation state: {row['sample_id']} {field}={v}")
                    m[i,index[field,v]]=1
    return m

def greedy(m,ids,budget,cap,tier=None,quota=None):
    counts=np.zeros(m.shape[1],dtype=int);chosen=[];available=np.ones(len(m),dtype=bool);used=Counter();trace=[]
    for step in range(budget):
        feasible=np.flatnonzero(available)
        if tier is not None:feasible=np.array([i for i in feasible if used[int(tier[i])]<quota[int(tier[i])-1]])
        if not len(feasible):raise ValueError('No feasible candidate under requested quota')
        weights=np.where(counts<cap,np.log(counts+2)-np.log(counts+1),0)
        gains=m[feasible]@weights;maximum=float(gains.max())
        ties=feasible[np.isclose(gains,maximum,atol=1e-12,rtol=0)];i=int(min(ties,key=lambda i:ids[i]))
        before=float(np.minimum(counts/cap,1).mean());chosen.append(i);available[i]=False;counts+=m[i]
        if tier is not None:used[int(tier[i])]+=1
        trace.append(dict(rank=step+1,sample_id=ids[i],concave_gain=maximum,frequency_coverage_gain=float(np.minimum(counts/cap,1).mean())-before,presence_coverage=float((counts>0).mean()),support_completion=float(np.minimum(counts/cap,1).mean()),zero_gain=maximum<=1e-12))
    return np.array(chosen,dtype=np.int16),trace

def select(args):
    root=args.root.resolve();out=args.output.resolve()
    if args.per_class_budget is None or args.layer_ratios is None or args.coverage_aggregation is None:raise ValueError('Selection requires explicit --per-class-budget, --layer-ratios and --coverage-aggregation; no sampled20percent default.')
    budget=args.per_class_budget
    if not 1<=budget<=1300:raise ValueError('Budget must be between1 and1300 per class')
    if args.support_cap<1 or args.random_repeats<1:raise ValueError('Positive support threshold and repetition count required')
    ratios=parse_ratios(args.layer_ratios);wanted=quotas(ratios,budget)
    rows=read_csv(root/'manifest.csv');assert len(rows)==7800 and len({r['sample_id'] for r in rows})==7800
    assert Counter(r['class_name'] for r in rows)==Counter(dict.fromkeys(CLASSES,1300))
    # Every annotation and every score must be present, not merely a successful smoke batch.
    expected={r['sample_id'] for r in rows}
    actual={p.stem for p in (root/'annotations').glob('*.json')};scorefiles={p.stem for p in (root/'scores').glob('*.json')}
    if not expected<=actual or not expected<=scorefiles:raise ValueError(f'Full fine annotation prerequisite incomplete: validatable annotation files {len(expected&actual)}/7800; score files {len(expected&scorefiles)}/7800')
    prepare(root,out)
    freeze={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(root/'manifest.csv'):sha(root/'manifest.csv'),str(root/'scoring_protocol.json'):sha(root/'scoring_protocol.json')}
    freeze.update({str(path.resolve()):sha(path) for path in code_dependencies()})
    for name in ('prompt.txt','runtime.json','protocol.json','model_provenance.json'):
        p=root/name
        if p.exists():freeze[str(p)]=sha(p)
    annotations={};scores={};schemas={c:json.loads((root/'schemas'/f'{c}.json').read_text()) for c in CLASSES};score_protocol=json.loads((root/'scoring_protocol.json').read_text())
    for c in CLASSES:freeze[str(root/'schemas'/f'{c}.json')]=sha(root/'schemas'/f'{c}.json')
    for row in rows:
        sid=row['sample_id'];ap=root/'annotations'/f'{sid}.json';sp=root/'scores'/f'{sid}.json'
        annotations[sid]=fresh.validate_annotation(json.loads(ap.read_text()),row,schemas[row['class_name']]);score=json.loads(sp.read_text());assert score['sample_id']==sid and score['class_name']==row['class_name']
        recomputed=fresh.fresh_score(row,annotations[sid],score_protocol)
        for key,value in recomputed.items():
            if isinstance(value,bool):assert score[key] is value
            else:assert np.isclose(score[key],value,atol=1e-10,rtol=0),(sid,key)
        for bound in ('lower','midpoint','upper'):
            assert np.isclose(score['score_'+bound],20*sum(score[d+'_'+bound] for d in DIMENSIONS),atol=1e-8)
        assert 0<=score['score_lower']<=score['score_midpoint']<=score['score_upper']<=100+1e-8
        if 'schema_sha256' in annotations[sid]:assert annotations[sid]['schema_sha256']==sha(root/'schemas'/f"{row['class_name']}.json")
        if 'prompt_sha256' in annotations[sid]:assert annotations[sid]['prompt_sha256']==sha(root/'prompt.txt')
        scores[sid]=score;freeze[str(ap)]=sha(ap);freeze[str(sp)]=sha(sp)
    strata=[];tier_profiles=[];summaries=[];state_stats=[];layer_stats=[];dimension_stats=[];group_stats=[];traces=[];selected={method:[] for method in ('coverage','risk','risk_coverage')}
    (out/'selection').mkdir(exist_ok=True);(out/'test_sets').mkdir(exist_ok=True)
    for ci,c in enumerate(CLASSES):
        subset=[r for r in rows if r['class_name']==c];ids=[r['sample_id'] for r in subset]
        score=np.array([scores[s]['score_midpoint'] for s in ids]);low=np.array([scores[s]['score_lower'] for s in ids]);high=np.array([scores[s]['score_upper'] for s in ids]);evaluable=np.array([scores[s]['evaluable'] for s in ids]);known=np.array([scores[s]['fully_known'] for s in ids])
        tier=np.zeros(1300,dtype=np.int8);ordered=sorted(range(1300),key=lambda i:(score[i],ids[i]))
        for t,portion in enumerate(np.array_split(ordered,5),1):tier[portion]=t
        for t in range(1,6):
            member=tier==t;next_min=float(score[tier==t+1].min()) if t<5 else None
            tier_profiles.append(dict(class_name=c,risk_tier=t,targets=int(member.sum()),score_midpoint_min=float(score[member].min()),score_midpoint_max=float(score[member].max()),score_midpoint_mean=float(score[member].mean()),score_lower_mean=float(low[member].mean()),score_upper_mean=float(high[member].mean()),fully_known_fraction=float(known[member].mean()),evaluable_fraction=float(evaluable[member].mean()),tied_with_next=next_min is not None and np.isclose(score[member].max(),next_min)))
        states=dictionary(schemas[c]);m=matrix(subset,annotations,states)
        for i,sid in enumerate(ids):strata.append(dict(sample_id=sid,class_name=c,risk_tier=int(tier[i]),score_lower=low[i],score_midpoint=score[i],score_upper=high[i],fully_known=bool(known[i]),evaluable=bool(evaluable[i]),selected_layer_target_quota=wanted[int(tier[i])-1]))
        if args.shared_coverage:
            with np.load(args.shared_coverage/'selection'/f'{c}.npz') as prior:
                assert prior['sample_ids'].tolist()==ids
                coverage=prior['coverage'][0];assert len(coverage)==budget
            trace=[]
        else:coverage,trace=greedy(m,ids,budget,args.support_cap)
        combo,ctrace=greedy(m,ids,budget,args.support_cap,tier,wanted)
        traces.extend(dict(class_name=c,method='coverage',**r) for r in trace);traces.extend(dict(class_name=c,method='risk_coverage',**r) for r in ctrace)
        random=np.empty((args.random_repeats,budget),dtype=np.int16);rng=np.random.default_rng(args.seed+ci)
        for rep in range(args.random_repeats):random[rep]=np.concatenate([rng.permutation(np.flatnonzero(tier==t))[:wanted[t-1]] for t in range(1,6)])
        choices={'coverage':coverage[None],'risk':random,'risk_coverage':combo[None]}
        for method in ('risk','risk_coverage'):
            for t in range(1,6):assert np.all((tier[choices[method]]==t).sum(1)==wanted[t-1])
        np.savez_compressed(out/'selection'/f'{c}.npz',sample_ids=np.array(ids),tiers=tier,matrix=m,**choices)
        for method,selection in choices.items():
            selected[method].append(selection);support=support_counts(m,selection);presence=(support>0).mean(1);completion=np.minimum(support/args.support_cap,1).mean(1)
            summaries.append(dict(class_name=c,method=method,per_class_budget=budget,repetitions=len(selection),dictionary_states=len(states),presence_coverage_mean=float(presence.mean()),presence_selection_p025=float(np.quantile(presence,.025)),presence_selection_p975=float(np.quantile(presence,.975)),support_completion_mean=float(completion.mean()),support_selection_p025=float(np.quantile(completion,.025)),support_selection_p975=float(np.quantile(completion,.975)),risk_lower_mean=float(low[selection].mean()),risk_midpoint_mean=float(score[selection].mean()),risk_upper_mean=float(high[selection].mean()),unknown_fraction_mean=float((~known[selection]).mean()),evaluable_fraction_mean=float(evaluable[selection].mean()),Q5_fraction_mean=float((tier[selection]==5).mean())))
            for t in range(1,6):
                count=(tier[selection]==t).sum(1);layer_stats.append(dict(class_name=c,method=method,risk_tier=t,count_mean=float(count.mean()),count_p025=float(np.quantile(count,.025)),count_p975=float(np.quantile(count,.975)),fraction_mean=float(count.mean()/budget),requested_fraction=float(ratios[t-1]) if method!='coverage' else '',quota=wanted[t-1] if method!='coverage' else 'unrestricted'))
            for j,state in enumerate(states):state_stats.append(dict(class_name=c,method=method,**state,pool_support=int(m[:,j].sum()),selected_support_mean=float(support[:,j].mean()),selected_support_p025=float(np.quantile(support[:,j],.025)),selected_support_p975=float(np.quantile(support[:,j],.975)),presence_probability=float((support[:,j]>0).mean()),zero_support_probability=float((support[:,j]==0).mean()),support_completion_mean=float(np.minimum(support[:,j]/args.support_cap,1).mean())))
            for d in DIMENSIONS:
                components={b:np.array([20*scores[s][d+'_'+b] for s in ids]) for b in ('lower','midpoint','upper')}
                unknown=np.array([scores[s][d+'_known_weight']<1-1e-9 for s in ids])
                dimension_stats.append(dict(class_name=c,method=method,dimension=d,**{b+'_points_mean':float(v[selection].mean()) for b,v in components.items()},unknown_fraction_mean=float(unknown[selection].mean())))
            for group in ('scale','visibility','occlusion','truncation','orientation','auxiliary'):
                columns=[i for i,s in enumerate(states) if s['coverage_group']==group]
                if columns:
                    group_stats.append(dict(class_name=c,method=method,coverage_group=group,declared_states=len(columns),presence_coverage_mean=float((support[:,columns]>0).mean()),support_completion_mean=float(np.minimum(support[:,columns]/args.support_cap,1).mean())))
    totals=[];tier_lookup={row['sample_id']:row['risk_tier'] for row in strata}
    for method in ('coverage','risk','risk_coverage'):
        section=[r for r in summaries if r['method']==method];den=sum(r['dictionary_states'] for r in section)
        totals.append(dict(method=method,total_targets=budget*6,coverage_aggregation=args.coverage_aggregation,class_macro_presence_coverage=float(np.mean([r['presence_coverage_mean'] for r in section])),pooled_micro_presence_coverage=sum(r['presence_coverage_mean']*r['dictionary_states'] for r in section)/den,class_macro_support_completion=float(np.mean([r['support_completion_mean'] for r in section])),pooled_micro_support_completion=sum(r['support_completion_mean']*r['dictionary_states'] for r in section)/den,risk_midpoint_mean=float(np.mean([r['risk_midpoint_mean'] for r in section])),risk_lower_mean=float(np.mean([r['risk_lower_mean'] for r in section])),risk_upper_mean=float(np.mean([r['risk_upper_mean'] for r in section])),unknown_fraction=float(np.mean([r['unknown_fraction_mean'] for r in section])),evaluable_fraction=float(np.mean([r['evaluable_fraction_mean'] for r in section])),Q5_fraction=float(np.mean([r['Q5_fraction_mean'] for r in section]))))
        primary=[]
        for c,selection in zip(CLASSES,selected[method]):
            subset=[r for r in rows if r['class_name']==c]
            for i in selection[0]:
                row=dict(subset[int(i)]);sid=row['sample_id'];row.update(scores[sid]);row['risk_tier']=tier_lookup[sid];row['selection_method']=method;row['selection_repetition']=0
                row['fine_annotation_path']=str(root/'annotations'/f'{sid}.json');row['fine_score_path']=str(root/'scores'/f'{sid}.json');row['fine_schema_path']=str(root/'schemas'/f'{c}.json')
                for field,value in annotations[sid]['features'].items():row['fine_feature_'+field]=json.dumps(value,ensure_ascii=False) if isinstance(value,list) else value
                primary.append(row)
        write_csv(out/'test_sets'/f'{method}_rep0.csv',primary)
    for name,values in [('target_fine_risk_tiers.csv',strata),('risk_tier_profiles.csv',tier_profiles),('class_method_summary.csv',summaries),('method_summary.csv',totals),('selected_feature_state_support.csv',state_stats),('selected_risk_layers.csv',layer_stats),('selected_dimension_scores.csv',dimension_stats),('selected_coverage_groups.csv',group_stats),('greedy_trace.csv',traces)]:write_csv(out/name,values)
    chosen=dict(per_class_budget=budget,total_budget=budget*6,selected_layer_ratios=[str(r) for r in ratios],selected_layer_counts_per_class=wanted,coverage_aggregation=args.coverage_aggregation,support_cap=args.support_cap,random_repeats=args.random_repeats,seed=args.seed,classification_feedback=False,coverage_risk_quota='unrestricted',risk_and_combo_quota='same user-specified risk layer counts; no extra joint cells',scoring_protocol_sha256=sha(root/'scoring_protocol.json'),shared_coverage_reference=str(args.shared_coverage) if args.shared_coverage else None)
    dump(out/'selection_parameters.json',chosen);dump(out/'selection_input_freeze.sha256.json',freeze)
    if args.shared_coverage:
        for c in CLASSES:
            p=(args.shared_coverage/'selection'/f'{c}.npz').resolve();freeze[str(p)]=sha(p)
        dump(out/'selection_input_freeze.sha256.json',freeze)
    produced=[out/name for name in ('target_fine_risk_tiers.csv','risk_tier_profiles.csv','class_method_summary.csv','method_summary.csv','selected_feature_state_support.csv','selected_risk_layers.csv','selected_dimension_scores.csv','selected_coverage_groups.csv','greedy_trace.csv','selection_parameters.json')]+[out/'selection'/f'{c}.npz' for c in CLASSES]+[out/'test_sets'/f'{m}_rep0.csv' for m in ('coverage','risk','risk_coverage')]
    produced+=[out/'design_protocol.json',out/'dictionary_counts.csv',out/'excluded_dictionary_states.csv']+[out/'coverage_dictionaries'/f'{c}.json' for c in CLASSES]
    dump(out/'selection_result_freeze.sha256.json',{str(p):sha(p) for p in produced})
    dump(out/'selection_audit.json',dict(status='pass',all7800_annotations_and_scores_validated=True,targets=7800,selected_targets_per_method=budget*6,predictions_read=False,coverage_schema_denominators_fixed=True,methods_differ_in_selection_rules=True,unknown_scoring_retained=True))
    resultfreeze={str(p):sha(p) for p in produced};resultfreeze[str(out/'selection_audit.json')]=sha(out/'selection_audit.json');dump(out/'selection_result_freeze.sha256.json',resultfreeze)
    print(json.dumps(dict(status='selected',total_targets_per_set=budget*6,risk_layer_quota_per_class=wanted,summary=totals),ensure_ascii=False))

def verify_prediction_binding(root,predictions_path,out):
    """Recheck96 original logits caches and exact model/seed/target binding."""
    predictions_path=predictions_path.resolve();manifest={r['sample_id']:r for r in read_csv(root/'manifest.csv')}
    with np.load(predictions_path) as z:cache={k:z[k] for k in z.files}
    combinations=list(itertools.product(MODELS,SEEDS))
    assert cache['predictions'].shape==(24,7800) and cache['sample_ids'].shape==(7800,)
    assert len(set(cache['sample_ids'].tolist()))==7800 and set(cache['sample_ids'].tolist())==set(manifest)
    assert list(zip(cache['models'].tolist(),cache['seeds'].tolist()))==combinations
    index={str(s):i for i,s in enumerate(cache['sample_ids'])};frozen={str(predictions_path):sha(predictions_path),str(root/'manifest.csv'):sha(root/'manifest.csv')}
    for run,(model,seed) in enumerate(combinations):
        seen=set()
        for name,pool in POOLS.items():
            path=pool/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz';source_manifest=pool/'manifests/candidate_pool.csv';rows=read_csv(source_manifest)
            with np.load(path) as z:
                ids=z['sample_ids'].tolist();logits=z['logits'];assert ids==[r['sample_id'] for r in rows]
                assert logits.shape==(len(ids),6) and np.isfinite(logits).all();prediction=logits.argmax(1)
            for row,observed in zip(rows,prediction):
                sid=row['sample_id'];assert sid not in seen;seen.add(sid);official=manifest[sid]
                assert official['source_pool']==name and official['class_name']==row['class_name'] and int(official['label'])==int(row['label'])==CLASSES.index(row['class_name'])
                assert official['image_sha256']==row['image_sha256'] and official['crop_sha256']==row['crop_sha256']
                assert json.loads(official['bbox_norm_xyxy'])==json.loads(row['bbox_norm_xyxy'])
                assert cache['predictions'][run,index[sid]]==int(observed)
            frozen[str(path.resolve())]=sha(path);frozen[str(source_manifest.resolve())]=sha(source_manifest)
        assert seen==set(manifest)
        print(json.dumps(dict(stage='verified_classifier_cache',model=model,seed=seed,targets=7800)),flush=True)
    dump(out/'classifier_cache_binding_freeze.sha256.json',frozen)
    dump(out/'classifier_cache_binding_audit.json',dict(status='pass',raw_caches=96,models_and_seed_runs=24,all_target_ids_official_labels_boxes_image_and_crop_hashes_validated=7800,merged_predictions_exactly_match_original_logits=True,new_inference=False,retraining=False))

def evaluate(root,out,predictions_path):
    audit=json.loads((out/'selection_audit.json').read_text());assert audit['all7800_annotations_and_scores_validated'] and audit['predictions_read'] is False
    freeze=json.loads((out/'selection_result_freeze.sha256.json').read_text());assert all(sha(Path(p))==v for p,v in freeze.items())
    manifest={r['sample_id']:r for r in read_csv(root/'manifest.csv')}
    with np.load(predictions_path) as z:cache={k:z[k] for k in z.files}
    assert cache['predictions'].shape==(24,7800) and set(cache['sample_ids'].tolist())==set(manifest)
    assert np.isin(cache['predictions'],range(6)).all() and len(cache['models'])==24 and len(cache['seeds'])==24
    index={str(s):i for i,s in enumerate(cache['sample_ids'])};selected={m:[] for m in ('coverage','risk','risk_coverage')}
    for c in CLASSES:
        with np.load(out/'selection'/f'{c}.npz') as z:
            globalid=np.array([index[str(s)] for s in z['sample_ids']]);assert all(manifest[str(s)]['class_name']==c for s in z['sample_ids'])
            for m in selected:selected[m].append(globalid[z[m]])
    results=[];raw={};wrong_by_method={}
    for method,parts in selected.items():
        ids=np.stack(parts,axis=1);reps,_,budget=ids.shape
        wrongs=[];all_metrics=[]
        for run,(model,seed) in enumerate(zip(cache['models'],cache['seeds'])):
            prediction=cache['predictions'][run,ids];confusion=np.zeros((reps,6,6),dtype=int)
            for c in range(6):
                for label in range(6):confusion[:,c,label]=(prediction[:,c]==label).sum(1)
            correct=np.diagonal(confusion,axis1=1,axis2=2);den=confusion.sum(1)+confusion.sum(2)
            f1=np.divide(2*correct,den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1);accuracy=correct.sum(1)/(6*budget)
            wrong=confusion.copy();wrong[:,np.arange(6),np.arange(6)]=0
            wrongs.append(wrong);all_metrics.append(dict(accuracy=accuracy,macro_f1=f1,errors=6*budget-correct.sum(1),confusion_types=(wrong>0).sum((1,2))))
            results.append(dict(method=method,model=str(model),seed=int(seed),repetitions=reps,total_targets=6*budget,accuracy_mean=float(accuracy.mean()),macro_f1_mean=float(f1.mean()),macro_f1_selection_p025=float(np.quantile(f1,.025)),macro_f1_selection_p975=float(np.quantile(f1,.975)),errors_mean=float((6*budget-correct.sum(1)).mean()),confusion_type_mean=float((wrong>0).sum((1,2)).mean()),**{c+'_recall':float((correct[:,i]/budget).mean()) for i,c in enumerate(CLASSES)}))
        wrong_by_method[method]=np.stack(wrongs);raw['wrong_counts_'+method]=wrong_by_method[method]
        for key in all_metrics[0]:raw[method+'_'+key]=np.stack([m[key] for m in all_metrics])
    from run_qe_three_strategy_full_experiment import rarefaction,exact_signflip
    pairs=[];architecture_tests=[]
    for method in ('coverage','risk_coverage'):
        fixed=wrong_by_method[method][:,0];random=wrong_by_method['risk'];q=np.minimum(fixed.sum(-1)[:,None],random.sum(-1))
        left=rarefaction(np.broadcast_to(fixed[:,None],random.shape),q).mean(-1);right=rarefaction(random,q).mean(-1);delta=left-right
        raw['paired_'+method+'_equal_error_breadth_delta']=delta
        for run,(model,seed) in enumerate(zip(cache['models'],cache['seeds'])):pairs.append(dict(method=method,model=str(model),seed=int(seed),equal_error_breadth_delta=float(delta[run].mean()),evaluable_class_fraction=float((q[run]>0).mean())))
        vals=[float(np.mean([r['equal_error_breadth_delta'] for r in pairs if r['method']==method and r['model']==m])) for m in MODELS]
        architecture_tests.append(dict(method=method,architectures=8,mean_equal_error_breadth_delta=float(np.mean(vals)),positive_architectures=sum(v>0 for v in vals),signflip_two_sided_p=exact_signflip(vals)))
    write_csv(out/'model_run_metrics.csv',results)
    write_csv(out/'paired_model_run_metrics.csv',pairs);write_csv(out/'diagnostic_tests_uncorrected.csv',architecture_tests);np.savez_compressed(out/'model_raw_arrays.npz',**raw)
    dump(out/'evaluation_freeze.sha256.json',{str(predictions_path.resolve()):sha(predictions_path),str(out/'selection_result_freeze.sha256.json'):sha(out/'selection_result_freeze.sha256.json'),**{str(out/name):sha(out/name) for name in ('model_run_metrics.csv','paired_model_run_metrics.csv','diagnostic_tests_uncorrected.csv','model_raw_arrays.npz')}})
    dump(out/'evaluation_audit.json',dict(status='pass',targets_identity_verified=7800,model_runs=24,new_classifier_inference=False,retraining=False,selection_frozen_before_predictions=True))

def experiment(args):
    if args.coverage_aggregation!='class_macro':raise ValueError('User chose class_macro aggregation; provide --coverage-aggregation class_macro')
    requested=args.budgets or list(DEFAULT_BUDGETS);assert all(1<=b<=800 for b in requested)
    base=args.output;allsummary=[];allclasses=[];index=[]
    prepare(args.root,base)
    for budget in requested:
        reference=None
        for name,ratios in MIXTURES.items():
            child=argparse.Namespace(**vars(args));child.output=base/f'n{6*budget}'/name;child.per_class_budget=budget;child.layer_ratios=ratios;child.shared_coverage=reference
            dump(base/'workflow_progress.json',dict(stage='selection',budget_per_class=budget,mixture=name,updated_at=fresh.now(),complete=False))
            select(child)
            if name=='balanced':reference=child.output
            for row in read_csv(child.output/'method_summary.csv'):
                allsummary.append(dict(mixture=name,per_class_budget=budget,coverage_reference='balanced shared same budget',**row))
            allclasses.extend(dict(mixture=name,**row) for row in read_csv(child.output/'class_method_summary.csv'))
            index.append(dict(mixture=name,per_class_budget=budget,total_targets=6*budget,output=str(child.output),coverage_reference=str(reference),coverage_is_shared_not_an_independent_mix_algorithm=name!='balanced',evaluation_completed=False))
    write_csv(base/'all_setting_summary.csv',allsummary);write_csv(base/'all_class_summary.csv',allclasses);write_csv(base/'setting_index.csv',index)
    if args.predictions:
        # Finish and freeze ALL selected targets before loading classifier outputs.
        verify_prediction_binding(args.root,args.predictions,base)
        for row in index:
            dump(base/'workflow_progress.json',dict(stage='evaluation',budget_per_class=row['per_class_budget'],mixture=row['mixture'],updated_at=fresh.now(),complete=False))
            evaluate(args.root,Path(row['output']),args.predictions);row['evaluation_completed']=True
        write_csv(base/'setting_index.csv',index)
    files=[base/'all_setting_summary.csv',base/'all_class_summary.csv',base/'setting_index.csv']
    for r in index:
        child=Path(r['output']);files.extend([child/'selection_parameters.json',child/'selection_audit.json',child/'selection_input_freeze.sha256.json',child/'selection_result_freeze.sha256.json'])
        if r['evaluation_completed']:files.extend([child/'model_run_metrics.csv',child/'evaluation_audit.json',child/'evaluation_freeze.sha256.json'])
    dump(base/'experiment_result_freeze.sha256.json',{str(p):sha(p) for p in files})

def report(root,out):
    """Generate final evidence only after real7800 annotation and evaluation."""
    from summarize_qe_three_strategy_full_experiment import md_table,docx_from_md
    from run_qe_three_strategy_full_experiment import holm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=read_csv(out/'setting_index.csv');summary=read_csv(out/'all_setting_summary.csv');models=[];dimensions=[];layers=[];groups=[];tests=[]
    assert settings and all(row['evaluation_completed']=='True' for row in settings)
    for setting in settings:
        child=Path(setting['output']);params=json.loads((child/'selection_parameters.json').read_text());meta=dict(mixture=setting['mixture'],per_class_budget=int(setting['per_class_budget']),total_targets=int(setting['total_targets']))
        models.extend({**meta,**r} for r in read_csv(child/'model_run_metrics.csv'))
        dimensions.extend({**meta,**r} for r in read_csv(child/'selected_dimension_scores.csv'))
        layers.extend({**meta,**r} for r in read_csv(child/'selected_risk_layers.csv'))
        groups.extend({**meta,**r} for r in read_csv(child/'selected_coverage_groups.csv'))
        test=next(r for r in read_csv(child/'diagnostic_tests_uncorrected.csv') if r['method']=='risk_coverage');tests.append({**meta,**test})
    adjusted=holm([float(row['signflip_two_sided_p']) for row in tests])
    for row,p in zip(tests,adjusted):row['holm_family_size']=len(tests);row['holm_p']=float(p)
    outputfiles=[]
    for name,rows in [('all_model_run_metrics.csv',models),('all_selected_dimension_scores.csv',dimensions),('all_selected_risk_layers.csv',layers),('all_selected_coverage_groups.csv',groups),('primary_diagnostic_tests.csv',tests)]:write_csv(out/name,rows);outputfiles.append(out/name)
    for row in summary:
        selected=[m for m in models if m['mixture']==row['mixture'] and int(m['per_class_budget'])==int(row['per_class_budget']) and m['method']==row['method']];assert len(selected)==24
        for key in ('accuracy_mean','macro_f1_mean','errors_mean','confusion_type_mean'):row[key]=float(np.mean([float(m[key]) for m in selected]))
    write_csv(out/'all_strategy_results.csv',summary);outputfiles.append(out/'all_strategy_results.csv')
    primary=[row for row in summary if int(row['per_class_budget'])==200]
    write_csv(out/'primary_1200_results.csv',primary);outputfiles.append(out/'primary_1200_results.csv')
    dictionary_counts=read_csv(out/'dictionary_counts.csv')
    fine_annotations=[json.loads((root/'annotations'/f"{row['sample_id']}.json").read_text()) for row in read_csv(root/'manifest.csv')]
    actual_features=sum(len(a['features']) for a in fine_annotations)
    lines=['# QE全目标细粒度标注后的覆盖与风险选样实验','',
        f'本轮先完成7800个目标的类别专用细粒度VLM标注，实际保留{actual_features}项逐目标特征及证据，然后依据冻结规则评分和选样。犬猫各35字段、马羊各27字段、人32字段、车26字段；原犬猫23字段保留。覆盖率采用各类完整字典的全部合法状态，加原图目标框尺度4状态。原始schema合计931状态，v4协议预先排除非车躯干与肢体的10个not_in_view状态，有效覆盖字典合计921。它们是规则不允许的状态，不作为需要补样的缺口；原因见excluded_dictionary_states.csv。所有其他未出现的合法状态仍保留在分母。unknown/not_observable不算已覆盖，其余允许的正常视角not_in_view仍是合法状态。六类别覆盖率分别计算后宏平均，不强制各类达到相同覆盖数值。','',
        '风险仍采用Q1最低至Q5最高：每类1300目标按总风险中点及sample_id排序，等分成5个260目标层。池中的每层20%是分层边界，选样比例另行规定。主基线各层20%；挑战对照Q1—Q5分别10%、15%、20%、25%、30%，两种配额在各类别一致。主预算1200总、每类200为沿用讨论的默认值，不是用户另行指定的固定数量。','',
        '每类预算为20、40、100、200、400、600、800，六类同时增加，风险随机在各层内使用固定随机排列前缀且重复1000次。组合贪心针对每个预算独立求选样，不保证不同预算集合嵌套。挑战Q5最多260目标，因此不越过可行配额去别层补数，也不运行7800全池挑战集合。','',
        '覆盖组不受风险层配额限制，每个预算只有一份参照，两种混合图中的覆盖线指同一集合；风险组按各层配额随机，组合组在同配额下增加覆盖。没有额外加入风险分中点/未知维数联合格配额。因此三种方法的规则不同，但组合与随机仅层比例匹配，总风险均值、未知程度和五维贡献不保证相同。','',
        '## 固定覆盖字典','']
    lines+=md_table(['类别','字段数','原schema状态数','协议排除数','有效固定状态数','正常视角状态数','辅助外观场景状态数'],[[r['class_name'],r['fields'],r['raw_schema_states'],r['excluded_structural_states'],r['effective_valid_states'],r['normal_not_in_view_states'],r['auxiliary_states']] for r in dictionary_counts])
    lines+=['','头部、面部、嘴等层级重叠可以同时提供状态证据，不假定统计独立。毛色、头发颜色、质地、场景等用于覆盖和细分描述，不人为赋予风险顺序。评分主协议中，非车可见性仍汇总头部/躯干/肢体三个区域、车汇总八部件；眼嘴等细项不全部直接计分。五维各20分，未知部分保留下界/中点/上界，不能把有细字段误写成全部细项都评分。','',
        '## 主预算1200目标','']
    headers=['混合','方法','宏平均出现覆盖%','支持完成%','风险下/中/上','Q5%','未知%','Macro-F1%','错误数']
    def cells(row):return [row['mixture'],row['method'],f"{100*float(row['class_macro_presence_coverage']):.3f}",f"{100*float(row['class_macro_support_completion']):.3f}",f"{float(row['risk_lower_mean']):.3f}/{float(row['risk_midpoint_mean']):.3f}/{float(row['risk_upper_mean']):.3f}",f"{100*float(row['Q5_fraction']):.2f}",f"{100*float(row['unknown_fraction']):.2f}",f"{100*float(row['macro_f1_mean']):.3f}",f"{float(row['errors_mean']):.3f}"]
    lines+=md_table(headers,[cells(r) for r in primary])
    lines+=['','风险随机表值是1000次抽样均值；test_sets/*_rep0.csv只对应第一份实体集合，全部索引在各类NPZ中。低F1反映选入条件下的模型表现，不是训练参数退化；错误数增加不单独证明选样更好。模型分类依据官方真值，24架构种子运行的旧预测经96份原始缓存、官方目标ID、类别、框和图像/裁剪哈希重新绑定核验，未训练或新推理。','',
        '## 同层配额下的诊断比较','']
    lines+=md_table(['混合','目标数','等错误数广度差/百分点','正向架构','Holm校正p'],[[t['mixture'],t['total_targets'],f"{100*float(t['mean_equal_error_breadth_delta']):+.3f}",f"{t['positive_architectures']}/8",f"{float(t['holm_p']):.6f}"] for t in tests])
    lines+=['',f'以8架构为独立单位、每架构先平均3种子，{len(tests)}个组合—随机检验统一Holm校正。双侧精确符号翻转最小p为0.0078125；在当前14项族中低于0.05/14的显著性门槛无法达到。未显著不能解释为无效，报告效应量及方向，不通过事后减少检验取得显著。层配额匹配不控制具体外观、总风险分和数据源，因此不作覆盖的单因素因果解释。','',
        '## 全预算结果与停止','',
        '达到预先设定预算后停止；无法满足配额或未完成7800标注则阻止选样。支持阈值20沿用旧约定，固定后不根据分类结果改变。覆盖缺口、五维贡献与风险层计数分别在all_selected_coverage_groups.csv、all_selected_dimension_scores.csv、all_selected_risk_layers.csv以及每配置selected_feature_state_support.csv中，所有未出现状态可追溯。','']
    lines+=md_table(headers,[cells(r) for r in summary]);text='\n'.join(lines)+'\n'
    out.joinpath('figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for col,mix in enumerate(MIXTURES):
        for ri,(metric,label,scale) in enumerate([('class_macro_presence_coverage','Class-macro full-fine state coverage (%)',100),('class_macro_support_completion','Support-cap20 completion (%)',100),('risk_midpoint_mean','Risk midpoint score',1),('macro_f1_mean','Macro-F1 (%)',100)]):
            ax=axes[ri,col]
            for method,color in [('coverage','#0072B2'),('risk','#777777'),('risk_coverage','#D55E00')]:
                values=sorted([r for r in summary if r['mixture']==mix and r['method']==method],key=lambda r:int(r['per_class_budget']))
                ax.plot([int(r['total_targets']) for r in values],[scale*float(r[metric]) for r in values],marker='o',markersize=3,color=color,label='Shared coverage reference' if method=='coverage' else method)
            ax.set_title(mix if ri==0 else '');ax.set_xlabel('Total targets');ax.set_ylabel(label);ax.grid(alpha=.2)
            if ri==0:ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','svg'):
        p=out/'figures'/f'fine_two_mixtures_three_rules.{ext}';fig.savefig(p,dpi=180);outputfiles.append(p)
    plt.close(fig)
    (out/'report.md').write_text(text);outputfiles.append(out/'report.md');docx_from_md(text,out/'report.docx',[out/'figures/fine_two_mixtures_three_rules.png']);outputfiles.append(out/'report.docx')
    freeze={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(out/'experiment_result_freeze.sha256.json'):sha(out/'experiment_result_freeze.sha256.json'),str(out/'classifier_cache_binding_freeze.sha256.json'):sha(out/'classifier_cache_binding_freeze.sha256.json')}
    freeze.update({str(p):sha(p) for p in outputfiles});dump(out/'final_report_freeze.sha256.json',freeze)
    dump(out/'final_audit.json',dict(status='pass',real_fine_targets=7800,real_fine_field_records=actual_features,mixtures=2,mixture_budget_settings=len(settings),group_name_summary_rows=len(summary),shared_coverage_not_double_independent=True,prediction_cache_binding='classifier_cache_binding_audit.json',new_training=False,new_classifier_inference=False,all_annotated_before_selection=True))
    dump(out/'workflow_progress.json',dict(stage='complete',complete=True,updated_at=fresh.now(),targets_annotated=7800,mixture_budget_settings=len(settings)))
    freeze[str(out/'final_audit.json')]=sha(out/'final_audit.json');dump(out/'final_report_freeze.sha256.json',freeze)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--output',type=Path)
    parser.add_argument('--phase',choices=('prepare','select','experiment','evaluate','report','all'),default='prepare');parser.add_argument('--per-class-budget',type=int);parser.add_argument('--layer-ratios');parser.add_argument('--coverage-aggregation',choices=('class_macro','pooled_micro'));parser.add_argument('--support-cap',type=int,default=20);parser.add_argument('--random-repeats',type=int,default=1000);parser.add_argument('--seed',type=int,default=20261013);parser.add_argument('--shared-coverage',type=Path);parser.add_argument('--budgets',type=int,nargs='+');parser.add_argument('--predictions',type=Path)
    args=parser.parse_args();args.output=args.output or args.root/'selection_analysis'
    if args.phase=='prepare':print(json.dumps(dict(status='design_prepared',counts=prepare(args.root,args.output),selection_requires_explicit_parameters=True),ensure_ascii=False))
    elif args.phase=='select':select(args)
    elif args.phase=='experiment':experiment(args)
    elif args.phase=='all':
        # Exporter enforces7800 score rows and preserves descriptions without inventing scores.
        subprocess.run([sys.executable,str(ROOT/'scripts/export_qe_full_fine_feature_scores.py'),'--input',str(args.root)],check=True)
        args.predictions=args.predictions or ROOT/'artifacts/QE_ThreeStrategy_Full_20261013/merged_predictions.npz'
        experiment(args);report(args.root,args.output)
    elif args.phase=='report':report(args.root,args.output)
    else:
        if not args.predictions:raise ValueError('Evaluation requires explicit --predictions with frozen cache')
        evaluate(args.root,args.output,args.predictions)

if __name__=='__main__':main()
