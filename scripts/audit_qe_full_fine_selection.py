#!/usr/bin/env python3
"""Independent end-to-end full-fine arithmetic/provenance audit.

Requires real complete 7800 labels. No producer selection/metric functions are
imported. API-label semantics remain checked only in the documented smoke set.
"""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,subprocess,sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
from audit_qe_full_fine_annotations import score as independent_score

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('dog','cat','horse','sheep','person','car')
DIMS=('scale','visibility','occlusion','truncation','orientation')
METHODS=('coverage','risk','risk_coverage')
BUDGETS=(20,40,100,200,400,600,800)
MIXES={'balanced':(Fraction(1,5),)*5,'challenge':tuple(Fraction(x,100) for x in (10,15,20,25,30))}
RAW_STATE_COUNTS=(182,184,137,137,163,128)
STATE_COUNTS=(180,182,135,135,161,128)

def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def stream_csv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def close(a,b,label):assert np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=1e-9,rtol=0),(label,a,b)
def freeze(p):
 values=load(p)
 for name,v in values.items():assert sha(name)==v,('freeze mismatch',name)
 return len(values)
def quota(mix,k):
 values=[r*k for r in MIXES[mix]];q=[int(v) for v in values]
 for i in sorted(range(5),key=lambda j:(-(values[j]-q[j]),j))[:k-sum(q)]:q[i]+=1
 assert sum(q)==k and max(q)<=260
 return np.array(q)
def boolvalue(v):return v is True or v=='True'
def independent_rarefaction(counts,q):
 """Sequential non-hit products independently avoid producer log-gamma."""
 n=counts.sum(-1);absent=np.ones(counts.shape,dtype=float)
 for draw in range(int(q.max(initial=0))):
  active=(q>draw)[...,None];den=(n-draw)[...,None];num=np.maximum(0,den-counts)
  fraction=np.divide(num,den,out=np.ones_like(num,dtype=float),where=den>0)
  absent*=np.where(active,fraction,1)
 return np.sum(1-absent,axis=-1)/5
def support(matrix,choices):
 result=np.empty((len(choices),matrix.shape[1]),dtype=np.int64)
 for start in range(0,len(choices),32):result[start:start+32]=matrix[choices[start:start+32]].sum(1)
 return result

def greedy_audit(m,ids,choices,tiers,q,cap,trace,label):
 counts=np.zeros(m.shape[1],dtype=np.int64);available=np.ones(len(m),dtype=bool);used=np.zeros(5,dtype=int)
 assert len(trace)==len(choices)
 for rank,index in enumerate(choices,1):
  feasible=np.flatnonzero(available if q is None else available & (used[tiers-1]<q[tiers-1]))
  weights=np.zeros(len(counts));active=counts<cap;weights[active]=np.log1p(1/(counts[active]+1))
  gains=m[feasible]@weights;maximum=gains.max();tied=feasible[np.abs(gains-maximum)<=1e-12]
  chosen=min(tied,key=lambda j:ids[j]);assert chosen==index,(label,rank,'maximal feasible gain/tie')
  old=np.minimum(counts,cap).sum()/(cap*len(counts));counts+=m[index];available[index]=False;used[tiers[index]-1]+=1
  row=trace[rank-1];assert int(row['rank'])==rank and row['sample_id']==ids[index]
  close(row['concave_gain'],maximum,(label,rank,'gain'));close(row['presence_coverage'],np.count_nonzero(counts)/len(counts),(label,rank,'presence'))
  completion=np.minimum(counts,cap).sum()/(cap*len(counts));close(row['support_completion'],completion,(label,rank,'support'))
  if 'frequency_coverage_gain' in row:close(row['frequency_coverage_gain'],completion-old,(label,rank,'frequency gain'))
  assert boolvalue(row['zero_gain'])==(maximum<=1e-12)

def ordinal_export(root,scores,annotations,schemas,protocol):
 path=root/'feature_scoring/feature_ordinal_scores.csv';n=0;main_sums={sid:np.zeros(3) for sid in scores};seen=set()
 for r in stream_csv(path):
  n+=1;sid=r['sample_id'];field=r['feature_id'];c=r['class_name'];assert (sid,field) not in seen;seen.add((sid,field));assert c==scores[sid]['class_name']
  fields={d['feature_id']:d for d in schemas[c]['universal_features']+schemas[c].get('category_specific_features',[])}
  value=json.loads(r['value_json']);region=protocol['visibility_regions']['car' if c=='car' else 'noncar'];direction=['body_orientation'] if c=='car' else ['body_orientation','head_orientation'];legacy=c in ('dog','cat');maps=protocol['maps']
  mapped=None;name=''
  if field=='target_scale':
   name='geometric_scale';area=float(scores[sid]['bbox_area_ratio']);expected='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large';assert value==expected;v=sum(area<edge for edge in protocol['scale_edges'])/3;bounds=np.array([v,v,v]);known=True
  else:
   assert value==annotations[sid]['features'][field] and r['evidence']==annotations[sid]['evidence'][field]
   if field in ('body_orientation','head_orientation','occlusion_level','truncation_level'):name=('legacy_' if legacy else '')+field
   elif field in ('body_visibility','leg_visibility'):name=field
   elif field=='leg_visibility_count':name='legacy_leg_visibility'
   elif field=='tail_visibility' and legacy:name='legacy_tail_visibility'
   elif {'clear','partial','not_visible'}<=set(fields[field]['possible_values']):name='part_visibility'
   mapped=maps[name] if name else None
   known=False;bounds=None
   if name and value not in protocol['not_applicable_values']:
    if value in protocol['unknown_values']:bounds=np.array([0,.5,1])
    else:v=mapped[value];bounds=np.array([v,v,v]);known=True
  assert r['ordinal_mapping']==name,(sid,field,'mapping')
  main=field=='target_scale' or field in region or field in direction or field in ('occlusion_level','truncation_level');assert boolvalue(r['participates_main_score'])==main
  applicable=[x for x in region if annotations[sid]['features'][x] not in protocol['not_applicable_values']]
  weight=1 if field in ('target_scale','occlusion_level','truncation_level') else 1/len(direction) if field in direction else 1/len(applicable) if field in applicable else 0
  if main:close(r['within_dimension_weight'],weight,(sid,field,'weight'))
  if not name:
   assert all(r['feature_score_'+b+'_0_100']=='' for b in ('lower','midpoint','upper'));assert r['risk_order_status']=='no_risk_order_defined'
  elif bounds is None:assert all(r['feature_score_'+b+'_0_100']=='' for b in ('lower','midpoint','upper'))
  else:
   close([r['feature_score_'+b+'_0_100'] for b in ('lower','midpoint','upper')],bounds*100,(sid,field,'ordinal score'));assert boolvalue(r['ordinal_known'])==known
  if main and bounds is not None:
   contribution=bounds*20*weight;close([r['main_contribution_'+b+'_0_20'] for b in ('lower','midpoint','upper')],contribution,(sid,field,'contribution'));main_sums[sid]+=contribution
  else:assert all(r['main_contribution_'+b+'_0_20']=='' for b in ('lower','midpoint','upper'))
 assert n==244400
 for sid,s in scores.items():
  c=s['class_name'];region=protocol['visibility_regions']['car' if c=='car' else 'noncar']
  if not any(annotations[sid]['features'][x] not in protocol['not_applicable_values'] for x in region):main_sums[sid]+=np.array([0,10,20])
  close(main_sums[sid],[s['score_'+b] for b in ('lower','midpoint','upper')],(sid,'main sum'))
 freeze(root/'feature_scoring/feature_score_freeze.sha256.json')
 return n

def reconstruct(root,out):
 rows=read(root/'manifest.csv');assert len(rows)==7800 and len({r['sample_id'] for r in rows})==7800;assert Counter(r['class_name'] for r in rows)==Counter(dict.fromkeys(CLASSES,1300))
 assert len(list((root/'annotations').glob('*.json')))==7800,'Require real complete7800, not smoke'
 assert len(list((root/'scores').glob('*.json')))==7800
 schemas={c:load(root/'schemas'/f'{c}.json') for c in CLASSES};protocol=load(root/'scoring_protocol.json');annotations={};scores={}
 frozen_annotation_protocol=load(root/'protocol.json');assert frozen_annotation_protocol['prompt_version'].startswith(('v4_','v5_')),'Effective921 dictionary requires frozen semantic exclusion protocol'
 for row in rows:
  sid=row['sample_id'];a=load(root/'annotations'/f'{sid}.json');s=load(root/'scores'/f'{sid}.json');annotations[sid]=a;computed=independent_score(row,a['features'],protocol)
  assert a['image_id']==sid and s['sample_id']==sid and s['class_name']==row['class_name'] and int(row['label'])==CLASSES.index(row['class_name'])
  if row['class_name']!='car':assert a['features']['torso_visibility']!='not_in_view' and a['features']['limb_visibility']!='not_in_view'
  for key,v in computed.items():close(s[key],v,(sid,key))
  scores[sid]={**row,**s}
 assert sum(len(a['features']) for a in annotations.values())==236600
 arrays={}
 for ci,c in enumerate(CLASSES):
  states=load(out/'coverage_dictionaries'/f'{c}.json')['states'];expected=[('target_scale',v) for v in ('tiny','small','medium','large')]
  for d in schemas[c]['universal_features']+schemas[c].get('category_specific_features',[]):expected.extend((d['feature_id'],v) for v in d['possible_values'] if v not in ('unknown','not_observable') and not (c!='car' and d['feature_id'] in ('torso_visibility','limb_visibility') and v=='not_in_view'))
  assert [(s['field'],s['value']) for s in states]==expected and len(expected)==STATE_COUNTS[ci]
  dictionary_meta=load(out/'coverage_dictionaries'/f'{c}.json');assert dictionary_meta['raw_schema_state_count']==RAW_STATE_COUNTS[ci] and dictionary_meta['effective_valid_state_count']==STATE_COUNTS[ci]
  subset=[r for r in rows if r['class_name']==c];ids=[r['sample_id'] for r in subset];m=np.zeros((1300,len(states)),dtype=np.uint8)
  for i,row in enumerate(subset):
   area=float(row['bbox_area_ratio']);scale='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large';fields={**annotations[row['sample_id']]['features'],'target_scale':scale}
   for j,state in enumerate(states):v=fields[state['field']];m[i,j]=int(state['value'] in v if isinstance(v,list) else state['value']==v)
  order=sorted(range(1300),key=lambda i:(scores[ids[i]]['score_midpoint'],ids[i]));tier=np.zeros(1300,dtype=int)
  for rank,index in enumerate(order):tier[index]=rank//260+1
  arrays[c]=dict(ids=ids,matrix=m,tiers=tier,states=states,scores=[scores[sid] for sid in ids])
 assert sum(STATE_COUNTS)==921 and sum(RAW_STATE_COUNTS)==931
 excluded=read(out/'excluded_dictionary_states.csv');assert len(excluded)==10
 assert {(r['class_name'],r['field'],r['value']) for r in excluded}=={(c,f,'not_in_view') for c in CLASSES[:-1] for f in ('torso_visibility','limb_visibility')}
 assert all(r['reason'] and 'structurally inadmissible' in r['exclusion_basis'] for r in excluded)
 return rows,arrays,scores,annotations,schemas,protocol

def condition_metrics(array,selection,cap):
 count=support(array['matrix'],selection);s=array['scores'];tiers=array['tiers'];result=dict(presence_coverage_mean=(count>0).mean(),presence_selection_p025=np.quantile((count>0).mean(1),.025),presence_selection_p975=np.quantile((count>0).mean(1),.975),support_completion_mean=np.minimum(count,cap).mean()/cap,support_selection_p025=np.quantile(np.minimum(count,cap).mean(1)/cap,.025),support_selection_p975=np.quantile(np.minimum(count,cap).mean(1)/cap,.975),unknown_fraction_mean=np.mean([[not t['fully_known'] for t in s][i] for i in selection.ravel()]),evaluable_fraction_mean=np.array([t['evaluable'] for t in s])[selection].mean(),Q5_fraction_mean=(tiers[selection]==5).mean())
 for b in ('lower','midpoint','upper'):result['risk_'+b+'_mean']=np.array([t['score_'+b] for t in s])[selection].mean()
 return result,count

def audit_setting(child,budget,mix,arrays,scores,annotations,cache):
 params=load(child/'selection_parameters.json');assert params['per_class_budget']==budget and params['total_budget']==6*budget and params['coverage_aggregation']=='class_macro' and params['classification_feedback'] is False
 assert params['coverage_risk_quota']=='unrestricted';q=quota(mix,budget);assert params['selected_layer_counts_per_class']==q.tolist();assert [Fraction(x) for x in params['selected_layer_ratios']]==list(MIXES[mix])
 cap=int(params['support_cap']);assert cap==20 and int(params['random_repeats'])==1000
 trace=read(child/'greedy_trace.csv');class_summary={(r['class_name'],r['method']):r for r in read(child/'class_method_summary.csv')};state_rows={(r['class_name'],r['method'],r['field'],r['value']):r for r in read(child/'selected_feature_state_support.csv')};dim_rows={(r['class_name'],r['method'],r['dimension']):r for r in read(child/'selected_dimension_scores.csv')};layer_rows={(r['class_name'],r['method'],int(r['risk_tier'])):r for r in read(child/'selected_risk_layers.csv')};group_rows={(r['class_name'],r['method'],r['coverage_group']):r for r in read(child/'selected_coverage_groups.csv')}
 assert len(class_summary)==18 and len(state_rows)==2763 and len(dim_rows)==90 and len(layer_rows)==90
 strata=read(child/'target_fine_risk_tiers.csv');assert len(strata)==7800
 strata={r['sample_id']:r for r in strata};choices={method:[] for method in METHODS};results={method:[] for method in METHODS};global_index={str(s):i for i,s in enumerate(cache['sample_ids'])}
 profiles={(r['class_name'],int(r['risk_tier'])):r for r in read(child/'risk_tier_profiles.csv')};assert len(profiles)==30
 for ci,c in enumerate(CLASSES):
  a=arrays[c];ids=a['ids'];m=a['matrix'];tier=a['tiers'];z=dict(np.load(child/'selection'/f'{c}.npz'))
  assert z['sample_ids'].tolist()==ids and np.array_equal(z['matrix'],m) and np.array_equal(z['tiers'],tier)
  for t in range(1,6):
   pr=profiles[c,t];member=tier==t;assert int(pr['targets'])==260
   for b in ('lower','midpoint','upper'):
    values=np.array([s['score_'+b] for s in a['scores']]);close(pr['score_'+b+'_mean'],values[member].mean(),'risk tier profile')
    if b=='midpoint':close(pr['score_midpoint_min'],values[member].min(),'risk tier min');close(pr['score_midpoint_max'],values[member].max(),'risk tier max')
   close(pr['fully_known_fraction'],np.array([s['fully_known'] for s in a['scores']])[member].mean(),'tier known');close(pr['evaluable_fraction'],np.array([s['evaluable'] for s in a['scores']])[member].mean(),'tier evaluable')
  for i,sid in enumerate(ids):
   r=strata[sid];assert r['class_name']==c and int(r['risk_tier'])==tier[i] and int(r['selected_layer_target_quota'])==q[tier[i]-1]
   for b in ('lower','midpoint','upper'):close(r['score_'+b],scores[sid]['score_'+b],(sid,'tier'))
  rng=np.random.default_rng(params['seed']+ci)
  for rep in range(1000):
   replay=np.concatenate([rng.permutation(np.flatnonzero(tier==t))[:q[t-1]] for t in range(1,6)]);assert np.array_equal(z['risk'][rep],replay),(child,c,rep,'random replay')
  for method in METHODS:
   selection=z[method];assert selection.shape==(1000 if method=='risk' else 1,budget);assert np.all((0<=selection)&(selection<1300));assert np.all(np.diff(np.sort(selection,axis=1),axis=1)>0),'without replacement'
   if method!='coverage':assert np.array_equal(np.stack([(tier[selection]==t).sum(1) for t in range(1,6)],1),np.broadcast_to(q,(len(selection),5)))
   if method=='coverage' and mix=='challenge':
    balanced=dict(np.load(child.parent/'balanced/selection'/f'{c}.npz'));assert np.array_equal(selection,balanced['coverage'])
   elif method!='risk':greedy_audit(m,ids,selection[0],tier,q if method=='risk_coverage' else None,cap,[r for r in trace if r['class_name']==c and r['method']==method],(mix,budget,c,method))
   metrics,count=condition_metrics(a,selection,cap);results[method].append(metrics)
   r=class_summary[c,method];assert int(r['dictionary_states'])==len(a['states']) and int(r['repetitions'])==len(selection)
   for key,v in metrics.items():close(r[key],v,(child,c,method,key))
   for j,state in enumerate(a['states']):
    sr=state_rows[c,method,state['field'],state['value']];close(sr['pool_support'],m[:,j].sum(),'pool state');close(sr['selected_support_mean'],count[:,j].mean(),'state support');close(sr['selected_support_p025'],np.quantile(count[:,j],.025),'state quantile');close(sr['selected_support_p975'],np.quantile(count[:,j],.975),'state quantile');close(sr['presence_probability'],(count[:,j]>0).mean(),'state presence');close(sr['zero_support_probability'],(count[:,j]==0).mean(),'state absent');close(sr['support_completion_mean'],np.minimum(count[:,j],cap).mean()/cap,'state completion')
   for d in DIMS:
    dr=dim_rows[c,method,d]
    for b in ('lower','midpoint','upper'):close(dr[b+'_points_mean'],np.array([s[d+'_'+b] for s in a['scores']])[selection].mean()*20,'dimension contribution')
    close(dr['unknown_fraction_mean'],(np.array([s[d+'_known_weight'] for s in a['scores']])[selection]<1-1e-9).mean(),'dimension unknown')
   for t in range(1,6):
    lr=layer_rows[c,method,t];ct=(tier[selection]==t).sum(1)
    for key,v in [('count_mean',ct.mean()),('count_p025',np.quantile(ct,.025)),('count_p975',np.quantile(ct,.975)),('fraction_mean',ct.mean()/budget)]:close(lr[key],v,'layer counts')
    if method!='coverage':assert int(lr['quota'])==q[t-1];close(lr['requested_fraction'],float(MIXES[mix][t-1]),'layer requested fraction')
   for g in ('scale','visibility','occlusion','truncation','orientation','auxiliary'):
    cols=[j for j,s in enumerate(a['states']) if s['coverage_group']==g]
    if cols:
     gr=group_rows[c,method,g];assert int(gr['declared_states'])==len(cols);close(gr['presence_coverage_mean'],(count[:,cols]>0).mean(),'group presence');close(gr['support_completion_mean'],np.minimum(count[:,cols],cap).mean()/cap,'group completion')
   choices[method].append(np.array([global_index[s] for s in ids])[selection])
 summary={r['method']:r for r in read(child/'method_summary.csv')}
 for method in METHODS:
  r=summary[method];parts=results[method]
  for suffix,metric in [('presence_coverage','presence_coverage_mean'),('support_completion','support_completion_mean')]:
   close(r['class_macro_'+suffix],np.mean([s[metric] for s in parts]),'macro denominator');close(r['pooled_micro_'+suffix],np.dot(STATE_COUNTS,[s[metric] for s in parts])/921,'micro denominator')
  for key,metric in [('unknown_fraction','unknown_fraction_mean'),('evaluable_fraction','evaluable_fraction_mean'),('Q5_fraction','Q5_fraction_mean')]+[(f'risk_{b}_mean',f'risk_{b}_mean') for b in ('lower','midpoint','upper')]:close(r[key],np.mean([s[metric] for s in parts]),key)
  manifest=read(child/'test_sets'/f'{method}_rep0.csv');expected_ids=[str(cache['sample_ids'][i]) for cls in choices[method] for i in cls[0]];assert [r['sample_id'] for r in manifest]==expected_ids
  for r in manifest:
   sid=r['sample_id'];assert r['image_path']==scores[sid]['image_path'] and Path(r['image_path']).is_file();assert r['selection_method']==method and int(r['selection_repetition'])==0
   for key,v in scores[sid].items():
    if key in ('score_lower','score_midpoint','score_upper') or any(key.startswith(d+'_') for d in DIMS):close(r[key],v,(sid,'entity score'))
   for field,v in annotations[sid]['features'].items():assert (json.loads(r['fine_feature_'+field]) if isinstance(v,list) else r['fine_feature_'+field])==v
 freeze(child/'selection_input_freeze.sha256.json');freeze(child/'selection_result_freeze.sha256.json');freeze(child/'evaluation_freeze.sha256.json')
 return audit_models(child,choices,budget,cache),summary

def signflip(values):
 values=np.array(values);signs=np.array(list(itertools.product((-1,1),repeat=len(values))));return float(np.mean(np.abs(signs@values/len(values))>=abs(values.mean())-1e-12))
def audit_models(child,choices,k,cache):
 raw=dict(np.load(child/'model_raw_arrays.npz'));rows={(r['method'],r['model'],int(r['seed'])):r for r in read(child/'model_run_metrics.csv')};paired={(r['method'],r['model'],int(r['seed'])):r for r in read(child/'paired_model_run_metrics.csv')};assert len(rows)==72 and len(paired)==48
 wrongs={}
 for method in METHODS:
  selected=np.stack(choices[method],axis=1);rep=len(selected);wrong=np.zeros((24,rep,6,6),dtype=np.int64);metrics={key:np.zeros((24,rep)) for key in ('accuracy','macro_f1','errors','confusion_types')}
  for run in range(24):
   conf=np.zeros((rep,6,6),dtype=np.int64)
   for ci in range(6):
    prediction=cache['predictions'][run,selected[:,ci]]
    for pi in range(6):conf[:,ci,pi]=(prediction==pi).sum(1)
   tp=np.diagonal(conf,axis1=1,axis2=2);den=conf.sum(1)+conf.sum(2);f1=np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1)
   wrong[run]=conf;wrong[run,:,np.arange(6),np.arange(6)]=0
   metrics['accuracy'][run]=tp.sum(1)/(6*k);metrics['macro_f1'][run]=f1;metrics['errors'][run]=6*k-tp.sum(1);metrics['confusion_types'][run]=(wrong[run]>0).sum((1,2))
   row=rows[method,str(cache['models'][run]),int(cache['seeds'][run])]
   for key,column in [('accuracy','accuracy_mean'),('macro_f1','macro_f1_mean'),('errors','errors_mean'),('confusion_types','confusion_type_mean')]:close(row[column],metrics[key][run].mean(),(child,method,run,column))
   close(row['macro_f1_selection_p025'],np.quantile(f1,.025),'model quantile');close(row['macro_f1_selection_p975'],np.quantile(f1,.975),'model quantile')
   for ci,c in enumerate(CLASSES):close(row[c+'_recall'],(tp[:,ci]/k).mean(),'class recall')
  assert np.array_equal(raw['wrong_counts_'+method],wrong)
  for key,v in metrics.items():close(raw[method+'_'+key],v,(child,method,key,'raw'))
  wrongs[method]=wrong
 tests={r['method']:r for r in read(child/'diagnostic_tests_uncorrected.csv')}
 for method in ('coverage','risk_coverage'):
  fixed=wrongs[method][:,0];random=wrongs['risk'];q=np.minimum(fixed.sum(-1)[:,None],random.sum(-1));left=independent_rarefaction(np.broadcast_to(fixed[:,None],random.shape),q).mean(-1);right=independent_rarefaction(random,q).mean(-1);delta=left-right;close(raw['paired_'+method+'_equal_error_breadth_delta'],delta,(child,'exact sequential rarefaction'))
  for run in range(24):
   r=paired[method,str(cache['models'][run]),int(cache['seeds'][run])];close(r['equal_error_breadth_delta'],delta[run].mean(),'paired means');close(r['evaluable_class_fraction'],(q[run]>0).mean(),'paired evaluable')
  vals=[delta[np.array(cache['models'])==model].mean() for model in dict.fromkeys(cache['models'].tolist())];r=tests[method];close(r['mean_equal_error_breadth_delta'],np.mean(vals),'architecture effect');close(r['signflip_two_sided_p'],signflip(vals),'architecture signflip');assert int(r['positive_architectures'])==sum(v>0 for v in vals)
 return tests['risk_coverage']

def final_report_audit(out,setting_summaries,tests):
 index=read(out/'setting_index.csv');assert len(index)==14 and all(r['evaluation_completed']=='True' for r in index)
 allresults=read(out/'all_strategy_results.csv');assert len(allresults)==42
 for row in read(out/'all_setting_summary.csv'):
  source=setting_summaries[row['mixture'],int(row['per_class_budget']),row['method']]
  for key,v in source.items():
   if key not in ('method','coverage_aggregation'):close(row[key],v,('all setting aggregate',key))
 for row in read(out/'all_class_summary.csv'):
  child=out/f'n{int(row["per_class_budget"])*6}'/row['mixture'];source=next(r for r in read(child/'class_method_summary.csv') if r['class_name']==row['class_name'] and r['method']==row['method'])
  for key,v in source.items():assert row[key]==v,('all class aggregate',key)
 for row in allresults:
  key=(row['mixture'],int(row['per_class_budget']),row['method']);source=setting_summaries[key]
  for name in ('class_macro_presence_coverage','pooled_micro_presence_coverage','class_macro_support_completion','pooled_micro_support_completion','risk_lower_mean','risk_midpoint_mean','risk_upper_mean','unknown_fraction','evaluable_fraction','Q5_fraction'):close(row[name],source[name],('final aggregation',key,name))
  child=out/f'n{int(row["per_class_budget"])*6}'/row['mixture'];models=[r for r in read(child/'model_run_metrics.csv') if r['method']==row['method']]
  for name in ('accuracy_mean','macro_f1_mean','errors_mean','confusion_type_mean'):close(row[name],np.mean([float(r[name]) for r in models]),('final modelmeans',key))
 primary=read(out/'primary_1200_results.csv');assert len(primary)==6
 for row in primary:assert row==next(r for r in allresults if (r['mixture'],r['method'],int(r['per_class_budget']))==(row['mixture'],row['method'],200))
 reported=read(out/'primary_diagnostic_tests.csv');assert len(reported)==14;p=np.array([float(tests[r['mixture'],int(r['per_class_budget'])]['signflip_two_sided_p']) for r in reported]);order=np.argsort(p,kind='stable');adjusted=np.zeros(14);acc=0
 for rank,i in enumerate(order):acc=max(acc,(14-rank)*p[i]);adjusted[i]=min(acc,1)
 for i,r in enumerate(reported):
  original=tests[r['mixture'],int(r['per_class_budget'])];close(r['mean_equal_error_breadth_delta'],original['mean_equal_error_breadth_delta'],'final effect');close(r['signflip_two_sided_p'],p[i],'final p');close(r['holm_p'],adjusted[i],'Holm14');assert int(r['holm_family_size'])==14
 expected={'all_setting_summary.csv':42,'all_class_summary.csv':252,'all_model_run_metrics.csv':1008,'all_selected_dimension_scores.csv':1260,'all_selected_risk_layers.csv':1260}
 for name,n in expected.items():assert len(read(out/name))==n,(name,n)
 freeze(out/'experiment_result_freeze.sha256.json');freeze(out/'classifier_cache_binding_freeze.sha256.json');freeze(out/'final_report_freeze.sha256.json')
 text=(out/'report.md').read_text();assert '7800' in text and '236600' in text and '14' in text and '0.0078125' in text and '20%' in text
 assert 'unknown' in text and '不' in text and '不全部直接计分' in text
 from docx import Document
 doc=Document(out/'report.docx');assert len(doc.tables)>=4
 progress=load(out/'workflow_progress.json');assert progress['stage']=='complete' and progress['complete'] is True

def main():
 p=argparse.ArgumentParser();p.add_argument('--input','--inputv2',dest='input',required=True,type=Path);p.add_argument('--skip-annotation-reaudit',action='store_true');args=p.parse_args();root=args.input.resolve();out=root/'selection_analysis';out.mkdir(exist_ok=True)
 try:
  if not args.skip_annotation_reaudit:subprocess.run([sys.executable,str(ROOT/'scripts/audit_qe_full_fine_annotations.py'),'--input',str(root)],check=True)
  ann=load(root/'independent_annotation_audit.json');assert ann['status']=='pass' and ann['valid_annotations']==7800 and ann['evidence_feature_records']==236600
  rows,arrays,scores,annotations,schemas,protocol=reconstruct(root,out);n=ordinal_export(root,scores,annotations,schemas,protocol)
  binding=load(out/'classifier_cache_binding_audit.json');assert binding['status']=='pass' and binding['raw_caches']==96
  frozen=load(out/'classifier_cache_binding_freeze.sha256.json');merged=[Path(name) for name in frozen if Path(name).name=='merged_predictions.npz'];assert len(merged)==1
  cache=dict(np.load(merged[0]));assert cache['predictions'].shape==(24,7800) and set(cache['sample_ids'])==set(scores)
  assert np.isin(cache['predictions'],np.arange(6)).all()
  # Rebind against actual logits independently, using only declared cached paths.
  cache_index={str(s):i for i,s in enumerate(cache['sample_ids'])}
  logit_paths=[Path(name) for name in frozen if Path(name).name=='candidate_outputs.npz'];assert len(logit_paths)==96
  bound_sets=[set() for _ in range(24)]
  for path in logit_paths:
   z=dict(np.load(path));model=path.parent.parent.name;seed=int(path.parent.name);run=next(i for i in range(24) if str(cache['models'][i])==model and int(cache['seeds'][i])==seed);ids=z['sample_ids'].tolist();assert np.array_equal(cache['predictions'][run,[cache_index[str(s)] for s in ids]],z['logits'].argmax(1))
   assert not bound_sets[run].intersection(ids);bound_sets[run].update(ids)
  assert all(s==set(scores) for s in bound_sets)
  summaries={};tests={}
  for k in BUDGETS:
   for mix in MIXES:
    child=out/f'n{k*6}'/mix;test,summary=audit_setting(child,k,mix,arrays,scores,annotations,cache);tests[mix,k]=test
    for method,r in summary.items():summaries[mix,k,method]=r
    print(json.dumps({'stage':'independently_verified','mixture':mix,'per_class_budget':k}),flush=True)
  final_report_audit(out,summaries,tests)
  result=dict(status='pass',targets=7800,real_evidence_feature_records=236600,ordinal_export_rows=n,fixed_states_per_class=dict(zip(CLASSES,STATE_COUNTS)),total_fixed_states=921,raw_schema_states=931,structurally_forbidden_states_excluded=10,mixtures=2,budgets_per_class=list(BUDGETS),settings=14,group_summary_rows=42,random_repetitions_per_setting=1000,all_random_indices_and_quota_counts_replayed=True,all_greedy_steps_maximal_feasible_gain_and_id_tie_verified=True,coverage_unrestricted_shared_between_mixtures=True,risk_combo_same_tier_quota_only=True,all_classifier_metrics_and_exact_rarefaction_recomputed=True,holm_family_size=14,raw_model_caches_rebound=96,nominal_and_complex_eye_have_no_invented_ordinal_score=True,main_descendant_scores_not_double_counted=True,all_report_freezes_verified=True,semantic_all7800_vlm_accuracy_verified=False)
  (out/'independent_selection_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
 except Exception as exc:
  result=dict(status='blocked',full_completion_claim_allowed=False,reason=str(exc),exception=type(exc).__name__,required='real7800 complete labels, unchanged frozen inputs, complete selections/evaluation/report')
  (out/'independent_selection_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');raise

if __name__=='__main__':main()
