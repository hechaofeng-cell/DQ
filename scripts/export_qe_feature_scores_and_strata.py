#!/usr/bin/env python3
"""Export frozen dual annotations, risk components, state support and Q strata."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import numpy as np
from build_qe_visualstate_coverrisk_dataset import read_csv, write_csv, write_json, sha, scale
from run_category_five_dimension_pilot import score_sample, scale_risk

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/QE_VisualState_CoverRisk_v1_20261013'
FULL=ROOT/'artifacts/QE_ThreeStrategy_Full_20261013'
OUT=ROOT/'artifacts/QE_FeatureRisk_Diagnostics_20261013'
CLASSES=('dog','cat','horse','sheep','person','car')
DIMENSIONS=('scale','visibility','occlusion','truncation','orientation')
MAIN_FIELDS=dict(scale='target_scale',visibility='body_visibility',occlusion='occlusion_level',truncation='truncation_level',orientation='body_orientation')
FIELD_DIMENSION=dict(target_scale='scale',body_visibility='visibility',head_visibility='visibility',leg_visibility='visibility',wheel_visibility='visibility',visible_vehicle_parts='visibility',occlusion_level='occlusion',occluder_type='occlusion',truncation_level='truncation',truncation_parts='truncation',body_orientation='orientation',head_orientation='orientation',pose_action='orientation')
FEATURE_MAP=dict(body_visibility='body_visibility',head_visibility='part_visibility',leg_visibility='leg_visibility',body_orientation='body_orientation',head_orientation='head_orientation',occlusion_level='occlusion_level',truncation_level='truncation_level')

def summary(values):
    values=np.asarray(values,dtype=float)
    return dict(min=float(values.min()),max=float(values.max()),mean=float(values.mean()),distinct_scores=int(len(np.unique(values))))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((DATA/'risk_protocol.json').read_text())
    dictionary=json.loads((DATA/'coverage_dictionary.json').read_text())
    states=[tuple(x) for x in dictionary['states']]
    assert len(states)==71
    fixed=set(states)
    manifest=read_csv(DATA/'manifest.csv');assert len(manifest)==7800
    tier_rows=read_csv(FULL/'selection/tiers.csv')
    tiers={(r['source'],r['sample_id']):r for r in tier_rows}
    assert len(tiers)==15600
    freeze={str(p):sha(p) for p in [Path(__file__),DATA/'manifest.csv',DATA/'risk_protocol.json',DATA/'coverage_dictionary.json',FULL/'selection/tiers.csv']}
    wide=[];long=[];state_counter=Counter();field_presence=Counter();score_groups=defaultdict(list)
    field_catalog={};parts_changes=defaultdict(list)
    for index,row in enumerate(manifest):
        sid=row['sample_id'];category=row['class_name'];area=float(row['bbox_area_ratio'])
        path=DATA/row['annotation_path'];assert path.is_file()
        packed=json.loads(path.read_text());freeze[str(path)]=sha(path)
        assert packed['sample_id']==sid and packed['class_name']==category and int(packed['official_label'])==int(row['label'])
        for backend in ('local','online'):
            annotation=packed[backend]['native'];features=annotation['features'];evidence=annotation.get('evidence',{})
            risk=score_sample(area,features,category,'equal_global',config)
            stored=packed[backend]['risk']
            for key,value in risk.items():
                if isinstance(value,(int,float,bool)):assert np.isclose(float(value),float(stored[key]),atol=1e-12,rtol=0),(sid,backend,key)
            tier=tiers[backend,sid]
            for key in ('lower','midpoint','upper'):assert np.isclose(risk['score_'+key],float(tier['risk_'+key]),atol=1e-12,rtol=0)
            parts=score_sample(area,features,category,'equal_parts',config)
            values={'target_scale':scale(area),**features}
            target=dict(sample_id=sid,class_name=category,official_label=int(row['label']),source_pool=row['source_pool'],annotation_backend=backend,
                bbox_norm_xyxy=row['bbox_norm_xyxy'],bbox_area_ratio=area,native_schema=annotation['schema_version'],native_field_count=len(features),
                risk_tier='Q'+str(tier['tier']),risk_tier_integer=int(tier['tier']),scoring_scheme='equal_global',
                global_risk_lower=risk['score_lower'],global_risk_midpoint=risk['score_midpoint'],global_risk_upper=risk['score_upper'],
                fully_known=risk['fully_known'],evaluable=risk['evaluable'],unknown_dimensions=risk['unknown_components'],
                sensitivity_equal_parts_lower=parts['score_lower'],sensitivity_equal_parts_midpoint=parts['score_midpoint'],sensitivity_equal_parts_upper=parts['score_upper'],
                sensitivity_equal_parts_fully_known=parts['fully_known'],sensitivity_equal_parts_evaluable=parts['evaluable'],
                sensitivity_parts_minus_global_midpoint=parts['score_midpoint']-risk['score_midpoint'])
            for dim in DIMENSIONS:
                target[dim+'_label']=values[MAIN_FIELDS[dim]]
                target[dim+'_known_weight']=risk[dim+'_known_weight']
                for bound in ('lower','midpoint','upper'):
                    target[dim+'_risk_'+bound+'_0_100']=100*risk[dim+'_'+bound]
                    target[dim+'_contribution_'+bound+'_0_20']=20*risk[dim+'_'+bound]
                for bound in ('lower','midpoint','upper'):
                    target['sensitivity_parts_'+dim+'_risk_'+bound+'_0_100']=100*parts[dim+'_'+bound]
            for bound in ('lower','midpoint','upper'):
                assert np.isclose(sum(target[dim+'_contribution_'+bound+'_0_20'] for dim in DIMENSIONS),target['global_risk_'+bound],atol=1e-12,rtol=0)
            wide.append(target);score_groups[backend,category,int(tier['tier'])].append(target)
            parts_changes[backend,category].append(target)
            for field,value in values.items():
                dim=FIELD_DIMENSION.get(field,'auxiliary')
                primary=field in MAIN_FIELDS.values()
                map_name=FEATURE_MAP.get(field)
                known=False;normalized=None
                if field=='target_scale':normalized=scale_risk(area,config);known=True
                elif map_name:
                    normalized=config['maps'][map_name].get(value);known=normalized is not None
                scoring=field=='target_scale' or map_name is not None
                record=dict(sample_id=sid,class_name=category,source_pool=row['source_pool'],annotation_backend=backend,feature_id=field,
                    feature_value=json.dumps(value,ensure_ascii=False) if isinstance(value,list) else value,
                    evidence=f'Official original-image bounding box area ratio {area:.10f}; frozen edges {config["scale_edges"]}' if field=='target_scale' else evidence.get(field,''),
                    evidence_origin='official_geometry' if field=='target_scale' else 'stored_VLM_annotation',dimension=dim,
                    feature_type='ordinal_risk_proxy' if scoring else 'descriptive_not_scored',
                    participates_equal_global=primary,participates_equal_parts=primary if category=='car' else field in ('target_scale','head_visibility','leg_visibility','occlusion_level','truncation_level','body_orientation','head_orientation'),
                    feature_score_lower_0_100=(100*normalized if known else 0) if scoring else '',
                    feature_score_midpoint_0_100=(100*normalized if known else 50) if scoring else '',
                    feature_score_upper_0_100=(100*normalized if known else 100) if scoring else '',
                    feature_score_known=known if scoring else '',mapping_name='geometric_scale' if field=='target_scale' else map_name or '',
                    scoring_note=('Main five-dimension component; unknown midpoint is not calibrated risk' if primary else 'Ordinal sensitivity only; excluded from main equal_global score') if scoring else 'No frozen ordinal risk mapping; retained as descriptive coverage/conditioning feature',
                    fixed_coverage_field=any(f==field for f,s in states))
                long.append(record)
                field_catalog[category,field]={k:record[k] for k in ('dimension','feature_type','participates_equal_global','participates_equal_parts','mapping_name','scoring_note','fixed_coverage_field')}
                items=value if isinstance(value,list) else [value]
                assert len(items)==len(set(items))
                field_presence[backend,category,field]+=1
                for state in items:state_counter[backend,category,field,state]+=1
        if (index+1)%1300==0:print(f'exported targets {index+1}/7800',flush=True)
    assert len(wide)==15600 and len(long)==309400
    write_csv(OUT/'target_scores.csv',wide);write_csv(OUT/'feature_scores_long.csv',long)
    strata=[];dimensions=[];sensitivity=[]
    for backend in ('local','online'):
        for category in CLASSES:
            ordered=[]
            for tier in range(1,6):
                group=score_groups[backend,category,tier];assert len(group)==260
                vals=[r['global_risk_midpoint'] for r in group]
                ordered.append((min(vals),max(vals)))
                strata.append(dict(annotation_backend=backend,class_name=category,risk_tier='Q'+str(tier),targets=len(group),**summary(vals),
                    lower_mean=float(np.mean([r['global_risk_lower'] for r in group])),upper_mean=float(np.mean([r['global_risk_upper'] for r in group])),
                    unknown_targets=sum(not r['fully_known'] for r in group),unknown_fraction=float(np.mean([not r['fully_known'] for r in group])),
                    fully_known_targets=sum(r['fully_known'] for r in group),evaluable_targets=sum(r['evaluable'] for r in group)))
            for i in range(5):
                entry=strata[-5+i];low,high=ordered[i]
                entry['lower_boundary_tied_previous_tier']=i>0 and low==ordered[i-1][1]
                entry['upper_boundary_tied_next_tier']=i<4 and high==ordered[i+1][0]
                entry['tier_rule']='source/class midpoint ascending then sample_id; 260 per tier; equal scores may straddle tiers'
            group=parts_changes[backend,category]
            for dim in DIMENSIONS:
                field=dim+'_risk_midpoint_0_100'
                dimensions.append(dict(annotation_backend=backend,class_name=category,dimension=dim,targets=len(group),**summary([r[field] for r in group]),
                    unknown_targets=sum(r[dim+'_known_weight']<1 for r in group),
                    weighted_contribution_mean_0_20=float(np.mean([r[dim+'_contribution_midpoint_0_20'] for r in group])),
                    risk_lower_mean_0_100=float(np.mean([r[dim+'_risk_lower_0_100'] for r in group])),risk_upper_mean_0_100=float(np.mean([r[dim+'_risk_upper_0_100'] for r in group]))))
            deltas=[r['sensitivity_parts_minus_global_midpoint'] for r in group]
            sensitivity.append(dict(annotation_backend=backend,class_name=category,targets=len(group),global_mean=float(np.mean([r['global_risk_midpoint'] for r in group])),
                equal_parts_mean=float(np.mean([r['sensitivity_equal_parts_midpoint'] for r in group])),mean_delta=float(np.mean(deltas)),changed_targets=int(np.sum(np.abs(deltas)>1e-12)),
                parts_fully_known_targets=sum(r['sensitivity_equal_parts_fully_known'] for r in group),scope='existing mapped head/leg and head/body sensitivity; not fresh torso/limb/hair annotations; does not alter main selections'))
    state_rows=[]
    for backend in ('local','online'):
        for category in CLASSES:
            tuples=set(states)|{(f,s) for b,c,f,s in state_counter if b==backend and c==category}
            tuples|={(f,'unknown') for f,_ in states}
            for field,state in sorted(tuples):
                count=state_counter[backend,category,field,state]
                in_fixed=(field,state) in fixed
                state_rows.append(dict(annotation_backend=backend,class_name=category,feature_id=field,state=state,dimension=FIELD_DIMENSION.get(field,'auxiliary'),
                    count=count,target_count=1300,proportion=count/1300,feature_present_targets=field_presence[backend,category,field],
                    in_fixed_coverage_denominator=in_fixed,unknown_excluded_from_coverage=state in ('unknown','not_observable'),
                    support_requirement=20 if in_fixed else '',support_completion=min(count/20,1) if in_fixed else '',
                    missing=in_fixed and count==0,shortfall=max(0,20-count) if in_fixed else ''))
    write_csv(OUT/'risk_strata_summary.csv',strata);write_csv(OUT/'dimension_score_summary.csv',dimensions)
    write_csv(OUT/'feature_state_counts.csv',state_rows);write_csv(OUT/'equal_parts_sensitivity_summary.csv',sensitivity)
    write_csv(OUT/'feature_dimension_catalog.csv',[dict(class_name=c,feature_id=f,**v) for (c,f),v in field_catalog.items()])
    protocol=dict(dataset=str(DATA),scope='all7800 merged candidates; all dual stored native labels',new_VLM_inference=False,main_scheme='equal_global',
        score_units={'feature_ordinal':'0-100','dimension':'0-100','weighted_dimension':'0-20','global':'0-100'},
        ordinal_mappings_source=str(DATA/'risk_protocol.json'),unknown_rule='interval [0,100] and midpoint50 for unobserved ordinal component; not expected true risk',
        risk_tiers_source=str(FULL/'selection/tiers.csv'),risk_tiers_rule='per annotation source/class score then targetID; exactly260/tier; tied scores may appear in adjacent tiers',
        nominal_policy='descriptive scores blank; pose/appearance/environment have no frozen ordinal mapping; auxiliary fields not forced into five dimensions',
        sensitivity='equal_parts existing head/leg + body/head orientation; car falls back global; never changes main tiers/selections',
        fine_feature_scope='stored native19 noncar/18car plus geometric scale; latest60 richer feature labels remain separate and are not imputed to7800',
        coverage='fixed71 states/class=426/source; missing states remain denominator; unknown separately recorded; support20',
        limitations=['VLM labels not fully manually reviewed','Visual-state score is not classifier failure probability','Equal-rank strata do not guarantee distinct risk thresholds'])
    write_json(OUT/'protocol.json',protocol)
    audit=dict(status='pass',targets=7800,target_source_score_rows=len(wide),feature_score_rows=len(long),strata_rows=len(strata),dimension_summary_rows=len(dimensions),
        fixed_state_rows=sum(r['in_fixed_coverage_denominator'] for r in state_rows),all_stored_main_scores_recomputed=True,all_tiers_match_frozen_experiment=True,
        all_weighted_dimensions_sum_global=True,nominal_fields_have_no_invented_scores=True,
        targets_by_backend={b:sum(r['annotation_backend']==b for r in wide) for b in ('local','online')})
    write_json(OUT/'audit.json',audit);write_json(OUT/'input_freeze.sha256.json',freeze)
    write_json(OUT/'result_freeze.sha256.json',{str(p):sha(p) for p in OUT.glob('*') if p.suffix in ('.csv','.json') and p.name!='result_freeze.sha256.json'})
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
