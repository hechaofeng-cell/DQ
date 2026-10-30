#!/usr/bin/env python3
"""Map genuine fine VLM fields to existing ordinal risk rules without double counting."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import run_fresh_vlm_feature_scoring as fresh

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/QE_FullFine_v5_20261013'

def mapping_for(field,category,protocol):
    old=category in ('dog','cat');maps=protocol['maps']
    if field in ('body_orientation','head_orientation','occlusion_level','truncation_level'):
        name='legacy_'+field if old else field
    elif field=='leg_visibility_count':name='legacy_leg_visibility'
    elif field=='tail_visibility' and old:name='legacy_tail_visibility'
    elif field in ('body_visibility','leg_visibility'):name=field
    elif field.endswith('_visibility') and field not in ('eye_visibility','tongue_visibility','wheel_visibility'):name='part_visibility'
    else:return None,None
    return name,maps[name]

def dimension_for(field):
    if field=='target_scale':return 'scale'
    if 'visibility' in field or field=='visible_vehicle_parts':return 'visibility'
    if field in ('occlusion_level','occluder_type','occluded_parts'):return 'occlusion'
    if field in ('truncation_level','truncation_parts'):return 'truncation'
    if field in ('body_orientation','head_orientation','pose_action','visible_action','tail_posture','ear_posture'):return 'orientation'
    return 'auxiliary'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT);ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    source=args.source.resolve();out=source/('component_preview' if args.allow_partial else 'component_score_exports');out.mkdir(exist_ok=True)
    manifest=fresh.pilot.read_csv(source/'manifest.csv');protocol=json.loads((source/'scoring_protocol.json').read_text());freeze=json.loads((source/'input_freeze.sha256.json').read_text())
    assert len(manifest)==7800
    for p,v in freeze.items():assert fresh.pilot.sha256(p)==v,(p,'input changed')
    missing=[r['sample_id'] for r in manifest if not (source/'annotations'/f"{r['sample_id']}.json").exists()]
    if missing and not args.allow_partial:raise ValueError(f'Full scoring gate: {len(missing)} fine annotations missing; no partial selection permitted')
    scores=[];fields=[];catalog={};inputs={str(Path(__file__).resolve()):fresh.pilot.sha256(__file__),str(source/'input_freeze.sha256.json'):fresh.pilot.sha256(source/'input_freeze.sha256.json')}
    for row in manifest:
        sid=row['sample_id'];c=row['class_name'];path=source/'annotations'/f'{sid}.json'
        if not path.exists():continue
        schema_path=source/'schemas'/f'{c}.json';schema=json.loads(schema_path.read_text());value=json.loads(path.read_text());fresh.validate_annotation(value,row,schema)
        assert value['schema_sha256']==fresh.pilot.sha256(schema_path) and value['prompt_sha256']==fresh.pilot.sha256(source/'prompt.txt')
        expected_version=json.loads((source/'protocol.json').read_text())['prompt_version']
        assert value['annotation_prompt_version']==expected_version
        assert value['input_view_sha256']=={k:freeze[row[k]] for k in ('image_path','marked_image_path','feature_crop_path')}
        if c!='car':assert all(value['features'][k]!='not_in_view' for k in ('torso_visibility','limb_visibility'))
        inputs[str(path)]=fresh.pilot.sha256(path)
        score=fresh.fresh_score(row,value,protocol);score_row=dict(sample_id=sid,class_name=c,source_pool=row['source_pool'],**score)
        for dim in ('scale','visibility','occlusion','truncation','orientation'):
            for bound in ('lower','midpoint','upper'):
                score_row[dim+'_risk_'+bound+'_0_100']=100*score[dim+'_'+bound]
                score_row[dim+'_contribution_'+bound+'_0_20']=20*score[dim+'_'+bound]
        for bound in ('lower','midpoint','upper'):assert np.isclose(sum(score_row[d+'_contribution_'+bound+'_0_20'] for d in ('scale','visibility','occlusion','truncation','orientation')),score['score_'+bound],atol=1e-12,rtol=0)
        scores.append(score_row)
        visible_regions=protocol['visibility_regions']['car' if c=='car' else 'noncar']
        main={'target_scale','occlusion_level','truncation_level','body_orientation',*visible_regions}
        if c!='car':main.add('head_orientation')
        observations={'target_scale':(3 if float(row['bbox_area_ratio'])<protocol['scale_edges'][0] else 2 if float(row['bbox_area_ratio'])<protocol['scale_edges'][1] else 1 if float(row['bbox_area_ratio'])<protocol['scale_edges'][2] else 0),**value['features']}
        for field,state in observations.items():
            name,mapping=mapping_for(field,c,protocol)
            if field=='target_scale':name='original_bbox_scale';mapping={0:0,1:1/3,2:2/3,3:1}
            is_na=state=='not_in_view';is_ordinal=mapping is not None
            known=is_ordinal and state in mapping
            if is_ordinal and not known and not is_na:assert state in protocol['unknown_values'],(c,field,state,'no validated mapping')
            component=dict(sample_id=sid,class_name=c,source_pool=row['source_pool'],field=field,dimension=dimension_for(field),
                value=json.dumps(state,ensure_ascii=False) if isinstance(state,list) else state,
                evidence=f"Original bbox area ratio {row['bbox_area_ratio']}; edges {protocol['scale_edges']}" if field=='target_scale' else value['evidence'][field],
                ordinal_mapping=name or '',participates_main_aggregate=field in main,
                applicability='not_applicable_normal_view' if is_na else 'unknown' if is_ordinal and not known else 'known' if known else 'descriptive_only',
                risk_lower_0_100=(mapping[state]*100 if known else 0) if is_ordinal and not is_na else '',
                risk_midpoint_0_100=(mapping[state]*100 if known else 50) if is_ordinal and not is_na else '',
                risk_upper_0_100=(mapping[state]*100 if known else 100) if is_ordinal and not is_na else '',
                scoring_role='main_component' if field in main else 'ordinal_description_not_added_to_main' if is_ordinal else 'nominal_or_unmapped_description_not_scored',
                scoring_note='Unknown midpoint is interval midpoint, not calibrated risk. Visibility main uses applicable regions equally; no double-counting facial/hair detail.' if is_ordinal else 'No frozen validated risk ordering; numeric risk intentionally blank.')
            fields.append(component)
            catalog[c,field]={k:component[k] for k in ('dimension','ordinal_mapping','participates_main_aggregate','scoring_role')}
    fresh.pilot.write_csv(out/'fine_component_scores.csv',fields);fresh.pilot.write_csv(out/'fine_target_scores.csv',scores)
    fresh.pilot.write_csv(out/'fine_feature_mapping_catalog.csv',[dict(class_name=c,field=f,**r) for (c,f),r in catalog.items()])
    fresh.atomic_json(out/'audit.json',dict(status='pass' if not missing else 'partial_preview',expected=7800,validated_targets=len(scores),missing=len(missing),component_records=len(fields),by_class=dict(Counter(r['class_name'] for r in scores)),main_score_recomputed=True,main_aggregate_not_expanded=True,nominal_fields_not_given_invented_scores=True,predictions_read=False))
    fresh.atomic_json(out/'input_freeze.sha256.json',inputs)
    own=[out/'fine_component_scores.csv',out/'fine_target_scores.csv',out/'fine_feature_mapping_catalog.csv',out/'audit.json',out/'input_freeze.sha256.json']
    fresh.atomic_json(out/'result_freeze.sha256.json',{str(p):fresh.pilot.sha256(p) for p in own})
    print(json.dumps(dict(status='pass' if not missing else 'partial_preview',valid=len(scores),fields=len(fields),output=str(out)),ensure_ascii=False))

if __name__=='__main__':main()
