#!/usr/bin/env python3
"""Export ordinal fine scores after all targets are annotated, without double counting.

Ordinal values reuse the frozen mappings. Additional fields remain separate
descriptors; they do not silently modify the five-dimensional main score.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np

CLASSES=('dog','cat','horse','sheep','person','car')

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mapping_for(field,definition,category,protocol):
    maps=protocol['maps'];legacy=category in ('dog','cat')
    if field=='target_scale':return 'geometric_scale',None
    if field in ('body_orientation','head_orientation','occlusion_level','truncation_level'):
        name='legacy_'+field if legacy else field
        return name,maps[name]
    if field=='body_visibility':return field,maps[field]
    if field=='leg_visibility':return field,maps[field]
    if field=='leg_visibility_count':return 'legacy_leg_visibility',maps['legacy_leg_visibility']
    if field=='tail_visibility' and legacy:return 'legacy_tail_visibility',maps['legacy_tail_visibility']
    if {'clear','partial','not_visible'}<=set(definition.get('possible_values',[])):
        return 'part_visibility',maps['part_visibility']
    return '',None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
    source=args.input.resolve();out=args.output.resolve() if args.output else source/'feature_scoring';out.mkdir(parents=True,exist_ok=True)
    manifest=read(source/'manifest.csv');assert len(manifest)==7800 and Counter(r['class_name'] for r in manifest)==Counter(dict.fromkeys(CLASSES,1300))
    schemas={c:json.loads((source/'schemas'/f'{c}.json').read_text()) for c in CLASSES}
    protocol=json.loads((source/'scoring_protocol.json').read_text());scores=read(source/'scores.csv');lookup={r['sample_id']:r for r in scores}
    assert len(scores)==len(lookup)==7800,'Require all 7800 existing main score rows; no partial export as complete.'
    annotations={};freeze_paths=[source/'manifest.csv',source/'scores.csv',source/'scoring_protocol.json',Path(__file__).resolve()]+[source/'schemas'/f'{c}.json' for c in CLASSES]
    for row in manifest:
        path=source/'annotations'/f"{row['sample_id']}.json";assert path.is_file(),('Missing annotation',row['sample_id'])
        annotation=json.loads(path.read_text());schema=schemas[row['class_name']];definitions=schema['universal_features']+schema.get('category_specific_features',[])
        assert annotation['image_id']==row['sample_id'] and annotation['schema_version']==schema['schema_version']
        assert set(annotation['features'])==set(annotation['evidence'])=={f['feature_id'] for f in definitions}
        annotations[row['sample_id']]=annotation;freeze_paths.append(path)
    destination=out/'feature_ordinal_scores.csv';field_summary=Counter();states_summary=Counter();ordinal_count=0;unmapped_count=0;napp_count=0
    with destination.open('w',encoding='utf-8',newline='') as stream:
        names=['sample_id','class_name','schema_version','feature_id','feature_name','value_json','evidence','dimension_or_context','ordinal_mapping','risk_order_status','feature_score_lower_0_100','feature_score_midpoint_0_100','feature_score_upper_0_100','ordinal_known','not_applicable','participates_main_score','within_dimension_weight','main_contribution_lower_0_20','main_contribution_midpoint_0_20','main_contribution_upper_0_20','main_score_protocol','mapping_validation_scope','score_note']
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        for row in manifest:
            sid=row['sample_id'];c=row['class_name'];annotation=annotations[sid];schema=schemas[c];features=annotation['features'];definitions=schema['universal_features']+schema.get('category_specific_features',[])
            regionkeys=protocol['visibility_regions']['car' if c=='car' else 'noncar'];applicable=[k for k in regionkeys if features[k] not in protocol['not_applicable_values']]
            direction=['body_orientation'] if c=='car' else ['body_orientation','head_orientation']
            sums=dict(lower=0.,midpoint=0.,upper=0.)
            extra=dict(feature_id='target_scale',feature_name_zh='官方原图目标尺度',group='scale',possible_values=['tiny','small','medium','large'])
            area=float(row['bbox_area_ratio']);scale='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large'
            for definition in [extra]+definitions:
                field=definition['feature_id'];value=scale if field=='target_scale' else features[field];mapping_name,mapping=mapping_for(field,definition,c,protocol)
                dimension='scale' if field=='target_scale' else 'visibility' if field in regionkeys or mapping_name in ('part_visibility','legacy_tail_visibility','legacy_leg_visibility','leg_visibility','body_visibility') else 'orientation' if field in direction or field=='head_orientation' else 'occlusion' if field=='occlusion_level' else 'truncation' if field=='truncation_level' else 'descriptive_context'
                ordered=bool(mapping_name);napp=value in protocol['not_applicable_values'] if isinstance(value,str) else False
                known=False;bounds=None
                if field=='target_scale':v=sum(area<edge for edge in protocol['scale_edges'])/3;bounds=(v,v,v);known=True
                elif ordered:
                    if napp:bounds=None
                    elif value in protocol['unknown_values']:bounds=(0.,.5,1.)
                    elif value in mapping:v=float(mapping[value]);bounds=(v,v,v);known=True
                    else:raise ValueError((sid,field,value,'ordinal state lacks frozen mapping'))
                main=field=='target_scale' or field in regionkeys or field in direction or field in ('occlusion_level','truncation_level')
                weight=1. if field in ('target_scale','occlusion_level','truncation_level') else 1/len(direction) if field in direction else 1/len(applicable) if field in applicable else 0.
                contribution=[20*weight*v for v in bounds] if main and bounds is not None else []
                if contribution:
                    for bound,value_score in zip(('lower','midpoint','upper'),contribution):sums[bound]+=value_score
                output=dict(sample_id=sid,class_name=c,schema_version=schema['schema_version'],feature_id=field,feature_name=definition.get('feature_name_zh',field),value_json=json.dumps(value,ensure_ascii=False),evidence=('Official original image bounding box area ratio '+str(area)) if field=='target_scale' else annotation['evidence'][field],dimension_or_context=dimension,ordinal_mapping=mapping_name,risk_order_status='frozen_ordinal_visual_proxy_not_calibrated_failure_risk' if ordered else 'no_risk_order_defined',ordinal_known=known if ordered else '',not_applicable=napp,participates_main_score=main,within_dimension_weight=weight if main else '',main_score_protocol=protocol['version'],mapping_validation_scope='existing frozen design; not independently validated feature-wise classifier failure probability',score_note='Main component, with explicit weight; do not also add descendant descriptions' if main else 'Separate ordinal descriptor, excluded from main total' if ordered else 'Nominal or complex state retained for description and coverage; score blank')
                for index,bound in enumerate(('lower','midpoint','upper')):
                    output['feature_score_'+bound+'_0_100']=100*bounds[index] if bounds is not None else ''
                    output['main_contribution_'+bound+'_0_20']=contribution[index] if contribution else ''
                writer.writerow(output);field_summary[c,'ordinal' if ordered else 'unmapped',str(main)]+=1;states_summary[c,field,str(value),'known' if known else 'unknown_or_not_scored']+=1
                ordinal_count+=int(ordered);unmapped_count+=int(not ordered);napp_count+=int(napp)
            # No applicable visibility region is itself an unknown whole region,
            # as in the frozen main protocol. Do not invent a scored child.
            if not applicable:
                sums['midpoint']+=10;sums['upper']+=20
            for bound,value in sums.items():assert np.isclose(value,float(lookup[sid]['score_'+bound]),atol=1e-9,rtol=0),(sid,bound,'feature contribution sum mismatch')
    assert ordinal_count+unmapped_count==244400
    freeze={str(path):sha(path) for path in freeze_paths};freeze[str(destination)]=sha(destination)
    audit=dict(status='pass',input_targets=7800,output_feature_rows=244400,ordinal_feature_rows=ordinal_count,unmapped_description_rows=unmapped_count,normal_view_not_applicable_rows=napp_count,
               complete_annotation_gate=True,main_score_unchanged=True,all_main_contributions_sum_existing_score=True,descendant_fields_not_double_counted=True,
               complex_eye_visibility_no_risk_order=True,nominal_color_texture_hair_style_no_risk_order=True,
               existing_ordered_mappings_only=True,featurewise_failure_risk_validated=False,
               empty_visibility_policy='Zero applicable child regions retains frozen unknown whole-region [0,10,20] contribution, not fabricated child scores.')
    (out/'feature_score_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');freeze[str(out/'feature_score_audit.json')]=sha(out/'feature_score_audit.json')
    (out/'feature_score_freeze.sha256.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
