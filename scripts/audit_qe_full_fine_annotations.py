#!/usr/bin/env python3
"""Independent frozen-input, raw-response and fine-score verification.

Partial runs are audited as partial; a complete result requires all 7,800
validated labels and all 236,600 evidence-bearing feature records.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/QE_VisualState_CoverRisk_v1_20261013'
OUT=ROOT/'artifacts/QE_FullFine_20261013'
OLD=ROOT/'artifacts/fresh_vlm_feature_scoring_20261012'
CLASSES=('dog','cat','horse','sheep','person','car')

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def close(a,b,label):assert np.isclose(float(a),float(b),atol=1e-9,rtol=0),(label,a,b)

def agg(values):
    if not values:values=[None]
    n=len(values);known=[v for v in values if v is not None];lo=sum(known)/n;unknown=n-len(known)
    hi=lo+unknown/n
    return dict(lower=lo,upper=hi,midpoint=(lo+hi)/2,known_weight=len(known)/n,unknown_components=unknown)

def score(row,f,p):
    c=row['class_name'];maps=p['maps'];old=c in ('dog','cat')
    regionkeys=p['visibility_regions']['car' if c=='car' else 'noncar']
    visible=[f[k] for k in regionkeys if f[k]!='not_in_view']
    parts={'visibility':agg([None if v in p['unknown_values'] else maps['part_visibility'][v] for v in visible])}
    area=float(row['bbox_area_ratio']);parts['scale']=agg([sum(area<e for e in p['scale_edges'])/3])
    for dim,field in [('occlusion','occlusion_level'),('truncation','truncation_level')]:
        mapping=maps['legacy_'+field if old else field]
        parts[dim]=agg([None if f[field]=='unknown' else mapping[f[field]]])
    direction=['body_orientation'] if c=='car' else ['body_orientation','head_orientation']
    parts['orientation']=agg([None if f[field]=='unknown' else maps['legacy_'+field if old else field][f[field]] for field in direction])
    low=20*sum(v['lower'] for v in parts.values());high=20*sum(v['upper'] for v in parts.values())
    result=dict(score_lower=low,score_upper=high,score_midpoint=(low+high)/2,
                fully_known=all(math.isclose(v['known_weight'],1) for v in parts.values()),
                evaluable=all(v['known_weight']>=p['minimum_known_weight']-1e-12 for v in parts.values()),
                unknown_dimensions=sum(v['unknown_components']>0 for v in parts.values()),applicable_visibility_regions=len(visible))
    for dim,values in parts.items():
        for key,value in values.items():result[dim+'_'+key]=value
    return result

def strict_annotation(value,row,schema):
    assert value['schema_version']==schema['schema_version']
    assert value['image_id']==row['sample_id'] and value['target_instance_id']==row['annotation_target_instance_id']=='target_1'
    definitions=schema['universal_features']+schema.get('category_specific_features',[])
    fields=value['features'];evidence=value['evidence'];assert set(fields)==set(evidence)=={d['feature_id'] for d in definitions}
    for definition in definitions:
        field=definition['feature_id'];v=fields[field]
        if definition['value_type']=='set':
            assert isinstance(v,list) and v and len(v)==len(set(v))
            assert all(x not in v or len(v)==1 for x in definition.get('exclusive_values',[]))
            states=v
        else:assert isinstance(v,str);states=[v]
        assert all(s in definition['possible_values'] for s in states),(row['sample_id'],field,'invalid enum')
        assert isinstance(evidence[field],str) and evidence[field].strip()
    if fields.get('mouth_visibility')=='not_visible':
        for field in ('mouth_state','tongue_visibility'):
            if field in fields:assert fields[field]=='unknown'
    if fields.get('tail_visibility')=='not_observable' and 'tail_posture' in fields:assert fields['tail_posture']=='unknown'

def parse_content(content):
    content=content.strip()
    if content.startswith('```'):
        content='\n'.join(content.splitlines()[1:-1]).strip()
    return json.loads(content)

def semantic_admission(value,row,protocol):
    """Independently repeat frozen v4/v5 rules, not semantic ground truth."""
    if not protocol.get('prompt_version','').startswith(('v4_','v5_')):return
    fields=value['features'];evidence=value['evidence']
    words=('模糊','低清','光照','灯光','舞台光','烟雾','烟','雾','逆光','阴影','过曝','blur','lighting','smoke','fog','backlight','shadow')
    if row['class_name']!='car':
        assert fields['torso_visibility']!='not_in_view' and fields['limb_visibility']!='not_in_view'
        for key in ('head_visibility','torso_visibility','limb_visibility'):
            if fields[key]=='not_visible':assert not any(word in evidence[key].lower() for word in words),(row['sample_id'],key,'frozen low-detail semantic admission')
    if fields['occlusion_level'] not in ('none','unknown'):
        assert not any(word in evidence['occlusion_level'].lower() for word in words),(row['sample_id'],'frozen opaque-object semantic admission')
    box=json.loads(row['bbox_norm_xyxy']);w=int(row['image_width']);h=int(row['image_height'])
    distances=[box[0]*w,box[1]*h,(1-box[2])*w,(1-box[3])*h]
    if min(distances)>2:
        assert fields['truncation_level']=='none'
        if 'truncation_parts' in fields:assert fields['truncation_parts']==['none']

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');parser.add_argument('--input',type=Path,default=OUT);args=parser.parse_args();OUT=args.input.resolve()
    manifest=read(OUT/'manifest.csv');master={r['sample_id']:r for r in read(DATA/'manifest.csv')}
    assert len(manifest)==len(master)==7800 and Counter(r['class_name'] for r in manifest)==Counter(dict.fromkeys(CLASSES,1300))
    assert set(r['sample_id'] for r in manifest)==set(master)
    freeze=json.loads((OUT/'input_freeze.sha256.json').read_text())
    for path,expected in freeze.items():assert sha(path)==expected,('input freeze changed',path)
    schemas={c:json.loads((OUT/'schemas'/f'{c}.json').read_text()) for c in CLASSES}
    for c,expected in zip(CLASSES,(35,35,27,27,32,26)):
        assert schemas[c]==json.loads((OLD/'schemas'/f'{c}.json').read_text())
        assert len(schemas[c]['universal_features'])==expected
    p=json.loads((OUT/'scoring_protocol.json').read_text());assert p==json.loads((OLD/'scoring_protocol.json').read_text())
    runtime=json.loads((OUT/'runtime.json').read_text());oldruntime=json.loads((OLD/'runtime.json').read_text())
    frozen_protocol=json.loads((OUT/'protocol.json').read_text())
    oldmanifest={r['sample_id']:r for r in read(OLD/'manifest.csv')};oldfreeze=json.loads((OLD/'input_freeze.sha256.json').read_text())
    records=0;newcount=0;reuse=0;fields=0;known=0;evaluable=0;byclass=Counter();scores={}
    for row in manifest:
        sid=row['sample_id'];m=master[sid]
        for key in ('class_name','label','source_pool','bbox_norm_xyxy','bbox_area_ratio','annotation_target_instance_id'):assert row[key]==m[key]
        for key in ('image_path','marked_image_path','feature_crop_path'):
            assert row[key]==str((DATA/m[key]).resolve()) and freeze[row[key]]==sha(row[key])
        assert row['official_target_instance_id']==m['target_instance_id'] and row['target_instance_id']=='target_1'
        with Image.open(row['image_path']) as im:
            assert im.size==(int(m['image_width']),int(m['image_height']))
            b=json.loads(row['bbox_norm_xyxy']);expected=[round(b[0]*im.width),round(b[1]*im.height),round(b[2]*im.width),round(b[3]*im.height)]
            assert json.loads(row['annotation_bbox'])==expected
        path=OUT/'annotations'/f'{sid}.json'
        if not path.exists():continue
        c=row['class_name'];schema=schemas[c];value=json.loads(path.read_text());strict_annotation(value,row,schema)
        semantic_admission(value,row,frozen_protocol)
        if 'prompt_version' in frozen_protocol:
            assert value['annotation_prompt_version']==frozen_protocol['prompt_version'],(sid,'prompt version differs from frozen protocol')
        assert value['schema_sha256']==sha(OUT/'schemas'/f'{c}.json') and value['prompt_sha256']==sha(OUT/'prompt.txt')
        if 'reuse_provenance' in value:
            prior=oldmanifest[sid];provenance=value['reuse_provenance'];original=Path(provenance['original_annotation'])
            assert provenance['original_annotation_sha256']==sha(original)
            sourcevalue=json.loads(original.read_text());assert value['features']==sourcevalue['features'] and value['evidence']==sourcevalue['evidence']
            assert prior['bbox_norm_xyxy']==row['bbox_norm_xyxy'] and prior['annotation_target_instance_id']==row['annotation_target_instance_id']
            for key in ('image_path','marked_image_path','feature_crop_path'):
                sourcepath=str(Path(prior[key]).resolve());assert oldfreeze[sourcepath]==freeze[row[key]]==sha(sourcepath)
            for key in ('model','num_ctx','num_predict','temperature','seed'):assert runtime[key]==oldruntime[key]
            reuse+=1
        else:
            assert value['annotation_model']==runtime['model']
            for key in ('image_path','marked_image_path','feature_crop_path'):assert value['input_view_sha256'][key]==freeze[row[key]]
            matched=[]
            for raw in (OUT/'raw').glob(sid+'_*.json'):
                try:
                    response=json.loads(raw.read_text());extracted=parse_content(response['message']['content'])
                    if extracted.get('features')==value['features'] and extracted.get('evidence')==value['evidence']:
                        assert response['done'] is True and response['model']==runtime['model'];matched.append(raw)
                except (ValueError,KeyError,TypeError):continue
            assert matched,(sid,'no raw model response matches final label')
            newcount+=1
        computed=score(row,value['features'],p);record=json.loads((OUT/'scores'/f'{sid}.json').read_text())
        for key,expected in computed.items():close(record[key],expected,(sid,key))
        scores[sid]=computed;records+=1;fields+=len(value['features']);known+=int(computed['fully_known']);evaluable+=int(computed['evaluable']);byclass[c]+=1
    if not args.partial:assert records==7800 and fields==236600,(records,fields,'full annotation not complete')
    if (OUT/'scores.csv').exists():
        rows=read(OUT/'scores.csv');assert len(rows)==len({r['sample_id'] for r in rows})
        for row in rows:
            sid=row['sample_id'];assert sid in scores
            for key,expected in scores[sid].items():
                if isinstance(expected,bool):assert row[key]==str(expected)
                else:close(row[key],expected,(sid,key,'export'))
    result=dict(status='partial_verified' if records<7800 else 'pass',planned=7800,valid_annotations=records,missing=7800-records,
                evidence_feature_records=fields,by_class=dict(byclass),actual_new_raw_responses_verified=newcount,reused_exactly_bound=reuse,
                scores_independently_recomputed=records,fully_known=known,evaluable=evaluable,input_frozen_paths_verified=len(freeze),
                schemas_equal_original_fine_schemas=True,main_scoring_protocol_unchanged=True,
                full_completion_claim_allowed=records==7800,semantic_vlm_accuracy_fully_verified=False)
    name='independent_partial_audit.json' if records<7800 else 'independent_annotation_audit.json'
    (OUT/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
