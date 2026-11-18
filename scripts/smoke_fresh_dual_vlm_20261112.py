#!/usr/bin/env python3
"""Twenty frozen native/fine jobs on six qualified targets, two independent VLMs."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_fresh_images_20261112 import READY, OUT as DATA, read, sha
from qe_quality.dta.clients import load_online_config, post_json
from qe_quality.dta.io import atomic_json, fingerprint, unique_json_object
from run_dog_local_pipeline import feature_schema, validate_feature
from annotate_dual_vlm import format_reminder
from annotate_other_classes_specific_v2_20261018 import specification, validate as validate_fine
from run_other_classes_specific_experiment_20261018 import contact
from jsonschema import Draft7Validator

OUT=DATA/'dual_vlm_smoke'
CLASSES=['dog','cat','horse','sheep','person','car']
NATIVE=ROOT/'artifacts/sixclass_independent_confirmation_20261004'
FINE=ROOT/'artifacts/other_classes_specific_experiment_20261018/frozen_dictionaries'


class AccessFailure(RuntimeError):pass


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen smoke input changed')
    else:atomic_json(path,value)


def require_data_gate():
    q=json.loads((DATA/'qualification_summary.json').read_text())
    audit=json.loads((DATA/'independent_data_audit.json').read_text())
    manifest=sha(DATA/'qualified_manifest.csv')
    if q['status']!='pass' or not q['data_qualified'] or audit['status']!='pass' or audit['manifest_sha256']!=manifest:
        raise ValueError('Actual data qualification and independent audit must pass before any VLM call')
    return manifest


def choose_targets(rows):
    return [min((r for r in rows if r['class_name']==c and r['role']=='selection'),key=lambda r:int(r['rank'])) for c in CLASSES]


def definition(row,kind):
    sid=row['sample_id'];c=row['class_name']
    if kind=='native':
        domain='car' if c=='car' else 'old'
        sp=NATIVE/'configs'/f'{domain}_annotation_schema.json'
        schema=json.loads(sp.read_text())
        compact={k:schema[k] for k in ('schema_version','category')}
        compact['features']=[{k:item.get(k,True) for k in ('feature_id','value_type','possible_values','allow_unknown')}
            for item in schema['universal_features']+schema.get('category_specific_features',[])]
        prompt=(NATIVE/'prompts'/f'{domain}.txt').read_text()+'\n特征字典如下：\n'+json.dumps(compact,ensure_ascii=False)+format_reminder(schema)
        prompt+='\nimage_id: '+sid+'\ntarget_instance_id: target_1'
        response=feature_schema(sp)
        paths=[row[k] for k in ('image_path','marked_image_path','feature_crop_path')]
        options=dict(temperature=0,seed=20260911,num_ctx=8192,num_predict=2600)
    else:
        sp=FINE/f'{c}.json';schema=json.loads(sp.read_text())
        prompt,response=specification(c,schema['features'])
        p=OUT/'evidence'/f'{sid}_fine.jpg';p.parent.mkdir(parents=True,exist_ok=True)
        content=contact(row,0)
        if p.exists() and p.read_bytes()!=content:raise ValueError('Contact evidence changed')
        if not p.exists():p.write_bytes(content)
        paths=[str(p)]
        options=dict(temperature=0,seed=20261018,num_ctx=16384,num_predict=2000)
    return dict(sample_id=sid,class_name=c,kind=kind,prompt=prompt,response_schema=response,
        semantic_schema=schema,schema_path=str(sp),schema_sha256=sha(sp),evidence_paths=paths,
        evidence_sha256=[sha(p) for p in paths],options=options,
        original_sha256=row['image_sha256'],crop_sha256=row['crop_sha256'])


def validate_output(value,job):
    json.dumps(value,allow_nan=False)
    Draft7Validator(job['response_schema']).validate(value)
    if job['kind']=='native':validate_feature(value,{'image_id':job['sample_id']},job['semantic_schema'])
    else:validate_fine(value,job['semantic_schema']['features'])


def request_body(job,source,model,extra=None):
    images=[base64.b64encode(Path(p).read_bytes()).decode('ascii') for p in job['evidence_paths']]
    if source=='A':
        return dict(model=model,stream=False,keep_alive='30m',format=job['response_schema'],options=job['options'],
            messages=[dict(role='user',content=job['prompt'],images=images)])
    return {**(extra or {}),'model':model,'temperature':0,'max_tokens':job['options']['num_predict'],
        'response_format':dict(type='json_object'),'messages':[dict(role='user',content=[dict(type='text',text=job['prompt'])]+
            [dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+i)) for i in images])]}


def run_job(job,source,cfg,pacing):
    folder=OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}"
    folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source=source,model=cfg['model']))
    parsed=folder/'parsed.json'
    if parsed.exists():
        old=json.loads(parsed.read_text())
        if old['input_binding']!=binding:raise ValueError('Parsed result belongs to another job')
        validate_output(old['result'],job);return old
    body=request_body(job,source,cfg['model'],cfg.get('request_parameters'))
    for attempt in range(1,4):
        stamp=folder/f'attempt{attempt}_started.json';rawpath=folder/f'attempt{attempt}_raw.json'
        meta=folder/f'attempt{attempt}_transport.json'
        if not stamp.exists():
            if source=='B':time.sleep(max(0,3.1-(time.monotonic()-pacing[0])))
            # A lost response consumes an attempt, rather than silently repeating a charged request.
            atomic_json(stamp,dict(input_binding=binding,source=source,attempt=attempt,started_unix=time.time()))
            try:
                status,text,elapsed,_=post_json(cfg['url'],body,
                    {} if source=='A' else {'Authorization':'Bearer '+cfg['api_key']},900)
                if source=='B':text=text.replace(cfg['api_key'],'[REDACTED]');pacing[0]=time.monotonic()
                rawpath.write_text(text)
                atomic_json(meta,dict(status=status,elapsed_seconds=elapsed))
            except Exception as exc:
                atomic_json(folder/f'attempt{attempt}_error.json',dict(error_type=type(exc).__name__,phase='transport'))
                continue
        elif json.loads(stamp.read_text())['input_binding']!=binding:raise ValueError('Attempt input changed')
        if not rawpath.exists() or not meta.exists():continue
        status=json.loads(meta.read_text())['status']
        if status in (401,402,403,404):raise AccessFailure(f'{source}_access_http_{status}')
        if status!=200:
            if status==429 and attempt<3:time.sleep(60)
            continue
        try:
            response=json.loads(rawpath.read_text())
            if source=='A':
                if response.get('done_reason')!='stop':raise ValueError('Truncated generation')
                content=response['message']['content']
            else:
                if response['choices'][0].get('finish_reason')!='stop':raise ValueError('Truncated generation')
                content=response['choices'][0]['message']['content']
            result=unique_json_object(content);validate_output(result,job)
            value=dict(input_binding=binding,source=source,model=cfg['model'],attempt=attempt,result=result,
                returned_model=response.get('model'),
                evidence_sha256=job['evidence_sha256'],prompt_sha256=hashlib.sha256(job['prompt'].encode()).hexdigest(),
                response_schema_sha256=fingerprint(job['response_schema']))
            atomic_json(parsed,value);return value
        except Exception as exc:
            # No answers/labels or validation-dependent semantic hints are fed back to either model.
            atomic_json(folder/f'attempt{attempt}_error.json',dict(error_type=type(exc).__name__,phase='schema_or_completion'))
    raise RuntimeError(f"Unresolved job after at most3 attempts: {source}/{job['sample_id']}/{job['kind']}")


def main():
    manifest=require_data_gate();OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((READY/'frozen_validation_protocol.json').read_text())['annotations']
    targets=choose_targets(read(DATA/'qualified_manifest.csv'))
    jobs=[definition(r,k) for r in targets for k in (['native'] if r['class_name'] in ('dog','cat') else ['native','fine'])]
    freeze(OUT/'inputs.json',dict(manifest_sha256=manifest,jobs=jobs,semantic_jobs=20,
        human_verification=False,revised_pilot_prompt_used=False,
        sampling='first qualified selection rank per class; targets stay in original pools',
        backend_difference='A uses native constrained-schema decoding; B uses JSON-object mode; both use identical task/schema validation/evidence',
        no_semantic_changes_on_retry=True))
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    if cfg['model']!=protocol['models']['B']:raise ValueError('B model differs from frozen protocol')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:11434/api/tags',timeout=20) as r:tags=json.load(r)
    if not any(m.get('name')==protocol['models']['A'] and m.get('digest')==protocol['frozen_native_model_A_digest'] for m in tags['models']):
        raise ValueError('Frozen local model digest unavailable')
    freeze(OUT/'runtime.json',dict(A_model=protocol['models']['A'],A_digest=protocol['frozen_native_model_A_digest'],
        B_model=cfg['model'],B_url=cfg['url'],B_request_parameters=cfg.get('request_parameters',{}),max_attempts_per_job=3))
    sources={'A':dict(model=protocol['models']['A'],url='http://127.0.0.1:11434/api/chat'),'B':cfg}
    pacing=[0.0];done=0;failures=[]
    for job in jobs:
        for source in ('A','B'):
            try:run_job(job,source,sources[source],pacing);done+=1
            except AccessFailure:raise
            except RuntimeError as exc:failures.append(str(exc))
            atomic_json(OUT/'progress.json',dict(completed_jobs=done,planned_jobs=20,failed_jobs=len(failures)))
            print(json.dumps(dict(stage='dual_vlm_smoke',done=done,planned=20,failed=len(failures))),flush=True)
    agreement={};unknown={}
    for job in jobs:
        paths=[OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_{s}"/'parsed.json' for s in ('A','B')]
        if not all(p.exists() for p in paths):continue
        values=[json.loads(p.read_text())['result'] for p in paths]
        features=[v['features'] if job['kind']=='native' else v for v in values]
        key=job['class_name']+'/'+job['kind']
        agreement[key]=dict(equal_fields=sum(features[0][k]==features[1][k] for k in features[0]),fields=len(features[0]))
        unknown[key]={s:sum(v=='unknown' or isinstance(v,list) and 'unknown' in v for v in f.values()) for s,f in zip(('A','B'),features)}
    attempts=len(list((OUT/'jobs').glob('*/attempt*_started.json')))
    result=dict(status='pass' if done==20 and not failures else 'failed',completed_jobs=done,planned_jobs=20,
        attempted_VLM_calls=attempts,max_attempts=60,unresolved_jobs=failures,field_agreement=agreement,unknown_fields=unknown,
        human_verification=False,target_classifier_calls=0,full_annotation_started=False,
        boundary='Automatic target/schema/transport smoke only; agreement does not certify visual label accuracy.')
    atomic_json(OUT/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if result['status']!='pass':raise RuntimeError('Smoke unresolved; full annotation remains blocked')


if __name__=='__main__':main()
