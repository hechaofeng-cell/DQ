#!/usr/bin/env python3
"""Validate a universal two-component strict transport on a third fresh B holdout."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from jsonschema import Draft7Validator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_chunked_normalized_holdout_20261115 as previous
import run_fresh_vlm_chunked_strict_completion_20261115 as chunk
import run_fresh_vlm_flat_strict_completion_20261115 as flat
import run_fresh_vlm_holdout_20261115 as held
import smoke_fresh_dual_vlm_20261112 as smoke
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint

PARENT=DATA/'dual_vlm_chunked_normalized_holdout_20261115'
OUT=DATA/'dual_vlm_universal_chunked_holdout_20261115'
SEED=2026111503
PER_CLASS_ROLE=5
MAX_CALLS=204
MAX_BODY_BYTES=4*1024**2
SYSTEM_PROMPT=chunk.SYSTEM_PROMPT
NORMALIZATION=previous.NORMALIZATION


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def split_ids(ids):
    midpoint=math.ceil(len(ids)/2);parts=[ids[:midpoint],ids[midpoint:]]
    if not all(parts) or max(map(len,parts))>10:raise ValueError('Component exceeds10 fields')
    return parts


def field_chunks(job):
    if job['kind']=='native':return chunk.feature_chunks(job)
    return split_ids(list(job['response_schema']['properties']))


def fine_schema(job,field_ids):
    source=job['response_schema']['properties'];properties={key:deepcopy(source[key]) for key in field_ids}
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def fine_prompt(job,field_ids):
    return job['prompt']+'\n\n兼容传输要求：本次只输出以下字段，直接放在顶层，不输出其他字段：'+json.dumps(field_ids,ensure_ascii=False)+'。只使用原字典合法枚举ID。'


def request_body(job,cfg,field_ids):
    changed=deepcopy(job)
    if job['kind']=='native':changed['prompt']=chunk.component_prompt(job,field_ids);schema=chunk.component_schema(job,field_ids)
    else:changed['prompt']=fine_prompt(job,field_ids);schema=fine_schema(job,field_ids)
    body=smoke.request_body(changed,'B',cfg['model'],cfg.get('request_parameters'))
    body['response_format']=dict(type='json_schema',json_schema=dict(name='visual_state_component',strict=True,schema=schema))
    body['messages']=[dict(role='system',content=SYSTEM_PROMPT),*body['messages']]
    return body,schema


def parse_component(response,job,field_ids):
    if job['kind']=='native':
        features,evidence,wire,changes=previous.parse_component(response,job,field_ids)
        return dict(features=features,evidence=evidence),wire,changes
    wire=held.response_value(response,'B');Draft7Validator(fine_schema(job,field_ids)).validate(wire)
    return dict(values={k:wire[k] for k in field_ids}),wire,[]


def prior_pilot_jobs():
    inputs=json.loads((PARENT/'inputs.json').read_text())
    selected=[]
    for class_name in ('horse','person'):
        job=next(j for j in inputs['fresh_jobs'] if j['class_name']==class_name and j['kind']=='fine')
        outcome=json.loads((PARENT/'fresh/components'/f"{job['sample_id']}_{job['kind']}_B_full"/'outcome.json').read_text())
        if outcome.get('reason')!='http_400':raise ValueError('Expected prior fine HTTP400')
        selected.append(job)
    return selected


def excluded_ids():
    ids=set(flat.excluded_ids())
    ids.update(r['sample_id'] for r in json.loads((PARENT/'inputs.json').read_text())['fresh_targets'])
    return ids


def select_targets(rows):
    excluded=excluded_ids();selected=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            eligible=[r for r in rows if r['class_name']==class_name and r['role']==role and r['sample_id'] not in excluded]
            eligible.sort(key=lambda r:hashlib.sha256(f'{SEED}|{class_name}|{role}|{r["sample_id"]}'.encode()).hexdigest())
            if len(eligible)<PER_CLASS_ROLE:raise ValueError('Insufficient third-holdout targets')
            selected.extend(eligible[:PER_CLASS_ROLE])
    if len(selected)!=60 or len({r['sample_id'] for r in selected})!=60:raise ValueError('Target count mismatch')
    return selected


def make_jobs(targets):
    held.OUT=OUT
    return held.make_jobs(targets)


def folder_for(phase,job,part):
    return OUT/phase/'components'/f"{job['sample_id']}_{job['kind']}_B_{part}"


def execute_call(job,cfg,phase,part,field_ids):
    folder=folder_for(phase,job,part);folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source='B',model=cfg['model'],transport='universal_chunked_strict',
        part=part,field_ids=field_ids,system=SYSTEM_PROMPT,normalization=NORMALIZATION))
    started=folder/'call_started.json';raw=folder/'raw.json';meta=folder/'transport.json'
    if started.exists():
        if json.loads(started.read_text())['input_binding']!=binding:raise ValueError('Started binding changed')
        if not raw.exists() or not meta.exists():
            result=dict(status='invalid',reason='started_without_complete_response');atomic_json(folder/'outcome.json',result);return result
    else:
        if len(list(OUT.glob('*/components/*/call_started.json')))>=MAX_CALLS:raise ValueError('204-call budget exhausted')
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:raise ValueError('Evidence changed before call')
        body,schema=request_body(job,cfg,field_ids);size=len(json.dumps(body,ensure_ascii=False).encode())
        if size>MAX_BODY_BYTES:raise ValueError('Serialized body exceeds4MiB')
        atomic_json(started,dict(input_binding=binding,started_at=datetime.now(timezone.utc).isoformat(),
            one_attempt_only=True,body_bytes=size,part=part,field_ids=field_ids,response_schema_sha256=fingerprint(schema)))
        try:
            status,text,elapsed,_=post_json(cfg['url'],body,{'Authorization':'Bearer '+cfg['api_key']},900)
            text=text.replace(cfg['api_key'],'[REDACTED]');atomic_text(raw,text);atomic_json(meta,dict(status=status,elapsed_seconds=elapsed))
        except Exception as exc:
            result=dict(status='invalid',reason='transport',error_type=type(exc).__name__);atomic_json(folder/'outcome.json',result);return result
    status=json.loads(meta.read_text())['status']
    if status in (401,402,403,404):raise RuntimeError('B access failure HTTP '+str(status))
    if status!=200:
        result=dict(status='invalid',reason='http_'+str(status));atomic_json(folder/'outcome.json',result);return result
    try:
        response=json.loads(raw.read_text());parsed,wire,changes=parse_component(response,job,field_ids)
        atomic_json(folder/'parsed.json',dict(input_binding=binding,parsed=parsed,wire=wire,normalization=changes,raw_sha256=sha(raw)))
        result=dict(status='valid',part=part,normalization_fields=len(changes))
    except Exception as exc:
        result=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
    atomic_json(folder/'outcome.json',result);return result


def execute_job(job,cfg,phase):
    calls=[];features={};evidence={};values={};changes=[];hashes=[]
    for index,fields in enumerate(field_chunks(job),1):
        part='part'+str(index);outcome=execute_call(job,cfg,phase,part,fields);calls.append(outcome);time.sleep(3.1)
        if outcome['status']=='valid':
            parsed=json.loads((folder_for(phase,job,part)/'parsed.json').read_text());changes.extend(parsed['normalization'])
            hashes.append(parsed['raw_sha256'])
            if job['kind']=='native':features.update(parsed['parsed']['features']);evidence.update(parsed['parsed']['evidence'])
            else:values.update(parsed['parsed']['values'])
    if all(x['status']=='valid' for x in calls):
        standard=(dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],target_instance_id='target_1',
            features=features,evidence=evidence) if job['kind']=='native' else values)
        held.validate_result(standard,job)
        atomic_json(OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B.json",dict(status='valid',standard=standard,
            normalization=changes,component_raw_sha256=hashes,reconstruction='merge_two_disjoint_strict_components'))
        return dict(status='valid',component_calls=calls,normalization_fields=len(changes))
    return dict(status='invalid',component_calls=calls,normalization_fields=sum(x.get('normalization_fields',0) for x in calls))


def summarize(phase,records,planned,status='running'):
    value=dict(status=status,phase=phase,processed_jobs=len(records),planned_jobs=planned,
        valid_jobs=sum(r['status']=='valid' for r in records),invalid_jobs=sum(r['status']=='invalid' for r in records),
        model_calls=len(list(OUT.glob('*/components/*/call_started.json'))),maximum_calls=MAX_CALLS,
        normalization_fields=sum(r['normalization_fields'] for r in records),records=records,
        full_annotation_started=False,target_classifier_calls=0,human_verification=False)
    atomic_json(OUT/(phase+'_summary.json'),value);atomic_json(OUT/'progress.json',value);return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Universal holdout already finished')
    pilots=prior_pilot_jobs();targets=select_targets(read(DATA/'qualified_manifest.csv'));jobs=make_jobs(targets)
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    parent_inputs=json.loads((PARENT/'inputs.json').read_text())
    if cfg['model']!=parent_inputs['model']:raise ValueError('B model changed')
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_universal_chunked_holdout_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_universal_chunked_holdout_20261115.py',
        ROOT/'tests/test_fresh_vlm_universal_chunked_holdout_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (PARENT/'inputs.json',PARENT/'summary.json',PARENT/'independent_audit.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_parent_freeze.json',frozen)
    target_rows=[{k:r[k] for k in ('sample_id','class_name','role','rank','image_sha256','crop_sha256')} for r in targets]
    freeze(OUT/'inputs.json',dict(seed=SEED,pilot_jobs=pilots,fresh_targets=target_rows,fresh_jobs=jobs,
        excluded_sample_ids=sorted(excluded_ids()),model=cfg['model'],system_prompt=SYSTEM_PROMPT,
        normalization=NORMALIZATION,pilot_model_calls=4,fresh_model_calls=200))
    freeze(OUT/'protocol.json',dict(question='Does universal two-component strict transport validate all native and fine B jobs?',
        pilot='One prior horse and one prior person fine failure; four calls; proceed only at2/2 reconstructed jobs.',
        fresh='Third disjoint balanced60-image holdout;100 semantic jobs; two calls each; success requires100/100.',
        attempts_per_component=1,maximum_calls=204,no_retry=True,
        reconstruction='Merge two disjoint <=10-field components; native evidence arrays allow only reversible compact-JSON serialization.',
        boundary='Structural transport reliability only; no semantic accuracy claim.'))
    pilot_records=[]
    for job in pilots:
        result=execute_job(job,cfg,'pilot');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],**result)
        pilot_records.append(record);summarize('pilot',pilot_records,2);print(json.dumps(record,ensure_ascii=False),flush=True)
    pilot=summarize('pilot',pilot_records,2,'pass' if all(r['status']=='valid' for r in pilot_records) else 'failed')
    if pilot['status']!='pass':
        atomic_json(OUT/'summary.json',dict(status='failed',stage='pilot',pilot=pilot,fresh_started=False,
            success_criterion_met=False,total_model_calls=pilot['model_calls'],full_annotation_started=False));return
    records=[]
    for job in jobs:
        result=execute_job(job,cfg,'fresh');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],
            role=job['holdout_role'],kind=job['kind'],**result);records.append(record)
        summarize('fresh',records,100);print(json.dumps(record,ensure_ascii=False),flush=True)
    fresh=summarize('fresh',records,100,'pass' if all(r['status']=='valid' for r in records) else 'failed')
    atomic_json(OUT/'summary.json',dict(status=fresh['status'],stage='fresh_holdout',pilot=pilot,fresh=fresh,
        fresh_started=True,success_criterion_met=fresh['status']=='pass',total_model_calls=fresh['model_calls'],
        full_annotation_started=False,target_classifier_calls=0))


if __name__=='__main__':main()
