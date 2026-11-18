#!/usr/bin/env python3
"""Test and validate a two-component strict transport for native B jobs."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import json
import math
from pathlib import Path
import sys
import time

from jsonschema import Draft7Validator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_flat_strict_completion_20261115 as flat
import run_fresh_vlm_holdout_20261115 as held
import smoke_fresh_dual_vlm_20261112 as smoke
from prepare_fresh_images_20261112 import OUT as DATA,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint,unique_json_object

PARENT=DATA/'dual_vlm_flat_strict_completion_20261115'
OUT=DATA/'dual_vlm_chunked_strict_completion_20261115'
MAX_CALLS=164
MAX_BODY_BYTES=4*1024**2
SYSTEM_PROMPT=flat.SYSTEM_PROMPT


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def feature_chunks(job):
    ids=list(job['response_schema']['properties']['features']['properties'])
    midpoint=math.ceil(len(ids)/2)
    chunks=[ids[:midpoint],ids[midpoint:]]
    if not all(chunks) or max(map(len,chunks))>10:raise ValueError('Native feature chunk exceeds10 fields')
    return chunks


def component_schema(job,field_ids):
    source=job['response_schema']['properties']['features']['properties']
    properties={key:deepcopy(source[key]) for key in field_ids};properties['evidence_json']={'type':'string'}
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def component_prompt(job,field_ids):
    return job['prompt']+'\n\n本次只输出以下特征子集，不输出其他特征、features对象或身份字段：'+json.dumps(field_ids,ensure_ascii=False)+(
        '。这些特征直接放在顶层。另输出evidence_json字符串，其内容必须是可由json.loads解析的JSON对象，键恰好为相同特征子集，值为简短可见证据字符串。')


def component_body(job,cfg,field_ids):
    changed=deepcopy(job);changed['prompt']=component_prompt(job,field_ids);schema=component_schema(job,field_ids)
    body=smoke.request_body(changed,'B',cfg['model'],cfg.get('request_parameters'))
    body['response_format']=dict(type='json_schema',json_schema=dict(name='visual_state_component',strict=True,schema=schema))
    body['messages']=[dict(role='system',content=SYSTEM_PROMPT),*body['messages']]
    return body,schema


def fine_body(job,cfg):
    return flat.request_body(job,cfg,'strict_fine')


def parse_component(response,job,field_ids):
    value=held.response_value(response,'B');Draft7Validator(component_schema(job,field_ids)).validate(value)
    evidence=unique_json_object(value['evidence_json'])
    if set(evidence)!=set(field_ids) or any(not isinstance(v,str) for v in evidence.values()):
        raise ValueError('Component evidence keys/types mismatch')
    return {k:value[k] for k in field_ids},evidence,value


def folder_for(phase,job,part):
    return OUT/phase/'components'/f"{job['sample_id']}_{job['kind']}_B_{part}"


def execute_call(job,cfg,phase,part,field_ids=None):
    folder=folder_for(phase,job,part);folder.mkdir(parents=True,exist_ok=True)
    transport='chunked_native' if field_ids is not None else 'strict_fine'
    binding=fingerprint(dict(job=job,source='B',model=cfg['model'],transport=transport,part=part,
        field_ids=field_ids,system=SYSTEM_PROMPT))
    started=folder/'call_started.json';raw=folder/'raw.json';meta=folder/'transport.json'
    if started.exists():
        if json.loads(started.read_text())['input_binding']!=binding:raise ValueError('Started binding changed')
        if not raw.exists() or not meta.exists():
            result=dict(status='invalid',reason='started_without_complete_response');atomic_json(folder/'outcome.json',result);return result
    else:
        if len(list(OUT.glob('*/components/*/call_started.json')))>=MAX_CALLS:raise ValueError('164-call budget exhausted')
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:raise ValueError('Evidence changed before call')
        body,schema=component_body(job,cfg,field_ids) if field_ids is not None else fine_body(job,cfg)
        size=len(json.dumps(body,ensure_ascii=False).encode())
        if size>MAX_BODY_BYTES:raise ValueError('Serialized body exceeds4MiB')
        atomic_json(started,dict(input_binding=binding,started_at=datetime.now(timezone.utc).isoformat(),
            one_attempt_only=True,body_bytes=size,transport=transport,part=part,field_ids=field_ids,
            response_schema_sha256=fingerprint(schema)))
        try:
            status,text,elapsed,_=post_json(cfg['url'],body,{'Authorization':'Bearer '+cfg['api_key']},900)
            text=text.replace(cfg['api_key'],'[REDACTED]');atomic_text(raw,text);atomic_json(meta,dict(status=status,elapsed_seconds=elapsed))
        except Exception as exc:
            result=dict(status='invalid',reason='transport',error_type=type(exc).__name__)
            atomic_json(folder/'outcome.json',result);return result
    status=json.loads(meta.read_text())['status']
    if status in (401,402,403,404):raise RuntimeError('B access failure HTTP '+str(status))
    if status!=200:
        result=dict(status='invalid',reason='http_'+str(status));atomic_json(folder/'outcome.json',result);return result
    try:
        response=json.loads(raw.read_text())
        if field_ids is not None:
            features,evidence,wire=parse_component(response,job,field_ids)
            parsed=dict(input_binding=binding,features=features,evidence=evidence,wire=wire,raw_sha256=sha(raw))
        else:
            wire=held.response_value(response,'B');held.validate_result(wire,job)
            parsed=dict(input_binding=binding,standard=wire,wire=wire,raw_sha256=sha(raw))
        atomic_json(folder/'parsed.json',parsed);result=dict(status='valid',transport=transport,part=part)
    except Exception as exc:
        result=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
    atomic_json(folder/'outcome.json',result);return result


def execute_job(job,cfg,phase):
    calls=[]
    if job['kind']=='native':
        features={};evidence={}
        for index,field_ids in enumerate(feature_chunks(job),1):
            outcome=execute_call(job,cfg,phase,'part'+str(index),field_ids);calls.append(outcome);time.sleep(3.1)
            if outcome['status']=='valid':
                parsed=json.loads((folder_for(phase,job,'part'+str(index))/'parsed.json').read_text())
                features.update(parsed['features']);evidence.update(parsed['evidence'])
        if all(x['status']=='valid' for x in calls):
            standard=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
                target_instance_id='target_1',features=features,evidence=evidence)
            held.validate_result(standard,job)
            atomic_json(OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B.json",dict(
                status='valid',standard=standard,component_raw_sha256=[sha(folder_for(phase,job,'part1')/'raw.json'),
                    sha(folder_for(phase,job,'part2')/'raw.json')],reconstruction='merge_disjoint_feature_and_evidence_components'))
            return dict(status='valid',component_calls=calls)
    else:
        outcome=execute_call(job,cfg,phase,'full');calls.append(outcome);time.sleep(3.1)
        if outcome['status']=='valid':
            standard=json.loads((folder_for(phase,job,'full')/'parsed.json').read_text())['standard']
            atomic_json(OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B.json",dict(
                status='valid',standard=standard,component_raw_sha256=[sha(folder_for(phase,job,'full')/'raw.json')],
                reconstruction='identity'))
            return dict(status='valid',component_calls=calls)
    return dict(status='invalid',component_calls=calls)


def summarize(phase,records,planned,status='running'):
    calls=len(list(OUT.glob('*/components/*/call_started.json')))
    value=dict(status=status,phase=phase,processed_jobs=len(records),planned_jobs=planned,
        valid_jobs=sum(r['status']=='valid' for r in records),invalid_jobs=sum(r['status']=='invalid' for r in records),
        model_calls=calls,maximum_calls=MAX_CALLS,records=records,full_annotation_started=False,target_classifier_calls=0)
    atomic_json(OUT/(phase+'_summary.json'),value);atomic_json(OUT/'progress.json',value);return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Chunked strict completion already finished')
    parent_inputs=json.loads((PARENT/'inputs.json').read_text());parent_summary=json.loads((PARENT/'summary.json').read_text())
    if parent_summary['fresh_started']:raise ValueError('Parent fresh holdout unexpectedly started')
    pilots=parent_inputs['pilot_jobs'];jobs=parent_inputs['fresh_jobs'];targets=parent_inputs['fresh_targets']
    if len(pilots)!=2 or len(jobs)!=100 or len(targets)!=60:raise ValueError('Frozen parent input count mismatch')
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    if cfg['model']!=parent_inputs['model']:raise ValueError('B model changed')
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_chunked_strict_completion_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_chunked_strict_completion_20261115.py',
        ROOT/'tests/test_fresh_vlm_chunked_strict_completion_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (PARENT/'inputs.json',PARENT/'summary.json',PARENT/'independent_audit.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_parent_freeze.json',frozen)
    freeze(OUT/'inputs.json',dict(pilot_jobs=pilots,fresh_jobs=jobs,fresh_targets=targets,
        excluded_sample_ids=parent_inputs['excluded_sample_ids'],model=cfg['model'],system_prompt=SYSTEM_PROMPT,
        pilot_model_calls=4,fresh_model_calls=160))
    freeze(OUT/'protocol.json',dict(question='Does splitting native strict output into two <=10-feature components avoid B schema rejection?',
        pilot='Two known native failures, four component calls; proceed only if both reconstructed jobs validate.',
        fresh='Same sixty unseen balanced targets and100 semantic jobs;60 native jobs use two calls,40 fine jobs use one; success requires100/100 reconstructed jobs and160/160 completed calls.',
        attempts_per_component=1,maximum_calls=164,no_retry=True,
        reconstruction='Merge two disjoint strict feature maps and exact-key evidence_json maps; inject only frozen identity fields; validate original standard schema.',
        boundary='Structural transport reliability only; no semantic accuracy claim.'))
    pilot_records=[]
    for job in pilots:
        result=execute_job(job,cfg,'pilot');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],**result)
        pilot_records.append(record);summarize('pilot',pilot_records,2);print(json.dumps(record,ensure_ascii=False),flush=True)
    pilot=summarize('pilot',pilot_records,2,'pass' if all(r['status']=='valid' for r in pilot_records) else 'failed')
    if pilot['status']!='pass':
        atomic_json(OUT/'summary.json',dict(status='failed',stage='pilot',pilot=pilot,fresh_started=False,
            success_criterion_met=False,total_model_calls=pilot['model_calls'],full_annotation_started=False));return
    fresh_records=[]
    for job in jobs:
        result=execute_job(job,cfg,'fresh');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],
            role=job['holdout_role'],kind=job['kind'],**result);fresh_records.append(record)
        summarize('fresh',fresh_records,100);print(json.dumps(record,ensure_ascii=False),flush=True)
    fresh=summarize('fresh',fresh_records,100,'pass' if all(r['status']=='valid' for r in fresh_records) else 'failed')
    atomic_json(OUT/'summary.json',dict(status=fresh['status'],stage='fresh_holdout',pilot=pilot,fresh=fresh,
        fresh_started=True,success_criterion_met=fresh['status']=='pass',total_model_calls=fresh['model_calls'],
        full_annotation_started=False,target_classifier_calls=0))


if __name__=='__main__':main()
