#!/usr/bin/env python3
"""Pilot and independently held-out test a B-compatible flat strict transport."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

from jsonschema import Draft7Validator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_holdout_20261115 as held
import run_fresh_vlm_strict_schema_repair_20261115 as strict
import smoke_fresh_dual_vlm_20261112 as smoke
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint,unique_json_object

PARENT=DATA/'dual_vlm_holdout_20261115'
STRICT_PARENT=DATA/'dual_vlm_strict_schema_repair_20261115'
OUT=DATA/'dual_vlm_flat_strict_completion_20261115'
SEED=2026111502
PER_CLASS_ROLE=5
MAX_CALLS=102
MAX_BODY_BYTES=4*1024**2
SYSTEM_PROMPT=(
    '你是严格按给定JSON Schema输出的视觉标注器。只能使用schema中的字段名、类型和枚举ID，不创造新字段或新ID。'
    '清楚可见但颜色字典未单列的主色使用mixed_or_other；被人抱持的目标在pose_action中使用interacting；'
    '所有证据只描述图中直接可见内容。只输出一个JSON对象。')


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def flat_schema(job):
    standard=job['response_schema'];feature_props=deepcopy(standard['properties']['features']['properties'])
    properties={
        'schema_version':{'type':'string','enum':[job['semantic_schema']['schema_version']]},
        'image_id':{'type':'string','enum':[job['sample_id']]},
        'target_instance_id':{'type':'string','enum':['target_1']},
        **feature_props,
        'evidence_json':{'type':'string'},
    }
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def flat_prompt(job):
    feature_ids=list(job['response_schema']['properties']['features']['properties'])
    evidence_ids=list(job['response_schema']['properties']['evidence']['properties'])
    return job['prompt']+'\n\n兼容传输要求：不要输出嵌套features/evidence对象。将以下features字段直接放在顶层：'+json.dumps(feature_ids,ensure_ascii=False)+(
        '。另输出evidence_json字符串；该字符串内容必须是一个JSON对象，键恰好为：'+json.dumps(evidence_ids,ensure_ascii=False)+
        '，每个值都是简短证据字符串。evidence_json必须可由json.loads无损解析。')


def request_body(job,cfg,transport):
    changed=deepcopy(job)
    if transport=='flat_native':changed['prompt']=flat_prompt(job);schema=flat_schema(job)
    elif transport=='strict_fine':schema=job['response_schema']
    else:raise ValueError('Unknown transport')
    body=smoke.request_body(changed,'B',cfg['model'],cfg.get('request_parameters'))
    body['response_format']=dict(type='json_schema',json_schema=dict(name='frozen_visual_state',strict=True,schema=schema))
    body['messages']=[dict(role='system',content=SYSTEM_PROMPT),*body['messages']]
    return body,schema


def reconstruct_flat(value,job):
    Draft7Validator(flat_schema(job)).validate(value)
    feature_ids=list(job['response_schema']['properties']['features']['properties'])
    evidence=unique_json_object(value['evidence_json'])
    expected=set(job['response_schema']['properties']['evidence']['properties'])
    if set(evidence)!=expected or any(not isinstance(v,str) for v in evidence.values()):
        raise ValueError('evidence_json keys/types differ from standard schema')
    standard=dict(schema_version=value['schema_version'],image_id=value['image_id'],
        target_instance_id=value['target_instance_id'],features={k:value[k] for k in feature_ids},evidence=evidence)
    held.validate_result(standard,job)
    return standard


def parse_response(response,job,transport):
    wire=held.response_value(response,'B')
    if transport=='flat_native':standard=reconstruct_flat(wire,job)
    else:held.validate_result(wire,job);standard=wire
    return wire,standard


def pilot_jobs():
    inputs=json.loads((PARENT/'inputs.json').read_text())
    jobs=[]
    for job in inputs['jobs']:
        if job['kind']!='native':continue
        prior=json.loads((STRICT_PARENT/'jobs'/f"{job['sample_id']}_{job['kind']}_B"/'outcome.json').read_text()) \
            if (STRICT_PARENT/'jobs'/f"{job['sample_id']}_{job['kind']}_B"/'outcome.json').exists() else None
        if prior and prior.get('reason')=='http_400':jobs.append(job)
    if len(jobs)!=2:raise ValueError('Expected two strict-native HTTP400 pilot jobs')
    return sorted(jobs,key=lambda j:j['sample_id'])


def excluded_ids():
    ids=set(held.debug_sample_ids())
    ids.update(r['sample_id'] for r in json.loads((PARENT/'inputs.json').read_text())['targets'])
    return ids


def select_targets(rows):
    excluded=excluded_ids();selected=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            eligible=[r for r in rows if r['class_name']==class_name and r['role']==role and r['sample_id'] not in excluded]
            eligible.sort(key=lambda r:hashlib.sha256(f'{SEED}|{class_name}|{role}|{r["sample_id"]}'.encode()).hexdigest())
            if len(eligible)<PER_CLASS_ROLE:raise ValueError('Insufficient fresh targets')
            selected.extend(eligible[:PER_CLASS_ROLE])
    if len(selected)!=60 or len({r['sample_id'] for r in selected})!=60:raise ValueError('Fresh target count mismatch')
    return selected


def make_fresh_jobs(targets):
    held.OUT=OUT
    return held.make_jobs(targets)


def execute(job,cfg,phase):
    transport='flat_native' if job['kind']=='native' else 'strict_fine'
    folder=OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B";folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source='B',model=cfg['model'],transport=transport,system=SYSTEM_PROMPT))
    started=folder/'call_started.json';raw=folder/'raw.json';meta=folder/'transport.json'
    if started.exists():
        if json.loads(started.read_text())['input_binding']!=binding:raise ValueError('Started binding changed')
        if not raw.exists() or not meta.exists():
            result=dict(status='invalid',reason='started_without_complete_response');atomic_json(folder/'outcome.json',result);return result
    else:
        if len(list(OUT.glob('*/jobs/*/call_started.json')))>=MAX_CALLS:raise ValueError('102-call budget exhausted')
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:raise ValueError('Evidence changed before call')
        body,schema=request_body(job,cfg,transport);size=len(json.dumps(body,ensure_ascii=False).encode())
        if size>MAX_BODY_BYTES:raise ValueError('Serialized body exceeds4MiB')
        atomic_json(started,dict(input_binding=binding,started_at=datetime.now(timezone.utc).isoformat(),one_attempt_only=True,
            body_bytes=size,transport=transport,response_schema_sha256=fingerprint(schema)))
        try:
            status,text,elapsed,_=post_json(cfg['url'],body,{'Authorization':'Bearer '+cfg['api_key']},900)
            text=text.replace(cfg['api_key'],'[REDACTED]');atomic_text(raw,text)
            atomic_json(meta,dict(status=status,elapsed_seconds=elapsed))
        except Exception as exc:
            result=dict(status='invalid',reason='transport',error_type=type(exc).__name__)
            atomic_json(folder/'outcome.json',result);return result
    status=json.loads(meta.read_text())['status']
    if status in (401,402,403,404):raise RuntimeError('B access failure HTTP '+str(status))
    if status!=200:
        result=dict(status='invalid',reason='http_'+str(status));atomic_json(folder/'outcome.json',result);return result
    try:
        response=json.loads(raw.read_text());wire,standard=parse_response(response,job,transport)
        atomic_json(folder/'parsed_wire.json',dict(input_binding=binding,result=wire,raw_sha256=sha(raw)))
        atomic_json(folder/'parsed_standard.json',dict(input_binding=binding,result=standard,raw_sha256=sha(raw),
            reconstruction='flat_features_plus_json_loads_evidence' if transport=='flat_native' else 'identity'))
        result=dict(status='valid',transport=transport)
    except Exception as exc:
        result=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
    atomic_json(folder/'outcome.json',result);return result


def summarize(phase,records,planned,status='running'):
    value=dict(status=status,phase=phase,processed_calls=len(records),planned_calls=planned,
        valid=sum(r['status']=='valid' for r in records),invalid=sum(r['status']=='invalid' for r in records),records=records,
        total_model_calls=len(list(OUT.glob('*/jobs/*/call_started.json'))),maximum_calls=MAX_CALLS,
        full_annotation_started=False,target_classifier_calls=0,human_verification=False)
    atomic_json(OUT/(phase+'_summary.json'),value);atomic_json(OUT/'progress.json',value)
    return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Flat strict completion already finished')
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    parent_runtime=json.loads((PARENT/'runtime.json').read_text())
    if cfg['model']!=parent_runtime['B_model']:raise ValueError('B model changed')
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_flat_strict_completion_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_flat_strict_completion_20261115.py',
        ROOT/'tests/test_fresh_vlm_flat_strict_completion_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (PARENT/'inputs.json',PARENT/'summary.json',PARENT/'independent_audit.json',PARENT/'runtime.json',
                 STRICT_PARENT/'summary.json',STRICT_PARENT/'independent_audit.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_parent_freeze.json',frozen)
    pilots=pilot_jobs();targets=select_targets(read(DATA/'qualified_manifest.csv'));jobs=make_fresh_jobs(targets)
    target_rows=[{k:r[k] for k in ('sample_id','class_name','role','rank','image_sha256','crop_sha256')} for r in targets]
    freeze(OUT/'inputs.json',dict(seed=SEED,pilot_jobs=pilots,fresh_targets=target_rows,fresh_jobs=jobs,
        excluded_sample_ids=sorted(excluded_ids()),pilot_source_jobs=2,fresh_source_jobs=100,model=cfg['model'],system_prompt=SYSTEM_PROMPT))
    freeze(OUT/'protocol.json',dict(
        question='Does a flat strict native envelope plus direct strict fine schema yield reliable B outputs?',
        phase1='Two known native HTTP400 cases; proceed only at2/2 valid.',
        phase2='Sixty unseen balanced originals,100 B jobs; success requires100/100 valid.',
        native_transport='Feature fields at top level under strict schema; evidence_json is parsed with duplicate-key rejection and exact evidence keys, then standard output is deterministically reconstructed.',
        fine_transport='Original fine schema supplied as strict server-side JSON Schema.',
        calls=dict(maximum=102,attempts_per_job=1,pilot=2,fresh=100),
        stop='Any pilot invalid prevents fresh calls; any fresh invalid blocks full annotation. No retry.',
        boundary='Structural/transport reliability only; no semantic accuracy claim.'))
    pilot_records=[]
    for job in pilots:
        result=execute(job,cfg,'pilot');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],**result)
        pilot_records.append(record);summarize('pilot',pilot_records,2);print(json.dumps(record,ensure_ascii=False),flush=True);time.sleep(3.1)
    pilot=summarize('pilot',pilot_records,2,'pass' if all(r['status']=='valid' for r in pilot_records) else 'failed')
    if pilot['status']!='pass':
        final=dict(status='failed',stage='pilot',pilot=pilot,fresh_started=False,success_criterion_met=False,
            total_model_calls=pilot['total_model_calls'],full_annotation_started=False,target_classifier_calls=0)
        atomic_json(OUT/'summary.json',final);return
    fresh_records=[]
    for job in jobs:
        result=execute(job,cfg,'fresh');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],
            role=job['holdout_role'],kind=job['kind'],**result);fresh_records.append(record)
        summarize('fresh',fresh_records,100);print(json.dumps(record,ensure_ascii=False),flush=True);time.sleep(3.1)
    fresh=summarize('fresh',fresh_records,100,'pass' if all(r['status']=='valid' for r in fresh_records) else 'failed')
    final=dict(status=fresh['status'],stage='fresh_holdout',pilot=pilot,fresh=fresh,fresh_started=True,
        success_criterion_met=fresh['status']=='pass',total_model_calls=fresh['total_model_calls'],
        full_annotation_started=False,target_classifier_calls=0)
    atomic_json(OUT/'summary.json',final)


if __name__=='__main__':main()
