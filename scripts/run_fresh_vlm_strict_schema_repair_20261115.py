#!/usr/bin/env python3
"""Run four one-shot strict-schema B repairs for held-out structural failures."""
from __future__ import annotations

from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_holdout_20261115 as held
import smoke_fresh_dual_vlm_20261112 as smoke
from prepare_fresh_images_20261112 import OUT as DATA,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint

PARENT=DATA/'dual_vlm_holdout_20261115'
OUT=DATA/'dual_vlm_strict_schema_repair_20261115'
MAX_CALLS=4
MAX_BODY_BYTES=4*1024**2
SYSTEM_PROMPT=(
    '你是严格按给定JSON Schema输出的视觉标注器。只能使用schema中的字段名、类型和枚举ID，不创造新字段或新ID。'
    '清楚可见但颜色字典未单列的主色使用mixed_or_other；被人抱持的目标在pose_action中使用interacting；'
    'evidence的键必须与features的键完全一致。只输出一个JSON对象。')


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def parent_failures():
    inputs=json.loads((PARENT/'inputs.json').read_text())
    failures=[]
    for job in inputs['jobs']:
        folder=PARENT/'jobs'/f"{job['sample_id']}_{job['kind']}_B"
        outcome=json.loads((folder/'outcome.json').read_text())
        if outcome['status']=='invalid':
            failures.append(dict(job=job,parent_outcome=outcome,parent_raw_sha256=sha(folder/'raw.json')))
    failures.sort(key=lambda item:(item['job']['sample_id'],item['job']['kind']))
    if len(failures)!=4:raise ValueError('Expected exactly four frozen B failures')
    return failures


def strict_body(job,cfg):
    body=smoke.request_body(job,'B',cfg['model'],cfg.get('request_parameters'))
    body['response_format']=dict(type='json_schema',json_schema=dict(
        name='frozen_visual_state',strict=True,schema=job['response_schema']))
    body['messages']=[dict(role='system',content=SYSTEM_PROMPT),*body['messages']]
    return body


def body_size(body):
    return len(json.dumps(body,ensure_ascii=False).encode())


def run_one(item,cfg):
    job=item['job'];folder=OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_B";folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source='B',model=cfg['model'],response_format='strict_json_schema',system=SYSTEM_PROMPT))
    started=folder/'call_started.json';raw=folder/'raw.json';transport=folder/'transport.json'
    if started.exists():raise ValueError('Strict repair call already started; no retry permitted')
    if len(list((OUT/'jobs').glob('*/call_started.json')))>=MAX_CALLS:raise ValueError('Strict repair call budget exhausted')
    for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
        if sha(path)!=digest:raise ValueError('Evidence changed before strict repair')
    body=strict_body(job,cfg);size=body_size(body)
    if size>MAX_BODY_BYTES:raise ValueError('Strict repair body exceeds4MiB')
    atomic_json(started,dict(input_binding=binding,started_at=datetime.now(timezone.utc).isoformat(),
        source='B',one_attempt_only=True,body_bytes=size,response_format='strict_json_schema'))
    try:
        status,text,elapsed,_=post_json(cfg['url'],body,{'Authorization':'Bearer '+cfg['api_key']},900)
        text=text.replace(cfg['api_key'],'[REDACTED]')
        atomic_text(raw,text);atomic_json(transport,dict(status=status,elapsed_seconds=elapsed))
    except Exception as exc:
        result=dict(status='invalid',reason='transport',error_type=type(exc).__name__)
        atomic_json(folder/'outcome.json',result);return result
    if status in (401,402,403,404):raise RuntimeError('B access failure HTTP '+str(status))
    if status!=200:
        result=dict(status='invalid',reason='http_'+str(status));atomic_json(folder/'outcome.json',result);return result
    try:
        response=json.loads(raw.read_text());value=held.response_value(response,'B');held.validate_result(value,job)
        result=dict(status='raw_valid')
        atomic_json(folder/'parsed.json',dict(input_binding=binding,result=value,model=cfg['model'],
            returned_model=response.get('model'),raw_sha256=sha(raw),evidence_sha256=job['evidence_sha256']))
    except Exception as exc:
        result=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
    atomic_json(folder/'outcome.json',result);return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Strict repair already complete')
    failures=parent_failures()
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    parent_runtime=json.loads((PARENT/'runtime.json').read_text())
    if cfg['model']!=parent_runtime['B_model']:raise ValueError('B model changed')
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_strict_schema_repair_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_strict_schema_repair_20261115.py',
        ROOT/'tests/test_fresh_vlm_strict_schema_repair_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (PARENT/'inputs.json',PARENT/'summary.json',PARENT/'independent_audit.json',PARENT/'runtime.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_parent_freeze.json',frozen)
    targets=[dict(sample_id=x['job']['sample_id'],kind=x['job']['kind'],class_name=x['job']['class_name'],
        parent_outcome=x['parent_outcome'],parent_raw_sha256=x['parent_raw_sha256'],job=x['job']) for x in failures]
    freeze(OUT/'inputs.json',dict(targets=targets,source_jobs=4,model=cfg['model'],system_prompt=SYSTEM_PROMPT,
        change='B response_format json_object -> strict json_schema plus frozen legal-ID/key reminder'))
    freeze(OUT/'protocol.json',dict(question='Can strict server-side JSON Schema resolve the four frozen B structural failures?',
        success='4/4 raw responses validate without local normalization',attempts_per_job=1,maximum_calls=4,
        fixed=['model','images','image bytes','task prompt','semantic schema','temperature','four target IDs'],
        changed=['strict JSON Schema response format','system reminder for existing legal IDs and exact evidence keys'],
        stop='No retry; any invalid result blocks a fresh holdout validation.',
        boundary='Debug-set repair feasibility only; success is not independent holdout evidence or semantic accuracy.'))
    records=[]
    for item in failures:
        outcome=run_one(item,cfg);record=dict(sample_id=item['job']['sample_id'],class_name=item['job']['class_name'],
            kind=item['job']['kind'],source='B',**outcome);records.append(record)
        atomic_json(OUT/'progress.json',dict(status='running',processed_calls=len(records),planned_calls=4,
            raw_valid=sum(r['status']=='raw_valid' for r in records),invalid=sum(r['status']=='invalid' for r in records)))
        print(json.dumps(record,ensure_ascii=False),flush=True);time.sleep(3.1)
    valid=sum(r['status']=='raw_valid' for r in records)
    summary=dict(status='pass' if valid==4 else 'failed',processed_calls=4,model_calls=4,raw_valid=valid,
        invalid=4-valid,success_criterion_met=valid==4,records=records,full_annotation_started=False,
        fresh_holdout_started=False,target_classifier_calls=0)
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'progress.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
