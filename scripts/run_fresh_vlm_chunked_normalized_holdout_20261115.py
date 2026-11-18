#!/usr/bin/env python3
"""Run a fresh B holdout with chunked strict native transport and reversible evidence normalization."""
from __future__ import annotations

from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import sys
import time

from jsonschema import Draft7Validator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_chunked_strict_completion_20261115 as chunk
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint,unique_json_object

PARENT=DATA/'dual_vlm_chunked_strict_completion_20261115'
SOURCE=DATA/'dual_vlm_flat_strict_completion_20261115'
OUT=DATA/'dual_vlm_chunked_normalized_holdout_20261115'
MAX_CALLS=160
MAX_BODY_BYTES=4*1024**2
NORMALIZATION='Only nonempty list[str] evidence values are compact-JSON serialized to strings; reversible by json.loads.'


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def parse_component(response,job,field_ids):
    value=held.response_value(response,'B');Draft7Validator(chunk.component_schema(job,field_ids)).validate(value)
    evidence=unique_json_object(value['evidence_json'])
    if set(evidence)!=set(field_ids):raise ValueError('Component evidence keys mismatch')
    normalized={};changes=[]
    for key in field_ids:
        item=evidence[key]
        if isinstance(item,str):normalized[key]=item
        elif isinstance(item,list) and item and all(isinstance(x,str) for x in item):
            encoded=json.dumps(item,ensure_ascii=False,separators=(',',':'))
            if json.loads(encoded)!=item:raise ValueError('Evidence serialization is not reversible')
            normalized[key]=encoded;changes.append(dict(path='/evidence/'+key,operation='JSON_array_to_JSON_string',
                before=item,after=encoded,reversible_by='json.loads'))
        else:raise ValueError('Unsupported evidence value type')
    return {k:value[k] for k in field_ids},normalized,value,changes


def offline_pilot_audit(pilots):
    records=[]
    for job in pilots:
        features={};evidence={};changes=[]
        for index,fields in enumerate(chunk.feature_chunks(job),1):
            raw=chunk.folder_for('pilot',job,'part'+str(index))/'raw.json'
            response=json.loads(raw.read_text());part_features,part_evidence,wire,part_changes=parse_component(response,job,fields)
            features.update(part_features);evidence.update(part_evidence);changes.extend(part_changes)
        standard=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
            target_instance_id='target_1',features=features,evidence=evidence);held.validate_result(standard,job)
        records.append(dict(sample_id=job['sample_id'],status='valid',changes=changes,
            parent_component_raw_sha256=[sha(chunk.folder_for('pilot',job,'part1')/'raw.json'),
                sha(chunk.folder_for('pilot',job,'part2')/'raw.json')]))
    audit=dict(status='pass' if len(records)==2 else 'failed',jobs=2,valid=len(records),invalid=2-len(records),
        normalization_fields=sum(len(r['changes']) for r in records),records=records,boundary='Offline replay only; no new calls.')
    freeze(OUT/'offline_pilot_audit.json',audit)
    return audit


def folder_for(job,part):
    return OUT/'fresh'/'components'/f"{job['sample_id']}_{job['kind']}_B_{part}"


def execute_call(job,cfg,part,field_ids=None):
    folder=folder_for(job,part);folder.mkdir(parents=True,exist_ok=True)
    transport='chunked_native_normalized' if field_ids is not None else 'strict_fine'
    binding=fingerprint(dict(job=job,source='B',model=cfg['model'],transport=transport,part=part,
        field_ids=field_ids,system=chunk.SYSTEM_PROMPT,normalization=NORMALIZATION))
    started=folder/'call_started.json';raw=folder/'raw.json';meta=folder/'transport.json'
    if started.exists():
        if json.loads(started.read_text())['input_binding']!=binding:raise ValueError('Started binding changed')
        if not raw.exists() or not meta.exists():
            result=dict(status='invalid',reason='started_without_complete_response');atomic_json(folder/'outcome.json',result);return result
    else:
        if len(list((OUT/'fresh/components').glob('*/call_started.json')))>=MAX_CALLS:raise ValueError('160-call budget exhausted')
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:raise ValueError('Evidence changed before call')
        body,schema=chunk.component_body(job,cfg,field_ids) if field_ids is not None else chunk.fine_body(job,cfg)
        size=len(json.dumps(body,ensure_ascii=False).encode())
        if size>MAX_BODY_BYTES:raise ValueError('Serialized body exceeds4MiB')
        atomic_json(started,dict(input_binding=binding,started_at=datetime.now(timezone.utc).isoformat(),one_attempt_only=True,
            body_bytes=size,transport=transport,part=part,field_ids=field_ids,response_schema_sha256=fingerprint(schema)))
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
        response=json.loads(raw.read_text())
        if field_ids is not None:
            features,evidence,wire,changes=parse_component(response,job,field_ids)
            parsed=dict(input_binding=binding,features=features,evidence=evidence,wire=wire,normalization=changes,raw_sha256=sha(raw))
        else:
            wire=held.response_value(response,'B');held.validate_result(wire,job);changes=[]
            parsed=dict(input_binding=binding,standard=wire,wire=wire,normalization=changes,raw_sha256=sha(raw))
        atomic_json(folder/'parsed.json',parsed);result=dict(status='valid',transport=transport,part=part,normalization_fields=len(changes))
    except Exception as exc:
        result=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
    atomic_json(folder/'outcome.json',result);return result


def execute_job(job,cfg):
    calls=[];features={};evidence={};changes=[]
    if job['kind']=='native':
        for index,fields in enumerate(chunk.feature_chunks(job),1):
            part='part'+str(index);outcome=execute_call(job,cfg,part,fields);calls.append(outcome);time.sleep(3.1)
            if outcome['status']=='valid':
                parsed=json.loads((folder_for(job,part)/'parsed.json').read_text());features.update(parsed['features'])
                evidence.update(parsed['evidence']);changes.extend(parsed['normalization'])
        if all(x['status']=='valid' for x in calls):
            standard=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
                target_instance_id='target_1',features=features,evidence=evidence);held.validate_result(standard,job)
            atomic_json(OUT/'fresh/jobs'/f"{job['sample_id']}_{job['kind']}_B.json",dict(status='valid',standard=standard,
                normalization=changes,component_raw_sha256=[sha(folder_for(job,'part1')/'raw.json'),sha(folder_for(job,'part2')/'raw.json')],
                reconstruction='merge_disjoint_components_with_reversible_evidence_serialization'))
            return dict(status='valid',component_calls=calls,normalization_fields=len(changes))
    else:
        outcome=execute_call(job,cfg,'full');calls.append(outcome);time.sleep(3.1)
        if outcome['status']=='valid':
            parsed=json.loads((folder_for(job,'full')/'parsed.json').read_text())
            atomic_json(OUT/'fresh/jobs'/f"{job['sample_id']}_{job['kind']}_B.json",dict(status='valid',standard=parsed['standard'],
                normalization=[],component_raw_sha256=[sha(folder_for(job,'full')/'raw.json')],reconstruction='identity'))
            return dict(status='valid',component_calls=calls,normalization_fields=0)
    return dict(status='invalid',component_calls=calls,normalization_fields=sum(x.get('normalization_fields',0) for x in calls))


def summarize(records,status='running'):
    value=dict(status=status,processed_jobs=len(records),planned_jobs=100,valid_jobs=sum(r['status']=='valid' for r in records),
        invalid_jobs=sum(r['status']=='invalid' for r in records),model_calls=len(list((OUT/'fresh/components').glob('*/call_started.json'))),
        maximum_calls=MAX_CALLS,normalization_fields=sum(r['normalization_fields'] for r in records),records=records,
        full_annotation_started=False,target_classifier_calls=0,human_verification=False)
    atomic_json(OUT/'progress.json',value);return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Normalized holdout already finished')
    source_inputs=json.loads((SOURCE/'inputs.json').read_text());pilots=source_inputs['pilot_jobs'];jobs=source_inputs['fresh_jobs']
    if len(jobs)!=100:raise ValueError('Fresh job count mismatch')
    offline=offline_pilot_audit(pilots)
    if offline['valid']!=2:raise ValueError('Offline pilot normalization failed')
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    if cfg['model']!=source_inputs['model']:raise ValueError('B model changed')
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_chunked_normalized_holdout_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_chunked_normalized_holdout_20261115.py',
        ROOT/'tests/test_fresh_vlm_chunked_normalized_holdout_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (SOURCE/'inputs.json',PARENT/'inputs.json',PARENT/'summary.json',PARENT/'independent_audit.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_parent_freeze.json',frozen)
    freeze(OUT/'inputs.json',dict(fresh_jobs=jobs,fresh_targets=source_inputs['fresh_targets'],
        excluded_sample_ids=source_inputs['excluded_sample_ids'],model=cfg['model'],system_prompt=chunk.SYSTEM_PROMPT,
        normalization=NORMALIZATION,semantic_jobs=100,model_calls=160))
    freeze(OUT/'protocol.json',dict(question='Does chunked strict transport plus the frozen reversible evidence-array serialization validate on unseen B targets?',
        offline_gate='Both prior pilot jobs must replay valid under the exact normalization rule.',
        fresh='Sixty unseen balanced targets;100 semantic jobs;60 native use two calls and40 fine use one.',
        success='100/100 reconstructed semantic jobs valid after at most the frozen evidence-only reversible serialization.',
        attempts_per_component=1,maximum_calls=160,no_retry=True,
        boundary='Structural transport reliability only; no semantic accuracy claim.'))
    records=[]
    for job in jobs:
        result=execute_job(job,cfg);record=dict(sample_id=job['sample_id'],class_name=job['class_name'],
            role=job['holdout_role'],kind=job['kind'],**result);records.append(record)
        summarize(records);print(json.dumps(record,ensure_ascii=False),flush=True)
    final=summarize(records,'pass' if all(r['status']=='valid' for r in records) else 'failed')
    final.update(success_criterion_met=final['status']=='pass',offline_pilot_audit_sha256=sha(OUT/'offline_pilot_audit.json'))
    atomic_json(OUT/'summary.json',final)


if __name__=='__main__':main()
