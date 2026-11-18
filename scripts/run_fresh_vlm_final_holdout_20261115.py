#!/usr/bin/env python3
"""Run the fourth disjoint B holdout with the final frozen transport cleanup."""
from __future__ import annotations

from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

from jsonschema import Draft7Validator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_universal_chunked_holdout_20261115 as transport
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.clients import load_online_config
from qe_quality.dta.io import atomic_json,atomic_text,unique_json_object

PARENT=DATA/'dual_vlm_universal_chunked_holdout_20261115'
OUT=DATA/'dual_vlm_final_holdout_20261115'
SEED=2026111504
PER_CLASS_ROLE=5
MAX_CALLS=200
MAX_BODY_BYTES=4*1024**2
NORMALIZATION=('Required evidence keys must all exist. A sole extra string key named evidence_json is dropped as a transport '
    'self-reference. Required nonempty list[str] evidence values are compact-JSON serialized. No other mutation is allowed.')


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def parse_component(response,job,field_ids):
    if job['kind']=='fine':
        wire=held.response_value(response,'B');Draft7Validator(transport.fine_schema(job,field_ids)).validate(wire)
        return dict(values={k:wire[k] for k in field_ids}),wire,[]
    value=held.response_value(response,'B');Draft7Validator(transport.chunk.component_schema(job,field_ids)).validate(value)
    evidence=unique_json_object(value['evidence_json']);required=set(field_ids);keys=set(evidence);changes=[]
    extras=keys-required;missing=required-keys
    if missing:raise ValueError('Component evidence is missing required keys')
    if extras:
        if extras!={'evidence_json'} or not isinstance(evidence['evidence_json'],str):
            raise ValueError('Unsupported extra evidence keys')
        removed=evidence.pop('evidence_json')
        changes.append(dict(path='/evidence/evidence_json',operation='drop_transport_self_reference',
            before=removed,after=None,reversible_by='restore recorded before value'))
    normalized={}
    for key in field_ids:
        item=evidence[key]
        if isinstance(item,str):normalized[key]=item
        elif isinstance(item,list) and item and all(isinstance(x,str) for x in item):
            encoded=json.dumps(item,ensure_ascii=False,separators=(',',':'))
            if json.loads(encoded)!=item:raise ValueError('Evidence serialization is not reversible')
            normalized[key]=encoded;changes.append(dict(path='/evidence/'+key,operation='JSON_array_to_JSON_string',
                before=item,after=encoded,reversible_by='json.loads'))
        else:raise ValueError('Unsupported evidence value type')
    return dict(features={k:value[k] for k in field_ids},evidence=normalized),value,changes


def configure_transport():
    transport.OUT=OUT;transport.MAX_CALLS=MAX_CALLS;transport.MAX_BODY_BYTES=MAX_BODY_BYTES
    transport.NORMALIZATION=NORMALIZATION;transport.parse_component=parse_component


configure_transport()


def offline_parent_audit():
    inputs=json.loads((PARENT/'inputs.json').read_text());records=[]
    for job in inputs['fresh_jobs']:
        features={};evidence={};values={};changes=[]
        for index,fields in enumerate(transport.field_chunks(job),1):
            raw=PARENT/'fresh/components'/f"{job['sample_id']}_{job['kind']}_B_part{index}"/'raw.json'
            response=json.loads(raw.read_text());parsed,wire,part_changes=parse_component(response,job,fields);changes.extend(part_changes)
            if job['kind']=='native':features.update(parsed['features']);evidence.update(parsed['evidence'])
            else:values.update(parsed['values'])
        standard=(dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
            target_instance_id='target_1',features=features,evidence=evidence) if job['kind']=='native' else values)
        held.validate_result(standard,job)
        records.append(dict(sample_id=job['sample_id'],kind=job['kind'],status='valid',changes=changes))
    cleanup=sum(c['operation']=='drop_transport_self_reference' for r in records for c in r['changes'])
    arrays=sum(c['operation']=='JSON_array_to_JSON_string' for r in records for c in r['changes'])
    audit=dict(status='pass' if len(records)==100 and cleanup==1 else 'failed',jobs=100,valid=len(records),invalid=100-len(records),
        self_reference_drops=cleanup,array_serializations=arrays,records=records,
        boundary='Offline replay of the previous holdout only; not independent confirmation.')
    freeze(OUT/'offline_parent_audit.json',audit)
    if audit['status']!='pass':raise ValueError('Offline parent audit failed')
    return audit


def excluded_ids():
    ids=set(transport.excluded_ids())
    ids.update(r['sample_id'] for r in json.loads((PARENT/'inputs.json').read_text())['fresh_targets'])
    return ids


def select_targets(rows):
    excluded=excluded_ids();selected=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            eligible=[r for r in rows if r['class_name']==class_name and r['role']==role and r['sample_id'] not in excluded]
            eligible.sort(key=lambda r:hashlib.sha256(f'{SEED}|{class_name}|{role}|{r["sample_id"]}'.encode()).hexdigest())
            if len(eligible)<PER_CLASS_ROLE:raise ValueError('Insufficient fourth-holdout targets')
            selected.extend(eligible[:PER_CLASS_ROLE])
    if len(selected)!=60 or len({r['sample_id'] for r in selected})!=60:raise ValueError('Target count mismatch')
    return selected


def make_jobs(targets):
    held.OUT=OUT
    return held.make_jobs(targets)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Final holdout already finished')
    offline=offline_parent_audit();targets=select_targets(read(DATA/'qualified_manifest.csv'));jobs=make_jobs(targets)
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    parent_inputs=json.loads((PARENT/'inputs.json').read_text())
    if cfg['model']!=parent_inputs['model']:raise ValueError('B model changed')
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_final_holdout_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_final_holdout_20261115.py',ROOT/'tests/test_fresh_vlm_final_holdout_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (PARENT/'inputs.json',PARENT/'summary.json',PARENT/'independent_audit.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_parent_freeze.json',frozen)
    target_rows=[{k:r[k] for k in ('sample_id','class_name','role','rank','image_sha256','crop_sha256')} for r in targets]
    freeze(OUT/'inputs.json',dict(seed=SEED,fresh_targets=target_rows,fresh_jobs=jobs,
        excluded_sample_ids=sorted(excluded_ids()),model=cfg['model'],system_prompt=transport.SYSTEM_PROMPT,
        normalization=NORMALIZATION,semantic_jobs=100,model_calls=200))
    freeze(OUT/'protocol.json',dict(question='Does the final universal chunked transport validate on a fourth disjoint holdout?',
        offline_gate='Previous100 jobs must replay100/100 with exactly one self-reference drop and no other key mismatch.',
        fresh='Fourth disjoint balanced60-image holdout;100 semantic jobs; two calls each.',
        success='100/100 reconstructed jobs valid;200/200 one-attempt component calls completed.',
        self_reference_rule='Drop only a sole extra string evidence key named evidence_json when every required key exists.',
        attempts_per_component=1,maximum_calls=200,no_retry=True,
        boundary='Structural transport reliability only; no semantic accuracy claim.'))
    records=[]
    for job in jobs:
        result=transport.execute_job(job,cfg,'fresh');record=dict(sample_id=job['sample_id'],class_name=job['class_name'],
            role=job['holdout_role'],kind=job['kind'],**result);records.append(record)
        transport.summarize('fresh',records,100);print(json.dumps(record,ensure_ascii=False),flush=True)
    fresh=transport.summarize('fresh',records,100,'pass' if all(r['status']=='valid' for r in records) else 'failed')
    final=dict(status=fresh['status'],stage='fourth_fresh_holdout',fresh=fresh,success_criterion_met=fresh['status']=='pass',
        total_model_calls=fresh['model_calls'],offline_parent_audit_sha256=sha(OUT/'offline_parent_audit.json'),
        full_annotation_started=False,target_classifier_calls=0)
    atomic_json(OUT/'summary.json',final)


if __name__=='__main__':main()
