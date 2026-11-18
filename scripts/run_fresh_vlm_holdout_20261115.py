#!/usr/bin/env python3
"""Run a frozen 60-image/200-call holdout for VLM format and transport reliability."""
from __future__ import annotations

import base64
from collections import Counter,defaultdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

from jsonschema import Draft7Validator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import smoke_fresh_dual_vlm_20261112 as smoke
import validate_fresh_vlm_format_repair_20261114 as repair
from prepare_fresh_images_20261112 import READY,OUT as DATA,read,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint,unique_json_object

OUT=DATA/'dual_vlm_holdout_20261115'
V1=DATA/'dual_vlm_smoke'
FORMAT_PILOT=DATA/'dual_vlm_format_repair_20261114'
CLASSES=['dog','cat','horse','sheep','person','car']
ROLES=['selection','reference']
SEED=2026111501
PER_CLASS_ROLE=5
MAX_CALLS=200
MAX_BODY_BYTES=4*1024**2


class AccessFailure(RuntimeError):pass


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)


def normalize_evidence_arrays(value,job):
    """Reversibly serialize only list-valued native evidence fields.

    Feature values, fine fields, identities and scalar evidence are immutable.
    """
    source=json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    result=json.loads(json.dumps(source,ensure_ascii=False,allow_nan=False))
    changes=[]
    if job['kind']!='native':return result,changes
    evidence=result.get('evidence')
    if not isinstance(evidence,dict):return result,changes
    schema=job['response_schema']['properties']['evidence']['properties']
    for key in sorted(evidence):
        current=evidence[key]
        if schema.get(key,{}).get('type')!='string' or not isinstance(current,list):continue
        if not current or any(not isinstance(item,str) for item in current):continue
        encoded=json.dumps(current,ensure_ascii=False,separators=(',',':'))
        if json.loads(encoded)!=current:raise ValueError('Evidence serialization is not reversible')
        evidence[key]=encoded
        changes.append(dict(path='/evidence/'+key,operation='JSON_array_to_JSON_string',before=current,after=encoded,
            reversible_by='json.loads'))
    # This explicit equality check prevents accidental changes outside evidence.
    before_other={k:v for k,v in source.items() if k!='evidence'}
    after_other={k:v for k,v in result.items() if k!='evidence'}
    if before_other!=after_other:raise ValueError('Normalization changed a non-evidence field')
    return result,changes


def validate_result(value,job):
    smoke.validate_output(value,job)
    return value


def validate_raw_or_normalized(value,job):
    try:
        validate_result(value,job)
        return dict(status='raw_valid',result=value,changes=[])
    except Exception as raw_error:
        normalized,changes=normalize_evidence_arrays(value,job)
        if not changes:
            return dict(status='invalid',result=None,changes=[],raw_error_type=type(raw_error).__name__,
                raw_error_path=list(getattr(raw_error,'absolute_path',[])))
        try:
            validate_result(normalized,job)
            return dict(status='normalized_valid',result=normalized,changes=changes,
                raw_error_type=type(raw_error).__name__,raw_error_path=list(getattr(raw_error,'absolute_path',[])))
        except Exception as normalized_error:
            return dict(status='invalid',result=None,changes=changes,
                raw_error_type=type(raw_error).__name__,raw_error_path=list(getattr(raw_error,'absolute_path',[])),
                normalized_error_type=type(normalized_error).__name__,
                normalized_error_path=list(getattr(normalized_error,'absolute_path',[])))


def response_value(response,source):
    if source=='A':
        if response.get('done_reason')!='stop':raise ValueError('generation_not_complete')
        content=response['message']['content']
    else:
        choice=response['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('generation_not_complete')
        content=choice['message']['content']
    return unique_json_object(content)


def format_pilot_jobs():
    return json.loads((FORMAT_PILOT/'inputs.json').read_text())['jobs']


def offline_normalization_audit():
    jobs=format_pilot_jobs();records=[]
    for job in jobs:
        for source in ('A','B'):
            folder=FORMAT_PILOT/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}"
            raw=folder/'raw.json';meta=folder/'transport.json'
            if not raw.exists() or not meta.exists():raise ValueError('Prior bounded pilot is incomplete')
            status=json.loads(meta.read_text())['status']
            if status!=200:
                records.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],source=source,
                    raw_sha256=sha(raw),status='invalid',reason='http_'+str(status),changes=[]));continue
            response=json.loads(raw.read_text())
            try:value=response_value(response,source)
            except Exception as exc:
                records.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],source=source,
                    raw_sha256=sha(raw),status='invalid',reason=type(exc).__name__,changes=[]));continue
            outcome=validate_raw_or_normalized(value,job)
            records.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],source=source,
                raw_sha256=sha(raw),status=outcome['status'],changes=outcome['changes'],
                normalized_result_sha256=fingerprint(outcome['result']) if outcome['result'] is not None else None))
    counts=Counter(r['status'] for r in records)
    audit=dict(status='pass' if counts['invalid']==0 and len(records)==20 else 'failed',previous_jobs=len(records),
        raw_valid=counts['raw_valid'],normalized_valid=counts['normalized_valid'],invalid=counts['invalid'],
        transformed_fields=sum(len(r['changes']) for r in records),records=records,
        source_summary={s:dict(Counter(r['status'] for r in records if r['source']==s)) for s in ('A','B')},
        boundary='Offline structural normalization only; no model calls and no visual accuracy claim.')
    freeze(OUT/'offline_normalization_audit.json',audit)
    if audit['status']!='pass':raise RuntimeError('Offline normalization does not resolve all prior format failures')
    return audit


def debug_sample_ids():
    ids=set()
    for path in (V1/'inputs.json',FORMAT_PILOT/'inputs.json'):
        if not path.exists():continue
        ids.update(j['sample_id'] for j in json.loads(path.read_text())['jobs'])
    return ids


def select_targets(rows):
    excluded=debug_sample_ids();selected=[]
    for class_name in CLASSES:
        for role in ROLES:
            eligible=[r for r in rows if r['class_name']==class_name and r['role']==role and r['sample_id'] not in excluded]
            eligible.sort(key=lambda r:hashlib.sha256(f'{SEED}|{class_name}|{role}|{r["sample_id"]}'.encode()).hexdigest())
            if len(eligible)<PER_CLASS_ROLE:raise ValueError('Insufficient held-out targets')
            selected.extend(eligible[:PER_CLASS_ROLE])
    if len(selected)!=60 or len({r['sample_id'] for r in selected})!=60:raise ValueError('Held-out target count mismatch')
    return selected


def make_jobs(targets):
    # The imported constructor writes fine contact sheets under its OUT. Redirect it
    # only inside this isolated process before any constructor call.
    smoke.OUT=OUT;repair.OUT=OUT
    jobs=[]
    for row in targets:
        kinds=['native'] if row['class_name'] in ('dog','cat') else ['native','fine']
        for kind in kinds:
            job=smoke.definition(row,kind)
            job['prompt']=repair.explicit_schema_prompt(job)
            job['transport_evidence']=repair.prepare_evidence(job)
            job['evidence_paths']=job['transport_evidence']['paths']
            job['evidence_sha256']=job['transport_evidence']['sha256']
            job['holdout_role']=row['role'];job['holdout_rank']=int(row['rank'])
            jobs.append(job)
    if len(jobs)!=100:raise ValueError('Expected100 semantic jobs')
    return jobs


def body_size(body):
    return len(json.dumps(body,ensure_ascii=False).encode())


def execute(job,source,cfg):
    folder=OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}";folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source=source,model=cfg['model']))
    started=folder/'call_started.json';raw=folder/'raw.json';transport=folder/'transport.json'
    def finish(value):
        atomic_json(folder/'outcome.json',value)
        return value
    if started.exists():
        if json.loads(started.read_text())['input_binding']!=binding:raise ValueError('Held-out call binding changed')
        if not raw.exists() or not transport.exists():
            return finish(dict(status='invalid',reason='started_without_complete_response'))
    else:
        if len(list((OUT/'jobs').glob('*/call_started.json')))>=MAX_CALLS:raise ValueError('Held-out200-call budget exhausted')
        for p,h in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(p)!=h:raise ValueError('Held-out evidence changed before call')
        body=smoke.request_body(job,source,cfg['model'],cfg.get('request_parameters'))
        if body_size(body)>MAX_BODY_BYTES:raise ValueError('Held-out serialized body exceeds4MiB')
        atomic_json(started,dict(input_binding=binding,source=source,started_at=datetime.now(timezone.utc).isoformat(),
            one_attempt_only=True,body_bytes=body_size(body)))
        try:
            status,text,elapsed,_=post_json(cfg['url'],body,{} if source=='A' else {'Authorization':'Bearer '+cfg['api_key']},900)
            if source=='B':text=text.replace(cfg['api_key'],'[REDACTED]')
            atomic_text(raw,text);atomic_json(transport,dict(status=status,elapsed_seconds=elapsed))
        except Exception as exc:
            atomic_json(folder/'error.json',dict(reason='transport',error_type=type(exc).__name__))
            return finish(dict(status='invalid',reason='transport'))
    status=json.loads(transport.read_text())['status']
    if status in (401,402,403,404):raise AccessFailure(f'{source}_http_{status}')
    if status!=200:return finish(dict(status='invalid',reason='http_'+str(status)))
    try:
        response=json.loads(raw.read_text());value=response_value(response,source)
    except Exception as exc:
        return finish(dict(status='invalid',reason=type(exc).__name__))
    outcome=validate_raw_or_normalized(value,job)
    result={k:v for k,v in outcome.items() if k!='result'}
    if outcome['result'] is not None:
        atomic_json(folder/('parsed_raw.json' if outcome['status']=='raw_valid' else 'parsed_normalized.json'),dict(
            input_binding=binding,result=outcome['result'],source=source,model=cfg['model'],returned_model=response.get('model'),
            raw_sha256=sha(raw),evidence_sha256=job['evidence_sha256'],normalization=outcome['changes']))
    return finish(result)


def summarize(records,finished=False):
    counts=Counter(r['status'] for r in records)
    value=dict(updated_at=datetime.now(timezone.utc).isoformat(),status=('pass' if counts['invalid']==0 else 'failed') if finished else 'running',
        processed_calls=len(records),planned_calls=200,raw_valid=counts['raw_valid'],normalized_valid=counts['normalized_valid'],
        invalid=counts['invalid'],model_calls=len(list((OUT/'jobs').glob('*/call_started.json'))),maximum_calls=200,
        source_counts={s:dict(Counter(r['status'] for r in records if r['source']==s)) for s in ('A','B')},
        full_annotation_started=False,target_classifier_calls=0,human_verification=False,
        boundary='Held-out format/transport reliability only; no semantic accuracy or method efficacy claim.')
    atomic_json(OUT/'progress.json',value)
    atomic_json(DATA/'pipeline_progress.json',dict(value,stage='dual_vlm_heldout_format_validation',data_qualified=True,
        packages_passed=18,source_verified_originals=5400))
    atomic_text(DATA/'当前进度.md',f"更新时间：{value['updated_at']}\n\n双 VLM 未调试样本格式验证：{value['status']}。\n\n"
        f"- 已处理 {len(records)}/200 次调用；原始格式通过 {counts['raw_valid']}，规范化后通过 {counts['normalized_valid']}，失败 {counts['invalid']}。\n"
        "- 60张样本在调用前已冻结，均未参与前两轮调试；每个模型/协议任务仅调用一次。\n"
        "- 图片资格与独立审计已通过；全量标注、选样和分类器推理尚未启动。\n")
    return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise ValueError('Held-out run already complete; do not rerun')
    q=json.loads((DATA/'qualification_summary.json').read_text());audit=json.loads((DATA/'independent_data_audit.json').read_text())
    if not q['data_qualified'] or audit['status']!='pass':raise ValueError('Data gate failed')
    offline=offline_normalization_audit()
    targets=select_targets(read(DATA/'qualified_manifest.csv'));jobs=make_jobs(targets)
    frozen_sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_vlm_holdout_20261115.py',
        ROOT/'scripts/continue_fresh_vlm_holdout_20261115.py',ROOT/'tests/test_fresh_vlm_holdout_20261115.py')
    code={str(path):sha(path) for path in frozen_sources}
    for p in (DATA/'box_exclusion_recovery_code_freeze_20261114.json',FORMAT_PILOT/'code_freeze.json',
              FORMAT_PILOT/'summary.json',FORMAT_PILOT/'independent_audit.json',DATA/'qualified_manifest.csv',DATA/'independent_data_audit.json'):
        code[str(p)]=sha(p)
    freeze(OUT/'code_and_parent_freeze.json',code)
    target_records=[{k:r[k] for k in ('sample_id','class_name','role','rank','image_sha256','crop_sha256')} for r in targets]
    freeze(OUT/'inputs.json',dict(seed=SEED,targets=target_records,jobs=jobs,semantic_jobs=100,source_jobs=200,
        excluded_debug_sample_ids=sorted(debug_sample_ids()),qualified_manifest_sha256=sha(DATA/'qualified_manifest.csv')))
    freeze(OUT/'protocol.json',dict(version='heldout-format-transport-v1-20261115',question='Does the fixed protocol yield structurally usable outputs on unseen holdout targets?',
        hypothesis='All200 source jobs validate raw or after the single frozen reversible evidence-array serialization.',
        setting='Six classes; five selection and five reference originals per class; deterministic hash seed; prior debug targets excluded.',
        primary_metric='post-normalization valid source-job fraction; success requires200/200',
        secondary_metrics=['raw-valid fraction','normalization rate','source/class/role/protocol breakdown'],
        calls=dict(maximum=200,attempts_per_job=1,A=100,B=100),
        fixed=['qualified manifest','target IDs','native/fine schemas and definitions','A/B models','sampling options','transport cap','normalization rule'],
        normalization='Only nonempty list[str] under native evidence fields typed string is serialized as compact JSON text; reversible by json.loads; feature values immutable.',
        evidence='Identical task text and image bytes for A/B within each job; fixed4MiB transport rule; resized cases disclosed.',
        stop='Complete all jobs unless access/authentication fails; never retry a started job; any post-normalization invalid blocks full annotation.',
        paper_decision='Only200/200 permits consideration of full annotation; passing does not establish semantic accuracy.',
        reproduction=str(ROOT/'.venv/bin/python')+' scripts/run_fresh_vlm_holdout_20261115.py',human_verification=False))
    atomic_text(OUT/'research_log.md','# Held-out VLM format validation research log\n\n'
        '- Claim decision: whether the fixed annotation transport can proceed beyond smoke testing.\n'
        '- Frozen setting: 60 unseen originals, 100 semantic jobs, two independent sources, 200 one-attempt calls.\n'
        '- Primary success: 200/200 raw-valid or valid after the one allowed reversible evidence serialization.\n'
        '- Raw outputs and failures are retained; no target replacement, semantic retry, classifier prediction or human review.\n')
    protocol=json.loads((READY/'frozen_validation_protocol.json').read_text())['annotations']
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    if cfg['model']!=protocol['models']['B']:raise ValueError('Unexpected online model')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:11434/api/tags',timeout=20) as r:tags=json.load(r)
    if not any(m.get('name')==protocol['models']['A'] and m.get('digest')==protocol['frozen_native_model_A_digest'] for m in tags['models']):
        raise ValueError('Frozen local model unavailable')
    freeze(OUT/'runtime.json',dict(A_model=protocol['models']['A'],A_digest=protocol['frozen_native_model_A_digest'],
        B_model=cfg['model'],B_url=cfg['url'],B_request_parameters=cfg.get('request_parameters',{})))
    sources={'A':dict(model=protocol['models']['A'],url='http://127.0.0.1:11434/api/chat'),'B':cfg}
    records=[];summarize(records)
    fatal=None
    try:
        for job in jobs:
            for source in ('A','B'):
                outcome=execute(job,source,sources[source]);outcome.update(sample_id=job['sample_id'],class_name=job['class_name'],
                    role=job['holdout_role'],kind=job['kind'],source=source)
                records.append(outcome);summarize(records)
                print(json.dumps(outcome,ensure_ascii=False),flush=True)
                if source=='B':time.sleep(3.1)
    except AccessFailure as exc:
        fatal=type(exc).__name__+':'+str(exc)
    final=summarize(records,finished=True)
    final.update(records=records,offline_normalization_audit_sha256=sha(OUT/'offline_normalization_audit.json'),
        target_count=len(targets),semantic_jobs=len(jobs),fatal_error=fatal)
    atomic_json(OUT/'summary.json',final)
    if fatal:raise RuntimeError(fatal)
    print(json.dumps({k:v for k,v in final.items() if k!='records'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
