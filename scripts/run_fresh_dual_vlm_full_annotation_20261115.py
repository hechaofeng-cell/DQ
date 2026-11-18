#!/usr/bin/env python3
"""Run the frozen 4,800-image dual-VLM annotation with resumable provenance."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_final_holdout_20261115 as final
import run_fresh_vlm_holdout_20261115 as held
import run_fresh_vlm_universal_chunked_holdout_20261115 as transport
import smoke_fresh_dual_vlm_20261112 as smoke
import validate_fresh_vlm_format_repair_20261114 as repair
from prepare_fresh_images_20261112 import READY,OUT as DATA,read,sha
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint

OUT=DATA/'dual_vlm_full_annotation_20261115'
MANIFEST=DATA/'qualified_manifest.csv'
FINAL_GATE=DATA/'dual_vlm_final_holdout_20261115'
CLASSES=('dog','cat','horse','sheep','person','car')
ROLES=('selection','reference')
SEMANTIC_JOBS=8000
MAX_ATTEMPTS=3
MAX_BODY_BYTES=4*1024**2
EXPECTED_CALLS={'A':8000,'B':16000}
MAX_CALLS={'A':24000,'B':48000}


class AccessFailure(RuntimeError):pass


def configure():
    smoke.OUT=OUT;repair.OUT=OUT;held.OUT=OUT;transport.OUT=OUT
    transport.MAX_BODY_BYTES=MAX_BODY_BYTES;transport.NORMALIZATION=final.NORMALIZATION


configure()


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError('Frozen artifact changed: '+str(path))
    else:
        atomic_json(path,value)


def kinds(row):
    return ('native',) if row['class_name'] in ('dog','cat') else ('native','fine')


def build_job(row,kind):
    job=smoke.definition(row,kind)
    original_prompt=job['prompt']
    job['prompt']=repair.explicit_schema_prompt(job)
    job['original_prompt_sha256']=hashlib.sha256(original_prompt.encode()).hexdigest()
    evidence=repair.prepare_evidence(job)
    job.update(transport_evidence=evidence,evidence_paths=evidence['paths'],evidence_sha256=evidence['sha256'],
        holdout_role=row['role'],holdout_rank=int(row['rank']))
    return job


def iter_jobs(index=None):
    lookup={(item['sample_id'],item['kind']):item for item in index or []}
    for row in read(MANIFEST):
        for kind in kinds(row):
            job=build_job(row,kind);digest=fingerprint(job)
            if lookup and lookup[(job['sample_id'],kind)]['job_sha256']!=digest:
                raise ValueError('Job reconstruction changed')
            yield job,digest


def check_gates():
    qualification=json.loads((DATA/'qualification_summary.json').read_text())
    data_audit=json.loads((DATA/'independent_data_audit.json').read_text())
    final_audit=json.loads((FINAL_GATE/'independent_audit.json').read_text())
    if not qualification['data_qualified'] or data_audit['status']!='pass':
        raise ValueError('Data qualification gate failed')
    if final_audit['status']!='pass' or not final_audit['success_criterion_met'] or final_audit['valid']!=100:
        raise ValueError('Final transport gate failed')


def runtime():
    protocol=json.loads((READY/'frozen_validation_protocol.json').read_text())['annotations']
    online=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    if online['model']!=protocol['models']['B']:
        raise ValueError('B model differs from frozen protocol')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:11434/api/tags',timeout=20) as response:
        tags=json.load(response)
    if not any(item.get('name')==protocol['models']['A'] and item.get('digest')==protocol['frozen_native_model_A_digest']
            for item in tags['models']):
        raise ValueError('Frozen A model digest unavailable')
    return protocol,online


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);check_gates()
    rows=read(MANIFEST)
    counts=Counter((row['class_name'],row['role']) for row in rows)
    if len(rows)!=4800 or set(counts.values())!={400}:
        raise ValueError('Full manifest balance mismatch')
    protocol,online=runtime()
    sources=(Path(__file__).resolve(),ROOT/'scripts/audit_fresh_dual_vlm_full_annotation_20261115.py',
        ROOT/'scripts/continue_fresh_dual_vlm_full_annotation_20261115.py',
        ROOT/'tests/test_fresh_dual_vlm_full_annotation_20261115.py')
    frozen={str(path):sha(path) for path in sources}
    for path in (MANIFEST,DATA/'independent_data_audit.json',READY/'frozen_validation_protocol.json',
            FINAL_GATE/'inputs.json',FINAL_GATE/'summary.json',FINAL_GATE/'independent_audit.json'):
        frozen[str(path)]=sha(path)
    freeze(OUT/'code_and_input_freeze.json',frozen)
    freeze(OUT/'runtime.json',dict(A_model=protocol['models']['A'],
        A_digest=protocol['frozen_native_model_A_digest'],B_model=online['model'],B_url=online['url'],
        B_request_parameters=online.get('request_parameters',{}),maximum_semantic_attempts=MAX_ATTEMPTS))
    freeze(OUT/'protocol.json',dict(version='fresh-full-dual-vlm-v1-20261115',images=4800,semantic_jobs=SEMANTIC_JOBS,
        source_semantic_jobs=16000,expected_physical_calls=EXPECTED_CALLS,maximum_physical_calls=MAX_CALLS,
        A_transport='One frozen constrained-schema response per semantic attempt.',
        B_transport='Two disjoint <=10-field strict-schema components per semantic attempt; deterministic reconstruction.',
        normalization=final.NORMALIZATION,attempts='At most three identical semantic attempts; no response-dependent hint.',
        pacing='A local sequential; B online sequential with at least3.1 seconds after every component call.',
        inputs='All 4,800 qualified selection/reference targets; no replacement based on annotations.',
        boundary='Automatic labels and A/B agreement are not human semantic ground truth.'))
    index=[]
    for job,digest in iter_jobs():
        index.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job['holdout_role'],
            rank=job['holdout_rank'],kind=job['kind'],job_sha256=digest,evidence_sha256=job['evidence_sha256'],
            evidence_mode=job['transport_evidence']['mode']))
        if len(index)%100==0:
            atomic_json(OUT/'prepare_progress.json',dict(status='running',prepared_jobs=len(index),planned_jobs=SEMANTIC_JOBS,
                updated_at=datetime.now(timezone.utc).isoformat()))
    if len(index)!=SEMANTIC_JOBS or len({(x['sample_id'],x['kind']) for x in index})!=SEMANTIC_JOBS:
        raise ValueError('Semantic job count mismatch')
    freeze(OUT/'job_index.json',index)
    atomic_json(OUT/'prepare_progress.json',dict(status='complete',prepared_jobs=len(index),planned_jobs=SEMANTIC_JOBS,
        bounded_evidence_jobs=sum(x['evidence_mode']!='unchanged' for x in index),
        updated_at=datetime.now(timezone.utc).isoformat()))
    atomic_json(DATA/'pipeline_progress.json',dict(stage='full_dual_vlm_annotation_prepared',data_qualified=True,
        full_annotation_started=True,full_annotation_complete=False,images=4800,semantic_jobs=SEMANTIC_JOBS,
        source_jobs=2*SEMANTIC_JOBS,expected_physical_calls=EXPECTED_CALLS,human_verification=False,
        target_classifier_calls=0))
    return index


def source_folder(source,job):
    return OUT/'jobs'/source/f"{job['sample_id']}_{job['kind']}"


def call_folder(source,job,attempt,part):
    return source_folder(source,job)/f'attempt{attempt}_{part}'


def perform_call(source,job,cfg,attempt,part,fields=None):
    folder=call_folder(source,job,attempt,part);folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source=source,model=cfg['model'],transport=('A_full_strict' if source=='A' else 'B_universal_chunked_strict'),
        attempt=attempt,part=part,field_ids=fields,system=(None if source=='A' else transport.SYSTEM_PROMPT),
        normalization=(None if source=='A' else final.NORMALIZATION)))
    started=folder/'call_started.json';raw=folder/'raw.json';meta=folder/'transport.json'
    if started.exists():
        if json.loads(started.read_text())['input_binding']!=binding:
            raise ValueError('Started call binding changed')
        if not raw.exists() or not meta.exists():
            return dict(status='invalid',reason='started_without_complete_response',binding=binding)
    else:
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:
                raise ValueError('Evidence changed before call')
        if source=='A':
            body=smoke.request_body(job,'A',cfg['model'])
        else:
            body,_=transport.request_body(job,cfg,fields)
        size=len(json.dumps(body,ensure_ascii=False).encode())
        if size>MAX_BODY_BYTES:
            raise ValueError('Serialized request exceeds4MiB')
        atomic_json(started,dict(input_binding=binding,source=source,attempt=attempt,part=part,field_ids=fields,
            one_request=True,body_bytes=size,started_at=datetime.now(timezone.utc).isoformat()))
        try:
            status,text,elapsed,_=post_json(cfg['url'],body,{} if source=='A' else {'Authorization':'Bearer '+cfg['api_key']},900)
            if source=='B':
                text=text.replace(cfg['api_key'],'[REDACTED]')
            atomic_text(raw,text);atomic_json(meta,dict(status=status,elapsed_seconds=elapsed))
        except Exception as exc:
            atomic_json(folder/'error.json',dict(error_type=type(exc).__name__,phase='transport'))
            return dict(status='invalid',reason='transport',binding=binding)
    status=json.loads(meta.read_text())['status']
    if status in (401,402,403,404):
        raise AccessFailure(f'{source}_http_{status}')
    if status!=200:
        return dict(status='invalid',reason='http_'+str(status),binding=binding)
    return dict(status='response',binding=binding,raw=raw)


def parse_a(call,job):
    response=json.loads(call['raw'].read_text());value=held.response_value(response,'A')
    outcome=held.validate_raw_or_normalized(value,job)
    if outcome['result'] is None:
        return dict(status='invalid',reason='schema',normalization=outcome['changes'])
    parsed=dict(status='valid',standard=outcome['result'],wire=value,normalization=outcome['changes'],
        input_binding=call['binding'],raw_sha256=sha(call['raw']))
    return parsed


def execute_a(job,cfg):
    folder=source_folder('A',job);result_path=folder/'result.json'
    if result_path.exists():
        value=json.loads(result_path.read_text());
        if value['status']=='valid':held.validate_result(value['standard'],job)
        return value
    attempts=[]
    for attempt in range(1,MAX_ATTEMPTS+1):
        call=perform_call('A',job,cfg,attempt,'full')
        if call['status']=='response':
            try:
                parsed=parse_a(call,job);atomic_json(call_folder('A',job,attempt,'full')/'parsed.json',parsed)
                if parsed['status']=='valid':
                    result=dict(status='valid',attempt=attempt,standard=parsed['standard'],normalization=parsed['normalization'],
                        raw_sha256=[parsed['raw_sha256']],reconstruction='full_strict_response')
                    atomic_json(result_path,result);return result
                call=parsed
            except Exception as exc:
                call=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
        attempts.append({k:v for k,v in call.items() if k not in ('raw','binding')})
        atomic_json(call_folder('A',job,attempt,'full')/'outcome.json',attempts[-1])
    result=dict(status='invalid',attempts=attempts);atomic_json(result_path,result);return result


def execute_b(job,cfg):
    folder=source_folder('B',job);result_path=folder/'result.json'
    if result_path.exists():
        value=json.loads(result_path.read_text())
        if value['status']=='valid':held.validate_result(value['standard'],job)
        return value
    history=[]
    chunks=transport.field_chunks(job)
    for attempt in range(1,MAX_ATTEMPTS+1):
        features={};evidence={};values={};changes=[];hashes=[];outcomes=[]
        for index,fields in enumerate(chunks,1):
            part='part'+str(index);call=perform_call('B',job,cfg,attempt,part,fields);time.sleep(3.1)
            if call['status']=='response':
                try:
                    response=json.loads(call['raw'].read_text());parsed,wire,part_changes=final.parse_component(response,job,fields)
                    record=dict(status='valid',parsed=parsed,wire=wire,normalization=part_changes,
                        input_binding=call['binding'],raw_sha256=sha(call['raw']))
                    atomic_json(call_folder('B',job,attempt,part)/'parsed.json',record)
                    if job['kind']=='native':features.update(parsed['features']);evidence.update(parsed['evidence'])
                    else:values.update(parsed['values'])
                    changes.extend(part_changes);hashes.append(record['raw_sha256']);call=dict(status='valid')
                except Exception as exc:
                    call=dict(status='invalid',reason=type(exc).__name__,path=list(getattr(exc,'absolute_path',[])))
            outcome={k:v for k,v in call.items() if k not in ('raw','binding')};outcomes.append(outcome)
            atomic_json(call_folder('B',job,attempt,part)/'outcome.json',outcome)
        if all(item['status']=='valid' for item in outcomes):
            standard=(dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
                target_instance_id='target_1',features=features,evidence=evidence) if job['kind']=='native' else values)
            held.validate_result(standard,job)
            result=dict(status='valid',attempt=attempt,standard=standard,normalization=changes,
                raw_sha256=hashes,reconstruction='merge_two_disjoint_strict_components')
            atomic_json(result_path,result);return result
        history.append(dict(attempt=attempt,components=outcomes))
    result=dict(status='invalid',attempts=history);atomic_json(result_path,result);return result


def summarize_source(source,records,finished=False):
    counts=Counter(record['status'] for record in records)
    calls=len(list((OUT/'jobs'/source).glob('*/*/call_started.json')))
    value=dict(status=('pass' if counts['valid']==SEMANTIC_JOBS else 'failed') if finished else 'running',source=source,
        processed_jobs=len(records),planned_jobs=SEMANTIC_JOBS,valid_jobs=counts['valid'],invalid_jobs=counts['invalid'],
        physical_calls=calls,expected_physical_calls=EXPECTED_CALLS[source],maximum_physical_calls=MAX_CALLS[source],
        normalization_fields=sum(len(record.get('normalization',[])) for record in records),
        updated_at=datetime.now(timezone.utc).isoformat(),full_annotation_started=True,human_verification=False)
    atomic_json(OUT/f'{source}_progress.json',value)
    return value


def run_source(source):
    index=json.loads((OUT/'job_index.json').read_text());protocol,online=runtime()
    cfg=(dict(model=protocol['models']['A'],url='http://127.0.0.1:11434/api/chat') if source=='A' else online)
    lock=(OUT/f'{source}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=[]
    for job,_ in iter_jobs(index):
        result=execute_a(job,cfg) if source=='A' else execute_b(job,cfg)
        records.append(result);summarize_source(source,records)
        print(json.dumps(dict(source=source,processed=len(records),sample_id=job['sample_id'],kind=job['kind'],
            status=result['status']),ensure_ascii=False),flush=True)
    summary=summarize_source(source,records,True);atomic_json(OUT/f'{source}_summary.json',summary)
    return summary


def status():
    value={}
    for source in ('A','B'):
        path=OUT/f'{source}_progress.json';value[source]=json.loads(path.read_text()) if path.exists() else None
    print(json.dumps(value,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','source','status'),required=True)
    parser.add_argument('--source',choices=('A','B'));args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='source':
        if not args.source:parser.error('--source is required for source phase')
        run_source(args.source)
    else:status()


if __name__=='__main__':main()
