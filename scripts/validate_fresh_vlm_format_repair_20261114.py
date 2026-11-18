#!/usr/bin/env python3
"""One bounded, separate transport/schema pilot; never retries exhausted v1 jobs."""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

from PIL import Image,ImageOps
from jsonschema import Draft7Validator

import smoke_fresh_dual_vlm_20261112 as original
from prepare_fresh_images_20261112 import ROOT,READY,OUT as DATA,sha,read
from qe_quality.dta.clients import load_online_config,post_json
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint,unique_json_object

OUT=DATA/'dual_vlm_format_repair_20261114'
MAX_BODY_BYTES=4*1024**2
MAX_CALLS=20


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Repair input changed: '+str(path))
    else:atomic_json(path,value)


def diagnose():
    old=original.OUT
    jobs=json.loads((old/'inputs.json').read_text())['jobs']
    records=[]
    for job in jobs:
        for source in ('A','B'):
            folder=old/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}"
            for raw in sorted(folder.glob('attempt*_raw.json')):
                meta=raw.with_name(raw.name.replace('_raw.json','_transport.json'))
                status=json.loads(meta.read_text())['status'];response=json.loads(raw.read_text())
                entry=dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],source=source,
                    raw_path=str(raw),raw_sha256=sha(raw),http_status=status,valid=False)
                if status!=200:entry.update(reason='http_'+str(status))
                else:
                    try:
                        if source=='A':content=response['message']['content'];finish=response.get('done_reason')
                        else:content=response['choices'][0]['message']['content'];finish=response['choices'][0].get('finish_reason')
                        if finish!='stop':raise ValueError('generation_not_complete')
                        value=unique_json_object(content);original.validate_output(value,job)
                        entry.update(valid=True,reason='valid')
                    except Exception as exc:
                        entry.update(reason=type(exc).__name__,validator=getattr(exc,'validator',None),
                            path=list(getattr(exc,'absolute_path',[])))
                records.append(entry)
    freeze(OUT/'previous_failure_diagnosis.json',dict(
        status='completed',previous_summary_sha256=sha(old/'summary.json'),
        previous_calls=len(list((old/'jobs').glob('*/attempt*_started.json'))),
        previous_attempt_outcomes=dict(Counter(r['reason'] for r in records)),records=records,
        conclusion='Six B jobs failed field typing; one B job failed HTTP413. No v1 attempt budget is reset.'))
    return records


def explicit_schema_prompt(job):
    # The exact existing schema is made visible to both backends. No answer, per-image
    # label suggestion, extra attribute definition or validation-dependent hint is included.
    return job['prompt']+('\n\n以下为完整输出 JSON Schema，仅规定原有字段和 JSON 类型，不改变任何视觉状态定义。'
        '\n必须严格按此结构输出；evidence 的每个字段均为一个字符串，set 类型字段为非空数组。'
        '\n不得输出 schema 本身、Markdown 或额外字段。\n'+
        json.dumps(job['response_schema'],ensure_ascii=False,separators=(',',':')))


def payload_size(job,images):
    # Use the larger of A and B serializations including transport parameters.
    options=job['options'];prompt=job['prompt']
    a=dict(model='qwen3-vl:30b-a3b-instruct',stream=False,keep_alive='30m',format=job['response_schema'],options=options,
        messages=[dict(role='user',content=prompt,images=images)])
    b=dict(model='sensenova-6.8-flash-lite',temperature=0,max_tokens=options['num_predict'],response_format={'type':'json_object'},
        messages=[dict(role='user',content=[dict(type='text',text=prompt)]+
            [dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+v)) for v in images])])
    return max(len(json.dumps(v,ensure_ascii=False).encode()) for v in (a,b))


def prepare_evidence(job):
    paths=[Path(p) for p in job['evidence_paths']]
    for p,h in zip(paths,job['evidence_sha256']):
        if sha(p)!=h:raise ValueError('Original frozen smoke evidence changed')
    content=[p.read_bytes() for p in paths]
    images=[base64.b64encode(v).decode('ascii') for v in content]
    if payload_size(job,images)<=MAX_BODY_BYTES:
        return dict(paths=[str(p) for p in paths],sha256=[sha(p) for p in paths],mode='unchanged',max_side=None,
            source_sha256=job['evidence_sha256'],serialized_body_bytes=payload_size(job,images))
    # The same fixed transport-size rule applies before every call and is independent
    # of model responses, labels, unknowns, disagreement and classifier outcomes.
    for side in (2048,1536,1024):
        encoded=[]
        for p in paths:
            with Image.open(p) as source:
                im=ImageOps.exif_transpose(source).convert('RGB')
                im.thumbnail((side,side),Image.Resampling.LANCZOS)
                b=io.BytesIO();im.save(b,format='JPEG',quality=90,optimize=True,subsampling=2)
                encoded.append(b.getvalue())
        images=[base64.b64encode(v).decode('ascii') for v in encoded]
        size=payload_size(job,images)
        if size>MAX_BODY_BYTES:continue
        outputs=[]
        for i,v in enumerate(encoded):
            p=OUT/'evidence'/f"{job['sample_id']}_{job['kind']}_{i}_max{side}.jpg"
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.read_bytes()!=v:raise ValueError('Transport evidence changed')
            if not p.exists():p.write_bytes(v)
            outputs.append(str(p))
        return dict(paths=outputs,sha256=[sha(p) for p in outputs],mode='bounded_JPEG_transport',max_side=side,
            JPEG_quality=90,aspect_ratio_preserved=True,source_sha256=job['evidence_sha256'],serialized_body_bytes=size)
    raise ValueError('Fixed transport encoding did not meet4MiB cap; no API call')


def prepare_jobs():
    old=json.loads((original.OUT/'inputs.json').read_text())
    if old['manifest_sha256']!=original.require_data_gate():raise ValueError('Qualified manifest changed')
    jobs=[]
    for j in old['jobs']:
        job={**j,'prompt':explicit_schema_prompt(j),'original_prompt_sha256':hashlib.sha256(j['prompt'].encode()).hexdigest()}
        evidence=prepare_evidence(job)
        job.update(transport_evidence=evidence,evidence_paths=evidence['paths'],evidence_sha256=evidence['sha256'])
        jobs.append(job)
    return jobs


def validate_returned(response,job,source):
    if source=='A':
        if response.get('done_reason')!='stop':raise ValueError('generation_not_complete')
        text=response['message']['content']
    else:
        if response['choices'][0].get('finish_reason')!='stop':raise ValueError('generation_not_complete')
        text=response['choices'][0]['message']['content']
    value=unique_json_object(text);original.validate_output(value,job)
    return value


def execute_one(job,source,cfg):
    folder=OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}";folder.mkdir(parents=True,exist_ok=True)
    binding=fingerprint(dict(job=job,source=source,model=cfg['model']))
    stamp=folder/'call_started.json';raw=folder/'raw.json';meta=folder/'transport.json'
    if stamp.exists():
        if json.loads(stamp.read_text())['input_binding']!=binding:raise ValueError('Pilot call binding changed')
        if not raw.exists() or not meta.exists():return dict(valid=False,reason='started_without_complete_response',source=source)
    else:
        if len(list((OUT/'jobs').glob('*/call_started.json')))>=MAX_CALLS:raise ValueError('Pilot20-call budget exhausted')
        for p,h in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(p)!=h:raise ValueError('Transport evidence changed before call')
        body=original.request_body(job,source,cfg['model'],cfg.get('request_parameters'))
        if len(json.dumps(body,ensure_ascii=False).encode())>MAX_BODY_BYTES:raise ValueError('Actual serialized body exceeds4MiB')
        atomic_json(stamp,dict(input_binding=binding,source=source,started_unix=time.time(),one_attempt_only=True))
        try:
            status,text,elapsed,_=post_json(cfg['url'],body,{} if source=='A' else {'Authorization':'Bearer '+cfg['api_key']},900)
            if source=='B':text=text.replace(cfg['api_key'],'[REDACTED]')
            atomic_text(raw,text);atomic_json(meta,dict(status=status,elapsed_seconds=elapsed))
        except Exception as exc:
            atomic_json(folder/'error.json',dict(reason='transport',error_type=type(exc).__name__))
            return dict(valid=False,reason='transport',source=source)
    status=json.loads(meta.read_text())['status']
    if status in (401,402,403,404):raise original.AccessFailure(f'{source}_http_{status}')
    if status!=200:return dict(valid=False,reason='http_'+str(status),source=source)
    try:
        response=json.loads(raw.read_text());value=validate_returned(response,job,source)
        atomic_json(folder/'parsed.json',dict(input_binding=binding,result=value,source=source,model=cfg['model'],
            returned_model=response.get('model'),evidence_sha256=job['evidence_sha256'],response_schema_sha256=fingerprint(job['response_schema'])))
        return dict(valid=True,reason='valid',source=source)
    except Exception as exc:
        failure=dict(valid=False,reason=type(exc).__name__,source=source,path=list(getattr(exc,'absolute_path',[])),
            validator=getattr(exc,'validator',None))
        atomic_json(folder/'error.json',failure);return failure


def progress(done,results,finished=False):
    passed=sum(r['valid'] for r in results);failed=len(results)-passed
    value=dict(updated_at=datetime.now(timezone.utc).isoformat(),status=('pass' if passed==20 else 'failed') if finished else 'running',
        attempted_jobs=done,planned_jobs=20,passed_jobs=passed,failed_jobs=failed,
        VLM_calls=len(list((OUT/'jobs').glob('*/call_started.json'))),maximum_calls=20,
        full_annotation_started=False,original_v1_status='failed_preserved',human_verification=False,
        scope='Separate bounded format/transport pilot; no semantic accuracy or method efficacy claim')
    atomic_json(OUT/'progress.json',value)
    atomic_json(DATA/'pipeline_progress.json',dict(value,stage='format_transport_repair_pilot',data_qualified=True,
        packages_passed=18,source_verified_originals=5400))
    atomic_text(DATA/'当前进度.md',f"更新时间：{value['updated_at']}\n\n格式与传输修复验证：{value['status']}。\n\n"
        f"- 图片资格与独立审计通过：4800张正式样本。\n- 新验证已处理{done}/20；通过{passed}，失败{failed}；上限20次调用。\n"
        "- 上一轮13通过/7失败的结果保留，未重置其重试上限。\n- 全量标注和分类器推理尚未启动。\n")
    return value


def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original.require_data_gate()
    if (OUT/'summary.json').exists():raise ValueError('Bounded pilot already complete; do not rerun')
    diagnose();jobs=prepare_jobs()
    parent=json.loads((DATA/'box_exclusion_recovery_code_freeze_20261114.json').read_text())
    for p,h in parent.items():
        if sha(p)!=h:raise ValueError('Parent code/input changed: '+p)
    freeze(OUT/'code_freeze.json',{**parent,str(Path(__file__).resolve()):sha(__file__),
        str(ROOT/'tests/test_fresh_vlm_format_repair_20261114.py'):sha(ROOT/'tests/test_fresh_vlm_format_repair_20261114.py')})
    freeze(OUT/'inputs.json',dict(jobs=jobs,qualified_manifest_sha256=original.require_data_gate()))
    freeze(OUT/'protocol.json',dict(version='format-transport-diagnostic-pilot-v2',maximum_calls=20,attempts_per_job=1,
        previous_VLM_calls=34,previous_failures_preserved=True,no_v1_retry_reset=True,
        inputs='Same six frozen targets and native/fine semantic definitions; exact output JSON Schema appended identically for A/B',
        transport=dict(max_body_bytes=MAX_BODY_BYTES,raw_evidence_if_fits=True,
            otherwise='EXIF-oriented JPEG, aspect preserved, maxside2048/1536/1024 in fixed order, quality90/subsampling2; identical evidence bytes for A/B'),
        changes_disclosed='Large-image transport resizing changes evidence resolution; no claim of identical bytes to v1 or semantic improvement.',
        success='All20 jobs parse/validate on first and only attempt; no agreement/unknown/successful-label optimization',
        failure='Preserve failed job and stop expansion; no further repair loop',full_annotation_automatic=False,human_verification=False))
    protocol=json.loads((READY/'frozen_validation_protocol.json').read_text())['annotations']
    cfg=load_online_config(ROOT/'artifacts/feature_pilot_20260908/api_config.local.json')
    if cfg['model']!=protocol['models']['B']:raise ValueError('Unexpected online model')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:11434/api/tags',timeout=20) as r:tags=json.load(r)
    if not any(m.get('name')==protocol['models']['A'] and m.get('digest')==protocol['frozen_native_model_A_digest'] for m in tags['models']):
        raise ValueError('Frozen model unavailable')
    freeze(OUT/'runtime.json',dict(A_model=protocol['models']['A'],A_digest=protocol['frozen_native_model_A_digest'],B_model=cfg['model'],
        B_url=cfg['url'],B_request_parameters=cfg.get('request_parameters',{})))
    sources={'A':dict(model=protocol['models']['A'],url='http://127.0.0.1:11434/api/chat'),'B':cfg}
    results=[];progress(0,results)
    try:
        for job in jobs:
            for source in ('A','B'):
                result=execute_one(job,source,sources[source]);result.update(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'])
                results.append(result);progress(len(results),results)
                print(json.dumps(result,ensure_ascii=False),flush=True)
                if source=='B':time.sleep(3.1)
        summary=progress(len(results),results,finished=True)
        summary.update(job_outcomes=results,source_pass_counts={s:sum(r['valid'] for r in results if r['source']==s) for s in ('A','B')},
            original_v1_summary_sha256=sha(original.OUT/'summary.json'))
        atomic_json(OUT/'summary.json',summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        atomic_json(OUT/'fatal_error.json',dict(error_type=type(exc).__name__))
        progress(len(results),results,finished=True)
        raise


if __name__=='__main__':main()
