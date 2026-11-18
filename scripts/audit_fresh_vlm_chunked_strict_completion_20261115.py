#!/usr/bin/env python3
"""Independently replay chunked strict native and direct strict fine outputs."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_chunked_strict_completion_20261115 as run
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def replay_call(job,phase,part,model,system,field_ids=None):
    folder=run.folder_for(phase,job,part);transport='chunked_native' if field_ids is not None else 'strict_fine'
    binding=fingerprint(dict(job=job,source='B',model=model,transport=transport,part=part,field_ids=field_ids,system=system))
    started=json.loads((folder/'call_started.json').read_text())
    if started['input_binding']!=binding or not started['one_attempt_only'] or started['field_ids']!=field_ids:
        raise ValueError('Component binding mismatch')
    if started['body_bytes']>run.MAX_BODY_BYTES:raise ValueError('Body cap exceeded')
    for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
        if sha(path)!=digest:raise ValueError('Evidence changed')
    raw=folder/'raw.json';meta=json.loads((folder/'transport.json').read_text());status='invalid';parsed=None
    if meta['status']==200:
        try:
            response=json.loads(raw.read_text())
            if field_ids is not None:
                features,evidence,wire=run.parse_component(response,job,field_ids)
                parsed=dict(features=features,evidence=evidence,wire=wire)
            else:
                wire=held.response_value(response,'B');held.validate_result(wire,job);parsed=dict(standard=wire,wire=wire)
            recorded=json.loads((folder/'parsed.json').read_text())
            for key,value in parsed.items():
                if recorded[key]!=value:raise ValueError('Parsed component differs from raw replay')
            if recorded['input_binding']!=binding or recorded['raw_sha256']!=sha(raw):raise ValueError('Parsed binding mismatch')
            status='valid'
        except Exception:status='invalid'
    outcome=json.loads((folder/'outcome.json').read_text())
    if outcome['status']!=status:raise ValueError('Outcome differs from replay')
    return status,parsed,sha(raw)


def replay_job(job,phase,model,system):
    raw_hashes=[]
    if job['kind']=='native':
        features={};evidence={};statuses=[]
        for index,fields in enumerate(run.feature_chunks(job),1):
            status,parsed,digest=replay_call(job,phase,'part'+str(index),model,system,fields)
            statuses.append(status);raw_hashes.append(digest)
            if parsed:features.update(parsed['features']);evidence.update(parsed['evidence'])
        valid=all(s=='valid' for s in statuses)
        if valid:
            standard=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
                target_instance_id='target_1',features=features,evidence=evidence);held.validate_result(standard,job)
    else:
        status,parsed,digest=replay_call(job,phase,'full',model,system);valid=status=='valid';raw_hashes=[digest]
        if valid:standard=parsed['standard']
    job_path=run.OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B.json"
    if valid:
        recorded=json.loads(job_path.read_text())
        if recorded['standard']!=standard or recorded['component_raw_sha256']!=raw_hashes:raise ValueError('Merged job mismatch')
    elif job_path.exists():raise ValueError('Invalid job has merged output')
    return dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job.get('holdout_role'),
        kind=job['kind'],status='valid' if valid else 'invalid',component_calls=len(raw_hashes),raw_sha256=raw_hashes)


def main():
    frozen=json.loads((run.OUT/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen code/input changed: '+path)
    inputs=json.loads((run.OUT/'inputs.json').read_text());summary=json.loads((run.OUT/'summary.json').read_text())
    model=inputs['model'];system=inputs['system_prompt']
    pilot=[replay_job(job,'pilot',model,system) for job in inputs['pilot_jobs']];pilot_valid=sum(r['status']=='valid' for r in pilot)
    fresh=[]
    if summary['fresh_started']:
        targets=inputs['fresh_targets'];jobs=inputs['fresh_jobs']
        if len(targets)!=60 or len(jobs)!=100:raise ValueError('Fresh input count mismatch')
        balance=Counter((r['class_name'],r['role']) for r in targets)
        if len(balance)!=12 or set(balance.values())!={5}:raise ValueError('Target balance mismatch')
        if {r['sample_id'] for r in targets}&set(inputs['excluded_sample_ids']):raise ValueError('Debug target leakage')
        manifest={r['sample_id']:r for r in read(DATA/'qualified_manifest.csv')}
        for row in targets:
            source=manifest[row['sample_id']]
            if any(str(source[k])!=str(row[k]) for k in ('class_name','role','rank','image_sha256','crop_sha256')):
                raise ValueError('Target binding mismatch')
        fresh=[replay_job(job,'fresh',model,system) for job in jobs]
    calls=len(list(run.OUT.glob('*/components/*/call_started.json')));fresh_valid=sum(r['status']=='valid' for r in fresh)
    expected=4+(160 if summary['fresh_started'] else 0)
    if calls!=expected:raise ValueError('Call count mismatch')
    success=pilot_valid==2 and summary['fresh_started'] and fresh_valid==100
    breakdown=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            subset=[r for r in fresh if r['class_name']==class_name and r['role']==role]
            breakdown.append(dict(class_name=class_name,role=role,jobs=len(subset),
                valid=sum(r['status']=='valid' for r in subset),invalid=sum(r['status']=='invalid' for r in subset)))
    audit=dict(status='pass',model_calls=calls,pilot_jobs=2,pilot_valid=pilot_valid,pilot_invalid=2-pilot_valid,
        fresh_started=summary['fresh_started'],fresh_jobs=len(fresh),fresh_valid=fresh_valid,fresh_invalid=len(fresh)-fresh_valid,
        success_criterion_met=success,one_attempt_per_component=True,deterministic_reconstruction=True,
        semantic_values_rewritten_locally=False,debug_target_overlap=0,full_annotation_started=False,
        target_classifier_calls=0,pilot_records=pilot,fresh_records=fresh,breakdown=breakdown,
        boundary='B structure/transport reliability only; no human truth or semantic accuracy claim.')
    atomic_json(run.OUT/'independent_audit.json',audit)
    lines=['# B 分块严格传输与全新留出验证（2026-11-15）','',
        f'两例原生兼容性试验：通过 {pilot_valid}/2。全新留出验证：通过 {fresh_valid}/{len(fresh) if fresh else 100}。预设条件：{"达到" if success else "未达到"}。','',
        '原生任务的特征被确定性拆成两个不重叠且各不超过10项的严格schema组件；每个组件携带同字段证据JSON字符串。审计重放原始响应后合并组件，并用原标准schema复验。细粒度任务直接使用原严格schema。','',
        f'实际模型调用 {calls} 次，每个组件一次，无重试。完整标注、选样和分类器推理均未启动。','',
        '该结果只验证B的结构与传输可靠性，不提供人工真值准确率或方法有效性证据。']
    atomic_text(run.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/双VLM分块严格传输验证_20261115.md','\n'.join(lines)+'\n')
    stage='chunked_strict_fresh_holdout_pass' if success else 'chunked_strict_validation_failed'
    atomic_json(DATA/'pipeline_progress.json',dict(stage=stage,data_qualified=True,model_calls=calls,pilot_valid=pilot_valid,
        fresh_valid=fresh_valid,fresh_invalid=len(fresh)-fresh_valid,full_annotation_started=False,
        target_classifier_calls=0,human_verification=False))
    atomic_text(DATA/'当前进度.md',f'B分块严格传输验证已完成。\n\n兼容性试验 {pilot_valid}/2；全新留出验证 {fresh_valid}/{len(fresh) if fresh else 100}。\n\n完整标注尚未启动。\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('pilot_records','fresh_records','breakdown')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
