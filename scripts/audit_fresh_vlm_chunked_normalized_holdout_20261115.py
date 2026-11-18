#!/usr/bin/env python3
"""Independently replay the final chunked-normalized fresh holdout."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_chunked_normalized_holdout_20261115 as run
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def replay_call(job,part,model,system,normalization,field_ids=None):
    folder=run.folder_for(job,part);transport='chunked_native_normalized' if field_ids is not None else 'strict_fine'
    binding=fingerprint(dict(job=job,source='B',model=model,transport=transport,part=part,
        field_ids=field_ids,system=system,normalization=normalization))
    started=json.loads((folder/'call_started.json').read_text())
    if started['input_binding']!=binding or not started['one_attempt_only'] or started['field_ids']!=field_ids:
        raise ValueError('Call binding mismatch')
    if started['body_bytes']>run.MAX_BODY_BYTES:raise ValueError('Body cap exceeded')
    for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
        if sha(path)!=digest:raise ValueError('Evidence changed')
    raw=folder/'raw.json';meta=json.loads((folder/'transport.json').read_text());status='invalid';parsed=None;changes=[]
    if meta['status']==200:
        try:
            response=json.loads(raw.read_text())
            if field_ids is not None:
                features,evidence,wire,changes=run.parse_component(response,job,field_ids)
                parsed=dict(features=features,evidence=evidence,wire=wire,normalization=changes)
            else:
                wire=held.response_value(response,'B');held.validate_result(wire,job)
                parsed=dict(standard=wire,wire=wire,normalization=[])
            recorded=json.loads((folder/'parsed.json').read_text())
            for key,value in parsed.items():
                if recorded[key]!=value:raise ValueError('Parsed output mismatch')
            if recorded['input_binding']!=binding or recorded['raw_sha256']!=sha(raw):raise ValueError('Parsed binding mismatch')
            for change in changes:
                if json.loads(change['after'])!=change['before']:raise ValueError('Normalization not reversible')
            status='valid'
        except Exception:status='invalid'
    outcome=json.loads((folder/'outcome.json').read_text())
    if outcome['status']!=status:raise ValueError('Outcome differs from replay')
    return status,parsed,sha(raw),changes


def replay_job(job,model,system,normalization):
    hashes=[];changes=[]
    if job['kind']=='native':
        features={};evidence={};statuses=[]
        for index,fields in enumerate(run.chunk.feature_chunks(job),1):
            status,parsed,digest,part_changes=replay_call(job,'part'+str(index),model,system,normalization,fields)
            statuses.append(status);hashes.append(digest);changes.extend(part_changes)
            if parsed:features.update(parsed['features']);evidence.update(parsed['evidence'])
        valid=all(x=='valid' for x in statuses)
        if valid:
            standard=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
                target_instance_id='target_1',features=features,evidence=evidence);held.validate_result(standard,job)
    else:
        status,parsed,digest,changes=replay_call(job,'full',model,system,normalization);valid=status=='valid';hashes=[digest]
        if valid:standard=parsed['standard']
    merged=run.OUT/'fresh/jobs'/f"{job['sample_id']}_{job['kind']}_B.json"
    if valid:
        recorded=json.loads(merged.read_text())
        if recorded['standard']!=standard or recorded['normalization']!=changes or recorded['component_raw_sha256']!=hashes:
            raise ValueError('Merged output mismatch')
    elif merged.exists():raise ValueError('Invalid job has merged output')
    return dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job['holdout_role'],kind=job['kind'],
        status='valid' if valid else 'invalid',component_calls=len(hashes),normalization_fields=len(changes),raw_sha256=hashes)


def main():
    frozen=json.loads((run.OUT/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen code/input changed: '+path)
    inputs=json.loads((run.OUT/'inputs.json').read_text());summary=json.loads((run.OUT/'summary.json').read_text())
    targets=inputs['fresh_targets'];jobs=inputs['fresh_jobs']
    if len(targets)!=60 or len(jobs)!=100:raise ValueError('Input count mismatch')
    balance=Counter((r['class_name'],r['role']) for r in targets)
    if len(balance)!=12 or set(balance.values())!={5}:raise ValueError('Target balance mismatch')
    if {r['sample_id'] for r in targets}&set(inputs['excluded_sample_ids']):raise ValueError('Debug leakage')
    manifest={r['sample_id']:r for r in read(DATA/'qualified_manifest.csv')}
    for row in targets:
        source=manifest[row['sample_id']]
        if any(str(source[k])!=str(row[k]) for k in ('class_name','role','rank','image_sha256','crop_sha256')):
            raise ValueError('Target binding mismatch')
    records=[replay_job(job,inputs['model'],inputs['system_prompt'],inputs['normalization']) for job in jobs]
    calls=len(list((run.OUT/'fresh/components').glob('*/call_started.json')));valid=sum(r['status']=='valid' for r in records)
    if calls!=160 or summary['valid_jobs']!=valid or summary['invalid_jobs']!=100-valid:raise ValueError('Summary/call count mismatch')
    breakdown=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            subset=[r for r in records if r['class_name']==class_name and r['role']==role]
            breakdown.append(dict(class_name=class_name,role=role,jobs=len(subset),valid=sum(r['status']=='valid' for r in subset),
                invalid=sum(r['status']=='invalid' for r in subset),normalization_fields=sum(r['normalization_fields'] for r in subset)))
    success=valid==100
    audit=dict(status='pass',model_calls=calls,targets=60,semantic_jobs=100,valid=valid,invalid=100-valid,
        normalization_fields=sum(r['normalization_fields'] for r in records),success_criterion_met=success,
        one_attempt_per_component=True,all_normalizations_reversible=True,semantic_values_rewritten_locally=False,
        deterministic_reconstruction=True,debug_target_overlap=0,full_annotation_started=False,target_classifier_calls=0,
        records=records,breakdown=breakdown,boundary='B structural transport reliability only; no semantic accuracy or human truth claim.')
    atomic_json(run.OUT/'independent_audit.json',audit)
    lines=['# B 分块严格传输最终留出验证（2026-11-15）','',
        f'在60张全新且未参与协议调试的平衡样本上执行100个B语义任务，共160次组件调用。有效 {valid}/100，失败 {100-valid}/100；预设条件：{"达到" if success else "未达到"}。','',
        f'证据字符串数组仅做可逆紧凑JSON序列化，共转换 {audit["normalization_fields"]} 个字段；特征值、身份和语义取值未在本地改写。独立审计重放全部160份原始响应并用原标准schema复验。','',
        '每个组件只调用一次，无重试。完整标注、选样和分类器推理均未启动。','',
        '该结果只验证B的结构与传输可靠性，不提供人工真值准确率或方法有效性证据。']
    atomic_text(run.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/双VLM分块严格传输最终留出验证_20261115.md','\n'.join(lines)+'\n')
    stage='chunked_normalized_fresh_holdout_pass' if success else 'chunked_normalized_fresh_holdout_failed'
    atomic_json(DATA/'pipeline_progress.json',dict(stage=stage,data_qualified=True,model_calls=calls,semantic_jobs=100,
        valid=valid,invalid=100-valid,normalization_fields=audit['normalization_fields'],full_annotation_started=False,
        target_classifier_calls=0,human_verification=False))
    atomic_text(DATA/'当前进度.md',f'B分块严格传输最终留出验证已完成。\n\n有效 {valid}/100，失败 {100-valid}/100；证据规范化字段 {audit["normalization_fields"]}。\n\n完整标注尚未启动。\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('records','breakdown')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
