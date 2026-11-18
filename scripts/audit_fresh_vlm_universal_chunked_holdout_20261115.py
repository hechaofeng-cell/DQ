#!/usr/bin/env python3
"""Independently replay the universal chunked pilot and third fresh holdout."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_universal_chunked_holdout_20261115 as run
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def replay_job(job,phase,model,system,normalization):
    features={};evidence={};values={};changes=[];hashes=[];statuses=[]
    for index,fields in enumerate(run.field_chunks(job),1):
        part='part'+str(index);folder=run.folder_for(phase,job,part)
        binding=fingerprint(dict(job=job,source='B',model=model,transport='universal_chunked_strict',part=part,
            field_ids=fields,system=system,normalization=normalization))
        started=json.loads((folder/'call_started.json').read_text())
        if started['input_binding']!=binding or not started['one_attempt_only'] or started['field_ids']!=fields:
            raise ValueError('Call binding mismatch')
        if started['body_bytes']>run.MAX_BODY_BYTES:raise ValueError('Body cap exceeded')
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:raise ValueError('Evidence changed')
        raw=folder/'raw.json';meta=json.loads((folder/'transport.json').read_text());status='invalid'
        if meta['status']==200:
            try:
                response=json.loads(raw.read_text());parsed,wire,part_changes=run.parse_component(response,job,fields)
                recorded=json.loads((folder/'parsed.json').read_text())
                if recorded['input_binding']!=binding or recorded['raw_sha256']!=sha(raw) or recorded['wire']!=wire or recorded['parsed']!=parsed:
                    raise ValueError('Parsed component mismatch')
                if recorded['normalization']!=part_changes:raise ValueError('Normalization ledger mismatch')
                for change in part_changes:
                    if json.loads(change['after'])!=change['before']:raise ValueError('Normalization not reversible')
                if job['kind']=='native':features.update(parsed['features']);evidence.update(parsed['evidence'])
                else:values.update(parsed['values'])
                changes.extend(part_changes);status='valid'
            except Exception:status='invalid'
        outcome=json.loads((folder/'outcome.json').read_text())
        if outcome['status']!=status:raise ValueError('Outcome differs from replay')
        statuses.append(status);hashes.append(sha(raw))
    valid=all(x=='valid' for x in statuses)
    if valid:
        standard=(dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],target_instance_id='target_1',
            features=features,evidence=evidence) if job['kind']=='native' else values);held.validate_result(standard,job)
        merged=json.loads((run.OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B.json").read_text())
        if merged['standard']!=standard or merged['normalization']!=changes or merged['component_raw_sha256']!=hashes:
            raise ValueError('Merged output mismatch')
    return dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job.get('holdout_role'),kind=job['kind'],
        status='valid' if valid else 'invalid',component_calls=2,normalization_fields=len(changes),raw_sha256=hashes)


def main():
    frozen=json.loads((run.OUT/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen code/input changed: '+path)
    inputs=json.loads((run.OUT/'inputs.json').read_text());summary=json.loads((run.OUT/'summary.json').read_text())
    pilot=[replay_job(job,'pilot',inputs['model'],inputs['system_prompt'],inputs['normalization']) for job in inputs['pilot_jobs']]
    pilot_valid=sum(r['status']=='valid' for r in pilot);fresh=[]
    if summary['fresh_started']:
        targets=inputs['fresh_targets'];jobs=inputs['fresh_jobs']
        if len(targets)!=60 or len(jobs)!=100:raise ValueError('Fresh count mismatch')
        balance=Counter((r['class_name'],r['role']) for r in targets)
        if len(balance)!=12 or set(balance.values())!={5}:raise ValueError('Target balance mismatch')
        if {r['sample_id'] for r in targets}&set(inputs['excluded_sample_ids']):raise ValueError('Debug leakage')
        manifest={r['sample_id']:r for r in read(DATA/'qualified_manifest.csv')}
        for row in targets:
            source=manifest[row['sample_id']]
            if any(str(source[k])!=str(row[k]) for k in ('class_name','role','rank','image_sha256','crop_sha256')):
                raise ValueError('Target binding mismatch')
        fresh=[replay_job(job,'fresh',inputs['model'],inputs['system_prompt'],inputs['normalization']) for job in jobs]
    calls=len(list(run.OUT.glob('*/components/*/call_started.json')));valid=sum(r['status']=='valid' for r in fresh)
    expected=4+(200 if summary['fresh_started'] else 0)
    if calls!=expected:raise ValueError('Call count mismatch')
    success=pilot_valid==2 and summary['fresh_started'] and valid==100
    breakdown=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            subset=[r for r in fresh if r['class_name']==class_name and r['role']==role]
            breakdown.append(dict(class_name=class_name,role=role,jobs=len(subset),valid=sum(r['status']=='valid' for r in subset),
                invalid=sum(r['status']=='invalid' for r in subset),normalization_fields=sum(r['normalization_fields'] for r in subset)))
    audit=dict(status='pass',model_calls=calls,pilot_valid=pilot_valid,pilot_invalid=2-pilot_valid,
        fresh_started=summary['fresh_started'],targets=60 if summary['fresh_started'] else 0,semantic_jobs=len(fresh),
        valid=valid,invalid=len(fresh)-valid,normalization_fields=sum(r['normalization_fields'] for r in fresh),
        success_criterion_met=success,one_attempt_per_component=True,all_normalizations_reversible=True,
        semantic_values_rewritten_locally=False,deterministic_reconstruction=True,debug_target_overlap=0,
        full_annotation_started=False,target_classifier_calls=0,pilot_records=pilot,fresh_records=fresh,breakdown=breakdown,
        boundary='B structural transport reliability only; no semantic accuracy or human truth claim.')
    atomic_json(run.OUT/'independent_audit.json',audit)
    lines=['# B 统一分块严格传输最终验证（2026-11-15）','',
        f'已知horse/person细粒度失败兼容性试验：通过 {pilot_valid}/2。第三批全新留出验证：通过 {valid}/{len(fresh) if fresh else 100}。预设条件：{"达到" if success else "未达到"}。','',
        '原生和细粒度任务均确定性拆成两个互不重叠且各不超过10字段的严格schema组件；原生证据字符串数组仅允许可逆紧凑JSON序列化。审计重放所有原始响应并按原标准schema复验。','',
        f'实际模型调用 {calls} 次，每个组件一次，无重试。完整标注、选样和分类器推理均未启动。','',
        '该结果只验证B的结构与传输可靠性，不提供人工真值准确率或方法有效性证据。']
    atomic_text(run.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/双VLM统一分块严格传输最终验证_20261115.md','\n'.join(lines)+'\n')
    stage='universal_chunked_fresh_holdout_pass' if success else 'universal_chunked_fresh_holdout_failed'
    atomic_json(DATA/'pipeline_progress.json',dict(stage=stage,data_qualified=True,model_calls=calls,semantic_jobs=len(fresh),
        valid=valid,invalid=len(fresh)-valid,normalization_fields=audit['normalization_fields'],full_annotation_started=False,
        target_classifier_calls=0,human_verification=False))
    atomic_text(DATA/'当前进度.md',f'B统一分块严格传输最终验证已完成。\n\n兼容性试验 {pilot_valid}/2；全新留出验证 {valid}/{len(fresh) if fresh else 100}。\n\n完整标注尚未启动。\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('pilot_records','fresh_records','breakdown')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
