#!/usr/bin/env python3
"""Independently replay the final fourth B holdout and verify its provenance."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_final_holdout_20261115 as run
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def verify_change(change):
    operation=change.get('operation')
    if operation=='JSON_array_to_JSON_string':
        if json.loads(change['after'])!=change['before']:
            raise ValueError('Array normalization is not reversible')
    elif operation=='drop_transport_self_reference':
        if (change.get('path')!='/evidence/evidence_json' or not isinstance(change.get('before'),str)
                or change.get('after','missing') is not None):
            raise ValueError('Invalid self-reference cleanup ledger')
    else:
        raise ValueError('Unsupported normalization operation')


def replay_job(job,model,system,normalization):
    features={};evidence={};values={};changes=[];hashes=[];statuses=[]
    for index,fields in enumerate(run.transport.field_chunks(job),1):
        part='part'+str(index);folder=run.transport.folder_for('fresh',job,part)
        binding=fingerprint(dict(job=job,source='B',model=model,transport='universal_chunked_strict',part=part,
            field_ids=fields,system=system,normalization=normalization))
        started=json.loads((folder/'call_started.json').read_text())
        if (started['input_binding']!=binding or not started['one_attempt_only'] or started['field_ids']!=fields
                or started['part']!=part):
            raise ValueError('Call binding mismatch')
        if started['body_bytes']>run.MAX_BODY_BYTES:
            raise ValueError('Body cap exceeded')
        for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=digest:
                raise ValueError('Evidence changed')
        raw=folder/'raw.json';meta=json.loads((folder/'transport.json').read_text());status='invalid'
        if meta['status']==200:
            try:
                response=json.loads(raw.read_text());parsed,wire,part_changes=run.parse_component(response,job,fields)
                recorded=json.loads((folder/'parsed.json').read_text())
                if (recorded['input_binding']!=binding or recorded['raw_sha256']!=sha(raw)
                        or recorded['wire']!=wire or recorded['parsed']!=parsed
                        or recorded['normalization']!=part_changes):
                    raise ValueError('Parsed component mismatch')
                for change in part_changes:
                    verify_change(change)
                if job['kind']=='native':
                    features.update(parsed['features']);evidence.update(parsed['evidence'])
                else:
                    values.update(parsed['values'])
                changes.extend(part_changes);status='valid'
            except Exception:
                status='invalid'
        outcome=json.loads((folder/'outcome.json').read_text())
        if outcome['status']!=status:
            raise ValueError('Outcome differs from replay')
        statuses.append(status);hashes.append(sha(raw))
    valid=all(value=='valid' for value in statuses)
    if valid:
        standard=(dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
            target_instance_id='target_1',features=features,evidence=evidence) if job['kind']=='native' else values)
        held.validate_result(standard,job)
        merged=json.loads((run.OUT/'fresh/jobs'/f"{job['sample_id']}_{job['kind']}_B.json").read_text())
        if (merged['standard']!=standard or merged['normalization']!=changes
                or merged['component_raw_sha256']!=hashes):
            raise ValueError('Merged output mismatch')
    return dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job['holdout_role'],kind=job['kind'],
        status='valid' if valid else 'invalid',component_calls=2,
        normalization_fields=len(changes),array_serializations=sum(c['operation']=='JSON_array_to_JSON_string' for c in changes),
        self_reference_drops=sum(c['operation']=='drop_transport_self_reference' for c in changes),raw_sha256=hashes)


def main():
    frozen=json.loads((run.OUT/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:
            raise ValueError('Frozen code/input changed: '+path)
    offline=json.loads((run.OUT/'offline_parent_audit.json').read_text())
    if (offline['status']!='pass' or offline['valid']!=100 or offline['invalid']!=0
            or offline['self_reference_drops']!=1):
        raise ValueError('Offline parent gate mismatch')
    inputs=json.loads((run.OUT/'inputs.json').read_text());summary=json.loads((run.OUT/'summary.json').read_text())
    targets=inputs['fresh_targets'];jobs=inputs['fresh_jobs']
    if len(targets)!=60 or len(jobs)!=100 or inputs['model_calls']!=200:
        raise ValueError('Fresh count mismatch')
    balance=Counter((row['class_name'],row['role']) for row in targets)
    if len(balance)!=12 or set(balance.values())!={5}:
        raise ValueError('Target balance mismatch')
    overlap={row['sample_id'] for row in targets}&set(inputs['excluded_sample_ids'])
    if overlap:
        raise ValueError('Prior target leakage')
    manifest={row['sample_id']:row for row in read(DATA/'qualified_manifest.csv')}
    for row in targets:
        source=manifest[row['sample_id']]
        if any(str(source[key])!=str(row[key]) for key in ('class_name','role','rank','image_sha256','crop_sha256')):
            raise ValueError('Target binding mismatch')
    records=[replay_job(job,inputs['model'],inputs['system_prompt'],inputs['normalization']) for job in jobs]
    calls=len(list(run.OUT.glob('*/components/*/call_started.json')))
    if calls!=200:
        raise ValueError('Call count mismatch')
    valid=sum(record['status']=='valid' for record in records);invalid=len(records)-valid
    if (summary['fresh']['processed_jobs']!=100 or summary['fresh']['valid_jobs']!=valid
            or summary['fresh']['invalid_jobs']!=invalid or summary['total_model_calls']!=calls):
        raise ValueError('Summary differs from replay')
    array_serializations=sum(record['array_serializations'] for record in records)
    self_reference_drops=sum(record['self_reference_drops'] for record in records)
    success=valid==100 and invalid==0
    breakdown=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            subset=[record for record in records if record['class_name']==class_name and record['role']==role]
            breakdown.append(dict(class_name=class_name,role=role,jobs=len(subset),
                valid=sum(record['status']=='valid' for record in subset),
                invalid=sum(record['status']=='invalid' for record in subset),
                array_serializations=sum(record['array_serializations'] for record in subset),
                self_reference_drops=sum(record['self_reference_drops'] for record in subset)))
    audit=dict(status='pass',model_calls=calls,targets=60,semantic_jobs=100,valid=valid,invalid=invalid,
        normalization_fields=sum(record['normalization_fields'] for record in records),
        array_serializations=array_serializations,self_reference_drops=self_reference_drops,
        success_criterion_met=success,one_attempt_per_component=True,all_normalizations_reversible=True,
        deterministic_reconstruction=True,prior_target_overlap=len(overlap),full_annotation_started=False,
        target_classifier_calls=0,records=records,breakdown=breakdown,
        boundary='B structural transport reliability only; no semantic accuracy or human truth claim.')
    atomic_json(run.OUT/'independent_audit.json',audit)
    lines=['# B 最终独立留出验证（2026-11-15）','',
        f'上一轮100项离线重放：100/100，命中1次唯一自引用清理。第四批全新留出验证：通过 {valid}/100。预设成功条件：{"达到" if success else "未达到"}。','',
        '60张图像与所有先前调试、兼容性试验和留出集完全不重叠；每类、每角色各5张，共100个语义任务。每个任务固定拆成两个不超过10字段的严格schema组件。','',
        f'实际模型调用 {calls} 次，每个组件仅调用一次，无重试。可逆数组序列化 {array_serializations} 次；额外evidence_json自引用清理 {self_reference_drops} 次。','',
        '独立审计重放了全部原始响应，核对调用绑定、输入证据哈希、解析结果、规范化台账和确定性重建。完整标注与分类器推理均未启动。','',
        '该结果只验证B模型的结构化传输可靠性，不代表语义标注正确率，也不包含人工真值核验。']
    atomic_text(run.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/双VLM最终独立留出验证_20261115.md','\n'.join(lines)+'\n')
    stage='final_fresh_holdout_pass' if success else 'final_fresh_holdout_failed'
    atomic_json(DATA/'pipeline_progress.json',dict(stage=stage,data_qualified=True,model_calls=calls,
        semantic_jobs=100,valid=valid,invalid=invalid,array_serializations=array_serializations,
        self_reference_drops=self_reference_drops,full_annotation_started=False,target_classifier_calls=0,
        human_verification=False))
    atomic_text(DATA/'当前进度.md',f'B最终独立留出验证已完成。\n\n第四批全新留出验证 {valid}/100；预设成功条件{"已达到" if success else "未达到"}。\n\n完整标注尚未启动。\n')
    print(json.dumps({key:value for key,value in audit.items() if key not in ('records','breakdown')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
