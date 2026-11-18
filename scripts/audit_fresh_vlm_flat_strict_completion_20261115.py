#!/usr/bin/env python3
"""Independently replay the flat strict pilot and fresh B-only holdout."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_flat_strict_completion_20261115 as run
import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import OUT as DATA,read,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def replay(job,phase,model,system):
    transport='flat_native' if job['kind']=='native' else 'strict_fine'
    folder=run.OUT/phase/'jobs'/f"{job['sample_id']}_{job['kind']}_B"
    started=json.loads((folder/'call_started.json').read_text())
    binding=fingerprint(dict(job=job,source='B',model=model,transport=transport,system=system))
    if started['input_binding']!=binding or not started['one_attempt_only'] or started['transport']!=transport:
        raise ValueError('Call binding/attempt mismatch')
    if started['body_bytes']>run.MAX_BODY_BYTES:raise ValueError('Body cap exceeded')
    for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
        if sha(path)!=digest:raise ValueError('Evidence changed')
    raw=folder/'raw.json';meta=json.loads((folder/'transport.json').read_text())
    status='invalid';reason='http_'+str(meta['status'])
    if meta['status']==200:
        try:
            response=json.loads(raw.read_text());wire,standard=run.parse_response(response,job,transport)
            parsed_wire=json.loads((folder/'parsed_wire.json').read_text());parsed=json.loads((folder/'parsed_standard.json').read_text())
            if parsed_wire['input_binding']!=binding or parsed_wire['raw_sha256']!=sha(raw) or parsed_wire['result']!=wire:
                raise ValueError('Wire parse mismatch')
            if parsed['input_binding']!=binding or parsed['raw_sha256']!=sha(raw) or parsed['result']!=standard:
                raise ValueError('Standard reconstruction mismatch')
            status='valid';reason=None
        except Exception as exc:reason=type(exc).__name__
    recorded=json.loads((folder/'outcome.json').read_text())
    if recorded['status']!=status:raise ValueError('Recorded/replayed outcome mismatch')
    return dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],
        role=job.get('holdout_role'),transport=transport,status=status,reason=reason,raw_sha256=sha(raw))


def main():
    frozen=json.loads((run.OUT/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen code/input changed: '+path)
    inputs=json.loads((run.OUT/'inputs.json').read_text());summary=json.loads((run.OUT/'summary.json').read_text())
    model=inputs['model'];system=inputs['system_prompt']
    pilot=[replay(job,'pilot',model,system) for job in inputs['pilot_jobs']]
    pilot_valid=sum(r['status']=='valid' for r in pilot)
    fresh=[]
    if summary['fresh_started']:
        targets=inputs['fresh_targets'];jobs=inputs['fresh_jobs']
        if len(targets)!=60 or len(jobs)!=100:raise ValueError('Fresh input count mismatch')
        counts=Counter((r['class_name'],r['role']) for r in targets)
        if len(counts)!=12 or set(counts.values())!={5}:raise ValueError('Fresh target balance mismatch')
        if {r['sample_id'] for r in targets}&set(inputs['excluded_sample_ids']):raise ValueError('Debug leakage')
        manifest={r['sample_id']:r for r in read(DATA/'qualified_manifest.csv')}
        for row in targets:
            source=manifest[row['sample_id']]
            if any(str(source[k])!=str(row[k]) for k in ('class_name','role','rank','image_sha256','crop_sha256')):
                raise ValueError('Target binding mismatch')
        fresh=[replay(job,'fresh',model,system) for job in jobs]
    total_calls=len(list(run.OUT.glob('*/jobs/*/call_started.json')));fresh_valid=sum(r['status']=='valid' for r in fresh)
    expected=2+(100 if summary['fresh_started'] else 0)
    if total_calls!=expected:raise ValueError('Actual call count mismatch')
    success=pilot_valid==2 and summary['fresh_started'] and fresh_valid==100
    breakdown=[]
    for class_name in held.CLASSES:
        for role in held.ROLES:
            subset=[r for r in fresh if r['class_name']==class_name and r['role']==role]
            breakdown.append(dict(class_name=class_name,role=role,jobs=len(subset),
                valid=sum(r['status']=='valid' for r in subset),invalid=sum(r['status']=='invalid' for r in subset)))
    audit=dict(status='pass',model_calls=total_calls,pilot_valid=pilot_valid,pilot_invalid=2-pilot_valid,
        fresh_started=summary['fresh_started'],fresh_jobs=len(fresh),fresh_valid=fresh_valid,
        fresh_invalid=len(fresh)-fresh_valid,success_criterion_met=success,one_attempt_each=True,
        local_semantic_rewrite=False,deterministic_native_reconstruction=True,debug_target_overlap=0,
        full_annotation_started=False,target_classifier_calls=0,pilot_records=pilot,fresh_records=fresh,breakdown=breakdown,
        boundary='B structural transport reliability only; no human truth and no semantic accuracy claim.')
    atomic_json(run.OUT/'independent_audit.json',audit)
    result='达到' if success else '未达到'
    lines=['# B 扁平严格传输与全新留出验证（2026-11-15）','',
        f'两例兼容性试验：通过 {pilot_valid}/2。全新留出验证：通过 {fresh_valid}/{len(fresh) if fresh else 100}。预设条件：{result}。','',
        '原生任务使用扁平严格schema传输特征，并将证据作为evidence_json字符串返回；审计通过json.loads、重复键拒绝、精确键集合与原标准schema完成确定性重建。细粒度任务直接使用原严格schema。','',
        f'实际模型调用 {total_calls} 次，每个任务一次，无重试。完整标注、选样和分类器推理均未启动。','',
        '该验证只支持结构与传输可靠性结论，不提供人工真值准确率或方法有效性证据。']
    atomic_text(run.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/双VLM扁平严格传输验证_20261115.md','\n'.join(lines)+'\n')
    stage='flat_strict_fresh_holdout_pass' if success else 'flat_strict_validation_failed'
    atomic_json(DATA/'pipeline_progress.json',dict(stage=stage,data_qualified=True,model_calls=total_calls,
        pilot_valid=pilot_valid,fresh_valid=fresh_valid,fresh_invalid=len(fresh)-fresh_valid,
        full_annotation_started=False,target_classifier_calls=0,human_verification=False))
    atomic_text(DATA/'当前进度.md',f'B扁平严格传输验证已完成。\n\n兼容性试验 {pilot_valid}/2；全新留出验证 {fresh_valid}/{len(fresh) if fresh else 100}。\n\n完整标注尚未启动。\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('pilot_records','fresh_records','breakdown')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
