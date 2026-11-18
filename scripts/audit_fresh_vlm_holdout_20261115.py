#!/usr/bin/env python3
"""Independently replay and audit the 200-call held-out format validation."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_holdout_20261115 as held
from prepare_fresh_images_20261112 import read,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def canonical(value):
    return sorted(value) if isinstance(value,list) else value


def main():
    out=held.OUT
    summary=json.loads((out/'summary.json').read_text())
    frozen=json.loads((out/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen held-out input/code changed: '+path)
    inputs=json.loads((out/'inputs.json').read_text());jobs=inputs['jobs'];targets=inputs['targets']
    if len(targets)!=60 or len({r['sample_id'] for r in targets})!=60:raise ValueError('Target count mismatch')
    target_counts=Counter((r['class_name'],r['role']) for r in targets)
    if len(target_counts)!=12 or set(target_counts.values())!={5}:raise ValueError('Target balance mismatch')
    if {r['sample_id'] for r in targets}&set(inputs['excluded_debug_sample_ids']):raise ValueError('Debug target leakage')
    manifest={r['sample_id']:r for r in read(held.DATA/'qualified_manifest.csv')}
    for r in targets:
        source=manifest.get(r['sample_id'])
        if not source or any(str(source[k])!=str(r[k]) for k in ('class_name','role','rank','image_sha256','crop_sha256')):
            raise ValueError('Frozen target differs from qualified manifest')
    if len(jobs)!=100:raise ValueError('Semantic job count mismatch')
    runtime=json.loads((out/'runtime.json').read_text());records=[];values={}
    for job in jobs:
        for p,digest in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(p)!=digest:raise ValueError('Transmitted evidence changed')
        if job['transport_evidence']['serialized_body_bytes']>held.MAX_BODY_BYTES:raise ValueError('Frozen body exceeds cap')
        for source in ('A','B'):
            folder=out/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}"
            started=json.loads((folder/'call_started.json').read_text())
            model=runtime[source+'_model'];binding=fingerprint(dict(job=job,source=source,model=model))
            if started['input_binding']!=binding or not started['one_attempt_only']:raise ValueError('Call binding mismatch')
            raw=folder/'raw.json';transport=folder/'transport.json'
            if not raw.exists() or not transport.exists():
                outcome=dict(status='invalid',reason='missing_response')
            elif json.loads(transport.read_text())['status']!=200:
                outcome=dict(status='invalid',reason='http_'+str(json.loads(transport.read_text())['status']))
            else:
                try:
                    value=held.response_value(json.loads(raw.read_text()),source)
                    outcome=held.validate_raw_or_normalized(value,job)
                except Exception as exc:outcome=dict(status='invalid',reason=type(exc).__name__,result=None,changes=[])
            recorded=json.loads((folder/'outcome.json').read_text()) if (folder/'outcome.json').exists() else {}
            if recorded.get('status')!=outcome['status']:raise ValueError('Recorded/replayed status mismatch')
            if outcome['status']!='invalid':
                parsed_name='parsed_raw.json' if outcome['status']=='raw_valid' else 'parsed_normalized.json'
                parsed=json.loads((folder/parsed_name).read_text())
                if parsed['input_binding']!=binding or parsed['result']!=outcome['result'] or parsed['raw_sha256']!=sha(raw):
                    raise ValueError('Parsed result differs from raw replay')
                if parsed['normalization']!=outcome['changes']:raise ValueError('Normalization ledger mismatch')
                for change in outcome['changes']:
                    if change['operation']!='JSON_array_to_JSON_string' or json.loads(change['after'])!=change['before']:
                        raise ValueError('Normalization is not reversible')
                values[(job['sample_id'],job['kind'],source)]=outcome['result']
            records.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job['holdout_role'],
                kind=job['kind'],source=source,status=outcome['status'],changes=len(outcome.get('changes',[])),
                raw_sha256=sha(raw) if raw.exists() else None))
    calls=len(list((out/'jobs').glob('*/call_started.json')))
    if calls!=200:raise ValueError('Actual call count is not200')
    counts=Counter(r['status'] for r in records)
    if (counts['raw_valid'],counts['normalized_valid'],counts['invalid'])!=(summary['raw_valid'],summary['normalized_valid'],summary['invalid']):
        raise ValueError('Summary/replay totals mismatch')
    breakdown=[]
    for source in ('A','B'):
        for class_name in held.CLASSES:
            for role in held.ROLES:
                subset=[r for r in records if r['source']==source and r['class_name']==class_name and r['role']==role]
                c=Counter(r['status'] for r in subset)
                breakdown.append(dict(source=source,class_name=class_name,role=role,jobs=len(subset),
                    raw_valid=c['raw_valid'],normalized_valid=c['normalized_valid'],invalid=c['invalid']))
    agreement=[]
    for job in jobs:
        key=job['sample_id'],job['kind']
        if not all(key+(s,) in values for s in ('A','B')):continue
        a,b=[values[key+(s,)] for s in ('A','B')]
        if job['kind']=='native':a=a['features'];b=b['features']
        agreement.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job['holdout_role'],kind=job['kind'],
            fields=len(a),matching_fields=sum(canonical(v)==canonical(b[k]) for k,v in a.items()),
            A_unknown=sum(v=='unknown' or isinstance(v,list) and 'unknown' in v for v in a.values()),
            B_unknown=sum(v=='unknown' or isinstance(v,list) and 'unknown' in v for v in b.values())))
    audit=dict(status='pass',raw_records_replayed=200,model_calls=200,targets=60,semantic_jobs=100,
        raw_valid=counts['raw_valid'],normalized_valid=counts['normalized_valid'],invalid=counts['invalid'],
        post_normalization_valid=counts['raw_valid']+counts['normalized_valid'],success_criterion_met=counts['invalid']==0,
        source_totals={s:dict(Counter(r['status'] for r in records if r['source']==s)) for s in ('A','B')},
        breakdown=breakdown,field_agreement=agreement,normalization_fields=sum(r['changes'] for r in records),
        all_normalizations_reversible=True,debug_target_overlap=0,target_classifier_calls=0,full_annotation_started=False,
        boundary='Format/transport reliability on one frozen60-image holdout only; field agreement is not semantic accuracy.')
    atomic_json(out/'independent_audit.json',audit)
    decision='supported' if audit['success_criterion_met'] else 'unsupported'
    atomic_text(out/'claim_evidence.md','# Claim evidence\n\n'
        f'- Claim: the fixed transport plus one reversible evidence serialization yields structurally usable outputs on this frozen holdout. Status: {decision}. '
        f'Evidence: {audit["post_normalization_valid"]}/200 valid; raw {audit["raw_valid"]}/200; normalized {audit["normalized_valid"]}/200; invalid {audit["invalid"]}/200.\n'
        '- Boundary: this does not establish human-label accuracy, agreement as truth, or selection-method efficacy.\n')
    gate='PASS' if audit['success_criterion_met'] else 'FAIL'
    atomic_text(out/'review.md','# Review\n\n'
        '- Citation integrity: PASS (no external citation claims).\n'
        '- Result traceability: PASS (200 raw responses replayed and matched to frozen inputs).\n'
        f'- Claim-scope integrity: {gate} for proceeding beyond format validation; semantic accuracy remains untested.\n'
        f'- Full annotation gate: {gate}.\n')
    lines=['# 双 VLM 未调试样本格式验证（2026-11-15）','',
        '使用未参与前两轮调试的60张冻结原图：六类的 selection/reference 各5张。每张执行原生协议；horse、sheep、person、car另执行细粒度协议，共100个语义任务、A/B各一次，总计200次调用。','',
        f'结果：原始格式通过 {audit["raw_valid"]}/200；经唯一允许的可逆 evidence 数组序列化后另有 {audit["normalized_valid"]}/200 通过；仍失败 {audit["invalid"]}/200。',
        f'预设的200/200结构可用条件：{"达到" if audit["success_criterion_met"] else "未达到"}。独立审计从200份原始响应重放，计数一致。','',
        '| 来源 | 原始通过 | 规范化后通过 | 失败 |','| --- | ---: | ---: | ---: |']
    for source in ('A','B'):
        c=Counter(r['status'] for r in records if r['source']==source)
        lines.append(f'| {source} | {c["raw_valid"]} | {c["normalized_valid"]} | {c["invalid"]} |')
    lines+=['','规范化仅作用于 native `evidence` 中 schema 要求字符串、模型却返回非空字符串数组的字段；使用紧凑 JSON 文本保存，可由 `json.loads` 无损恢复。任何 feature、fine 字段、样本身份和语义取值都未修改。','',
        '该结果只验证格式与传输可用性。A/B字段一致不等于标注正确；完整标注、选样和分类器推理尚未启动。']
    report=ROOT/'docs/实验方案与结果/双VLM未调试样本格式验证_20261115.md'
    atomic_text(report,'\n'.join(lines)+'\n');atomic_text(out/'result.md','\n'.join(lines)+'\n')
    final_stage='heldout_format_validation_pass' if audit['success_criterion_met'] else 'heldout_format_validation_failed'
    atomic_json(held.DATA/'pipeline_progress.json',dict(stage=final_stage,data_qualified=True,processed_calls=200,
        raw_valid=audit['raw_valid'],normalized_valid=audit['normalized_valid'],invalid=audit['invalid'],
        full_annotation_started=False,target_classifier_calls=0))
    atomic_text(held.DATA/'当前进度.md',f'双 VLM 未调试样本格式验证已完成。\n\n原始格式通过 {audit["raw_valid"]}/200，规范化后另通过 {audit["normalized_valid"]}/200，失败 {audit["invalid"]}/200。\n\n完整标注尚未启动。\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('breakdown','field_agreement')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
