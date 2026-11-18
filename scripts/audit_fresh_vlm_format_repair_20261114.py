#!/usr/bin/env python3
"""Audit the bounded pilot from raw responses; publish only reproducible results."""
from collections import Counter
import hashlib
import json

import validate_fresh_vlm_format_repair_20261114 as pilot
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def main():
    out=pilot.OUT
    summary=json.loads((out/'summary.json').read_text())
    frozen=json.loads((out/'code_freeze.json').read_text())
    for path,digest in frozen.items():
        if pilot.sha(path)!=digest:raise ValueError('Frozen input/code changed')
    if summary['original_v1_summary_sha256']!=pilot.sha(pilot.original.OUT/'summary.json'):
        raise ValueError('Original failure evidence changed')
    jobs=json.loads((out/'inputs.json').read_text())['jobs']
    runtime=json.loads((out/'runtime.json').read_text())
    planned=[(j,s) for j in jobs for s in ('A','B')]
    if len(planned)!=20:raise ValueError('Expected twenty fixed source jobs')
    verified=[];values={};changed=[]
    for job in jobs:
        for p,h in zip(job['evidence_paths'],job['evidence_sha256']):
            if pilot.sha(p)!=h:raise ValueError('Actual transmitted evidence changed')
        if job['transport_evidence']['mode']!='unchanged':changed.append(job['class_name']+'/'+job['kind'])
    for job,source in planned:
        folder=out/'jobs'/f"{job['sample_id']}_{job['kind']}_{source}"
        stamp=json.loads((folder/'call_started.json').read_text())
        model=runtime[source+'_model'];binding=fingerprint(dict(job=job,source=source,model=model))
        if stamp['input_binding']!=binding:raise ValueError('Request binding mismatch')
        raw=folder/'raw.json';meta=folder/'transport.json';valid=False;reason='missing_response'
        if raw.exists() and meta.exists():
            status=json.loads(meta.read_text())['status']
            if status!=200:reason='http_'+str(status)
            else:
                try:
                    value=pilot.validate_returned(json.loads(raw.read_text()),job,source)
                    parsed=json.loads((folder/'parsed.json').read_text())
                    if parsed['result']!=value or parsed['input_binding']!=binding:raise ValueError('Parsed/raw mismatch')
                    if parsed['evidence_sha256']!=job['evidence_sha256']:raise ValueError('Parsed evidence mismatch')
                    values[(job['sample_id'],job['kind'],source)]=value;valid=True;reason='valid'
                except Exception as exc:reason=type(exc).__name__
        verified.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],kind=job['kind'],source=source,valid=valid,reason=reason,
            raw_sha256=pilot.sha(raw) if raw.exists() else None))
    attempted=len(list((out/'jobs').glob('*/call_started.json')))
    if attempted!=20 or attempted>pilot.MAX_CALLS:raise ValueError('Pilot attempt accounting mismatch')
    passed=sum(v['valid'] for v in verified)
    if passed!=summary['passed_jobs'] or len(verified)-passed!=summary['failed_jobs']:
        raise ValueError('Reported success counts differ from raw outputs')
    agreement=[]
    for j in jobs:
        sid=j['sample_id'];kind=j['kind']
        if not all((sid,kind,s) in values for s in ('A','B')):continue
        a,b=[values[sid,kind,s] for s in ('A','B')]
        if kind=='native':a=a['features'];b=b['features']
        def canonical(v):return sorted(v) if isinstance(v,list) else v
        agreement.append(dict(class_name=j['class_name'],kind=kind,fields=len(a),
            matching_fields=sum(canonical(v)==canonical(b[k]) for k,v in a.items()),
            A_unknown=sum(v=='unknown' or isinstance(v,list) and 'unknown' in v for v in a.values()),
            B_unknown=sum(v=='unknown' or isinstance(v,list) and 'unknown' in v for v in b.values())))
    audit=dict(status='pass',raw_source_jobs_verified=len(verified),passed_jobs=passed,failed_jobs=len(verified)-passed,
        source_pass_counts={s:sum(v['valid'] for v in verified if v['source']==s) for s in ('A','B')},
        actual_calls=attempted,maximum_calls=20,previous_calls=34,previous_failed_smoke_preserved=True,
        jobs_with_transport_resize=changed,matching_task_text_and_evidence_for_A_B=True,
        outcomes=verified,field_agreement=agreement,full_annotation_started=False,
        boundary='This verifies format/transport and raw record accounting; neither visual accuracy nor method efficacy.')
    atomic_json(out/'independent_audit.json',audit)
    lines=['# 双 VLM 格式与传输修复验证（2026-11-14）','',
        '前一轮检查为13/20通过、7/20失败，34次调用。原始失败记录和重试上限保留；本轮是另行冻结的格式/传输诊断验证。','',
        '失败原因：6个 B 模型任务的 JSON 字段类型不符；1个 B 模型任务因大型羊图请求体过大而返回 HTTP413。','',
        '本轮同时向两个模型提供原有完整 JSON Schema。每任务仅一次调用，总计20次；不按输出调整字段、词典、目标或标签。',
        '发送前按统一规则检查4MiB请求体上限；超限时保持宽高比按2048/1536/1024顺序生成 JPEG90 编码，两个模型使用相同字节。大型图片分辨率变化单独披露，不能声称与前一轮证据完全相同。','',
        f'本轮实际结果：{passed}/20通过，{20-passed}/20失败；A通过{audit["source_pass_counts"]["A"]}/10，B通过{audit["source_pass_counts"]["B"]}/10。独立从20份调用记录与原始响应重新核算，审计通过。','',
        '更改图片传输编码的任务：'+', '.join(changed)+'。','',
        '| 类别/协议 | 相同字段/全部字段 | A unknown | B unknown |','| --- | --- | --- | --- |']
    for r in agreement:lines.append(f'| {r["class_name"]}/{r["kind"]} | {r["matching_fields"]}/{r["fields"]} | {r["A_unknown"]} | {r["B_unknown"]} |')
    lines+=['','字段一致不代表准确。该检查复用少量固定目标，不能证明全量标注通过率、标注真值或选样方法有效性。',
        '完整16,000任务标注、风险计算与选样、冻结分类器推理均未启动。', '',
        '原始证据与可复核结果：artifacts/fresh_image_qualification_20261112/dual_vlm_format_repair_20261114/。']
    report=pilot.ROOT/'docs/实验方案与结果/双VLM格式与传输修复验证_20261114.md'
    atomic_text(report,'\n'.join(lines)+'\n')
    atomic_text(out/'result.md','\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('outcomes','field_agreement')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
