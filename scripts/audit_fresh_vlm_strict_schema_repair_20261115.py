#!/usr/bin/env python3
"""Independently replay the four strict-schema repair calls."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_vlm_holdout_20261115 as held
import run_fresh_vlm_strict_schema_repair_20261115 as repair
from prepare_fresh_images_20261112 import sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def main():
    frozen=json.loads((repair.OUT/'code_and_parent_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen code/input changed: '+path)
    inputs=json.loads((repair.OUT/'inputs.json').read_text());summary=json.loads((repair.OUT/'summary.json').read_text())
    if len(inputs['targets'])!=4:raise ValueError('Target count mismatch')
    records=[]
    for target in inputs['targets']:
        job=target['job'];folder=repair.OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_B"
        started=json.loads((folder/'call_started.json').read_text())
        binding=fingerprint(dict(job=job,source='B',model=inputs['model'],response_format='strict_json_schema',system=inputs['system_prompt']))
        if started['input_binding']!=binding or not started['one_attempt_only']:raise ValueError('Call binding mismatch')
        if started['response_format']!='strict_json_schema' or started['body_bytes']>repair.MAX_BODY_BYTES:
            raise ValueError('Strict transport record mismatch')
        raw=folder/'raw.json';transport=json.loads((folder/'transport.json').read_text())
        status='invalid';reason='http_'+str(transport['status'])
        if transport['status']==200:
            try:
                response=json.loads(raw.read_text());value=held.response_value(response,'B');held.validate_result(value,job)
                parsed=json.loads((folder/'parsed.json').read_text())
                if parsed['input_binding']!=binding or parsed['raw_sha256']!=sha(raw) or parsed['result']!=value:
                    raise ValueError('Parsed result mismatch')
                status='raw_valid';reason=None
            except Exception as exc:
                reason=type(exc).__name__
        recorded=json.loads((folder/'outcome.json').read_text())
        if recorded['status']!=status:raise ValueError('Recorded/replayed status mismatch')
        records.append(dict(sample_id=job['sample_id'],kind=job['kind'],status=status,reason=reason,raw_sha256=sha(raw)))
    calls=len(list((repair.OUT/'jobs').glob('*/call_started.json')));valid=sum(r['status']=='raw_valid' for r in records)
    if calls!=4 or summary['raw_valid']!=valid or summary['invalid']!=4-valid:raise ValueError('Summary count mismatch')
    audit=dict(status='pass',model_calls=4,raw_records_replayed=4,raw_valid=valid,invalid=4-valid,
        success_criterion_met=valid==4,one_attempt_each=True,local_normalization_used=False,records=records,
        boundary='Repair feasibility on the four previously failed debug targets; not fresh holdout evidence.')
    atomic_json(repair.OUT/'independent_audit.json',audit)
    lines=['# B 严格 JSON Schema 四例修复试验（2026-11-15）','',
        '对未调试样本验证中的4个B结构失败任务保持模型、图片、任务提示和语义schema不变，只启用服务端严格JSON Schema并增加合法ID与字段名提醒。每个任务调用一次，不重试。','',
        f'结果：原始结构通过 {valid}/4，失败 {4-valid}/4。预设的4/4条件：{"达到" if valid==4 else "未达到"}。独立审计重放4份原始响应，计数一致。','',
        '该结果只说明已知失败案例的修复可行性。即使4/4通过，也必须使用全新未参与调试的样本验证后才能进入全量标注。']
    atomic_text(repair.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/双VLM严格Schema四例修复_20261115.md','\n'.join(lines)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
