#!/usr/bin/env python3
"""Independently replay completed full dual-VLM annotations and agreement."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import run_fresh_dual_vlm_full_annotation_20261115 as run
from prepare_fresh_images_20261112 import OUT as DATA,sha
from qe_quality.dta.io import atomic_json,atomic_text,fingerprint


def expected_binding(source,job,model,attempt,part,fields=None):
    return fingerprint(dict(job=job,source=source,model=model,
        transport=('A_full_strict' if source=='A' else 'B_universal_chunked_strict'),attempt=attempt,part=part,
        field_ids=fields,system=(None if source=='A' else run.transport.SYSTEM_PROMPT),
        normalization=(None if source=='A' else run.final.NORMALIZATION)))


def verify_call(source,job,model,attempt,part,fields=None):
    folder=run.call_folder(source,job,attempt,part);started=json.loads((folder/'call_started.json').read_text())
    binding=expected_binding(source,job,model,attempt,part,fields)
    if (started['input_binding']!=binding or not started['one_request'] or started['body_bytes']>run.MAX_BODY_BYTES
            or started['attempt']!=attempt or started['part']!=part):
        raise ValueError('Call provenance mismatch')
    raw=folder/'raw.json';meta=json.loads((folder/'transport.json').read_text())
    if meta['status']!=200:
        raise ValueError('Valid result is bound to non-200 response')
    return dict(status='response',binding=binding,raw=raw)


def replay_valid(source,job,result,model):
    attempt=result['attempt']
    if not 1<=attempt<=run.MAX_ATTEMPTS:
        raise ValueError('Attempt outside frozen cap')
    if source=='A':
        parsed=run.parse_a(verify_call('A',job,model,attempt,'full'),job)
        if parsed['status']!='valid':raise ValueError('A replay invalid')
        standard=parsed['standard'];changes=parsed['normalization'];hashes=[parsed['raw_sha256']]
    else:
        features={};evidence={};values={};changes=[];hashes=[]
        for index,fields in enumerate(run.transport.field_chunks(job),1):
            call=verify_call('B',job,model,attempt,'part'+str(index),fields)
            response=json.loads(call['raw'].read_text());parsed,_,part_changes=run.final.parse_component(response,job,fields)
            if job['kind']=='native':features.update(parsed['features']);evidence.update(parsed['evidence'])
            else:values.update(parsed['values'])
            changes.extend(part_changes);hashes.append(sha(call['raw']))
        standard=(dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],
            target_instance_id='target_1',features=features,evidence=evidence) if job['kind']=='native' else values)
        run.held.validate_result(standard,job)
    if result['standard']!=standard or result['normalization']!=changes or result['raw_sha256']!=hashes:
        raise ValueError('Final result differs from raw replay')
    return standard,changes


def main():
    frozen=json.loads((run.OUT/'code_and_input_freeze.json').read_text())
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Frozen code/input changed: '+path)
    index=json.loads((run.OUT/'job_index.json').read_text())
    if len(index)!=run.SEMANTIC_JOBS:raise ValueError('Job index count mismatch')
    runtime=json.loads((run.OUT/'runtime.json').read_text());models={'A':runtime['A_model'],'B':runtime['B_model']}
    class_lookup={(item['sample_id'],item['kind']):item['class_name'] for item in index}
    jobs=list(run.iter_jobs(index));valid_by_source=Counter();invalid_by_source=Counter();normalizations=Counter()
    standards={'A':{},'B':{}};records=[]
    for job,digest in jobs:
        for path,expected in zip(job['evidence_paths'],job['evidence_sha256']):
            if sha(path)!=expected:raise ValueError('Evidence changed')
        for source in ('A','B'):
            result=json.loads((run.source_folder(source,job)/'result.json').read_text())
            if result['status']=='valid':
                standard,changes=replay_valid(source,job,result,models[source]);valid_by_source[source]+=1
                standards[source][(job['sample_id'],job['kind'])]=standard
                for change in changes:normalizations[source+':'+change['operation']]+=1
            else:
                invalid_by_source[source]+=1
            records.append(dict(sample_id=job['sample_id'],class_name=job['class_name'],role=job['holdout_role'],
                kind=job['kind'],source=source,status=result['status'],attempt=result.get('attempt'),job_sha256=digest))
    physical={source:len(list((run.OUT/'jobs'/source).glob('*/*/call_started.json'))) for source in ('A','B')}
    for source in ('A','B'):
        if physical[source]>run.MAX_CALLS[source]:raise ValueError('Physical call cap exceeded')
        summary=json.loads((run.OUT/f'{source}_summary.json').read_text())
        if (summary['processed_jobs']!=run.SEMANTIC_JOBS or summary['valid_jobs']!=valid_by_source[source]
                or summary['invalid_jobs']!=invalid_by_source[source] or summary['physical_calls']!=physical[source]):
            raise ValueError('Source summary differs from replay')
    agreement=defaultdict(lambda:dict(jobs=0,fields=0,equal_fields=0,exact_jobs=0,A_unknown=0,B_unknown=0))
    common=set(standards['A'])&set(standards['B'])
    for key in common:
        sample_id,kind=key;a=standards['A'][key];b=standards['B'][key]
        features_a=a['features'] if kind=='native' else a;features_b=b['features'] if kind=='native' else b
        class_name=class_lookup[key]
        bucket=agreement[class_name+'/'+kind];bucket['jobs']+=1;bucket['fields']+=len(features_a)
        equal=sum(features_a[field]==features_b[field] for field in features_a);bucket['equal_fields']+=equal
        bucket['exact_jobs']+=equal==len(features_a)
        for source,features in (('A',features_a),('B',features_b)):
            bucket[source+'_unknown']+=sum(value=='unknown' or isinstance(value,list) and 'unknown' in value for value in features.values())
    success=all(valid_by_source[source]==run.SEMANTIC_JOBS for source in ('A','B'))
    audit=dict(status='pass',success_criterion_met=success,images=4800,semantic_jobs=run.SEMANTIC_JOBS,
        source_jobs=2*run.SEMANTIC_JOBS,valid_by_source=dict(valid_by_source),invalid_by_source=dict(invalid_by_source),
        physical_calls=physical,normalizations=dict(normalizations),common_valid_jobs=len(common),agreement=dict(agreement),
        maximum_three_semantic_attempts=True,inputs_and_evidence_frozen=True,raw_responses_replayed=True,
        human_verification=False,target_classifier_calls=0,records=records,
        boundary='Automatic VLM agreement and schema validity do not establish semantic accuracy.')
    atomic_json(run.OUT/'independent_audit.json',audit)
    lines=['# 全量双VLM自动标注结果（2026-11-15）','',
        f'A有效 {valid_by_source["A"]}/8000，B有效 {valid_by_source["B"]}/8000；完整成功条件：{"达到" if success else "未达到"}。','',
        f'实际底层调用：A {physical["A"]}，B {physical["B"]}。所有有效结果均从原始响应独立重放，并核对输入、证据和代码冻结哈希。','',
        'A/B一致性按类别和协议分别保存在 independent_audit.json；unknown和分歧均原样保留，没有人工裁决或第三模型多数替换。','',
        '该阶段完成自动标签冻结，不将模型一致性解释为人工语义准确率。分类器推理与选样实验仍是后续阶段。']
    atomic_text(run.OUT/'result.md','\n'.join(lines)+'\n')
    atomic_text(ROOT/'docs/实验方案与结果/全量双VLM自动标注结果_20261115.md','\n'.join(lines)+'\n')
    atomic_json(DATA/'pipeline_progress.json',dict(stage='full_dual_vlm_annotation_complete' if success else 'full_dual_vlm_annotation_incomplete',
        data_qualified=True,full_annotation_started=True,full_annotation_complete=success,
        valid_by_source=dict(valid_by_source),invalid_by_source=dict(invalid_by_source),physical_calls=physical,
        human_verification=False,target_classifier_calls=0))
    print(json.dumps({key:value for key,value in audit.items() if key!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
