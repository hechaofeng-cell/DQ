#!/usr/bin/env python3
"""Bounded supplemental rerun; never overwrite the completed confirmatory run."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_fresh_dual_vlm_full_annotation_20261115 as parent
from qe_quality.dta.io import atomic_json, atomic_text, fingerprint

PARENT = parent.OUT
OUT = PARENT.parent / 'failed_annotation_retry_20261118'
ORIGINAL_CHUNKS = parent.transport.field_chunks


def now():
    return datetime.now(timezone.utc).isoformat()


def bounded_chunks(job):
    ids = [x for group in ORIGINAL_CHUNKS(job) for x in group]
    return [ids[i:i + 5] for i in range(0, len(ids), 5)]


def constrained_job(job):
    """Encode existing fine-label semantics in decoding, without choosing answers."""
    changed = deepcopy(job)
    if job['kind'] == 'native':
        # The original native context can leave too little room for generation.
        changed['options']['num_ctx'] = 16384
        return changed
    for field in job['semantic_schema']['features']:
        if field['value_type'] != 'set':
            continue
        key = field['feature_id']
        rule = changed['response_schema']['properties'][key]
        allowed = rule['items']['enum']
        exclusive = set(field.get('exclusive_values', ['unknown']))
        ordinary = [x for x in allowed if x not in exclusive]
        alternatives = [dict(type='array', items=dict(type='string', enum=[x]),
                             minItems=1, maxItems=1) for x in allowed if x in exclusive]
        if ordinary:
            alternatives.append(dict(type='array', items=dict(type='string', enum=ordinary),
                                     minItems=1, maxItems=len(ordinary), uniqueItems=True))
        changed['response_schema']['properties'][key] = dict(oneOf=alternatives)
    return changed


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'inputs.json').exists():
        return
    rows = {r['sample_id']: r for r in parent.read(parent.MANIFEST)}
    index = {(x['sample_id'], x['kind']): x for x in json.loads((PARENT / 'job_index.json').read_text())}
    jobs = []
    frozen = {}
    for source in ('A', 'B'):
        for path in sorted((PARENT / 'jobs' / source).glob('*/result.json')):
            if json.loads(path.read_text())['status'] == 'valid':
                continue
            sid, kind = path.parent.name.rsplit('_', 1)
            job = parent.build_job(rows[sid], kind)
            if fingerprint(job) != index[(sid, kind)]['job_sha256']:
                raise ValueError('Parent input reconstruction changed')
            jobs.append(dict(source=source, job=job, parent_result=str(path)))
            for file in path.parent.rglob('*.json'):
                frozen[str(file)] = parent.sha(file)
    for file in (Path(__file__).resolve(), ROOT / 'tests/test_failed_fresh_vlm_retry_20261118.py',
                 PARENT / 'completion.json', PARENT / 'independent_audit.json',
                 PARENT / 'job_index.json', PARENT / 'protocol.json', parent.MANIFEST):
        frozen[str(file)] = parent.sha(file)
    for module in (parent, parent.smoke, parent.held, parent.transport, parent.final, parent.repair):
        file = Path(module.__file__)
        frozen[str(file)] = parent.sha(file)
    parent.freeze(OUT / 'freeze.json', frozen)
    parent.freeze(OUT / 'inputs.json', dict(created_at=now(), jobs=jobs))
    parent.freeze(OUT / 'protocol.json', dict(
        question='Do identical requests recover prior failures, and do bounded delivery changes recover the rest?',
        population='All failed source jobs from the completed full run; unchanged image bytes and target.',
        baseline='One additional semantic attempt using the identical parent request builder and settings.',
        repair_A='Only unresolved A: native context 8192 to 16384; fine existing exclusivity and uniqueness encoded in decoding schema.',
        repair_B='Only unresolved B without explicit safety refusal: same evidence, <=5 fields per component instead of <=10.',
        maximum_semantic_attempts_per_source_job=2, maximum_physical_calls=214,
        original_validation='Every accepted result must also pass the unchanged parent semantic validator.',
        refusal='No alternate request following an explicit provider safety refusal.',
        boundary='Supplemental post hoc recovery, not an independent holdout or proof of visual correctness.',
        no_human_verification=True, target_classifier_calls=0))
    print(json.dumps(dict(prepared=len(jobs), by_source=dict(Counter(x['source'] for x in jobs)))), flush=True)


def explicit_refusal(folder):
    for raw in folder.glob('*/raw.json'):
        try:
            value = json.loads(raw.read_text())
        except ValueError:
            continue
        error = value.get('error', {})
        message = str(error.get('message', '')).lower() if isinstance(error, dict) else str(error).lower()
        if any(x in message for x in ('inappropriate input/output', 'sensitive image', 'sensitive content', 'content policy')):
            return True
    return False


def execute(source, phase, job, cfg):
    parent.OUT = OUT / phase
    parent.MAX_ATTEMPTS = 1
    parent.transport.field_chunks = bounded_chunks if phase == 'repair' and source == 'B' else ORIGINAL_CHUNKS
    submitted = constrained_job(job) if phase == 'repair' and source == 'A' else job
    result_file = parent.source_folder(source, job) / 'result.json'
    try:
        result = parent.execute_a(submitted, cfg) if source == 'A' else parent.execute_b(submitted, cfg)
        if result['status'] == 'valid':
            parent.held.validate_result(result['standard'], job)
        return result
    except Exception as exc:
        result = dict(status='invalid', reason=type(exc).__name__, message=str(exc)[:1000])
        atomic_json(result_file, result)
        return result


def run_source(source):
    lock = (OUT / (source + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, online = parent.runtime()
    cfg = dict(model=protocol['models']['A'], url='http://127.0.0.1:11434/api/chat') if source == 'A' else online
    jobs = [x for x in json.loads((OUT / 'inputs.json').read_text())['jobs'] if x['source'] == source]
    records = []
    for item in jobs:
        job = item['job']
        baseline = execute(source, 'baseline', job, cfg)
        phase = 'baseline'
        result = baseline
        refused = explicit_refusal(OUT / 'baseline' / 'jobs' / source / f"{job['sample_id']}_{job['kind']}")
        if result['status'] != 'valid' and not refused:
            phase = 'repair'
            result = execute(source, phase, job, cfg)
        record = dict(source=source, sample_id=job['sample_id'], kind=job['kind'],
                      baseline=baseline['status'], final=result['status'], final_phase=phase,
                      explicit_refusal=refused)
        records.append(record)
        value = dict(status='complete' if len(records) == len(jobs) else 'running', updated_at=now(),
                     planned=len(jobs), processed=len(records),
                     baseline_recovered=sum(x['baseline'] == 'valid' for x in records),
                     final_recovered=sum(x['final'] == 'valid' for x in records), records=records)
        atomic_json(OUT / (source + '_progress.json'), value)
        print(json.dumps(record), flush=True)


def finalize():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    changed = [path for path, digest in frozen.items() if parent.sha(path) != digest]
    inputs = json.loads((OUT / 'inputs.json').read_text())['jobs']
    records = []
    for source in ('A', 'B'):
        progress = json.loads((OUT / (source + '_progress.json')).read_text())
        if progress['status'] != 'complete':
            raise ValueError('Incomplete supplementary run')
        records.extend(progress['records'])
    # Replay accepted responses from raw files, and independently reapply original semantics.
    replayed = 0
    for record in records:
        if record['final'] != 'valid':
            continue
        job = next(x['job'] for x in inputs if x['source'] == record['source'] and
                   x['job']['sample_id'] == record['sample_id'] and x['job']['kind'] == record['kind'])
        phase = record['final_phase']; source = record['source']
        folder = OUT / phase / 'jobs' / source / f"{job['sample_id']}_{job['kind']}"
        stored = json.loads((folder / 'result.json').read_text())
        if source == 'A':
            raw = folder / 'attempt1_full/raw.json'
            value = parent.held.response_value(json.loads(raw.read_text()), 'A')
            replay = parent.held.validate_raw_or_normalized(value, job)['result']
            if [parent.sha(raw)] != stored['raw_sha256']:
                raise ValueError('A raw hash changed')
        else:
            fields = bounded_chunks(job) if phase == 'repair' else ORIGINAL_CHUNKS(job)
            features = {}; evidence = {}; values = {}; hashes = []
            for i, ids in enumerate(fields, 1):
                raw = folder / f'attempt1_part{i}/raw.json'
                part, _, _ = parent.final.parse_component(json.loads(raw.read_text()), job, ids)
                hashes.append(parent.sha(raw))
                if job['kind'] == 'native':
                    features.update(part['features']); evidence.update(part['evidence'])
                else:
                    values.update(part['values'])
            replay = dict(schema_version=job['semantic_schema']['schema_version'], image_id=job['sample_id'],
                          target_instance_id='target_1', features=features, evidence=evidence) if job['kind'] == 'native' else values
            if hashes != stored['raw_sha256']:
                raise ValueError('B raw hash changed')
        parent.held.validate_result(replay, job)
        if replay != stored['standard']:
            raise ValueError('Raw reconstruction differs from result')
        replayed += 1
    calls = len(list(OUT.glob('*/jobs/*/*/*/call_started.json')))
    if calls > 214 or changed:
        raise ValueError('Budget exceeded or frozen files changed: ' + str(changed))
    summary = dict(status='complete', finished_at=now(), failed_source_jobs=len(inputs),
                   baseline_recovered=sum(r['baseline'] == 'valid' for r in records),
                   final_recovered=replayed, still_invalid=len(inputs) - replayed,
                   physical_calls=calls, raw_replay='pass', parent_unchanged=True, records=records,
                   boundary='Recovery and structural/semantic consistency only; no semantic ground truth.')
    atomic_json(OUT / 'summary.json', summary)
    baseline = summary['baseline_recovered']
    report = f'''# 失败标注补充复测（2026-11-18）

原全量实验已结束，本次只复测其 {len(inputs)} 个失败的模型标注任务，不覆盖原结果。

| 指标 | 结果 |
|---|---:|
| 相同请求补跑恢复 | {baseline}/{len(inputs)} |
| 调整传输/约束表达后累计恢复 | {replayed}/{len(inputs)} |
| 仍未得到合格完整标注 | {len(inputs)-replayed} |
| 新增实际模型调用 | {calls} |

同一图片、相同请求也能恢复的任务，说明原失败不能归因为图片不可识别。
修复阶段仍使用原始图片和原始标签含义；本地模型只增加上下文空间，或把原有互斥要求编码进生成约束；在线模型只缩小每次输出的字段组。
所有计入恢复的结果均通过未修改的原标注验证器，且从原始返回独立回放一致。原实验文件哈希未变。

这项结果证明的是标注流程的可恢复性，不是图像语义标注的准确率，也不能证明“有图片就不可能失败”。

详细逐项记录见 summary.json；每次请求的原始响应、耗时和验证结果在 baseline/ 与 repair/。
'''
    atomic_text(OUT / 'result.md', report)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['A', 'B'])
    args = parser.parse_args()
    if args.source:
        run_source(args.source)
        return
    prepare()
    children = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--source', x], cwd=ROOT)
                for x in ('A', 'B')]
    codes = [child.wait() for child in children]
    if any(codes):
        raise RuntimeError('Source process failed: ' + str(codes))
    finalize()


if __name__ == '__main__':
    main()
