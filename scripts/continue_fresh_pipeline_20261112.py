#!/usr/bin/env python3
"""Durable continuation for the already-running frozen acquisition; no second downloader."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_fresh_images_20261112 import ROOT, READY, OUT, sha, read, dump

LOG=Path('/tmp/qe_fresh_acquisition_full_20261112.log')


def identity(pid):
    try:
        base=Path('/proc')/str(pid)
        rest=(base/'stat').read_text().rsplit(')',1)[1].split()
        if rest[0]=='Z':return None
        return dict(start_ticks=rest[19],command=(base/'cmdline').read_bytes().split(b'\0'))
    except (FileNotFoundError,ProcessLookupError):return None


def alive(pid,start_ticks):
    value=identity(pid)
    return bool(value and value['start_ticks']==start_ticks and
        b'scripts/acquire_fresh_images_20261112.py' in value['command'])


def recent_transport():
    if not LOG.exists():return {}
    with LOG.open('rb') as f:
        f.seek(max(0,f.seek(0,2)-65536));lines=f.read().decode(errors='replace').splitlines()
    result={}
    for line in lines:
        try:r=json.loads(line)
        except ValueError:continue
        if r.get('stage')=='package_start':result=dict(package=r['package'],compressed_bytes=0)
        elif r.get('stage')=='range_progress':result.update(r)
        elif r.get('status')=='pass':result={}
    return result


def snapshot(stage,detail='',**extra):
    packages=[json.loads(p.read_text()) for p in (OUT/'package_audits').glob('*.json')]
    passed=[p for p in packages if p['status']=='pass']
    budget=json.loads((OUT/'network_budget.json').read_text())
    required=read(READY/'alternative_packages.csv')
    sizes=json.loads((READY/'package_size_audit.json').read_text())['records']
    completed={r['package'] for r in passed};remaining=sum(r['bytes'] for r in sizes if r['package'] not in completed)
    transport=recent_transport();left=max(0,remaining-transport.get('compressed_bytes',0))
    rate=transport.get('MiB_per_second',0)
    value=dict(updated_at=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),stage=stage,
        detail=detail,packages_passed=len(passed),packages_required=len(required),
        source_verified_originals=sum(p['targets'] for p in passed),planned_originals=5400,
        conservative_network_reserved_bytes=budget['reserved_bytes'],remaining_source_bytes_estimate=left,
        remaining_download_hours_estimate=round(left/(rate*1024**2)/3600,2) if rate else None,
        current_transport=transport,data_qualified=False,full_annotation_started=False,**extra)
    qualification=OUT/'qualification_summary.json'
    if qualification.exists():value['data_qualified']=json.loads(qualification.read_text()).get('data_qualified',False)
    dump('pipeline_progress.json',value)
    labels={'waiting_for_acquisition':'正在采集冻结名单中的图片','qualifying':'正在执行完整图片资格检查',
        'independent_data_audit':'正在独立复核图片和名单绑定','dual_vlm_smoke':'正在执行六类双 VLM 小样本检查',
        'smoke_complete_ready_for_full_annotation':'图片资格与双 VLM 小样本检查已完成','failed':'流程已停止并保留错误'}
    p=OUT/'当前进度.md';tmp=p.with_suffix('.md.tmp')
    tmp.write_text(f"更新时间：{value['updated_at']}\n\n{labels.get(stage,stage)}。{detail}\n\n"
        f"- 已完整验证来源包：{len(passed)}/{len(required)}；已验证原图：{value['source_verified_originals']}/5400。\n"
        f"- 图片全池资格：{'通过' if value['data_qualified'] else '尚未通过'}。\n"
        f"- 剩余传输时间粗估：{value['remaining_download_hours_estimate']} 小时；随网络速度变化。\n"
        "- 自动顺序：采集 → 完整去重与名单冻结 → 独立像素审计 → 六类双 VLM 检查（20 个任务）。\n"
        "- 不需要人工核验；未启动全量标注和目标分类器推理。小样本一致性不等于标注准确率。\n"
        "\n完整机器状态见 pipeline_progress.json；阶段日志见 pipeline_logs/。\n")
    tmp.replace(p)
    return value


def check_freeze(frozen):
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError('Pipeline code/input changed: '+path)


def run_stage(stage,args,frozen):
    check_freeze(frozen);snapshot(stage)
    path=OUT/'pipeline_logs'/f'{stage}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            snapshot(stage,child_pid=child.pid);time.sleep(15)
        if child.returncode:raise RuntimeError(f'{stage} exited {child.returncode}; see {path}')
    check_freeze(frozen)


def main(pid):
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline_supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ident=identity(pid)
    if not ident or b'scripts/acquire_fresh_images_20261112.py' not in ident['command']:
        raise ValueError('Expected existing acquisition process is unavailable')
    ticks=ident['start_ticks']
    source=['scripts/prepare_fresh_images_20261112.py','scripts/acquire_fresh_images_20261112.py',
        'scripts/qualify_fresh_images_20261112.py','scripts/audit_qualified_fresh_images_20261112.py',
        'scripts/smoke_fresh_dual_vlm_20261112.py','scripts/continue_fresh_pipeline_20261112.py',
        'scripts/prepare_multiclass_object_candidate_pool.py','scripts/run_dog_local_pipeline.py',
        'scripts/annotate_dual_vlm.py','scripts/annotate_other_classes_specific_v2_20261018.py',
        'scripts/run_other_classes_specific_experiment_20261018.py','qe_quality/dta/clients.py','qe_quality/dta/io.py',
        'tests/test_fresh_image_acquisition.py','tests/test_fresh_image_qualification.py','tests/test_fresh_pipeline_continuation.py']
    inputs=json.loads((READY/'final_freeze.json').read_text())
    for p in source:inputs[str(ROOT/p)]=sha(ROOT/p)
    for p in ('historical_fingerprints.jsonl','historical_binding_summary.json','historical_references.csv',
              'historical_relocations.csv','historical_unresolved.csv','historical_hash_mismatches.csv',
              'initial_code_freeze.json','acquisition_code_freeze.json','qualification_code_freeze.json'):
        inputs[str(OUT/p)]=sha(OUT/p)
    parent=json.loads((READY/'frozen_validation_protocol.json').read_text())
    encoder=Path.home()/'.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth'
    inputs[str(encoder)]=sha(encoder)
    for p in (ROOT/'artifacts/other_classes_specific_experiment_20261018/frozen_dictionaries').glob('*.json'):
        inputs[str(p)]=sha(p)
    # Freeze the exact original prompt/schema paths actually loaded by the smoke adapter as well.
    for domain in ('old','car'):
        for relative in (f'configs/{domain}_annotation_schema.json',f'prompts/{domain}.txt'):
            p=ROOT/'artifacts/sixclass_independent_confirmation_20261004'/relative;inputs[str(p)]=sha(p)
    dependencies={name:version(name) for name in ('numpy','scipy','Pillow','torch','torchvision','requests',
        'jsonschema','attrs','jsonschema-specifications','referencing','rpds-py')}
    runtime=OUT/'pipeline_runtime.json'
    runtime_value=dict(python=sys.version,executable=sys.executable,dependencies=dependencies)
    if runtime.exists():
        if json.loads(runtime.read_text())!=runtime_value:raise ValueError('Pipeline runtime changed')
    else:dump(runtime.name,runtime_value)
    inputs[str(runtime)]=sha(runtime)
    freeze=OUT/'pipeline_code_and_input_freeze.json'
    if freeze.exists():
        if json.loads(freeze.read_text())!=inputs:raise ValueError('Supervisor freeze changed')
    else:dump(freeze.name,inputs)
    check_freeze(inputs)
    dump('pipeline_supervisor.json',dict(pid=os.getpid(),watched_acquisition_pid=pid,acquisition_start_ticks=ticks,
        human_verification=False,automatic_stages=['qualify','independent_data_audit','dual_vlm_smoke'],
        full_annotation_automatic=False,maximum_smoke_attempts=60,parent_protocol=parent['version']))
    try:
        while alive(pid,ticks):snapshot('waiting_for_acquisition');time.sleep(15)
        state=snapshot('waiting_for_acquisition','采集进程已结束，正在检查完整性。')
        if state['packages_passed']!=18 or state['source_verified_originals']!=5400:
            raise RuntimeError('Acquisition stopped before all18 packages/5400 originals passed; no downstream model calls')
        run_stage('qualifying',['scripts/qualify_fresh_images_20261112.py','qualify'],inputs)
        q=json.loads((OUT/'qualification_summary.json').read_text())
        if q['status']!='pass' or not q['data_qualified']:raise RuntimeError('Insufficient supply after frozen duplicate exclusions')
        run_stage('independent_data_audit',['scripts/audit_qualified_fresh_images_20261112.py'],inputs)
        run_stage('dual_vlm_smoke',['scripts/smoke_fresh_dual_vlm_20261112.py'],inputs)
        snapshot('smoke_complete_ready_for_full_annotation','20 个自动检查任务通过；完整标注与选样适配仍需后续执行。')
        dump('pipeline_completion.json',dict(status='data_and_smoke_complete',full_annotation_started=False,
            completed_at=datetime.now(timezone.utc).isoformat()))
    except Exception as exc:
        # No secret config is included in any supervisor state or error output.
        detail=type(exc).__name__+':'+str(exc)
        dump('pipeline_failure.json',dict(status='failed',error=detail,at=datetime.now(timezone.utc).isoformat()))
        snapshot('failed',detail)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch-pid',type=int,required=True)
    main(p.parse_args().watch_pid)
