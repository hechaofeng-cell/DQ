#!/usr/bin/env python3
"""Resume after completed acquisition, preserving the original protocol and failure evidence."""
import fcntl
import json
from pathlib import Path

from prepare_fresh_images_20261112 import ROOT, OUT, sha, dump
from continue_fresh_pipeline_20261112 import run_stage, snapshot


def main():
    lock=(OUT/'pipeline_supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=OUT/'pipeline_code_and_input_freeze.json'
    inputs=json.loads(parent.read_text())
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('Frozen parent changed: '+p)
    inputs[str(parent)]=sha(parent)
    for relative in ('scripts/qualify_fresh_images_v2_20261114.py','scripts/audit_qualified_fresh_images_v2_20261114.py',
                     'scripts/continue_fresh_after_acquisition_20261114.py','tests/test_fresh_box_exclusion_20261114.py'):
        p=ROOT/relative;inputs[str(p)]=sha(p)
    freeze=OUT/'box_exclusion_recovery_code_freeze_20261114.json'
    if freeze.exists():
        if json.loads(freeze.read_text())!=inputs:raise ValueError('Recovery implementation changed')
    else:dump(freeze.name,inputs)
    dump('box_exclusion_recovery_protocol_20261114.json',dict(
        scientific_protocol_unchanged=True,parent_code_freeze_sha256=sha(parent),
        repair='Record invalid official boxes as technical exclusions, then use already frozen own-class/role reserves in order',
        boundary_tolerance_unchanged=1e-9,clipping_or_membership_reordering=False,
        old_qualifier_and_all_failure_logs_preserved=True,human_verification=False))
    try:
        state=snapshot('qualifying','采集已完成；记录官方框技术排除后按冻结规则继续。')
        if state['packages_passed']!=18 or state['source_verified_originals']!=5400:raise ValueError('Acquisition incomplete')
        run_stage('qualifying_v2',['scripts/qualify_fresh_images_v2_20261114.py','qualify'],inputs)
        q=json.loads((OUT/'qualification_summary.json').read_text())
        if q['status']!='pass':raise RuntimeError('Insufficient qualified supply; stop before VLM')
        run_stage('independent_data_audit',['scripts/audit_qualified_fresh_images_v2_20261114.py'],inputs)
        run_stage('dual_vlm_smoke',['scripts/smoke_fresh_dual_vlm_20261112.py'],inputs)
        snapshot('smoke_complete_ready_for_full_annotation','完整图片审计与20个双 VLM 检查任务通过。')
        dump('pipeline_completion.json',dict(status='data_and_smoke_complete',full_annotation_started=False))
    except Exception as exc:
        detail=type(exc).__name__+':'+str(exc)
        dump('pipeline_failure.json',dict(status='failed',error=detail,recovery='official_box_technical_exclusion_v2'))
        snapshot('failed',detail)
        raise


if __name__=='__main__':main()
