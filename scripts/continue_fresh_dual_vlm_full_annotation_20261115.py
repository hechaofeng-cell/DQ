#!/usr/bin/env python3
"""Prepare, run both annotation sources, and audit under one durable owner."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fresh_image_qualification_20261112/dual_vlm_full_annotation_20261115'
PYTHON=ROOT/'.venv/bin/python'


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def command(log,*arguments):
    stream=(OUT/log).open('a')
    return subprocess.Popen([str(PYTHON),*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT),stream


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        prepare,stream=command('prepare.log',ROOT/'scripts/run_fresh_dual_vlm_full_annotation_20261115.py','--phase','prepare')
        code=prepare.wait();stream.close()
        if code:raise RuntimeError('prepare failed; see prepare.log')
        processes=[]
        for source in ('A','B'):
            process,stream=command(source+'.log',ROOT/'scripts/run_fresh_dual_vlm_full_annotation_20261115.py',
                '--phase','source','--source',source);processes.append((source,process,stream))
        failures=[]
        for source,process,stream in processes:
            code=process.wait();stream.close()
            if code:failures.append(source)
        if failures:raise RuntimeError('source runners failed: '+','.join(failures))
        audit,stream=command('audit.log',ROOT/'scripts/audit_fresh_dual_vlm_full_annotation_20261115.py')
        code=audit.wait();stream.close()
        if code:raise RuntimeError('audit failed; see audit.log')
        result=json.loads((OUT/'independent_audit.json').read_text())
        atomic(OUT/'completion.json',dict(status='complete',success_criterion_met=result['success_criterion_met'],
            valid_by_source=result['valid_by_source'],invalid_by_source=result['invalid_by_source'],
            physical_calls=result['physical_calls'],completed_at=datetime.now(timezone.utc).isoformat()))
    except Exception as exc:
        atomic(OUT/'supervisor_failure.json',dict(status='failed',error=type(exc).__name__+':'+str(exc),
            at=datetime.now(timezone.utc).isoformat()))
        raise


if __name__=='__main__':main()
