#!/usr/bin/env python3
"""Durably run and audit the universal chunked holdout."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fresh_image_qualification_20261112/dual_vlm_universal_chunked_holdout_20261115'
PYTHON=ROOT/'.venv/bin/python'


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def run(name,script):
    with (OUT/(name+'.log')).open('a') as stream:
        result=subprocess.run([str(PYTHON),str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; see {name}.log')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        run('runner','run_fresh_vlm_universal_chunked_holdout_20261115.py')
        run('audit','audit_fresh_vlm_universal_chunked_holdout_20261115.py')
        audit=json.loads((OUT/'independent_audit.json').read_text())
        atomic(OUT/'completion.json',dict(status='complete',success_criterion_met=audit['success_criterion_met'],
            full_annotation_started=False,completed_at=datetime.now(timezone.utc).isoformat()))
    except Exception as exc:
        atomic(OUT/'supervisor_failure.json',dict(status='failed',error=type(exc).__name__+':'+str(exc),
            full_annotation_started=False,at=datetime.now(timezone.utc).isoformat()))
        raise


if __name__=='__main__':main()
