#!/usr/bin/env python3
"""Run then audit the bounded strict-schema repair."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fresh_image_qualification_20261112/dual_vlm_strict_schema_repair_20261115'
PYTHON=ROOT/'.venv/bin/python'


def run(name,script):
    with (OUT/(name+'.log')).open('a') as stream:
        result=subprocess.run([str(PYTHON),str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; see {name}.log')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    run('runner','run_fresh_vlm_strict_schema_repair_20261115.py')
    run('audit','audit_fresh_vlm_strict_schema_repair_20261115.py')
    audit=json.loads((OUT/'independent_audit.json').read_text())
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',success_criterion_met=audit['success_criterion_met'],
        fresh_holdout_started=False,full_annotation_started=False),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
