#!/usr/bin/env python3
"""Wait for this task's active annotation process, repair missing calls once, then audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cross_vlm_verification_20261112'
PYTHON = ROOT / '.venv/bin/python'


def state(stage, **kw):
    value = dict(stage=stage, **kw)
    temporary = OUT / 'pipeline_status.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUT / 'pipeline_status.json')
    print(json.dumps(value), flush=True)


def active(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2] != 'Z'


def run(pid):
    while active(pid):
        count = sum(1 for _ in (OUT / 'specific_B').glob('**/*.json'))
        state('annotating', valid_specific=count, expected=2400, annotation_pid=pid)
        time.sleep(30)
    count = sum(1 for _ in (OUT / 'specific_B').glob('**/*.json'))
    if count < 2400:
        state('repair_missing_once', valid_specific=count, expected=2400)
        subprocess.run([str(PYTHON), 'scripts/repair_cross_vlm_format_20261112.py'],
                       cwd=ROOT, check=False)
    count = sum(1 for _ in (OUT / 'specific_B').glob('**/*.json'))
    if count < 2400:
        state('incomplete_annotation', valid_specific=count, expected=2400)
        return
    # analyze already runs the complete audit before it can succeed.
    for stage in ['analyze']:
        state(stage, valid_specific=count, expected=2400)
        subprocess.run([str(PYTHON), 'scripts/verify_cross_vlm_20261112.py', stage], cwd=ROOT, check=True)
    state('writing_report', valid_specific=count, expected=2400)
    subprocess.run([str(PYTHON), 'scripts/report_cross_vlm_20261112.py'], cwd=ROOT, check=True)
    state('complete', valid_specific=count, expected=2400,
          report=str(ROOT / 'docs/实验方案与结果/跨VLM核验结果_20261112.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-pid', type=int, required=True)
    args = parser.parse_args()
    run(args.annotation_pid)
