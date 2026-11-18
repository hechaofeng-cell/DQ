#!/usr/bin/env python3
"""Continue the already authorized bounded pilot after the development process."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import run_label_protocol_pilot_20261112 as run

PYTHON=run.ROOT/'.venv/bin/python'


def stage(name,**kw):
    value=dict(stage=name,updated_at=run.now(),**kw)
    run.dump(run.OUT/'pipeline_status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def command(script,*args):
    subprocess.run([str(PYTHON),f'scripts/{script}',*args],cwd=run.ROOT,check=True)


def main(pid):
    while True:
        p=Path(f'/proc/{pid}/stat')
        if not p.exists() or p.read_text().split()[2]=='Z':break
        count=sum(1 for _ in (run.OUT/'labels/development').glob('*/*/*.json'))
        stage('development',valid=count,expected=480)
        time.sleep(30)
    stage('development_analysis')
    command('analyze_label_protocol_pilot_20261112.py','--split','development')
    summary=run.OUT/'analysis/development/summary.json'
    stage('freezing_final_prompts')
    command('run_label_protocol_pilot_20261112.py','freeze-final')
    run.freeze(run.OUT/'development_decision.json',dict(
        decision='Keep candidate revised-v1 unchanged; release locked repair check once',
        files={str(summary):run.sha(summary),
               str(run.OUT/'analysis/development/validation.json'):run.sha(run.OUT/'analysis/development/validation.json')},
        no_locked_outputs_examined=True))
    stage('locked_repair_check',expected=480)
    command('run_label_protocol_pilot_20261112.py','annotate','--split','locked_repair_check')
    stage('locked_analysis')
    command('analyze_label_protocol_pilot_20261112.py','--split','locked_repair_check')
    stage('writing_report')
    command('report_label_protocol_pilot_20261112.py')
    stage('complete',report=str(run.ROOT/'docs/实验方案与结果/跨VLM协议修订小试验结果_20261112.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-pid',type=int,required=True)
    args=parser.parse_args();main(args.development_pid)
