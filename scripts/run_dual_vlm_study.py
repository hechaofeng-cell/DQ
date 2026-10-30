#!/usr/bin/env python3
"""Background orchestration for aligned online/local annotation and diagnostics."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

from dual_vlm_common import PYTHON, ROOT, STUDY, job_path, now, prepare, verify_freeze, write_json


def status(stage, **extra):
    value = {'status': 'running', 'stage': stage, 'pid': os.getpid(), 'heartbeat': now(), **extra}
    write_json(STUDY / 'progress.json', value)
    print(f'[{now()}] {stage} {json.dumps(extra, ensure_ascii=False)}', flush=True)


def spawn(name, arguments):
    logs = STUDY / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stream = (logs / f'{name}.log').open('a', encoding='utf-8')
    child = subprocess.Popen([PYTHON, '-u', *arguments], cwd=ROOT, stdout=stream,
                             stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED': '1',
                             'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'})
    write_json(STUDY / f'{name}_process.json', {'pid': child.pid, 'started_at': now(),
               'arguments': arguments, 'log': str(logs / f'{name}.log')})
    return child, stream


def annotation_snapshot():
    output = {}
    for backend in ('local', 'online'):
        count = sum(len(list((job_path(backend, domain) / 'parsed').glob('*.json'))) for domain in ('old', 'car'))
        output[backend] = {'completed': count, 'planned': 2400}
        statuses = {}
        for domain in ('old', 'car'):
            path = job_path(backend, domain) / 'pipeline_status.json'
            if path.exists():
                statuses[domain] = json.loads(path.read_text())
        output[backend]['jobs'] = statuses
    return output


def wait_jobs(jobs, stage):
    while any(child.poll() is None for child, _stream in jobs.values()):
        status(stage, annotations=annotation_snapshot(),
               children={name: {'pid': child.pid, 'exit': child.poll()} for name, (child, _stream) in jobs.items()})
        time.sleep(20)
    for name, (child, stream) in jobs.items():
        stream.close()
        if child.returncode:
            raise RuntimeError(f'{name}: exit {child.returncode}; see logs/{name}.log')


def pipeline():
    STUDY.mkdir(parents=True, exist_ok=True)
    lock = (STUDY / 'pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('another dual-VLM orchestrator is running')
    status('prepare')
    prepare()
    # The paired classifier arrays, weights and all image views are immutable.
    status('verify_frozen_inputs')
    write_json(STUDY / 'freeze_check_before.json', verify_freeze())
    if not (STUDY / 'annotation_smoke_audit.json').exists():
        smoke_jobs = {f'{backend}_{domain}_smoke': spawn(f'{backend}_{domain}_smoke',
                      ['scripts/annotate_dual_vlm.py', '--backend', backend, '--domain', domain, '--smoke'])
                      for backend, domain in (('online', 'old'), ('local', 'car'))}
        wait_jobs(smoke_jobs, 'annotation_smoke')
        reports = {}
        for backend, domain in (('online', 'old'), ('local', 'car')):
            report = json.loads((job_path(backend, domain) / 'report.json').read_text())
            if report['failed'] != 0 or report['successful'] != report['planned'] or report['planned'] != 3:
                raise ValueError(f'annotation smoke failed: {backend}/{domain}: {report}')
            reports[f'{backend}_{domain}'] = report
        write_json(STUDY / 'annotation_smoke_audit.json', {'status': 'pass', 'checked_at': now(), 'reports': reports})
    jobs = {}
    for backend, domain in (('online', 'old'), ('local', 'car')):
        name = f'{backend}_{domain}'
        jobs[name] = spawn(name, ['scripts/annotate_dual_vlm.py', '--backend', backend, '--domain', domain])
    wait_jobs(jobs, 'paired_annotation')
    for backend, domain in (('online', 'old'), ('local', 'car')):
        expected = 2000 if domain == 'old' else 400
        for recovery in range(3):
            report = json.loads((job_path(backend, domain) / 'report.json').read_text())
            if report['successful'] == expected and report['failed'] == 0:
                break
            if recovery == 2:
                raise ValueError(f'annotations incomplete: {backend}/{domain}: {report}')
            name = f'{backend}_{domain}_recovery{recovery + 1}'
            child = spawn(name, ['scripts/annotate_dual_vlm.py', '--backend', backend, '--domain', domain])
            wait_jobs({name: child}, 'annotation_recovery')
    child = spawn('analysis_smoke', ['scripts/analyze_dual_vlm_study.py', '--smoke'])
    wait_jobs({'analysis_smoke': child}, 'analysis_smoke')
    child = spawn('analysis', ['scripts/analyze_dual_vlm_study.py'])
    wait_jobs({'analysis': child}, 'paired_analysis')
    write_json(STUDY / 'freeze_check_after.json', verify_freeze())
    audit = json.loads((STUDY / 'analysis/final_audit.json').read_text())
    if audit['status'] != 'pass':
        raise ValueError('final result audit did not pass')
    write_json(STUDY / 'progress.json', {'status': 'complete', 'stage': 'complete', 'pid': os.getpid(),
               'heartbeat': now(), 'annotations': annotation_snapshot(), 'report': str(STUDY / 'analysis/report_zh.md')})
    print(f'[{now()}] all stages complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    if args.background:
        STUDY.mkdir(parents=True, exist_ok=True)
        # Refuse another live run, including one still loading modules.
        path = STUDY / 'background_process.json'
        if path.exists():
            previous = json.loads(path.read_text())
            try:
                os.kill(previous['pid'], 0)
                cmdline = open(f"/proc/{previous['pid']}/cmdline", 'rb').read()
                if b'run_dual_vlm_study.py' in cmdline:
                    raise RuntimeError(f"background study already running, pid={previous['pid']}")
            except (ProcessLookupError, FileNotFoundError):
                pass
        stream = (STUDY / 'pipeline.log').open('a', encoding='utf-8')
        child = subprocess.Popen([PYTHON, '-u', str(ROOT / 'scripts/run_dual_vlm_study.py')],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        write_json(path, {'pid': child.pid, 'started_at': now(), 'log': str(STUDY / 'pipeline.log')})
        stream.close()
        print(json.dumps({'pid': child.pid, 'log': str(STUDY / 'pipeline.log')}), flush=True)
        return
    try:
        pipeline()
    except Exception as error:
        write_json(STUDY / 'failure.json', {'status': 'failed', 'at': now(), 'type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc()})
        write_json(STUDY / 'progress.json', {'status': 'failed', 'heartbeat': now(), 'pid': os.getpid(),
                   'error': str(error), 'details': str(STUDY / 'failure.json')})
        raise


if __name__ == '__main__':
    main()
