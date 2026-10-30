#!/usr/bin/env python3
"""Durable, single-owner full annotation -> audited selection workflow.

This supervisor never treats a partial VLM batch as a finished experiment.
Run it detached with stdout/stderr redirected to workflow_supervisor.log.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def plan(args):
    root = args.root.resolve()
    py = str(ROOT / '.venv/bin/python')
    scripts = ROOT / 'scripts'
    def command(name, *argv):
        return [py, '-u', str(scripts / name), *map(str, argv)]
    stages = [
        dict(name='full_annotations', command=command(args.annotation_runner.name,
             '--phase', 'annotate', '--workers', '2', '--force-fresh')),
        dict(name='independent_annotation_audit', command=command(
             'audit_qe_full_fine_annotations.py', '--input', root)),
        dict(name='selection_evaluation_report', command=command(
             'run_qe_full_fine_selection.py', '--root', root, '--phase', 'all',
             '--coverage-aggregation', 'class_macro', '--support-cap', '20',
             '--random-repeats', '1000', '--budgets', '20', '40', '100', '200',
             '400', '600', '800', '--predictions', args.predictions.resolve())),
        dict(name='independent_selection_audit', command=command(
             'audit_qe_full_fine_selection.py', '--input', root)),
    ]
    names = {
        args.annotation_runner.name, Path(__file__).name,
        'audit_qe_full_fine_annotations.py', 'audit_qe_full_fine_selection.py',
        'run_qe_full_fine_selection.py', 'export_qe_full_fine_feature_scores.py',
        'run_fresh_vlm_feature_scoring.py', 'run_category_five_dimension_pilot.py',
        'run_dog_local_pipeline.py', 'build_category_dimension_review.py',
        'build_qe_visualstate_coverrisk_dataset.py',
        'run_qe_three_strategy_full_experiment.py',
        'summarize_qe_three_strategy_full_experiment.py',
    }
    paths = [scripts / name for name in sorted(names)]
    paths += [args.predictions.resolve(), root / 'protocol.json',
              root / 'prompt.txt', root / 'scoring_protocol.json',
              root / 'input_freeze.sha256.json', args.smoke_review.resolve()]
    freeze = {str(p.resolve()): sha(p) for p in paths}
    return dict(version='qe_full_fine_supervised_v1', root=str(root),
        stages=stages, frozen_dependencies=freeze,
        semantic_smoke_review=str(args.smoke_review.resolve()),
        full_annotation_required=7800, max_annotation_passes=3,
        user_approved_ratios=dict(balanced=[.2] * 5,
            challenge=[.1, .15, .2, .25, .3]),
        class_balance='same budget in each of six classes',
        primary_total_budget=1200,
        primary_budget_origin='context default, not separately specified by user',
        inference='real local VLM; classifier predictions reused only after identity audit',
        completion='all full gates and independent selection audit pass',
        pending_policy='semantic failures remain pending; collecting other targets is allowed; no partial selection or fabricated unknown labels')


def check_freeze(value):
    for path, expected in value['frozen_dependencies'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Frozen dependency changed: {path}')


class Supervisor:
    def __init__(self, root, specification):
        self.root = root
        self.specification = specification
        self.child = None
        self.state = dict(status='running', complete=False, pid=os.getpid(),
                          started_at=now(), stages_completed=[])

    def save(self, **fields):
        self.state.update(fields, updated_at=now())
        annotation = self.root / 'progress.json'
        if annotation.exists():
            progress = read(annotation)
            self.state['annotation_progress'] = {
                key: progress.get(key) for key in (
                    'successful', 'planned', 'pending', 'status',
                    'fresh_targets_per_minute', 'eta_seconds', 'updated_at')}
        atomic(self.root / 'workflow_progress.json', self.state)

    def event(self, **fields):
        row = dict(updated_at=now(), supervisor_pid=os.getpid(), **fields)
        with (self.root / 'workflow_execution.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def stop(self, signum, frame):
        if self.child and self.child.poll() is None:
            self.child.terminate()
        self.save(status='interrupted', complete=False, signal=signum)
        raise SystemExit(128 + signum)

    def stage(self, value, attempt=1):
        check_freeze(self.specification)
        name = value['name']
        log = self.root / 'workflow_logs' / f'{name}_attempt{attempt}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        self.event(stage=name, attempt=attempt, event='started',
                   command=value['command'], log=str(log))
        with log.open('a') as stream:
            self.child = subprocess.Popen(value['command'], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
            self.save(stage=name, attempt=attempt, child_pid=self.child.pid,
                      current_log=str(log))
            while self.child.poll() is None:
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.save()
            code = self.child.returncode
        self.child = None
        self.event(stage=name, attempt=attempt, event='finished', returncode=code)
        self.save(child_pid=None)
        if code:
            raise RuntimeError(f'{name} exited with code {code}; see {log}')
        check_freeze(self.specification)

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        try:
            for value in self.specification['stages']:
                if value['name'] == 'full_annotations':
                    for attempt in range(1, self.specification['max_annotation_passes'] + 1):
                        self.stage(value, attempt)
                        audit = read(self.root / 'annotation_audit.json')
                        if audit['status'] == 'pass' and audit['valid'] == 7800:
                            break
                        self.event(stage=value['name'], event='incomplete_retry',
                                   valid=audit['valid'], missing=audit['missing'])
                    else:
                        raise RuntimeError('Full annotation still incomplete after three resumable passes')
                else:
                    self.stage(value)
                self.state['stages_completed'].append(value['name'])
                self.save()
            # The final independent CLI must emit this full-size verdict.
            independent = read(self.root / 'selection_analysis/independent_selection_audit.json')
            if independent['status'] != 'pass':
                raise RuntimeError('Independent selection audit did not pass')
            self.save(status='complete', stage='complete', complete=True,
                finished_at=now(), report=str(self.root / 'selection_analysis/report.docx'))
            self.event(event='workflow_complete')
        except Exception as exc:
            self.save(status='needs_repair', complete=False, error=str(exc))
            self.event(event='workflow_failed', error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--annotation-runner', type=Path, required=True)
    parser.add_argument('--smoke-review', type=Path, required=True)
    parser.add_argument('--predictions', type=Path,
        default=ROOT / 'artifacts/QE_ThreeStrategy_Full_20261013/merged_predictions.npz')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        review = read(args.smoke_review)
        allowed = review['status'] == 'pass' or (
            review['status'] == 'pass_with_pending'
            and review.get('annotation_collection_allowed') is True
            and review.get('full_selection_allowed') is False)
        if not allowed:
            raise RuntimeError('Independent review has not approved retained-label collection')
        specification = plan(args)
        destination = root / 'workflow_plan.json'
        if destination.exists():
            if read(destination) != specification:
                raise RuntimeError('Existing workflow plan differs; preserve and version the plan')
        else:
            atomic(destination, specification)
        if args.prepare_only:
            print(json.dumps(dict(status='prepared', plan=str(destination)), ensure_ascii=False))
            return
        Supervisor(root, specification).run()


if __name__ == '__main__':
    main()
