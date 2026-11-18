#!/usr/bin/env python3
"""Paired historical VLM verification; never treats either annotator as truth.

Stages: prepare, annotate [--smoke], analyze, audit. All historical inputs are read-only.
"""
from __future__ import annotations

import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'),
               str(ROOT / 'output/experiments/论文关键结论验证_20261106')]
import common as historical
import run_design_robustness_20261110 as robustness
from annotate_coverage_validation_20261018 import SPEC, VALIDATE
from run_dog_local_pipeline import validate_feature
from qe_quality.dta.clients import load_online_config, post_json
from qe_quality.dta.io import unique_json_object

cc = historical.cc
OUT = ROOT / 'artifacts/cross_vlm_verification_20261112'
BASE = historical.OUT
CLASSES = historical.CLASSES
METHODS = ['R', 'C20', 'Greedy20', 'P20']
SOURCES = {'A': 'qwen3-vl:30b-a3b-instruct', 'B': 'sensenova-6.8-flash-lite'}
POOLS = {'B': ROOT / 'artifacts/sixclass_independent_confirmation_20261004',
         'D': ROOT / 'artifacts/objects365_independent_confirmation_20261005'}
CONFIG = ROOT / 'artifacts/feature_pilot_20260908/api_config.local.json'


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return cc.sha(Path(path))


def dump(name, value):
    cc.dump(OUT / name, value)


def write(name, rows):
    cc.write(OUT / name, rows)


def event(stage, **kw):
    value = dict(time=cc.old.now(), stage=stage, **kw)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    with (OUT / 'research_log.md').open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def manifests():
    return {p: read(BASE / f'{p}_manifest.csv') for p in POOLS}


def dictionary(c):
    return historical.SPEC / 'frozen_dictionaries' / f'{c}.json'


def specific_original(phase, row):
    c, sid = row['class_name'], row['sample_id']
    folder = (historical.SPEC / 'annotations_v2' / c if phase == 'B' else
              historical.EXT / 'annotations/external' / c)
    return folder / f'{sid}_parsed.json', folder / f'{sid}_evidence.jpg'


def validate_specific(features, fields):
    if not isinstance(features, dict) or set(features) != {f['feature_id'] for f in fields}:
        raise ValueError('specific JSON field names differ from the dictionary')
    for field in fields:
        key = field['feature_id']; value = features[key]
        allowed = {v['value_id'] for v in field['values']}
        if field['value_type'] == 'set':
            if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
                raise ValueError(f'{key} must be a nonempty JSON array, including unknown as ["unknown"]')
            if len(value) != len(set(value)) or not set(value) <= allowed:
                raise ValueError(f'{key} has duplicate or illegal array values')
            if len(value) > 1 and set(value) & set(field.get('exclusive_values', ['unknown'])):
                raise ValueError(f'{key} mixes an exclusive value with other values')
        elif not isinstance(value, str) or value not in allowed:
            raise ValueError(f'{key} must be a listed enum string')
    VALIDATE(features, fields)


def native_path(phase, source, row):
    return POOLS[phase] / 'annotations' / ('local' if source == 'A' else 'online') / (
        'car' if row['class_name'] == 'car' else 'old') / 'parsed' / f"{row['sample_id']}.json"


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT / 'protocol.json').exists(), 'Freeze protocol before any outputs'
    rows = manifests()
    expected = {'B': 2400, 'D': 1200}
    paths = {Path(__file__), OUT / 'protocol.json',
             ROOT / 'configs/category_five_dimension_pilot_v1.json',
             Path(historical.__file__), Path(robustness.__file__), Path(cc.__file__),
             ROOT / 'scripts/annotate_other_classes_specific_v2_20261018.py'}
    provenance, jobs, image_paths = [], [], set()
    native_frozen = robustness.native_features()
    input_index = {}
    for phase, pool in POOLS.items():
        for domain in ('old', 'car'):
            # Actual target/view binding is recorded in the original pool input manifests.
            p = pool / 'inputs' / f'{domain}.csv'
            if not p.exists():
                p = pool / 'inputs' / f'{domain}_feature_manifest.csv'
            if p.exists():
                paths.add(p)
                input_index[phase, domain] = {r['image_id']: r for r in read(p)}
            for source in ('local', 'online'):
                p = pool / 'annotations' / source / domain / 'protocol.json'
                if p.exists():
                    paths.add(p)
                    if source == 'online':
                        assert json.loads(p.read_text())['model'] == SOURCES['B']
    for phase, rr in rows.items():
        assert len(rr) == expected[phase] == len({r['sample_id'] for r in rr})
        paths.add(BASE / f'{phase}_manifest.csv')
        for r in rr:
            c, sid = r['class_name'], r['sample_id']
            domain = 'car' if c == 'car' else 'old'
            sp = POOLS[phase] / 'configs' / f'{domain}_annotation_schema.json'
            schema = json.loads(sp.read_text()); paths.add(sp)
            local_features = None
            for source in SOURCES:
                p = native_path(phase, source, r)
                j = json.loads(p.read_text())
                validate_feature(j, {'image_id': sid}, schema)
                if source == 'A':
                    assert j.get('annotation_model', SOURCES['A']) == SOURCES['A']
                    local_features = j['features']
                paths.add(p)
                provenance.append(dict(phase=phase, sample_id=sid, namespace='native',
                                       source=source, model=SOURCES[source], path=str(p), sha256=sha(p)))
            if phase == 'B':
                assert native_frozen[sid] == local_features, ('native baseline mismatch', sid)
            for field, expected_sha in [('image_path', r['image_sha256']), ('crop_path', r['crop_sha256'])]:
                p = Path(r[field]); image_paths.add(p)
                assert sha(p) == expected_sha, (sid, field)
            indexed = input_index.get((phase, domain), {}).get(sid)
            if indexed:
                for k, alt in [('image_path', 'image_path'), ('marked_image_path', 'marked_image_path'),
                               ('feature_crop_path', 'crop_image_path')]:
                    assert sha(r[k]) == sha(indexed[alt]), (sid, k, 'paired view differs')
            if c not in ('dog', 'cat'):
                fields = json.loads(dictionary(c).read_text())['features']
                p, ev = specific_original(phase, r)
                j = json.loads(p.read_text()); VALIDATE(j['features'], fields)
                assert j['sample_id'] == sid and j['category'] == c
                assert j['dictionary_sha256'] == sha(dictionary(c))
                assert j['evidence_sha256'] == sha(ev)
                assert j['original_sha256'] == r['image_sha256'] and j['crop_sha256'] == r['crop_sha256']
                assert j['model'] == SOURCES['A']
                prompt, response_schema = SPEC(c, fields)
                if 'prompt_sha256' in j:
                    assert j['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
                elif phase == 'B':
                    assert (p.parent / 'prompt.txt').read_text() == prompt
                    paths.add(p.parent / 'prompt.txt')
                paths.update([p, ev, dictionary(c)])
                jobs.append(dict(phase=phase, class_name=c, sample_id=sid, evidence_path=str(ev),
                                 evidence_sha256=sha(ev), dictionary_sha256=sha(dictionary(c)),
                                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
                provenance.append(dict(phase=phase, sample_id=sid, namespace='specific',
                                       source='A', model=SOURCES['A'], path=str(p), sha256=sha(p)))
        event('input_phase_checked', phase=phase, targets=len(rr))
    assert not {r['image_sha256'] for r in rows['B']} & {r['image_sha256'] for r in rows['D']}
    for n in ['main_sets.npz', 'thresholds.json', 'reference_all_states.csv']:
        paths.add(BASE / n)
    paths.update(ROOT / 'artifacts/empirical_completion_20261108' / n
                 for n in ['greedy_sets.npz', 'summary_corrected.json'])
    for c in CLASSES:
        for p in ('B', 'D'):
            paths.add(BASE / f'{p}_{c}_matrix.npz')
    paths.update(image_paths)
    freeze = {str(p): sha(p) for p in sorted(paths)}
    cc.freeze(OUT / 'input_freeze.json', {'files': freeze})
    write('annotation_provenance.csv', provenance)
    write('specific_jobs.csv', jobs)
    dump('preparation_audit.json', dict(status='pass', native_paired_targets=3600,
          native_annotation_files=7200, specific_second_model_planned=len(jobs),
          source_A_native_reproduced=True, frozen_files=len(freeze),
          paired_view_input_manifests_found=[list(x) for x in input_index],
          current_labels_are_ground_truth=False))
    event('prepared', paired_native=3600, specific_jobs=len(jobs), frozen_files=len(freeze))


def annotate(smoke=False, workers=4):
    cc.verify_freeze(OUT / 'input_freeze.json')
    cfg = load_online_config(CONFIG)
    assert cfg['model'] == SOURCES['B']
    parameters = {**cfg.get('request_parameters', {}), 'max_tokens': 3000, 'temperature': 0}
    cc.freeze(OUT / 'specific_runtime.json', dict(model=cfg['model'], parameters=parameters,
              schema_policy='same frozen specific dictionary and exact original prompt/evidence',
              attempts=6, workers=workers, old_answers_visible=False,
              start_interval_seconds=3., rate_error_cooldown_seconds=60))
    jobs = read(OUT / 'specific_jobs.csv')
    if smoke:
        jobs = [next(j for j in jobs if j['phase'] == p and j['class_name'] == c)
                for p in POOLS for c in CLASSES if c not in ('dog', 'cat')]
    stop = threading.Event()
    lock = threading.Lock()
    next_start = [0.]
    interval = [3.]

    def rate_cooldown():
        with lock:
            next_start[0] = max(next_start[0], time.monotonic() + 60)

    def one(job):
        phase, c, sid = job['phase'], job['class_name'], job['sample_id']
        dest = OUT / 'specific_B' / phase / c / f'{sid}.json'
        fields = json.loads(dictionary(c).read_text())['features']
        prompt, schema = SPEC(c, fields)
        array_fields = [f['feature_id'] for f in fields if f['value_type'] == 'set']
        format_reminder = ('\nJSON格式补充（不改变字典语义）：下列set字段必须输出非空JSON数组，'
                           '包括无法判断时写["unknown"]而不是字符串"unknown"：' +
                           json.dumps(array_fields, ensure_ascii=False) + '。其他字段使用枚举字符串。')
        if dest.exists():
            j = json.loads(dest.read_text()); VALIDATE(j['features'], fields)
            assert all(j[k] == job[k] for k in ('sample_id', 'evidence_sha256', 'prompt_sha256', 'dictionary_sha256'))
            assert j['model'] == SOURCES['B']
            return 'cached'
        if stop.is_set():
            return 'not_called_after_access_error'
        image = 'data:image/jpeg;base64,' + base64.b64encode(Path(job['evidence_path']).read_bytes()).decode()
        request_prompt = prompt + format_reminder
        body = {**parameters, 'model': cfg['model'], 'response_format': {'type': 'json_object'},
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': request_prompt}, {'type': 'image_url', 'image_url': {'url': image}}]}]}
        errors = []
        for attempt in range(1, 7):
            if stop.is_set():
                break
            while True:
                with lock:
                    delay = max(0., next_start[0] - time.monotonic())
                    if delay <= 0:
                        next_start[0] = time.monotonic() + interval[0]
                        break
                time.sleep(min(delay, 1.))
            try:
                status, text, elapsed, _ = post_json(cfg['url'], body,
                            {'Authorization': 'Bearer ' + cfg['api_key']}, 90)
                safe = text.replace(cfg['api_key'], '[REDACTED]')
                raw = OUT / 'raw_specific' / phase / c / f'{sid}_{time.time_ns()}_attempt{attempt}.json'
                raw.parent.mkdir(parents=True, exist_ok=True); raw.write_text(safe)
                cc.dump(raw.with_suffix('.meta.json'), dict(status=status, elapsed_seconds=elapsed,
                        request_prompt_sha256=hashlib.sha256(body['messages'][0]['content'][0]['text'].encode()).hexdigest()))
                if status in (401, 402, 403, 404):
                    stop.set()
                if status == 429:
                    rate_cooldown()
                if status != 200:
                    raise ValueError(f'http_{status}')
                outer = json.loads(safe)
                if outer.get('error'):
                    err = outer['error']
                    if isinstance(err, dict) and (err.get('code') == '8' or 'rpm exhausted' in err.get('message', '')):
                        rate_cooldown()
                        raise ValueError('rate_limit_global_cooldown')
                    raise ValueError(f'API_error:{str(err)[:200]}')
                choice = outer['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('generation_not_complete')
                features = unique_json_object(choice['message']['content'])
                validate_specific(features, fields)
                cc.dump(dest, dict(job, features=features, model=cfg['model'],
                                  annotated_at=cc.old.now(), attempts=attempt,
                                  response_model=outer.get('model'), raw_sha256=sha(raw),
                                  raw_path=str(raw), request_prompt_sha256=hashlib.sha256(body['messages'][0]['content'][0]['text'].encode()).hexdigest(),
                                  format_only_reminder=format_reminder,
                                  elapsed_seconds=elapsed, usage=outer.get('usage', {})))
                return 'new'
            except Exception as exc:
                errors.append(dict(attempt=attempt, error=str(exc).replace(cfg['api_key'], '[REDACTED]')))
                if str(exc).startswith(('specific JSON', 'generation_not_complete')) or 'must be' in str(exc) or 'exclusive' in str(exc) or 'illegal array' in str(exc):
                    body['messages'][0]['content'][0]['text'] = request_prompt + '\n上次输出格式未通过校验：' + errors[-1]['error'] + '。请重新输出完整JSON。'
                if attempt < 6 and not stop.is_set():
                    time.sleep(min(5 * attempt, 10))
        cc.dump(OUT / 'specific_failures' / phase / f'{sid}.json', dict(job, errors=errors))
        return 'failed'

    counts = defaultdict(int)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, j): j for j in jobs}
        for f in as_completed(futures):
            state = f.result(); counts[state] += 1
            completed = sum(counts.values())
            if smoke or completed % 20 == 0:
                event('annotation_progress', smoke=smoke, processed=completed, total=len(jobs), counts=dict(counts))
                dump('specific_progress.json', dict(smoke=smoke, processed=completed, total=len(jobs), counts=dict(counts)))
    result = dict(status='pass' if counts['failed'] == counts['not_called_after_access_error'] == 0 else 'incomplete',
                  smoke=smoke, planned=len(jobs), counts=dict(counts))
    dump('specific_smoke_audit.json' if smoke else 'specific_annotation_audit.json', result)
    if result['status'] != 'pass':
        raise RuntimeError('Incomplete independent specific-field annotations; retain failures')


def analyze():
    import verify_cross_vlm_analysis_20261112 as analysis
    analysis.run()


def audit():
    import verify_cross_vlm_analysis_20261112 as analysis
    analysis.audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'annotate', 'analyze', 'audit'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'annotate':
        annotate(args.smoke, args.workers)
    else:
        globals()[args.stage]()


if __name__ == '__main__':
    main()
