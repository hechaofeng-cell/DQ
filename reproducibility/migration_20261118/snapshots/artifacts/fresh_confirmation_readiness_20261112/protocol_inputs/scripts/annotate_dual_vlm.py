#!/usr/bin/env python3
"""Resume one annotation branch without changing the paired native schema."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import annotate_multiclass_online_sensenova as online
import run_dog_local_pipeline as local
from dual_vlm_common import (LOCAL_MODEL, ONLINE_MODEL, STUDY, job_path, now, prompt_path,
                             read_csv, schema_path, write_csv, write_json)


def robust_transport(original, prefix):
    def request(*args, **kwargs):
        for attempt in range(1, 4):
            try:
                return original(*args, **kwargs)
            except (OSError, TimeoutError) as error:
                print(f'[{prefix} transport] {attempt}/3 {type(error).__name__}', flush=True)
                if attempt == 3:
                    raise
                time.sleep(5 * attempt)
    return request


def format_reminder(schema):
    arrays = [item['feature_id'] for item in schema['universal_features'] + schema.get('category_specific_features', [])
              if item['value_type'] == 'set']
    return ('\n输出格式约束（不改变视觉状态定义）：回复只能包含一个JSON对象。'
            '\n以下set字段必须使用非空JSON数组：' + json.dumps(arrays, ensure_ascii=False)
            + '\nnone或unknown必须单独出现，例如["none"]或["unknown"]，不能写成标量字符串。'
            '\n全部特征值严格依据原字典，每个字段附简短可见证据，不添加、删除或合并字段。')


def run_online(domain, limit, smoke=False):
    original = online.request_one
    online.post_json = robust_transport(online.post_json, 'online')
    def one(config, prompt, row, schema, raw_dir, limiter):
        if config['model'] != ONLINE_MODEL:
            raise ValueError('online model differs from registration')
        return original(config, prompt + format_reminder(schema), row, schema, raw_dir, limiter)
    online.request_one = one
    manifest = STUDY / 'inputs' / f'{domain}.csv'
    if smoke:
        pending = [row for row in read_csv(manifest) if not (job_path('online', domain) / 'parsed' / f"{row['image_id']}.json").exists()]
        manifest = STUDY / 'inputs' / f'{domain}_online_smoke.csv'
        write_csv(manifest, pending[:3])
    sys.argv = [sys.argv[0], '--input', str(manifest),
                '--output', str(job_path('online', domain)), '--schema', str(schema_path(domain)),
                '--prompt', str(prompt_path(domain)), '--workers', '4', '--start-interval', '3', '--resume']
    if limit:
        sys.argv.extend(['--limit', str(limit)])
    online.main()


def run_local(domain, limit, smoke=False):
    rows = read_csv(STUDY / 'inputs' / f'{domain}.csv')
    if smoke:
        rows = [row for row in rows if not (job_path('local', domain) / 'parsed' / f"{row['image_id']}.json").exists()][:3]
    if limit:
        rows = rows[:limit]
    schema = json.loads(schema_path(domain).read_text())
    compact = {'schema_version': schema['schema_version'], 'category': schema['category'],
               'features': [{key: item.get(key, True) for key in ('feature_id', 'value_type', 'possible_values', 'allow_unknown')}
                            for item in schema['universal_features'] + schema.get('category_specific_features', [])]}
    prompt = prompt_path(domain).read_text() + '\n特征字典如下：\n' + json.dumps(compact, ensure_ascii=False)
    prompt += format_reminder(schema)
    config = {'url': 'http://127.0.0.1:11434/api/chat', 'model': LOCAL_MODEL, 'num_ctx': 8192,
              'num_predict': 2600, 'timeout_seconds': 900, 'max_retries': 3,
              'retry_wait_seconds': 20, 'seed': 20260911}
    local.post = robust_transport(local.post, 'local')
    output = job_path('local', domain)
    (output / 'parsed').mkdir(parents=True, exist_ok=True)
    results, errors, pending = {}, {}, []
    for row in rows:
        path = output / 'parsed' / f"{row['image_id']}.json"
        if path.exists():
            try:
                results[row['image_id']] = local.validate_feature(json.loads(path.read_text()), row, schema)
                continue
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        pending.append(row)
    started = now()
    def progress(state='running', current=''):
        write_json(output / 'pipeline_status.json', {'status': state, 'started_at': started, 'heartbeat': now(),
                   'model': LOCAL_MODEL, 'planned': len(rows), 'completed': len(results), 'failed': len(errors),
                   'pending': len(rows) - len(results) - len(errors), 'current_image': current})
    progress()
    for row in pending:
        key = row['image_id']
        progress(current=key)
        try:
            validation_error = None
            result = None
            for attempt in range(1, 4):
                correction = '\n上一份JSON校验失败：' + validation_error + '。请重新输出完整JSON。' if validation_error else ''
                stamp = time.time_ns()
                candidate = local.ask(config, prompt + correction, row,
                                      [Path(row[field]) for field in ('image_path', 'marked_image_path', 'crop_image_path')],
                                      output / 'raw' / f'{key}_{stamp}_semantic{attempt}.json', local.feature_schema(schema_path(domain)))
                try:
                    result = local.validate_feature(candidate, row, schema)
                    break
                except ValueError as error:
                    validation_error = str(error)
            if result is None:
                raise ValueError(validation_error or 'semantic_validation_failed')
            result.update(annotation_backend='local', annotation_model=LOCAL_MODEL)
            write_json(output / 'parsed' / f'{key}.json', result)
            results[key] = result
            print(f'[local] {len(results) + len(errors)}/{len(rows)} {key} ok', flush=True)
        except Exception as error:
            errors[key] = str(error)
            print(f'[local] {len(results) + len(errors)}/{len(rows)} {key} ERROR {error}', flush=True)
        progress()
    flat = [online.flatten(row, results.get(row['image_id']), errors.get(row['image_id'], '')) for row in rows]
    for row in flat:
        row.update(annotation_backend='local', annotation_model=LOCAL_MODEL)
    write_csv(output / 'results.csv', flat)
    write_json(output / 'errors.json', errors)
    write_json(output / 'report.json', {'model': LOCAL_MODEL, 'schema_version': schema['schema_version'],
               'planned': len(rows), 'successful': len(results), 'failed': len(errors), 'cached': len(rows) - len(pending),
               'inputs': ['original', 'A_marked', 'enlarged_crop'], 'target_label_visible': True})
    progress('needs_retry' if errors else 'complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('local', 'online'), required=True)
    parser.add_argument('--domain', choices=('old', 'car'), required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = job_path(args.backend, args.domain)
    if (output / 'errors.json').exists():
        history = output / 'history' / str(time.time_ns())
        history.mkdir(parents=True, exist_ok=True)
        for name in ('errors.json', 'report.json', 'protocol.json', 'pipeline_status.json'):
            source = output / name
            if source.exists():
                shutil.copyfile(source, history / name)
    (run_local if args.backend == 'local' else run_online)(args.domain, args.limit, args.smoke)


if __name__ == '__main__':
    main()
