#!/usr/bin/env python3
"""Retry only missing independent labels, exposing exact JSON types and enum IDs."""
import base64
import hashlib
import json
import time

import verify_cross_vlm_20261112 as study


def run():
    study.cc.verify_freeze(study.OUT / 'input_freeze.json')
    cfg = study.load_online_config(study.CONFIG)
    assert cfg['model'] == study.SOURCES['B']
    jobs = study.read(study.OUT / 'specific_jobs.csv')
    missing = [j for j in jobs if not (study.OUT / 'specific_B' / j['phase'] /
                j['class_name'] / f"{j['sample_id']}.json").exists()]
    study.dump('format_repair_protocol.json', dict(
        purpose='Retry missing fixed targets only; no annotation-value rewriting or sample replacement',
        change='Append literal frozen response JSON schema, including type and legal enum IDs; no new semantic definitions',
        targets=[j['sample_id'] for j in missing], attempts=3, original_model=cfg['model'],
        original_answers_visible=False, classifier_outputs_visible=False,
        code_sha256=study.sha(__file__)))
    done, failed = [], []
    for job in missing:
        c, sid, phase = job['class_name'], job['sample_id'], job['phase']
        fields = json.loads(study.dictionary(c).read_text())['features']
        prompt, schema = study.SPEC(c, fields)
        assert hashlib.sha256(prompt.encode()).hexdigest() == job['prompt_sha256']
        reminder = '\nJSON输出格式（不改变上述字典定义，请逐字段使用下面的合法枚举ID）：\n' + json.dumps(schema, ensure_ascii=False)
        request_prompt = prompt + reminder
        image = 'data:image/jpeg;base64,' + base64.b64encode(study.Path(job['evidence_path']).read_bytes()).decode()
        body = {**cfg.get('request_parameters', {}), 'model': cfg['model'], 'max_tokens': 3000,
                'temperature': 0, 'response_format': {'type': 'json_object'},
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': request_prompt},
                              {'type': 'image_url', 'image_url': {'url': image}}]}]}
        errors = []
        for attempt in range(1, 4):
            tick = time.monotonic()
            try:
                status, text, elapsed, _ = study.post_json(cfg['url'], body,
                    {'Authorization': 'Bearer ' + cfg['api_key']}, 90)
                safe = text.replace(cfg['api_key'], '[REDACTED]')
                raw_path = study.OUT / 'raw_specific' / phase / c / f'{sid}_formatrepair_{time.time_ns()}_attempt{attempt}.json'
                raw_path.parent.mkdir(parents=True, exist_ok=True); raw_path.write_text(safe)
                if status in (401, 402, 403, 404):
                    raise RuntimeError(f'access_error_http_{status}')
                if status != 200:
                    raise ValueError(f'http_{status}')
                outer = json.loads(safe)
                choice = outer['choices'][0]
                assert choice['finish_reason'] == 'stop'
                features = study.unique_json_object(choice['message']['content'])
                study.validate_specific(features, fields)
                dest = study.OUT / 'specific_B' / phase / c / f'{sid}.json'
                assert not dest.exists(), 'Never overwrite a completed independent answer'
                study.cc.dump(dest, dict(job, features=features, model=cfg['model'],
                    annotated_at=study.cc.old.now(), attempts=attempt, response_model=outer.get('model'),
                    raw_path=str(raw_path), raw_sha256=study.sha(raw_path),
                    request_prompt_sha256=hashlib.sha256(body['messages'][0]['content'][0]['text'].encode()).hexdigest(),
                    format_only_reminder=reminder, format_repair=True, elapsed_seconds=elapsed, usage=outer.get('usage', {})))
                done.append(sid)
                study.event('format_repair_success', sample_id=sid, attempt=attempt)
                break
            except Exception as exc:
                error = str(exc).replace(cfg['api_key'], '[REDACTED]')
                errors.append(dict(attempt=attempt, error=error))
                study.dump(f'format_repair_errors/{phase}/{sid}.json', dict(job, errors=errors))
                if isinstance(exc, RuntimeError):
                    raise
                body['messages'][0]['content'][0]['text'] = request_prompt + '\n上次JSON格式校验错误：' + error + '。必须逐字段使用schema枚举ID。'
                if attempt < 3:
                    time.sleep(60 if '429' in error else 5)
        else:
            failed.append(sid)
        # Serial repair calls avoid overloading the already limited endpoint.
        time.sleep(max(0., 3 - (time.monotonic() - tick)))
    study.dump('format_repair_audit.json', dict(status='pass' if not failed else 'incomplete',
        attempted=len(missing), completed=len(done), failed=failed, old_answers_visible=False))
    if failed:
        raise RuntimeError('Unresolved independent annotations after bounded format-only repair')


if __name__ == '__main__':
    run()
