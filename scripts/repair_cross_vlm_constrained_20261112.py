#!/usr/bin/env python3
"""Re-request unresolved original targets with schema and original-category reminders."""
import base64
import hashlib
import json
import time

import verify_cross_vlm_20261112 as study


def run():
    study.cc.verify_freeze(study.OUT / 'input_freeze.json')
    cfg = study.load_online_config(study.CONFIG)
    jobs = [j for j in study.read(study.OUT / 'specific_jobs.csv') if not
            (study.OUT / 'specific_B' / j['phase'] / j['class_name'] / f"{j['sample_id']}.json").exists()]
    system = ('你是严格按给定字典输出JSON的视觉标注器。只使用schema中的字段和合法enum ID，'
              '不创造新ID。所有语义判据以原字典为准。颜色字段中，清楚可见但未单列的主色，'
              '应按原字典mixed_or_other的“其他主色”判据归类；不要仅因未单列色名而用unknown。'
              '证据不足时仍应使用unknown。不要输出解释。')
    study.dump('constrained_repair_protocol.json', dict(
        targets=[j['sample_id'] for j in jobs], attempts=3, model=cfg['model'],
        change='JSON-schema transport; system-level legal-ID requirement and restatement of original other-color criterion',
        original_semantic_prompt_retained=True, original_evidence_retained=True,
        original_answers_visible=False, classifier_outputs_visible=False,
        values_rewritten_locally=False, code_sha256=study.sha(__file__), system_prompt=system))
    done, failed = [], []
    for job in jobs:
        c, phase, sid = job['class_name'], job['phase'], job['sample_id']
        fields = json.loads(study.dictionary(c).read_text())['features']
        prompt, schema = study.SPEC(c, fields)
        assert hashlib.sha256(prompt.encode()).hexdigest() == job['prompt_sha256']
        reminder = '\n原响应JSON schema（不得创造新枚举ID）：\n' + json.dumps(schema, ensure_ascii=False)
        user_prompt = prompt + reminder
        request_hash = hashlib.sha256(json.dumps(dict(system=system, user=user_prompt),
                                                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        evidence = 'data:image/jpeg;base64,' + base64.b64encode(study.Path(job['evidence_path']).read_bytes()).decode()
        body = {**cfg.get('request_parameters', {}), 'model': cfg['model'], 'temperature': 0,
                'max_tokens': 3000,
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': 'frozen_visual_state', 'strict': True, 'schema': schema}},
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt},
                              {'type': 'image_url', 'image_url': {'url': evidence}}]}]}
        errors = []
        for attempt in range(1, 4):
            tick = time.monotonic()
            try:
                status, text, elapsed, _ = study.post_json(cfg['url'], body,
                    {'Authorization': 'Bearer ' + cfg['api_key']}, 90)
                safe = text.replace(cfg['api_key'], '[REDACTED]')
                raw_path = study.OUT / 'raw_specific' / phase / c / f'{sid}_constrained_{time.time_ns()}_attempt{attempt}.json'
                raw_path.write_text(safe)
                study.cc.dump(raw_path.with_suffix('.meta.json'), dict(status=status,
                    response_format=body['response_format'], request_prompt_sha256=request_hash))
                if status in (401, 402, 403, 404):
                    raise RuntimeError(f'access_error_http_{status}')
                if status != 200:
                    raise ValueError(f'http_{status}')
                outer = json.loads(safe)
                if outer.get('code') == 8:
                    raise ValueError('rpm_exhausted')
                choice = outer['choices'][0]
                assert choice['finish_reason'] == 'stop'
                features = study.unique_json_object(choice['message']['content'])
                study.validate_specific(features, fields)
                dest = study.OUT / 'specific_B' / phase / c / f'{sid}.json'
                assert not dest.exists()
                study.cc.dump(dest, dict(job, features=features, model=cfg['model'],
                    annotated_at=study.cc.old.now(), attempts=attempt, response_model=outer.get('model'),
                    raw_path=str(raw_path), raw_sha256=study.sha(raw_path),
                    request_prompt_sha256=request_hash, system_prompt=system,
                    format_only_reminder=reminder, constrained_repair=True,
                    response_format=body['response_format'], elapsed_seconds=elapsed, usage=outer.get('usage', {})))
                done.append(sid)
                study.event('constrained_repair_success', sample_id=sid, attempt=attempt)
                break
            except Exception as exc:
                error = str(exc).replace(cfg['api_key'], '[REDACTED]')
                errors.append(dict(attempt=attempt, error=error))
                study.dump(f'constrained_repair_errors/{phase}/{sid}.json', dict(job, errors=errors))
                if isinstance(exc, RuntimeError):
                    raise
                # If this compatible endpoint rejects/ignores json_schema, retain
                # the same schema in text and validate the actual model output.
                body['response_format'] = {'type': 'json_object'}
                if attempt < 3:
                    time.sleep(60 if '429' in error or 'rpm' in error else 5)
            finally:
                time.sleep(max(0., 3 - (time.monotonic() - tick)))
        else:
            failed.append(sid)
    study.dump('constrained_repair_audit.json', dict(status='pass' if not failed else 'incomplete',
        attempted=len(jobs), completed=len(done), failed=failed, values_rewritten_locally=False))
    if failed:
        raise RuntimeError('Unresolved targets after constrained re-request')


if __name__ == '__main__':
    run()
