#!/usr/bin/env python3
"""Freeze a prediction-blind development/repair-check pilot; makes no VLM calls."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/label_protocol_pilot_20261112'
PREVIOUS = ROOT / 'artifacts/cross_vlm_verification_20261112'
BASE = ROOT / 'artifacts/paper_claim_validation_20261106'
PLAN = ROOT / 'docs/实验方案与结果/跨VLM分歧修订与复验方案_20261112.md'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
SEED = 'label-protocol-pilot-20261112-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save_once(path, content):
    if path.exists():
        assert path.read_text() == content, f'Frozen pilot differs: {path}'
    else:
        path.write_text(content)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [BASE / f'{phase}_manifest.csv' for phase in ('B', 'D')]
    inputs += [PREVIOUS / 'risk_per_target.csv', PLAN, Path(__file__).resolve()]
    risks = {(r['phase'], r['sample_id']): int(r['A_Q'])
             for r in read(PREVIOUS / 'risk_per_target.csv')}
    buckets = defaultdict(list)
    for phase in ('B', 'D'):
        for row in read(BASE / f'{phase}_manifest.csv'):
            q = risks[phase, row['sample_id']]
            buckets[phase, row['class_name'], q].append(row)
    chosen, seen_images = [], set()
    for phase in ('B', 'D'):
        per_stratum = 6 if phase == 'B' else 2
        for category in CLASSES:
            for q in range(1, 6):
                candidates = sorted(buckets[phase, category, q], key=lambda row:
                    hashlib.sha256((SEED + '|' + row['sample_id']).encode()).hexdigest())
                selected = [row for row in candidates if row['image_sha256'] not in seen_images][:per_stratum]
                assert len(selected) == per_stratum
                for j, row in enumerate(selected):
                    assert row['image_sha256'] not in seen_images
                    seen_images.add(row['image_sha256'])
                    record = dict(row, phase=phase, old_Q=q,
                                  pilot_split='development' if j < per_stratum // 2 else 'locked_repair_check')
                    for key in ('image_path', 'marked_image_path', 'feature_crop_path'):
                        path = Path(row[key])
                        assert path.is_file(), path
                        record[key + '_sha256'] = sha(path)
                        inputs.append(path)
                    assert record['image_path_sha256'] == row['image_sha256']
                    chosen.append(record)
    assert len(chosen) == len(seen_images) == 240
    assert Counter(r['class_name'] for r in chosen) == dict.fromkeys(CLASSES, 40)
    assert Counter(r['pilot_split'] for r in chosen) == dict(development=120, locked_repair_check=120)
    for split in ('development', 'locked_repair_check'):
        rr = [r for r in chosen if r['pilot_split'] == split]
        assert Counter(r['class_name'] for r in rr) == dict.fromkeys(CLASSES, 20)
        assert Counter(r['phase'] for r in rr) == dict(B=90, D=30)
        assert Counter(r['old_Q'] for r in rr) == dict.fromkeys(range(1, 6), 24)
    import io
    buffer = io.StringIO(newline='')
    columns = list(dict.fromkeys(key for row in chosen for key in row))
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader(); writer.writerows(chosen)
    save_once(OUT / 'manifest.csv', buffer.getvalue())
    protocol = dict(
        status='prepared_only_no_new_vlm_calls', seed=SEED, targets=240,
        development_targets=120, locked_repair_check_targets=120,
        per_class=40, pools=dict(B=180, D=60),
        selection_inputs='class, pool, old Qwen Q tier, target identity and image hashes only',
        global_source_images_unique=True, classifier_outputs_read=False,
        pilot_split_is_new_but_images_are_historical=True, fresh_generalization_test=False,
        intended_comparison='Both VLMs x old semantic protocol versus frozen revised semantic protocol',
        base_requests_if_run=960, semantic_revision_rounds_max=1,
        model_and_prompt_freeze_required_before_calls=True,
        keep_all_disagreements=True, agreement_is_not_accuracy=True)
    save_once(OUT / 'protocol.json', json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    inputs += [OUT / 'manifest.csv', OUT / 'protocol.json']
    freeze = dict(files={str(p): sha(p) for p in sorted(set(inputs))})
    save_once(OUT / 'input_freeze.json', json.dumps(freeze, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
