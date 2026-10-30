"""Paths and strict preparation for the paired six-class annotation study."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/sixclass_car_extension_20261004'
STUDY = ROOT / 'artifacts/sixclass_dual_vlm_20261004'
OLD = ROOT / 'artifacts/multiclass_object_pool_v1_20260928'
OLD_ONLINE = ROOT / 'artifacts/online_vlm_reannotation_20260929'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
MODELS = ('resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large',
          'densenet121', 'efficientnet_b0', 'convnext_tiny', 'swin_t')
SEEDS = (20260916, 20260917, 20260918)
BUDGETS = (20, 40, 60, 80, 100)
PYTHON = '/home/hcf/project/QE/.venv/bin/python'
LOCAL_MODEL = 'qwen3-vl:30b-a3b-instruct'
ONLINE_MODEL = 'sensenova-6.8-flash-lite'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def schema_path(domain):
    return STUDY / 'configs' / ('old_annotation_schema.json' if domain == 'old' else 'car_annotation_schema.json')


def prompt_path(domain):
    return STUDY / 'prompts' / f'{domain}.txt'


def job_path(backend, domain):
    return STUDY / 'annotations' / backend / domain


def verify_freeze():
    freeze = json.loads((STUDY / 'source_freeze.sha256.json').read_text())
    changed = [path for path, digest in freeze.items() if not Path(path).is_file() or sha256(path) != digest]
    if changed:
        raise ValueError(f'frozen input changed: {changed[:5]}')
    return {'status': 'pass', 'files': len(freeze), 'checked_at': now()}


def prepare():
    from run_dog_local_pipeline import validate_feature

    STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY / 'preparation_audit.json').exists():
        return verify_freeze()
    if json.loads((SOURCE / 'final_audit.json').read_text())['status'] != 'pass':
        raise ValueError('source six-class study is incomplete')
    freeze = {}
    previous = json.loads((SOURCE / 'artifact_manifest.sha256.json').read_text())
    for relative, digest in previous.items():
        path = SOURCE / relative
        if sha256(path) != digest:
            raise ValueError(f'source artifact hash mismatch: {path}')
        freeze[str(path)] = digest
    print(f'[prepare] verified {len(previous)} six-class artifacts', flush=True)
    copies = {
        SOURCE / 'manifests/candidate_pool.csv': STUDY / 'manifests/candidate_pool.csv',
        SOURCE / 'manifests/train.csv': STUDY / 'manifests/train.csv',
        SOURCE / 'manifests/eval.csv': STUDY / 'manifests/eval.csv',
        ROOT / 'configs/multiclass_object_state_schema_v1_0.json': schema_path('old'),
        SOURCE / 'configs/car_annotation_schema.json': schema_path('car'),
        SOURCE / 'configs/shared_object_schema.json': STUDY / 'configs/shared_object_schema.json',
        ROOT / 'prompts/feature_annotation_multiclass_object_v1_0.txt': prompt_path('old'),
        ROOT / 'prompts/feature_annotation_car_shared_v1_0.txt': prompt_path('car'),
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        freeze[str(source)] = sha256(source)
        freeze[str(target)] = sha256(target)
    inputs = {'old': read_csv(OLD / 'inputs/object_state_feature_manifest.csv'),
              'car': read_csv(SOURCE / 'inputs/car_feature_manifest.csv')}
    candidates = read_csv(STUDY / 'manifests/candidate_pool.csv')
    index = {row['sample_id']: row for row in candidates}
    if len(index) != 2400 or Counter(row['class_name'] for row in candidates) != Counter(dict.fromkeys(CLASSES, 400)):
        raise ValueError('candidate balance/identity mismatch')
    identities = set()
    for domain, rows in inputs.items():
        for row in rows:
            key = row['image_id']
            if key in identities or key not in index or row['target_class'] != index[key]['class_name']:
                raise ValueError(f'annotation target mismatch: {key}')
            identities.add(key)
            for field, candidate_field in (('image_path', 'image_path'), ('marked_image_path', 'marked_image_path'),
                                           ('crop_image_path', 'feature_crop_path')):
                path = Path(row[field])
                candidate_path = Path(index[key][candidate_field])
                digest = sha256(path)
                if path != candidate_path and digest != sha256(candidate_path):
                    raise ValueError(f'three-view mismatch: {key}/{field}')
                freeze[str(path)] = digest
            row['split'] = 'candidate'
        manifest = STUDY / 'inputs' / f'{domain}.csv'
        write_csv(manifest, rows)
        freeze[str(manifest)] = sha256(manifest)
    if identities != set(index):
        raise ValueError('missing annotation targets')
    cached = {('local', 'old'): (OLD / 'annotations/parsed/features', inputs['old']),
              ('online', 'old'): (OLD_ONLINE / 'parsed', read_csv(OLD_ONLINE / 'input_manifest.csv')),
              ('online', 'car'): (SOURCE / 'online_car_annotations/parsed', inputs['car'])}
    provenance = []
    for (backend, domain), (directory, source_rows) in cached.items():
        current = {row['image_id']: row for row in inputs[domain]}
        schema = json.loads(schema_path(domain).read_text())
        for row in source_rows:
            key = row['image_id']
            if key not in current:
                raise ValueError(f'cached target absent: {key}')
            for field in ('image_path', 'marked_image_path', 'crop_image_path'):
                if sha256(row[field]) != sha256(current[key][field]):
                    raise ValueError(f'cached annotation views differ: {key}')
            source = directory / f'{key}.json'
            validate_feature(json.loads(source.read_text()), current[key], schema)
            target = job_path(backend, domain) / 'parsed' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = sha256(source)
            freeze[str(source)] = digest
            provenance.append({'backend': backend, 'domain': domain, 'image_id': key,
                               'source_parsed': str(source), 'source_sha256': digest,
                               'destination': str(target), 'cached': True})
    write_csv(STUDY / 'cached_label_provenance.csv', provenance)
    shared = json.loads((STUDY / 'configs/shared_object_schema.json').read_text())
    registration = {
        'registered_at': now(), 'study': 'paired six-class online/local annotation sensitivity',
        'classes': list(CLASSES), 'models': list(MODELS), 'seeds': list(SEEDS),
        'candidates_per_class': 400, 'training_per_class': 300, 'budgets_per_class': list(BUDGETS),
        'frozen_classifier_runs': 24, 'retrain': False, 'random_replicates': 1000,
        'local_model': LOCAL_MODEL, 'online_model': ONLINE_MODEL,
        'native_fields': {'old_five': 19, 'car': 18}, 'shared_fields': 14,
        'shared_field_ids': [item['feature_id'] for item in shared['universal_features']],
        'primary_tests': ['local_coverage_vs_random_equal_error_confusion',
                          'local_coverage_vs_random_equal_error_clusters',
                          'online_coverage_vs_random_equal_error_confusion',
                          'online_coverage_vs_random_equal_error_clusters'],
        'primary_multiplicity': 'Holm across all four paired tests; architecture-level sign flips',
        'uncertainty': '5000 architecture/seed hierarchical bootstrap; seeds are not independent architectures',
        'cross_evaluation': '2 selection label sources x 2 evaluation label sources on same frozen IDs',
        'state_universe': 'per-class union of feasible local and online states; unknown excluded and reported',
        'holdout': 'five disjoint groups; exclude group before selection and evaluate it using opposite annotator',
        'independent_diagnostics': 'predicted confusion classes and frozen classifier-feature error clusters',
        'baselines': ['random', 'deepgini', 'dsa', 'kcenter'],
        'annotations': 'both sources are automatic labels, neither is human ground truth',
        'decision': 'support requires both independent metrics positive with Holm p<.05 for both annotators; otherwise report partial or unstable evidence',
        'limitations': ['historical cached labels and current completions use model aliases at different dates',
                        'native schema/prompt differs by class family but is matched within each paired image',
                        'Ollama constrained JSON decoding and online JSON mode are different transport mechanisms',
                        'same finite candidate pool does not demonstrate universal category generalization'],
        'stopping': 'complete one frozen sweep; retain negative results; stop incomplete on unresolved validation failure',
    }
    write_json(STUDY / 'preregistration.json', registration)
    freeze[str(STUDY / 'preregistration.json')] = sha256(STUDY / 'preregistration.json')
    write_json(STUDY / 'source_freeze.sha256.json', freeze)
    audit = {'status': 'pass', 'created_at': now(), 'candidate_targets': 2400,
             'cached_local': 2000, 'cached_online': 600, 'new_local_planned': 400,
             'new_online_planned': 1800, 'frozen_files': len(freeze), 'three_views_match': True}
    write_json(STUDY / 'preparation_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return audit
