"""New execution destinations; frozen historical algorithms are not edited."""
from pathlib import Path
import json
import os
from dual_vlm_common import CLASSES, MODELS, SEEDS, read_csv, write_csv, write_json, sha256, now

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/hcf/project/QE')
BASE = ROOT / 'artifacts/automatic_strengthening_20261005'
D = BASE / 'pool_D'
E = BASE / 'suite_E'
A = ROOT / 'artifacts/sixclass_dual_vlm_replaced_20261004'
B = ROOT / 'artifacts/sixclass_independent_confirmation_20261004'
C = ROOT / 'artifacts/sixclass_voc_external_confirmation_20261005'
PROTOCOL = ROOT / 'docs/visual_state_automatic_strengthening_protocol_20261005.md'
PYTHON = PROJECT / '.venv/bin/python'
LABELS = dict(zip(CLASSES, range(6)))
MID = dict(zip(CLASSES, ('/m/0bt9lr', '/m/01yrx', '/m/03k3r', '/m/07bgp', '/m/01g317', '/m/0k4j')))


def note(stage, **extra):
    value = dict(status='running', stage=stage, heartbeat=now(), pid=os.getpid(), **extra)
    write_json(BASE / 'progress.json', value)
    print(f'[{now()}] {stage} {json.dumps(extra, ensure_ascii=False)}', flush=True)


def activate_d():
    import independent_confirmation_common as common
    common.BASE = D
    common.PROTOCOL = PROTOCOL
    import analyze_dual_vlm_study as analysis
    analysis.STUDY = D
    analysis.job_path = lambda backend, domain: D / 'annotations' / backend / domain
    analysis.schema_path = lambda domain: D / 'configs' / ('car_annotation_schema.json' if domain == 'car' else 'old_annotation_schema.json')
    import freeze_independent_sixclass_selection as selection
    selection.BASE = D
    selection.PROTOCOL = PROTOCOL
    import analyze_independent_sixclass as confirmation
    confirmation.BASE = D
    confirmation.PROTOCOL = PROTOCOL
    return common, analysis, selection, confirmation


def copy_setup(destination):
    import shutil
    for directory in ('configs', 'prompts'):
        for path in (A / directory).glob('*'):
            if path.is_file():
                target = destination / directory / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(path, target)
    for name in ('train.csv', 'eval.csv'):
        target = destination / 'manifests' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(A / 'manifests' / name, target)


def annotation_inputs(rows, destination):
    from PIL import Image, ImageOps
    from prepare_multiclass_object_candidate_pool import absolute_box
    for domain in ('old', 'car'):
        output = []
        for row in rows:
            if (row['class_name'] == 'car') != (domain == 'car'):
                continue
            with Image.open(row['image_path']) as original:
                image = ImageOps.exif_transpose(original)
                box = absolute_box(json.loads(row['bbox_norm_xyxy']), image.width, image.height, 0)
            output.append(dict(image_id=row['sample_id'], target_instance_id='target_1', target_class=row['class_name'],
                               target_class_id=row['target_class_id'], bbox=json.dumps(box), image_path=row['image_path'],
                               marked_image_path=row['marked_image_path'], crop_image_path=row['feature_crop_path'],
                               protocol_version='car_shared_state_v1.0' if domain == 'car' else 'object_state_v1.0'))
        write_csv(destination / 'inputs' / f'{domain}.csv', output)


def labels_for(rows, destination):
    from run_dog_local_pipeline import validate_feature
    result = {backend: {} for backend in ('local', 'online')}
    failures = []
    for row in rows:
        domain = 'car' if row['class_name'] == 'car' else 'old'
        schema = json.loads((destination / 'configs' / ('car_annotation_schema.json' if domain == 'car' else 'old_annotation_schema.json')).read_text())
        for backend in result:
            path = destination / 'annotations' / backend / domain / 'parsed' / f"{row['sample_id']}.json"
            try:
                result[backend][row['sample_id']] = validate_feature(json.loads(path.read_text()),
                    dict(image_id=row['sample_id'], target_instance_id='target_1'), schema)['features']
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                failures.append(dict(sample_id=row['sample_id'], backend=backend, class_name=row['class_name']))
    return result, failures
