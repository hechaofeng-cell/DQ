#!/usr/bin/env python3
"""Export small, immutable migration records from the complete local workspace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'reproducibility/migration_20261118'
READY = Path('artifacts/fresh_confirmation_readiness_20261112')
FRESH = Path('artifacts/fresh_image_qualification_20261112')
BASE_URL = 'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def relative(value):
    """Normalize both historical worktree paths and the current project root."""
    prefix = str(ROOT) + '/'
    if value.startswith(prefix):
        return value[len(prefix):]
    match = re.match(r'^/home/hcf/\.codex/worktrees/[^/]+/QE/(.+)$', value)
    if match:
        return match[1]
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def source_catalog():
    return dict(schema_version=1, scope='Latest fresh run and six-class historical pools; old exploratory DTA/Imagenette/head-face branches are not required for this scope', datasets=[
        dict(id='fresh_objects365', name='Objects365 v2 train', official_page='https://www.objects365.org/download.html',
             frozen_originals=5400, qualified_targets=4800, precise_download_manifest='objects365_fresh_downloads.json',
             restore='python3 scripts/restore_qe_data.py download',
             annotation_archive=BASE_URL+'zhiyuan_objv2_train.tar.gz',
             annotation_sha256='d5c10d84863718a332421c2aeaa0563a1dc3d6ce9c4b20f680cb4dd239da8e21',
             scope='Restorer needs frozen image members only, not the full official annotation JSON'),
        dict(id='historical_coco_B', name='COCO 2017 train', frozen_targets=2400,
             manifest='manifests/artifacts/sixclass_independent_confirmation_20261004/manifests/candidate_pool.csv',
             per_image_url='http://images.cocodataset.org/{source_split}/{image_id}.jpg',
             official_annotations='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
             restore='Use exact image_id/source_split/image_sha256 in the manifest; regenerate views using the frozen bbox, never resample',
             existing_code='scripts/prepare_independent_sixclass_pool.py'),
        dict(id='historical_openimages_A', name='Open Images V7 plus 111 VOC2012 sheep targets', frozen_targets=2400,
             manifest='manifests/artifacts/sixclass_dual_vlm_replaced_20261004/manifests/candidate_pool.csv',
             openimages_image_url='https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg',
             split_rule='Use only train/validation/test rows for the Open Images URL; voc2012 rows come from the VOC2012 archive',
             metadata_config='configs/openimages_v7_dog_gap_experiment_v1.json',
             existing_code='scripts/prepare_openimages_v7_experiment.py'),
        dict(id='historical_voc2012', name='PASCAL VOC2012 trainval',
             archive='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
             archive_member='VOCdevkit/VOC2012/JPEGImages/{image_id_without_voc2012_prefix}.jpg',
             existing_code='scripts/prepare_pascal_voc2012_dog_validation.py'),
        dict(id='historical_voc2007_C', name='PASCAL VOC2007 trainval and test', frozen_targets=600,
             manifest='manifests/artifacts/sixclass_voc_external_confirmation_20261005/manifests/candidate_pool.csv',
             checksum_record='snapshots/artifacts/sixclass_voc_external_confirmation_20261005/download_audit.json',
             archives=['https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
                       'https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'],
             existing_code='scripts/download_sixclass_voc_sources.py'),
        dict(id='historical_objects365_D', name='Objects365 v2 train historical confirmation', frozen_targets=2400,
             manifest='manifests/artifacts/objects365_independent_confirmation_20261005/manifests/candidate_pool.csv',
             source_record='snapshots/artifacts/objects365_independent_confirmation_20261005/metadata/source_audit.json',
             archive_url=BASE_URL+'{patch_name}.tar.gz',
             note='These 2400 members are a different pool; the fresh-only restorer does not restore them. Use the source_file_name/package records from this pool.'),
        dict(id='classifier_training', name='Six-class COCO training and validation', train_targets=1800, validation_targets=390,
             manifests=['manifests/artifacts/sixclass_car_extension_20261004/manifests/train.csv',
                        'manifests/artifacts/sixclass_car_extension_20261004/manifests/eval.csv'],
             precise_crop_bindings='training_crops.json',
             note='Originals can be reacquired from COCO, but some old rows lack crop geometry. Transfer the exact crops and the original YOLO-format coco2017labels.zip if preserving historical training inputs.')])


def export(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    # Original byte copies remain historical evidence, not relocated runtime files.
    readiness_names = [
        'alternative_packages.csv', 'alternative_provisional_plan.csv',
        'alternative_supply.json', 'package_size_audit.json',
        'frozen_validation_protocol.json', 'protocol_input_binding.json',
        'checkpoint_lineage.csv', 'training_lineage.json', 'final_freeze.json',
    ]
    sources += [READY / name for name in readiness_names]
    sources += [p.relative_to(ROOT) for p in (ROOT / READY / 'protocol_inputs').rglob('*') if p.is_file()]
    sources += [FRESH / name for name in [
        'qualified_manifest.csv', 'qualification_exclusions.csv', 'qualification_summary.json',
        'independent_data_audit.json', 'dedup_encoder_freeze.json',
        'official_box_technical_exclusions.csv', 'near_duplicate_evidence.csv',
    ]]
    sources += [p.relative_to(ROOT) for p in (ROOT / FRESH / 'package_audits').glob('*.json')]
    historical_manifests = [
        'artifacts/sixclass_car_extension_20261004/manifests/train.csv',
        'artifacts/sixclass_car_extension_20261004/manifests/eval.csv',
        'artifacts/sixclass_dual_vlm_replaced_20261004/manifests/candidate_pool.csv',
        'artifacts/sixclass_independent_confirmation_20261004/manifests/candidate_pool.csv',
        'artifacts/sixclass_voc_external_confirmation_20261005/manifests/candidate_pool.csv',
        'artifacts/objects365_independent_confirmation_20261005/manifests/candidate_pool.csv',
    ]
    sources += list(map(Path, historical_manifests))
    sources += [Path('artifacts/objects365_independent_confirmation_20261005/metadata/source_audit.json'),
                Path('artifacts/sixclass_voc_external_confirmation_20261005/download_audit.json')]
    bindings = []
    for rel in sorted(set(sources)):
        source = ROOT / rel
        dest = output / 'snapshots' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        bindings.append(dict(project_relative_path=rel.as_posix(), bytes=source.stat().st_size,
                             sha256=sha(dest)))

    plan = read(ROOT / READY / 'alternative_provisional_plan.csv')
    packages = []
    for audit_path in sorted((ROOT / FRESH / 'package_audits').glob('*.json')):
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'pass'
        images = []
        for name, record in sorted(audit['found'].items()):
            images.append(dict(name=name, bytes=record['bytes'], sha256=record['sha256'],
                               relative_path=relative(record['path'])))
        packages.append(dict(package=audit['package'], url=BASE_URL + audit['package'].split('/')[-1] + '.tar.gz',
                             bytes=audit['archive_bytes'], sha256=audit['archive_sha256'], images=images))
    assert len(packages) == 18 and sum(len(p['images']) for p in packages) == len(plan) == 5400
    qualified = read(ROOT / FRESH / 'qualified_manifest.csv')
    assert len(qualified) == 4800
    write_json(output / 'objects365_fresh_downloads.json', dict(
        schema_version=1, source='Objects365 v2 train', scope='Frozen 5400 originals, including technical reserves; 4800 qualified targets',
        source_audits='snapshots/' + FRESH.as_posix() + '/package_audits',
        archive_bytes=sum(p['bytes'] for p in packages),
        extracted_bytes=sum(i['bytes'] for p in packages for i in p['images']),
        packages=packages))

    # Portable CSVs use project-relative paths; source snapshots are untouched.
    portable = []
    manifests = [FRESH / 'qualified_manifest.csv', *map(Path, historical_manifests)]
    for rel in manifests:
        rows = read(ROOT / rel)
        fields = list(rows[0])
        for row in rows:
            for key, value in row.items():
                if value and (key.endswith('_path') or key == 'path'):
                    row[key] = relative(value)
        dest = output / 'manifests' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        portable.append(dest.relative_to(output).as_posix())

    checkpoints = []
    for row in read(ROOT / READY / 'checkpoint_lineage.csv'):
        rel = relative(row['path']); path = ROOT / rel
        checkpoints.append(dict(model=row['model'], seed=int(row['seed']), relative_path=rel,
                                bytes=path.stat().st_size, sha256=row['sha256'],
                                acquisition='Transfer the original trained checkpoint; public pretrained weights are not a substitute'))
    write_json(output / 'trained_checkpoints.json', dict(schema_version=1, checkpoints=checkpoints,
        total_bytes=sum(p['bytes'] for p in checkpoints), public_download_available=False))

    crops = {}
    for rel in historical_manifests[:2]:
        for row in read(ROOT / rel):
            name = relative(row['crop_path']); path = ROOT / name
            if name not in crops:
                crops[name] = dict(relative_path=name, bytes=path.stat().st_size, sha256=sha(path))
    labels = ROOT / 'data/coco2017/coco2017labels.zip'
    write_json(output / 'training_crops.json', dict(schema_version=1,
        files=[crops[p] for p in sorted(crops)], total_bytes=sum(p['bytes'] for p in crops.values()),
        labels_archive=dict(relative_path='data/coco2017/coco2017labels.zip', bytes=labels.stat().st_size, sha256=sha(labels)),
        boundary='Exact historical crops, not public pretrained weights. Public originals alone do not reconstruct missing historical crop geometry.'))
    write_json(output / 'data_sources.json', source_catalog())

    protocol = json.loads((ROOT / READY / 'frozen_validation_protocol.json').read_text())
    dedup = json.loads((ROOT / FRESH / 'dedup_encoder_freeze.json').read_text())
    write_json(output / 'models.json', dict(
        annotation=protocol['annotations']['models'],
        annotation_A_digest=protocol['annotations']['frozen_native_model_A_digest'],
        annotation_A_install='ollama pull qwen3-vl:30b-a3b-instruct',
        annotation_B='Online API, configure URL/model/key locally; no credential is included',
        supplemental_C='gemma3:12b; only used in the explicitly separate 2026-11-18 supplemental check',
        public_classifier_backbones={
            'yolo26s_cls': 'Ultralytics/assets/yolo26s-cls.pt',
            'pe_core_b16': 'facebook/PE-Core-B16-224; additionally needs perception_models source code',
            'siglip2_base': 'google/siglip2-base-patch16-224',
            'mobileclip2_s0': 'timm/MobileCLIP2-S0-OpenCLIP/open_clip_pytorch_model.bin; needs open_clip',
            'aimv2_large': 'apple/aimv2-large-patch14-224'},
        dedup_encoder=dict(model=dedup['model'], sha256=dedup['sha256'],
                           url='https://download.pytorch.org/models/vit_b_16-c867db91.pth',
                           cache_relative_to_home='.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth'),
        trained_models='trained_checkpoints.json'))
    transfer = sorted({r['relative_path'] for r in checkpoints} | set(crops) | {'data/coco2017/coco2017labels.zip'} | {
        (FRESH / 'dual_vlm_full_annotation_20261115').as_posix(),
        (FRESH / 'failed_annotation_retry_20261118').as_posix(),
    })
    # Only existing supplemental paths are listed for tar --files-from.
    transfer = [p for p in transfer if (ROOT / p).exists()]
    (output / 'transfer_paths.txt').write_text('\n'.join(transfer) + '\n')
    write_json(output / 'provenance.json', dict(
        schema_version=1, exported_date='2026-11-18',
        historical_project_root=str(ROOT),
        legacy_path_pattern='/home/hcf/.codex/worktrees/*/QE/',
        scope='Latest frozen fresh data, six-class historical pool membership, protocol inputs and 15 trained checkpoint bindings; not every historical dataset',
        snapshots=bindings, portable_manifests=portable,
        boundary='Historical freezes refer to additional local files not all included. Relocated copies do not certify a new independent data audit. No image pixels, API credentials, checkpoints or raw API responses are in this bundle.'))
    inventory = {}
    for p in sorted(output.rglob('*')):
        if p.is_file(): inventory[p.relative_to(output).as_posix()] = sha(p)
    write_json(output / 'manifest.sha256.json', inventory)
    print(json.dumps(dict(output=str(output), snapshots=len(bindings), portable_manifests=len(portable),
                         packages=len(packages), originals=5400, qualified_targets=len(qualified),
                         bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    export(parser.parse_args().output)
