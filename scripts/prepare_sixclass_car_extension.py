#!/usr/bin/env python3
"""Extend frozen five-class manifests with image-disjoint COCO/OI car targets."""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/hcf/project/QE')
OUT = ROOT / 'artifacts/sixclass_car_extension_20261004'
OLD_POOL = ROOT / 'artifacts/multiclass_object_pool_v1_20260928'
TRAIN = SOURCE / 'artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv'
EVAL = SOURCE / 'artifacts/resnet18_coco5_dog496_vs_dog596_20260916/data/eval_fixed.csv'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
LABELS = {name: index for index, name in enumerate(CLASSES)}
SEED = 20261004
REMOVED = {'head_visibility', 'head_orientation', 'pose_action', 'leg_visibility', 'truncation_parts'}

sys.path.insert(0, str(SOURCE / 'scripts'))
from run_resnet18_dog_coverage_comparison import crop_target, download_one as download_coco, parse_labels
from prepare_openimages_v7_experiment import absolute_box, image_targets, manifest_row, stream_rows
from prepare_multiclass_object_candidate_pool import (
    download_one as download_oi, ensure_views, read_csv, sha256, stratified_select,
    write_csv, write_json,
)


def create_schemas(output: Path) -> None:
    schema = json.loads((ROOT / 'configs/multiclass_object_state_schema_v1_0.json').read_text())
    shared = {**schema, 'schema_version': 'shared_object_state_v1.0',
              'universal_features': [item for item in schema['universal_features'] if item['feature_id'] not in REMOVED]}
    write_json(output / 'configs/shared_object_schema.json', shared)
    car = {**shared, 'schema_version': 'car_shared_state_v1.0', 'category': 'car',
           'category_specific_features': [
               {'feature_id': 'car_color', 'group': 'car_appearance', 'value_type': 'enum',
                'possible_values': ['white', 'black', 'gray_or_silver', 'red', 'blue', 'green', 'yellow_or_orange', 'brown', 'multicolor', 'other', 'unknown']},
               {'feature_id': 'wheel_visibility', 'group': 'car_parts', 'value_type': 'enum',
                'possible_values': ['zero', 'one', 'two', 'three_or_more', 'unknown']},
               {'feature_id': 'visible_vehicle_parts', 'group': 'car_parts', 'value_type': 'set',
                'possible_values': ['front', 'rear', 'side_body', 'roof', 'windshield', 'windows', 'wheels', 'lights', 'unknown'],
                'exclusive_values': ['unknown']},
               {'feature_id': 'reflective_surface', 'group': 'car_appearance', 'value_type': 'enum',
                'possible_values': ['minimal', 'moderate', 'strong', 'unknown']},
           ]}
    write_json(output / 'configs/car_annotation_schema.json', car)


def car_coco_targets(split: str, forbidden: set[str], count: int) -> list[dict]:
    candidates = []
    with zipfile.ZipFile(SOURCE / 'data/coco2017/coco2017labels.zip') as archive:
        prefix = f'coco/labels/{split}/'
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith('.txt'):
                continue
            image_id = Path(name).stem
            if image_id in forbidden:
                continue
            boxes = [line for line in parse_labels(archive.read(name)) if len(line) == 5 and int(line[0]) == 2]
            if boxes:
                candidates.append({'image_id': image_id, 'target': max(boxes, key=lambda box: float(box[3]) * float(box[4]))})
    candidates.sort(key=lambda row: sha_key(split, row['image_id']))
    if len(candidates) < count:
        raise ValueError(f'{split} car: need {count}, got {len(candidates)}')
    return candidates[:count]


def sha_key(*parts) -> str:
    import hashlib
    return hashlib.sha256((':'.join(map(str, (SEED, *parts)))).encode()).hexdigest()


def coco_rows(targets: list[dict], split: str, output: Path, workers: int) -> list[dict]:
    images = output / 'data/coco_images' / split
    images.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_coco, row['image_id'], split, images) for row in targets]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f'[car COCO download] {split} {index}/{len(futures)}', flush=True)
    rows = []
    for row in targets:
        path = output / 'data/coco_crops' / split / f"{row['image_id']}.jpg"
        if not path.is_file():
            crop_target(images / f"{row['image_id']}.jpg", row['target'], path)
        rows.append({'sample_id': f"car_{row['image_id']}", 'image_id': row['image_id'],
                     'crop_path': str(path.resolve()), 'label': 5, 'class_name': 'car',
                     'source': split, 'yolo_class_id': 2, 'target_class_id': 2,
                     'bbox_area_ratio': float(row['target'][3]) * float(row['target'][4]),
                     'image_path': str((images / f"{row['image_id']}.jpg").resolve()),
                     'image_sha256': sha256(images / f"{row['image_id']}.jpg"), 'crop_sha256': sha256(path)})
    return rows


def official_car_annotations(output: Path) -> tuple[str, list[dict]]:
    target = output / 'data/annotations/openimages_test_car.csv'
    descriptions_url = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv'
    with urllib.request.urlopen(descriptions_url, timeout=90) as response:
        descriptions = list(csv.reader(io.StringIO(response.read().decode('utf-8'))))
    matches = [row[0] for row in descriptions if len(row) == 2 and row[1] == 'Car']
    if len(matches) != 1:
        raise ValueError('official Car class ID is not unique')
    mid = matches[0]
    if target.is_file():
        return mid, read_csv(target)
    url = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'
    retained = []
    for index, row in enumerate(stream_rows(url), 1):
        if row['LabelName'] == mid:
            retained.append(row)
        if index % 100000 == 0:
            print(f'[Open Images car boxes] scanned={index} retained={len(retained)}', flush=True)
    write_csv(target, retained)
    write_json(output / 'data/annotations/source.json', {'class_mid': mid, 'class_name': 'Car',
               'box_url': url, 'descriptions_url': descriptions_url})
    return mid, retained


def prepare(output: Path, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    create_schemas(output)
    base_train, base_eval = read_csv(TRAIN), read_csv(EVAL)
    base_candidates = read_csv(OLD_POOL / 'manifests/candidate_pool.csv')
    if len(base_train) != 1500 or len(base_candidates) != 2000:
        raise ValueError('frozen five-class counts changed')
    used = {row['image_id'] for row in base_train + base_eval + base_candidates}
    car_train = coco_rows(car_coco_targets('train2017', used, 300), 'train2017', output, workers)
    used.update(row['image_id'] for row in car_train)
    car_eval = coco_rows(car_coco_targets('val2017', used, 65), 'val2017', output, workers)
    for index, row in enumerate(car_eval):
        row['split'] = 'val' if index < 15 else 'test'
    train = [*base_train, *car_train]
    evaluation = [*base_eval, *car_eval]
    write_csv(output / 'manifests/train.csv', train)
    write_csv(output / 'manifests/eval.csv', evaluation)
    forbidden_hashes = {sha256(Path(row['crop_path'])) for row in train + evaluation}
    forbidden_hashes.update(row['crop_sha256'] for row in base_candidates)
    mid, annotations = official_car_annotations(output)
    filters = {'confidence': 1, 'is_group_of': 0, 'is_depiction': 0, 'is_inside': 0, 'minimum_area_ratio': 0.001}
    targets = image_targets(annotations, {mid: 'car'}, filters)['car']
    for row in targets:
        row['_source_split'] = 'test'
    used.update(row['image_id'] for row in evaluation)
    selected, stages = stratified_select(targets, 'car', 400, used)
    reserve = sorted([row for row in targets if row['ImageID'] not in used], key=lambda row: sha_key('reserve', row['ImageID']))
    config = {'classes': {'car': mid}, 'class_labels': LABELS, 'experiment_id': 'sixclass-car-extension-v1'}
    exclusions, cars = [], []
    def materialize(target):
        item = manifest_row(target, config, 'test', 'car_candidate', 1)
        _, status = download_oi(item)
        if status.startswith('error:'):
            return None, status
        item['image_path'] = str((SOURCE / 'data/openimages_v7_dog_gap_v1/images/test' / f"{item['image_id']}.jpg").resolve())
        return ensure_views(item, output), ''
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(materialize, target): target for target in selected}
        for index, future in enumerate(as_completed(futures), 1):
            target = futures[future]
            item, error = future.result()
            if error or item['crop_sha256'] in forbidden_hashes:
                exclusions.append({'image_id': target['ImageID'], 'reason': error or 'duplicate_crop_hash'})
            else:
                forbidden_hashes.add(item['crop_sha256'])
                cars.append(item)
            if index % 50 == 0:
                print(f'[car candidate] processed={index}/400 accepted={len(cars)}', flush=True)
    for target in reserve:
        if len(cars) == 400:
            break
        item, error = materialize(target)
        if error or item['crop_sha256'] in forbidden_hashes:
            exclusions.append({'image_id': target['ImageID'], 'reason': error or 'duplicate_crop_hash'})
            continue
        forbidden_hashes.add(item['crop_sha256'])
        cars.append(item)
    if len(cars) != 400:
        raise ValueError(f'could only prepare {len(cars)} unique car candidates')
    cars.sort(key=lambda row: sha_key(row['sample_id']))
    for index, row in enumerate(cars, 1):
        row.update({'class_candidate_rank': index, 'candidate_source': 'openimages_test_car_extension', 'pool_protocol': 'sixclass_car_extension_v1'})
    candidates = [*base_candidates, *cars]
    write_csv(output / 'manifests/candidate_pool.csv', candidates)
    write_csv(output / 'manifests/car_exclusions.csv', exclusions or [{'image_id': '', 'reason': 'none'}])
    feature_input = []
    for row in cars:
        with Image.open(row['image_path']) as image:
            box = absolute_box(json.loads(row['bbox_norm_xyxy']), image.width, image.height, 0)
        feature_input.append({'image_id': row['sample_id'], 'target_instance_id': 'target_1',
                              'target_class_id': mid, 'target_class': 'car', 'bbox': json.dumps(box),
                              'image_path': row['image_path'], 'marked_image_path': row['marked_image_path'],
                              'crop_image_path': row['feature_crop_path'], 'protocol_version': 'car_shared_state_v1.0'})
    write_csv(output / 'inputs/car_feature_manifest.csv', feature_input)
    partitions = {'train': train, 'val': [row for row in evaluation if row['split'] == 'val'],
                  'base_test': [row for row in evaluation if row['split'] == 'test'], 'candidate': candidates}
    overlap = {}
    for left, left_rows in partitions.items():
        for right, right_rows in partitions.items():
            if left >= right:
                continue
            left_ids = {row['image_id'] for row in left_rows}
            right_ids = {row['image_id'] for row in right_rows}
            intersection = left_ids & right_ids
            overlap[f'{left}_vs_{right}'] = sorted(intersection)
    if any(overlap.values()):
        raise ValueError(f'source image overlap: {overlap}')
    counts = {name: dict(Counter(row['class_name'] for row in rows)) for name, rows in partitions.items()}
    expected = {'train': 300, 'val': 15, 'base_test': 50, 'candidate': 400}
    for name, counter in counts.items():
        if counter != dict.fromkeys(CLASSES, expected[name]):
            raise ValueError(f'unbalanced {name}: {counter}')
    write_json(output / 'data_audit.json', {'status': 'pass', 'counts': counts, 'source_overlaps': overlap,
               'car_selection_stages': stages, 'car_exclusions': exclusions,
               'old_train_sha256': sha256(TRAIN), 'old_pool_sha256': sha256(OLD_POOL / 'manifests/candidate_pool.csv')})
    print(f'[prepare] complete train={len(train)} candidates={len(candidates)} eval={len(evaluation)}', flush=True)


def merge_annotations(output: Path) -> None:
    shared = json.loads((output / 'configs/shared_object_schema.json').read_text())
    fields = [item['feature_id'] for item in shared['universal_features']]
    previous = read_csv(OLD_POOL / 'annotations/results.csv')
    cars = read_csv(output / 'online_car_annotations/results.csv')
    if len(cars) != 400 or any(row['status'] != 'ok' for row in cars):
        raise ValueError('car online annotation incomplete')
    merged = []
    for rows, annotator in ((previous, 'qwen3-vl:30b-a3b-instruct'), (cars, 'sensenova-6.8-flash-lite')):
        for row in rows:
            if any(field not in row for field in fields):
                raise ValueError(f"missing shared field in {row['image_id']}")
            merged.append({'image_id': row['image_id'], 'target_class': row.get('target_class', ''),
                           'status': 'ok', 'annotation_model': annotator, **{field: row[field] for field in fields}})
    write_csv(output / 'annotations/results.csv', merged)
    write_json(output / 'annotations/report.json', {'status': 'complete', 'planned': 2400, 'successful': len(merged),
               'shared_fields': fields, 'annotation_models': {'original_five': 'local Qwen3-VL', 'car': 'online SenseNova'},
               'limitation': 'category and annotator are not fully crossed'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--merge-annotations', action='store_true')
    args = parser.parse_args()
    if args.merge_annotations:
        merge_annotations(args.output)
    else:
        prepare(args.output, args.workers)


if __name__ == '__main__':
    main()
