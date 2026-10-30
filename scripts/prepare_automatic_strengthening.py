#!/usr/bin/env python3
"""Hash-sampled, prediction-blind Open Images pool and bounded E reserve."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from automatic_strengthening_common import *


def order_key(*parts):
    return hashlib.sha256(':'.join(map(str, (20261005, 'automaticD', *parts))).encode()).hexdigest()


def historical():
    marker = BASE / 'audit/historical.json'
    if marker.exists():
        value = json.loads(marker.read_text())
        return set(value['ids']), set(value['image_hashes']), set(value['crop_hashes']), value['reference']
    paths = set()
    for base in (ROOT, PROJECT):
        for path in (base / 'artifacts').glob('**/*.csv'):
            if BASE in path.parents:
                continue
            if any(token in path.name for token in ('manifest', 'candidate', 'train', 'test', 'eval', 'input', 'pool', 'target', 'mapping')):
                paths.add(path)
        paths.update((base / 'data').glob('**/manifests/*.csv'))
        paths.update((base / 'data').glob('**/manifest.csv'))
    ids = set(); image_hashes = set(); crop_hashes = set(); crops = {}; provenance = []
    for path in sorted(paths):
        try:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if not set(reader.fieldnames or []) & {'image_id', 'sample_id', 'image_path'}:
                    continue
                for row in reader:
                    for field in ('image_id', 'sample_id', 'image_path'):
                        value = row.get(field) or ''
                        ids.update(re.findall(r'(?<![0-9a-f])([0-9a-f]{16})(?![0-9a-f])', value.lower()))
                    if row.get('image_sha256'): image_hashes.add(row['image_sha256'])
                    if row.get('crop_sha256'): crop_hashes.add(row['crop_sha256'])
                    for field in ('crop_path', 'crop_image_path'):
                        image = Path(row.get(field) or '/nonexistent')
                        if image.is_file():
                            resolved = str(image.resolve())
                            crops[resolved] = dict(sample_id=resolved, crop_path=resolved, label=0, class_name='reference')
            provenance.append(dict(path=str(path), sha256=sha256(path)))
        except (OSError, UnicodeError, csv.Error):
            continue
    note('historical_audit', manifests=len(provenance), forbidden_openimages_ids=len(ids), reference_crops=len(crops))
    # Content hashes deduplicate paths shared by historical manifests.
    unique = {}
    for row in crops.values():
        digest = sha256(row['crop_path']); crop_hashes.add(digest); unique.setdefault(digest, row)
    reference = sorted(unique.values(), key=lambda r: r['sample_id'])
    write_json(BASE / 'audit/historical_manifest_provenance.json', provenance)
    write_json(marker, dict(ids=sorted(ids), image_hashes=sorted(image_hashes), crop_hashes=sorted(crop_hashes), reference=reference))
    return ids, image_hashes, crop_hashes, reference


def metadata():
    marker = BASE / 'audit/availability.json'
    if marker.exists(): return json.loads(marker.read_text())
    used, _, _, _ = historical()
    old = PROJECT / 'data/openimages_v7_dog_gap_v1/annotations/train_five_classes.csv'
    from prepare_openimages_v7_experiment import image_targets
    filters = dict(confidence=1, is_group_of=0, is_depiction=0, is_inside=0, minimum_area_ratio=.001)
    candidates = image_targets(read_csv(old), {v:k for k,v in MID.items() if k != 'car'}, filters)
    available = {name:sum(r['ImageID'] not in used for r in candidates[name]) for name in CLASSES if name != 'car'}
    if any(number < 400 for number in available.values()):
        value=dict(status='insufficient', available=available, failed_classes=[c for c,n in available.items() if n<400],
            minimum=400, prediction_read=False, split='train', car_availability='not_audited',
            cached_annotation_sha256=sha256(old), source_substitution=False)
        write_json(marker,value);note('metadata',operation='insufficient_before_new_predictions',**available)
        return value
    car = BASE / 'metadata/train_car.csv'
    if not car.exists():
        # The full CSV is streamed once; only car records are retained locally.
        url = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
        retained = []
        note('metadata', operation='stream_official_train_boxes_for_car', url=url)
        with urllib.request.urlopen(url,timeout=120) as response:
            # A proxy may return only a byte range even without a Range request.
            if response.status!=200 or response.headers.get('Content-Range'):
                raise RuntimeError('Official annotation response is partial; no availability claim from truncated CSV')
            for index,row in enumerate(csv.DictReader(line.decode('utf-8') for line in response),1):
                if row['LabelName']==MID['car']:retained.append(row)
                if index%100000==0:note('metadata',scanned=index,retained_car_boxes=len(retained))
        write_csv(car, retained)
        write_json(BASE / 'metadata/car_source.json', dict(url=url, split='train', class_mid=MID['car']))
    candidates['car'] = image_targets(read_csv(car), {MID['car']:'car'}, filters)['car']
    available = {name:sum(r['ImageID'] not in used for r in candidates[name]) for name in CLASSES}
    allocation = sorted(CLASSES, key=lambda c:(available[c], c)); claimed = set(used); plan = []; counts = {}
    for name in allocation:
        choices = sorted((r for r in candidates[name] if r['ImageID'] not in claimed), key=lambda r:order_key(name, r['ImageID']))
        counts[name] = len(choices)
        if len(choices) < 400:
            write_json(marker, dict(status='insufficient', available=available, after_class_allocation=counts, failed_class=name,
                                   minimum=400, prediction_read=False, split='train'))
            raise ValueError(f'{name}: {len(choices)} unseen eligible train images; need 400. No source substitution.')
        for rank, row in enumerate(choices[:1100], 1):
            plan.append({**row, 'selection_rank':rank}); claimed.add(row['ImageID'])
    write_csv(BASE / 'metadata/target_plan.csv', plan)
    value = dict(status='pass', available=available, after_class_allocation=counts, allocation=allocation,
                 filters=filters, split='train', prediction_read=False,
                 sources={str(old):sha256(old), str(car):sha256(car)}, plan_sha256=sha256(BASE / 'metadata/target_plan.csv'))
    write_json(marker, value); note('metadata', operation='availability_pass', available=available)
    return value


def prepare():
    marker = D / 'preparation_audit.json'
    if marker.exists(): return
    availability = metadata()
    if availability['status'] != 'pass': raise ValueError('Availability audit did not pass')
    used, images, digests, reference = historical()
    copy_setup(D); copy_setup(E)
    weak = B / 'transfer/weak_states_A_frozen.csv'
    (D / 'transfer').mkdir(parents=True, exist_ok=True); shutil.copy2(weak, D / 'transfer/weak_states_A_frozen.csv')
    import torch
    torch.set_num_threads(8)
    import sixclass_car_support as support
    support.configure()
    b = support.benchmark
    from audit_multiclass_near_duplicates import phash, hamming_matrix
    from prepare_multiclass_object_candidate_pool import ensure_views
    from prepare_openimages_v7_experiment import manifest_row
    note('prepare', operation='historical_ViT_embeddings', reference_crops=len(reference))
    ref = b.candidate_embeddings({'rows':reference}, BASE / 'audit/reference_vit.npz', 4)
    ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    ph = np.asarray([phash(Path(r['crop_path'])) for r in reference], dtype=np.uint64)
    cfg = dict(classes=MID, class_labels=LABELS, experiment_id='openimages-automatic-confirmation-D-20261005')
    plan = read_csv(BASE / 'metadata/target_plan.csv'); accepted = []; accepted_v = []; accepted_p = []; excluded = []; counts = Counter()
    checkpoint = BASE / 'audit/materialized_checkpoint.csv'
    if checkpoint.exists():
        # Resume from cached accepted vectors, never re-evaluate acceptance from predictions.
        accepted = read_csv(checkpoint)
        with np.load(BASE / 'audit/materialized_checkpoint.npz') as saved:
            assert saved['sample_ids'].tolist() == [r['sample_id'] for r in accepted]
            accepted_v = list(saved['embeddings']); accepted_p = saved['phashes'].tolist()
        counts.update(r['class_name'] for r in accepted)
        excluded = read_csv(BASE / 'audit/exclusions.csv') if (BASE / 'audit/exclusions.csv').exists() else []
        for r in accepted: images.add(r['image_sha256']); digests.add(r['crop_sha256'])
    attempted = {r.get('image_id') for r in excluded} | {r['image_id'] for r in accepted}
    def materialize(target):
        iid = target['ImageID']; path = BASE / 'images/train' / f'{iid}.jpg'
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://open-images-dataset.s3.amazonaws.com/train/{iid}.jpg'
                last = None
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(url, timeout=60) as response, path.with_suffix('.part').open('wb') as stream:
                            shutil.copyfileobj(response, stream)
                        path.with_suffix('.part').replace(path); break
                    except OSError as error: last = error
                if not path.exists(): raise OSError(type(last).__name__)
            row = manifest_row(target, cfg, 'train', 'automatic_confirmation', 1)
            row.update(sample_id=f'oiD-train-{iid}-{target["class_name"]}', image_path=str(path.resolve()),
                       target_class_id=MID[target['class_name']], selection_rank=target['selection_rank'])
            row = ensure_views(row, BASE)
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original); box = json.loads(row['bbox_norm_xyxy'])
                row.update(image_width=image.width, image_height=image.height,
                           target_pixel_width=(box[2]-box[0])*image.width, target_pixel_height=(box[3]-box[1])*image.height)
            return row, None
        except Exception as error:
            return None, dict(image_id=iid, class_name=target['class_name'], reason=f'download_or_input_error:{type(error).__name__}')
    for name in availability['allocation']:
        options = [r for r in plan if r['class_name'] == name and r['ImageID'] not in attempted]
        for offset in range(0, len(options), 48):
            if counts[name] >= 760: break
            note('prepare', operation='download_and_deduplicate', class_name=name, accepted=dict(counts), offset=offset)
            with ThreadPoolExecutor(max_workers=12) as executor: outputs = list(executor.map(materialize, options[offset:offset+48]))
            rows = []
            for row, error in outputs:
                if error: excluded.append(error); continue
                if row['image_sha256'] in images or row['crop_sha256'] in digests:
                    excluded.append(dict(image_id=row['image_id'], class_name=name, reason='historical_exact_hash')); continue
                rows.append(row)
            if not rows: continue
            vectors = b.candidate_embeddings({'rows':rows}, BASE / f'audit/batches/{name}_{offset}_{order_key(*[r["image_id"] for r in rows])[:12]}.npz', 4)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            hashes = np.asarray([phash(Path(r['crop_path'])) for r in rows], dtype=np.uint64)
            cosine = vectors @ ref.T; distance = hamming_matrix(hashes, ph)
            bad = ((distance<=4)&(cosine>=.980)) | ((distance<=8)&(cosine>=.995))
            for i, row in enumerate(rows):
                if counts[name] >= 760: break
                reject = bool(bad[i].any())
                if accepted:
                    cc = np.asarray(accepted_v) @ vectors[i]
                    dd = hamming_matrix(hashes[i:i+1], np.asarray(accepted_p, dtype=np.uint64))[0]
                    reject |= bool((((dd<=4)&(cc>=.980))|((dd<=8)&(cc>=.995))).any())
                if reject:
                    excluded.append(dict(image_id=row['image_id'], class_name=name, reason='near_duplicate_threshold')); continue
                accepted.append(row); accepted_v.append(vectors[i]); accepted_p.append(int(hashes[i])); counts[name]+=1
                images.add(row['image_sha256']); digests.add(row['crop_sha256'])
            write_csv(checkpoint, accepted); write_csv(BASE / 'audit/exclusions.csv', excluded)
            np.savez_compressed(BASE / 'audit/materialized_checkpoint.npz', sample_ids=np.asarray([r['sample_id'] for r in accepted]),
                                embeddings=np.asarray(accepted_v), phashes=np.asarray(accepted_p, dtype=np.uint64))
        if counts[name] < 400: raise ValueError(f'{name}: only {counts[name]} unique targets; D incomplete')
    lookup = {r['sample_id']:v for r,v in zip(accepted, accepted_v)}; candidates = []; reserve = []; suite = []
    for name in CLASSES:
        rows = sorted((r for r in accepted if r['class_name']==name), key=lambda r:int(r['selection_rank']))
        candidates.extend({**r,'role':'candidate'} for r in rows[:400])
        reserve.extend({**r,'role':'technical_reserve'} for r in rows[400:460])
        suite.extend({**r,'role':'E_candidate'} for r in rows[460:760])
    assert len(candidates)==2400 and len({r['image_id'] for r in accepted})==len(accepted)
    write_csv(D / 'manifests/candidate_pool_initial.csv', candidates); write_csv(D / 'manifests/candidate_pool.csv', candidates)
    write_csv(D / 'manifests/technical_reserves.csv', reserve); write_csv(E / 'manifests/reservoir.csv', suite)
    np.savez_compressed(D / 'audit/accepted_vit.npz', sample_ids=np.asarray([r['sample_id'] for r in candidates+reserve]),
                        embeddings=np.asarray([lookup[r['sample_id']] for r in candidates+reserve]))
    from independent_confirmation_common import freeze
    files = [PROTOCOL, BASE / 'registration.json', BASE / 'metadata/target_plan.csv', BASE / 'audit/historical_manifest_provenance.json',
             D / 'manifests/candidate_pool_initial.csv', D / 'manifests/technical_reserves.csv', E / 'manifests/reservoir.csv',
             D / 'transfer/weak_states_A_frozen.csv', D / 'manifests/train.csv', D / 'manifests/eval.csv']
    files += list((D / 'configs').glob('*')) + list((D / 'prompts').glob('*'))
    freeze(D / 'preparation_freeze.sha256.json', files)
    write_json(marker, dict(status='pass', targets=2400, per_class=400, reserve_counts=dict(Counter(r['class_name'] for r in reserve)),
        E_reservoir_counts=dict(Counter(r['class_name'] for r in suite)), reference_crops=len(reference), prediction_read=False,
        source='Open Images train, unused images; source previously participated in A discovery', manual_review=False))
    note('prepare', operation='complete', targets=2400, E_candidates=len(suite))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('metadata','prepare'), default='prepare')
    globals()[parser.parse_args().stage]()
