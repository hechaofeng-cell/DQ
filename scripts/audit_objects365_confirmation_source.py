#!/usr/bin/env python3
"""Complete metadata audit and deterministic target plan; no model predictions."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time
import tarfile
import shutil
import tempfile
import requests
from dual_vlm_common import CLASSES, now, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/objects365_independent_confirmation_20261005'
REMOTE = 'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/'
ANNOTATION = REMOTE + 'val/zhiyuan_objv2_val.json'
NAMES = dict(zip(CLASSES, ('Dog', 'Cat', 'Horse', 'Sheep', 'Person', 'Car')))
MAX_ARCHIVE_BYTES=80*1024**3


def note(stage, **kwargs):
    item = dict(status='running', stage=stage, pid=os.getpid(), heartbeat=now(), **kwargs)
    # Several package threads and the parent process publish progress concurrently.
    # Each writer owns a temporary file; the destination always contains one full JSON.
    BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=BASE,
                                     prefix='progress.', suffix='.tmp', delete=False) as stream:
        json.dump(item, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        temporary = Path(stream.name)
    try:
        temporary.replace(BASE / 'progress.json')
    finally:
        temporary.unlink(missing_ok=True)
    print(f'[{now()}] {stage} {json.dumps(kwargs, ensure_ascii=False)}', flush=True)


def remote_size(url):
    with requests.get(url + '?qe-size-audit=20261005', headers={'Range':'bytes=0-0'}, stream=True, timeout=45) as response:
        response.raise_for_status()
        value = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes 0-0/(\d+)', value)
        if response.status_code == 206 and match:
            return int(match.group(1))
        if response.status_code == 200 and response.headers.get('Content-Length'):
            return int(response.headers['Content-Length'])
        raise ValueError(f'Unverified remote size/status: {response.status_code} {value}')


def download_complete(url, destination, workers=4):
    """Validated disjoint ranges make HTTP proxy truncation explicit and resumable."""
    size = remote_size(url)
    audit = destination.with_suffix(destination.suffix + '.download.json')
    if destination.exists() and audit.exists():
        saved = json.loads(audit.read_text())
        if destination.stat().st_size == size and saved['sha256'] == sha256(destination):
            return saved
        raise ValueError(f'Changed cached file: {destination}')
    pieces = destination.parent / (destination.name + '.parts')
    pieces.mkdir(parents=True, exist_ok=True)
    block = 8 * 1024 * 1024
    bounds = [(start, min(start + block, size) - 1) for start in range(0, size, block)]

    def fetch(pair):
        start, end = pair
        target = pieces / f'{start:012d}.part'
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        last = None
        for attempt in range(3):
            try:
                # Cache key carries the range; never concatenate responses for other ranges.
                with requests.get(url + f'?qe-confirmation-range={start}-{end}',
                                  headers={'Range':f'bytes={start}-{end}'}, timeout=(30,90), stream=True) as response:
                    response.raise_for_status()
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Unverified range response')
                    temporary = target.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        for content in response.iter_content(1024*1024):
                            stream.write(content)
                    if temporary.stat().st_size != end - start + 1:
                        raise ValueError('Truncated range response')
                    temporary.replace(target)
                    return target
            except (requests.RequestException, OSError, ValueError) as error:
                last = error
                time.sleep(min(attempt + 1, 3))
        raise RuntimeError(f'Range {start}-{end} failed: {type(last).__name__}')

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, pair) for pair in bounds]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            note('source_download', file=destination.name, completed_parts=done, total_parts=len(bounds),
                 elapsed_seconds=round(time.monotonic()-started,1))
    temporary = destination.with_suffix(destination.suffix + '.assembling')
    with temporary.open('wb') as output:
        for start, _ in bounds:
            with (pieces / f'{start:012d}.part').open('rb') as stream:
                while content := stream.read(1024*1024):
                    output.write(content)
    assert temporary.stat().st_size == size
    temporary.replace(destination)
    value = dict(url=url, bytes=size, sha256=sha256(destination), complete=True, downloaded_at=now())
    write_json(audit,value)
    # Keep validated chunks for interrupted-download recovery; no broad deletion.
    return value


def order_key(*values):
    return hashlib.sha256(':'.join(map(str,(20261005,'objects365-v2-val',*values))).encode()).hexdigest()


def audit(split='val'):
    folder = BASE / 'metadata'; folder.mkdir(parents=True,exist_ok=True)
    if split == 'val':
        path = folder / 'zhiyuan_objv2_val.json'
        download = download_complete(ANNOTATION,path)
        data = json.loads(path.read_text())
        category_rows = data['categories']
        image_rows = data['images']
        annotations = iter(data['annotations'])
    else:
        import ijson
        archive = folder/'zhiyuan_objv2_train.tar.gz'
        download = download_complete(REMOTE+'train/zhiyuan_objv2_train.tar.gz',archive)
        path = folder/'zhiyuan_objv2_train.json'
        if not path.exists():
            with tarfile.open(archive) as container:
                members = [m for m in container.getmembers() if m.isfile() and m.name.endswith('.json')]
                if len(members)!=1:raise ValueError('Expected exactly one official train annotation JSON')
                temporary = path.with_suffix('.extracting')
                with container.extractfile(members[0]) as stream,temporary.open('wb') as out:
                    shutil.copyfileobj(stream,out)
                assert temporary.stat().st_size==members[0].size
                temporary.replace(path)
        note('source_metadata',operation='read_train_categories',annotation_bytes=path.stat().st_size)
        with path.open('rb') as stream:category_rows=list(ijson.items(stream,'categories.item',use_float=True))
        with path.open('rb') as stream:image_rows=list(ijson.items(stream,'images.item',use_float=True))
        annotation_stream=path.open('rb')
        annotations=ijson.items(annotation_stream,'annotations.item',use_float=True)
    note('source_metadata', operation='parse_complete_annotations', bytes=download['bytes'])
    categories = {r['name'].casefold():r['id'] for r in category_rows}
    mapping = {name:categories[label.casefold()] for name,label in NAMES.items()}
    reverse = {value:name for name,value in mapping.items()}
    images = {r['id']:r for r in image_rows}
    best = {}; exclusions = Counter()
    annotation_count=0
    for ann in annotations:
        annotation_count+=1
        if annotation_count%1000000==0:note('source_metadata',operation='scan_annotations',records=annotation_count)
        if ann['category_id'] not in reverse: continue
        name = reverse[ann['category_id']]; image = images[ann['image_id']]
        if ann.get('iscrowd',0) or ann.get('ignore',0):
            exclusions['crowd_or_ignore'] += 1; continue
        x,y,w,h = map(float,ann['bbox']); iw,ih = image['width'],image['height']
        if min(w,h,iw,ih) <= 0 or x<0 or y<0 or x+w>iw+1e-3 or y+h>ih+1e-3:
            exclusions['invalid_bbox'] += 1; continue
        area = w*h/(iw*ih)
        if area < .001:
            exclusions['area_below_0.001'] += 1; continue
        key = name,image['id']
        old = best.get(key)
        if old is None or (w*h, -ann['id']) > (old['bbox'][2]*old['bbox'][3],-old['annotation_id']):
            best[key] = dict(class_name=name,image_id=image['id'],file_name=image['file_name'],
                width=iw,height=ih,bbox=[x,y,w,h],bbox_area_ratio=area,annotation_id=ann['id'],category_id=ann['category_id'])
    if split=='train':annotation_stream.close()
    def package(filename):
        match=re.search(r'(?:images/)?(v[12]/patch\d+|patch\d+)/',filename)
        if not match:raise ValueError(f'Unrecognized official image path {filename}')
        return match.group(1)
    full_available=dict(Counter(r['class_name'] for r in best.values()))
    prefix_audit=None
    if split=='train':
        package_names=sorted({package(r['file_name']) for r in best.values()},key=lambda p:order_key('package',p))
        urls={p:REMOTE+'train/'+p.rsplit('/',1)[-1]+'.tar.gz' for p in package_names}
        note('source_metadata',operation='train_package_size_audit',packages=len(urls),counts=full_available)
        with ThreadPoolExecutor(max_workers=8) as executor:sizes=dict(zip(urls,executor.map(remote_size,urls.values())))
        selected=[];total=0;counts=Counter();prefix=[]
        by_package=defaultdict(Counter)
        for r in best.values():by_package[package(r['file_name'])][r['class_name']]+=1
        for p in package_names:
            if total+sizes[p]>MAX_ARCHIVE_BYTES:break
            selected.append(p);total+=sizes[p];counts.update(by_package[p])
            prefix.append(dict(package=p,bytes=sizes[p],cumulative_counts=dict(counts)))
            if all(counts[c]>=600 for c in CLASSES):break
        prefix_audit=dict(rule='SHA256 package order; first prefix with >=600 eligible images per class; <=80GiB',
                          steps=prefix,selected_packages=selected,bytes=total,full_qualified_counts=full_available)
        write_json(folder/'train_package_prefix.json',prefix_audit)
        best={k:r for k,r in best.items() if package(r['file_name']) in selected}
    grouped = defaultdict(list)
    for row in best.values():grouped[row['class_name']].append(row)
    available = {name:len(grouped[name]) for name in CLASSES}
    allocation = sorted(CLASSES,key=lambda name:(available[name],name))
    claimed = set(); plan = []; disjoint_available = {}
    for name in allocation:
        rows = [r for r in sorted(grouped[name],key=lambda r:order_key(name,r['image_id'])) if r['image_id'] not in claimed]
        disjoint_available[name] = len(rows)
        for rank,row in enumerate(rows[:900],1):
            plan.append(dict(**row,selection_rank=rank,order_sha256=order_key(name,row['image_id'])))
            claimed.add(row['image_id'])
    write_json(folder / 'categories.json',category_rows)
    write_json(folder / 'target_plan.json',plan)
    write_csv(folder / 'target_plan.csv',[{**r,'bbox':json.dumps(r['bbox'])} for r in plan])
    patches = {}
    for row in plan:
        filename = row['file_name']
        patch=package(filename)
        patches[patch] = REMOTE + ('val/images/'+patch if split=='val' else 'train/'+patch.rsplit('/',1)[-1]) + '.tar.gz'
    note('source_metadata',operation='audit_required_packages',packages=len(patches),available=available)
    with ThreadPoolExecutor(max_workers=8) as executor:
        sizes = dict(zip(patches,executor.map(remote_size,patches.values())))
    total = sum(sizes.values())
    status = 'pass' if min(disjoint_available.values())>=400 and total<=MAX_ARCHIVE_BYTES else 'insufficient_or_over_budget'
    value = dict(status=status,source=f'Objects365 v2 {split}',split=split,annotation=download,annotation_json_sha256=sha256(path),images=len(images),
        annotations=annotation_count,mapping=mapping,qualified_image_counts=available,train_package_prefix=prefix_audit,
        image_disjoint_available_counts=disjoint_available,allocation_order=allocation,qualified_filters=dict(exclusions),
        plan_count=len(plan),plan_sha256=sha256(folder/'target_plan.json'),packages=[dict(name=k,url=v,bytes=sizes[k]) for k,v in patches.items()],
        required_archive_bytes=total,max_archive_bytes=MAX_ARCHIVE_BYTES,history_dedup_pending=True,
        prediction_read=False,manual_review=False,completed_at=now())
    write_json(folder/f'{split}_source_audit.json',value)
    write_json(folder/'source_audit.json',value)
    note('source_metadata',result=status,counts=available,archive_gib=round(total/1024**3,2))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=('train','val'),default='val')
    try:
        audit(parser.parse_args().split)
    except Exception as error:
        write_json(BASE/'progress.json',dict(status='failed',stage='source_audit',error=str(error),heartbeat=now(),pid=os.getpid()))
        raise
