#!/usr/bin/env python3
"""Build an image-disjoint, model-output-blind COCO confirmation pool."""
import csv
import hashlib
import itertools
import json
import re
import shutil
import zipfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from independent_confirmation_common import *
import sixclass_car_support as support
from prepare_multiclass_object_candidate_pool import ensure_views
from audit_multiclass_near_duplicates import phash,hamming_matrix
import sys
sys.path.insert(0,str(PROJECT/'scripts'))
from run_resnet18_dog_coverage_comparison import download_one

def key(*items):return hashlib.sha256(':'.join(map(str,(20261004,'independentB',*items))).encode()).hexdigest()

def historical():
    paths=set()
    for base in (ROOT,PROJECT):
        for p in (base/'artifacts').glob('**/*.csv'):
            if BASE in p.parents:continue
            if any(t in p.name for t in ('manifest','candidate','train','test','eval','input','pool','target','mapping')):paths.add(p)
        paths.update((base/'data').glob('**/manifests/*.csv'))
        paths.update((base/'data').glob('**/manifest.csv'))
    ids=set();images=set();crops={};digests=set();provenance=[]
    for path in sorted(paths):
        try:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream)
                if not set(reader.fieldnames or [])&{'image_id','sample_id','image_path'}:continue
                for row in reader:
                    for field in ('image_id','sample_id','image_path'):
                        ids.update(re.findall(r'(?<!\d)(\d{12})(?!\d)',row.get(field,'') or ''))
                    if row.get('image_sha256'):images.add(row['image_sha256'])
                    if row.get('crop_sha256'):digests.add(row['crop_sha256'])
                    for field in ('crop_path','crop_image_path'):
                        p=Path(row.get(field,'') or '/nonexistent')
                        if p.is_file():crops[str(p.resolve())]=dict(sample_id=str(p.resolve()),crop_path=str(p.resolve()),label=0,class_name='reference')
            provenance.append(dict(path=str(path),sha256=sha256(path)))
        except (UnicodeError,csv.Error,OSError):continue
    write_json(BASE/'audit/historical_manifest_provenance.json',provenance)
    write_json(BASE/'audit/forbidden_coco_ids.json',sorted(ids))
    rows=list(crops.values())
    for row in rows:digests.add(sha256(row['crop_path']))
    return ids,images,digests,rows

def freeze_weak_states():
    import pandas as pd
    records=pd.read_csv(ROOT/'artifacts/sixclass_difficulty_supplement_20261004/tables/state_recall.csv')
    state=records.groupby(['source','class_name','model','feature','value','support'],as_index=False).recall.mean()
    rows=read_csv(A/'manifests/candidate_pool.csv');result=[]
    means={}
    for model in MODELS:
        parts=[]
        for seed in SEEDS:
            with np.load(A/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as cache:
                assert cache['sample_ids'].tolist()==[r['sample_id'] for r in rows]
                parts.append(cache['logits'].argmax(1))
        predictions=np.asarray(parts)
        for name in CLASSES:
            indices=[i for i,r in enumerate(rows) if r['class_name']==name]
            means[model,name]=float((predictions[:,indices]==LABELS[name]).mean())
    for source,model,name in itertools.product(('local','online'),MODELS,CLASSES):
        part=state[(state.source==source)&(state.model==model)&(state.class_name==name)].copy()
        part['class_recall']=means[model,name];part['gap']=part.recall-part.class_recall
        selected=part[(part.support>=20)&(part.gap<=-.10)].sort_values(['gap','feature','value']).head(3)
        result.extend(selected.to_dict('records'))
    write_csv(BASE/'transfer/weak_states_A_frozen.csv',result)
    write_json(BASE/'transfer/selection_rule.json',dict(status='frozen_before_B_predictions',minimum_A_support=20,
              maximum_states_per_class_model_source=3,maximum_recall_gap=-.10,selected=len(result)))

def prepare():
    if (BASE/'preparation_audit.json').exists():return
    note('prepare',operation='historical_exclusion_scan')
    BASE.mkdir(parents=True,exist_ok=True)
    for directory in ('configs','prompts'):
        (BASE/directory).mkdir(exist_ok=True)
        for p in (A/directory).glob('*'):
            if p.is_file():shutil.copy2(p,BASE/directory/p.name)
    (BASE/'manifests').mkdir(exist_ok=True)
    (BASE/'data/images').mkdir(parents=True,exist_ok=True)
    for name in ('train.csv','eval.csv'):shutil.copy2(A/'manifests'/name,BASE/'manifests'/name)
    freeze_weak_states()
    ids,images,digests,reference=historical()
    note('prepare',operation='enumerate_unused_COCO',historical_ids=len(ids),reference_crops=len(reference))
    candidates=defaultdict(list);archive=PROJECT/'data/coco2017/coco2017labels.zip'
    reverse={i:c for c,i in COCO.items()}
    with zipfile.ZipFile(archive) as z:
        for filename in z.namelist():
            if not filename.startswith('coco/labels/train2017/') or not filename.endswith('.txt'):continue
            iid=Path(filename).stem
            if iid in ids:continue
            grouped=defaultdict(list)
            for line in z.read(filename).decode().splitlines():
                vals=line.split()
                if len(vals)==5 and int(vals[0]) in reverse:grouped[reverse[int(vals[0])]].append(list(map(float,vals)))
            for name,boxes in grouped.items():
                target=max(boxes,key=lambda v:v[3]*v[4])
                if target[3]>0 and target[4]>0:candidates[name].append(dict(image_id=iid,class_name=name,target=target))
    available={name:len(rows) for name,rows in candidates.items()}
    allocation=sorted(CLASSES,key=lambda c:(available[c],c));claimed=set();plan=[]
    for name in allocation:
        chosen=[r for r in sorted(candidates[name],key=lambda r:key(name,r['image_id'])) if r['image_id'] not in claimed]
        if len(chosen)<460:raise ValueError(f'Insufficient unseen {name}: {len(chosen)}')
        # Register extra deterministic download/duplicate reserves, not outcome reserves.
        for rank,row in enumerate(chosen[:min(650,len(chosen))],1):row['selection_rank']=rank;plan.append(row);claimed.add(row['image_id'])
    write_json(BASE/'selection_plan.json',dict(available=available,allocation_order=allocation,targets=plan,
               label_archive_sha256=sha256(archive),rules='hash-sorted, largest same-class box, no classifier outputs'))
    import torch
    torch.set_num_threads(8)
    support.configure();b=support.benchmark
    note('prepare',operation='reference_ViT_embeddings',reference_crops=len(reference))
    ref=b.candidate_embeddings({'rows':reference},BASE/'audit/reference_vit.npz',4)
    ref=ref/np.maximum(np.linalg.norm(ref,axis=1,keepdims=True),1e-12)
    ph=np.asarray([phash(Path(r['crop_path'])) for r in reference],dtype=np.uint64)
    accepted=[];accepted_v=[];accepted_p=[];excluded=[];counts=Counter()
    def materialize(target):
        try:
            iid=target['image_id'];name=target['class_name'];image=download_one(iid,'train2017',BASE/'data/images')
            _,cx,cy,w,h=target['target'];box=[max(0,cx-w/2),max(0,cy-h/2),min(1,cx+w/2),min(1,cy+h/2)]
            row=dict(sample_id=f'cocoB-train2017-{iid}-{name}',image_id=iid,class_name=name,label=LABELS[name],
                     source_split='train2017',split='independent_candidate_B',dataset_version='COCO2017',
                     bbox_norm_xyxy=json.dumps(box),bbox_area_ratio=w*h,target_class_id=COCO[name],
                     image_path=str(image.resolve()),selection_rank=target['selection_rank'])
            row=ensure_views(row,BASE)
            with Image.open(image) as im:
                row.update(image_width=im.width,image_height=im.height,target_pixel_width=w*im.width,target_pixel_height=h*im.height)
            return row,None
        except Exception as exc:return None,dict(image_id=target['image_id'],class_name=target['class_name'],reason=str(exc))
    for name in allocation:
        class_plan=[r for r in plan if r['class_name']==name]
        for offset in range(0,len(class_plan),60):
            if counts[name]>=460:break
            batch=class_plan[offset:offset+60]
            note('prepare',operation='download_and_duplicate_audit',class_name=name,accepted=dict(counts),offset=offset)
            with ThreadPoolExecutor(max_workers=12) as executor:outputs=list(executor.map(materialize,batch))
            rows=[]
            for row,error in outputs:
                if error:excluded.append(error);continue
                if row['image_sha256'] in images or row['crop_sha256'] in digests:
                    excluded.append(dict(sample_id=row['sample_id'],reason='exact_historical_hash'));continue
                rows.append(row)
            if not rows:continue
            vectors=b.candidate_embeddings({'rows':rows},BASE/f'audit/batches/{name}_{offset}.npz',4)
            vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            hashes=np.asarray([phash(Path(r['crop_path'])) for r in rows],dtype=np.uint64)
            cosine=vectors@ref.T;dist=hamming_matrix(hashes,ph)
            bad=((dist<=4)&(cosine>=.980))|((dist<=8)&(cosine>=.995))
            for i,row in enumerate(rows):
                if counts[name]>=460:break
                reject=bool(bad[i].any())
                if accepted:
                    cc=np.asarray(accepted_v)@vectors[i];dd=hamming_matrix(hashes[i:i+1],np.asarray(accepted_p,dtype=np.uint64))[0]
                    reject=reject or bool((((dd<=4)&(cc>=.980))|((dd<=8)&(cc>=.995))).any())
                if reject:
                    excluded.append(dict(sample_id=row['sample_id'],reason='near_duplicate_threshold'));continue
                row['role']='candidate' if counts[name]<400 else 'technical_reserve'
                accepted.append(row);accepted_v.append(vectors[i]);accepted_p.append(hashes[i]);counts[name]+=1
                images.add(row['image_sha256']);digests.add(row['crop_sha256'])
        if counts[name]!=460:raise ValueError(f'{name}: only {counts[name]} accepted targets')
    vector_lookup={r['sample_id']:v for r,v in zip(accepted,accepted_v)}
    accepted.sort(key=lambda r:(LABELS[r['class_name']],r['selection_rank']))
    write_csv(BASE/'manifests/accepted_targets_with_reserves.csv',accepted)
    write_csv(BASE/'manifests/candidate_pool_initial.csv',[r for r in accepted if r['role']=='candidate'])
    write_csv(BASE/'manifests/candidate_pool.csv',[r for r in accepted if r['role']=='candidate'])
    write_csv(BASE/'manifests/technical_reserves.csv',[r for r in accepted if r['role']=='technical_reserve'])
    write_csv(BASE/'audit/exclusions.csv',excluded or [dict(reason='none')])
    np.savez_compressed(BASE/'audit/accepted_vit.npz',sample_ids=np.asarray([r['sample_id'] for r in accepted]),
                        embeddings=np.asarray([vector_lookup[r['sample_id']] for r in accepted]))
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/independent_confirmation_common.py',archive,
           BASE/'selection_plan.json',BASE/'manifests/candidate_pool_initial.csv',BASE/'manifests/technical_reserves.csv',
           BASE/'transfer/weak_states_A_frozen.csv',BASE/'manifests/train.csv',BASE/'manifests/eval.csv']
    paths+=list((BASE/'configs').glob('*'))+list((BASE/'prompts').glob('*'))
    paths+=[WEIGHTS/model/'oi_gap_add0'/str(seed)/'best.pt' for model,seed in itertools.product(MODELS,SEEDS)]
    freeze(BASE/'preparation_freeze.sha256.json',paths)
    write_json(BASE/'preparation_audit.json',dict(status='pass',candidate_counts={c:400 for c in CLASSES},reserve_counts={c:60 for c in CLASSES},
               reference_crops=len(reference),historical_coco_ids=len(ids),available_before_class_disjoint=available,
               rejected=len(excluded),independence='new samples, same COCO source as training; not external-source confirmation'))
    note('prepare',operation='complete',targets=2400,reserves=360)

if __name__=='__main__':prepare()
