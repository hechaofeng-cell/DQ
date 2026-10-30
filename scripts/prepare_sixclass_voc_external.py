#!/usr/bin/env python3
"""Materialize unused, image-disjoint VOC targets before any new predictions."""
import csv
import hashlib
import itertools
import json
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
from voc_external_common import *
common.BASE=BASE;common.PROTOCOL=PROTOCOL
from independent_confirmation_common import CLASSES,LABELS,COCO,MODELS,SEEDS,WEIGHTS,PROJECT,read_csv,write_csv,write_json,sha256,freeze,note
from prepare_multiclass_object_candidate_pool import ensure_views
from audit_multiclass_near_duplicates import phash,hamming_matrix
import prepare_independent_sixclass_pool as old
old.BASE=BASE

def key(*items):return hashlib.sha256(':'.join(map(str,(20261005,'externalVOC',*items))).encode()).hexdigest()

def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'preparation_audit.json').exists():return
    for sub in ('configs','prompts','manifests','data/images','audit','transfer'):
        (BASE/sub).mkdir(parents=True,exist_ok=True)
    for directory in ('configs','prompts'):
        for p in (COCO_B/directory).glob('*'):
            if p.is_file():shutil.copy2(p,BASE/directory/p.name)
    for name in ('train.csv','eval.csv'):shutil.copy2(COCO_B/'manifests'/name,BASE/'manifests'/name)
    shutil.copy2(COCO_B/'transfer/weak_states_A_frozen.csv',BASE/'transfer/weak_states_A_frozen.csv')
    note('prepare',operation='historical_exclusion_scan')
    _,image_hashes,crop_hashes,reference=old.historical()
    provenance=json.loads((BASE/'audit/historical_manifest_provenance.json').read_text());used=set()
    for item in provenance:
        with Path(item['path']).open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                for field in ('image_id','sample_id','image_path','crop_path','crop_image_path'):
                    value=row.get(field,'') or ''
                    used.update(re.findall(r'(?<!\d)(20\d{2}_\d{6})(?!\d)',value))
                    used.update(re.findall(r'(?:VOC2007/JPEGImages/|voc2007[-/]|vocC-)(\d{6})(?!\d)',value,re.I))
    write_json(BASE/'audit/forbidden_voc_ids.json',sorted(used))
    candidates=defaultdict(list)
    archives={p.name:tarfile.open(p) for p in ARCHIVES}
    members={f'{p.name}/{m.name}':(archives[p.name],m) for p in ARCHIVES for m in archives[p.name].getmembers()}
    for name,member in members.items():
        if '/Annotations/' not in name or not name.endswith('.xml'):continue
        stem=Path(name).stem
        if stem in used:continue
        handle,member=member
        root=ET.parse(handle.extractfile(member)).getroot();w=int(root.findtext('size/width','0'));h=int(root.findtext('size/height','0'))
        if min(w,h)<=0:continue
        groups=defaultdict(list)
        for obj in root.findall('object'):
            c=obj.findtext('name')
            if c not in CLASSES:continue
            bb=obj.find('bndbox')
            if bb is None:continue
            box=[max(0,float(bb.findtext('xmin'))-1),max(0,float(bb.findtext('ymin'))-1),
                 min(w,float(bb.findtext('xmax'))),min(h,float(bb.findtext('ymax')))]
            if box[2]>box[0] and box[3]>box[1]:groups[c].append((box,obj))
        for c,targets in groups.items():
            box,obj=max(targets,key=lambda v:(v[0][2]-v[0][0])*(v[0][3]-v[0][1]))
            candidates[c].append(dict(stem=stem,class_name=c,box=box,width=w,height=h,
                 difficult=int(obj.findtext('difficult','0')),truncated=int(obj.findtext('truncated','0')),
                 member=name.replace('/Annotations/','/JPEGImages/').removesuffix('.xml')+'.jpg'))
    available={c:len(candidates[c]) for c in CLASSES};allocation=sorted(CLASSES,key=lambda c:(available[c],c))
    plan=[];claimed=set()
    for c in allocation:
        rows=[r for r in sorted(candidates[c],key=lambda r:key(c,r['stem'])) if r['stem'] not in claimed]
        if len(rows)<N+20:raise ValueError(f'{c}: unused image-disjoint targets {len(rows)} < {N+20}')
        # Each class claims a deterministic reserve of possible candidates; no model outputs are consulted.
        for rank,row in enumerate(rows[:min(len(rows),N+80)],1):row['selection_rank']=rank;plan.append(row);claimed.add(row['stem'])
    write_json(BASE/'selection_plan.json',dict(available=available,allocation_order=allocation,targets=plan,archive_sha256={str(p):sha256(p) for p in ARCHIVES},
               candidate_count_per_class=N,technical_reserve_per_class=20,used_VOC_ids=len(used),prediction_read=False))
    import torch
    torch.set_num_threads(8)
    import sixclass_car_support as support
    support.configure();b=support.benchmark
    # Reuse reference vectors only on exact path IDs; newly referenced historical crops are embedded as well.
    with np.load(COCO_B/'audit/reference_vit.npz') as cache:
        lookup={str(k):v for k,v in zip(cache['sample_ids'],cache['embeddings'])}
    additional=[r for r in reference if r['sample_id'] not in lookup]
    note('prepare',operation='historical_reference_embeddings',reference_crops=len(reference),additional=len(additional))
    if additional:
        values=b.candidate_embeddings({'rows':additional},BASE/'audit/additional_reference_vit.npz',4)
        lookup.update({r['sample_id']:v for r,v in zip(additional,values)})
    ref=np.asarray([lookup[r['sample_id']] for r in reference]);ref/=np.maximum(np.linalg.norm(ref,axis=1,keepdims=True),1e-12)
    hashes=np.asarray([phash(Path(r['crop_path'])) for r in reference],dtype=np.uint64)
    accepted=[];vectors=[];accepted_ph=[];excluded=[];counts=Counter()
    for c in allocation:
        rows=[r for r in plan if r['class_name']==c]
        for offset in range(0,len(rows),40):
            if counts[c]>=N+20:break
            batch=[]
            for target in rows[offset:offset+40]:
                stem=target['stem'];path=BASE/'data/images'/f'{stem}.jpg'
                try:
                    if not path.exists():
                        handle,member=members[target['member']];path.write_bytes(handle.extractfile(member).read())
                    box=target['box'];w=target['width'];h=target['height'];normal=[box[0]/w,box[1]/h,box[2]/w,box[3]/h]
                    row=dict(sample_id=f'vocC-{stem}-{c}',image_id=f'voc2007-{stem}',class_name=c,label=LABELS[c],
                         source_split='VOC2007_trainval_and_test',split='external_confirmation',dataset_version='VOC2007',
                         bbox_norm_xyxy=json.dumps(normal),bbox_area_ratio=(box[2]-box[0])*(box[3]-box[1])/(w*h),
                         target_class_id=COCO[c],image_path=str(path.resolve()),selection_rank=target['selection_rank'],
                         image_width=w,image_height=h,target_pixel_width=box[2]-box[0],target_pixel_height=box[3]-box[1],
                         voc_difficult=target['difficult'],voc_truncated=target['truncated'])
                    row=ensure_views(row,BASE)
                    if row['image_sha256'] in image_hashes or row['crop_sha256'] in crop_hashes:
                        excluded.append(dict(sample_id=row['sample_id'],reason='historical_exact_hash'));continue
                    batch.append(row)
                except (OSError,ValueError,KeyError) as error:excluded.append(dict(stem=stem,reason=str(error)))
            if not batch:continue
            note('prepare',operation='duplicate_audit',class_name=c,accepted=dict(counts),offset=offset)
            vv=b.candidate_embeddings({'rows':batch},BASE/f'audit/batches/{c}_{offset}.npz',4)
            vv/=np.maximum(np.linalg.norm(vv,axis=1,keepdims=True),1e-12)
            pp=np.asarray([phash(Path(r['crop_path'])) for r in batch],dtype=np.uint64)
            cc=vv@ref.T;dd=hamming_matrix(pp,hashes)
            bad=((dd<=4)&(cc>=.980))|((dd<=8)&(cc>=.995))
            for i,row in enumerate(batch):
                if counts[c]>=N+20:break
                rejected=bool(bad[i].any())
                if accepted:
                    cc2=np.asarray(vectors)@vv[i];dd2=hamming_matrix(pp[i:i+1],np.asarray(accepted_ph,dtype=np.uint64))[0]
                    rejected|=bool((((dd2<=4)&(cc2>=.980))|((dd2<=8)&(cc2>=.995))).any())
                if rejected:excluded.append(dict(sample_id=row['sample_id'],reason='near_duplicate_threshold'));continue
                row['role']='candidate' if counts[c]<N else 'technical_reserve'
                accepted.append(row);vectors.append(vv[i]);accepted_ph.append(pp[i]);counts[c]+=1
                image_hashes.add(row['image_sha256']);crop_hashes.add(row['crop_sha256'])
        if counts[c]!=N+20:raise ValueError(f'{c}: only {counts[c]} targets after duplicate checks; stop')
    for handle in archives.values():handle.close()
    mapping={r['sample_id']:v for r,v in zip(accepted,vectors)}
    accepted.sort(key=lambda r:(LABELS[r['class_name']],int(r['selection_rank'])))
    main=[r for r in accepted if r['role']=='candidate'];reserves=[r for r in accepted if r['role']=='technical_reserve']
    assert len(main)==TOTAL and len({r['image_id'] for r in accepted})==len(accepted)
    for name,rows in [('candidate_pool_initial',main),('candidate_pool',main),('technical_reserves',reserves),('accepted_targets_with_reserves',accepted)]:
        write_csv(BASE/'manifests'/f'{name}.csv',rows)
    write_csv(BASE/'audit/exclusions.csv',excluded or [dict(reason='none')])
    np.savez_compressed(BASE/'audit/accepted_vit.npz',sample_ids=np.asarray([r['sample_id'] for r in accepted]),embeddings=np.asarray([mapping[r['sample_id']] for r in accepted]))
    files=[PROTOCOL,*ARCHIVES,BASE/'selection_plan.json',BASE/'manifests/candidate_pool_initial.csv',BASE/'manifests/technical_reserves.csv',
           BASE/'transfer/weak_states_A_frozen.csv',BASE/'manifests/train.csv',BASE/'manifests/eval.csv',*list((BASE/'configs').glob('*')),*list((BASE/'prompts').glob('*'))]
    files+=[WEIGHTS/m/'oi_gap_add0'/str(s)/'best.pt' for m,s in itertools.product(MODELS,SEEDS)]
    freeze(BASE/'preparation_freeze.sha256.json',files)
    write_json(BASE/'preparation_audit.json',dict(status='pass',targets=TOTAL,candidate_counts={c:N for c in CLASSES},
               reserve_counts={c:20 for c in CLASSES},available_before_class_disjoint=available,reference_crops=len(reference),historical_voc_ids=len(used),
               rejected=len(excluded),independence='unused VOC2007 source images; distinct dataset source from COCO training; pretraining overlap not ruled out'))
    note('prepare',operation='complete',targets=TOTAL,reserves=120)

if __name__=='__main__':prepare()
