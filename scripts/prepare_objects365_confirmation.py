#!/usr/bin/env python3
"""Download selected official packages and deduplicate targets before any predictions."""
import itertools
import json
import shutil
import tarfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image
from objects365_confirmation_common import *
from audit_objects365_confirmation_source import download_complete


def prepare():
    marker=BASE/'preparation_audit.json'
    if marker.exists():return
    source=json.loads((BASE/'metadata/source_audit.json').read_text())
    if source['status']!='pass' or source['split']!='train':raise ValueError('Source qualification did not pass')
    import automatic_strengthening_common as helper
    helper.copy_setup(BASE)
    (BASE/'transfer').mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD_B/'transfer/weak_states_A_frozen.csv',BASE/'transfer/weak_states_A_frozen.csv')
    import prepare_automatic_strengthening as historical
    historical.BASE=BASE;historical.note=note
    _,images,digests,reference=historical.historical()
    plan=json.loads((BASE/'metadata/target_plan.json').read_text())
    by_filename={Path(r['file_name']).name:r for r in plan}
    if len(by_filename)!=len(plan):raise ValueError('Repeated source file basename')
    notes=[]
    folder=BASE/'images';folder.mkdir(parents=True,exist_ok=True)

    def materialize_package(record):
        patch=record['name'];package=BASE/'data/packages'/f"{patch.replace('/','_')}.tar.gz"
        wanted={name for name,row in by_filename.items() if ('/'+patch.rsplit('/',1)[-1]+'/') in ('/'+row['file_name'])}
        if not wanted:return
        package.parent.mkdir(parents=True,exist_ok=True)
        note('download_images',package=patch,selected_images=len(wanted),archive_bytes=record['bytes'])
        download=download_complete(record['url'],package,workers=4)
        found=[]
        # Never extract an archive path; only predefined basenames go into the image directory.
        with tarfile.open(package,'r|gz') as container:
            for member in container:
                name=Path(member.name).name
                if not member.isfile() or name not in wanted:continue
                destination=folder/name
                if not destination.exists():
                    temporary=destination.with_suffix('.extracting')
                    with container.extractfile(member) as stream,temporary.open('wb') as out:shutil.copyfileobj(stream,out)
                    assert temporary.stat().st_size==member.size
                    temporary.replace(destination)
                found.append(name)
        value=dict(package=patch,download=download,wanted=len(wanted),found=len(found),missing=sorted(wanted-set(found)))
        write_json(BASE/'audit/packages'/f"{patch.replace('/','_')}.json",value)
        return value

    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(materialize_package,p) for p in source['packages']]):
            value=future.result()
            if value:notes.append(value)
    write_json(BASE/'audit/image_package_audit.json',notes)
    import torch
    torch.set_num_threads(8)
    import sixclass_car_support as support
    support.configure();b=support.benchmark
    from audit_multiclass_near_duplicates import phash,hamming_matrix
    from prepare_multiclass_object_candidate_pool import ensure_views
    reference_mapping={}
    for old in (OLD_B,ROOT/'artifacts/sixclass_voc_external_confirmation_20261005',ROOT/'artifacts/automatic_strengthening_20261005'):
        for path in (old/'audit').glob('*reference*vit*.npz'):
            with np.load(path) as cache:
                if {'sample_ids','embeddings'}<=set(cache.files):
                    reference_mapping.update({str(i):v for i,v in zip(cache['sample_ids'],cache['embeddings'])})
    extra=[r for r in reference if r['sample_id'] not in reference_mapping]
    note('prepare',operation='historical_reference_embeddings',total=len(reference),uncached=len(extra))
    if extra:
        vectors=b.candidate_embeddings({'rows':extra},BASE/'audit/additional_reference_vit.npz',4)
        reference_mapping.update({r['sample_id']:v for r,v in zip(extra,vectors)})
    ref=np.asarray([reference_mapping[r['sample_id']] for r in reference]);ref/=np.maximum(np.linalg.norm(ref,axis=1,keepdims=True),1e-12)
    ph=np.asarray([phash(Path(r['crop_path'])) for r in reference],dtype=np.uint64)
    accepted=[];accepted_v=[];accepted_p=[];excluded=[];counts=Counter()
    checkpoint=BASE/'audit/materialization_checkpoint.csv'
    if checkpoint.exists():
        accepted=read_csv(checkpoint)
        with np.load(BASE/'audit/materialization_checkpoint.npz') as cache:
            assert cache['sample_ids'].tolist()==[r['sample_id'] for r in accepted]
            accepted_v=list(cache['embeddings']);accepted_p=cache['phashes'].tolist()
        excluded=read_csv(BASE/'audit/exclusions.csv')
        counts.update(r['class_name'] for r in accepted)
        for row in accepted:images.add(row['image_sha256']);digests.add(row['crop_sha256'])
    attempted={r['source_image_id'] for r in excluded}|{str(r['source_image_id']) for r in accepted}
    for name in source['allocation_order']:
        options=[r for r in plan if r['class_name']==name and str(r['image_id']) not in attempted]
        for offset in range(0,len(options),40):
            if counts[name]>=460:break
            batch=[]
            for target in options[offset:offset+40]:
                path=folder/Path(target['file_name']).name;iid=target['image_id']
                try:
                    if not path.exists():raise OSError('Image absent in complete source package')
                    with Image.open(path) as image:
                        if (image.width,image.height)!=(target['width'],target['height']):raise ValueError('Official/image dimensions disagree')
                    x,y,w,h=target['bbox'];iw,ih=target['width'],target['height']
                    row=dict(sample_id=f'obj365-train-{iid}-{name}',image_id=f'objects365-v2-train-{iid}',source_image_id=iid,
                        class_name=name,label=LABELS[name],target_class_id=target['category_id'],annotation_id=target['annotation_id'],
                        source_split='Objects365_v2_train_package_prefix',dataset_version='Objects365_v2',split='external_confirmation',
                        bbox_norm_xyxy=json.dumps([x/iw,y/ih,(x+w)/iw,(y+h)/ih]),bbox_area_ratio=target['bbox_area_ratio'],
                        image_path=str(path.resolve()),image_width=iw,image_height=ih,target_pixel_width=w,target_pixel_height=h,
                        selection_rank=target['selection_rank'],source_file_name=target['file_name'])
                    row=ensure_views(row,BASE)
                    if row['image_sha256'] in images or row['crop_sha256'] in digests:
                        excluded.append(dict(source_image_id=iid,class_name=name,reason='historical_exact_hash'));continue
                    batch.append(row)
                except (OSError,ValueError,KeyError) as error:
                    excluded.append(dict(source_image_id=iid,class_name=name,reason='input_error:'+str(error)))
            if not batch:continue
            note('prepare',operation='near_duplicate_check',class_name=name,counts=dict(counts),offset=offset)
            batch_key=sha256(BASE/'metadata/target_plan.json')[:12]
            vectors=b.candidate_embeddings({'rows':batch},BASE/f'audit/batches/{name}_{offset}_{batch_key}.npz',4)
            vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            hashes=np.asarray([phash(Path(r['crop_path'])) for r in batch],dtype=np.uint64)
            cosine=vectors@ref.T;distance=hamming_matrix(hashes,ph)
            bad=((distance<=4)&(cosine>=.980))|((distance<=8)&(cosine>=.995))
            for i,row in enumerate(batch):
                if counts[name]>=460:break
                reject=bool(bad[i].any())
                if accepted:
                    cc=np.asarray(accepted_v)@vectors[i]
                    dd=hamming_matrix(hashes[i:i+1],np.asarray(accepted_p,dtype=np.uint64))[0]
                    reject|=bool((((dd<=4)&(cc>=.980))|((dd<=8)&(cc>=.995))).any())
                if reject:
                    excluded.append(dict(source_image_id=row['source_image_id'],class_name=name,reason='near_duplicate_threshold'));continue
                row['role']='candidate' if counts[name]<400 else 'technical_reserve'
                accepted.append(row);accepted_v.append(vectors[i]);accepted_p.append(int(hashes[i]));counts[name]+=1
                images.add(row['image_sha256']);digests.add(row['crop_sha256'])
            write_csv(checkpoint,accepted);write_csv(BASE/'audit/exclusions.csv',excluded)
            np.savez_compressed(BASE/'audit/materialization_checkpoint.npz',sample_ids=np.asarray([r['sample_id'] for r in accepted]),
                embeddings=np.asarray(accepted_v),phashes=np.asarray(accepted_p,dtype=np.uint64))
        if counts[name]<400:raise ValueError(f'{name}: only {counts[name]} unused targets after duplicates; stop before classification')
    lookup={r['sample_id']:v for r,v in zip(accepted,accepted_v)}
    accepted.sort(key=lambda r:(LABELS[r['class_name']],int(r['selection_rank'])))
    candidates=[r for r in accepted if r['role']=='candidate'];reserves=[r for r in accepted if r['role']=='technical_reserve']
    assert len(candidates)==2400 and len({r['image_id'] for r in accepted})==len(accepted)
    for filename,rows in [('candidate_pool_initial',candidates),('candidate_pool',candidates),('technical_reserves',reserves)]:
        write_csv(BASE/'manifests'/f'{filename}.csv',rows)
    np.savez_compressed(BASE/'audit/accepted_vit.npz',sample_ids=np.asarray([r['sample_id'] for r in accepted]),
                       embeddings=np.asarray([lookup[r['sample_id']] for r in accepted]))
    from independent_confirmation_common import freeze
    paths=[PROTOCOL,BASE/'registration.json',BASE/'metadata/source_audit.json',BASE/'metadata/target_plan.json',
        BASE/'audit/historical_manifest_provenance.json',BASE/'manifests/candidate_pool_initial.csv',BASE/'manifests/technical_reserves.csv',
        BASE/'transfer/weak_states_A_frozen.csv',BASE/'manifests/train.csv',BASE/'manifests/eval.csv',BASE/'audit/accepted_vit.npz',
        *list((BASE/'configs').glob('*')),*list((BASE/'prompts').glob('*'))]
    paths += [WEIGHTS/m/'oi_gap_add0'/str(s)/'best.pt' for m,s in itertools.product(MODELS,SEEDS)]
    freeze(BASE/'preparation_freeze.sha256.json',paths)
    write_json(marker,dict(status='pass',targets=2400,candidate_counts=dict(Counter(r['class_name'] for r in candidates)),
        reserve_counts=dict(Counter(r['class_name'] for r in reserves)),historical_reference_crops=len(reference),
        exclusions=len(excluded),prediction_read=False,manual_review=False,
        independence='new Objects365 release source; image/crop hashes and frozen near-duplicate rules checked; pretraining overlap not excluded'))
    note('prepare',result='pass',targets=2400,reserves=len(reserves))


if __name__=='__main__':prepare()
