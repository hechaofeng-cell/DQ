#!/usr/bin/env python3
"""Independent identity and immutable prediction audit for merged QE test pool."""
from pathlib import Path
import csv, hashlib, json
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/QE_VisualState_CoverRisk_v1_20261013'
OUT = ROOT / 'artifacts/QE_ThreeStrategy_Test_20261013/independent_prediction_audit'
CLASSES = ('dog','cat','horse','sheep','person','car')
MODELS = ('resnet18','resnet50','mobilenet_v3_small','mobilenet_v3_large','densenet121','efficientnet_b0','convnext_tiny','swin_t')
SEEDS = (20260916,20260917,20260918)

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def lookup_digest(mapping,path):
    suffix=str(path).split('/artifacts/',1)[1]
    values=[v for k,v in mapping.items() if k.endswith('/artifacts/'+suffix)]
    assert len(values)==1,(path,values)
    return values[0]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=rows(DATA/'manifest.csv'); ids=[r['sample_id'] for r in manifest]
    assert len(ids)==7800 and len(set(ids))==7800
    assert Counter(r['class_name'] for r in manifest)==dict.fromkeys(CLASSES,1300)
    pools=json.loads((DATA/'protocol.json').read_text())['source_pools']
    evidence=[]; digest_map={}; provenance=[]
    a_audit=json.loads((Path(pools['A_OpenImages'])/'inference_audit.json').read_text())
    weight_hash={(x['model'],x['seed']):x['checkpoint_sha256'] for x in a_audit['runs']}
    assert set(weight_hash)=={(m,s) for m in MODELS for s in SEEDS}
    for m,s in weight_hash:
        p=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'/m/'oi_gap_add0'/str(s)/'best.pt'
        h=sha(p);assert h==weight_hash[m,s]
        digest_map[str(p)]=h
    first_train=None
    for name,folder in pools.items():
        p=Path(folder); candidates=rows(p/'manifests/candidate_pool.csv')
        cids=[r['sample_id'] for r in candidates]
        merged=[r for r in manifest if r['source_pool']==name]
        assert cids==[r['sample_id'] for r in merged]
        assert len(set(cids))==len(cids)
        for old,new in zip(candidates,merged):
            for k in ('class_name','label','bbox_norm_xyxy','crop_sha256','image_sha256'):
                assert old[k]==new[k],(name,old['sample_id'],k)
            assert int(new['label'])==CLASSES.index(new['class_name'])
        train=rows(p/'manifests/train.csv')
        train_identity=[(r['sample_id'],r['label'],r['crop_sha256']) for r in train]
        if first_train is None:first_train=train_identity
        else:assert train_identity==first_train,(name,'train identity')
        assert not(set(cids)&{r['sample_id'] for r in train})
        assert not({r['crop_sha256'] for r in candidates}&{r['crop_sha256'] for r in train})
        files=list((p/'benchmark/runs').glob('*/*/candidate_outputs.npz'))
        assert {(f.parent.parent.name,int(f.parent.name)) for f in files}==set(weight_hash)
        freeze=json.loads((p/'prediction_freeze.sha256.json').read_text()) if name!='A_OpenImages' else None
        if name!='A_OpenImages':
            prep=json.loads((p/'preparation_freeze.sha256.json').read_text())
            for m,s in weight_hash:
                path=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'/m/'oi_gap_add0'/str(s)/'best.pt'
                assert lookup_digest(prep,path)==weight_hash[m,s]
            inf=json.loads((p/'inference_audit.json').read_text())
            assert inf['weights_retrained'] is False and inf['runs']==24
            provenance.append({'source_pool':name,**inf})
        for f in files:
            m,s=f.parent.parent.name,int(f.parent.name)
            with np.load(f,allow_pickle=False) as data:
                assert data['sample_ids'].tolist()==cids
                logits=data['logits'];assert logits.shape==(len(cids),6) and np.isfinite(logits).all()
                assert data['features'].shape[0]==len(cids)
            actual=sha(f)
            expected=lookup_digest(freeze,f) if freeze else next(x['revised_cache_sha256'] for x in a_audit['runs'] if x['model']==m and x['seed']==s)
            assert actual==expected,(f,'frozen digest mismatch')
            digest_map[str(f)]=actual
            evidence.append(dict(source_pool=name,model=m,seed=s,targets=len(cids),id_order_verified=True,labels_and_crop_identity_verified=True,finite_logits=True,frozen_sha256_verified=True,sha256=actual))
    fields=list(evidence[0])
    with (OUT/'cache_audit.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(evidence)
    for f in [DATA/'manifest.csv',DATA/'protocol.json',Path(__file__)]:digest_map[str(f)]=sha(f)
    (OUT/'input_freeze.sha256.json').write_text(json.dumps(digest_map,indent=2)+'\n')
    report=dict(status='pass',targets=7800,source_pools=4,classification_runs=24,verified_cache_files=96,verified_checkpoints=24,train_targets=len(first_train),all_pool_training_identity_equal=True,candidate_ID_label_crop_bindings_verified=True,all_ID_orders_and_saved_prediction_hashes_verified=True,weights_retrained=False,inference_inputs='original frozen 5% margin target crops; Resize256/CenterCrop224/ImageNet normalization',reuse_valid=True,limitations=['No new classifier inference performed by this audit','Cross-source near duplicate exclusion and VLM semantic truth are not re-established by cache identity checks'])
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (OUT/'external_inference_provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
