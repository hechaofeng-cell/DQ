#!/usr/bin/env python3
"""Check actual extracted smoke targets against historical content; not a pool qualification."""
from collections import defaultdict
import json
import numpy as np

import qualify_fresh_images_20261112 as q


def main():
    rows=q.materialize()
    root=q.OUT
    stage=root/"partial_dedup_smoke";stage.mkdir(exist_ok=True)
    if (stage/"summary.json").exists():raise FileExistsError("Smoke already recorded")
    hist={}
    for line in (root/"historical_fingerprints.jsonl").open():
        r=json.loads(line)
        if r["status"]=="ok":hist.setdefault(r["sha256"],r)
    history=list(hist.values());hp=np.array([int(r["phash"],16) for r in history],dtype=np.uint64)
    hd=np.array([int(r[k],16) for r in history for k in ("dhash","dhash_flip")],dtype=np.uint64)
    excluded=defaultdict(list);pairs=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        dh=q.distances([int(r["original_dhash"],16) for r in batch],hd)
        ph=q.distances([int(r["crop_phash"],16) for r in batch],hp)
        for k,r in enumerate(batch):
            i=start+k
            if r["image_sha256"] in hist or r["crop_sha256"] in hist:excluded[i].append("historical_exact_SHA")
            if dh[k].min()<=4:excluded[i].append("historical_dhash_le4")
            if not excluded[i]:
                for j in np.flatnonzero(ph[k]<=8):pairs.append((i,int(j),int(ph[k,j])))
    paths=sorted({rows[i]["crop_path"] for i,_,_ in pairs}|{history[j]["path"] for _,j,_ in pairs})
    q.OUT=stage
    def stage_dump(name,value):
        (stage/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    q.dump=stage_dump
    vectors=q.embeddings(paths) if paths else None;lookup={p:i for i,p in enumerate(paths)}
    confirmed=[]
    for i,j,d in pairs:
        cosine=float(vectors[lookup[rows[i]["crop_path"]]]@vectors[lookup[history[j]["path"]]])
        if q.fixed_near(d,cosine):
            excluded[i].append("historical_phash_and_ViT")
            confirmed.append(dict(sample_id=rows[i]["sample_id"],other_path=history[j]["path"],distance=d,cosine=cosine))
    q.dump("confirmed_crop_pairs.json",confirmed)
    q.dump("summary.json",dict(status="actual_content_smoke_complete",targets=len(rows),historical_unique_contents=len(hist),
        exact_or_dhash_or_crop_rejected=len([i for i,reasons in excluded.items() if reasons]),crop_candidate_pairs=len(pairs),
        confirmed_crop_pairs=len(confirmed),dedup_encoder_images=len(paths),full_pool_qualification=False,
        VLM_calls=0,target_classifier_calls=0,
        caution="Partial source-verified targets only; no threshold or sample tuning follows this smoke."))
    print((stage/"summary.json").read_text(),flush=True)


if __name__=="__main__":main()
