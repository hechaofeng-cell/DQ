#!/usr/bin/env python3
"""Qualify frozen fresh originals using content hashes and fixed near-duplicate rules."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from prepare_fresh_images_20261112 import ROOT, READY, OUT, prepare, read, write, dump, sha, fingerprints
from prepare_multiclass_object_candidate_pool import ensure_views

CLASSES=["dog","cat","horse","sheep","person","car"]


def materialize():
    prepare()
    code={str(Path(__file__)):sha(__file__)}
    freeze=OUT/"qualification_code_freeze.json"
    if freeze.exists():
        if json.loads(freeze.read_text())!=code:raise ValueError("Qualification code changed")
    else:dump(freeze.name,code)
    plan=read(READY/"alternative_provisional_plan.csv")
    packages={p.stem.replace("_","/",1):json.loads(p.read_text()) for p in (OUT/"package_audits").glob("*.json")}
    rows=[]
    def one(r):
        package=packages.get(r["package"])
        if not package or package["status"]!="pass":return None
        sid=f"fresh-obj365-train-{r['source_image_id']}-{r['class_name']}"
        binding=OUT/"bindings"/(sid+".json")
        if binding.exists():return json.loads(binding.read_text())
        actual=package["found"][Path(r["source_file_name"]).name]
        if sha(actual["path"])!=actual["sha256"]:raise ValueError("Extracted original changed")
        box=json.loads(r["bbox_norm_xyxy"])
        if not 0<=box[0]<box[2]<=1.000000001 or not 0<=box[1]<box[3]<=1.000000001:raise ValueError("Invalid official normalized box")
        reserve_for=""
        if r["role"]=="technical_reserve":reserve_for="selection" if int(r["rank"])<=850 else "reference"
        row=dict(sample_id=sid,image_id="objects365-v2-train-"+r["source_image_id"],class_name=r["class_name"],
            label=CLASSES.index(r["class_name"]),target_class_id=int(r["category_id"]),source_image_id=r["source_image_id"],
            source_split="Objects365_v2_train",dataset_version="Objects365_v2",source_file_name=r["source_file_name"],
            annotation_id=r["annotation_id"],package=r["package"],rank=int(r["rank"]),planned_role=r["role"],reserve_for=reserve_for,
            bbox_norm_xyxy=r["bbox_norm_xyxy"],bbox_area_ratio=float(r["bbox_area_ratio"]),
            image_width=int(r["width"]),image_height=int(r["height"]),image_path=actual["path"],image_sha256=actual["sha256"])
        row=ensure_views(row,OUT)
        row["feature_crop_sha256"]=sha(row["feature_crop_path"]);row["marked_image_sha256"]=sha(row["marked_image_path"])
        for kind,path in (("original",row["image_path"]),("crop",row["crop_path"])):
            value=fingerprints(path)
            if value["status"]!="ok":raise ValueError("New image decode failure")
            for name in ("dhash","dhash_flip","phash"):row[kind+"_"+name]=value[name]
        dump("bindings/"+sid+".json",row)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(one,plan):
            if row:rows.append(row)
    if rows:write("materialized_targets.csv",rows,list(rows[0]))
    dump("materialization_progress.json",dict(source_verified_and_rendered=len(rows),planned=len(plan),
        full_materialization_complete=len(rows)==len(plan),new_VLM_calls=0,new_classifier_calls=0))
    print(json.dumps(dict(stage="materialized",targets=len(rows),planned=len(plan))),flush=True)
    return rows


def distances(a,b):
    xor=np.bitwise_xor(np.asarray(a,dtype=np.uint64)[:,None],np.asarray(b,dtype=np.uint64)[None,:])
    if hasattr(np,"bitwise_count"):return np.bitwise_count(xor)
    lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    return lut[xor.view(np.uint8).reshape(xor.shape+(8,))].sum(-1,dtype=np.uint8)


def fixed_near(distance,cosine):
    return (distance<=4 and cosine>=.980) or (distance<=8 and cosine>=.995)


def choose_qualified(rows,reasons,edges):
    """Prioritize frozen primary members; only their assigned reserves may fill."""
    accepted=[];selected=set();excluded=[];counts=Counter();roles={}
    primary=[i for i,r in enumerate(rows) if r["planned_role"]!="technical_reserve"]
    reserves=[i for i,r in enumerate(rows) if r["planned_role"]=="technical_reserve"]
    for i in primary+reserves:
        r=rows[i];role=r["planned_role"] if i in primary else r["reserve_for"]
        k=r["class_name"],role
        if r["planned_role"]=="technical_reserve" and counts[k]>=400:continue
        why=list(reasons.get(i,[]))
        duplicate=edges.get(i,set())&selected
        if duplicate:why.append("within_new_duplicate_of:"+rows[min(duplicate)]["sample_id"])
        if why:
            excluded.append(dict(sample_id=r["sample_id"],class_name=r["class_name"],role=role,reason=";".join(why)));continue
        selected.add(i);accepted.append({**r,"role":role,"technical_replacement":r["planned_role"]=="technical_reserve"})
        roles[i]=role;counts[k]+=1
    return accepted,excluded,counts


def embeddings(paths):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,Dataset
    from torchvision import models
    from PIL import Image
    cache=OUT/"dedup_vit_embeddings.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            if z["paths"].tolist()!=paths:raise ValueError("Stale embedding cache")
            return z["embeddings"]
    weights=models.ViT_B_16_Weights.IMAGENET1K_V1
    weight_path=Path.home()/".cache/torch/hub/checkpoints/vit_b_16-c867db91.pth"
    digest=sha(weight_path)
    dump("dedup_encoder_freeze.json",dict(model="torchvision ViT_B_16_Weights.IMAGENET1K_V1",weight_path=str(weight_path),
        sha256=digest,transform=repr(weights.transforms()),precision="float32",batch_size=48,
        role="near-duplicate embedding only, not target-classifier predictions"))
    torch.set_num_threads(8)
    transform=weights.transforms()
    class Images(Dataset):
        def __len__(self):return len(paths)
        def __getitem__(self,i):
            with Image.open(paths[i]) as im:return transform(im.convert("RGB"))
    model=models.vit_b_16(weights=weights);model.heads.head=nn.Identity();model.cuda().eval()
    values=[]
    with torch.inference_mode():
        for i,x in enumerate(DataLoader(Images(),batch_size=48,shuffle=False,num_workers=4,pin_memory=True)):
            values.append(model(x.cuda(non_blocking=True)).float().cpu().numpy())
            if i%20==0:print(json.dumps(dict(stage="dedup_embeddings",batches=i,images=len(paths))),flush=True)
    result=np.concatenate(values);result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
    np.savez_compressed(cache,paths=np.array(paths),embeddings=result)
    del model;torch.cuda.empty_cache()
    return result


def qualify():
    rows=materialize()
    if len(rows)!=5400:raise RuntimeError("All18 source packages and5400 planned targets must complete first")
    history=json.loads((OUT/"historical_binding_summary.json").read_text())
    if history["status"]!="pass":raise RuntimeError("Historical original/crop binding incomplete")
    historical={}
    for line in (OUT/"historical_fingerprints.jsonl").open():
        r=json.loads(line)
        if r["status"]=="ok":historical.setdefault(r["sha256"],r)
    hist=list(historical.values())
    historical_hashes=set(historical)|set((READY/"historical_hash_tokens.txt").read_text().splitlines())
    reasons=defaultdict(list);edges=defaultdict(set);pairs=[];evidence=[]
    seen_hashes={}
    for i,r in enumerate(rows):
        for digest in (r["image_sha256"],r["crop_sha256"]):
            if digest in seen_hashes and seen_hashes[digest]!=i:
                j=seen_hashes[digest];edges[i].add(j);edges[j].add(i)
            else:seen_hashes[digest]=i
    hp=np.array([int(r["phash"],16) for r in hist],dtype=np.uint64)
    hd=np.array([int(r[k],16) for r in hist for k in ("dhash","dhash_flip")],dtype=np.uint64)
    nd=np.array([int(r["original_dhash"],16) for r in rows],dtype=np.uint64)
    nf=np.array([int(r["original_dhash_flip"],16) for r in rows],dtype=np.uint64)
    np_=np.array([int(r["crop_phash"],16) for r in rows],dtype=np.uint64)
    for i,r in enumerate(rows):
        if r["image_sha256"] in historical_hashes or r["crop_sha256"] in historical_hashes:reasons[i].append("historical_exact_SHA")
    for start in range(0,len(rows),32):
        end=min(start+32,len(rows))
        dh=distances(nd[start:end],hd)
        pp=distances(np_[start:end],hp)
        for k in range(end-start):
            i=start+k
            if dh[k].min()<=4:reasons[i].append("historical_original_dhash_le4")
            if not reasons[i]:
                for j in np.flatnonzero(pp[k]<=8):pairs.append((i,int(j),int(pp[k,j]),"historical"))
        dd=np.minimum(distances(nd[start:end],nd),distances(nd[start:end],nf))
        pc=distances(np_[start:end],np_)
        for k in range(end-start):
            i=start+k
            for j in np.flatnonzero(dd[k]<=4):
                if j<i:edges[i].add(int(j));edges[int(j)].add(i)
            if not reasons[i]:
                for j in np.flatnonzero(pc[k]<=8):
                    if j<i and not reasons[j]:pairs.append((i,int(j),int(pc[k,j]),"within_new"))
        if start%320==0:print(json.dumps(dict(stage="hash_pair_screen",done=end,total=len(rows),crop_pairs=len(pairs))),flush=True)
    paths=sorted({rows[i]["crop_path"] for i,_,_,_ in pairs}|
        {hist[j]["path"] if scope=="historical" else rows[j]["crop_path"] for _,j,_,scope in pairs})
    lookup={p:i for i,p in enumerate(paths)}
    vectors=embeddings(paths) if paths else None
    for i,j,d,scope in pairs:
        left=rows[i]["crop_path"];right=hist[j]["path"] if scope=="historical" else rows[j]["crop_path"]
        cosine=float(vectors[lookup[left]]@vectors[lookup[right]])
        reject=fixed_near(d,cosine)
        if reject:
            evidence.append(dict(sample_id=rows[i]["sample_id"],other_path=right,scope=scope,phash_distance=d,cosine=cosine))
            if scope=="historical":reasons[i].append("historical_crop_phash_and_ViT")
            else:edges[i].add(j);edges[j].add(i)
    dump("near_duplicate_screen_summary.json",dict(historical_unique_content=len(hist),new_targets=len(rows),
        crop_candidate_pairs=len(pairs),ViT_images=len(paths),confirmed_crop_pairs=len(evidence),
        original_or_exact_historical_rejected=len(reasons),within_new_graph_edges=sum(map(len,edges.values()))//2))
    write("near_duplicate_evidence.csv",evidence,["sample_id","other_path","scope","phash_distance","cosine"])
    accepted,excluded,counts=choose_qualified(rows,reasons,edges)
    write("qualification_exclusions.csv",excluded,["sample_id","class_name","role","reason"])
    good=all(counts[c,role]==400 for c in CLASSES for role in ("selection","reference"))
    write("qualified_manifest.csv" if good else "incomplete_manifest.csv",accepted,list(accepted[0]))
    result=dict(status="pass" if good else "insufficient_after_fixed_deduplication",data_qualified=good,
        accepted=len(accepted),counts={c:{role:counts[c,role] for role in ("selection","reference")} for c in CLASSES},
        technical_replacements=sum(r["technical_replacement"] for r in accepted),exclusions=len(excluded),
        historical_unresolved=history["unresolved_references"],historical_hash_mismatches=history["hash_mismatch_references"],
        target_classifier_calls=0,VLM_calls=0,near_duplicate_encoder_images=len(paths),
        boundary="Project-accessible content checked; foundation pretraining independence and semantic label truth are not certified.")
    dump("qualification_summary.json",result)
    dump("qualified_data_freeze.json",dict(parent_protocol_sha256=sha(READY/"frozen_validation_protocol.json"),
        manifest_sha256=sha(OUT/("qualified_manifest.csv" if good else "incomplete_manifest.csv")),
        images_and_views={r["sample_id"]:{k:r[k] for k in ("image_sha256","crop_sha256","feature_crop_sha256","marked_image_sha256")} for r in accepted}))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["materialize","qualify"])
    args=p.parse_args();globals()[args.stage]()
