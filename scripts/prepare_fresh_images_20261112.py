#!/usr/bin/env python3
"""Bind historical image content before qualifying the frozen fresh pools."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "artifacts/fresh_confirmation_readiness_20261112"
OUT = ROOT / "artifacts/fresh_image_qualification_20261112"
FIELDS = {"image_path", "original_image_path", "source_image_path", "crop_path", "source_crop_path", "crop_image_path"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, value):
    p = OUT / name; p.parent.mkdir(parents=True,exist_ok=True)
    tmp = p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)


def read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows, fields):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def prepare():
    if (OUT/"protocol.json").exists():
        return
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((READY/"final_freeze.json").read_text())
    for p,digest in frozen.items():
        if sha(p)!=digest:raise ValueError("Readiness input changed: "+p)
    protocol=dict(parent_protocol_sha256=sha(READY/"frozen_validation_protocol.json"),
        parent_plan_sha256=sha(READY/"alternative_provisional_plan.csv"),
        scope="Image acquisition and exact/near-duplicate qualification only; no semantic labels or target-classifier predictions",
        historical_scope="CSV image/original/crop/source paths in the frozen project registry, including portable datasets and previous worktree aliases; union of all physical image filenames for exact-hash fallback",
        relative_paths="Resolve against CSV parent and dataset ancestors; old /worktrees/*/QE suffix may resolve under current project. Verify expected SHA where present; otherwise hash actual content and retain relocation provenance.",
        unresolved_policy="Search actual registered images by expected SHA or unambiguous basename content; unresolved actual original/crop references block qualification",
        near_duplicates="exact SHA; full-original dHash<=4 including mirror; crop pHash<=4/cos>=.980 OR pHash<=8/cos>=.995, frozen pretrained ViT-B16",
        source_membership="Frozen original-image/class/role assignment and technical reserves only; no outcome-based replacement",
        data_root=str(OUT),human_verification=False,model_classification_or_annotation_calls=False)
    dump("protocol.json",protocol)
    dump("initial_code_freeze.json",{str(Path(__file__)):sha(__file__)})


@lru_cache(maxsize=None)
def resolve_path(csv_parent, value):
    p=Path(value)
    if p.is_absolute():
        options=[p]
        marker="/QE/"
        if marker in str(p):options.append(ROOT/str(p).split(marker,1)[1])
    else:
        parent=Path(csv_parent)
        options=[parent/p,ROOT/p]
        options += [x/p for x in parent.parents if x.is_relative_to(ROOT)]
    for candidate in options:
        if candidate.is_file():return str(candidate.resolve())
    return ""


def fingerprints(path):
    p=Path(path)
    try:
        h=sha(p)
        with Image.open(p) as source:
            im=ImageOps.exif_transpose(source).convert("RGB")
            width,height=im.size
            def dh(x):
                a=np.asarray(x.convert("L").resize((9,8),Image.Resampling.LANCZOS))
                return np.packbits((a[:,1:]>a[:,:-1]).ravel()).tobytes().hex()
            low=dctn(np.asarray(im.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32),type=2,norm="ortho")[:8,:8]
            bits=(low>=float(np.median(low.reshape(-1)[1:]))).ravel()
            ph=np.packbits(bits).tobytes().hex()
            return dict(path=str(p),sha256=h,width=width,height=height,dhash=dh(im),dhash_flip=dh(ImageOps.mirror(im)),phash=ph,status="ok",error="")
    except (OSError,ValueError) as exc:
        return dict(path=str(p),sha256="",width="",height="",dhash="",dhash_flip="",phash="",status="error",error=type(exc).__name__+":"+str(exc)[:160])


def historical():
    prepare()
    if (OUT/"historical_binding_summary.json").exists():return
    registry=read(READY/"historical_file_registry.csv")
    csvs=[Path(r["path"]) for r in registry if r["path"].endswith(".csv")]
    refs={}; expected=defaultdict(set); tables=0
    for p in csvs:
        with p.open(encoding="utf-8-sig",newline="",errors="replace") as f:
            reader=csv.DictReader(f); cols=set(reader.fieldnames or [])&FIELDS
            if not cols:continue
            tables+=1
            for r in reader:
                for col in cols:
                    value=r.get(col) or ""
                    if not value:continue
                    digest=""
                    if col in {"image_path","original_image_path","source_image_path"}:digest=r.get("image_sha256","") or ""
                    elif col in {"crop_path","source_crop_path"}:digest=r.get("crop_sha256","") or ""
                    if not re.fullmatch("[0-9a-fA-F]{64}",digest):digest=""
                    resolved=resolve_path(str(p.parent),value)
                    k=(str(p.parent),value,col,digest)
                    refs.setdefault(k,dict(csv=str(p),field=col,recorded_path=value,resolved_path=resolved,
                        expected_sha256=digest.lower(),sample_id=r.get("sample_id",r.get("image_id",""))))
                    if resolved and digest:expected[resolved].add(digest.lower())
        if tables%200==0:print(json.dumps(dict(stage="historical_references",tables=tables,unique_refs=len(refs))),flush=True)
    paths=sorted({r["resolved_path"] for r in refs.values() if r["resolved_path"]})
    # Persist content fingerprints incrementally so interruption does not lose work.
    cache=OUT/"historical_fingerprints.jsonl"
    known={}
    if cache.exists():
        for line in cache.open():
            r=json.loads(line);known[r["path"]]=r
    pending=[p for p in paths if p not in known]
    with cache.open("a") as f,ThreadPoolExecutor(max_workers=12) as pool:
        for i,r in enumerate(pool.map(fingerprints,pending),1):
            known[r["path"]]=r;f.write(json.dumps(r,ensure_ascii=False)+"\n")
            if i%1000==0:
                f.flush();print(json.dumps(dict(stage="historical_fingerprints",done=i,total=len(pending))),flush=True)
    by_hash={r["sha256"]:r["path"] for r in known.values() if r["status"]=="ok"}
    # Registered physical files can recover renamed copies that no longer appear
    # in a current manifest. Hash basename candidates before accepting any alias.
    basename=defaultdict(list)
    for item in registry:
        p=Path(item["path"])
        if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:basename[p.name].append(str(p))
    missing=[];mismatches=[];relocated=[]
    for r in refs.values():
        p=r["resolved_path"];digest=r["expected_sha256"]
        if not p:
            p=by_hash.get(digest,"") if digest else ""
            options=basename.get(Path(r["recorded_path"]).name,[])
            if not p and options:
                values=[]
                for candidate in options:
                    if candidate not in known:
                        value=fingerprints(candidate);known[candidate]=value
                        with cache.open("a") as f:f.write(json.dumps(value,ensure_ascii=False)+"\n")
                        if value["status"]=="ok":by_hash[value["sha256"]]=candidate
                    if known[candidate]["status"]=="ok":values.append(known[candidate])
                if digest:
                    p=next((v["path"] for v in values if v["sha256"]==digest),"")
                elif len({v["sha256"] for v in values})==1:p=values[0]["path"]
            r["resolved_path"]=p
        if not p or known[p]["status"]!="ok":missing.append(r);continue
        if digest and known[p]["sha256"]!=digest:mismatches.append({**r,"actual_sha256":known[p]["sha256"]});continue
        if str(Path(r["recorded_path"]))!=p:relocated.append(r)
    columns=["csv","field","recorded_path","resolved_path","expected_sha256","sample_id"]
    write("historical_references.csv",refs.values(),columns)
    write("historical_relocations.csv",relocated,columns)
    write("historical_unresolved.csv",missing,columns)
    write("historical_hash_mismatches.csv",mismatches,columns+["actual_sha256"])
    errors=[r for r in known.values() if r["status"]!="ok"]
    result=dict(status="pass" if not missing and not mismatches and not errors else "historical_binding_incomplete",
        historical_tables=tables,unique_references=len(refs),unique_actual_files=len(known),
        resolved_relative_or_relocated_references=len(relocated),unresolved_references=len(missing),
        hash_mismatch_references=len(mismatches),decode_failures=len(errors),
        exact_hashes=len(by_hash),new_VLM_calls=0,new_classifier_calls=0)
    dump("historical_binding_summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","historical"])
    args=p.parse_args();globals()[args.stage]()
