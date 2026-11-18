#!/usr/bin/env python3
"""Independent membership, pixel binding and exclusion checks after qualification."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_fresh_images_20261112 import ROOT, READY, OUT, read, dump, sha, fingerprints


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    summary=json.loads((OUT/'qualification_summary.json').read_text())
    require(summary['status']=='pass' and summary['data_qualified'], 'Qualification did not pass')
    frozen=json.loads((OUT/'qualified_data_freeze.json').read_text())
    require(sha(READY/'frozen_validation_protocol.json')==frozen['parent_protocol_sha256'], 'Parent protocol changed')
    require(sha(OUT/'qualified_manifest.csv')==frozen['manifest_sha256'], 'Qualified manifest changed')
    rows=read(OUT/'qualified_manifest.csv');plan=read(READY/'alternative_provisional_plan.csv')
    by_id={r['source_image_id']:r for r in plan}
    require(len(rows)==4800 and len({r['source_image_id'] for r in rows})==4800, 'Expected 4800 unique originals')
    counts=Counter((r['class_name'],r['role']) for r in rows)
    require(len(counts)==12 and set(counts.values())=={400}, 'Class/role counts changed')
    excluded={r['sample_id']:r for r in read(OUT/'qualification_exclusions.csv')}
    accepted={r['sample_id'] for r in rows}
    packages={p.stem.replace('_','/',1):json.loads(p.read_text()) for p in (OUT/'package_audits').glob('*.json')}
    require(len(packages)==18 and all(p['status']=='pass' for p in packages.values()), 'Source archive gate incomplete')
    history=json.loads((OUT/'historical_binding_summary.json').read_text())
    require(history['status']=='pass' and not any(history[k] for k in
        ('unresolved_references','hash_mismatch_references','decode_failures')), 'Historical binding failed')
    hist={}
    for line in (OUT/'historical_fingerprints.jsonl').open():
        r=json.loads(line)
        require(r['status']=='ok', 'Historical fingerprint failed')
        hist.setdefault(r['sha256'],r)
    forbidden=set(hist)|set((READY/'historical_hash_tokens.txt').read_text().splitlines())
    for r in rows:
        p=by_id[r['source_image_id']]
        for new,old in (('class_name','class_name'),('rank','rank'),('planned_role','role'),
                        ('annotation_id','annotation_id'),('package','package'),('source_file_name','source_file_name'),
                        ('bbox_norm_xyxy','bbox_norm_xyxy')):
            require(r[new]==p[old], 'Frozen membership/official target changed: '+r['sample_id'])
        require(r['role'] in ('selection','reference'), 'Invalid final role')
        if p['role']=='technical_reserve':
            role='selection' if int(p['rank'])<=850 else 'reference'
            require(r['role']==role, 'Reserve crossed its frozen role')
        else:require(r['role']==p['role'], 'Primary crossed its frozen role')
        source=packages[r['package']]['found'][Path(r['source_file_name']).name]
        require(source['sha256']==r['image_sha256'], 'Source stream/original hash mismatch')
        for k in ('image_sha256','crop_sha256','feature_crop_sha256','marked_image_sha256'):
            require(frozen['images_and_views'][r['sample_id']][k]==r[k], 'View digest changed')
        require(r['image_sha256'] not in forbidden and r['crop_sha256'] not in forbidden, 'Historical exact duplicate accepted')
    # Every omitted primary needs a pre-annotation technical exclusion. No replacements by labels/outcomes.
    for p in plan:
        if p['role']=='technical_reserve':continue
        sid=f"fresh-obj365-train-{p['source_image_id']}-{p['class_name']}"
        require(sid in accepted or sid in excluded, 'Primary disappeared without technical exclusion')
    for c,role in counts:
        selected_reserves=[int(r['rank']) for r in rows if r['class_name']==c and r['role']==role and r['planned_role']=='technical_reserve']
        if not selected_reserves:continue
        for p in plan:
            reserve_role='selection' if int(p['rank'])<=850 else 'reference'
            if p['class_name']!=c or p['role']!='technical_reserve' or reserve_role!=role or int(p['rank'])>=max(selected_reserves):continue
            sid=f"fresh-obj365-train-{p['source_image_id']}-{c}"
            require(sid in accepted or sid in excluded, 'Technical reserve skipped out of frozen rank order')
    def check_pixels(r):
        for path_key,hash_key in (('image_path','image_sha256'),('crop_path','crop_sha256'),
                                  ('feature_crop_path','feature_crop_sha256'),('marked_image_path','marked_image_sha256')):
            require(sha(r[path_key])==r[hash_key], 'Actual image/view content changed')
            with Image.open(r[path_key]) as im:im.verify()
        with Image.open(r['image_path']) as im:
            require(im.size==(int(r['image_width']),int(r['image_height'])), 'Actual dimensions changed')
        # Recompute perceptual hashes from the accepted pixels, independently of stored binding rows.
        for kind,key in (('original','image_path'),('crop','crop_path')):
            f=fingerprints(r[key])
            require(f['status']=='ok', 'Decode failure during independent audit')
            for name in ('dhash','dhash_flip','phash'):
                require(f[name]==r[kind+'_'+name], 'Stored perceptual hash differs from pixels')
        return r['sample_id']
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(check_pixels,rows),1):
            if i%500==0:print(json.dumps(dict(stage='independent_pixel_audit',done=i,total=4800)),flush=True)
    # A separate uint64 implementation audits full-original dHash, including mirror.
    hp=np.array([int(h[k],16) for h in hist.values() for k in ('dhash','dhash_flip')],dtype=np.uint64)
    original=np.array([int(r['original_dhash'],16) for r in rows],dtype=np.uint64)
    mirror=np.array([int(r['original_dhash_flip'],16) for r in rows],dtype=np.uint64)
    lut=np.array([n.bit_count() for n in range(256)],dtype=np.uint8)
    def distance(a,b):
        x=a[:,None]^b[None,:]
        return lut[x.view(np.uint8).reshape(x.shape+(8,))].sum(axis=-1)
    for start in range(0,len(rows),16):
        block=original[start:start+16]
        require(not (distance(block,hp)<=4).any(), 'Historical original near duplicate accepted')
        d=np.minimum(distance(block,original),distance(block,mirror))
        for i in range(len(block)):d[i,start+i]=65
        require(not (d<=4).any(), 'New cross-role/class original near duplicate accepted')
    screen=json.loads((OUT/'near_duplicate_screen_summary.json').read_text())
    require(screen['new_targets']+screen.get('official_box_failures',0)==5400 and screen['historical_unique_content']==len(hist), 'Crop screen coverage incomplete')
    invalid={}
    for p in plan:
        b=json.loads(p['bbox_norm_xyxy'])
        if len(b)!=4 or not (0<=b[0]<b[2]<=1.000000001 and 0<=b[1]<b[3]<=1.000000001):
            sid=f"fresh-obj365-train-{p['source_image_id']}-{p['class_name']}"
            invalid[sid]=p
            require(sid not in accepted and excluded.get(sid,{}).get('reason')=='invalid_official_normalized_box', 'Official box failure not excluded before selection')
    require(screen.get('official_box_failures',0)==len(invalid), 'Official box failure accounting mismatch')
    path_to_id={r['crop_path']:r['sample_id'] for r in read(OUT/'materialized_targets.csv')}
    for e in read(OUT/'near_duplicate_evidence.csv'):
        if e['sample_id'] not in accepted:continue
        require(e['scope']!='historical', 'Historical confirmed crop duplicate accepted')
        require(path_to_id.get(e['other_path']) not in accepted, 'Both confirmed new crop duplicates accepted')
    result=dict(status='pass',independent_membership_and_pixel_audit=True,accepted_originals=4800,
        actual_image_and_view_files_rehashed=19200,perceptual_hashes_recomputed=True,
        historical_unique_contents=len(hist),full_original_dhash_rechecked=True,
        confirmed_crop_exclusions_checked=True,manifest_sha256=sha(OUT/'qualified_manifest.csv'),
        human_review=False,new_VLM_calls=0,target_classifier_calls=0,
        boundary='Crop ViT decisions use the frozen qualification evidence; this is not semantic ground-truth certification.')
    dump('independent_data_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
