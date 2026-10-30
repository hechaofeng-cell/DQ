#!/usr/bin/env python3
"""Check new targets against frozen train/eval and amended candidates."""
from pathlib import Path
import json
import numpy as np
from dual_vlm_common import ROOT, SOURCE, read_csv, write_json, now, sha256
from audit_multiclass_near_duplicates import phash, hamming_matrix


def main():
    study = ROOT/'artifacts/sixclass_dual_vlm_replaced_20261004'
    rows = read_csv(study/'manifests/candidate_pool.csv')
    replacements = read_csv(study/'manifests/replacement_mapping.csv')
    new_ids = {r['new_sample_id'] for r in replacements}
    indices = [i for i,r in enumerate(rows) if r['sample_id'] in new_ids]
    new_rows = [rows[i] for i in indices]
    with np.load(study/'benchmark/embeddings/vit_b16_candidates.npz') as saved:
        assert saved['sample_ids'].tolist() == [r['sample_id'] for r in rows]
        candidate_embeddings = saved['embeddings']
    new_embeddings = candidate_embeddings[indices]
    new_embeddings /= np.maximum(np.linalg.norm(new_embeddings,axis=1,keepdims=True),1e-12)
    new_phash = np.asarray([phash(Path(r['crop_path'])) for r in new_rows],dtype=np.uint64)
    new_hashes = [sha256(r['crop_path']) for r in new_rows]
    flagged, counts = [], {}
    for scope, manifest, cache in (
        ('train',SOURCE/'manifests/train.csv',SOURCE/'audit/train_vit_embeddings.npz'),
        ('validation_and_base_test',SOURCE/'manifests/eval.csv',SOURCE/'audit/validation_and_base_test_vit_embeddings.npz'),
        ('candidate',study/'manifests/candidate_pool.csv',study/'benchmark/embeddings/vit_b16_candidates.npz')):
        other_rows = read_csv(manifest)
        with np.load(cache) as saved:
            assert saved['sample_ids'].tolist() == [r['sample_id'] for r in other_rows]
            embeddings = saved['embeddings'].copy()
        embeddings /= np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
        cosine = new_embeddings @ embeddings.T
        distances = hamming_matrix(new_phash,np.asarray([phash(Path(r['crop_path'])) for r in other_rows],dtype=np.uint64))
        counts[scope] = len(other_rows)
        for i,new in enumerate(new_rows):
            for j,other in enumerate(other_rows):
                if new['sample_id'] == other['sample_id']:
                    continue
                exact = new_hashes[i] == sha256(other['crop_path'])
                same_image = new['image_id'] == other['image_id']
                near = (distances[i,j]<=4 and cosine[i,j]>=.980) or (distances[i,j]<=8 and cosine[i,j]>=.995)
                if exact or same_image or near:
                    flagged.append(dict(scope=scope,new_target=new['sample_id'],other_target=other['sample_id'],
                                   phash_distance=int(distances[i,j]),vit_cosine=float(cosine[i,j]),
                                   exact_crop=exact,same_image=same_image))
    report = dict(status='pass' if not flagged else 'needs_resolution',checked_at=now(),new_targets=2,
                  comparison_counts=counts,flagged_pairs=flagged,rule='pHash<=4 & cosine>=.980 OR pHash<=8 & cosine>=.995; exact crop/image always flagged',
                  limits='automated threshold audit does not guarantee absence of all near duplicates',script_sha256=sha256(__file__))
    write_json(study/'replacement_near_duplicate_audit.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if flagged:
        raise ValueError('replacement duplicate candidates need data-only resolution')


if __name__=='__main__':
    main()
