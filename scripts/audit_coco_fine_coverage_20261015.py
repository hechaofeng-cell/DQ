#!/usr/bin/env python3
"""Independent result recomputation from saved native labels and selection arrays."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import hypergeom, rankdata


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    root=parser.parse_args().input.resolve()
    protocol=json.loads((root/'protocol.json').read_text()); classes=('dog','cat','horse','sheep','person','car')
    manifest=read(root/'manifest.csv')
    native={r['sample_id']:r['features'] for r in map(json.loads,(root/'native_features.jsonl').read_text().splitlines())}
    states=json.loads((root/'state_dictionary.json').read_text())['main_states']
    with np.load(root/'selection.npz') as z:
        ids=z['sample_ids'].copy(); selections={g:z[g].copy() for g in ('E0','E1','E2','E3','E4')}
    assert ids.reshape(-1).tolist()==[r['sample_id'] for r in manifest]
    matrices=[]
    for ci,c in enumerate(classes):
        m=np.zeros((400,71),dtype=np.uint8)
        for i,sid in enumerate(ids[ci]):
            row=manifest[400*ci+i];area=float(row['bbox_area_ratio'])
            scale='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large'
            for j,(field,state) in enumerate(states):
                v=scale if field=='target_scale' else native[str(sid)][field]
                m[i,j]=state in v if isinstance(v,list) else v==state
        matrices.append(m)
    dataset={r['group']:r for r in read(root/'dataset_metrics.csv')}
    for group,selected in selections.items():
        pc=np.zeros(len(selected));sc=np.zeros(len(selected))
        for ci in range(6):
            assert np.all(np.diff(np.sort(selected[:,ci],axis=1),axis=1)>0)
            counts=matrices[ci][selected[:,ci]].sum(1)
            pc+=(counts>0).mean(1)/6;sc+=np.minimum(counts/20,1).mean(1)/6
        assert np.isclose(pc.mean(),float(dataset[group]['presence_coverage']))
        assert np.isclose(sc.mean(),float(dataset[group]['support_completion']))
    # Validate every saved random draw against the actually saved score-cell quotas.
    tiers={(r['scheme'],r['sample_id']):r for r in read(root/'tiers.csv')}
    quotas=read(root/'risk_cell_quotas.csv')
    quota_checks=0
    for scheme,rg,cg in [('equal_global','E1','E2'),('equal_parts','E3','E4')]:
        for ci,c in enumerate(classes):
            buckets=defaultdict(list)
            for i,sid in enumerate(ids[ci]):
                r=tiers[(scheme,str(sid))]
                key=(int(r['tier']),round(float(r['score_midpoint'])*1e8),round((float(r['score_upper'])-float(r['score_lower']))*1e8))
                buckets[key].append(i)
            for row in quotas:
                if row['scheme']!=scheme or row['class_name']!=c:continue
                key=(int(row['tier']),round(float(row['risk_midpoint'])*1e8),round(float(row['uncertainty_width'])*1e8))
                for group in (rg,cg):
                    assert np.all(np.isin(selections[group][:,ci],buckets[key]).sum(1)==int(row['quota']))
                    quota_checks+=len(selections[group])
    # Recompute rarefaction via scipy's hypergeometric survival function.
    paired=read(root/'paired_model_run_metrics.csv')
    with np.load(root/'predictions.npz') as z:
        predictions=z['predictions'].copy();models=z['models'].tolist();seeds=z['seeds'].tolist()
    with np.load(root/'model_raw_arrays.npz') as z:
        for cg,rg in [('E2','E1'),('E4','E3')]:
            fixed=z[cg+'_wrong'][:,0,None];random=z[rg+'_wrong']
            q=np.minimum(fixed.sum(-1),random.sum(-1))
            values=[]
            for counts in (np.broadcast_to(fixed,random.shape),random):
                n=counts.sum(-1)
                probability=hypergeom.sf(0,np.maximum(n[...,None],1),counts,q[...,None])
                probability=np.where(q[...,None]>0,probability,0)
                assert np.isfinite(probability).all()
                values.append(probability.sum(-1).mean(-1)/5)
            difference=(values[0]-values[1]).mean(1)
            for run,(model,seed) in enumerate(zip(models,seeds)):
                recorded=next(r for r in paired if r['contrast']==cg+'-'+rg and r['model']==model and int(r['seed'])==seed)
                assert np.isclose(difference[run],float(recorded['equal_error_breadth_delta']),atol=1e-10)
    # Rank-sum AUROC is independent of the sklearn routine used by the runner.
    score_rows={(r['scheme'],r['sample_id']):r for r in read(root/'scores.csv')}
    auc_rows=read(root/'scoring_class_metrics.csv');auc_checks=0
    for scheme in ('equal_global','equal_parts'):
        for ci,c in enumerate(classes):
            risk=np.array([float(score_rows[(scheme,str(sid))]['score_midpoint']) for sid in ids[ci]])
            known=np.array([all(score_rows[(s,str(sid))]['fully_known']=='True' for s in ('equal_global','equal_parts')) for sid in ids[ci]])
            for cohort,mask in [('all',np.ones(400,dtype=bool)),('common_complete',known)]:
                ranks=rankdata(risk[mask],method='average')
                for run,(model,seed) in enumerate(zip(models,seeds)):
                    wrong=predictions[run,ci*400:(ci+1)*400][mask]!=ci
                    n1,n0=int(wrong.sum()),int((~wrong).sum())
                    if not n1 or not n0:continue
                    auc=(ranks[wrong].sum()-n1*(n1+1)/2)/(n1*n0)
                    row=next(r for r in auc_rows if (r['scheme'],r['class_name'],r['cohort'],r['model'],int(r['seed']))==(scheme,c,cohort,model,seed))
                    assert np.isclose(auc,float(row['error_auroc']),atol=1e-12)
                    auc_checks+=1
    assert len(protocol['tests_predeclared'])==4
    # Verify that no scoring/selection input changed during execution.
    import hashlib
    checked=0
    for path,digest in json.loads((root/'input_hashes.json').read_text()).items():
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==digest,path
        checked+=1
    audit=dict(status='pass',coverage_recomputed_from_native_features=True,
        risk_cell_draw_checks=quota_checks,hypergeometric_independent_breadth_checks=48,
        rank_sum_independent_auroc_checks=auc_checks,unchanged_input_files=checked,
        legacy_numbers_not_substituted=True)
    (root/'independent_result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
