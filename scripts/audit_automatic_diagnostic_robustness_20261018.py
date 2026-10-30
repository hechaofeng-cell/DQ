"""Recompute overlap and weighted point contrasts without the experiment runner."""
from collections import Counter
import numpy as np
from coverage_validation_common_20261018 import *

DEST=OUT/'automatic_robustness'


def main():
    for name in ['inputs_freeze.json','external_label_hashes.json']:
        for path,digest in json.loads((DEST/name).read_text()).items():assert sha(path)==digest
    rows=read(OUT/'external_manifest.csv');risk={r['sample_id']:r for r in read(DEST/'external_risk_scores.csv')}
    pred,_=predictions('D',rows);checks=read(DEST/'unique_state_model_checks.csv');count=0
    for c in CLASSES:
        positions=[i for i,r in enumerate(rows) if r['class_name']==c]
        z=np.load(OUT/f'external_{c}_matrix.npz',allow_pickle=False);X=z['matrix'];ids=z['sample_ids'].astype(str)
        cells=[(int(risk[s]['score_bin']),int(risk[s]['uncertain'])) for s in ids]
        for rec in [r for r in checks if r['class_name']==c]:
            j=int(rec['state_index']);present=X[:,j].astype(bool);ys=(pred[rec['model']][:,positions]==ALL_CLASSES.index(c)).mean(0)
            a=Counter(cells[i] for i in range(200) if present[i]);b=Counter(cells[i] for i in range(200) if not present[i])
            use=[key for key in a if a[key]>=5 and b[key]>=5];na=sum(a[k] for k in use);nb=sum(b[k] for k in use)
            assert na==int(rec['retained_state_targets']) and nb==int(rec['retained_control_targets'])
            fraction=na/int(present.sum()) if present.any() else 0
            valid=na>=20 and nb>=20 and fraction>=.8-1e-12
            assert valid==bool(int(rec['adjustment_evaluable']))
            if valid:
                delta=0
                for key in use:
                    ia=[i for i in range(200) if cells[i]==key and present[i]]
                    ib=[i for i in range(200) if cells[i]==key and not present[i]]
                    delta+=a[key]/na*(sum(ys[i] for i in ia)/len(ia)-sum(ys[i] for i in ib)/len(ib))
                assert np.isclose(delta*100,float(rec['adjusted_delta_pp']),atol=1e-10)
            count+=1
    detail=read(DEST/'condition_checks.csv');summ=read(DEST/'summary.csv')
    for row in summ:
        rr=[r for r in detail if r['method']==row['method'] and r['scope']==row['scope']];div=100 if row['method']=='same_cell_random' else 1
        assert len(rr)/div==float(row['proposed'])
        for t in [0,5,10,20]:
            n=sum(int(r['external_support'])>=20 and (float(r['external_delta_pp'])<0 if t==0 else float(r['external_delta_pp'])<=-t+1e-10) for r in rr)
            assert np.isclose(n/div,float(row[f'confirmed_{t}pp']))
        for key in ['adjustment_evaluable','adjusted_direction','adjusted_10pp','all_seeds_negative','all_seeds_at_least_10pp']:
            assert np.isclose(sum(int(r[key]) for r in rr)/div,float(row[key]))
    audit=dict(status='pass',independent_overlap_and_contrast_checks=count,summary_rows_recomputed=len(summ),frozen_hashes_verified=True)
    dump(DEST/'independent_audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
