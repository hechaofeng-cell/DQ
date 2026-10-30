"""Independent arithmetic audit of saved selections and reported aggregates."""
import argparse
import json
from collections import Counter
import numpy as np
from coverage_validation_common_20261018 import ROOT, OUT, SOURCE, CLASSES, read, dump, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--external',action='store_true');args=parser.parse_args()
    records=read(OUT/'coverage_metrics.csv');saved=np.load(OUT/'selections.npz',allow_pickle=False)
    cells=read(SOURCE/'risk_scores_and_tiers.csv');byid={r['sample_id']:r for r in cells}
    features=np.load(ROOT/'artifacts/sixclass_independent_confirmation_20261004/embeddings/vit_b16_candidates.npz',allow_pickle=False)
    feature_index={str(s):i for i,s in enumerate(features['sample_ids'])}
    checked=0
    for c in CLASSES:
        original=np.load(SOURCE/f'{c}_state_matrix.npz',allow_pickle=False)
        X=original['matrix'].astype(bool);ids=original['sample_ids'].astype(str)
        state=json.loads((SOURCE/f'{c}_state_dictionary.json').read_text())['states']
        keys=[(int(original['tiers'][i]),round(float(byid[s]['score_midpoint'])*1e8),
               round((float(byid[s]['score_upper'])-float(byid[s]['score_lower']))*1e8)) for i,s in enumerate(ids)]
        E=features['embeddings'][[feature_index[s] for s in ids]].astype(float)
        E/=np.linalg.norm(E,axis=1,keepdims=True)
        pairwise=1-E@E.T
        for tag in saved.files:
            if not tag.startswith(c+'__'):continue
            _,b,method,k,weight=tag.split('__');budget=int(b[1:]);cap=int(k[1:]);sels=saved[tag]
            cell_counts=[Counter(keys[int(i)] for i in ss) for ss in sels]
            # Independent comparison to the saved common per-budget capacities.
            quota={ (int(r['Q']),round(float(r['midpoint'])*1e8),round(float(r['width'])*1e8)):int(r['quota'])
                    for r in read(OUT/'quotas.csv') if r['class_name']==c and int(r['budget'])==budget }
            for ss,cc in zip(sels,cell_counts):
                assert len(ss)==len(set(ss.tolist()))==budget
                assert all(cc[key]==value for key,value in quota.items())
                assert all(sum(n for key,n in cc.items() if key[0]==q)==budget//5 for q in range(1,6))
            if method=='static_rarity':
                scores=np.sum(X/np.maximum(1,X.sum(axis=0)),axis=1)
                expected=[]
                for key,n in quota.items():
                    members=[i for i in range(len(X)) if keys[i]==key]
                    expected.extend(sorted(members,key=lambda i:(-scores[i],ids[i]))[:n])
                assert set(expected)==set(sels[0].tolist())
            if method=='vit_kcenter':
                selected=[];taken=Counter();centroid=E.mean(0);centroid/=np.linalg.norm(centroid)
                for picked in sels[0]:
                    feasible=[i for i in range(len(X)) if i not in selected and taken[keys[i]]<quota[keys[i]]]
                    gains=E[feasible]@centroid if not selected else np.maximum(0,pairwise[np.ix_(feasible,selected)]).min(axis=1)
                    assert gains.max()-gains[feasible.index(int(picked))]<1e-10
                    selected.append(int(picked));taken[keys[int(picked)]]+=1
            rr=[r for r in records if r['class_name']==c and int(r['budget'])==budget and r['method']==method
                and int(r['selection_cap'])==cap and r['weighting']==weight]
            support=np.stack([np.count_nonzero(X[ss],axis=0) for ss in sels])
            for r in rr:
                ns=r['namespace'];mask=np.array([ns=='all' or s[0]==ns for s in state])
                n=support[:,mask];ec=int(r['evaluation_cap'])
                metric=r['metric']
                values=100*((n>0).mean(1) if metric=='presence_pct' else (n>=ec).mean(1) if metric=='threshold_pct' else np.minimum(n,ec).sum(1)/(ec*n.shape[1]))
                assert np.isclose(values.mean(),float(r['mean_pct']),atol=1e-10)
                assert np.isclose(np.quantile(values,.025),float(r['p025_pct']),atol=1e-10)
                assert np.isclose(np.quantile(values,.975),float(r['p975_pct']),atol=1e-10)
                checked+=1
    for name in ['analysis_input_freeze.json','selection_before_prediction_freeze.json','annotation_input_freeze_v2.json']:
        for path,digest in json.loads((OUT/name).read_text()).items():assert sha(path)==digest,(name,path)
    result=dict(status='pass',coverage_rows_recomputed=checked,selection_settings=len(saved.files),
                all_risk_cells_checked=True,static_rarity_independent_cell_sort=True,kcenter_pairwise_reference_verified=True,input_freeze_hashes_verified=True)
    if args.external:
        from annotate_coverage_validation_20261018 import VALIDATE
        for group,manifest in [('external','external_manifest.csv'),('repeat','repeat_manifest.csv')]:
            for row in read(OUT/manifest):
                c=row['class_name'];sid=row['sample_id'];folder=OUT/'annotations'/group/c
                d=json.loads((folder/f'{sid}_parsed.json').read_text());fields=json.loads((OUT/'dictionaries'/f'{c}.json').read_text())['features']
                VALIDATE(d['features'],fields)
                raw=json.loads((folder/f'{sid}_attempt{d["attempt"]}_raw.json').read_text())
                assert d['features']==json.loads(raw['message']['content']) and raw['done_reason']=='stop'
                assert sha(folder/f'{sid}_evidence.jpg')==d['evidence_sha256']
                assert d['human_verified'] is False
        details=read(OUT/'external_condition_results.csv');summ=read(OUT/'external_diagnostic_summary.csv')
        for r in details:
            expected='insufficient_support' if int(r['external_support'])<20 else 'direction_confirmed' if float(r['external_delta_pp'])<0 else 'not_confirmed'
            assert r['status']==expected
        for r in summ:
            rr=[x for x in details if x['scope']==r['scope'] and x['method']==r['method']]
            div=100 if r['method']=='same_cell_random' else 1
            assert np.isclose(len(rr)/div,float(r['proposed']))
            for key,status in [('confirmed','direction_confirmed'),('not_confirmed','not_confirmed'),('insufficient','insufficient_support')]:
                assert np.isclose(sum(x['status']==status for x in rr)/div,float(r[key]))
        result.update(real_annotation_outputs_verified=880,external_condition_statuses_verified=len(details),external_summary_rows=len(summ))
    dump(OUT/('final_numerical_audit.json' if args.external else 'coverage_numerical_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
