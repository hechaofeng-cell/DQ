#!/usr/bin/env python3
"""Reference agreement and reproducible sensitivity; no manual truth claims."""
import argparse
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint
from automatic_strengthening_common import *


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def audit(include_d=False):
    destination=BASE/'automatic_audit';records=[];agreement=[];risk_rows=[];inputs=[];stratification=[];confusions=[]
    pools=[('A',A),('B',B),('C',C)]+([('D',D)] if include_d else [])
    from analyze_sixclass_difficulty import risk
    for pool,path in pools:
        rows=read_csv(path/'manifests/candidate_pool.csv');labels,failures=labels_for(rows,path)
        if failures:raise ValueError(f'{pool}: strict paired annotations missing: {len(failures)}')
        groups=defaultdict(list);field_groups=defaultdict(list);risk_groups=defaultdict(list)
        for row in rows:
            iid=row['sample_id'];name=row['class_name'];box=json.loads(row['bbox_norm_xyxy']) if row.get('bbox_norm_xyxy') else None
            with Image.open(row['image_path']) as original:
                image=ImageOps.exif_transpose(original);width,height=image.size
            area=(box[2]-box[0])*(box[3]-box[1]) if box else None
            in_bounds=bool(box and 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1)
            difference=abs(area-float(row['bbox_area_ratio'])) if area is not None else None
            paths_present=all(Path(row[k]).is_file() for k in ('image_path','crop_path','marked_image_path','feature_crop_path'))
            hash_valid=all(not row.get(k) or sha256(row[p])==row[k] for k,p in [('image_sha256','image_path'),('crop_sha256','crop_path')])
            inputs.append(dict(pool=pool,sample_id=iid,class_name=name,width=width,height=height,bbox_in_bounds=in_bounds,
                               area_difference=difference,paths_present=paths_present,hash_valid=hash_valid))
            local=labels['local'][iid];online=labels['online'][iid]
            shared=json.loads((path/'configs/shared_object_schema.json').read_text())['universal_features']
            for definition in shared:
                field=definition['feature_id'];left=local[field];right=online[field]
                left=set(left) if isinstance(left,list) else {left};right=set(right) if isinstance(right,list) else {right}
                field_groups[name,field].append((left,right))
            risks={s:risk(row,labels[s][iid]) for s in labels}
            for source,value in risks.items():risk_rows.append(dict(pool=pool,source=source,class_name=name,sample_id=iid,**value))
            risk_groups[name].append((iid,risks['local']['h_midpoint'],risks['online']['h_midpoint']))
            known=all('unknown' not in (v if isinstance(v,list) else [v]) for v in list(local.values())+list(online.values()))
            shared_fields=[x['feature_id'] for x in shared]
            unknown=any('unknown' in (labels[s][iid][f] if isinstance(labels[s][iid][f],list) else [labels[s][iid][f]]) for s in labels for f in shared_fields)
            same=all(local[f]==online[f] for f in shared_fields)
            stratification.append(dict(pool=pool,sample_id=iid,class_name=name,group='at_least_one_unknown' if unknown else 'all_shared_agree' if same else 'shared_disagreement'))
            for field,official in [('occlusion_level','is_occluded'),('truncation_level','is_truncated')]:
                if row.get(official) not in ('0','1') or row.get('is_depiction') is None or not row.get('image_id'):
                    continue
                # VOC truncation has different annotation semantics and is not the OI reference study.
                if len(row['image_id'])!=16 or any(c not in '0123456789abcdef' for c in row['image_id'].lower()):continue
                reference=int(row[official])
                for source in labels:
                    state=labels[source][iid][field]
                    prediction=None if state=='unknown' else int(state!='none')
                    groups[source,name,field].append((reference,prediction))
        for (source,name,field),pairs in groups.items():
            known=[(a,b) for a,b in pairs if b is not None];n=len(known);true=np.asarray([a for a,b in known]);pred=np.asarray([b for a,b in known])
            tp=int(((true==1)&(pred==1)).sum());tn=int(((true==0)&(pred==0)).sum());fp=int(((true==0)&(pred==1)).sum());fn=int(((true==1)&(pred==0)).sum())
            low,high=proportion_confint(tp+tn,n,method='wilson') if n else (None,None)
            records.append(dict(pool=pool,source=source,class_name=name,feature=field,reference='Open Images box binary flags',planned=len(pairs),known=n,
                                unknown=len(pairs)-n,unknown_rate=(len(pairs)-n)/len(pairs),tp=tp,tn=tn,fp=fp,fn=fn,
                                exact_agreement=(tp+tn)/n if n else None,ci95_low=finite(low),ci95_high=finite(high),
                                positive_agreement=tp/(tp+fn) if tp+fn else None,negative_agreement=tn/(tn+fp) if tn+fp else None,
                                cohen_kappa=finite(cohen_kappa_score(true,pred)) if len(set(true)|set(pred))>1 else None))
            confusions.extend(dict(pool=pool,source=source,class_name=name,feature=field,reference=a,predicted=b,count=sum(x==a and y==b for x,y in known)) for a in (0,1) for b in (0,1))
        for (name,field),pairs in field_groups.items():
            left=['|'.join(sorted(x)) for x,y in pairs];right=['|'.join(sorted(y)) for x,y in pairs]
            agreement.append(dict(pool=pool,class_name=name,feature=field,n=len(pairs),exact_agreement=float(np.mean([x==y for x,y in pairs])),
                jaccard=float(np.mean([len(x&y)/len(x|y) for x,y in pairs])),cohen_kappa=finite(cohen_kappa_score(left,right)) if len(set(left+right))>1 else None,
                local_unknown_rate=float(np.mean(['unknown' in x for x,y in pairs])),online_unknown_rate=float(np.mean(['unknown' in y for x,y in pairs]))))
        for name,values in risk_groups.items():
            local=[x[1] for x in values];online=[x[2] for x in values]
            left=sorted(range(len(values)),key=lambda i:(local[i],values[i][0]));right=sorted(range(len(values)),key=lambda i:(online[i],values[i][0]))
            l={i:rank//(len(values)//5)+1 for rank,i in enumerate(left)};r={i:rank//(len(values)//5)+1 for rank,i in enumerate(right)}
            agreement.append(dict(pool=pool,class_name=name,feature='derived_risk',n=len(values),exact_agreement=float(np.mean([l[i]==r[i] for i in l])),
                                  risk_spearman=finite(spearmanr(local,online).statistic),mean_abs_H_difference=float(np.mean(np.abs(np.asarray(local)-online)))))
        # Conditional error rates use the original prediction fact once per run.
        group_map={r['sample_id']:r['group'] for r in stratification if r['pool']==pool}
        for model in MODELS:
            for seed in SEEDS:
                cache=path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz'
                if not cache.exists():continue
                with np.load(cache) as saved:
                    if saved['sample_ids'].tolist()!=[r['sample_id'] for r in rows]:raise ValueError('Prediction identity mismatch')
                    pred=saved['logits'].argmax(1)
                for name in CLASSES:
                    for group in ('all_shared_agree','shared_disagreement','at_least_one_unknown'):
                        idx=[i for i,row in enumerate(rows) if row['class_name']==name and group_map[row['sample_id']]==group]
                        if idx: confusions.append(dict(pool=pool,model=model,seed=seed,class_name=name,feature='annotation_group_error',group=group,
                                                       support=len(idx),error_rate=float((pred[idx]!=LABELS[name]).mean())))
        note('automatic_audit',pool=pool,targets=len(rows),manual_review=False)
    write_csv(destination/'official_attribute_reference.csv',records);write_csv(destination/'dual_vlm_agreement.csv',agreement)
    write_csv(destination/'risk_scores.csv',risk_rows);write_csv(destination/'input_audit.csv',inputs)
    write_csv(destination/'annotation_groups.csv',stratification);write_csv(destination/'reference_confusion_and_group_errors.csv',confusions)
    sensitivity=[]
    for pool,path in pools:
        candidate=path/'robustness/tables/risk_variant_comparisons.csv'
        if candidate.exists():sensitivity.extend(dict(pool=pool,**r) for r in read_csv(candidate))
    write_csv(destination/'existing_risk_sensitivity.csv',sensitivity)
    broken=[r for r in inputs if not r['bbox_in_bounds'] or not r['paths_present'] or not r['hash_valid'] or r['area_difference'] is None or r['area_difference']>1e-6]
    write_json(destination/'audit.json',dict(status='pass' if not broken else 'needs_input_investigation',pools=[p for p,_ in pools],
        targets=len(inputs),official_comparison_rows=len(records),input_errors=broken,manual_review=False,agreement_is_accuracy=False,
        uncertainty='Wilson marginal intervals; descriptive, not joint all-field accuracy'))
    if broken:raise ValueError(f'Input audit has {len(broken)} issues; see automatic_audit/audit.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-D',action='store_true');audit(parser.parse_args().include_D)
