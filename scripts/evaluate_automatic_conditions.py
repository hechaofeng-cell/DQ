#!/usr/bin/env python3
"""Bounded, state-enriched confirmation separate from general pool D."""
import argparse
import itertools
import json
import shutil
from collections import defaultdict
import numpy as np
from automatic_strengthening_common import *


def state(row,features,field):
    if field=='target_scale':
        from run_multiclass_object_selection_benchmark import scale_state
        return {scale_state(float(row['bbox_area_ratio']))}
    value=features[field]
    return set(value) if isinstance(value,list) else {value}


def select():
    from analyze_sixclass_difficulty import risk
    from prepare_automatic_strengthening import order_key
    rows=read_csv(E/'manifests/reservoir.csv');labels,failures=labels_for(rows,E)
    failed={r['sample_id'] for r in failures};rows=[r for r in rows if r['sample_id'] not in failed]
    conditions=json.loads((BASE/'frozen_E_conditions.json').read_text());used=set();selected=[];audit=[]
    for index,condition in enumerate(conditions):
        name=condition['class_name'];field=condition['feature'];value=condition['value'];groups=defaultdict(lambda: {'cases':[],'references':[]})
        for row in rows:
            if row['class_name']!=name or row['sample_id'] in used:continue
            features=labels['local'][row['sample_id']];states=state(row,features,field)
            if 'unknown' in states:continue
            score=risk(row,features);cell=score['h_midpoint'],score['unknown_dimensions']
            groups[cell]['cases' if value in states else 'references'].append(row)
        pairs=[]
        for cell in sorted(groups):
            left=sorted(groups[cell]['cases'],key=lambda r:order_key(index,'case',r['sample_id']))
            right=sorted(groups[cell]['references'],key=lambda r:order_key(index,'reference',r['sample_id']))
            pairs.extend((cell,l,r) for l,r in zip(left,right))
        # Fixed hash order across available exact-risk pairs, no model outputs.
        pairs=sorted(pairs,key=lambda p:order_key(index,p[1]['sample_id'],p[2]['sample_id']))[:30]
        for pair_index,(cell,left,right) in enumerate(pairs):
            for role,row in [('case',left),('reference',right)]:
                selected.append({**row,'condition_id':f'condition_{index+1:02d}','condition_feature':field,'condition_value':value,
                                 'condition_role':role,'pair_id':f'{index+1:02d}_{pair_index+1:02d}','matched_h_midpoint':cell[0],'matched_unknown_dimensions':cell[1]})
                used.add(row['sample_id'])
        audit.append(dict(condition_id=f'condition_{index+1:02d}',**condition,pairs=len(pairs),evaluable=len(pairs)>=20,
                          status='evaluable' if len(pairs)>=20 else 'insufficient_support_or_matching',selection_prediction_read=False))
    assert len({r['sample_id'] for r in selected})==len(selected)<=720
    write_csv(E/'manifests/selected.csv',selected);write_json(E/'selection_audit.json',dict(status='pass',conditions=audit,selected=len(selected),
        exclusions_due_to_technical_annotation_failure=len(failed),manual_review=False))
    from independent_confirmation_common import freeze
    paths=[BASE/'frozen_E_conditions.json',E/'manifests/reservoir.csv',E/'manifests/selected.csv',E/'selection_audit.json']
    paths+=list((E/'annotations').glob('*/*/parsed/*.json'))
    freeze(E/'pre_prediction_freeze.sha256.json',paths)


def infer():
    from independent_confirmation_common import verify
    verify(E/'pre_prediction_freeze.sha256.json')
    rows=read_csv(E/'manifests/selected.csv')
    if not rows:return
    import torch
    torch.set_num_threads(8)
    import sixclass_car_support as support
    support.configure();b=support.benchmark
    b.CHECKPOINT_ROOT=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'
    ctx=dict(rows=rows,train_rows=read_csv(E/'manifests/train.csv'))
    for i,(model,seed) in enumerate(itertools.product(MODELS,SEEDS),1):
        note('infer_E',run=i,total=24,model=model,seed=seed,targets=len(rows))
        folder=E/'benchmark/runs'/model/str(seed);folder.mkdir(parents=True,exist_ok=True)
        original=ROOT/'artifacts/sixclass_car_extension_20261004/benchmark/runs'/model/str(seed)/'train_features.npz'
        if not (folder/'train_features.npz').exists():shutil.copy2(original,folder/'train_features.npz')
        b.infer_and_extract(E/'benchmark',ctx,model,seed,4)


def evaluate():
    from independent_confirmation_common import verify
    verify(E/'pre_prediction_freeze.sha256.json')
    rows=read_csv(E/'manifests/selected.csv');labels,_=labels_for(rows,E);records=[];suite=[]
    audit=json.loads((E/'selection_audit.json').read_text());condition_lookup={r['condition_id']:r for r in audit['conditions']}
    for model,seed in itertools.product(MODELS,SEEDS):
        if not rows:break
        with np.load(E/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
            assert saved['sample_ids'].tolist()==[r['sample_id'] for r in rows]
            logits=saved['logits'];pred=logits.argmax(1)
        for condition_id,condition in condition_lookup.items():
            section=[i for i,r in enumerate(rows) if r['condition_id']==condition_id]
            for source in labels:
                cases=[i for i in section if rows[i]['condition_role']=='case' and condition['value'] in state(rows[i],labels[source][rows[i]['sample_id']],condition['feature'])]
                references=[i for i in section if rows[i]['condition_role']=='reference' and condition['value'] not in state(rows[i],labels[source][rows[i]['sample_id']],condition['feature']) and
                            'unknown' not in state(rows[i],labels[source][rows[i]['sample_id']],condition['feature'])]
                case_by_pair={rows[i]['pair_id']:i for i in cases};reference_by_pair={rows[i]['pair_id']:i for i in references}
                pairs=sorted(set(case_by_pair)&set(reference_by_pair));n=len(pairs)
                left=np.asarray([pred[case_by_pair[p]]==LABELS[condition['class_name']] for p in pairs],dtype=float)
                right=np.asarray([pred[reference_by_pair[p]]==LABELS[condition['class_name']] for p in pairs],dtype=float)
                records.append(dict(condition_id=condition_id,source=source,model=model,seed=seed,class_name=condition['class_name'],feature=condition['feature'],value=condition['value'],
                    historical_weak_model=model in condition['frozen_models'][source],case_support=len(cases),reference_support=len(references),matched_pairs=n,evaluable=n>=20,
                    case_recall=float(left.mean()) if n else None,reference_recall=float(right.mean()) if n else None,
                    recall_gap=float((left-right).mean()) if n else None))
        weight=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'/model/'oi_gap_add0'/str(seed)/'best.pt'
        for i,row in enumerate(rows):suite.append(dict(**row,model=model,seed=seed,expected_predicted_label=int(pred[i]),
            expected_logits=json.dumps(logits[i].tolist()),checkpoint_path=str(weight),checkpoint_sha256=sha256(weight)))
    write_csv(E/'tables/condition_validation.csv',records);write_csv(E/'regression_suite.csv',suite)
    lines=['# 预选条件确认E','', '按状态富集，与一般候选池D分开；不得估计自然总体F1。','',
        '| 条件 | 类别 | 状态 | 匹配对 | 是否达到20对 |','|---|---|---|---:|---|']
    for row in audit['conditions']:lines.append(f"| {row['condition_id']} | {row['class_name']} | {row['feature']}={row['value']} | {row['pairs']} | {row['evaluable']} |")
    lines+=['','全部24运行与两种标注来源的正负差值见tables/condition_validation.csv；来源条件支持分别计算。',
            '当前实现提供可追溯条件差与匹配支持，不作新增多条件显著性声明；后续整体区间或检验必须按冻结观测单位计算。']
    (E/'report_zh.md').write_text('\n'.join(lines)+'\n')
    write_json(E/'final_audit.json',dict(status='pass',selected_targets=len(rows),evaluated_runs=24 if rows else 0,conditions=len(audit['conditions']),
        evaluable_local_conditions=sum(r['evaluable'] for r in audit['conditions']),manual_review=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('select_E','infer_E','evaluate_E'),required=True)
    {'select_E':select,'infer_E':infer,'evaluate_E':evaluate}[parser.parse_args().stage]()
